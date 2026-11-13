//! Providers bound per store (agent D83, migration 0016): `GET|PUT|DELETE
//! /api/stores/{id}/providers`, resolution store → nearest ancestor → tenant, the model rule
//! (`embed_model_mismatch` on a store with content), and `POST|GET /api/stores/{id}/reembed-copy`
//! (with its delta passes, D91). Also D90 (`ambiguous_root` once a tenant has two roots) and D92
//! (chat bound per store).
//! Providers are mock axum servers speaking the OpenAI `/embeddings` and Cohere `/rerank` shapes;
//! "model B" answers the fake embedder's vectors reversed, so a vector it made is told apart from
//! one model A made.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use axum::Json;
use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use common::{OTHER_TENANT, TENANT, TestDb, other, test_config, tn};
use pgvector::Vector;
use serde_json::{Value, json};
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::inference::{Embedder, FakeEmbedder, FakeReranker, Reranker};
use strata_rag::providers::{Deployment, KeyCipher, LaneLimits, Providers};
use strata_rag::vecmath::cosine;
use tower::ServiceExt;
use uuid::Uuid;

const KEY: [u8; 32] = [42u8; 32];
const MODEL_A: &str = "acme/a@1024";
const MODEL_B: &str = "acme/b@1024";

#[derive(Default)]
struct Mock {
    /// `Some(k)`: anything but `Bearer k` is 401.
    key: Option<String>,
    /// Answer the fake embedder's vectors reversed ("model B").
    reverse: AtomicBool,
    embeds: AtomicUsize,
    /// Texts embedded, over all calls.
    texts: AtomicUsize,
    reranks: AtomicUsize,
    chats: AtomicUsize,
    /// Armed by a test: the first embed request holding a text with "harpoon" signals the first
    /// sender, then waits for the second channel before it answers.
    hold: std::sync::Mutex<Option<Hold>>,
}

struct Hold {
    reached: tokio::sync::oneshot::Sender<()>,
    release: tokio::sync::oneshot::Receiver<()>,
}

struct MockServer {
    url: String,
    m: Arc<Mock>,
}

impl MockServer {
    fn texts(&self) -> usize {
        self.m.texts.load(Ordering::SeqCst)
    }
    fn embeds(&self) -> usize {
        self.m.embeds.load(Ordering::SeqCst)
    }
    fn chats(&self) -> usize {
        self.m.chats.load(Ordering::SeqCst)
    }
    /// Arms the hold (see `Mock::hold`): returns (reached, release).
    fn arm_hold(
        &self,
    ) -> (
        tokio::sync::oneshot::Receiver<()>,
        tokio::sync::oneshot::Sender<()>,
    ) {
        let (reached_tx, reached_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        *self.m.hold.lock().unwrap() = Some(Hold {
            reached: reached_tx,
            release: release_rx,
        });
        (reached_rx, release_tx)
    }
}

fn unauthorized(m: &Mock, headers: &HeaderMap) -> bool {
    let auth = headers.get("authorization").and_then(|v| v.to_str().ok());
    m.key
        .as_ref()
        .is_some_and(|k| auth != Some(format!("Bearer {k}").as_str()))
}

/// The vector "model B" gives `text`.
fn reversed(text: &str) -> Vec<f32> {
    let mut v = FakeEmbedder.embed(&[text.to_string()]).unwrap().remove(0);
    v.reverse();
    v
}

/// The vector "model A" gives `text`.
fn plain(text: &str) -> Vec<f32> {
    FakeEmbedder.embed(&[text.to_string()]).unwrap().remove(0)
}

async fn embeddings(
    State(m): State<Arc<Mock>>,
    headers: HeaderMap,
    Json(b): Json<Value>,
) -> Response {
    m.embeds.fetch_add(1, Ordering::SeqCst);
    if unauthorized(&m, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let texts: Vec<String> = serde_json::from_value(b["input"].clone()).unwrap();
    m.texts.fetch_add(texts.len(), Ordering::SeqCst);
    if texts.iter().any(|t| t.contains("harpoon")) {
        let hold = m.hold.lock().unwrap().take();
        if let Some(h) = hold {
            let _ = h.reached.send(());
            let _ = h.release.await;
        }
    }
    let reverse = m.reverse.load(Ordering::SeqCst);
    let data: Vec<Value> = FakeEmbedder
        .embed(&texts)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, mut v)| {
            if reverse {
                v.reverse();
            }
            json!({"index": i, "embedding": v})
        })
        .collect();
    Json(json!({"data": data})).into_response()
}

async fn rerank(State(m): State<Arc<Mock>>, headers: HeaderMap, Json(b): Json<Value>) -> Response {
    m.reranks.fetch_add(1, Ordering::SeqCst);
    if unauthorized(&m, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let docs: Vec<String> = serde_json::from_value(b["documents"].clone()).unwrap();
    let scores = FakeReranker
        .score(b["query"].as_str().unwrap(), &docs)
        .unwrap();
    let results: Vec<Value> = scores
        .iter()
        .enumerate()
        .map(|(i, s)| json!({"index": i, "relevance_score": s}))
        .collect();
    Json(json!({"results": results})).into_response()
}

/// Answers `from-b` when the mock is "model B" (reversed), else `from-a`: which provider spoke.
async fn chat(State(m): State<Arc<Mock>>, headers: HeaderMap) -> Response {
    m.chats.fetch_add(1, Ordering::SeqCst);
    if unauthorized(&m, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let who = if m.reverse.load(Ordering::SeqCst) {
        "from-b"
    } else {
        "from-a"
    };
    Json(json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": who}}]}))
        .into_response()
}

async fn mock(key: Option<&str>, reverse: bool) -> MockServer {
    let m = Arc::new(Mock {
        key: key.map(str::to_string),
        reverse: AtomicBool::new(reverse),
        ..Mock::default()
    });
    let app = Router::new()
        .route("/embeddings", post(embeddings))
        .route("/rerank", post(rerank))
        .route("/chat/completions", post(chat))
        .with_state(m.clone());
    MockServer {
        url: common::mock::spawn(app).await,
        m,
    }
}

/// No defaults (resolves like production), but registrations may name the loopback mocks.
const LOCAL_EU: &str = "region = \"eu\"\nmode = \"dev\"\n\
                        [policy]\nallowed_dialects = [\"openai\", \"cohere\"]\n\
                        allow_internal_provider_hosts = true\n";

fn embed_reg(m: &MockServer, model: &str, key: Option<&str>) -> Value {
    let mut v = json!({
        "base_url": m.url, "dialect": "openai", "model": model, "dims": 1024,
        "regions": ["eu"], "zdr": true, "timeout_ms": 5000, "concurrency": 4
    });
    if let Some(k) = key {
        v["api_key"] = json!(k);
    }
    v
}

fn chat_reg(m: &MockServer, key: Option<&str>) -> Value {
    let mut v = json!({
        "base_url": m.url, "dialect": "openai", "model": "local-llm", "regions": ["eu"], "zdr": true
    });
    if let Some(k) = key {
        v["api_key"] = json!(k);
    }
    v
}

fn rerank_reg(m: &MockServer) -> Value {
    json!({"base_url": m.url, "dialect": "cohere", "model": "rr", "regions": ["eu"], "zdr": true})
}

struct Setup {
    engine: Arc<Engine>,
    app: Router,
    db: TestDb,
    /// The tenant's provider ("model A").
    a: MockServer,
    /// A store provider ("model B", reversed vectors).
    b: MockServer,
}

/// A registry engine; `TENANT` and `OTHER_TENANT` both register model A for embed + rerank.
async fn setup() -> Setup {
    let db = TestDb::start().await;
    let dep = Deployment::parse(LOCAL_EU, &Default::default()).expect("deployment");
    let providers = Providers::registry(
        db.pool.clone(),
        dep,
        Some(KeyCipher::new(&KEY)),
        LaneLimits::default(),
    )
    .await
    .expect("registry");
    let engine = Arc::new(Engine::with_source(
        db.pool.clone(),
        providers,
        Arc::new(strata_rag::text::WhitespaceCounter),
        test_config(),
    ));
    let app = strata_rag::api::router(engine.clone());
    let a = mock(None, false).await;
    let b = mock(None, true).await;
    for tenant in [TENANT, OTHER_TENANT] {
        let (s, v) = api(
            &app,
            "PUT",
            "/api/tenant/providers",
            tenant,
            Some(json!({"embed": embed_reg(&a, "acme/a", None), "rerank": rerank_reg(&a)})),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
    }
    Setup {
        engine,
        app,
        db,
        a,
        b,
    }
}

async fn api(
    app: &Router,
    method: &str,
    uri: &str,
    tenant: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("x-tenant-id", tenant)
        .header("content-type", "application/json")
        .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    (status, v)
}

fn providers_uri(store: Uuid) -> String {
    format!("/api/stores/{store}/providers")
}

async fn put_store(app: &Router, store: Uuid, body: Value) -> (StatusCode, Value) {
    api(app, "PUT", &providers_uri(store), TENANT, Some(body)).await
}

async fn get_store(app: &Router, store: Uuid) -> Value {
    let (s, v) = api(app, "GET", &providers_uri(store), TENANT, None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v
}

fn new_store(parent_id: Option<Uuid>, kind: StoreKind, name: &str) -> NewStore {
    NewStore {
        parent_id,
        kind,
        name: name.to_string(),
        description: format!("{name} holds documents about the sea"),
    }
}

async fn create(engine: &Engine, parent: Option<Uuid>, kind: StoreKind, name: &str) -> Uuid {
    engine
        .create_store(tn(), new_store(parent, kind, name))
        .await
        .unwrap()
        .id
}

async fn embed_model(db: &TestDb, store: Uuid) -> String {
    sqlx::query_scalar("SELECT embed_model FROM stores WHERE id = $1")
        .bind(store)
        .fetch_one(&db.pool)
        .await
        .unwrap()
}

/// An empty root re-tagged to model B by its own store binding.
async fn gen2_root(s: &Setup) -> Uuid {
    let gen2 = create(&s.engine, None, StoreKind::Router, "gen2").await;
    assert_eq!(
        embed_model(&s.db, gen2).await,
        MODEL_A,
        "born in the tenant's model"
    );
    let (st, v) = put_store(
        &s.app,
        gen2,
        json!({"embed": embed_reg(&s.b, "acme/b", None)}),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["embed_model"], MODEL_B, "{v}");
    gen2
}

const SEA: &str = "The harpoon struck the white whale. The crew cheered on the deck of the ship.";
const ABYSS: &str = "Anglerfish hunt in the dark abyss with a glowing lure far below the surface.";
const REEF: &str = "Coral reefs shelter clownfish among the anemones in warm shallow water.";

/// main root → leaf `sea` (two documents), router `deep` → leaf `abyss` (one), in model A.
async fn main_tree(engine: &Engine) -> (Uuid, Uuid) {
    let root = create(engine, None, StoreKind::Router, "main").await;
    let sea = create(engine, Some(root), StoreKind::Leaf, "sea").await;
    let deep = create(engine, Some(root), StoreKind::Router, "deep").await;
    let abyss = create(engine, Some(deep), StoreKind::Leaf, "abyss").await;
    engine
        .ingest(tn(), sea, "sea.txt".into(), Some("Sea".into()), SEA)
        .await
        .unwrap();
    engine
        .ingest(tn(), sea, "reef.txt".into(), None, REEF)
        .await
        .unwrap();
    engine
        .ingest(tn(), abyss, "abyss.txt".into(), None, ABYSS)
        .await
        .unwrap();
    (root, sea)
}

fn query(root: Option<Uuid>) -> strata_rag::query::QueryRequest {
    let mut q = json!({"query": "white whale harpoon", "route_mode": "embed", "min_score": -1.0});
    if let Some(r) = root {
        q["root_id"] = json!(r);
    }
    serde_json::from_value(q).unwrap()
}

#[tokio::test]
async fn store_binding_wins_over_tenant() {
    let s = setup().await;
    let (main, _) = main_tree(&s.engine).await;
    let gen2 = gen2_root(&s).await;
    let v = get_store(&s.app, gen2).await;
    assert_eq!(v["embed"]["source"], "store", "{v}");
    assert_eq!(v["embed"]["model"], "acme/b", "{v}");
    assert_eq!(v["rerank"]["source"], "tenant", "{v}");
    assert_eq!(embed_model(&s.db, gen2).await, MODEL_B);
    // the re-tag embedded the store's description with B
    let desc: Vector = sqlx::query_scalar("SELECT desc_vec FROM stores WHERE id = $1")
        .bind(gen2)
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    let want = reversed(&strata_rag::engine::desc_text(
        "gen2",
        "gen2 holds documents about the sea",
    ));
    assert!(cosine(&desc.to_vec(), &want) > 0.9999);

    // a leaf under gen2 is embedded and filled by B; A is not called for it
    let leaf = create(&s.engine, Some(gen2), StoreKind::Leaf, "bay").await;
    let (a_before, b_before) = (s.a.texts(), s.b.texts());
    s.engine
        .ingest(tn(), leaf, "bay.txt".into(), None, SEA)
        .await
        .unwrap();
    assert_eq!(
        s.a.texts(),
        a_before,
        "the tenant's provider was not called"
    );
    assert!(s.b.texts() > b_before);
    // a query from gen2 embeds with B, a query from main with A
    let b_before = s.b.embeds();
    let hits = s.engine.query(tn(), &query(Some(gen2))).await.unwrap();
    assert!(!hits.results.is_empty());
    assert!(s.b.embeds() > b_before);
    let a_before = s.a.embeds();
    let hits = s.engine.query(tn(), &query(Some(main))).await.unwrap();
    assert!(!hits.results.is_empty());
    assert!(s.a.embeds() > a_before);
    assert!(
        hits.results
            .iter()
            .all(|r| r.store_path.first().map(String::as_str) == Some("main")),
        "{:?}",
        hits.results
            .iter()
            .map(|r| &r.store_path)
            .collect::<Vec<_>>()
    );
    // two roots and none named: no tree is guessed (D90)
    let err = s.engine.query(tn(), &query(None)).await.unwrap_err();
    assert_eq!(err.code(), "ambiguous_root");
}

#[tokio::test]
async fn ancestor_binding_inherited() {
    let s = setup().await;
    let gen2 = gen2_root(&s).await;
    let deep = create(&s.engine, Some(gen2), StoreKind::Router, "deep").await;
    let leaf = create(&s.engine, Some(deep), StoreKind::Leaf, "abyss").await;
    for id in [deep, leaf] {
        assert_eq!(
            embed_model(&s.db, id).await,
            MODEL_B,
            "created with the parent's binding"
        );
        let v = get_store(&s.app, id).await;
        assert_eq!(v["embed"]["source"], "ancestor", "{v}");
        assert_eq!(v["embed"]["from_store"], json!(gen2), "{v}");
        assert_eq!(v["embed"]["model"], "acme/b", "{v}");
        assert!(v["embed"].get("api_key").is_none());
    }
    let b_before = s.b.texts();
    s.engine
        .ingest(tn(), leaf, "abyss.txt".into(), None, ABYSS)
        .await
        .unwrap();
    assert!(s.b.texts() > b_before);
    let emb: Vector =
        sqlx::query_scalar("SELECT embedding FROM chunks WHERE store_id = $1 LIMIT 1")
            .bind(leaf)
            .fetch_one(&s.db.pool)
            .await
            .unwrap();
    let text: String = sqlx::query_scalar("SELECT text FROM chunks WHERE store_id = $1 LIMIT 1")
        .bind(leaf)
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    assert!(
        cosine(&emb.to_vec(), &reversed(&text)) > 0.9999,
        "B made the vector"
    );
}

#[tokio::test]
async fn tenant_binding_used_without_store_binding() {
    let s = setup().await;
    let (main, sea) = main_tree(&s.engine).await;
    for id in [main, sea] {
        let v = get_store(&s.app, id).await;
        assert_eq!(v["embed"]["source"], "tenant", "{v}");
        assert_eq!(v["rerank"]["source"], "tenant", "{v}");
        assert!(v["embed"].get("from_store").is_none(), "{v}");
        assert_eq!(v["embed_model"], MODEL_A, "{v}");
    }
    let (a_before, b_before) = (s.a.texts(), s.b.texts());
    s.engine
        .ingest(tn(), sea, "more.txt".into(), None, ABYSS)
        .await
        .unwrap();
    assert!(s.a.texts() > a_before);
    assert_eq!(s.b.texts(), b_before);

    // a removed store binding falls back to the tenant's
    let gen2 = gen2_root(&s).await;
    let (st, v) = api(&s.app, "DELETE", &providers_uri(gen2), TENANT, None).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["embed"]["source"], "tenant", "{v}");
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM store_providers")
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    assert_eq!(n, 0);
    // gen2 still holds B's vectors: with A resolved again it is refused, not mixed
    let err = s.engine.query(tn(), &query(Some(gen2))).await.unwrap_err();
    assert_eq!(err.code(), "embed_model_mismatch");
}

#[tokio::test]
async fn store_embed_mismatch_409() {
    let s = setup().await;
    let (main, sea) = main_tree(&s.engine).await;
    // a store with documents keeps its model
    let (st, v) = put_store(
        &s.app,
        sea,
        json!({"embed": embed_reg(&s.b, "acme/b", None)}),
    )
    .await;
    assert_eq!(st, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "embed_model_mismatch", "{v}");
    // so does a router whose subtree has content
    let (st, v) = put_store(
        &s.app,
        main,
        json!({"embed": embed_reg(&s.b, "acme/b", None)}),
    )
    .await;
    assert_eq!(st, StatusCode::CONFLICT, "{v}");
    // an empty store under an ancestor in another model cannot change model either
    let empty = create(&s.engine, Some(main), StoreKind::Leaf, "empty").await;
    let (st, v) = put_store(
        &s.app,
        empty,
        json!({"embed": embed_reg(&s.b, "acme/b", None)}),
    )
    .await;
    assert_eq!(st, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "embed_model_mismatch", "{v}");
    // nothing was stored, nothing re-tagged
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM store_providers")
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    assert_eq!(n, 0);
    for id in [main, sea, empty] {
        assert_eq!(embed_model(&s.db, id).await, MODEL_A);
    }
    // the same model through another provider is fine on a store with content
    let mut same_model = embed_reg(&s.b, "other-name", None);
    same_model["model_id"] = json!(MODEL_A);
    let (st, v) = put_store(&s.app, sea, json!({"embed": same_model})).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["embed"]["source"], "store", "{v}");
    // chat is a store capability too (D92): `null` removes it; an unknown capability is 422
    let (st, v) = put_store(&s.app, sea, json!({"chat": null})).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(
        v["embed"]["source"], "tenant",
        "the whole set was replaced: {v}"
    );
    assert_eq!(v["chat"]["source"], "unconfigured", "{v}");
    let (st, v) = put_store(&s.app, sea, json!({"summarize": null})).await;
    assert_eq!(st, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
}

#[tokio::test]
async fn a_store_key_is_sealed_and_never_returned() {
    const SECRET: &str = "sk-store-b-0123456789";
    let s = setup().await;
    let keyed = mock(Some(SECRET), true).await;
    let gen2 = create(&s.engine, None, StoreKind::Router, "gen2").await;
    let (st, v) = put_store(
        &s.app,
        gen2,
        json!({"embed": embed_reg(&keyed, "acme/b", Some(SECRET))}),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    let got = get_store(&s.app, gen2).await;
    assert_eq!(got, v, "PUT answers the stored view");
    assert_eq!(got["embed"]["has_api_key"], true, "{got}");
    assert!(!got.to_string().contains(SECRET), "key leaked: {got}");
    let (ct, nonce, version, text): (Vec<u8>, Vec<u8>, i32, String) = sqlx::query_as(
        "SELECT api_key_ct, api_key_nonce, key_version, p::text FROM store_providers p \
         WHERE store = $1",
    )
    .bind(gen2)
    .fetch_one(&s.db.pool)
    .await
    .unwrap();
    assert_eq!((nonce.len(), version), (12, 1));
    assert!(!ct.windows(SECRET.len()).any(|w| w == SECRET.as_bytes()));
    assert!(!text.contains(SECRET));
    // the sealed key works: a leaf under gen2 is embedded through the keyed provider
    let leaf = create(&s.engine, Some(gen2), StoreKind::Leaf, "bay").await;
    s.engine
        .ingest(tn(), leaf, "bay.txt".into(), None, SEA)
        .await
        .unwrap();
    // a PUT without api_key keeps it for the same base_url only
    let (st, _) = put_store(
        &s.app,
        gen2,
        json!({"embed": embed_reg(&keyed, "acme/b", None)}),
    )
    .await;
    assert_eq!(st, StatusCode::OK);
    let other_host = mock(None, true).await;
    let (st, v) = put_store(
        &s.app,
        gen2,
        json!({"embed": embed_reg(&other_host, "acme/b", None)}),
    )
    .await;
    assert_eq!(st, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "api_key_required", "{v}");
}

async fn wait_done(app: &Router, source: Uuid, job: &str) -> Value {
    for _ in 0..300 {
        let (st, v) = api(
            app,
            "GET",
            &format!("/api/stores/{source}/reembed-copy/{job}"),
            TENANT,
            None,
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        if v["state"] != "running" {
            return v;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    panic!("reembed copy {job} still running after 30 s");
}

#[tokio::test]
async fn reembed_copy_moves_all_chunks_with_target_model() {
    let s = setup().await;
    let (main, _sea) = main_tree(&s.engine).await;
    let source_chunks: Vec<(String, Vector)> =
        sqlx::query_as("SELECT text, embedding FROM chunks ORDER BY text")
            .fetch_all(&s.db.pool)
            .await
            .unwrap();
    assert!(source_chunks.len() >= 3);
    let gen2 = gen2_root(&s).await;
    let b_before = s.b.texts();

    let (st, v) = api(
        &s.app,
        "POST",
        &format!("/api/stores/{main}/reembed-copy"),
        TENANT,
        Some(json!({"target_store": gen2})),
    )
    .await;
    assert_eq!(st, StatusCode::ACCEPTED, "{v}");
    assert_eq!(v["state"], "running", "{v}");
    assert_eq!(v["embed_model"], MODEL_B, "{v}");
    assert_eq!(v["chunks_total"], source_chunks.len(), "{v}");
    let job = v["job"].as_str().unwrap().to_string();
    let done = wait_done(&s.app, main, &job).await;
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["chunks_done"], source_chunks.len(), "{done}");
    assert!(done.get("error").is_none(), "{done}");
    assert!(s.b.texts() >= b_before + source_chunks.len());

    // the mirror: same names and kinds under gen2, all in model B
    let tree = s.engine.store_tree(tn()).await.unwrap();
    let under = |parent: Uuid| -> Vec<(String, StoreKind, Uuid)> {
        let mut v: Vec<_> = tree
            .iter()
            .filter(|t| t.parent_id == Some(parent))
            .map(|t| (t.name.clone(), t.kind, t.id))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    };
    let top = under(gen2);
    assert_eq!(
        top.iter().map(|t| (t.0.as_str(), t.1)).collect::<Vec<_>>(),
        vec![("deep", StoreKind::Router), ("sea", StoreKind::Leaf)]
    );
    let deep = top[0].2;
    let below = under(deep);
    assert_eq!(below.len(), 1);
    assert_eq!(
        (below[0].0.as_str(), below[0].1),
        ("abyss", StoreKind::Leaf)
    );
    for id in [gen2, deep, top[1].2, below[0].2] {
        assert_eq!(embed_model(&s.db, id).await, MODEL_B);
    }

    // every chunk copied, re-embedded by B; the source untouched
    let target_ids: Vec<Uuid> = vec![top[1].2, below[0].2];
    let copied: Vec<(String, Vector)> =
        sqlx::query_as("SELECT text, embedding FROM chunks WHERE store_id = ANY($1) ORDER BY text")
            .bind(target_ids.as_slice())
            .fetch_all(&s.db.pool)
            .await
            .unwrap();
    assert_eq!(copied.len(), source_chunks.len());
    for ((ct, cv), (st_, sv)) in copied.iter().zip(&source_chunks) {
        assert_eq!(ct, st_);
        assert!(
            cosine(&cv.to_vec(), &reversed(ct)) > 0.9999,
            "B made {ct:?}"
        );
        assert!(
            cosine(&sv.to_vec(), &plain(st_)) > 0.9999,
            "the source kept A's"
        );
    }
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM chunks")
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    assert_eq!(n as usize, 2 * source_chunks.len());
    // documents copied with their labels and titles
    let docs: Vec<(String, Option<String>)> = sqlx::query_as(
        "SELECT source, title FROM documents WHERE store_id = ANY($1) ORDER BY source",
    )
    .bind(target_ids.as_slice())
    .fetch_all(&s.db.pool)
    .await
    .unwrap();
    assert_eq!(
        docs,
        vec![
            ("abyss.txt".to_string(), None),
            ("reef.txt".to_string(), None),
            ("sea.txt".to_string(), Some("Sea".to_string())),
        ]
    );
    // rollups: gen2 counts its copied content
    let (tokens, chunks): (i64, i64) =
        sqlx::query_as("SELECT token_count, chunk_count FROM stores WHERE id = $1")
            .bind(gen2)
            .fetch_one(&s.db.pool)
            .await
            .unwrap();
    let (src_tokens, src_chunks): (i64, i64) =
        sqlx::query_as("SELECT token_count, chunk_count FROM stores WHERE id = $1")
            .bind(main)
            .fetch_one(&s.db.pool)
            .await
            .unwrap();
    assert_eq!((tokens, chunks), (src_tokens, src_chunks));

    // the copy answers queries in B; the source still answers in A
    let hits = s.engine.query(tn(), &query(Some(gen2))).await.unwrap();
    assert!(hits.results.iter().any(|r| r.text.contains("harpoon")));
    let hits = s.engine.query(tn(), &query(Some(main))).await.unwrap();
    assert!(hits.results.iter().any(|r| r.text.contains("harpoon")));

    // the target is no longer empty: a second copy into it is refused
    let (st, v) = api(
        &s.app,
        "POST",
        &format!("/api/stores/{main}/reembed-copy"),
        TENANT,
        Some(json!({"target_store": gen2})),
    )
    .await;
    assert_eq!(st, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "invalid_target", "{v}");
}

#[tokio::test]
async fn reembed_copy_refuses_bad_targets() {
    let s = setup().await;
    let (main, sea) = main_tree(&s.engine).await;
    let post = |source: Uuid, body: Value| {
        let app = s.app.clone();
        async move {
            api(
                &app,
                "POST",
                &format!("/api/stores/{source}/reembed-copy"),
                TENANT,
                Some(body),
            )
            .await
        }
    };
    // itself, inside its own subtree, a kind mismatch, unknown fields
    let gen2 = gen2_root(&s).await;
    for (source, body, status, code) in [
        (main, json!({"target_store": main}), 422, "invalid_target"),
        (main, json!({"target_store": sea}), 422, "invalid_target"),
        (sea, json!({"target_store": gen2}), 422, "invalid_target"),
        (
            main,
            json!({"target_store": gen2, "x": 1}),
            422,
            "invalid_input",
        ),
        (
            main,
            json!({"target_store": Uuid::now_v7()}),
            404,
            "not_found",
        ),
    ] {
        let (st, v) = post(source, body.clone()).await;
        assert_eq!(st.as_u16(), status, "{body}: {v}");
        assert_eq!(v["error"]["code"], code, "{body}: {v}");
    }
    // a target whose model is not its resolved one (binding removed after the re-tag)
    let (st, _) = api(&s.app, "DELETE", &providers_uri(gen2), TENANT, None).await;
    assert_eq!(st, StatusCode::OK);
    let (st, v) = post(main, json!({"target_store": gen2})).await;
    assert_eq!(st, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "embed_model_mismatch", "{v}");
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM reembed_copy_jobs")
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    assert_eq!(n, 0, "no refused request stored a job");
}

#[tokio::test]
async fn reembed_copy_is_tenant_scoped() {
    let s = setup().await;
    let (main, _) = main_tree(&s.engine).await;
    let gen2 = gen2_root(&s).await;
    let theirs = s
        .engine
        .create_store(other(), new_store(None, StoreKind::Router, "theirs"))
        .await
        .unwrap()
        .id;
    // another tenant names neither our source nor our target, and cannot read our bindings
    for (source, target) in [(main, theirs), (theirs, gen2)] {
        let (st, v) = api(
            &s.app,
            "POST",
            &format!("/api/stores/{source}/reembed-copy"),
            OTHER_TENANT,
            Some(json!({"target_store": target})),
        )
        .await;
        assert_eq!(st, StatusCode::NOT_FOUND, "{v}");
    }
    let (st, _) = api(&s.app, "GET", &providers_uri(gen2), OTHER_TENANT, None).await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, _) = api(
        &s.app,
        "PUT",
        &providers_uri(gen2),
        OTHER_TENANT,
        Some(json!({})),
    )
    .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    assert_eq!(get_store(&s.app, gen2).await["embed"]["source"], "store");

    let (st, v) = api(
        &s.app,
        "POST",
        &format!("/api/stores/{main}/reembed-copy"),
        TENANT,
        Some(json!({"target_store": gen2})),
    )
    .await;
    assert_eq!(st, StatusCode::ACCEPTED, "{v}");
    let job = v["job"].as_str().unwrap().to_string();
    // the job is ours: another tenant, or another source path, is 404
    let (st, _) = api(
        &s.app,
        "GET",
        &format!("/api/stores/{main}/reembed-copy/{job}"),
        OTHER_TENANT,
        None,
    )
    .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    let (st, _) = api(
        &s.app,
        "GET",
        &format!("/api/stores/{gen2}/reembed-copy/{job}"),
        TENANT,
        None,
    )
    .await;
    assert_eq!(st, StatusCode::NOT_FOUND);
    assert_eq!(wait_done(&s.app, main, &job).await["state"], "done");
    // the other tenant's tree got nothing
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM stores WHERE tenant = $1")
        .bind(other().id())
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    assert_eq!(n, 1);
}

#[tokio::test]
async fn a_stale_running_job_reads_as_interrupted() {
    let s = setup().await;
    let (main, _) = main_tree(&s.engine).await;
    let gen2 = gen2_root(&s).await;
    let job = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO reembed_copy_jobs (id, tenant, source_store, target_store, embed_model, \
                                        state, chunks_total, heartbeat_at) \
         VALUES ($1, $2, $3, $4, $5, 'running', 3, now() - interval '5 minutes')",
    )
    .bind(job)
    .bind(tn().id())
    .bind(main)
    .bind(gen2)
    .bind(MODEL_B)
    .execute(&s.db.pool)
    .await
    .unwrap();
    let got = s.engine.reembed_copy_job(tn(), main, job).await.unwrap();
    assert_eq!(got.state, "failed");
    assert!(got.error.as_deref().unwrap().starts_with("interrupted"));
    assert!(got.finished_at.is_some());
    // and the target is free again for a new copy
    let started = s
        .engine
        .start_reembed_copy(tn(), main, gen2, false)
        .await
        .unwrap();
    assert_eq!(started.state, "running");
}

/// D90 over HTTP: `POST /api/query` and `/api/ask` without `root_id` work while the tenant has
/// one root, are 409 `ambiguous_root` once it has two, and work again naming either root.
#[tokio::test]
async fn query_without_root_id_is_ambiguous_once_there_are_two_roots() {
    let s = setup().await;
    let (main, _) = main_tree(&s.engine).await;
    let body = |root: Option<Uuid>| {
        let mut q = json!({"query": "white whale harpoon", "route_mode": "embed",
                           "min_score": -1.0});
        if let Some(r) = root {
            q["root_id"] = json!(r);
        }
        q
    };
    // one root, omitted
    let (st, v) = api(&s.app, "POST", "/api/query", TENANT, Some(body(None))).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert!(!v["results"].as_array().unwrap().is_empty(), "{v}");

    let gen2 = gen2_root(&s).await;
    // two roots, omitted: 409, for query and ask alike
    for uri in ["/api/query", "/api/ask"] {
        let (st, v) = api(&s.app, "POST", uri, TENANT, Some(body(None))).await;
        assert_eq!(st, StatusCode::CONFLICT, "{uri}: {v}");
        assert_eq!(v["error"]["code"], "ambiguous_root", "{uri}: {v}");
        assert!(
            v["error"]["message"].as_str().unwrap().contains("root_id"),
            "{v}"
        );
    }
    // two roots, named
    let (st, v) = api(&s.app, "POST", "/api/query", TENANT, Some(body(Some(main)))).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert!(!v["results"].as_array().unwrap().is_empty(), "{v}");
    let (st, v) = api(&s.app, "POST", "/api/query", TENANT, Some(body(Some(gen2)))).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    // the other tenant (one root of its own, none yet) is not affected by ours
    let (st, v) = api(&s.app, "POST", "/api/query", OTHER_TENANT, Some(body(None))).await;
    assert_eq!(st, StatusCode::OK, "{v}");
}

fn ask_body(root: Uuid) -> Value {
    json!({"query": "white whale harpoon", "route_mode": "embed", "min_score": -1.0,
           "root_id": root, "max_tokens": 16})
}

/// D92: chat resolves like embed / rerank — the store's own row, else the nearest ancestor's,
/// else the tenant's, else 409 provider_not_configured. A store chat key is sealed for its store
/// and never returned; the chat options are stored per store.
#[tokio::test]
async fn chat_resolves_store_then_ancestor_then_tenant() {
    const SECRET: &str = "sk-store-chat-0123456789";
    let s = setup().await;
    let (main, sea) = main_tree(&s.engine).await;
    // no chat anywhere: ask is 409 provider_not_configured naming chat
    let (st, v) = api(&s.app, "POST", "/api/ask", TENANT, Some(ask_body(main))).await;
    assert_eq!(st, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "provider_not_configured", "{v}");
    assert_eq!(
        get_store(&s.app, sea).await["chat"]["source"],
        "unconfigured"
    );

    // a store chat on main (keyed, model "B" answers from-b): main and everything below use it
    let keyed = mock(Some(SECRET), true).await;
    let mut reg = chat_reg(&keyed, Some(SECRET));
    reg["thinking"] = json!(false);
    reg["temperature"] = json!(0.2);
    let (st, v) = put_store(&s.app, main, json!({"chat": reg})).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["chat"]["source"], "store", "{v}");
    assert_eq!(v["chat"]["has_api_key"], true, "{v}");
    assert_eq!(v["chat"]["thinking"], false, "{v}");
    assert_eq!(v["chat"]["temperature"], 0.2, "{v}");
    assert_eq!(v["embed"]["source"], "tenant", "{v}");
    assert!(!v.to_string().contains(SECRET), "key leaked: {v}");
    let (cap, thinking, temperature, ct): (String, Option<bool>, Option<f64>, Vec<u8>) =
        sqlx::query_as(
            "SELECT capability, thinking, temperature, api_key_ct FROM store_providers \
             WHERE store = $1",
        )
        .bind(main)
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    assert_eq!(
        (cap.as_str(), thinking, temperature),
        ("chat", Some(false), Some(0.2))
    );
    assert!(!ct.windows(SECRET.len()).any(|w| w == SECRET.as_bytes()));
    let v = get_store(&s.app, sea).await;
    assert_eq!(v["chat"]["source"], "ancestor", "{v}");
    assert_eq!(v["chat"]["from_store"], json!(main), "{v}");
    let before = keyed.chats();
    let (st, v) = api(&s.app, "POST", "/api/ask", TENANT, Some(ask_body(main))).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["answer"], "from-b", "the store's chat answered: {v}");
    assert_eq!(keyed.chats(), before + 1);
    // from a leaf below: the ancestor's chat (the sealed key opened for main, not for sea)
    let (st, v) = api(&s.app, "POST", "/api/ask", TENANT, Some(ask_body(sea))).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["answer"], "from-b", "{v}");

    // the tenant registers chat A; a second root without a binding uses it, main keeps its own
    let (st, v) = api(
        &s.app,
        "PUT",
        "/api/tenant/providers",
        TENANT,
        Some(
            json!({"embed": embed_reg(&s.a, "acme/a", None), "rerank": rerank_reg(&s.a),
                    "chat": chat_reg(&s.a, None)}),
        ),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    let other_root = create(&s.engine, None, StoreKind::Router, "plain").await;
    assert_eq!(
        get_store(&s.app, other_root).await["chat"]["source"],
        "tenant"
    );
    let (st, v) = api(
        &s.app,
        "POST",
        "/api/ask",
        TENANT,
        Some(ask_body(other_root)),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["answer"], "from-a", "the tenant's chat answered: {v}");
    let (st, v) = api(&s.app, "POST", "/api/ask", TENANT, Some(ask_body(sea))).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(
        v["answer"], "from-b",
        "an ancestor binding beats the tenant: {v}"
    );
    // a binding on the leaf itself beats its ancestor's
    let (st, v) = put_store(&s.app, sea, json!({"chat": chat_reg(&s.a, None)})).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["chat"]["source"], "store", "{v}");
    let (st, v) = api(&s.app, "POST", "/api/ask", TENANT, Some(ask_body(sea))).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["answer"], "from-a", "{v}");

    // removing main's binding: main falls back to the tenant's chat
    let (st, v) = api(&s.app, "DELETE", &providers_uri(main), TENANT, None).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["chat"]["source"], "tenant", "{v}");
    let (st, v) = api(&s.app, "POST", "/api/ask", TENANT, Some(ask_body(main))).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["answer"], "from-a", "{v}");
    // chat options stay chat-only
    let mut bad = rerank_reg(&s.a);
    bad["thinking"] = json!(true);
    let (st, v) = put_store(&s.app, main, json!({"rerank": bad})).await;
    assert_eq!(st, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
}

const LATE: &str = "Sperm whales dive deep to hunt giant squid in the midnight zone of the ocean.";
const BAY: &str = "Seals bask on the rocks of the quiet bay while gulls circle overhead.";

/// D91: documents ingested into the source while the copy embeds its snapshot (into an existing
/// leaf, and into a leaf created meanwhile) end up in the target; a document deleted from the
/// source meanwhile does not (the target mirrors the source as of the copy's commit).
#[tokio::test]
async fn reembed_copy_takes_documents_ingested_mid_copy() {
    let s = setup().await;
    let (main, sea) = main_tree(&s.engine).await;
    let gen2 = gen2_root(&s).await;
    let reef: Uuid = sqlx::query_scalar("SELECT id FROM documents WHERE source = 'reef.txt'")
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    let (reached, release) = s.b.arm_hold();
    let (st, v) = api(
        &s.app,
        "POST",
        &format!("/api/stores/{main}/reembed-copy"),
        TENANT,
        Some(json!({"target_store": gen2})),
    )
    .await;
    assert_eq!(st, StatusCode::ACCEPTED, "{v}");
    let job = v["job"].as_str().unwrap().to_string();
    // the snapshot's chunks are being embedded by B, held: the source changes under the copy
    tokio::time::timeout(std::time::Duration::from_secs(30), reached)
        .await
        .expect("the copy reached the snapshot embed")
        .unwrap();
    s.engine
        .ingest(tn(), sea, "late.txt".into(), None, LATE)
        .await
        .unwrap();
    let bay = create(&s.engine, Some(main), StoreKind::Leaf, "bay").await;
    s.engine
        .ingest(tn(), bay, "bay.txt".into(), None, BAY)
        .await
        .unwrap();
    s.engine.delete_document(tn(), reef).await.unwrap();
    release.send(()).unwrap();

    let done = wait_done(&s.app, main, &job).await;
    assert_eq!(done["state"], "done", "{done}");
    // the target mirrors the source as it is now: bay included, reef gone
    let target_docs: Vec<(String, String)> = sqlx::query_as(
        "SELECT st.name, d.source FROM documents d JOIN stores st ON st.id = d.store_id \
         WHERE st.tenant = $1 AND d.store_id IN ( \
             WITH RECURSIVE sub(id) AS (SELECT $2::uuid UNION ALL \
                 SELECT s.id FROM stores s JOIN sub ON s.parent_id = sub.id) \
             SELECT id FROM sub) \
         ORDER BY d.source",
    )
    .bind(tn().id())
    .bind(gen2)
    .fetch_all(&s.db.pool)
    .await
    .unwrap();
    assert_eq!(
        target_docs,
        vec![
            ("abyss".to_string(), "abyss.txt".to_string()),
            ("bay".to_string(), "bay.txt".to_string()),
            ("sea".to_string(), "late.txt".to_string()),
            ("sea".to_string(), "sea.txt".to_string()),
        ]
    );
    let source_docs: Vec<String> = sqlx::query_scalar(
        "SELECT d.source FROM documents d JOIN stores st ON st.id = d.store_id \
         WHERE st.tenant = $1 AND st.embed_model = $2 ORDER BY d.source",
    )
    .bind(tn().id())
    .bind(MODEL_A)
    .fetch_all(&s.db.pool)
    .await
    .unwrap();
    assert_eq!(
        source_docs,
        vec!["abyss.txt", "bay.txt", "late.txt", "sea.txt"]
    );
    // every copied chunk was embedded by B, the late ones included; the job counts what it wrote
    let copied: Vec<(String, Vector)> = sqlx::query_as(
        "SELECT c.text, c.embedding FROM chunks c JOIN stores st ON st.id = c.store_id \
         WHERE st.tenant = $1 AND st.embed_model = $2 ORDER BY c.text",
    )
    .bind(tn().id())
    .bind(MODEL_B)
    .fetch_all(&s.db.pool)
    .await
    .unwrap();
    assert!(copied.iter().any(|(t, _)| t.contains("giant squid")));
    assert!(copied.iter().any(|(t, _)| t.contains("Seals bask")));
    assert!(!copied.iter().any(|(t, _)| t.contains("clownfish")));
    for (t, v) in &copied {
        assert!(cosine(&v.to_vec(), &reversed(t)) > 0.9999, "B made {t:?}");
    }
    assert_eq!(done["chunks_total"], copied.len(), "{done}");
    assert_eq!(done["chunks_done"], copied.len(), "{done}");
}

/// `copy_vectors` with a target in a DIFFERENT model is 409 `embed_model_mismatch`, and stores no
/// job: copied vectors only mean something in the model that made them.
#[tokio::test]
async fn copy_vectors_refuses_a_target_in_another_model() {
    let s = setup().await;
    let (main, _) = main_tree(&s.engine).await;
    let gen2 = gen2_root(&s).await;
    let (st, v) = api(
        &s.app,
        "POST",
        &format!("/api/stores/{main}/reembed-copy"),
        TENANT,
        Some(json!({"target_store": gen2, "copy_vectors": true})),
    )
    .await;
    assert_eq!(st, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "embed_model_mismatch", "{v}");
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM reembed_copy_jobs")
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
    assert_eq!(n, 0);
}

/// `copy_vectors` with a target in the SAME model (here: a new root that binds only its own
/// rerank provider) copies the source's vectors without one embed call; a document ingested
/// while the copy waits for its final transaction is copied by the delta pass the same way.
#[tokio::test]
async fn copy_vectors_copies_the_source_vectors_without_embedding() {
    let s = setup().await;
    let (main, sea) = main_tree(&s.engine).await;
    let draft = create(&s.engine, None, StoreKind::Router, "draft").await;
    let (st, v) = put_store(&s.app, draft, json!({"rerank": rerank_reg(&s.b)})).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["embed_model"], MODEL_A, "{v}");

    // hold the target's row: the copy's final transaction waits on it after its first pass
    let mut hold = s.db.pool.begin().await.unwrap();
    sqlx::query("SELECT 1 FROM stores WHERE id = $1 FOR UPDATE")
        .bind(draft)
        .execute(&mut *hold)
        .await
        .unwrap();
    let (a_before, b_before) = (s.a.texts(), s.b.texts());
    let (st, v) = api(
        &s.app,
        "POST",
        &format!("/api/stores/{main}/reembed-copy"),
        TENANT,
        Some(json!({"target_store": draft, "copy_vectors": true})),
    )
    .await;
    assert_eq!(st, StatusCode::ACCEPTED, "{v}");
    assert_eq!(v["copy_vectors"], true, "{v}");
    assert_eq!(v["embed_model"], MODEL_A, "{v}");
    let job = v["job"].as_str().unwrap().to_string();
    let mut waiting = false;
    for _ in 0..200 {
        let n: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM pg_stat_activity \
             WHERE wait_event_type = 'Lock' AND datname = current_database()",
        )
        .fetch_one(&s.db.pool)
        .await
        .unwrap();
        if n > 0 {
            waiting = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(waiting, "the copy reached its final transaction");
    s.engine
        .ingest(tn(), sea, "late.txt".into(), None, LATE)
        .await
        .unwrap();
    let a_after_ingest = s.a.texts();
    assert!(a_after_ingest > a_before, "the ingest itself embeds with A");
    hold.commit().await.unwrap();

    let done = wait_done(&s.app, main, &job).await;
    assert_eq!(done["state"], "done", "{done}");
    assert_eq!(done["copy_vectors"], true, "{done}");
    assert_eq!(
        s.a.texts(),
        a_after_ingest,
        "the copy embedded nothing with A"
    );
    assert_eq!(s.b.texts(), b_before, "nor with B");
    // the target's chunks carry exactly the source's vectors, the late document's included
    let pairs: Vec<(String, Vector, Vector)> = sqlx::query_as(
        "WITH RECURSIVE tgt(id) AS (SELECT $1::uuid UNION ALL \
             SELECT s.id FROM stores s JOIN tgt ON s.parent_id = tgt.id), \
         src(id) AS (SELECT $2::uuid UNION ALL \
             SELECT s.id FROM stores s JOIN src ON s.parent_id = src.id) \
         SELECT t.text, t.embedding, o.embedding FROM chunks t \
         JOIN chunks o ON o.text = t.text AND o.store_id IN (SELECT id FROM src) \
         WHERE t.store_id IN (SELECT id FROM tgt) ORDER BY t.text",
    )
    .bind(draft)
    .bind(main)
    .fetch_all(&s.db.pool)
    .await
    .unwrap();
    let source_chunks: i64 = sqlx::query_scalar(
        "WITH RECURSIVE src(id) AS (SELECT $1::uuid UNION ALL \
             SELECT s.id FROM stores s JOIN src ON s.parent_id = src.id) \
         SELECT count(*) FROM chunks WHERE store_id IN (SELECT id FROM src)",
    )
    .bind(main)
    .fetch_one(&s.db.pool)
    .await
    .unwrap();
    assert!(source_chunks >= 4, "{source_chunks}");
    assert_eq!(pairs.len() as i64, source_chunks, "every chunk copied once");
    assert!(pairs.iter().any(|(t, _, _)| t.contains("giant squid")));
    for (t, copied, original) in &pairs {
        assert_eq!(copied.to_vec(), original.to_vec(), "{t:?}");
    }
    assert_eq!(done["chunks_total"], source_chunks, "{done}");
    // the copy answers queries from its own root
    let hits = s.engine.query(tn(), &query(Some(draft))).await.unwrap();
    assert!(hits.results.iter().any(|r| r.text.contains("harpoon")));
}
