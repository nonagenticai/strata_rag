//! Tenant-registered providers (spec §4.1): `PUT/GET /api/tenant/providers`, registration-time
//! validation + live probe, keys encrypted at rest and never returned, per-request resolution
//! (tenant → dev default → 409 provider_not_configured) with a cache that follows every PUT.
//! Providers are mock axum servers speaking the OpenAI `/embeddings` and Cohere `/rerank` shapes.

mod common;

use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use common::{OTHER_TENANT, TENANT, TestDb, other, test_config, test_key, tn};
use serde_json::{Value, json};
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::inference::{Embedder, FakeEmbedder, FakeReranker, Reranker};
use strata_rag::providers::{Deployment, KeyCipher, LaneLimits, Providers};
use strata_rag::query::QueryRequest;
use tower::ServiceExt;
use uuid::Uuid;

/// A mock provider: counts calls, records the Authorization header of each, and can be told to
/// answer another status, other dims, or vectors that are not unit length.
#[derive(Default)]
struct Mock {
    /// `Some(k)`: anything but `Bearer k` is 401.
    key: Option<String>,
    embeds: AtomicUsize,
    reranks: AtomicUsize,
    /// 0 = answer normally.
    fail_status: AtomicU16,
    /// 0 = 1024.
    dims: AtomicUsize,
    /// Multiply every vector by this many (1 = unit vectors).
    scale: AtomicUsize,
    auth: Mutex<Vec<Option<String>>>,
    chats: AtomicUsize,
    /// Every chat request body, in order.
    chat_bodies: Mutex<Vec<Value>>,
}

/// One reply serves both chat uses: expansion parses it, ask returns it as the answer.
const CHAT_REPLY: &str =
    r#"{"hyde_passage": "The harpoon struck the white whale.", "rewrites": []}"#;

struct MockServer {
    url: String,
    m: Arc<Mock>,
}

impl MockServer {
    fn embeds(&self) -> usize {
        self.m.embeds.load(Ordering::SeqCst)
    }
    fn reranks(&self) -> usize {
        self.m.reranks.load(Ordering::SeqCst)
    }
    fn chats(&self) -> usize {
        self.m.chats.load(Ordering::SeqCst)
    }
    fn auth(&self) -> Vec<Option<String>> {
        self.m.auth.lock().unwrap().clone()
    }
}

fn gate(m: &Mock, headers: &HeaderMap) -> Option<Response> {
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    m.auth.lock().unwrap().push(auth.clone());
    if let Some(k) = &m.key
        && auth.as_deref() != Some(format!("Bearer {k}").as_str())
    {
        return Some(StatusCode::UNAUTHORIZED.into_response());
    }
    match m.fail_status.load(Ordering::SeqCst) {
        0 => None,
        s => Some(StatusCode::from_u16(s).unwrap().into_response()),
    }
}

async fn embeddings(
    State(m): State<Arc<Mock>>,
    headers: HeaderMap,
    Json(b): Json<Value>,
) -> Response {
    m.embeds.fetch_add(1, Ordering::SeqCst);
    if let Some(r) = gate(&m, &headers) {
        return r;
    }
    let texts: Vec<String> = serde_json::from_value(b["input"].clone()).unwrap();
    let dims = match m.dims.load(Ordering::SeqCst) {
        0 => 1024,
        d => d,
    };
    let scale = m.scale.load(Ordering::SeqCst).max(1) as f32;
    let vecs = FakeEmbedder.embed(&texts).unwrap();
    let data: Vec<Value> = vecs
        .into_iter()
        .enumerate()
        .map(|(i, mut v)| {
            v.resize(dims, 0.0);
            if dims != 1024 {
                v[0] = 1.0;
            }
            let v: Vec<f32> = v.into_iter().map(|x| x * scale).collect();
            json!({"index": i, "embedding": v})
        })
        .collect();
    Json(json!({"data": data})).into_response()
}

async fn rerank(State(m): State<Arc<Mock>>, headers: HeaderMap, Json(b): Json<Value>) -> Response {
    m.reranks.fetch_add(1, Ordering::SeqCst);
    if let Some(r) = gate(&m, &headers) {
        return r;
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

async fn chat(State(m): State<Arc<Mock>>, headers: HeaderMap, Json(b): Json<Value>) -> Response {
    m.chats.fetch_add(1, Ordering::SeqCst);
    m.chat_bodies.lock().unwrap().push(b);
    if let Some(r) = gate(&m, &headers) {
        return r;
    }
    Json(
        json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": CHAT_REPLY}}]}),
    )
    .into_response()
}

async fn mock(key: Option<&str>) -> MockServer {
    let m = Arc::new(Mock {
        key: key.map(str::to_string),
        ..Mock::default()
    });
    let app = Router::new()
        .route("/embeddings", post(embeddings))
        .route("/rerank", post(rerank))
        .route("/chat/completions", post(chat))
        .route("/models", axum::routing::get(|| async { "{}" }))
        .with_state(m.clone());
    MockServer {
        url: common::mock::spawn(app).await,
        m,
    }
}

/// A production deployment: no defaults, the SSRF guard on.
const PROD_EU: &str = "region = \"eu\"\nmode = \"production\"\n\
                       [policy]\nallowed_dialects = [\"openai\", \"cohere\"]\n";

/// No defaults (resolves like production), but tenants may register the loopback mock servers:
/// `allow_internal_provider_hosts` is a dev-only flag.
const LOCAL_EU: &str = "region = \"eu\"\nmode = \"dev\"\n\
                        [policy]\nallowed_dialects = [\"openai\", \"cohere\"]\n\
                        allow_internal_provider_hosts = true\n";

/// Dev with defaults on the (loopback) `default` mock; `allow` = tenants may register internal hosts.
fn dev_eu(default: &MockServer, allow: bool) -> String {
    format!(
        "region = \"eu\"\nmode = \"dev\"\n[policy]\nallowed_dialects = [\"openai\", \"cohere\"]\n\
         allow_internal_provider_hosts = {allow}\n\
         [default.embed]\nbase_url = \"{u}\"\ndialect = \"openai\"\nmodel = \"BAAI/bge-m3\"\n\
         dims = 1024\nregions = [\"eu\"]\nzdr = true\n\
         [default.rerank]\nbase_url = \"{u}\"\ndialect = \"cohere\"\nmodel = \"rr\"\n\
         regions = [\"eu\"]\nzdr = true\n",
        u = default.url,
        allow = allow
    )
}

/// `dev_eu` (tenants may register internal hosts) plus a `[default.chat]` on the same mock.
fn dev_eu_with_chat(default: &MockServer) -> String {
    format!(
        "{}[default.chat]\nbase_url = \"{}\"\ndialect = \"openai\"\nmodel = \"local-llm\"\n\
         regions = [\"eu\"]\nzdr = true\n",
        dev_eu(default, true),
        default.url
    )
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

async fn registry(db: &TestDb, toml: &str, cipher: bool) -> Providers {
    let dep = Deployment::parse(toml, &Default::default()).expect("deployment");
    Providers::registry(
        db.pool.clone(),
        dep,
        cipher.then(|| KeyCipher::new(&test_key())),
        LaneLimits::default(),
    )
    .await
    .expect("registry")
}

async fn engine_on(db: &TestDb, toml: &str) -> Arc<Engine> {
    Arc::new(Engine::with_source(
        db.pool.clone(),
        registry(db, toml, true).await,
        Arc::new(strata_rag::text::WhitespaceCounter),
        test_config(),
    ))
}

async fn setup(toml: &str) -> (Arc<Engine>, Router, TestDb) {
    let db = TestDb::start().await;
    let engine = engine_on(&db, toml).await;
    let app = strata_rag::api::router(engine.clone());
    (engine, app, db)
}

async fn call(
    app: &Router,
    method: &str,
    tenant: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri("/api/tenant/providers")
        .header("x-tenant-id", tenant)
        .header("content-type", "application/json")
        .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn put(app: &Router, tenant: &str, body: Value) -> (StatusCode, Value) {
    call(app, "PUT", tenant, Some(body)).await
}

async fn get(app: &Router, tenant: &str) -> Value {
    let (s, v) = call(app, "GET", tenant, None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v
}

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

fn rerank_reg(m: &MockServer, key: Option<&str>) -> Value {
    let mut v = json!({
        "base_url": m.url, "dialect": "cohere", "model": "rr", "regions": ["eu"], "zdr": true
    });
    if let Some(k) = key {
        v["api_key"] = json!(k);
    }
    v
}

fn store(parent_id: Option<Uuid>, kind: StoreKind, name: &str) -> NewStore {
    NewStore {
        parent_id,
        kind,
        name: name.to_string(),
        description: format!("{name} holds documents about whales and ships"),
    }
}

/// root → leaf with one document, as `tenant`.
async fn tree(engine: &Engine, tenant: strata_rag::Tenant) -> (Uuid, Uuid) {
    let root = engine
        .create_store(tenant, store(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    let leaf = engine
        .create_store(tenant, store(Some(root), StoreKind::Leaf, "sea"))
        .await
        .unwrap()
        .id;
    engine
        .ingest(
            tenant,
            leaf,
            "sea.txt".into(),
            None,
            "The harpoon struck the white whale.",
        )
        .await
        .unwrap();
    (root, leaf)
}

fn query() -> QueryRequest {
    serde_json::from_value(json!({"query": "white whale harpoon", "route_mode": "embed"})).unwrap()
}

#[tokio::test]
async fn a_registration_is_stored_encrypted_and_never_returned() {
    const SECRET: &str = "sk-tenant-a-0123456789";
    let a = mock(Some(SECRET)).await;
    let (_engine, app, db) = setup(LOCAL_EU).await;

    let (s, v) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&a, "acme/embed", Some(SECRET)), "rerank": rerank_reg(&a, Some(SECRET))}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let got = get(&app, TENANT).await;
    assert_eq!(v, got, "PUT answers the stored view");
    for cap in ["embed", "rerank"] {
        assert_eq!(got[cap]["source"], "tenant", "{got}");
        assert_eq!(got[cap]["has_api_key"], true, "{got}");
        assert_eq!(got[cap]["base_url"], a.url.as_str(), "{got}");
        assert!(got[cap]["updated_at"].is_string(), "{got}");
        assert!(got[cap].get("api_key").is_none(), "{got}");
    }
    assert_eq!(got["embed"]["model"], "acme/embed");
    assert_eq!(got["embed"]["dims"], 1024);
    assert!(!got.to_string().contains(SECRET), "key leaked: {got}");
    // the probes sent the key
    assert!(
        a.auth()
            .iter()
            .all(|h| h.as_deref() == Some(&format!("Bearer {SECRET}")[..]))
    );
    assert_eq!((a.embeds(), a.reranks()), (1, 1));

    // at rest: ciphertext + nonce + key version, never the key bytes in any column
    let rows: Vec<(Vec<u8>, Vec<u8>, i32, String)> = sqlx::query_as(
        "SELECT api_key_ct, api_key_nonce, key_version, t::text FROM tenant_providers t \
         WHERE tenant = $1 ORDER BY capability",
    )
    .bind(tn().id())
    .fetch_all(&db.pool)
    .await
    .unwrap();
    assert_eq!(rows.len(), 2);
    for (ct, nonce, version, text) in &rows {
        assert_eq!(nonce.len(), 12);
        assert_eq!(*version, 1);
        assert!(!ct.windows(SECRET.len()).any(|w| w == SECRET.as_bytes()));
        assert!(!text.contains(SECRET), "{text}");
    }
    assert_ne!(rows[0].0, rows[1].0, "one nonce per write");
}

#[tokio::test]
async fn each_invalid_registration_is_422_with_its_code_and_nothing_is_stored() {
    let ok = mock(None).await;
    let keyed = mock(Some("sk-needed")).await;
    let short = mock(None).await;
    short.m.dims.store(768, Ordering::SeqCst);
    let long = mock(None).await;
    long.m.scale.store(3, Ordering::SeqCst);
    let broken = mock(None).await;
    broken.m.fail_status.store(500, Ordering::SeqCst);
    let (_engine, app, _db) = setup(LOCAL_EU).await;

    let with = |m: &MockServer, f: &dyn Fn(&mut Value)| {
        let mut v = embed_reg(m, "acme/embed", None);
        f(&mut v);
        json!({ "embed": v })
    };
    let cases: Vec<(Value, &str)> = vec![
        (
            with(&ok, &|v| v["regions"] = json!(["us"])),
            "region_not_allowed",
        ),
        (with(&ok, &|v| v["zdr"] = json!(false)), "zdr_required"),
        (with(&ok, &|v| v["dims"] = json!(768)), "dims_must_be_1024"),
        (
            with(&ok, &|v| v["dialect"] = json!("voyage")),
            "dialect_not_allowed",
        ),
        (
            with(&ok, &|v| v["dialect"] = json!("cohere")),
            "dialect_not_allowed",
        ),
        (
            json!({"rerank": {"base_url": ok.url, "dialect": "openai", "model": "rr",
                              "regions": ["eu"], "zdr": true}}),
            "dialect_not_allowed",
        ),
        (
            with(&ok, &|v| v["base_url"] = json!("ftp://x")),
            "invalid_provider",
        ),
        // the mock wants a key: 401 on the probe
        (with(&keyed, &|_| {}), "probe_failed"),
        (
            with(&keyed, &|v| v["api_key"] = json!("sk-wrong")),
            "probe_failed",
        ),
        // a 768-d provider registered as 1024-d
        (with(&short, &|_| {}), "probe_failed"),
        // vectors that are not L2-normalised
        (with(&long, &|_| {}), "probe_failed"),
        (with(&broken, &|_| {}), "probe_failed"),
        (
            json!({"rerank": {"base_url": keyed.url, "dialect": "cohere", "model": "rr",
                              "regions": ["eu"], "zdr": true}}),
            "probe_failed",
        ),
    ];
    for (body, code) in cases {
        let (s, v) = put(&app, TENANT, body.clone()).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{body} -> {v}");
        assert_eq!(v["error"]["code"], code, "{body} -> {v}");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(!msg.contains("sk-"), "key in message: {msg}");
        assert!(!msg.contains("127.0.0.1"), "URL in message: {msg}");
    }
    let (_, v) = put(&app, TENANT, with(&keyed, &|_| {})).await;
    assert!(
        v["error"]["message"].as_str().unwrap().contains("auth"),
        "{v}"
    );
    let (_, v) = put(&app, TENANT, with(&short, &|_| {})).await;
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid_response"),
        "{v}"
    );
    let got = get(&app, TENANT).await;
    assert_eq!(got["embed"]["source"], "unconfigured", "{got}");
    assert_eq!(got["rerank"]["source"], "unconfigured", "{got}");
}

#[tokio::test]
async fn an_omitted_key_is_kept_for_the_same_base_url_only() {
    const SECRET: &str = "sk-keep-me-42";
    let a = mock(Some(SECRET)).await;
    let b = mock(Some(SECRET)).await;
    let (_engine, app, _db) = setup(LOCAL_EU).await;

    let (s, v) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&a, "m", Some(SECRET))}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    // same base_url, no key: the stored key is reused (the probe still authenticates)
    let mut again = embed_reg(&a, "m", None);
    again["concurrency"] = json!(2);
    let (s, v) = put(&app, TENANT, json!({"embed": again})).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["embed"]["has_api_key"], true, "{v}");
    assert_eq!(v["embed"]["concurrency"], 2, "{v}");
    assert_eq!(a.embeds(), 2);
    assert!(
        a.auth()
            .iter()
            .all(|h| h.as_deref() == Some(&format!("Bearer {SECRET}")[..]))
    );

    // another base_url without a key: the stored key never goes to a new host
    let (s, v) = put(&app, TENANT, json!({"embed": embed_reg(&b, "m", None)})).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "api_key_required", "{v}");
    assert_eq!(b.embeds(), 0, "no call to the new host");
    assert_eq!(get(&app, TENANT).await["embed"]["base_url"], a.url.as_str());
    // ... with the key it is accepted
    let (s, v) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&b, "m", Some(SECRET))}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["embed"]["base_url"], b.url.as_str());

    // a capability left out (or null) is removed
    let (s, v) = put(&app, TENANT, json!({"embed": null})).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["embed"]["source"], "unconfigured", "{v}");
}

#[tokio::test]
async fn a_tenant_timeout_is_capped() {
    let a = mock(None).await;
    let (_engine, app, _db) = setup(LOCAL_EU).await;
    let mut e = embed_reg(&a, "m", None);
    e["timeout_ms"] = json!(30_001);
    let (s, v) = put(&app, TENANT, json!({ "embed": e })).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "invalid_provider", "{v}");
    let mut r = rerank_reg(&a, None);
    r["timeout_ms"] = json!(15_001);
    let (s, v) = put(&app, TENANT, json!({ "rerank": r })).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    let mut r = rerank_reg(&a, None);
    r["timeout_ms"] = json!(15_000);
    let (s, v) = put(&app, TENANT, json!({ "rerank": r })).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(a.embeds(), 0, "refused before any probe");
}

#[tokio::test]
async fn a_tenant_cannot_see_or_overwrite_another_tenants_providers() {
    let a = mock(Some("sk-a")).await;
    let b = mock(None).await;
    let (engine, app, db) = setup(LOCAL_EU).await;

    let (s, _) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&a, "m-a", Some("sk-a"))}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let theirs = get(&app, OTHER_TENANT).await;
    assert_eq!(theirs["embed"]["source"], "unconfigured", "{theirs}");
    assert!(!theirs.to_string().contains(&a.url), "{theirs}");

    // B registers its own and removes "its" rerank: A's rows are untouched
    let (s, _) = put(
        &app,
        OTHER_TENANT,
        json!({"embed": embed_reg(&b, "m-b", None)}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let mine = get(&app, TENANT).await;
    assert_eq!(mine["embed"]["model"], "m-a", "{mine}");
    assert_eq!(mine["embed"]["has_api_key"], true, "{mine}");
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM tenant_providers")
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(n, 2);

    // and each resolves its own provider
    let before = (a.embeds(), b.embeds());
    engine
        .create_store(other(), store(None, StoreKind::Router, "root"))
        .await
        .unwrap();
    assert_eq!((a.embeds(), b.embeds()), (before.0, before.1 + 1));
}

#[tokio::test]
async fn without_a_registration_or_default_it_is_409_provider_not_configured() {
    let a = mock(None).await;
    let (engine, app, _db) = setup(LOCAL_EU).await;

    let err = engine
        .create_store(tn(), store(None, StoreKind::Router, "root"))
        .await
        .unwrap_err();
    assert_eq!((err.code(), err.status()), ("provider_not_configured", 409));
    assert!(err.to_string().contains("embed"), "{err}");
    // through the API
    let req = Request::builder()
        .method("POST")
        .uri("/api/stores")
        .header("x-tenant-id", TENANT)
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"kind": "router", "name": "root", "description": "d"}).to_string(),
        ))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    let v: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(v["error"]["code"], "provider_not_configured", "{v}");

    // embed registered, rerank not: stores and ingest work, a query that reranks is 409 (rerank)
    let (s, _) = put(&app, TENANT, json!({"embed": embed_reg(&a, "m", None)})).await;
    assert_eq!(s, StatusCode::OK);
    tree(&engine, tn()).await;
    let err = engine.query(tn(), &query()).await.unwrap_err();
    assert_eq!(err.code(), "provider_not_configured", "{err:?}");
    assert!(err.to_string().contains("rerank"), "{err}");
    let (s, _) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&a, "m", None), "rerank": rerank_reg(&a, None)}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(
        !engine
            .query(tn(), &query())
            .await
            .unwrap()
            .results
            .is_empty()
    );
}

#[tokio::test]
async fn dev_resolves_the_tenant_first_then_the_default() {
    let default = mock(None).await;
    let mine = mock(None).await;
    let (engine, app, _db) = setup(&dev_eu(&default, true)).await;

    // unregistered tenant: the dev default, shown as such
    let v = get(&app, TENANT).await;
    // a default says what it is, never where
    assert_eq!(
        v["embed"],
        json!({"source": "default", "dialect": "openai", "model": "BAAI/bge-m3", "dims": 1024}),
        "{v}"
    );
    assert_eq!(
        v["rerank"],
        json!({"source": "default", "dialect": "cohere", "model": "rr", "dims": null}),
        "{v}"
    );
    tree(&engine, tn()).await;
    assert!(
        !engine
            .query(tn(), &query())
            .await
            .unwrap()
            .results
            .is_empty()
    );
    assert!(default.embeds() > 0 && default.reranks() > 0);

    // the tenant registers the same model (same store identity): its provider is used, not ours
    let (s, v) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&mine, "BAAI/bge-m3", None), "rerank": rerank_reg(&mine, None)}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["rerank"]["source"], "tenant", "{v}");
    let (d, m) = (
        (default.embeds(), default.reranks()),
        (mine.embeds(), mine.reranks()),
    );
    assert!(
        !engine
            .query(tn(), &query())
            .await
            .unwrap()
            .results
            .is_empty()
    );
    assert_eq!(
        (default.embeds(), default.reranks()),
        d,
        "default untouched"
    );
    assert!(
        mine.embeds() > m.0 && mine.reranks() > m.1,
        "tenant provider used"
    );
}

#[tokio::test]
async fn a_put_switches_the_cached_provider_on_every_replica() {
    let first = mock(None).await;
    let second = mock(None).await;
    let (engine, app, db) = setup(LOCAL_EU).await;
    // a second replica on the same database, with its own cache
    let replica = engine_on(&db, LOCAL_EU).await;

    let reg = |m: &MockServer| json!({"embed": embed_reg(m, "acme/embed", None), "rerank": rerank_reg(m, None)});
    let (s, _) = put(&app, TENANT, reg(&first)).await;
    assert_eq!(s, StatusCode::OK);
    tree(&engine, tn()).await;
    engine.query(tn(), &query()).await.unwrap();
    replica.query(tn(), &query()).await.unwrap(); // both replicas now cache `first`

    let (s, _) = put(&app, TENANT, reg(&second)).await;
    assert_eq!(s, StatusCode::OK);
    let f = (first.embeds(), first.reranks());
    let s0 = (second.embeds(), second.reranks());
    engine.query(tn(), &query()).await.unwrap();
    replica.query(tn(), &query()).await.unwrap();
    assert_eq!(
        (first.embeds(), first.reranks()),
        f,
        "no call to the old provider"
    );
    assert_eq!(second.embeds(), s0.0 + 2, "one query embed per replica");
    assert_eq!(second.reranks(), s0.1 + 2);

    // removing the registration: the next request has no provider (production)
    let (s, _) = put(&app, TENANT, json!({})).await;
    assert_eq!(s, StatusCode::OK);
    let err = replica.query(tn(), &query()).await.unwrap_err();
    assert_eq!(err.code(), "provider_not_configured");
}

#[tokio::test]
async fn changing_the_embed_model_makes_existing_stores_409() {
    let a = mock(None).await;
    let (engine, app, _db) = setup(LOCAL_EU).await;
    let reg = |model: &str, id: Option<&str>| {
        let mut e = embed_reg(&a, model, None);
        if let Some(id) = id {
            e["model_id"] = json!(id);
        }
        json!({"embed": e, "rerank": rerank_reg(&a, None)})
    };
    let (s, _) = put(&app, TENANT, reg("acme/a", None)).await;
    assert_eq!(s, StatusCode::OK);
    let (_root, leaf) = tree(&engine, tn()).await;

    let (s, _) = put(&app, TENANT, reg("acme/b", None)).await;
    assert_eq!(s, StatusCode::OK);
    let err = engine.query(tn(), &query()).await.unwrap_err();
    assert_eq!(
        (err.code(), err.status()),
        ("embed_model_mismatch", 409),
        "{err:?}"
    );
    assert!(err.to_string().contains("acme/a@1024") && err.to_string().contains("acme/b@1024"));
    let err = engine
        .ingest(tn(), leaf, "more.txt".into(), None, "Another whale story.")
        .await
        .unwrap_err();
    assert_eq!(err.code(), "embed_model_mismatch");

    // the same weights under another served name: model_id keeps the identity
    let (s, _) = put(&app, TENANT, reg("acme/b", Some("acme/a@1024"))).await;
    assert_eq!(s, StatusCode::OK);
    assert!(
        !engine
            .query(tn(), &query())
            .await
            .unwrap()
            .results
            .is_empty()
    );
}

#[tokio::test]
async fn health_reports_the_defaults_only() {
    let default = mock(None).await;
    let tenant_mock = mock(None).await;
    let (engine, app, _db) = setup(LOCAL_EU).await;
    let (s, _) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&tenant_mock, "m", None), "rerank": rerank_reg(&tenant_mock, None)}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let calls = (tenant_mock.embeds(), tenant_mock.reranks());
    let h = engine.health().await;
    assert_eq!((h.embed, h.rerank), ("unconfigured", "unconfigured"));
    assert_eq!(
        (tenant_mock.embeds(), tenant_mock.reranks()),
        calls,
        "no tenant probe"
    );

    // operator-set defaults are exempt from the SSRF guard: loopback works with the flag off
    let (dev, _app, _db2) = setup(&dev_eu(&default, false)).await;
    let h = dev.health().await;
    assert_eq!((h.embed, h.rerank), ("ok", "ok"));
}

#[tokio::test]
async fn keys_need_strata_provider_key() {
    const SECRET: &str = "sk-sealed";
    let a = mock(Some(SECRET)).await;
    let db = TestDb::start().await;
    let no_key = Arc::new(Engine::with_source(
        db.pool.clone(),
        registry(&db, LOCAL_EU, false).await,
        Arc::new(strata_rag::text::WhitespaceCounter),
        test_config(),
    ));
    let app = strata_rag::api::router(no_key);
    let (s, v) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&a, "m", Some(SECRET))}),
    )
    .await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
    assert_eq!(v["error"]["code"], "provider_key_unconfigured", "{v}");

    // a stored key makes a process without STRATA_PROVIDER_KEY refuse to start
    let with_key = engine_on(&db, LOCAL_EU).await;
    let app = strata_rag::api::router(with_key);
    let (s, _) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&a, "m", Some(SECRET))}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let dep = Deployment::parse(LOCAL_EU, &Default::default()).unwrap();
    let err = match Providers::registry(db.pool.clone(), dep, None, LaneLimits::default()).await {
        Ok(_) => panic!("must refuse without STRATA_PROVIDER_KEY"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("STRATA_PROVIDER_KEY"), "{err}");
    // and a different key cannot open it: resolving fails (500), never a keyless call
    let dep = Deployment::parse(LOCAL_EU, &Default::default()).unwrap();
    let wrong = Providers::registry(
        db.pool.clone(),
        dep,
        Some(KeyCipher::new(&test_key())),
        LaneLimits::default(),
    )
    .await
    .unwrap();
    // Resolution is per lane: the undecryptable embed row faults only the embed lane, and using
    // it fails `internal` — never a keyless call to the provider, never the key in the error.
    let handle = wrong.resolve(tn()).await.unwrap();
    let err = handle.embed_one("q".into()).await.unwrap_err();
    assert_eq!(err.code(), "internal", "{err:?}");
    assert!(!err.to_string().contains(SECRET));
    assert!(!format!("{handle:?}").contains(SECRET));
}

#[tokio::test]
async fn the_ssrf_guard_refuses_internal_hosts_and_plain_http() {
    let a = mock(None).await;
    let (engine, app, _db) = setup(PROD_EU).await;
    let reg = |url: &str| {
        json!({"embed": {"base_url": url, "dialect": "openai", "model": "m", "dims": 1024,
                         "regions": ["eu"], "zdr": true}})
    };
    for url in [
        a.url.as_str(), // http://127.0.0.1:<port>: plain http
        "http://api.example.com/v1",
        "https://127.0.0.1:6432/v1",
        "https://[::1]/v1",
        "https://10.43.0.10/v1",
        "https://172.16.5.4/v1",
        "https://192.168.1.119/v1",
        "https://169.254.169.254/latest",
        "https://100.64.0.2:5432/v1",
        "https://100.64.0.1/v1",
        "https://0.0.0.0/v1",
        "https://224.0.0.1/v1",
        "https://255.255.255.255/v1",
        "https://[fe80::1]/v1",
        "https://[fd00::1]/v1",
        "https://[::ffff:10.0.0.1]/v1",
        "https://localhost/v1",
        "https://db.example.svc.cluster.local:6432/v1",
        "https://provider.example.internal:9131/v1",
        "https://printer.local/v1",
        "https://pgbouncer/v1",
        "https://2130706433/v1",
    ] {
        let (s, v) = put(&app, TENANT, reg(url)).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{url} -> {v}");
        assert_eq!(
            v["error"]["code"], "provider_host_not_allowed",
            "{url} -> {v}"
        );
        let msg = v["error"]["message"].as_str().unwrap();
        for leak in ["10.", "127.0.0.1", "169.254", "100.", "::"] {
            assert!(
                !msg.contains(leak),
                "{url}: message maps the network: {msg}"
            );
        }
    }
    // parser differentials: refused as written, never re-read as another host
    for url in [
        "https://10.0.0.1\\.x.attacker.com/v1",
        "https://evil@10.0.0.1/v1",
        "https://public.com@10.0.0.1",
        "HTTPS://FOO.INTERNAL",
        "https://foo.internal./v1",
        "https://%31%30.0.0.1/v1",
        "https://10.0.0.1\t.example.com/v1",
    ] {
        let (s, v) = put(&app, TENANT, reg(url)).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{url:?} -> {v}");
        assert_eq!(
            v["error"]["code"], "provider_host_not_allowed",
            "{url:?} -> {v}"
        );
    }
    assert_eq!(a.embeds(), 0, "no request reached the internal mock");
    assert_eq!(get(&app, TENANT).await["embed"]["source"], "unconfigured");
    let err = engine
        .create_store(tn(), store(None, StoreKind::Router, "r"))
        .await
        .unwrap_err();
    assert_eq!(err.code(), "provider_not_configured");
}

/// Rows that bypass the PUT-time check (stored before the guard, or written directly) are
/// re-checked whenever a tenant client is built from them: refused with
/// `provider_host_not_allowed`, and no provider is ever called.
#[tokio::test]
async fn a_stored_internal_host_is_refused_without_calling_out() {
    let a = mock(None).await;
    let db = TestDb::start().await;
    // a tree built through a dev engine that allows the loopback mock ...
    let local = engine_on(&db, LOCAL_EU).await;
    let (s, _) = put(
        &strata_rag::api::router(local.clone()),
        TENANT,
        json!({"embed": embed_reg(&a, "m", None), "rerank": rerank_reg(&a, None)}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    tree(&local, tn()).await;
    let calls = (a.embeds(), a.reranks());

    // ... served by a guarded (production) engine: the stored http://127.0.0.1 row is refused,
    // and so is an internal IP literal written straight into the table
    let guarded = engine_on(&db, PROD_EU).await;
    for base_url in [a.url.clone(), "https://10.0.0.5/v1".to_string()] {
        sqlx::query("UPDATE tenant_providers SET base_url = $1, updated_at = clock_timestamp()")
            .bind(&base_url)
            .execute(&db.pool)
            .await
            .unwrap();
        let err = guarded.query(tn(), &query()).await.unwrap_err();
        assert_eq!(
            err.code(),
            "provider_host_not_allowed",
            "{base_url}: {err:?}"
        );
        assert!(!err.to_string().contains("10.0.0.5"), "{err}");
        assert!(!err.to_string().contains("127.0.0.1"), "{err}");
    }
    assert_eq!(
        (a.embeds(), a.reranks()),
        calls,
        "no request reached any provider"
    );
}

fn ask(expand: bool) -> strata_rag::ask::AskRequest {
    let mut q = query();
    q.expand = expand;
    strata_rag::ask::AskRequest {
        query: q,
        max_tokens: 64,
    }
}

fn expanded() -> QueryRequest {
    let mut q = query();
    q.expand = true;
    q
}

#[tokio::test]
async fn ask_and_expand_use_the_tenants_chat_before_the_default() {
    let default = mock(None).await;
    let mine = mock(Some("sk-chat")).await;
    let (engine, app, _db) = setup(&dev_eu_with_chat(&default)).await;
    tree(&engine, tn()).await;

    // the dev default
    let v = get(&app, TENANT).await;
    assert_eq!(
        v["chat"],
        json!({"source": "default", "dialect": "openai", "model": "local-llm", "dims": null})
    );
    let a = engine.ask(tn(), &ask(false)).await.unwrap();
    assert!(a.answer.contains("white whale"), "{}", a.answer);
    engine.query(tn(), &expanded()).await.unwrap();
    assert_eq!(default.chats(), 2, "one answer + one expansion");

    // a tenant registration wins (only chat is registered: embed / rerank stay the default)
    let (s, v) = put(
        &app,
        TENANT,
        json!({"chat": chat_reg(&mine, Some("sk-chat"))}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["chat"]["source"], "tenant", "{v}");
    // options not registered are not shown (and not sent)
    assert!(v["chat"].get("thinking").is_none(), "{v}");
    assert!(v["chat"].get("temperature").is_none(), "{v}");
    assert_eq!(v["chat"]["has_api_key"], true, "{v}");
    assert!(!v.to_string().contains("sk-chat"), "{v}");
    assert_eq!(mine.chats(), 1, "the registration probe");
    assert_eq!(mine.m.chat_bodies.lock().unwrap()[0]["max_tokens"], 1);

    let r = engine.ask(tn(), &ask(true)).await.unwrap();
    assert!(r.expansion.unwrap().fallback.is_none());
    assert_eq!(
        mine.chats(),
        3,
        "expansion + answer through the tenant's chat"
    );
    assert_eq!(default.chats(), 2, "the default is not called");
    assert!(
        mine.auth()
            .iter()
            .all(|h| h.as_deref() == Some("Bearer sk-chat"))
    );
}

#[tokio::test]
async fn without_a_chat_provider_ask_and_expansion_are_409_and_plain_queries_work() {
    let a = mock(None).await;
    let (engine, app, _db) = setup(LOCAL_EU).await;
    let (s, _) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&a, "m", None), "rerank": rerank_reg(&a, None)}),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(get(&app, TENANT).await["chat"]["source"], "unconfigured");
    tree(&engine, tn()).await;

    assert!(
        !engine
            .query(tn(), &query())
            .await
            .unwrap()
            .results
            .is_empty()
    );
    for err in [
        engine.ask(tn(), &ask(false)).await.unwrap_err(),
        engine.query(tn(), &expanded()).await.unwrap_err(),
    ] {
        assert_eq!((err.code(), err.status()), ("provider_not_configured", 409));
        assert!(err.to_string().contains("chat"), "{err}");
    }
    assert_eq!(a.chats(), 0);
    let h = engine.health().await;
    assert_eq!(h.chat, "unconfigured");
}

#[tokio::test]
async fn chat_registrations_are_validated_probed_and_guarded() {
    let ok = mock(None).await;
    let keyed = mock(Some("sk-needed")).await;
    let (_engine, app, _db) = setup(LOCAL_EU).await;
    let with = |m: &MockServer, f: &dyn Fn(&mut Value)| {
        let mut v = chat_reg(m, None);
        f(&mut v);
        json!({ "chat": v })
    };
    for (body, code) in [
        (with(&keyed, &|_| {}), "probe_failed"),
        (
            with(&ok, &|v| v["timeout_ms"] = json!(300_001)),
            "invalid_provider",
        ),
        (with(&ok, &|v| v["dims"] = json!(1024)), "invalid_provider"),
        (
            with(&ok, &|v| v["dialect"] = json!("cohere")),
            "dialect_not_allowed",
        ),
        (with(&ok, &|v| v["zdr"] = json!(false)), "zdr_required"),
        (
            with(&ok, &|v| v["temperature"] = json!(3.0)),
            "invalid_provider",
        ),
    ] {
        let (s, v) = put(&app, TENANT, body.clone()).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{body} -> {v}");
        assert_eq!(v["error"]["code"], code, "{body} -> {v}");
    }
    let (_, v) = put(&app, TENANT, with(&keyed, &|_| {})).await;
    assert!(
        v["error"]["message"].as_str().unwrap().contains("auth"),
        "{v}"
    );
    // thinking is chat-only
    let mut e = embed_reg(&ok, "m", None);
    e["thinking"] = json!(true);
    let (s, v) = put(&app, TENANT, json!({ "embed": e })).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "invalid_provider", "{v}");
    // a valid one, thinking on, at the 300 s cap
    let (s, v) = put(
        &app,
        TENANT,
        with(&ok, &|v| {
            v["thinking"] = json!(true);
            v["timeout_ms"] = json!(300_000);
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["chat"]["thinking"], true, "{v}");
    assert_eq!(
        ok.m.chat_bodies.lock().unwrap()[0]["chat_template_kwargs"],
        json!({"enable_thinking": false}),
        "the 1-token probe never thinks"
    );

    // the SSRF guard applies to chat like the other capabilities
    let (_prod, prod_app, _db2) = setup(PROD_EU).await;
    for url in [
        ok.url.as_str(),
        "https://10.0.0.1/v1",
        "https://llm-provider.internal/v1",
    ] {
        let mut c = chat_reg(&ok, None);
        c["base_url"] = json!(url);
        let (s, v) = put(&prod_app, TENANT, json!({ "chat": c })).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{url} -> {v}");
        assert_eq!(
            v["error"]["code"], "provider_host_not_allowed",
            "{url} -> {v}"
        );
    }
}

#[tokio::test]
async fn the_migration_allows_chat_and_keeps_the_chat_options_chat_only() {
    let db = TestDb::start().await;
    let insert = |cap: &'static str, thinking: Option<bool>| {
        let pool = db.pool.clone();
        async move {
            sqlx::query(
                "INSERT INTO tenant_providers (tenant, capability, base_url, dialect, model, \
                                               regions, zdr, timeout_ms, concurrency, thinking) \
                 VALUES ($1, $2, 'https://x.example/v1', 'openai', 'm', '{eu}', true, 1, 1, $3)",
            )
            .bind(Uuid::now_v7())
            .bind(cap)
            .bind(thinking)
            .execute(&pool)
            .await
        }
    };
    insert("chat", Some(true))
        .await
        .expect("chat is a capability");
    insert("chat", None)
        .await
        .expect("NULL = the option is not sent");
    insert("rerank", None).await.expect("rerank still is");
    let err = insert("vision", None).await.unwrap_err();
    assert_eq!(err.as_database_error().unwrap().code().unwrap(), "23514");
    let err = insert("embed", Some(false)).await.unwrap_err();
    assert_eq!(err.as_database_error().unwrap().code().unwrap(), "23514");
    let err = sqlx::query(
        "INSERT INTO tenant_providers (tenant, capability, base_url, dialect, model, regions, zdr, \
                                       timeout_ms, concurrency, temperature) \
         VALUES ($1, 'rerank', 'https://x.example/v1', 'cohere', 'm', '{eu}', true, 1, 1, 0.2)",
    )
    .bind(Uuid::now_v7())
    .execute(&db.pool)
    .await
    .unwrap_err();
    assert_eq!(err.as_database_error().unwrap().code().unwrap(), "23514");
}

async fn api(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("x-tenant-id", TENANT)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// One unusable registration (a stored chat row whose host the guard refuses) fails only the
/// requests that need chat: plain queries keep working on the other capabilities.
#[tokio::test]
async fn a_bad_chat_row_fails_only_chat() {
    let default = mock(None).await;
    // guarded dev deployment: embed / rerank defaults on the (exempt) loopback mock, no chat default
    let (engine, app, db) = setup(&dev_eu(&default, false)).await;
    tree(&engine, tn()).await;
    sqlx::query(
        "INSERT INTO tenant_providers (tenant, capability, base_url, dialect, model, regions, zdr, \
                                       timeout_ms, concurrency) \
         VALUES ($1, 'chat', 'https://10.0.0.5/v1', 'openai', 'local-llm', '{eu}', true, 5000, 2)",
    )
    .bind(tn().id())
    .execute(&db.pool)
    .await
    .unwrap();

    let q = json!({"query": "white whale harpoon", "route_mode": "embed"});
    let (s, v) = api(&app, "/api/query", q.clone()).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(!v["results"].as_array().unwrap().is_empty(), "{v}");
    let (s, v) = api(&app, "/api/ask", q.clone()).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "provider_host_not_allowed", "{v}");
    assert!(!v.to_string().contains("10.0.0.5"), "{v}");
    let mut expanded = q;
    expanded["expand"] = json!(true);
    let (s, v) = api(&app, "/api/query", expanded).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "provider_host_not_allowed", "{v}");
    assert_eq!(default.chats(), 0);
}

// ---- POST /api/tenant/providers/test: try a provider without saving it (06 §3 A3, B3 T0t) ----

const TEST_PATH: &str = "/api/tenant/providers/test";

/// A test body: `reg` (a registration body) plus its `capability`.
fn draft(cap: &str, mut reg: Value) -> Value {
    reg["capability"] = json!(cap);
    reg
}

/// The minimal body the agent sends: `{capability, dialect, base_url, model, api_key?}`.
fn minimal(cap: &str, m: &MockServer, key: Option<&str>) -> Value {
    let (dialect, model) = match cap {
        "rerank" => ("cohere", "rr"),
        "chat" => ("openai", "local-llm"),
        _ => ("openai", "acme/embed"),
    };
    let mut v = json!({"capability": cap, "base_url": m.url, "dialect": dialect, "model": model});
    if let Some(k) = key {
        v["api_key"] = json!(k);
    }
    v
}

async fn tenant_rows(db: &TestDb) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM tenant_providers")
        .fetch_one(&db.pool)
        .await
        .unwrap()
}

#[tokio::test]
async fn test_probe_stores_nothing() {
    const SECRET: &str = "sk-probe-store-nothing";
    let a = mock(Some(SECRET)).await;
    let (_engine, app, db) = setup(LOCAL_EU).await;

    for cap in ["embed", "rerank", "chat"] {
        let (s, v) = api(&app, TEST_PATH, minimal(cap, &a, Some(SECRET))).await;
        assert_eq!(s, StatusCode::OK, "{cap}: {v}");
        assert_eq!(v["ok"], true, "{cap}: {v}");
        assert_eq!(v["capability"], cap, "{v}");
        assert!(v["latency_ms"].is_u64(), "{v}");
        assert!(v.get("error_class").is_none(), "{v}");
    }
    let (_, v) = api(&app, TEST_PATH, minimal("embed", &a, Some(SECRET))).await;
    assert_eq!(v["dims"], 1024, "{v}");
    assert_eq!(v["model"], "acme/embed", "{v}");
    // every call is a fresh probe (nothing cached): two embeds, one rerank, one chat
    assert_eq!((a.embeds(), a.reranks(), a.chats()), (2, 1, 1));
    assert_eq!(tenant_rows(&db).await, 0, "a probe stores no registration");
    let got = get(&app, TENANT).await;
    for cap in ["embed", "rerank", "chat"] {
        assert_eq!(got[cap]["source"], "unconfigured", "{got}");
    }
}

#[tokio::test]
async fn test_probe_401_is_ok_false() {
    let a = mock(Some("sk-right")).await;
    let (_engine, app, _db) = setup(LOCAL_EU).await;
    let (s, v) = api(&app, TEST_PATH, minimal("embed", &a, Some("sk-wrong"))).await;
    assert_eq!(
        s,
        StatusCode::OK,
        "a failing provider is not a 4xx/5xx: {v}"
    );
    assert_eq!(v["ok"], false, "{v}");
    assert_eq!(v["error_class"], "auth", "{v}");
    assert!(v.get("dims").is_none(), "{v}");
    // a provider that errors (not auth) is also ok:false, never a 5xx
    let b = mock(None).await;
    b.m.fail_status.store(500, Ordering::SeqCst);
    let (s, v) = api(&app, TEST_PATH, minimal("rerank", &b, None)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["ok"], false, "{v}");
    assert_eq!(v["error_class"], "retryable", "{v}");
}

#[tokio::test]
async fn test_probe_uses_stored_key_same_host() {
    const SECRET: &str = "sk-stored-for-probe";
    let a = mock(Some(SECRET)).await;
    let (_engine, app, db) = setup(LOCAL_EU).await;
    let (s, v) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&a, "acme/embed", Some(SECRET))}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let before: String =
        sqlx::query_scalar("SELECT updated_at::text FROM tenant_providers WHERE tenant = $1")
            .bind(tn().id())
            .fetch_one(&db.pool)
            .await
            .unwrap();

    // no api_key, same base_url, another model: the stored key authenticates the probe
    let mut body = minimal("embed", &a, None);
    body["model"] = json!("acme/embed-v2");
    let (s, v) = api(&app, TEST_PATH, body).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["ok"], true, "{v}");
    assert_eq!(v["model"], "acme/embed-v2", "{v}");
    assert!(
        a.auth()
            .iter()
            .all(|h| h.as_deref() == Some(&format!("Bearer {SECRET}")[..])),
        "{:?}",
        a.auth()
    );
    // the registration is untouched
    let after: String =
        sqlx::query_scalar("SELECT updated_at::text FROM tenant_providers WHERE tenant = $1")
            .bind(tn().id())
            .fetch_one(&db.pool)
            .await
            .unwrap();
    assert_eq!(before, after);
    assert_eq!(get(&app, TENANT).await["embed"]["model"], "acme/embed");

    // another tenant has no stored key: the same body probes keyless and the mock refuses it
    let (s, v) = {
        let req = Request::builder()
            .method("POST")
            .uri(TEST_PATH)
            .header("x-tenant-id", OTHER_TENANT)
            .header("content-type", "application/json")
            .body(Body::from(minimal("embed", &a, None).to_string()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice::<Value>(&bytes).unwrap())
    };
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        v["error_class"], "auth",
        "another tenant's key is never used: {v}"
    );
    assert_eq!(a.auth().last().cloned().flatten(), None);
}

#[tokio::test]
async fn test_probe_new_host_without_key_422() {
    const SECRET: &str = "sk-never-to-a-new-host";
    let a = mock(Some(SECRET)).await;
    let b = mock(None).await;
    let (_engine, app, _db) = setup(LOCAL_EU).await;
    let (s, v) = put(
        &app,
        TENANT,
        json!({"rerank": rerank_reg(&a, Some(SECRET))}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");

    let (s, v) = api(&app, TEST_PATH, minimal("rerank", &b, None)).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "secret_required_for_new_host", "{v}");
    assert_eq!(b.reranks(), 0, "no call to the new host");
    // with a key the new host is probed
    let (s, v) = api(&app, TEST_PATH, minimal("rerank", &b, Some("sk-new"))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["ok"], true, "{v}");
    assert_eq!(b.reranks(), 1);
}

#[tokio::test]
async fn test_probe_invalid_bodies_are_422_and_call_nothing() {
    let a = mock(None).await;
    let (_engine, app, db) = setup(LOCAL_EU).await;
    let mut wrong_region = draft("embed", embed_reg(&a, "m", None));
    wrong_region["regions"] = json!(["us"]);
    let mut extra = minimal("embed", &a, None);
    extra["secret"] = json!("x");
    let cases = [
        (minimal("transcribe", &a, None), "unknown_capability"),
        (
            {
                let mut v = minimal("embed", &a, None);
                v["dialect"] = json!("cohere");
                v
            },
            "dialect_not_allowed",
        ),
        (
            {
                let mut v = minimal("embed", &a, None);
                v["dims"] = json!(768);
                v
            },
            "dims_must_be_1024",
        ),
        (
            {
                let mut v = minimal("chat", &a, None);
                v["base_url"] = json!("ftp://x");
                v
            },
            "invalid_provider",
        ),
        (wrong_region, "region_not_allowed"),
        (extra, "invalid_input"),
        (json!({"capability": "embed"}), "invalid_input"),
    ];
    for (body, code) in cases {
        let (s, v) = api(&app, TEST_PATH, body.clone()).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{body}: {v}");
        assert_eq!(v["error"]["code"], code, "{body}: {v}");
    }
    assert_eq!((a.embeds(), a.reranks(), a.chats()), (0, 0, 0));
    assert_eq!(tenant_rows(&db).await, 0);
}

#[tokio::test]
async fn test_probe_guard_refuses_internal_hosts() {
    let (_engine, app, _db) = setup(PROD_EU).await;
    let body = json!({"capability": "embed", "base_url": "https://10.0.0.5/v1",
                      "dialect": "openai", "model": "m"});
    let (s, v) = api(&app, TEST_PATH, body).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "provider_host_not_allowed", "{v}");
    assert!(!v.to_string().contains("10.0.0.5"), "{v}");
}

#[tokio::test]
async fn test_probe_never_echoes_key() {
    const SECRET: &str = "sk-echo-check-31337";
    let a = mock(Some(SECRET)).await;
    let (_engine, app, _db) = setup(LOCAL_EU).await;
    // success, provider failure, and validation failure: the key is in none of the answers
    let ok = api(&app, TEST_PATH, minimal("chat", &a, Some(SECRET))).await;
    let mut bad_key = minimal("chat", &a, Some(SECRET));
    a.m.fail_status.store(401, Ordering::SeqCst);
    let failed = api(&app, TEST_PATH, bad_key.clone()).await;
    bad_key["dims"] = json!(5);
    let invalid = api(&app, TEST_PATH, bad_key).await;
    assert_eq!(ok.1["ok"], true, "{}", ok.1);
    assert_eq!(failed.1["ok"], false, "{}", failed.1);
    assert_eq!(invalid.0, StatusCode::UNPROCESSABLE_ENTITY, "{}", invalid.1);
    for (_, v) in [ok, failed, invalid] {
        let text = v.to_string();
        assert!(!text.contains(SECRET), "key leaked: {text}");
        assert!(!text.contains(&a.url), "url echoed: {text}");
        assert!(v.get("api_key").is_none(), "{text}");
    }
}

#[tokio::test]
async fn test_probe_emits_no_cost_event() {
    let a = mock(None).await;
    let db = TestDb::start().await;
    let (metering, mut rx) =
        strata_rag::metering::Metering::capture(Arc::new(strata_rag::text::WhitespaceCounter));
    let engine = Arc::new(
        Engine::with_source(
            db.pool.clone(),
            registry(&db, LOCAL_EU, true).await,
            Arc::new(strata_rag::text::WhitespaceCounter),
            test_config(),
        )
        .with_metering(metering),
    );
    let app = strata_rag::api::router(engine.clone());
    let drain = |rx: &mut tokio::sync::mpsc::Receiver<strata_rag::metering::CostEvent>| {
        let mut n = 0;
        while rx.try_recv().is_ok() {
            n += 1;
        }
        n
    };

    for cap in ["embed", "rerank", "chat"] {
        let (s, v) = api(&app, TEST_PATH, minimal(cap, &a, None)).await;
        assert_eq!((s, &v["ok"]), (StatusCode::OK, &json!(true)), "{cap}: {v}");
    }
    assert_eq!(a.embeds() + a.reranks() + a.chats(), 3, "the probes ran");
    assert_eq!(
        drain(&mut rx),
        0,
        "a probe is not a tenant's use: no cost event"
    );
    let outbox: i64 = sqlx::query_scalar("SELECT count(*) FROM metering_outbox")
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(outbox, 0);

    // control: the same provider, registered and used, does emit cost events
    let (s, v) = put(
        &app,
        TENANT,
        json!({"embed": embed_reg(&a, "acme/embed", None)}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    tree(&engine, tn()).await;
    assert!(drain(&mut rx) > 0, "the capture sees real provider use");
}
