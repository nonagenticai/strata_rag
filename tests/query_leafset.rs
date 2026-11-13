//! `POST /api/query` over a named leaf set (`leaf_ids`) and over several trees (`root_ids`),
//! including leaves in two embedding models: the question is embedded once per model, each
//! model's leaves are searched with its own vector, and RRF fuses the lists (spec §6.3).
//! Model A / model B are mock providers wrapping the fake embedder (B answers its vectors
//! reversed, so the two vector spaces differ).

mod common;

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use common::{TENANT, other, test_config, test_engine, tn};
use serde_json::{Value, json};
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::error::StrataError;
use strata_rag::inference::{Embedder, FakeEmbedder, FakeReranker, Reranker};
use strata_rag::ingest::{IngestRequest, SourceTag};
use strata_rag::providers::{Deployment, KeyCipher, LaneLimits, Providers};
use strata_rag::query::QueryRequest;
use strata_rag::routing::RouteMode;
use tower::ServiceExt;
use uuid::Uuid;

const QUANTUM: &str = "quantum electrons photons wavefunction superposition entanglement spin";
const MUSIC: &str = "melody harmony rhythm violin piano symphony orchestra chords";
const PAINTING: &str = "pigments canvas brushstrokes oil watercolor fresco portrait landscape";

async fn store_in(
    engine: &Engine,
    tenant: strata_rag::Tenant,
    parent: Option<Uuid>,
    kind: StoreKind,
    name: &str,
) -> Uuid {
    engine
        .create_store(
            tenant,
            NewStore {
                parent_id: parent,
                kind,
                name: name.to_string(),
                description: format!("{name} documents"),
            },
        )
        .await
        .expect("create store")
        .id
}

async fn store(engine: &Engine, parent: Option<Uuid>, kind: StoreKind, name: &str) -> Uuid {
    store_in(engine, tn(), parent, kind, name).await
}

async fn ingest(engine: &Engine, leaf: Uuid, source: &str, text: &str) {
    engine
        .ingest(tn(), leaf, source.to_string(), None, text)
        .await
        .expect("ingest");
}

/// Two roots: `science` → leaf `physics`; `arts` → leaves `music`, `painting`.
struct Two {
    science: Uuid,
    arts: Uuid,
    physics: Uuid,
    music: Uuid,
    painting: Uuid,
}

async fn two_trees(engine: &Engine) -> Two {
    let science = store(engine, None, StoreKind::Router, "science").await;
    let physics = store(engine, Some(science), StoreKind::Leaf, "physics").await;
    let arts = store(engine, None, StoreKind::Router, "arts").await;
    let music = store(engine, Some(arts), StoreKind::Leaf, "music").await;
    let painting = store(engine, Some(arts), StoreKind::Leaf, "painting").await;
    ingest(engine, physics, "quantum.txt", QUANTUM).await;
    // tagged like the agent platform tags knowledge (`metadata.kind`)
    engine
        .ingest_document(
            tn(),
            IngestRequest {
                store_id: music,
                source: "music.txt".to_string(),
                title: None,
                text: MUSIC,
                tag: SourceTag {
                    source_id: None,
                    source_run: None,
                    metadata: Some(json!({"kind": "arts"})),
                },
            },
        )
        .await
        .expect("ingest");
    ingest(engine, painting, "painting.txt", PAINTING).await;
    Two {
        science,
        arts,
        physics,
        music,
        painting,
    }
}

fn roots(r: &strata_rag::query::QueryResponse) -> Vec<String> {
    r.results.iter().map(|h| h.store_path[0].clone()).collect()
}

#[tokio::test]
async fn leaf_ids_search_exactly_those_leaves_without_routing() {
    let (engine, _db) = test_engine().await;
    let t = two_trees(&engine).await;
    // a question that routing by cosine would send to one side only
    let req = QueryRequest {
        leaf_ids: Some(vec![t.physics, t.music, t.physics]),
        top_k: 10,
        ..QueryRequest::new("quantum photons violin melody")
    };
    let resp = engine.query(tn(), &req).await.unwrap();
    assert_eq!(
        resp.routing.leaves,
        vec![t.physics, t.music],
        "deduplicated"
    );
    assert_eq!(resp.routing.mode_used, None, "nothing was routed");
    assert!(!resp.routing.laya_consulted);
    let texts: Vec<&str> = resp.results.iter().map(|r| r.text.as_str()).collect();
    assert!(texts.contains(&QUANTUM), "{texts:?}");
    assert!(texts.contains(&MUSIC), "{texts:?}");
    assert!(
        !texts.contains(&PAINTING),
        "an unnamed leaf is never searched"
    );
    assert_eq!(resp.timings_by_model.len(), 1);
    assert_eq!(resp.timings_by_model[0].leaves, 2);

    // the document filters apply inside the leaf-set search
    let filtered = engine
        .query(
            tn(),
            &QueryRequest {
                metadata: Some(json!({"kind": "arts"})),
                ..req.clone()
            },
        )
        .await
        .unwrap();
    assert!(!filtered.results.is_empty());
    assert!(filtered.results.iter().all(|r| r.text == MUSIC));
}

#[tokio::test]
async fn leaf_ids_reject_routers_unknown_and_other_tenants_leaves() {
    let (engine, _db) = test_engine().await;
    let t = two_trees(&engine).await;
    let theirs_root = store_in(&engine, other(), None, StoreKind::Router, "theirs").await;
    let theirs = store_in(&engine, other(), Some(theirs_root), StoreKind::Leaf, "x").await;
    let leaf_set = |ids: Vec<Uuid>| QueryRequest {
        leaf_ids: Some(ids),
        ..QueryRequest::new("quantum")
    };
    // another tenant's leaf is not silently dropped: the whole request is 404
    for ids in [vec![t.physics, theirs], vec![Uuid::now_v7()]] {
        let err = engine.query(tn(), &leaf_set(ids)).await.unwrap_err();
        assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");
    }
    let err = engine
        .query(tn(), &leaf_set(vec![t.science]))
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::InvalidInput(_)), "{err:?}");
    // one target kind per request
    let err = engine
        .query(
            tn(),
            &QueryRequest {
                root_id: Some(t.arts),
                ..leaf_set(vec![t.music])
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::InvalidInput(_)), "{err:?}");
}

#[tokio::test]
async fn root_ids_route_under_each_root_and_union_the_leaves() {
    let (engine, _db) = test_engine().await;
    let t = two_trees(&engine).await;
    let req = QueryRequest {
        root_ids: Some(vec![t.science, t.arts]),
        route_mode: RouteMode::Embed,
        beam: 4,
        min_score: -1.0,
        top_k: 10,
        ..QueryRequest::new("quantum photons violin melody")
    };
    let resp = engine.query(tn(), &req).await.unwrap();
    assert_eq!(resp.routing.mode_used, Some(RouteMode::Embed));
    let mut leaves = resp.routing.leaves.clone();
    leaves.sort();
    let mut want = vec![t.physics, t.music, t.painting];
    want.sort();
    assert_eq!(leaves, want);
    let paths = roots(&resp);
    assert!(paths.iter().any(|p| p == "science"), "{paths:?}");
    assert!(paths.iter().any(|p| p == "arts"), "{paths:?}");

    // two roots and no target named is still 409 ambiguous_root (D90)
    let err = engine
        .query(tn(), &QueryRequest::new("quantum"))
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::AmbiguousRoot(_)), "{err:?}");
    // an unknown root is 404
    let err = engine
        .query(
            tn(),
            &QueryRequest {
                root_ids: Some(vec![t.science, Uuid::now_v7()]),
                ..QueryRequest::new("quantum")
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");
}

// ---------- two embedding models ----------

const KEY: [u8; 32] = [42u8; 32];
const MODEL_A: &str = "acme/a@1024";
const MODEL_B: &str = "acme/b@1024";
const SEA: &str = "The harpoon struck the white whale. The crew cheered on the deck of the ship.";
const BAY: &str = "A white whale breached in the bay while the harpoon boats waited by the pier.";

async fn embeddings(State(reverse): State<bool>, Json(b): Json<Value>) -> Json<Value> {
    let texts: Vec<String> = serde_json::from_value(b["input"].clone()).unwrap();
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
    Json(json!({"data": data}))
}

async fn rerank(Json(b): Json<Value>) -> Json<Value> {
    let docs: Vec<String> = serde_json::from_value(b["documents"].clone()).unwrap();
    let scores = FakeReranker
        .score(b["query"].as_str().unwrap(), &docs)
        .unwrap();
    let results: Vec<Value> = scores
        .iter()
        .enumerate()
        .map(|(i, s)| json!({"index": i, "relevance_score": s}))
        .collect();
    Json(json!({"results": results}))
}

/// A mock provider: model A (`reverse` false) or model B (the fake vectors reversed).
async fn mock(reverse: bool) -> String {
    let app = Router::new()
        .route("/embeddings", post(embeddings))
        .route("/rerank", post(rerank))
        .with_state(reverse);
    common::mock::spawn(app).await
}

const LOCAL_EU: &str = "region = \"eu\"\nmode = \"dev\"\n\
                        [policy]\nallowed_dialects = [\"openai\", \"cohere\"]\n\
                        allow_internal_provider_hosts = true\n";

fn embed_reg(url: &str, model: &str) -> Value {
    json!({
        "base_url": url, "dialect": "openai", "model": model, "dims": 1024,
        "regions": ["eu"], "zdr": true, "timeout_ms": 5000, "concurrency": 4
    })
}

async fn put(app: &Router, uri: &str, body: Value) {
    let req = Request::builder()
        .method("PUT")
        .uri(uri)
        .header("x-tenant-id", TENANT)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
}

/// Tenant in model A: root `main` → leaf `sea`. Root `gen2` bound to model B → leaf `bay`.
async fn mixed() -> (Arc<Engine>, common::TestDb, [Uuid; 4]) {
    let db = common::TestDb::start().await;
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
    let (a, b) = (mock(false).await, mock(true).await);
    put(
        &app,
        "/api/tenant/providers",
        json!({"embed": embed_reg(&a, "acme/a"),
               "rerank": {"base_url": a, "dialect": "cohere", "model": "rr",
                          "regions": ["eu"], "zdr": true}}),
    )
    .await;
    let main = store(&engine, None, StoreKind::Router, "main").await;
    let sea = store(&engine, Some(main), StoreKind::Leaf, "sea").await;
    let gen2 = store(&engine, None, StoreKind::Router, "gen2").await;
    put(
        &app,
        &format!("/api/stores/{gen2}/providers"),
        json!({"embed": embed_reg(&b, "acme/b")}),
    )
    .await;
    let bay = store(&engine, Some(gen2), StoreKind::Leaf, "bay").await;
    ingest(&engine, sea, "sea.txt", SEA).await;
    ingest(&engine, bay, "bay.txt", BAY).await;
    (engine, db, [main, sea, gen2, bay])
}

fn models(r: &strata_rag::query::QueryResponse) -> Vec<(String, usize)> {
    r.timings_by_model
        .iter()
        .map(|m| (m.embed_model.clone(), m.leaves))
        .collect()
}

#[tokio::test]
async fn leaves_in_two_embedding_models_are_fused() {
    let (engine, _db, [main, sea, gen2, bay]) = mixed().await;
    let req = QueryRequest {
        leaf_ids: Some(vec![sea, bay]),
        ..QueryRequest::new("white whale harpoon")
    };
    let resp = engine.query(tn(), &req).await.unwrap();
    let paths = roots(&resp);
    assert!(paths.iter().any(|p| p == "main"), "{paths:?}");
    assert!(paths.iter().any(|p| p == "gen2"), "{paths:?}");
    assert_eq!(
        models(&resp),
        vec![(MODEL_A.to_string(), 1), (MODEL_B.to_string(), 1)]
    );
    // each hit was found by its own model's dense search (the same question, embedded per model)
    assert!(
        resp.results
            .iter()
            .all(|r| r.scores.dense.unwrap_or(0.0) > 0.0),
        "{:?}",
        resp.results
    );

    // the same over two roots in different models, routed
    let routed = engine
        .query(
            tn(),
            &QueryRequest {
                root_ids: Some(vec![main, gen2]),
                route_mode: RouteMode::Embed,
                min_score: -1.0,
                ..QueryRequest::new("white whale harpoon")
            },
        )
        .await
        .unwrap();
    let mut leaves = routed.routing.leaves.clone();
    leaves.sort();
    let mut want = vec![sea, bay];
    want.sort();
    assert_eq!(leaves, want);
    let paths = roots(&routed);
    assert!(paths.iter().any(|p| p == "main"), "{paths:?}");
    assert!(paths.iter().any(|p| p == "gen2"), "{paths:?}");
    assert_eq!(routed.timings_by_model.len(), 2);
}

/// Each model's dense search reports its path: a small leaf set (here 2 chunks, far below the
/// default `STRATA_EXACT_SCAN_MAX_CHUNKS`) is scanned exactly; with the threshold at 0 the same
/// query takes the HNSW path. Both answer the same hits on this tiny corpus.
#[tokio::test]
async fn small_leaf_sets_take_the_exact_path_and_threshold_0_forces_hnsw() {
    use strata_rag::db::search::DensePath;

    let (engine, db) = test_engine().await;
    assert_eq!(engine.cfg.exact_scan_max_chunks, 5_000, "the default");
    let t = two_trees(&engine).await;
    let req = QueryRequest {
        leaf_ids: Some(vec![t.physics, t.music]),
        top_k: 10,
        ..QueryRequest::new("quantum photons violin melody")
    };
    let exact = engine.query(tn(), &req).await.unwrap();
    assert_eq!(exact.timings_by_model.len(), 1);
    assert_eq!(exact.timings_by_model[0].path, Some(DensePath::Exact));
    let v = serde_json::to_value(&exact).unwrap();
    assert_eq!(v["timings_by_model"][0]["path"], "exact", "{v}");

    let mut cfg = test_config();
    cfg.exact_scan_max_chunks = 0;
    let hnsw_engine = Engine::new(
        db.pool.clone(),
        strata_rag::inference::InferenceHandle::fake(),
        Arc::new(strata_rag::text::WhitespaceCounter),
        cfg,
    );
    let hnsw = hnsw_engine.query(tn(), &req).await.unwrap();
    assert_eq!(hnsw.timings_by_model[0].path, Some(DensePath::Hnsw));
    let v = serde_json::to_value(&hnsw).unwrap();
    assert_eq!(v["timings_by_model"][0]["path"], "hnsw", "{v}");
    let ids = |r: &strata_rag::query::QueryResponse| -> Vec<Uuid> {
        r.results.iter().map(|h| h.chunk_id).collect()
    };
    assert_eq!(ids(&exact), ids(&hnsw));
    assert!(!ids(&exact).is_empty());
}
