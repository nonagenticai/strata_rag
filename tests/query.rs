//! Engine::query end-to-end against Postgres (testcontainers) with the fake embedder/reranker.
//! Covers Review Focus #3 (empty tree / childless router / chunkless leaf → 200 empty) and
//! Review Focus #5 (laya slow / 401 / erroring → still answers, laya_fallback set, one timeout max).

mod common;

use common::tn;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use strata_rag::clients::laya::LayaClient;
use strata_rag::config::Secret;
use strata_rag::db::search::{ChunkFilter, dense_search, fetch_chunks, lexical_search};
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::error::{Result, StrataError};
use strata_rag::query::QueryRequest;
use strata_rag::routing::{BoxFut, ChoiceOracle, RouteMode, RoutingCache};
use strata_rag::search::{CANDIDATES, RRF_K, rrf, search_leaves};
use uuid::Uuid;

use common::mock::spawn;
use common::test_engine;

const PHYSICS_QUANTUM: &str =
    "quantum electrons photons wavefunction superposition entanglement spin";
const PHYSICS_GRAVITY: &str = "gravity orbits planets mass acceleration newton einstein spacetime";
const CHEMISTRY: &str = "molecules covalent bonds reactions catalysts acids bases titration";
const MUSIC: &str = "melody harmony rhythm violin piano symphony orchestra chords";
const PAINTING: &str = "pigments canvas brushstrokes oil watercolor fresco portrait landscape";

async fn store_for(
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
                description: name.to_string(),
            },
        )
        .await
        .expect("create store")
        .id
}

async fn store(
    engine: &Engine,
    parent: Option<Uuid>,
    kind: StoreKind,
    name: &str,
    description: &str,
) -> Uuid {
    engine
        .create_store(
            tn(),
            NewStore {
                parent_id: parent,
                kind,
                name: name.to_string(),
                description: description.to_string(),
            },
        )
        .await
        .expect("create store")
        .id
}

async fn ingest(engine: &Engine, leaf: Uuid, source: &str, text: &str) {
    engine
        .ingest(tn(), leaf, source.to_string(), None, text)
        .await
        .expect("ingest");
}

/// library ─┬─ science ─┬─ physics  (quantum, gravity)
///          │           └─ chemistry
///          └─ arts    ─┬─ music
///                      └─ painting
struct Library {
    root: Uuid,
    science: Uuid,
    arts: Uuid,
    physics: Uuid,
    chemistry: Uuid,
    music: Uuid,
    painting: Uuid,
}

async fn library(engine: &Engine) -> Library {
    let root = store(
        engine,
        None,
        StoreKind::Router,
        "library",
        "everything we know",
    )
    .await;
    let science = store(
        engine,
        Some(root),
        StoreKind::Router,
        "science",
        "natural sciences physics chemistry",
    )
    .await;
    let arts = store(
        engine,
        Some(root),
        StoreKind::Router,
        "arts",
        "creative arts music painting",
    )
    .await;
    let physics = store(
        engine,
        Some(science),
        StoreKind::Leaf,
        "physics",
        "physics of particles and forces",
    )
    .await;
    let chemistry = store(
        engine,
        Some(science),
        StoreKind::Leaf,
        "chemistry",
        "chemistry of molecules",
    )
    .await;
    let music = store(
        engine,
        Some(arts),
        StoreKind::Leaf,
        "music",
        "music composition and instruments",
    )
    .await;
    let painting = store(
        engine,
        Some(arts),
        StoreKind::Leaf,
        "painting",
        "painting pigments and canvas",
    )
    .await;
    ingest(engine, physics, "quantum.txt", PHYSICS_QUANTUM).await;
    ingest(engine, physics, "gravity.txt", PHYSICS_GRAVITY).await;
    ingest(engine, chemistry, "chemistry.txt", CHEMISTRY).await;
    ingest(engine, music, "music.txt", MUSIC).await;
    ingest(engine, painting, "painting.txt", PAINTING).await;
    Library {
        root,
        science,
        arts,
        physics,
        chemistry,
        music,
        painting,
    }
}

fn strict(query: &str) -> QueryRequest {
    QueryRequest {
        route_mode: RouteMode::Embed,
        beam: 1,
        min_score: 0.99,
        ..QueryRequest::new(query)
    }
}

/// Every router ambiguous: min_score −1 means all children clear it, and 2 > beam 1.
fn forced_ambiguity() -> QueryRequest {
    QueryRequest {
        route_mode: RouteMode::Hybrid,
        beam: 1,
        min_score: -1.0,
        ..QueryRequest::new("quantum electrons photons")
    }
}

struct FailingOracle {
    calls: AtomicUsize,
}

impl ChoiceOracle for FailingOracle {
    fn choose<'a>(
        &'a self,
        _q: &'a str,
        _o: &'a [(String, String)],
    ) -> BoxFut<'a, Result<HashMap<String, f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(StrataError::Upstream {
                service: "laya",
                message: "status 401".to_string(),
            })
        })
    }
}

struct PickOracle(HashMap<String, f32>);

impl ChoiceOracle for PickOracle {
    fn choose<'a>(
        &'a self,
        _q: &'a str,
        options: &'a [(String, String)],
    ) -> BoxFut<'a, Result<HashMap<String, f32>>> {
        let out: HashMap<String, f32> = options
            .iter()
            .filter_map(|(n, _)| self.0.get(n).map(|p| (n.clone(), *p)))
            .collect();
        Box::pin(async move { Ok(out) })
    }
}

// ---------- routing + retrieval ----------

#[tokio::test]
async fn query_routes_to_the_matching_leaf_and_ranks_its_chunk_first() {
    let (engine, _db) = test_engine().await;
    let lib = library(&engine).await;

    let resp = engine
        .query(tn(), &strict("quantum electrons photons"))
        .await
        .unwrap();
    assert_eq!(resp.routing.leaves, vec![lib.physics]);
    assert_eq!(resp.routing.mode_used, Some(RouteMode::Embed));
    assert_eq!(resp.results.len(), 2, "both physics chunks, nothing else");
    assert_eq!(resp.results[0].text, PHYSICS_QUANTUM);
    for r in &resp.results {
        assert_eq!(r.store_path, vec!["library", "science", "physics"]);
    }
    // 2 fused candidates ≤ the default rerank budget (8): both reranked, best first
    let (r0, r1) = (resp.results[0].scores.rerank, resp.results[1].scores.rerank);
    assert!(r0.is_some() && r1.is_some(), "{r0:?} {r1:?}");
    assert!(r0 >= r1);
    assert!(resp.results[0].scores.dense.is_some());
    assert!(
        resp.results[0].scores.lexical.is_some(),
        "query words occur in the chunk"
    );
}

#[tokio::test]
async fn default_hybrid_query_caps_each_level_and_still_ranks_the_answer_first() {
    let (engine, _db) = test_engine().await;
    let lib = library(&engine).await;
    let resp = engine
        .query(tn(), &QueryRequest::new("quantum electrons photons"))
        .await
        .unwrap();
    assert_eq!(resp.routing.mode_used, Some(RouteMode::Hybrid));
    // beam 3 ≥ 2 children everywhere: a per-router cap would keep all four leaves (a flat search);
    // the per-level cap keeps the best 3 (≤ 2·beam, but the extras need ≥ min_score), best first
    let leaves = &resp.routing.leaves;
    assert_eq!(leaves.len(), 3, "{leaves:?}");
    assert_eq!(leaves[0], lib.physics);
    for l in leaves {
        assert!(
            [lib.physics, lib.chemistry, lib.music, lib.painting].contains(l),
            "{leaves:?}"
        );
    }
    assert_eq!(resp.results[0].text, PHYSICS_QUANTUM);
}

#[tokio::test]
async fn root_id_restricts_routing_to_that_subtree() {
    let (engine, _db) = test_engine().await;
    let lib = library(&engine).await;

    let under_arts = engine
        .query(
            tn(),
            &QueryRequest {
                root_id: Some(lib.arts),
                ..strict("quantum electrons photons")
            },
        )
        .await
        .unwrap();
    assert_eq!(under_arts.routing.leaves.len(), 1);
    assert!([lib.music, lib.painting].contains(&under_arts.routing.leaves[0]));

    let under_science = engine
        .query(
            tn(),
            &QueryRequest {
                root_id: Some(lib.science),
                ..strict("quantum electrons photons")
            },
        )
        .await
        .unwrap();
    assert_eq!(under_science.routing.leaves, vec![lib.physics]);

    let from_root = engine
        .query(
            tn(),
            &QueryRequest {
                root_id: Some(lib.root),
                ..strict("quantum electrons photons")
            },
        )
        .await
        .unwrap();
    assert_eq!(from_root.routing.leaves, vec![lib.physics]);
}

#[tokio::test]
async fn laya_mode_follows_the_oracle() {
    let (engine, _db) = test_engine().await;
    let lib = library(&engine).await;
    let oracle = PickOracle(HashMap::from([
        ("science".to_string(), 0.05),
        ("arts".to_string(), 0.9),
        ("music".to_string(), 0.9),
        ("painting".to_string(), 0.05),
    ]));
    let engine = engine.with_laya(Arc::new(oracle));
    let resp = engine
        .query(
            tn(),
            &QueryRequest {
                route_mode: RouteMode::Laya,
                ..strict("quantum electrons photons")
            },
        )
        .await
        .unwrap();
    assert_eq!(resp.routing.leaves, vec![lib.music]);
    assert!(resp.routing.laya_consulted);
    assert_eq!(resp.routing.laya_fallback, None);
    assert!(resp.results.iter().all(|r| r.text == MUSIC));
}

// ---------- Review Focus #3 ----------

#[tokio::test]
async fn empty_tree_returns_empty_results() {
    let (engine, _db) = test_engine().await;
    let resp = engine
        .query(tn(), &QueryRequest::new("anything at all"))
        .await
        .unwrap();
    assert!(resp.results.is_empty());
    assert!(resp.routing.leaves.is_empty());
    assert!(!resp.routing.laya_consulted);
}

#[tokio::test]
async fn router_without_children_returns_empty_results() {
    let (engine, _db) = test_engine().await;
    store(&engine, None, StoreKind::Router, "library", "nothing yet").await;
    let resp = engine
        .query(tn(), &QueryRequest::new("anything at all"))
        .await
        .unwrap();
    assert!(resp.results.is_empty());
    assert!(resp.routing.leaves.is_empty());
}

#[tokio::test]
async fn leaf_without_chunks_returns_empty_results() {
    let (engine, _db) = test_engine().await;
    let root = store(&engine, None, StoreKind::Router, "library", "root").await;
    let leaf = store(
        &engine,
        Some(root),
        StoreKind::Leaf,
        "empty",
        "no documents",
    )
    .await;
    let resp = engine
        .query(tn(), &QueryRequest::new("anything at all"))
        .await
        .unwrap();
    assert_eq!(resp.routing.leaves, vec![leaf]);
    assert!(resp.results.is_empty());
}

// ---------- validation ----------

#[tokio::test]
async fn invalid_requests_are_rejected() {
    let (engine, _db) = test_engine().await;
    let bad = [
        QueryRequest::new("   "),
        QueryRequest {
            top_k: 0,
            ..QueryRequest::new("x")
        },
        QueryRequest {
            top_k: 101,
            ..QueryRequest::new("x")
        },
        // top_k above the 50 fused candidates would be silently capped: reject it instead
        QueryRequest {
            top_k: CANDIDATES + 1,
            ..QueryRequest::new("x")
        },
        QueryRequest {
            beam: 0,
            ..QueryRequest::new("x")
        },
        QueryRequest {
            min_score: f32::NAN,
            ..QueryRequest::new("x")
        },
    ];
    for req in &bad {
        let err = engine.query(tn(), req).await.unwrap_err();
        assert!(
            matches!(err, StrataError::InvalidInput(_)),
            "{req:?} → {err:?}"
        );
    }
    let err = engine
        .query(
            tn(),
            &QueryRequest {
                root_id: Some(Uuid::now_v7()),
                ..QueryRequest::new("x")
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");
}

// ---------- Review Focus #5 ----------

#[tokio::test]
async fn erroring_oracle_still_answers_with_fallback() {
    let (engine, _db) = test_engine().await;
    library(&engine).await;
    let oracle = Arc::new(FailingOracle {
        calls: AtomicUsize::new(0),
    });
    let engine = engine.with_laya(oracle.clone());
    let resp = engine.query(tn(), &forced_ambiguity()).await.unwrap();
    assert!(resp.routing.laya_consulted);
    assert_eq!(resp.routing.laya_fallback.as_deref(), Some("status 401"));
    assert_eq!(
        resp.routing.leaves.len(),
        2,
        "embed fallback: every child clears min_score -1, cap 2·beam = 2 per LEVEL (was 2 per router = 4)"
    );
    assert_eq!(resp.results[0].text, PHYSICS_QUANTUM);
    assert_eq!(oracle.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn laya_401_over_http_still_answers_with_fallback() {
    let (engine, _db) = test_engine().await;
    library(&engine).await;
    let base =
        spawn(Router::new().route("/v1/systemone", post(|| async { StatusCode::UNAUTHORIZED })))
            .await;
    let client = LayaClient::new(&base, Secret::new("t"), "jev-latest", 800).unwrap();
    let engine = engine.with_laya(Arc::new(client));
    let resp = engine.query(tn(), &forced_ambiguity()).await.unwrap();
    assert_eq!(resp.routing.laya_fallback.as_deref(), Some("status 401"));
    assert_eq!(resp.results[0].text, PHYSICS_QUANTUM);
}

#[tokio::test]
async fn laya_transport_error_fallback_leaks_neither_token_nor_url() {
    let (engine, _db) = test_engine().await;
    library(&engine).await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let client = LayaClient::new(&dead, Secret::new("ltok-secret"), "jev-latest", 800).unwrap();
    let engine = engine.with_laya(Arc::new(client));
    let resp = engine.query(tn(), &forced_ambiguity()).await.unwrap();
    let fallback = resp.routing.laya_fallback.expect("laya fallback set");
    assert!(fallback.starts_with("request: "), "{fallback}");
    assert!(
        !fallback.contains("ltok-secret") && !fallback.contains("127.0.0.1"),
        "{fallback}"
    );
    assert_eq!(resp.results[0].text, PHYSICS_QUANTUM);
}

#[tokio::test]
async fn slow_laya_costs_at_most_one_timeout() {
    let (engine, _db) = test_engine().await;
    library(&engine).await;
    let hits = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(
            "/v1/systemone",
            post(|State(h): State<Arc<AtomicUsize>>| async move {
                h.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_secs(3)).await;
                StatusCode::OK
            }),
        )
        .with_state(hits.clone());
    let base = spawn(app).await;
    let client = LayaClient::new(&base, Secret::new("t"), "jev-latest", 500).unwrap();
    let engine = engine.with_laya(Arc::new(client));

    let t = Instant::now();
    let resp = engine.query(tn(), &forced_ambiguity()).await.unwrap();
    let took = t.elapsed();
    assert_eq!(resp.routing.laya_fallback.as_deref(), Some("timeout"));
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "three ambiguous routers, one laya call"
    );
    // The hits == 1 assertion above is the real guard (three ambiguous routers, one laya call).
    // Wall clock only bounds it generously: three timeouts plus a loaded box would still be < 5 s.
    assert!(took < Duration::from_secs(5), "took {took:?}");
    assert_eq!(resp.results[0].text, PHYSICS_QUANTUM);
}

/// Answers like `PickOracle`, after `delay` — slow but successful (below the 800 ms per-call timeout).
struct DelayedPickOracle {
    probs: HashMap<String, f32>,
    delay: Duration,
    calls: AtomicUsize,
}

impl ChoiceOracle for DelayedPickOracle {
    fn choose<'a>(
        &'a self,
        _q: &'a str,
        options: &'a [(String, String)],
    ) -> BoxFut<'a, Result<HashMap<String, f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let out: HashMap<String, f32> = options
            .iter()
            .filter_map(|(n, _)| self.probs.get(n).map(|p| (n.clone(), *p)))
            .collect();
        let delay = self.delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            Ok(out)
        })
    }
}

/// laya picks both science and arts at the root, so laya mode consults all three routers.
fn picks_everything(delay: Duration) -> Arc<DelayedPickOracle> {
    Arc::new(DelayedPickOracle {
        probs: HashMap::from([
            ("science".to_string(), 0.5),
            ("arts".to_string(), 0.5),
            ("physics".to_string(), 0.9),
            ("chemistry".to_string(), 0.05),
            ("music".to_string(), 0.9),
            ("painting".to_string(), 0.05),
        ]),
        delay,
        calls: AtomicUsize::new(0),
    })
}

fn laya_mode() -> QueryRequest {
    QueryRequest {
        route_mode: RouteMode::Laya,
        beam: 1,
        min_score: 0.99,
        ..QueryRequest::new("quantum electrons photons")
    }
}

#[tokio::test]
async fn slow_successful_laya_is_bounded_by_the_query_budget() {
    let (engine, _db) = test_engine().await;
    let lib = library(&engine).await;
    // A level's laya calls run concurrently: root (700 ms) then science + arts together (another
    // 700 ms) = 1.4 s of successful calls; the default query budget is 1.2 s, so level 2 is cut.
    let oracle = picks_everything(Duration::from_millis(700));
    let engine = engine.with_laya(oracle.clone());

    let t = Instant::now();
    let resp = engine.query(tn(), &laya_mode()).await.unwrap();
    let took = t.elapsed();
    assert_eq!(
        resp.routing.laya_fallback.as_deref(),
        Some("query_budget_exhausted"),
        "calls made: {}, took {took:?}, routing {:?}",
        oracle.calls.load(Ordering::SeqCst),
        resp.routing
    );
    assert!(resp.routing.laya_consulted);
    // both level-2 routers were cut at the deadline and routed by embed; the level cut at beam 1
    // keeps the best embed child — level semantics unchanged.
    assert_eq!(resp.routing.leaves, vec![lib.physics]);
    assert_eq!(
        oracle.calls.load(Ordering::SeqCst),
        3,
        "the level-2 calls were cut, not skipped"
    );
    // Budget 1.2 s + generous slack for a loaded box; unbounded it would be >= 1.4 s of laya alone,
    // so the fallback text above is the real proof, this only bounds the wall clock.
    assert!(took < Duration::from_millis(3000), "took {took:?}");
}

#[tokio::test]
async fn the_configured_laya_budget_reaches_the_query_path() {
    let (mut engine, _db) = test_engine().await;
    let _lib = library(&engine).await;
    // 2 concurrent levels x 300 ms = 600 ms fits the 1.2 s default, so only a configured 100 ms budget can cut it.
    engine.cfg.laya_query_budget_ms = 100;
    let oracle = picks_everything(Duration::from_millis(300));
    let engine = engine.with_laya(oracle.clone());
    let resp = engine.query(tn(), &laya_mode()).await.unwrap();
    assert_eq!(
        resp.routing.laya_fallback.as_deref(),
        Some("query_budget_exhausted"),
        "routing {:?}",
        resp.routing
    );
    assert_eq!(oracle.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn fast_laya_does_not_exhaust_the_query_budget() {
    let (engine, _db) = test_engine().await;
    let lib = library(&engine).await;
    let oracle = picks_everything(Duration::ZERO);
    let engine = engine.with_laya(oracle.clone());
    let resp = engine.query(tn(), &laya_mode()).await.unwrap();
    assert_eq!(resp.routing.laya_fallback, None);
    assert_eq!(oracle.calls.load(Ordering::SeqCst), 3);
    let mut leaves = resp.routing.leaves.clone();
    leaves.sort();
    let mut want = vec![lib.physics, lib.music];
    want.sort();
    assert_eq!(leaves, want);
}

// ---------- db::search + cache ----------

#[tokio::test]
async fn dense_search_finds_an_identical_text_first() {
    let (engine, _db) = test_engine().await;
    let lib = library(&engine).await;
    let leaves = [lib.physics, lib.chemistry, lib.music, lib.painting];
    let q = engine
        .inference(tn())
        .await
        .unwrap()
        .embed_one(PHYSICS_QUANTUM.to_string())
        .await
        .unwrap();
    let hits = dense_search(&engine.pool, tn(), &leaves, &q, 10, &ChunkFilter::default())
        .await
        .unwrap();
    assert_eq!(hits.len(), 5);
    assert!(hits[0].1 > 0.999, "sim {}", hits[0].1);
    assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1), "best first");
    let top = fetch_chunks(&engine.pool, tn(), &[hits[0].0])
        .await
        .unwrap();
    assert_eq!(top[0].text, PHYSICS_QUANTUM);
    assert_eq!(top[0].store_id, lib.physics);
    assert!(
        dense_search(&engine.pool, tn(), &[], &q, 10, &ChunkFilter::default())
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn lexical_search_matches_words_and_ignores_empty_queries() {
    let (engine, _db) = test_engine().await;
    let lib = library(&engine).await;
    let hits = lexical_search(
        &engine.pool,
        tn(),
        &[lib.physics],
        "quantum",
        10,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    assert_eq!(hits.len(), 1);
    let chunk = fetch_chunks(&engine.pool, tn(), &[hits[0].0])
        .await
        .unwrap();
    assert_eq!(chunk[0].text, PHYSICS_QUANTUM);
    // a word that only exists in another leaf
    assert!(
        lexical_search(
            &engine.pool,
            tn(),
            &[lib.physics],
            "violin",
            10,
            &ChunkFilter::default()
        )
        .await
        .unwrap()
        .is_empty()
    );
    assert!(
        lexical_search(
            &engine.pool,
            tn(),
            &[lib.physics],
            "",
            10,
            &ChunkFilter::default()
        )
        .await
        .unwrap()
        .is_empty()
    );
    assert!(
        lexical_search(
            &engine.pool,
            tn(),
            &[lib.physics],
            "!!! ???",
            10,
            &ChunkFilter::default()
        )
        .await
        .unwrap()
        .is_empty()
    );
    assert!(
        fetch_chunks(&engine.pool, tn(), &[])
            .await
            .unwrap()
            .is_empty()
    );
}

/// English stopwords leave the OR query: "the" no longer makes every chunk a lexical match. On the
/// seed library one natural question matched 10,076 of 10,103 chunks through its stopwords.
#[tokio::test]
async fn lexical_search_ignores_stopwords_in_the_query() {
    let (engine, _db) = test_engine().await;
    let root = store(&engine, None, StoreKind::Router, "library", "root").await;
    let leaf = store(&engine, Some(root), StoreKind::Leaf, "pets", "pets").await;
    ingest(&engine, leaf, "cat.txt", "the cat sat on the mat").await;
    ingest(&engine, leaf, "dog.txt", "the dog ran in the park").await;
    let hits = lexical_search(
        &engine.pool,
        tn(),
        &[leaf],
        "what is the cat",
        10,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(
        fetch_chunks(&engine.pool, tn(), &[hits[0].0])
            .await
            .unwrap()[0]
            .text,
        "the cat sat on the mat"
    );
    // a query of nothing but stopwords has no lexemes left: no lexical rows (dense still answers)
    assert!(
        lexical_search(
            &engine.pool,
            tn(),
            &[leaf],
            "what is the",
            10,
            &ChunkFilter::default()
        )
        .await
        .unwrap()
        .is_empty()
    );
}

/// A `RoutingCache` other than the engine's (what another replica, or the CLI, holds) follows the
/// database: it keeps serving its snapshot while no store changes, and reloads after a store write
/// made through the engine, which never told this cache (`store_generations`, migration 0010).
#[tokio::test]
async fn routing_cache_follows_store_writes_made_elsewhere() {
    let (engine, _db) = test_engine().await;
    let cache = RoutingCache::default();
    // an empty tree is read but never kept (a `*` caller may name any tenant)
    assert!(
        cache
            .get(&engine.pool, tn())
            .await
            .unwrap()
            .nodes
            .is_empty()
    );
    assert!(cache.cached_tenants().await.is_empty());
    let root = store(&engine, None, StoreKind::Router, "library", "root").await;
    let tree = cache.get(&engine.pool, tn()).await.unwrap();
    assert_eq!(
        tree.root,
        Some(root),
        "the empty tree was not served from a cache"
    );
    assert_eq!(cache.cached_tenants().await, vec![tn()]);
    let again = cache.get(&engine.pool, tn()).await.unwrap();
    assert!(
        Arc::ptr_eq(&tree, &again),
        "unchanged stores: the cached snapshot"
    );
    // another tenant's writes and invalidations leave this tenant's snapshot alone
    cache.invalidate(common::other()).await;
    store_for(&engine, common::other(), None, StoreKind::Router, "theirs").await;
    assert!(Arc::ptr_eq(
        &tree,
        &cache.get(&engine.pool, tn()).await.unwrap()
    ));
    // a write through the engine (not through this cache) is seen on the next read
    let leaf = store(&engine, Some(root), StoreKind::Leaf, "sea", "sea").await;
    let tree = cache.get(&engine.pool, tn()).await.unwrap();
    assert!(
        tree.nodes.contains_key(&leaf),
        "reloaded after a write made elsewhere"
    );
    assert_eq!(tree.path(leaf), vec!["library", "sea"]);
    // so is a write that bypasses strata entirely
    sqlx::query("UPDATE stores SET name = 'ocean' WHERE id = $1")
        .bind(leaf)
        .execute(&engine.pool)
        .await
        .unwrap();
    let tree = cache.get(&engine.pool, tn()).await.unwrap();
    assert_eq!(tree.path(leaf), vec!["library", "ocean"]);
}

// ---------- lexical OR-of-terms (PoC defect) ----------

const PIGEON: &str = "the rock dove is the wild ancestor of every pigeon kept by fanciers today";

#[tokio::test]
async fn lexical_search_matches_any_query_word_not_all_of_them() {
    let (engine, _db) = test_engine().await;
    let root = store(&engine, None, StoreKind::Router, "library", "root").await;
    let birds = store(&engine, Some(root), StoreKind::Leaf, "birds", "birds").await;
    ingest(&engine, birds, "pigeon.txt", PIGEON).await;
    ingest(&engine, birds, "music.txt", MUSIC).await;
    let query = "how are domestic pigeon breeds descended";
    // precondition: most query words are absent from the chunk, so an AND query would match nothing
    let absent = query
        .split(' ')
        .filter(|w| !PIGEON.split(' ').any(|p| p == *w))
        .count();
    assert!(absent >= 5, "{absent} absent words");

    let hits = lexical_search(
        &engine.pool,
        tn(),
        &[birds],
        query,
        10,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        hits.len(),
        1,
        "only the pigeon chunk shares a word: {hits:?}"
    );
    assert_eq!(
        fetch_chunks(&engine.pool, tn(), &[hits[0].0])
            .await
            .unwrap()[0]
            .text,
        PIGEON
    );

    let resp = engine.query(tn(), &QueryRequest::new(query)).await.unwrap();
    let pigeon = resp
        .results
        .iter()
        .find(|r| r.text == PIGEON)
        .expect("pigeon chunk returned");
    assert!(
        pigeon.scores.lexical.is_some(),
        "found by the lexical leg: {:?}",
        pigeon.scores
    );
}

// ---------- rerank budget (stitching note 9) ----------

const BIRD_QUERY: &str = "pigeon dove feather wing nest egg";
/// 0 is RRF-first (dense and lexical: four query words, each twice) but holds only 4 of 6 query
/// words; 1 holds all 6 diluted by filler, so it is RRF-second yet rerank-first (fake reranker =
/// fraction of query words present). That disagreement is what exercises the head sort.
const BIRDS: [&str; 6] = [
    "pigeon dove feather wing pigeon dove feather wing",
    "pigeon dove feather wing nest egg seen far away over the old grey city roofs in winter",
    "dove feather and some other unrelated filler text",
    "a nest with one egg and nothing else of note",
    "wing span of a bird is long and the sky is wide",
    "pigeon lofts",
];

async fn birds(engine: &Engine) -> Uuid {
    let root = store(engine, None, StoreKind::Router, "library", "root").await;
    let leaf = store(engine, Some(root), StoreKind::Leaf, "birds", "birds").await;
    for (i, text) in BIRDS.iter().enumerate() {
        ingest(engine, leaf, &format!("bird{i}.txt"), text).await;
    }
    leaf
}

/// The RRF order computed independently of `search_leaves`, from the two legs it fuses.
async fn fused_order(engine: &Engine, leaf: Uuid, q: &[f32]) -> Vec<Uuid> {
    let limit = CANDIDATES as i64;
    let dense = dense_search(
        &engine.pool,
        tn(),
        &[leaf],
        q,
        limit,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    let lexical = lexical_search(
        &engine.pool,
        tn(),
        &[leaf],
        BIRD_QUERY,
        limit,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    let ids = |v: &[(Uuid, f32)]| v.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    rrf(&[ids(&dense), ids(&lexical)], RRF_K)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

#[tokio::test]
async fn reranked_rows_come_first_then_the_rest_in_rrf_order() {
    let (engine, _db) = test_engine().await;
    let leaf = birds(&engine).await;
    let q = engine
        .inference(tn())
        .await
        .unwrap()
        .embed_one(BIRD_QUERY.to_string())
        .await
        .unwrap();
    let fused = fused_order(&engine, leaf, &q).await;
    assert_eq!(
        fused.len(),
        BIRDS.len(),
        "every bird chunk is a fused candidate"
    );

    let (hits, _) = search_leaves(
        &engine.pool,
        tn(),
        &engine.inference(tn()).await.unwrap(),
        &[leaf],
        std::slice::from_ref(&q),
        BIRD_QUERY,
        10,
        2,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    assert_eq!(hits.len(), BIRDS.len());
    let ids: Vec<Uuid> = hits.iter().map(|h| h.chunk.chunk_id).collect();

    // head: exactly the first 2 fused candidates, reranked, rerank score desc
    let mut head = ids[..2].to_vec();
    let mut want = fused[..2].to_vec();
    head.sort();
    want.sort();
    assert_eq!(head, want, "the rerank budget takes the RRF top 2");
    let r: Vec<f32> = hits[..2]
        .iter()
        .map(|h| h.scores.rerank.expect("reranked"))
        .collect();
    assert!(r[0] >= r[1], "{r:?}");
    // the fixture must make the reranker disagree with RRF, or the head sort is untested
    assert_ne!(
        ids[..2],
        fused[..2],
        "fixture: rerank reorders the RRF head"
    );

    // tail: the remaining fused candidates, in RRF order, not reranked
    assert_eq!(ids[2..], fused[2..]);
    assert!(hits[2..].iter().all(|h| h.scores.rerank.is_none()));
    assert!(hits.iter().all(|h| h.scores.dense.is_some()));

    // top_k cuts across the boundary: 3 = 2 reranked + 1 from the tail
    let (top3, _) = search_leaves(
        &engine.pool,
        tn(),
        &engine.inference(tn()).await.unwrap(),
        &[leaf],
        std::slice::from_ref(&q),
        BIRD_QUERY,
        3,
        2,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    let top3: Vec<Uuid> = top3.iter().map(|h| h.chunk.chunk_id).collect();
    assert_eq!(top3, ids[..3]);

    // top_k inside the budget: only reranked rows
    let (top1, _) = search_leaves(
        &engine.pool,
        tn(),
        &engine.inference(tn()).await.unwrap(),
        &[leaf],
        std::slice::from_ref(&q),
        BIRD_QUERY,
        1,
        2,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    assert_eq!(top1[0].chunk.chunk_id, ids[0]);
    assert!(top1[0].scores.rerank.is_some());
}

#[tokio::test]
async fn query_reranks_only_the_configured_budget() {
    let (mut engine, _db) = test_engine().await;
    engine.cfg.rerank_candidates = 2;
    let leaf = birds(&engine).await;
    let resp = engine
        .query(
            tn(),
            &QueryRequest {
                top_k: 10,
                ..QueryRequest::new(BIRD_QUERY)
            },
        )
        .await
        .unwrap();
    assert_eq!(resp.routing.leaves, vec![leaf]);
    assert_eq!(resp.results.len(), BIRDS.len());
    let reranked: Vec<bool> = resp
        .results
        .iter()
        .map(|r| r.scores.rerank.is_some())
        .collect();
    assert_eq!(reranked, [true, true, false, false, false, false]);
    let v = serde_json::to_value(&resp.results[5].scores).unwrap();
    assert_eq!(
        v["rerank"],
        serde_json::Value::Null,
        "un-reranked rows serialise rerank: null"
    );
}

// ---------- tsquery-hostile input ----------

const URL: &str = "http://example.com:8080/a?b=1&c=2";
const URL_CHUNK: &str = "the mirror lives at http://example.com:8080/a?b=1&c=2 for everyone";
/// Inputs whose 'simple' lexemes contain tsquery operators (`:`, `&`, `(`, `!`, `?`): re-parsing
/// them with to_tsquery is a syntax error or silently turns `&` into AND.
const HOSTILE: [&str; 5] = [
    "see http://example.com:8080/a?b=1&c=2 now",
    "how does http://a.com/x?q=(1)&r=!2 work",
    "what about a|b & c:* ! (d) <-> e",
    r"a backslash in http://a.com/x\y and C:\dir\file o'brien's",
    "quotes in http://a.com/it's'a'trap here",
];

#[tokio::test]
async fn lexical_search_treats_operator_characters_as_plain_text() {
    let (engine, _db) = test_engine().await;
    let root = store(&engine, None, StoreKind::Router, "library", "root").await;
    let leaf = store(&engine, Some(root), StoreKind::Leaf, "web", "web").await;
    ingest(&engine, leaf, "url.txt", URL_CHUNK).await;
    ingest(&engine, leaf, "music.txt", MUSIC).await;
    for q in HOSTILE {
        lexical_search(&engine.pool, tn(), &[leaf], q, 10, &ChunkFilter::default())
            .await
            .unwrap_or_else(|e| panic!("{q:?} → {e}"));
    }
    // the URL is one 'simple' lexeme: the chunk holding it matches, nothing else does
    let hits = lexical_search(
        &engine.pool,
        tn(),
        &[leaf],
        &format!("where is {URL}"),
        10,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(
        fetch_chunks(&engine.pool, tn(), &[hits[0].0])
            .await
            .unwrap()[0]
            .text,
        URL_CHUNK
    );
}

#[tokio::test]
async fn query_with_operator_characters_answers() {
    let (engine, _db) = test_engine().await;
    let root = store(&engine, None, StoreKind::Router, "library", "root").await;
    let leaf = store(&engine, Some(root), StoreKind::Leaf, "web", "web").await;
    ingest(&engine, leaf, "url.txt", URL_CHUNK).await;
    for q in HOSTILE {
        engine
            .query(tn(), &QueryRequest::new(q))
            .await
            .unwrap_or_else(|e| panic!("{q:?} → {e}"));
    }
    let resp = engine
        .query(tn(), &QueryRequest::new(format!("where is {URL}")))
        .await
        .unwrap();
    assert_eq!(resp.results[0].text, URL_CHUNK);
    assert!(
        resp.results[0].scores.lexical.is_some(),
        "{:?}",
        resp.results[0].scores
    );
}

#[tokio::test]
async fn query_results_carry_pages_title_and_source() {
    let (engine, _db) = test_engine().await;
    let root = store(&engine, None, StoreKind::Router, "library", "everything").await;
    let leaf = store(
        &engine,
        Some(root),
        StoreKind::Leaf,
        "harbour",
        "harbour surveys",
    )
    .await;
    // A Bundle-shaped document: title, a section spanning pages 1–2, a section on page 3.
    let text = "# Quillmere Harbour Survey\n\n<!-- page:1 -->\n## Soundings\n\nsoundings depth fathoms.\n\n<!-- page:2 -->\nsoundings continue tidal.\n\n<!-- page:3 -->\n## Moorings\n\nmoorings buoys chains anchors.";
    engine
        .ingest(
            tn(),
            leaf,
            "survey.pdf".to_string(),
            Some("Quillmere Harbour Survey".to_string()),
            text,
        )
        .await
        .expect("ingest");

    let resp = engine
        .query(tn(), &strict("moorings buoys chains"))
        .await
        .unwrap();
    let top = &resp.results[0];
    assert!(top.text.contains("moorings buoys"), "{}", top.text);
    assert_eq!((top.page_start, top.page_end), (Some(3), Some(3)));
    assert_eq!(top.title.as_deref(), Some("Quillmere Harbour Survey"));
    assert_eq!(top.source, "survey.pdf");
    assert_eq!(top.ord, 1);

    let resp = engine
        .query(tn(), &strict("soundings depth fathoms tidal"))
        .await
        .unwrap();
    let top = &resp.results[0];
    assert!(top.text.contains("soundings depth"), "{}", top.text);
    assert_eq!((top.page_start, top.page_end), (Some(1), Some(2)));
    assert_eq!(top.ord, 0);
}
