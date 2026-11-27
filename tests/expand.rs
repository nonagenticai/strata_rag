//! Query expansion end-to-end: real search against Postgres + a mock chat provider (axum on
//! 127.0.0.1:0).
//! The fake embedder is bag-of-words, so a variant identical to a chunk scores cosine 1.0 —
//! which is how these tests see that the variant vectors were actually searched.

mod common;

use common::tn;

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::query::{QueryRequest, QueryResponse};
use strata_rag::routing::RouteMode;
use uuid::Uuid;

use common::mock::spawn;
use common::test_engine;

const PHOTONS: &str = "quantum photons carry light energy in discrete packets";
const LEVELS: &str = "electrons change energy levels and emit radiation";
const QUESTION: &str = "why does light come in packets";

#[derive(Clone)]
struct Mock {
    reply: String,
    seen: Arc<Mutex<Vec<Value>>>,
}

async fn chat(State(m): State<Mock>, Json(body): Json<Value>) -> Json<Value> {
    m.seen.lock().unwrap().push(body);
    Json(json!({"choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": m.reply}}]}))
}

/// An engine whose chat provider always answers `reply`; returns the recorded request bodies.
async fn with_mock_vllm(engine: Engine, reply: &str) -> (Engine, Arc<Mutex<Vec<Value>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let base = spawn(
        Router::new()
            .route("/v1/chat/completions", post(chat))
            .with_state(Mock {
                reply: reply.to_string(),
                seen: seen.clone(),
            }),
    )
    .await;
    let engine = engine.with_chat(common::chat_at(&base, 5));
    (engine, seen)
}

async fn corpus(engine: &Engine) {
    let root = engine
        .create_store(
            tn(),
            NewStore {
                parent_id: None,
                kind: StoreKind::Router,
                name: "library".to_string(),
                description: "root".to_string(),
            },
        )
        .await
        .unwrap()
        .id;
    let leaf = engine
        .create_store(
            tn(),
            NewStore {
                parent_id: Some(root),
                kind: StoreKind::Leaf,
                name: "physics".to_string(),
                description: "physics of light".to_string(),
            },
        )
        .await
        .unwrap()
        .id;
    for (src, text) in [("photons.txt", PHOTONS), ("levels.txt", LEVELS)] {
        engine
            .ingest(tn(), leaf, src.to_string(), None, text)
            .await
            .unwrap();
    }
}

fn req(expand: bool) -> QueryRequest {
    QueryRequest {
        route_mode: RouteMode::Embed,
        min_score: 0.0,
        expand,
        ..QueryRequest::new(QUESTION)
    }
}

fn ids(r: &QueryResponse) -> Vec<Uuid> {
    r.results.iter().map(|x| x.chunk_id).collect()
}

fn best_dense(r: &QueryResponse) -> f32 {
    r.results
        .iter()
        .filter_map(|x| x.scores.dense)
        .fold(f32::MIN, f32::max)
}

#[tokio::test]
async fn expansion_searches_every_variant_and_reports_them() {
    let (engine, _db) = test_engine().await;
    corpus(&engine).await;
    let reply = json!({
        "hyde_passage": LEVELS,
        "rewrites": ["light quanta", "photon packets", "discrete light energy"]
    })
    .to_string();
    let (engine, seen) = with_mock_vllm(engine, &format!("```json\n{reply}\n```")).await;

    let plain = engine.query(tn(), &req(false)).await.unwrap();
    assert!(plain.expansion.is_none());
    assert!(seen.lock().unwrap().is_empty(), "no expand → no vLLM call");
    assert!(
        best_dense(&plain) < 0.99,
        "fixture: the question matches no chunk exactly"
    );

    let expanded = engine.query(tn(), &req(true)).await.unwrap();
    let info = expanded.expansion.as_ref().expect("expansion reported");
    assert_eq!(info.hyde_passage.as_deref(), Some(LEVELS));
    assert_eq!(info.rewrites.len(), 3);
    assert!(info.fallback.is_none());
    // the HyDE vector found its identical chunk: dense = best similarity over all vectors
    let levels = expanded
        .results
        .iter()
        .find(|x| x.text == LEVELS)
        .expect("LEVELS retrieved");
    assert!(levels.scores.dense.unwrap() > 0.999, "{:?}", levels.scores);

    let calls = seen.lock().unwrap();
    assert_eq!(calls.len(), 1, "one vLLM call per expanded query");
    assert_eq!(calls[0]["max_tokens"], 600);
    let prompt = calls[0]["messages"][0]["content"].as_str().unwrap();
    assert!(
        prompt.ends_with(&format!("Query: {QUESTION}\n")),
        "{prompt}"
    );

    let v = serde_json::to_value(&expanded).unwrap();
    assert!(v["expansion"]["ms"].is_u64(), "{v}");
    assert!(v["expansion"].get("fallback").is_none(), "{v}");
    assert!(
        serde_json::to_value(&plain)
            .unwrap()
            .get("expansion")
            .is_none()
    );
}

#[tokio::test]
async fn unusable_expansion_falls_back_to_the_question_alone() {
    let (engine, _db) = test_engine().await;
    corpus(&engine).await;
    let (engine, seen) = with_mock_vllm(engine, "I cannot help with that.").await;

    let plain = engine.query(tn(), &req(false)).await.unwrap();
    let degraded = engine.query(tn(), &req(true)).await.unwrap();
    assert_eq!(seen.lock().unwrap().len(), 1);
    let info = degraded.expansion.as_ref().unwrap();
    assert!(info.hyde_passage.is_none() && info.rewrites.is_empty());
    let why = info.fallback.as_deref().unwrap();
    assert!(why.contains("no JSON object"), "{why}");
    assert_eq!(
        ids(&degraded),
        ids(&plain),
        "fallback = the unexpanded query"
    );
}

#[tokio::test]
async fn expansion_without_a_chat_provider_is_409_but_a_plain_query_works() {
    let (engine, _db) = test_engine().await;
    corpus(&engine).await;
    let plain = engine.query(tn(), &req(false)).await.unwrap();
    assert!(!plain.results.is_empty());
    let err = engine.query(tn(), &req(true)).await.unwrap_err();
    assert_eq!((err.code(), err.status()), ("provider_not_configured", 409));
    assert!(err.to_string().contains("chat"), "{err}");
}

/// A slow chat provider (3 s, over its own 1 s client timeout) cannot hold the query: the call is
/// cut at the client timeout and the query answers with the question alone. (`EXPAND_TIMEOUT` is
/// 180 s, the outer bound, and is pinned by a unit test in `providers::registry`.)
#[tokio::test]
async fn a_slow_chat_provider_degrades_expansion_within_the_bound() {
    let (engine, _db) = test_engine().await;
    corpus(&engine).await;
    let base = spawn(Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            Json(json!({"choices": [{"message": {"role": "assistant", "content": "late"}}]}))
        }),
    ))
    .await;
    let engine = engine.with_chat(common::chat_at(&base, 1));
    let plain = engine.query(tn(), &req(false)).await.unwrap();
    let t = std::time::Instant::now();
    let degraded = engine.query(tn(), &req(true)).await.unwrap();
    let took = t.elapsed();
    assert!(
        took < std::time::Duration::from_millis(2500),
        "took {took:?}"
    );
    let why = degraded.expansion.as_ref().unwrap().fallback.as_deref();
    assert!(
        why.is_some_and(|w| w.starts_with("generation unavailable")),
        "{why:?}"
    );
    assert_eq!(ids(&degraded), ids(&plain));
}

/// A chat provider that is configured but failing still degrades (the question alone answers).
#[tokio::test]
async fn a_failing_chat_provider_degrades_expansion_instead_of_failing() {
    let (engine, _db) = test_engine().await;
    corpus(&engine).await;
    let base = spawn(Router::new().route(
        "/v1/chat/completions",
        post(|| async { axum::http::StatusCode::UNAUTHORIZED }),
    ))
    .await;
    let engine = engine.with_chat(common::chat_at(&base, 5));
    let plain = engine.query(tn(), &req(false)).await.unwrap();
    let degraded = engine.query(tn(), &req(true)).await.unwrap();
    let why = degraded.expansion.as_ref().unwrap().fallback.as_deref();
    assert_eq!(why, Some("generation unavailable: chat: auth: status 401"));
    assert_eq!(ids(&degraded), ids(&plain));
}

#[tokio::test]
async fn ask_reports_the_expansion_it_searched_with() {
    let (engine, _db) = test_engine().await;
    corpus(&engine).await;
    // one canned reply serves both calls: expansion parses it, the answer is the same text
    let reply = json!({"hyde_passage": LEVELS, "rewrites": []}).to_string();
    let (engine, seen) = with_mock_vllm(engine, &reply).await;
    let resp = engine
        .ask(
            tn(),
            &strata_rag::ask::AskRequest {
                query: req(true),
                max_tokens: 100,
                instructions: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(seen.lock().unwrap().len(), 2, "expand call + answer call");
    let info = resp.expansion.expect("ask carries the expansion");
    assert_eq!(info.hyde_passage.as_deref(), Some(LEVELS));
    assert!(info.fallback.is_none());
}

#[tokio::test]
async fn expand_is_accepted_on_the_wire() {
    let r: QueryRequest =
        serde_json::from_str(r#"{"query":"x","expand":true,"route_mode":"embed"}"#).unwrap();
    assert!(r.expand);
    let a: strata_rag::ask::AskRequest =
        serde_json::from_str(r#"{"query":"x","expand":true}"#).unwrap();
    assert!(
        a.query.expand,
        "ask flattens the query params, expand included"
    );
}
