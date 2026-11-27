//! Engine::ask end-to-end: real query against Postgres + a mock OpenAI-dialect chat provider
//! (axum on 127.0.0.1:0).

mod common;

use common::tn;

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};
use strata_rag::ask::AskRequest;
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::error::StrataError;
use strata_rag::query::QueryRequest;
use uuid::Uuid;

use common::mock::spawn;
use common::test_engine;

const PHOTONS: &str = "quantum photons carry light energy in discrete packets";
const LEVELS: &str = "quantum photons are emitted when electrons change energy levels";

/// (Authorization header, request body)
type Call = (Option<String>, Value);

#[derive(Clone, Default)]
struct Seen(Arc<Mutex<Vec<Call>>>);

async fn chat(
    State(seen): State<Seen>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Json<Value> {
    let auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    seen.0.lock().unwrap().push((auth, body));
    // thinking-model shape: leading whitespace in content plus a separate reasoning field
    Json(json!({"choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant",
                    "content": "\n\nPhotons are emitted on transitions [2] and carry energy [1].",
                    "reasoning": "thinking..."}}]}))
}

async fn physics_corpus(engine: &Engine) -> Uuid {
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
    engine
        .ingest(tn(), leaf, "photons.txt".to_string(), None, PHOTONS)
        .await
        .unwrap();
    engine
        .ingest(tn(), leaf, "levels.txt".to_string(), None, LEVELS)
        .await
        .unwrap();
    leaf
}

#[tokio::test]
async fn ask_generates_from_numbered_sources_and_maps_citations() {
    let (engine, _db) = test_engine().await;
    physics_corpus(&engine).await;
    let seen = Seen::default();
    let base = spawn(
        Router::new()
            .route("/v1/chat/completions", post(chat))
            .with_state(seen.clone()),
    )
    .await;
    let engine = engine.with_chat(common::chat_at(&base, 5));

    let req = AskRequest {
        query: QueryRequest::new("quantum photons"),
        max_tokens: 800,
        instructions: None,
    };
    let resp = engine.ask(tn(), &req).await.unwrap();
    assert_eq!(resp.results.len(), 2);
    assert_eq!(
        resp.answer,
        "Photons are emitted on transitions [2] and carry energy [1]."
    );
    assert_eq!(
        resp.citations,
        vec![resp.results[1].chunk_id, resp.results[0].chunk_id]
    );

    let calls = seen.0.lock().unwrap();
    assert_eq!(calls.len(), 1);
    let (auth, body) = &calls[0];
    assert_eq!(auth.as_deref(), Some("Bearer vtok"));
    assert_eq!(body["model"], "local-llm");
    assert_eq!(body["max_tokens"], 800);
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["role"], "user");
    let user = body["messages"][1]["content"].as_str().unwrap();
    assert!(
        user.contains(&format!(
            "[1] (library / physics)\n{}",
            resp.results[0].text
        )),
        "{user}"
    );
    assert!(
        user.contains(&format!(
            "[2] (library / physics)\n{}",
            resp.results[1].text
        )),
        "{user}"
    );
    assert!(user.ends_with("Question: quantum photons"), "{user}");
}

#[tokio::test]
async fn ask_without_a_chat_provider_is_409_provider_not_configured() {
    let (engine, _db) = test_engine().await;
    let err = engine
        .ask(
            tn(),
            &AskRequest {
                query: QueryRequest::new("anything"),
                max_tokens: 800,
                instructions: None,
            },
        )
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            StrataError::ProviderNotConfigured { capability: "chat" }
        ),
        "{err:?}"
    );
    assert_eq!((err.code(), err.status()), ("provider_not_configured", 409));
}

#[tokio::test]
async fn ask_with_chat_down_is_generation_unavailable_but_query_still_works() {
    let (engine, _db) = test_engine().await;
    physics_corpus(&engine).await;
    let base = spawn(Router::new().route(
        "/v1/chat/completions",
        post(|| async { StatusCode::BAD_GATEWAY }),
    ))
    .await;
    let engine = engine.with_chat(common::chat_at(&base, 5));
    let err = engine
        .ask(
            tn(),
            &AskRequest {
                query: QueryRequest::new("quantum photons"),
                max_tokens: 800,
                instructions: None,
            },
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, StrataError::GenerationUnavailable(ref m) if m == "chat: retryable: status 502"),
        "{err:?}"
    );
    // /api/query without expansion never depends on chat
    assert_eq!(
        engine
            .query(tn(), &QueryRequest::new("quantum photons"))
            .await
            .unwrap()
            .results
            .len(),
        2
    );
}

#[tokio::test]
async fn ask_rejects_zero_max_tokens() {
    let (engine, _db) = test_engine().await;
    let base = spawn(Router::new()).await;
    let engine = engine.with_chat(common::chat_at(&base, 5));
    let err = engine
        .ask(
            tn(),
            &AskRequest {
                query: QueryRequest::new("x"),
                max_tokens: 0,
                instructions: None,
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::InvalidInput(_)), "{err:?}");
}
