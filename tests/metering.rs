//! Cost events (`strata_rag::metering`): one per provider call that did work, attributed to the
//! request's tenant and caller headers; none for health probes; chat (`/api/ask`) included.

mod common;

use common::{TENANT, test_engine, tn};

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
use axum::routing::post;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use strata_rag::ask::AskRequest;
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::metering::{CostEvent, Metering, tenant_level_project_id};
use strata_rag::query::QueryRequest;
use strata_rag::text::WhitespaceCounter;
use tokio::sync::mpsc::Receiver;
use tower::ServiceExt;
use uuid::Uuid;

fn metered(engine: Engine) -> (Engine, Receiver<CostEvent>) {
    let (m, rx) = Metering::capture(Arc::new(WhitespaceCounter));
    (engine.with_metering(m), rx)
}

fn drain(rx: &mut Receiver<CostEvent>) -> Vec<CostEvent> {
    let mut out = Vec::new();
    while let Ok(e) = rx.try_recv() {
        out.push(e);
    }
    out
}

fn keys(events: &[CostEvent]) -> Vec<&str> {
    events
        .iter()
        .map(|e| e.get("service_key").unwrap())
        .collect()
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn post_json(uri: &str, body: Value, headers: &[(&str, String)]) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .header("x-tenant-id", TENANT);
    for (k, v) in headers {
        b = b.header(*k, v);
    }
    b.body(Body::from(body.to_string())).unwrap()
}

/// Through the HTTP API: the caller's X-Project-Id / X-App-Id / X-Actor-Id / X-License-Id reach
/// the events of that request only; without them the project is the tenant-level one.
#[tokio::test]
async fn api_calls_emit_one_event_per_provider_call_with_the_callers_attribution() {
    let (engine, _db) = test_engine().await;
    let (engine, mut rx) = metered(engine);
    let app = strata_rag::api::router(Arc::new(engine));
    let (project, app_id, actor, licence) = (
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
    );
    let attribution = [
        ("x-project-id", project.to_string()),
        ("x-app-id", app_id.to_string()),
        ("x-actor-id", actor.to_string()),
        ("x-license-id", licence.to_string()),
    ];

    // store creation embeds its description: one embed call, no caller headers
    let (s, root) = send(
        &app,
        post_json(
            "/api/stores",
            json!({"kind": "router", "name": "library", "description": "all the books"}),
            &[],
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{root}");
    let events = drain(&mut rx);
    assert_eq!(keys(&events), ["strata-rag-embed"]);
    let e = &events[0];
    assert_eq!(e.get("tenant_id"), Some(TENANT));
    let tenant_project = tenant_level_project_id(tn().id()).to_string();
    assert_eq!(e.get("project_id"), Some(tenant_project.as_str()));
    assert_eq!(e.get("app_id"), Some(""));
    assert_eq!(e.get("cost_source"), Some("strata-rag"));
    assert_eq!(e.get("provider"), Some("in-process"));
    assert_eq!(e.get("funding"), Some("local_gpu"));
    assert_eq!(e.get("quantity_unit"), Some("tokens"));
    // "library\n\nall the books": four whitespace tokens
    assert_eq!(e.get("input_tokens"), Some("4"));

    let root_id = root["id"].as_str().unwrap();
    let (s, leaf) = send(
        &app,
        post_json(
            "/api/stores",
            json!({"kind": "leaf", "name": "sea", "description": "sea stories", "parent_id": root_id}),
            &attribution,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{leaf}");
    let leaf_id = leaf["id"].as_str().unwrap();
    drain(&mut rx);

    let ingest = Request::builder()
        .method("POST")
        .uri(format!("/api/stores/{leaf_id}/documents?source=whale.txt"))
        .header(CONTENT_TYPE, "text/plain; charset=utf-8")
        .header("x-tenant-id", TENANT)
        .body(Body::from(
            "The white whale surfaced beside the ship. Ahab watched the white whale.",
        ))
        .unwrap();
    let (s, _) = send(&app, ingest).await;
    assert_eq!(s, StatusCode::CREATED);
    let events = drain(&mut rx);
    assert!(!events.is_empty());
    assert!(
        events
            .iter()
            .all(|e| e.get("service_key") == Some("strata-rag-embed"))
    );
    let extra: Value = serde_json::from_str(events[0].get("usage_extra").unwrap()).unwrap();
    assert_eq!(extra["input_type"], "document");

    let (s, v) = send(
        &app,
        post_json(
            "/api/query",
            json!({"query": "white whale", "route_mode": "embed", "min_score": 0.0}),
            &attribution,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let events = drain(&mut rx);
    assert_eq!(keys(&events), ["strata-rag-embed", "strata-rag-rerank"]);
    for e in &events {
        assert_eq!(e.get("project_id"), Some(project.to_string().as_str()));
        assert_eq!(e.get("app_id"), Some(app_id.to_string().as_str()));
        assert_eq!(e.get("actor_id"), Some(actor.to_string().as_str()));
        assert_eq!(e.get("license_id"), Some(licence.to_string().as_str()));
    }
    let rerank = &events[1];
    assert_eq!(rerank.get("quantity_unit"), Some("passages"));
    let passages = v["results"].as_array().unwrap().len().to_string();
    assert_eq!(rerank.get("input_tokens"), Some(passages.as_str()));
    let event_keys: std::collections::HashSet<&str> =
        events.iter().map(|e| e.get("event_key").unwrap()).collect();
    assert_eq!(event_keys.len(), events.len(), "a fresh event_key per call");

    // the unauthenticated health probe is not metered
    let (s, _) = send(
        &app,
        Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(drain(&mut rx).is_empty());
}

#[tokio::test]
async fn ask_meters_its_generation_as_chat() {
    let (engine, _db) = test_engine().await;
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
        .ingest(
            tn(),
            leaf,
            "photons.txt".to_string(),
            None,
            "quantum photons carry light energy",
        )
        .await
        .unwrap();
    let base = common::mock::spawn(Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            axum::Json(json!({"choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Photons carry energy [1]."}}]}))
        }),
    ))
    .await;
    let engine = engine.with_chat(common::chat_at(&base, 5));
    let (engine, mut rx) = metered(engine);

    let resp = engine
        .ask(
            tn(),
            &AskRequest {
                query: QueryRequest::new("quantum photons"),
                max_tokens: 100,
                instructions: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(
        resp.providers.chat.as_ref().map(|p| p.provider.as_str()),
        Some("in-process"),
        "a fixed (test) chat provider"
    );
    let events = drain(&mut rx);
    assert_eq!(
        keys(&events),
        ["strata-rag-embed", "strata-rag-rerank", "strata-rag-chat"]
    );
    let chat = &events[2];
    assert_eq!(chat.get("provider"), Some("in-process"));
    assert_eq!(chat.get("model"), Some("in-process-chat"));
    assert_eq!(chat.get("funding"), Some("local_gpu"));
    assert_eq!(
        chat.get("output_tokens"),
        Some("4"),
        "Photons carry energy [1]."
    );
    assert!(chat.get("input_tokens").unwrap().parse::<u64>().unwrap() > 10);
    let extra: Value = serde_json::from_str(chat.get("usage_extra").unwrap()).unwrap();
    assert_eq!(extra["max_tokens"], 100);
    // outside a request (the engine called directly): the tenant-level project
    let tenant_project = tenant_level_project_id(tn().id()).to_string();
    assert!(
        events
            .iter()
            .all(|e| e.get("project_id") == Some(tenant_project.as_str()))
    );
}

/// Without metering configured nothing is emitted and the handle is not metered.
#[tokio::test]
async fn metering_is_off_without_a_redis_url() {
    let (engine, _db) = test_engine().await;
    assert!(engine.metering.is_none());
    assert!(!engine.inference(tn()).await.unwrap().is_metered());
}
