//! `GET /metrics` through `router()`: Prometheus text exposition of the request counter, the
//! latency histograms and the pool gauges; open without a token, like `/health`.
//!
//! The recorder is process-wide, so assertions look for this test's own series and never for
//! exact counts.

mod common;

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use strata_rag::config::{ApiToken, TokenScope};
use strata_rag::engine::Engine;
use tower::ServiceExt;

fn app_of(engine: Engine) -> Router {
    strata_rag::api::router(Arc::new(engine))
}

/// Status, content type and body text of `req`.
async fn call(app: &Router, req: Request<Body>) -> (StatusCode, String, String) {
    let resp = app
        .clone()
        .oneshot(req)
        .await
        .expect("router is infallible");
    let status = resp.status();
    let ctype = resp
        .headers()
        .get(CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    (status, ctype, String::from_utf8_lossy(&bytes).into_owned())
}

fn json_req(method: &str, uri: &str, body: Value) -> Request<Body> {
    common::with_tenant(
        Request::builder()
            .method(method)
            .uri(uri)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
}

fn get(uri: &str) -> Request<Body> {
    Request::builder().uri(uri).body(Body::empty()).unwrap()
}

/// A line of `exposition` for `metric` (exactly that name, then labels) carrying every label.
fn has_series(exposition: &str, metric: &str, labels: &[&str]) -> bool {
    let prefix = format!("{metric}{{");
    exposition
        .lines()
        .any(|l| l.starts_with(&prefix) && labels.iter().all(|lab| l.contains(lab)))
}

fn id_of(body: &str) -> String {
    let v: Value = serde_json::from_str(body).expect("json body");
    v["id"].as_str().expect("id string").to_owned()
}

#[tokio::test]
async fn query_shows_in_metrics() {
    let (engine, _db) = common::test_engine().await;
    let app = app_of(engine);
    let (s, _, root) = call(
        &app,
        json_req(
            "POST",
            "/api/stores",
            json!({"kind":"router","name":"library","description":"all books"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{root}");
    let (s, _, leaf) = call(
        &app,
        json_req(
            "POST",
            "/api/stores",
            json!({"parent_id": id_of(&root), "kind":"leaf","name":"sea","description":"whales and ships"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{leaf}");
    let ingest = common::with_tenant(
        Request::builder()
            .method("POST")
            .uri(format!(
                "/api/stores/{}/documents?source=whale.txt",
                id_of(&leaf)
            ))
            .header(CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(Body::from("The white whale surfaced beside the ship."))
            .unwrap(),
    );
    let (s, _, v) = call(&app, ingest).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let (s, _, v) = call(
        &app,
        json_req(
            "POST",
            "/api/query",
            json!({"query":"white whale","route_mode":"embed","min_score":0.0}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");

    // no tenant header, no token: the scrape is open
    let (s, ctype, text) = call(&app, get("/metrics")).await;
    assert_eq!(s, StatusCode::OK, "{text}");
    assert!(ctype.starts_with("text/plain"), "{ctype}");
    assert!(
        has_series(
            &text,
            "strata_http_requests_total",
            &[
                r#"method="POST""#,
                r#"route="/api/query""#,
                r#"status="200""#
            ],
        ),
        "{text}"
    );
    // the route template, never the raw path
    assert!(
        has_series(
            &text,
            "strata_http_requests_total",
            &[r#"route="/api/stores/{id}/documents""#, r#"status="201""#],
        ),
        "{text}"
    );
    assert!(
        has_series(
            &text,
            "strata_http_request_duration_seconds_bucket",
            &[r#"route="/api/query""#, r#"le="+Inf""#],
        ),
        "{text}"
    );
    for stage in ["embed", "route", "search", "rerank", "total"] {
        assert!(
            has_series(
                &text,
                "strata_query_stage_duration_seconds_count",
                &[r#"endpoint="query""#, &format!(r#"stage="{stage}""#)],
            ),
            "{stage}: {text}"
        );
    }
    assert!(text.contains("\nstrata_db_pool_size "), "{text}");
    assert!(text.contains("\nstrata_db_pool_idle "), "{text}");
    assert!(
        !text.contains(common::TENANT),
        "no tenant in labels: {text}"
    );
}

#[tokio::test]
async fn metrics_is_open_when_tokens_are_set() {
    let (mut engine, _db) = common::test_engine().await;
    engine.cfg.api_tokens = vec![ApiToken::new(TokenScope::AnyTenant, "metrics-test-token")];
    let app = app_of(engine);

    let (s, _, _) = call(&app, get("/api/eval/runs")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let (s, _, _) = call(&app, get("/no/such/route")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);

    let (s, _, text) = call(&app, get("/metrics")).await;
    assert_eq!(s, StatusCode::OK, "{text}");
    // rejected requests are counted under their route; unknown paths under `unmatched`
    assert!(
        has_series(
            &text,
            "strata_http_requests_total",
            &[r#"route="/api/eval/runs""#, r#"status="401""#],
        ),
        "{text}"
    );
    assert!(
        has_series(
            &text,
            "strata_http_requests_total",
            &[r#"route="unmatched""#, r#"status="401""#],
        ),
        "{text}"
    );

    // a token is not required, and a bad one is not rejected either
    let req = Request::builder()
        .uri("/metrics")
        .header(AUTHORIZATION, "Bearer wrong")
        .body(Body::empty())
        .unwrap();
    let (s, _, _) = call(&app, req).await;
    assert_eq!(s, StatusCode::OK);
}

#[tokio::test]
async fn metrics_listener_is_off_unless_configured() {
    assert!(
        strata_rag::api::bind_metrics(None)
            .await
            .expect("unset is not an error")
            .is_none(),
        "no STRATA_METRICS_BIND, no listener"
    );
}

#[tokio::test]
async fn metrics_listener_serves_only_metrics() {
    let (engine, _db) = common::test_engine().await;
    let listener = strata_rag::api::bind_metrics(Some("127.0.0.1:0".parse().unwrap()))
        .await
        .expect("bind")
        .expect("set means a listener");
    let addr = listener.local_addr().unwrap();
    let app = strata_rag::api::metrics_router(Arc::new(engine));
    tokio::spawn(async move { axum::serve(listener, app).await });

    let client = reqwest::Client::new();
    let ok = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("scrape");
    assert_eq!(ok.status(), 200);
    let body = ok.text().await.unwrap();
    assert!(body.contains("strata_db_pool_size"), "{body}");

    // no API route is reachable here, with or without tenant/auth headers
    for path in ["/api/stores", "/health", "/"] {
        let r = client
            .get(format!("http://{addr}{path}"))
            .header("X-Tenant-Id", common::TENANT)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404, "{path} must not exist on the metrics port");
    }
    let r = client
        .post(format!("http://{addr}/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 405);
}
