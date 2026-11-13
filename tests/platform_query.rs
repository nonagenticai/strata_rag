//! Platform knowledge on the query path (spec 2026-10-20-platform-knowledge-design.md §5-§7):
//! which platform stores a tenant may search (`query::Readable`), the `layer` / `pack` labels,
//! tenant-over-platform precedence (shadowing, margin, near-duplicate dedup), the unchanged
//! tenant-only query, and the metering of the platform embed.

mod common;

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use strata_rag::engine::Engine;
use strata_rag::metering::{CostEvent, Metering};
use strata_rag::precedence::Layer;
use strata_rag::query::QueryRequest;
use strata_rag::text::WhitespaceCounter;
use tokio::sync::mpsc::Receiver;
use tower::ServiceExt;
use uuid::Uuid;

use common::{OTHER_TENANT, PLATFORM, TENANT};

/// The tenant's own leave rule: shares "annual leave days" with `PLATFORM_TEXT`, not
/// "jurisdiction".
const TENANT_TEXT: &str = "acme staff accrue thirty days of annual leave each year";
/// The platform pack's leave rule, about topic `hr/leave`.
const PLATFORM_TEXT: &str =
    "statutory annual leave in the jurisdiction is twenty working days per year";

/// A platform-knowledge engine (`common::platform_config`, owner `PLATFORM`) behind the router,
/// with captured cost events when `metered`.
async fn app(metered: bool) -> (Router, common::TestDb, Option<Receiver<CostEvent>>) {
    let (mut engine, db) = common::test_engine().await;
    engine.cfg = common::platform_config();
    let (engine, rx) = if metered {
        let (m, rx) = Metering::capture(Arc::new(WhitespaceCounter));
        (engine.with_metering(m), Some(rx))
    } else {
        (engine, None)
    };
    (strata_rag::api::router(Arc::new(engine)), db, rx)
}

/// Sends `method uri` as `tenant`, with a JSON `body` when given.
async fn call(
    app: &Router,
    tenant: &str,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let b = Request::builder()
        .method(method)
        .uri(uri)
        .header("x-tenant-id", tenant);
    let req = match body {
        Some(v) => b
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    };
    let resp = app.clone().oneshot(req).await.expect("infallible");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let v = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, v)
}

fn code(v: &Value) -> &str {
    v["error"]["code"].as_str().unwrap_or("")
}

/// A store of `tenant`; a root when `parent` is `None`.
async fn store(app: &Router, tenant: &str, parent: Option<Uuid>, kind: &str, name: &str) -> Uuid {
    let (s, v) = call(
        app,
        tenant,
        "POST",
        "/api/stores",
        Some(json!({"parent_id": parent, "kind": kind, "name": name, "description": name})),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    v["id"].as_str().expect("id").parse().expect("uuid")
}

/// Ingests `body` into `leaf` as `tenant`; must be 201.
async fn ingest(app: &Router, tenant: &str, leaf: Uuid, body: Value) {
    let (s, v) = call(
        app,
        tenant,
        "POST",
        &format!("/api/stores/{leaf}/documents"),
        Some(body),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
}

/// A platform document body: `text` about `topic_keys`, with an allowed licence.
fn platform_doc(text: &str, topic_keys: &[&str]) -> Value {
    json!({
        "text": text,
        "topic_keys": topic_keys,
        "provenance": {
            "source_url": "https://example.org/leave",
            "license": "CC0-1.0",
            "retrieved_at": "2026-10-01T12:00:00Z"
        }
    })
}

fn pack_uri(key: &str, version: i32, action: &str) -> String {
    format!(
        "/api/platform/packs/{}/{version}/{action}",
        key.replace('/', "%2F")
    )
}

/// As the platform owner: `key@version` registered on `root` and (when `publish`) published.
async fn pack(app: &Router, key: &str, version: i32, root: Uuid, publish: bool) {
    let (s, v) = call(
        app,
        PLATFORM,
        "POST",
        "/api/platform/packs",
        Some(json!({"key": key, "version": version, "root_id": root})),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    if publish {
        let (s, v) = call(
            app,
            PLATFORM,
            "POST",
            &pack_uri(key, version, "publish"),
            None,
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
    }
}

struct World {
    tenant_root: Uuid,
    tenant_leaf: Uuid,
    platform_root: Uuid,
    platform_leaf: Uuid,
}

/// The tenant's `acme / hr` holding `TENANT_TEXT`, and the published pack `base@1`
/// (`employment / leave`) holding `PLATFORM_TEXT` about `hr/leave`.
async fn world(app: &Router) -> World {
    let tenant_root = store(app, TENANT, None, "router", "acme").await;
    let tenant_leaf = store(app, TENANT, Some(tenant_root), "leaf", "hr").await;
    ingest(app, TENANT, tenant_leaf, json!({"text": TENANT_TEXT})).await;
    let platform_root = store(app, PLATFORM, None, "router", "employment").await;
    let platform_leaf = store(app, PLATFORM, Some(platform_root), "leaf", "leave").await;
    ingest(
        app,
        PLATFORM,
        platform_leaf,
        platform_doc(PLATFORM_TEXT, &["hr/leave"]),
    )
    .await;
    pack(app, "base", 1, platform_root, true).await;
    World {
        tenant_root,
        tenant_leaf,
        platform_root,
        platform_leaf,
    }
}

/// `POST /api/query` as `tenant`; returns the status and body.
async fn query(app: &Router, tenant: &str, body: Value) -> (StatusCode, Value) {
    call(app, tenant, "POST", "/api/query", Some(body)).await
}

/// `POST /api/query` that must answer 200; returns its results.
async fn results(app: &Router, tenant: &str, body: Value) -> Vec<Value> {
    let (s, v) = query(app, tenant, body).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v["results"].as_array().expect("results").clone()
}

/// (layer, text) of each result, in order.
fn layers(rs: &[Value]) -> Vec<(String, String)> {
    rs.iter()
        .map(|r| {
            (
                r["layer"].as_str().expect("layer").to_string(),
                r["text"].as_str().expect("text").to_string(),
            )
        })
        .collect()
}

#[tokio::test]
async fn a_tenant_searches_its_leaf_and_a_platform_leaf_with_both_layers_labelled() {
    let (app, _db, _) = app(false).await;
    let w = world(&app).await;
    let rs = results(
        &app,
        TENANT,
        json!({"query": "annual leave days", "leaf_ids": [w.tenant_leaf, w.platform_leaf], "top_k": 10}),
    )
    .await;
    let tenant = rs
        .iter()
        .find(|r| r["text"] == TENANT_TEXT)
        .expect("tenant hit");
    assert_eq!(tenant["layer"], "tenant");
    assert!(
        tenant.get("pack").is_none(),
        "no pack on a tenant hit: {tenant}"
    );
    assert_eq!(tenant["store_path"], json!(["acme", "hr"]));
    let platform = rs
        .iter()
        .find(|r| r["text"] == PLATFORM_TEXT)
        .expect("platform hit");
    assert_eq!(platform["layer"], "platform");
    assert_eq!(platform["pack"], "base@1");
    assert_eq!(
        platform["store_path"],
        json!(["base", "employment", "leave"]),
        "pack key, then the path within the platform tree"
    );
    assert_eq!(platform["metadata"]["provenance"]["license"], "CC0-1.0");
    // equal rerank scores: the tenant margin puts the tenant hit first
    assert_eq!(rs[0]["layer"], "tenant", "{rs:?}");

    // a platform root is routed like any root, within its pack only
    for body in [
        json!({"query": "annual leave days", "root_ids": [w.tenant_root, w.platform_root],
               "route_mode": "embed", "min_score": -1.0, "top_k": 10}),
        json!({"query": "annual leave days", "root_id": w.platform_root,
               "route_mode": "embed", "min_score": -1.0, "top_k": 10}),
    ] {
        let rs = results(&app, TENANT, body.clone()).await;
        assert!(
            rs.iter()
                .any(|r| r["layer"] == "platform" && r["pack"] == "base@1"),
            "{body}: {rs:?}"
        );
    }
    // without a named store only the tenant's own tree is searched
    let rs = results(
        &app,
        TENANT,
        json!({"query": "annual leave days", "top_k": 10}),
    )
    .await;
    assert!(!rs.is_empty());
    assert!(rs.iter().all(|r| r["layer"] == "tenant"), "{rs:?}");
}

#[tokio::test]
async fn an_excluded_or_draft_pack_is_not_found_and_a_retired_one_stays_queryable() {
    let (app, _db, _) = app(false).await;
    let w = world(&app).await;
    let leaf_query = |leaf: Uuid| json!({"query": "annual leave", "leaf_ids": [leaf]});

    // a draft pack: the platform owner's only
    let draft_root = store(&app, PLATFORM, None, "router", "retail").await;
    let draft_leaf = store(&app, PLATFORM, Some(draft_root), "leaf", "rota").await;
    ingest(
        &app,
        PLATFORM,
        draft_leaf,
        platform_doc("retail rota rules for annual leave", &["retail/rota"]),
    )
    .await;
    pack(&app, "industry/retail", 1, draft_root, false).await;
    let (s, v) = query(&app, TENANT, leaf_query(draft_leaf)).await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");
    let rs = results(&app, PLATFORM, leaf_query(draft_leaf)).await;
    assert!(!rs.is_empty(), "the owner reads its draft");
    assert!(
        rs.iter().all(|r| r["layer"] == "tenant"),
        "the owner's own stores are its tenant layer: {rs:?}"
    );

    // an exclusion hides every version of the pack from that tenant, routed or not
    let (s, v) = call(
        &app,
        TENANT,
        "PUT",
        "/api/platform/exclusions",
        Some(json!({"key_prefix": "base", "reason": "regulation"})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let (s, v) = query(&app, TENANT, leaf_query(w.platform_leaf)).await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");
    let (s, v) = query(
        &app,
        TENANT,
        json!({"query": "annual leave", "root_ids": [w.tenant_root, w.platform_root]}),
    )
    .await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");
    // another tenant is not excluded
    let rs = results(&app, OTHER_TENANT, leaf_query(w.platform_leaf)).await;
    assert_eq!(
        layers(&rs),
        [("platform".to_string(), PLATFORM_TEXT.to_string())]
    );

    // retired: hidden from the listing, still readable by its pin
    let (s, v) = call(&app, PLATFORM, "POST", &pack_uri("base", 1, "retire"), None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let (s, v) = call(&app, OTHER_TENANT, "GET", "/api/platform/packs", None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["packs"], json!([]), "{v}");
    let rs = results(&app, OTHER_TENANT, leaf_query(w.platform_leaf)).await;
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["pack"], "base@1");
}

#[tokio::test]
async fn a_tenant_document_superseding_a_topic_shadows_the_platform_document() {
    let (app, _db, _) = app(false).await;
    let w = world(&app).await;
    let both = json!({"query": "annual leave days", "leaf_ids": [w.tenant_leaf, w.platform_leaf], "top_k": 10});
    let rs = results(&app, TENANT, both.clone()).await;
    assert!(
        rs.iter().any(|r| r["layer"] == "platform"),
        "premise: {rs:?}"
    );

    // per request: `supersedes` drops the platform hit about that topic
    let rs = results(
        &app,
        TENANT,
        json!({"query": "annual leave days", "leaf_ids": [w.platform_leaf], "supersedes": ["hr/leave"]}),
    )
    .await;
    assert!(rs.is_empty(), "{rs:?}");
    let rs = results(
        &app,
        TENANT,
        json!({"query": "annual leave days", "leaf_ids": [w.platform_leaf], "supersedes": ["hr/pension"]}),
    )
    .await;
    assert_eq!(rs.len(), 1, "another topic shadows nothing");

    // stored: a tenant document in a SEARCHED leaf that supersedes the topic
    ingest(
        &app,
        TENANT,
        w.tenant_leaf,
        json!({"text": "acme leave policy replaces every statutory rule", "supersedes": ["hr/leave"]}),
    )
    .await;
    let rs = results(&app, TENANT, both).await;
    assert!(!rs.is_empty());
    assert!(rs.iter().all(|r| r["layer"] == "tenant"), "{rs:?}");
    // the platform leaf searched alone: the tenant leaf is not searched, its supersedes do not apply
    let rs = results(
        &app,
        TENANT,
        json!({"query": "annual leave days", "leaf_ids": [w.platform_leaf]}),
    )
    .await;
    assert_eq!(rs.len(), 1, "{rs:?}");
}

#[tokio::test]
async fn the_platform_penalty_reorders_and_the_minimum_score_drops_platform_hits() {
    let (app, _db, _) = app(false).await;
    let w = world(&app).await;
    // fake rerank = share of query words in the passage: platform 4/4 = 1.0, tenant 3/4 = 0.75
    let q = |extra: Value| {
        let mut body = json!({"query": "annual leave days jurisdiction",
                              "leaf_ids": [w.tenant_leaf, w.platform_leaf], "top_k": 10});
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        body
    };
    let rs = results(&app, TENANT, q(json!({"platform_penalty": 0.0}))).await;
    assert_eq!(
        rs[0]["layer"], "platform",
        "no margin: the better platform hit leads"
    );
    assert!((rs[0]["scores"]["rerank"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    let rs = results(&app, TENANT, q(json!({"platform_penalty": 0.3}))).await;
    assert_eq!(
        layers(&rs),
        [
            ("tenant".to_string(), TENANT_TEXT.to_string()),
            ("platform".to_string(), PLATFORM_TEXT.to_string())
        ]
    );
    assert!((rs[1]["scores"]["rerank"].as_f64().unwrap() - 0.7).abs() < 1e-5);
    // the minimum applies after the penalty, to platform hits only
    let rs = results(
        &app,
        TENANT,
        q(json!({"platform_penalty": 0.3, "platform_min_score": 0.8})),
    )
    .await;
    assert_eq!(
        layers(&rs),
        [("tenant".to_string(), TENANT_TEXT.to_string())]
    );
    for bad in [
        json!({"platform_penalty": 1.5}),
        json!({"platform_penalty": -0.1}),
    ] {
        let (s, v) = query(&app, TENANT, q(bad.clone())).await;
        assert_eq!(
            (s, code(&v)),
            (StatusCode::UNPROCESSABLE_ENTITY, "invalid_input"),
            "{bad}: {v}"
        );
    }
}

#[tokio::test]
async fn a_platform_near_duplicate_of_a_tenant_hit_is_dropped() {
    let (app, _db, _) = app(false).await;
    let w = world(&app).await;
    // the tenant holds the platform text verbatim (copied in): its copy wins
    ingest(&app, TENANT, w.tenant_leaf, json!({"text": PLATFORM_TEXT})).await;
    let rs = results(
        &app,
        TENANT,
        json!({"query": "statutory annual leave jurisdiction", "leaf_ids": [w.tenant_leaf, w.platform_leaf],
               "top_k": 10, "platform_penalty": 0.0}),
    )
    .await;
    let copies: Vec<&Value> = rs.iter().filter(|r| r["text"] == PLATFORM_TEXT).collect();
    assert_eq!(copies.len(), 1, "{rs:?}");
    assert_eq!(copies[0]["layer"], "tenant");
    assert!(rs.iter().all(|r| r["layer"] == "tenant"), "{rs:?}");
}

#[tokio::test]
async fn a_tenant_only_query_is_unchanged_by_platform_knowledge() {
    let (app, db, _) = app(false).await;
    let w = world(&app).await;
    let off = Engine::new(
        db.pool.clone(),
        strata_rag::inference::InferenceHandle::fake(),
        Arc::new(WhitespaceCounter),
        common::test_config(),
    );
    let on = Engine::new(
        db.pool.clone(),
        strata_rag::inference::InferenceHandle::fake(),
        Arc::new(WhitespaceCounter),
        common::platform_config(),
    );
    for req in [
        QueryRequest {
            leaf_ids: Some(vec![w.tenant_leaf]),
            ..QueryRequest::new("annual leave days")
        },
        QueryRequest {
            min_score: -1.0,
            ..QueryRequest::new("annual leave days")
        },
    ] {
        let a = off.query(common::tn(), &req).await.unwrap();
        let b = on.query(common::tn(), &req).await.unwrap();
        let view = |r: &strata_rag::query::QueryResponse| {
            r.results
                .iter()
                .map(|h| {
                    (
                        h.chunk_id,
                        h.store_path.clone(),
                        h.layer,
                        h.pack.clone(),
                        h.scores.rerank.map(f32::to_bits),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert!(!a.results.is_empty());
        assert_eq!(view(&a), view(&b));
        assert!(
            b.results
                .iter()
                .all(|h| h.layer == Layer::Tenant && h.pack.is_none())
        );
        assert_eq!(a.routing.leaves, b.routing.leaves);
    }
}

#[tokio::test]
async fn the_platform_embed_is_attributed_to_the_caller_and_funded_by_the_platform() {
    let (app, _db, rx) = app(true).await;
    let mut rx = rx.expect("metered");
    let w = world(&app).await;
    while rx.try_recv().is_ok() {} // ingest and store embeds
    results(
        &app,
        TENANT,
        json!({"query": "annual leave days", "leaf_ids": [w.tenant_leaf, w.platform_leaf]}),
    )
    .await;
    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }
    let embeds: Vec<&CostEvent> = events
        .iter()
        .filter(|e| e.get("service_key") == Some("strata-rag-embed"))
        .collect();
    // one question embed per (owner, model) group: the tenant's and the platform's
    assert_eq!(embeds.len(), 2, "{events:?}");
    for e in &events {
        assert_eq!(e.get("tenant_id"), Some(TENANT), "{e:?}");
        assert_ne!(e.get("funding"), Some("tenant_byo"), "{e:?}");
    }
    assert_eq!(
        events
            .iter()
            .filter(|e| e.get("service_key") == Some("strata-rag-rerank"))
            .count(),
        1,
        "one rerank, the caller's"
    );
}
