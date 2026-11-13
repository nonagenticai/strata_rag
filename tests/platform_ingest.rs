//! Platform-knowledge ingest (spec platform-knowledge §1 D4, §6, §8): per-document `topic_keys` /
//! `supersedes` / `provenance` over `POST /api/stores/{id}/documents`, the licence allow-list on
//! platform stores, and `db::documents::shadow_keys`.
mod common;

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sqlx::Row;
use strata_rag::Tenant;
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use tower::ServiceExt;
use uuid::Uuid;

/// The platform owner of these tests: `common::OTHER_TENANT`.
fn platform() -> Tenant {
    common::other()
}

/// `common::test_engine` with platform knowledge on: `platform()` owns the platform stores, and
/// the licence allow-list is the spec default.
async fn platform_app() -> (Router, Arc<Engine>, common::TestDb) {
    let db = common::TestDb::start().await;
    let mut cfg = common::test_config();
    cfg.platform_owner = Some(platform());
    cfg.platform_licenses = [
        "CC0-1.0",
        "PDDL-1.0",
        "public-domain",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "OGL-UK-3.0",
        "ODC-BY-1.0",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let engine = Arc::new(Engine::new(
        db.pool.clone(),
        strata_rag::inference::InferenceHandle::fake(),
        Arc::new(strata_rag::text::WhitespaceCounter),
        cfg,
    ));
    (strata_rag::api::router(Arc::clone(&engine)), engine, db)
}

/// A root router with one leaf, owned by `tenant`; returns the leaf.
async fn leaf_of(engine: &Engine, tenant: Tenant) -> Uuid {
    let mk = |parent_id, kind, name: &str| NewStore {
        parent_id,
        kind,
        name: name.to_string(),
        description: format!("{name} store"),
    };
    let root = engine
        .create_store(tenant, mk(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    engine
        .create_store(tenant, mk(Some(root), StoreKind::Leaf, "leaf"))
        .await
        .unwrap()
        .id
}

/// POSTs `body` as JSON into `store`'s documents, acting for `tenant`.
async fn ingest(app: &Router, tenant: Tenant, store: Uuid, body: Value) -> (StatusCode, Value) {
    let mut req = Request::builder()
        .method("POST")
        .uri(format!("/api/stores/{store}/documents"))
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    req.headers_mut().insert(
        "x-tenant-id",
        HeaderValue::from_str(&tenant.to_string()).unwrap(),
    );
    let resp = app.clone().oneshot(req).await.expect("infallible");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let v = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, v)
}

fn provenance(license: &str) -> Value {
    json!({
        "source_url": "https://example.org/leave",
        "license": license,
        "retrieved_at": "2026-10-01T12:00:00Z"
    })
}

#[tokio::test]
async fn tenant_ingest_persists_topic_keys_and_supersedes_without_provenance() {
    let (app, engine, db) = platform_app().await;
    let leaf = leaf_of(&engine, common::tn()).await;
    let (status, v) = ingest(
        &app,
        common::tn(),
        leaf,
        json!({
            "text": "our leave policy grants thirty days",
            "topic_keys": ["hr/leave", "hr/leave"],
            "supersedes": ["hr/leave", "hr/leave/accrual"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{v}");
    assert_eq!(v["status"], "created");
    assert_eq!(v["topic_keys"], json!(["hr/leave"]), "deduplicated");
    assert_eq!(v["supersedes"], json!(["hr/leave", "hr/leave/accrual"]));

    let row = sqlx::query("SELECT topic_keys, supersedes, metadata FROM documents WHERE id = $1")
        .bind(Uuid::parse_str(v["id"].as_str().unwrap()).unwrap())
        .fetch_one(&db.pool)
        .await
        .unwrap();
    let topic: Vec<String> = row.get("topic_keys");
    let sup: Vec<String> = row.get("supersedes");
    assert_eq!(topic, ["hr/leave"]);
    assert_eq!(sup, ["hr/leave", "hr/leave/accrual"]);
    let sqlx::types::Json(md): sqlx::types::Json<Value> = row.get("metadata");
    assert_eq!(md, json!({}), "no provenance given, none stored");

    // a tenant store takes any licence, stored as metadata.provenance
    let (status, v) = ingest(
        &app,
        common::tn(),
        leaf,
        json!({"text": "another tenant note", "provenance": provenance("CC-BY-NC-4.0")}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{v}");
    assert_eq!(v["metadata"]["provenance"]["license"], "CC-BY-NC-4.0");

    // invalid keys are refused before anything is written
    let (status, v) = ingest(
        &app,
        common::tn(),
        leaf,
        json!({"text": "bad keys", "topic_keys": ["HR Leave"]}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "invalid_input");
}

#[tokio::test]
async fn platform_store_requires_provenance_with_an_allowed_licence() {
    let (app, engine, db) = platform_app().await;
    let leaf = leaf_of(&engine, platform()).await;
    let text = "employees accrue annual leave every month";

    let (status, v) = ingest(&app, platform(), leaf, json!({"text": text})).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert_eq!(v["error"]["code"], "invalid_input");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("provenance"),
        "{v}"
    );

    let (status, v) = ingest(
        &app,
        platform(),
        leaf,
        json!({"text": text, "provenance": provenance("CC-BY-NC-4.0")}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("CC-BY-NC-4.0"),
        "{v}"
    );

    // incomplete provenance is refused too
    let (status, v) = ingest(
        &app,
        platform(),
        leaf,
        json!({"text": text, "provenance": {"source_url": "https://example.org", "license": "CC0-1.0"}}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{v}");

    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM documents WHERE store_id = $1")
        .bind(leaf)
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(count, 0, "nothing written by a refused ingest");

    // allowed (case-insensitive)
    let (status, v) = ingest(
        &app,
        platform(),
        leaf,
        json!({
            "text": text,
            "topic_keys": ["hr/leave"],
            "metadata": {"lang": "en"},
            "provenance": provenance("cc-by-4.0")
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{v}");
    assert_eq!(v["topic_keys"], json!(["hr/leave"]));
    assert_eq!(v["metadata"]["lang"], "en");
    assert_eq!(
        v["metadata"]["provenance"],
        json!({
            "source_url": "https://example.org/leave",
            "license": "cc-by-4.0",
            "retrieved_at": "2026-10-01T12:00:00Z"
        })
    );
}

#[tokio::test]
async fn shadow_keys_are_the_callers_supersedes_in_the_given_leaves() {
    let (app, engine, db) = platform_app().await;
    let mine = leaf_of(&engine, common::tn()).await;
    let plat = leaf_of(&engine, platform()).await;
    for (text, sup) in [
        ("first tenant document", json!(["hr/leave", "hr/pay"])),
        ("second tenant document", json!(["hr/pay", "it/laptops"])),
        ("third tenant document", json!([])),
    ] {
        let (status, v) = ingest(
            &app,
            common::tn(),
            mine,
            json!({"text": text, "supersedes": sup}),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED, "{v}");
    }
    let (status, v) = ingest(
        &app,
        platform(),
        plat,
        json!({
            "text": "platform document",
            "supersedes": ["platform/only"],
            "provenance": provenance("CC0-1.0")
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{v}");

    let keys = strata_rag::db::documents::shadow_keys(&db.pool, common::tn(), &[mine, plat])
        .await
        .unwrap();
    assert_eq!(
        keys,
        ["hr/leave", "hr/pay", "it/laptops"],
        "sorted, distinct, own stores only"
    );
    assert!(
        strata_rag::db::documents::shadow_keys(&db.pool, common::tn(), &[plat])
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        strata_rag::db::documents::shadow_keys(&db.pool, common::tn(), &[])
            .await
            .unwrap()
            .is_empty()
    );
}

/// `GET /api/stores/{store}/documents` as `tenant`.
async fn list(app: &Router, tenant: Tenant, store: Uuid) -> (StatusCode, Value) {
    let mut req = Request::builder()
        .method("GET")
        .uri(format!("/api/stores/{store}/documents"))
        .body(Body::empty())
        .unwrap();
    req.headers_mut().insert(
        "x-tenant-id",
        HeaderValue::from_str(&tenant.to_string()).unwrap(),
    );
    let resp = app.clone().oneshot(req).await.expect("infallible");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let v = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, v)
}

#[tokio::test]
async fn document_listing_returns_topic_keys_and_supersedes() {
    let (app, engine, _db) = platform_app().await;
    let leaf = leaf_of(&engine, common::tn()).await;
    let (status, keyed) = ingest(
        &app,
        common::tn(),
        leaf,
        json!({
            "text": "our leave policy grants thirty days",
            "topic_keys": ["hr/leave/accrual", "hr/leave"],
            "supersedes": ["hr/leave"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{keyed}");
    let (status, plain) = ingest(
        &app,
        common::tn(),
        leaf,
        json!({"text": "the canteen opens at eight"}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{plain}");

    let (status, page) = list(&app, common::tn(), leaf).await;
    assert_eq!(status, StatusCode::OK, "{page}");
    let items = page["items"].as_array().expect("items");
    assert_eq!(items.len(), 2, "{page}");
    let item = |id: &Value| {
        items
            .iter()
            .find(|d| d["id"] == *id)
            .unwrap_or_else(|| panic!("{id} not listed: {page}"))
    };

    // the same shape (and stored, sorted values) as the ingest response
    let k = item(&keyed["id"]);
    assert_eq!(
        k["topic_keys"],
        json!(["hr/leave", "hr/leave/accrual"]),
        "{k}"
    );
    assert_eq!(k["supersedes"], json!(["hr/leave"]), "{k}");
    assert_eq!(k["topic_keys"], keyed["topic_keys"]);
    assert_eq!(k["supersedes"], keyed["supersedes"]);

    // unset: present as empty arrays, never null or missing
    let p = item(&plain["id"]);
    assert_eq!(p["topic_keys"], json!([]), "{p}");
    assert_eq!(p["supersedes"], json!([]), "{p}");
    assert_eq!(p["topic_keys"], plain["topic_keys"]);
    assert_eq!(p["supersedes"], plain["supersedes"]);
}
