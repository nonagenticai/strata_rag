//! Platform knowledge packs and exclusions (spec 2026-10-20-platform-knowledge-design.md §2-§5,
//! §8): the platform-owner-only pack lifecycle, exclusion prefix semantics, `readable_packs`, the
//! freeze of a published pack's stores, and 404 everywhere while platform knowledge is off.

mod common;

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use strata_rag::db::platform::{assert_writable, readable_packs};
use tower::ServiceExt;
use uuid::Uuid;

use common::{OTHER_TENANT, PLATFORM, TENANT};

async fn app(enabled: bool) -> (Router, sqlx::PgPool, common::TestDb) {
    let (mut engine, db) = common::test_engine().await;
    if enabled {
        engine.cfg = common::platform_config();
    }
    let pool = db.pool.clone();
    (strata_rag::api::router(Arc::new(engine)), pool, db)
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

fn id_of(v: &Value) -> Uuid {
    v["id"].as_str().expect("id").parse().expect("uuid")
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
    id_of(&v)
}

/// Registers `root` as `key@version` (as the platform owner) and returns the pack body.
async fn register(app: &Router, key: &str, version: i32, root: Uuid) -> Value {
    let (s, v) = call(
        app,
        PLATFORM,
        "POST",
        "/api/platform/packs",
        Some(json!({"key": key, "version": version, "root_id": root})),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    v
}

fn pack_uri(key: &str, version: i32, action: &str) -> String {
    format!(
        "/api/platform/packs/{}/{version}/{action}",
        key.replace('/', "%2F")
    )
}

async fn publish(app: &Router, key: &str, version: i32) -> Value {
    let (s, v) = call(
        app,
        PLATFORM,
        "POST",
        &pack_uri(key, version, "publish"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v
}

/// `key@version` of every pack `tenant` lists.
async fn listed(app: &Router, tenant: &str) -> Vec<String> {
    let (s, v) = call(app, tenant, "GET", "/api/platform/packs", None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v["packs"]
        .as_array()
        .expect("packs")
        .iter()
        .map(|p| format!("{}@{}", p["key"].as_str().unwrap(), p["version"]))
        .collect()
}

#[tokio::test]
async fn every_platform_route_is_404_while_disabled() {
    let (app, _pool, _db) = app(false).await;
    let root = store(&app, TENANT, None, "router", "lib").await;
    for (method, uri, body) in [
        ("GET", "/api/platform/packs".to_string(), None),
        (
            "POST",
            "/api/platform/packs".to_string(),
            Some(json!({"key": "base", "version": 1, "root_id": root})),
        ),
        ("POST", pack_uri("base", 1, "publish"), None),
        ("POST", pack_uri("base", 1, "retire"), None),
        ("GET", pack_uri("base", 1, "stores"), None),
        ("GET", "/api/platform/exclusions".to_string(), None),
        (
            "PUT",
            "/api/platform/exclusions".to_string(),
            Some(json!({"key_prefix": "base", "reason": "x"})),
        ),
        (
            "DELETE",
            "/api/platform/exclusions?key_prefix=base".to_string(),
            None,
        ),
    ] {
        let (s, v) = call(&app, TENANT, method, &uri, body).await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{method} {uri}: {v}");
        assert_eq!(code(&v), "not_found");
    }
}

#[tokio::test]
async fn pack_lifecycle_register_publish_retire() {
    let (app, _pool, _db) = app(true).await;
    let root = store(&app, PLATFORM, None, "router", "retail v1").await;
    let v = register(&app, "industry/retail", 1, root).await;
    assert_eq!(v["key"], "industry/retail");
    assert_eq!(v["version"], 1);
    assert_eq!(v["root_id"], root.to_string());
    assert!(v["published_at"].is_null(), "a new pack is a draft: {v}");

    // a draft is the platform owner's only
    assert_eq!(listed(&app, PLATFORM).await, ["industry/retail@1"]);
    assert!(listed(&app, TENANT).await.is_empty());

    let v = publish(&app, "industry/retail", 1).await;
    let published_at = v["published_at"]
        .as_str()
        .expect("published_at")
        .to_string();
    // idempotent: publishing again keeps the first timestamp
    let v = publish(&app, "industry/retail", 1).await;
    assert_eq!(v["published_at"], published_at.as_str());
    assert_eq!(listed(&app, TENANT).await, ["industry/retail@1"]);
    assert_eq!(listed(&app, OTHER_TENANT).await, ["industry/retail@1"]);

    let (s, v) = call(
        &app,
        PLATFORM,
        "POST",
        &pack_uri("industry/retail", 1, "retire"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(v["retired_at"].is_string(), "{v}");
    assert!(listed(&app, TENANT).await.is_empty(), "retired = hidden");
    assert_eq!(listed(&app, PLATFORM).await, ["industry/retail@1"]);

    // unknown packs
    let (s, v) = call(
        &app,
        PLATFORM,
        "POST",
        &pack_uri("nope", 1, "publish"),
        None,
    )
    .await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");
}

#[tokio::test]
async fn pack_writes_are_the_platform_owners_only() {
    let (app, _pool, _db) = app(true).await;
    let mine = store(&app, TENANT, None, "router", "mine").await;
    let root = store(&app, PLATFORM, None, "router", "base v1").await;
    register(&app, "base", 1, root).await;
    for (method, uri, body) in [
        (
            "POST",
            "/api/platform/packs".to_string(),
            Some(json!({"key": "mine", "version": 1, "root_id": mine})),
        ),
        ("POST", pack_uri("base", 1, "publish"), None),
        ("POST", pack_uri("base", 1, "retire"), None),
    ] {
        let (s, v) = call(&app, TENANT, method, &uri, body).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{method} {uri}: {v}");
        assert_eq!(code(&v), "forbidden");
    }
    // still a draft
    assert!(listed(&app, TENANT).await.is_empty());
}

#[tokio::test]
async fn register_validates_key_version_and_root() {
    let (app, _pool, _db) = app(true).await;
    let root = store(&app, PLATFORM, None, "router", "root").await;
    let child = store(&app, PLATFORM, Some(root), "leaf", "child").await;
    let tenants = store(&app, TENANT, None, "router", "tenant root").await;
    let post = |body: Value| {
        let app = app.clone();
        async move { call(&app, PLATFORM, "POST", "/api/platform/packs", Some(body)).await }
    };
    for bad in ["", "Base", "a//b", "/base", "a/b/c/d/e/f/g", "retail.hr"] {
        let (s, v) = post(json!({"key": bad, "version": 1, "root_id": root})).await;
        assert_eq!(
            (s, code(&v)),
            (StatusCode::UNPROCESSABLE_ENTITY, "invalid_input"),
            "{bad}: {v}"
        );
    }
    let (s, v) = post(json!({"key": "base", "version": 0, "root_id": root})).await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::UNPROCESSABLE_ENTITY, "invalid_input"),
        "{v}"
    );
    let (s, v) = post(json!({"key": "base", "version": 1, "root_id": child})).await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::UNPROCESSABLE_ENTITY, "invalid_target"),
        "{v}"
    );
    // another tenant's store is not the platform's: 404, as if it did not exist
    let (s, v) = post(json!({"key": "base", "version": 1, "root_id": tenants})).await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");

    register(&app, "base", 1, root).await;
    // same key@version, and the same root under another version
    let other_root = store(&app, PLATFORM, None, "router", "root 2").await;
    let (s, v) = post(json!({"key": "base", "version": 1, "root_id": other_root})).await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::CONFLICT, "duplicate_name"),
        "{v}"
    );
    let (s, v) = post(json!({"key": "base", "version": 2, "root_id": root})).await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::CONFLICT, "duplicate_name"),
        "{v}"
    );
    register(&app, "base", 2, other_root).await;
}

#[tokio::test]
async fn exclusions_hide_a_prefix_segment_wise_and_only_for_their_tenant() {
    let (app, pool, _db) = app(true).await;
    for (i, key) in [
        "base",
        "industry/retail",
        "industry/retail/hr",
        "industry/retailx",
    ]
    .iter()
    .enumerate()
    {
        let root = store(&app, PLATFORM, None, "router", &format!("r{i}")).await;
        register(&app, key, 1, root).await;
        publish(&app, key, 1).await;
    }
    let all = [
        "base@1",
        "industry/retail/hr@1",
        "industry/retail@1",
        "industry/retailx@1",
    ];
    let mut seen = listed(&app, TENANT).await;
    seen.sort();
    assert_eq!(seen, all);

    let (s, v) = call(
        &app,
        TENANT,
        "PUT",
        "/api/platform/exclusions",
        Some(json!({"key_prefix": "industry/retail", "reason": "regulation"})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["key_prefix"], "industry/retail");
    assert_eq!(v["reason"], "regulation");
    let mut seen = listed(&app, TENANT).await;
    seen.sort();
    assert_eq!(seen, ["base@1", "industry/retailx@1"]);
    // another tenant, and the platform owner, are not affected
    let mut seen = listed(&app, OTHER_TENANT).await;
    seen.sort();
    assert_eq!(seen, all);

    // the db predicate agrees
    let packs = readable_packs(&pool, common::platform(), common::tn())
        .await
        .unwrap();
    let keys: Vec<&str> = packs.iter().map(|p| p.key.as_str()).collect();
    assert_eq!(keys, ["base", "industry/retailx"]);

    // upsert replaces the reason
    let (s, v) = call(
        &app,
        TENANT,
        "PUT",
        "/api/platform/exclusions",
        Some(json!({"key_prefix": "industry/retail", "reason": "performance"})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let (s, v) = call(&app, TENANT, "GET", "/api/platform/exclusions", None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let ex = v["exclusions"].as_array().expect("exclusions");
    assert_eq!(ex.len(), 1, "{v}");
    assert_eq!(ex[0]["reason"], "performance");
    let (s, v) = call(&app, OTHER_TENANT, "GET", "/api/platform/exclusions", None).await;
    assert_eq!(s, StatusCode::OK);
    assert!(v["exclusions"].as_array().unwrap().is_empty(), "{v}");

    // validation
    for body in [
        json!({"key_prefix": "Industry", "reason": "x"}),
        json!({"key_prefix": "industry/", "reason": "x"}),
        json!({"key_prefix": "base", "reason": "  "}),
    ] {
        let (s, v) = call(&app, TENANT, "PUT", "/api/platform/exclusions", Some(body)).await;
        assert_eq!(
            (s, code(&v)),
            (StatusCode::UNPROCESSABLE_ENTITY, "invalid_input"),
            "{v}"
        );
    }

    let uri = "/api/platform/exclusions?key_prefix=industry%2Fretail";
    let (s, v) = call(&app, TENANT, "DELETE", uri, None).await;
    assert_eq!(s, StatusCode::NO_CONTENT, "{v}");
    let (s, v) = call(&app, TENANT, "DELETE", uri, None).await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");
    assert_eq!(listed(&app, TENANT).await.len(), all.len());
}

#[tokio::test]
async fn a_published_pack_is_frozen() {
    let (app, pool, _db) = app(true).await;
    let root = store(&app, PLATFORM, None, "router", "base v1").await;
    let leaf = store(&app, PLATFORM, Some(root), "leaf", "facts").await;
    register(&app, "base", 1, root).await;

    // a draft is writable
    assert_writable(&pool, leaf)
        .await
        .expect("draft is writable");
    let scratch = store(&app, PLATFORM, Some(root), "leaf", "scratch").await;
    let (s, v) = call(
        &app,
        PLATFORM,
        "DELETE",
        &format!("/api/stores/{scratch}"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT, "{v}");

    publish(&app, "base", 1).await;
    let err = assert_writable(&pool, leaf).await.unwrap_err();
    assert_eq!((err.code(), err.status()), ("pack_published", 409));
    assert_eq!(err.to_string(), "pack base@1 is published");

    let frozen = |(s, v): (StatusCode, Value)| {
        assert_eq!(
            (s, code(&v)),
            (StatusCode::CONFLICT, "pack_published"),
            "{v}"
        );
    };
    frozen(
        call(
            &app,
            PLATFORM,
            "POST",
            "/api/stores",
            Some(json!({"parent_id": root, "kind": "leaf", "name": "new", "description": "d"})),
        )
        .await,
    );
    frozen(
        call(
            &app,
            PLATFORM,
            "PATCH",
            &format!("/api/stores/{leaf}"),
            Some(json!({"description": "changed"})),
        )
        .await,
    );
    frozen(
        call(
            &app,
            PLATFORM,
            "DELETE",
            &format!("/api/stores/{leaf}"),
            None,
        )
        .await,
    );
    frozen(
        call(
            &app,
            PLATFORM,
            "DELETE",
            &format!("/api/stores/{root}?recursive=true"),
            None,
        )
        .await,
    );
    frozen(
        call(
            &app,
            PLATFORM,
            "DELETE",
            &format!("/api/stores/{leaf}/providers"),
            None,
        )
        .await,
    );
    // a reembed-copy into a published pack's store
    let src = store(&app, PLATFORM, None, "leaf", "src").await;
    let empty_root = store(&app, PLATFORM, None, "leaf", "empty").await;
    register(&app, "base", 2, empty_root).await;
    publish(&app, "base", 2).await;
    frozen(
        call(
            &app,
            PLATFORM,
            "POST",
            &format!("/api/stores/{src}/reembed-copy"),
            Some(json!({"target_store": empty_root})),
        )
        .await,
    );

    // another tenant touching the platform's stores gets 404, never the 409 hint
    let (s, v) = call(&app, TENANT, "DELETE", &format!("/api/stores/{leaf}"), None).await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");

    // a store outside any pack stays writable
    assert_writable(&pool, src).await.expect("not in a pack");

    // retired: the curator may delete the whole pack root; the pack row goes with it
    let (s, v) = call(&app, PLATFORM, "POST", &pack_uri("base", 1, "retire"), None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    frozen(
        call(
            &app,
            PLATFORM,
            "DELETE",
            &format!("/api/stores/{leaf}"),
            None,
        )
        .await,
    );
    let (s, v) = call(
        &app,
        PLATFORM,
        "DELETE",
        &format!("/api/stores/{root}?recursive=true"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT, "{v}");
    assert_eq!(listed(&app, PLATFORM).await, ["base@2"]);
}

#[tokio::test]
async fn a_published_packs_store_providers_are_frozen() {
    let (app, _pool, _db) = app(true).await;
    let root = store(&app, PLATFORM, None, "router", "base v1").await;
    let mid = store(&app, PLATFORM, Some(root), "router", "hr").await;
    let leaf = store(&app, PLATFORM, Some(mid), "leaf", "leave").await;
    register(&app, "base", 1, root).await;
    let uri = format!("/api/stores/{leaf}/providers");

    // control: while the pack is a draft the freeze does not fire (this engine has fixed
    // providers, so the write then fails further on, in the provider registry)
    for (method, body) in [("PUT", Some(json!({}))), ("DELETE", None)] {
        let (s, v) = call(&app, PLATFORM, method, &uri, body).await;
        assert_ne!(code(&v), "pack_published", "{method} on a draft: {s} {v}");
        assert_ne!(s, StatusCode::CONFLICT, "{method} on a draft: {v}");
    }

    publish(&app, "base", 1).await;
    let frozen = |(s, v): (StatusCode, Value)| {
        assert_eq!(
            (s, code(&v)),
            (StatusCode::CONFLICT, "pack_published"),
            "{v}"
        );
    };
    for target in [leaf, mid, root] {
        let uri = format!("/api/stores/{target}/providers");
        frozen(call(&app, PLATFORM, "PUT", &uri, Some(json!({}))).await);
        // a non-empty set is refused by the freeze too, before any validation or live probe
        frozen(
            call(
                &app,
                PLATFORM,
                "PUT",
                &uri,
                Some(json!({"rerank": {
                    "base_url": "http://127.0.0.1:9", "dialect": "cohere", "model": "rr",
                    "regions": ["eu"], "zdr": true
                }})),
            )
            .await,
        );
        frozen(call(&app, PLATFORM, "DELETE", &uri, None).await);
    }
}

#[tokio::test]
async fn deleting_a_draft_root_drops_the_draft() {
    let (app, _pool, _db) = app(true).await;
    let root = store(&app, PLATFORM, None, "router", "draft").await;
    register(&app, "job/store_manager", 1, root).await;
    let (s, v) = call(
        &app,
        PLATFORM,
        "DELETE",
        &format!("/api/stores/{root}"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT, "{v}");
    assert!(listed(&app, PLATFORM).await.is_empty());
}

#[tokio::test]
async fn readable_packs_owner_sees_drafts_and_tenants_keep_retired_pins() {
    let (app, pool, _db) = app(true).await;
    let a = store(&app, PLATFORM, None, "router", "a").await;
    let b = store(&app, PLATFORM, None, "router", "b").await;
    let c = store(&app, PLATFORM, None, "router", "c").await;
    register(&app, "base", 1, a).await;
    register(&app, "base", 2, b).await;
    register(&app, "base", 3, c).await;
    publish(&app, "base", 1).await;
    publish(&app, "base", 2).await;
    let (s, _) = call(&app, PLATFORM, "POST", &pack_uri("base", 1, "retire"), None).await;
    assert_eq!(s, StatusCode::OK);
    // a draft cannot be retired
    let (s, v) = call(&app, PLATFORM, "POST", &pack_uri("base", 3, "retire"), None).await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::UNPROCESSABLE_ENTITY, "invalid_target"),
        "{v}"
    );

    let owner = readable_packs(&pool, common::platform(), common::platform())
        .await
        .unwrap();
    let v: Vec<i32> = owner.iter().map(|p| p.version).collect();
    assert_eq!(v, [1, 2, 3]);
    // a tenant reads every published pack, the retired one included (its pins keep working,
    // spec §3), but never a draft
    let tenant = readable_packs(&pool, common::platform(), common::tn())
        .await
        .unwrap();
    let v: Vec<(i32, Uuid)> = tenant.iter().map(|p| (p.version, p.root_id)).collect();
    assert_eq!(v, [(1, a), (2, b)]);
    assert!(tenant[0].retired_at.is_some() && tenant[1].retired_at.is_none());
    assert!(tenant.iter().all(|p| p.published_at.is_some()));
    // ... while the listing hides the retired one from tenants, not from the owner
    assert_eq!(listed(&app, TENANT).await, ["base@2"]);
    assert_eq!(listed(&app, PLATFORM).await, ["base@1", "base@2", "base@3"]);
}

/// `(path joined by "/", kind)` of every store in a pack-stores body.
fn store_paths(v: &Value) -> Vec<(String, String)> {
    v["stores"]
        .as_array()
        .expect("stores")
        .iter()
        .map(|s| {
            let path: Vec<&str> = s["path"]
                .as_array()
                .expect("path")
                .iter()
                .map(|n| n.as_str().expect("name"))
                .collect();
            (
                path.join("/"),
                s["kind"].as_str().expect("kind").to_string(),
            )
        })
        .collect()
}

#[tokio::test]
async fn a_tenant_lists_a_readable_packs_stores_and_queries_its_leaves() {
    let (app, _pool, _db) = app(true).await;
    let root = store(&app, PLATFORM, None, "router", "employment").await;
    let hr = store(&app, PLATFORM, Some(root), "router", "hr").await;
    let leave = store(&app, PLATFORM, Some(hr), "leaf", "leave").await;
    let pay = store(&app, PLATFORM, Some(root), "leaf", "pay").await;
    let (s, v) = call(
        &app,
        PLATFORM,
        "POST",
        &format!("/api/stores/{leave}/documents"),
        Some(json!({
            "text": "statutory annual leave is twenty working days per year",
            "topic_keys": ["hr/leave"],
            "provenance": {
                "source_url": "https://example.org/leave",
                "license": "CC0-1.0",
                "retrieved_at": "2026-10-01T12:00:00Z"
            }
        })),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    // a store of the platform owner outside the pack is not listed
    store(&app, PLATFORM, None, "router", "elsewhere").await;
    register(&app, "industry/retail", 1, root).await;
    let uri = pack_uri("industry/retail", 1, "stores");

    // a draft: the owner sees it, a tenant gets 404 as for a pack that does not exist
    let (s, v) = call(&app, PLATFORM, "GET", &uri, None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let (s, v) = call(&app, TENANT, "GET", &uri, None).await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");

    publish(&app, "industry/retail", 1).await;
    let (s, v) = call(&app, TENANT, "GET", &uri, None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["pack"], "industry/retail@1");
    assert_eq!(v["root_id"], root.to_string());
    assert_eq!(
        store_paths(&v),
        [
            ("employment".to_string(), "router".to_string()),
            ("employment/hr".to_string(), "router".to_string()),
            ("employment/hr/leave".to_string(), "leaf".to_string()),
            ("employment/pay".to_string(), "leaf".to_string()),
        ]
    );
    let stores = v["stores"].as_array().unwrap();
    assert_eq!(stores[0]["id"], root.to_string());
    assert!(stores[0]["parent_id"].is_null(), "{v}");
    assert_eq!(stores[1]["id"], hr.to_string());
    assert_eq!(stores[2]["id"], leave.to_string());
    assert_eq!(stores[2]["parent_id"], hr.to_string());
    assert_eq!(stores[2]["name"], "leave");
    assert!(stores[2]["chunk_count"].as_i64().unwrap() >= 1, "{v}");
    assert_eq!(stores[3]["id"], pay.to_string());
    assert_eq!(stores[3]["chunk_count"], 0);
    assert!(stores.iter().all(|s| s["embed_model"].is_string()), "{v}");
    assert!(
        stores.iter().all(|s| s.get("description").is_none()
            && s.get("desc_vec").is_none()
            && s.get("profile_vec").is_none()),
        "no description or vectors: {v}"
    );

    // the listed leaf ids are accepted by a query's leaf_ids
    let leaves: Vec<&str> = stores
        .iter()
        .filter(|s| s["kind"] == "leaf")
        .map(|s| s["id"].as_str().unwrap())
        .collect();
    let (s, q) = call(
        &app,
        TENANT,
        "POST",
        "/api/query",
        Some(json!({"query": "annual leave", "leaf_ids": leaves, "top_k": 5})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{q}");
    let rs = q["results"].as_array().expect("results");
    assert!(!rs.is_empty(), "{q}");
    assert!(rs.iter().all(|r| r["pack"] == "industry/retail@1"), "{q}");

    // retired: still listed for existing pins
    let (s, _) = call(
        &app,
        PLATFORM,
        "POST",
        &pack_uri("industry/retail", 1, "retire"),
        None,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let (s, v) = call(&app, TENANT, "GET", &uri, None).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(store_paths(&v).len(), 4);

    // excluded: 404 for that tenant only
    let (s, v) = call(
        &app,
        TENANT,
        "PUT",
        "/api/platform/exclusions",
        Some(json!({"key_prefix": "industry", "reason": "not our sector"})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let (s, v) = call(&app, TENANT, "GET", &uri, None).await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");
    let (s, v) = call(&app, OTHER_TENANT, "GET", &uri, None).await;
    assert_eq!(s, StatusCode::OK, "{v}");

    // unknown pack / version
    for uri in [
        pack_uri("industry/retail", 2, "stores"),
        pack_uri("nope", 1, "stores"),
    ] {
        let (s, v) = call(&app, PLATFORM, "GET", &uri, None).await;
        assert_eq!(
            (s, code(&v)),
            (StatusCode::NOT_FOUND, "not_found"),
            "{uri}: {v}"
        );
    }
}
