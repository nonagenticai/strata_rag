//! Tenant + source scoping (spec §8.6): token → tenant resolution over HTTP, isolation of every
//! tenanted route, the per-tenant routing cache, and SQL-side query filters.

mod common;

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{Request, StatusCode};
use axum::routing::post;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use strata_rag::Tenant;
use strata_rag::ask::AskRequest;
use strata_rag::config::{ApiToken, TokenScope};
use strata_rag::db::search::{ChunkFilter, dense_search, fetch_chunks, lexical_search};
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::error::StrataError;
use strata_rag::ingest::{IngestOutcome, IngestRequest, SourceTag};
use strata_rag::query::QueryRequest;
use strata_rag::routing::RouteMode;
use tower::ServiceExt;
use uuid::Uuid;

use common::{OTHER_TENANT, TENANT, other, test_engine, tn};

const SVC: &str = "svc-token";
const TOK_A: &str = "tenant-a-token";
const TOK_B: &str = "tenant-b-token";

/// `*:SVC`, `A:TOK_A`, `B:TOK_B`.
fn tokens() -> Vec<ApiToken> {
    vec![
        ApiToken::new(TokenScope::AnyTenant, SVC),
        ApiToken::new(TokenScope::Tenant(tn()), TOK_A),
        ApiToken::new(TokenScope::Tenant(other()), TOK_B),
    ]
}

/// Sends `req` exactly as built (no default tenant header).
async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
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

fn req(
    method: &str,
    uri: &str,
    token: Option<&str>,
    tenant: Option<&str>,
    body: Option<Value>,
) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(t) = token {
        b = b.header(AUTHORIZATION, format!("Bearer {t}"));
    }
    if let Some(t) = tenant {
        b = b.header("x-tenant-id", t);
    }
    match body {
        Some(v) => b
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

/// A caller acting for one tenant with its pinned token.
#[derive(Clone, Copy)]
struct As<'a> {
    token: &'a str,
}

impl As<'_> {
    fn r(&self, method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
        req(method, uri, Some(self.token), None, body)
    }
}

fn code(v: &Value) -> &str {
    v["error"]["code"].as_str().unwrap_or("")
}

fn id_of(v: &Value) -> Uuid {
    v["id"].as_str().expect("id").parse().expect("uuid")
}

// ---------- auth: token → tenant ----------

#[tokio::test]
async fn token_and_header_resolve_the_tenant() {
    let (mut engine, _db) = test_engine().await;
    engine.cfg.api_tokens = tokens();
    let app = strata_rag::api::router(Arc::new(engine));
    let list =
        |token: Option<&str>, tenant: Option<&str>| req("GET", "/api/stores", token, tenant, None);

    // pinned token alone, or with its own tenant repeated
    assert_eq!(send(&app, list(Some(TOK_A), None)).await.0, StatusCode::OK);
    assert_eq!(
        send(&app, list(Some(TOK_A), Some(TENANT))).await.0,
        StatusCode::OK
    );
    // pinned token naming another tenant
    let (s, v) = send(&app, list(Some(TOK_A), Some(OTHER_TENANT))).await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::FORBIDDEN, "tenant_mismatch"),
        "{v}"
    );
    // `*` token must name a tenant
    let (s, v) = send(&app, list(Some(SVC), None)).await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::BAD_REQUEST, "tenant_required"),
        "{v}"
    );
    assert_eq!(
        send(&app, list(Some(SVC), Some(OTHER_TENANT))).await.0,
        StatusCode::OK
    );
    // a licence token in the tenant header
    let bearer = format!("Bearer {TENANT}");
    let (s, v) = send(&app, list(Some(SVC), Some(&bearer))).await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::BAD_REQUEST, "invalid_tenant"),
        "{v}"
    );
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("licence token"),
        "{v}"
    );
    for bad in ["nope", "00000000-0000-0000-0000-000000000000"] {
        let (s, v) = send(&app, list(Some(SVC), Some(bad))).await;
        assert_eq!(
            (s, code(&v)),
            (StatusCode::BAD_REQUEST, "invalid_tenant"),
            "{bad}: {v}"
        );
    }
    // no token: 401 before any tenant question, also with a header
    for tenant in [None, Some(TENANT)] {
        let (s, v) = send(&app, list(None, tenant)).await;
        assert_eq!(
            (s, code(&v)),
            (StatusCode::UNAUTHORIZED, "unauthorized"),
            "{v}"
        );
    }
    // /health stays public and untenanted
    assert_eq!(
        send(&app, req("GET", "/health", None, None, None)).await.0,
        StatusCode::OK
    );
}

#[tokio::test]
async fn open_mode_still_requires_a_tenant() {
    let (engine, _db) = test_engine().await;
    assert!(engine.cfg.api_tokens.is_empty());
    let app = strata_rag::api::router(Arc::new(engine));
    let (s, v) = send(&app, req("GET", "/api/stores", None, None, None)).await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::BAD_REQUEST, "tenant_required"),
        "{v}"
    );
    let (s, v) = send(
        &app,
        req("POST", "/api/query", None, None, Some(json!({"query":"x"}))),
    )
    .await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::BAD_REQUEST, "tenant_required"),
        "{v}"
    );
    assert_eq!(
        send(&app, req("GET", "/api/stores", None, Some(TENANT), None))
            .await
            .0,
        StatusCode::OK
    );
    assert_eq!(
        send(&app, req("GET", "/health", None, None, None)).await.0,
        StatusCode::OK
    );
}

// ---------- isolation of every tenanted route ----------

/// Tenant A's tree over HTTP: root router `lib` → leaf `sea` with one document of source `s1`.
struct ATree {
    root: Uuid,
    leaf: Uuid,
    doc: Uuid,
}

async fn a_tree(app: &Router, a: As<'_>) -> ATree {
    let (s, v) = send(
        app,
        a.r(
            "POST",
            "/api/stores",
            Some(json!({"kind":"router","name":"lib","description":"books"})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let root = id_of(&v);
    let (s, v) = send(
        app,
        a.r(
            "POST",
            "/api/stores",
            Some(json!({"parent_id":root,"kind":"leaf","name":"sea","description":"whales ships"})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let leaf = id_of(&v);
    let (s, v) = send(
        app,
        a.r(
            "POST",
            &format!("/api/stores/{leaf}/documents"),
            Some(json!({"text":"the white whale breached beside the ship","source_id":"s1","source_run":"r1"})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    ATree {
        root,
        leaf,
        doc: id_of(&v),
    }
}

#[tokio::test]
async fn another_tenants_stores_documents_and_chunks_are_invisible() {
    let (mut engine, _db) = test_engine().await;
    engine.cfg.api_tokens = tokens();
    let app = strata_rag::api::router(Arc::new(engine));
    let a = As { token: TOK_A };
    let b = As { token: TOK_B };
    let t = a_tree(&app, a).await;

    // B sees nothing
    let (s, v) = send(&app, b.r("GET", "/api/stores", None)).await;
    assert_eq!((s, v.clone()), (StatusCode::OK, json!({"stores": []})));

    // every route given one of A's ids answers B 404, never 403 (existence does not leak)
    let leaf = t.leaf;
    let cases: Vec<(&str, String, Option<Value>)> = vec![
        ("GET", format!("/api/stores/{leaf}"), None),
        (
            "PATCH",
            format!("/api/stores/{leaf}"),
            Some(json!({"name":"stolen"})),
        ),
        ("DELETE", format!("/api/stores/{leaf}"), None),
        (
            "DELETE",
            format!("/api/stores/{}?recursive=true", t.root),
            None,
        ),
        (
            "POST",
            format!("/api/stores/{leaf}/documents"),
            Some(json!({"text":"injected into another tenant"})),
        ),
        ("GET", format!("/api/stores/{leaf}/documents"), None),
        ("DELETE", format!("/api/documents/{}", t.doc), None),
        (
            "DELETE",
            format!("/api/sources/s1/documents?all=true&store_id={leaf}"),
            None,
        ),
        (
            "POST",
            "/api/stores".to_string(),
            Some(json!({"parent_id":t.root,"kind":"leaf","name":"x","description":"y"})),
        ),
        (
            "POST",
            "/api/query".to_string(),
            Some(json!({"query":"white whale","root_id":t.root})),
        ),
    ];
    for (m, uri, body) in cases {
        let (s, v) = send(&app, b.r(m, &uri, body)).await;
        assert_eq!(
            (s, code(&v)),
            (StatusCode::NOT_FOUND, "not_found"),
            "{m} {uri}: {v}"
        );
    }
    // a prune of the same source_id without store_id touches only B's (no) documents
    let (s, v) = send(
        &app,
        b.r("DELETE", "/api/sources/s1/documents?all=true", None),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        (v["deleted"].as_u64(), v["kept"].as_i64()),
        (Some(0), Some(0)),
        "{v}"
    );
    // B's query from its (empty) tree: nothing
    let (s, v) = send(
        &app,
        b.r("POST", "/api/query", Some(json!({"query":"white whale"}))),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["results"], json!([]), "{v}");
    // B may reuse A's root name (decision 1: names are unique per tenant)
    let (s, v) = send(
        &app,
        b.r(
            "POST",
            "/api/stores",
            Some(json!({"kind":"leaf","name":"lib","description":"b's own"})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");

    // nothing of A's changed
    let (s, v) = send(&app, a.r("GET", &format!("/api/stores/{leaf}"), None)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["name"], "sea");
    assert_eq!(v["chunk_count"], 1);
    let (_, v) = send(
        &app,
        a.r("GET", &format!("/api/stores/{leaf}/documents"), None),
    )
    .await;
    assert_eq!(v["items"].as_array().unwrap().len(), 1, "{v}");
    assert_eq!(v["items"][0]["id"], json!(t.doc));
    let (_, v) = send(&app, a.r("GET", "/api/stores", None)).await;
    assert_eq!(v["stores"].as_array().unwrap().len(), 2, "{v}");
    let (s, v) = send(
        &app,
        a.r("POST", "/api/query", Some(json!({"query":"white whale"}))),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["results"][0]["document_id"], json!(t.doc), "{v}");
    // and A's own delete of the document still works
    let (s, _) = send(
        &app,
        a.r("DELETE", &format!("/api/documents/{}", t.doc), None),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
}

/// Two tenants with overlapping content: each query (with and without root_id) and each ask
/// returns only the caller's chunks.
#[tokio::test]
async fn query_and_ask_never_return_another_tenants_chunks() {
    let (engine, _db) = test_engine().await;
    let mut leaves = Vec::new();
    for (tenant, text) in [
        (tn(), "whale harpoon ship ocean tenant alpha"),
        (other(), "whale harpoon ship ocean tenant beta"),
    ] {
        let root = engine
            .create_store(tenant, new_store(None, StoreKind::Router, "lib"))
            .await
            .unwrap()
            .id;
        let leaf = engine
            .create_store(tenant, new_store(Some(root), StoreKind::Leaf, "sea"))
            .await
            .unwrap()
            .id;
        engine
            .ingest(tenant, leaf, "doc".into(), None, text)
            .await
            .unwrap();
        leaves.push((tenant, root, leaf));
    }
    let base = common::mock::spawn(Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            axum::Json(
                json!({"choices":[{"message":{"role":"assistant","content":"A whale [1]."}}]}),
            )
        }),
    ))
    .await;
    let engine = engine.with_chat(common::chat_at(&base, 5));

    for &(tenant, root, leaf) in &leaves {
        let mine = |text: &str| text.contains(if tenant == tn() { "alpha" } else { "beta" });
        for root_id in [None, Some(root), Some(leaf)] {
            let req = QueryRequest {
                root_id,
                route_mode: RouteMode::Embed,
                min_score: -1.0,
                top_k: 10,
                ..QueryRequest::new("whale harpoon ship ocean")
            };
            let resp = engine.query(tenant, &req).await.unwrap();
            assert!(!resp.results.is_empty(), "{root_id:?}");
            assert!(
                resp.results.iter().all(|r| mine(&r.text)),
                "{tenant} {root_id:?}"
            );
            let ask = engine
                .ask(
                    tenant,
                    &AskRequest {
                        query: req,
                        max_tokens: 50,
                        instructions: None,
                    },
                )
                .await
                .unwrap();
            assert!(ask.results.iter().all(|r| mine(&r.text)), "ask {tenant}");
        }
        // the other tenant's root and leaf ids are unknown here
        for &(t2, root2, leaf2) in &leaves {
            if t2 == tenant {
                continue;
            }
            for id in [root2, leaf2] {
                let req = QueryRequest {
                    root_id: Some(id),
                    ..QueryRequest::new("whale")
                };
                let err = engine.query(tenant, &req).await.unwrap_err();
                assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");
            }
        }
    }
}

/// The routing cache is keyed by tenant: B's query right after A's tree was cached routes over
/// B's tree, never A's.
#[tokio::test]
async fn routing_cache_never_serves_one_tenants_tree_to_another() {
    let (engine, _db) = test_engine().await;
    let a_root = engine
        .create_store(tn(), new_store(None, StoreKind::Router, "lib"))
        .await
        .unwrap()
        .id;
    let a_leaf = engine
        .create_store(tn(), new_store(Some(a_root), StoreKind::Leaf, "sea"))
        .await
        .unwrap()
        .id;
    engine
        .ingest(tn(), a_leaf, "a".into(), None, "whale harpoon ship")
        .await
        .unwrap();
    let req = QueryRequest {
        route_mode: RouteMode::Embed,
        min_score: -1.0,
        ..QueryRequest::new("whale harpoon ship")
    };
    let a = engine.query(tn(), &req).await.unwrap();
    assert_eq!(a.results.len(), 1);
    assert_eq!(
        engine.cache.cached_tenants().await,
        vec![tn()],
        "A's tree is cached"
    );

    // B has no stores: an empty answer, not A's cached tree
    let b = engine.query(other(), &req).await.unwrap();
    assert!(b.results.is_empty() && b.routing.leaves.is_empty(), "{b:?}");
    // B gets a tree of its own: its query routes over it, A's cache entry untouched
    let b_leaf = engine
        .create_store(other(), new_store(None, StoreKind::Leaf, "solo"))
        .await
        .unwrap()
        .id;
    engine
        .ingest(other(), b_leaf, "b".into(), None, "whale harpoon ship beta")
        .await
        .unwrap();
    let b = engine.query(other(), &req).await.unwrap();
    assert_eq!(b.routing.leaves, vec![b_leaf]);
    assert!(b.results.iter().all(|r| r.text.contains("beta")), "{b:?}");
    assert_eq!(engine.cache.cached_tenants().await.len(), 2);
    let a2 = engine.query(tn(), &req).await.unwrap();
    assert_eq!(a2.routing.leaves, vec![a_leaf]);
}

fn new_store(parent_id: Option<Uuid>, kind: StoreKind, name: &str) -> NewStore {
    NewStore {
        parent_id,
        kind,
        name: name.to_string(),
        description: format!("{name} whale harpoon ship"),
    }
}

/// Engine-level twin of the HTTP 404s for the paths the HTTP test cannot reach directly
/// (`delete_store`, `update_store`, `prune_source` and `ingest` via the engine API).
#[tokio::test]
async fn engine_operations_on_another_tenants_ids_are_not_found() {
    let (engine, _db) = test_engine().await;
    let root = engine
        .create_store(tn(), new_store(None, StoreKind::Router, "lib"))
        .await
        .unwrap()
        .id;
    let leaf = engine
        .create_store(tn(), new_store(Some(root), StoreKind::Leaf, "sea"))
        .await
        .unwrap()
        .id;
    let IngestOutcome::Created(doc) = engine
        .ingest_document(
            tn(),
            IngestRequest {
                store_id: leaf,
                source: "x".into(),
                title: None,
                text: "whale",
                tag: SourceTag {
                    source_id: Some("s1".into()),
                    source_run: Some("r1".into()),
                    metadata: None,
                },
            },
        )
        .await
        .unwrap()
    else {
        panic!("created")
    };
    let b = other();
    let nf = |e: StrataError| assert!(matches!(e, StrataError::NotFound(_)), "{e:?}");
    nf(engine.get_store(b, leaf).await.unwrap_err());
    nf(engine
        .update_store(b, leaf, Some("x".into()), None)
        .await
        .unwrap_err());
    nf(engine.delete_store(b, root, true).await.unwrap_err());
    nf(engine.delete_store(b, leaf, false).await.unwrap_err());
    nf(engine
        .ingest(b, leaf, "x".into(), None, "y")
        .await
        .unwrap_err());
    nf(engine.delete_document(b, doc.id).await.unwrap_err());
    nf(engine
        .create_store(b, new_store(Some(root), StoreKind::Leaf, "x"))
        .await
        .unwrap_err());
    let report = engine
        .prune_source(
            b,
            strata_rag::ingest::PruneRequest {
                source_id: "s1".into(),
                all: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!((report.deleted, report.kept), (0, 0));
    assert!(engine.store_tree(b).await.unwrap().is_empty());
    // A: untouched
    let view = engine.get_store(tn(), leaf).await.unwrap();
    assert_eq!((view.name.as_str(), view.chunk_count), ("sea", 1));
    assert_eq!(engine.store_tree(tn()).await.unwrap().len(), 2);
}

// ---------- query filters ----------

/// 60 decoy documents (source `x`, lang en) that match the query better than the 12 wanted ones
/// (source `y`, lang nl), all in one leaf. The unfiltered candidate list (`CANDIDATES` = 50 rows)
/// holds no wanted chunk, so a filter applied after the search would return nothing; a filter in
/// SQL still returns a full `top_k` of wanted chunks.
async fn filter_fixture(engine: &Engine, tenant: Tenant) -> Uuid {
    let leaf = engine
        .create_store(tenant, new_store(None, StoreKind::Leaf, "mixed"))
        .await
        .unwrap()
        .id;
    let docs = (0..60)
        .map(|i| (format!("harbor lighthouse beacon d{i}"), "x", "en"))
        .chain((0..12).map(|i| {
            (
                format!("harbor tide kelp gull pier rope net crab w{i}"),
                "y",
                "nl",
            )
        }));
    for (text, source, lang) in docs {
        engine
            .ingest_document(
                tenant,
                IngestRequest {
                    store_id: leaf,
                    source: text.clone(),
                    title: None,
                    text: &text,
                    tag: SourceTag {
                        source_id: Some(source.into()),
                        source_run: None,
                        metadata: Some(json!({"lang": lang, "n": 1})),
                    },
                },
            )
            .await
            .unwrap();
    }
    leaf
}

#[tokio::test]
async fn filters_apply_in_sql_and_still_fill_top_k() {
    let (engine, _db) = test_engine().await;
    let leaf = filter_fixture(&engine, tn()).await;
    let base = QueryRequest {
        root_id: Some(leaf),
        top_k: 50,
        ..QueryRequest::new("harbor lighthouse beacon")
    };
    // the premise: the unfiltered candidates are all decoys
    let all = engine.query(tn(), &base).await.unwrap();
    assert_eq!(all.results.len(), 50);
    assert!(
        all.results
            .iter()
            .all(|r| r.source_id.as_deref() == Some("x")),
        "fixture: no wanted chunk among the unfiltered top 50"
    );
    for (name, req) in [
        (
            "source_ids",
            QueryRequest {
                top_k: 10,
                source_ids: Some(vec!["y".into()]),
                ..base.clone()
            },
        ),
        (
            "metadata",
            QueryRequest {
                top_k: 10,
                metadata: Some(json!({"lang": "nl"})),
                ..base.clone()
            },
        ),
        (
            "both",
            QueryRequest {
                top_k: 10,
                source_ids: Some(vec!["y".into(), "nope".into()]),
                metadata: Some(json!({"lang": "nl", "n": 1})),
                ..base.clone()
            },
        ),
    ] {
        let resp = engine.query(tn(), &req).await.unwrap();
        assert_eq!(resp.results.len(), 10, "{name}: a full top_k");
        assert!(
            resp.results
                .iter()
                .all(|r| r.source_id.as_deref() == Some("y") && r.metadata["lang"] == "nl"),
            "{name}"
        );
    }
    // a filter nothing matches
    let none = QueryRequest {
        metadata: Some(json!({"lang": "fr"})),
        ..base.clone()
    };
    assert!(engine.query(tn(), &none).await.unwrap().results.is_empty());
    // invalid filters are 422
    for bad in [
        QueryRequest {
            source_ids: Some(vec![]),
            ..base.clone()
        },
        QueryRequest {
            metadata: Some(json!(["lang"])),
            ..base.clone()
        },
    ] {
        let err = engine.query(tn(), &bad).await.unwrap_err();
        assert!(matches!(err, StrataError::InvalidInput(_)), "{err:?}");
    }
}

/// `/api/ask` takes the same filters (its body flattens the query request).
#[tokio::test]
async fn ask_passes_filters_through() {
    let (engine, _db) = test_engine().await;
    let leaf = filter_fixture(&engine, tn()).await;
    let base = common::mock::spawn(Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            axum::Json(json!({"choices":[{"message":{"role":"assistant","content":"Kelp [1]."}}]}))
        }),
    ))
    .await;
    let engine = Arc::new(engine.with_chat(common::chat_at(&base, 5)));
    let app = strata_rag::api::router(engine);
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/ask",
            None,
            Some(TENANT),
            Some(json!({
                "query": "harbor lighthouse beacon",
                "root_id": leaf,
                "top_k": 5,
                "source_ids": ["y"],
            })),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 5, "{v}");
    assert!(results.iter().all(|r| r["source_id"] == "y"), "{v}");
}

// ---------- review round: direct DB helpers, cursors, ask/eval over HTTP, writes, slots ----------

/// The search helpers are tenant-scoped in SQL: handed another tenant's leaf or chunk ids
/// directly (no routing in front), they return nothing.
#[tokio::test]
async fn search_helpers_return_nothing_for_another_tenants_ids() {
    let (engine, _db) = test_engine().await;
    let leaf = engine
        .create_store(tn(), new_store(None, StoreKind::Leaf, "sea"))
        .await
        .unwrap()
        .id;
    engine
        .ingest(tn(), leaf, "a".into(), None, "whale harpoon ship")
        .await
        .unwrap();
    let q = engine
        .inference(tn())
        .await
        .unwrap()
        .embed_one("whale harpoon ship".into())
        .await
        .unwrap();
    let f = ChunkFilter::default();
    let filtered = ChunkFilter {
        source_ids: None,
        metadata: Some(json!({})),
    };
    // the owner finds the chunk, through both the plain and the filtered statements
    for filter in [&f, &filtered] {
        let dense = dense_search(&engine.pool, tn(), &[leaf], &q, 10, filter)
            .await
            .unwrap();
        assert_eq!(dense.len(), 1);
        let lexical = lexical_search(&engine.pool, tn(), &[leaf], "whale", 10, filter)
            .await
            .unwrap();
        assert_eq!(lexical.len(), 1);
        assert_eq!(
            fetch_chunks(&engine.pool, tn(), &[dense[0].0])
                .await
                .unwrap()
                .len(),
            1
        );
        // another tenant, same ids: nothing
        assert!(
            dense_search(&engine.pool, other(), &[leaf], &q, 10, filter)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            lexical_search(&engine.pool, other(), &[leaf], "whale", 10, filter)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            fetch_chunks(&engine.pool, other(), &[dense[0].0])
                .await
                .unwrap()
                .is_empty()
        );
    }
}

/// A document-list cursor taken from tenant A's store: B gets 404 on A's store, and on B's own
/// store the cursor only positions the page, it never yields A's rows.
#[tokio::test]
async fn a_document_cursor_never_crosses_tenants() {
    let (mut engine, _db) = test_engine().await;
    engine.cfg.api_tokens = tokens();
    let app = strata_rag::api::router(Arc::new(engine));
    let (a, b) = (As { token: TOK_A }, As { token: TOK_B });
    let mut leaves = Vec::new();
    for (who, tag) in [(a, "alpha"), (b, "beta")] {
        let (s, v) = send(
            &app,
            who.r(
                "POST",
                "/api/stores",
                Some(json!({"kind":"leaf","name":"docs","description":"d"})),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{v}");
        let leaf = id_of(&v);
        for i in 0..3 {
            let (s, v) = send(
                &app,
                who.r(
                    "POST",
                    &format!("/api/stores/{leaf}/documents"),
                    Some(json!({"text": format!("{tag} document number {i}")})),
                ),
            )
            .await;
            assert_eq!(s, StatusCode::CREATED, "{v}");
        }
        leaves.push(leaf);
    }
    let (a_leaf, b_leaf) = (leaves[0], leaves[1]);
    let (s, v) = send(
        &app,
        a.r(
            "GET",
            &format!("/api/stores/{a_leaf}/documents?limit=1"),
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let cursor = v["next"].as_str().expect("a next cursor").to_string();

    let (s, v) = send(
        &app,
        b.r(
            "GET",
            &format!("/api/stores/{a_leaf}/documents?cursor={cursor}"),
            None,
        ),
    )
    .await;
    assert_eq!((s, code(&v)), (StatusCode::NOT_FOUND, "not_found"), "{v}");
    let (s, v) = send(
        &app,
        b.r(
            "GET",
            &format!("/api/stores/{b_leaf}/documents?cursor={cursor}&limit=500"),
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    for item in v["items"].as_array().unwrap() {
        assert_eq!(item["store_id"], json!(b_leaf), "{v}");
    }
}

/// A pinned token naming another tenant on a WRITE route is 403 and writes nothing.
#[tokio::test]
async fn a_mismatched_tenant_header_on_a_write_is_403_and_writes_nothing() {
    let (mut engine, _db) = test_engine().await;
    engine.cfg.api_tokens = tokens();
    let engine = Arc::new(engine);
    let app = strata_rag::api::router(Arc::clone(&engine));
    let a = As { token: TOK_A };
    let t = a_tree(&app, a).await;
    let writes: Vec<(&str, String, Value)> = vec![
        (
            "POST",
            "/api/stores".into(),
            json!({"kind":"router","name":"evil","description":"x"}),
        ),
        (
            "POST",
            "/api/stores".into(),
            json!({"parent_id":t.root,"kind":"leaf","name":"evil","description":"x"}),
        ),
        (
            "POST",
            format!("/api/stores/{}/documents", t.leaf),
            json!({"text":"written under the wrong tenant"}),
        ),
    ];
    for (m, uri, body) in writes {
        let (s, v) = send(
            &app,
            req(m, &uri, Some(TOK_A), Some(OTHER_TENANT), Some(body)),
        )
        .await;
        assert_eq!(
            (s, code(&v)),
            (StatusCode::FORBIDDEN, "tenant_mismatch"),
            "{m} {uri}: {v}"
        );
    }
    assert!(engine.store_tree(other()).await.unwrap().is_empty());
    let mine = engine.store_tree(tn()).await.unwrap();
    assert_eq!(mine.len(), 2, "no store added for A either");
    assert!(mine.iter().all(|v| v.name != "evil"));
    let docs: i64 = sqlx::query_scalar("SELECT count(*) FROM documents")
        .fetch_one(&engine.pool)
        .await
        .unwrap();
    assert_eq!(docs, 1, "only A's own document");
}

/// `/api/ask` over HTTP: each tenant's answer is built from its own chunks only.
#[tokio::test]
async fn ask_over_http_is_isolated() {
    let (mut engine, _db) = test_engine().await;
    engine.cfg.api_tokens = tokens();
    let base = common::mock::spawn(Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            axum::Json(
                json!({"choices":[{"message":{"role":"assistant","content":"Whales [1]."}}]}),
            )
        }),
    ))
    .await;
    let engine = engine.with_chat(common::chat_at(&base, 5));
    for (tenant, tag) in [(tn(), "alpha"), (other(), "beta")] {
        let leaf = engine
            .create_store(tenant, new_store(None, StoreKind::Leaf, "sea"))
            .await
            .unwrap()
            .id;
        engine
            .ingest(
                tenant,
                leaf,
                tag.into(),
                None,
                &format!("whale harpoon ship {tag}"),
            )
            .await
            .unwrap();
    }
    let app = strata_rag::api::router(Arc::new(engine));
    for (token, tenant, tag) in [
        (TOK_A, None, "alpha"),
        (TOK_B, None, "beta"),
        (SVC, Some(TENANT), "alpha"),
        (SVC, Some(OTHER_TENANT), "beta"),
    ] {
        let (s, v) = send(
            &app,
            req(
                "POST",
                "/api/ask",
                Some(token),
                tenant,
                Some(json!({"query":"whale harpoon ship","route_mode":"embed","min_score":-1.0})),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let results = v["results"].as_array().unwrap();
        assert!(!results.is_empty(), "{v}");
        assert!(
            results
                .iter()
                .all(|r| r["text"].as_str().unwrap().contains(tag)),
            "{token}: {v}"
        );
    }
}

const EVAL_TREE: &str = r#"{"corpus_dir":"corpus","root":{"name":"lib","kind":"router","description":"books","children":[
    {"name":"sea","kind":"leaf","description":"whales harpoons ships ocean","files":["sea.txt"]},
    {"name":"stars","kind":"leaf","description":"astronomy telescope planets orbit","files":["stars.txt"]}]}}"#;

const EVAL_QUESTIONS: &str = concat!(
    r#"{"query":"harpoon whale ship","expected_document":"sea.txt","expected_leaf":"sea"}"#,
    "\n",
    r#"{"query":"telescope planet orbit","expected_document":"stars.txt","expected_leaf":"stars"}"#,
    "\n"
);

/// `/api/eval` over HTTP: it scores the caller's tenant only. A loads the tree; B (nothing
/// loaded) is told so, and B's stray stores never contaminate A's report.
#[tokio::test]
async fn eval_over_http_is_isolated() {
    let (mut engine, _db) = test_engine().await;
    engine.cfg.api_tokens = tokens();
    let dir = std::env::temp_dir().join(format!("strata-tenancy-eval-{}", Uuid::now_v7()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tree.json"), EVAL_TREE).unwrap();
    std::fs::write(dir.join("questions.jsonl"), EVAL_QUESTIONS).unwrap();
    engine.cfg.eval_questions = dir.join("questions.jsonl");
    let app = strata_rag::api::router(Arc::new(engine));
    let (a, b) = (As { token: TOK_A }, As { token: TOK_B });

    let create = |who: As<'static>, body: Value| {
        let app = app.clone();
        async move {
            let (s, v) = send(&app, who.r("POST", "/api/stores", Some(body))).await;
            assert_eq!(s, StatusCode::CREATED, "{v}");
            id_of(&v)
        }
    };
    let root = create(
        a,
        json!({"kind":"router","name":"lib","description":"books"}),
    )
    .await;
    for (name, desc, file, text) in [
        (
            "sea",
            "whales harpoons ships ocean",
            "sea.txt",
            "The harpoon struck the whale ship.",
        ),
        (
            "stars",
            "astronomy telescope planets orbit",
            "stars.txt",
            "The telescope found a planet orbit.",
        ),
    ] {
        let leaf = create(
            a,
            json!({"parent_id":root,"kind":"leaf","name":name,"description":desc}),
        )
        .await;
        let (s, v) = send(
            &app,
            a.r(
                "POST",
                &format!("/api/stores/{leaf}/documents"),
                Some(json!({"text":text,"source":file})),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{v}");
    }
    // B: a stray store of its own, nothing of the eval tree
    create(
        b,
        json!({"kind":"leaf","name":"bench-tmp-x","description":"stray"}),
    )
    .await;

    let (s, v) = send(
        &app,
        a.r("POST", "/api/eval", Some(json!({"modes":["embed","flat"]}))),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        v["contaminated"], false,
        "B's store is not A's foreign store: {v}"
    );
    assert_eq!(v["foreign_stores"], json!([]), "{v}");

    let (s, v) = send(
        &app,
        b.r("POST", "/api/eval", Some(json!({"modes":["embed"]}))),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    // B's own stray store is foreign to the tree, and it never sees A's stores as loaded
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("bench-tmp-x") || code(&v) == "eval_not_ready",
        "{v}"
    );
    assert!(
        !msg.contains(&root.to_string()),
        "A's ids never appear: {v}"
    );
    let (s, v) = send(
        &app,
        b.r(
            "POST",
            "/api/eval",
            Some(json!({"modes":["embed"],"allow_foreign_stores":true})),
        ),
    )
    .await;
    assert_eq!(
        (s, code(&v)),
        (StatusCode::CONFLICT, "eval_not_ready"),
        "{v}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
