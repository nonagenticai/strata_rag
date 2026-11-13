//! care-bus readiness endpoints through `router()`: `GET /api/stores/{id}/documents` paging,
//! bearer service-token auth (`STRATA_API_TOKENS`), and `POST /api/eval`.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, WWW_AUTHENTICATE};
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use strata_rag::config::{ApiToken, TokenScope};
use strata_rag::engine::Engine;
use tower::ServiceExt;
use uuid::Uuid;

const TOKEN: &str = "s3cret-token-1";

fn app_of(engine: Engine) -> Router {
    strata_rag::api::router(Arc::new(engine))
}

async fn send_full(app: &Router, req: Request<Body>) -> (StatusCode, Value, axum::http::HeaderMap) {
    let resp = app
        .clone()
        .oneshot(common::with_tenant(req))
        .await
        .expect("infallible");
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let v = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, v, headers)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let (s, v, _) = send_full(app, req).await;
    (s, v)
}

fn req(method: &str, uri: &str, auth: Option<&str>, body: Option<Value>) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(a) = auth {
        b = b.header(AUTHORIZATION, a);
    }
    match body {
        Some(v) => b
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

fn id_of(v: &Value) -> Uuid {
    v["id"].as_str().expect("id").parse().expect("uuid")
}

async fn create(app: &Router, body: Value) -> Uuid {
    let (s, v) = send(app, req("POST", "/api/stores", None, Some(body))).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    id_of(&v)
}

/// root router `lib` with leaves `sea` and `stars`.
async fn tree(app: &Router) -> (Uuid, Uuid, Uuid) {
    let root = create(
        app,
        json!({"kind":"router","name":"lib","description":"books"}),
    )
    .await;
    let sea = create(
        app,
        json!({"parent_id":root,"kind":"leaf","name":"sea","description":"whales harpoons ships ocean"}),
    )
    .await;
    let stars = create(
        app,
        json!({"parent_id":root,"kind":"leaf","name":"stars","description":"astronomy telescope planets orbit"}),
    )
    .await;
    (root, sea, stars)
}

async fn ingest(app: &Router, leaf: Uuid, source: &str, text: &str) -> Uuid {
    let (s, v) = send(
        app,
        req(
            "POST",
            &format!("/api/stores/{leaf}/documents"),
            None,
            Some(json!({"text": text, "source": source, "title": format!("t-{source}")})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    v["document"]["id"]
        .as_str()
        .or_else(|| v["id"].as_str())
        .unwrap_or_else(|| panic!("document id in {v}"))
        .parse()
        .unwrap()
}

// ---------- GET /api/stores/{id}/documents ----------

#[tokio::test]
async fn documents_page_through_with_an_opaque_cursor() {
    let (engine, _db) = common::test_engine().await;
    let app = app_of(engine);
    let (root, sea, stars) = tree(&app).await;
    let mut ids = Vec::new();
    for (i, text) in ["the whale breached", "the harpoon flew", "the ship sank"]
        .iter()
        .enumerate()
    {
        ids.push(ingest(&app, sea, &format!("doc{i}.txt"), text).await);
    }
    ingest(&app, stars, "other.txt", "a telescope saw a planet").await;

    let (s, p1) = send(
        &app,
        req(
            "GET",
            &format!("/api/stores/{sea}/documents?limit=2"),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{p1}");
    let items = p1["items"].as_array().unwrap();
    assert_eq!(items.len(), 2, "{p1}");
    for k in [
        "id",
        "store_id",
        "source",
        "title",
        "content_hash",
        "token_count",
        "chunk_count",
        "created_at",
    ] {
        assert!(items[0].get(k).is_some(), "missing {k}: {}", items[0]);
    }
    assert_eq!(items[0]["content_hash"].as_str().unwrap().len(), 64);
    assert!(items[0]["chunk_count"].as_i64().unwrap() >= 1);
    let next = p1["next"].as_str().expect("a next cursor").to_string();

    let (s, p2) = send(
        &app,
        req(
            "GET",
            &format!("/api/stores/{sea}/documents?limit=2&cursor={next}"),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{p2}");
    assert_eq!(p2["items"].as_array().unwrap().len(), 1, "{p2}");
    assert_eq!(p2["next"], Value::Null);

    // newest first, every sea document exactly once, nothing from `stars`
    let seen: Vec<Uuid> = items
        .iter()
        .chain(p2["items"].as_array().unwrap())
        .map(id_of)
        .collect();
    let mut want = ids.clone();
    want.reverse();
    assert_eq!(seen, want);

    // default limit returns everything with next = null
    let (_, all) = send(
        &app,
        req("GET", &format!("/api/stores/{sea}/documents"), None, None),
    )
    .await;
    assert_eq!(all["items"].as_array().unwrap().len(), 3);
    assert_eq!(all["next"], Value::Null);

    // a router holds no documents of its own: empty page
    let (s, r) = send(
        &app,
        req("GET", &format!("/api/stores/{root}/documents"), None, None),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(r, json!({"items": [], "next": null}));

    // a cursor from another store positions the page but never leaks that store's rows
    let (_, st) = send(
        &app,
        req(
            "GET",
            &format!("/api/stores/{stars}/documents?limit=1"),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(st["items"].as_array().unwrap().len(), 1);
    let (_, leak) = send(
        &app,
        req(
            "GET",
            &format!("/api/stores/{stars}/documents?cursor={next}"),
            None,
            None,
        ),
    )
    .await;
    for d in leak["items"].as_array().unwrap() {
        assert_eq!(d["store_id"], json!(stars));
    }
}

#[tokio::test]
async fn documents_errors() {
    let (engine, _db) = common::test_engine().await;
    let app = app_of(engine);
    let (_, sea, _) = tree(&app).await;
    let (s, v) = send(
        &app,
        req(
            "GET",
            &format!("/api/stores/{}/documents", Uuid::now_v7()),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{v}");
    assert_eq!(v["error"]["code"], "not_found");
    for q in ["limit=0", "limit=-1", "limit=x", "cursor=nope"] {
        let (s, v) = send(
            &app,
            req(
                "GET",
                &format!("/api/stores/{sea}/documents?{q}"),
                None,
                None,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{q}: {v}");
        assert_eq!(v["error"]["code"], "invalid_input");
    }
    // above the max is capped, not refused
    let (s, _) = send(
        &app,
        req(
            "GET",
            &format!("/api/stores/{sea}/documents?limit=100000"),
            None,
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
}

// ---------- bearer token auth ----------

async fn app_with_token() -> (Router, common::TestDb) {
    let (mut engine, db) = common::test_engine().await;
    engine.cfg.api_tokens = vec![
        ApiToken::new(TokenScope::Tenant(common::tn()), "other-token"),
        ApiToken::new(TokenScope::AnyTenant, TOKEN),
    ];
    (app_of(engine), db)
}

#[tokio::test]
async fn tokens_gate_every_route_but_health() {
    let (app, _db) = app_with_token().await;
    let (s, _) = send(&app, req("GET", "/health", None, None)).await;
    assert_eq!(s, StatusCode::OK, "/health stays open");

    let (s, v, h) = send_full(&app, req("GET", "/api/stores", None, None)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_eq!(
        v,
        json!({"error":{"code":"unauthorized","message":"missing or invalid bearer token"}})
    );
    assert_eq!(h.get(WWW_AUTHENTICATE).unwrap(), "Bearer");

    let prefix = &TOKEN[..TOKEN.len() - 1];
    for bad in [
        "Bearer wrong".to_string(),
        format!("Bearer {prefix}"),
        format!("Bearer {TOKEN}x"),
        format!("Basic {TOKEN}"),
        TOKEN.to_string(),
        "Bearer ".to_string(),
    ] {
        let (s, v) = send(&app, req("GET", "/api/stores", Some(&bad), None)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{bad}: {v}");
    }
    // token in the query string is ignored
    let (s, _) = send(
        &app,
        req("GET", &format!("/api/stores?token={TOKEN}"), None, None),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    // unknown paths and writes are gated too
    for (m, uri) in [
        ("GET", "/nope"),
        ("GET", "/healthz"),
        ("GET", "/health-anything"),
        ("POST", "/api/query"),
        ("POST", "/api/eval"),
    ] {
        let (s, _) = send(&app, req(m, uri, None, Some(json!({})))).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{m} {uri}");
    }

    for good in [
        format!("Bearer {TOKEN}"),
        format!("  bearer   {TOKEN} "),
        "Bearer other-token".to_string(),
    ] {
        let (s, v) = send(&app, req("GET", "/api/stores", Some(&good), None)).await;
        assert_eq!(s, StatusCode::OK, "{good}: {v}");
    }
}

#[tokio::test]
async fn no_tokens_means_open() {
    let (engine, _db) = common::test_engine().await;
    assert!(engine.cfg.api_tokens.is_empty());
    let app = app_of(engine);
    let (s, _) = send(&app, req("GET", "/api/stores", None, None)).await;
    assert_eq!(s, StatusCode::OK);
}

#[tokio::test]
async fn serve_refuses_a_public_bind_without_tokens() {
    let (mut engine, _db) = common::test_engine().await;
    engine.cfg.bind = "0.0.0.0:0".parse().unwrap();
    let err = strata_rag::api::serve(Arc::new(engine)).await.unwrap_err();
    assert!(err.to_string().contains("STRATA_API_TOKENS"), "{err}");
}

/// The address checked is the address bound: `serve` takes no second address that could differ
/// from the validated `cfg.bind` (that let a loopback cfg serve unauthenticated on 0.0.0.0). It
/// listens on exactly `cfg.bind`.
#[tokio::test]
async fn serve_listens_on_the_address_it_checked() {
    let (mut engine, _db) = common::test_engine().await;
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    engine.cfg.bind = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    assert!(engine.cfg.api_tokens.is_empty());
    let server = tokio::spawn(strata_rag::api::serve(Arc::new(engine)));
    let url = format!("http://127.0.0.1:{port}/health");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let status = loop {
        match reqwest::get(&url).await {
            Ok(r) => break r.status(),
            Err(e) => {
                assert!(
                    !server.is_finished() && std::time::Instant::now() < deadline,
                    "serve is not listening on cfg.bind {url}: {e}"
                );
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
    };
    assert_eq!(status, reqwest::StatusCode::OK);
    server.abort();
}

// ---------- POST /api/eval ----------

const TREE: &str = r#"{"corpus_dir":"corpus","root":{"name":"lib","kind":"router","description":"books","children":[
    {"name":"sea","kind":"leaf","description":"whales harpoons ships ocean","files":["sea.txt"]},
    {"name":"stars","kind":"leaf","description":"astronomy telescope planets orbit","files":["stars.txt"]}]}}"#;

const QUESTIONS: &str = concat!(
    r#"{"query":"harpoon whale ship","expected_document":"sea.txt","expected_leaf":"sea"}"#,
    "\n",
    r#"{"query":"telescope planet orbit","expected_document":"stars.txt","expected_leaf":"stars"}"#,
    "\n"
);

/// questions.jsonl + tree.json in a temp dir (tree resolved next to the questions).
fn eval_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("strata-api-eval-{}", Uuid::now_v7()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tree.json"), TREE).unwrap();
    std::fs::write(dir.join("questions.jsonl"), QUESTIONS).unwrap();
    dir
}

async fn eval_app() -> (Router, common::TestDb, PathBuf) {
    let (mut engine, db) = common::test_engine().await;
    let dir = eval_dir();
    engine.cfg.eval_questions = dir.join("questions.jsonl");
    (app_of(engine), db, dir)
}

#[tokio::test]
async fn eval_reports_over_a_loaded_tree() {
    let (app, _db, dir) = eval_app().await;
    let (_, sea, stars) = tree(&app).await;
    ingest(
        &app,
        sea,
        "sea.txt",
        "The harpoon struck the great whale and the ship heaved on the ocean.",
    )
    .await;
    ingest(
        &app,
        stars,
        "stars.txt",
        "Through the telescope the planet traced its orbit across the night sky.",
    )
    .await;

    let (s, v) = send(&app, req("POST", "/api/eval", None, Some(json!({})))).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let modes: Vec<&str> = v["reports"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["mode"].as_str().unwrap())
        .collect();
    assert_eq!(modes, ["embed", "hybrid", "flat"]);
    for r in v["reports"].as_array().unwrap() {
        assert_eq!(r["n"], 2, "{r}");
    }
    assert!(v["verdict"] == "PASS" || v["verdict"] == "FAIL", "{v}");
    assert_eq!(v["contaminated"], false, "{v}");
    assert_eq!(v["foreign_stores"], json!([]), "{v}");

    // empty body = defaults; options select modes and routing params
    let (s, d) = send(&app, req("POST", "/api/eval", None, None)).await;
    assert_eq!(s, StatusCode::OK, "{d}");
    assert_eq!(d["reports"].as_array().unwrap().len(), 3);
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval",
            None,
            Some(json!({"modes":["flat","embed","embed"],"beam":1,"min_score":0.0,"allow_foreign_stores":false})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let modes: Vec<&str> = v["reports"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["mode"].as_str().unwrap())
        .collect();
    assert_eq!(modes, ["embed", "flat"]);
    assert!(
        (v["reports"][1]["recall_at_10"].as_f64().unwrap() - 1.0).abs() < 1e-9,
        "{v}"
    );

    for bad in [
        json!({"modes":[]}),
        json!({"modes":["bogus"]}),
        json!({"beam":0}),
        json!({"unknown":1}),
    ] {
        let (s, v) = send(&app, req("POST", "/api/eval", None, Some(bad.clone()))).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{bad}: {v}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn eval_refuses_a_tree_with_foreign_stores_unless_allowed() {
    let (app, _db, dir) = eval_app().await;
    let (root, sea, stars) = tree(&app).await;
    ingest(&app, sea, "sea.txt", "The harpoon struck the great whale.").await;
    ingest(
        &app,
        stars,
        "stars.txt",
        "The telescope traced the planet orbit.",
    )
    .await;
    // a leftover bench leaf under the tree's router: it takes a beam slot in routed modes
    let stray = create(
        &app,
        json!({"parent_id":root,"kind":"leaf","name":"bench-tmp-x","description":"stray"}),
    )
    .await;

    let (s, v) = send(&app, req("POST", "/api/eval", None, Some(json!({})))).await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("lib/bench-tmp-x") && msg.contains(&stray.to_string()),
        "{msg}"
    );

    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval",
            None,
            Some(json!({"allow_foreign_stores": true})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["contaminated"], true, "{v}");
    assert_eq!(v["verdict"], "CONTAMINATED", "never a bare PASS: {v}");
    assert_eq!(v["foreign_stores"][0]["path"], "lib/bench-tmp-x", "{v}");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn eval_is_409_when_the_tree_is_not_loaded() {
    let (app, _db, dir) = eval_app().await;
    // nothing loaded at all
    let (s, v) = send(&app, req("POST", "/api/eval", None, Some(json!({})))).await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "eval_not_ready", "{v}");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        !msg.contains(dir.to_str().unwrap()),
        "no server path: {msg}"
    );
    assert!(
        msg.contains("leaf `sea`") && msg.contains("leaf `stars`"),
        "{msg}"
    );

    // stores exist but stars.txt was never ingested: named, and no store was created by eval
    let (_, sea, _) = tree(&app).await;
    ingest(&app, sea, "sea.txt", "The harpoon struck the whale.").await;
    let (s, v) = send(&app, req("POST", "/api/eval", None, Some(json!({})))).await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("stars.txt") && !msg.contains("sea.txt"),
        "{msg}"
    );
    let (_, stores) = send(&app, req("GET", "/api/stores", None, None)).await;
    assert_eq!(
        stores["stores"].to_string().matches("\"name\"").count(),
        3,
        "{stores}"
    );

    // a broken tree file → 409 eval_not_ready, without the parser's text or the path
    std::fs::write(dir.join("tree.json"), "{ not json").unwrap();
    let (s, v) = send(&app, req("POST", "/api/eval", None, Some(json!({})))).await;
    assert_eq!(
        (s, v["error"]["code"].as_str()),
        (StatusCode::CONFLICT, Some("eval_not_ready")),
        "{v}"
    );
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("tree") && !msg.contains(dir.to_str().unwrap()),
        "{msg}"
    );
    assert!(
        !msg.contains("line") && !msg.contains("column"),
        "no parser text: {msg}"
    );

    // missing questions file → 409 eval_not_ready, never the server path or the OS error
    std::fs::remove_file(dir.join("questions.jsonl")).unwrap();
    let (s, v) = send(&app, req("POST", "/api/eval", None, Some(json!({})))).await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "eval_not_ready", "{v}");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("questions"), "{msg}");
    for leak in [
        dir.to_str().unwrap(),
        "questions.jsonl",
        "No such file",
        "os error",
    ] {
        assert!(!msg.contains(leak), "{leak:?} leaked: {msg}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Embeds pass straight through until `hold` is set; from then on each embed reports on `entered`
/// and waits for a permit, which holds an eval open mid-run without any timing guess.
struct HoldEmbedder {
    hold: Arc<std::sync::atomic::AtomicBool>,
    entered: std::sync::mpsc::Sender<()>,
    permits: std::sync::mpsc::Receiver<()>,
}

impl strata_rag::inference::Embedder for HoldEmbedder {
    fn embed(&mut self, texts: &[String]) -> strata_rag::error::Result<Vec<Vec<f32>>> {
        if self.hold.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = self.entered.send(());
            let _ = self.permits.recv();
        }
        strata_rag::inference::FakeEmbedder.embed(texts)
    }
}

/// One eval at a time: a second `POST /api/eval` while one runs is refused at once (409,
/// retryable), not queued behind it, and the slot frees when the first finishes.
#[tokio::test]
async fn a_second_concurrent_eval_is_refused_not_queued() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use strata_rag::inference::{FakeReranker, InferenceHandle};

    let db = common::TestDb::start().await;
    let hold = Arc::new(AtomicBool::new(false));
    let (entered_tx, entered) = std::sync::mpsc::channel();
    let (permit_tx, permits) = std::sync::mpsc::channel();
    let embedder = HoldEmbedder {
        hold: hold.clone(),
        entered: entered_tx,
        permits,
    };
    let inf = InferenceHandle::in_process(Box::new(embedder), Box::new(FakeReranker));
    let dir = eval_dir();
    let mut cfg = common::test_config();
    cfg.eval_questions = dir.join("questions.jsonl");
    let app = app_of(Engine::new(
        db.pool.clone(),
        inf,
        Arc::new(strata_rag::text::WhitespaceCounter),
        cfg,
    ));
    let (_, sea, stars) = tree(&app).await;
    ingest(&app, sea, "sea.txt", "The harpoon struck the whale.").await;
    ingest(&app, stars, "stars.txt", "The telescope found a planet.").await;

    let body = || Some(json!({"modes":["embed"]}));
    hold.store(true, Ordering::SeqCst);
    let first = tokio::spawn({
        let app = app.clone();
        async move { send(&app, req("POST", "/api/eval", None, body())).await }
    });
    // The first eval is now inside a model call, holding its slot.
    tokio::task::spawn_blocking(move || entered.recv_timeout(std::time::Duration::from_secs(15)))
        .await
        .unwrap()
        .expect("the first eval reached the model");

    let (s, v) = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        send(&app, req("POST", "/api/eval", None, body())),
    )
    .await
    .expect("a second eval must be refused at once, not queued behind the first");
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "conflict", "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("already running"),
        "{v}"
    );

    // the slot is per tenant: another tenant's eval is not refused as "already running" (it has
    // nothing loaded, so it answers eval_not_ready before any model call)
    let mut other = req("POST", "/api/eval", None, body());
    other
        .headers_mut()
        .insert("x-tenant-id", common::OTHER_TENANT.parse().unwrap());
    let (s, v) = tokio::time::timeout(std::time::Duration::from_secs(5), send(&app, other))
        .await
        .expect("another tenant's eval is not queued behind the first");
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "eval_not_ready", "{v}");

    hold.store(false, Ordering::SeqCst);
    for _ in 0..64 {
        let _ = permit_tx.send(());
    }
    let (s, v) = first.await.unwrap();
    assert_eq!(s, StatusCode::OK, "{v}");
    // the slot is released when the eval finishes
    let (s, v) = send(&app, req("POST", "/api/eval", None, body())).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    std::fs::remove_dir_all(&dir).ok();
}
