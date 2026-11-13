mod common;

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;
use uuid::Uuid;

async fn app() -> (Router, common::TestDb) {
    let (engine, db) = common::test_engine().await;
    (strata_rag::api::router(Arc::new(engine)), db)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(common::with_tenant(req))
        .await
        .expect("router is infallible");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let value = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
    };
    (status, value)
}

fn json_req(method: &str, uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn text_req(uri: &str, body: impl Into<Body>) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(body.into())
        .unwrap()
}

fn bare_req(method: &str, uri: &str) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn id_of(v: &Value) -> Uuid {
    v["id"].as_str().expect("id string").parse().expect("uuid")
}

fn assert_error(status: StatusCode, body: &Value, want_status: StatusCode, code: &str) {
    assert_eq!(status, want_status, "body: {body}");
    let err = body["error"]
        .as_object()
        .unwrap_or_else(|| panic!("no error object: {body}"));
    assert_eq!(err.len(), 2, "error has exactly code + message: {body}");
    assert_eq!(err["code"], code);
    assert!(err["message"].as_str().is_some_and(|m| !m.is_empty()));
}

/// root router "library" → leaf "sea"
async fn make_tree(app: &Router) -> (Uuid, Uuid) {
    let (s, root) = send(
        app,
        json_req(
            "POST",
            "/api/stores",
            json!({"kind":"router","name":"library","description":"all books"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{root}");
    let root_id = id_of(&root);
    let (s, leaf) = send(
        app,
        json_req(
            "POST",
            "/api/stores",
            json!({"parent_id": root_id, "kind":"leaf","name":"sea","description":"whales ships and the ocean"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{leaf}");
    (root_id, id_of(&leaf))
}

#[tokio::test]
async fn create_store_returns_201_with_view_shape() {
    let (app, _db) = app().await;
    let (s, v) = send(
        &app,
        json_req(
            "POST",
            "/api/stores",
            json!({"kind":"router","name":"library","description":"all books"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED);
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "children",
            "chunk_count",
            "description",
            "id",
            "kind",
            "name",
            "parent_id",
            "token_count"
        ]
    );
    assert_eq!(v["kind"], "router");
    assert_eq!(v["parent_id"], Value::Null);
    assert_eq!(v["token_count"], 0);
    assert_eq!(v["children"], json!([]));

    let root_id = id_of(&v);
    let (s, leaf) = send(
        &app,
        json_req(
            "POST",
            "/api/stores",
            json!({"parent_id": root_id, "kind":"leaf","name":"sea","description":"ocean"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED);
    let (s, one) = send(&app, bare_req("GET", &format!("/api/stores/{root_id}"))).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(one["children"], json!([leaf["id"]]));
    let (s, all) = send(&app, bare_req("GET", "/api/stores")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(all["stores"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn unknown_store_is_404_error_body() {
    let (app, _db) = app().await;
    let (s, v) = send(
        &app,
        bare_req("GET", &format!("/api/stores/{}", Uuid::now_v7())),
    )
    .await;
    assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");
}

#[tokio::test]
async fn malformed_id_and_body_are_422_invalid_input() {
    let (app, _db) = app().await;
    let (s, v) = send(&app, bare_req("GET", "/api/stores/not-a-uuid")).await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    let (s, v) = send(
        &app,
        json_req(
            "POST",
            "/api/stores",
            json!({"kind":"shelf","name":"x","description":"y"}),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
}

#[tokio::test]
async fn leaf_as_parent_is_422_invalid_target() {
    let (app, _db) = app().await;
    let (_root, leaf) = make_tree(&app).await;
    let (s, v) = send(
        &app,
        json_req(
            "POST",
            "/api/stores",
            json!({"parent_id": leaf, "kind":"leaf","name":"x","description":"y"}),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_target");
}

#[tokio::test]
async fn ingest_text_plain_201_then_200_then_delete_204() {
    let (app, _db) = app().await;
    let (_root, leaf) = make_tree(&app).await;
    let uri = format!("/api/stores/{leaf}/documents?source=whale.txt&title=Whales");
    let text = "The white whale rose beside the ship.\n\nAhab watched from the deck.";

    let (s, created) = send(&app, text_req(&uri, text)).await;
    assert_eq!(s, StatusCode::CREATED, "{created}");
    assert_eq!(created["status"], "created");
    assert_eq!(created["source"], "whale.txt");
    assert_eq!(created["title"], "Whales");

    let (s, again) = send(&app, text_req(&uri, text)).await;
    assert_eq!(s, StatusCode::OK, "{again}");
    assert_eq!(again["status"], "existing");
    assert_eq!(again["id"], created["id"]);

    let (s, j) = send(
        &app,
        json_req(
            "POST",
            &format!("/api/stores/{leaf}/documents"),
            json!({"text":"Harpoons and rope lay coiled on the deck.","source":"rope.txt"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{j}");
    assert_eq!(j["source"], "rope.txt");

    let doc = id_of(&created);
    let (s, body) = send(&app, bare_req("DELETE", &format!("/api/documents/{doc}"))).await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    assert_eq!(body, Value::Null);
    let (s, v) = send(&app, bare_req("DELETE", &format!("/api/documents/{doc}"))).await;
    assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");
}

#[tokio::test]
async fn ingest_non_utf8_and_empty_are_422_invalid_input() {
    let (app, _db) = app().await;
    let (_root, leaf) = make_tree(&app).await;
    let uri = format!("/api/stores/{leaf}/documents");
    let (s, v) = send(&app, text_req(&uri, vec![0xff_u8, 0xfe, 0x00, 0x41])).await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    let (s, v) = send(&app, text_req(&uri, "   \n\t  ")).await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    let (s, v) = send(&app, text_req(&uri, "")).await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
}

#[tokio::test]
async fn ingest_over_leaf_cap_is_409_leaf_full() {
    // test_config: leaf_max_tokens 10_000, WhitespaceCounter (1 token per word)
    let (app, _db) = app().await;
    let (_root, leaf) = make_tree(&app).await;
    let words: Vec<String> = (0..10_050).map(|i| format!("w{i}")).collect();
    let (s, v) = send(
        &app,
        text_req(&format!("/api/stores/{leaf}/documents"), words.join(" ")),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "leaf_full");
}

#[tokio::test]
async fn body_over_16_mib_is_413() {
    let (app, _db) = app().await;
    let (_root, leaf) = make_tree(&app).await;
    let big = vec![b'a'; strata_rag::api::MAX_BODY_BYTES + 1];
    let (s, v) = send(
        &app,
        text_req(&format!("/api/stores/{leaf}/documents"), big),
    )
    .await;
    assert_error(s, &v, StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large");
    assert_eq!(v["error"]["message"], "request body exceeds 16 MiB");
}

#[tokio::test]
async fn delete_router_with_children_is_409_unless_recursive() {
    let (app, _db) = app().await;
    let (root, _leaf) = make_tree(&app).await;
    let (s, v) = send(&app, bare_req("DELETE", &format!("/api/stores/{root}"))).await;
    assert_error(s, &v, StatusCode::CONFLICT, "has_children");
    let (s, _) = send(
        &app,
        bare_req("DELETE", &format!("/api/stores/{root}?recursive=true")),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    let (s, all) = send(&app, bare_req("GET", "/api/stores")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(all["stores"], json!([]));
}

#[tokio::test]
async fn patch_store_updates_name() {
    let (app, _db) = app().await;
    let (_root, leaf) = make_tree(&app).await;
    let (s, v) = send(
        &app,
        json_req(
            "PATCH",
            &format!("/api/stores/{leaf}"),
            json!({"name":"ocean"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["name"], "ocean");
    let (s, v) = send(
        &app,
        json_req("PATCH", &format!("/api/stores/{leaf}"), json!({})),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
}

#[tokio::test]
async fn query_returns_results_routing_timings() {
    let (app, _db) = app().await;
    let (_root, leaf) = make_tree(&app).await;
    let (s, _) = send(
        &app,
        text_req(
            &format!("/api/stores/{leaf}/documents?source=whale.txt"),
            "The white whale surfaced beside the ship. Ahab watched the white whale.",
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED);

    let (s, v) = send(
        &app,
        json_req(
            "POST",
            "/api/query",
            json!({"query":"white whale ship","route_mode":"embed","min_score":0.0}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "providers",
            "results",
            "routing",
            "timings_by_model",
            "timings_ms"
        ]
    );
    // the fake providers run in-process; no expansion asked, so no chat stage
    assert_eq!(v["providers"]["embed"]["provider"], "in-process");
    assert_eq!(v["providers"]["rerank"]["provider"], "in-process");
    assert!(v["providers"].get("chat").is_none(), "{v}");
    assert!(
        v["routing"]["leaves"]
            .as_array()
            .unwrap()
            .contains(&json!(leaf))
    );
    assert_eq!(v["routing"]["laya_consulted"], false);
    let first = &v["results"][0];
    for k in ["chunk_id", "document_id", "store_path", "text", "scores"] {
        assert!(first.get(k).is_some(), "missing {k}: {first}");
    }
    assert_eq!(
        first["store_path"].as_array().unwrap().last(),
        Some(&json!("sea"))
    );
    for k in ["embed", "route", "search", "rerank", "total"] {
        assert!(v["timings_ms"][k].is_u64(), "timings_ms.{k}: {v}");
    }
}

#[tokio::test]
async fn health_reports_every_component() {
    let (app, _db) = app().await;
    let (s, v) = send(&app, bare_req("GET", "/health")).await;
    assert_eq!(s, StatusCode::OK);
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    // `chat` replaced the old `vllm` component (spec §8.2)
    assert_eq!(keys, ["chat", "db", "embed", "laya", "rerank"]);
    assert_eq!(v["db"], "ok");
    assert_eq!((&v["embed"], &v["rerank"]), (&json!("ok"), &json!("ok")));
    assert_eq!(v["laya"], "unconfigured");
    assert_eq!(v["chat"], "unconfigured");
}

/// M3: a hung upstream must not hold /health for its client timeout (the chat client's is 60 s):
/// a provider that does not answer within the check timeout is "busy".
#[tokio::test]
async fn health_reports_a_hung_upstream_as_busy_quickly() {
    use axum::routing::get;

    let (engine, _db) = common::test_engine().await;
    let hung = common::mock::spawn(Router::new().route(
        "/v1/models",
        get(|| async {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            "never"
        }),
    ))
    .await;
    let engine = Arc::new(engine.with_chat(common::chat_at(&hung, 60)));
    let t = std::time::Instant::now();
    let report = engine.health().await;
    let took = t.elapsed();
    assert_eq!(report.chat, "busy");
    assert_eq!((report.db, report.embed, report.rerank), ("ok", "ok", "ok"));
    assert!(took < std::time::Duration::from_secs(5), "took {took:?}");
}

/// `/health` is unauthenticated: within the cache TTL (default 10 s) concurrent and repeated calls
/// share ONE upstream probe, so nobody can make the server spend laya / chat calls per request.
#[tokio::test]
async fn health_probes_upstreams_once_per_ttl() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::routing::get;
    use strata_rag::clients::laya::LayaClient;
    use strata_rag::config::Secret;

    let counting = |hits: Arc<AtomicUsize>| {
        move || {
            let hits = hits.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                "{}"
            }
        }
    };
    let laya_hits = Arc::new(AtomicUsize::new(0));
    let vllm_hits = Arc::new(AtomicUsize::new(0));
    let laya =
        common::mock::spawn(Router::new().route("/health", get(counting(laya_hits.clone())))).await;
    let vllm =
        common::mock::spawn(Router::new().route("/v1/models", get(counting(vllm_hits.clone()))))
            .await;

    let (engine, _db) = common::test_engine().await;
    let mut engine = engine.with_chat(common::chat_at(&vllm, 60));
    engine.laya_health = Some(Arc::new(
        LayaClient::new(&laya, Secret::new("ltok"), "jev-latest", 800).unwrap(),
    ));
    let app = strata_rag::api::router(Arc::new(engine));

    let ((s1, a), (s2, b)) = tokio::join!(
        send(&app, bare_req("GET", "/health")),
        send(&app, bare_req("GET", "/health"))
    );
    let (s3, c) = send(&app, bare_req("GET", "/health")).await;
    assert_eq!(
        (s1, s2, s3),
        (StatusCode::OK, StatusCode::OK, StatusCode::OK)
    );
    assert_eq!(
        a,
        json!({"db":"ok","embed":"ok","rerank":"ok","chat":"ok","laya":"ok"})
    );
    assert_eq!((&a, &a), (&b, &c));
    assert_eq!(laya_hits.load(Ordering::SeqCst), 1, "laya probes");
    assert_eq!(vllm_hits.load(Ordering::SeqCst), 1, "chat probes");
}

/// Blocks every embed call until the test hands out a permit (a provider that stopped answering).
struct GatedEmbedder(std::sync::mpsc::Receiver<()>);

impl strata_rag::inference::Embedder for GatedEmbedder {
    fn embed(&mut self, texts: &[String]) -> strata_rag::error::Result<Vec<Vec<f32>>> {
        let _ = self.0.recv();
        strata_rag::inference::FakeEmbedder.embed(texts)
    }
}

/// Every embed call fails, as a provider rejecting our key would.
struct BrokenEmbedder;

impl strata_rag::inference::Embedder for BrokenEmbedder {
    fn embed(&mut self, _texts: &[String]) -> strata_rag::error::Result<Vec<Vec<f32>>> {
        Err(strata_rag::error::StrataError::Upstream {
            service: "embed",
            message: "auth: status 401".into(),
        })
    }
}

fn engine_with_embedder(
    pool: sqlx::PgPool,
    embedder: Box<dyn strata_rag::inference::Embedder>,
) -> Arc<strata_rag::engine::Engine> {
    use strata_rag::inference::{FakeReranker, InferenceHandle};
    let inf = InferenceHandle::in_process(embedder, Box::new(FakeReranker));
    Arc::new(strata_rag::engine::Engine::new(
        pool,
        inf,
        Arc::new(strata_rag::text::WhitespaceCounter),
        common::test_config(),
    ))
}

/// Polls `/health` (each call must answer from the cache at once) until `done` holds for its report.
async fn health_until(
    engine: &Arc<strata_rag::engine::Engine>,
    done: impl Fn(&strata_rag::engine::HealthReport) -> bool,
) -> strata_rag::engine::HealthReport {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let t = std::time::Instant::now();
        let report = engine.health().await;
        assert!(
            t.elapsed() < std::time::Duration::from_millis(300),
            "/health waited on a probe: {:?}",
            t.elapsed()
        );
        if done(&report) {
            return report;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "never reached: {report:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

/// An embed provider that does not answer within the check timeout is "busy", not "down".
#[tokio::test]
async fn health_reports_a_slow_embed_provider_as_busy_not_down() {
    let db = common::TestDb::start().await;
    let (permit_tx, permits) = std::sync::mpsc::channel();
    let engine = engine_with_embedder(db.pool.clone(), Box::new(GatedEmbedder(permits)));
    let report = engine.health().await;
    assert_eq!(report.embed, "busy");
    assert_eq!(report.embed_error, None);
    assert_eq!((report.db, report.rerank), ("ok", "ok"));
    for _ in 0..4 {
        let _ = permit_tx.send(());
    }
}

/// The cache must not freeze "busy": once the provider answers again, a call inside the TTL but past
/// `HEALTH_MIN_REPROBE` starts a re-probe (busy provider only; the other components are served from
/// cache) and "ok" shows up right after it. Inside the minimum interval it is not re-probed.
#[tokio::test]
async fn health_cache_does_not_freeze_a_busy_embed_provider() {
    let db = common::TestDb::start().await;
    let (permit_tx, permits) = std::sync::mpsc::channel();
    let engine = engine_with_embedder(db.pool.clone(), Box::new(GatedEmbedder(permits)));
    assert!(engine.cfg.health_cache_s > 0, "cache on (default)");
    assert_eq!(engine.health().await.embed, "busy");
    // the "ingest" finishes: every later embed passes
    for _ in 0..16 {
        let _ = permit_tx.send(());
    }
    assert_eq!(
        engine.health().await.embed,
        "busy",
        "re-probed inside the minimum interval"
    );
    tokio::time::sleep(strata_rag::engine::HEALTH_MIN_REPROBE).await;
    let report = health_until(&engine, |r| r.embed == "ok").await;
    assert_eq!(report.db, "ok");
}

/// Live 2026-10-16: the embed provider (over a relayed private-network path) stopped answering within the
/// 2 s check for ~25 min. Every kubelet readiness call (1 s timeout) landed on a due busy re-probe
/// and waited up to 2 s for it inside the request, so the pod flapped not-Ready although `/health`
/// is always 200. Once a report is cached, a call must answer at once while the re-probe runs
/// behind it.
#[tokio::test]
async fn health_answers_at_once_while_a_busy_provider_is_re_probed() {
    let db = common::TestDb::start().await;
    let (permit_tx, permits) = std::sync::mpsc::channel();
    let engine = engine_with_embedder(db.pool.clone(), Box::new(GatedEmbedder(permits)));
    assert_eq!(engine.health().await.embed, "busy");
    tokio::time::sleep(strata_rag::engine::HEALTH_MIN_REPROBE).await;
    // the re-probe is due and the provider still hangs
    let t = std::time::Instant::now();
    let report = engine.health().await;
    assert!(
        t.elapsed() < std::time::Duration::from_millis(300),
        "waited on the re-probe: {:?}",
        t.elapsed()
    );
    assert_eq!((report.db, report.embed), ("ok", "busy"), "the last report");
    for _ in 0..16 {
        let _ = permit_tx.send(());
    }
}

/// Same for an expired TTL: the full re-probe (here a laya taking 0.8 s — well under the 2 s check, well over the 300 ms bound) runs behind the call.
#[tokio::test]
async fn health_answers_at_once_while_the_full_probe_is_slow() {
    use std::sync::atomic::Ordering;

    let (laya, hits) = counting_laya(std::time::Duration::from_millis(800)).await;
    let (mut engine, _db) = common::test_engine().await;
    engine.cfg.health_cache_s = 0;
    with_laya(&mut engine, &laya);
    let engine = Arc::new(engine);
    assert_eq!(engine.health().await.laya, "ok");
    tokio::time::sleep(strata_rag::engine::HEALTH_MIN_REPROBE).await;
    let t = std::time::Instant::now();
    assert_eq!(engine.health().await.laya, "ok", "the last report");
    assert!(
        t.elapsed() < std::time::Duration::from_millis(300),
        "waited on the probe: {:?}",
        t.elapsed()
    );
    health_until(&engine, |_| hits.load(Ordering::SeqCst) == 2).await;
}

/// The first probe (nothing cached yet) runs in its own task: a caller that gives up on it (the
/// kubelet's timeout drops the request) does not cancel it, and the next caller gets its result.
#[tokio::test]
async fn an_abandoned_first_health_call_does_not_cancel_its_probe() {
    use std::sync::atomic::Ordering;

    let (laya, hits) = counting_laya(std::time::Duration::from_millis(1000)).await;
    let (mut engine, _db) = common::test_engine().await;
    with_laya(&mut engine, &laya);
    let engine = Arc::new(engine);
    let abandoned = tokio::time::timeout(std::time::Duration::from_millis(200), engine.health());
    assert!(abandoned.await.is_err(), "the first probe takes 1 s");
    assert_eq!(engine.health().await.laya, "ok");
    assert_eq!(hits.load(Ordering::SeqCst), 1, "one probe, not restarted");
}

/// A laya mock whose `/health` counts hits and answers after `delay`.
async fn counting_laya(
    delay: std::time::Duration,
) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::routing::get;

    let hits = Arc::new(AtomicUsize::new(0));
    let h = hits.clone();
    let url = common::mock::spawn(Router::new().route(
        "/health",
        get(move || {
            let h = h.clone();
            async move {
                h.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(delay).await;
                "ok"
            }
        }),
    ))
    .await;
    (url, hits)
}

fn with_laya(engine: &mut strata_rag::engine::Engine, url: &str) {
    use strata_rag::clients::laya::LayaClient;
    use strata_rag::config::Secret;
    engine.laya_health = Some(Arc::new(
        LayaClient::new(url, Secret::new("ltok"), "jev-latest", 1500).unwrap(),
    ));
}

/// `STRATA_HEALTH_CACHE_S=0` turns the TTL cache off, but probes stay at least
/// `HEALTH_MIN_REPROBE` apart: back-to-back calls share one probe, a later call probes again.
#[tokio::test]
async fn health_cache_zero_still_keeps_the_minimum_reprobe_interval() {
    use std::sync::atomic::Ordering;

    let (laya, hits) = counting_laya(std::time::Duration::ZERO).await;
    let (mut engine, _db) = common::test_engine().await;
    engine.cfg.health_cache_s = 0;
    with_laya(&mut engine, &laya);
    let engine = Arc::new(engine);
    assert_eq!(engine.health().await.laya, "ok");
    assert_eq!(engine.health().await.laya, "ok");
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "second call inside the minimum interval"
    );
    tokio::time::sleep(strata_rag::engine::HEALTH_MIN_REPROBE).await;
    health_until(&engine, |_| hits.load(Ordering::SeqCst) == 2).await;
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert_eq!(engine.health().await.laya, "ok");
    assert_eq!(hits.load(Ordering::SeqCst), 2, "one probe per due interval");
}

/// One probe at a time, and a caller arriving while it runs gets the last report at once instead
/// of queueing behind it (or starting a second probe).
#[tokio::test]
async fn a_health_call_during_a_running_probe_returns_the_last_report_without_waiting() {
    use std::sync::atomic::Ordering;

    let (laya, hits) = counting_laya(std::time::Duration::from_millis(1000)).await;
    let (mut engine, _db) = common::test_engine().await;
    engine.cfg.health_cache_s = 0;
    with_laya(&mut engine, &laya);
    let engine = Arc::new(engine);
    assert_eq!(engine.health().await.laya, "ok");
    tokio::time::sleep(strata_rag::engine::HEALTH_MIN_REPROBE).await;

    let probing = tokio::spawn({
        let engine = engine.clone();
        async move { engine.health().await }
    });
    // wait until the second probe is inside laya
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while hits.load(Ordering::SeqCst) < 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "second probe never started"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let t = std::time::Instant::now();
    let report = engine.health().await;
    assert!(
        t.elapsed() < std::time::Duration::from_millis(300),
        "queued behind the running probe: {:?}",
        t.elapsed()
    );
    assert_eq!(report.laya, "ok", "the last report");
    assert_eq!(probing.await.unwrap().laya, "ok");
    assert_eq!(hits.load(Ordering::SeqCst), 2, "no second concurrent probe");
}

/// A provider that fails answers at once with an error: that is "down", with the error class.
#[tokio::test]
async fn health_reports_a_failing_embed_provider_as_down_with_its_class() {
    let db = common::TestDb::start().await;
    let engine = engine_with_embedder(db.pool.clone(), Box::new(BrokenEmbedder));
    let t = std::time::Instant::now();
    let report = engine.health().await;
    assert_eq!(report.embed, "down");
    assert_eq!(report.embed_error, Some("auth"));
    let body = serde_json::to_value(&report).unwrap();
    assert_eq!(body["embed_error"], "auth", "{body}");
    assert!(body.get("rerank_error").is_none(), "{body}");
    assert!(
        t.elapsed() < std::time::Duration::from_secs(2),
        "answered fast"
    );
}

/// M11: an oversized body without Content-Length (chunked) is only caught while the Json extractor
/// reads it; that rejection must stay 413 payload_too_large, not become 422.
#[tokio::test]
async fn chunked_json_over_16_mib_is_413_not_422() {
    let (app, _db) = app().await;
    let mut big = br#"{"query":""#.to_vec();
    big.resize(strata_rag::api::MAX_BODY_BYTES + 16, b'a');
    big.extend_from_slice(br#""}"#);
    let req = Request::builder()
        .method("POST")
        .uri("/api/query")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(big))
        .unwrap();
    assert!(req.headers().get("content-length").is_none());
    let (s, v) = send(&app, req).await;
    assert_error(s, &v, StatusCode::PAYLOAD_TOO_LARGE, "payload_too_large");
}

/// M8: query text over MAX_QUERY_CHARS is rejected before any model call.
#[tokio::test]
async fn query_over_4096_chars_is_422() {
    let (app, _db) = app().await;
    let long = "x".repeat(strata_rag::query::MAX_QUERY_CHARS + 1);
    let (s, v) = send(&app, json_req("POST", "/api/query", json!({"query": long}))).await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
}

#[tokio::test]
async fn ask_without_a_chat_provider_is_409_provider_not_configured() {
    let (app, _db) = app().await;
    let (_root, leaf) = make_tree(&app).await;
    let (s, _) = send(
        &app,
        text_req(
            &format!("/api/stores/{leaf}/documents"),
            "The white whale surfaced.",
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED);
    let (s, v) = send(
        &app,
        json_req("POST", "/api/ask", json!({"query":"white whale"})),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "provider_not_configured");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("chat"),
        "{v}"
    );
}

#[tokio::test]
async fn query_top_k_over_cap_is_422_and_bad_json_is_422() {
    let (app, _db) = app().await;
    let (s, v) = send(
        &app,
        json_req("POST", "/api/query", json!({"query":"x","top_k":51})),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    let (s, v) = send(&app, json_req("POST", "/api/query", json!({"top_k":3}))).await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
}

#[tokio::test]
async fn unversioned_routes_only() {
    let (app, _db) = app().await;
    let resp = app
        .clone()
        .oneshot(common::with_tenant(bare_req("GET", "/api/v1/stores")))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// A store write that keeps losing a serialization race (40001, injected twice by a test-only
/// statement trigger) is retried once and then answered 409 `conflict` — retryable by the client,
/// not a 500.
#[tokio::test]
async fn patch_store_losing_a_race_twice_is_409_conflict() {
    let (app, db) = app().await;
    let (_root, leaf) = make_tree(&app).await;
    for sql in [
        "CREATE SEQUENCE fault_seq",
        "CREATE FUNCTION fault() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN \
             IF nextval('fault_seq') <= 2 THEN \
                 RAISE EXCEPTION 'injected serialization failure' USING ERRCODE = '40001'; \
             END IF; \
             RETURN NULL; \
         END $$",
        "CREATE TRIGGER fault BEFORE UPDATE ON stores \
         FOR EACH STATEMENT EXECUTE FUNCTION fault()",
    ] {
        sqlx::query(sql).execute(&db.pool).await.unwrap();
    }
    let (s, v) = send(
        &app,
        json_req(
            "PATCH",
            &format!("/api/stores/{leaf}"),
            json!({"name":"ocean"}),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "conflict");
    let calls: i64 = sqlx::query_scalar("SELECT last_value FROM fault_seq")
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(calls, 2, "one attempt plus exactly one retry");
}
