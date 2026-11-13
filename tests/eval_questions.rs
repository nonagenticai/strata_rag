//! Tenant eval questions through `router()` against a real Postgres (migrations 0012-0014) with the
//! fake embedder/reranker: `/api/eval/questions` CRUD + import, `POST /api/eval` with tenant /
//! inline questions (asynchronous: 202, then polled to `completed` / `failed`), stored runs
//! (`/api/eval/runs`), single flight per tenant, the service-start sweep, and tenant isolation of
//! all of it.

mod common;

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::header::{CONTENT_TYPE, LOCATION};
use axum::http::{HeaderMap, Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sqlx::PgPool;
use tower::ServiceExt;
use uuid::Uuid;

const ACTOR: &str = "0192f0c8-0000-7000-8000-0000000000ac";

async fn app() -> (Router, common::TestDb) {
    let (engine, db) = common::test_engine().await;
    (strata_rag::api::router(Arc::new(engine)), db)
}

async fn send_full(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, Value) {
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
    (status, headers, v)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let (status, _, v) = send_full(app, req).await;
    (status, v)
}

fn req(method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
    let b = Request::builder()
        .method(method)
        .uri(uri)
        .header("x-actor-id", ACTOR);
    match body {
        Some(v) => b
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

/// The same request as the other tenant.
fn as_other(mut r: Request<Body>) -> Request<Body> {
    r.headers_mut().insert(
        "x-tenant-id",
        axum::http::HeaderValue::from_static(common::OTHER_TENANT),
    );
    r
}

fn csv_req(body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/api/eval/questions/import")
        .header(CONTENT_TYPE, "text/csv")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn id_of(v: &Value) -> Uuid {
    v["id"].as_str().expect("id").parse().expect("uuid")
}

fn assert_error(s: StatusCode, v: &Value, want: StatusCode, code: &str) {
    assert_eq!(s, want, "{v}");
    assert_eq!(v["error"]["code"], code, "{v}");
}

async fn question_count(app: &Router) -> usize {
    let (s, v) = send(app, req("GET", "/api/eval/questions", None)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["next"], Value::Null, "{v}");
    v["items"].as_array().unwrap().len()
}

async fn create_question(app: &Router, body: Value) -> Uuid {
    let (s, v) = send(app, req("POST", "/api/eval/questions", Some(body))).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    id_of(&v)
}

/// A request as the default tenant (`common::TENANT`), unchanged.
fn as_self(r: Request<Body>) -> Request<Body> {
    r
}

async fn create_store_as(
    app: &Router,
    who: fn(Request<Body>) -> Request<Body>,
    body: Value,
) -> Uuid {
    let (s, v) = send(app, who(req("POST", "/api/stores", Some(body)))).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    id_of(&v)
}

/// router `lib` with leaves `sea` (sea.txt) and `stars` (stars.txt, title "Star guide").
async fn loaded_tree(app: &Router) -> (Uuid, Uuid, Uuid) {
    loaded_tree_as(app, as_self).await
}

/// `loaded_tree` as the tenant `who` sends as.
async fn loaded_tree_as(
    app: &Router,
    who: fn(Request<Body>) -> Request<Body>,
) -> (Uuid, Uuid, Uuid) {
    let root = create_store_as(
        app,
        who,
        json!({"kind":"router","name":"lib","description":"books"}),
    )
    .await;
    let sea = create_store_as(
        app,
        who,
        json!({"parent_id":root,"kind":"leaf","name":"sea","description":"whales harpoons ships ocean"}),
    )
    .await;
    let stars = create_store_as(
        app,
        who,
        json!({"parent_id":root,"kind":"leaf","name":"stars","description":"astronomy telescope planets orbit"}),
    )
    .await;
    for (leaf, source, title, text) in [
        (
            sea,
            "sea.txt",
            "Sea tales",
            "The harpoon struck the great whale and the ship heaved on the ocean.",
        ),
        (
            stars,
            "stars.txt",
            "Star guide",
            "Through the telescope the planet traced its orbit across the night sky.",
        ),
    ] {
        let (s, v) = send(
            app,
            who(req(
                "POST",
                &format!("/api/stores/{leaf}/documents"),
                Some(json!({"text": text, "source": source, "title": title})),
            )),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{v}");
    }
    (root, sea, stars)
}

/// `POST /api/eval` with tenant questions as `who`: 202, the `running` run's summary, and a
/// `Location` naming it.
async fn start_eval_as(
    app: &Router,
    who: fn(Request<Body>) -> Request<Body>,
    body: Value,
) -> Value {
    let (s, h, v) = send_full(app, who(req("POST", "/api/eval", Some(body)))).await;
    assert_eq!(s, StatusCode::ACCEPTED, "{v}");
    let location = format!("/api/eval/runs/{}", id_of(&v));
    assert_eq!(
        h.get(LOCATION).and_then(|l| l.to_str().ok()),
        Some(location.as_str()),
        "{v}"
    );
    assert_eq!(v["status"], "running", "{v}");
    assert_eq!(v["error"], Value::Null, "{v}");
    assert_eq!(v["verdict"], Value::Null, "{v}");
    assert_eq!(v["reports"], Value::Null, "{v}");
    assert_eq!(v["finished_at"], Value::Null, "{v}");
    assert_eq!(v["took_ms"], Value::Null, "{v}");
    assert_eq!(v["progress"]["done"], 0, "{v}");
    assert!(v.get("results").is_none(), "a summary: {v}");
    v
}

async fn start_eval(app: &Router, body: Value) -> Value {
    start_eval_as(app, as_self, body).await
}

/// Polls `GET /api/eval/runs/{id}` as `who` until the run is no longer `running`.
async fn wait_run_as(app: &Router, who: fn(Request<Body>) -> Request<Body>, id: Uuid) -> Value {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        let (s, v) = send(app, who(req("GET", &format!("/api/eval/runs/{id}"), None))).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        if v["status"] != "running" {
            return v;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "run {id} still running after 60 s: {v}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

async fn wait_run(app: &Router, id: Uuid) -> Value {
    wait_run_as(app, as_self, id).await
}

/// A tenant eval started and polled to its end, which must be `completed`: the stored run with
/// `results`.
async fn run_eval(app: &Router, body: Value) -> Value {
    let started = start_eval(app, body).await;
    let run = wait_run(app, id_of(&started)).await;
    assert_eq!(run["status"], "completed", "{run}");
    assert_eq!(run["error"], Value::Null, "{run}");
    assert_eq!(
        run["progress"]["done"], run["progress"]["total"],
        "a completed run is fully scored: {run}"
    );
    assert_eq!(
        run["progress"]["total"], started["progress"]["total"],
        "{run}"
    );
    run
}

async fn chunk_of(pool: &PgPool, source: &str) -> Uuid {
    sqlx::query_scalar(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id \
         WHERE d.source = $1 ORDER BY c.ord LIMIT 1",
    )
    .bind(source)
    .fetch_one(pool)
    .await
    .expect("a chunk of the document")
}

fn modes_of(v: &Value) -> Vec<String> {
    v["reports"]
        .as_array()
        .unwrap_or_else(|| panic!("reports in {v}"))
        .iter()
        .map(|r| r["mode"].as_str().unwrap().to_string())
        .collect()
}

fn report<'a>(v: &'a Value, mode: &str) -> &'a Value {
    v["reports"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["mode"] == mode)
        .unwrap_or_else(|| panic!("no {mode} report in {v}"))
}

fn approx(v: &Value, want: f64) -> bool {
    v.as_f64().is_some_and(|x| (x - want).abs() < 1e-9)
}

// ---------- questions ----------

#[tokio::test]
async fn questions_crud_validation_and_tenant_isolation() {
    let (app, _db) = app().await;
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval/questions",
            Some(json!({"query":"  How big is a whale? ","expected_document":"sea.txt","expected_leaf":"sea"})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let id = id_of(&v);
    assert_eq!(v["query"], "How big is a whale?", "trimmed: {v}");
    assert_eq!(v["expected_document"], "sea.txt");
    assert_eq!(v["expected_leaf"], "sea");
    assert_eq!(v["expected_document_ids"], json!([]));
    assert_eq!(v["expected_text"], Value::Null);
    assert_eq!(v["created_by"], ACTOR, "X-Actor-Id: {v}");
    assert!(v["created_at"].as_str().unwrap().ends_with('Z'), "{v}");
    assert_eq!(v["created_at"], v["updated_at"]);

    // same query, other case: 409; no expectation / unknown field: 422
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval/questions",
            Some(json!({"query":"HOW BIG IS A WHALE?","expected_text":"whale"})),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "question_exists");
    for bad in [
        json!({"query":"q","expected_leaf":"sea"}),
        json!({"query":" ","expected_document":"sea.txt"}),
        json!({"query":"q","expected_document":"sea.txt","bogus":1}),
        json!({"query":"q","expected_chunk_ids":["not-a-uuid"]}),
    ] {
        let (s, v) = send(&app, req("POST", "/api/eval/questions", Some(bad.clone()))).await;
        assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    }
    assert_eq!(question_count(&app).await, 1);

    // the other tenant sees nothing and cannot touch it (404, as for an unknown id)
    let (s, v) = send(&app, as_other(req("GET", "/api/eval/questions", None))).await;
    assert_eq!((s, v["items"].clone()), (StatusCode::OK, json!([])));
    let uri = format!("/api/eval/questions/{id}");
    let (s, v) = send(
        &app,
        as_other(req("PATCH", &uri, Some(json!({"expected_text":"x"})))),
    )
    .await;
    assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");
    let (s, v) = send(&app, as_other(req("DELETE", &uri, None))).await;
    assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");
    // ... and may store the same query for itself
    let (s, v) = send(
        &app,
        as_other(req(
            "POST",
            "/api/eval/questions",
            Some(json!({"query":"How big is a whale?","expected_text":"whale"})),
        )),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");

    // PATCH: absent kept, null cleared
    let (s, v) = send(
        &app,
        req(
            "PATCH",
            &uri,
            Some(json!({"expected_leaf":null,"expected_text":"great whale"})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["expected_leaf"], Value::Null);
    assert_eq!(v["expected_document"], "sea.txt");
    assert_eq!(v["expected_text"], "great whale");
    assert_ne!(v["updated_at"], v["created_at"]);
    // clearing every expectation is refused; an empty patch too
    let (s, v) = send(
        &app,
        req(
            "PATCH",
            &uri,
            Some(json!({"expected_document":null,"expected_text":null})),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    let (s, v) = send(&app, req("PATCH", &uri, Some(json!({})))).await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    // renaming onto another stored query is a conflict
    let other_id = create_question(
        &app,
        json!({"query":"Which planet?","expected_document":"stars.txt"}),
    )
    .await;
    let (s, v) = send(
        &app,
        req(
            "PATCH",
            &format!("/api/eval/questions/{other_id}"),
            Some(json!({"query":"how big is a WHALE?"})),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "question_exists");

    let (s, _) = send(&app, req("DELETE", &uri, None)).await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    let (s, v) = send(&app, req("DELETE", &uri, None)).await;
    assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");
    assert_eq!(question_count(&app).await, 1);
}

#[tokio::test]
async fn import_is_all_or_nothing_and_capped() {
    let (app, _db) = app().await;
    let chunk = Uuid::now_v7();
    let csv = format!(
        "query,expected_document,expected_leaf,expected_chunk_ids\n\
         \"Whale, big?\",sea.txt,sea,\n\
         Planet orbit,,stars,{chunk}\n"
    );
    let (s, v) = send(&app, csv_req(&csv)).await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    assert_eq!(v, json!({"added":2,"total":2}));
    let (_, list) = send(&app, req("GET", "/api/eval/questions", None)).await;
    assert_eq!(list["items"][0]["query"], "Whale, big?", "{list}");
    assert_eq!(list["items"][1]["expected_chunk_ids"], json!([chunk]));

    // one bad row: nothing imported, the line is named
    let (s, v) = send(
        &app,
        csv_req("query,expected_document\nfine,a.txt\n,b.txt\n"),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("line 3"),
        "{v}"
    );
    // a query already stored: nothing imported (the first row is rolled back too)
    let (s, v) = send(
        &app,
        csv_req("query,expected_document\nbrand new,a.txt\n\"WHALE, big?\",a.txt\n"),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "question_exists");
    assert_eq!(question_count(&app).await, 2, "rolled back");

    // JSON import
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval/questions/import",
            Some(json!({"questions":[{"query":"brand new","expected_text":"x"}]})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    assert_eq!(v, json!({"added":1,"total":3}));

    // cap: 3 stored + 198 = 201 > 200
    let many: Vec<Value> = (0..198)
        .map(|i| json!({"query": format!("q{i}"), "expected_text": "x"}))
        .collect();
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval/questions/import",
            Some(json!({"questions": many})),
        ),
    )
    .await;
    assert_error(
        s,
        &v,
        StatusCode::UNPROCESSABLE_ENTITY,
        "too_many_questions",
    );
    assert_eq!(question_count(&app).await, 3);
    // 197 fits exactly; then a single create is refused
    let fits: Vec<Value> = (0..197)
        .map(|i| json!({"query": format!("q{i}"), "expected_text": "x"}))
        .collect();
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval/questions/import",
            Some(json!({"questions": fits})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    assert_eq!(v["total"], 200);
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval/questions",
            Some(json!({"query":"one more","expected_text":"x"})),
        ),
    )
    .await;
    assert_error(
        s,
        &v,
        StatusCode::UNPROCESSABLE_ENTITY,
        "too_many_questions",
    );

    let (s, v) = send(
        &app,
        Request::builder()
            .method("POST")
            .uri("/api/eval/questions/import")
            .header(CONTENT_TYPE, "application/xml")
            .body(Body::from("<x/>"))
            .unwrap(),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
}

// ---------- POST /api/eval with tenant questions, stored runs ----------

#[tokio::test]
async fn tenant_eval_scores_stores_and_compares_runs() {
    let (app, db) = app().await;
    loaded_tree(&app).await;
    let stars_chunk = chunk_of(&db.pool, "stars.txt").await;
    let q_sea = create_question(
        &app,
        json!({"query":"harpoon whale ship","expected_document":"sea.txt","expected_leaf":"sea"}),
    )
    .await;
    let q_stars = create_question(
        &app,
        json!({"query":"telescope planet orbit","expected_chunk_ids":[stars_chunk],"expected_leaf":"lib/stars"}),
    )
    .await;
    // an expectation no chunk satisfies: a miss in every mode, so recall is not trivially 1
    let q_miss = create_question(
        &app,
        json!({"query":"whale ocean","expected_text":"this sentence appears in no document"}),
    )
    .await;

    let started = start_eval(
        &app,
        json!({"source":"tenant","modes":["flat","embed","embed"]}),
    )
    .await;
    assert_eq!(started["modes"], json!(["embed", "flat"]), "{started}");
    assert_eq!(started["question_count"], 3, "{started}");
    assert_eq!(
        started["progress"],
        json!({"done":0,"total":6}),
        "3 questions x 2 modes: {started}"
    );
    let run = wait_run(&app, id_of(&started)).await;
    assert_eq!(run["status"], "completed", "{run}");
    assert_eq!(run["progress"], json!({"done":6,"total":6}), "{run}");
    assert_eq!(run["error"], Value::Null, "{run}");
    assert_eq!(run["started_at"], started["started_at"]);
    assert_eq!(
        modes_of(&run),
        ["embed", "flat"],
        "routed first, deduplicated"
    );
    let flat = report(&run, "flat");
    assert_eq!(flat["n"], 3, "{run}");
    // two of three in the top 10 (only two documents exist), the third never
    assert!(approx(&flat["recall_at_10"], 2.0 / 3.0), "{flat}");
    assert_eq!(flat["leaf_accuracy"], Value::Null);
    assert_eq!(run["source"], "tenant");
    assert_eq!(run["started_by"], ACTOR);
    assert!(
        run["verdict"] == "PASS" || run["verdict"] == "FAIL",
        "{run}"
    );
    assert_eq!(run["modes"], json!(["embed", "flat"]));
    let qids: Vec<Value> = run["questions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| q["id"].clone())
        .collect();
    assert_eq!(qids, [json!(q_sea), json!(q_stars), json!(q_miss)]);
    let results = run["results"].as_array().unwrap();
    assert_eq!(results.len(), 6, "3 questions x 2 modes: {run}");
    for r in results.iter().filter(|r| r["index"] == 2) {
        assert_eq!(
            r["rank"],
            Value::Null,
            "the impossible question never hits: {r}"
        );
        assert_eq!(r["leaf_hit"], Value::Null, "no expected_leaf: {r}");
    }
    for r in results
        .iter()
        .filter(|r| r["mode"] == "flat" && r["index"] != 2)
    {
        assert!(
            r["rank"].as_u64().is_some_and(|k| (1..=10).contains(&k)),
            "{r}"
        );
    }
    let embed_sea = results
        .iter()
        .find(|r| r["mode"] == "embed" && r["index"] == 0)
        .unwrap();
    assert!(
        embed_sea["leaf_hit"].is_boolean(),
        "expected_leaf given: {embed_sea}"
    );
    let run1 = id_of(&run);

    // a subset by id, in the order asked
    let run2 = run_eval(
        &app,
        json!({"question_ids":[q_stars, q_stars],"modes":["flat"]}),
    )
    .await;
    assert_eq!(report(&run2, "flat")["n"], 1);
    assert!(
        approx(&report(&run2, "flat")["recall_at_10"], 1.0),
        "{run2}"
    );

    // inline questions: scored and stored as a run, not stored as questions
    let run3 = run_eval(
        &app,
        json!({"questions":[{"query":"telescope","expected_document":"Star guide"}],"modes":["flat"]}),
    )
    .await;
    assert_eq!(run3["source"], "inline");
    assert_eq!(run3["questions"][0]["id"], Value::Null);
    assert!(
        approx(&report(&run3, "flat")["recall_at_10"], 1.0),
        "matched by title: {run3}"
    );
    assert_eq!(question_count(&app).await, 3);

    // the run list: newest first, paged, summaries only (a question count, no snapshot, no
    // per-question results)
    let (s, page) = send(&app, req("GET", "/api/eval/runs?limit=2", None)).await;
    assert_eq!(s, StatusCode::OK, "{page}");
    let items = page["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["id"], run3["id"]);
    assert_eq!(items[1]["id"], run2["id"]);
    for item in items {
        assert!(item.get("results").is_none(), "{page}");
        assert!(item.get("questions").is_none(), "{page}");
    }
    assert_eq!(items[0]["question_count"], 1, "{page}");
    assert_eq!(items[1]["question_count"], 1, "deduplicated ids: {page}");
    assert_eq!(items[0]["reports"], run3["reports"]);
    assert_eq!(items[0]["verdict"], run3["verdict"]);
    assert_eq!(items[0]["started_at"], run3["started_at"]);
    assert_eq!(items[0]["took_ms"], run3["took_ms"]);
    assert_eq!(items[0]["status"], "completed");
    assert_eq!(items[0]["progress"], run3["progress"]);
    let mut keys: Vec<&str> = items[0]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "beam",
            "error",
            "finished_at",
            "id",
            "min_score",
            "modes",
            "progress",
            "question_count",
            "reports",
            "root_id",
            "source",
            "started_at",
            "started_by",
            "status",
            "took_ms",
            "verdict"
        ],
        "the list item shape: {page}"
    );
    let next = page["next"].as_str().expect("a next cursor");
    let (s, rest) = send(
        &app,
        req(
            "GET",
            &format!("/api/eval/runs?limit=2&cursor={next}"),
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{rest}");
    assert_eq!(rest["items"].as_array().unwrap().len(), 1);
    assert_eq!(rest["items"][0]["id"], json!(run1));
    assert_eq!(rest["items"][0]["question_count"], 3, "{rest}");
    assert_eq!(rest["next"], Value::Null);

    // a stored run keeps its snapshot after its question is deleted
    let (s, _) = send(
        &app,
        req("DELETE", &format!("/api/eval/questions/{q_sea}"), None),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    let (s, got) = send(&app, req("GET", &format!("/api/eval/runs/{run1}"), None)).await;
    assert_eq!(s, StatusCode::OK, "{got}");
    assert_eq!(got["questions"], run["questions"]);
    assert_eq!(got["reports"], run["reports"]);
    assert_eq!(got["results"], run["results"]);
    assert!(got["took_ms"].as_i64().is_some_and(|ms| ms >= 0), "{got}");

    // the other tenant sees no runs and cannot read this one
    let (s, v) = send(&app, as_other(req("GET", "/api/eval/runs", None))).await;
    assert_eq!((s, v["items"].clone()), (StatusCode::OK, json!([])));
    let (s, v) = send(
        &app,
        as_other(req("GET", &format!("/api/eval/runs/{run1}"), None)),
    )
    .await;
    assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");
}

#[tokio::test]
async fn tenant_eval_scopes_to_root_id() {
    let (app, _db) = app().await;
    let (_root, sea, _stars) = loaded_tree(&app).await;
    create_question(
        &app,
        json!({"query":"harpoon whale","expected_document":"sea.txt","expected_leaf":"sea"}),
    )
    .await;
    let v = run_eval(
        &app,
        json!({"source":"tenant","root_id":sea,"modes":["flat"]}),
    )
    .await;
    assert_eq!(v["root_id"], json!(sea));
    // stars.txt lives outside the `sea` subtree: refused, not silently scored as a miss
    create_question(
        &app,
        json!({"query":"telescope","expected_document":"stars.txt"}),
    )
    .await;
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval",
            Some(json!({"source":"tenant","store_id":sea,"modes":["flat"]})),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "eval_not_ready");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("stars.txt"),
        "{v}"
    );
    // a root that is not this tenant's
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval",
            Some(json!({"source":"tenant","root_id":Uuid::now_v7()})),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");
}

/// D90: with one root a tenant eval may omit root_id (the only tree); with two roots and no
/// root_id it is 409 `ambiguous_root` (no tree is guessed), and naming one runs it.
#[tokio::test]
async fn tenant_eval_needs_root_id_once_the_tenant_has_two_roots() {
    let (app, _db) = app().await;
    let (root, _sea, _stars) = loaded_tree(&app).await;
    create_question(
        &app,
        json!({"query":"harpoon whale","expected_document":"sea.txt"}),
    )
    .await;
    // one root, omitted: scored over that tree
    let v = run_eval(&app, json!({"source":"tenant","modes":["flat"]})).await;
    assert_eq!(v["root_id"], Value::Null, "{v}");

    create_store_as(
        &app,
        as_self,
        json!({"kind":"router","name":"lib2","description":"more books"}),
    )
    .await;
    // two roots, omitted: refused before any run is stored
    let runs_before = send(&app, req("GET", "/api/eval/runs", None)).await.1["items"]
        .as_array()
        .unwrap()
        .len();
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval",
            Some(json!({"source":"tenant","modes":["flat"]})),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "ambiguous_root");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("root_id"),
        "{v}"
    );
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/eval",
            Some(json!({"questions":[{"query":"whale","expected_document":"sea.txt"}]})),
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "ambiguous_root");
    let runs_after = send(&app, req("GET", "/api/eval/runs", None)).await.1["items"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(runs_after, runs_before, "a refused eval stores no run");
    // two roots, named: runs
    let v = run_eval(
        &app,
        json!({"source":"tenant","root_id":root,"modes":["flat"]}),
    )
    .await;
    assert_eq!(v["root_id"], json!(root));
}

#[tokio::test]
async fn tenant_eval_refuses_what_it_cannot_score() {
    let (app, _db) = app().await;
    // no questions yet
    let (s, v) = send(
        &app,
        req("POST", "/api/eval", Some(json!({"source":"tenant"}))),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "eval_not_ready");

    loaded_tree(&app).await;
    let bad_leaf = create_question(
        &app,
        json!({"query":"whale","expected_document":"sea.txt","expected_leaf":"nope"}),
    )
    .await;
    create_question(
        &app,
        json!({"query":"comet","expected_document":"missing.txt","expected_document_ids":[Uuid::now_v7()]}),
    )
    .await;
    let (s, v) = send(
        &app,
        req("POST", "/api/eval", Some(json!({"source":"tenant"}))),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "eval_not_ready");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("question 1: expected_leaf `nope`"), "{msg}");
    assert!(
        msg.contains("question 2: no document `missing.txt`"),
        "{msg}"
    );
    assert!(msg.contains("question 2: document "), "{msg}");
    let (s, v) = send(&app, req("GET", "/api/eval/runs", None)).await;
    assert_eq!(
        (s, v["items"].clone()),
        (StatusCode::OK, json!([])),
        "nothing stored"
    );

    for (body, status, code) in [
        (
            json!({"source":"builtin","question_ids":[bad_leaf]}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"root_id":Uuid::now_v7()}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"source":"tenant","allow_foreign_stores":true}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"question_ids":[bad_leaf],"questions":[{"query":"q","expected_text":"x"}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"questions":[{"query":"q"}]}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"question_ids":[]}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"question_ids":[Uuid::now_v7()]}),
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            json!({"source":"everything"}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
    ] {
        let (s, v) = send(&app, req("POST", "/api/eval", Some(body.clone()))).await;
        assert_eq!(s, status, "{body}: {v}");
        assert_eq!(v["error"]["code"], code, "{body}: {v}");
    }
}

#[tokio::test]
async fn concurrent_patches_of_different_fields_both_persist() {
    let (app, db) = app().await;
    let id = create_question(
        &app,
        json!({"query":"How deep is the sea?","expected_document":"sea.txt"}),
    )
    .await;
    let uri = format!("/api/eval/questions/{id}");

    // Hold the row so both PATCHes are in flight at once: each must wait, then apply its field on
    // top of the other's result. A read-then-write PATCH would read the same old row twice and the
    // second write would put the first one's field back.
    let mut holder = db.pool.begin().await.unwrap();
    sqlx::query("SELECT 1 FROM eval_questions WHERE id = $1 FOR UPDATE")
        .bind(id)
        .execute(&mut *holder)
        .await
        .unwrap();
    let spawn_patch = |body: Value| {
        let app = app.clone();
        let uri = uri.clone();
        tokio::spawn(async move { send(&app, req("PATCH", &uri, Some(body))).await })
    };
    let a = spawn_patch(json!({"expected_text":"fathoms"}));
    let b = spawn_patch(json!({"expected_leaf":"sea"}));
    // both backends waiting on a lock (a condition, not a sleep), then let them go
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        let waiting: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM pg_stat_activity \
             WHERE datname = current_database() AND wait_event_type = 'Lock'",
        )
        .fetch_one(&db.pool)
        .await
        .unwrap();
        if waiting >= 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "both PATCHes never blocked on the held row ({waiting} waiting)"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    holder.commit().await.unwrap();
    for h in [a, b] {
        let (s, v) = h.await.unwrap();
        assert_eq!(s, StatusCode::OK, "{v}");
    }

    let (s, v) = send(&app, req("GET", "/api/eval/questions", None)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let q = &v["items"][0];
    assert_eq!(q["id"], json!(id), "{v}");
    assert_eq!(q["expected_text"], "fathoms", "first PATCH lost: {q}");
    assert_eq!(q["expected_leaf"], "sea", "second PATCH lost: {q}");
    assert_eq!(
        q["expected_document"], "sea.txt",
        "untouched field kept: {q}"
    );
}

#[tokio::test]
async fn question_ids_naming_another_tenants_question_evaluate_nothing() {
    let (app, db) = app().await;
    loaded_tree(&app).await;
    let mine = create_question(
        &app,
        json!({"query":"harpoon whale","expected_document":"sea.txt"}),
    )
    .await;
    let foreign_query = "telescope secret of the other tenant";
    let (s, v) = send(
        &app,
        as_other(req(
            "POST",
            "/api/eval/questions",
            Some(json!({"query":foreign_query,"expected_text":"orbit"})),
        )),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    let foreign = id_of(&v);

    // a foreign id answers 404 exactly like an unknown one, alone or next to an own id
    for ids in [json!([mine, foreign]), json!([foreign])] {
        let (s, v) = send(
            &app,
            req(
                "POST",
                "/api/eval",
                Some(json!({"question_ids":ids,"modes":["flat"]})),
            ),
        )
        .await;
        assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains(&foreign.to_string()), "{msg}");
        assert!(
            !msg.contains(&mine.to_string()),
            "only the missing id: {msg}"
        );
        assert!(
            !v.to_string().contains(foreign_query),
            "no foreign data: {v}"
        );
    }

    // nothing was evaluated or stored, for either tenant
    let (s, v) = send(&app, req("GET", "/api/eval/runs", None)).await;
    assert_eq!((s, v["items"].clone()), (StatusCode::OK, json!([])));
    let (s, v) = send(&app, as_other(req("GET", "/api/eval/runs", None))).await;
    assert_eq!((s, v["items"].clone()), (StatusCode::OK, json!([])));
    let stored: i64 = sqlx::query_scalar("SELECT count(*) FROM eval_run_results")
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(stored, 0);

    // the own id alone still runs, and its stored snapshot holds only this tenant's question
    let run = run_eval(&app, json!({"question_ids":[mine],"modes":["flat"]})).await;
    assert_eq!(run["questions"].as_array().unwrap().len(), 1, "{run}");
    assert_eq!(run["questions"][0]["id"], json!(mine));
    assert!(!run.to_string().contains(foreign_query), "{run}");
}

/// Migration 0013: a result row's tenant is its run's tenant, and never the nil uuid.
#[tokio::test]
async fn run_results_are_tied_to_their_runs_tenant() {
    let (app, db) = app().await;
    loaded_tree(&app).await;
    create_question(
        &app,
        json!({"query":"harpoon whale","expected_document":"sea.txt"}),
    )
    .await;
    let run = run_eval(&app, json!({"source":"tenant","modes":["flat"]})).await;
    let run_id = id_of(&run);

    for (tenant, sqlstate, constraint) in [
        (
            common::OTHER_TENANT,
            "23503",
            "eval_run_results_run_tenant_fkey",
        ),
        (
            "00000000-0000-0000-0000-000000000000",
            "23514",
            "eval_run_results_tenant_not_nil",
        ),
    ] {
        let err = sqlx::query(
            "INSERT INTO eval_run_results (run_id, tenant, idx, mode) VALUES ($1, $2, 99, 'flat')",
        )
        .bind(run_id)
        .bind(Uuid::parse_str(tenant).unwrap())
        .execute(&db.pool)
        .await
        .expect_err("a result row with a tenant other than its run's is refused");
        let e = err.as_database_error().expect("a database error");
        assert_eq!(e.code().as_deref(), Some(sqlstate), "{tenant}: {e}");
        assert_eq!(e.constraint(), Some(constraint), "{tenant}: {e}");
    }

    // deleting the run still cascades to its results through the composite key
    sqlx::query("DELETE FROM eval_runs WHERE id = $1")
        .bind(run_id)
        .execute(&db.pool)
        .await
        .unwrap();
    let left: i64 = sqlx::query_scalar("SELECT count(*) FROM eval_run_results WHERE run_id = $1")
        .bind(run_id)
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(left, 0);
}

// ---------- asynchronous runs (migration 0014) ----------

/// Holds an ACCESS EXCLUSIVE lock on `chunks` until committed: a run's scoring (the flat search
/// reads chunks) waits on it, while starting a run (stores, documents, eval_* tables) and reading
/// runs do not. So a run stays `running` for as long as the test needs, without a sleep.
async fn block_scoring(pool: &PgPool) -> sqlx::Transaction<'static, sqlx::Postgres> {
    let mut tx = pool.begin().await.unwrap();
    sqlx::query("LOCK TABLE chunks IN ACCESS EXCLUSIVE MODE")
        .execute(&mut *tx)
        .await
        .unwrap();
    tx
}

#[tokio::test]
async fn tenant_eval_runs_in_the_background_one_at_a_time() {
    let (app, db) = app().await;
    loaded_tree(&app).await;
    loaded_tree_as(&app, as_other).await;
    create_question(
        &app,
        json!({"query":"harpoon whale ship","expected_document":"sea.txt"}),
    )
    .await;
    create_question(
        &app,
        json!({"query":"whale ocean","expected_text":"this sentence appears in no document"}),
    )
    .await;
    let other_query = "telescope planet orbit";
    let (s, v) = send(
        &app,
        as_other(req(
            "POST",
            "/api/eval/questions",
            Some(json!({"query":other_query,"expected_document":"stars.txt"})),
        )),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");

    let holder = block_scoring(&db.pool).await;
    // answered while the scoring cannot even begin: the request does not wait for the run
    let started = start_eval(&app, json!({"source":"tenant","modes":["flat"]})).await;
    let id = id_of(&started);
    assert_eq!(
        started["progress"],
        json!({"done":0,"total":2}),
        "{started}"
    );
    assert_eq!(started["question_count"], 2, "{started}");

    let (s, v) = send(&app, req("GET", &format!("/api/eval/runs/{id}"), None)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["status"], "running", "{v}");
    assert_eq!(v["results"], json!([]), "nothing scored is shown: {v}");
    assert_eq!(v["reports"], Value::Null, "{v}");
    assert_eq!(v["verdict"], Value::Null, "{v}");
    assert_eq!(v["finished_at"], Value::Null, "{v}");
    assert_eq!(v["took_ms"], Value::Null, "{v}");
    assert_eq!(v["progress"]["total"], 2, "{v}");
    assert_eq!(v["questions"].as_array().unwrap().len(), 2, "{v}");
    let (s, page) = send(&app, req("GET", "/api/eval/runs", None)).await;
    assert_eq!(s, StatusCode::OK, "{page}");
    assert_eq!(page["items"][0]["id"], json!(id), "{page}");
    assert_eq!(page["items"][0]["status"], "running", "{page}");

    // single flight: any second tenant run is refused with its own code while this one runs
    for body in [
        json!({"source":"tenant"}),
        json!({"questions":[{"query":"q","expected_text":"x"}],"modes":["flat"]}),
    ] {
        let (s, v) = send(&app, req("POST", "/api/eval", Some(body.clone()))).await;
        assert_error(s, &v, StatusCode::CONFLICT, "eval_running");
    }
    // the built-in eval of the tenant is refused as before (`conflict`)
    let (s, v) = send(&app, req("POST", "/api/eval", None)).await;
    assert_error(s, &v, StatusCode::CONFLICT, "conflict");

    // isolation: the other tenant neither sees the running run nor is blocked by it
    let (s, v) = send(
        &app,
        as_other(req("GET", &format!("/api/eval/runs/{id}"), None)),
    )
    .await;
    assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");
    let (s, v) = send(&app, as_other(req("GET", "/api/eval/runs", None))).await;
    assert_eq!((s, v["items"].clone()), (StatusCode::OK, json!([])));
    let other = start_eval_as(&app, as_other, json!({"source":"tenant","modes":["flat"]})).await;
    let other_id = id_of(&other);
    let (s, v) = send(
        &app,
        req("GET", &format!("/api/eval/runs/{other_id}"), None),
    )
    .await;
    assert_error(s, &v, StatusCode::NOT_FOUND, "not_found");

    holder.commit().await.unwrap();
    let run = wait_run(&app, id).await;
    assert_eq!(run["status"], "completed", "{run}");
    assert_eq!(run["error"], Value::Null, "{run}");
    assert_eq!(run["progress"], json!({"done":2,"total":2}), "{run}");
    assert!(
        run["verdict"] == "PASS" || run["verdict"] == "FAIL",
        "{run}"
    );
    assert!(run["took_ms"].as_i64().is_some_and(|ms| ms >= 0), "{run}");
    let flat = report(&run, "flat");
    assert_eq!(flat["n"], 2, "{run}");
    assert!(approx(&flat["recall_at_10"], 0.5), "one hit of two: {flat}");
    let results = run["results"].as_array().unwrap();
    assert_eq!(results.len(), 2, "{run}");
    assert!(
        results[0]["rank"]
            .as_u64()
            .is_some_and(|k| (1..=10).contains(&k)),
        "{run}"
    );
    assert_eq!(results[1]["rank"], Value::Null, "{run}");
    assert!(!run.to_string().contains(other_query), "{run}");

    let other_run = wait_run_as(&app, as_other, other_id).await;
    assert_eq!(other_run["status"], "completed", "{other_run}");
    assert!(
        approx(&report(&other_run, "flat")["recall_at_10"], 1.0),
        "{other_run}"
    );
    assert_eq!(other_run["questions"].as_array().unwrap().len(), 1);

    // finished: the tenant's next run starts at once
    run_eval(&app, json!({"source":"tenant","modes":["flat"]})).await;
}

#[tokio::test]
async fn a_failed_run_is_stored_failed_with_its_error() {
    let (app, db) = app().await;
    loaded_tree(&app).await;
    create_question(
        &app,
        json!({"query":"harpoon whale","expected_document":"sea.txt"}),
    )
    .await;
    // scoring succeeds, storing its results fails
    sqlx::query(
        "CREATE FUNCTION refuse_results() RETURNS trigger LANGUAGE plpgsql AS \
         $$ BEGIN RAISE EXCEPTION 'results refused'; END $$",
    )
    .execute(&db.pool)
    .await
    .unwrap();
    sqlx::query(
        "CREATE TRIGGER refuse_results BEFORE INSERT ON eval_run_results \
         FOR EACH STATEMENT EXECUTE FUNCTION refuse_results()",
    )
    .execute(&db.pool)
    .await
    .unwrap();

    let started = start_eval(&app, json!({"source":"tenant","modes":["flat"]})).await;
    let run = wait_run(&app, id_of(&started)).await;
    assert_eq!(run["status"], "failed", "{run}");
    assert_eq!(
        run["error"], "internal server error",
        "a database error's detail stays in the server log: {run}"
    );
    assert!(run["finished_at"].is_string(), "{run}");
    assert!(run["took_ms"].as_i64().is_some_and(|ms| ms >= 0), "{run}");
    assert_eq!(run["verdict"], Value::Null, "{run}");
    assert_eq!(run["reports"], Value::Null, "{run}");
    assert_eq!(
        run["results"],
        json!([]),
        "the failed transaction stored none: {run}"
    );
    let (s, page) = send(&app, req("GET", "/api/eval/runs", None)).await;
    assert_eq!(s, StatusCode::OK, "{page}");
    assert_eq!(page["items"][0]["status"], "failed", "{page}");
    assert_eq!(page["items"][0]["error"], "internal server error", "{page}");

    // a failed run frees the tenant: with the fault gone the next run completes
    sqlx::query("DROP TRIGGER refuse_results ON eval_run_results")
        .execute(&db.pool)
        .await
        .unwrap();
    run_eval(&app, json!({"source":"tenant","modes":["flat"]})).await;
}

/// Inserts a `running` run of `tenant` as a worker would have left it, beating last `beat_age`
/// seconds ago (`None`: never beat, as a row from before migration 0015, started that long ago).
async fn insert_running(
    pool: &PgPool,
    tenant: &str,
    beat_age: Option<f64>,
    started_age: f64,
) -> Result<Uuid, sqlx::Error> {
    sqlx::query_scalar::<_, Uuid>(
        "INSERT INTO eval_runs (tenant, source, modes, beam, min_score, status, \
           started_at, questions, progress_total, heartbeat_at) \
         VALUES ($1, 'tenant', '{flat}', 3, 0, 'running', now() - make_interval(secs => $3), \
           '[]', 1, now() - make_interval(secs => $2)) RETURNING id",
    )
    .bind(Uuid::parse_str(tenant).unwrap())
    .bind(beat_age)
    .bind(started_age)
    .fetch_one(pool)
    .await
}

async fn status_of(pool: &PgPool, id: Uuid) -> (String, Option<String>, i32) {
    sqlx::query_as("SELECT status, error, progress_done FROM eval_runs WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .unwrap()
}

#[tokio::test]
async fn only_runs_whose_heartbeat_went_stale_are_swept() {
    use strata_rag::db::eval::{EVAL_STALE_AFTER, fail_stale_runs};
    let (app, db) = app().await;
    loaded_tree(&app).await;
    create_question(
        &app,
        json!({"query":"harpoon whale","expected_document":"sea.txt"}),
    )
    .await;
    // runs whose worker still beats (on another replica), though
    // they started long ago
    let mine = insert_running(&db.pool, common::TENANT, Some(1.0), 3600.0)
        .await
        .unwrap();
    let theirs = insert_running(&db.pool, common::OTHER_TENANT, Some(1.0), 3600.0)
        .await
        .unwrap();
    // one running run per tenant, in the table itself
    let err = insert_running(&db.pool, common::TENANT, Some(0.0), 0.0)
        .await
        .expect_err("a second running run of a tenant is refused");
    let e = err.as_database_error().expect("a database error");
    assert_eq!(e.code().as_deref(), Some("23505"), "{e}");
    assert_eq!(e.constraint(), Some("eval_runs_one_running"), "{e}");

    // a fresh heartbeat survives every sweep: at start / periodic, and the lazy ones on reads
    assert_eq!(
        fail_stale_runs(&db.pool, None, EVAL_STALE_AFTER)
            .await
            .unwrap(),
        0
    );
    let (s, v) = send(&app, req("GET", &format!("/api/eval/runs/{mine}"), None)).await;
    assert_eq!((s, v["status"].clone()), (StatusCode::OK, json!("running")));
    // the stored row is the guard, not this process's memory: it blocks the tenant's next run
    let (s, v) = send(
        &app,
        req("POST", "/api/eval", Some(json!({"source":"tenant"}))),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "eval_running");
    let (s, v) = send(&app, req("POST", "/api/eval", None)).await;
    assert_error(s, &v, StatusCode::CONFLICT, "conflict");

    // its worker died: the heartbeat goes past the threshold
    let stale = EVAL_STALE_AFTER.as_secs_f64() + 1.0;
    sqlx::query(
        "UPDATE eval_runs SET heartbeat_at = now() - make_interval(secs => $2) WHERE id = $1",
    )
    .bind(mine)
    .bind(stale)
    .execute(&db.pool)
    .await
    .unwrap();
    // single flight unblocks at once: the POST sweeps the tenant's stale run itself
    let run = run_eval(&app, json!({"source":"tenant","modes":["flat"]})).await;
    let (s, v) = send(&app, req("GET", &format!("/api/eval/runs/{mine}"), None)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["status"], "failed", "{v}");
    assert_eq!(v["error"], "interrupted", "{v}");
    assert!(v["finished_at"].is_string(), "{v}");
    assert_eq!(v["verdict"], Value::Null, "{v}");
    // the lazy sweep is the tenant's own: the other tenant's live run is untouched
    assert_eq!(status_of(&db.pool, theirs).await.0, "running");

    // a row that never beat (before migration 0015) ages from its start
    sqlx::query(
        "UPDATE eval_runs SET heartbeat_at = NULL, started_at = now() - make_interval(secs => $2) \
         WHERE id = $1",
    )
    .bind(theirs)
    .bind(stale)
    .execute(&db.pool)
    .await
    .unwrap();
    assert_eq!(
        fail_stale_runs(&db.pool, None, EVAL_STALE_AFTER)
            .await
            .unwrap(),
        1,
        "the other tenant's stale run, by the all-tenant sweep"
    );
    assert_eq!(
        status_of(&db.pool, theirs).await,
        ("failed".to_string(), Some("interrupted".to_string()), 0)
    );
    // finished runs are left alone
    assert_eq!(
        fail_stale_runs(&db.pool, None, std::time::Duration::ZERO)
            .await
            .unwrap(),
        0
    );
    let (s, v) = send(
        &app,
        req("GET", &format!("/api/eval/runs/{}", id_of(&run)), None),
    )
    .await;
    assert_eq!(
        (s, v["status"].clone()),
        (StatusCode::OK, json!("completed"))
    );
}

#[tokio::test]
async fn a_swept_runs_worker_writes_nothing_more() {
    use strata_rag::db::eval as db_eval;
    let db = common::TestDb::start().await;
    let id = insert_running(&db.pool, common::TENANT, Some(120.0), 120.0)
        .await
        .unwrap();
    assert_eq!(
        db_eval::fail_stale_runs(&db.pool, Some(common::tn()), db_eval::EVAL_STALE_AFTER)
            .await
            .unwrap(),
        1
    );
    let swept = ("failed".to_string(), Some("interrupted".to_string()), 0);
    assert_eq!(status_of(&db.pool, id).await, swept);
    // the worker's next beat tells it to stop, and writes nothing
    assert!(
        !db_eval::heartbeat(&db.pool, common::tn(), id, 1)
            .await
            .unwrap()
    );
    // its final writes are no-ops: the `failed` / `interrupted` row stands
    assert!(
        !db_eval::complete_run(&db.pool, common::tn(), id, "PASS", &json!([]), &[],)
            .await
            .unwrap()
    );
    assert!(
        !db_eval::fail_run(&db.pool, common::tn(), id, "boom")
            .await
            .unwrap()
    );
    assert_eq!(status_of(&db.pool, id).await, swept);
    let beat_still_old: bool = sqlx::query_scalar(
        "SELECT heartbeat_at < now() - interval '100 seconds' FROM eval_runs WHERE id = $1",
    )
    .bind(id)
    .fetch_one(&db.pool)
    .await
    .unwrap();
    assert!(
        beat_still_old,
        "the swept run's heartbeat was not refreshed"
    );
}

#[tokio::test]
async fn a_live_worker_beats_and_a_swept_one_stops() {
    use strata_rag::db::eval::fail_stale_runs;
    let (app, db) = app().await;
    loaded_tree(&app).await;
    create_question(
        &app,
        json!({"query":"harpoon whale","expected_document":"sea.txt"}),
    )
    .await;

    // a real worker, its scoring held: it keeps beating while it waits
    let holder = block_scoring(&db.pool).await;
    let id = id_of(&start_eval(&app, json!({"source":"tenant","modes":["flat"]})).await);
    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
    assert_eq!(
        fail_stale_runs(&db.pool, None, std::time::Duration::from_secs(3))
            .await
            .unwrap(),
        0,
        "a worker that beats about once a second is never 3 s stale"
    );
    assert_eq!(status_of(&db.pool, id).await.0, "running");

    // swept anyway (as if its beats had not landed): the worker must not overwrite `failed`
    assert_eq!(
        fail_stale_runs(&db.pool, None, std::time::Duration::ZERO)
            .await
            .unwrap(),
        1
    );
    holder.commit().await.unwrap();
    // long enough for a beat to see the sweep, or for scoring to finish and try to complete
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    assert_eq!(
        status_of(&db.pool, id).await,
        ("failed".to_string(), Some("interrupted".to_string()), 0)
    );
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM eval_run_results WHERE run_id = $1")
        .bind(id)
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(n, 0, "the swept run's results were not stored");
    // the tenant is free again
    run_eval(&app, json!({"source":"tenant","modes":["flat"]})).await;
}

async fn panicking_scoring() -> strata_rag::error::Result<()> {
    panic!("scoring panicked")
}

#[tokio::test]
async fn a_panicking_eval_task_fails_its_run() {
    use strata_rag::api::eval_api::spawn_supervised;
    let (engine, db) = common::test_engine().await;
    let engine = Arc::new(engine);
    let id = insert_running(&db.pool, common::TENANT, Some(0.0), 0.0)
        .await
        .unwrap();
    let task = spawn_supervised(&engine, common::tn(), id, panicking_scoring());
    assert_eq!(engine.eval_tasks.in_flight(), vec![(common::tn(), id)]);
    task.await.expect("the supervisor itself does not panic");
    assert_eq!(
        status_of(&db.pool, id).await,
        (
            "failed".to_string(),
            Some("internal server error".to_string()),
            0
        ),
        "a panic fails the run with the generic message, not `running` until swept"
    );
    let has_finish: bool =
        sqlx::query_scalar("SELECT finished_at IS NOT NULL FROM eval_runs WHERE id = $1")
            .bind(id)
            .fetch_one(&db.pool)
            .await
            .unwrap();
    assert!(has_finish);
    assert!(
        engine.eval_tasks.in_flight().is_empty(),
        "the ended task is untracked"
    );
}

#[tokio::test]
async fn shutdown_fails_this_process_running_runs_only() {
    use strata_rag::api::eval_api::spawn_supervised;
    use strata_rag::engine::EVAL_SHUTDOWN;
    let (engine, db) = common::test_engine().await;
    let engine = Arc::new(engine);
    // this process's run, its scoring still going
    let mine = insert_running(&db.pool, common::TENANT, Some(0.0), 0.0)
        .await
        .unwrap();
    // another tenant's run, scored by another replica
    let theirs = insert_running(&db.pool, common::OTHER_TENANT, Some(0.0), 0.0)
        .await
        .unwrap();
    let (release, wait) = tokio::sync::oneshot::channel::<()>();
    let task = spawn_supervised(&engine, common::tn(), mine, async move {
        let _ = wait.await;
        Ok(())
    });
    assert_eq!(engine.eval_tasks.in_flight(), vec![(common::tn(), mine)]);

    let n = engine
        .eval_tasks
        .fail_on_shutdown(&db.pool, std::time::Duration::from_secs(5))
        .await;
    assert_eq!(n, 1);
    assert_eq!(
        status_of(&db.pool, mine).await,
        ("failed".to_string(), Some(EVAL_SHUTDOWN.to_string()), 0)
    );
    let (took, finished): (Option<f64>, bool) = sqlx::query_as(
        "SELECT extract(epoch FROM finished_at - started_at)::float8, finished_at IS NOT NULL \
         FROM eval_runs WHERE id = $1",
    )
    .bind(mine)
    .fetch_one(&db.pool)
    .await
    .unwrap();
    assert!(finished && took.unwrap() >= 0.0, "{took:?}");
    assert_eq!(
        status_of(&db.pool, theirs).await,
        ("running".to_string(), None, 0),
        "another tenant's run (another process's) is untouched"
    );
    // its worker's late writes are no-ops: the `shutdown` row stands
    assert!(
        !strata_rag::db::eval::complete_run(&db.pool, common::tn(), mine, "PASS", &json!([]), &[])
            .await
            .unwrap()
    );
    release.send(()).unwrap();
    task.await.unwrap();
    assert_eq!(
        status_of(&db.pool, mine).await.1.as_deref(),
        Some(EVAL_SHUTDOWN)
    );
    assert!(engine.eval_tasks.in_flight().is_empty());
    // nothing left to fail: a second call (after the drain) is a no-op
    assert_eq!(
        engine
            .eval_tasks
            .fail_on_shutdown(&db.pool, std::time::Duration::from_secs(5))
            .await,
        0
    );
}
