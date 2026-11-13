//! The metering outbox (`strata_rag::metering`) against a real Postgres and a real Redis:
//! - every paid call is recorded as it returns, independent of the business transaction: an
//!   ingest that rolls back, or a reembed killed half-way, has recorded each call it paid for;
//! - the batch writer: one INSERT per batch, the CLI's exit flush (`cli::run_command`), a burst
//!   beyond 10 000 events, a row Postgres refuses (dropped, counted, the writer goes on);
//! - the drainer: billing's 20-field map, rows marked sent, no transaction open across Redis,
//!   an unreadable row skipped, a Redis outage survived, two drainers never double-send, prune;
//! - `strata serve`'s wiring (drainer running; flush before its last pass) and `/health`.
//!
//! Redis: `REDIS_URL` when set (CI: the archetype's `redis_image`, started in dind), otherwise one
//! throwaway Redis container per test binary (the pattern of `tests/common`). Never skipped.

mod common;

use common::{TENANT, TestDb, test_config, test_engine, tn};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sqlx::PgPool;
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::inference::{
    EmbedProvider, Embedder, FakeEmbedder, InferenceHandle, InputType, Limits, RerankProvider,
};
use strata_rag::metering::{
    Attribution, CallKind, Caller, Drainer, DrainerConfig, Funding, Metering, ProviderMeta,
    QuantityUnit, Usage, WriterConfig, extra, tenant_level_project_id,
};
use strata_rag::routing::BoxFut;
use strata_rag::text::WhitespaceCounter;
use testcontainers_modules::redis::{REDIS_PORT, Redis};
use testcontainers_modules::testcontainers::runners::AsyncRunner;
use testcontainers_modules::testcontainers::{ContainerAsync, ImageExt};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::OnceCell;
use uuid::Uuid;

struct Server {
    url: String,
    _container: Option<ContainerAsync<Redis>>,
}

static SERVER: OnceCell<Server> = OnceCell::const_new();

async fn redis_url() -> &'static str {
    &SERVER
        .get_or_init(|| async {
            if let Ok(url) = std::env::var("REDIS_URL")
                && !url.is_empty()
            {
                return Server {
                    url,
                    _container: None,
                };
            }
            let c = Redis::default()
                .with_tag("8.10.1-alpine")
                .start()
                .await
                .expect("start a redis container (is Docker running?)");
            let port = c.get_host_port_ipv4(REDIS_PORT).await.expect("redis port");
            Server {
                url: format!("redis://127.0.0.1:{port}/"),
                _container: Some(c),
            }
        })
        .await
        .url
}

/// A fresh stream name per test: tests share the server.
fn stream() -> String {
    format!("billing:costs:test:{}", Uuid::now_v7())
}

/// Every entry of `stream`, oldest first, as field maps.
async fn entries(url: &str, stream: &str) -> Vec<HashMap<String, String>> {
    let client = redis::Client::open(url).unwrap();
    let mut conn = client.get_multiplexed_async_connection().await.unwrap();
    let rows: Vec<(String, HashMap<String, String>)> = redis::cmd("XRANGE")
        .arg(stream)
        .arg("-")
        .arg("+")
        .query_async(&mut conn)
        .await
        .unwrap();
    rows.into_iter().map(|(_, fields)| fields).collect()
}

/// A Redis URL whose traffic goes through a local proxy: every client write is delayed by
/// `delay`, and while `up` is false every connection is closed at once (Redis "down").
async fn proxy(url: &str, delay: Duration, up: Arc<AtomicBool>) -> String {
    let rest = url.strip_prefix("redis://").expect("a redis:// url");
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let (userinfo, upstream) = match authority.rsplit_once('@') {
        Some((u, h)) => (format!("{u}@"), h.to_string()),
        None => (String::new(), authority.to_string()),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        while let Ok((client, _)) = listener.accept().await {
            if !up.load(Ordering::SeqCst) {
                drop(client);
                continue;
            }
            let upstream = upstream.clone();
            tokio::spawn(async move {
                let Ok(server) = tokio::net::TcpStream::connect(&upstream).await else {
                    return;
                };
                let (mut cr, mut cw) = client.into_split();
                let (mut sr, mut sw) = server.into_split();
                let up = async move {
                    let mut buf = vec![0u8; 64 * 1024];
                    loop {
                        let n = match cr.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => n,
                        };
                        if !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                        if sw.write_all(&buf[..n]).await.is_err() {
                            break;
                        }
                    }
                };
                let down = async move {
                    let _ = tokio::io::copy(&mut sr, &mut cw).await;
                };
                tokio::join!(up, down);
            });
        }
    });
    format!("redis://{userinfo}127.0.0.1:{port}/{path}")
}

/// Short backoffs so the outage test recovers quickly.
fn fast() -> DrainerConfig {
    DrainerConfig {
        interval: Duration::from_millis(20),
        idle_max: Duration::from_millis(200),
        batch: 500,
        retry_min: Duration::from_millis(20),
        retry_max: Duration::from_millis(200),
        ..DrainerConfig::default()
    }
}

/// A batch writer that only INSERTs on a flush (or 10 000 queued events): whatever reaches the
/// outbox without a flush was written by someone else (a write transaction).
fn held_writer(pool: &PgPool) -> Arc<Metering> {
    Metering::start_with(
        pool.clone(),
        Arc::new(WhitespaceCounter),
        WriterConfig {
            batch_max: 10_000,
            linger: Duration::from_secs(3600),
        },
    )
}

#[derive(Debug)]
struct Row {
    event_key: Uuid,
    status: String,
    attempts: i32,
    payload: Value,
    /// the inserting transaction
    xmin: String,
}

async fn rows(pool: &PgPool) -> Vec<Row> {
    let raw: Vec<(Uuid, String, i32, String, String)> = sqlx::query_as(
        "SELECT event_key, status, attempts, payload::text, xmin::text \
         FROM metering_outbox ORDER BY id",
    )
    .fetch_all(pool)
    .await
    .unwrap();
    raw.into_iter()
        .map(|(event_key, status, attempts, payload, xmin)| Row {
            event_key,
            status,
            attempts,
            payload: serde_json::from_str(&payload).unwrap(),
            xmin,
        })
        .collect()
}

fn service_keys(rows: &[Row]) -> Vec<String> {
    let mut k: Vec<String> = rows
        .iter()
        .map(|r| r.payload["service_key"].as_str().unwrap().to_string())
        .collect();
    k.sort();
    k
}

fn an_embed_event(project: Uuid) -> (Attribution, ProviderMeta, Usage) {
    (
        Attribution {
            tenant: tn(),
            caller: Caller {
                project_id: Some(project),
                ..Caller::default()
            },
        },
        ProviderMeta::tenant_registered("https://api.voyageai.com/v1", "voyage-3"),
        Usage {
            call: CallKind::Embed,
            input: 7,
            output: 0,
            unit: QuantityUnit::Tokens,
            latency_ms: 12,
            extra: extra(&[("texts", json!(1))], true),
        },
    )
}

/// A root router and one leaf under it; returns (root, leaf).
async fn tree(engine: &Engine) -> (Uuid, Uuid) {
    let root = engine
        .create_store(
            tn(),
            NewStore {
                parent_id: None,
                kind: StoreKind::Router,
                name: "library".into(),
                description: "all books".into(),
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
                name: "sea".into(),
                description: "whales".into(),
            },
        )
        .await
        .unwrap()
        .id;
    (root, leaf)
}

/// Emit → batch writer → outbox row → drainer → billing's field map on the stream, the row marked
/// sent with the same `event_key`; a second pass sends nothing.
#[tokio::test]
async fn the_drainer_xadds_billings_field_map_and_marks_the_row_sent() {
    let url = redis_url().await;
    let stream = stream();
    let db = TestDb::start().await;
    let m = Metering::start(db.pool.clone(), Arc::new(WhitespaceCounter));
    let project = Uuid::now_v7();
    let (attr, meta, usage) = an_embed_event(project);
    m.emit(&attr, &meta, usage);
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0, "in the outbox");
    let before = rows(&db.pool).await;
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].status, "pending");
    assert!(entries(url, &stream).await.is_empty(), "nothing sent yet");

    let mut d = Drainer::new(db.pool.clone(), url, &stream, fast()).unwrap();
    let r = d.drain_once().await.unwrap();
    assert_eq!((r.sent, r.failed, r.redis_error), (1, 0, None));

    let got = entries(url, &stream).await;
    assert_eq!(got.len(), 1);
    let e = &got[0];
    for (k, v) in [
        ("tenant_id", TENANT),
        ("project_id", project.to_string().as_str()),
        ("service_key", "strata-rag-embed"),
        ("cost_source", "strata-rag"),
        ("provider", "voyageai"),
        ("model", "voyage-3"),
        ("served_model", "voyage-3"),
        ("input_tokens", "7"),
        ("output_tokens", "0"),
        ("quantity_unit", "tokens"),
        ("funding", "tenant_byo"),
        ("latency_ms", "12"),
        ("app_id", ""),
        ("actor_id", ""),
        ("license_id", ""),
        ("subject_id", ""),
        ("provider_request_id", ""),
    ] {
        assert_eq!(e.get(k).map(String::as_str), Some(v), "{k}: {e:?}");
    }
    assert_eq!(e.len(), 20, "every field of the event, nothing else: {e:?}");
    let key = Uuid::parse_str(&e["event_key"]).unwrap();
    assert_eq!(key.get_version_num(), 7);
    assert_eq!(key, before[0].event_key, "the row's key is the event's key");
    assert!(e["timestamp"].ends_with('Z'), "{}", e["timestamp"]);
    let extra: Value = serde_json::from_str(&e["usage_extra"]).unwrap();
    assert_eq!(extra["byo"], json!(true));
    assert_eq!(extra["call"], "embed");

    let after = rows(&db.pool).await;
    assert_eq!(after[0].status, "sent");
    let sent_at: Option<sqlx::types::chrono::DateTime<sqlx::types::chrono::Utc>> =
        sqlx::query_scalar("SELECT sent_at FROM metering_outbox")
            .fetch_one(&db.pool)
            .await
            .unwrap();
    assert!(sent_at.is_some());
    let r = d.drain_once().await.unwrap();
    assert_eq!((r.sent, r.failed), (0, 0), "a sent row is not sent again");
    assert_eq!(entries(url, &stream).await.len(), 1);
}

/// The query path has no transaction: its embed + rerank events wait in memory (nothing in the
/// outbox before the flush), then reach the outbox in ONE multi-row INSERT.
#[tokio::test]
async fn a_querys_events_reach_the_outbox_through_the_batch_writer() {
    let (engine, db) = test_engine().await;
    let (_, leaf) = tree(&engine).await;
    engine
        .ingest(
            tn(),
            leaf,
            "w.txt".into(),
            None,
            "The white whale surfaced beside the ship.",
        )
        .await
        .unwrap();
    let m = held_writer(&db.pool);
    let engine = engine.with_metering(Arc::clone(&m));
    let req: strata_rag::query::QueryRequest = serde_json::from_value(
        json!({"query": "white whale", "route_mode": "embed", "min_score": 0.0}),
    )
    .unwrap();
    engine.query(tn(), &req).await.unwrap();
    assert!(
        rows(&db.pool).await.is_empty(),
        "queued, not written inline"
    );
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let got = rows(&db.pool).await;
    assert_eq!(
        service_keys(&got),
        ["strata-rag-embed", "strata-rag-rerank"]
    );
    assert_eq!(got[0].xmin, got[1].xmin, "one INSERT for the batch");
    let tenant_project = tenant_level_project_id(tn().id()).to_string();
    assert!(
        got.iter()
            .all(|r| r.payload["project_id"] == tenant_project)
    );
    assert!(got.iter().all(|r| r.status == "pending" && r.attempts == 0));
}

/// An ingest's embed calls are recorded as they return, through the batch writer (not held for
/// the ingest's transaction): queued at once, in the outbox after the flush.
#[tokio::test]
async fn ingest_records_each_paid_call_as_it_returns() {
    let (engine, db) = test_engine().await;
    let (_, leaf) = tree(&engine).await;
    let m = held_writer(&db.pool);
    let engine = engine.with_metering(Arc::clone(&m));
    let text = "The white whale surfaced beside the ship. Ahab watched the white whale. ".repeat(6);
    engine
        .ingest(tn(), leaf, "w.txt".into(), None, &text)
        .await
        .unwrap();
    let queued = m.queued();
    assert!(queued >= 1, "handed to the writer as the calls returned");
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let got = rows(&db.pool).await;
    assert_eq!(got.len(), queued);
    assert!(got.iter().all(|r| {
        r.payload["usage_extra"]
            .as_str()
            .unwrap()
            .contains("document")
    }));
}

/// An embed provider that, on a document embed, fills the target leaf behind ingest's back, so
/// ingest's locked cap re-check fails and its transaction rolls back AFTER the provider was paid.
struct FillsTheLeaf {
    pool: PgPool,
    leaf: std::sync::OnceLock<Uuid>,
    document_calls: AtomicUsize,
}

impl EmbedProvider for FillsTheLeaf {
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        input_type: InputType,
    ) -> BoxFut<'a, strata_rag::Result<Vec<Vec<f32>>>> {
        Box::pin(async move {
            if input_type == InputType::Document {
                self.document_calls.fetch_add(1, Ordering::SeqCst);
                sqlx::query("UPDATE stores SET token_count = 1000000000 WHERE id = $1")
                    .bind(self.leaf.get().copied())
                    .execute(&self.pool)
                    .await
                    .unwrap();
            }
            FakeEmbedder.embed(texts)
        })
    }
}

struct ZeroReranker;

impl RerankProvider for ZeroReranker {
    fn rerank<'a>(
        &'a self,
        _query: &'a str,
        passages: &'a [String],
    ) -> BoxFut<'a, strata_rag::Result<Vec<f32>>> {
        let n = passages.len();
        Box::pin(async move { Ok(vec![0.0; n]) })
    }
}

/// The ingest transaction rolls back after the embed calls were paid: no document, and each paid
/// call is still recorded once (commonpractice: a paid provider call is always billed).
#[tokio::test]
async fn a_rolled_back_ingest_still_records_its_paid_calls() {
    let db = TestDb::start().await;
    let provider = Arc::new(FillsTheLeaf {
        pool: db.pool.clone(),
        leaf: std::sync::OnceLock::new(),
        document_calls: AtomicUsize::new(0),
    });
    let engine = Engine::new(
        db.pool.clone(),
        InferenceHandle::new(provider.clone(), Arc::new(ZeroReranker), Limits::default()),
        Arc::new(WhitespaceCounter),
        test_config(),
    );
    let (_, leaf) = tree(&engine).await;
    provider.leaf.set(leaf).unwrap();
    let m = held_writer(&db.pool);
    let engine = engine.with_metering(Arc::clone(&m));

    let err = engine
        .ingest(
            tn(),
            leaf,
            "w.txt".into(),
            None,
            "The white whale surfaced beside the ship.",
        )
        .await
        .unwrap_err();
    assert_eq!(err.code(), "leaf_full", "{err:?}");
    let docs: i64 = sqlx::query_scalar("SELECT count(*) FROM documents")
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(docs, 0, "the ingest rolled back");
    let paid = provider.document_calls.load(Ordering::SeqCst);
    assert!(paid >= 1);
    assert_eq!(m.queued(), paid, "every paid call handed to the writer");
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let got = rows(&db.pool).await;
    assert_eq!(got.len(), paid, "each paid call recorded once: {got:?}");
}

/// Document embeds succeed until the `stall_at`-th, which never returns (a reembed killed
/// half-way: the task is aborted while it waits).
struct StallsMidReembed {
    armed: AtomicBool,
    stall_at: usize,
    started: AtomicUsize,
    returned: AtomicUsize,
    /// armed non-document calls that returned (reembed's store descriptions)
    others: AtomicUsize,
    stalled: tokio::sync::Notify,
}

impl EmbedProvider for StallsMidReembed {
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        input_type: InputType,
    ) -> BoxFut<'a, strata_rag::Result<Vec<Vec<f32>>>> {
        Box::pin(async move {
            if input_type == InputType::Document && self.armed.load(Ordering::SeqCst) {
                if self.started.fetch_add(1, Ordering::SeqCst) + 1 == self.stall_at {
                    self.stalled.notify_one();
                    std::future::pending::<()>().await;
                }
                let v = FakeEmbedder.embed(texts);
                self.returned.fetch_add(1, Ordering::SeqCst);
                return v;
            }
            let v = FakeEmbedder.embed(texts);
            if self.armed.load(Ordering::SeqCst) {
                self.others.fetch_add(1, Ordering::SeqCst);
            }
            v
        })
    }
}

/// A reembed killed half-way (its task aborted while a provider call hangs, before its write
/// transaction ever starts) has recorded every call that already returned: they are queued for
/// the writer at once, and in the outbox after the flush.
#[tokio::test]
async fn a_reembed_killed_half_way_has_recorded_every_call_it_paid_for() {
    let db = TestDb::start().await;
    let provider = Arc::new(StallsMidReembed {
        armed: AtomicBool::new(false),
        stall_at: 3,
        started: AtomicUsize::new(0),
        returned: AtomicUsize::new(0),
        others: AtomicUsize::new(0),
        stalled: tokio::sync::Notify::new(),
    });
    let engine = Engine::new(
        db.pool.clone(),
        InferenceHandle::new(provider.clone(), Arc::new(ZeroReranker), Limits::default()),
        Arc::new(WhitespaceCounter),
        test_config(),
    );
    let (root, leaf) = tree(&engine).await;
    for i in 0..5 {
        engine
            .ingest(
                tn(),
                leaf,
                format!("doc{i}.txt"),
                None,
                &format!("Document number {i} tells of whales and ships and the sea."),
            )
            .await
            .unwrap();
    }
    provider.armed.store(true, Ordering::SeqCst);
    let m = held_writer(&db.pool);
    let engine = Arc::new(engine.with_metering(Arc::clone(&m)));
    let task = tokio::spawn({
        let engine = Arc::clone(&engine);
        async move { engine.reembed_tree(tn(), root, 1).await }
    });
    provider.stalled.notified().await;
    // still running (stalled on the third call): the two returned calls are already queued for
    // the writer, not held for a transaction that has not even begun
    assert_eq!(provider.returned.load(Ordering::SeqCst), 2);
    assert_eq!(
        m.queued(),
        2 + provider.others.load(Ordering::SeqCst),
        "recorded as each call returned"
    );
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    let paid = provider.returned.load(Ordering::SeqCst);
    assert_eq!(paid, 2, "two chunk embeds returned before the kill");
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let document_rows = rows(&db.pool)
        .await
        .iter()
        .filter(|r| {
            r.payload["usage_extra"]
                .as_str()
                .unwrap()
                .contains("\"input_type\":\"document\"")
        })
        .count();
    assert_eq!(document_rows, paid, "every returned call has its row");
}

/// A burst far beyond the old 10 000-event queue is recorded in full, nothing dropped.
#[tokio::test]
async fn a_burst_of_more_than_ten_thousand_events_is_not_dropped() {
    let db = TestDb::start().await;
    let m = Metering::start(db.pool.clone(), Arc::new(WhitespaceCounter));
    const N: usize = 15_000;
    for _ in 0..N {
        let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
        m.emit(&attr, &meta, usage);
    }
    assert_eq!(m.dropped(), 0);
    assert_eq!(m.flush(Duration::from_secs(60)).await, 0);
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM metering_outbox")
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(n as usize, N);
}

/// A row Postgres refuses (a NUL in the payload: jsonb rejects `\u0000`) does not block the
/// writer: the rest of its batch lands, the refused event is counted as dropped, later events
/// still land.
#[tokio::test]
async fn a_row_postgres_refuses_is_dropped_without_blocking_the_writer() {
    let db = TestDb::start().await;
    let m = Metering::start(db.pool.clone(), Arc::new(WhitespaceCounter));
    let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
    m.emit(&attr, &meta, usage.clone());
    let bad = ProviderMeta {
        provider: "x".into(),
        model: "bad\u{0}model".into(),
        funding: Funding::LocalGpu,
        local: true,
    };
    m.emit(&attr, &bad, usage.clone());
    m.emit(&attr, &meta, usage.clone());
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0, "not stuck");
    assert_eq!(rows(&db.pool).await.len(), 2);
    assert_eq!(m.dropped(), 1);
    m.emit(&attr, &meta, usage);
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    assert_eq!(rows(&db.pool).await.len(), 3, "the writer goes on");
}

/// `strata serve`'s wiring (`cli::start_drainer` / `cli::stop_metering`): the drainer runs in the
/// background while serving, and the shutdown flushes the writer BEFORE the drainer's last pass,
/// so an event recorded just before shutdown is on the stream when it returns.
#[tokio::test]
async fn serve_wiring_drains_while_running_and_flushes_before_stopping() {
    let url = redis_url().await;
    let stream = stream();
    let db = TestDb::start().await;
    let mut cfg = test_config();
    cfg.metering_redis_url = Some(strata_rag::config::Secret::new(url));
    cfg.metering_stream = stream.clone();
    let off = Engine::new(
        db.pool.clone(),
        InferenceHandle::fake(),
        Arc::new(WhitespaceCounter),
        cfg.clone(),
    );
    assert!(
        strata_rag::cli::start_drainer(&off).unwrap().is_none(),
        "metering off: no drainer"
    );
    let engine = off.with_metering(Metering::start(
        db.pool.clone(),
        Arc::new(WhitespaceCounter),
    ));
    let m = Arc::clone(engine.metering.as_ref().unwrap());
    let drainer = strata_rag::cli::start_drainer(&engine)
        .unwrap()
        .expect("metering on: a drainer");

    let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
    m.emit(&attr, &meta, usage.clone());
    let deadline = Instant::now() + Duration::from_secs(15);
    while entries(url, &stream).await.is_empty() {
        assert!(Instant::now() < deadline, "the running drainer never sent");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    m.emit(&attr, &meta, usage);
    strata_rag::cli::stop_metering(&engine, Some(drainer)).await;
    assert_eq!(
        entries(url, &stream).await.len(),
        2,
        "the event recorded just before shutdown was flushed, then sent"
    );
    assert!(rows(&db.pool).await.iter().all(|r| r.status == "sent"));
}

/// `/health` reports the backlog: pending rows, the oldest one's age, the queue, the drops.
#[tokio::test]
async fn health_reports_the_metering_backlog() {
    let db = TestDb::start().await;
    let m = Metering::start(db.pool.clone(), Arc::new(WhitespaceCounter));
    for _ in 0..2 {
        let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
        m.emit(&attr, &meta, usage);
    }
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let engine = Arc::new(
        Engine::new(
            db.pool.clone(),
            InferenceHandle::fake(),
            Arc::new(WhitespaceCounter),
            test_config(),
        )
        .with_metering(m),
    );
    let report = engine.health().await;
    let v = serde_json::to_value(&report).unwrap();
    assert_eq!(v["metering"]["outbox_pending"], 2, "{v}");
    assert!(
        v["metering"]["outbox_oldest_pending_age_s"]
            .as_i64()
            .unwrap()
            >= 0
    );
    assert_eq!(v["metering"]["queued"], 0);
    assert_eq!(v["metering"]["dropped_total"], 0);
}

/// The drainer holds no database transaction while Redis answers: during a slow XADD there is no
/// `idle in transaction` session, and the claimed rows are leased (not due) meanwhile.
#[tokio::test]
async fn no_transaction_stays_open_across_the_xadd() {
    let url = redis_url().await;
    let stream = stream();
    let db = TestDb::start().await;
    let m = Metering::start(db.pool.clone(), Arc::new(WhitespaceCounter));
    for _ in 0..3 {
        let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
        m.emit(&attr, &meta, usage);
    }
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let slow = proxy(
        url,
        Duration::from_millis(1500),
        Arc::new(AtomicBool::new(true)),
    )
    .await;
    let mut d = Drainer::new(db.pool.clone(), &slow, &stream, fast()).unwrap();
    // connect first (the handshake is slow too), then time the pass
    assert_eq!(d.drain_once().await.unwrap().sent, 3);
    for _ in 0..3 {
        let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
        m.emit(&attr, &meta, usage);
    }
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let pass = tokio::spawn(async move { d.drain_once().await });
    tokio::time::sleep(Duration::from_millis(600)).await;
    let open: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pg_stat_activity \
         WHERE datname = current_database() AND state LIKE 'idle in transaction%'",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap();
    assert_eq!(open, 0, "a transaction is open while the XADD runs");
    let leased: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM metering_outbox WHERE status = 'pending' AND next_attempt_at > now()",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap();
    assert_eq!(leased, 3, "the claimed rows are leased during the XADD");
    assert_eq!(pass.await.unwrap().unwrap().sent, 3);
    assert_eq!(entries(url, &stream).await.len(), 6);
}

/// One unreadable payload does not fail its batch: the other rows are sent, the bad one is
/// rescheduled with its error.
#[tokio::test]
async fn an_unreadable_row_does_not_fail_the_batch() {
    let url = redis_url().await;
    let stream = stream();
    let db = TestDb::start().await;
    let m = Metering::start(db.pool.clone(), Arc::new(WhitespaceCounter));
    for _ in 0..4 {
        let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
        m.emit(&attr, &meta, usage);
    }
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    sqlx::query(
        "UPDATE metering_outbox SET payload = '[1, 2]'::jsonb \
         WHERE id = (SELECT id FROM metering_outbox ORDER BY id LIMIT 1)",
    )
    .execute(&db.pool)
    .await
    .unwrap();
    let mut d = Drainer::new(db.pool.clone(), url, &stream, fast()).unwrap();
    let r = d.drain_once().await.unwrap();
    assert_eq!((r.sent, r.failed), (3, 1), "{r:?}");
    assert_eq!(entries(url, &stream).await.len(), 3);
    let (status, attempts, error): (String, i32, Option<String>) = sqlx::query_as(
        "SELECT status, attempts, last_error FROM metering_outbox ORDER BY id LIMIT 1",
    )
    .fetch_one(&db.pool)
    .await
    .unwrap();
    assert_eq!((status.as_str(), attempts), ("pending", 1));
    assert!(error.unwrap().contains("JSON object"));
}

/// The CLI's exit flush: a `strata query` run through `cli::run_command` returns with its events
/// in the outbox although the writer (held) would not have written them on its own. Removing the
/// flush from `run_command` fails this test.
#[tokio::test]
async fn the_cli_exit_flush_writes_queued_events_to_the_outbox() {
    let (engine, db) = test_engine().await;
    let (_, leaf) = tree(&engine).await;
    engine
        .ingest(tn(), leaf, "w.txt".into(), None, "whales and ships")
        .await
        .unwrap();
    let m = held_writer(&db.pool);
    let engine = Arc::new(engine.with_metering(Arc::clone(&m)));
    let code = strata_rag::cli::run_command(
        engine,
        strata_rag::cli::Command::Query {
            tenant: strata_rag::cli::TenantArg { tenant: tn() },
            query: "whales".into(),
            top_k: 8,
            mode: strata_rag::cli::ModeArg::Embed,
            beam: 3,
            min_score: 0.0,
            expand: false,
            source_ids: Vec::new(),
            metadata: None,
            json: true,
        },
    )
    .await
    .unwrap();
    assert_eq!(
        format!("{code:?}"),
        format!("{:?}", std::process::ExitCode::SUCCESS)
    );
    assert_eq!(
        m.flush(Duration::ZERO).await,
        0,
        "run_command returned with events still queued: the exit flush did not run"
    );
    assert_eq!(
        service_keys(&rows(&db.pool).await),
        ["strata-rag-embed", "strata-rag-rerank"]
    );
}

/// Redis down: the drainer delivers nothing and every row stays pending; once Redis is back the
/// same rows (same event keys) are delivered, once each.
#[tokio::test]
async fn redis_down_keeps_the_rows_and_delivers_them_on_recovery() {
    let url = redis_url().await;
    let stream = stream();
    let db = TestDb::start().await;
    let m = Metering::start(db.pool.clone(), Arc::new(WhitespaceCounter));
    for _ in 0..3 {
        let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
        m.emit(&attr, &meta, usage);
    }
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let up = Arc::new(AtomicBool::new(false));
    let via = proxy(url, Duration::ZERO, Arc::clone(&up)).await;
    let mut d = Drainer::new(db.pool.clone(), &via, &stream, fast()).unwrap();
    for _ in 0..3 {
        let r = d.drain_once().await.unwrap();
        assert_eq!(r.sent, 0);
        assert!(r.redis_error.is_some(), "{r:?}");
    }
    let held = rows(&db.pool).await;
    assert_eq!(held.len(), 3);
    assert!(held.iter().all(|r| r.status == "pending"), "{held:?}");
    assert!(entries(url, &stream).await.is_empty());

    up.store(true, Ordering::SeqCst);
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut sent = 0;
    while sent < 3 {
        assert!(Instant::now() < deadline, "not delivered after recovery");
        sent += d.drain_once().await.unwrap().sent;
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    let got = entries(url, &stream).await;
    assert_eq!(got.len(), 3, "each row once");
    let on_stream: HashSet<Uuid> = got
        .iter()
        .map(|e| Uuid::parse_str(&e["event_key"]).unwrap())
        .collect();
    let in_outbox: HashSet<Uuid> = held.iter().map(|r| r.event_key).collect();
    assert_eq!(on_stream, in_outbox);
    assert!(rows(&db.pool).await.iter().all(|r| r.status == "sent"));
}

/// Two drainers on one outbox (two replicas): every row reaches the stream exactly once, and both
/// drainers did part of the work (`FOR UPDATE SKIP LOCKED`). Each pipeline is held back by a slow
/// proxy, so their claims overlap in time.
#[tokio::test]
async fn two_drainers_never_send_a_row_twice() {
    let url = redis_url().await;
    let stream = stream();
    let db = TestDb::start().await;
    let m = Metering::start(db.pool.clone(), Arc::new(WhitespaceCounter));
    const N: usize = 300;
    for _ in 0..N {
        let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
        m.emit(&attr, &meta, usage);
    }
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let slow = proxy(
        url,
        Duration::from_millis(100),
        Arc::new(AtomicBool::new(true)),
    )
    .await;
    let cfg = DrainerConfig {
        batch: 20,
        ..fast()
    };
    let run = |mut d: Drainer| async move {
        let mut sent = 0;
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let r = d.drain_once().await.unwrap();
            assert!(r.redis_error.is_none(), "{r:?}");
            sent += r.sent;
            if r.sent == 0 {
                return sent;
            }
            assert!(Instant::now() < deadline);
        }
    };
    let a = Drainer::new(db.pool.clone(), &slow, &stream, cfg).unwrap();
    let b = Drainer::new(db.pool.clone(), &slow, &stream, cfg).unwrap();
    let (sa, sb) = tokio::join!(run(a), run(b));
    assert_eq!(sa + sb, N, "{sa} + {sb}");
    assert!(sa > 0 && sb > 0, "both drained: {sa} / {sb}");
    let got = entries(url, &stream).await;
    assert_eq!(got.len(), N, "no row sent twice");
    let keys: HashSet<&str> = got.iter().map(|e| e["event_key"].as_str()).collect();
    assert_eq!(keys.len(), N);
    assert!(rows(&db.pool).await.iter().all(|r| r.status == "sent"));
}

/// Retention: sent rows older than the window are deleted; recent sent rows and pending rows stay.
#[tokio::test]
async fn sent_rows_are_pruned_after_the_retention_window() {
    let url = redis_url().await;
    let stream = stream();
    let db = TestDb::start().await;
    let m = Metering::start(db.pool.clone(), Arc::new(WhitespaceCounter));
    for _ in 0..3 {
        let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
        m.emit(&attr, &meta, usage);
    }
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    let mut d = Drainer::new(db.pool.clone(), url, &stream, fast()).unwrap();
    assert_eq!(d.drain_once().await.unwrap().sent, 3);
    // two of the sent rows are old; one more event is still pending
    sqlx::query(
        "UPDATE metering_outbox SET sent_at = now() - interval '8 days' \
         WHERE id IN (SELECT id FROM metering_outbox ORDER BY id LIMIT 2)",
    )
    .execute(&db.pool)
    .await
    .unwrap();
    let (attr, meta, usage) = an_embed_event(Uuid::now_v7());
    m.emit(&attr, &meta, usage);
    assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    assert_eq!(d.prune().await.unwrap(), 2);
    let left = rows(&db.pool).await;
    let mut statuses: Vec<&str> = left.iter().map(|r| r.status.as_str()).collect();
    statuses.sort_unstable();
    assert_eq!(statuses, ["pending", "sent"]);
}
