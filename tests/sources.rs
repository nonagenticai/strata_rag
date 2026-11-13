//! Document source provenance (migration 0004): `source_id` / `source_run` / `metadata` on ingest,
//! list and query hits; same-source re-ingest refreshes without re-embedding; and the re-sync prune
//! `DELETE /api/sources/{source_id}/documents`.
mod common;

use common::tn;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Router;
use axum::body::Body;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use strata_rag::config::{ApiToken, TokenScope};
use strata_rag::db::stores::{self, NewStore, StoreKind, StoreRow};
use strata_rag::engine::Engine;
use strata_rag::error::{Result, StrataError};
use strata_rag::inference::{Embedder, FakeEmbedder, FakeReranker, InferenceHandle};
use strata_rag::ingest::{IngestOutcome, IngestRequest, PruneReport, PruneRequest, SourceTag};
use tower::ServiceExt;
use uuid::Uuid;

/// The fake embedder, counting its calls.
struct CountingEmbedder(Arc<AtomicUsize>);

impl Embedder for CountingEmbedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        FakeEmbedder.embed(texts)
    }
}

/// `common::test_engine` with a counting embedder.
async fn counting_engine() -> (Engine, common::TestDb, Arc<AtomicUsize>) {
    let db = common::TestDb::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = Engine::new(
        db.pool.clone(),
        InferenceHandle::in_process(
            Box::new(CountingEmbedder(Arc::clone(&calls))),
            Box::new(FakeReranker),
        ),
        Arc::new(strata_rag::text::WhitespaceCounter),
        common::test_config(),
    );
    (engine, db, calls)
}

struct Tree {
    root: Uuid,
    mid: Uuid,
    leaf: Uuid,
    leaf2: Uuid,
}

/// root(router) ── mid(router) ── leaf ; root ── leaf2
async fn build_tree(engine: &Engine) -> Tree {
    let mk = |parent_id, kind, name: &str| NewStore {
        parent_id,
        kind,
        name: name.to_string(),
        description: format!("{name} store"),
    };
    let root = engine
        .create_store(tn(), mk(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    let mid = engine
        .create_store(tn(), mk(Some(root), StoreKind::Router, "mid"))
        .await
        .unwrap()
        .id;
    let leaf = engine
        .create_store(tn(), mk(Some(mid), StoreKind::Leaf, "leaf"))
        .await
        .unwrap()
        .id;
    let leaf2 = engine
        .create_store(tn(), mk(Some(root), StoreKind::Leaf, "leaf2"))
        .await
        .unwrap()
        .id;
    Tree {
        root,
        mid,
        leaf,
        leaf2,
    }
}

fn tag(id: Option<&str>, run: Option<&str>, metadata: Option<Value>) -> SourceTag {
    SourceTag {
        source_id: id.map(str::to_string),
        source_run: run.map(str::to_string),
        metadata,
    }
}

async fn ingest(
    engine: &Engine,
    store_id: Uuid,
    title: Option<&str>,
    text: &str,
    tag: SourceTag,
) -> IngestOutcome {
    engine
        .ingest_document(
            tn(),
            IngestRequest {
                store_id,
                source: "test".to_string(),
                title: title.map(str::to_string),
                text,
                tag,
            },
        )
        .await
        .unwrap()
}

fn created(o: IngestOutcome) -> strata_rag::db::documents::DocumentRow {
    match o {
        IngestOutcome::Created(d) => d,
        other => panic!("expected Created, got {other:?}"),
    }
}

fn existing(o: IngestOutcome) -> strata_rag::db::documents::DocumentRow {
    match o {
        IngestOutcome::Existing(d) => d,
        other => panic!("expected Existing, got {other:?}"),
    }
}

/// `n` distinct whitespace tokens.
fn words(prefix: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{prefix}{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

async fn count(pool: &PgPool, sql: &'static str) -> i64 {
    sqlx::query_scalar(sql).fetch_one(pool).await.unwrap()
}

async fn chunk_ids(pool: &PgPool, doc: Uuid) -> Vec<Uuid> {
    sqlx::query_scalar("SELECT id FROM chunks WHERE document_id = $1 ORDER BY ord")
        .bind(doc)
        .fetch_all(pool)
        .await
        .unwrap()
}

async fn row(engine: &Engine, id: Uuid) -> StoreRow {
    let mut conn = engine.pool.acquire().await.unwrap();
    stores::get_store(&mut conn, tn(), id).await.unwrap()
}

fn assert_vec_close(a: Option<&[f32]>, b: Option<&[f32]>, what: &str) {
    match (a, b) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(a.len(), b.len(), "{what}");
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-5, "{what}: {x} vs {y}");
            }
        }
        _ => panic!("{what}: {a:?} vs {b:?}"),
    }
}

// ---------- engine: dedupe refresh ----------

#[tokio::test]
async fn same_source_reingest_refreshes_run_metadata_and_title_without_embedding() {
    let (engine, _db, calls) = counting_engine().await;
    let t = build_tree(&engine).await;
    let text = words("alpha", 80);

    let first = created(
        ingest(
            &engine,
            t.leaf,
            Some("T1"),
            &text,
            tag(Some("web-1"), Some("r1"), Some(json!({"v": 1}))),
        )
        .await,
    );
    assert_eq!(first.source_id.as_deref(), Some("web-1"));
    assert_eq!(first.source_run.as_deref(), Some("r1"));
    assert_eq!(first.metadata["v"], 1);
    let embeds = calls.load(Ordering::SeqCst);
    assert!(embeds >= 1);
    let chunks = chunk_ids(&engine.pool, first.id).await;
    let leaf_before = row(&engine, t.leaf).await;

    // same text, same source, new run + metadata, no title: refreshed, title kept, nothing embedded
    let again = existing(
        ingest(
            &engine,
            t.leaf,
            None,
            &text,
            tag(
                Some(" web-1 "),
                Some("r2"),
                Some(json!({"v": 2, "url": "u"})),
            ),
        )
        .await,
    );
    assert_eq!(again.id, first.id);
    assert_eq!(again.source_run.as_deref(), Some("r2"));
    assert_eq!(
        again.metadata,
        json!({"v": 2, "url": "u"}).as_object().unwrap().clone()
    );
    assert_eq!(
        again.title.as_deref(),
        Some("T1"),
        "an untitled re-delivery keeps the title"
    );
    assert_eq!(again.chunk_count, first.chunk_count);
    assert_eq!(calls.load(Ordering::SeqCst), embeds, "no embed call");
    assert_eq!(
        chunk_ids(&engine.pool, first.id).await,
        chunks,
        "chunks untouched"
    );
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        1
    );
    let leaf_after = row(&engine, t.leaf).await;
    assert_eq!(leaf_after.token_count, leaf_before.token_count);
    assert_eq!(leaf_after.content_vec, leaf_before.content_vec);

    // a title given → replaced; no run / no metadata given → both KEPT (never erased); the
    // `source` label always takes the request's
    let titled = existing(
        engine
            .ingest_document(
                tn(),
                IngestRequest {
                    store_id: t.leaf,
                    source: "renamed.md".to_string(),
                    title: Some("T2".to_string()),
                    text: &text,
                    tag: tag(Some("web-1"), None, None),
                },
            )
            .await
            .unwrap(),
    );
    assert_eq!(titled.title.as_deref(), Some("T2"));
    assert_eq!(titled.source, "renamed.md");
    assert_eq!(titled.source_run.as_deref(), Some("r2"));
    assert_eq!(
        Value::Object(titled.metadata.clone()),
        json!({"v": 2, "url": "u"})
    );
    type Stored = (String, Option<String>, Option<String>, Value);
    let stored = || async {
        sqlx::query_as::<_, Stored>(
            "SELECT source, title, source_run, metadata FROM documents WHERE id = $1",
        )
        .bind(first.id)
        .fetch_one(&engine.pool)
        .await
        .unwrap()
    };
    assert_eq!(
        stored().await,
        (
            "renamed.md".to_string(),
            Some("T2".to_string()),
            Some("r2".to_string()),
            json!({"v": 2, "url": "u"})
        )
    );
    // metadata given is a full replace (no merge); an empty object clears it
    let replaced = existing(
        ingest(
            &engine,
            t.leaf,
            None,
            &text,
            tag(Some("web-1"), Some("r3"), Some(json!({"only": true}))),
        )
        .await,
    );
    assert_eq!(Value::Object(replaced.metadata), json!({"only": true}));
    assert_eq!(replaced.source_run.as_deref(), Some("r3"));
    let cleared = existing(
        ingest(
            &engine,
            t.leaf,
            None,
            &text,
            tag(Some("web-1"), None, Some(json!({}))),
        )
        .await,
    );
    assert!(cleared.metadata.is_empty());
    assert_eq!(cleared.source_run.as_deref(), Some("r3"));
    assert_eq!(stored().await.3, json!({}));
    assert_eq!(calls.load(Ordering::SeqCst), embeds, "still no embed call");
}

#[tokio::test]
async fn same_text_from_another_or_no_source_leaves_the_original_untouched() {
    let (engine, _db) = common::test_engine().await;
    let t = build_tree(&engine).await;
    let text = words("beta", 40);
    let original = created(
        ingest(
            &engine,
            t.leaf,
            Some("Orig"),
            &text,
            tag(Some("s1"), Some("r1"), Some(json!({"k": "v"}))),
        )
        .await,
    );
    for other in [
        tag(Some("s2"), Some("r9"), Some(json!({"k": "other"}))),
        tag(None, None, Some(json!({"k": "other"}))),
    ] {
        let got = existing(ingest(&engine, t.leaf, Some("New"), &text, other).await);
        assert_eq!(got.id, original.id);
        assert_eq!(got.source_id.as_deref(), Some("s1"));
        assert_eq!(got.source_run.as_deref(), Some("r1"));
        assert_eq!(got.metadata, original.metadata);
        assert_eq!(got.title.as_deref(), Some("Orig"));
    }
    // a source-less document is not adopted by a source either
    let plain = words("gamma", 20);
    let doc = created(ingest(&engine, t.leaf, None, &plain, SourceTag::default()).await);
    let got = existing(
        ingest(
            &engine,
            t.leaf,
            None,
            &plain,
            tag(Some("s1"), Some("r2"), None),
        )
        .await,
    );
    assert_eq!(got.id, doc.id);
    assert_eq!((got.source_id, got.source_run), (None, None));
    let (sid, md): (Option<String>, Value) =
        sqlx::query_as("SELECT source_id, metadata FROM documents WHERE id = $1")
            .bind(doc.id)
            .fetch_one(&engine.pool)
            .await
            .unwrap();
    assert_eq!((sid, md), (None, json!({})));
}

/// Both race windows around the unlocked dedupe: (1) the same source's document is committed while
/// a new ingest embeds — the locked re-check in `write_document` must refresh it, not just report it;
/// (2) the document found by the unlocked lookup is deleted before the refresh takes the leaf lock —
/// the ingest must create it afresh, not fail.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dedupe_refresh_is_race_safe_under_the_leaf_lock() {
    let (engine, _db) = common::test_engine().await;
    let t = build_tree(&engine).await;

    // (1) ingest blocks on the leaf lock in write_document; meanwhile the holder commits the doc
    let text = words("delta", 30);
    let hash = Sha256::digest(strata_rag::text::normalize(&text).as_bytes()).to_vec();
    let mut holder = engine.pool.begin().await.unwrap();
    let holder_pid = common::backend_pid(&mut holder).await;
    sqlx::query("SELECT 1 FROM stores WHERE id = $1 FOR UPDATE")
        .bind(t.leaf)
        .execute(&mut *holder)
        .await
        .unwrap();
    let pool = engine.pool.clone();
    let leaf = t.leaf;
    let raced_id = Uuid::now_v7();
    let (out, ()) = tokio::join!(
        ingest(
            &engine,
            t.leaf,
            None,
            &text,
            tag(Some("s"), Some("r2"), Some(json!({"run": 2}))),
        ),
        async move {
            common::wait_until_blocked_by(&pool, holder_pid).await;
            sqlx::query(
                "INSERT INTO documents (id, store_id, source, content_hash, token_count, source_id, \
                 source_run, metadata) VALUES ($1, $2, 'x', $3, 30, 's', 'r1', '{\"run\":1}')",
            )
            .bind(raced_id)
            .bind(leaf)
            .bind(&hash)
            .execute(&mut *holder)
            .await
            .unwrap();
            holder.commit().await.unwrap();
        }
    );
    let doc = existing(out);
    assert_eq!(doc.id, raced_id);
    assert_eq!(doc.source_run.as_deref(), Some("r2"));
    assert_eq!(doc.metadata["run"], 2);
    let run: Option<String> = sqlx::query_scalar("SELECT source_run FROM documents WHERE id = $1")
        .bind(raced_id)
        .fetch_one(&engine.pool)
        .await
        .unwrap();
    assert_eq!(run.as_deref(), Some("r2"));

    // (2) the refresh blocks on the leaf lock; the holder deletes the document first
    let text2 = words("eps", 30);
    let gone = created(
        ingest(
            &engine,
            t.leaf,
            None,
            &text2,
            tag(Some("s"), Some("r1"), None),
        )
        .await,
    );
    let mut holder = engine.pool.begin().await.unwrap();
    let holder_pid = common::backend_pid(&mut holder).await;
    sqlx::query("SELECT 1 FROM stores WHERE id = $1 FOR UPDATE")
        .bind(t.leaf)
        .execute(&mut *holder)
        .await
        .unwrap();
    let pool = engine.pool.clone();
    let gone_id = gone.id;
    let (out, ()) = tokio::join!(
        ingest(
            &engine,
            t.leaf,
            None,
            &text2,
            tag(Some("s"), Some("r2"), None)
        ),
        async move {
            common::wait_until_blocked_by(&pool, holder_pid).await;
            sqlx::query("DELETE FROM documents WHERE id = $1")
                .bind(gone_id)
                .execute(&mut *holder)
                .await
                .unwrap();
            holder.commit().await.unwrap();
        }
    );
    let fresh = created(out);
    assert_ne!(fresh.id, gone_id);
    assert_eq!(fresh.source_run.as_deref(), Some("r2"));
    let chunks: i64 = sqlx::query_scalar("SELECT count(*) FROM chunks WHERE document_id = $1")
        .bind(fresh.id)
        .fetch_one(&engine.pool)
        .await
        .unwrap();
    assert_eq!(chunks, fresh.chunk_count);
}

// ---------- engine: prune ----------

fn prune(
    source: &str,
    except: Option<&str>,
    store: Option<Uuid>,
    allow_empty: bool,
) -> PruneRequest {
    PruneRequest {
        source_id: source.to_string(),
        except_run: except.map(str::to_string),
        // no run to keep = an explicit whole-source wipe
        all: except.is_none(),
        store_id: store,
        allow_empty,
    }
}

fn report(deleted: u64, kept: i64, stores: Vec<Uuid>, refused_stores: Vec<Uuid>) -> PruneReport {
    PruneReport {
        deleted,
        kept,
        stores,
        refused_stores,
    }
}

fn sorted(mut v: Vec<Uuid>) -> Vec<Uuid> {
    v.sort();
    v
}

/// The 409's report, or a panic.
fn refused(err: StrataError) -> PruneReport {
    assert_eq!(
        (err.code(), err.status()),
        ("empty_sync_run", 409),
        "{err:?}"
    );
    match err {
        StrataError::EmptySyncRun { report, .. } => *report,
        other => panic!("expected EmptySyncRun, got {other:?}"),
    }
}

#[tokio::test]
async fn prune_keeps_the_run_recomputes_the_tree_and_invalidates_the_cache() {
    let (engine, _db) = common::test_engine().await;
    let t = build_tree(&engine).await;
    let s_r2 = tag(Some("s"), Some("r2"), None);
    // the state a fresh sync of run r2 alone would produce
    let k1 = created(ingest(&engine, t.leaf, None, &words("keep", 50), s_r2.clone()).await);
    let k2 = created(ingest(&engine, t.leaf2, None, &words("also", 40), s_r2.clone()).await);
    let other = created(
        ingest(
            &engine,
            t.leaf,
            None,
            &words("other", 30),
            tag(Some("t"), None, None),
        )
        .await,
    );
    let ids = [t.root, t.mid, t.leaf, t.leaf2];
    let mut fresh = Vec::new();
    for id in ids {
        fresh.push(row(&engine, id).await);
    }
    // stale documents of earlier runs (and one with no run)
    ingest(
        &engine,
        t.leaf,
        None,
        &words("stale", 60),
        tag(Some("s"), Some("r1"), None),
    )
    .await;
    ingest(
        &engine,
        t.leaf,
        None,
        &words("norun", 25),
        tag(Some("s"), None, None),
    )
    .await;
    ingest(
        &engine,
        t.leaf2,
        None,
        &words("old", 35),
        tag(Some("s"), Some("r1"), None),
    )
    .await;
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        6
    );

    // a run nobody carries → 409, every leaf refused, nothing deleted
    let mut rep = refused(
        engine
            .prune_source(tn(), prune("s", Some("r9"), None, false))
            .await
            .unwrap_err(),
    );
    rep.refused_stores.sort();
    assert_eq!(rep, report(0, 5, vec![], sorted(vec![t.leaf, t.leaf2])));
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        6
    );

    // a router as store scope is refused; an unknown store is 404
    let err = engine
        .prune_source(tn(), prune("s", Some("r2"), Some(t.mid), false))
        .await
        .unwrap_err();
    assert_eq!(err.code(), "invalid_target");
    let err = engine
        .prune_source(tn(), prune("s", Some("r2"), Some(Uuid::now_v7()), false))
        .await
        .unwrap_err();
    assert_eq!(err.code(), "not_found");

    // a whole-source wipe must be explicit: no except_run needs all=true; both is ambiguous
    for (except, all) in [(None, false), (Some("r2"), true)] {
        let mut req = prune("s", except, None, false);
        req.all = all;
        let err = engine.prune_source(tn(), req).await.unwrap_err();
        assert_eq!(err.code(), "invalid_input", "{except:?} all={all}");
    }
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        6
    );

    // store scope: only leaf2's stale document
    let rep = engine
        .prune_source(tn(), prune("s", Some("r2"), Some(t.leaf2), false))
        .await
        .unwrap();
    assert_eq!(rep, report(1, 1, vec![t.leaf2], vec![]));

    // prime the routing cache, then prune the rest of the source
    let cached = engine.cache.get(&engine.pool, tn()).await.unwrap();
    let rep = engine
        .prune_source(tn(), prune(" s ", Some("r2"), None, false))
        .await
        .unwrap();
    assert_eq!(rep, report(2, 2, vec![t.leaf], vec![]));
    let reloaded = engine.cache.get(&engine.pool, tn()).await.unwrap();
    assert!(
        !Arc::ptr_eq(&cached, &reloaded),
        "the routing cache was invalidated"
    );

    let mut left: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM documents")
        .fetch_all(&engine.pool)
        .await
        .unwrap();
    left.sort();
    let mut want = vec![k1.id, k2.id, other.id];
    want.sort();
    assert_eq!(left, want);
    let orphans = count(
        &engine.pool,
        "SELECT count(*) FROM chunks c WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id)",
    )
    .await;
    assert_eq!(orphans, 0);

    // every store is back to the fresh-sync state, and the reloaded cache serves it
    for (id, want) in ids.iter().zip(&fresh) {
        let got = row(&engine, *id).await;
        assert_eq!(
            (got.token_count, got.chunk_count),
            (want.token_count, want.chunk_count),
            "{}",
            want.name
        );
        assert_vec_close(
            got.content_vec.as_deref(),
            want.content_vec.as_deref(),
            &want.name,
        );
        assert_vec_close(Some(&got.profile_vec), Some(&want.profile_vec), &want.name);
        assert_vec_close(
            Some(&reloaded.nodes[id].profile_vec),
            Some(&want.profile_vec),
            &want.name,
        );
    }

    // idempotent: nothing stale left
    let rep = engine
        .prune_source(tn(), prune("s", Some("r2"), None, false))
        .await
        .unwrap();
    assert_eq!(rep, report(0, 2, vec![], vec![]));

    // allow_empty prunes even though nobody carries the run
    let rep = engine
        .prune_source(tn(), prune("s", Some("r9"), None, true))
        .await
        .unwrap();
    assert_eq!(rep.deleted, 2);
    assert_eq!(rep.kept, 0);
    assert!(rep.refused_stores.is_empty());
    let mut stores = rep.stores.clone();
    stores.sort();
    let mut want = vec![t.leaf, t.leaf2];
    want.sort();
    assert_eq!(stores, want);
    for id in ids {
        let r = row(&engine, id).await;
        let want_tokens = if id == t.leaf2 { 0 } else { 30 };
        assert_eq!(r.token_count, want_tokens, "{}", r.name);
    }

    // unknown source → zeros (also with except_run), no-run prune deletes the whole source
    for req in [
        prune("nobody", None, None, false),
        prune("nobody", Some("r1"), None, false),
    ] {
        let rep = engine.prune_source(tn(), req).await.unwrap();
        assert_eq!(rep, report(0, 0, vec![], vec![]));
    }
    let rep = engine
        .prune_source(tn(), prune("t", None, None, false))
        .await
        .unwrap();
    assert_eq!(rep, report(1, 0, vec![t.leaf], vec![]));
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        0
    );
    assert_eq!(count(&engine.pool, "SELECT count(*) FROM chunks").await, 0);
    assert_eq!(row(&engine, t.root).await.token_count, 0);

    // blank ids are 422
    let err = engine
        .prune_source(tn(), prune("  ", None, None, false))
        .await
        .unwrap_err();
    assert_eq!(err.code(), "invalid_input");
    let err = engine
        .prune_source(tn(), prune("s", Some(""), None, false))
        .await
        .unwrap_err();
    assert_eq!(err.code(), "invalid_input");
}

/// The guard is per leaf: a leaf that carries the run is pruned, one that does not is left
/// untouched and named in `refused_stores`, and the call is still a 409 carrying the report.
#[tokio::test]
async fn prune_guard_refuses_only_the_leaf_without_the_run() {
    let (engine, _db) = common::test_engine().await;
    let t = build_tree(&engine).await;
    let keep = created(
        ingest(
            &engine,
            t.leaf,
            None,
            &words("fresh", 20),
            tag(Some("s"), Some("R"), None),
        )
        .await,
    );
    ingest(
        &engine,
        t.leaf,
        None,
        &words("stale", 20),
        tag(Some("s"), Some("Q"), None),
    )
    .await;
    // leaf2's sync failed: it only has the old run
    let untouched = created(
        ingest(
            &engine,
            t.leaf2,
            None,
            &words("old", 20),
            tag(Some("s"), Some("Q"), None),
        )
        .await,
    );
    let leaf2_before = row(&engine, t.leaf2).await;

    let rep = refused(
        engine
            .prune_source(tn(), prune("s", Some("R"), None, false))
            .await
            .unwrap_err(),
    );
    assert_eq!(rep, report(1, 2, vec![t.leaf], vec![t.leaf2]));
    let mut left: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM documents")
        .fetch_all(&engine.pool)
        .await
        .unwrap();
    left.sort();
    assert_eq!(left, sorted(vec![keep.id, untouched.id]));
    assert_eq!(row(&engine, t.leaf).await.token_count, 20);
    assert_eq!(
        row(&engine, t.leaf2).await.token_count,
        leaf2_before.token_count
    );

    // scoped to the refused leaf: still refused; allow_empty prunes it
    let rep = refused(
        engine
            .prune_source(tn(), prune("s", Some("R"), Some(t.leaf2), false))
            .await
            .unwrap_err(),
    );
    assert_eq!(rep, report(0, 1, vec![], vec![t.leaf2]));
    let rep = engine
        .prune_source(tn(), prune("s", Some("R"), None, true))
        .await
        .unwrap();
    assert_eq!(rep, report(1, 1, vec![t.leaf2], vec![]));
}

/// The guard is decided under the leaf lock, not from a snapshot taken before it: a run-R document
/// that lands in the leaf while the prune waits for the lock lets the prune proceed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prune_guard_is_decided_under_the_leaf_lock() {
    let (engine, _db) = common::test_engine().await;
    let t = build_tree(&engine).await;
    let stale = created(
        ingest(
            &engine,
            t.leaf,
            None,
            &words("stale", 20),
            tag(Some("s"), Some("Q"), None),
        )
        .await,
    );
    let mut holder = engine.pool.begin().await.unwrap();
    let holder_pid = common::backend_pid(&mut holder).await;
    sqlx::query("SELECT 1 FROM stores WHERE id = $1 FOR UPDATE")
        .bind(t.leaf)
        .execute(&mut *holder)
        .await
        .unwrap();
    let pool = engine.pool.clone();
    let leaf = t.leaf;
    let landed = Uuid::now_v7();
    let (rep, ()) = tokio::join!(
        engine.prune_source(tn(), prune("s", Some("R"), None, false)),
        async move {
            common::wait_until_blocked_by(&pool, holder_pid).await;
            sqlx::query(
                "INSERT INTO documents (id, store_id, source, content_hash, token_count, source_id, \
                 source_run) VALUES ($1, $2, 'x', '\\x01', 5, 's', 'R')",
            )
            .bind(landed)
            .bind(leaf)
            .execute(&mut *holder)
            .await
            .unwrap();
            holder.commit().await.unwrap();
        }
    );
    assert_eq!(rep.unwrap(), report(1, 1, vec![t.leaf], vec![]));
    let left: Vec<Uuid> = sqlx::query_scalar("SELECT id FROM documents")
        .fetch_all(&engine.pool)
        .await
        .unwrap();
    assert_eq!(left, vec![landed]);
    assert_ne!(left[0], stale.id);
}

// ---------- HTTP ----------

const TOKEN: &str = "sources-token";

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(common::with_tenant(req))
        .await
        .expect("infallible");
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

fn req(method: &str, uri: &str, body: Option<Value>) -> Request<Body> {
    let b = Request::builder()
        .method(method)
        .uri(uri)
        .header(AUTHORIZATION, format!("Bearer {TOKEN}"));
    match body {
        Some(v) => b
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

async fn app_with_tree() -> (Router, Arc<Engine>, Tree, common::TestDb) {
    let (mut engine, db) = common::test_engine().await;
    engine.cfg.api_tokens = vec![ApiToken::new(TokenScope::AnyTenant, TOKEN)];
    let t = build_tree(&engine).await;
    let engine = Arc::new(engine);
    (strata_rag::api::router(Arc::clone(&engine)), engine, t, db)
}

fn assert_error(s: StatusCode, v: &Value, want: StatusCode, code: &str) {
    assert_eq!(s, want, "{v}");
    assert_eq!(v["error"]["code"], code, "{v}");
}

#[tokio::test]
async fn http_source_fields_round_trip_through_ingest_list_and_query() {
    let (app, engine, t, _db) = app_with_tree().await;
    let md = json!({"url": "https://example.test/whales", "lang": "en", "tags": ["sea"]});
    let (s, v) = send(
        &app,
        req(
            "POST",
            &format!("/api/stores/{}/documents", t.leaf),
            Some(json!({
                "text": "The white whale breached beside the ship at dawn.",
                "title": "Whales",
                "source_id": "web-1",
                "source_run": "run-1",
                "metadata": md,
            })),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{v}");
    assert_eq!(v["status"], "created");
    assert_eq!(v["source_id"], "web-1");
    assert_eq!(v["source_run"], "run-1");
    assert_eq!(v["metadata"], md);
    let doc_id = v["id"].as_str().unwrap().to_string();

    // text/plain with the ids as query params; no metadata → {}
    let (s, plain) = send(
        &app,
        Request::builder()
            .method("POST")
            .uri(format!(
                "/api/stores/{}/documents?source_id=web-2&source_run=run-1",
                t.leaf
            ))
            .header(AUTHORIZATION, format!("Bearer {TOKEN}"))
            .header(CONTENT_TYPE, "text/plain")
            .body(Body::from("A lighthouse keeper watched the storm."))
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED, "{plain}");
    assert_eq!(plain["source_id"], "web-2");
    assert_eq!(plain["metadata"], json!({}));

    // same text, same source, new run → 200 existing with the new fields
    let md2 = json!({"url": "https://example.test/whales", "rev": 2});
    let (s, v) = send(
        &app,
        req(
            "POST",
            &format!("/api/stores/{}/documents", t.leaf),
            Some(json!({
                "text": "The white whale breached beside the ship at dawn.",
                "source_id": "web-1",
                "source_run": "run-2",
                "metadata": md2,
            })),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["status"], "existing");
    assert_eq!(v["id"], doc_id.as_str());
    assert_eq!(v["source_run"], "run-2");
    assert_eq!(v["metadata"], md2);
    assert_eq!(v["title"], "Whales");

    let (s, page) = send(
        &app,
        req("GET", &format!("/api/stores/{}/documents", t.leaf), None),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{page}");
    let item = page["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["id"] == doc_id.as_str())
        .expect("listed");
    assert_eq!(item["source_id"], "web-1");
    assert_eq!(item["source_run"], "run-2");
    assert_eq!(item["metadata"], md2);

    // `/api/ask` answers with the same `QueryResult`s, so its sources carry these fields too
    let (s, v) = send(
        &app,
        req(
            "POST",
            "/api/query",
            Some(json!({"query": "white whale ship", "route_mode": "embed", "min_score": 0.0})),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let results = v["results"].as_array().unwrap();
    let hit = results
        .iter()
        .find(|h| h["document_id"] == doc_id.as_str())
        .unwrap_or_else(|| panic!("hit for the document in {v}"));
    assert_eq!(hit["source_id"], "web-1");
    assert_eq!(hit["source_run"], "run-2");
    assert_eq!(hit["metadata"], md2);
    for h in results.iter().filter(|h| h["document_id"] == plain["id"]) {
        assert_eq!(h["source_id"], "web-2");
        assert_eq!(h["metadata"], json!({}));
    }
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        2
    );
}

#[tokio::test]
async fn http_rejects_bad_source_fields_and_writes_nothing() {
    let (app, engine, t, _db) = app_with_tree().await;
    let uri = format!("/api/stores/{}/documents", t.leaf);
    let base = |extra: Value| {
        let mut b = json!({"text": "Some text about ships."});
        b.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        b
    };
    for (extra, want, code) in [
        (
            json!({"metadata": [1, 2]}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"metadata": "x"}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"source_run": "r1"}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"source_id": "   "}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"source_id": "s".repeat(201)}),
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_input",
        ),
        (
            json!({"source_id": "s", "metadata": {"blob": "x".repeat(64 * 1024)}}),
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
        ),
    ] {
        let (s, v) = send(&app, req("POST", &uri, Some(base(extra.clone())))).await;
        assert_error(s, &v, want, code);
    }
    // source_run as a query param without a source_id
    let (s, v) = send(
        &app,
        Request::builder()
            .method("POST")
            .uri(format!("{uri}?source_run=r1"))
            .header(AUTHORIZATION, format!("Bearer {TOKEN}"))
            .header(CONTENT_TYPE, "text/plain")
            .body(Body::from("Some text about ships."))
            .unwrap(),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        0
    );
}

#[tokio::test]
async fn http_list_filters_by_source_id_across_pages() {
    let (app, _engine, t, _db) = app_with_tree().await;
    let uri = format!("/api/stores/{}/documents", t.leaf);
    let mut want = Vec::new();
    for i in 0..5 {
        let (s, v) = send(
            &app,
            req(
                "POST",
                &uri,
                Some(json!({"text": format!("whale number {i} swam"), "source_id": "s"})),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{v}");
        want.push(v["id"].as_str().unwrap().to_string());
        let (s, _) = send(
            &app,
            req(
                "POST",
                &uri,
                Some(json!({"text": format!("ship number {i} sailed"), "source_id": "t"})),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);
    }
    let (s, _) = send(
        &app,
        req("POST", &uri, Some(json!({"text": "no source at all"}))),
    )
    .await;
    assert_eq!(s, StatusCode::CREATED);

    let mut got = Vec::new();
    let mut sizes = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let q = match &cursor {
            Some(c) => format!("{uri}?source_id=s&limit=2&cursor={c}"),
            None => format!("{uri}?source_id=s&limit=2"),
        };
        let (s, page) = send(&app, req("GET", &q, None)).await;
        assert_eq!(s, StatusCode::OK, "{page}");
        let items = page["items"].as_array().unwrap();
        sizes.push(items.len());
        for i in items {
            assert_eq!(i["source_id"], "s", "{i}");
            got.push(i["id"].as_str().unwrap().to_string());
        }
        match page["next"].as_str() {
            Some(n) => cursor = Some(n.to_string()),
            None => break,
        }
    }
    assert_eq!(sizes, [2, 2, 1]);
    want.reverse(); // newest first
    assert_eq!(got, want);

    let (s, v) = send(&app, req("GET", &format!("{uri}?source_id=%20"), None)).await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    let (s, page) = send(&app, req("GET", &format!("{uri}?source_id=nobody"), None)).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(page["items"], json!([]));
    let (s, page) = send(&app, req("GET", &uri, None)).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(page["items"].as_array().unwrap().len(), 11);
}

#[tokio::test]
async fn http_prune_route() {
    let (app, engine, t, _db) = app_with_tree().await;
    let uri = format!("/api/stores/{}/documents", t.leaf);
    for (text, run) in [("old whale text", "r1"), ("new whale text", "r2")] {
        let (s, v) = send(
            &app,
            req(
                "POST",
                &uri,
                Some(json!({"text": text, "source_id": "web/1", "source_run": run})),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED, "{v}");
    }
    // the route needs the bearer token like every other /api route
    let (s, _) = send(
        &app,
        Request::builder()
            .method("DELETE")
            .uri("/api/sources/web%2F1/documents?except_run=r2")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);

    let (s, v) = send(
        &app,
        req(
            "DELETE",
            "/api/sources/web%2F1/documents?except_run=r9",
            None,
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::CONFLICT, "empty_sync_run");
    // the 409 still reports what the prune did, next to the error
    assert_eq!(v["deleted"], 0, "{v}");
    assert_eq!(v["kept"], 2, "{v}");
    assert_eq!(v["stores"], json!([]), "{v}");
    assert_eq!(v["refused_stores"], json!([t.leaf]), "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("allow_empty"),
        "{v}"
    );
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        2
    );

    let (s, v) = send(
        &app,
        req(
            "DELETE",
            &format!(
                "/api/sources/web%2F1/documents?except_run=r2&store_id={}",
                t.leaf2
            ),
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        v,
        json!({"deleted": 0, "kept": 0, "stores": [], "refused_stores": []})
    );

    let (s, v) = send(
        &app,
        req(
            "DELETE",
            "/api/sources/web%2F1/documents?except_run=r2",
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        v,
        json!({"deleted": 1, "kept": 1, "stores": [t.leaf], "refused_stores": []})
    );

    let (s, v) = send(&app, req("DELETE", "/api/sources/nobody/documents", None)).await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    let (s, v) = send(
        &app,
        req("DELETE", "/api/sources/nobody/documents?all=true", None),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        v,
        json!({"deleted": 0, "kept": 0, "stores": [], "refused_stores": []})
    );

    let (s, v) = send(
        &app,
        req(
            "DELETE",
            "/api/sources/web%2F1/documents?store_id=nope",
            None,
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");
    let (s, v) = send(
        &app,
        req(
            "DELETE",
            "/api/sources/web%2F1/documents?allow_empty=maybe",
            None,
        ),
    )
    .await;
    assert_error(s, &v, StatusCode::UNPROCESSABLE_ENTITY, "invalid_input");

    let (s, v) = send(
        &app,
        req(
            "DELETE",
            "/api/sources/web%2F1/documents?except_run=r9&allow_empty=true",
            None,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["deleted"], 1);
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        0
    );
}
