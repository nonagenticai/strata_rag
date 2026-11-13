mod common;

use common::tn;

use common::test_engine;
use sqlx::PgPool;
use strata_rag::db::stores::{self, NewStore, StoreKind, StoreRow};
use strata_rag::engine::Engine;
use strata_rag::error::StrataError;
use strata_rag::ingest::IngestOutcome;
use uuid::Uuid;

struct Tree {
    root: Uuid,
    mid: Uuid,
    leaf: Uuid,
}

/// root(router) ── mid(router) ── leaf
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
    Tree { root, mid, leaf }
}

/// `n` distinct whitespace tokens, a full stop every 10 words.
fn words(prefix: &str, n: usize) -> String {
    (0..n)
        .map(|i| {
            if i % 10 == 9 {
                format!("{prefix}{i}.")
            } else {
                format!("{prefix}{i}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

async fn row(engine: &Engine, id: Uuid) -> StoreRow {
    let mut conn = engine.pool.acquire().await.unwrap();
    stores::get_store(&mut conn, tn(), id).await.unwrap()
}

async fn count(pool: &PgPool, sql: &'static str) -> i64 {
    sqlx::query_scalar(sql).fetch_one(pool).await.unwrap()
}

#[tokio::test]
async fn ingest_creates_chunks_and_propagates_counts() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;

    let out = engine
        .ingest(
            tn(),
            t.leaf,
            "notes.txt".to_string(),
            Some("Notes".to_string()),
            &words("alpha", 100),
        )
        .await
        .unwrap();
    let doc = match out {
        IngestOutcome::Created(d) => d,
        other => panic!("expected Created, got {other:?}"),
    };
    assert_eq!(doc.store_id, t.leaf);
    assert_eq!(doc.source, "notes.txt");
    assert_eq!(doc.title.as_deref(), Some("Notes"));
    assert_eq!(doc.token_count, 100);
    assert!(doc.chunk_count > 1, "100 words at target 32 must split");

    let db_chunks: i64 = sqlx::query_scalar("SELECT count(*) FROM chunks WHERE document_id = $1")
        .bind(doc.id)
        .fetch_one(&engine.pool)
        .await
        .unwrap();
    assert_eq!(db_chunks, doc.chunk_count);
    let max_chunk_tokens: i32 =
        sqlx::query_scalar("SELECT max(token_count) FROM chunks WHERE document_id = $1")
            .bind(doc.id)
            .fetch_one(&engine.pool)
            .await
            .unwrap();
    assert!(max_chunk_tokens <= engine.cfg.chunk_target as i32);

    for id in [t.leaf, t.mid, t.root] {
        let r = row(&engine, id).await;
        assert_eq!(r.token_count, 100, "{}", r.name);
        assert_eq!(r.chunk_count, doc.chunk_count, "{}", r.name);
        let content = r.content_vec.as_deref().expect("content_vec set");
        let norm: f32 = content.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "{} content norm {norm}", r.name);
        assert!(
            r.profile_vec != r.desc_vec,
            "{} profile must include content",
            r.name
        );
    }
    // a single-child chain carries the leaf's content unchanged to the root
    let leaf = row(&engine, t.leaf).await;
    let root = row(&engine, t.root).await;
    for (a, b) in leaf
        .content_vec
        .unwrap()
        .iter()
        .zip(root.content_vec.unwrap().iter())
    {
        assert!((a - b).abs() < 1e-5);
    }
}

#[tokio::test]
async fn identical_text_after_normalisation_is_deduplicated() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    let text = words("beta", 60);

    let first = match engine
        .ingest(tn(), t.leaf, "a.txt".into(), None, &text)
        .await
        .unwrap()
    {
        IngestOutcome::Created(d) => d,
        other => panic!("expected Created, got {other:?}"),
    };
    let chunks_before = count(&engine.pool, "SELECT count(*) FROM chunks").await;

    let messy = format!("   {}   \n\n\n", text.replace(' ', "  \t "));
    let second = engine
        .ingest(tn(), t.leaf, "b.txt".into(), None, &messy)
        .await
        .unwrap();
    match second {
        IngestOutcome::Existing(d) => assert_eq!(d.id, first.id),
        other => panic!("expected Existing, got {other:?}"),
    }
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        1
    );
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM chunks").await,
        chunks_before
    );
    assert_eq!(row(&engine, t.leaf).await.token_count, 60);
}

#[tokio::test]
async fn empty_input_is_rejected_and_writes_nothing() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;

    for input in ["", "   ", "\n\n\t  \n\r\n", "\u{0001}\u{0007}\u{001b}"] {
        let err = engine
            .ingest(tn(), t.leaf, "empty.txt".into(), None, input)
            .await
            .unwrap_err();
        assert!(
            matches!(err, StrataError::InvalidInput(_)),
            "{input:?}: {err:?}"
        );
    }
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        0
    );
    assert_eq!(count(&engine.pool, "SELECT count(*) FROM chunks").await, 0);
    assert_eq!(row(&engine, t.leaf).await.token_count, 0);
}

#[tokio::test]
async fn non_leaf_target_is_rejected() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;

    for target in [t.mid, t.root] {
        let err = engine
            .ingest(tn(), target, "x.txt".into(), None, "some real text")
            .await
            .unwrap_err();
        assert!(matches!(err, StrataError::InvalidTarget(_)), "{err:?}");
    }
    let err = engine
        .ingest(tn(), Uuid::now_v7(), "x.txt".into(), None, "some real text")
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        0
    );
}

#[tokio::test]
async fn leaf_cap_is_enforced() {
    let (mut engine, _db) = test_engine().await;
    engine.cfg.leaf_max_tokens = 50;
    let t = build_tree(&engine).await;

    let err = engine
        .ingest(tn(), t.leaf, "big.txt".into(), None, &words("gamma", 60))
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            StrataError::LeafFull {
                current: 0,
                adding: 60,
                max: 50,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        0
    );

    engine
        .ingest(tn(), t.leaf, "fits.txt".into(), None, &words("delta", 40))
        .await
        .unwrap();
    let err = engine
        .ingest(tn(), t.leaf, "more.txt".into(), None, &words("epsilon", 20))
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            StrataError::LeafFull {
                current: 40,
                adding: 20,
                max: 50,
                ..
            }
        ),
        "{err:?}"
    );
    // exactly filling the cap is allowed
    engine
        .ingest(tn(), t.leaf, "exact.txt".into(), None, &words("zeta", 10))
        .await
        .unwrap();
    assert_eq!(row(&engine, t.leaf).await.token_count, 50);
}

/// The in-transaction cap re-check must read the leaf under `FOR UPDATE`. A second connection
/// holds the leaf row lock; ingest A passes its pre-check (the leaf is empty) and must then block on
/// that lock. Only once Postgres reports A blocked does the holder fill the leaf to 100/150 and
/// commit, so A must see 100 and refuse. Without `FOR UPDATE`, A reads 0 before it blocks (later, on
/// the document FK check) and overfills the leaf. No timing involved: the order is forced.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_rechecks_the_cap_under_the_leaf_lock() {
    let (mut engine, _db) = test_engine().await;
    engine.cfg.leaf_max_tokens = 150;
    let t = build_tree(&engine).await;
    let leaf = t.leaf;
    let a = words("eta", 100);

    let mut holder = engine.pool.begin().await.unwrap();
    let holder_pid = common::backend_pid(&mut holder).await;
    sqlx::query("SELECT 1 FROM stores WHERE id = $1 FOR UPDATE")
        .bind(leaf)
        .execute(&mut *holder)
        .await
        .unwrap();
    let pool = engine.pool.clone();
    let (ra, ()) = tokio::join!(
        engine.ingest(tn(), leaf, "a.txt".into(), None, &a),
        async move {
            common::wait_until_blocked_by(&pool, holder_pid).await;
            // a concurrent ingest that committed while A was embedding
            sqlx::query("UPDATE stores SET token_count = 100 WHERE id = $1")
                .bind(leaf)
                .execute(&mut *holder)
                .await
                .unwrap();
            holder.commit().await.unwrap();
        }
    );

    assert!(
        matches!(
            ra,
            Err(StrataError::LeafFull {
                current: 100,
                adding: 100,
                max: 150,
                ..
            })
        ),
        "{ra:?}"
    );
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        0
    );
}

/// delete_document must take the leaf lock BEFORE it touches the document, the same order ingest
/// uses (leaf, then document). Otherwise an ingest of the same content holding the leaf and a delete
/// holding the document deadlock, and ingest's locked dedupe can report `Existing` for a document
/// that is being deleted. A holder takes the leaf lock; once the delete is blocked behind it, the
/// document row must still be free (`FOR UPDATE NOWAIT` succeeds).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_document_locks_the_leaf_before_the_document() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    let doc = match engine
        .ingest(tn(), t.leaf, "d.txt".into(), None, &words("kappa", 30))
        .await
        .unwrap()
    {
        IngestOutcome::Created(d) => d,
        other => panic!("expected Created, got {other:?}"),
    };
    let doc_id = doc.id;

    let mut holder = engine.pool.begin().await.unwrap();
    let holder_pid = common::backend_pid(&mut holder).await;
    sqlx::query("SELECT 1 FROM stores WHERE id = $1 FOR UPDATE")
        .bind(t.leaf)
        .execute(&mut *holder)
        .await
        .unwrap();
    let pool = engine.pool.clone();
    let (rd, probe) = tokio::join!(engine.delete_document(tn(), doc_id), async move {
        common::wait_until_blocked_by(&pool, holder_pid).await;
        let probe = sqlx::query("SELECT 1 FROM documents WHERE id = $1 FOR UPDATE NOWAIT")
            .bind(doc_id)
            .execute(&mut *holder)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string());
        // an aborted transaction commits as a rollback: either way the leaf lock is released
        holder.commit().await.unwrap();
        probe
    });

    assert!(
        probe.is_ok(),
        "delete_document held the document row while waiting for the leaf: {probe:?}"
    );
    rd.unwrap();
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        0
    );
    let leaf = row(&engine, t.leaf).await;
    assert_eq!((leaf.token_count, leaf.chunk_count), (0, 0));
}

#[tokio::test]
async fn delete_document_recomputes_to_empty() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    let doc = match engine
        .ingest(tn(), t.leaf, "d.txt".into(), None, &words("iota", 70))
        .await
        .unwrap()
    {
        IngestOutcome::Created(d) => d,
        other => panic!("expected Created, got {other:?}"),
    };

    engine.delete_document(tn(), doc.id).await.unwrap();

    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        0
    );
    assert_eq!(count(&engine.pool, "SELECT count(*) FROM chunks").await, 0);
    for id in [t.leaf, t.mid, t.root] {
        let r = row(&engine, id).await;
        assert_eq!((r.token_count, r.chunk_count), (0, 0), "{}", r.name);
        assert_eq!(r.content_vec, None, "{}", r.name);
        assert_eq!(r.profile_vec, r.desc_vec, "{}", r.name);
    }

    let err = engine.delete_document(tn(), doc.id).await.unwrap_err();
    assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");
}

/// The true centroid of `leaf` from its chunk rows: (normalised mean, raw f64 sum, chunk count).
async fn true_centroid(pool: &PgPool, leaf: Uuid) -> (Vec<f32>, Vec<f64>, i64) {
    let vecs: Vec<pgvector::Vector> =
        sqlx::query_scalar("SELECT embedding FROM chunks WHERE store_id = $1")
            .bind(leaf)
            .fetch_all(pool)
            .await
            .unwrap();
    let mut sum = vec![0.0_f64; strata_rag::vecmath::DIM];
    for v in &vecs {
        for (s, x) in sum.iter_mut().zip(v.to_vec()) {
            *s += f64::from(x);
        }
    }
    let n = vecs.len() as f64;
    let mut mean: Vec<f32> = sum.iter().map(|s| (s / n) as f32).collect();
    strata_rag::vecmath::l2_normalize(&mut mean);
    (mean, sum, vecs.len() as i64)
}

/// The leaf's running sums (migration 0020): (content_sum, content_chunks, content_tokens).
async fn running_sums(pool: &PgPool, leaf: Uuid) -> (Option<Vec<f64>>, i64, i64) {
    sqlx::query_as("SELECT content_sum, content_chunks, content_tokens FROM stores WHERE id = $1")
        .bind(leaf)
        .fetch_one(pool)
        .await
        .unwrap()
}

/// The leaf's running sums, counts and centroid equal the true ones computed from its rows.
async fn check_centroid(engine: &Engine, leaf: Uuid, label: &str) {
    let pool = &engine.pool;
    let (mean, sum, n) = true_centroid(pool, leaf).await;
    let tokens: i64 = sqlx::query_scalar(
        "SELECT coalesce(sum(token_count), 0)::bigint FROM documents WHERE store_id = $1",
    )
    .bind(leaf)
    .fetch_one(pool)
    .await
    .unwrap();
    let (stored_sum, stored_n, stored_tokens) = running_sums(pool, leaf).await;
    assert_eq!((stored_n, stored_tokens), (n, tokens), "{label}");
    let stored_sum = stored_sum.expect("content_sum set");
    assert_eq!(stored_sum.len(), sum.len(), "{label}");
    for (i, (a, b)) in stored_sum.iter().zip(&sum).enumerate() {
        assert!((a - b).abs() < 1e-4, "{label}: sum[{i}] {a} != {b}");
    }
    let r = row(engine, leaf).await;
    assert_eq!((r.token_count, r.chunk_count), (tokens, n), "{label}");
    let content = r.content_vec.expect("content_vec set");
    for (i, (a, b)) in content.iter().zip(&mean).enumerate() {
        assert!((a - b).abs() < 1e-5, "{label}: mean[{i}] {a} != {b}");
    }
}

/// Ingest no longer rescans the leaf: its centroid comes from a running sum the chunk / document
/// triggers keep. After N ingests and a delete it must still equal the true
/// mean of the chunks left, and the counts their true totals; an emptied leaf drops the sum.
#[tokio::test]
async fn running_centroid_equals_the_true_mean_after_ingests_and_deletes() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    let mut docs = Vec::new();
    for (i, prefix) in ["red", "green", "blue", "cyan", "plum", "teal"]
        .iter()
        .enumerate()
    {
        let out = engine
            .ingest(
                tn(),
                t.leaf,
                format!("{prefix}.txt"),
                None,
                &words(prefix, 40 + 15 * i),
            )
            .await
            .unwrap();
        match out {
            IngestOutcome::Created(d) => docs.push(d),
            other => panic!("expected Created, got {other:?}"),
        }
    }
    engine.delete_document(tn(), docs[2].id).await.unwrap();

    check_centroid(&engine, t.leaf, "after 6 ingests and a delete").await;

    for d in docs.iter().filter(|d| d.id != docs[2].id).skip(1) {
        engine.delete_document(tn(), d.id).await.unwrap();
    }
    check_centroid(&engine, t.leaf, "one document left").await;

    engine.delete_document(tn(), docs[0].id).await.unwrap();
    let (sum, n, tokens) = running_sums(&engine.pool, t.leaf).await;
    assert_eq!(
        (sum, n, tokens),
        (None, 0, 0),
        "an emptied leaf drops its sum"
    );
    let r = row(&engine, t.leaf).await;
    assert_eq!(r.content_vec, None);
}
