//! HNSW dense search (migration 0009, spec §8.3): the chunk embeddings carry an HNSW index and the
//! dense search runs with `hnsw.iterative_scan = relaxed_order`, so the tenant / store / document
//! filters still fill the candidate list. Without the iterative scan an HNSW scan returns at most
//! `hnsw.ef_search` rows BEFORE the filter: a leaf holding 3% of the corpus would get ~3 of its 50
//! candidates.

mod common;

use common::{TestDb, other, test_engine, tn};

use pgvector::Vector;
use serde_json::json;
use sqlx::PgPool;
use strata_rag::db::search::{
    ChunkFilter, DEFAULT_EF_SEARCH, DEFAULT_EXACT_SCAN_MAX_CHUNKS, DENSE_SQL, DensePath,
    choose_dense_path, dense_search, dense_search_ef, dense_search_path, leaf_set_chunks,
    set_exact_scan, set_hnsw_scan,
};
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::search::CANDIDATES;
use strata_rag::vecmath::{DIM, l2_normalize};
use uuid::Uuid;

/// Dimensions the seeded vectors vary in (see `Rng::unit_vector`).
const INTRINSIC: usize = 16;

/// xorshift64*: deterministic, no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// A random unit vector in the first `INTRINSIC` dimensions (zero elsewhere). Uniform random
    /// 1024-dim vectors are all nearly equidistant, so "the exact top 50" would be a coin toss for
    /// any ANN index; real embeddings have low intrinsic dimension, and so does this corpus.
    fn unit_vector(&mut self) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM];
        for x in v.iter_mut().take(INTRINSIC) {
            *x = self.signed();
        }
        l2_normalize(&mut v);
        v
    }
}

fn store(parent_id: Option<Uuid>, kind: StoreKind, name: &str) -> NewStore {
    NewStore {
        parent_id,
        kind,
        name: name.to_string(),
        description: format!("{name} store"),
    }
}

/// One document in `leaf` holding `n` chunks with random unit embeddings; returns (chunk id, vector).
async fn seed_leaf(pool: &PgPool, leaf: Uuid, n: usize, rng: &mut Rng) -> Vec<(Uuid, Vec<f32>)> {
    let mut tx = pool.begin().await.unwrap();
    let doc = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO documents (id, store_id, source, content_hash, token_count) \
         VALUES ($1, $2, 'seed', $3, $4)",
    )
    .bind(doc)
    .bind(leaf)
    .bind(doc.as_bytes().to_vec())
    .bind(n as i32)
    .execute(&mut *tx)
    .await
    .unwrap();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let id = Uuid::now_v7();
        let v = rng.unit_vector();
        sqlx::query(
            "INSERT INTO chunks (id, document_id, store_id, ord, text, token_count, embedding) \
             VALUES ($1, $2, $3, $4, $5, 1, $6)",
        )
        .bind(id)
        .bind(doc)
        .bind(leaf)
        .bind(i as i32)
        .bind(format!("seed chunk {i}"))
        .bind(Vector::from(v.clone()))
        .execute(&mut *tx)
        .await
        .unwrap();
        out.push((id, v));
    }
    tx.commit().await.unwrap();
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Exact top `k` of `rows` by cosine (all unit vectors), best first.
fn exact_top(rows: &[(Uuid, Vec<f32>)], q: &[f32], k: usize) -> Vec<Uuid> {
    let mut scored: Vec<(Uuid, f32)> = rows.iter().map(|(id, v)| (*id, dot(v, q))).collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// A query near `v`: `v` plus noise, normalised.
fn near(v: &[f32], rng: &mut Rng) -> Vec<f32> {
    let mut q: Vec<f32> = v
        .iter()
        .take(INTRINSIC)
        .map(|x| x + 0.05 * rng.signed())
        .chain(std::iter::repeat_n(0.0, DIM - INTRINSIC))
        .collect();
    l2_normalize(&mut q);
    q
}

struct Corpus {
    small: Uuid,
    big: Uuid,
    small_rows: Vec<(Uuid, Vec<f32>)>,
    big_rows: Vec<(Uuid, Vec<f32>)>,
}

/// Tenant A: root → `small` (exactly CANDIDATES chunks) + `big` (1000). Tenant B: root → `b`
/// (500). `small` is ~3% of the 1550 indexed chunks.
async fn corpus(engine: &Engine, rng: &mut Rng) -> Corpus {
    let root = engine
        .create_store(tn(), store(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    let small = engine
        .create_store(tn(), store(Some(root), StoreKind::Leaf, "small"))
        .await
        .unwrap()
        .id;
    let big = engine
        .create_store(tn(), store(Some(root), StoreKind::Leaf, "big"))
        .await
        .unwrap()
        .id;
    let other_root = engine
        .create_store(other(), store(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    let b = engine
        .create_store(other(), store(Some(other_root), StoreKind::Leaf, "b"))
        .await
        .unwrap()
        .id;
    let small_rows = seed_leaf(&engine.pool, small, CANDIDATES, rng).await;
    let big_rows = seed_leaf(&engine.pool, big, 1000, rng).await;
    seed_leaf(&engine.pool, b, 500, rng).await;
    Corpus {
        small,
        big,
        small_rows,
        big_rows,
    }
}

/// A second pool on the same database in which the planner can only order by the HNSW index:
/// sequential and bitmap scans are off and a sort is priced out, so the btree on `store_id` plus a
/// sort loses. The tiny test tables would otherwise always be scanned exactly, proving nothing.
async fn hnsw_only_pool(db: &TestDb) -> PgPool {
    let name: String = sqlx::query_scalar("SELECT current_database()")
        .fetch_one(&db.pool)
        .await
        .unwrap();
    for setting in ["enable_seqscan", "enable_bitmapscan", "enable_sort"] {
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "ALTER DATABASE {name} SET {setting} = off"
        )))
        .execute(&db.pool)
        .await
        .unwrap();
    }
    PgPool::connect(&db.url).await.unwrap()
}

#[tokio::test]
async fn migration_creates_the_hnsw_index() {
    let (engine, _db) = test_engine().await;
    let def: String = sqlx::query_scalar(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks' AND indexname = 'chunks_embedding_hnsw'",
    )
    .fetch_one(&engine.pool)
    .await
    .unwrap();
    assert!(def.contains("USING hnsw"), "{def}");
    assert!(def.contains("vector_cosine_ops"), "{def}");
}

#[tokio::test]
async fn the_dense_search_uses_the_index_when_the_planner_can() {
    let (engine, db) = test_engine().await;
    let mut rng = Rng(0x5eed_0001);
    let c = corpus(&engine, &mut rng).await;
    let pool = hnsw_only_pool(&db).await;
    let mut tx = pool.begin().await.unwrap();
    set_hnsw_scan(&mut tx, DEFAULT_EF_SEARCH).await.unwrap();
    let plan: Vec<String> = sqlx::query_scalar(sqlx::AssertSqlSafe(format!("EXPLAIN {DENSE_SQL}")))
        .bind(Vector::from(rng.unit_vector()))
        .bind(vec![c.small])
        .bind(CANDIDATES as i64)
        .bind(vec![tn().id()])
        .fetch_all(&mut *tx)
        .await
        .unwrap();
    let plan = plan.join("\n");
    assert!(plan.contains("chunks_embedding_hnsw"), "{plan}");
}

/// The leaf holding 3% of the corpus still gets all CANDIDATES of its chunks through the index:
/// the iterative scan keeps walking the graph until the store filter has passed enough rows.
#[tokio::test]
async fn a_selective_store_filter_still_fills_the_candidate_list() {
    let (engine, db) = test_engine().await;
    let mut rng = Rng(0x5eed_0002);
    let c = corpus(&engine, &mut rng).await;
    let pool = hnsw_only_pool(&db).await;
    let all_small: std::collections::HashSet<Uuid> =
        c.small_rows.iter().map(|(id, _)| *id).collect();
    for i in 0..5 {
        let q = near(&c.big_rows[i * 7].1, &mut rng);
        for filter in [
            ChunkFilter::default(),
            // the documents-join variant of the statement
            ChunkFilter {
                source_ids: None,
                metadata: Some(json!({})),
            },
        ] {
            let hits = dense_search(&pool, tn(), &[c.small], &q, CANDIDATES as i64, &filter)
                .await
                .unwrap();
            assert_eq!(hits.len(), CANDIDATES, "query {i} {filter:?}");
            let got: std::collections::HashSet<Uuid> = hits.iter().map(|(id, _)| *id).collect();
            assert_eq!(got, all_small, "query {i} {filter:?}");
            assert!(
                hits.windows(2).all(|w| w[0].1 >= w[1].1),
                "best first despite relaxed_order"
            );
        }
    }
}

/// Recall of the indexed search against an exact scan over the same filtered set, for a leaf
/// larger than the candidate list, and the other tenant's rows never appear.
#[tokio::test]
async fn filtered_results_match_an_exact_scan() {
    let (engine, db) = test_engine().await;
    let mut rng = Rng(0x5eed_0003);
    let c = corpus(&engine, &mut rng).await;
    let pool = hnsw_only_pool(&db).await;
    let tenant_rows: Vec<(Uuid, Vec<f32>)> = c
        .small_rows
        .iter()
        .chain(&c.big_rows)
        .map(|(id, v)| (*id, v.clone()))
        .collect();
    let mut total = 0usize;
    let mut found = 0usize;
    for i in 0..5 {
        let q = near(&c.big_rows[i * 11 + 3].1, &mut rng);
        let want = exact_top(&tenant_rows, &q, CANDIDATES);
        let hits = dense_search_ef(
            &pool,
            tn(),
            &[c.small, c.big],
            &q,
            CANDIDATES as i64,
            &ChunkFilter::default(),
            DEFAULT_EF_SEARCH,
        )
        .await
        .unwrap();
        assert_eq!(hits.len(), CANDIDATES, "query {i}");
        assert_eq!(
            hits[0].0, want[0],
            "query {i}: the nearest chunk comes first"
        );
        let got: std::collections::HashSet<Uuid> = hits.iter().map(|(id, _)| *id).collect();
        assert!(
            got.iter()
                .all(|id| tenant_rows.iter().any(|(t, _)| t == id)),
            "only the tenant's own leaves"
        );
        total += want.len();
        found += want.iter().filter(|id| got.contains(id)).count();
    }
    let recall = found as f64 / total as f64;
    assert!(recall >= 0.9, "recall@{CANDIDATES} {recall:.3}");
}

/// The exact path (the planner's choice on small tables) and the HNSW path agree on the top hit.
#[tokio::test]
async fn the_default_pool_still_answers_exactly() {
    let (engine, _db) = test_engine().await;
    let mut rng = Rng(0x5eed_0004);
    let c = corpus(&engine, &mut rng).await;
    let q = near(&c.big_rows[42].1, &mut rng);
    let want = exact_top(&c.big_rows, &q, CANDIDATES);
    let hits = dense_search(
        &engine.pool,
        tn(),
        &[c.big],
        &q,
        CANDIDATES as i64,
        &ChunkFilter::default(),
    )
    .await
    .unwrap();
    assert_eq!(hits.len(), CANDIDATES);
    assert_eq!(hits[0].0, want[0]);
    assert_eq!(hits[0].0, c.big_rows[42].0);
}

/// A small leaf set takes the exact path: its chunk count (migration 0020's `content_chunks`) is
/// below the default threshold, and with index scans off the statement cannot use the HNSW index
/// (an index scan is a disabled node, a bitmap or sequential scan + sort is not, so no cost
/// estimate can bring the index back). It returns the true top k. Not on `hnsw_only_pool`: there
/// every exact plan carries two disabled nodes (scan + sort) against the index scan's one.
#[tokio::test]
async fn a_small_leaf_set_takes_the_exact_path_and_returns_the_true_top_k() {
    let (engine, _db) = test_engine().await;
    let mut rng = Rng(0x5eed_0005);
    let c = corpus(&engine, &mut rng).await;
    let pool = engine.pool.clone();
    let leaves = [c.small, c.big];

    let n = leaf_set_chunks(&pool, tn(), &leaves).await.unwrap();
    assert_eq!(n, (CANDIDATES + 1000) as i64);
    // tenant-scoped: another tenant's leaves count nothing
    assert_eq!(leaf_set_chunks(&pool, other(), &leaves).await.unwrap(), 0);
    assert_eq!(leaf_set_chunks(&pool, tn(), &[]).await.unwrap(), 0);
    assert_eq!(
        choose_dense_path(n, DEFAULT_EXACT_SCAN_MAX_CHUNKS),
        DensePath::Exact
    );
    assert_eq!(
        choose_dense_path(n, n as u64),
        DensePath::Exact,
        "inclusive"
    );
    assert_eq!(choose_dense_path(n, n as u64 - 1), DensePath::Hnsw);
    assert_eq!(choose_dense_path(n, 0), DensePath::Hnsw, "0 disables");
    assert_eq!(choose_dense_path(0, 0), DensePath::Hnsw, "0 disables");

    // the HNSW-only pool would take the index; the exact path's settings keep it off
    let mut tx = pool.begin().await.unwrap();
    set_exact_scan(&mut tx).await.unwrap();
    let plan: Vec<String> = sqlx::query_scalar(sqlx::AssertSqlSafe(format!("EXPLAIN {DENSE_SQL}")))
        .bind(Vector::from(rng.unit_vector()))
        .bind(leaves.to_vec())
        .bind(CANDIDATES as i64)
        .bind(vec![tn().id()])
        .fetch_all(&mut *tx)
        .await
        .unwrap();
    tx.rollback().await.unwrap();
    let plan = plan.join("\n");
    assert!(!plan.contains("chunks_embedding_hnsw"), "{plan}");

    let tenant_rows: Vec<(Uuid, Vec<f32>)> = c
        .small_rows
        .iter()
        .chain(&c.big_rows)
        .map(|(id, v)| (*id, v.clone()))
        .collect();
    for i in 0..5 {
        let q = near(&c.big_rows[i * 13 + 5].1, &mut rng);
        let want = exact_top(&tenant_rows, &q, CANDIDATES);
        for filter in [
            ChunkFilter::default(),
            ChunkFilter {
                source_ids: None,
                metadata: Some(json!({})),
            },
        ] {
            let hits = dense_search_path(
                &pool,
                tn(),
                &leaves,
                &q,
                CANDIDATES as i64,
                &filter,
                DEFAULT_EF_SEARCH,
                DensePath::Exact,
            )
            .await
            .unwrap();
            let got: Vec<Uuid> = hits.iter().map(|(id, _)| *id).collect();
            assert_eq!(got.len(), CANDIDATES, "query {i} {filter:?}");
            assert_eq!(got[0], want[0], "query {i} {filter:?}");
            let got: std::collections::HashSet<Uuid> = got.into_iter().collect();
            let want: std::collections::HashSet<Uuid> = want.iter().copied().collect();
            assert_eq!(got, want, "query {i} {filter:?}: exactly the true top k");
            assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1), "best first");
        }
        // the other tenant gets nothing from these leaves on the exact path either
        let none = dense_search_path(
            &pool,
            other(),
            &leaves,
            &q,
            CANDIDATES as i64,
            &ChunkFilter::default(),
            DEFAULT_EF_SEARCH,
            DensePath::Exact,
        )
        .await
        .unwrap();
        assert!(none.is_empty());
    }
}
