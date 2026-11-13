//! Leaf-scoped chunk search queries. The dense search runs one of two paths (`DensePath`): the
//! HNSW index on `chunks.embedding` (migration 0009, spec §8.3), or, for a leaf set of at most
//! `STRATA_EXACT_SCAN_MAX_CHUNKS` chunks, an exact scan with index scans off (bitmap on the
//! `store_id` btree, then a sort), see `choose_dense_path`.

use pgvector::Vector;
use serde_json::{Map, Value};
use sqlx::types::Json;
use uuid::Uuid;

use crate::error::Result;
use crate::precedence::Layer;
use crate::tenant::Tenant;

// Every search is owner-scoped in SQL ($4, `Owners::ids`): only the leaves of `$2` that belong to
// the caller (or, for a query granted platform packs, to the platform owner) are searched, so a
// caller passing another tenant's leaf ids gets nothing. `$2` is already the authorised leaf set
// (`crate::query::Readable`); the owner filter never widens it. It is applied to the (short) leaf
// list, not joined per chunk, so the chunk scan's plan is unchanged.
//
// The dense statements are written for the HNSW index: the inner, MATERIALIZED query is the
// index-ordered scan (`ORDER BY embedding <=> $1 LIMIT $3`), the filters are applied to the rows
// it yields, and the session runs it with `hnsw.iterative_scan = relaxed_order` (`set_hnsw_scan`)
// so the scan keeps walking the graph until enough rows have passed the filters. `relaxed_order`
// may return rows slightly out of distance order, so the outer query re-sorts them (the pgvector
// documented pattern); the materialisation keeps the planner from merging the two sorts.
pub const DENSE_SQL: &str = "WITH r AS MATERIALIZED (SELECT id, embedding <=> $1 AS dist FROM chunks WHERE store_id = ANY(ARRAY(SELECT s.id FROM stores s WHERE s.id = ANY($2) AND s.tenant = ANY($4))) ORDER BY embedding <=> $1 LIMIT $3) SELECT id, 1 - dist AS sim FROM r ORDER BY dist, id";
// OR-of-terms query (NOT websearch_to_tsquery, which ANDs every word: the PoC measured a
// median of 0 lexical candidates on natural-language questions with it). Each lexeme is
// quote_literal'd and the whole cast to tsquery, never re-parsed by to_tsquery: 'simple' keeps
// URLs/hosts whole, and their `:` `&` `(` `!` would be read as operators (syntax error, or a
// silent AND). Stopwords are dropped from the QUERY only (a lexeme the 'english' config reduces to
// nothing): 'simple' keeps them, and an OR over "the"/"of"/"what" matched 10,076 of 10,103 chunks,
// so ts_rank_cd scored the whole corpus (113 ms → 12 ms on the seed library). The index stays
// 'simple', so non-English text still matches; only English function words leave the query.
// Input without (non-stopword) lexemes -> NULL -> no rows.
const LEXICAL_SQL: &str = "SELECT id, ts_rank_cd(tsv, q) AS r FROM chunks, (SELECT NULLIF(array_to_string(ARRAY(SELECT quote_literal(l) FROM unnest(tsvector_to_array(to_tsvector('simple', $1))) l WHERE to_tsvector('english', l) <> ''::tsvector), ' | '), '')::tsquery) AS t(q) WHERE store_id = ANY(ARRAY(SELECT s.id FROM stores s WHERE s.id = ANY($2) AND s.tenant = ANY($4))) AND tsv @@ q ORDER BY r DESC LIMIT $3";
// The same two searches restricted by the request's document filters (`ChunkFilter`), applied in
// SQL BEFORE the LIMIT, so a filtered search still fills its candidate list from the matching
// chunks instead of returning the unfiltered top rows minus the ones that do not match. $5: the
// accepted `documents.source_id`s (NULL = any), $6: a JSON object `documents.metadata` must contain
// (NULL = any). Separate statements, so an unfiltered query keeps its join-free plan.
pub const DENSE_FILTERED_SQL: &str = "WITH r AS MATERIALIZED (SELECT c.id, c.embedding <=> $1 AS dist FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.store_id = ANY(ARRAY(SELECT s.id FROM stores s WHERE s.id = ANY($2) AND s.tenant = ANY($4))) AND ($5::text[] IS NULL OR d.source_id = ANY($5)) AND ($6::jsonb IS NULL OR d.metadata @> $6) ORDER BY c.embedding <=> $1 LIMIT $3) SELECT id, 1 - dist AS sim FROM r ORDER BY dist, id";
const LEXICAL_FILTERED_SQL: &str = "SELECT c.id, ts_rank_cd(c.tsv, q) AS r FROM chunks c JOIN documents d ON d.id = c.document_id, (SELECT NULLIF(array_to_string(ARRAY(SELECT quote_literal(l) FROM unnest(tsvector_to_array(to_tsvector('simple', $1))) l WHERE to_tsvector('english', l) <> ''::tsvector), ' | '), '')::tsquery) AS t(q) WHERE c.store_id = ANY(ARRAY(SELECT s.id FROM stores s WHERE s.id = ANY($2) AND s.tenant = ANY($4))) AND c.tsv @@ q AND ($5::text[] IS NULL OR d.source_id = ANY($5)) AND ($6::jsonb IS NULL OR d.metadata @> $6) ORDER BY r DESC LIMIT $3";
// The chunk's document-level provenance rides along in the same query (no per-hit lookup), with
// its store's owner and the document's `topic_keys` for precedence (`crate::precedence`).
// Owner-scoped like the searches: another tenant's chunk ids return no rows.
const FETCH_SQL: &str = "SELECT c.id AS chunk_id, c.document_id, c.store_id, s.tenant AS owner, c.ord, c.text, c.page_start, c.page_end, d.source, d.title, d.source_id, d.source_run, d.metadata, d.topic_keys FROM chunks c JOIN documents d ON d.id = c.document_id JOIN stores s ON s.id = c.store_id WHERE c.id = ANY($1) AND s.tenant = ANY($2)";

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct ChunkText {
    pub chunk_id: Uuid,
    pub document_id: Uuid,
    pub store_id: Uuid,
    /// `stores.tenant` of the chunk's store: the caller, or the platform owner for a platform hit.
    pub owner: Uuid,
    /// The chunk's position in its document.
    pub ord: i32,
    pub text: String,
    /// First / last page the chunk spans (migration 0007); NULL without page anchors.
    pub page_start: Option<i32>,
    pub page_end: Option<i32>,
    /// The document's `source` label and `title`.
    pub source: String,
    pub title: Option<String>,
    /// The document's `source_id` / `source_run` / `metadata` (migration 0004).
    pub source_id: Option<String>,
    pub source_run: Option<String>,
    #[sqlx(json)]
    pub metadata: Map<String, Value>,
    /// The document's `topic_keys` (migration 0022; set on platform documents for shadowing).
    pub topic_keys: Vec<String>,
}

/// Whose stores one search may read (spec platform-knowledge §5): the caller's, plus the platform
/// owner's when the query was granted platform packs. Only a filter over the already authorised
/// leaf set (`crate::query::Readable`), never a grant of its own. A `Tenant` converts to the
/// caller alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Owners {
    pub caller: Tenant,
    /// The platform owner, when platform stores are among the searched leaves (never the caller).
    pub platform: Option<Tenant>,
}

impl Owners {
    /// The `stores.tenant` values a search accepts (`$4` / `$2` above).
    pub fn ids(&self) -> Vec<Uuid> {
        let mut ids = vec![self.caller.id()];
        if let Some(p) = self.platform.filter(|p| *p != self.caller) {
            ids.push(p.id());
        }
        ids
    }

    /// The layer of a chunk whose store `owner` (`ChunkText::owner`) is given.
    pub fn layer_of(&self, owner: Uuid) -> Layer {
        match self.platform {
            Some(p) if p != self.caller && p.id() == owner => Layer::Platform,
            _ => Layer::Tenant,
        }
    }
}

impl From<Tenant> for Owners {
    fn from(caller: Tenant) -> Self {
        Owners {
            caller,
            platform: None,
        }
    }
}

/// Restricts a search to chunks whose DOCUMENT matches (`QueryRequest::source_ids` / `metadata`).
/// Validated by the caller (`crate::query`): `metadata` is a JSON object.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChunkFilter {
    /// The document's `source_id` is one of these.
    pub source_ids: Option<Vec<String>>,
    /// The document's `metadata` contains this object (`@>`).
    pub metadata: Option<Value>,
}

impl ChunkFilter {
    pub fn is_empty(&self) -> bool {
        self.source_ids.is_none() && self.metadata.is_none()
    }
}

/// `hnsw.ef_search` when none is configured (`STRATA_HNSW_EF_SEARCH`): the size of the HNSW
/// candidate list per scan step. Above the `CANDIDATES` a search asks for, so one step usually
/// fills an unfiltered search.
pub const DEFAULT_EF_SEARCH: u32 = 100;
/// pgvector's own bounds for `hnsw.ef_search`.
pub const MIN_EF_SEARCH: u32 = 1;
pub const MAX_EF_SEARCH: u32 = 1000;

/// `STRATA_EXACT_SCAN_MAX_CHUNKS` when unset: leaf sets of at most this many chunks are searched
/// exactly. The scale harness (`scripts/scale_harness/RESULTS.md`) measured, at ~1M chunks dim 256,
/// the HNSW path on a 61-chunk tenant leaf set at recall@10 0.66–0.71 with 75–82% short results (the
/// planner estimates the `ARRAY(subquery)` leaf filter at ~5,040 rows whatever its real size, so it
/// always takes the index, and the filtered iterative scan runs out of memory budget first). At the
/// production dim 1024 (~200k chunks, work_mem 16MB) the exact scan costs ~2 ms per 1k chunks
/// (p95 7 / 10–15 / 15–35 / 31–35 ms at 1k / 2.5k / 5k / 10k); HNSW was broken at 1k (0.5% of the
/// table: p95 ~120 ms, 21/50 short), and from 2.5k on p95 7–20 ms at recall@10 0.99–1.0. The p95
/// crossover is ~2.5k chunks; 5k keeps the exact path down to 0.5% selectivity on a ~1M-chunk table.
pub const DEFAULT_EXACT_SCAN_MAX_CHUNKS: u64 = 5_000;

/// How one dense search was answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DensePath {
    /// Every chunk of the leaf set is scored (index scans off: bitmap on `chunks_store_idx` + sort).
    Exact,
    /// The HNSW index with `hnsw.iterative_scan = relaxed_order` (`set_hnsw_scan`).
    Hnsw,
}

impl DensePath {
    pub fn as_str(self) -> &'static str {
        match self {
            DensePath::Exact => "exact",
            DensePath::Hnsw => "hnsw",
        }
    }
}

/// Exact when the leaf set holds at most `exact_max_chunks` chunks; `exact_max_chunks = 0`
/// disables the exact path (always HNSW).
pub fn choose_dense_path(chunks: i64, exact_max_chunks: u64) -> DensePath {
    if exact_max_chunks > 0 && u64::try_from(chunks).is_ok_and(|n| n <= exact_max_chunks) {
        DensePath::Exact
    } else {
        DensePath::Hnsw
    }
}

/// The chunk count of `owners`' `leaves` (another tenant's leaves count 0): the sum of
/// `stores.content_chunks`, which migration 0020's triggers keep exact. One index lookup per leaf.
pub async fn leaf_set_chunks(
    pool: &sqlx::PgPool,
    owners: impl Into<Owners>,
    leaves: &[Uuid],
) -> Result<i64> {
    if leaves.is_empty() {
        return Ok(0);
    }
    let n: i64 = sqlx::query_scalar(
        "SELECT coalesce(sum(s.content_chunks), 0)::bigint FROM stores s \
         WHERE s.id = ANY($1) AND s.tenant = ANY($2)",
    )
    .bind(leaves)
    .bind(owners.into().ids())
    .fetch_one(pool)
    .await?;
    Ok(n)
}

/// For the rest of the current transaction: the planner may not use index scans, so the dense
/// statement cannot take the HNSW index and scores every chunk of the leaf set (a bitmap scan of
/// `chunks_store_idx`, then a sort). This is the exact statement the scale harness validated
/// (`SET LOCAL enable_indexscan = off` around the unchanged `DENSE_SQL`): no rewrite of the
/// statement keeps the planner off the index as reliably, since `ORDER BY embedding <=> $1` is
/// exactly what the HNSW index answers. `SET LOCAL`, so it never leaks to the pooled connection.
pub async fn set_exact_scan(conn: &mut sqlx::PgConnection) -> Result<()> {
    sqlx::query("SELECT set_config('enable_indexscan', 'off', true)")
        .execute(&mut *conn)
        .await?;
    Ok(())
}

/// For the rest of the current transaction: HNSW scans continue past `hnsw.ef_search` while the
/// filters reject rows (`iterative_scan = relaxed_order`), with `ef_search` candidates per step.
/// `set_config(..., true)` is `SET LOCAL`, so the setting never leaks to the pooled connection's
/// next user.
pub async fn set_hnsw_scan(conn: &mut sqlx::PgConnection, ef_search: u32) -> Result<()> {
    sqlx::query(
        "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true), \
                set_config('hnsw.ef_search', $1, true)",
    )
    .bind(ef_search.clamp(MIN_EF_SEARCH, MAX_EF_SEARCH).to_string())
    .execute(&mut *conn)
    .await?;
    Ok(())
}

/// (chunk_id, cosine similarity = 1 − distance), best first, among `filter`'s chunks of
/// `owners`' `leaves` (another tenant's leaves are skipped). `DEFAULT_EF_SEARCH`.
pub async fn dense_search(
    pool: &sqlx::PgPool,
    owners: impl Into<Owners>,
    leaves: &[Uuid],
    q: &[f32],
    limit: i64,
    filter: &ChunkFilter,
) -> Result<Vec<(Uuid, f32)>> {
    dense_search_ef(pool, owners, leaves, q, limit, filter, DEFAULT_EF_SEARCH).await
}

/// `dense_search` with an explicit `hnsw.ef_search`, always on the HNSW path.
pub async fn dense_search_ef(
    pool: &sqlx::PgPool,
    owners: impl Into<Owners>,
    leaves: &[Uuid],
    q: &[f32],
    limit: i64,
    filter: &ChunkFilter,
    ef_search: u32,
) -> Result<Vec<(Uuid, f32)>> {
    dense_search_path(
        pool,
        owners,
        leaves,
        q,
        limit,
        filter,
        ef_search,
        DensePath::Hnsw,
    )
    .await
}

/// `dense_search_ef` on the given path: the same statement and owner scoping either way; `path`
/// only sets the transaction's planner settings (`set_hnsw_scan` / `set_exact_scan`). Runs in its
/// own short transaction, the scope of the `SET LOCAL`s.
#[allow(clippy::too_many_arguments)]
pub async fn dense_search_path(
    pool: &sqlx::PgPool,
    owners: impl Into<Owners>,
    leaves: &[Uuid],
    q: &[f32],
    limit: i64,
    filter: &ChunkFilter,
    ef_search: u32,
    path: DensePath,
) -> Result<Vec<(Uuid, f32)>> {
    if leaves.is_empty() || limit <= 0 {
        return Ok(Vec::new());
    }
    let owners = owners.into().ids();
    let mut tx = pool.begin().await?;
    match path {
        DensePath::Hnsw => set_hnsw_scan(&mut tx, ef_search).await?,
        DensePath::Exact => set_exact_scan(&mut tx).await?,
    }
    let query = if filter.is_empty() {
        sqlx::query_as::<_, (Uuid, f64)>(DENSE_SQL)
    } else {
        sqlx::query_as::<_, (Uuid, f64)>(DENSE_FILTERED_SQL)
    };
    let mut query = query
        .bind(Vector::from(q.to_vec()))
        .bind(leaves)
        .bind(limit)
        .bind(owners);
    if !filter.is_empty() {
        query = query
            .bind(filter.source_ids.as_deref())
            .bind(filter.metadata.as_ref().map(Json));
    }
    let rows: Vec<(Uuid, f64)> = query.fetch_all(&mut *tx).await?;
    tx.commit().await?;
    Ok(rows.into_iter().map(|(id, sim)| (id, sim as f32)).collect())
}

/// (chunk_id, ts_rank_cd), best first, among `filter`'s chunks of `owners`' `leaves`. A chunk matches if it contains ANY
/// query lexeme. Text without lexemes (empty, punctuation only) → [].
pub async fn lexical_search(
    pool: &sqlx::PgPool,
    owners: impl Into<Owners>,
    leaves: &[Uuid],
    text: &str,
    limit: i64,
    filter: &ChunkFilter,
) -> Result<Vec<(Uuid, f32)>> {
    if leaves.is_empty() || limit <= 0 || text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let query = if filter.is_empty() {
        sqlx::query_as::<_, (Uuid, f32)>(LEXICAL_SQL)
    } else {
        sqlx::query_as::<_, (Uuid, f32)>(LEXICAL_FILTERED_SQL)
    };
    let mut query = query
        .bind(text)
        .bind(leaves)
        .bind(limit)
        .bind(owners.into().ids());
    if !filter.is_empty() {
        query = query
            .bind(filter.source_ids.as_deref())
            .bind(filter.metadata.as_ref().map(Json));
    }
    let rows: Vec<(Uuid, f32)> = query.fetch_all(pool).await?;
    Ok(rows)
}

/// Texts for the given chunk ids, in arbitrary order (callers re-order).
/// Chunk ids of stores not `owners`' are skipped.
pub async fn fetch_chunks(
    pool: &sqlx::PgPool,
    owners: impl Into<Owners>,
    ids: &[Uuid],
) -> Result<Vec<ChunkText>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let rows: Vec<ChunkText> = sqlx::query_as(FETCH_SQL)
        .bind(ids)
        .bind(owners.into().ids())
        .fetch_all(pool)
        .await?;
    Ok(rows)
}
