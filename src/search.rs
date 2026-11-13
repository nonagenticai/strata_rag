//! Leaf search: dense (HNSW, or exact for a leaf set of at most `STRATA_EXACT_SCAN_MAX_CHUNKS`
//! chunks) + Postgres full-text, fused by RRF, then cross-encoder rerank of the
//! first `rerank_candidates` fused rows (spec §6.3–6.4, stitching note 9). With platform packs
//! among the leaves, tenant-over-platform precedence (`crate::precedence`) is applied around the
//! rerank (`Precedence`).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use uuid::Uuid;

use crate::db::search::{
    ChunkFilter, ChunkText, DEFAULT_EF_SEARCH, DEFAULT_EXACT_SCAN_MAX_CHUNKS, DensePath, Owners,
    choose_dense_path, dense_search_path, fetch_chunks, leaf_set_chunks, lexical_search,
};
use crate::error::{Result, StrataError};
use crate::inference::InferenceHandle;
use crate::precedence::{self, Layer, PrecedenceOpts, Ranked};
use crate::tenant::Tenant;

pub const RRF_K: f32 = 60.0;
pub const CANDIDATES: usize = 50;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Scores {
    pub dense: Option<f32>,
    pub lexical: Option<f32>,
    /// `None` = not reranked: the row fell outside the rerank budget (stitching note 9).
    pub rerank: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ScoredChunk {
    pub chunk: ChunkText,
    pub scores: Scores,
    /// `Platform` for a chunk of the platform owner's stores (`Owners::layer_of`).
    pub layer: Layer,
}

/// Precedence ranks by the rerank score; a row outside the rerank budget (`None`) ranks below
/// every reranked row, keeps its RRF order, and is never rescored.
impl Ranked for ScoredChunk {
    fn layer(&self) -> Layer {
        self.layer
    }
    fn score(&self) -> f32 {
        self.scores.rerank.unwrap_or(f32::NEG_INFINITY)
    }
    fn set_score(&mut self, s: f32) {
        if self.scores.rerank.is_some() {
            self.scores.rerank = Some(s);
        }
    }
    fn text(&self) -> &str {
        &self.chunk.text
    }
}

/// Tenant-over-platform precedence for one search (spec platform-knowledge §6): platform
/// candidates whose document `topic_keys` meet `shadow` are dropped before the rerank
/// (`precedence::shadowed`); after it, `precedence::apply` with `opts` (margin, minimum score,
/// the cut to `top_k`, near-duplicate dedup against the kept tenant hits). Nothing changes without platform candidates.
#[derive(Debug, Clone)]
pub struct Precedence {
    pub shadow: HashSet<String>,
    pub opts: PrecedenceOpts,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct SearchTimings {
    pub search: u64,
    pub rerank: u64,
}

pub(crate) fn elapsed_ms(t: Instant) -> u64 {
    u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Reciprocal rank fusion: score(id) = Σ 1 / (k + rank), rank 1-based. Sorted by score desc;
/// equal scores keep first-appearance order (list 0 scanned fully, then list 1, …).
pub fn rrf(lists: &[Vec<Uuid>], k: f32) -> Vec<(Uuid, f32)> {
    rrf_weighted(lists, &vec![1.0; lists.len()], k)
}

/// `rrf` with a per-list weight: score(id) = Σ w_list / (k + rank). `weights` must have one
/// entry per list; a missing weight is a caller bug and panics in debug, counts as 1.0 in release.
pub fn rrf_weighted(lists: &[Vec<Uuid>], weights: &[f32], k: f32) -> Vec<(Uuid, f32)> {
    debug_assert_eq!(lists.len(), weights.len(), "one RRF weight per list");
    let mut order: Vec<Uuid> = Vec::new();
    let mut score: HashMap<Uuid, f32> = HashMap::new();
    for (li, list) in lists.iter().enumerate() {
        let w = weights.get(li).copied().unwrap_or(1.0);
        for (rank, id) in list.iter().enumerate() {
            let s = w / (k + (rank + 1) as f32);
            match score.get_mut(id) {
                Some(v) => *v += s,
                None => {
                    score.insert(*id, s);
                    order.push(*id);
                }
            }
        }
    }
    let mut fused: Vec<(Uuid, f32)> = order.into_iter().map(|id| (id, score[&id])).collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1)); // stable: ties keep first appearance
    fused
}

/// Dense top `CANDIDATES` per query vector + lexical top `CANDIDATES` over `leaves`
/// (concurrently), fused by RRF and cut to `CANDIDATES`. `q_vecs[0]` is the question itself; any
/// further vectors are expansion variants (see `expand`), and then the question's dense list
/// carries `expand::ORIGINAL_QUERY_WEIGHT`. `scores.dense` is the best similarity over all vectors. The first `min(rerank_candidates, fused)` are reranked and come first
/// (rerank score desc, ties in RRF order); the remaining fused rows follow in RRF order with
/// `scores.rerank = None`. Truncated to `top_k`. No leaves → no SQL, empty result. Only
/// `tenant`'s leaves and chunks are read. `filter`
/// restricts both candidate searches in SQL (see `ChunkFilter`).
#[allow(clippy::too_many_arguments)]
pub async fn search_leaves(
    pool: &sqlx::PgPool,
    tenant: Tenant,
    inf: &InferenceHandle,
    leaves: &[Uuid],
    q_vecs: &[Vec<f32>],
    q_text: &str,
    top_k: usize,
    rerank_candidates: usize,
    filter: &ChunkFilter,
) -> Result<(Vec<ScoredChunk>, SearchTimings)> {
    search_leaves_ef(
        pool,
        tenant,
        inf,
        leaves,
        q_vecs,
        q_text,
        top_k,
        rerank_candidates,
        filter,
        DEFAULT_EF_SEARCH,
        DEFAULT_EXACT_SCAN_MAX_CHUNKS,
    )
    .await
}

/// `search_leaves` with the dense searches' `hnsw.ef_search` (`STRATA_HNSW_EF_SEARCH`) and exact-scan
/// threshold (`STRATA_EXACT_SCAN_MAX_CHUNKS`, see `choose_dense_path`).
#[allow(clippy::too_many_arguments)]
pub async fn search_leaves_ef(
    pool: &sqlx::PgPool,
    tenant: Tenant,
    inf: &InferenceHandle,
    leaves: &[Uuid],
    q_vecs: &[Vec<f32>],
    q_text: &str,
    top_k: usize,
    rerank_candidates: usize,
    filter: &ChunkFilter,
    ef_search: u32,
    exact_max_chunks: u64,
) -> Result<(Vec<ScoredChunk>, SearchTimings)> {
    let (hits, timings, _) = search_groups_ef(
        pool,
        tenant.into(),
        inf,
        &[DenseGroup { leaves, q_vecs }],
        q_text,
        top_k,
        rerank_candidates,
        filter,
        ef_search,
        exact_max_chunks,
        None,
    )
    .await?;
    Ok((hits, timings))
}

/// One embedding model's share of a search: the leaves embedded by that model and the query
/// vectors that model made (`q_vecs[0]` the question, then any expansion variants).
#[derive(Debug, Clone, Copy)]
pub struct DenseGroup<'a> {
    pub leaves: &'a [Uuid],
    pub q_vecs: &'a [Vec<f32>],
}

/// How one group's dense searches ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupSearch {
    /// Wall time of the group's chunk count and dense searches.
    pub ms: u64,
    pub path: DensePath,
}

/// `search_leaves_ef` over leaves in several embedding models (spec §6.3): each group's dense
/// searches run over its own leaves with its own vectors, all groups concurrently; the lexical
/// search (model-free) runs once over every group's leaves. One RRF fuses every list: it is
/// rank-based, so similarities from different models are never compared. `inf` reranks the fused
/// candidates. Each group first counts its leaves' chunks (`leaf_set_chunks`) and takes the exact
/// path at most `exact_max_chunks` (0 = never), else the HNSW path (`choose_dense_path`). Also
/// answers each non-empty group's dense-search time and path, in `groups` order. Only `owners`'
/// leaves and chunks are read; `prec` (when given) ranks platform rows under tenant rows.
#[allow(clippy::too_many_arguments)]
pub async fn search_groups_ef(
    pool: &sqlx::PgPool,
    owners: Owners,
    inf: &InferenceHandle,
    groups: &[DenseGroup<'_>],
    q_text: &str,
    top_k: usize,
    rerank_candidates: usize,
    filter: &ChunkFilter,
    ef_search: u32,
    exact_max_chunks: u64,
    prec: Option<&Precedence>,
) -> Result<(Vec<ScoredChunk>, SearchTimings, Vec<GroupSearch>)> {
    let groups: Vec<DenseGroup<'_>> = groups
        .iter()
        .copied()
        .filter(|g| !g.leaves.is_empty() && !g.q_vecs.is_empty())
        .collect();
    if groups.is_empty() || top_k == 0 {
        return Ok((Vec::new(), SearchTimings::default(), Vec::new()));
    }
    let all_leaves: Vec<Uuid> = groups
        .iter()
        .flat_map(|g| g.leaves.iter().copied())
        .collect();
    let t_search = Instant::now();
    let limit = CANDIDATES as i64;
    let (dense_groups, lexical) = tokio::join!(
        futures_util::future::try_join_all(groups.iter().map(|g| async move {
            let t = Instant::now();
            let path = if exact_max_chunks == 0 {
                DensePath::Hnsw // no count needed
            } else {
                choose_dense_path(
                    leaf_set_chunks(pool, owners, g.leaves).await?,
                    exact_max_chunks,
                )
            };
            crate::metrics::record_dense_path(path);
            let lists = futures_util::future::try_join_all(g.q_vecs.iter().map(|v| {
                dense_search_path(pool, owners, g.leaves, v, limit, filter, ef_search, path)
            }))
            .await?;
            Ok::<_, StrataError>((
                lists,
                GroupSearch {
                    ms: elapsed_ms(t),
                    path,
                },
            ))
        })),
        lexical_search(pool, owners, &all_leaves, q_text, limit, filter)
    );
    let (dense_groups, lexical) = (dense_groups?, lexical?);
    let group_searches: Vec<GroupSearch> = dense_groups.iter().map(|(_, g)| *g).collect();
    let mut lists: Vec<Vec<Uuid>> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    for (group_lists, _) in &dense_groups {
        lists.extend(
            group_lists
                .iter()
                .map(|l| l.iter().map(|(id, _)| *id).collect::<Vec<Uuid>>()),
        );
        weights.extend(crate::expand::dense_weights(group_lists.len()));
    }
    let dense_lists: Vec<&Vec<(Uuid, f32)>> =
        dense_groups.iter().flat_map(|(l, _)| l.iter()).collect();
    // the best similarity over the chunk's own model's vectors (a chunk sits in one leaf, so in
    // one group: similarities of different models never meet here)
    let mut dense_by: HashMap<Uuid, f32> = HashMap::new();
    for (id, sim) in dense_lists.iter().copied().flatten() {
        let best = dense_by.entry(*id).or_insert(*sim);
        *best = best.max(*sim);
    }
    let lexical_by: HashMap<Uuid, f32> = lexical.iter().copied().collect();
    lists.push(lexical.iter().map(|(id, _)| *id).collect());
    weights.push(1.0);
    let fused = rrf_weighted(&lists, &weights, RRF_K);
    // Only rows that can reach the output need their text: the rerank budget or top_k, whichever is larger.
    let ids: Vec<Uuid> = fused
        .iter()
        .take(CANDIDATES.min(top_k.max(rerank_candidates)))
        .map(|(id, _)| *id)
        .collect();
    let mut by_id: HashMap<Uuid, ChunkText> = fetch_chunks(pool, owners, &ids)
        .await?
        .into_iter()
        .map(|c| (c.chunk_id, c))
        .collect();
    // RRF order; a chunk deleted between search and fetch simply drops out.
    let mut candidates: Vec<ScoredChunk> = ids
        .iter()
        .filter_map(|id| by_id.remove(id))
        .map(|chunk| ScoredChunk {
            scores: Scores {
                dense: dense_by.get(&chunk.chunk_id).copied(),
                lexical: lexical_by.get(&chunk.chunk_id).copied(),
                rerank: None,
            },
            layer: owners.layer_of(chunk.owner),
            chunk,
        })
        .collect();
    // shadowing, strongest precedence rule: before the rerank, so a shadowed platform row never
    // takes a rerank slot
    if let Some(p) = prec {
        candidates.retain(|c| {
            c.layer != Layer::Platform || !precedence::shadowed(&c.chunk.topic_keys, &p.shadow)
        });
    }
    let search = elapsed_ms(t_search);
    let n_rerank = rerank_candidates.min(candidates.len());
    if n_rerank == 0 {
        let candidates = apply_precedence(candidates, prec, top_k);
        return Ok((
            candidates,
            SearchTimings { search, rerank: 0 },
            group_searches,
        ));
    }

    let t_rerank = Instant::now();
    let passages: Vec<String> = candidates[..n_rerank]
        .iter()
        .map(|c| c.chunk.text.clone())
        .collect();
    let scores = inf.rerank(q_text.to_string(), passages).await?;
    if scores.len() != n_rerank {
        return Err(StrataError::Model(format!(
            "reranker returned {} scores for {} passages",
            scores.len(),
            n_rerank
        )));
    }
    for (c, s) in candidates.iter_mut().zip(scores) {
        c.scores.rerank = Some(s);
    }
    // stable: rerank ties keep RRF order; the un-reranked tail is untouched
    candidates[..n_rerank].sort_by(|a, b| {
        let (a, b) = (
            a.scores.rerank.unwrap_or(f32::MIN),
            b.scores.rerank.unwrap_or(f32::MIN),
        );
        b.total_cmp(&a)
    });
    let candidates = apply_precedence(candidates, prec, top_k);
    Ok((
        candidates,
        SearchTimings {
            search,
            rerank: elapsed_ms(t_rerank),
        },
        group_searches,
    ))
}

/// `precedence::apply` (which cuts to `top_k` itself, so the near-duplicate dedup only counts
/// tenant hits that are kept) when there is a precedence and at least one platform row; else
/// `hits` cut to `top_k` (a tenant-only search keeps exactly its pre-platform order).
fn apply_precedence(
    mut hits: Vec<ScoredChunk>,
    prec: Option<&Precedence>,
    top_k: usize,
) -> Vec<ScoredChunk> {
    match prec {
        Some(p) if hits.iter().any(|h| h.layer == Layer::Platform) => {
            precedence::apply(hits, &p.opts, top_k)
        }
        _ => {
            hits.truncate(top_k);
            hits
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::InferenceHandle;
    use uuid::Uuid;

    fn id(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    #[test]
    fn rrf_sums_reciprocal_ranks() {
        let fused = rrf(&[vec![id(1), id(2)], vec![id(2), id(3)]], 60.0);
        let ids: Vec<Uuid> = fused.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![id(2), id(1), id(3)]);
        assert!((fused[0].1 - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-7);
        assert!((fused[1].1 - 1.0 / 61.0).abs() < 1e-7);
        assert!((fused[2].1 - 1.0 / 62.0).abs() < 1e-7);
    }

    #[test]
    fn rrf_ties_keep_first_appearance() {
        // a and b: 1/61 + 1/62 each, exactly equal → a (seen first) wins
        let fused = rrf(&[vec![id(1), id(2)], vec![id(2), id(1)]], 60.0);
        assert_eq!(fused[0].0, id(1));
        assert_eq!(fused[1].0, id(2));
        assert_eq!(fused[0].1, fused[1].1);
    }

    #[test]
    fn rrf_item_in_one_list_only() {
        // a = 1/61 (list 0 rank 1), c = 1/61 (list 1 rank 1), b = 1/62: a before c by first appearance
        let fused = rrf(&[vec![id(1), id(2)], vec![id(3)]], 60.0);
        let ids: Vec<Uuid> = fused.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids, vec![id(1), id(3), id(2)]);
    }

    #[test]
    fn rrf_weighted_scales_each_list() {
        // id 2 is rank 1 in a weight-2 list: 2/61 beats id 1's 1/61 + 1/62
        let fused = rrf_weighted(&[vec![id(2)], vec![id(1), id(2)]], &[2.0, 1.0], 60.0);
        assert_eq!(fused[0].0, id(2));
        assert!((fused[0].1 - (2.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-7);
        assert!((fused[1].1 - 1.0 / 61.0).abs() < 1e-7);
        // unit weights are exactly `rrf`
        let lists = [vec![id(1), id(2)], vec![id(2), id(3)]];
        assert_eq!(rrf_weighted(&lists, &[1.0, 1.0], 60.0), rrf(&lists, 60.0));
    }

    #[test]
    fn rrf_of_nothing_is_empty() {
        assert!(rrf(&[], 60.0).is_empty());
        assert!(rrf(&[vec![], vec![]], 60.0).is_empty());
    }

    #[tokio::test]
    async fn search_leaves_with_no_leaves_issues_no_sql() {
        // a lazy pool to a port nobody listens on: any query would error
        let pool = sqlx::postgres::PgPoolOptions::new()
            .connect_lazy("postgres://nobody:nothing@127.0.0.1:1/none")
            .unwrap();
        let inf = InferenceHandle::fake();
        let (hits, timings) = search_leaves(
            &pool,
            Tenant::new(Uuid::from_u128(1)).unwrap(),
            &inf,
            &[],
            &[vec![0.0; 4]],
            "anything",
            8,
            8,
            &ChunkFilter::default(),
        )
        .await
        .unwrap();
        assert!(hits.is_empty());
        assert_eq!(timings.search, 0);
        assert_eq!(timings.rerank, 0);
    }
}
