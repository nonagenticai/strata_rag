//! Routing over the store tree (spec §6 step 2).
//!
//! Each tenant's tree (ids, kinds, names, descriptions, `profile_vec`s) is cached in memory, keyed by
//! tenant, and rebuilt lazily after any write to that tenant's stores (`RoutingCache::invalidate`). `route()` walks it breadth-first from a start node and returns
//! the reached leaves; laya (a `ChoiceOracle`) is consulted per router depending on the mode.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use uuid::Uuid;

use crate::db::stores::{RoutingRow, StoreKind, list_routing_rows};
use crate::error::{Result, StrataError};
use crate::tenant::Tenant;
use crate::vecmath::cosine;

pub const LAYA_PICK_THRESHOLD: f32 = 0.15;
pub const AMBIGUITY_MARGIN: f32 = 0.05;
/// laya (Jev) choice questions accept 2..=255 options.
const LAYA_MIN_OPTIONS: usize = 2;
const LAYA_MAX_OPTIONS: usize = 255;
/// Default wall-clock budget for all laya calls of one query (`STRATA_LAYA_QUERY_BUDGET_MS`).
pub const DEFAULT_LAYA_QUERY_BUDGET_MS: u64 = 1200;
/// `laya_fallback` once the query's laya budget is spent: the remaining routers are routed by embed.
pub const LAYA_BUDGET_EXHAUSTED: &str = "query_budget_exhausted";

pub type BoxFut<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

pub trait ChoiceOracle: Send + Sync {
    /// options: (name, description). Returns name → probability.
    fn choose<'a>(
        &'a self,
        query: &'a str,
        options: &'a [(String, String)],
    ) -> BoxFut<'a, Result<HashMap<String, f32>>>;
}

#[derive(Debug, Clone)]
pub struct TreeNode {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub kind: StoreKind,
    pub name: String,
    pub description: String,
    pub profile_vec: Vec<f32>,
    /// `stores.embed_model` (`"{model}@{dims}"`).
    pub embed_model: String,
}

#[derive(Debug, Default)]
pub struct Tree {
    pub nodes: HashMap<Uuid, TreeNode>,
    pub children: HashMap<Uuid, Vec<Uuid>>,
    pub root: Option<Uuid>,
    /// How many stores have no parent (D90: more than one makes a query without `root_id`
    /// ambiguous).
    pub root_count: usize,
}

impl Tree {
    /// Builds the tree from all store rows, in any order. Children are sorted by id (uuid v7 = creation
    /// order) so routing is deterministic. If several rows have no parent, the smallest id is the root.
    pub fn from_rows(rows: Vec<RoutingRow>) -> Tree {
        let mut tree = Tree::default();
        for row in rows {
            match row.parent_id {
                Some(parent) => tree.children.entry(parent).or_default().push(row.id),
                None => {
                    tree.root_count += 1;
                    if tree.root.is_none_or(|r| row.id < r) {
                        tree.root = Some(row.id);
                    }
                }
            }
            tree.nodes.insert(
                row.id,
                TreeNode {
                    id: row.id,
                    parent_id: row.parent_id,
                    kind: row.kind,
                    name: row.name,
                    description: row.description,
                    profile_vec: row.profile_vec,
                    embed_model: row.embed_model,
                },
            );
        }
        for kids in tree.children.values_mut() {
            kids.sort();
        }
        tree
    }

    /// The first store at or below `start`, or above it up to the root, whose `embed_model` is not
    /// `configured`, as an `EmbedModelMismatch`. Every store a query routes through must share the
    /// query's vector space; an ancestor is checked too because its `profile_vec` already averages
    /// the subtree's vectors, so a mismatch there means the tree is mixed (refuse rather than serve
    /// part of it).
    pub fn check_embed_model(&self, start: Uuid, configured: &str) -> Result<()> {
        let mismatch = |node: &TreeNode| StrataError::EmbedModelMismatch {
            store_id: node.id,
            store_model: node.embed_model.clone(),
            configured: configured.to_string(),
        };
        let mut up = self.nodes.get(&start).and_then(|n| n.parent_id);
        let mut hops = 0;
        while let Some(id) = up {
            let Some(node) = self.nodes.get(&id) else {
                break;
            };
            if node.embed_model != configured {
                return Err(mismatch(node));
            }
            hops += 1;
            if hops > self.nodes.len() {
                break; // cycle guard, as in `path`
            }
            up = node.parent_id;
        }
        let mut stack = vec![start];
        let mut seen = HashSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue; // cycle guard, as in `path`
            }
            let Some(node) = self.nodes.get(&id) else {
                continue;
            };
            if node.embed_model != configured {
                return Err(mismatch(node));
            }
            if let Some(kids) = self.children.get(&id) {
                stack.extend(kids.iter().copied());
            }
        }
        Ok(())
    }

    /// Names from the root down to `id` (inclusive). Unknown id → empty.
    pub fn path(&self, id: Uuid) -> Vec<String> {
        let mut names = Vec::new();
        let mut cur = Some(id);
        while let Some(c) = cur {
            let Some(node) = self.nodes.get(&c) else {
                break;
            };
            names.push(node.name.clone());
            if names.len() > self.nodes.len() {
                break; // cycle guard; the schema forbids cycles, this only bounds a corrupt tree
            }
            cur = node.parent_id;
        }
        names.reverse();
        names
    }
}

/// Lazily rebuilt snapshots of each tenant's store tree, keyed by tenant: a tree is built from
/// that tenant's rows only, so a query can never route into another tenant's stores. Every write
/// path calls `invalidate(tenant)` (this process sees its own writes at once).
///
/// Writes by OTHER processes (another replica, `strata reembed` from the CLI) are seen through the
/// database: every store write bumps the tenant's `store_generations` row at commit (migration
/// 0010), `get` reads that one indexed row first, and a snapshot is served only while its database
/// generation is still current. There is no TTL to wait out.
///
/// The database load runs with NO lock held (one tenant's slow load never stalls another tenant's
/// cached reads). A per-tenant generation, bumped by `invalidate`, makes that safe: a load keeps
/// its result only if no invalidation of its tenant happened since it read the generation, so a
/// load that started before a write's invalidation can never park the pre-write tree in the cache.
/// An empty tree is never kept (a caller allowed to name any tenant could otherwise grow the map
/// with one entry per made-up tenant).
///
/// Loads are single-flight per tenant: concurrent misses (every query during an ingest, which
/// bumps the generation per batch) queue on that tenant's load lock, and those behind the first
/// are answered from the tree it cached instead of each reloading the whole tree.
#[derive(Clone, Default)]
pub struct RoutingCache {
    inner: Arc<tokio::sync::RwLock<CacheState>>,
}

#[derive(Default)]
struct CacheState {
    /// The snapshot and the database generation (`store_generations`) it was loaded under.
    trees: HashMap<Tenant, (u64, Arc<Tree>)>,
    /// Invalidations per tenant (absent = 0).
    generations: HashMap<Tenant, u64>,
    /// Per-tenant load locks, present only while a load is running or awaited (removed by the
    /// last holder, so made-up tenants leave nothing behind).
    loading: HashMap<Tenant, Arc<tokio::sync::Mutex<()>>>,
}

impl RoutingCache {
    pub async fn get(&self, pool: &sqlx::PgPool, tenant: Tenant) -> Result<Arc<Tree>> {
        // read BEFORE the rows: a tree loaded after this read is at least this new, so it is never
        // cached under a generation newer than its contents
        let db_generation = store_generation(pool, tenant).await?;
        self.get_or_load(tenant, db_generation, list_routing_rows(pool, tenant))
            .await
    }

    /// `get` with the database generation and the row load supplied by the caller (`get` passes
    /// `store_generations` and the tenant's `list_routing_rows`). A cached tree is served only when
    /// it was loaded under `db_generation`. On a miss the caller takes the tenant's load lock; if a
    /// load that held it before cached a tree at least as new as `db_generation`, that tree is
    /// served and `load` is never polled. The load holds only that tenant's lock (never the cache
    /// lock); its tree is cached only when non-empty and no `invalidate(tenant)` ran since it began.
    pub async fn get_or_load<F>(
        &self,
        tenant: Tenant,
        db_generation: u64,
        load: F,
    ) -> Result<Arc<Tree>>
    where
        F: std::future::Future<Output = Result<Vec<RoutingRow>>>,
    {
        {
            let state = self.inner.read().await;
            if let Some((at, tree)) = state.trees.get(&tenant)
                && *at == db_generation
            {
                return Ok(Arc::clone(tree));
            }
        }
        let lock = Arc::clone(self.inner.write().await.loading.entry(tenant).or_default());
        let result = {
            let _held = lock.lock().await;
            self.load_locked(tenant, db_generation, load).await
        };
        // the last holder removes the lock: clones are only taken and dropped under the write
        // lock, so once ours is gone a count of 1 (the map's) means no other holder remains
        let mut state = self.inner.write().await;
        drop(lock);
        if state
            .loading
            .get(&tenant)
            .is_some_and(|l| Arc::strong_count(l) == 1)
        {
            state.loading.remove(&tenant);
        }
        result
    }

    /// The miss path of `get_or_load`, run under `tenant`'s load lock.
    async fn load_locked<F>(&self, tenant: Tenant, db_generation: u64, load: F) -> Result<Arc<Tree>>
    where
        F: std::future::Future<Output = Result<Vec<RoutingRow>>>,
    {
        let generation = {
            let state = self.inner.read().await;
            // the load we queued behind cached a tree at least this new: it is current
            if let Some((at, tree)) = state.trees.get(&tenant)
                && *at >= db_generation
            {
                return Ok(Arc::clone(tree));
            }
            state.generations.get(&tenant).copied().unwrap_or(0)
        };
        let tree = Arc::new(Tree::from_rows(load.await?));
        if tree.nodes.is_empty() {
            return Ok(tree);
        }
        let mut state = self.inner.write().await;
        if state.generations.get(&tenant).copied().unwrap_or(0) == generation {
            match state.trees.get(&tenant) {
                // a concurrent load of the same (or a newer) database generation won; it is current
                Some((at, cached)) if *at >= db_generation => return Ok(Arc::clone(cached)),
                _ => {
                    state
                        .trees
                        .insert(tenant, (db_generation, Arc::clone(&tree)));
                    return Ok(tree);
                }
            }
        }
        // invalidated while loading: answer from what was read, but do not cache it
        Ok(tree)
    }

    pub async fn invalidate(&self, tenant: Tenant) {
        let mut state = self.inner.write().await;
        state.trees.remove(&tenant);
        *state.generations.entry(tenant).or_default() += 1;
    }

    /// Tenants with a cached tree (tests and diagnostics).
    pub async fn cached_tenants(&self) -> Vec<Tenant> {
        let mut t: Vec<Tenant> = self.inner.read().await.trees.keys().copied().collect();
        t.sort();
        t
    }
}

/// `tenant`'s store-tree generation (`store_generations`, bumped at commit of every store write;
/// 0 before the first).
pub async fn store_generation(pool: &sqlx::PgPool, tenant: Tenant) -> Result<u64> {
    let g: Option<i64> =
        sqlx::query_scalar("SELECT generation FROM store_generations WHERE tenant = $1")
            .bind(tenant.id())
            .fetch_optional(pool)
            .await?;
    Ok(g.map_or(0, |g| u64::try_from(g).unwrap_or(0)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RouteMode {
    Embed,
    Laya,
    #[default]
    Hybrid,
}

#[derive(Debug, Clone, Copy)]
pub struct RouteParams {
    pub mode: RouteMode,
    pub beam: usize,
    pub min_score: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RouteOutcome {
    pub mode_used: RouteMode,
    pub leaves: Vec<Uuid>,
    pub laya_consulted: bool,
    pub laya_fallback: Option<String>,
}

/// Top `beam` children plus every child scoring ≥ `min_score`, capped at `2·beam`. Input sorted desc.
pub fn embed_select(scored_desc: &[(Uuid, f32)], beam: usize, min_score: f32) -> Vec<Uuid> {
    scored_desc
        .iter()
        .enumerate()
        .filter(|(i, (_, s))| *i < beam || *s >= min_score)
        .take(beam.saturating_mul(2))
        .map(|(_, (id, _))| *id)
        .collect()
}

/// Ambiguous when the top two are within `AMBIGUITY_MARGIN`, or more than `beam` children clear `min_score`.
pub fn is_ambiguous(scored_desc: &[(Uuid, f32)], beam: usize, min_score: f32) -> bool {
    let close = scored_desc.len() >= 2 && scored_desc[0].1 - scored_desc[1].1 < AMBIGUITY_MARGIN;
    let crowded = scored_desc.iter().filter(|(_, s)| *s >= min_score).count() > beam;
    close || crowded
}

/// Per-query laya bookkeeping.
struct LayaState<'o> {
    oracle: Option<&'o dyn ChoiceOracle>,
    consulted: bool,
    fallback: Option<String>,
    /// Set after a level with an oracle error: no further calls in this query, so a slow or broken
    /// laya costs at most one (concurrent) round of timeouts per query (Review Focus #5).
    tripped: bool,
    /// End of this query's laya budget. `tripped` bounds a FAILING laya; this bounds a slow but
    /// SUCCESSFUL one, which BFS would otherwise call once per level at up to the per-call timeout
    /// each. Every call is cut at the deadline (the client's own per-call timeout still applies, so a
    /// call gets min(remaining budget, per-call timeout)); once it passes, no further calls.
    deadline: tokio::time::Instant,
}

/// Whether one router's laya call goes ahead (decided before a level's calls start).
enum Gate {
    /// No call, nothing to record (laya not wanted here, or already tripped).
    Skip,
    /// No call: record this fallback reason; `trip` stops the remaining calls of the query.
    Fail {
        reason: &'static str,
        trip: bool,
    },
    Call,
}

impl LayaState<'_> {
    fn fail(&mut self, reason: impl Into<String>) {
        if self.fallback.is_none() {
            self.fallback = Some(reason.into());
        }
    }

    /// Whether a router with `scored` children may be put to laya. Pure: the caller records the
    /// outcome in frontier order (`settle`), so the first fallback reason is deterministic.
    fn gate(&self, scored: &[(Uuid, f32)]) -> Gate {
        if self.oracle.is_none() {
            return Gate::Fail {
                reason: "unconfigured",
                trip: false,
            };
        }
        if self.tripped {
            return Gate::Skip;
        }
        if !(LAYA_MIN_OPTIONS..=LAYA_MAX_OPTIONS).contains(&scored.len()) {
            return Gate::Fail {
                reason: "option_count",
                trip: false,
            };
        }
        if tokio::time::Instant::now() >= self.deadline {
            return Gate::Fail {
                reason: LAYA_BUDGET_EXHAUSTED,
                trip: true,
            };
        }
        Gate::Call
    }

    /// One laya call, cut at the query's deadline. Shared borrow: a level's calls run concurrently.
    async fn call(
        &self,
        tree: &Tree,
        scored: &[(Uuid, f32)],
        query_text: &str,
        beam: usize,
    ) -> std::result::Result<Vec<Uuid>, String> {
        let Some(oracle) = self.oracle else {
            return Err("unconfigured".to_string());
        };
        let call = laya_pick(oracle, tree, scored, query_text, beam);
        match tokio::time::timeout_at(self.deadline, call).await {
            Ok(result) => result,
            Err(_elapsed) => Err(LAYA_BUDGET_EXHAUSTED.to_string()),
        }
    }

    /// Records one router's gate and (for `Gate::Call`) its answer. `Some(picks)` when laya
    /// answered; `None` (with the reason recorded) when the caller must use embed.
    fn settle(
        &mut self,
        gate: Gate,
        answer: Option<std::result::Result<Vec<Uuid>, String>>,
    ) -> Option<Vec<Uuid>> {
        match gate {
            Gate::Skip => None,
            Gate::Fail { reason, trip } => {
                self.tripped |= trip;
                self.fail(reason);
                None
            }
            Gate::Call => {
                self.consulted = true;
                match answer? {
                    Ok(picks) => Some(picks),
                    Err(reason) => {
                        self.tripped = true;
                        self.fail(reason);
                        None
                    }
                }
            }
        }
    }
}

fn fallback_reason(e: &StrataError) -> String {
    match e {
        StrataError::Upstream { message, .. } => message.clone(),
        other => other.to_string(),
    }
}

/// Children with probability ≥ `LAYA_PICK_THRESHOLD` (highest first, capped `2·beam`), else the argmax.
async fn laya_pick(
    oracle: &dyn ChoiceOracle,
    tree: &Tree,
    scored: &[(Uuid, f32)],
    query_text: &str,
    beam: usize,
) -> std::result::Result<Vec<Uuid>, String> {
    let options: Vec<(String, String)> = scored
        .iter()
        .filter_map(|(id, _)| {
            tree.nodes
                .get(id)
                .map(|n| (n.name.clone(), n.description.clone()))
        })
        .collect();
    let probs = oracle
        .choose(query_text, &options)
        .await
        .map_err(|e| fallback_reason(&e))?;
    let mut ranked: Vec<(Uuid, f32)> = scored
        .iter()
        .zip(options.iter())
        .filter_map(|((id, _), (name, _))| probs.get(name).map(|p| (*id, *p)))
        .collect();
    if ranked.is_empty() {
        return Err("no_probabilities".to_string());
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1)); // stable: equal probabilities keep embed order (score desc, name asc)
    let picked: Vec<Uuid> = ranked
        .iter()
        .filter(|(_, p)| *p >= LAYA_PICK_THRESHOLD)
        .take(beam.saturating_mul(2))
        .map(|(id, _)| *id)
        .collect();
    if picked.is_empty() {
        Ok(vec![ranked[0].0])
    } else {
        Ok(picked)
    }
}

fn score_children(tree: &Tree, kids: &[Uuid], q: &[f32]) -> Vec<(Uuid, f32)> {
    let mut scored: Vec<(Uuid, f32)> = kids
        .iter()
        .filter_map(|id| tree.nodes.get(id).map(|n| (*id, cosine(q, &n.profile_vec))))
        .collect();
    // score desc, then name asc: a deterministic tie-break (stitching note 8)
    scored.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| tree.nodes[&a.0].name.cmp(&tree.nodes[&b.0].name))
    });
    scored
}

/// Whether `mode` puts this router to laya at all.
fn wants_laya(scored: &[(Uuid, f32)], p: &RouteParams) -> bool {
    match p.mode {
        RouteMode::Embed => false,
        RouteMode::Laya => true,
        RouteMode::Hybrid => is_ambiguous(scored, p.beam, p.min_score),
    }
}

/// One router's picks, and whether laya chose them (laya-chosen children count as clearing `min_score`
/// in `level_select`). `laya`: the router's laya picks, `None` when laya was not consulted or failed.
fn pick(scored: &[(Uuid, f32)], p: &RouteParams, laya: Option<Vec<Uuid>>) -> (Vec<Uuid>, bool) {
    let Some(picks) = laya else {
        return (embed_select(scored, p.beam, p.min_score), false);
    };
    if p.mode != RouteMode::Hybrid {
        return (picks, true);
    }
    // hybrid: laya's picks join the embed top-1
    let mut out = vec![scored[0].0];
    for id in picks {
        if !out.contains(&id) {
            out.push(id);
        }
    }
    (out, true)
}

/// The next level's frontier: every child a router picked, re-ranked across routers by score (desc, then
/// name asc) and cut with the rule `embed_select` applies to one router — top `beam` plus any admitted
/// extra, capped at `2·beam` — where a child is admitted when it scores ≥ `min_score` or laya chose it.
/// Applied per LEVEL, not per router: with a cap per router only, a tree whose fan-out is ≤ `beam` (the
/// seed library: 3 routers of 2–3 leaves at beam 3) keeps every child of every router, so routing
/// degrades to a flat search of all leaves.
fn level_select(
    tree: &Tree,
    mut picked: Vec<(Uuid, f32, bool)>,
    beam: usize,
    min_score: f32,
) -> Vec<Uuid> {
    picked.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| tree.nodes[&a.0].name.cmp(&tree.nodes[&b.0].name))
    });
    picked
        .iter()
        .enumerate()
        .filter(|(i, (_, s, by_laya))| *i < beam || *by_laya || *s >= min_score)
        .take(beam.saturating_mul(2))
        .map(|(_, (id, _, _))| *id)
        .collect()
}

/// `now + budget`; an overflowing budget (absurdly large config) means "no practical limit", not a panic.
fn budget_deadline(budget: std::time::Duration) -> tokio::time::Instant {
    let now = tokio::time::Instant::now();
    now.checked_add(budget)
        .unwrap_or_else(|| now + std::time::Duration::from_secs(86_400 * 365))
}

/// `route_with_budget` at the default laya budget (`DEFAULT_LAYA_QUERY_BUDGET_MS`).
pub async fn route(
    tree: &Tree,
    start: Uuid,
    q: &[f32],
    query_text: &str,
    p: &RouteParams,
    oracle: Option<&dyn ChoiceOracle>,
) -> RouteOutcome {
    let budget = std::time::Duration::from_millis(DEFAULT_LAYA_QUERY_BUDGET_MS);
    route_with_budget(tree, start, q, query_text, p, oracle, budget).await
}

/// Routes `q` from `start` down to leaves, one tree level at a time. Never fails: laya problems become
/// `laya_fallback`. The laya calls of one level run concurrently; their results are applied in
/// frontier order, so the outcome does not depend on which answer arrives first. Leaves come out in selection order (best first within each level). All laya calls
/// of this query share `laya_budget`; once it is spent the remaining routers are routed by embed and
/// `laya_fallback` is `LAYA_BUDGET_EXHAUSTED` (unless an earlier reason was already recorded).
pub async fn route_with_budget(
    tree: &Tree,
    start: Uuid,
    q: &[f32],
    query_text: &str,
    p: &RouteParams,
    oracle: Option<&dyn ChoiceOracle>,
    laya_budget: std::time::Duration,
) -> RouteOutcome {
    let mut laya = LayaState {
        oracle,
        consulted: false,
        fallback: None,
        tripped: false,
        deadline: budget_deadline(laya_budget),
    };
    let mut leaves = Vec::new();
    let mut seen = HashSet::new();
    let mut frontier = Vec::new();
    if tree.nodes.contains_key(&start) {
        seen.insert(start);
        frontier.push(start);
    }
    while !frontier.is_empty() {
        // this level's routers with children, in frontier order, and whether each goes to laya
        let mut routers: Vec<(Vec<(Uuid, f32)>, Gate)> = Vec::new();
        for id in frontier {
            let Some(node) = tree.nodes.get(&id) else {
                continue;
            };
            match node.kind {
                StoreKind::Leaf => leaves.push(id),
                StoreKind::Router => {
                    let Some(kids) = tree.children.get(&id) else {
                        continue;
                    };
                    if kids.is_empty() {
                        continue;
                    }
                    let scored = score_children(tree, kids, q);
                    let gate = if wants_laya(&scored, p) {
                        laya.gate(&scored)
                    } else {
                        Gate::Skip
                    };
                    routers.push((scored, gate));
                }
            }
        }
        let answers = {
            let laya = &laya;
            futures_util::future::join_all(routers.iter().map(|(scored, gate)| async move {
                match gate {
                    Gate::Call => Some(laya.call(tree, scored, query_text, p.beam).await),
                    Gate::Skip | Gate::Fail { .. } => None,
                }
            }))
            .await
        };
        let mut picked: Vec<(Uuid, f32, bool)> = Vec::new();
        for ((scored, gate), answer) in routers.into_iter().zip(answers) {
            let laya_picks = laya.settle(gate, answer);
            let (children, by_laya) = pick(&scored, p, laya_picks);
            for child in children {
                if !seen.insert(child) {
                    continue;
                }
                if let Some((_, score)) = scored.iter().find(|(k, _)| *k == child) {
                    picked.push((child, *score, by_laya));
                }
            }
        }
        frontier = level_select(tree, picked, p.beam, p.min_score);
    }
    RouteOutcome {
        mode_used: p.mode,
        leaves,
        laya_consulted: laya.consulted,
        laya_fallback: laya.fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::stores::RoutingRow;
    use crate::error::StrataError;
    use crate::vecmath::DIM;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // ---------- helpers ----------

    fn axis(parts: &[(usize, f32)]) -> Vec<f32> {
        let mut v = vec![0.0; DIM];
        for (i, x) in parts {
            v[*i] = *x;
        }
        v
    }

    fn id(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn add(
        tree: &mut Tree,
        parent: Option<Uuid>,
        kind: StoreKind,
        name: &str,
        profile: Vec<f32>,
    ) -> Uuid {
        let id = Uuid::now_v7();
        tree.nodes.insert(
            id,
            TreeNode {
                id,
                parent_id: parent,
                kind,
                name: name.to_string(),
                description: format!("{name} description"),
                profile_vec: profile,
                embed_model: "BAAI/bge-m3@1024".to_string(),
            },
        );
        match parent {
            Some(p) => tree.children.entry(p).or_default().push(id),
            None => {
                tree.root = Some(id);
                tree.root_count += 1;
            }
        }
        id
    }

    fn params(mode: RouteMode, beam: usize, min_score: f32) -> RouteParams {
        RouteParams {
            mode,
            beam,
            min_score,
        }
    }

    fn some(o: &dyn ChoiceOracle) -> Option<&dyn ChoiceOracle> {
        Some(o)
    }

    struct FixedOracle {
        probs: HashMap<String, f32>,
        calls: AtomicUsize,
    }

    impl FixedOracle {
        fn new(pairs: &[(&str, f32)]) -> Self {
            Self {
                probs: pairs.iter().map(|(n, p)| (n.to_string(), *p)).collect(),
                calls: AtomicUsize::new(0),
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl ChoiceOracle for FixedOracle {
        fn choose<'a>(
            &'a self,
            _query: &'a str,
            options: &'a [(String, String)],
        ) -> BoxFut<'a, Result<HashMap<String, f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let out: HashMap<String, f32> = options
                .iter()
                .filter_map(|(n, _)| self.probs.get(n).map(|p| (n.clone(), *p)))
                .collect();
            Box::pin(async move { Ok(out) })
        }
    }

    struct ErrOracle {
        message: &'static str,
        calls: AtomicUsize,
    }

    impl ErrOracle {
        fn new(message: &'static str) -> Self {
            Self {
                message,
                calls: AtomicUsize::new(0),
            }
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl ChoiceOracle for ErrOracle {
        fn choose<'a>(
            &'a self,
            _query: &'a str,
            _options: &'a [(String, String)],
        ) -> BoxFut<'a, Result<HashMap<String, f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let message = self.message.to_string();
            Box::pin(async move {
                Err(StrataError::Upstream {
                    service: "laya",
                    message,
                })
            })
        }
    }

    /// library ─┬─ sci (e0)  ─┬─ phys (e0+e2)
    ///           │             └─ chem (e0+e3)
    ///           └─ arts (e1) ─┬─ music (e1+e4)
    ///                         └─ paint (e1+e5)
    /// Physics query q = e0+e2: root scores sci 0.707 / arts 0.0; sci scores phys 1.0 / chem 0.5.
    struct Library {
        tree: Tree,
        root: Uuid,
        sci: Uuid,
        arts: Uuid,
        phys: Uuid,
        chem: Uuid,
        music: Uuid,
        paint: Uuid,
    }

    fn library() -> Library {
        let mut t = Tree::default();
        let root = add(
            &mut t,
            None,
            StoreKind::Router,
            "library",
            axis(&[(9, 1.0)]),
        );
        let sci = add(
            &mut t,
            Some(root),
            StoreKind::Router,
            "sci",
            axis(&[(0, 1.0)]),
        );
        let arts = add(
            &mut t,
            Some(root),
            StoreKind::Router,
            "arts",
            axis(&[(1, 1.0)]),
        );
        let phys = add(
            &mut t,
            Some(sci),
            StoreKind::Leaf,
            "phys",
            axis(&[(0, 1.0), (2, 1.0)]),
        );
        let chem = add(
            &mut t,
            Some(sci),
            StoreKind::Leaf,
            "chem",
            axis(&[(0, 1.0), (3, 1.0)]),
        );
        let music = add(
            &mut t,
            Some(arts),
            StoreKind::Leaf,
            "music",
            axis(&[(1, 1.0), (4, 1.0)]),
        );
        let paint = add(
            &mut t,
            Some(arts),
            StoreKind::Leaf,
            "paint",
            axis(&[(1, 1.0), (5, 1.0)]),
        );
        Library {
            tree: t,
            root,
            sci,
            arts,
            phys,
            chem,
            music,
            paint,
        }
    }

    fn physics_q() -> Vec<f32> {
        axis(&[(0, 1.0), (2, 1.0)])
    }

    /// root ─ x (e0), y (e0 + 0.01·e1), z (e0 + 0.02·e2). Query e0 scores 1.0 / 0.99995 / 0.9998:
    /// top1 − top2 < AMBIGUITY_MARGIN, and all three ≥ 0.35 > beam 1. Ambiguous by both rules.
    struct Close {
        tree: Tree,
        root: Uuid,
        x: Uuid,
        y: Uuid,
        z: Uuid,
    }

    fn close_siblings() -> Close {
        let mut t = Tree::default();
        let root = add(&mut t, None, StoreKind::Router, "root", axis(&[(9, 1.0)]));
        let x = add(&mut t, Some(root), StoreKind::Leaf, "x", axis(&[(0, 1.0)]));
        let y = add(
            &mut t,
            Some(root),
            StoreKind::Leaf,
            "y",
            axis(&[(0, 1.0), (1, 0.01)]),
        );
        let z = add(
            &mut t,
            Some(root),
            StoreKind::Leaf,
            "z",
            axis(&[(0, 1.0), (2, 0.02)]),
        );
        Close {
            tree: t,
            root,
            x,
            y,
            z,
        }
    }

    fn e0() -> Vec<f32> {
        axis(&[(0, 1.0)])
    }

    /// A load that began before `invalidate(tenant)` must not be cached: deterministic, the load
    /// is held open on a channel while the invalidation runs.
    #[tokio::test]
    async fn a_load_racing_an_invalidation_is_not_cached() {
        let tenant = Tenant::new(Uuid::from_u128(7)).unwrap();
        let cache = RoutingCache::default();
        let stale = vec![row(id(1), None, StoreKind::Router, "stale")];
        let fresh = vec![row(id(1), None, StoreKind::Router, "fresh")];
        let (release, gate) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started) = tokio::sync::oneshot::channel::<()>();
        let racing = tokio::spawn({
            let cache = cache.clone();
            let stale = stale.clone();
            async move {
                cache
                    .get_or_load(tenant, 0, async move {
                        let _ = started_tx.send(());
                        gate.await.unwrap();
                        Ok(stale)
                    })
                    .await
                    .unwrap()
            }
        });
        started.await.unwrap(); // the load has read the generation and is in flight
        cache.invalidate(tenant).await; // a write committed meanwhile
        release.send(()).unwrap();
        let served = racing.await.unwrap();
        assert_eq!(
            served.nodes[&id(1)].name,
            "stale",
            "the racing caller gets what it read"
        );
        assert!(
            cache.cached_tenants().await.is_empty(),
            "but it is not cached"
        );
        let next = cache
            .get_or_load(tenant, 0, async { Ok(fresh) })
            .await
            .unwrap();
        assert_eq!(next.nodes[&id(1)].name, "fresh");
        assert_eq!(cache.cached_tenants().await, vec![tenant]);
        // a later load sees the cached fresh tree, not a reload
        let again = cache
            .get_or_load(tenant, 0, async { Ok(stale) })
            .await
            .unwrap();
        assert_eq!(again.nodes[&id(1)].name, "fresh");
    }

    /// A snapshot is served only under the database generation it was loaded at: another
    /// process's write (a newer `store_generations` value) reloads, an older value never replaces
    /// a newer snapshot.
    #[tokio::test]
    async fn a_newer_database_generation_reloads() {
        let tenant = Tenant::new(Uuid::from_u128(9)).unwrap();
        let cache = RoutingCache::default();
        let v1 = vec![row(id(1), None, StoreKind::Router, "v1")];
        let v2 = vec![row(id(1), None, StoreKind::Router, "v2")];
        let t = cache
            .get_or_load(tenant, 1, async { Ok(v1.clone()) })
            .await
            .unwrap();
        assert_eq!(t.nodes[&id(1)].name, "v1");
        let same = cache
            .get_or_load(tenant, 1, async {
                Err(StrataError::Model("must be served from the cache".into()))
            })
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&t, &same));
        let t2 = cache
            .get_or_load(tenant, 2, async { Ok(v2.clone()) })
            .await
            .unwrap();
        assert_eq!(t2.nodes[&id(1)].name, "v2", "generation moved: reloaded");
        // a slow reader that saw generation 1 does not put its older tree back
        let old = cache
            .get_or_load(tenant, 1, async { Ok(v1.clone()) })
            .await
            .unwrap();
        assert_eq!(old.nodes[&id(1)].name, "v2");
        let cur = cache
            .get_or_load(tenant, 2, async {
                Err(StrataError::Model("must be served from the cache".into()))
            })
            .await
            .unwrap();
        assert_eq!(cur.nodes[&id(1)].name, "v2");
    }

    /// The load runs with no lock held: another tenant's cached read completes while it is open.
    #[tokio::test]
    async fn a_slow_load_does_not_block_other_tenants() {
        let (a, b) = (
            Tenant::new(Uuid::from_u128(1)).unwrap(),
            Tenant::new(Uuid::from_u128(2)).unwrap(),
        );
        let cache = RoutingCache::default();
        cache
            .get_or_load(b, 0, async {
                Ok(vec![row(id(2), None, StoreKind::Leaf, "b")])
            })
            .await
            .unwrap();
        let (release, gate) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started) = tokio::sync::oneshot::channel::<()>();
        let slow = tokio::spawn({
            let cache = cache.clone();
            async move {
                cache
                    .get_or_load(a, 0, async move {
                        let _ = started_tx.send(());
                        gate.await.unwrap();
                        Ok(vec![row(id(1), None, StoreKind::Leaf, "a")])
                    })
                    .await
                    .unwrap()
            }
        });
        started.await.unwrap();
        let b_tree = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            cache.get_or_load(b, 0, async {
                Err(StrataError::Model("b must be served from the cache".into()))
            }),
        )
        .await
        .expect("b's cached read is not blocked by a's load")
        .unwrap();
        assert!(b_tree.nodes.contains_key(&id(2)));
        tokio::time::timeout(std::time::Duration::from_secs(2), cache.invalidate(b))
            .await
            .expect("an invalidation is not blocked by a's load");
        release.send(()).unwrap();
        assert!(slow.await.unwrap().nodes.contains_key(&id(1)));
    }

    fn row(id: Uuid, parent: Option<Uuid>, kind: StoreKind, name: &str) -> RoutingRow {
        RoutingRow {
            id,
            parent_id: parent,
            kind,
            name: name.to_string(),
            description: format!("{name} store"),
            profile_vec: axis(&[(0, 1.0)]),
            embed_model: "BAAI/bge-m3@1024".to_string(),
        }
    }

    // ---------- embed_select / is_ambiguous ----------

    #[test]
    fn embed_select_takes_top_beam_plus_above_min_capped_at_twice_beam() {
        let s: Vec<(Uuid, f32)> = [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]
            .iter()
            .enumerate()
            .map(|(i, v)| (id(i as u128), *v))
            .collect();
        // beam 2 → top 2, plus 0.7 and 0.6 (≥ 0.55) = 4 = cap
        assert_eq!(embed_select(&s, 2, 0.55), vec![id(0), id(1), id(2), id(3)]);
        // 0.5 also ≥ 0.4, but the cap is 2·beam = 4
        assert_eq!(embed_select(&s, 2, 0.4), vec![id(0), id(1), id(2), id(3)]);
        // nothing above min: exactly the top beam
        assert_eq!(embed_select(&s, 2, 0.95), vec![id(0), id(1)]);
    }

    #[test]
    fn embed_select_keeps_top_beam_below_min_and_handles_empty() {
        let s = vec![(id(1), 0.2), (id(2), 0.1)];
        assert_eq!(embed_select(&s, 3, 0.35), vec![id(1), id(2)]);
        assert!(embed_select(&[], 3, 0.35).is_empty());
    }

    #[test]
    fn is_ambiguous_on_close_top_two() {
        // values exactly representable in f32: margins 0.03125 and 0.0625
        assert!(is_ambiguous(&[(id(1), 0.5), (id(2), 0.46875)], 3, 0.9));
        assert!(!is_ambiguous(&[(id(1), 0.5), (id(2), 0.4375)], 3, 0.9));
        assert!(!is_ambiguous(&[(id(1), 0.9)], 3, 0.35));
        assert!(!is_ambiguous(&[], 3, 0.35));
    }

    #[test]
    fn is_ambiguous_when_more_than_beam_clear_min_score() {
        let s = vec![(id(1), 0.9), (id(2), 0.7), (id(3), 0.6), (id(4), 0.5)];
        assert!(is_ambiguous(&s, 2, 0.4)); // 4 ≥ 0.4 > beam 2
        assert!(!is_ambiguous(&s, 2, 0.65)); // 2 ≥ 0.65, not > 2; margin 0.2
    }

    // ---------- Tree ----------

    #[test]
    fn tree_from_rows_builds_children_root_and_paths() {
        // rows deliberately out of order: children before parents
        let rows = vec![
            row(id(5), Some(id(2)), StoreKind::Leaf, "chemistry"),
            row(id(4), Some(id(2)), StoreKind::Leaf, "physics"),
            row(id(2), Some(id(1)), StoreKind::Router, "science"),
            row(id(1), None, StoreKind::Router, "library"),
            row(id(3), Some(id(1)), StoreKind::Router, "arts"),
        ];
        let t = Tree::from_rows(rows);
        assert_eq!(t.nodes.len(), 5);
        assert_eq!(t.root, Some(id(1)));
        assert_eq!(t.children[&id(1)], vec![id(2), id(3)]);
        assert_eq!(t.children[&id(2)], vec![id(4), id(5)]);
        assert!(!t.children.contains_key(&id(3)));
        assert_eq!(t.path(id(4)), vec!["library", "science", "physics"]);
        assert_eq!(t.path(id(1)), vec!["library"]);
        assert!(t.path(id(99)).is_empty());
    }

    // ---------- route: embed ----------

    #[tokio::test]
    async fn embed_strict_follows_a_single_branch() {
        let l = library();
        let out = route(
            &l.tree,
            l.root,
            &physics_q(),
            "q",
            &params(RouteMode::Embed, 1, 0.9),
            None,
        )
        .await;
        assert_eq!(out.leaves, vec![l.phys]);
        assert_eq!(out.mode_used, RouteMode::Embed);
        assert!(!out.laya_consulted);
        assert_eq!(out.laya_fallback, None);
    }

    #[tokio::test]
    async fn embed_wide_caps_each_level_not_each_router() {
        let l = library();
        let out = route(
            &l.tree,
            l.root,
            &physics_q(),
            "q",
            &params(RouteMode::Embed, 3, 0.35),
            None,
        )
        .await;
        // per router, beam 3 would keep both children of sci AND of arts (fan-out 2 ≤ beam): all four
        // leaves, a flat search. Per level: phys 1.0, chem 0.5, then music/paint tie at 0.0 → top 3,
        // name asc breaks the tie.
        assert_eq!(out.leaves, vec![l.phys, l.chem, l.music]);
    }

    #[tokio::test]
    async fn route_can_start_below_the_root() {
        let l = library();
        let wide = params(RouteMode::Embed, 3, 0.35);
        let out = route(&l.tree, l.sci, &physics_q(), "q", &wide, None).await;
        assert_eq!(out.leaves, vec![l.phys, l.chem]);
        // under arts both leaves score 0.0; beam 1 keeps the first child
        let strict = params(RouteMode::Embed, 1, 0.9);
        let out = route(&l.tree, l.arts, &physics_q(), "q", &strict, None).await;
        assert_eq!(out.leaves, vec![l.music]);
    }

    #[tokio::test]
    async fn start_at_a_leaf_returns_it_without_consulting() {
        let l = library();
        let oracle = FixedOracle::new(&[("phys", 0.9)]);
        let out = route(
            &l.tree,
            l.phys,
            &physics_q(),
            "q",
            &params(RouteMode::Laya, 1, 0.9),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![l.phys]);
        assert!(!out.laya_consulted);
        assert_eq!(oracle.calls(), 0);
    }

    #[tokio::test]
    async fn router_without_children_contributes_nothing() {
        let mut t = Tree::default();
        let root = add(&mut t, None, StoreKind::Router, "root", e0());
        let out = route(
            &t,
            root,
            &e0(),
            "q",
            &params(RouteMode::Hybrid, 3, 0.35),
            None,
        )
        .await;
        assert!(out.leaves.is_empty());
        assert_eq!(out.laya_fallback, None);
    }

    #[tokio::test]
    async fn unknown_start_returns_no_leaves() {
        let l = library();
        let out = route(
            &l.tree,
            id(42),
            &physics_q(),
            "q",
            &params(RouteMode::Embed, 3, 0.35),
            None,
        )
        .await;
        assert!(out.leaves.is_empty());
    }

    // ---------- route: laya ----------

    #[tokio::test]
    async fn laya_follows_oracle_picks_over_embed_scores() {
        let l = library();
        let oracle = FixedOracle::new(&[
            ("sci", 0.1),
            ("arts", 0.9),
            ("phys", 0.5),
            ("chem", 0.5),
            ("music", 0.95),
            ("paint", 0.02),
        ]);
        let out = route(
            &l.tree,
            l.root,
            &physics_q(),
            "q",
            &params(RouteMode::Laya, 1, 0.9),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![l.music]);
        assert_eq!(out.mode_used, RouteMode::Laya);
        assert!(out.laya_consulted);
        assert_eq!(out.laya_fallback, None);
        assert_eq!(oracle.calls(), 2);
    }

    #[tokio::test]
    async fn laya_caps_picks_at_twice_beam() {
        let mut t = Tree::default();
        let root = add(&mut t, None, StoreKind::Router, "root", axis(&[(9, 1.0)]));
        let l0 = add(&mut t, Some(root), StoreKind::Leaf, "l0", axis(&[(0, 1.0)]));
        let l1 = add(&mut t, Some(root), StoreKind::Leaf, "l1", axis(&[(1, 1.0)]));
        add(&mut t, Some(root), StoreKind::Leaf, "l2", axis(&[(2, 1.0)]));
        add(&mut t, Some(root), StoreKind::Leaf, "l3", axis(&[(3, 1.0)]));
        let oracle = FixedOracle::new(&[("l0", 0.5), ("l1", 0.5), ("l2", 0.5), ("l3", 0.5)]);
        let out = route(
            &t,
            root,
            &e0(),
            "q",
            &params(RouteMode::Laya, 1, 0.9),
            some(&oracle),
        )
        .await;
        // all four clear 0.15; cap 2·beam = 2; equal probabilities keep embed order
        assert_eq!(out.leaves, vec![l0, l1]);
    }

    #[tokio::test]
    async fn laya_takes_argmax_when_nothing_clears_the_threshold() {
        let l = library();
        let oracle = FixedOracle::new(&[
            ("sci", 0.10),
            ("arts", 0.12),
            ("music", 0.05),
            ("paint", 0.11),
        ]);
        let out = route(
            &l.tree,
            l.root,
            &physics_q(),
            "q",
            &params(RouteMode::Laya, 1, 0.9),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![l.paint]);
        assert_eq!(out.laya_fallback, None);
    }

    #[tokio::test]
    async fn laya_error_falls_back_to_embed_and_stops_consulting() {
        let l = library();
        let oracle = ErrOracle::new("status 401");
        let out = route(
            &l.tree,
            l.root,
            &physics_q(),
            "q",
            &params(RouteMode::Laya, 1, 0.9),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![l.phys]);
        assert!(out.laya_consulted);
        assert_eq!(out.laya_fallback.as_deref(), Some("status 401"));
        assert_eq!(
            oracle.calls(),
            1,
            "tripped after the first failure: one timeout per query at most"
        );
    }

    #[tokio::test]
    async fn laya_without_oracle_is_unconfigured() {
        let l = library();
        let out = route(
            &l.tree,
            l.root,
            &physics_q(),
            "q",
            &params(RouteMode::Laya, 1, 0.9),
            None,
        )
        .await;
        assert_eq!(out.leaves, vec![l.phys]);
        assert!(!out.laya_consulted);
        assert_eq!(out.laya_fallback.as_deref(), Some("unconfigured"));
    }

    #[tokio::test]
    async fn laya_with_one_child_is_option_count_and_not_called() {
        let mut t = Tree::default();
        let root = add(&mut t, None, StoreKind::Router, "root", e0());
        let only = add(&mut t, Some(root), StoreKind::Leaf, "only", e0());
        let oracle = FixedOracle::new(&[("only", 0.9)]);
        let out = route(
            &t,
            root,
            &e0(),
            "q",
            &params(RouteMode::Laya, 1, 0.9),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![only]);
        assert_eq!(out.laya_fallback.as_deref(), Some("option_count"));
        assert!(!out.laya_consulted);
        assert_eq!(oracle.calls(), 0);
    }

    #[tokio::test]
    async fn first_fallback_reason_is_kept() {
        // root has ONE child (option_count), mid has two (oracle error): the first reason wins
        let mut t = Tree::default();
        let root = add(&mut t, None, StoreKind::Router, "root", axis(&[(9, 1.0)]));
        let mid = add(&mut t, Some(root), StoreKind::Router, "mid", e0());
        let a = add(&mut t, Some(mid), StoreKind::Leaf, "a", e0());
        add(&mut t, Some(mid), StoreKind::Leaf, "b", axis(&[(1, 1.0)]));
        let oracle = ErrOracle::new("status 500");
        let out = route(
            &t,
            root,
            &e0(),
            "q",
            &params(RouteMode::Laya, 1, 0.9),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![a]);
        assert_eq!(out.laya_fallback.as_deref(), Some("option_count"));
        assert!(out.laya_consulted);
        assert_eq!(oracle.calls(), 1);
    }

    #[tokio::test]
    async fn laya_answer_naming_no_option_is_no_probabilities() {
        let l = library();
        let oracle = FixedOracle::new(&[]);
        let out = route(
            &l.tree,
            l.root,
            &physics_q(),
            "q",
            &params(RouteMode::Laya, 1, 0.9),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![l.phys]);
        assert_eq!(out.laya_fallback.as_deref(), Some("no_probabilities"));
        assert_eq!(oracle.calls(), 1);
    }

    // ---------- route: hybrid ----------

    #[tokio::test]
    async fn hybrid_unambiguous_never_consults() {
        let l = library();
        let oracle = FixedOracle::new(&[("arts", 0.99), ("music", 0.99)]);
        let out = route(
            &l.tree,
            l.root,
            &physics_q(),
            "q",
            &params(RouteMode::Hybrid, 1, 0.9),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![l.phys]);
        assert_eq!(out.mode_used, RouteMode::Hybrid);
        assert!(!out.laya_consulted);
        assert_eq!(out.laya_fallback, None);
        assert_eq!(oracle.calls(), 0);
    }

    #[tokio::test]
    async fn hybrid_ambiguous_unions_laya_picks_with_embed_top1() {
        let c = close_siblings();
        let p = params(RouteMode::Hybrid, 1, 0.35);
        let oracle = FixedOracle::new(&[("x", 0.01), ("y", 0.01), ("z", 0.9)]);
        let out = route(&c.tree, c.root, &e0(), "q", &p, some(&oracle)).await;
        assert_eq!(out.leaves, vec![c.x, c.z]);
        assert!(out.laya_consulted);
        assert_eq!(out.laya_fallback, None);
        // contrast: embed alone takes top-1 + y (≥ min) under the cap of 2
        let embed = route(
            &c.tree,
            c.root,
            &e0(),
            "q",
            &params(RouteMode::Embed, 1, 0.35),
            None,
        )
        .await;
        assert_eq!(embed.leaves, vec![c.x, c.y]);
    }

    #[tokio::test]
    async fn hybrid_ambiguous_oracle_error_falls_back_to_embed() {
        let c = close_siblings();
        let oracle = ErrOracle::new("timeout");
        let out = route(
            &c.tree,
            c.root,
            &e0(),
            "q",
            &params(RouteMode::Hybrid, 1, 0.35),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![c.x, c.y]);
        assert!(out.laya_consulted);
        assert_eq!(out.laya_fallback.as_deref(), Some("timeout"));
    }

    #[tokio::test]
    async fn hybrid_ambiguous_without_oracle_is_unconfigured() {
        let c = close_siblings();
        let out = route(
            &c.tree,
            c.root,
            &e0(),
            "q",
            &params(RouteMode::Hybrid, 1, 0.35),
            None,
        )
        .await;
        assert_eq!(out.leaves, vec![c.x, c.y]);
        assert!(!out.laya_consulted);
        assert_eq!(out.laya_fallback.as_deref(), Some("unconfigured"));
    }

    #[tokio::test]
    async fn hybrid_union_does_not_duplicate_top1() {
        let c = close_siblings();
        let oracle = FixedOracle::new(&[("x", 0.9), ("y", 0.01), ("z", 0.01)]);
        let out = route(
            &c.tree,
            c.root,
            &e0(),
            "q",
            &params(RouteMode::Hybrid, 1, 0.35),
            some(&oracle),
        )
        .await;
        assert_eq!(out.leaves, vec![c.x]);
    }

    #[tokio::test]
    async fn tied_scores_break_by_name_ascending_not_creation_order() {
        // created zebra, mango, apple (all score 0.0 against e0); name asc must win over id order
        let mut t = Tree::default();
        let root = add(&mut t, None, StoreKind::Router, "root", axis(&[(9, 1.0)]));
        let _zebra = add(
            &mut t,
            Some(root),
            StoreKind::Leaf,
            "zebra",
            axis(&[(1, 1.0)]),
        );
        let mango = add(
            &mut t,
            Some(root),
            StoreKind::Leaf,
            "mango",
            axis(&[(2, 1.0)]),
        );
        let apple = add(
            &mut t,
            Some(root),
            StoreKind::Leaf,
            "apple",
            axis(&[(3, 1.0)]),
        );
        let out = route(
            &t,
            root,
            &e0(),
            "q",
            &params(RouteMode::Embed, 2, 0.9),
            None,
        )
        .await;
        assert_eq!(out.leaves, vec![apple, mango]);
    }

    // ---------- single-flight tree loads ----------

    /// Misses that arrive while a load of the same tenant is in flight wait for it: N concurrent
    /// misses run one load, and all get the tree it cached.
    #[tokio::test]
    async fn concurrent_misses_share_one_load() {
        let tenant = Tenant::new(Uuid::from_u128(11)).unwrap();
        let cache = RoutingCache::default();
        let loads = Arc::new(AtomicUsize::new(0));
        let (release, gate) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started) = tokio::sync::oneshot::channel::<()>();
        let first = tokio::spawn({
            let cache = cache.clone();
            let loads = Arc::clone(&loads);
            async move {
                cache
                    .get_or_load(tenant, 0, async move {
                        loads.fetch_add(1, Ordering::SeqCst);
                        let _ = started_tx.send(());
                        gate.await.unwrap();
                        Ok(vec![row(id(1), None, StoreKind::Router, "first")])
                    })
                    .await
                    .unwrap()
            }
        });
        started.await.unwrap(); // the first load is in flight
        let waiters: Vec<_> = (0..8)
            .map(|_| {
                let cache = cache.clone();
                let loads = Arc::clone(&loads);
                tokio::spawn(async move {
                    cache
                        .get_or_load(tenant, 0, async move {
                            loads.fetch_add(1, Ordering::SeqCst);
                            Ok(vec![row(id(1), None, StoreKind::Router, "again")])
                        })
                        .await
                        .unwrap()
                })
            })
            .collect();
        // let every waiter reach the tenant's load lock; without single-flight each loads now
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(loads.load(Ordering::SeqCst), 1, "waiters do not load");
        release.send(()).unwrap();
        let first = first.await.unwrap();
        assert_eq!(first.nodes[&id(1)].name, "first");
        for w in waiters {
            assert!(
                Arc::ptr_eq(&first, &w.await.unwrap()),
                "served the cached tree"
            );
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(
            cache.inner.read().await.loading.is_empty(),
            "an idle tenant keeps no load lock"
        );
    }

    // ---------- concurrent laya calls per level ----------

    /// Answers from `probs`; a call whose first option is not `sci` (the library's second level)
    /// waits until `barrier` many such calls are in flight at once.
    struct BarrierOracle {
        probs: HashMap<String, f32>,
        barrier: tokio::sync::Barrier,
        calls: AtomicUsize,
    }

    impl ChoiceOracle for BarrierOracle {
        fn choose<'a>(
            &'a self,
            _query: &'a str,
            options: &'a [(String, String)],
        ) -> BoxFut<'a, Result<HashMap<String, f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let out: HashMap<String, f32> = options
                .iter()
                .filter_map(|(n, _)| self.probs.get(n).map(|p| (n.clone(), *p)))
                .collect();
            let wait = options[0].0 != "sci";
            Box::pin(async move {
                if wait {
                    self.barrier.wait().await;
                }
                Ok(out)
            })
        }
    }

    #[tokio::test]
    async fn routers_on_one_level_consult_laya_concurrently() {
        let l = library();
        let oracle = BarrierOracle {
            probs: [
                ("sci", 0.5),
                ("arts", 0.5),
                ("phys", 0.9),
                ("chem", 0.05),
                ("music", 0.9),
                ("paint", 0.05),
            ]
            .iter()
            .map(|(n, p)| (n.to_string(), *p))
            .collect(),
            // sci's and arts's calls (level 2) only finish once both are in flight
            barrier: tokio::sync::Barrier::new(2),
            calls: AtomicUsize::new(0),
        };
        let out = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            route_with_budget(
                &l.tree,
                l.root,
                &physics_q(),
                "q",
                &params(RouteMode::Laya, 2, 0.9),
                some(&oracle),
                std::time::Duration::from_secs(30),
            ),
        )
        .await
        .expect("level 2's two laya calls run at the same time");
        assert_eq!(out.leaves, vec![l.phys, l.music]);
        assert!(out.laya_consulted);
        assert_eq!(out.laya_fallback, None);
        assert_eq!(oracle.calls.load(Ordering::SeqCst), 3);
    }

    /// Fails the library's second-level calls: sci's slowly, arts's at once.
    struct SlowFirstErrOracle {
        calls: AtomicUsize,
    }

    impl ChoiceOracle for SlowFirstErrOracle {
        fn choose<'a>(
            &'a self,
            _query: &'a str,
            options: &'a [(String, String)],
        ) -> BoxFut<'a, Result<HashMap<String, f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let first = options[0].0.clone();
            Box::pin(async move {
                let laya_err = |message: &str| StrataError::Upstream {
                    service: "laya",
                    message: message.to_string(),
                };
                match first.as_str() {
                    "sci" => Ok([("sci".to_string(), 0.5), ("arts".to_string(), 0.5)]
                        .into_iter()
                        .collect()),
                    "phys" => {
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        Err(laya_err("sci down"))
                    }
                    _ => Err(laya_err("arts down")),
                }
            })
        }
    }

    /// Concurrent answers are applied in frontier order: the first router's failure is the
    /// reported reason even when a later router's failure arrives first.
    #[tokio::test]
    async fn concurrent_laya_failures_report_in_frontier_order() {
        let l = library();
        let oracle = SlowFirstErrOracle {
            calls: AtomicUsize::new(0),
        };
        let out = route(
            &l.tree,
            l.root,
            &physics_q(),
            "q",
            &params(RouteMode::Laya, 2, 0.9),
            some(&oracle),
        )
        .await;
        assert_eq!(out.laya_fallback.as_deref(), Some("sci down"));
        assert!(out.laya_consulted);
        assert_eq!(
            out.leaves,
            vec![l.phys, l.chem],
            "both routers fell back to embed"
        );
        assert_eq!(oracle.calls.load(Ordering::SeqCst), 3);
    }
}
