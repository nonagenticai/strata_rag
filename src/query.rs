//! `POST /api/query` (spec §6): embed → route → leaf search → rerank.
//!
//! A query names what it searches in one of three ways: `root_id` (one tree, routed), `root_ids`
//! (several trees, each routed, the selected leaves unioned) or `leaf_ids` (exactly those leaves,
//! no routing, no pruning). Leaves may sit in different embedding models (D83): the question is
//! embedded once per model, each model's leaves are searched with its own vector, and RRF fuses
//! the lists (`search::search_groups_ef`).
//!
//! Platform knowledge (spec 2026-10-20-platform-knowledge-design.md §5-§7): `root_ids` /
//! `leaf_ids` / `root_id` may also name stores of the platform packs granted to the caller
//! (`Readable`); a default query searches the caller's own tree only. Platform leaves embed the
//! question with the platform owner's providers (one group per owner and model), and their hits
//! rank under the caller's own by `crate::precedence` (shadowing, margin, dedup), labelled with
//! `layer` and `pack`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use uuid::Uuid;

use crate::db::platform::PlatformPack;
use crate::db::search::{ChunkFilter, Owners};
use crate::db::stores::StoreKind;
use crate::engine::Engine;
use crate::error::{Result, StrataError};
use crate::expand::{EXPAND_TIMEOUT, ExpansionInfo, expand};
use crate::inference::InferenceHandle;
use crate::ingest::{source_field, valid_topic_key};
use crate::metering::ProviderMeta;
use crate::precedence::{DEFAULT_DEDUP_JACCARD, Layer, PrecedenceOpts};
use crate::routing::{RouteMode, RouteParams, Tree, TreeNode, route_with_budget};
use crate::search::{CANDIDATES, DenseGroup, Precedence, Scores, elapsed_ms, search_groups_ef};
use crate::tenant::Tenant;

/// Targets whose providers are resolved concurrently (each resolution may query the database).
const RESOLVE_CONCURRENCY: usize = 8;

/// Search never returns more than `CANDIDATES` rows, so a larger `top_k` is rejected, not silently capped.
pub const MAX_TOP_K: usize = CANDIDATES;

/// Longest accepted query, in chars (a question, not a document; bounds tokenizer + laya work).
pub const MAX_QUERY_CHARS: usize = 4096;

// The query defaults (spec §6), defined here once: serde, `QueryRequest::new` and callers use these.
pub fn d_top_k() -> usize {
    8
}
pub fn d_beam() -> usize {
    3
}
pub fn d_min_score() -> f32 {
    0.35
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct QueryRequest {
    pub query: String,
    #[serde(default = "d_top_k")]
    pub top_k: usize,
    /// Defaults to `RouteMode::Hybrid` (`RouteMode`'s `#[default]`).
    #[serde(default)]
    pub route_mode: RouteMode,
    #[serde(default = "d_beam")]
    pub beam: usize,
    #[serde(default = "d_min_score")]
    pub min_score: f32,
    /// Route from this store (one tree). At most one of `root_id` / `root_ids` / `leaf_ids`; none
    /// of them = the tenant's only root (several roots: 409 `ambiguous_root`, D90).
    #[serde(default)]
    pub root_id: Option<Uuid>,
    /// Route from each of these stores and search the union of the selected leaves.
    #[serde(default)]
    pub root_ids: Option<Vec<Uuid>>,
    /// Search exactly these leaves: no routing (`route_mode`, `beam` and `min_score` do not apply).
    /// Each must be a leaf the caller may read (`readable_leaves`).
    #[serde(default)]
    pub leaf_ids: Option<Vec<Uuid>>,
    /// Query expansion (HyDE passage + rewrites, one chat-provider call) for the dense search; see
    /// `expand`.
    #[serde(default)]
    pub expand: bool,
    /// Only chunks of documents whose `source_id` is one of these (non-empty list).
    #[serde(default)]
    pub source_ids: Option<Vec<String>>,
    /// Only chunks of documents whose `metadata` contains this JSON object (`@>`).
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    /// Topic keys the caller's knowledge supersedes for this query, on top of the stored
    /// `supersedes` of its documents in the searched leaves: a platform hit whose document
    /// `topic_keys` meet them is dropped (spec platform-knowledge §6).
    #[serde(default)]
    pub supersedes: Option<Vec<String>>,
    /// Subtracted from each platform hit's rerank score, within [0, 1] (default
    /// `STRATA_PLATFORM_PENALTY`).
    #[serde(default)]
    pub platform_penalty: Option<f32>,
    /// Platform hits scoring below this after the penalty are dropped (default: none).
    #[serde(default)]
    pub platform_min_score: Option<f32>,
}

impl QueryRequest {
    /// A request with every parameter at its default.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            top_k: d_top_k(),
            route_mode: RouteMode::default(),
            beam: d_beam(),
            min_score: d_min_score(),
            root_id: None,
            root_ids: None,
            leaf_ids: None,
            expand: false,
            source_ids: None,
            metadata: None,
            supersedes: None,
            platform_penalty: None,
            platform_min_score: None,
        }
    }
}

/// Most `source_ids` one query may filter on.
pub const MAX_FILTER_SOURCE_IDS: usize = 1000;

/// Most stores one query may name in `root_ids` or `leaf_ids`.
pub const MAX_QUERY_STORES: usize = 256;

/// Most topic keys one query may name in `supersedes`.
pub const MAX_QUERY_SUPERSEDES: usize = 256;

#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryResult {
    pub chunk_id: Uuid,
    pub document_id: Uuid,
    /// Store names from the root down to the chunk's leaf. For a platform hit: the pack key, then
    /// the path within the platform owner's tree from the pack's root (`Readable::store_path`).
    pub store_path: Vec<String>,
    /// `tenant` (the caller's own knowledge) or `platform` (a platform pack).
    pub layer: Layer,
    /// The platform hit's pack, `"<key>@<version>"`; absent for a tenant hit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pack: Option<String>,
    /// The chunk's position in its document.
    pub ord: i32,
    pub text: String,
    /// First / last page the chunk spans; `null` when the document has no pages.
    pub page_start: Option<i32>,
    pub page_end: Option<i32>,
    /// The document's `source` label and `title`, for citations.
    pub source: String,
    pub title: Option<String>,
    pub scores: Scores,
    /// The hit's document provenance (see `db::documents::SourceFields`).
    pub source_id: Option<String>,
    pub source_run: Option<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RoutingInfo {
    /// `null` when nothing was routed (`leaf_ids`).
    pub mode_used: Option<RouteMode>,
    pub leaves: Vec<Uuid>,
    pub laya_consulted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub laya_fallback: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Timings {
    /// The question's embedding; with `expand` this stage also covers the overlapped chat call and
    /// the variant embeddings (`expansion.ms` isolates the chat part).
    pub embed: u64,
    pub route: u64,
    pub search: u64,
    pub rerank: u64,
    pub total: u64,
}

/// One embedding model's share of a query: the leaves in that model, how long embedding the
/// question in it and searching its leaves took.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ModelTimings {
    /// `stores.embed_model` of these leaves (`"{model}@{dims}"`).
    pub embed_model: String,
    pub leaves: usize,
    /// The question's embedding in this model (expansion variants not included).
    pub embed: u64,
    /// This model's dense searches (they run concurrently with the other models').
    pub search: u64,
    /// How this model's leaves were searched: `"exact"` (a leaf set of at most
    /// `STRATA_EXACT_SCAN_MAX_CHUNKS` chunks) or `"hnsw"`. Absent when the model had no leaves.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<crate::db::search::DensePath>,
}

/// The provider and model behind one stage of a query or ask.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct StageProvider {
    pub provider: String,
    pub model: String,
    /// The stage was answered from the query-embedding cache: no provider call (embed only;
    /// omitted when false).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub cached: bool,
}

impl From<&ProviderMeta> for StageProvider {
    fn from(m: &ProviderMeta) -> Self {
        StageProvider {
            provider: m.provider.clone(),
            model: m.model.clone(),
            cached: false,
        }
    }
}

/// Per stage, the provider configured for it in this request (absent: not configured, or the
/// stage did not run: `chat` only with `expand` or on `/api/ask`).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct StageProviders {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed: Option<StageProvider>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<StageProvider>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat: Option<StageProvider>,
}

impl StageProviders {
    /// The embed and rerank providers of `inf`.
    pub fn of(inf: &InferenceHandle) -> Self {
        StageProviders {
            embed: inf.embed_provider().map(StageProvider::from),
            rerank: inf.rerank_provider().map(StageProvider::from),
            chat: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QueryResponse {
    pub results: Vec<QueryResult>,
    pub routing: RoutingInfo,
    pub timings_ms: Timings,
    /// Which provider served each stage (`timings_ms` says how long it took). With leaves in
    /// several embedding models, `embed` names the first target store's (see `timings_by_model`).
    pub providers: StageProviders,
    /// Per embedding model searched, in target order (one entry for a single-model query).
    pub timings_by_model: Vec<ModelTimings>,
    /// Present only when the request asked for `expand`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansion: Option<ExpansionInfo>,
}

fn validate(req: &QueryRequest) -> Result<()> {
    if req.query.trim().is_empty() {
        return Err(StrataError::InvalidInput(
            "query must not be empty".to_string(),
        ));
    }
    let chars = req.query.chars().count();
    if chars > MAX_QUERY_CHARS {
        return Err(StrataError::InvalidInput(format!(
            "query must be at most {MAX_QUERY_CHARS} characters, got {chars}"
        )));
    }
    if !(1..=MAX_TOP_K).contains(&req.top_k) {
        return Err(StrataError::InvalidInput(format!(
            "top_k must be between 1 and {MAX_TOP_K}, got {}",
            req.top_k
        )));
    }
    if req.beam == 0 {
        return Err(StrataError::InvalidInput(
            "beam must be at least 1".to_string(),
        ));
    }
    if !req.min_score.is_finite() {
        return Err(StrataError::InvalidInput(
            "min_score must be a finite number".to_string(),
        ));
    }
    if let Some(p) = req.platform_penalty
        && !(0.0..=1.0).contains(&p)
    {
        return Err(StrataError::InvalidInput(format!(
            "platform_penalty must be between 0 and 1, got {p}"
        )));
    }
    if let Some(m) = req.platform_min_score
        && !m.is_finite()
    {
        return Err(StrataError::InvalidInput(
            "platform_min_score must be a finite number".to_string(),
        ));
    }
    if let Some(keys) = &req.supersedes {
        if keys.len() > MAX_QUERY_SUPERSEDES {
            return Err(StrataError::InvalidInput(format!(
                "supersedes may name at most {MAX_QUERY_SUPERSEDES} topic keys, got {}",
                keys.len()
            )));
        }
        if let Some(bad) = keys.iter().find(|k| !valid_topic_key(k)) {
            return Err(StrataError::InvalidInput(format!(
                "supersedes: `{bad}` is not a topic key (1-{} chars of [a-z0-9_./-])",
                crate::ingest::MAX_TOPIC_KEY_CHARS
            )));
        }
    }
    let named = [
        req.root_id.is_some(),
        req.root_ids.is_some(),
        req.leaf_ids.is_some(),
    ]
    .into_iter()
    .filter(|n| *n)
    .count();
    if named > 1 {
        return Err(StrataError::InvalidInput(
            "name at most one of root_id, root_ids and leaf_ids".to_string(),
        ));
    }
    for (field, ids) in [("root_ids", &req.root_ids), ("leaf_ids", &req.leaf_ids)] {
        match ids {
            Some(ids) if ids.is_empty() => {
                return Err(StrataError::InvalidInput(format!(
                    "{field} must name at least one store"
                )));
            }
            Some(ids) if ids.len() > MAX_QUERY_STORES => {
                return Err(StrataError::InvalidInput(format!(
                    "{field} may name at most {MAX_QUERY_STORES} stores, got {}",
                    ids.len()
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

/// `ids` without repeats, first occurrence kept.
fn dedup(ids: &[Uuid]) -> Vec<Uuid> {
    let mut seen = HashSet::new();
    ids.iter().copied().filter(|id| seen.insert(*id)).collect()
}

/// The request's document filters, validated: `source_ids` a non-empty list (at most
/// `MAX_FILTER_SOURCE_IDS`) of valid source ids, `metadata` a JSON object.
fn chunk_filter(req: &QueryRequest) -> Result<ChunkFilter> {
    let source_ids = match &req.source_ids {
        None => None,
        Some(ids) if ids.is_empty() => {
            return Err(StrataError::InvalidInput(
                "source_ids must name at least one source (omit it to search every source)"
                    .to_string(),
            ));
        }
        Some(ids) if ids.len() > MAX_FILTER_SOURCE_IDS => {
            return Err(StrataError::InvalidInput(format!(
                "source_ids may name at most {MAX_FILTER_SOURCE_IDS} sources, got {}",
                ids.len()
            )));
        }
        Some(ids) => Some(
            ids.iter()
                .map(|id| {
                    source_field("source_ids[]", Some(id.clone()))?.ok_or_else(|| {
                        StrataError::InvalidInput("source_ids[] must not be empty".to_string())
                    })
                })
                .collect::<Result<Vec<String>>>()?,
        ),
    };
    let metadata = match &req.metadata {
        None => None,
        Some(v @ serde_json::Value::Object(_)) => Some(v.clone()),
        Some(_) => {
            return Err(StrataError::InvalidInput(
                "metadata must be a JSON object".to_string(),
            ));
        }
    };
    Ok(ChunkFilter {
        source_ids,
        metadata,
    })
}

/// The store a query starts from, in `tree`: `root_id` (404 when it is not in the tree), else the
/// tenant's only root. More than one root and no `root_id` is 409 `ambiguous_root` (D90): each
/// root is a tree of its own (D83), so which one to search is the caller's choice. `None`: the
/// tenant has no store at all.
pub(crate) fn start_store(
    tree: &crate::routing::Tree,
    root_id: Option<Uuid>,
) -> Result<Option<Uuid>> {
    match (root_id, tree.root) {
        (Some(id), _) if tree.nodes.contains_key(&id) => Ok(Some(id)),
        (Some(id), _) => Err(StrataError::NotFound(format!("store {id}"))),
        (None, Some(_)) if tree.root_count > 1 => Err(ambiguous_root(tree.root_count)),
        (None, root) => Ok(root),
    }
}

/// 409 `ambiguous_root` for a tenant holding `roots` root stores.
pub(crate) fn ambiguous_root(roots: usize) -> StrataError {
    StrataError::AmbiguousRoot(format!(
        "this tenant has {roots} root stores; name what to search with root_id, root_ids or leaf_ids"
    ))
}

/// The stores one query may read (spec platform-knowledge §5): THE predicate for "which stores
/// may this caller search". Always the caller's own tree; with platform knowledge on and a request
/// naming a store outside that tree, also the subtrees of the packs `db::platform::readable_packs`
/// grants the caller (published, not excluded; retired packs stay readable for their pins). This
/// is the only place cross-owner reads are granted: search SQL then filters on `Owners` AND this
/// authorised set. Own and platform stores never overlap (a store has one tenant). The platform
/// owner reads its packs as its own tree (`tenant` layer): it curates them, nothing outranks them.
pub(crate) struct Readable {
    own: Arc<Tree>,
    platform: Option<PlatformStores>,
}

/// The platform stores granted to one query.
struct PlatformStores {
    owner: Tenant,
    /// The platform owner's whole tree (routing within a pack never leaves the pack's subtree).
    tree: Arc<Tree>,
    packs: Vec<PlatformPack>,
    /// Every store in a granted pack's subtree → its pack, as an index into `packs`.
    pack_of: HashMap<Uuid, usize>,
}

impl PlatformStores {
    fn new(owner: Tenant, tree: Arc<Tree>, packs: Vec<PlatformPack>) -> Self {
        let mut pack_of = HashMap::new();
        for (i, pack) in packs.iter().enumerate() {
            let mut stack = vec![pack.root_id];
            while let Some(id) = stack.pop() {
                // a root missing from the tree (deleted meanwhile) grants nothing; the insert
                // doubles as the cycle guard
                if !tree.nodes.contains_key(&id) || pack_of.insert(id, i).is_some() {
                    continue;
                }
                if let Some(kids) = tree.children.get(&id) {
                    stack.extend(kids.iter().copied());
                }
            }
        }
        PlatformStores {
            owner,
            tree,
            packs,
            pack_of,
        }
    }
}

impl Readable {
    /// The caller's own tree only (platform knowledge off, or nothing outside it named).
    #[cfg(test)]
    fn own(tree: Arc<Tree>) -> Self {
        Readable {
            own: tree,
            platform: None,
        }
    }

    /// The layer of `id` when the caller may read it; `None` = not readable (404).
    fn layer(&self, id: Uuid) -> Option<Layer> {
        if self.own.nodes.contains_key(&id) {
            return Some(Layer::Tenant);
        }
        self.platform
            .as_ref()
            .filter(|p| p.pack_of.contains_key(&id))
            .map(|_| Layer::Platform)
    }

    fn node(&self, id: Uuid) -> Option<&TreeNode> {
        match self.layer(id)? {
            Layer::Tenant => self.own.nodes.get(&id),
            Layer::Platform => self.platform.as_ref()?.tree.nodes.get(&id),
        }
    }

    /// The tree a store of `layer` routes in.
    fn tree(&self, layer: Layer) -> &Tree {
        match (layer, &self.platform) {
            (Layer::Platform, Some(p)) => &p.tree,
            _ => &self.own,
        }
    }

    /// The tree a readable store `id` sits in (the caller's for an unknown id).
    fn tree_of(&self, id: Uuid) -> &Tree {
        self.tree(self.layer(id).unwrap_or(Layer::Tenant))
    }

    /// The platform owner when platform stores were granted to this query.
    fn platform_owner(&self) -> Option<Tenant> {
        self.platform.as_ref().map(|p| p.owner)
    }

    /// Whose providers embed for `id`: the platform owner for a platform store, else the caller.
    fn provider_owner(&self, id: Uuid, caller: Tenant) -> Tenant {
        match (self.layer(id), &self.platform) {
            (Some(Layer::Platform), Some(p)) => p.owner,
            _ => caller,
        }
    }

    /// The pack a platform store belongs to.
    fn pack(&self, id: Uuid) -> Option<&PlatformPack> {
        let p = self.platform.as_ref()?;
        p.pack_of.get(&id).map(|i| &p.packs[*i])
    }

    /// `QueryResult::store_path`: the caller's tree path for its own store; for a platform store,
    /// the pack key, then the path within the platform tree from the pack's root (the key names
    /// the pack in a citation, the root's own name is the curator's).
    fn store_path(&self, id: Uuid) -> Vec<String> {
        match self.pack(id) {
            Some(pack) => {
                let mut path = vec![pack.key.clone()];
                path.extend(self.tree(Layer::Platform).path(id));
                path
            }
            None => self.own.path(id),
        }
    }
}

/// The leaves among `ids` the caller may read (`Readable`), deduplicated, in request order; any
/// other id fails the whole request (never silently dropped). Another tenant's store, a store of
/// a pack the caller is not granted (draft, excluded) or an unknown id is 404, as everywhere (it
/// does not exist for the caller); a router is 400.
pub(crate) fn readable_leaves(r: &Readable, ids: &[Uuid]) -> Result<Vec<Uuid>> {
    let ids = dedup(ids);
    for id in &ids {
        match r.node(*id) {
            None => return Err(StrataError::NotFound(format!("store {id}"))),
            Some(node) if node.kind != StoreKind::Leaf => {
                return Err(StrataError::InvalidInput(format!(
                    "leaf_ids: store {id} is a router, not a leaf"
                )));
            }
            Some(_) => {}
        }
    }
    Ok(ids)
}

/// What one query searches, resolved against the caller's tree.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Targets {
    /// Route from each store (`root_id` / `root_ids` / the only root).
    Roots(Vec<Uuid>),
    /// Search exactly these leaves (`leaf_ids`).
    Leaves(Vec<Uuid>),
}

impl Targets {
    fn stores(&self) -> &[Uuid] {
        match self {
            Targets::Roots(ids) | Targets::Leaves(ids) => ids.as_slice(),
        }
    }
}

/// The request's targets among the readable stores (`req` validated). `None`: no store named and
/// the tenant has none (an empty tree answers 200 with no results). Without a named store only
/// the caller's own tree is searched.
fn targets(r: &Readable, req: &QueryRequest) -> Result<Option<Targets>> {
    if let Some(ids) = &req.leaf_ids {
        return Ok(Some(Targets::Leaves(readable_leaves(r, ids)?)));
    }
    if let Some(ids) = &req.root_ids {
        let ids = dedup(ids);
        if let Some(id) = ids.iter().find(|id| r.layer(**id).is_none()) {
            return Err(StrataError::NotFound(format!("store {id}")));
        }
        return Ok(Some(Targets::Roots(ids)));
    }
    if let Some(id) = req.root_id
        && r.layer(id) == Some(Layer::Platform)
    {
        return Ok(Some(Targets::Roots(vec![id])));
    }
    Ok(start_store(&r.own, req.root_id)?.map(|s| Targets::Roots(vec![s])))
}

/// The target stores that share one provider owner and embedding model, and the providers that
/// embed for them (the platform owner's for platform stores, spec platform-knowledge §7).
struct ModelGroup {
    model: String,
    owner: Tenant,
    inf: InferenceHandle,
}

impl Engine {
    /// The stores `tenant` may read for `req` (see [`Readable`]): its own tree, plus the granted
    /// platform packs when platform knowledge is on and `req` names a store outside its tree (a
    /// query naming only its own stores costs no platform lookup).
    pub(crate) async fn readable(&self, tenant: Tenant, req: &QueryRequest) -> Result<Readable> {
        let own = self.cache.get(&self.pool, tenant).await?;
        let names_foreign = req
            .root_id
            .iter()
            .chain(req.root_ids.iter().flatten())
            .chain(req.leaf_ids.iter().flatten())
            .any(|id| !own.nodes.contains_key(id));
        let platform = match self.cfg.platform_owner {
            Some(owner) if owner != tenant && names_foreign => {
                // the platform owner's tree is cached and invalidated like any tenant's
                let tree = self.cache.get(&self.pool, owner).await?;
                let packs = crate::db::platform::readable_packs(&self.pool, owner, tenant).await?;
                Some(PlatformStores::new(owner, tree, packs))
            }
            _ => None,
        };
        Ok(Readable { own, platform })
    }

    /// The providers that embed the question for target `store`: the caller's (its nearest store
    /// binding, else its own), or for a platform store the platform owner's, metered to the caller
    /// and funded by the platform (`Engine::platform_inference_for`).
    async fn target_inference(
        &self,
        tenant: Tenant,
        r: &Readable,
        store: Uuid,
    ) -> Result<InferenceHandle> {
        match (r.layer(store), r.platform_owner()) {
            (Some(Layer::Platform), Some(owner)) => {
                self.platform_inference_for(tenant, owner, store).await
            }
            _ => self.inference_for(tenant, Some(store)).await,
        }
    }

    /// The store whose providers serve a query's single-provider stages (rerank; chat for `ask`
    /// and `expand`): the first of the caller's own stores it names, else the tenant's only root
    /// (see [`start_store`]); `None` (the tenant's providers) when it names platform stores only:
    /// rerank and chat are always the caller's. Fails as the query would on an unknown, forbidden
    /// or ambiguous target.
    pub async fn query_start(&self, tenant: Tenant, req: &QueryRequest) -> Result<Option<Uuid>> {
        validate(req)?;
        let r = self.readable(tenant, req).await?;
        Ok(targets(&r, req)?.and_then(|t| {
            t.stores()
                .iter()
                .copied()
                .find(|s| r.layer(*s) == Some(Layer::Tenant))
        }))
    }

    /// Searches what `tenant` may read (`Readable`): from `root_id` / `root_ids` / `leaf_ids`,
    /// which must be its stores or stores of platform packs granted to it (anything else is 404),
    /// or from its only root (several roots: 409 `ambiguous_root`).
    pub async fn query(&self, tenant: Tenant, req: &QueryRequest) -> Result<QueryResponse> {
        let t_total = Instant::now();
        validate(req)?;
        let filter = chunk_filter(req)?;
        let text = req.query.trim();

        let r = self.readable(tenant, req).await?;
        let targets = match targets(&r, req)? {
            Some(targets) => targets,
            // empty tree: 200 with no results (Review Focus #3)
            None => {
                return Ok(QueryResponse {
                    results: Vec::new(),
                    routing: RoutingInfo {
                        mode_used: Some(req.route_mode),
                        leaves: Vec::new(),
                        laya_consulted: false,
                        laya_fallback: None,
                    },
                    timings_ms: Timings {
                        total: elapsed_ms(t_total),
                        ..Timings::default()
                    },
                    providers: StageProviders::default(),
                    timings_by_model: Vec::new(),
                    expansion: None,
                });
            }
        };
        let stores = targets.stores();

        // Each target's providers (its nearest store binding, else the tenant's; a platform store's
        // through the platform owner's lineage) embed the query for it; a store embedded by
        // another model than its provider would be compared in the wrong vector space. Targets
        // sharing a provider owner and a model share one embedding of the question.
        // Resolution can hit the database, so at most RESOLVE_CONCURRENCY run at once (a 256-leaf
        // request must not drain the pool); chunks keep the targets' order. (Not a `buffered`
        // stream: its closure makes this future fail the higher-ranked `Send` check of callers
        // that spawn it.)
        let mut handles: Vec<InferenceHandle> = Vec::with_capacity(stores.len());
        for chunk in stores.chunks(RESOLVE_CONCURRENCY) {
            handles.extend(
                futures_util::future::try_join_all(
                    chunk.iter().map(|s| self.target_inference(tenant, &r, *s)),
                )
                .await?,
            );
        }
        let mut groups: Vec<ModelGroup> = Vec::new();
        let mut group_of: Vec<usize> = Vec::with_capacity(stores.len());
        for (store, inf) in stores.iter().zip(handles) {
            let model = inf.embed_model_id()?.to_string();
            r.tree_of(*store).check_embed_model(*store, &model)?;
            let owner = r.provider_owner(*store, tenant);
            let g = match groups
                .iter()
                .position(|g| g.owner == owner && g.model == model)
            {
                Some(g) => g,
                None => {
                    groups.push(ModelGroup { model, owner, inf });
                    groups.len() - 1
                }
            };
            group_of.push(g);
        }
        // the first target's providers rerank and expand (groups[0] holds target 0) when it is
        // the caller's own; rerank and chat are never the platform's, so with a platform store
        // first, the caller's first own target's (or its own) providers serve them
        let inf = if r.layer(stores[0]) == Some(Layer::Platform) {
            let first_own = stores
                .iter()
                .copied()
                .find(|s| r.layer(*s) == Some(Layer::Tenant));
            self.inference_for(tenant, first_own).await?
        } else {
            groups[0].inf.clone()
        };
        // expansion needs a chat provider: none configured (or an unusable registration) is the
        // caller's error, not a silent degrade (a configured provider that fails or is slow still
        // degrades, below)
        if req.expand {
            inf.require_chat()?;
        }
        let mut providers = StageProviders::of(&inf);
        // the embed stage names the first target's provider (the platform's for a platform target)
        providers.embed = groups[0].inf.embed_provider().map(StageProvider::from);
        if req.expand {
            providers.chat = inf.chat_provider().map(StageProvider::from);
        }

        // The chat call overlaps the question's embeddings (one per model, concurrently); its
        // variants are embedded after, in every model.
        let t = Instant::now();
        let (embedded, expansion) = tokio::join!(
            futures_util::future::try_join_all(groups.iter().map(|g| async move {
                let t = Instant::now();
                let (v, cached) = self
                    .query_cache
                    .embed_one_for(tenant, g.owner, &g.inf, text)
                    .await?;
                Ok::<_, StrataError>((v, cached, elapsed_ms(t)))
            })),
            async {
                if !req.expand {
                    return None;
                }
                let t = Instant::now();
                // bounded: a slow chat provider degrades the query instead of holding it
                let outcome = match tokio::time::timeout(EXPAND_TIMEOUT, expand(&inf, text)).await {
                    Ok(outcome) => outcome,
                    Err(_) => Err(StrataError::GenerationUnavailable(format!(
                        "expansion timed out after {} s",
                        EXPAND_TIMEOUT.as_secs()
                    ))),
                };
                Some((outcome, elapsed_ms(t)))
            }
        );
        let embedded = embedded?;
        if let Some(p) = providers.embed.as_mut() {
            p.cached = embedded.iter().all(|(_, cached, _)| *cached);
        }
        let embed_ms: Vec<u64> = embedded.iter().map(|(_, _, ms)| *ms).collect();
        let mut q_vecs: Vec<Vec<Vec<f32>>> =
            embedded.into_iter().map(|(v, _, _)| vec![v]).collect();
        let expansion = match expansion {
            None => None,
            Some((Ok(e), ms)) => {
                let variants = futures_util::future::try_join_all(groups.iter().map(|g| {
                    self.query_cache
                        .embed_queries_for(tenant, g.owner, &g.inf, e.variants())
                }))
                .await?;
                for (vecs, more) in q_vecs.iter_mut().zip(variants) {
                    vecs.extend(more);
                }
                Some(ExpansionInfo {
                    hyde_passage: Some(e.hyde_passage),
                    rewrites: e.rewrites,
                    fallback: None,
                    ms,
                })
            }
            // degraded, not failed: the question alone still answers (same contract as laya_fallback)
            Some((Err(err), ms)) => {
                tracing::warn!(error = %err, "query expansion failed; searching with the question alone");
                Some(ExpansionInfo {
                    hyde_passage: None,
                    rewrites: Vec::new(),
                    fallback: Some(err.to_string()),
                    ms,
                })
            }
        };
        let embed = elapsed_ms(t);

        // The leaves to search, per model group: routed under each root (concurrently, each with
        // its own model's vector), or exactly the named leaves. A leaf reached twice counts once.
        let t = Instant::now();
        let mut group_leaves: Vec<Vec<Uuid>> = vec![Vec::new(); groups.len()];
        let mut leaves: Vec<Uuid> = Vec::new();
        let mut seen: HashSet<Uuid> = HashSet::new();
        let mut routing = RoutingInfo {
            mode_used: None,
            leaves: Vec::new(),
            laya_consulted: false,
            laya_fallback: None,
        };
        match &targets {
            Targets::Leaves(ids) => {
                for (i, leaf) in ids.iter().enumerate() {
                    if seen.insert(*leaf) {
                        group_leaves[group_of[i]].push(*leaf);
                        leaves.push(*leaf);
                    }
                }
            }
            Targets::Roots(roots) => {
                let params = RouteParams {
                    mode: req.route_mode,
                    beam: req.beam,
                    min_score: req.min_score,
                };
                let budget = std::time::Duration::from_millis(self.cfg.laya_query_budget_ms);
                let r_ref: &Readable = &r;
                let outcomes =
                    futures_util::future::join_all(roots.iter().enumerate().map(|(i, root)| {
                        // a platform root routes in the platform tree, below the root only: it
                        // never leaves its pack
                        route_with_budget(
                            r_ref.tree_of(*root),
                            *root,
                            &q_vecs[group_of[i]][0],
                            text,
                            &params,
                            self.laya.as_deref(),
                            budget,
                        )
                    }))
                    .await;
                routing.mode_used = Some(req.route_mode);
                for (i, outcome) in outcomes.into_iter().enumerate() {
                    routing.laya_consulted |= outcome.laya_consulted;
                    if routing.laya_fallback.is_none() {
                        routing.laya_fallback = outcome.laya_fallback;
                    }
                    for leaf in outcome.leaves {
                        if seen.insert(leaf) {
                            group_leaves[group_of[i]].push(leaf);
                            leaves.push(leaf);
                        }
                    }
                }
            }
        }
        let route_ms = elapsed_ms(t);

        // Precedence (spec platform-knowledge §6), only when platform leaves are searched: the
        // shadow set is the stored `supersedes` of the caller's documents in its searched leaves
        // plus the request's.
        let owners = Owners {
            caller: tenant,
            platform: r.platform_owner(),
        };
        let precedence = if leaves.iter().any(|l| r.layer(*l) == Some(Layer::Platform)) {
            let own_leaves: Vec<Uuid> = leaves
                .iter()
                .copied()
                .filter(|l| r.layer(*l) == Some(Layer::Tenant))
                .collect();
            let mut shadow: HashSet<String> =
                crate::db::documents::shadow_keys(&self.pool, tenant, &own_leaves)
                    .await?
                    .into_iter()
                    .collect();
            shadow.extend(req.supersedes.iter().flatten().cloned());
            Some(Precedence {
                shadow,
                opts: PrecedenceOpts {
                    platform_penalty: req.platform_penalty.unwrap_or(self.cfg.platform_penalty),
                    platform_min_score: req.platform_min_score,
                    dedup_jaccard: DEFAULT_DEDUP_JACCARD,
                },
            })
        } else {
            None
        };

        let dense_groups: Vec<DenseGroup<'_>> = group_leaves
            .iter()
            .zip(&q_vecs)
            .map(|(leaves, q_vecs)| DenseGroup {
                leaves: leaves.as_slice(),
                q_vecs: q_vecs.as_slice(),
            })
            .collect();
        let (hits, st, searches) = search_groups_ef(
            &self.pool,
            owners,
            &inf,
            &dense_groups,
            text,
            req.top_k,
            self.cfg.rerank_candidates,
            &filter,
            self.cfg.hnsw_ef_search,
            self.cfg.exact_scan_max_chunks,
            precedence.as_ref(),
        )
        .await?;
        // search_groups_ef skips groups with no leaves: its timings follow the non-empty ones
        let mut searches = searches.into_iter();
        let timings_by_model = groups
            .iter()
            .enumerate()
            .map(|(g, group)| {
                let searched = if group_leaves[g].is_empty() {
                    None
                } else {
                    searches.next()
                };
                ModelTimings {
                    embed_model: group.model.clone(),
                    leaves: group_leaves[g].len(),
                    embed: embed_ms[g],
                    search: searched.map_or(0, |s| s.ms),
                    path: searched.map(|s| s.path),
                }
            })
            .collect();
        let results = hits
            .into_iter()
            .map(|h| QueryResult {
                chunk_id: h.chunk.chunk_id,
                document_id: h.chunk.document_id,
                store_path: r.store_path(h.chunk.store_id),
                layer: h.layer,
                pack: (h.layer == Layer::Platform)
                    .then(|| r.pack(h.chunk.store_id))
                    .flatten()
                    .map(|p| format!("{}@{}", p.key, p.version)),
                ord: h.chunk.ord,
                text: h.chunk.text,
                page_start: h.chunk.page_start,
                page_end: h.chunk.page_end,
                source: h.chunk.source,
                title: h.chunk.title,
                scores: h.scores,
                source_id: h.chunk.source_id,
                source_run: h.chunk.source_run,
                metadata: h.chunk.metadata,
            })
            .collect();
        routing.leaves = leaves;

        Ok(QueryResponse {
            results,
            routing,
            timings_ms: Timings {
                embed,
                route: route_ms,
                search: st.search,
                rerank: st.rerank,
                total: elapsed_ms(t_total),
            },
            providers,
            timings_by_model,
            expansion,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::RouteMode;

    #[test]
    fn query_request_defaults() {
        let r: QueryRequest = serde_json::from_str(r#"{"query":"x"}"#).unwrap();
        assert_eq!(r.top_k, 8);
        assert_eq!(r.beam, 3);
        assert!((r.min_score - 0.35).abs() < 1e-6);
        assert_eq!(r.route_mode, RouteMode::Hybrid);
        assert_eq!(r.root_id, None);
        assert!(!r.expand);
        assert!(!QueryRequest::new("x").expand);
        let n = QueryRequest::new("x");
        assert_eq!((n.top_k, n.beam, n.route_mode), (8, 3, RouteMode::Hybrid));
        assert!((n.min_score - 0.35).abs() < 1e-6);
    }

    #[test]
    fn filters_parse_and_validate() {
        let r: QueryRequest = serde_json::from_str(
            r#"{"query":"x","source_ids":["a","b"],"metadata":{"lang":"en"}}"#,
        )
        .unwrap();
        let f = chunk_filter(&r).unwrap();
        assert_eq!(f.source_ids, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(f.metadata, Some(serde_json::json!({"lang": "en"})));
        assert!(chunk_filter(&QueryRequest::new("x")).unwrap().is_empty());
        for bad in [
            r#"{"query":"x","source_ids":[]}"#,
            r#"{"query":"x","source_ids":[" "]}"#,
            r#"{"query":"x","metadata":[1]}"#,
            r#"{"query":"x","metadata":"a"}"#,
        ] {
            let r: QueryRequest = serde_json::from_str(bad).unwrap();
            assert!(
                matches!(chunk_filter(&r), Err(StrataError::InvalidInput(_))),
                "{bad}"
            );
        }
    }

    #[test]
    fn query_length_is_capped_in_chars() {
        let at_cap = QueryRequest::new("é".repeat(MAX_QUERY_CHARS));
        assert!(
            validate(&at_cap).is_ok(),
            "4096 chars (8192 bytes) is allowed"
        );
        let over = QueryRequest::new("é".repeat(MAX_QUERY_CHARS + 1));
        assert!(matches!(validate(&over), Err(StrataError::InvalidInput(_))));
    }

    #[test]
    fn at_most_one_target_kind_and_no_empty_lists() {
        let id = Uuid::now_v7();
        let ok: QueryRequest =
            serde_json::from_value(serde_json::json!({"query": "x", "leaf_ids": [id, id]}))
                .unwrap();
        assert!(validate(&ok).is_ok());
        for bad in [
            serde_json::json!({"query": "x", "root_id": id, "leaf_ids": [id]}),
            serde_json::json!({"query": "x", "root_id": id, "root_ids": [id]}),
            serde_json::json!({"query": "x", "root_ids": [id], "leaf_ids": [id]}),
            serde_json::json!({"query": "x", "leaf_ids": []}),
            serde_json::json!({"query": "x", "root_ids": []}),
            serde_json::json!({"query": "x", "leaf_ids": vec![id; MAX_QUERY_STORES + 1]}),
        ] {
            let r: QueryRequest = serde_json::from_value(bad.clone()).unwrap();
            assert!(
                matches!(validate(&r), Err(StrataError::InvalidInput(_))),
                "{bad}"
            );
        }
    }

    fn node(id: Uuid, parent_id: Option<Uuid>, kind: StoreKind) -> crate::routing::TreeNode {
        crate::routing::TreeNode {
            id,
            parent_id,
            kind,
            name: id.to_string(),
            description: String::new(),
            profile_vec: Vec::new(),
            embed_model: "m@4".to_string(),
        }
    }

    /// A router `root` with one leaf `leaf`.
    fn small_tree(root: Uuid, leaf: Uuid) -> Tree {
        let mut tree = Tree::default();
        tree.nodes.insert(root, node(root, None, StoreKind::Router));
        tree.nodes
            .insert(leaf, node(leaf, Some(root), StoreKind::Leaf));
        tree.children.insert(root, vec![leaf]);
        tree.root = Some(root);
        tree.root_count = 1;
        tree
    }

    #[test]
    fn readable_leaves_admits_own_leaves_only() {
        let (root, leaf, other) = (Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3));
        let r = Readable::own(Arc::new(small_tree(root, leaf)));
        assert_eq!(readable_leaves(&r, &[leaf, leaf]).unwrap(), vec![leaf]);
        assert!(matches!(
            readable_leaves(&r, &[leaf, other]),
            Err(StrataError::NotFound(_))
        ));
        assert!(matches!(
            readable_leaves(&r, &[root]),
            Err(StrataError::InvalidInput(_))
        ));
    }

    fn pack(key: &str, version: i32, root_id: Uuid) -> PlatformPack {
        PlatformPack {
            key: key.to_string(),
            version,
            root_id,
            published_at: None,
            retired_at: None,
        }
    }

    #[test]
    fn readable_admits_granted_pack_subtrees_only() {
        let id = Uuid::from_u128;
        let (own_root, own_leaf) = (id(1), id(2));
        // the platform owner's tree: a granted pack (10 → 11) and a store outside every granted
        // pack (20 → 21: a draft or excluded pack's root, not in `packs`)
        let mut platform = small_tree(id(10), id(11));
        platform
            .nodes
            .insert(id(20), node(id(20), None, StoreKind::Router));
        platform
            .nodes
            .insert(id(21), node(id(21), Some(id(20)), StoreKind::Leaf));
        platform.children.insert(id(20), vec![id(21)]);
        let owner = Tenant::new(id(99)).unwrap();
        let r = Readable {
            own: Arc::new(small_tree(own_root, own_leaf)),
            platform: Some(PlatformStores::new(
                owner,
                Arc::new(platform),
                vec![pack("industry/retail", 2, id(10))],
            )),
        };
        assert_eq!(
            readable_leaves(&r, &[own_leaf, id(11)]).unwrap(),
            vec![own_leaf, id(11)]
        );
        assert!(matches!(
            readable_leaves(&r, &[id(21)]),
            Err(StrataError::NotFound(_))
        ));
        assert_eq!(r.layer(own_leaf), Some(Layer::Tenant));
        assert_eq!(r.layer(id(10)), Some(Layer::Platform));
        assert_eq!(r.layer(id(20)), None);
        assert_eq!(r.provider_owner(id(11), Tenant::new(id(5)).unwrap()), owner);
        assert_eq!(r.pack(id(11)).map(|p| p.version), Some(2));
        assert_eq!(r.pack(own_leaf), None);
        assert_eq!(
            r.store_path(id(11)),
            [
                "industry/retail".to_string(),
                id(10).to_string(),
                id(11).to_string()
            ]
        );
        assert_eq!(
            r.store_path(own_leaf),
            [own_root.to_string(), own_leaf.to_string()]
        );
        // a platform root names a target; an ungranted one is 404
        let req = |root_id: Uuid| QueryRequest {
            root_id: Some(root_id),
            ..QueryRequest::new("x")
        };
        assert_eq!(
            targets(&r, &req(id(10))).unwrap(),
            Some(Targets::Roots(vec![id(10)]))
        );
        assert!(matches!(
            targets(&r, &req(id(20))),
            Err(StrataError::NotFound(_))
        ));
        // no store named: the caller's own tree only
        assert_eq!(
            targets(&r, &QueryRequest::new("x")).unwrap(),
            Some(Targets::Roots(vec![own_root]))
        );
    }

    #[test]
    fn platform_query_fields_validate() {
        let ok: QueryRequest = serde_json::from_value(serde_json::json!({
            "query": "x", "supersedes": ["hr/leave"], "platform_penalty": 0.0,
            "platform_min_score": -2.5
        }))
        .unwrap();
        assert!(validate(&ok).is_ok());
        for bad in [
            serde_json::json!({"query": "x", "platform_penalty": 1.5}),
            serde_json::json!({"query": "x", "platform_penalty": -0.1}),
            serde_json::json!({"query": "x", "supersedes": ["Not A Key"]}),
            serde_json::json!({"query": "x", "supersedes": vec!["k"; MAX_QUERY_SUPERSEDES + 1]}),
        ] {
            let r: QueryRequest = serde_json::from_value(bad.clone()).unwrap();
            assert!(
                matches!(validate(&r), Err(StrataError::InvalidInput(_))),
                "{bad}"
            );
        }
    }

    #[test]
    fn routing_info_omits_absent_fallback() {
        let info = RoutingInfo {
            mode_used: Some(RouteMode::Embed),
            leaves: vec![],
            laya_consulted: false,
            laya_fallback: None,
        };
        let v = serde_json::to_value(&info).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"mode_used": "embed", "leaves": [], "laya_consulted": false})
        );
    }
}
