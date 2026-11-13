//! The shared engine used by the API and the CLI: store operations and health.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use uuid::Uuid;

use crate::clients::laya::LayaClient;
use crate::clients::provider::error_class;
use crate::config::{Config, redact_url};
use crate::db::stores::{self, NewStore, StoreKind, StoreRow};
use crate::error::{Result, StrataError};
use crate::inference::InferenceHandle;
use crate::metering::{Attribution, Metering, current_caller};
use crate::providers::{KeyCipher, LaneLimits, Mode, Providers};
use crate::routing::{ChoiceOracle, RoutingCache};
use crate::tenant::Tenant;
use crate::text::{HfTokenCounter, TokenCounter, bge_m3_tokenizer_path};
use crate::vecmath::DIM;

/// Upper bound for each `/health` component check.
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(2);

/// `false` when `check` fails or exceeds `HEALTH_CHECK_TIMEOUT`.
async fn bounded(check: impl std::future::Future<Output = bool>) -> bool {
    tokio::time::timeout(HEALTH_CHECK_TIMEOUT, check)
        .await
        .unwrap_or(false)
}

/// A provider probe that did not answer within `HEALTH_CHECK_TIMEOUT`: the provider (or the path to
/// it) is slow; probes take no query slot. Re-probed once `HEALTH_MIN_REPROBE` has passed.
const PROVIDER_BUSY: &str = "busy";

fn status(ok: bool) -> &'static str {
    if ok { "ok" } else { "down" }
}

pub struct Engine {
    pub pool: sqlx::PgPool,
    /// Embed / rerank providers, resolved per request's tenant ([`Engine::inference`]).
    pub providers: Providers,
    pub tc: Arc<dyn TokenCounter>,
    pub cache: RoutingCache,
    pub laya: Option<Arc<dyn ChoiceOracle>>,
    pub laya_health: Option<Arc<LayaClient>>,
    /// Records cost events of provider calls in `metering_outbox` (`crate::metering`); `None` = off.
    pub metering: Option<Arc<Metering>>,
    /// Query vectors by (tenant, model, text) (`STRATA_QUERY_EMBED_CACHE`).
    pub query_cache: crate::query_cache::QueryEmbedCache,
    pub cfg: Config,
    /// One `POST /api/eval` at a time PER TENANT (each runs a query per question per mode, laya
    /// calls included); a second concurrent request of the same tenant is refused, not queued, and
    /// one tenant's eval never refuses another's.
    pub(crate) eval_slots: EvalSlots,
    /// The tenant eval runs this process is scoring, so a shutdown can fail them
    /// ([`EvalTasks::fail_on_shutdown`]) instead of leaving them `running` until the stale sweep.
    pub eval_tasks: Arc<EvalTasks>,
    /// Last `/health` report (`STRATA_HEALTH_CACHE_S`, at least `HEALTH_MIN_REPROBE`).
    health_cache: std::sync::Mutex<Option<HealthCache>>,
    /// One `/health` probe at a time, held by the task running it; callers never wait on it once a
    /// report is cached.
    health_slot: Arc<tokio::sync::Semaphore>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StoreView {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub kind: StoreKind,
    pub name: String,
    pub description: String,
    pub token_count: i64,
    pub chunk_count: i64,
    pub children: Vec<Uuid>,
}

impl StoreView {
    fn from_row(row: StoreRow, children: Vec<Uuid>) -> Self {
        StoreView {
            id: row.id,
            parent_id: row.parent_id,
            kind: row.kind,
            name: row.name,
            description: row.description,
            token_count: row.token_count,
            chunk_count: row.chunk_count,
            children,
        }
    }
}

/// `/health` body. Each component is "ok", "down" or "unconfigured" (laya without a licence
/// token). `embed` / `rerank` / `chat` are the deployment's DEFAULT providers (dev only; a production
/// deployment has none, so "unconfigured"), never a tenant's registration: `/health` is
/// unauthenticated. Each is probed with one real call, in the background (see `Engine::health`):
/// "busy" when the call did not answer within the check timeout (probes take no query slot, so: a
/// slow provider or path), and on "down" `embed_error` / `rerank_error` / `chat_error` carry the error class (`auth`,
/// `retryable`, `invalid_request`, `invalid_response`, `internal`), never a message, URL or key.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthReport {
    pub db: &'static str,
    pub embed: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed_error: Option<&'static str>,
    pub rerank: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_error: Option<&'static str>,
    /// The default chat provider (`GET {base_url}/models`, no generation); replaces the old
    /// `vllm` component.
    pub chat: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_error: Option<&'static str>,
    pub laya: &'static str,
    /// Cost-event backlog (`crate::metering`); absent when metering is off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metering: Option<MeteringHealth>,
}

/// `/health` `metering`: what is not yet on billing's stream. `outbox_pending` /
/// `outbox_oldest_pending_age_s` are rows not yet delivered (`null` when the outbox could not be
/// read in time); an old pending row is undelivered spend. `queued` is events in memory not yet in
/// the outbox; `dropped_total` events lost since start (queue full, or a row Postgres refused).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct MeteringHealth {
    pub outbox_pending: Option<i64>,
    pub outbox_oldest_pending_age_s: Option<i64>,
    pub queued: usize,
    pub dropped_total: u64,
}

/// Minimum time between two provider probes, whatever `STRATA_HEALTH_CACHE_S` says (0 included):
/// `/health` is unauthenticated, and each probe is a real embed + rerank call.
pub const HEALTH_MIN_REPROBE: Duration = Duration::from_secs(2);

#[derive(Clone)]
struct HealthCache {
    /// Last full probe (all components).
    full_at: Instant,
    /// Last probe of any kind (a full one, or a re-probe of busy providers).
    probed_at: Instant,
    report: HealthReport,
}

impl HealthCache {
    fn busy(&self) -> bool {
        self.report.embed == PROVIDER_BUSY
            || self.report.rerank == PROVIDER_BUSY
            || self.report.chat == PROVIDER_BUSY
    }
}

/// Which probe `cache` needs now: `Some(false)` a full one (nothing cached, or the TTL ran out),
/// `Some(true)` a re-probe of the busy providers only, `None` none.
fn health_due(cache: Option<&HealthCache>, ttl: Duration) -> Option<bool> {
    match cache {
        None => Some(false),
        Some(c) if c.full_at.elapsed() >= ttl => Some(false),
        Some(c) if c.busy() && c.probed_at.elapsed() >= HEALTH_MIN_REPROBE => Some(true),
        Some(_) => None,
    }
}

/// Logs a provider whose `/health` status changed (the first report counts as a change from "ok"),
/// so a busy or failing provider is visible in the pod log, not only on `/health`.
fn log_provider_changes(before: Option<&HealthReport>, after: &HealthReport) {
    for (provider, was, now, error) in [
        (
            "embed",
            before.map_or("ok", |b| b.embed),
            after.embed,
            after.embed_error,
        ),
        (
            "rerank",
            before.map_or("ok", |b| b.rerank),
            after.rerank,
            after.rerank_error,
        ),
        (
            "chat",
            before.map_or("ok", |b| b.chat),
            after.chat,
            after.chat_error,
        ),
    ] {
        if was == now {
            continue;
        }
        if now == "ok" || now == "unconfigured" {
            tracing::info!(provider, was, "provider health probe ok again");
        } else {
            tracing::warn!(
                provider,
                status = now,
                error = error.unwrap_or(""),
                "provider health probe not ok"
            );
        }
    }
}

/// One provider probe: status plus, when down, the error class.
type Probe = (&'static str, Option<&'static str>);

fn probe_result<T>(
    r: std::result::Result<Result<T>, tokio::time::error::Elapsed>,
    ok: impl FnOnce(&T) -> bool,
) -> Probe {
    match r {
        Ok(Ok(v)) if ok(&v) => ("ok", None),
        Ok(Ok(_)) => ("down", Some("invalid_response")),
        Ok(Err(e)) => ("down", Some(error_class(&e))),
        Err(_elapsed) => (PROVIDER_BUSY, None),
    }
}

/// The tenants with an eval running (see `Engine::eval_slots`).
#[derive(Default)]
pub(crate) struct EvalSlots(std::sync::Mutex<std::collections::HashSet<Tenant>>);

impl EvalSlots {
    /// `tenant`'s slot, or `None` while another eval of that tenant holds it. Released on drop.
    pub(crate) fn try_acquire(&self, tenant: Tenant) -> Option<EvalSlot<'_>> {
        let mut running = self.0.lock().unwrap_or_else(|p| p.into_inner());
        // Lazy `then`, never `then_some`: an eagerly built EvalSlot is dropped right here when the
        // tenant is already running, and its Drop re-locks this (non-reentrant) mutex while
        // `running` still holds it — a deadlock that also freed the running eval's slot.
        running.insert(tenant).then(|| EvalSlot {
            slots: self,
            tenant,
        })
    }
}

/// A held eval slot; dropping it (also on error or a cancelled request) frees the tenant's slot.
pub(crate) struct EvalSlot<'a> {
    slots: &'a EvalSlots,
    tenant: Tenant,
}

impl Drop for EvalSlot<'_> {
    fn drop(&mut self) {
        self.slots
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.tenant);
    }
}

/// The error a run gets when the process scoring it shuts down.
pub const EVAL_SHUTDOWN: &str = "shutdown";
/// How long a shutdown waits for its runs to be stored `failed` before it gives up (they are then
/// swept as interrupted once stale).
pub const EVAL_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// The tenant eval runs (id → tenant) whose background task runs in this process.
#[derive(Default)]
pub struct EvalTasks(std::sync::Mutex<std::collections::HashMap<Uuid, Tenant>>);

impl EvalTasks {
    /// Records `run` as scored by this process until the returned guard is dropped (the task
    /// ended: completed, failed, panicked or dropped).
    pub fn track(self: &Arc<Self>, tenant: Tenant, run: Uuid) -> EvalTask {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(run, tenant);
        EvalTask {
            tasks: Arc::clone(self),
            run,
        }
    }

    /// The runs in flight now, as `(tenant, run id)`.
    pub fn in_flight(&self) -> Vec<(Tenant, Uuid)> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|(run, tenant)| (*tenant, *run))
            .collect()
    }

    /// On shutdown: stores every run this process is scoring as `failed` with error
    /// [`EVAL_SHUTDOWN`] (only those still `running`: one that completed meanwhile keeps its
    /// result), within `timeout`. The tasks die with the process; a run they still touch is no
    /// longer `running`, so they write nothing more. Answers how many runs it failed; on an error
    /// or a timeout it logs and answers 0, and the stale sweep fails those runs later.
    pub async fn fail_on_shutdown(&self, pool: &sqlx::PgPool, timeout: std::time::Duration) -> u64 {
        let runs = self.in_flight();
        if runs.is_empty() {
            return 0;
        }
        match tokio::time::timeout(
            timeout,
            crate::db::eval::fail_runs(pool, &runs, EVAL_SHUTDOWN),
        )
        .await
        {
            Ok(Ok(n)) => {
                tracing::warn!(failed = n, "tenant eval runs failed on shutdown");
                n
            }
            Ok(Err(e)) => {
                tracing::error!(error = %e, runs = runs.len(), "cannot fail tenant eval runs on shutdown");
                0
            }
            Err(_elapsed) => {
                tracing::error!(
                    runs = runs.len(),
                    "failing tenant eval runs on shutdown timed out"
                );
                0
            }
        }
    }
}

/// A run tracked in [`EvalTasks`]; dropping it untracks the run.
pub struct EvalTask {
    tasks: Arc<EvalTasks>,
    run: Uuid,
}

impl Drop for EvalTask {
    fn drop(&mut self) {
        self.tasks
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.run);
    }
}

/// A store's name and description: what its `desc_vec` embeds.
struct StoreMeta {
    name: String,
    description: String,
}

/// The text embedded into a store's `desc_vec`.
pub fn desc_text(name: &str, description: &str) -> String {
    format!("{name}\n\n{description}")
}

fn clean_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(StrataError::InvalidInput(
            "store name must not be empty".to_string(),
        ));
    }
    Ok(name.to_string())
}

/// Any failure while loading the tokenizer surfaces as `StrataError::Model`.
fn as_model_error(e: StrataError) -> StrataError {
    match e {
        StrataError::Model(m) => StrataError::Model(m),
        other => StrataError::Model(other.to_string()),
    }
}

impl Engine {
    /// Production constructor: the deployment's `providers.toml` must be loaded
    /// (`Config::from_env`); build the dev default provider clients, connect + migrate the DB, load
    /// the chunking tokenizer, and build the laya client when its licence token exists.
    /// Absent tokens leave the client unconfigured (not an error).
    pub async fn from_config(cfg: Config) -> Result<Self> {
        let deployment = cfg.deployment.clone().ok_or_else(|| {
            StrataError::Config(format!(
                "no providers file loaded; set STRATA_PROVIDERS_FILE (default {})",
                crate::providers::deployment::DEFAULT_PROVIDERS_FILE
            ))
        })?;
        let cipher = cfg
            .provider_key
            .as_ref()
            .map(|k| KeyCipher::from_base64(k.expose()))
            .transpose()?;
        let limits = LaneLimits {
            embed_batch: cfg.embed_batch,
            bulk_concurrency: cfg.provider_bulk_concurrency,
        };

        let pool = crate::db::connect_with(&cfg.database_url, cfg.pool_settings())
            .await
            .map_err(|e| match e {
                StrataError::Db(cause) => StrataError::Config(format!(
                    "database unreachable at {}: {cause}",
                    redact_url(&cfg.database_url)
                )),
                other => other,
            })?;
        let providers =
            Providers::registry(pool.clone(), deployment.clone(), cipher, limits).await?;

        let tc = tokio::task::spawn_blocking(|| -> Result<HfTokenCounter> {
            HfTokenCounter::from_file(&bge_m3_tokenizer_path()?)
        })
        .await
        .map_err(|e| StrataError::Model(format!("tokenizer loader panicked: {e}")))?
        .map_err(as_model_error)?;
        let tc: Arc<dyn TokenCounter> = Arc::new(tc);

        // METERING_REDIS_URL on: record every metered call in `metering_outbox` (the drainer
        // that takes the rows to Redis runs in `strata serve` only, `cli::dispatch`)
        let metering = match &cfg.metering_redis_url {
            Some(url) => {
                crate::metering::check_redis_url(url.expose())?;
                Some(Metering::start(pool.clone(), tc.clone()))
            }
            None => None,
        };
        let mut engine = Engine::with_source(pool, providers, tc, cfg.clone());
        engine.metering = metering;
        if let Some(token) = cfg.laya_token.clone() {
            let laya = Arc::new(LayaClient::new(
                &cfg.laya_url,
                token,
                &cfg.laya_model,
                cfg.laya_timeout_ms,
            )?);
            engine = engine.with_laya(laya.clone());
            engine.laya_health = Some(laya);
        }
        let default_model = |s: Option<&crate::providers::ProviderSpec>| {
            s.map_or_else(|| "none".to_string(), |s| s.model.clone())
        };
        tracing::info!(
            db = %redact_url(&cfg.database_url),
            region = %deployment.region,
            mode = ?deployment.mode,
            default_embed = %default_model(deployment.default_embed.as_ref()),
            default_rerank = %default_model(deployment.default_rerank.as_ref()),
            default_chat = %default_model(deployment.default_chat.as_ref()),
            default_chat_routed = deployment.default_chat_routed,
            provider_key = cfg.provider_key.is_some(),
            laya = cfg.laya_token.is_some(),
            metering = engine.metering.is_some(),
            metering_stream = %cfg.metering_stream,
            "engine ready"
        );
        if deployment.mode == Mode::Production && cfg.provider_key.is_none() {
            tracing::warn!(
                "production deployment without STRATA_PROVIDER_KEY: tenants cannot register \
                 providers (PUT /api/tenant/providers answers 503)"
            );
        }
        Ok(engine)
    }

    /// An engine whose providers are `inf` for every tenant (tests, in-process tools); see
    /// [`Engine::with_providers`] for tenant registrations.
    pub fn new(
        pool: sqlx::PgPool,
        inf: InferenceHandle,
        tc: Arc<dyn TokenCounter>,
        cfg: Config,
    ) -> Self {
        Self::with_source(pool, Providers::fixed(inf), tc, cfg)
    }

    /// An engine resolving providers through `providers`.
    pub fn with_source(
        pool: sqlx::PgPool,
        providers: Providers,
        tc: Arc<dyn TokenCounter>,
        cfg: Config,
    ) -> Self {
        Engine {
            pool,
            providers,
            tc,
            cache: RoutingCache::default(),
            laya: None,
            laya_health: None,
            metering: None,
            query_cache: crate::query_cache::QueryEmbedCache::new(cfg.query_embed_cache),
            cfg,
            eval_slots: EvalSlots::default(),
            eval_tasks: Arc::default(),
            health_cache: std::sync::Mutex::new(None),
            health_slot: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    /// Ground truth: the first (cold) laya call takes ~2.2 s. One throw-away call warms the
    /// pooled connection and laya's model. For `serve` only: a one-shot CLI command would pay for
    /// a laya request it never uses. Result ignored; never blocks.
    pub fn spawn_laya_warmup(&self) {
        let Some(laya) = self.laya_health.clone() else {
            return;
        };
        tokio::spawn(async move {
            let options = vec![
                ("warm-a".to_string(), "connection warm-up".to_string()),
                ("warm-b".to_string(), "connection warm-up".to_string()),
            ];
            let _ = laya.choose("warm-up", &options).await;
        });
    }

    pub fn with_providers(self, providers: Providers) -> Self {
        Engine { providers, ..self }
    }

    /// `tenant`'s providers for one request (registration → dev default → unconfigured; a call
    /// needing an unconfigured capability fails with 409 `provider_not_configured`). With metering
    /// on, every call through the handle emits a cost event billed to `tenant` and the running
    /// request's caller headers (`metering::current_caller`).
    pub async fn inference(&self, tenant: Tenant) -> Result<InferenceHandle> {
        self.inference_for(tenant, None).await
    }

    /// The providers of one request acting on `store`: per capability the registration nearest
    /// to the store (`store_providers`: its own, else its nearest ancestor's), else the tenant's
    /// (as [`Engine::inference`]). `None` = the tenant's.
    pub async fn inference_for(
        &self,
        tenant: Tenant,
        store: Option<Uuid>,
    ) -> Result<InferenceHandle> {
        let inf = self.providers.resolve_for(tenant, store).await?;
        Ok(self.metered(tenant, inf))
    }

    /// The providers that embed a query for `store`, a store of the platform owner `owner` that
    /// `caller` was granted (spec platform-knowledge §7): resolved through the platform owner's
    /// lineage (its store bindings, then its own providers, then the deployment defaults), metered
    /// to `caller`, and funded by the platform (`InferenceHandle::platform_funded`).
    pub async fn platform_inference_for(
        &self,
        caller: Tenant,
        owner: Tenant,
        store: Uuid,
    ) -> Result<InferenceHandle> {
        let inf = self.providers.resolve_for(owner, Some(store)).await?;
        Ok(self.metered(caller, inf).platform_funded())
    }

    /// `inf` with metering on when the engine meters (see [`Engine::inference`]).
    pub fn metered(&self, tenant: Tenant, inf: InferenceHandle) -> InferenceHandle {
        match &self.metering {
            Some(m) => inf.metered(
                Arc::clone(m),
                Attribution {
                    tenant,
                    caller: current_caller(),
                },
            ),
            None => inf,
        }
    }

    pub fn with_metering(self, metering: Arc<Metering>) -> Self {
        Engine {
            metering: Some(metering),
            ..self
        }
    }

    pub fn with_laya(self, oracle: Arc<dyn ChoiceOracle>) -> Self {
        Engine {
            laya: Some(oracle),
            ..self
        }
    }

    /// Fixed providers (tests, in-process tools) plus `chat` for every tenant. An engine resolving
    /// through the tenant registry is returned unchanged (its chat comes from registrations and
    /// `[default.chat]`).
    pub fn with_chat(self, chat: Arc<dyn crate::inference::Chat>) -> Self {
        Engine {
            providers: self.providers.with_fixed_chat(chat),
            ..self
        }
    }

    /// `EmbedModelMismatch` unless `id` and every ancestor up to the root were embedded with
    /// `configured` (the resolved embed provider's identity). Ancestors matter because every write rolls the store's vectors up into
    /// them (`recompute_upwards` averages child `content_vec`s), so a mismatched ancestor would
    /// mix two vector spaces.
    /// `id` must already be resolved within the caller's tenant (ancestors share it).
    pub async fn check_lineage_embed_model(
        &self,
        conn: &mut sqlx::PgConnection,
        id: Uuid,
        configured: &str,
    ) -> Result<()> {
        match stores::lineage_embed_mismatch(conn, id, configured).await? {
            None => Ok(()),
            Some((store_id, store_model)) => Err(StrataError::EmbedModelMismatch {
                store_id,
                store_model,
                configured: configured.to_string(),
            }),
        }
    }

    /// A root (no parent) belongs to `tenant`. A tenant may hold several roots (D83: a store
    /// re-embedded in another model is a tree of its own, `reembed_copy`); root names are unique
    /// within the tenant, and with several roots a query must name its `root_id` (D90). A parent
    /// must be `tenant`'s: another tenant's parent is 404, as if it did not exist. The new store
    /// is embedded with its parent's providers (a store binding is inherited), a root with the
    /// tenant's.
    pub async fn create_store(&self, tenant: Tenant, new: NewStore) -> Result<StoreView> {
        let new = NewStore {
            name: clean_name(&new.name)?,
            description: new.description.trim().to_string(),
            ..new
        };
        // embed before opening the transaction: no locks held during inference
        let inf = self.inference_for(tenant, new.parent_id).await?;
        let model = inf.embed_model_id()?;
        let desc_vec = inf
            .embed_one(desc_text(&new.name, &new.description))
            .await?;

        let row =
            crate::db::retry_once(|| self.insert_store_tx(tenant, &new, &desc_vec, model)).await?;
        self.cache.invalidate(tenant).await;
        Ok(StoreView::from_row(row, Vec::new()))
    }

    /// `create_store`'s transaction: check the root / parent precondition under a lock, insert.
    async fn insert_store_tx(
        &self,
        tenant: Tenant,
        new: &NewStore,
        desc_vec: &[f32],
        model: &str,
    ) -> Result<StoreRow> {
        let mut tx = self.pool.begin().await?;
        // a root needs no check: a second root of the same name is 409 duplicate_name (unique
        // per tenant, 0005)
        {
            if let Some(parent_id) = new.parent_id {
                // lock the parent so it cannot be deleted under us; it must be this tenant's
                // (another tenant's parent is 404, never a hint that it exists)
                let parent = match stores::get_store_for_update(&mut tx, tenant, parent_id).await {
                    Ok(p) => p,
                    Err(StrataError::NotFound(_)) => {
                        return Err(StrataError::NotFound(format!("parent store {parent_id}")));
                    }
                    Err(e) => return Err(e),
                };
                if parent.kind != StoreKind::Router {
                    return Err(StrataError::InvalidTarget(format!(
                        "parent store {parent_id} is a leaf; only routers can have children"
                    )));
                }
                // the new store's desc_vec rolls up into the parent and every ancestor
                self.check_lineage_embed_model(&mut tx, parent_id, model)
                    .await?;
                // no new store inside a published platform pack
                crate::db::platform::assert_writable(&mut *tx, parent_id).await?;
            }
        }
        let row = stores::insert_store(&mut tx, tenant, new, desc_vec, model).await?;
        tx.commit().await?;
        Ok(row)
    }

    /// Flat list of `tenant`'s stores (uuid v7 order), each with its child ids.
    pub async fn store_tree(&self, tenant: Tenant) -> Result<Vec<StoreView>> {
        let rows = stores::list_stores(&self.pool, tenant).await?;
        let mut children: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        for r in &rows {
            if let Some(p) = r.parent_id {
                children.entry(p).or_default().push(r.id);
            }
        }
        Ok(rows
            .into_iter()
            .map(|r| {
                let kids = children.remove(&r.id).unwrap_or_default();
                StoreView::from_row(r, kids)
            })
            .collect())
    }

    pub async fn get_store(&self, tenant: Tenant, id: Uuid) -> Result<StoreView> {
        let mut conn = self.pool.acquire().await?;
        let row = stores::get_store(&mut conn, tenant, id).await?;
        let children = stores::child_ids(&mut conn, id).await?;
        Ok(StoreView::from_row(row, children))
    }

    /// Reads the store, embeds the new name/description with no lock held (inference is slow),
    /// then writes under the row lock. A field the request leaves out is taken from the unlocked
    /// read, so if a concurrent update changed such a field during the embed, writing would put the
    /// stale value back (and a `desc_vec` of text the row no longer holds): that answers 409
    /// `conflict` instead. A field the request sets is last-writer-wins.
    pub async fn update_store(
        &self,
        tenant: Tenant,
        id: Uuid,
        name: Option<String>,
        description: Option<String>,
    ) -> Result<StoreView> {
        let inf = self.inference_for(tenant, Some(id)).await?;
        let current = {
            let mut conn = self.pool.acquire().await?;
            let current = stores::get_store(&mut conn, tenant, id).await?;
            // the new desc_vec comes from the resolved model; it rolls up into the ancestors
            self.check_lineage_embed_model(&mut conn, id, inf.embed_model_id()?)
                .await?;
            current
        };
        let name = match name {
            Some(n) => Some(clean_name(&n)?),
            None => None,
        };
        let description = description.map(|d| d.trim().to_string());
        let meta = StoreMeta {
            name: name.clone().unwrap_or_else(|| current.name.clone()),
            description: description
                .clone()
                .unwrap_or_else(|| current.description.clone()),
        };
        let desc_vec = inf
            .embed_one(desc_text(&meta.name, &meta.description))
            .await?;

        // the fields we did not set must still be what we embedded
        let expect = StoreMeta {
            name: current.name,
            description: current.description,
        };
        let inherited = (name.is_none(), description.is_none());
        crate::db::retry_once(|| {
            self.update_store_tx(tenant, id, &meta, &desc_vec, &expect, inherited)
        })
        .await?;
        self.cache.invalidate(tenant).await;
        self.get_store(tenant, id).await
    }

    /// `update_store`'s transaction. Locks the store row first, then `recompute_upwards` walks to
    /// the root: child before parent, the order every writer uses.
    async fn update_store_tx(
        &self,
        tenant: Tenant,
        id: Uuid,
        meta: &StoreMeta,
        desc_vec: &[f32],
        expect: &StoreMeta,
        (name_inherited, description_inherited): (bool, bool),
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let locked = stores::get_store_for_update(&mut tx, tenant, id).await?;
        crate::db::platform::assert_writable(&mut *tx, id).await?;
        if (name_inherited && locked.name != expect.name)
            || (description_inherited && locked.description != expect.description)
        {
            return Err(StrataError::Conflict {
                message: format!("store {id} was changed concurrently; retry the request"),
                // rerunning this transaction would fail the same way: the embed is stale
                retryable: false,
            });
        }
        stores::update_store_meta(&mut tx, tenant, id, &meta.name, &meta.description, desc_vec)
            .await?;
        stores::recompute_upwards(&mut tx, tenant, id, self.cfg.profile_alpha).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn delete_store(&self, tenant: Tenant, id: Uuid, recursive: bool) -> Result<()> {
        crate::db::retry_once(|| self.delete_store_tx(tenant, id, recursive)).await?;
        self.cache.invalidate(tenant).await;
        Ok(())
    }

    async fn delete_store_tx(&self, tenant: Tenant, id: Uuid, recursive: bool) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        // the caller's store (404 otherwise) before the pack guard, so a 409 never reveals another
        // tenant's store; a published pack is frozen, a draft or retired pack goes with its root
        stores::get_store(&mut tx, tenant, id).await?;
        crate::db::platform::release_for_delete(&mut tx, id).await?;
        let parent = stores::delete_store(&mut tx, tenant, id, recursive).await?;
        if let Some(parent_id) = parent {
            stores::recompute_upwards(&mut tx, tenant, parent_id, self.cfg.profile_alpha).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// `/health` is unauthenticated, so its report is cached for `STRATA_HEALTH_CACHE_S`, never less
    /// than `HEALTH_MIN_REPROBE` (also when the setting is 0). A cached "busy" provider is re-probed
    /// (that provider only) once `HEALTH_MIN_REPROBE` has passed, so it clears soon after the
    /// provider answers again rather than staying stale for the whole TTL.
    ///
    /// Probes never run inside the caller's request: once anything is cached, `/health` answers from
    /// memory at once and a due probe runs in a spawned task (one at a time, `health_slot`). The
    /// kubelet's readiness probe calls this with a 1 s timeout; a probe that waits on a slow upstream
    /// (up to `HEALTH_CHECK_TIMEOUT` per check) inside the request flapped the pod not-Ready every
    /// time an upstream was slow, although `/health` always answers 200. Only the very first callers,
    /// with nothing cached yet, wait for the first probe; it too runs in a task, so a caller that
    /// gives up (the kubelet's timeout) cannot cancel it half-way.
    pub async fn health(self: &Arc<Self>) -> HealthReport {
        let ttl = self.health_ttl();
        loop {
            let snap = self.cached_health();
            if health_due(snap.as_ref(), ttl).is_some()
                && let Ok(slot) = Arc::clone(&self.health_slot).try_acquire_owned()
            {
                let engine = Arc::clone(self);
                tokio::spawn(async move {
                    engine.refresh_health().await;
                    drop(slot);
                });
            }
            if let Some(c) = snap {
                return c.report;
            }
            // nothing cached: wait for the running first probe, then read its result
            drop(self.health_slot.acquire().await);
        }
    }

    fn health_ttl(&self) -> Duration {
        Duration::from_secs(self.cfg.health_cache_s).max(HEALTH_MIN_REPROBE)
    }

    /// Runs the probe that is due, if one still is (another task may have just finished one), and
    /// caches it. The caller holds `health_slot`. Timestamps are taken when the probe ENDS: a probe
    /// can run for `HEALTH_CHECK_TIMEOUT`, and the minimum interval is between probes, not between
    /// their starts.
    async fn refresh_health(&self) {
        let snap = self.cached_health();
        let Some(busy_only) = health_due(snap.as_ref(), self.health_ttl()) else {
            return;
        };
        let entry = match snap.clone() {
            Some(mut c) if busy_only => {
                if c.report.embed == PROVIDER_BUSY {
                    (c.report.embed, c.report.embed_error) = self.probe_embed().await;
                }
                if c.report.rerank == PROVIDER_BUSY {
                    (c.report.rerank, c.report.rerank_error) = self.probe_rerank().await;
                }
                if c.report.chat == PROVIDER_BUSY {
                    (c.report.chat, c.report.chat_error) = self.probe_chat().await;
                }
                c.probed_at = Instant::now();
                c
            }
            _ => {
                let report = self.probe_health().await;
                let now = Instant::now();
                HealthCache {
                    full_at: now,
                    probed_at: now,
                    report,
                }
            }
        };
        log_provider_changes(snap.as_ref().map(|c| &c.report), &entry.report);
        *self.health_cache.lock().unwrap_or_else(|p| p.into_inner()) = Some(entry);
    }

    fn cached_health(&self) -> Option<HealthCache> {
        self.health_cache
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// One embed straight to the default provider (no query slot, see
    /// `InferenceHandle::probe_embed`). A failing provider answers with an error (its class is
    /// reported); no answer within the check timeout is "busy", not "down"; no default provider is
    /// "unconfigured".
    async fn probe_embed(&self) -> Probe {
        let inf = self.providers.health_handle();
        if !inf.has_embed() {
            return ("unconfigured", None);
        }
        probe_result(
            tokio::time::timeout(HEALTH_CHECK_TIMEOUT, inf.probe_embed()).await,
            |v| v.len() == DIM,
        )
    }

    /// One single-passage rerank straight to the default provider, classified like `probe_embed`.
    async fn probe_rerank(&self) -> Probe {
        let inf = self.providers.health_handle();
        if !inf.has_rerank() {
            return ("unconfigured", None);
        }
        probe_result(
            tokio::time::timeout(HEALTH_CHECK_TIMEOUT, inf.probe_rerank()).await,
            |s| s.len() == 1,
        )
    }

    /// The default chat provider's liveness check (`/models`, no generation), classified like
    /// `probe_embed`.
    async fn probe_chat(&self) -> Probe {
        let inf = self.providers.health_handle();
        if !inf.has_chat() {
            return ("unconfigured", None);
        }
        probe_result(
            tokio::time::timeout(HEALTH_CHECK_TIMEOUT, inf.probe_chat()).await,
            |_: &()| true,
        )
    }

    /// All checks run concurrently, each bounded by `HEALTH_CHECK_TIMEOUT`, so a hung upstream
    /// (a chat client's timeout is up to 60 s) cannot hold a probe. On timeout db/laya report
    /// "down"; the providers report "busy".
    async fn probe_health(&self) -> HealthReport {
        let db = async { sqlx::query("SELECT 1").execute(&self.pool).await.is_ok() };
        let laya = async {
            match &self.laya_health {
                None => None,
                Some(c) => Some(bounded(c.health()).await),
            }
        };
        let metering = async {
            let m = self.metering.as_ref()?;
            let backlog = tokio::time::timeout(
                HEALTH_CHECK_TIMEOUT,
                crate::metering::outbox_backlog(&self.pool),
            )
            .await
            .ok()
            .and_then(|r| r.ok());
            Some(MeteringHealth {
                outbox_pending: backlog.map(|b| b.0),
                outbox_oldest_pending_age_s: backlog.map(|b| b.1),
                queued: m.queued(),
                dropped_total: m.dropped(),
            })
        };
        let (db, (embed, embed_error), (rerank, rerank_error), (chat, chat_error), laya, metering) = tokio::join!(
            bounded(db),
            self.probe_embed(),
            self.probe_rerank(),
            self.probe_chat(),
            laya,
            metering
        );
        HealthReport {
            db: status(db),
            embed,
            embed_error,
            rerank,
            rerank_error,
            chat,
            chat_error,
            laya: laya.map_or("unconfigured", status),
            metering,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_slots_are_per_tenant_and_freed_on_drop() {
        let (a, b) = (
            Tenant::new(Uuid::from_u128(1)).unwrap(),
            Tenant::new(Uuid::from_u128(2)).unwrap(),
        );
        let slots = EvalSlots::default();
        let held = slots.try_acquire(a).expect("a's first eval");
        assert!(
            slots.try_acquire(a).is_none(),
            "a second eval of a is refused"
        );
        let other = slots.try_acquire(b).expect("b is not blocked by a");
        drop(held);
        assert!(slots.try_acquire(a).is_some(), "a's slot is freed on drop");
        drop(other);
    }
}
