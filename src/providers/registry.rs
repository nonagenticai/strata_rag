//! Per-request provider resolution and the tenant / store provider-config API (spec §4.1, D83).
//!
//! Resolution for a request, per capability: when the request acts on a store, the registration
//! nearest to it (`store_providers`: the store's own row, else its nearest ancestor's; embed,
//! rerank and chat, D92) → the tenant's registration → else (dev deployment only) the `providers.toml`
//! default → else the capability is unconfigured, and the first call that needs it fails with
//! `409 provider_not_configured`.
//!
//! Built clients are cached per `(tenant or store, capability)`, keyed by the row's `updated_at`.
//! Every resolution reads the tenant's `(capability, updated_at)` pairs and, for a store, its
//! lineage's (one indexed query each), so a PUT on any replica switches every replica on its next
//! request; the replica that took the PUT also drops its entries at once. A cached client keeps
//! its own query / bulk slots across requests.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sqlx::types::chrono::{DateTime, Utc};

use super::crypto::{KeyCipher, Sealed};
use super::deployment::{Deployment, Mode};
use super::{Capability, DEFAULT_CONCURRENCY, ProviderSpec, Rejection, validate};
use crate::clients::chat::{ChatClient, ChatOptions};
use crate::clients::embed::EmbedClient;
use crate::clients::provider::error_class;
use crate::clients::rerank::RerankClient;
use crate::config::{Secret, redact_url};
use crate::db::providers::{self as rows, ProviderRow};
use crate::error::{Result, StrataError};
use crate::inference::{ChatLane, EmbedLane, InferenceHandle, LaneFault, Limits, RerankLane};
use crate::metering::ProviderMeta;
use crate::tenant::Tenant;
use crate::vecmath::DIM;

/// Longest `timeout_ms` a tenant may register (a slow tenant provider holds query slots).
pub const MAX_TENANT_EMBED_TIMEOUT_MS: u64 = 30_000;
pub const MAX_TENANT_RERANK_TIMEOUT_MS: u64 = 15_000;
/// Chat generates (an answer of up to 8192 tokens), so it may be slower: 300 s covers the full
/// token cap at about 7.5 tok/s with headroom. Applies to tenant registrations only; the
/// deployment `[default.chat]` is bounded by `MAX_TIMEOUT_MS` (600 s) alone.
pub const MAX_TENANT_CHAT_TIMEOUT_MS: u64 = 300_000;

/// Upper bound for one registration probe (connect + call + retries).
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// Upper bound for a chat registration probe: the provider's `timeout_ms`, at most this.
pub const CHAT_PROBE_TIMEOUT_MAX: Duration = Duration::from_secs(30);

/// Deployment-wide lane settings applied to every resolved embed provider.
#[derive(Debug, Clone, Copy)]
pub struct LaneLimits {
    /// Texts per ingest embed request (`STRATA_BULK_EMBED_BATCH`).
    pub embed_batch: usize,
    /// Concurrent ingest embed requests per embed provider (`STRATA_PROVIDER_BULK_CONCURRENCY`),
    /// never more than the provider's own `concurrency`.
    pub bulk_concurrency: usize,
}

impl Default for LaneLimits {
    fn default() -> Self {
        LaneLimits {
            embed_batch: 32,
            bulk_concurrency: 2,
        }
    }
}

#[derive(Clone)]
enum Lane {
    Embed(Arc<EmbedLane>),
    Rerank(Arc<RerankLane>),
    Chat(Arc<ChatLane>),
}

struct Cached {
    updated_at: DateTime<Utc>,
    lane: Lane,
}

/// Whose registration a cached lane was built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Scope {
    Tenant(Tenant),
    /// A `store_providers` row of this store.
    Store(uuid::Uuid),
}

/// The engine's provider source.
pub struct Providers(Source);

enum Source {
    /// One handle for every tenant (tests, in-process tools): no registrations.
    Fixed(InferenceHandle),
    Registry(Box<Registry>),
}

struct Registry {
    pool: sqlx::PgPool,
    deployment: Deployment,
    cipher: Option<KeyCipher>,
    limits: LaneLimits,
    /// The dev defaults' lanes, built once (`None` in production or when not configured).
    default_embed: Option<Arc<EmbedLane>>,
    default_rerank: Option<Arc<RerankLane>>,
    default_chat: Option<Arc<ChatLane>>,
    cache: Mutex<HashMap<(Scope, Capability), Cached>>,
}

/// Who pays for a lane's calls (`crate::metering`): the tenant's own registration, or the
/// deployment's default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    Tenant,
    Deployment,
}

fn meta(owner: Owner, spec: &ProviderSpec) -> ProviderMeta {
    match owner {
        Owner::Tenant => ProviderMeta::tenant_registered(&spec.base_url, &spec.model),
        Owner::Deployment => ProviderMeta::deployment_default(&spec.base_url, &spec.model),
    }
}

/// `guarded`: a tenant registration under the SSRF guard (`super::ssrf::GuardedResolver`).
fn build_embed(
    spec: &ProviderSpec,
    limits: LaneLimits,
    guarded: bool,
    owner: Owner,
) -> Result<Arc<EmbedLane>> {
    let client = EmbedClient::with_guard(
        &spec.base_url,
        spec.api_key.clone(),
        &spec.model,
        spec.timeout_ms,
        guarded,
    )?;
    Ok(Arc::new(
        EmbedLane::new(
            Arc::new(client),
            spec.embed_model_id(),
            Limits {
                embed_batch: limits.embed_batch,
                query_concurrency: spec.concurrency,
                bulk_concurrency: limits.bulk_concurrency.min(spec.concurrency),
            },
        )
        .with_meta(meta(owner, spec)),
    ))
}

fn build_rerank(spec: &ProviderSpec, guarded: bool, owner: Owner) -> Result<Arc<RerankLane>> {
    let client = RerankClient::with_guard(
        &spec.base_url,
        spec.api_key.clone(),
        &spec.model,
        spec.timeout_ms,
        guarded,
    )?;
    Ok(Arc::new(
        RerankLane::new(Arc::new(client), spec.concurrency).with_meta(meta(owner, spec)),
    ))
}

fn chat_client(spec: &ProviderSpec, guarded: bool) -> Result<ChatClient> {
    ChatClient::with_guard(
        &spec.base_url,
        spec.api_key.clone(),
        &spec.model,
        spec.timeout_ms,
        ChatOptions {
            thinking: spec.thinking,
            temperature: spec.temperature,
        },
        guarded,
    )
}

/// `routed`: the deployment default's `[default.chat] routed` (always `false` for a tenant lane).
fn build_chat(
    spec: &ProviderSpec,
    guarded: bool,
    owner: Owner,
    routed: bool,
) -> Result<Arc<ChatLane>> {
    Ok(Arc::new(
        ChatLane::new(Arc::new(chat_client(spec, guarded)?), spec.concurrency)
            .with_meta(meta(owner, spec))
            .with_routed(routed),
    ))
}

/// A tenant registration's lane.
fn build(cap: Capability, spec: &ProviderSpec, limits: LaneLimits, guarded: bool) -> Result<Lane> {
    Ok(match cap {
        Capability::Embed => Lane::Embed(build_embed(spec, limits, guarded, Owner::Tenant)?),
        Capability::Rerank => Lane::Rerank(build_rerank(spec, guarded, Owner::Tenant)?),
        // a tenant's own provider is never the inference-router lane
        Capability::Chat => Lane::Chat(build_chat(spec, guarded, Owner::Tenant, false)?),
    })
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A stored row as a spec (key not yet attached).
fn spec_of(row: &ProviderRow) -> ProviderSpec {
    ProviderSpec {
        base_url: row.base_url.clone(),
        dialect: row.dialect.clone(),
        model: row.model.clone(),
        model_id: row.model_id.clone(),
        dims: row.dims,
        regions: row.regions.clone(),
        zdr: row.zdr,
        timeout_ms: u64::try_from(row.timeout_ms).unwrap_or(1),
        concurrency: usize::try_from(row.concurrency).unwrap_or(1),
        api_key: None,
        thinking: row.thinking,
        temperature: row.temperature,
    }
}

fn rejected(r: Rejection) -> StrataError {
    StrataError::ProviderRejected {
        code: r.code,
        message: r.message,
    }
}

/// `PUT /api/tenant/providers` body: the tenant's whole provider set. A capability left out or
/// `null` is removed.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PutProviders {
    #[serde(default)]
    pub embed: Option<ProviderInput>,
    #[serde(default)]
    pub rerank: Option<ProviderInput>,
    #[serde(default)]
    pub chat: Option<ProviderInput>,
}

/// One capability in a PUT. `api_key` is write-only: omitted = keep the stored key (same
/// `base_url` only). `Debug` redacts it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderInput {
    pub base_url: String,
    pub dialect: String,
    pub model: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub dims: Option<i32>,
    pub regions: Vec<String>,
    pub zdr: bool,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    #[serde(default)]
    pub concurrency: Option<usize>,
    #[serde(default)]
    pub api_key: Option<Secret>,
    /// Chat only: `chat_template_kwargs.enable_thinking` (a vLLM extension); omitted = not sent.
    #[serde(default)]
    pub thinking: Option<bool>,
    /// Chat only; omitted = not sent (the server's default).
    #[serde(default)]
    pub temperature: Option<f64>,
}

impl ProviderInput {
    fn into_spec(self, cap: Capability) -> ProviderSpec {
        ProviderSpec {
            // not trimmed: whitespace is refused by the URL check, never silently fixed
            base_url: self.base_url.trim_end_matches('/').to_string(),
            dialect: self.dialect.trim().to_string(),
            model: self.model.trim().to_string(),
            model_id: self.model_id.map(|m| m.trim().to_string()),
            dims: self.dims,
            regions: self.regions.iter().map(|r| r.trim().to_string()).collect(),
            zdr: self.zdr,
            timeout_ms: self.timeout_ms.unwrap_or(cap.default_timeout_ms()),
            concurrency: self.concurrency.unwrap_or(DEFAULT_CONCURRENCY),
            api_key: self.api_key,
            thinking: self.thinking,
            temperature: self.temperature,
        }
    }
}

/// `PUT /api/stores/{id}/providers` body: the store's own provider set. A capability left out or
/// `null` is removed (the store then inherits it). Any other field is refused (422).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PutStoreProviders {
    #[serde(default)]
    pub embed: Option<ProviderInput>,
    #[serde(default)]
    pub rerank: Option<ProviderInput>,
    /// Chat per store (D92, migration 0017): `/api/ask` and query expansion from this store.
    #[serde(default)]
    pub chat: Option<ProviderInput>,
}

/// `POST /api/tenant/providers/test` body: one provider to try without saving it (06 §3 A3).
/// The same fields as a registration plus `capability`; `regions`, `zdr` and (embed) `dims` are
/// save-time declarations a probe cannot check, so omitted they take the values this deployment
/// accepts (the deployment region, `zdr = true`, 1024) and the save still enforces them. `api_key`
/// omitted = the tenant's stored key of that capability when `base_url` is unchanged. `Debug`
/// redacts the key.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestProvider {
    pub capability: String,
    pub base_url: String,
    pub dialect: String,
    pub model: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub dims: Option<i32>,
    #[serde(default)]
    pub regions: Option<Vec<String>>,
    #[serde(default)]
    pub zdr: Option<bool>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Accepted so a registration body can be tried as is; validated, unused by the one call.
    #[serde(default)]
    pub concurrency: Option<usize>,
    #[serde(default)]
    pub api_key: Option<Secret>,
    #[serde(default)]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

/// `POST /api/tenant/providers/test` answer. A provider that answers badly is `ok: false` with
/// its error class (never a 5xx); `detail` never carries the key or the URL.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProviderTestResult {
    pub ok: bool,
    pub capability: &'static str,
    pub model: String,
    /// Wall time of the probe call.
    pub latency_ms: u64,
    /// Embed only, on success: the vector size the provider returned (the probe refuses any
    /// other than the schema's).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dims: Option<usize>,
    /// On failure: `auth`, `retryable`, `invalid_request`, `invalid_response`, `timeout` or
    /// `internal`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_class: Option<&'static str>,
    pub detail: String,
}

fn parse_capability(s: &str) -> Result<Capability> {
    Capability::ALL
        .into_iter()
        .find(|c| c.as_str() == s.trim())
        .ok_or_else(|| StrataError::ProviderRejected {
            code: "unknown_capability",
            message: format!(
                "capability {s:?} is not one strata_rag serves (embed, rerank or chat)"
            ),
        })
}

/// The capabilities a store can bind (`store_providers`, migrations 0016 and 0017).
pub const STORE_CAPABILITIES: [Capability; 3] =
    [Capability::Embed, Capability::Rerank, Capability::Chat];

/// A store PUT that passed validation and its live probes, keys sealed for the store: what the
/// engine writes in its store transaction ([`crate::db::providers::store_upsert`]).
pub struct PreparedStoreProviders {
    pub set: Vec<(Capability, ProviderSpec, Option<Sealed>)>,
    /// A lane of the new embed registration (for the store's own `desc_vec` when its model
    /// changes); `None` without an embed registration or on fixed providers.
    embed_lane: Option<Arc<EmbedLane>>,
}

impl std::fmt::Debug for PreparedStoreProviders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedStoreProviders")
            .field(
                "set",
                &self.set.iter().map(|(c, s, _)| (c, s)).collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl PreparedStoreProviders {
    /// The embed registration's model identity (`stores.embed_model` format), if one is set.
    pub fn embed_model_id(&self) -> Option<String> {
        self.set
            .iter()
            .find(|(c, _, _)| *c == Capability::Embed)
            .map(|(_, spec, _)| spec.embed_model_id())
    }

    /// The new embed registration as an inference handle (embed only).
    pub fn embed_handle(&self) -> Option<InferenceHandle> {
        self.embed_lane
            .clone()
            .map(|l| InferenceHandle::from_lanes(Some(l), None, None))
    }
}

/// `GET /api/stores/{id}/providers` (and the PUT / DELETE answer): per capability the store
/// binds, which registration a request on this store uses and its settings. Never a key.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StoreProvidersView {
    pub store_id: uuid::Uuid,
    /// The model that made the store's vectors (`stores.embed_model`).
    pub embed_model: String,
    pub embed: StoreCapabilityView,
    pub rerank: StoreCapabilityView,
    pub chat: StoreCapabilityView,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StoreCapabilityView {
    /// `store` (this store's own row), `ancestor` (inherited from `from_store`), `tenant`,
    /// `default` (the dev deployment default) or `unconfigured`.
    pub source: &'static str,
    /// The ancestor whose row is inherited (`source = "ancestor"` only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_store: Option<uuid::Uuid>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderView>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub default: Option<DefaultView>,
}

/// `GET /api/tenant/providers` (and the PUT answer): per capability, where the provider comes
/// from and its settings. Never a key.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProvidersView {
    pub embed: CapabilityView,
    pub rerank: CapabilityView,
    pub chat: CapabilityView,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CapabilityView {
    /// `tenant` (a registration), `default` (the dev deployment default) or `unconfigured`.
    pub source: &'static str,
    /// A tenant registration's settings.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderView>,
    /// A dev default: what it is, never where (no `base_url`, no regions).
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub default: Option<DefaultView>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DefaultView {
    pub dialect: String,
    pub model: String,
    pub dims: Option<i32>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProviderView {
    pub base_url: String,
    pub dialect: String,
    pub model: String,
    pub model_id: Option<String>,
    pub dims: Option<i32>,
    pub regions: Vec<String>,
    pub zdr: bool,
    pub timeout_ms: u64,
    pub concurrency: usize,
    pub has_api_key: bool,
    /// Chat only; omitted when not set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
    /// Chat only; omitted when not set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// RFC 3339 (UTC, microseconds); `null` for a default.
    pub updated_at: Option<String>,
}

impl ProviderView {
    fn of(
        cap: Capability,
        spec: &ProviderSpec,
        has_api_key: bool,
        updated_at: Option<DateTime<Utc>>,
    ) -> Self {
        ProviderView {
            base_url: redact_url(&spec.base_url),
            dialect: spec.dialect.clone(),
            model: spec.model.clone(),
            model_id: spec.model_id.clone(),
            dims: spec.dims,
            regions: spec.regions.clone(),
            zdr: spec.zdr,
            timeout_ms: spec.timeout_ms,
            concurrency: spec.concurrency,
            has_api_key,
            thinking: spec.thinking.filter(|_| cap == Capability::Chat),
            temperature: spec.temperature.filter(|_| cap == Capability::Chat),
            updated_at: updated_at.map(|t| t.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string()),
        }
    }
}

const UNCONFIGURED: CapabilityView = CapabilityView {
    source: "unconfigured",
    provider: None,
    default: None,
};

type DefaultLanes = (
    Option<Arc<EmbedLane>>,
    Option<Arc<RerankLane>>,
    Option<Arc<ChatLane>>,
);

/// The dev defaults' lanes (`None` each in production or when not configured).
fn default_lanes(deployment: &Deployment, limits: LaneLimits) -> Result<DefaultLanes> {
    if deployment.mode != Mode::Dev {
        return Ok((None, None, None));
    }
    Ok((
        deployment
            .default_embed
            .as_ref()
            // operator-set defaults are exempt from the SSRF guard
            .map(|s| build_embed(s, limits, false, Owner::Deployment))
            .transpose()?,
        deployment
            .default_rerank
            .as_ref()
            .map(|s| build_rerank(s, false, Owner::Deployment))
            .transpose()?,
        deployment
            .default_chat
            .as_ref()
            .map(|s| build_chat(s, false, Owner::Deployment, deployment.default_chat_routed))
            .transpose()?,
    ))
}

impl Providers {
    /// The same handle for every tenant (tests, in-process tools).
    pub fn fixed(inf: InferenceHandle) -> Self {
        Providers(Source::Fixed(inf))
    }

    /// Tenant registrations over `pool`, with `deployment`'s policy and (dev) defaults. Refuses to
    /// start when a stored key exists and `cipher` is `None` (`STRATA_PROVIDER_KEY` unset).
    pub async fn registry(
        pool: sqlx::PgPool,
        deployment: Deployment,
        cipher: Option<KeyCipher>,
        limits: LaneLimits,
    ) -> Result<Self> {
        if cipher.is_none() && rows::any_stored_key(&pool).await? {
            return Err(StrataError::Config(
                "tenant provider keys are stored but STRATA_PROVIDER_KEY is unset; set it to the \
                 key they were sealed with"
                    .to_string(),
            ));
        }
        let (default_embed, default_rerank, default_chat) = default_lanes(&deployment, limits)?;
        Ok(Providers(Source::Registry(Box::new(Registry {
            pool,
            deployment,
            cipher,
            limits,
            default_embed,
            default_rerank,
            default_chat,
            cache: Mutex::new(HashMap::new()),
        }))))
    }

    /// Fixed providers with `chat` added; a registry is returned unchanged.
    pub fn with_fixed_chat(self, chat: Arc<dyn crate::inference::Chat>) -> Self {
        match self.0 {
            Source::Fixed(inf) => Providers(Source::Fixed(inf.with_chat(chat))),
            other => Providers(other),
        }
    }

    /// The deployment layer, when this is a registry.
    pub fn deployment(&self) -> Option<&Deployment> {
        match &self.0 {
            Source::Fixed(_) => None,
            Source::Registry(r) => Some(&r.deployment),
        }
    }

    /// `tenant`'s providers for one request that acts on no particular store.
    pub async fn resolve(&self, tenant: Tenant) -> Result<InferenceHandle> {
        self.resolve_for(tenant, None).await
    }

    /// The providers of one request acting on `store` (of `tenant`): per capability the
    /// registration nearest to the store, else the tenant's, else the dev default. `None` = the
    /// tenant's. A store that is not `tenant`'s binds nothing (the caller's lookup answers 404).
    pub async fn resolve_for(
        &self,
        tenant: Tenant,
        store: Option<uuid::Uuid>,
    ) -> Result<InferenceHandle> {
        match &self.0 {
            Source::Fixed(inf) => Ok(inf.clone()),
            Source::Registry(r) => r.resolve(tenant, store).await,
        }
    }

    /// What the unauthenticated `/health` probes: the fixed handle, or the DEFAULT providers of a
    /// dev deployment. Never a tenant's registration (production: nothing).
    pub fn health_handle(&self) -> InferenceHandle {
        match &self.0 {
            Source::Fixed(inf) => inf.clone(),
            Source::Registry(r) => InferenceHandle::from_lanes(
                r.default_embed.clone(),
                r.default_rerank.clone(),
                r.default_chat.clone(),
            ),
        }
    }

    pub async fn view(&self, tenant: Tenant) -> Result<ProvidersView> {
        match &self.0 {
            Source::Fixed(_) => Err(fixed_has_no_registry()),
            Source::Registry(r) => r.view(tenant).await,
        }
    }

    pub async fn replace(&self, tenant: Tenant, body: PutProviders) -> Result<ProvidersView> {
        match &self.0 {
            Source::Fixed(_) => Err(fixed_has_no_registry()),
            Source::Registry(r) => r.replace(tenant, body).await,
        }
    }

    /// The store view; `embed_model` is the store's `stores.embed_model` (the caller read the
    /// store within `tenant`).
    pub async fn store_view(
        &self,
        tenant: Tenant,
        store: uuid::Uuid,
        embed_model: String,
    ) -> Result<StoreProvidersView> {
        match &self.0 {
            Source::Fixed(_) => Err(fixed_has_no_registry()),
            Source::Registry(r) => r.store_view(tenant, store, embed_model).await,
        }
    }

    /// Validates, probes and seals a store PUT (nothing is written: the engine writes it under
    /// the store's row lock).
    pub async fn prepare_store(
        &self,
        tenant: Tenant,
        store: uuid::Uuid,
        body: PutStoreProviders,
    ) -> Result<PreparedStoreProviders> {
        match &self.0 {
            Source::Fixed(_) => Err(fixed_has_no_registry()),
            Source::Registry(r) => r.prepare_store(tenant, store, body).await,
        }
    }

    /// Probes one provider the tenant has not saved: the same validation, SSRF guard and live
    /// probe as `PUT /api/tenant/providers`, storing nothing, caching nothing and metering nothing
    /// (a fresh client, never a metered lane: a probe is not a tenant's use).
    pub async fn test(&self, tenant: Tenant, body: TestProvider) -> Result<ProviderTestResult> {
        match &self.0 {
            Source::Fixed(_) => Err(fixed_has_no_registry()),
            Source::Registry(r) => r.test(tenant, body).await,
        }
    }

    /// Drops this replica's cached lanes of `store`'s rows (after a write, also a failed one).
    pub fn forget_store(&self, store: uuid::Uuid) {
        if let Source::Registry(r) = &self.0 {
            lock(&r.cache).retain(|(s, _), _| *s != Scope::Store(store));
        }
    }
}

fn fixed_has_no_registry() -> StrataError {
    StrataError::Config("this engine has fixed providers and no tenant registry".to_string())
}

impl Registry {
    async fn resolve(&self, tenant: Tenant, store: Option<uuid::Uuid>) -> Result<InferenceHandle> {
        let versions = rows::versions(&self.pool, tenant).await?;
        let bound = match store {
            Some(id) => rows::lineage_versions(&self.pool, tenant, id).await?,
            None => Vec::new(),
        };
        let mut embed = None;
        let mut rerank = None;
        let mut chat = None;
        let mut faults: Vec<(Capability, LaneFault)> = Vec::new();
        'caps: for cap in Capability::ALL {
            // the store's (or its nearest ancestor's) row first, then the tenant's
            let holder = bound
                .iter()
                .find(|(c, _, _)| c == cap.as_str())
                .map(|(_, holder, at)| (Scope::Store(*holder), *at));
            let own = versions
                .iter()
                .find(|(c, _)| c == cap.as_str())
                .map(|(_, at)| (Scope::Tenant(tenant), *at));
            if own.is_none() {
                lock(&self.cache).remove(&(Scope::Tenant(tenant), cap));
            }
            let mut lane = None;
            for (scope, at) in [holder, own].into_iter().flatten() {
                match self.scoped_lane(tenant, scope, cap, at).await {
                    Ok(Some(l)) => {
                        lane = Some(l);
                        break;
                    }
                    // deleted since the version read: the next layer
                    Ok(None) => {}
                    Err(e) => {
                        // this capability is unusable; the others still serve the request
                        tracing::warn!(
                            %tenant,
                            capability = %cap,
                            store = ?store,
                            code = e.code(),
                            "provider registration unusable"
                        );
                        faults.push((cap, LaneFault::of(&e)));
                        continue 'caps;
                    }
                }
            }
            match (cap, lane) {
                (Capability::Embed, Some(Lane::Embed(l))) => embed = Some(l),
                (Capability::Rerank, Some(Lane::Rerank(l))) => rerank = Some(l),
                (Capability::Chat, Some(Lane::Chat(l))) => chat = Some(l),
                (Capability::Embed, _) => embed = self.default_embed.clone(),
                (Capability::Rerank, _) => rerank = self.default_rerank.clone(),
                (Capability::Chat, _) => chat = self.default_chat.clone(),
            }
        }
        Ok(faults.into_iter().fold(
            InferenceHandle::from_lanes(embed, rerank, chat),
            |inf, (cap, fault)| inf.with_fault(cap.as_str(), fault),
        ))
    }

    /// The cached lane of `scope`'s row for the version `at`, else the row loaded, decrypted and
    /// built. `None` when the row was deleted since the version read (the caller falls back).
    async fn scoped_lane(
        &self,
        tenant: Tenant,
        scope: Scope,
        cap: Capability,
        at: DateTime<Utc>,
    ) -> Result<Option<Lane>> {
        if let Some(c) = lock(&self.cache).get(&(scope, cap))
            && c.updated_at == at
        {
            return Ok(Some(c.lane.clone()));
        }
        let row = match scope {
            Scope::Tenant(_) => rows::get(&self.pool, tenant, cap).await?,
            Scope::Store(id) => rows::store_get(&self.pool, tenant, id, cap).await?,
        };
        let Some(row) = row else {
            return Ok(None);
        };
        // A stored row is re-checked before any client is built from it: rows stored before the
        // guard existed, and IP literals (which never pass through the connect-time resolver).
        if self.guarded()
            && let Err(r) = super::ssrf::check_static(cap, &row.base_url)
        {
            tracing::warn!(%tenant, capability = %cap, "stored provider host not allowed; refusing");
            return Err(rejected(r));
        }
        let mut spec = spec_of(&row);
        spec.api_key = match scope {
            Scope::Tenant(_) => self.open_key(tenant, cap, row.sealed_key.as_ref())?,
            Scope::Store(id) => self.open_store_key(tenant, id, cap, row.sealed_key.as_ref())?,
        };
        let lane = build(cap, &spec, self.limits, self.guarded())?;
        let mut cache = lock(&self.cache);
        // a newer version cached by a concurrent request wins
        match cache.get(&(scope, cap)) {
            Some(c) if c.updated_at > row.updated_at => Ok(Some(c.lane.clone())),
            _ => {
                cache.insert(
                    (scope, cap),
                    Cached {
                        updated_at: row.updated_at,
                        lane: lane.clone(),
                    },
                );
                Ok(Some(lane))
            }
        }
    }

    /// Tenant registrations are under the SSRF guard unless the dev-only policy flag lifts it.
    fn guarded(&self) -> bool {
        !self.deployment.policy.allow_internal_provider_hosts
    }

    fn cipher(&self) -> Result<&KeyCipher> {
        self.cipher.as_ref().ok_or_else(|| {
            StrataError::ProviderKeyUnavailable(
                "STRATA_PROVIDER_KEY is not set on this deployment".to_string(),
            )
        })
    }

    fn open_key(
        &self,
        tenant: Tenant,
        cap: Capability,
        sealed: Option<&Sealed>,
    ) -> Result<Option<Secret>> {
        sealed
            .map(|s| self.cipher()?.open(tenant, cap, s))
            .transpose()
    }

    fn open_store_key(
        &self,
        tenant: Tenant,
        store: uuid::Uuid,
        cap: Capability,
        sealed: Option<&Sealed>,
    ) -> Result<Option<Secret>> {
        sealed
            .map(|s| self.cipher()?.open_store(tenant, store, cap, s))
            .transpose()
    }

    async fn view(&self, tenant: Tenant) -> Result<ProvidersView> {
        let stored = rows::list(&self.pool, tenant).await?;
        let view = |cap: Capability| {
            if let Some(row) = stored.iter().find(|r| r.capability == cap.as_str()) {
                return CapabilityView {
                    source: "tenant",
                    provider: Some(ProviderView::of(
                        cap,
                        &spec_of(row),
                        row.sealed_key.is_some(),
                        Some(row.updated_at),
                    )),
                    default: None,
                };
            }
            match (self.deployment.mode, self.deployment.default_for(cap)) {
                (Mode::Dev, Some(spec)) => CapabilityView {
                    source: "default",
                    provider: None,
                    default: Some(DefaultView {
                        dialect: spec.dialect.clone(),
                        model: spec.model.clone(),
                        dims: spec.dims,
                    }),
                },
                _ => UNCONFIGURED,
            }
        };
        Ok(ProvidersView {
            embed: view(Capability::Embed),
            rerank: view(Capability::Rerank),
            chat: view(Capability::Chat),
        })
    }

    async fn replace(&self, tenant: Tenant, body: PutProviders) -> Result<ProvidersView> {
        let cipher = self.cipher()?;
        let stored = rows::list(&self.pool, tenant).await?;
        let mut set: Vec<(Capability, ProviderSpec)> = Vec::new();
        for (cap, input) in [
            (Capability::Embed, body.embed),
            (Capability::Rerank, body.rerank),
            (Capability::Chat, body.chat),
        ] {
            let Some(input) = input else { continue };
            let mut spec = self.checked_spec(cap, input).await?;
            if spec.api_key.is_none() {
                spec.api_key = self.kept_key(tenant, cap, &spec, &stored)?;
            }
            set.push((cap, spec));
        }
        // live probes, concurrently, each bounded by PROBE_TIMEOUT
        futures_util::future::try_join_all(
            set.iter()
                .map(|(cap, spec)| probe(*cap, spec, self.guarded())),
        )
        .await?;

        let sealed: Vec<(Capability, &ProviderSpec, Option<Sealed>)> = set
            .iter()
            .map(|(cap, spec)| {
                let key = spec
                    .api_key
                    .as_ref()
                    .map(|k| cipher.seal(tenant, *cap, k))
                    .transpose()?;
                Ok((*cap, spec, key))
            })
            .collect::<Result<_>>()?;
        let result = async {
            let mut tx = self.pool.begin().await?;
            for cap in Capability::ALL {
                match sealed.iter().find(|(c, _, _)| *c == cap) {
                    Some((_, spec, key)) => {
                        rows::upsert(&mut tx, tenant, cap, spec, key.as_ref()).await?
                    }
                    None => rows::delete(&mut tx, tenant, cap).await?,
                }
            }
            tx.commit().await?;
            Ok::<(), StrataError>(())
        }
        .await;
        // also on failure: a partly applied write must not leave a stale client behind
        lock(&self.cache).retain(|(s, _), _| *s != Scope::Tenant(tenant));
        result?;
        tracing::info!(
            %tenant,
            embed = set.iter().any(|(c, _)| *c == Capability::Embed),
            rerank = set.iter().any(|(c, _)| *c == Capability::Rerank),
            chat = set.iter().any(|(c, _)| *c == Capability::Chat),
            "tenant providers replaced"
        );
        self.view(tenant).await
    }

    /// The stored key a PUT without `api_key` keeps: only for the same `base_url` (a stored key
    /// is never sent to a new host). A changed `base_url` with a stored key and no new key → 422
    /// `api_key_required`; no stored key → none.
    fn kept_key(
        &self,
        tenant: Tenant,
        cap: Capability,
        spec: &ProviderSpec,
        stored: &[ProviderRow],
    ) -> Result<Option<Secret>> {
        match kept_sealed(cap, spec, stored)? {
            Some(sealed) => self.open_key(tenant, cap, Some(sealed)),
            None => Ok(None),
        }
    }

    /// A registration's input as a spec, under every registration rule: the shared validation,
    /// the tenant timeout caps, and (guarded) the SSRF check of its host.
    async fn checked_spec(&self, cap: Capability, input: ProviderInput) -> Result<ProviderSpec> {
        let spec = input.into_spec(cap);
        validate(cap, &spec, self.deployment.region, &self.deployment.policy).map_err(rejected)?;
        let max_timeout = match cap {
            Capability::Embed => MAX_TENANT_EMBED_TIMEOUT_MS,
            Capability::Rerank => MAX_TENANT_RERANK_TIMEOUT_MS,
            Capability::Chat => MAX_TENANT_CHAT_TIMEOUT_MS,
        };
        if spec.timeout_ms > max_timeout {
            return Err(StrataError::ProviderRejected {
                code: "invalid_provider",
                message: format!("{cap}: timeout_ms must be at most {max_timeout}"),
            });
        }
        if self.guarded() {
            super::ssrf::check_registration(cap, &spec.base_url)
                .await
                .map_err(rejected)?;
        }
        Ok(spec)
    }

    async fn test(&self, tenant: Tenant, body: TestProvider) -> Result<ProviderTestResult> {
        let cap = parse_capability(&body.capability)?;
        let region = self.deployment.region;
        let input = ProviderInput {
            base_url: body.base_url,
            dialect: body.dialect,
            model: body.model,
            model_id: body.model_id,
            dims: match cap {
                Capability::Embed => body.dims.or(Some(DIM as i32)),
                _ => body.dims,
            },
            regions: body
                .regions
                .unwrap_or_else(|| vec![region.as_str().to_string()]),
            zdr: body.zdr.unwrap_or(true),
            timeout_ms: body.timeout_ms,
            concurrency: body.concurrency,
            api_key: body.api_key,
            thinking: body.thinking,
            temperature: body.temperature,
        };
        let mut spec = self.checked_spec(cap, input).await?;
        if spec.api_key.is_none() {
            let stored = rows::list(&self.pool, tenant).await?;
            spec.api_key = match kept_sealed(cap, &spec, &stored) {
                Ok(Some(sealed)) => self.open_key(tenant, cap, Some(sealed))?,
                Ok(None) => None,
                Err(StrataError::ProviderRejected { .. }) => {
                    return Err(StrataError::ProviderRejected {
                        code: "secret_required_for_new_host",
                        message: format!(
                            "{cap}: base_url differs from the stored registration, so its key is \
                             not sent there; send api_key"
                        ),
                    });
                }
                Err(e) => return Err(e),
            };
        }
        let started = std::time::Instant::now();
        let outcome = probe_outcome(cap, &spec, self.guarded()).await?;
        let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        tracing::info!(
            %tenant,
            capability = %cap,
            ok = outcome.is_ok(),
            latency_ms,
            "provider test probe"
        );
        Ok(match outcome {
            Ok(()) => ProviderTestResult {
                ok: true,
                capability: cap.as_str(),
                model: spec.model,
                latency_ms,
                dims: (cap == Capability::Embed).then_some(DIM),
                error_class: None,
                detail: format!("{cap}: the provider answered the probe"),
            },
            Err(class) => ProviderTestResult {
                ok: false,
                capability: cap.as_str(),
                model: spec.model,
                latency_ms,
                dims: None,
                error_class: Some(class),
                detail: probe_failed_message(cap, class),
            },
        })
    }

    async fn store_view(
        &self,
        tenant: Tenant,
        store: uuid::Uuid,
        embed_model: String,
    ) -> Result<StoreProvidersView> {
        let bound = rows::lineage_versions(&self.pool, tenant, store).await?;
        let tenant_view = self.view(tenant).await?;
        let mut views = Vec::with_capacity(STORE_CAPABILITIES.len());
        for cap in STORE_CAPABILITIES {
            let holder = bound
                .iter()
                .find(|(c, _, _)| c == cap.as_str())
                .map(|(_, h, _)| *h);
            let row = match holder {
                Some(h) => rows::store_get(&self.pool, tenant, h, cap).await?,
                None => None,
            };
            views.push(match (holder, row) {
                (Some(h), Some(row)) => StoreCapabilityView {
                    source: if h == store { "store" } else { "ancestor" },
                    from_store: (h != store).then_some(h),
                    provider: Some(ProviderView::of(
                        cap,
                        &spec_of(&row),
                        row.sealed_key.is_some(),
                        Some(row.updated_at),
                    )),
                    default: None,
                },
                _ => {
                    let t = match cap {
                        Capability::Embed => &tenant_view.embed,
                        Capability::Rerank => &tenant_view.rerank,
                        Capability::Chat => &tenant_view.chat,
                    };
                    StoreCapabilityView {
                        source: t.source,
                        from_store: None,
                        provider: t.provider.clone(),
                        default: t.default.clone(),
                    }
                }
            });
        }
        let chat = views.pop().unwrap_or(STORE_UNCONFIGURED);
        let rerank = views.pop().unwrap_or(STORE_UNCONFIGURED);
        let embed = views.pop().unwrap_or(STORE_UNCONFIGURED);
        Ok(StoreProvidersView {
            store_id: store,
            embed_model,
            embed,
            rerank,
            chat,
        })
    }

    async fn prepare_store(
        &self,
        tenant: Tenant,
        store: uuid::Uuid,
        body: PutStoreProviders,
    ) -> Result<PreparedStoreProviders> {
        let cipher = self.cipher()?;
        let stored = rows::store_list(&self.pool, tenant, store).await?;
        let mut set: Vec<(Capability, ProviderSpec)> = Vec::new();
        for (cap, input) in [
            (Capability::Embed, body.embed),
            (Capability::Rerank, body.rerank),
            (Capability::Chat, body.chat),
        ] {
            let Some(input) = input else { continue };
            let mut spec = self.checked_spec(cap, input).await?;
            if spec.api_key.is_none() {
                spec.api_key = match kept_sealed(cap, &spec, &stored)? {
                    Some(sealed) => Some(cipher.open_store(tenant, store, cap, sealed)?),
                    None => None,
                };
            }
            set.push((cap, spec));
        }
        futures_util::future::try_join_all(
            set.iter()
                .map(|(cap, spec)| probe(*cap, spec, self.guarded())),
        )
        .await?;
        let embed_lane = set
            .iter()
            .find(|(c, _)| *c == Capability::Embed)
            .map(|(_, spec)| build_embed(spec, self.limits, self.guarded(), Owner::Tenant))
            .transpose()?;
        let set: Vec<(Capability, ProviderSpec, Option<Sealed>)> = set
            .into_iter()
            .map(|(cap, spec)| {
                let key = spec
                    .api_key
                    .as_ref()
                    .map(|k| cipher.seal_store(tenant, store, cap, k))
                    .transpose()?;
                Ok((cap, spec, key))
            })
            .collect::<Result<_>>()?;
        Ok(PreparedStoreProviders { set, embed_lane })
    }
}

const STORE_UNCONFIGURED: StoreCapabilityView = StoreCapabilityView {
    source: "unconfigured",
    from_store: None,
    provider: None,
    default: None,
};

/// The stored sealed key a PUT without `api_key` keeps (see `Registry::kept_key`).
fn kept_sealed<'a>(
    cap: Capability,
    spec: &ProviderSpec,
    stored: &'a [ProviderRow],
) -> Result<Option<&'a Sealed>> {
    let Some(row) = stored.iter().find(|r| r.capability == cap.as_str()) else {
        return Ok(None);
    };
    let Some(sealed) = row.sealed_key.as_ref() else {
        return Ok(None);
    };
    if row.base_url != spec.base_url {
        return Err(StrataError::ProviderRejected {
            code: "api_key_required",
            message: format!(
                "{cap}: base_url changed, so the stored key is not reused; send api_key"
            ),
        });
    }
    Ok(Some(sealed))
}

/// The registration probe: a fresh client (never the cache), one minimal call through the same
/// client code requests use. Failure → 422 `probe_failed` naming the provider error class
/// (`auth`, `retryable`, `invalid_request`, `invalid_response`, `timeout`), never the key or URL.
async fn probe(cap: Capability, spec: &ProviderSpec, guarded: bool) -> Result<()> {
    match probe_outcome(cap, spec, guarded).await? {
        Ok(()) => Ok(()),
        Err(class) => Err(StrataError::ProviderRejected {
            code: "probe_failed",
            message: probe_failed_message(cap, class),
        }),
    }
}

fn probe_failed_message(cap: Capability, class: &str) -> String {
    format!("{cap}: the provider probe failed with error class {class}")
}

/// The probe call itself. `Ok(Err(class))` = the provider answered badly (or not in time);
/// `Err` = the input itself is refused (a client that cannot be built, a host the connect-time
/// SSRF guard refuses): the caller's error, 422.
async fn probe_outcome(
    cap: Capability,
    spec: &ProviderSpec,
    guarded: bool,
) -> Result<std::result::Result<(), &'static str>> {
    let call = async {
        match cap {
            Capability::Embed => {
                EmbedClient::with_guard(
                    &spec.base_url,
                    spec.api_key.clone(),
                    &spec.model,
                    spec.timeout_ms,
                    guarded,
                )
                .map_err(invalid_client)?
                .probe()
                .await
            }
            Capability::Rerank => {
                RerankClient::with_guard(
                    &spec.base_url,
                    spec.api_key.clone(),
                    &spec.model,
                    spec.timeout_ms,
                    guarded,
                )
                .map_err(invalid_client)?
                .probe()
                .await
            }
            Capability::Chat => {
                chat_client(spec, guarded)
                    .map_err(invalid_client)?
                    .probe()
                    .await
            }
        }
    };
    // a chat probe generates (slower); embed / rerank keep the 10 s bound
    let limit = match cap {
        Capability::Chat => Duration::from_millis(spec.timeout_ms).min(CHAT_PROBE_TIMEOUT_MAX),
        _ => PROBE_TIMEOUT,
    };
    match tokio::time::timeout(limit, call).await {
        Ok(Ok(())) => Ok(Ok(())),
        Ok(Err(e @ StrataError::ProviderRejected { .. })) => Err(e),
        Ok(Err(e)) => {
            tracing::info!(capability = %cap, class = error_class(&e), "provider probe failed");
            Ok(Err(error_class(&e)))
        }
        Err(_) => Ok(Err("timeout")),
    }
}

/// A client that cannot be built from the input (e.g. a key with a newline) is the caller's error.
fn invalid_client(e: StrataError) -> StrataError {
    match e {
        StrataError::Config(m) => StrataError::ProviderRejected {
            code: "invalid_provider",
            message: m,
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slow-model budgets nest: expansion fits the slot wait, the slot wait outlasts the
    /// deployment chat timeout (240 s in the deployment's `providers.toml`), and a tenant may register at
    /// least that timeout.
    #[test]
    fn slow_llm_budgets_nest_outer_over_inner() {
        const DEPLOYMENT_CHAT_TIMEOUT: Duration = Duration::from_secs(240);
        assert_eq!(crate::expand::EXPAND_TIMEOUT, Duration::from_secs(180));
        assert_eq!(crate::inference::CHAT_SLOT_WAIT, Duration::from_secs(300));
        assert_eq!(MAX_TENANT_CHAT_TIMEOUT_MS, 300_000);
        assert!(crate::inference::CHAT_SLOT_WAIT >= DEPLOYMENT_CHAT_TIMEOUT);
        assert!(crate::inference::CHAT_SLOT_WAIT >= crate::expand::EXPAND_TIMEOUT);
        assert!(Duration::from_millis(MAX_TENANT_CHAT_TIMEOUT_MS) >= DEPLOYMENT_CHAT_TIMEOUT);
        // the shared validation (the only one a deployment default sees) admits 240 s
        const { assert!(crate::providers::MAX_TIMEOUT_MS >= 240_000) };
        // the registration probe stays a short 1-token call
        assert_eq!(CHAT_PROBE_TIMEOUT_MAX, Duration::from_secs(30));
    }

    #[test]
    fn put_input_debug_redacts_the_key() {
        let body: PutProviders = serde_json::from_value(serde_json::json!({
            "embed": {
                "base_url": "https://e.test/v1/", "dialect": "openai", "model": "m", "dims": 1024,
                "regions": ["eu"], "zdr": true, "api_key": "sk-debug-777"
            }
        }))
        .unwrap();
        let dbg = format!("{body:?}");
        assert!(!dbg.contains("sk-debug-777"), "{dbg}");
        let spec = body.embed.unwrap().into_spec(Capability::Embed);
        assert_eq!(spec.base_url, "https://e.test/v1");
        assert_eq!(
            (spec.timeout_ms, spec.concurrency),
            (30_000, DEFAULT_CONCURRENCY)
        );
    }

    #[test]
    fn unknown_fields_are_refused() {
        // `chat` is a capability since §8.2; a capability strata does not serve is still refused.
        assert!(
            serde_json::from_value::<PutProviders>(serde_json::json!({"transcribe": null}))
                .is_err()
        );
        assert!(
            serde_json::from_value::<PutProviders>(serde_json::json!({
                "rerank": {"base_url": "x", "dialect": "cohere", "model": "m", "regions": [],
                           "zdr": true, "secret": "y"}
            }))
            .is_err()
        );
        let empty: PutProviders = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(empty.embed.is_none() && empty.rerank.is_none() && empty.chat.is_none());
    }

    #[test]
    fn a_store_put_takes_chat_and_refuses_unknown_capabilities() {
        // chat is a store capability since D92
        let body: PutStoreProviders =
            serde_json::from_value(serde_json::json!({"chat": null})).expect("chat is accepted");
        assert!(body.chat.is_none());
        assert!(
            serde_json::from_value::<PutStoreProviders>(serde_json::json!({"summarize": null}))
                .is_err()
        );
        let empty: PutStoreProviders = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(empty.embed.is_none() && empty.rerank.is_none() && empty.chat.is_none());
        assert_eq!(
            STORE_CAPABILITIES,
            [Capability::Embed, Capability::Rerank, Capability::Chat]
        );
    }

    #[test]
    fn a_test_body_redacts_the_key_and_refuses_unknown_fields() {
        let body: TestProvider = serde_json::from_value(serde_json::json!({
            "capability": "rerank", "base_url": "https://r.test/v1", "dialect": "cohere",
            "model": "m", "api_key": "sk-test-body-555"
        }))
        .unwrap();
        assert!(!format!("{body:?}").contains("sk-test-body-555"));
        assert_eq!(
            parse_capability(&body.capability).unwrap(),
            Capability::Rerank
        );
        assert_eq!(
            parse_capability("transcribe").unwrap_err().code(),
            "unknown_capability"
        );
        assert!(
            serde_json::from_value::<TestProvider>(serde_json::json!({
                "capability": "embed", "base_url": "x", "dialect": "openai", "model": "m",
                "secret": "y"
            }))
            .is_err()
        );
    }

    /// `routed` is deployment-only: no tenant or store registration body (nor a test body) takes it.
    #[test]
    fn a_registration_cannot_set_routed() {
        let chat = serde_json::json!({
            "base_url": "https://llm.test/v1", "dialect": "openai", "model": "m",
            "regions": ["eu"], "zdr": true, "routed": true
        });
        let err = serde_json::from_value::<PutProviders>(serde_json::json!({"chat": chat}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("routed"), "{err}");
        assert!(
            serde_json::from_value::<PutStoreProviders>(serde_json::json!({"chat": chat})).is_err()
        );
        let mut test = chat.clone();
        test["capability"] = serde_json::json!("chat");
        assert!(serde_json::from_value::<TestProvider>(test).is_err());
        // the same body without it is accepted: the refusal is about `routed`
        let mut ok = chat.clone();
        ok.as_object_mut().unwrap().remove("routed");
        serde_json::from_value::<PutProviders>(serde_json::json!({"chat": ok})).unwrap();
    }

    fn dev_with_chat(base: &str, routed: &str) -> Deployment {
        Deployment::parse(
            &format!(
                "region = \"eu\"\nmode = \"dev\"\n[policy]\nallowed_dialects = [\"openai\"]\n\
                 [default.chat]\nbase_url = \"{base}\"\ndialect = \"openai\"\n\
                 model = \"local-llm\"\nregions = [\"eu\"]\nzdr = true\n{routed}"
            ),
            &HashMap::new(),
        )
        .unwrap()
    }

    /// Only the deployment default chat lane carries `routed`; a tenant registration of the very
    /// same spec is a plain lane that never sends the router's identity headers
    /// (`X-Effective-Tenant-Id`, project, app, actor), even metered for a fully attributed caller.
    #[tokio::test]
    async fn only_the_default_chat_lane_is_routed_and_a_tenant_lane_never_names_the_tenant() {
        let (base, seen) = crate::inference::tests::tenant_header_mock().await;
        let routed = dev_with_chat(&base, "routed = true\n");
        let (_, _, chat) = default_lanes(&routed, LaneLimits::default()).unwrap();
        assert!(chat.unwrap().is_routed(), "routed = true reaches the lane");
        let direct = dev_with_chat(&base, "");
        let (_, _, chat) = default_lanes(&direct, LaneLimits::default()).unwrap();
        assert!(!chat.unwrap().is_routed(), "routed defaults to false");

        let spec = routed.default_chat.as_ref().unwrap();
        let Lane::Chat(tenant_lane) =
            build(Capability::Chat, spec, LaneLimits::default(), false).unwrap()
        else {
            panic!("a chat registration builds a chat lane");
        };
        assert!(!tenant_lane.is_routed());
        let tenant = Tenant::parse("0195f48b-d000-7000-8000-000000000188").unwrap();
        let (m, _rx) = crate::metering::Metering::capture(Arc::new(crate::text::WhitespaceCounter));
        let h = InferenceHandle::from_lanes(None, None, Some(tenant_lane)).metered(
            m,
            crate::metering::Attribution {
                tenant,
                caller: crate::metering::Caller {
                    project_id: Some(uuid::Uuid::now_v7()),
                    app_id: Some("agent".to_string()),
                    actor_id: Some(uuid::Uuid::now_v7()),
                    license_id: None,
                },
            },
        );
        h.chat(&[crate::clients::chat::ChatMessage::new("user", "hi")], 8)
            .await
            .unwrap();
        assert_eq!(
            seen.lock().unwrap().clone(),
            vec![crate::inference::tests::SeenIdentity::default()]
        );
    }

    #[test]
    fn a_view_serialises_without_a_key_field() {
        let spec = ProviderSpec {
            base_url: "https://e.test/v1".into(),
            dialect: "openai".into(),
            model: "m".into(),
            model_id: None,
            dims: Some(1024),
            regions: vec!["eu".into()],
            zdr: true,
            timeout_ms: 1,
            concurrency: 1,
            api_key: Some(Secret::new("sk-view-888")),
            thinking: None,
            temperature: None,
        };
        let v = ProvidersView {
            embed: CapabilityView {
                source: "tenant",
                provider: Some(ProviderView::of(Capability::Embed, &spec, true, None)),
                default: None,
            },
            rerank: UNCONFIGURED,
            chat: UNCONFIGURED,
        };
        let j = serde_json::to_value(&v).unwrap();
        assert!(!j.to_string().contains("sk-view-888"), "{j}");
        assert_eq!(j["embed"]["source"], "tenant");
        assert_eq!(j["embed"]["has_api_key"], true);
        assert!(j["embed"].get("api_key").is_none());
        assert_eq!(j["rerank"], serde_json::json!({"source": "unconfigured"}));
    }
}
