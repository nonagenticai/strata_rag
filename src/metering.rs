//! Cost events for provider calls, on billing's cost stream (billing-service
//! `docs/EMITTER_GUIDE.md`; spec §8.8), through a transactional outbox (`metering_outbox`, the
//! commonpractice model).
//!
//! One event per provider call that did real work: every embed request (query and ingest), every
//! rerank, every chat completion (`/api/ask`, query expansion). Health probes and the registration
//! probe are not metered, and a query-embedding cache hit makes no call, so it emits nothing.
//!
//! On when `METERING_REDIS_URL` is set (the one switch): then every process (`strata serve` and
//! every CLI command) RECORDS each event as a `metering_outbox` row, and `strata serve` also runs
//! the [`Drainer`], which XADDs pending rows to `METERING_STREAM` (default `billing:costs`,
//! `MAXLEN ~ 100000`) and marks them sent. Rows are written whether or not Redis is reachable; a
//! CLI process never drains, its rows wait for a serving replica.
//!
//! Every paid provider call gets a row, recorded AS THE CALL RETURNS and independent of any
//! business transaction (the commonpractice rule: the metering write never depends on the business
//! write). [`Metering::emit`] puts the event on a bounded in-memory queue (100 000 events, never
//! awaits) and one batch writer INSERTs up to `batch_max` events per statement, at most `linger`
//! (200 ms) after the first. So an ingest or reembed that rolls back, fails or is killed half-way
//! has already recorded (or queued) every call it paid for. Every process flushes the queue to
//! Postgres before it exits (`cli::run_command`; `strata serve` flushes before stopping its
//! drainer).
//!
//! Loss window: an event is lost only if the process dies without a graceful shutdown (crash, OOM
//! kill, SIGKILL) while the event is still in the in-memory queue, i.e. within ~`linger` (200 ms)
//! of the call while Postgres is up, and for as long as Postgres is unreachable (the writer holds
//! and retries); or if the queue is full (100 000 events), or Postgres refuses the row itself
//! (logged with its payload). Both are counted (`Metering::dropped`, `/health` `metering`). Once
//! the row is committed, a crash, a Redis outage or a pod restart loses nothing. Delivery is at
//! least once (a crash between the XADD and the `sent` update re-sends the row); billing
//! deduplicates on `event_key`, which is made once when the call is recorded and stored in the
//! row.
//!
//! Field map (the guide's canonical event, flat string → string):
//! - `cost_source` = `strata-rag` (the EMITTING service), `service_key` = `strata-rag-<call>`
//!   (`embed` / `rerank` / `chat`); the provider is in `provider`, its model in `model`.
//! - `funding`: `tenant_byo` (and `usage_extra.byo = true`) for a tenant-registered provider,
//!   `local_gpu` for self-hosted endpoints (mlx-api, vLLM, cluster-internal / private hosts),
//!   `platform_external` for a deployment default on a public host. The question's embed for
//!   platform-knowledge leaves runs on the platform owner's provider and is funded by the platform
//!   ([`ProviderMeta::platform_funded`]: never `tenant_byo`, even for a provider the platform owner
//!   registered as a tenant), while `tenant_id` stays the calling tenant.
//! - `tenant_id` = the request's tenant; `project_id` = `X-Project-Id` when the caller sent one,
//!   else the tenant-level project id ([`tenant_level_project_id`], billing's owner ruling
//!   2026-10-07 for project-less callers); `actor_id` / `license_id` from `X-Actor-Id` /
//!   `X-License-Id` when present (uuids; anything else is dropped, never guessed); `app_id` from
//!   `X-App-Id`: the calling app's licences-registry key (`agent`, `care-bus`), relayed verbatim
//!   after a header-safety screen, never defaulted and never strata's own (billing resolves a key
//!   or a uuid, `license_resolver.split_asserted_app`).
//! - `input_tokens` / `output_tokens` + `quantity_unit`: embed and chat count `tokens` (with
//!   strata's bge-m3 tokenizer; `usage_extra.tokens_counted_by` says so: providers do not report
//!   usage through these clients), rerank counts `passages` (`usage_extra.tokens` adds the token
//!   count of query + passages).
//! - `event_key` = a uuid v7 made when the call is recorded (billing dedups on it; every
//!   re-delivery carries the same key), `latency_ms`, RFC 3339 UTC `timestamp`, `usage_extra`
//!   JSON.

use std::future::Future;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::HeaderMap;
use serde_json::{Map, Value, json};
use sqlx::types::chrono::{DateTime, Utc};
use sqlx::{PgConnection, PgPool};
use tokio::sync::{Notify, mpsc};
use uuid::Uuid;

use crate::error::{Result, StrataError};
use crate::tenant::Tenant;
use crate::text::TokenCounter;

/// This service's `cost_source` (billing's registry of emitters).
pub const COST_SOURCE: &str = "strata-rag";
/// `METERING_STREAM` when unset: the stream billing consumes.
pub const DEFAULT_STREAM: &str = "billing:costs";
/// `XADD ... MAXLEN ~` (the guide's bound).
pub const STREAM_MAXLEN: usize = 100_000;
/// Events waiting for the batch writer; beyond this they are dropped (counted). The channel
/// allocates as it fills, so the bound costs nothing until an outage fills it (~1 KiB per event).
pub const QUEUE_CAPACITY: usize = 100_000;
/// `usage_extra.tokens_counted_by`: token counts are strata's own, not the provider's.
pub const TOKENS_COUNTED_BY: &str = "strata-bge-m3-tokenizer";

/// The kind of provider call; `service_key` = `strata-rag-<kind>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallKind {
    Embed,
    Rerank,
    Chat,
}

impl CallKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CallKind::Embed => "embed",
            CallKind::Rerank => "rerank",
            CallKind::Chat => "chat",
        }
    }

    pub fn service_key(self) -> &'static str {
        match self {
            CallKind::Embed => "strata-rag-embed",
            CallKind::Rerank => "strata-rag-rerank",
            CallKind::Chat => "strata-rag-chat",
        }
    }
}

/// Whose money (or hardware) paid for a call (billing's `funding`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Funding {
    LocalGpu,
    PlatformExternal,
    TenantByo,
}

impl Funding {
    pub fn as_str(self) -> &'static str {
        match self {
            Funding::LocalGpu => "local_gpu",
            Funding::PlatformExternal => "platform_external",
            Funding::TenantByo => "tenant_byo",
        }
    }
}

/// Billing's `quantity_unit` for `input_tokens` / `output_tokens`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantityUnit {
    Tokens,
    Passages,
}

impl QuantityUnit {
    pub fn as_str(self) -> &'static str {
        match self {
            QuantityUnit::Tokens => "tokens",
            QuantityUnit::Passages => "passages",
        }
    }
}

/// What served a call: `provider` (a short name derived from the endpoint host, see
/// [`provider_name`]), the `model` sent to it, and who pays.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderMeta {
    pub provider: String,
    pub model: String,
    pub funding: Funding,
    /// The endpoint is self-hosted ([`is_local_endpoint`]); decides the funding a platform-funded
    /// call of a tenant-registered provider takes ([`Self::platform_funded`]).
    pub local: bool,
}

impl ProviderMeta {
    /// A provider the tenant registered with its own key: `tenant_byo`.
    pub fn tenant_registered(base_url: &str, model: &str) -> Self {
        ProviderMeta {
            provider: provider_name(base_url),
            model: model.to_string(),
            funding: Funding::TenantByo,
            local: is_local_endpoint(base_url),
        }
    }

    /// The same provider with the call paid by the platform (spec platform-knowledge §7: the
    /// question's embed for platform leaves, on the platform owner's provider): a `tenant_byo`
    /// funding becomes `local_gpu` on a self-hosted endpoint, else `platform_external`; any other
    /// funding is already the platform's and stays.
    pub fn platform_funded(&self) -> Self {
        let funding = match self.funding {
            Funding::TenantByo if self.local => Funding::LocalGpu,
            Funding::TenantByo => Funding::PlatformExternal,
            other => other,
        };
        ProviderMeta {
            funding,
            ..self.clone()
        }
    }

    /// A deployment default (`providers.toml`): `local_gpu` on a self-hosted endpoint, else
    /// `platform_external` (the platform's key paid a vendor).
    pub fn deployment_default(base_url: &str, model: &str) -> Self {
        let local = is_local_endpoint(base_url);
        ProviderMeta {
            provider: provider_name(base_url),
            model: model.to_string(),
            funding: if local {
                Funding::LocalGpu
            } else {
                Funding::PlatformExternal
            },
            local,
        }
    }

    /// An in-process model (fakes, test doubles).
    pub fn in_process(model: &str) -> Self {
        ProviderMeta {
            provider: "in-process".to_string(),
            model: model.to_string(),
            funding: Funding::LocalGpu,
            local: true,
        }
    }

    /// The platform's vLLM (behind the gateway): self-hosted.
    pub fn vllm(model: &str) -> Self {
        ProviderMeta {
            provider: "vllm".to_string(),
            model: model.to_string(),
            funding: Funding::LocalGpu,
            local: true,
        }
    }
}

/// Domain suffixes that only resolve inside our own networks.
const LOCAL_SUFFIXES: [&str; 6] = [
    ".internal",
    ".local",
    ".localhost",
    ".cluster.local",
    ".svc",
    ".ts.net",
];

fn is_local_domain(host: &str) -> bool {
    host == "localhost" || !host.contains('.') || LOCAL_SUFFIXES.iter().any(|s| host.ends_with(s))
}

fn is_local_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                // 100.64/10: the carrier-grade NAT range
                || (a == 100 && (64..128).contains(&b))
        }
        IpAddr::V6(v6) => v6.is_loopback() || (v6.segments()[0] & 0xfe00) == 0xfc00,
    }
}

/// The platform's own API gateway (`api.nonagentic.ai/<service>/...`): the self-hosted services
/// behind it (vLLM, laya) are local GPU, named by their first path segment.
const PLATFORM_DOMAIN: &str = "nonagentic.ai";

fn is_platform_domain(host: &str) -> bool {
    host == PLATFORM_DOMAIN || host.ends_with(&format!(".{PLATFORM_DOMAIN}"))
}

/// `true` for an endpoint on our own hosts: a cluster-internal / private-network / `.local` name, a
/// single-label service name, a private, loopback, link-local or CGNAT address, or the
/// platform's own gateway.
pub fn is_local_endpoint(base_url: &str) -> bool {
    match url::Url::parse(base_url)
        .ok()
        .and_then(|u| u.host().map(|h| h.to_owned()))
    {
        Some(url::Host::Domain(d)) => {
            let d = d.trim_end_matches('.').to_ascii_lowercase();
            is_local_domain(&d) || is_platform_domain(&d)
        }
        Some(url::Host::Ipv4(ip)) => is_local_ip(IpAddr::V4(ip)),
        Some(url::Host::Ipv6(ip)) => is_local_ip(IpAddr::V6(ip)),
        None => false,
    }
}

/// A short provider name from an endpoint URL: the label of a single-label private name
/// (`mlx-api` → `mlx-api`), the registrable name of a public one (`api.voyageai.com` →
/// `voyageai`), the service path segment behind the platform gateway
/// (`api.nonagentic.ai/vllm/v1` → `vllm`), the address itself for an IP. Never the port or
/// credentials.
pub fn provider_name(base_url: &str) -> String {
    let parsed = url::Url::parse(base_url).ok();
    let host = parsed.as_ref().and_then(|u| u.host().map(|h| h.to_owned()));
    match host {
        Some(url::Host::Domain(d)) => {
            let d = d.trim_end_matches('.').to_ascii_lowercase();
            if is_platform_domain(&d) {
                let service = parsed
                    .as_ref()
                    .and_then(|u| u.path_segments())
                    .and_then(|mut segs| segs.next())
                    .filter(|s| !s.is_empty() && *s != "v1");
                return service.unwrap_or("nonagentic").to_ascii_lowercase();
            }
            let labels: Vec<&str> = d.split('.').filter(|l| !l.is_empty()).collect();
            let name = if is_local_domain(&d) {
                labels.first().copied()
            } else if labels.len() >= 2 {
                Some(labels[labels.len() - 2])
            } else {
                labels.first().copied()
            };
            name.unwrap_or("unknown").to_string()
        }
        Some(url::Host::Ipv4(ip)) => ip.to_string(),
        Some(url::Host::Ipv6(ip)) => ip.to_string(),
        None => "unknown".to_string(),
    }
}

/// Billing's tenant-level project id (`tenancy.tenant_level_project_id`): the 32 hex digits of
/// the tenant uuid, reversed, as a uuid. Sent as `project_id` when the caller named no project.
/// An involution. Fixed vector: `0195f48b-d000-7000-8000-000000000100` →
/// `00100000-0000-0008-0007-000db84f5910`.
pub fn tenant_level_project_id(tenant: Uuid) -> Uuid {
    // reversing the hex string = reversing the bytes, then the two nibbles of each byte
    let b = tenant.as_u128().swap_bytes();
    let lo = 0x0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_u128;
    Uuid::from_u128(((b & lo) << 4) | ((b >> 4) & lo))
}

/// The caller's attribution headers, read once per request by the API middleware.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Caller {
    pub project_id: Option<Uuid>,
    /// The calling app's licences-registry key (see [`app_header`]).
    pub app_id: Option<String>,
    pub actor_id: Option<Uuid>,
    pub license_id: Option<Uuid>,
}

/// One header holding one non-nil uuid; absent, repeated or malformed → `None`.
fn uuid_header(h: &HeaderMap, name: &str) -> Option<Uuid> {
    let mut values = h.get_all(name).iter();
    let v = values.next()?;
    if values.next().is_some() {
        return None;
    }
    Uuid::parse_str(v.to_str().ok()?.trim())
        .ok()
        .filter(|u| !u.is_nil())
}

/// Longest relayed `X-App-Id`: registry keys are short slugs, anything longer is not one.
const APP_ID_MAX_LEN: usize = 128;

/// One `X-App-Id` holding a relayable app key: trimmed, non-empty, at most [`APP_ID_MAX_LEN`]
/// printable ASCII characters (it rides the next hop's header and lands in billing), not a nil
/// uuid. Absent, repeated or unrelayable → `None` (an absence, never repaired). Same screen as
/// `kag_common.attribution`.
fn app_header(h: &HeaderMap) -> Option<String> {
    let mut values = h.get_all("x-app-id").iter();
    let v = values.next()?;
    if values.next().is_some() {
        return None;
    }
    let v = v.to_str().ok()?.trim_matches([' ', '\t']);
    let relayable = !v.is_empty()
        && v.len() <= APP_ID_MAX_LEN
        && v.chars().all(|c| (' '..='~').contains(&c))
        && !matches!(Uuid::parse_str(v), Ok(u) if u.is_nil());
    relayable.then(|| v.to_string())
}

impl Caller {
    pub fn from_headers(h: &HeaderMap) -> Self {
        Caller {
            project_id: uuid_header(h, "x-project-id"),
            app_id: app_header(h),
            actor_id: uuid_header(h, "x-actor-id"),
            license_id: uuid_header(h, "x-license-id"),
        }
    }
}

tokio::task_local! {
    static CALLER: Caller;
}

/// Runs `f` (a request) with `caller` as its attribution: every provider call it makes reads it
/// through [`current_caller`]. Provider calls run inside the request's task (joins, not spawns).
pub async fn with_caller<F: Future>(caller: Caller, f: F) -> F::Output {
    CALLER.scope(caller, f).await
}

/// The running request's caller; none outside a request (CLI, background work).
pub fn current_caller() -> Caller {
    CALLER.try_with(Caller::clone).unwrap_or_default()
}

/// Who a metered call is billed to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attribution {
    pub tenant: Tenant,
    pub caller: Caller,
}

impl Attribution {
    pub fn project_id(&self) -> Uuid {
        self.caller
            .project_id
            .unwrap_or_else(|| tenant_level_project_id(self.tenant.id()))
    }
}

/// What one call consumed.
#[derive(Clone, Debug)]
pub struct Usage {
    pub call: CallKind,
    pub input: u64,
    pub output: u64,
    pub unit: QuantityUnit,
    pub latency_ms: u64,
    pub extra: Map<String, Value>,
}

/// One cost event: the flat field list `XADD` writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CostEvent(pub Vec<(&'static str, String)>);

impl CostEvent {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// The event for one call (pure; see the module docs for the field map).
pub fn cost_event(
    attr: &Attribution,
    meta: &ProviderMeta,
    usage: &Usage,
    event_key: Uuid,
    at: DateTime<Utc>,
) -> CostEvent {
    let mut extra = usage.extra.clone();
    extra.insert("call".to_string(), json!(usage.call.as_str()));
    if meta.funding == Funding::TenantByo {
        extra.insert("byo".to_string(), Value::Bool(true));
    }
    let opt = |u: Option<Uuid>| u.map(|u| u.to_string()).unwrap_or_default();
    CostEvent(vec![
        ("tenant_id", attr.tenant.id().to_string()),
        ("project_id", attr.project_id().to_string()),
        ("app_id", attr.caller.app_id.clone().unwrap_or_default()),
        ("actor_id", opt(attr.caller.actor_id)),
        ("license_id", opt(attr.caller.license_id)),
        ("subject_id", String::new()),
        ("service_key", usage.call.service_key().to_string()),
        ("cost_source", COST_SOURCE.to_string()),
        ("provider", meta.provider.clone()),
        ("model", meta.model.clone()),
        ("served_model", meta.model.clone()),
        ("input_tokens", usage.input.to_string()),
        ("output_tokens", usage.output.to_string()),
        ("quantity_unit", usage.unit.as_str().to_string()),
        ("funding", meta.funding.as_str().to_string()),
        ("latency_ms", usage.latency_ms.to_string()),
        ("provider_request_id", String::new()),
        ("usage_extra", Value::Object(extra).to_string()),
        ("event_key", event_key.to_string()),
        // RFC 3339, UTC, microseconds
        ("timestamp", at.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string()),
    ])
}

impl CostEvent {
    /// The event's `event_key` (always set by [`cost_event`]).
    pub fn event_key(&self) -> Option<Uuid> {
        self.get("event_key").and_then(|k| Uuid::parse_str(k).ok())
    }

    /// The event's `tenant_id` (always set by [`cost_event`]).
    pub fn tenant_id(&self) -> Option<Uuid> {
        self.get("tenant_id").and_then(|t| Uuid::parse_str(t).ok())
    }

    /// The field map as a JSON object (`metering_outbox.payload`).
    pub fn payload(&self) -> String {
        let m: Map<String, Value> = self
            .0
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.clone())))
            .collect();
        Value::Object(m).to_string()
    }
}

/// A new event for `usage`: a fresh uuid v7 `event_key` and the current time. The key is made here,
/// once, and travels in the outbox row, so every re-delivery of the event carries the same key.
pub fn record(attr: &Attribution, meta: &ProviderMeta, usage: &Usage) -> CostEvent {
    cost_event(attr, meta, usage, Uuid::now_v7(), Utc::now())
}

async fn insert_rows(
    conn: &mut PgConnection,
    events: &[CostEvent],
) -> std::result::Result<u64, sqlx::Error> {
    let mut tenants = Vec::with_capacity(events.len());
    let mut keys = Vec::with_capacity(events.len());
    let mut payloads = Vec::with_capacity(events.len());
    for e in events {
        let (Some(t), Some(k)) = (e.tenant_id(), e.event_key()) else {
            tracing::error!("metering: a cost event without tenant_id / event_key; not recorded");
            continue;
        };
        tenants.push(t);
        keys.push(k);
        payloads.push(e.payload());
    }
    if keys.is_empty() {
        return Ok(0);
    }
    let done = sqlx::query(
        "INSERT INTO metering_outbox (tenant_id, event_key, payload) \
         SELECT t, k, p::jsonb FROM unnest($1::uuid[], $2::uuid[], $3::text[]) AS u(t, k, p) \
         ON CONFLICT (event_key) DO NOTHING",
    )
    .bind(tenants)
    .bind(keys)
    .bind(payloads)
    .execute(conn)
    .await?;
    Ok(done.rows_affected())
}

/// Inserts `events` into `metering_outbox` in one statement. An event already there (same
/// `event_key`) is skipped, so writing the same events twice records each once.
pub async fn insert_outbox(conn: &mut PgConnection, events: &[CostEvent]) -> Result<u64> {
    Ok(insert_rows(conn, events).await?)
}

/// At most one warning per `LOG_EVERY` for each kind of failure (queue full, outbox insert failed,
/// Redis unavailable, drain failed), carrying how many happened since the last one: an outage must
/// not turn every provider call into a log line.
pub const LOG_EVERY: Duration = Duration::from_secs(30);
/// Bound on one Redis connect or one XADD pipeline.
pub const REDIS_TIMEOUT: Duration = Duration::from_secs(5);
/// Bound on one outbox INSERT by the batch writer.
pub const DB_TIMEOUT: Duration = Duration::from_secs(10);
/// The batch writer's wait after a transient INSERT failure, doubling up to the max; the batch is
/// kept (Postgres down must not lose events).
pub const WRITE_RETRY_MIN: Duration = Duration::from_millis(100);
pub const WRITE_RETRY_MAX: Duration = Duration::from_secs(5);
/// Longest [`DrainerHandle::stop`] waits for the drainer's last pass.
pub const DRAINER_STOP_TIMEOUT: Duration = Duration::from_secs(10);

/// Rate-limits a repeated warning: [`Throttle::hit`] answers `Some(n)` (the `n` occurrences since
/// the last logged one, this one included) at most once per interval, else `None`.
#[derive(Debug)]
pub struct Throttle {
    interval: Duration,
    last: Option<Instant>,
    unlogged: u64,
}

impl Throttle {
    pub fn new(interval: Duration) -> Self {
        Throttle {
            interval,
            last: None,
            unlogged: 0,
        }
    }

    pub fn hit(&mut self, now: Instant) -> Option<u64> {
        self.unlogged += 1;
        match self.last {
            Some(last) if now.saturating_duration_since(last) < self.interval => None,
            _ => {
                self.last = Some(now);
                Some(std::mem::take(&mut self.unlogged))
            }
        }
    }
}

/// The batch writer's knobs: an INSERT is issued when `batch_max` events are waiting or `linger`
/// after the first of them, whichever comes first (or at once on [`Metering::flush`]).
#[derive(Clone, Copy, Debug)]
pub struct WriterConfig {
    pub batch_max: usize,
    pub linger: Duration,
}

impl Default for WriterConfig {
    fn default() -> Self {
        WriterConfig {
            batch_max: 256,
            linger: Duration::from_millis(200),
        }
    }
}

/// Where events go: the outbox batch writer, or a test receiver.
enum Sink {
    Outbox {
        /// Tells the writer to INSERT what it holds now ([`Metering::flush`]).
        wake: Arc<Notify>,
    },
    Capture,
}

/// The recorder: a bounded in-memory queue in front of one Postgres batch writer
/// (`metering_outbox`). [`Metering::emit`] never waits on the database; the drainer
/// ([`Drainer`], in `strata serve`) takes the rows to Redis.
pub struct Metering {
    tx: mpsc::Sender<CostEvent>,
    tc: Arc<dyn TokenCounter>,
    sink: Sink,
    /// Events queued and not yet in the outbox (outbox mode).
    pending: Arc<AtomicUsize>,
    /// Events lost: queue full, or a row Postgres refused (logged with its payload).
    dropped: Arc<AtomicU64>,
    queue_full: Mutex<Throttle>,
}

impl std::fmt::Debug for Metering {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metering").finish_non_exhaustive()
    }
}

/// `METERING_REDIS_URL` must parse as a Redis URL; the error never echoes it (it may carry a
/// password).
pub fn check_redis_url(url: &str) -> Result<redis::Client> {
    redis::Client::open(url).map_err(|_| {
        StrataError::Config("METERING_REDIS_URL is not a valid redis:// URL".to_string())
    })
}

impl Metering {
    /// Starts the batch writer on `pool` (needs a tokio runtime). `tc` counts tokens for events.
    pub fn start(pool: PgPool, tc: Arc<dyn TokenCounter>) -> Arc<Self> {
        Self::start_with(pool, tc, WriterConfig::default())
    }

    pub fn start_with(pool: PgPool, tc: Arc<dyn TokenCounter>, cfg: WriterConfig) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(QUEUE_CAPACITY);
        let pending = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicU64::new(0));
        let wake = Arc::new(Notify::new());
        tokio::spawn(write_batches(
            pool,
            rx,
            Arc::clone(&pending),
            Arc::clone(&dropped),
            Arc::clone(&wake),
            cfg,
        ));
        Arc::new(Metering {
            tx,
            tc,
            sink: Sink::Outbox { wake },
            pending,
            dropped,
            queue_full: Mutex::new(Throttle::new(LOG_EVERY)),
        })
    }

    /// Events go to the returned receiver instead of the outbox (tests).
    pub fn capture(tc: Arc<dyn TokenCounter>) -> (Arc<Self>, mpsc::Receiver<CostEvent>) {
        let (tx, rx) = mpsc::channel(QUEUE_CAPACITY);
        (
            Arc::new(Metering {
                tx,
                tc,
                sink: Sink::Capture,
                pending: Arc::new(AtomicUsize::new(0)),
                dropped: Arc::new(AtomicU64::new(0)),
                queue_full: Mutex::new(Throttle::new(LOG_EVERY)),
            }),
            rx,
        )
    }

    /// Token count of `texts` with strata's tokenizer.
    pub fn tokens<'a>(&self, texts: impl IntoIterator<Item = &'a str>) -> u64 {
        texts.into_iter().map(|t| self.tc.count(t) as u64).sum()
    }

    /// Records the event for `usage` as the call returns; never waits, never fails the caller.
    pub fn emit(&self, attr: &Attribution, meta: &ProviderMeta, usage: Usage) {
        self.enqueue(record(attr, meta, &usage));
    }

    /// Events queued and not yet in the outbox.
    pub fn queued(&self) -> usize {
        self.pending.load(Ordering::SeqCst)
    }

    /// Events lost since start (queue full, or refused by Postgres).
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::SeqCst)
    }

    /// Queues `event` for the batch writer; a full queue drops it (counted, throttled warning).
    pub fn enqueue(&self, event: CostEvent) {
        let outbox = matches!(self.sink, Sink::Outbox { .. });
        if outbox {
            self.pending.fetch_add(1, Ordering::SeqCst);
        }
        let Err(e) = self.tx.try_send(event) else {
            return;
        };
        if outbox {
            self.pending.fetch_sub(1, Ordering::SeqCst);
        }
        self.dropped.fetch_add(1, Ordering::SeqCst);
        let reason = match e {
            mpsc::error::TrySendError::Full(_) => "metering queue full",
            mpsc::error::TrySendError::Closed(_) => "metering writer stopped",
        };
        let logged = self
            .queue_full
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .hit(Instant::now());
        if let Some(dropped) = logged {
            tracing::warn!(
                dropped,
                capacity = QUEUE_CAPACITY,
                "{reason}; cost events dropped"
            );
        }
    }

    /// Has the writer INSERT everything queued now and waits, at most `timeout`, until it is in the
    /// outbox; returns how many events were still not written. Every process calls this before it
    /// exits (`cli::run_command`; `strata serve` before stopping its drainer).
    pub async fn flush(&self, timeout: Duration) -> usize {
        let Sink::Outbox { wake } = &self.sink else {
            return 0;
        };
        let deadline = Instant::now() + timeout;
        loop {
            let left = self.pending.load(Ordering::SeqCst);
            if left == 0 || Instant::now() >= deadline {
                return left;
            }
            wake.notify_one();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// `extra` for an event: the given pairs plus `tokens_counted_by` when tokens were counted here.
pub fn extra(pairs: &[(&str, Value)], counted: bool) -> Map<String, Value> {
    let mut m: Map<String, Value> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    if counted {
        m.insert("tokens_counted_by".to_string(), json!(TOKENS_COUNTED_BY));
    }
    m
}

/// An INSERT failure: transient (Postgres unreachable, pool exhausted, timeout, a lost race, a
/// missing privilege or any other server-side condition an operator can fix) is retried with the
/// batch kept; only a refusal of the DATA itself will fail again on every retry of that row.
fn is_poison(e: &sqlx::Error) -> bool {
    match e {
        sqlx::Error::Database(db) => db.code().is_some_and(|c| is_poison_code(&c)),
        sqlx::Error::Encode(_) => true,
        _ => false,
    }
}

/// SQLSTATE codes that say this row's data is bad: class 22 (data exception, e.g. 22P05 a NUL in
/// jsonb) and class 23 (integrity constraint). Never class 42 (42501 insufficient_privilege: a
/// broken grant would drop every event), 08 (connection), 53 (resources), 57 (shutdown), 40 (lost
/// race): those are retried.
fn is_poison_code(code: &str) -> bool {
    code.starts_with("22") || code.starts_with("23")
}

enum WriteFailure {
    Transient(String),
    Poison(String),
}

async fn write_batch(pool: &PgPool, events: &[CostEvent]) -> std::result::Result<(), WriteFailure> {
    let mut conn = pool
        .acquire()
        .await
        .map_err(|e| WriteFailure::Transient(e.to_string()))?;
    match tokio::time::timeout(DB_TIMEOUT, insert_rows(&mut conn, events)).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) if is_poison(&e) => Err(WriteFailure::Poison(e.to_string())),
        Ok(Err(e)) => Err(WriteFailure::Transient(e.to_string())),
        Err(_) => Err(WriteFailure::Transient("timeout".to_string())),
    }
}

/// The batch writer: collects queued events (up to `batch_max`, for at most `linger`, or until a
/// flush) and INSERTs them in one statement. A transient failure keeps the batch and retries
/// (`WRITE_RETRY_MIN` doubling to `WRITE_RETRY_MAX`). A batch Postgres REFUSES is not retried as a
/// batch (it would block every later event forever): its events are inserted one by one, and an
/// event refused on its own is logged at error with its payload and counted in `dropped`.
async fn write_batches(
    pool: PgPool,
    mut rx: mpsc::Receiver<CostEvent>,
    pending: Arc<AtomicUsize>,
    dropped: Arc<AtomicU64>,
    wake: Arc<Notify>,
    cfg: WriterConfig,
) {
    let batch_max = cfg.batch_max.max(1);
    let mut batch: Vec<CostEvent> = Vec::new();
    let mut open = true;
    let mut retry = WRITE_RETRY_MIN;
    let mut failed = Throttle::new(LOG_EVERY);
    loop {
        if batch.is_empty() {
            if !open {
                return;
            }
            match rx.recv().await {
                Some(e) => batch.push(e),
                None => return,
            }
        }
        if open {
            let deadline = tokio::time::Instant::now() + cfg.linger;
            while batch.len() < batch_max {
                tokio::select! {
                    biased;
                    e = rx.recv() => match e {
                        Some(e) => batch.push(e),
                        None => {
                            open = false;
                            break;
                        }
                    },
                    _ = wake.notified() => break,
                    _ = tokio::time::sleep_until(deadline) => break,
                }
            }
        }
        let transient = match write_batch(&pool, &batch).await {
            Ok(()) => {
                pending.fetch_sub(batch.len(), Ordering::SeqCst);
                batch.clear();
                None
            }
            Err(WriteFailure::Poison(error)) => {
                tracing::warn!(
                    events = batch.len(),
                    %error,
                    "metering: outbox refused a batch; inserting its events one by one"
                );
                let mut transient = None;
                let mut done = 0;
                for e in &batch {
                    match write_batch(&pool, std::slice::from_ref(e)).await {
                        Ok(()) => {}
                        Err(WriteFailure::Poison(error)) => {
                            dropped.fetch_add(1, Ordering::SeqCst);
                            tracing::error!(
                                event_key = e.get("event_key").unwrap_or_default(),
                                payload = %e.payload(),
                                %error,
                                "metering: outbox refused a cost event; dropped"
                            );
                        }
                        Err(WriteFailure::Transient(error)) => {
                            transient = Some(error);
                            break;
                        }
                    }
                    done += 1;
                }
                pending.fetch_sub(done, Ordering::SeqCst);
                batch.drain(..done);
                transient
            }
            Err(WriteFailure::Transient(error)) => Some(error),
        };
        match transient {
            None => retry = WRITE_RETRY_MIN,
            Some(error) => {
                if let Some(failures) = failed.hit(Instant::now()) {
                    tracing::warn!(
                        failures,
                        held = batch.len(),
                        queued = pending.load(Ordering::SeqCst),
                        %error,
                        "metering: outbox insert failed; retrying (events are held in memory)"
                    );
                }
                tokio::time::sleep(retry).await;
                retry = (retry * 2).min(WRITE_RETRY_MAX);
            }
        }
    }
}

/// The outbox backlog: pending rows and the age in seconds of the oldest one (0 when none). An old
/// pending row is undelivered spend.
pub async fn outbox_backlog(pool: &PgPool) -> Result<(i64, i64)> {
    let (pending, oldest): (i64, i64) = sqlx::query_as(
        "SELECT count(*), COALESCE(EXTRACT(EPOCH FROM now() - min(created_at)), 0)::int8 \
         FROM metering_outbox WHERE status = 'pending'",
    )
    .fetch_one(pool)
    .await?;
    Ok((pending, oldest))
}

/// The drainer's knobs.
#[derive(Clone, Copy, Debug)]
pub struct DrainerConfig {
    /// Wait after a pass that found work but less than a full batch, and the first wait once the
    /// outbox is empty.
    pub interval: Duration,
    /// Each pass that finds nothing doubles the wait, from `interval` up to this; any work found
    /// resets it to `interval`. An idle server polls the outbox ~twice a minute, not every second.
    /// (Polling, not LISTEN/NOTIFY: that does not survive a transaction-mode pgbouncer.)
    pub idle_max: Duration,
    /// Rows claimed (and XADDed in one pipeline) per pass.
    pub batch: i64,
    /// A claimed row is leased for this long (`next_attempt_at` moved ahead), so no other drainer
    /// takes it while its XADD runs; a drainer that dies mid-pass releases its rows when the lease
    /// runs out. Must exceed `REDIS_TIMEOUT` plus the time to mark the rows.
    pub lease: Duration,
    /// A row whose XADD failed is retried after `retry_min · 2^(attempts-1)`, at most `retry_max`;
    /// the drainer itself backs off the same way after a failed pass.
    pub retry_min: Duration,
    pub retry_max: Duration,
    /// Sent rows are deleted once older than this.
    pub retention: Duration,
    pub prune_every: Duration,
}

impl Default for DrainerConfig {
    fn default() -> Self {
        DrainerConfig {
            interval: Duration::from_secs(1),
            idle_max: Duration::from_secs(30),
            batch: 500,
            lease: Duration::from_secs(60),
            retry_min: Duration::from_secs(1),
            retry_max: Duration::from_secs(300),
            retention: Duration::from_secs(7 * 24 * 3600),
            prune_every: Duration::from_secs(3600),
        }
    }
}

/// What one drain pass did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DrainReport {
    /// Rows XADDed and marked sent.
    pub sent: usize,
    /// Rows claimed that were not delivered (rescheduled with backoff).
    pub failed: usize,
    /// Why Redis could not be used (`None`: it could).
    pub redis_error: Option<String>,
}

/// Delivers `metering_outbox` to billing's stream. A pass:
/// 1. CLAIMS due pending rows in one autocommitted statement (`FOR UPDATE SKIP LOCKED`, and a
///    lease: `next_attempt_at` moved `lease` ahead), so any number of replicas drain without
///    taking the same row;
/// 2. XADDs them (`MAXLEN ~ STREAM_MAXLEN`, one pipeline) with NO database transaction open;
/// 3. marks each row sent, or reschedules it with backoff, in short statements.
///
/// A row whose payload cannot be read is rescheduled on its own; the rest of the batch goes on.
/// At least once: a crash between the XADD and the marking re-sends the rows once the lease runs
/// out, and billing deduplicates on `event_key`.
pub struct Drainer {
    pool: PgPool,
    client: redis::Client,
    stream: String,
    cfg: DrainerConfig,
    conn: Option<redis::aio::ConnectionManager>,
}

/// One row's payload as XADD field / value pairs; `None` when it is not a JSON object.
fn xadd_fields(payload: &str) -> Option<Vec<(String, String)>> {
    let fields: Map<String, Value> = serde_json::from_str(payload).ok()?;
    Some(
        fields
            .into_iter()
            .map(|(k, v)| {
                let v = match v {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                (k, v)
            })
            .collect(),
    )
}

impl Drainer {
    pub fn new(pool: PgPool, redis_url: &str, stream: &str, cfg: DrainerConfig) -> Result<Self> {
        Ok(Drainer {
            pool,
            client: check_redis_url(redis_url)?,
            stream: stream.to_string(),
            cfg,
            conn: None,
        })
    }

    async fn connection(
        &mut self,
    ) -> std::result::Result<&mut redis::aio::ConnectionManager, String> {
        if self.conn.is_none() {
            // one connect attempt per pass: the drainer loop does its own backoff
            let config = redis::aio::ConnectionManagerConfig::new()
                .set_number_of_retries(1)
                .set_connection_timeout(Some(REDIS_TIMEOUT))
                .set_response_timeout(Some(REDIS_TIMEOUT));
            let connected = tokio::time::timeout(
                REDIS_TIMEOUT,
                redis::aio::ConnectionManager::new_with_config(self.client.clone(), config),
            )
            .await;
            match connected {
                Ok(Ok(c)) => self.conn = Some(c),
                Ok(Err(e)) => return Err(format!("connect: {:?}", e.kind())),
                Err(_) => return Err("connect: timeout".to_string()),
            }
        }
        self.conn
            .as_mut()
            .ok_or_else(|| "connect: no connection".to_string())
    }

    async fn reschedule(&self, ids: &[Uuid], error: &str) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        sqlx::query(
            "UPDATE metering_outbox \
             SET attempts = attempts + 1, last_error = $2, \
                 next_attempt_at = now() + make_interval(secs => \
                   LEAST($4::float8, $3::float8 * power(2, attempts))) \
             WHERE id = ANY($1) AND status = 'pending'",
        )
        .bind(ids)
        .bind(error)
        .bind(self.cfg.retry_min.as_secs_f64())
        .bind(self.cfg.retry_max.as_secs_f64())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// One pass (see [`Drainer`]). A database error is `Err`; a Redis failure is in the report.
    pub async fn drain_once(&mut self) -> Result<DrainReport> {
        // no row is claimed while Redis cannot even be reached
        if let Err(e) = self.connection().await {
            return Ok(DrainReport {
                redis_error: Some(e),
                ..DrainReport::default()
            });
        }
        // 1. claim + lease, committed at once: no transaction stays open across Redis
        let rows: Vec<(Uuid, String)> = sqlx::query_as(
            "UPDATE metering_outbox o \
             SET next_attempt_at = now() + make_interval(secs => $2::float8) \
             FROM (SELECT id FROM metering_outbox \
                   WHERE status = 'pending' AND next_attempt_at <= now() \
                   ORDER BY next_attempt_at, id \
                   LIMIT $1 \
                   FOR UPDATE SKIP LOCKED) c \
             WHERE o.id = c.id \
             RETURNING o.id, o.payload::text",
        )
        .bind(self.cfg.batch.max(1))
        .bind(self.cfg.lease.as_secs_f64())
        .fetch_all(&self.pool)
        .await?;
        if rows.is_empty() {
            return Ok(DrainReport::default());
        }
        // 2. one XADD per readable row; an unreadable one is set aside, the rest go on
        let mut ids = Vec::with_capacity(rows.len());
        let mut unreadable = Vec::new();
        let mut pipe = redis::pipe();
        pipe.ignore_errors();
        for (id, payload) in &rows {
            let Some(fields) = xadd_fields(payload) else {
                unreadable.push(*id);
                continue;
            };
            pipe.cmd("XADD")
                .arg(&self.stream)
                .arg("MAXLEN")
                .arg("~")
                .arg(STREAM_MAXLEN)
                .arg("*");
            for (k, v) in fields {
                pipe.arg(k).arg(v);
            }
            ids.push(*id);
        }
        let mut sent = Vec::new();
        let mut failed: Vec<Uuid> = Vec::new();
        let mut redis_error = None;
        if !ids.is_empty() {
            let answered = match self.connection().await {
                Ok(conn) => {
                    match tokio::time::timeout(
                        REDIS_TIMEOUT,
                        pipe.query_async::<Vec<redis::RedisResult<String>>>(conn),
                    )
                    .await
                    {
                        Ok(Ok(results)) => Ok(results),
                        Ok(Err(e)) => Err(format!("xadd: {:?}", e.kind())),
                        Err(_) => Err("xadd: timeout".to_string()),
                    }
                }
                Err(e) => Err(e),
            };
            match answered {
                Ok(results) => {
                    for (i, id) in ids.iter().enumerate() {
                        match results.get(i) {
                            Some(Ok(_)) => sent.push(*id),
                            Some(Err(e)) => {
                                redis_error = Some(format!("xadd: {:?}", e.kind()));
                                failed.push(*id);
                            }
                            None => failed.push(*id),
                        }
                    }
                }
                Err(error) => {
                    // a failed or hung pipeline: rebuild the connection for the next pass
                    self.conn = None;
                    redis_error = Some(error);
                    failed = ids.clone();
                }
            }
        }
        // 3. mark, in short statements
        if !sent.is_empty() {
            sqlx::query(
                "UPDATE metering_outbox \
                 SET status = 'sent', sent_at = now(), last_error = NULL \
                 WHERE id = ANY($1) AND status = 'pending'",
            )
            .bind(sent.as_slice())
            .execute(&self.pool)
            .await?;
        }
        let error = redis_error
            .clone()
            .unwrap_or_else(|| "xadd: no reply".to_string());
        self.reschedule(&failed, &error).await?;
        if !unreadable.is_empty() {
            tracing::error!(
                rows = unreadable.len(),
                "metering: outbox rows whose payload is not a JSON object; rescheduled"
            );
            self.reschedule(&unreadable, "payload is not a JSON object")
                .await?;
        }
        Ok(DrainReport {
            sent: sent.len(),
            failed: failed.len() + unreadable.len(),
            redis_error,
        })
    }

    /// Deletes sent rows older than the retention window; returns how many.
    pub async fn prune(&self) -> Result<u64> {
        let done = sqlx::query(
            "DELETE FROM metering_outbox WHERE id IN ( \
               SELECT id FROM metering_outbox \
               WHERE status = 'sent' AND sent_at < now() - make_interval(secs => $1::float8) \
               LIMIT 10000)",
        )
        .bind(self.cfg.retention.as_secs_f64())
        .execute(&self.pool)
        .await?;
        Ok(done.rows_affected())
    }

    /// Runs passes until [`DrainerHandle::stop`], which gets one last pass (the stop wakes any
    /// wait). The wait before the next pass is [`Pace`]'s: none after a full batch, `interval`
    /// after a partial one, `interval` doubling to `idle_max` while the outbox stays empty, and a
    /// backoff after a failed pass (database or Redis; `retry_min` doubling to `retry_max`);
    /// failures are logged at most once per `LOG_EVERY` with their count. Sent rows are pruned
    /// every `prune_every`.
    pub fn spawn(mut self) -> DrainerHandle {
        let stop = Arc::new(Notify::new());
        let stopped = Arc::clone(&stop);
        let task = tokio::spawn(async move {
            let mut pace = Pace::new(self.cfg);
            let mut failures = Throttle::new(LOG_EVERY);
            let mut pruned_at: Option<Instant> = None;
            let mut last = false;
            loop {
                let wait = match self.drain_once().await {
                    Ok(
                        report @ DrainReport {
                            redis_error: None, ..
                        },
                    ) => pace.delivered(&report),
                    Ok(DrainReport {
                        redis_error: Some(error),
                        failed,
                        ..
                    }) => {
                        if let Some(n) = failures.hit(Instant::now()) {
                            tracing::warn!(
                                failures = n,
                                rescheduled = failed,
                                %error,
                                "metering: redis unavailable; cost events stay in the outbox"
                            );
                        }
                        pace.failed()
                    }
                    Err(e) => {
                        if let Some(n) = failures.hit(Instant::now()) {
                            tracing::warn!(failures = n, error = %e, "metering: outbox drain failed");
                        }
                        pace.failed()
                    }
                };
                if last {
                    return;
                }
                if pruned_at.is_none_or(|t| t.elapsed() >= self.cfg.prune_every) {
                    pruned_at = Some(Instant::now());
                    match self.prune().await {
                        Ok(0) => {}
                        Ok(n) => tracing::debug!(pruned = n, "metering: sent outbox rows pruned"),
                        Err(e) => tracing::warn!(error = %e, "metering: outbox prune failed"),
                    }
                }
                tokio::select! {
                    // stop: one more pass for what the shutdown flush just wrote, then return
                    _ = stopped.notified() => last = true,
                    _ = tokio::time::sleep(wait) => {}
                }
            }
        });
        DrainerHandle { stop, task }
    }
}

/// The drainer loop's wait before its next pass.
#[derive(Debug)]
struct Pace {
    cfg: DrainerConfig,
    /// Next wait after an empty pass: `interval` doubling to `idle_max`.
    idle: Duration,
    /// Next wait after a failed pass: `retry_min` doubling to `retry_max`.
    backoff: Duration,
}

impl Pace {
    fn new(cfg: DrainerConfig) -> Self {
        Pace {
            cfg,
            idle: cfg.interval,
            backoff: cfg.retry_min,
        }
    }

    /// After a pass that reached Redis: no wait after a full batch, `interval` after any work,
    /// the growing idle wait after an empty pass.
    fn delivered(&mut self, report: &DrainReport) -> Duration {
        self.backoff = self.cfg.retry_min;
        let claimed = report.sent + report.failed;
        if claimed == 0 {
            let wait = self.idle;
            self.idle = (self.idle * 2).min(self.cfg.idle_max.max(self.cfg.interval));
            return wait;
        }
        self.idle = self.cfg.interval;
        if (report.sent as i64) >= self.cfg.batch.max(1) {
            Duration::ZERO
        } else {
            self.cfg.interval
        }
    }

    /// After a failed pass (database or Redis).
    fn failed(&mut self) -> Duration {
        let wait = self.backoff;
        self.backoff = (self.backoff * 2).min(self.cfg.retry_max);
        wait
    }
}

/// A running [`Drainer`].
pub struct DrainerHandle {
    stop: Arc<Notify>,
    task: tokio::task::JoinHandle<()>,
}

impl DrainerHandle {
    /// Stops after one last pass, waiting at most `DRAINER_STOP_TIMEOUT` (then the task is
    /// aborted; rows it had leased are taken again by any drainer once the lease runs out).
    pub async fn stop(self) {
        self.stop.notify_one();
        let abort = self.task.abort_handle();
        if tokio::time::timeout(DRAINER_STOP_TIMEOUT, self.task)
            .await
            .is_err()
        {
            abort.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn tenant() -> Tenant {
        Tenant::parse("0195f48b-d000-7000-8000-000000000100").unwrap()
    }

    #[test]
    fn tenant_level_project_id_matches_billings_vector() {
        let t = Uuid::parse_str("0195f48b-d000-7000-8000-000000000100").unwrap();
        let p = tenant_level_project_id(t);
        assert_eq!(p.to_string(), "00100000-0000-0008-0007-000db84f5910");
        assert_eq!(tenant_level_project_id(p), t, "an involution");
        // the definition itself: the hex digits reversed
        let other = Uuid::now_v7();
        let rev: String = other.simple().to_string().chars().rev().collect();
        assert_eq!(tenant_level_project_id(other).simple().to_string(), rev);
    }

    fn usage(call: CallKind) -> Usage {
        Usage {
            call,
            input: 12,
            output: 3,
            unit: QuantityUnit::Tokens,
            latency_ms: 41,
            extra: extra(&[("texts", json!(2))], true),
        }
    }

    #[test]
    fn the_field_map_follows_the_emitter_guide() {
        let attr = Attribution {
            tenant: tenant(),
            caller: Caller::default(),
        };
        let meta = ProviderMeta::deployment_default("http://mlx-api:9131/v1", "bge-m3");
        let key = Uuid::now_v7();
        let at = DateTime::parse_from_rfc3339("2026-10-16T12:00:00.5Z")
            .unwrap()
            .with_timezone(&Utc);
        let e = cost_event(&attr, &meta, &usage(CallKind::Embed), key, at);
        let want: [(&str, &str); 17] = [
            ("tenant_id", "0195f48b-d000-7000-8000-000000000100"),
            // no X-Project-Id: the tenant-level project, never the tenant id itself
            ("project_id", "00100000-0000-0008-0007-000db84f5910"),
            ("app_id", ""),
            ("actor_id", ""),
            ("license_id", ""),
            ("subject_id", ""),
            ("service_key", "strata-rag-embed"),
            ("cost_source", "strata-rag"),
            ("provider", "mlx-api"),
            ("model", "bge-m3"),
            ("served_model", "bge-m3"),
            ("input_tokens", "12"),
            ("output_tokens", "3"),
            ("quantity_unit", "tokens"),
            ("funding", "local_gpu"),
            ("latency_ms", "41"),
            ("timestamp", "2026-10-16T12:00:00.500000Z"),
        ];
        for (k, v) in want {
            assert_eq!(e.get(k), Some(v), "{k}");
        }
        assert_eq!(e.get("event_key"), Some(key.to_string().as_str()));
        assert_eq!(e.get("provider_request_id"), Some(""));
        let extra: Value = serde_json::from_str(e.get("usage_extra").unwrap()).unwrap();
        assert_eq!(
            extra,
            json!({"call": "embed", "texts": 2, "tokens_counted_by": TOKENS_COUNTED_BY})
        );
        // every key once, every value a string the stream can carry
        let mut keys: Vec<&str> = e.0.iter().map(|(k, _)| *k).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), e.0.len());
        assert_eq!(e.0.len(), 20);
    }

    #[test]
    fn a_tenant_registered_provider_is_byo_and_caller_headers_win() {
        let project = Uuid::now_v7();
        let (actor, licence) = (Uuid::now_v7(), Uuid::now_v7());
        let attr = Attribution {
            tenant: tenant(),
            caller: Caller {
                project_id: Some(project),
                app_id: Some("agent".to_string()),
                actor_id: Some(actor),
                license_id: Some(licence),
            },
        };
        let meta = ProviderMeta::tenant_registered("https://api.voyageai.com/v1", "rerank-2");
        let mut u = usage(CallKind::Rerank);
        u.unit = QuantityUnit::Passages;
        let e = cost_event(&attr, &meta, &u, Uuid::now_v7(), Utc::now());
        assert_eq!(e.get("project_id"), Some(project.to_string().as_str()));
        assert_eq!(e.get("app_id"), Some("agent"));
        assert_eq!(e.get("actor_id"), Some(actor.to_string().as_str()));
        assert_eq!(e.get("license_id"), Some(licence.to_string().as_str()));
        assert_eq!(e.get("funding"), Some("tenant_byo"));
        assert_eq!(e.get("provider"), Some("voyageai"));
        assert_eq!(e.get("service_key"), Some("strata-rag-rerank"));
        assert_eq!(e.get("quantity_unit"), Some("passages"));
        let extra: Value = serde_json::from_str(e.get("usage_extra").unwrap()).unwrap();
        assert_eq!(extra["byo"], json!(true));
        let ts = e.get("timestamp").unwrap();
        assert!(
            DateTime::parse_from_rfc3339(ts).is_ok() && ts.ends_with('Z'),
            "{ts}"
        );
    }

    #[test]
    fn funding_and_provider_names_from_the_endpoint() {
        for (url, name, local) in [
            ("http://provider.example.internal:9131/v1", "provider", true),
            ("http://127.0.0.1:9130", "127.0.0.1", true),
            ("http://100.64.0.9:9131/v1", "100.64.0.9", true),
            ("http://10.0.0.5/v1", "10.0.0.5", true),
            ("http://[::1]:9130", "::1", true),
            ("http://embed-svc:8080", "embed-svc", true),
            ("http://tei.example.svc.cluster.local/v1", "tei", true),
            ("https://api.voyageai.com/v1", "voyageai", false),
            ("https://api.cohere.com/v2", "cohere", false),
            ("https://user:pw@api.openai.com/v1", "openai", false),
            ("https://8.8.8.8/v1", "8.8.8.8", false),
            ("https://api.nonagentic.ai/vllm/v1", "vllm", true),
            ("https://api.nonagentic.ai/v1", "nonagentic", true),
            ("https://nonagentic.ai.evil.com/vllm", "evil", false),
            ("not a url", "unknown", false),
        ] {
            assert_eq!(provider_name(url), name, "{url}");
            assert_eq!(is_local_endpoint(url), local, "{url}");
            let want = if local {
                Funding::LocalGpu
            } else {
                Funding::PlatformExternal
            };
            assert_eq!(
                ProviderMeta::deployment_default(url, "m").funding,
                want,
                "{url}"
            );
            assert_eq!(
                ProviderMeta::tenant_registered(url, "m").funding,
                Funding::TenantByo,
                "{url}"
            );
            // the platform pays for its own registrations' platform-knowledge embeds
            assert_eq!(
                ProviderMeta::tenant_registered(url, "m")
                    .platform_funded()
                    .funding,
                want,
                "{url}"
            );
            assert_eq!(
                ProviderMeta::deployment_default(url, "m")
                    .platform_funded()
                    .funding,
                want,
                "{url}"
            );
        }
    }

    #[test]
    fn caller_headers_are_single_non_nil_uuids() {
        let p = Uuid::now_v7();
        let mut h = HeaderMap::new();
        h.insert(
            "x-project-id",
            HeaderValue::from_str(&p.to_string()).unwrap(),
        );
        h.insert("x-app-id", HeaderValue::from_static(" care-bus "));
        h.insert(
            "x-actor-id",
            HeaderValue::from_static("00000000-0000-0000-0000-000000000000"),
        );
        h.append(
            "x-license-id",
            HeaderValue::from_str(&p.to_string()).unwrap(),
        );
        h.append(
            "x-license-id",
            HeaderValue::from_str(&p.to_string()).unwrap(),
        );
        let c = Caller::from_headers(&h);
        assert_eq!(
            c,
            Caller {
                project_id: Some(p),
                app_id: Some("care-bus".to_string()),
                actor_id: None,
                license_id: None,
            }
        );
        assert_eq!(Caller::from_headers(&HeaderMap::new()), Caller::default());
    }

    /// `X-App-Id` is the caller's registry key, relayed verbatim; what cannot ride a header (or is
    /// not one value) is an absence, never repaired.
    #[test]
    fn an_app_key_is_relayed_only_when_it_is_one_header_safe_value() {
        let app = |values: &[&str]| {
            let mut h = HeaderMap::new();
            for v in values {
                h.append("x-app-id", HeaderValue::from_bytes(v.as_bytes()).unwrap());
            }
            Caller::from_headers(&h).app_id
        };
        assert_eq!(app(&["agent"]), Some("agent".to_string()));
        let id = Uuid::now_v7().to_string();
        assert_eq!(app(&[id.as_str()]), Some(id.clone()));
        assert_eq!(app(&["   "]), None);
        assert_eq!(app(&["agent", "agent"]), None);
        let (long, longest) = ("a".repeat(129), "a".repeat(128));
        assert_eq!(app(&[long.as_str()]), None);
        assert_eq!(app(&[longest.as_str()]), Some(longest.clone()));
        assert_eq!(app(&["caf\u{e9}"]), None);
        // a tab passes HeaderValue::to_str; only the printable-ASCII screen refuses it
        assert_eq!(app(&["care\tbus"]), None);
        assert_eq!(app(&["00000000-0000-0000-0000-000000000000"]), None);
    }

    #[tokio::test]
    async fn the_caller_is_scoped_to_the_request() {
        assert_eq!(current_caller(), Caller::default());
        let c = Caller {
            project_id: Some(Uuid::now_v7()),
            ..Caller::default()
        };
        let seen = with_caller(c.clone(), async {
            tokio::task::yield_now().await;
            current_caller()
        })
        .await;
        assert_eq!(seen, c);
        assert_eq!(current_caller(), Caller::default());
    }

    #[test]
    fn the_drainer_backs_off_while_idle_and_resets_on_work() {
        let cfg = DrainerConfig {
            interval: Duration::from_secs(1),
            idle_max: Duration::from_secs(30),
            batch: 10,
            retry_min: Duration::from_secs(1),
            retry_max: Duration::from_secs(300),
            ..DrainerConfig::default()
        };
        let report = |sent: usize, failed: usize| DrainReport {
            sent,
            failed,
            redis_error: None,
        };
        let secs = |s: u64| Duration::from_secs(s);
        let mut pace = Pace::new(cfg);
        let idle: Vec<Duration> = (0..7).map(|_| pace.delivered(&report(0, 0))).collect();
        assert_eq!(
            idle,
            [
                secs(1),
                secs(2),
                secs(4),
                secs(8),
                secs(16),
                secs(30),
                secs(30)
            ]
        );
        // work (sent or rescheduled rows) resets the idle wait to `interval`
        assert_eq!(pace.delivered(&report(3, 0)), secs(1));
        assert_eq!(pace.delivered(&report(0, 0)), secs(1));
        assert_eq!(pace.delivered(&report(0, 0)), secs(2));
        assert_eq!(pace.delivered(&report(0, 2)), secs(1));
        assert_eq!(pace.delivered(&report(0, 0)), secs(1));
        // a full batch: the next pass at once, and the idle wait starts over
        pace.delivered(&report(0, 0));
        assert_eq!(pace.delivered(&report(10, 0)), Duration::ZERO);
        assert_eq!(pace.delivered(&report(0, 0)), secs(1));
        // failures back off on their own and a delivered pass resets them
        assert_eq!(
            (pace.failed(), pace.failed(), pace.failed()),
            (secs(1), secs(2), secs(4))
        );
        pace.delivered(&report(0, 0));
        assert_eq!(pace.failed(), secs(1));
    }

    #[tokio::test]
    async fn a_bad_redis_url_is_a_config_error_that_does_not_echo_it() {
        let err = check_redis_url("nope://user:secret@host").unwrap_err();
        assert!(matches!(err, StrataError::Config(_)), "{err:?}");
        assert!(!err.to_string().contains("secret"));
        assert!(check_redis_url("redis://127.0.0.1:6379/").is_ok());
        let pool = sqlx::postgres::PgPoolOptions::new()
            .connect_lazy("postgres://u:p@127.0.0.1:1/db")
            .unwrap();
        assert!(Drainer::new(pool, "nope://x", DEFAULT_STREAM, DrainerConfig::default()).is_err());
    }

    #[tokio::test]
    async fn emit_never_blocks_and_capture_sees_the_event() {
        let tc: Arc<dyn TokenCounter> = Arc::new(crate::text::WhitespaceCounter);
        let (m, mut rx) = Metering::capture(tc);
        assert_eq!(m.tokens(["one two", "three"]), 3);
        let attr = Attribution {
            tenant: tenant(),
            caller: Caller::default(),
        };
        m.emit(
            &attr,
            &ProviderMeta::vllm("local-llm"),
            usage(CallKind::Chat),
        );
        let e = rx.recv().await.unwrap();
        assert_eq!(e.get("service_key"), Some("strata-rag-chat"));
        assert_eq!(e.get("provider"), Some("vllm"));
        assert_eq!(e.get("funding"), Some("local_gpu"));
        assert_eq!(
            Uuid::parse_str(e.get("event_key").unwrap())
                .unwrap()
                .get_version_num(),
            7
        );
    }

    #[test]
    fn the_throttle_logs_once_per_interval_with_the_dropped_count() {
        let mut t = Throttle::new(Duration::from_secs(30));
        let t0 = Instant::now();
        assert_eq!(t.hit(t0), Some(1), "the first drop is logged");
        assert_eq!(t.hit(t0 + Duration::from_secs(1)), None);
        assert_eq!(t.hit(t0 + Duration::from_secs(29)), None);
        assert_eq!(
            t.hit(t0 + Duration::from_secs(31)),
            Some(3),
            "two suppressed plus this one"
        );
        assert_eq!(t.hit(t0 + Duration::from_secs(32)), None);
    }

    /// Postgres down: `emit` still never waits (the request path does not touch the database),
    /// and `flush` gives up at its timeout and reports what was not written.
    #[tokio::test]
    async fn database_down_never_blocks_emit_and_flush_is_bounded() {
        let tc: Arc<dyn TokenCounter> = Arc::new(crate::text::WhitespaceCounter);
        // nothing listens on port 1
        let pool = sqlx::postgres::PgPoolOptions::new()
            .acquire_timeout(Duration::from_millis(200))
            .connect_lazy("postgres://u:p@127.0.0.1:1/db")
            .unwrap();
        let m = Metering::start(pool, tc);
        let attr = Attribution {
            tenant: tenant(),
            caller: Caller::default(),
        };
        let t = Instant::now();
        for _ in 0..50 {
            m.emit(
                &attr,
                &ProviderMeta::in_process("m"),
                usage(CallKind::Embed),
            );
        }
        assert!(t.elapsed() < Duration::from_millis(500), "emit never waits");
        let t = Instant::now();
        let left = m.flush(Duration::from_millis(500)).await;
        assert_eq!(left, 50, "nothing reached the outbox");
        assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    }

    #[test]
    fn the_payload_is_the_field_map_as_a_json_object() {
        let attr = Attribution {
            tenant: tenant(),
            caller: Caller::default(),
        };
        let e = record(
            &attr,
            &ProviderMeta::vllm("local-llm"),
            &usage(CallKind::Chat),
        );
        let v: Value = serde_json::from_str(&e.payload()).unwrap();
        let m = v.as_object().unwrap();
        assert_eq!(m.len(), 20);
        for (k, val) in &e.0 {
            assert_eq!(m[*k], json!(val), "{k}");
        }
        assert_eq!(e.tenant_id(), Some(tenant().id()));
        assert_eq!(e.event_key().unwrap().get_version_num(), 7);
    }

    #[test]
    fn only_bad_data_sqlstates_are_poison() {
        for code in ["22P05", "22001", "22P02", "23502", "23514"] {
            assert!(is_poison_code(code), "{code}");
        }
        for code in [
            "42501", "42P01", "42804", "08006", "53300", "57P01", "40P01", "XX000",
        ] {
            assert!(!is_poison_code(code), "{code}");
        }
    }

    #[test]
    fn an_unreadable_payload_yields_no_xadd_fields() {
        assert_eq!(xadd_fields("[1, 2]"), None);
        assert_eq!(xadd_fields("not json"), None);
        assert_eq!(
            xadd_fields(r#"{"a": "x", "n": 3}"#),
            Some(vec![
                ("a".to_string(), "x".to_string()),
                ("n".to_string(), "3".to_string())
            ])
        );
    }

    #[tokio::test]
    async fn flush_without_a_writer_returns_at_once() {
        let tc: Arc<dyn TokenCounter> = Arc::new(crate::text::WhitespaceCounter);
        let (m, _rx) = Metering::capture(tc);
        assert_eq!(m.flush(Duration::from_secs(10)).await, 0);
    }
}
