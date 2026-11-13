//! Runtime configuration: `STRATA_*` env vars with defaults, plus licence tokens from
//! `~/.config/strata-rag/licenses.env` (override with `STRATA_LICENSES_FILE`).
//! Process env wins over the licences file; empty values count as unset.
//!
//! Embed / rerank providers are not configured here: the deployment layer is `providers.toml`
//! (`STRATA_PROVIDERS_FILE`, see `crate::providers::deployment`, loaded by [`Config::from_env`]),
//! and tenants register their own (`crate::providers`).

use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::error::{Result, StrataError};
use crate::tenant::Tenant;

pub const DEFAULT_DATABASE_URL: &str = "postgres://strata_rag:strata_rag@127.0.0.1:5440/strata_rag";
pub const DEFAULT_BIND: &str = "127.0.0.1:8092";
pub const DEFAULT_LAYA_URL: &str = "https://api.nonagentic.ai/laya";
const LICENSES_RELATIVE: &str = ".config/strata-rag/licenses.env";
pub const DEFAULT_EVAL_QUESTIONS: &str = "eval/questions.jsonl";
/// Upper bound of `STRATA_EXACT_SCAN_MAX_CHUNKS`: an exact scan costs ~0.3 ms per 1k chunks at
/// dim 256 (~2 ms at dim 1024, scale harness), so a larger threshold would let one search scan for
/// seconds.
pub const MAX_EXACT_SCAN_MAX_CHUNKS: u64 = 1_000_000;
/// Default `STRATA_PLATFORM_LICENSES` (platform-knowledge spec D4): licences platform content may
/// carry. Non-commercial and no-derivatives licences are left out: tenants pay.
pub const DEFAULT_PLATFORM_LICENSES: &str =
    "CC0-1.0,PDDL-1.0,public-domain,CC-BY-4.0,CC-BY-SA-4.0,OGL-UK-3.0,ODC-BY-1.0";
/// Default `STRATA_LICENSE_APP_KEYS`: the licences-service app key whose licences grant access.
pub const DEFAULT_LICENSE_APP_KEYS: &str = "strata-rag";
/// Upper bound of `STRATA_LICENSE_CACHE_S`: a revoked licence keeps working at most this long.
pub const MAX_LICENSE_CACHE_S: u64 = 60;

/// A secret string (licence token, provider key). `Debug` never prints the value; there is no
/// `Display` and no `Serialize`.
#[derive(Clone, serde::Deserialize)]
#[serde(transparent)]
pub struct Secret(String);

impl Secret {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(***)")
    }
}

/// Which tenants an API token may act for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenScope {
    /// `<tenant-uuid>:<token>`: this tenant only; `X-Tenant-Id` may repeat it, never differ.
    Tenant(Tenant),
    /// `*:<token>`: a multi-tenant service caller; every request names its tenant in `X-Tenant-Id`.
    AnyTenant,
}

/// One `STRATA_API_TOKENS` entry. `Debug` shows the scope, never the token.
#[derive(Debug, Clone)]
pub struct ApiToken {
    pub scope: TokenScope,
    pub token: Secret,
}

impl ApiToken {
    pub fn new(scope: TokenScope, token: impl Into<String>) -> Self {
        Self {
            scope,
            token: Secret::new(token),
        }
    }
}

/// `STRATA_API_TOKENS`: comma-separated `<tenant-uuid>:<token>` or `*:<token>` entries (whitespace
/// around entries ignored, empty entries dropped). Errors name the variable and the entry's
/// position, never its content: a malformed entry may be a bare secret.
pub fn parse_api_tokens(raw: &str) -> Result<Vec<ApiToken>> {
    let bad = |n: usize, why: &str| {
        config_err(format!(
            "STRATA_API_TOKENS entry {n} {why}; each entry is <tenant-uuid>:<token> or *:<token>"
        ))
    };
    let mut out: Vec<ApiToken> = Vec::new();
    for (i, entry) in raw
        .split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .enumerate()
    {
        let n = i + 1;
        let Some((scope, token)) = entry.split_once(':') else {
            return Err(bad(n, "has no tenant (no `:`)"));
        };
        let token = token.trim();
        if token.is_empty() {
            return Err(bad(n, "has an empty token"));
        }
        let scope = match scope.trim() {
            "*" => TokenScope::AnyTenant,
            t => TokenScope::Tenant(
                Tenant::parse(t)
                    .map_err(|_| bad(n, "does not start with a tenant uuid (non-nil) or `*`"))?,
            ),
        };
        if out.iter().any(|t| t.token.expose() == token) {
            return Err(bad(n, "repeats the token of an earlier entry"));
        }
        out.push(ApiToken::new(scope, token));
    }
    Ok(out)
}

/// `Debug` is hand-written so the database password never reaches a log line.
#[derive(Clone)]
pub struct Config {
    pub database_url: String,
    pub bind: SocketAddr,
    /// `STRATA_METRICS_BIND`: a second, metrics-only listener (`GET /metrics`, nothing else).
    /// `None` (unset or empty) disables it.
    pub metrics_bind: Option<SocketAddr>,
    pub leaf_max_tokens: i64,
    pub profile_alpha: f32,
    pub chunk_target: usize,
    pub chunk_overlap: usize,
    pub rerank_candidates: usize,
    /// `STRATA_PROVIDERS_FILE`: the deployment's `providers.toml`.
    pub providers_file: PathBuf,
    /// The loaded `providers_file`: set by [`Config::from_env`] (which has the env its `env:` key
    /// references resolve against); `None` from [`Config::from_map`]. Required to build an engine.
    pub deployment: Option<crate::providers::Deployment>,
    /// `STRATA_PROVIDER_KEY`: base64 of the 32-byte AES-256-GCM key sealing tenant provider keys
    /// (validated here, never printed). Required once any tenant key is stored and for the PUT API.
    pub provider_key: Option<Secret>,
    /// Most texts per ingest embed request (`STRATA_BULK_EMBED_BATCH`).
    pub embed_batch: usize,
    /// Concurrent ingest embed requests per embed provider (`STRATA_PROVIDER_BULK_CONCURRENCY`),
    /// capped at the provider's `concurrency`; separate from its query slots so ingest cannot
    /// starve queries.
    pub provider_bulk_concurrency: usize,
    pub laya_url: String,
    pub laya_token: Option<Secret>,
    pub laya_model: String,
    pub laya_timeout_ms: u64,
    /// Wall-clock budget for ALL laya calls of one query; each call is also cut at `laya_timeout_ms`.
    pub laya_query_budget_ms: u64,
    /// Bearer service tokens for the HTTP API (`STRATA_API_TOKENS`, comma list of
    /// `<tenant-uuid>:<token>` / `*:<token>`, see [`parse_api_tokens`]). Empty = no auth, which
    /// `strata serve` only allows on a loopback bind ([`Config::check_serve_auth`]); requests then
    /// still name their tenant in `X-Tenant-Id`.
    pub api_tokens: Vec<ApiToken>,
    /// `STRATA_LICENSE_INTROSPECT_URL`: the licences service's `POST /license/introspect` endpoint
    /// (full URL). Set = a bearer that matches no `STRATA_API_TOKENS` entry is introspected there
    /// and accepted as an active licence of one of `license_app_keys`, acting for the licence's
    /// tenant (`crate::api::licence`). Unset = licence auth off.
    pub license_introspect_url: Option<String>,
    /// `STRATA_LICENSE_APP_KEYS` (comma list, default [`DEFAULT_LICENSE_APP_KEYS`]): the app keys
    /// whose licences are accepted.
    pub license_app_keys: Vec<String>,
    /// Seconds an accepted licence token is reused without asking again (`STRATA_LICENSE_CACHE_S`,
    /// default 30, 0..=[`MAX_LICENSE_CACHE_S`]; 0 = introspect every request).
    pub license_cache_s: u64,
    /// Timeout of one introspect call (`STRATA_LICENSE_TIMEOUT_MS`, default 3000).
    pub license_timeout_ms: u64,
    /// Questions `POST /api/eval` scores (`STRATA_EVAL_QUESTIONS`).
    pub eval_questions: PathBuf,
    /// Tree `POST /api/eval` scores against (`STRATA_EVAL_TREE`); unset = `tree.json` next to the
    /// questions file, as `strata eval` resolves it.
    pub eval_tree: Option<PathBuf>,
    /// Seconds a `/health` report is reused (`STRATA_HEALTH_CACHE_S`, default 10; 0 = probe on
    /// every request). `/health` is unauthenticated and probes laya and the default providers.
    pub health_cache_s: u64,
    /// `hnsw.ef_search` for the dense searches (`STRATA_HNSW_EF_SEARCH`, default 100, 1..=1000):
    /// candidates per HNSW scan step; higher = better recall, slower scan.
    pub hnsw_ef_search: u32,
    /// Leaf sets of at most this many chunks are searched exactly, without the HNSW index
    /// (`STRATA_EXACT_SCAN_MAX_CHUNKS`, default 5000, at most 1000000; 0 = always HNSW). See
    /// `db::search::choose_dense_path`.
    pub exact_scan_max_chunks: u64,
    /// `METERING_REDIS_URL`: billing's Redis, and the one metering switch: set = every process
    /// records cost events in `metering_outbox` and `strata serve` drains them to this Redis;
    /// unset = off (`crate::metering`). May carry a password, so it is a `Secret`.
    pub metering_redis_url: Option<Secret>,
    /// `METERING_STREAM` (default `billing:costs`): the stream cost events are `XADD`ed to.
    pub metering_stream: String,
    /// Query vectors kept in the LRU (`STRATA_QUERY_EMBED_CACHE`, default 1024; 0 = off).
    pub query_embed_cache: usize,
    /// Most pooled Postgres connections per process (`STRATA_DB_MAX_CONNECTIONS`, default 16).
    /// The deployment's pgbouncer `pool_size` must cover this x (replicas + surge).
    pub db_max_connections: u32,
    /// Seconds an idle pooled connection is kept (`STRATA_DB_IDLE_TIMEOUT_S`, default 120). Must
    /// stay below the server-side idle reapers (pgbouncer `server_idle_timeout` 240 s, postgres
    /// `idle_session_timeout` 600 s) so the pool closes a connection before the server drops it
    /// under us (the EOF errors on the first query after an idle spell).
    pub db_idle_timeout_s: u64,
    /// Seconds before a pooled connection is recycled regardless of use
    /// (`STRATA_DB_MAX_LIFETIME_S`, default 1800).
    pub db_max_lifetime_s: u64,
    /// `STRATA_PLATFORM_OWNER` (non-nil uuid): the tenant owning platform knowledge packs
    /// (`crate::platform`). Unset = platform knowledge off; every `/api/platform` route is 404.
    pub platform_owner: Option<Tenant>,
    /// `STRATA_PLATFORM_LICENSES` (comma list, default [`DEFAULT_PLATFORM_LICENSES`]): the licence
    /// ids a document ingested into a platform store may carry.
    pub platform_licenses: Vec<String>,
    /// `STRATA_PLATFORM_PENALTY` (default 0.05, within [0, 1]): subtracted from a platform hit's
    /// reranked score, so tenant knowledge wins near-ties.
    pub platform_penalty: f32,
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("database_url", &redact_url(&self.database_url))
            .field("bind", &self.bind)
            .field("metrics_bind", &self.metrics_bind)
            .field("leaf_max_tokens", &self.leaf_max_tokens)
            .field("profile_alpha", &self.profile_alpha)
            .field("chunk_target", &self.chunk_target)
            .field("chunk_overlap", &self.chunk_overlap)
            .field("rerank_candidates", &self.rerank_candidates)
            .field("providers_file", &self.providers_file)
            .field("deployment", &self.deployment)
            .field("provider_key", &self.provider_key)
            .field("embed_batch", &self.embed_batch)
            .field("provider_bulk_concurrency", &self.provider_bulk_concurrency)
            .field("laya_url", &self.laya_url)
            .field("laya_token", &self.laya_token)
            .field("laya_model", &self.laya_model)
            .field("laya_timeout_ms", &self.laya_timeout_ms)
            .field("laya_query_budget_ms", &self.laya_query_budget_ms)
            .field("api_tokens", &self.api_tokens)
            .field("license_introspect_url", &self.license_introspect_url)
            .field("license_app_keys", &self.license_app_keys)
            .field("license_cache_s", &self.license_cache_s)
            .field("license_timeout_ms", &self.license_timeout_ms)
            .field("eval_questions", &self.eval_questions)
            .field("eval_tree", &self.eval_tree)
            .field("health_cache_s", &self.health_cache_s)
            .field("hnsw_ef_search", &self.hnsw_ef_search)
            .field("exact_scan_max_chunks", &self.exact_scan_max_chunks)
            .field("metering_redis_url", &self.metering_redis_url)
            .field("metering_stream", &self.metering_stream)
            .field("query_embed_cache", &self.query_embed_cache)
            .field("db_max_connections", &self.db_max_connections)
            .field("db_idle_timeout_s", &self.db_idle_timeout_s)
            .field("db_max_lifetime_s", &self.db_max_lifetime_s)
            .field("platform_owner", &self.platform_owner)
            .field("platform_licenses", &self.platform_licenses)
            .field("platform_penalty", &self.platform_penalty)
            .finish()
    }
}

impl Config {
    /// Reads the process environment plus the licences file (process env wins).
    pub fn from_env() -> Result<Self> {
        let process: HashMap<String, String> = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
            .collect();
        let path = licenses_path(&process);
        let merged = merge_env(process, path.as_deref())?;
        let mut cfg = Self::from_map(&merged)?;
        cfg.deployment = Some(crate::providers::Deployment::load(
            &cfg.providers_file,
            &merged,
        )?);
        Ok(cfg)
    }

    /// Pure: builds and validates a config from a key/value map.
    pub fn from_map(env: &HashMap<String, String>) -> Result<Self> {
        let cfg = Config {
            database_url: get(env, "STRATA_DATABASE_URL")
                .unwrap_or(DEFAULT_DATABASE_URL)
                .to_string(),
            bind: parse(env, "STRATA_BIND", DEFAULT_BIND)?,
            metrics_bind: match get(env, "STRATA_METRICS_BIND") {
                None => None,
                Some(raw) => Some(raw.parse::<SocketAddr>().map_err(|e| {
                    config_err(format!("STRATA_METRICS_BIND={raw:?} is invalid: {e}"))
                })?),
            },
            leaf_max_tokens: parse(env, "STRATA_LEAF_MAX_TOKENS", "1000000")?,
            profile_alpha: parse(env, "STRATA_PROFILE_ALPHA", "0.7")?,
            chunk_target: parse(env, "STRATA_CHUNK_TARGET", "384")?,
            chunk_overlap: parse(env, "STRATA_CHUNK_OVERLAP", "48")?,
            rerank_candidates: parse(env, "STRATA_RERANK_CANDIDATES", "8")?,
            providers_file: PathBuf::from(
                get(env, "STRATA_PROVIDERS_FILE")
                    .unwrap_or(crate::providers::deployment::DEFAULT_PROVIDERS_FILE),
            ),
            deployment: None,
            provider_key: get(env, "STRATA_PROVIDER_KEY").map(Secret::new),
            embed_batch: parse(env, "STRATA_BULK_EMBED_BATCH", "32")?,
            provider_bulk_concurrency: parse(env, "STRATA_PROVIDER_BULK_CONCURRENCY", "2")?,
            laya_url: url(env, "STRATA_LAYA_URL", DEFAULT_LAYA_URL)?,
            laya_token: get(env, "LAYA_LICENSE_TOKEN").map(Secret::new),
            laya_model: get(env, "STRATA_LAYA_MODEL")
                .unwrap_or("jev-latest")
                .to_string(),
            laya_timeout_ms: parse(env, "STRATA_LAYA_TIMEOUT_MS", "800")?,
            laya_query_budget_ms: parse(env, "STRATA_LAYA_QUERY_BUDGET_MS", "1200")?,
            api_tokens: get(env, "STRATA_API_TOKENS")
                .map(parse_api_tokens)
                .transpose()?
                .unwrap_or_default(),
            license_introspect_url: get(env, "STRATA_LICENSE_INTROSPECT_URL")
                .map(|_| url(env, "STRATA_LICENSE_INTROSPECT_URL", ""))
                .transpose()?,
            license_app_keys: get(env, "STRATA_LICENSE_APP_KEYS")
                .unwrap_or(DEFAULT_LICENSE_APP_KEYS)
                .split(',')
                .map(str::trim)
                .filter(|k| !k.is_empty())
                .map(str::to_string)
                .collect(),
            license_cache_s: parse(env, "STRATA_LICENSE_CACHE_S", "30")?,
            license_timeout_ms: parse(env, "STRATA_LICENSE_TIMEOUT_MS", "3000")?,
            eval_questions: PathBuf::from(
                get(env, "STRATA_EVAL_QUESTIONS").unwrap_or(DEFAULT_EVAL_QUESTIONS),
            ),
            eval_tree: get(env, "STRATA_EVAL_TREE").map(PathBuf::from),
            health_cache_s: parse(env, "STRATA_HEALTH_CACHE_S", "10")?,
            hnsw_ef_search: parse(env, "STRATA_HNSW_EF_SEARCH", "100")?,
            exact_scan_max_chunks: parse(env, "STRATA_EXACT_SCAN_MAX_CHUNKS", "5000")?,
            metering_redis_url: get(env, "METERING_REDIS_URL").map(Secret::new),
            metering_stream: get(env, "METERING_STREAM")
                .unwrap_or(crate::metering::DEFAULT_STREAM)
                .to_string(),
            query_embed_cache: parse(env, "STRATA_QUERY_EMBED_CACHE", "1024")?,
            db_max_connections: parse(env, "STRATA_DB_MAX_CONNECTIONS", "16")?,
            db_idle_timeout_s: parse(env, "STRATA_DB_IDLE_TIMEOUT_S", "120")?,
            db_max_lifetime_s: parse(env, "STRATA_DB_MAX_LIFETIME_S", "1800")?,
            platform_owner: get(env, "STRATA_PLATFORM_OWNER")
                .map(|raw| {
                    Tenant::parse(raw).map_err(|_| {
                        config_err(format!(
                            "STRATA_PLATFORM_OWNER={raw:?} must be a non-nil uuid"
                        ))
                    })
                })
                .transpose()?,
            platform_licenses: get(env, "STRATA_PLATFORM_LICENSES")
                .unwrap_or(DEFAULT_PLATFORM_LICENSES)
                .split(',')
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect(),
            platform_penalty: parse(env, "STRATA_PLATFORM_PENALTY", "0.05")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.profile_alpha) {
            return Err(config_err(format!(
                "STRATA_PROFILE_ALPHA must be within [0, 1], got {}",
                self.profile_alpha
            )));
        }
        if self.chunk_target == 0 {
            return Err(config_err("STRATA_CHUNK_TARGET must be >= 1".to_string()));
        }
        if self.chunk_overlap >= self.chunk_target {
            return Err(config_err(format!(
                "STRATA_CHUNK_OVERLAP ({}) must be < STRATA_CHUNK_TARGET ({})",
                self.chunk_overlap, self.chunk_target
            )));
        }
        if self.rerank_candidates == 0 {
            return Err(config_err(
                "STRATA_RERANK_CANDIDATES must be >= 1".to_string(),
            ));
        }
        if self.leaf_max_tokens < 1 {
            return Err(config_err(
                "STRATA_LEAF_MAX_TOKENS must be >= 1".to_string(),
            ));
        }
        if self.embed_batch == 0 {
            return Err(config_err(
                "STRATA_BULK_EMBED_BATCH must be >= 1".to_string(),
            ));
        }
        if self.provider_bulk_concurrency == 0 {
            return Err(config_err(
                "STRATA_PROVIDER_BULK_CONCURRENCY must be >= 1".to_string(),
            ));
        }
        if let Some(k) = &self.provider_key {
            crate::providers::crypto::decode_key(k.expose())?;
        }
        if self.laya_timeout_ms == 0 {
            return Err(config_err(
                "STRATA_LAYA_TIMEOUT_MS must be >= 1".to_string(),
            ));
        }
        if self.laya_query_budget_ms == 0 {
            return Err(config_err(
                "STRATA_LAYA_QUERY_BUDGET_MS must be >= 1".to_string(),
            ));
        }
        if !(crate::db::search::MIN_EF_SEARCH..=crate::db::search::MAX_EF_SEARCH)
            .contains(&self.hnsw_ef_search)
        {
            return Err(config_err(format!(
                "STRATA_HNSW_EF_SEARCH must be within [{}, {}], got {}",
                crate::db::search::MIN_EF_SEARCH,
                crate::db::search::MAX_EF_SEARCH,
                self.hnsw_ef_search
            )));
        }
        if self.exact_scan_max_chunks > MAX_EXACT_SCAN_MAX_CHUNKS {
            return Err(config_err(format!(
                "STRATA_EXACT_SCAN_MAX_CHUNKS must be within [0, {MAX_EXACT_SCAN_MAX_CHUNKS}], got {}",
                self.exact_scan_max_chunks
            )));
        }
        if self.db_max_connections == 0 {
            return Err(config_err(
                "STRATA_DB_MAX_CONNECTIONS must be >= 1".to_string(),
            ));
        }
        if self.db_idle_timeout_s == 0 {
            return Err(config_err(
                "STRATA_DB_IDLE_TIMEOUT_S must be >= 1".to_string(),
            ));
        }
        if self.license_app_keys.is_empty() {
            return Err(config_err(
                "STRATA_LICENSE_APP_KEYS must name at least one app key".to_string(),
            ));
        }
        if self.license_cache_s > MAX_LICENSE_CACHE_S {
            return Err(config_err(format!(
                "STRATA_LICENSE_CACHE_S must be within [0, {MAX_LICENSE_CACHE_S}], got {}",
                self.license_cache_s
            )));
        }
        if self.license_timeout_ms == 0 {
            return Err(config_err(
                "STRATA_LICENSE_TIMEOUT_MS must be >= 1".to_string(),
            ));
        }
        if self.platform_licenses.is_empty() {
            return Err(config_err(
                "STRATA_PLATFORM_LICENSES must name at least one licence".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.platform_penalty) {
            return Err(config_err(format!(
                "STRATA_PLATFORM_PENALTY must be within [0, 1], got {}",
                self.platform_penalty
            )));
        }
        if self.metrics_bind == Some(self.bind) {
            return Err(config_err(format!(
                "STRATA_METRICS_BIND must differ from STRATA_BIND (both {})",
                self.bind
            )));
        }
        if self.db_max_lifetime_s < self.db_idle_timeout_s {
            return Err(config_err(format!(
                "STRATA_DB_MAX_LIFETIME_S ({}) must be >= STRATA_DB_IDLE_TIMEOUT_S ({})",
                self.db_max_lifetime_s, self.db_idle_timeout_s
            )));
        }
        Ok(())
    }

    /// The Postgres pool sizing and recycling knobs, for [`crate::db::connect_with`].
    pub fn pool_settings(&self) -> crate::db::PoolSettings {
        crate::db::PoolSettings {
            max_connections: self.db_max_connections,
            idle_timeout: std::time::Duration::from_secs(self.db_idle_timeout_s),
            max_lifetime: std::time::Duration::from_secs(self.db_max_lifetime_s),
        }
    }
}

impl Config {
    /// True when requests need a bearer: `STRATA_API_TOKENS` holds a token or licence auth is on.
    pub fn auth_enabled(&self) -> bool {
        !self.api_tokens.is_empty() || self.license_introspect_url.is_some()
    }

    /// `strata serve` refuses an open API on a non-loopback bind: with `STRATA_BIND` reachable from
    /// other hosts, `STRATA_API_TOKENS` must hold at least one token or
    /// `STRATA_LICENSE_INTROSPECT_URL` must be set. Only the serve path calls this; the CLI
    /// subcommands run in-process and never listen.
    pub fn check_serve_auth(&self) -> Result<()> {
        if !self.auth_enabled() && !self.bind.ip().is_loopback() {
            return Err(config_err(format!(
                "STRATA_BIND {} is not a loopback address and STRATA_API_TOKENS is empty; \
                 set STRATA_API_TOKENS (comma-separated <tenant-uuid>:<token> or *:<token>), \
                 STRATA_LICENSE_INTROSPECT_URL, or bind 127.0.0.1",
                self.bind
            )));
        }
        Ok(())
    }
}

fn config_err(msg: String) -> StrataError {
    StrataError::Config(msg)
}

/// Trimmed value, or `None` when absent / empty / whitespace-only.
fn get<'a>(env: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    env.get(key).map(|s| s.trim()).filter(|s| !s.is_empty())
}

fn parse<T>(env: &HashMap<String, String>, key: &str, default: &str) -> Result<T>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    let raw = get(env, key).unwrap_or(default);
    raw.parse::<T>()
        .map_err(|e| config_err(format!("{key}={raw:?} is invalid: {e}")))
}

fn url(env: &HashMap<String, String>, key: &str, default: &str) -> Result<String> {
    let raw = get(env, key).unwrap_or(default).trim_end_matches('/');
    if raw.starts_with("http://") || raw.starts_with("https://") {
        Ok(raw.to_string())
    } else {
        Err(config_err(format!(
            "{key}={raw:?} must start with http:// or https://"
        )))
    }
}

/// Parses `KEY=VALUE` lines: blank lines and `#` comments are skipped, an optional
/// `export ` prefix is accepted, the value may be wrapped in matching `"` or `'`.
/// Only the first `=` separates key from value. Inline `#` comments are NOT stripped.
pub fn parse_env_file(contents: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line
            .strip_prefix("export ")
            .map(str::trim_start)
            .unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        out.insert(key.to_string(), unquote(value.trim()).to_string());
    }
    out
}

fn unquote(v: &str) -> &str {
    for q in ['"', '\''] {
        if v.len() >= 2 && v.starts_with(q) && v.ends_with(q) {
            return &v[1..v.len() - 1];
        }
    }
    v
}

/// `STRATA_LICENSES_FILE` if set, else `$HOME/.config/strata-rag/licenses.env`, else `None`.
fn licenses_path(env: &HashMap<String, String>) -> Option<PathBuf> {
    if let Some(p) = get(env, "STRATA_LICENSES_FILE") {
        return Some(PathBuf::from(p));
    }
    get(env, "HOME").map(|home| Path::new(home).join(LICENSES_RELATIVE))
}

/// Licences file first, then non-empty process values on top. A missing file is fine;
/// any other read failure is a config error (never silently run without tokens).
fn merge_env(
    process: HashMap<String, String>,
    file: Option<&Path>,
) -> Result<HashMap<String, String>> {
    let mut merged = HashMap::new();
    if let Some(path) = file {
        match std::fs::read_to_string(path) {
            Ok(contents) => {
                warn_if_permissive(path);
                merged.extend(parse_env_file(&contents));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(config_err(format!(
                    "cannot read licences file {}: {e}",
                    path.display()
                )));
            }
        }
    }
    for (k, v) in process {
        if !v.trim().is_empty() {
            merged.insert(k, v);
        }
    }
    Ok(merged)
}

#[cfg(unix)]
fn warn_if_permissive(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(meta) = std::fs::metadata(path) {
        let mode = meta.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            tracing::warn!(
                path = %path.display(),
                mode = %format!("{mode:o}"),
                "licences file is readable by group/others; run chmod 600"
            );
        }
    }
}

#[cfg(not(unix))]
fn warn_if_permissive(_path: &Path) {}

/// Redact secrets from a connection URL, for logs and error messages: the password in
/// `scheme://user:password@host…` and the values of `password=` / `sslpassword=` query parameters
/// become `***`. A password holding an unencoded `/`, `?` or `#` is still redacted (the userinfo runs to
/// the last `@`). URLs without secrets are returned unchanged.
pub fn redact_url(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let (prefix, rest) = url.split_at(scheme_end + 3);
    match userinfo_bounds(rest) {
        Some((colon, at)) => format!(
            "{prefix}{}:***{}",
            &rest[..colon],
            mask_query_secrets(&rest[at..])
        ),
        None => format!("{prefix}{}", mask_query_secrets(rest)),
    }
}

/// Byte offsets in `rest` (the URL after `://`) of the `:` that ends the user name and the `@` that
/// ends the userinfo, when the URL carries a password.
fn userinfo_bounds(rest: &str) -> Option<(usize, usize)> {
    let delim = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..delim];
    if let Some(at) = authority.rfind('@') {
        return Some((authority[..at].find(':')?, at));
    }
    // No `@` before the first delimiter: either no userinfo at all, or a password that holds an
    // unencoded `/`, `?` or `#`. With a `:` before that delimiter and an `@` after it, the latter is
    // assumed, except for a plain `host:port` whose path/query merely contains an `@`.
    let colon = authority.find(':')?;
    let at = rest.rfind('@')?;
    let after_colon = &authority[colon + 1..];
    let numeric_port = !after_colon.is_empty() && after_colon.bytes().all(|b| b.is_ascii_digit());
    if numeric_port && !rest[at + 1..].contains('/') {
        return None;
    }
    Some((colon, at))
}

/// Mask the values of `password` / `sslpassword` query parameters (key match is case-insensitive).
fn mask_query_secrets(s: &str) -> String {
    let Some(q) = s.find('?') else {
        return s.to_string();
    };
    let (head, query) = s.split_at(q + 1);
    let (query, fragment) = match query.find('#') {
        Some(h) => query.split_at(h),
        None => (query, ""),
    };
    let masked: Vec<String> = query
        .split('&')
        .map(|param| match param.split_once('=') {
            Some((key, _))
                if key.eq_ignore_ascii_case("password")
                    || key.eq_ignore_ascii_case("sslpassword") =>
            {
                format!("{key}=***")
            }
            _ => param.to_string(),
        })
        .collect();
    format!("{head}{}{fragment}", masked.join("&"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn temp_file(contents: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("strata-cfg-{}.env", uuid::Uuid::now_v7()));
        std::fs::write(&p, contents).expect("write temp env file");
        p
    }

    #[test]
    fn metrics_bind_is_optional_and_validated() {
        for unset in [None, Some(""), Some("  ")] {
            let m = match unset {
                Some(v) => map(&[("STRATA_METRICS_BIND", v)]),
                None => HashMap::new(),
            };
            assert_eq!(Config::from_map(&m).unwrap().metrics_bind, None);
        }
        let c = Config::from_map(&map(&[("STRATA_METRICS_BIND", "0.0.0.0:8093")])).unwrap();
        assert_eq!(c.metrics_bind, Some("0.0.0.0:8093".parse().unwrap()));
        assert_eq!(c.bind.port(), 8092, "STRATA_BIND is unaffected");
        assert!(Config::from_map(&map(&[("STRATA_METRICS_BIND", "nope")])).is_err());
        assert!(Config::from_map(&map(&[("STRATA_METRICS_BIND", DEFAULT_BIND)])).is_err());
    }

    #[test]
    fn defaults_match_contract() {
        let c = Config::from_map(&HashMap::new()).expect("defaults are valid");
        assert_eq!(
            c.database_url,
            "postgres://strata_rag:strata_rag@127.0.0.1:5440/strata_rag"
        );
        assert_eq!(c.bind, "127.0.0.1:8092".parse::<SocketAddr>().unwrap());
        assert_eq!(c.metrics_bind, None);
        assert_eq!(c.leaf_max_tokens, 1_000_000);
        assert!((c.profile_alpha - 0.7).abs() < f32::EPSILON);
        assert_eq!(c.chunk_target, 384);
        assert_eq!(c.chunk_overlap, 48);
        assert_eq!(c.rerank_candidates, 8);
        assert_eq!(
            c.providers_file,
            PathBuf::from("/etc/strata/providers.toml")
        );
        assert!(c.deployment.is_none(), "from_map never reads the file");
        assert!(c.provider_key.is_none());
        assert_eq!((c.embed_batch, c.provider_bulk_concurrency), (32, 2));
        assert_eq!(c.laya_url, "https://api.nonagentic.ai/laya");
        assert!(c.laya_token.is_none());
        assert_eq!(c.laya_model, "jev-latest");
        assert_eq!(c.laya_timeout_ms, 800);
        assert_eq!(c.laya_query_budget_ms, 1200);
        assert_eq!(
            c.laya_query_budget_ms,
            crate::routing::DEFAULT_LAYA_QUERY_BUDGET_MS
        );
        assert_eq!(c.pool_settings(), crate::db::PoolSettings::default());
        assert!(
            c.platform_owner.is_none(),
            "platform knowledge is off by default"
        );
        assert_eq!(
            c.platform_licenses,
            [
                "CC0-1.0",
                "PDDL-1.0",
                "public-domain",
                "CC-BY-4.0",
                "CC-BY-SA-4.0",
                "OGL-UK-3.0",
                "ODC-BY-1.0"
            ]
        );
        assert!((c.platform_penalty - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn platform_settings_are_parsed_and_validated() {
        let owner = "0192f0c8-0000-7000-8000-0000000000ff";
        let c = Config::from_map(&map(&[
            ("STRATA_PLATFORM_OWNER", owner),
            ("STRATA_PLATFORM_LICENSES", " CC0-1.0 , ,CC-BY-4.0"),
            ("STRATA_PLATFORM_PENALTY", "0.2"),
        ]))
        .expect("valid platform settings");
        assert_eq!(c.platform_owner, Some(Tenant::parse(owner).unwrap()));
        assert_eq!(c.platform_licenses, ["CC0-1.0", "CC-BY-4.0"]);
        assert!((c.platform_penalty - 0.2).abs() < f32::EPSILON);
        for (key, value) in [
            ("STRATA_PLATFORM_OWNER", "not-a-uuid"),
            (
                "STRATA_PLATFORM_OWNER",
                "00000000-0000-0000-0000-000000000000",
            ),
            ("STRATA_PLATFORM_LICENSES", " , "),
            ("STRATA_PLATFORM_PENALTY", "1.5"),
            ("STRATA_PLATFORM_PENALTY", "-0.1"),
            ("STRATA_PLATFORM_PENALTY", "x"),
        ] {
            let err = Config::from_map(&map(&[(key, value)])).unwrap_err();
            assert!(
                matches!(err, StrataError::Config(_)),
                "{key}={value}: {err}"
            );
        }
    }

    #[test]
    fn overrides_are_parsed() {
        let c = Config::from_map(&map(&[
            ("STRATA_DATABASE_URL", "postgres://u:p@db:5440/x"),
            ("STRATA_BIND", "0.0.0.0:9999"),
            ("STRATA_LEAF_MAX_TOKENS", "5000"),
            ("STRATA_PROFILE_ALPHA", "0.25"),
            ("STRATA_CHUNK_TARGET", "128"),
            ("STRATA_CHUNK_OVERLAP", "16"),
            ("STRATA_RERANK_CANDIDATES", "16"),
            ("STRATA_PROVIDERS_FILE", "/srv/providers.toml"),
            ("STRATA_PROVIDER_KEY", KEY_B64),
            ("STRATA_BULK_EMBED_BATCH", "8"),
            ("STRATA_PROVIDER_BULK_CONCURRENCY", "1"),
            ("STRATA_LAYA_URL", "http://localhost:9000/laya/"),
            ("STRATA_LAYA_MODEL", "jev-test"),
            ("STRATA_LAYA_TIMEOUT_MS", "100"),
            ("STRATA_LAYA_QUERY_BUDGET_MS", "250"),
            ("STRATA_DB_MAX_CONNECTIONS", "4"),
            ("STRATA_DB_IDLE_TIMEOUT_S", "30"),
            ("STRATA_DB_MAX_LIFETIME_S", "600"),
            ("LAYA_LICENSE_TOKEN", "laya-secret-value"),
        ]))
        .expect("valid overrides");
        assert_eq!(c.database_url, "postgres://u:p@db:5440/x");
        assert_eq!(c.bind.port(), 9999);
        assert_eq!(c.leaf_max_tokens, 5000);
        assert!((c.profile_alpha - 0.25).abs() < f32::EPSILON);
        assert_eq!((c.chunk_target, c.chunk_overlap), (128, 16));
        assert_eq!(c.rerank_candidates, 16);
        assert_eq!(c.providers_file, PathBuf::from("/srv/providers.toml"));
        assert_eq!(c.provider_key.as_ref().map(Secret::expose), Some(KEY_B64));
        assert_eq!((c.embed_batch, c.provider_bulk_concurrency), (8, 1));
        assert_eq!(c.laya_url, "http://localhost:9000/laya");
        assert_eq!(c.laya_model, "jev-test");
        assert_eq!(c.laya_timeout_ms, 100);
        assert_eq!(c.laya_query_budget_ms, 250);
        assert_eq!(
            c.pool_settings(),
            crate::db::PoolSettings {
                max_connections: 4,
                idle_timeout: std::time::Duration::from_secs(30),
                max_lifetime: std::time::Duration::from_secs(600),
            }
        );
        assert_eq!(
            c.laya_token.as_ref().map(Secret::expose),
            Some("laya-secret-value")
        );
    }

    #[test]
    fn empty_values_count_as_unset() {
        let c = Config::from_map(&map(&[
            ("STRATA_CHUNK_TARGET", ""),
            ("LAYA_LICENSE_TOKEN", "   "),
            ("STRATA_LAYA_URL", ""),
        ]))
        .unwrap();
        assert_eq!(c.chunk_target, 384);
        assert!(c.laya_token.is_none());
        assert_eq!(c.laya_url, "https://api.nonagentic.ai/laya");
    }

    #[test]
    fn invalid_values_are_config_errors() {
        let bad: &[(&str, &str)] = &[
            ("STRATA_PROFILE_ALPHA", "1.5"),
            ("STRATA_PROFILE_ALPHA", "NaN"),
            ("STRATA_CHUNK_OVERLAP", "384"),
            ("STRATA_CHUNK_TARGET", "0"),
            ("STRATA_LEAF_MAX_TOKENS", "0"),
            ("STRATA_BULK_EMBED_BATCH", "0"),
            ("STRATA_RERANK_CANDIDATES", "0"),
            ("STRATA_PROVIDER_BULK_CONCURRENCY", "0"),
            ("STRATA_PROVIDER_KEY", "c2hvcnQ="),
            ("STRATA_PROVIDER_KEY", "not base64!"),
            ("STRATA_BIND", "not-an-addr"),
            ("STRATA_LEAF_MAX_TOKENS", "lots"),
            ("STRATA_LAYA_URL", "ftp://x"),
            ("STRATA_LAYA_TIMEOUT_MS", "0"),
            ("STRATA_LAYA_QUERY_BUDGET_MS", "0"),
            ("STRATA_LAYA_QUERY_BUDGET_MS", "-5"),
            ("STRATA_DB_MAX_CONNECTIONS", "0"),
            ("STRATA_DB_IDLE_TIMEOUT_S", "0"),
            ("STRATA_DB_MAX_LIFETIME_S", "60"),
        ];
        for (k, v) in bad {
            let err = match Config::from_map(&map(&[(*k, *v)])) {
                Ok(_) => panic!("{k}={v} must be rejected"),
                Err(e) => e,
            };
            assert!(
                matches!(err, StrataError::Config(_)),
                "{k}={v} gave {err:?}"
            );
            assert_eq!(err.code(), "internal");
        }
    }

    const TENANT_A: &str = "0192f0c8-0000-7000-8000-00000000000a";
    /// base64 of 32 bytes of 0x01.
    const KEY_B64: &str = "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=";

    #[test]
    fn api_tokens_are_trimmed_and_empty_entries_dropped() {
        let raw = format!(" {TENANT_A}:tok-a , ,*: tok-b ,");
        let c = Config::from_map(&map(&[("STRATA_API_TOKENS", raw.as_str())])).unwrap();
        let toks: Vec<&str> = c.api_tokens.iter().map(|t| t.token.expose()).collect();
        assert_eq!(toks, ["tok-a", "tok-b"]);
        let scopes: Vec<TokenScope> = c.api_tokens.iter().map(|t| t.scope).collect();
        assert_eq!(
            scopes,
            [
                TokenScope::Tenant(Tenant::parse(TENANT_A).unwrap()),
                TokenScope::AnyTenant
            ]
        );
        let dbg = format!("{c:?}");
        assert!(!dbg.contains("tok-a") && !dbg.contains("tok-b"), "{dbg}");
        for empty in ["", " , ,", ","] {
            let c = Config::from_map(&map(&[("STRATA_API_TOKENS", empty)])).unwrap();
            assert!(c.api_tokens.is_empty(), "{empty:?}");
        }
    }

    /// A malformed entry is a startup error naming STRATA_API_TOKENS and the entry's position, and
    /// never echoing the entry: a bare token without a tenant is still a secret.
    #[test]
    fn malformed_api_tokens_are_config_errors_without_the_secret() {
        for (raw, why) in [
            ("s3cret-bare-token", "no `:`"),
            (
                "00000000-0000-0000-0000-000000000000:s3cret-nil",
                "tenant uuid",
            ),
            ("not-a-uuid:s3cret-garbage", "tenant uuid"),
            ("Bearer s3cret-bearer:x", "tenant uuid"),
            ("*:", "empty token"),
            ("*:s3cret-dup,*:s3cret-dup", "repeats"),
        ] {
            let err = Config::from_map(&map(&[("STRATA_API_TOKENS", raw)])).unwrap_err();
            assert!(matches!(err, StrataError::Config(_)), "{raw}: {err:?}");
            let msg = err.to_string();
            assert!(msg.contains("STRATA_API_TOKENS"), "{msg}");
            assert!(msg.contains(why), "{raw}: {msg}");
            assert!(!msg.contains("s3cret"), "secret leaked: {msg}");
        }
        // the second entry is the bad one, and is named by position
        let raw = format!("{TENANT_A}:ok,s3cret-bare");
        let err = Config::from_map(&map(&[("STRATA_API_TOKENS", raw.as_str())])).unwrap_err();
        assert!(err.to_string().contains("entry 2"), "{err}");
        // a token may itself contain `:` (only the first separates the tenant)
        let raw = format!("{TENANT_A}:a:b");
        let c = Config::from_map(&map(&[("STRATA_API_TOKENS", raw.as_str())])).unwrap();
        assert_eq!(c.api_tokens[0].token.expose(), "a:b");
    }

    #[test]
    fn licence_auth_settings() {
        let c = Config::from_map(&map(&[])).unwrap();
        assert_eq!(c.license_introspect_url, None);
        assert_eq!(c.license_app_keys, ["strata-rag"]);
        assert_eq!(c.license_cache_s, 30);
        assert!(!c.auth_enabled());
        let c = Config::from_map(&map(&[
            (
                "STRATA_LICENSE_INTROSPECT_URL",
                " http://licenses.example.com/license/introspect/ ",
            ),
            ("STRATA_LICENSE_APP_KEYS", " strata-rag , ,care-bus "),
            ("STRATA_LICENSE_CACHE_S", "0"),
        ]))
        .unwrap();
        assert_eq!(
            c.license_introspect_url.as_deref(),
            Some("http://licenses.example.com/license/introspect")
        );
        assert_eq!(c.license_app_keys, ["strata-rag", "care-bus"]);
        assert_eq!(c.license_cache_s, 0);
        assert!(c.auth_enabled());
        for (k, v) in [
            (
                "STRATA_LICENSE_INTROSPECT_URL",
                "licenses.example.com/license/introspect",
            ),
            ("STRATA_LICENSE_APP_KEYS", " , "),
            ("STRATA_LICENSE_CACHE_S", "61"),
            ("STRATA_LICENSE_TIMEOUT_MS", "0"),
        ] {
            let err = Config::from_map(&map(&[(k, v)])).unwrap_err();
            assert!(err.to_string().contains(k), "{k}={v}: {err}");
        }
    }

    #[test]
    fn serve_refuses_an_open_non_loopback_bind() {
        for bind in ["0.0.0.0:8092", "192.168.1.5:8092", "[::]:8092"] {
            let c = Config::from_map(&map(&[("STRATA_BIND", bind)])).unwrap();
            let err = c.check_serve_auth().unwrap_err();
            assert!(matches!(err, StrataError::Config(_)), "{err:?}");
            assert!(err.to_string().contains("STRATA_API_TOKENS"), "{err}");
            let c = Config::from_map(&map(&[("STRATA_BIND", bind), ("STRATA_API_TOKENS", "*:t")]))
                .unwrap();
            c.check_serve_auth().expect("tokens set");
            let c = Config::from_map(&map(&[
                ("STRATA_BIND", bind),
                (
                    "STRATA_LICENSE_INTROSPECT_URL",
                    "http://licenses.example.com/license/introspect",
                ),
            ]))
            .unwrap();
            c.check_serve_auth().expect("licence auth set");
        }
        for bind in ["127.0.0.1:8092", "[::1]:8092"] {
            let c = Config::from_map(&map(&[("STRATA_BIND", bind)])).unwrap();
            c.check_serve_auth().expect("loopback stays open");
            assert!(c.api_tokens.is_empty());
        }
    }

    #[test]
    fn eval_paths_default_and_override() {
        let c = Config::from_map(&HashMap::new()).unwrap();
        assert_eq!(c.eval_questions, PathBuf::from("eval/questions.jsonl"));
        assert_eq!(c.eval_tree, None);
        let c = Config::from_map(&map(&[
            ("STRATA_EVAL_QUESTIONS", "/srv/q.jsonl"),
            ("STRATA_EVAL_TREE", "/srv/tree.json"),
        ]))
        .unwrap();
        assert_eq!(c.eval_questions, PathBuf::from("/srv/q.jsonl"));
        assert_eq!(c.eval_tree, Some(PathBuf::from("/srv/tree.json")));
    }

    #[test]
    fn health_cache_defaults_to_10s_and_0_disables() {
        let c = Config::from_map(&HashMap::new()).unwrap();
        assert_eq!(c.health_cache_s, 10);
        let c = Config::from_map(&map(&[("STRATA_HEALTH_CACHE_S", "0")])).unwrap();
        assert_eq!(c.health_cache_s, 0);
        assert!(Config::from_map(&map(&[("STRATA_HEALTH_CACHE_S", "-1")])).is_err());
    }

    #[test]
    fn exact_scan_max_chunks_defaults_to_5000_and_0_disables() {
        let c = Config::from_map(&HashMap::new()).unwrap();
        assert_eq!(
            c.exact_scan_max_chunks,
            crate::db::search::DEFAULT_EXACT_SCAN_MAX_CHUNKS
        );
        assert_eq!(c.exact_scan_max_chunks, 5_000);
        let c = Config::from_map(&map(&[("STRATA_EXACT_SCAN_MAX_CHUNKS", "0")])).unwrap();
        assert_eq!(c.exact_scan_max_chunks, 0);
        let c = Config::from_map(&map(&[("STRATA_EXACT_SCAN_MAX_CHUNKS", "1000000")])).unwrap();
        assert_eq!(c.exact_scan_max_chunks, 1_000_000);
        for bad in ["1000001", "-1", "lots"] {
            assert!(
                Config::from_map(&map(&[("STRATA_EXACT_SCAN_MAX_CHUNKS", bad)])).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn hnsw_ef_search_defaults_to_100_within_pgvector_bounds() {
        let c = Config::from_map(&HashMap::new()).unwrap();
        assert_eq!(c.hnsw_ef_search, 100);
        let c = Config::from_map(&map(&[("STRATA_HNSW_EF_SEARCH", "400")])).unwrap();
        assert_eq!(c.hnsw_ef_search, 400);
        for bad in ["0", "1001", "-1", "many"] {
            assert!(
                Config::from_map(&map(&[("STRATA_HNSW_EF_SEARCH", bad)])).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn query_embed_cache_defaults_to_1024_and_0_disables() {
        let c = Config::from_map(&HashMap::new()).unwrap();
        assert_eq!(c.query_embed_cache, 1024);
        let c = Config::from_map(&map(&[("STRATA_QUERY_EMBED_CACHE", "0")])).unwrap();
        assert_eq!(c.query_embed_cache, 0);
        assert!(Config::from_map(&map(&[("STRATA_QUERY_EMBED_CACHE", "-1")])).is_err());
    }

    #[test]
    fn metering_is_off_by_default_and_its_url_is_never_printed() {
        let c = Config::from_map(&HashMap::new()).unwrap();
        assert!(c.metering_redis_url.is_none());
        assert_eq!(c.metering_stream, "billing:costs");
        let c = Config::from_map(&map(&[
            (
                "METERING_REDIS_URL",
                "redis://:hunter2@billing-redis.billing:6379/0",
            ),
            ("METERING_STREAM", "billing:costs:test"),
        ]))
        .unwrap();
        assert_eq!(
            c.metering_redis_url.as_ref().map(|s| s.expose()),
            Some("redis://:hunter2@billing-redis.billing:6379/0")
        );
        assert_eq!(c.metering_stream, "billing:costs:test");
        assert!(!format!("{c:?}").contains("hunter2"));
    }

    /// The pre-providers.toml env vars are gone: setting one changes nothing (forward-only).
    #[test]
    fn the_flat_provider_env_vars_are_not_read() {
        let c = Config::from_map(&map(&[
            ("STRATA_EMBED_BASE_URL", "not-a-url"),
            ("STRATA_EMBED_DIMS", "768"),
            ("STRATA_RERANK_BASE_URL", "ftp://x"),
        ]))
        .expect("ignored");
        assert!(c.deployment.is_none());
    }

    #[test]
    fn a_bad_provider_key_is_refused_without_echoing_it() {
        let err =
            Config::from_map(&map(&[("STRATA_PROVIDER_KEY", "s3cret-not-b64!")])).unwrap_err();
        assert!(err.to_string().contains("STRATA_PROVIDER_KEY"), "{err}");
        assert!(!err.to_string().contains("s3cret"), "{err}");
    }

    #[test]
    fn secrets_never_appear_in_debug_output() {
        let c = Config::from_map(&map(&[
            ("LAYA_LICENSE_TOKEN", "tok-laya-123"),
            ("VLLM_LICENSE_TOKEN", "tok-vllm-456"),
            ("STRATA_PROVIDER_KEY", KEY_B64),
        ]))
        .unwrap();
        let dbg = format!("{c:?}");
        assert!(!dbg.contains("tok-laya-123"), "{dbg}");
        assert!(!dbg.contains("tok-vllm-456"), "{dbg}");
        assert!(!dbg.contains(KEY_B64), "{dbg}");
        assert!(dbg.contains("Secret(***)"), "{dbg}");
        assert_eq!(format!("{:?}", Secret::new("x")), "Secret(***)");
    }

    #[test]
    fn parse_env_file_handles_comments_quotes_export_and_equals() {
        let parsed = parse_env_file(
            "# a comment\n\
             \n\
             LAYA_LICENSE_TOKEN=abc.def\n\
             export VLLM_LICENSE_TOKEN=\"quoted value\"\r\n\
             SINGLE='single'\n\
             WITH_EQ=a=b=c\n\
               SPACED  =  padded  \n\
             NO_EQUALS_LINE\n\
             =no_key\n\
             LONE_QUOTE=\"\n",
        );
        assert_eq!(
            parsed.get("LAYA_LICENSE_TOKEN").map(String::as_str),
            Some("abc.def")
        );
        assert_eq!(
            parsed.get("VLLM_LICENSE_TOKEN").map(String::as_str),
            Some("quoted value")
        );
        assert_eq!(parsed.get("SINGLE").map(String::as_str), Some("single"));
        assert_eq!(parsed.get("WITH_EQ").map(String::as_str), Some("a=b=c"));
        assert_eq!(parsed.get("SPACED").map(String::as_str), Some("padded"));
        assert_eq!(parsed.get("LONE_QUOTE").map(String::as_str), Some("\""));
        assert!(!parsed.contains_key("NO_EQUALS_LINE"));
        assert!(!parsed.contains_key(""));
        assert_eq!(parsed.len(), 6);
    }

    #[test]
    fn merge_env_process_overrides_file_but_empty_process_value_does_not() {
        let p = temp_file("LAYA_LICENSE_TOKEN=file-laya\nVLLM_LICENSE_TOKEN='file-vllm'\n");
        let process = map(&[
            ("VLLM_LICENSE_TOKEN", "env-vllm"),
            ("LAYA_LICENSE_TOKEN", ""),
        ]);
        let merged = merge_env(process, Some(&p)).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert_eq!(merged["LAYA_LICENSE_TOKEN"], "file-laya");
        assert_eq!(merged["VLLM_LICENSE_TOKEN"], "env-vllm");
    }

    #[test]
    fn merge_env_missing_file_is_not_an_error() {
        let merged = merge_env(
            map(&[("A", "1")]),
            Some(Path::new("/nonexistent/strata-rag/licenses.env")),
        )
        .unwrap();
        assert_eq!(merged["A"], "1");
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn merge_env_unreadable_path_is_a_config_error() {
        // A directory exists but cannot be read as a file.
        let err = merge_env(HashMap::new(), Some(&std::env::temp_dir())).unwrap_err();
        assert!(matches!(err, StrataError::Config(_)), "{err:?}");
    }

    #[test]
    fn licenses_path_resolution() {
        assert_eq!(
            licenses_path(&map(&[
                ("STRATA_LICENSES_FILE", "/x/l.env"),
                ("HOME", "/home/u")
            ])),
            Some(PathBuf::from("/x/l.env"))
        );
        assert_eq!(
            licenses_path(&map(&[("STRATA_LICENSES_FILE", " "), ("HOME", "/home/u")])),
            Some(PathBuf::from("/home/u/.config/strata-rag/licenses.env"))
        );
        assert_eq!(licenses_path(&HashMap::new()), None);
    }

    #[test]
    fn tokens_flow_from_file_into_config() {
        let p = temp_file("LAYA_LICENSE_TOKEN=from-file\n");
        let merged = merge_env(HashMap::new(), Some(&p)).unwrap();
        std::fs::remove_file(&p).unwrap();
        let c = Config::from_map(&merged).unwrap();
        assert_eq!(c.laya_token.as_ref().map(Secret::expose), Some("from-file"));
    }

    #[test]
    fn redact_url_hides_password() {
        assert_eq!(
            redact_url("postgres://strata_rag:s3cret@127.0.0.1:5440/strata_rag"),
            "postgres://strata_rag:***@127.0.0.1:5440/strata_rag"
        );
    }

    #[test]
    fn redact_url_handles_at_in_password() {
        assert_eq!(
            redact_url("postgres://u:p@ss@h:1/db?x=1"),
            "postgres://u:***@h:1/db?x=1"
        );
    }

    #[test]
    fn redact_url_hides_passwords_with_unencoded_delimiters() {
        for (raw, want) in [
            ("postgres://u:p/ss@h:1/db", "postgres://u:***@h:1/db"),
            ("postgres://u:p?ss@h:1/db", "postgres://u:***@h:1/db"),
            ("postgres://u:p#ss@h:1/db", "postgres://u:***@h:1/db"),
            ("postgres://u:a/b?c#d@h/db?x=1", "postgres://u:***@h/db?x=1"),
            ("postgres://u:123/abc@h/db", "postgres://u:***@h/db"),
        ] {
            assert_eq!(redact_url(raw), want, "{raw}");
        }
    }

    #[test]
    fn redact_url_masks_password_query_parameters() {
        assert_eq!(
            redact_url("postgres://u@h/db?password=secret"),
            "postgres://u@h/db?password=***"
        );
        assert_eq!(
            redact_url("postgres://h/db?sslmode=require&sslpassword=k3y&PASSWORD=x&a=b"),
            "postgres://h/db?sslmode=require&sslpassword=***&PASSWORD=***&a=b"
        );
        assert_eq!(
            redact_url("postgres://u:p@h/db?password=q"),
            "postgres://u:***@h/db?password=***"
        );
        // a key that merely ends in "password" is not a password parameter
        assert_eq!(
            redact_url("postgres://h/db?mypassword=v"),
            "postgres://h/db?mypassword=v"
        );
    }

    #[test]
    fn config_debug_redacts_the_database_password() {
        let c = Config::from_map(&map(&[(
            "STRATA_DATABASE_URL",
            "postgres://u:db-pass-789@h:5440/x?sslpassword=ssl-pass-012",
        )]))
        .unwrap();
        let dbg = format!("{c:?}");
        assert!(!dbg.contains("db-pass-789"), "{dbg}");
        assert!(!dbg.contains("ssl-pass-012"), "{dbg}");
        assert!(dbg.contains("postgres://u:***@h:5440/x"), "{dbg}");
    }

    #[test]
    fn redact_url_leaves_passwordless_urls_alone() {
        assert_eq!(redact_url("postgres://u@h:1/db"), "postgres://u@h:1/db");
        assert_eq!(redact_url("postgres://h:1/db"), "postgres://h:1/db");
        assert_eq!(redact_url("not a url"), "not a url");
        assert_eq!(redact_url("postgres://h/db?u=a@b"), "postgres://h/db?u=a@b");
        assert_eq!(
            redact_url("postgres://h:5432/db?u=a@b"),
            "postgres://h:5432/db?u=a@b"
        );
    }
}
