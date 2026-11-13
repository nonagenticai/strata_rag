//! Licence-token auth (`STRATA_LICENSE_INTROSPECT_URL`): a bearer that matches no
//! `STRATA_API_TOKENS` entry is sent to the licences service's `POST /license/introspect`
//! (`{"token": "..."}` in, `{"active", "tenant_id", "app": {"key", "revoked"}, ...}` out) and
//! accepted when the licence is active, its app is not revoked and its app key is one of
//! `STRATA_LICENSE_APP_KEYS`. The caller then acts for the licence's tenant, exactly like a
//! tenant-pinned `<tenant>:<token>` entry.
//!
//! Accepted tokens are cached by SHA-256 digest (never the token) for `STRATA_LICENSE_CACHE_S`
//! (at most 60 s, so a revoked licence stops working within a minute), in a bounded LRU. Rejections
//! are not cached. Fail closed: when the licences service cannot answer (transport error, timeout,
//! non-2xx, unreadable body) the request is refused with 503, never let through and never reported
//! as a bad token. Tokens are never logged.

use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use lru::LruCache;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::error::{Result, StrataError};
use crate::tenant::Tenant;

/// Most accepted tokens remembered at once.
pub const LICENCE_CACHE_CAPACITY: usize = 4096;

/// What the licences service said about a presented token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LicenceCheck {
    /// An active licence of an accepted app: act for this tenant.
    Accepted(Tenant),
    /// Not a usable licence (unknown, inactive, app revoked, no tenant): 401.
    Rejected,
    /// An active licence of an app not in `STRATA_LICENSE_APP_KEYS`: 403.
    AppNotAllowed,
    /// The licences service could not answer: 503, fail closed.
    Unavailable,
}

#[derive(Deserialize)]
struct IntrospectBody {
    #[serde(default)]
    active: bool,
    tenant_id: Option<String>,
    app: Option<IntrospectApp>,
}

#[derive(Deserialize)]
struct IntrospectApp {
    key: Option<String>,
    revoked: Option<bool>,
}

/// Accepted licences: SHA-256 of the token → (tenant, cached at).
type LicenceCache = Mutex<LruCache<[u8; 32], (Tenant, Instant)>>;

pub struct LicenceIntrospector {
    http: reqwest::Client,
    url: String,
    app_keys: Vec<String>,
    ttl: Duration,
    /// `None` = no cache (`STRATA_LICENSE_CACHE_S=0`).
    cache: Option<LicenceCache>,
}

impl std::fmt::Debug for LicenceIntrospector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LicenceIntrospector")
            .field("url", &self.url)
            .field("app_keys", &self.app_keys)
            .field("ttl", &self.ttl)
            .finish()
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

impl LicenceIntrospector {
    pub fn new(url: &str, app_keys: &[String], cache_s: u64, timeout_ms: u64) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            // the token travels in the body: never replay it to wherever a redirect points
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| {
                StrataError::Config(format!("licence http client: {}", e.without_url()))
            })?;
        let ttl = Duration::from_secs(cache_s.min(crate::config::MAX_LICENSE_CACHE_S));
        let cache = if ttl.is_zero() {
            None
        } else {
            NonZeroUsize::new(LICENCE_CACHE_CAPACITY).map(|c| Mutex::new(LruCache::new(c)))
        };
        Ok(Self {
            http,
            url: url.to_string(),
            app_keys: app_keys.to_vec(),
            ttl,
            cache,
        })
    }

    /// `None` when `STRATA_LICENSE_INTROSPECT_URL` is unset (licence auth off).
    pub fn from_config(cfg: &Config) -> Result<Option<Self>> {
        cfg.license_introspect_url
            .as_deref()
            .map(|url| {
                Self::new(
                    url,
                    &cfg.license_app_keys,
                    cfg.license_cache_s,
                    cfg.license_timeout_ms,
                )
            })
            .transpose()
    }

    pub async fn check(&self, token: &str) -> LicenceCheck {
        let key = digest(token);
        if let Some(tenant) = self.cached(&key) {
            return LicenceCheck::Accepted(tenant);
        }
        match self.introspect(token).await {
            Ok(Ok(tenant)) => {
                if let Some(c) = &self.cache {
                    lock(c).put(key, (tenant, Instant::now() + self.ttl));
                }
                LicenceCheck::Accepted(tenant)
            }
            Ok(Err(refused)) => refused,
            Err(why) => {
                tracing::warn!(error = %why, "licence introspect failed; refusing the request");
                LicenceCheck::Unavailable
            }
        }
    }

    fn cached(&self, key: &[u8; 32]) -> Option<Tenant> {
        let mut c = lock(self.cache.as_ref()?);
        let hit = c.get(key).copied();
        match hit {
            Some((tenant, until)) if Instant::now() < until => Some(tenant),
            Some(_) => {
                c.pop(key);
                None
            }
            None => None,
        }
    }

    /// `Ok(Err(..))` = the service answered and the token is not accepted (`Rejected` or
    /// `AppNotAllowed`); `Err` = no usable answer.
    /// Error messages never carry the URL (reqwest's Display embeds it) or the token.
    async fn introspect(
        &self,
        token: &str,
    ) -> std::result::Result<std::result::Result<Tenant, LicenceCheck>, String> {
        let resp = self
            .http
            .post(&self.url)
            .json(&serde_json::json!({ "token": token }))
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    "timeout".to_string()
                } else {
                    format!("request: {}", e.without_url())
                }
            })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(format!("status {}", status.as_u16()));
        }
        let body: IntrospectBody = resp
            .json()
            .await
            .map_err(|e| format!("body: {}", e.without_url()))?;
        Ok(self.accept(&body))
    }

    fn accept(&self, b: &IntrospectBody) -> std::result::Result<Tenant, LicenceCheck> {
        let rejected = LicenceCheck::Rejected;
        if !b.active {
            return Err(rejected);
        }
        let app = b.app.as_ref().ok_or(rejected)?;
        if app.revoked.unwrap_or(false) {
            return Err(rejected);
        }
        let key = app.key.as_deref().ok_or(rejected)?;
        if !self.app_keys.iter().any(|k| k == key) {
            return Err(LicenceCheck::AppNotAllowed);
        }
        let tenant = b.tenant_id.as_deref().ok_or(rejected)?;
        Tenant::parse(tenant).map_err(|_| rejected)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Router;
    use axum::http::StatusCode;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use serde_json::{Value, json};

    use super::*;
    use crate::clients::testutil::spawn;

    pub(crate) const TENANT: &str = "0192f0c8-0000-7000-8000-00000000000a";

    pub(crate) fn active(app_key: &str) -> Value {
        json!({
            "active": true,
            "tenant_id": TENANT,
            "license_id": "l-1",
            "app": {"id": "a-1", "key": app_key, "revoked": false},
            "status": "active",
        })
    }

    /// A mock introspect endpoint: `reply(token)` answers; returns its URL and the call count.
    pub(crate) async fn mock(
        reply: impl Fn(&str) -> Response + Send + Sync + 'static,
    ) -> (String, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let reply = Arc::new(reply);
        let app = Router::new().route(
            "/license/introspect",
            post(move |axum::Json(b): axum::Json<Value>| {
                let (c, reply) = (c.clone(), reply.clone());
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    reply(b["token"].as_str().unwrap_or_default())
                }
            }),
        );
        (format!("{}/license/introspect", spawn(app).await), calls)
    }

    fn introspector(url: &str, cache_s: u64) -> LicenceIntrospector {
        LicenceIntrospector::new(url, &["strata-rag".to_string()], cache_s, 2000).unwrap()
    }

    fn tenant() -> Tenant {
        Tenant::parse(TENANT).unwrap()
    }

    #[tokio::test]
    async fn accepts_only_an_active_licence_of_an_accepted_app() {
        let (url, _) = mock(|token| {
            let body = match token {
                "good" => active("strata-rag"),
                "other-app" => active("kanban"),
                "inactive" => json!({"active": false, "reason": "license_inactive"}),
                "revoked-app" => {
                    let mut b = active("strata-rag");
                    b["app"]["revoked"] = json!(true);
                    b
                }
                "no-tenant" => {
                    let mut b = active("strata-rag");
                    b["tenant_id"] = Value::Null;
                    b
                }
                "nil-tenant" => {
                    let mut b = active("strata-rag");
                    b["tenant_id"] = json!(uuid::Uuid::nil().to_string());
                    b
                }
                "no-app" => {
                    let mut b = active("strata-rag");
                    b["app"] = Value::Null;
                    b
                }
                _ => json!({"active": false, "reason": "token_not_found"}),
            };
            axum::Json(body).into_response()
        })
        .await;
        let l = introspector(&url, 30);
        assert_eq!(l.check("good").await, LicenceCheck::Accepted(tenant()));
        assert_eq!(l.check("other-app").await, LicenceCheck::AppNotAllowed);
        for bad in [
            "inactive",
            "revoked-app",
            "no-tenant",
            "nil-tenant",
            "no-app",
            "unknown",
        ] {
            assert_eq!(l.check(bad).await, LicenceCheck::Rejected, "{bad}");
        }
    }

    #[tokio::test]
    async fn introspect_failures_fail_closed_as_unavailable() {
        let replies: [fn() -> Response; 4] = [
            || StatusCode::INTERNAL_SERVER_ERROR.into_response(),
            || StatusCode::UNAUTHORIZED.into_response(),
            || StatusCode::TEMPORARY_REDIRECT.into_response(),
            || "not json".into_response(),
        ];
        for reply in replies {
            let (url, _) = mock(move |_| reply()).await;
            assert_eq!(
                introspector(&url, 30).check("good").await,
                LicenceCheck::Unavailable
            );
        }
        // nothing listening
        let dead = {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            l.local_addr().unwrap()
        };
        let l = introspector(&format!("http://{dead}/license/introspect"), 30);
        assert_eq!(l.check("good").await, LicenceCheck::Unavailable);
        // a slow service times out
        let slow = Router::new().route(
            "/license/introspect",
            post(|| async {
                tokio::time::sleep(Duration::from_millis(500)).await;
                axum::Json(active("strata-rag"))
            }),
        );
        let url = format!("{}/license/introspect", spawn(slow).await);
        let l = LicenceIntrospector::new(&url, &["strata-rag".to_string()], 30, 50).unwrap();
        assert_eq!(l.check("good").await, LicenceCheck::Unavailable);
    }

    #[tokio::test]
    async fn accepted_tokens_are_cached_rejections_are_not() {
        let (url, calls) = mock(|token| match token {
            "good" => axum::Json(active("strata-rag")).into_response(),
            _ => axum::Json(json!({"active": false})).into_response(),
        })
        .await;
        let l = introspector(&url, 30);
        for _ in 0..3 {
            assert_eq!(l.check("good").await, LicenceCheck::Accepted(tenant()));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        for _ in 0..2 {
            assert_eq!(l.check("bad").await, LicenceCheck::Rejected);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 3);

        // cache off: every check asks
        let l = introspector(&url, 0);
        l.check("good").await;
        l.check("good").await;
        assert_eq!(calls.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn cache_ttl_is_capped_and_expired_entries_are_dropped() {
        let l = introspector("http://127.0.0.1:1/license/introspect", 3600);
        assert_eq!(
            l.ttl,
            Duration::from_secs(crate::config::MAX_LICENSE_CACHE_S)
        );
        let key = digest("good");
        let c = l.cache.as_ref().unwrap();
        lock(c).put(key, (tenant(), Instant::now() - Duration::from_secs(1)));
        assert_eq!(l.cached(&key), None);
        assert!(lock(c).is_empty());
        lock(c).put(key, (tenant(), Instant::now() + Duration::from_secs(10)));
        assert_eq!(l.cached(&key), Some(tenant()));
    }

    #[test]
    fn debug_never_shows_tokens() {
        let l = introspector("http://127.0.0.1:1/license/introspect", 30);
        lock(l.cache.as_ref().unwrap()).put(digest("s3cret"), (tenant(), Instant::now()));
        assert!(!format!("{l:?}").contains("s3cret"));
    }
}
