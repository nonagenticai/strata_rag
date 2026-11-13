//! Bearer service-token check (`STRATA_API_TOKENS`) and tenant resolution in front of every route
//! except `/health` and `/metrics`.
//!
//! Tokens are compared as SHA-256 digests with `subtle`'s constant-time equality, against every
//! configured token without an early exit, so neither the matching token nor any token's length
//! shows in the response time. Tokens are never logged.
//!
//! Tenant (spec §8.6): a `<tenant>:<token>` token acts for that tenant only (`X-Tenant-Id` may
//! repeat it; a different one is 403 `tenant_mismatch`). A `*:<token>` token, and every request
//! when auth is off (loopback only), must name the tenant in `X-Tenant-Id` (400 `tenant_required`).
//! The header is a bare tenant uuid; anything else is 400 `invalid_tenant`. The resolved tenant is
//! a request extension that handlers take as a `Tenant` extractor.
//!
//! Licence tokens (`STRATA_LICENSE_INTROSPECT_URL` set, see `super::licence`): a bearer matching no
//! static entry is introspected at the licences service; an active licence of an accepted app acts
//! for the licence's tenant like a `<tenant>:<token>` entry (a different `X-Tenant-Id` is 403
//! `tenant_mismatch`). A rejected licence is 401; an active licence of another app is 403
//! `licence_app_not_allowed`; a licences service that cannot answer is 503 `licence_unavailable`
//! (fail closed). Unset URL = static tokens only, unchanged.

use std::sync::Arc;

use axum::Json;
use axum::extract::{FromRequestParts, Request, State};
use axum::http::header::{AUTHORIZATION, WWW_AUTHENTICATE};
use axum::http::request::Parts;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use sha2::{Digest, Sha256};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::licence::{LicenceCheck, LicenceIntrospector};
use crate::config::{ApiToken, Config, TokenScope};
use crate::error::StrataError;
use crate::tenant::{Tenant, resolve_tenant};

/// The only paths served without a token (and without a tenant): the probe and the Prometheus
/// scrape. Matched exactly.
pub const PUBLIC_PATHS: [&str; 2] = ["/health", crate::metrics::METRICS_PATH];

/// The request header naming the tenant: a bare uuid.
pub const TENANT_HEADER: HeaderName = HeaderName::from_static("x-tenant-id");

const UNAUTHORIZED_MESSAGE: &str = "missing or invalid bearer token";
const LICENCE_UNAVAILABLE_MESSAGE: &str = "the licence service could not verify the token; retry";
const LICENCE_APP_NOT_ALLOWED_MESSAGE: &str =
    "this licence is for an app this service does not accept";

/// Digests of the accepted tokens with their scopes, plus the licence introspector when licence
/// auth is on. Neither = auth off (loopback-only, enforced at serve start).
#[derive(Clone, Default)]
pub struct ApiTokens {
    entries: Arc<Vec<([u8; 32], TokenScope)>>,
    licences: Option<Arc<LicenceIntrospector>>,
}

impl ApiTokens {
    pub fn new(tokens: &[ApiToken]) -> Self {
        Self {
            entries: Arc::new(
                tokens
                    .iter()
                    .map(|t| (digest(t.token.expose()), t.scope))
                    .collect(),
            ),
            licences: None,
        }
    }

    /// `STRATA_API_TOKENS` plus, when `STRATA_LICENSE_INTROSPECT_URL` is set, licence tokens.
    pub fn from_config(cfg: &Config) -> crate::error::Result<Self> {
        let mut t = Self::new(&cfg.api_tokens);
        t.licences = LicenceIntrospector::from_config(cfg)?.map(Arc::new);
        Ok(t)
    }

    pub fn with_licences(mut self, licences: LicenceIntrospector) -> Self {
        self.licences = Some(Arc::new(licences));
        self
    }

    pub fn is_enabled(&self) -> bool {
        !self.entries.is_empty() || self.licences.is_some()
    }

    /// The presented token's scope: a static entry's, else (licence auth on) the licence tenant's.
    async fn authenticate(&self, presented: Option<&str>) -> Result<TokenScope, Refusal> {
        let presented = presented.ok_or(Refusal::Unauthorized)?;
        if let Some(scope) = self.accepts(presented) {
            return Ok(scope);
        }
        match &self.licences {
            None => Err(Refusal::Unauthorized),
            Some(l) => match l.check(presented).await {
                LicenceCheck::Accepted(tenant) => Ok(TokenScope::Tenant(tenant)),
                LicenceCheck::Rejected => Err(Refusal::Unauthorized),
                LicenceCheck::AppNotAllowed => Err(Refusal::AppNotAllowed),
                LicenceCheck::Unavailable => Err(Refusal::LicenceUnavailable),
            },
        }
    }

    /// The matching token's scope. Constant time in the presented token's content and in which
    /// configured token matches: every entry is compared, and the match is selected with
    /// `conditional_assign`, never a branch or early exit. (Config rejects duplicate tokens, so at
    /// most one entry matches.)
    fn accepts(&self, presented: &str) -> Option<TokenScope> {
        let d = digest(presented);
        let mut hit = Choice::from(0);
        let mut index = 0_u64;
        for (i, (t, _)) in self.entries.iter().enumerate() {
            let eq = t.ct_eq(&d);
            index.conditional_assign(&(i as u64), eq);
            hit |= eq;
        }
        bool::from(hit).then(|| self.entries[index as usize].1)
    }
}

/// Why `authenticate` refused a bearer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// 401: no bearer, or one neither a static entry nor an active licence matches.
    Unauthorized,
    /// 403: an active licence whose app key is not in `STRATA_LICENSE_APP_KEYS`.
    AppNotAllowed,
    /// 503: licence auth is on and the licences service gave no usable answer (fail closed).
    LicenceUnavailable,
}

impl Refusal {
    fn response(self) -> Response {
        match self {
            Refusal::Unauthorized => unauthorized(),
            Refusal::AppNotAllowed => error_response(
                StatusCode::FORBIDDEN,
                "licence_app_not_allowed",
                LICENCE_APP_NOT_ALLOWED_MESSAGE,
            ),
            Refusal::LicenceUnavailable => error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "licence_unavailable",
                LICENCE_UNAVAILABLE_MESSAGE,
            ),
        }
    }
}

fn digest(s: &str) -> [u8; 32] {
    Sha256::digest(s.as_bytes()).into()
}

/// `Authorization: Bearer <token>`: scheme case-insensitive, surrounding whitespace ignored.
fn bearer(headers: &HeaderMap) -> Option<&str> {
    let v = headers.get(AUTHORIZATION)?.to_str().ok()?.trim();
    let (scheme, token) = v.split_once(' ')?;
    let token = token.trim();
    (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()).then_some(token)
}

/// `X-Tenant-Id`, parsed: `None` when absent. Exactly one header holding a bare, non-nil uuid.
fn tenant_header(headers: &HeaderMap) -> Result<Option<Tenant>, StrataError> {
    let mut values = headers.get_all(&TENANT_HEADER).iter();
    let Some(v) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(StrataError::InvalidTenant(
            "send X-Tenant-Id once".to_string(),
        ));
    }
    let v = v
        .to_str()
        .map_err(|_| StrataError::InvalidTenant("X-Tenant-Id must be a tenant uuid".to_string()))?;
    let t = v.trim();
    if t.len() >= 7 && t[..7].eq_ignore_ascii_case("bearer ") {
        return Err(StrataError::InvalidTenant(
            "X-Tenant-Id takes the bare tenant uuid, not a licence token or `Bearer ...` value"
                .to_string(),
        ));
    }
    Tenant::parse(t)
        .map(Some)
        .map_err(|_| StrataError::InvalidTenant("X-Tenant-Id must be a non-nil tenant uuid".into()))
}

pub async fn require_token(
    State(tokens): State<ApiTokens>,
    mut req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path();
    if PUBLIC_PATHS.contains(&path) {
        return next.run(req).await;
    }
    // `None` = auth off (loopback only)
    let scope = if tokens.is_enabled() {
        // owned: no borrow of the (non-`Sync`) request is held across the introspect await
        let presented = bearer(req.headers()).map(str::to_owned);
        match tokens.authenticate(presented.as_deref()).await {
            Ok(scope) => Some(scope),
            Err(refusal) => return refusal.response(),
        }
    } else {
        None
    };
    let tenant = match tenant_header(req.headers()).and_then(|h| resolve_tenant(scope, h)) {
        Ok(t) => t,
        Err(e) => return e.into_response(),
    };
    req.extensions_mut().insert(tenant);
    // the caller's attribution headers, for the cost events of this request's provider calls
    let caller = crate::metering::Caller::from_headers(req.headers());
    crate::metering::with_caller(caller, next.run(req)).await
}

/// Handlers take the tenant `require_token` resolved. A route served without that middleware has
/// none, and answers 400 `tenant_required` rather than guess one.
impl<S: Send + Sync> FromRequestParts<S> for Tenant {
    type Rejection = StrataError;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Self::Rejection> {
        parts.extensions.get::<Tenant>().copied().ok_or_else(|| {
            StrataError::TenantRequired("name the tenant in X-Tenant-Id".to_string())
        })
    }
}

/// 401 in the standard `{"error":{"code","message"}}` body, with `WWW-Authenticate: Bearer`.
fn unauthorized() -> Response {
    let mut resp = (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": {"code": "unauthorized", "message": UNAUTHORIZED_MESSAGE}})),
    )
        .into_response();
    resp.headers_mut()
        .insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    resp
}

/// `{"error":{"code","message"}}` with `status`.
fn error_response(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"error": {"code": code, "message": message}})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(AUTHORIZATION, HeaderValue::from_str(v).unwrap());
        h
    }

    #[test]
    fn bearer_parsing() {
        assert_eq!(bearer(&headers("Bearer abc")), Some("abc"));
        assert_eq!(bearer(&headers("  bearer   abc  ")), Some("abc"));
        assert_eq!(bearer(&headers("BEARER abc")), Some("abc"));
        assert_eq!(bearer(&headers("Basic abc")), None);
        assert_eq!(bearer(&headers("Bearer ")), None);
        assert_eq!(bearer(&headers("abc")), None);
        assert_eq!(bearer(&HeaderMap::new()), None);
    }

    fn tenant(n: u128) -> Tenant {
        Tenant::new(uuid::Uuid::from_u128(n)).unwrap()
    }

    #[test]
    fn only_exact_tokens_are_accepted_with_their_scope() {
        let t = ApiTokens::new(&[
            ApiToken::new(TokenScope::Tenant(tenant(1)), "tok-one"),
            ApiToken::new(TokenScope::AnyTenant, "tok-two"),
            ApiToken::new(TokenScope::Tenant(tenant(3)), "tok-three"),
        ]);
        assert!(t.is_enabled());
        assert_eq!(t.accepts("tok-one"), Some(TokenScope::Tenant(tenant(1))));
        assert_eq!(t.accepts("tok-two"), Some(TokenScope::AnyTenant));
        assert_eq!(t.accepts("tok-three"), Some(TokenScope::Tenant(tenant(3))));
        for bad in ["tok-on", "tok-one1", "tok-onetok-two", "", "TOK-ONE"] {
            assert_eq!(t.accepts(bad), None, "{bad}");
        }
        assert!(!ApiTokens::new(&[]).is_enabled());
    }

    fn tenant_headers(values: &[&str]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for v in values {
            h.append(TENANT_HEADER, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn tenant_header_is_one_bare_non_nil_uuid() {
        let id = uuid::Uuid::now_v7();
        assert_eq!(tenant_header(&HeaderMap::new()).unwrap(), None);
        assert_eq!(
            tenant_header(&tenant_headers(&[&format!(" {id} ")]))
                .unwrap()
                .map(Tenant::id),
            Some(id)
        );
        for bad in [
            vec![format!("Bearer {id}")],
            vec![format!("bearer {id}")],
            vec!["not-a-uuid".to_string()],
            vec![uuid::Uuid::nil().to_string()],
            vec![String::new()],
            vec![id.to_string(), id.to_string()],
        ] {
            let refs: Vec<&str> = bad.iter().map(String::as_str).collect();
            let err = tenant_header(&tenant_headers(&refs)).unwrap_err();
            assert_eq!(err.code(), "invalid_tenant", "{bad:?}");
            assert_eq!(err.status(), 400);
        }
        let err = tenant_header(&tenant_headers(&[&format!("Bearer {id}")])).unwrap_err();
        assert!(err.to_string().contains("licence token"), "{err}");
    }

    /// `/health` and `/metrics` are the only open paths, matched exactly: a prefix match
    /// (`starts_with("/health")`) or a trailing-slash variant would open every route that happens
    /// to share the prefix.
    #[tokio::test]
    async fn only_the_exact_public_paths_are_open() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let tokens = ApiTokens::new(&[ApiToken::new(TokenScope::AnyTenant, "tok")]);
        // Every path answers 200 behind the gate, so the status is the gate's decision alone.
        let app = Router::new()
            .fallback(|| async { StatusCode::OK })
            .layer(axum::middleware::from_fn_with_state(tokens, require_token));
        let status = |uri: &'static str| {
            let app = app.clone();
            async move {
                let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
                app.oneshot(req).await.unwrap().status()
            }
        };
        assert_eq!(status("/health").await, StatusCode::OK);
        assert_eq!(status("/health?verbose=1").await, StatusCode::OK);
        assert_eq!(status("/metrics").await, StatusCode::OK);
        for gated in [
            "/healthz",
            "/health-anything",
            "/health/",
            "/health/x",
            "/healthy",
            "/Health",
            "//health",
            "/api/health",
            "/metrics/",
            "/metricsx",
            "/api/metrics",
        ] {
            assert_eq!(status(gated).await, StatusCode::UNAUTHORIZED, "{gated}");
        }
    }

    /// End to end through the middleware: static tokens keep working unchanged, a licence token
    /// acts for its licence's tenant, a disagreeing `X-Tenant-Id` is 403, a rejected licence 401,
    /// and an unreachable licences service 503.
    #[tokio::test]
    async fn licence_tokens_through_the_middleware() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        use crate::api::licence::tests::{TENANT, active, mock};

        let (url, calls) = mock(|token| match token {
            "lic-good" => Json(active("strata-rag")).into_response(),
            "lic-down" => StatusCode::BAD_GATEWAY.into_response(),
            "lic-other-app" => Json(active("kanban")).into_response(),
            _ => Json(json!({"active": false, "reason": "token_not_found"})).into_response(),
        })
        .await;
        let licences =
            LicenceIntrospector::new(&url, &["strata-rag".to_string()], 30, 2000).unwrap();
        let tokens = ApiTokens::new(&[ApiToken::new(TokenScope::Tenant(tenant(1)), "static")])
            .with_licences(licences);
        // echoes the resolved tenant
        let app = Router::new()
            .fallback(|t: Tenant| async move { t.to_string() })
            .layer(axum::middleware::from_fn_with_state(tokens, require_token));
        let call = |bearer: Option<&'static str>, tenant_hdr: Option<String>| {
            let app = app.clone();
            async move {
                let mut b = Request::builder().uri("/api/stores");
                if let Some(t) = bearer {
                    b = b.header(AUTHORIZATION, format!("Bearer {t}"));
                }
                if let Some(h) = tenant_hdr {
                    b = b.header(TENANT_HEADER, h);
                }
                let resp = app.oneshot(b.body(Body::empty()).unwrap()).await.unwrap();
                let status = resp.status();
                let body = http_body_util::BodyExt::collect(resp.into_body())
                    .await
                    .unwrap()
                    .to_bytes();
                (status, String::from_utf8(body.to_vec()).unwrap())
            }
        };

        // the static token is matched first and never introspected
        assert_eq!(
            call(Some("static"), None).await,
            (StatusCode::OK, tenant(1).to_string())
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);

        assert_eq!(
            call(Some("lic-good"), None).await,
            (StatusCode::OK, TENANT.to_string())
        );
        assert_eq!(
            call(Some("lic-good"), Some(TENANT.to_string())).await,
            (StatusCode::OK, TENANT.to_string())
        );
        let (status, body) = call(Some("lic-good"), Some(tenant(2).to_string())).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert!(body.contains("tenant_mismatch"), "{body}");

        let (status, body) = call(Some("lic-unknown"), None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(body.contains("unauthorized"), "{body}");
        assert_eq!(call(None, None).await.0, StatusCode::UNAUTHORIZED);

        let (status, body) = call(Some("lic-down"), None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains("licence_unavailable"), "{body}");

        let (status, body) = call(Some("lic-other-app"), None).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert!(body.contains("licence_app_not_allowed"), "{body}");
        assert!(!body.contains("lic-down"), "{body}");
    }

    /// Without `STRATA_LICENSE_INTROSPECT_URL` an unknown bearer is 401, as before, and nothing
    /// is introspected.
    #[tokio::test]
    async fn without_licence_auth_unknown_bearers_are_401() {
        let cfg = Config::from_map(&std::collections::HashMap::from([(
            "STRATA_API_TOKENS".to_string(),
            "*:tok".to_string(),
        )]))
        .unwrap();
        let t = ApiTokens::from_config(&cfg).unwrap();
        assert!(t.licences.is_none() && t.is_enabled());
        assert_eq!(
            t.authenticate(Some("lic-anything")).await,
            Err(Refusal::Unauthorized)
        );
        assert_eq!(t.authenticate(None).await, Err(Refusal::Unauthorized));
        assert_eq!(t.authenticate(Some("tok")).await, Ok(TokenScope::AnyTenant));
        // licence auth alone turns auth on
        let cfg = Config::from_map(&std::collections::HashMap::from([(
            "STRATA_LICENSE_INTROSPECT_URL".to_string(),
            "http://127.0.0.1:1/license/introspect".to_string(),
        )]))
        .unwrap();
        assert!(ApiTokens::from_config(&cfg).unwrap().is_enabled());
    }
}
