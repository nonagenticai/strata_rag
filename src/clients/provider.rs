//! Shared HTTP plumbing for the embed and rerank providers: bearer auth held in a sensitive
//! `HeaderValue`, a per-client timeout from config, status classification and a small bounded retry.
//!
//! Every failure surfaces as `StrataError::Upstream { service, message }` with `message` starting with
//! its class (`auth: status 401`), which `/health` reports through [`error_class`]:
//! - `auth`: 401 / 403. Never retried.
//! - `retryable`: 408 / 429 / 5xx, a connection failure, or our own timeout. Statuses and connection
//!   failures are retried up to [`MAX_RETRIES`] times with a short backoff; a timeout is not retried
//!   (each attempt already spent the whole timeout, and the query path cannot afford three).
//! - `invalid_request`: 400 / 413 / 422 and any other 4xx. Never retried.
//! - `invalid_response`: a 2xx body that does not decode or breaks the contract (count, dims, index).

use std::fmt;
use std::time::Duration;

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::config::{Secret, redact_url};
use crate::error::{Result, StrataError};

/// Retries after the first attempt, for `retryable` statuses and connection failures only.
pub const MAX_RETRIES: usize = 2;
/// Sleep before retry 1, 2, ...
const BACKOFF_MS: [u64; MAX_RETRIES] = [100, 400];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorClass {
    Auth,
    Retryable,
    InvalidRequest,
    InvalidResponse,
}

impl ErrorClass {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorClass::Auth => "auth",
            ErrorClass::Retryable => "retryable",
            ErrorClass::InvalidRequest => "invalid_request",
            ErrorClass::InvalidResponse => "invalid_response",
        }
    }

    fn of_status(status: u16) -> ErrorClass {
        match status {
            401 | 403 => ErrorClass::Auth,
            408 | 429 | 500..=599 => ErrorClass::Retryable,
            400..=499 => ErrorClass::InvalidRequest,
            _ => ErrorClass::InvalidResponse,
        }
    }
}

const CLASSES: [ErrorClass; 4] = [
    ErrorClass::Auth,
    ErrorClass::Retryable,
    ErrorClass::InvalidRequest,
    ErrorClass::InvalidResponse,
];

/// The provider error class of `e` (`auth`, `retryable`, `invalid_request`, `invalid_response`) when
/// it came from a provider client, else `internal`. Never contains message text, so it is safe to
/// put on the unauthenticated `/health`.
pub fn error_class(e: &StrataError) -> &'static str {
    if let StrataError::Upstream { message, .. } = e {
        let head = message.split(':').next().unwrap_or("");
        if let Some(c) = CLASSES.iter().find(|c| c.as_str() == head) {
            return c.as_str();
        }
    }
    "internal"
}

struct Failure {
    class: ErrorClass,
    detail: String,
    retry: bool,
}

/// One provider endpoint family (`{base_url}/...`) with optional bearer auth.
pub struct ProviderHttp {
    service: &'static str,
    http: reqwest::Client,
    base_url: String,
    auth: Option<HeaderValue>,
    /// Retries for `retryable` statuses / connection failures ([`MAX_RETRIES`]; 0 for chat).
    max_retries: usize,
}

impl fmt::Debug for ProviderHttp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderHttp")
            .field("service", &self.service)
            .field("base_url", &redact_url(&self.base_url))
            .field("auth", &self.auth.as_ref().map(|_| "Bearer ***"))
            .finish_non_exhaustive()
    }
}

impl ProviderHttp {
    /// `key_var` names where the key came from, for the error message only.
    pub fn new(
        service: &'static str,
        base_url: &str,
        api_key: Option<Secret>,
        key_var: &str,
        timeout_ms: u64,
    ) -> Result<Self> {
        Self::build(service, base_url, api_key, key_var, timeout_ms, false)
    }

    /// `guarded`: resolve through `crate::providers::ssrf::GuardedResolver` (no internal address
    /// is ever connected to, also after a DNS change) and ignore proxy env vars. For tenant
    /// registrations; deployment defaults are operator-set and unguarded.
    pub fn build(
        service: &'static str,
        base_url: &str,
        api_key: Option<Secret>,
        key_var: &str,
        timeout_ms: u64,
        guarded: bool,
    ) -> Result<Self> {
        // no redirects: a 3xx is a misconfigured base URL (`invalid_response`), and following it
        // could carry the bearer key to another host
        let mut builder = reqwest::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .redirect(reqwest::redirect::Policy::none());
        if guarded {
            builder = builder
                .no_proxy()
                .dns_resolver(crate::providers::ssrf::GuardedResolver);
        }
        let http = builder.build().map_err(|e| {
            StrataError::Config(format!("{service} http client: {}", e.without_url()))
        })?;
        let auth = match api_key {
            None => None,
            Some(key) => {
                let mut v =
                    HeaderValue::from_str(&format!("Bearer {}", key.expose())).map_err(|_| {
                        StrataError::Config(format!("{key_var} contains invalid header characters"))
                    })?;
                v.set_sensitive(true);
                Some(v)
            }
        };
        Ok(Self {
            service,
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            auth,
            max_retries: MAX_RETRIES,
        })
    }

    /// No retry at all (chat: a generation is slow and costly; a retry can double it).
    pub fn without_retries(self) -> Self {
        Self {
            max_retries: 0,
            ..self
        }
    }

    pub fn service(&self) -> &'static str {
        self.service
    }

    /// `StrataError::Upstream` for this provider, message prefixed with the class.
    pub fn error(&self, class: ErrorClass, detail: impl fmt::Display) -> StrataError {
        StrataError::Upstream {
            service: self.service,
            message: format!("{}: {detail}", class.as_str()),
        }
    }

    /// POST `body` as JSON to `{base_url}{path}` and decode the 2xx response as `T`, retrying
    /// retryable statuses and connection failures (see the module docs).
    ///
    /// The response body is read up to `max_bytes`; a longer one is an `invalid_response` and is
    /// never parsed (a provider cannot make strata buffer an unbounded body).
    pub async fn post_json<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &Value,
        max_bytes: usize,
    ) -> Result<T> {
        self.post_json_with(path, body, max_bytes, &HeaderMap::new())
            .await
    }

    /// [`Self::post_json`] with `extra` request headers (e.g. chat's `X-Effective-Tenant-Id`).
    pub async fn post_json_with<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &Value,
        max_bytes: usize,
        extra: &HeaderMap,
    ) -> Result<T> {
        let bytes = serde_json::to_vec(body)
            .map_err(|e| self.error(ErrorClass::InvalidRequest, format!("encode: {e}")))?;
        let url = format!("{}{path}", self.base_url);
        let mut attempt = 0;
        loop {
            match self.post_once(&url, &bytes, max_bytes, extra).await {
                Ok(raw) => {
                    return serde_json::from_slice(&raw).map_err(|e| {
                        self.error(ErrorClass::InvalidResponse, format!("decode: {e}"))
                    });
                }
                Err(f) if f.retry && attempt < self.max_retries => {
                    tracing::debug!(
                        service = self.service,
                        attempt,
                        class = f.class.as_str(),
                        detail = %f.detail,
                        "provider call failed; retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(BACKOFF_MS[attempt])).await;
                    attempt += 1;
                }
                Err(f) => return Err(self.error(f.class, f.detail)),
            }
        }
    }

    /// GET `{base_url}{path}` and require a 2xx (a cheap liveness call, e.g. `/models` for
    /// `/health`). No retry; the body is not read.
    pub async fn get_ok(&self, path: &str) -> Result<()> {
        let mut req = self.http.get(format!("{}{path}", self.base_url));
        if let Some(auth) = &self.auth {
            req = req.header(AUTHORIZATION, auth.clone());
        }
        let resp = req.send().await.map_err(|e| {
            let f = transport(e);
            self.error(f.class, f.detail)
        })?;
        let status = resp.status();
        if !status.is_success() {
            return Err(self.error(
                ErrorClass::of_status(status.as_u16()),
                format!("status {}", status.as_u16()),
            ));
        }
        Ok(())
    }

    async fn post_once(
        &self,
        url: &str,
        body: &[u8],
        max_bytes: usize,
        extra: &HeaderMap,
    ) -> std::result::Result<Vec<u8>, Failure> {
        let mut req = self
            .http
            .post(url)
            .headers(extra.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(body.to_vec());
        if let Some(auth) = &self.auth {
            req = req.header(AUTHORIZATION, auth.clone());
        }
        let resp = req.send().await.map_err(transport)?;
        let status = resp.status();
        if !status.is_success() {
            let class = ErrorClass::of_status(status.as_u16());
            return Err(Failure {
                class,
                detail: format!("status {}", status.as_u16()),
                retry: class == ErrorClass::Retryable,
            });
        }
        read_capped(resp, max_bytes).await
    }
}

/// The body, streamed, refused as soon as it exceeds `max_bytes` (also up front from a declared
/// `Content-Length`).
async fn read_capped(
    mut resp: reqwest::Response,
    max_bytes: usize,
) -> std::result::Result<Vec<u8>, Failure> {
    let too_large = || Failure {
        class: ErrorClass::InvalidResponse,
        detail: format!("response body exceeds {max_bytes} bytes"),
        retry: false,
    };
    if resp.content_length().is_some_and(|n| n > max_bytes as u64) {
        return Err(too_large());
    }
    let mut out = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(transport)? {
        if out.len() + chunk.len() > max_bytes {
            return Err(too_large());
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

/// `without_url`: reqwest's Display embeds the request URL; keep it out of messages and logs.
fn transport(e: reqwest::Error) -> Failure {
    if e.is_timeout() {
        Failure {
            class: ErrorClass::Retryable,
            detail: "timeout".to_string(),
            retry: false,
        }
    } else {
        Failure {
            class: ErrorClass::Retryable,
            detail: format!("request: {}", e.without_url()),
            retry: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_map_to_their_class() {
        for (s, c) in [
            (401, ErrorClass::Auth),
            (403, ErrorClass::Auth),
            (408, ErrorClass::Retryable),
            (429, ErrorClass::Retryable),
            (500, ErrorClass::Retryable),
            (503, ErrorClass::Retryable),
            (400, ErrorClass::InvalidRequest),
            (413, ErrorClass::InvalidRequest),
            (422, ErrorClass::InvalidRequest),
            (404, ErrorClass::InvalidRequest),
            (302, ErrorClass::InvalidResponse),
        ] {
            assert_eq!(ErrorClass::of_status(s), c, "status {s}");
        }
    }

    #[test]
    fn error_class_reads_only_provider_errors() {
        let up = |m: &str| StrataError::Upstream {
            service: "embed",
            message: m.to_string(),
        };
        assert_eq!(error_class(&up("auth: status 401")), "auth");
        assert_eq!(error_class(&up("retryable: timeout")), "retryable");
        assert_eq!(
            error_class(&up("invalid_request: status 400")),
            "invalid_request"
        );
        assert_eq!(
            error_class(&up("invalid_response: decode: x")),
            "invalid_response"
        );
        assert_eq!(error_class(&up("timeout")), "internal");
        assert_eq!(error_class(&StrataError::Model("x".into())), "internal");
    }
}
