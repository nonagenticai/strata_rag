//! Embed provider client, OpenAI embeddings shape: `POST {base}/embeddings` with
//! `{model, input, dimensions, encoding_format: "float", input_type}` (`input_type` is an extension
//! a provider may ignore), answered by `{data: [{index, embedding}], usage?}`.
//!
//! The response is sorted by `index` and must hold exactly one `crate::vecmath::DIM`-long finite
//! vector per input; each vector is L2-normalised here, whatever the provider did. Auth, timeout,
//! error classes and retry: see `super::provider`.

use std::fmt;

use serde::Deserialize;
use serde_json::json;

use super::provider::{ErrorClass, ProviderHttp};
use crate::config::Secret;
use crate::error::Result;
use crate::inference::{EmbedProvider, InputType};
use crate::routing::BoxFut;
use crate::vecmath::DIM;

/// Largest accepted embed response for `n` texts: `n` × 1024 floats × ~24 bytes of JSON + 64 KiB.
pub fn max_response_bytes(n: usize) -> usize {
    n.saturating_mul(DIM * 24).saturating_add(64 * 1024)
}

/// How far from 1 a provider vector's norm may be in the registration probe.
pub const UNIT_NORM_TOLERANCE: f32 = 1e-2;

pub struct EmbedClient {
    http: ProviderHttp,
    model: String,
}

impl fmt::Debug for EmbedClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbedClient")
            .field("http", &self.http)
            .field("model", &self.model)
            .finish()
    }
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    index: usize,
    embedding: Vec<f32>,
}

impl EmbedClient {
    pub fn new(
        base_url: &str,
        api_key: Option<Secret>,
        model: &str,
        timeout_ms: u64,
    ) -> Result<Self> {
        Self::with_guard(base_url, api_key, model, timeout_ms, false)
    }

    /// `guarded`: see `ProviderHttp::build` (tenant-registered providers).
    pub fn with_guard(
        base_url: &str,
        api_key: Option<Secret>,
        model: &str,
        timeout_ms: u64,
        guarded: bool,
    ) -> Result<Self> {
        Ok(Self {
            http: ProviderHttp::build(
                "embed",
                base_url,
                api_key,
                "the embed api_key",
                timeout_ms,
                guarded,
            )?,
            model: model.to_string(),
        })
    }

    /// One unit vector per text, in input order. Empty input → no call.
    pub async fn embed(&self, texts: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({
            "model": self.model,
            "input": texts,
            "dimensions": DIM,
            "encoding_format": "float",
            "input_type": input_type,
        });
        let resp: EmbedResponse = self
            .http
            .post_json("/embeddings", &body, max_response_bytes(texts.len()))
            .await?;
        self.check(resp, texts.len(), false)
    }

    /// Registration probe: one 1-text query embed that must come back as one 1024-d vector the
    /// provider ALREADY normalised (the contract), not one strata had to normalise.
    pub async fn probe(&self) -> Result<()> {
        let texts = ["strata provider probe".to_string()];
        let body = json!({
            "model": self.model,
            "input": texts,
            "dimensions": DIM,
            "encoding_format": "float",
            "input_type": InputType::Query,
        });
        let resp: EmbedResponse = self
            .http
            .post_json("/embeddings", &body, max_response_bytes(texts.len()))
            .await?;
        self.check(resp, 1, true).map(|_| ())
    }

    /// `unit`: refuse a vector whose norm is off 1 by more than [`UNIT_NORM_TOLERANCE`].
    fn check(&self, resp: EmbedResponse, expected: usize, unit: bool) -> Result<Vec<Vec<f32>>> {
        let bad = |detail: String| self.http.error(ErrorClass::InvalidResponse, detail);
        let mut data = resp.data;
        if data.len() != expected {
            return Err(bad(format!(
                "{} embeddings for {expected} inputs",
                data.len()
            )));
        }
        data.sort_by_key(|item| item.index);
        let mut out = Vec::with_capacity(expected);
        for (pos, item) in data.into_iter().enumerate() {
            if item.index != pos {
                return Err(bad(format!(
                    "embedding indices are not 0..{expected} (missing {pos})"
                )));
            }
            let mut v = item.embedding;
            if v.len() != DIM {
                return Err(bad(format!(
                    "embedding {pos} has {} dims, expected {DIM}",
                    v.len()
                )));
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if !norm.is_finite() || norm == 0.0 {
                return Err(bad(format!("embedding {pos} has norm {norm}")));
            }
            if unit && (norm - 1.0).abs() > UNIT_NORM_TOLERANCE {
                return Err(bad(format!(
                    "embedding {pos} is not L2-normalised (norm {norm})"
                )));
            }
            for x in &mut v {
                *x /= norm;
            }
            out.push(v);
        }
        Ok(out)
    }
}

impl EmbedProvider for EmbedClient {
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        input_type: InputType,
    ) -> BoxFut<'a, Result<Vec<Vec<f32>>>> {
        Box::pin(EmbedClient::embed(self, texts, input_type))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use axum::Router;
    use axum::http::{HeaderMap, StatusCode, header};
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use serde_json::{Value, json};

    use super::*;
    use crate::clients::testutil::spawn;
    use crate::error::StrataError;

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("text {i}")).collect()
    }

    /// A vector of `DIM` with `scale` at position `hot` (so the norm is `scale`).
    fn vec_at(hot: usize, scale: f32) -> Vec<f32> {
        let mut v = vec![0.0; DIM];
        v[hot] = scale;
        v
    }

    /// Serves `/embeddings` with `respond(call_no, body, headers)`, recording every request body.
    async fn mock<F>(
        respond: F,
    ) -> (
        String,
        Arc<Mutex<Vec<(Option<String>, Value)>>>,
        Arc<AtomicUsize>,
    )
    where
        F: Fn(usize, &Value) -> Response + Send + Sync + 'static,
    {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let respond = Arc::new(respond);
        let (s, c) = (seen.clone(), calls.clone());
        let app = Router::new().route(
            "/embeddings",
            post(
                move |headers: HeaderMap, axum::Json(body): axum::Json<Value>| {
                    let (s, c, respond) = (s.clone(), c.clone(), respond.clone());
                    async move {
                        let auth = headers
                            .get(header::AUTHORIZATION)
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_owned);
                        let n = c.fetch_add(1, Ordering::SeqCst);
                        let r = respond(n, &body);
                        s.lock().unwrap().push((auth, body));
                        r
                    }
                },
            ),
        );
        (spawn(app).await, seen, calls)
    }

    /// `{data}` for `n` inputs, item i hot at i, listed in `order`.
    fn data(order: &[usize], scale: f32) -> Response {
        let items: Vec<Value> = order
            .iter()
            .map(|&i| json!({"index": i, "embedding": vec_at(i, scale)}))
            .collect();
        axum::Json(json!({"data": items, "usage": {"prompt_tokens": 3}})).into_response()
    }

    fn client(base: &str, key: Option<&str>) -> EmbedClient {
        EmbedClient::new(base, key.map(Secret::new), "BAAI/bge-m3", 2000).unwrap()
    }

    fn class_of(e: &StrataError) -> &'static str {
        crate::clients::provider::error_class(e)
    }

    #[tokio::test]
    async fn happy_path_sends_the_contract_body_and_bearer() {
        let (base, seen, _) = mock(|_, _| data(&[0, 1], 1.0)).await;
        let got = client(&base, Some("ek-123"))
            .embed(&texts(2), InputType::Document)
            .await
            .unwrap();
        assert_eq!(got, vec![vec_at(0, 1.0), vec_at(1, 1.0)]);
        let seen = seen.lock().unwrap();
        let (auth, body) = &seen[0];
        assert_eq!(auth.as_deref(), Some("Bearer ek-123"));
        assert_eq!(
            body,
            &json!({"model": "BAAI/bge-m3", "input": ["text 0", "text 1"], "dimensions": 1024,
                    "encoding_format": "float", "input_type": "document"})
        );
    }

    #[tokio::test]
    async fn no_key_sends_no_authorization_header_and_query_type() {
        let (base, seen, _) = mock(|_, _| data(&[0], 1.0)).await;
        client(&base, None)
            .embed(&texts(1), InputType::Query)
            .await
            .unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].0, None);
        assert_eq!(seen[0].1["input_type"], "query");
    }

    #[tokio::test]
    async fn out_of_order_indices_are_sorted_and_vectors_normalised() {
        let (base, _, _) = mock(|_, _| data(&[2, 0, 1], 3.0)).await;
        let got = client(&base, None)
            .embed(&texts(3), InputType::Document)
            .await
            .unwrap();
        assert_eq!(got, vec![vec_at(0, 1.0), vec_at(1, 1.0), vec_at(2, 1.0)]);
    }

    #[tokio::test]
    async fn wrong_dims_is_an_invalid_response() {
        let (base, _, _) = mock(|_, _| {
            axum::Json(json!({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})).into_response()
        })
        .await;
        let err = client(&base, None)
            .embed(&texts(1), InputType::Document)
            .await
            .unwrap_err();
        assert_eq!(class_of(&err), "invalid_response", "{err}");
        assert!(err.to_string().contains("2 dims"), "{err}");
    }

    #[tokio::test]
    async fn count_mismatch_and_duplicate_index_are_invalid_responses() {
        let (base, _, _) = mock(|_, _| data(&[0], 1.0)).await;
        let err = client(&base, None)
            .embed(&texts(2), InputType::Document)
            .await
            .unwrap_err();
        assert_eq!(class_of(&err), "invalid_response", "{err}");
        assert!(
            err.to_string().contains("1 embeddings for 2 inputs"),
            "{err}"
        );

        let (base, _, _) = mock(|_, _| data(&[0, 0], 1.0)).await;
        let err = client(&base, None)
            .embed(&texts(2), InputType::Document)
            .await
            .unwrap_err();
        assert_eq!(class_of(&err), "invalid_response", "{err}");
    }

    #[tokio::test]
    async fn status_401_is_auth_and_not_retried() {
        let (base, _, calls) = mock(|_, _| StatusCode::UNAUTHORIZED.into_response()).await;
        let err = client(&base, Some("bad"))
            .embed(&texts(1), InputType::Query)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                StrataError::Upstream {
                    service: "embed",
                    ..
                }
            ),
            "{err:?}"
        );
        assert_eq!(class_of(&err), "auth");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn status_429_is_retried_then_succeeds() {
        let (base, _, calls) = mock(|n, _| {
            if n < 2 {
                StatusCode::TOO_MANY_REQUESTS.into_response()
            } else {
                data(&[0], 1.0)
            }
        })
        .await;
        let got = client(&base, None)
            .embed(&texts(1), InputType::Document)
            .await
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn persistent_503_gives_up_after_the_bounded_retries() {
        let (base, _, calls) = mock(|_, _| StatusCode::SERVICE_UNAVAILABLE.into_response()).await;
        let err = client(&base, None)
            .embed(&texts(1), InputType::Document)
            .await
            .unwrap_err();
        assert_eq!(class_of(&err), "retryable");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1 + crate::clients::provider::MAX_RETRIES
        );
    }

    #[tokio::test]
    async fn status_400_is_invalid_request_and_not_retried() {
        let (base, _, calls) = mock(|_, _| StatusCode::BAD_REQUEST.into_response()).await;
        let err = client(&base, None)
            .embed(&texts(1), InputType::Document)
            .await
            .unwrap_err();
        assert_eq!(class_of(&err), "invalid_request");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn empty_input_makes_no_call() {
        let (base, _, calls) = mock(|_, _| data(&[0], 1.0)).await;
        let got = client(&base, None)
            .embed(&[], InputType::Document)
            .await
            .unwrap();
        assert!(got.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_redirect_is_an_invalid_response_and_is_not_followed() {
        let followed = Arc::new(AtomicUsize::new(0));
        let f = followed.clone();
        let app = Router::new()
            .route(
                "/embeddings",
                post(|| async {
                    (StatusCode::FOUND, [(header::LOCATION, "/elsewhere")]).into_response()
                }),
            )
            .route(
                "/elsewhere",
                post(move || {
                    let f = f.clone();
                    async move {
                        f.fetch_add(1, Ordering::SeqCst);
                        data(&[0], 1.0)
                    }
                })
                .get(|| async { data(&[0], 1.0) }),
            );
        let base = spawn(app).await;
        let err = client(&base, Some("k"))
            .embed(&texts(1), InputType::Query)
            .await
            .unwrap_err();
        assert_eq!(class_of(&err), "invalid_response", "{err}");
        assert!(err.to_string().contains("status 302"), "{err}");
        assert_eq!(
            followed.load(Ordering::SeqCst),
            0,
            "the redirect was followed"
        );
    }

    #[test]
    fn the_key_never_appears_in_debug_output() {
        let c = EmbedClient::new(
            "http://user:pw@127.0.0.1:1/",
            Some(Secret::new("ek-SECRET-42")),
            "m",
            1000,
        )
        .unwrap();
        let dbg = format!("{c:?}");
        assert!(!dbg.contains("ek-SECRET-42"), "{dbg}");
        assert!(!dbg.contains("pw@"), "{dbg}");
        assert!(dbg.contains("Bearer ***"), "{dbg}");
    }

    #[test]
    fn a_key_with_header_breaking_characters_is_a_config_error() {
        let err = EmbedClient::new("http://x", Some(Secret::new("a\nb")), "m", 1000).unwrap_err();
        assert!(matches!(err, StrataError::Config(_)), "{err:?}");
        assert!(err.to_string().contains("embed api_key"));
        assert!(!err.to_string().contains("a\nb"));
    }
}
