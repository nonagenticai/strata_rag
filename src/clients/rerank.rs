//! Rerank provider client, Cohere v2 shape: `POST {base}/rerank` with
//! `{model, query, documents, top_n: documents.len()}`, answered by
//! `{results: [{index, relevance_score}]}` in any order.
//!
//! Scores come back in INPUT order (mapped by `index`); a missing, duplicate or out-of-range index or
//! a non-finite score is an `invalid_response`. Auth, timeout, error classes and retry: see
//! `super::provider`.

use std::fmt;

use serde::Deserialize;
use serde_json::json;

use super::provider::{ErrorClass, ProviderHttp};
use crate::config::Secret;
use crate::error::Result;
use crate::inference::RerankProvider;
use crate::routing::BoxFut;

/// Largest accepted rerank response.
pub const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

pub struct RerankClient {
    http: ProviderHttp,
    model: String,
}

impl fmt::Debug for RerankClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RerankClient")
            .field("http", &self.http)
            .field("model", &self.model)
            .finish()
    }
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankItem>,
}

#[derive(Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f32,
}

impl RerankClient {
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
                "rerank",
                base_url,
                api_key,
                "the rerank api_key",
                timeout_ms,
                guarded,
            )?,
            model: model.to_string(),
        })
    }

    /// One score per passage, in input order. Empty input → no call.
    pub async fn rerank(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({
            "model": self.model,
            "query": query,
            "documents": passages,
            "top_n": passages.len(),
        });
        let resp: RerankResponse = self
            .http
            .post_json("/rerank", &body, MAX_RESPONSE_BYTES)
            .await?;
        self.check(resp, passages.len())
    }

    /// Registration probe: one 2-document rerank that must score each document exactly once
    /// (index in range, no duplicate) with a finite score in [0, 1] (the contract).
    pub async fn probe(&self) -> Result<()> {
        let docs = [
            "Paris is the capital of France.".to_string(),
            "Bananas are rich in potassium.".to_string(),
        ];
        let scores = self.rerank("What is the capital of France?", &docs).await?;
        if let Some(s) = scores.iter().find(|s| !(0.0..=1.0).contains(*s)) {
            return Err(self.http.error(
                ErrorClass::InvalidResponse,
                format!("relevance_score {s} outside [0, 1]"),
            ));
        }
        Ok(())
    }

    fn check(&self, resp: RerankResponse, expected: usize) -> Result<Vec<f32>> {
        let bad = |detail: String| self.http.error(ErrorClass::InvalidResponse, detail);
        let mut scores: Vec<Option<f32>> = vec![None; expected];
        for item in resp.results {
            let Some(slot) = scores.get_mut(item.index) else {
                return Err(bad(format!(
                    "result index {} out of range for {expected} documents",
                    item.index
                )));
            };
            if slot.is_some() {
                return Err(bad(format!("duplicate result index {}", item.index)));
            }
            if !item.relevance_score.is_finite() {
                return Err(bad(format!(
                    "non-finite relevance_score for index {}",
                    item.index
                )));
            }
            *slot = Some(item.relevance_score);
        }
        scores
            .into_iter()
            .enumerate()
            .map(|(i, s)| s.ok_or_else(|| bad(format!("no result for document {i}"))))
            .collect()
    }
}

impl RerankProvider for RerankClient {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        passages: &'a [String],
    ) -> BoxFut<'a, Result<Vec<f32>>> {
        Box::pin(RerankClient::rerank(self, query, passages))
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
    use crate::clients::provider::error_class;
    use crate::clients::testutil::spawn;
    use crate::error::StrataError;

    type Seen = Arc<Mutex<Vec<(Option<String>, Value)>>>;

    async fn mock<F>(respond: F) -> (String, Seen, Arc<AtomicUsize>)
    where
        F: Fn(usize) -> Response + Send + Sync + 'static,
    {
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let respond = Arc::new(respond);
        let (s, c) = (seen.clone(), calls.clone());
        let app = Router::new().route(
            "/rerank",
            post(
                move |headers: HeaderMap, axum::Json(body): axum::Json<Value>| {
                    let (s, c, respond) = (s.clone(), c.clone(), respond.clone());
                    async move {
                        let auth = headers
                            .get(header::AUTHORIZATION)
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_owned);
                        let n = c.fetch_add(1, Ordering::SeqCst);
                        s.lock().unwrap().push((auth, body));
                        respond(n)
                    }
                },
            ),
        );
        (spawn(app).await, seen, calls)
    }

    fn results(items: &[(usize, f32)]) -> Response {
        let r: Vec<Value> = items
            .iter()
            .map(|(i, s)| json!({"index": i, "relevance_score": s}))
            .collect();
        axum::Json(json!({"id": "x", "results": r})).into_response()
    }

    fn client(base: &str, key: Option<&str>) -> RerankClient {
        RerankClient::new(base, key.map(Secret::new), "BAAI/bge-reranker-base", 2000).unwrap()
    }

    fn docs(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("doc {i}")).collect()
    }

    /// A body over the cap is `invalid_response` and never parsed: with a declared length
    /// (refused up front) and streamed without one (refused once the cap is crossed).
    #[tokio::test]
    async fn a_response_body_over_the_cap_is_refused() {
        let padding = "x".repeat(MAX_RESPONSE_BYTES + 1);
        let declared = padding.clone();
        let (base, _, _) = mock(move |_| {
            let body = format!(r#"{{"results":[],"pad":"{declared}"}}"#);
            body.into_response()
        })
        .await;
        let err = client(&base, None).rerank("q", &docs(1)).await.unwrap_err();
        assert_eq!(error_class(&err), "invalid_response", "{err}");
        assert!(err.to_string().contains("exceeds"), "{err}");

        let (base, _, calls) = mock(move |_| {
            let chunk = bytes_chunk(64 * 1024);
            let stream = futures_util::stream::iter(
                (0..64).map(move |_| Ok::<_, std::io::Error>(chunk.clone())),
            );
            axum::body::Body::from_stream(stream).into_response()
        })
        .await;
        let err = client(&base, None).rerank("q", &docs(1)).await.unwrap_err();
        assert_eq!(error_class(&err), "invalid_response", "{err}");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "not retried");
        drop(padding);
    }

    fn bytes_chunk(n: usize) -> Vec<u8> {
        vec![b' '; n]
    }

    #[tokio::test]
    async fn sorted_by_score_results_come_back_in_input_order() {
        let (base, seen, _) = mock(|_| results(&[(2, 0.9), (0, 0.5), (1, 0.1)])).await;
        let got = client(&base, Some("rk-1"))
            .rerank("q", &docs(3))
            .await
            .unwrap();
        assert_eq!(got, vec![0.5, 0.1, 0.9]);
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].0.as_deref(), Some("Bearer rk-1"));
        assert_eq!(
            seen[0].1,
            json!({"model": "BAAI/bge-reranker-base", "query": "q",
                   "documents": ["doc 0", "doc 1", "doc 2"], "top_n": 3})
        );
    }

    #[tokio::test]
    async fn no_key_sends_no_authorization_header() {
        let (base, seen, _) = mock(|_| results(&[(0, 0.5)])).await;
        client(&base, None).rerank("q", &docs(1)).await.unwrap();
        assert_eq!(seen.lock().unwrap()[0].0, None);
    }

    #[tokio::test]
    async fn a_missing_index_is_an_invalid_response() {
        let (base, _, _) = mock(|_| results(&[(0, 0.5)])).await;
        let err = client(&base, None).rerank("q", &docs(2)).await.unwrap_err();
        assert_eq!(error_class(&err), "invalid_response", "{err}");
        assert!(
            err.to_string().contains("no result for document 1"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn out_of_range_and_duplicate_indices_are_invalid_responses() {
        for bad in [vec![(0, 0.5), (5, 0.2)], vec![(0, 0.5), (0, 0.2)]] {
            let (base, _, _) = mock(move |_| results(&bad)).await;
            let err = client(&base, None).rerank("q", &docs(2)).await.unwrap_err();
            assert_eq!(error_class(&err), "invalid_response", "{err}");
        }
    }

    #[tokio::test]
    async fn status_403_is_auth_and_not_retried() {
        let (base, _, calls) = mock(|_| StatusCode::FORBIDDEN.into_response()).await;
        let err = client(&base, Some("k"))
            .rerank("q", &docs(1))
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                StrataError::Upstream {
                    service: "rerank",
                    ..
                }
            ),
            "{err:?}"
        );
        assert_eq!(error_class(&err), "auth");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn status_429_is_retried_then_succeeds() {
        let (base, _, calls) = mock(|n| {
            if n == 0 {
                StatusCode::TOO_MANY_REQUESTS.into_response()
            } else {
                results(&[(0, 0.7)])
            }
        })
        .await;
        let got = client(&base, None).rerank("q", &docs(1)).await.unwrap();
        assert_eq!(got, vec![0.7]);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn status_422_is_invalid_request_and_not_retried() {
        let (base, _, calls) = mock(|_| StatusCode::UNPROCESSABLE_ENTITY.into_response()).await;
        let err = client(&base, None).rerank("q", &docs(1)).await.unwrap_err();
        assert_eq!(error_class(&err), "invalid_request");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_timeout_is_retryable_but_not_retried() {
        let calls = Arc::new(AtomicUsize::new(0));
        let c2 = calls.clone();
        let app = Router::new().route(
            "/rerank",
            post(move || {
                let c = c2.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    results(&[(0, 0.5)])
                }
            }),
        );
        let base = spawn(app).await;
        let c = RerankClient::new(&base, None, "m", 100).unwrap();
        let err = c.rerank("q", &docs(1)).await.unwrap_err();
        assert_eq!(error_class(&err), "retryable");
        assert!(err.to_string().contains("timeout"), "{err}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn the_key_never_appears_in_debug_output() {
        let c = RerankClient::new(
            "http://127.0.0.1:1",
            Some(Secret::new("rk-SECRET-7")),
            "m",
            1000,
        )
        .unwrap();
        let dbg = format!("{c:?}");
        assert!(!dbg.contains("rk-SECRET-7"), "{dbg}");
        assert!(dbg.contains("Bearer ***"), "{dbg}");
    }
}
