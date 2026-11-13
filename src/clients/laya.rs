//! laya System-1 classifier client: one Jev `choice` question per ambiguous router.
//!
//! `POST {base}/v1/systemone`, `Authorization: Bearer <licence token>`. Timeout from config (800 ms on the
//! query path), no retry: any failure is returned as `Upstream { service: "laya", .. }` and `route()` falls
//! back to embed routing. The token is held only in a sensitive `HeaderValue` and is never logged.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::config::Secret;
use crate::error::{Result, StrataError};
use crate::routing::{BoxFut, ChoiceOracle};

pub const ROUTE_INSTRUCTIONS: &str =
    "Which knowledge store most likely contains the answer to this question?";

pub struct LayaClient {
    http: reqwest::Client,
    base_url: String,
    auth: HeaderValue,
    model: String,
}

impl fmt::Debug for LayaClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayaClient")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
struct SystemOneResponse {
    answers: HashMap<String, ChoiceAnswer>,
}

#[derive(Deserialize)]
struct ChoiceAnswer {
    probabilities: HashMap<String, f32>,
}

fn upstream(message: impl Into<String>) -> StrataError {
    StrataError::Upstream {
        service: "laya",
        message: message.into(),
    }
}

/// `without_url`: reqwest's Display embeds the request URL; keep it out of messages and logs.
fn transport(e: reqwest::Error) -> StrataError {
    if e.is_timeout() {
        upstream("timeout")
    } else {
        upstream(format!("request: {}", e.without_url()))
    }
}

impl LayaClient {
    pub fn new(base_url: &str, token: Secret, model: &str, timeout_ms: u64) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .map_err(|e| StrataError::Config(format!("laya http client: {}", e.without_url())))?;
        let mut auth =
            HeaderValue::from_str(&format!("Bearer {}", token.expose())).map_err(|_| {
                StrataError::Config(
                    "LAYA_LICENSE_TOKEN contains invalid header characters".to_string(),
                )
            })?;
        auth.set_sensitive(true);
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            auth,
            model: model.to_string(),
        })
    }

    pub async fn health(&self) -> bool {
        // the api.nonagentic.ai licence gate may cover /laya/health too: always send the bearer
        match self
            .http
            .get(format!("{}/health", self.base_url))
            .header(AUTHORIZATION, self.auth.clone())
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn request_body(&self, query: &str, options: &[(String, String)]) -> Value {
        let criteria: Map<String, Value> = options
            .iter()
            .map(|(name, description)| (name.clone(), Value::String(description.clone())))
            .collect();
        json!({
            "model": self.model,
            "state": query,
            "questions": {"route": {
                "type": "choice",
                "instructions": ROUTE_INSTRUCTIONS,
                "criteria": criteria
            }}
        })
    }

    async fn systemone(
        &self,
        query: &str,
        options: &[(String, String)],
    ) -> Result<HashMap<String, f32>> {
        let body = serde_json::to_vec(&self.request_body(query, options))
            .map_err(|e| upstream(format!("encode: {e}")))?;
        let resp = self
            .http
            .post(format!("{}/v1/systemone", self.base_url))
            .header(AUTHORIZATION, self.auth.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
            .map_err(transport)?;
        let status = resp.status();
        if !status.is_success() {
            return Err(upstream(format!("status {}", status.as_u16())));
        }
        let bytes = resp.bytes().await.map_err(transport)?;
        let mut parsed: SystemOneResponse =
            serde_json::from_slice(&bytes).map_err(|e| upstream(format!("decode: {e}")))?;
        let probabilities = parsed
            .answers
            .remove("route")
            .map(|a| a.probabilities)
            .ok_or_else(|| upstream("decode: answers.route missing"))?;
        // route() ranks with total_cmp: a NaN/inf/negative value would win or distort argmax, so reject.
        if let Some((name, p)) = probabilities
            .iter()
            .find(|(_, p)| !p.is_finite() || **p < 0.0)
        {
            return Err(upstream(format!(
                "decode: invalid probability for {name}: {p}"
            )));
        }
        Ok(probabilities)
    }
}

impl ChoiceOracle for LayaClient {
    fn choose<'a>(
        &'a self,
        query: &'a str,
        options: &'a [(String, String)],
    ) -> BoxFut<'a, Result<HashMap<String, f32>>> {
        Box::pin(self.systemone(query, options))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::testutil::spawn;
    use crate::error::StrataError;
    use crate::routing::ChoiceOracle;
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode, header};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    type Call = (Option<String>, Value);

    #[derive(Clone, Default)]
    struct Seen(Arc<Mutex<Vec<Call>>>);

    async fn ok_handler(
        State(seen): State<Seen>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        let auth = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        seen.0.lock().unwrap().push((auth, body));
        Json(json!({
            "model": "laya-english-0.4.2",
            "answers": {"route": {"type": "choice", "choice": "physics",
                                  "probabilities": {"physics": 0.8, "music": 0.2}, "confidence": 0.6}},
            "usage": {"input_tokens": 12, "output_tokens": 1}
        }))
    }

    fn options() -> Vec<(String, String)> {
        vec![
            ("physics".to_string(), "forces and particles".to_string()),
            ("music".to_string(), "songs and scores".to_string()),
        ]
    }

    fn upstream_message(e: StrataError) -> String {
        match e {
            StrataError::Upstream { service, message } => {
                assert_eq!(service, "laya");
                message
            }
            other => panic!("expected Upstream, got {other:?}"),
        }
    }

    async fn choose_against(app: Router, timeout_ms: u64) -> Result<HashMap<String, f32>> {
        let base = spawn(app).await;
        let client =
            LayaClient::new(&base, Secret::new("tok-123"), "jev-latest", timeout_ms).unwrap();
        let opts = options();
        client.choose("what is a photon", &opts).await
    }

    #[tokio::test]
    async fn choose_sends_a_jev_choice_question_with_bearer_and_parses_probabilities() {
        let seen = Seen::default();
        let app = Router::new()
            .route("/v1/systemone", post(ok_handler))
            .with_state(seen.clone());
        let base = spawn(app).await;
        // trailing slash on the base must not produce `//v1/systemone`
        let client = LayaClient::new(
            &format!("{base}/"),
            Secret::new("tok-123"),
            "jev-latest",
            800,
        )
        .unwrap();
        let opts = options();
        let probs = client.choose("what is a photon", &opts).await.unwrap();
        assert_eq!(probs.len(), 2);
        assert!((probs["physics"] - 0.8).abs() < 1e-6);
        assert!((probs["music"] - 0.2).abs() < 1e-6);

        let calls = seen.0.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0.as_deref(), Some("Bearer tok-123"));
        assert_eq!(
            calls[0].1,
            json!({
                "model": "jev-latest",
                "state": "what is a photon",
                "questions": {"route": {
                    "type": "choice",
                    "instructions": "Which knowledge store most likely contains the answer to this question?",
                    "criteria": {"physics": "forces and particles", "music": "songs and scores"}
                }}
            })
        );
    }

    #[tokio::test]
    async fn status_401_is_upstream_status_401() {
        let app = Router::new().route("/v1/systemone", post(|| async { StatusCode::UNAUTHORIZED }));
        let err = choose_against(app, 800).await.unwrap_err();
        assert_eq!(upstream_message(err), "status 401");
    }

    #[tokio::test]
    async fn status_503_is_upstream_status_503() {
        let app = Router::new().route(
            "/v1/systemone",
            post(|| async { StatusCode::SERVICE_UNAVAILABLE }),
        );
        let err = choose_against(app, 800).await.unwrap_err();
        assert_eq!(upstream_message(err), "status 503");
    }

    #[tokio::test]
    async fn malformed_json_is_a_decode_error() {
        let app = Router::new().route(
            "/v1/systemone",
            post(|| async { ([(header::CONTENT_TYPE, "application/json")], "{not json") }),
        );
        let err = choose_against(app, 800).await.unwrap_err();
        assert!(upstream_message(err).starts_with("decode: "));
    }

    #[tokio::test]
    async fn answer_without_route_is_a_decode_error() {
        let app = Router::new().route(
            "/v1/systemone",
            post(|| async { Json(json!({"answers": {}})) }),
        );
        let err = choose_against(app, 800).await.unwrap_err();
        assert_eq!(upstream_message(err), "decode: answers.route missing");
    }

    #[tokio::test]
    async fn invalid_probabilities_are_decode_errors() {
        // negative; null; string; 1e39 is finite as f64 but overflows f32 to inf
        for bad in ["-0.1", "null", "\"0.5\"", "1e39"] {
            let body = format!(
                r#"{{"answers":{{"route":{{"type":"choice","choice":"physics","probabilities":{{"physics":{bad},"music":0.2}}}}}}}}"#
            );
            let app = Router::new().route(
                "/v1/systemone",
                post(move || {
                    let body = body.clone();
                    async move { ([(header::CONTENT_TYPE, "application/json")], body) }
                }),
            );
            let err = choose_against(app, 800).await.unwrap_err();
            assert!(upstream_message(err).starts_with("decode: "), "{bad}");
        }
    }

    #[tokio::test]
    async fn slow_laya_times_out_within_the_budget() {
        let app = Router::new().route(
            "/v1/systemone",
            post(|| async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                Json(json!({}))
            }),
        );
        let t = Instant::now();
        let err = choose_against(app, 200).await.unwrap_err();
        assert!(
            t.elapsed() < Duration::from_secs(1),
            "took {:?}",
            t.elapsed()
        );
        assert_eq!(upstream_message(err), "timeout");
    }

    #[tokio::test]
    async fn health_is_true_on_2xx_and_false_otherwise() {
        let up = spawn(Router::new().route("/health", get(|| async { "ok" }))).await;
        let client = LayaClient::new(&up, Secret::new("t"), "jev-latest", 800).unwrap();
        assert!(client.health().await);

        let degraded = spawn(
            Router::new().route("/health", get(|| async { StatusCode::SERVICE_UNAVAILABLE })),
        )
        .await;
        let client = LayaClient::new(&degraded, Secret::new("t"), "jev-latest", 800).unwrap();
        assert!(!client.health().await);

        // a port nobody listens on
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let client = LayaClient::new(&dead, Secret::new("t"), "jev-latest", 800).unwrap();
        assert!(!client.health().await);
    }

    #[tokio::test]
    async fn transport_error_neither_leaks_token_nor_url() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let client = LayaClient::new(&dead, Secret::new("ltok-secret"), "jev-latest", 800).unwrap();
        let msg = upstream_message(client.choose("q", &options()).await.unwrap_err());
        assert!(msg.starts_with("request: "), "{msg}");
        assert!(
            !msg.contains("ltok-secret") && !msg.contains("127.0.0.1") && !msg.contains("http"),
            "{msg}"
        );
    }

    #[test]
    fn debug_never_prints_the_token() {
        let client = LayaClient::new(
            "http://127.0.0.1:1",
            Secret::new("tok-123"),
            "jev-latest",
            800,
        )
        .unwrap();
        let shown = format!("{client:?}");
        assert!(!shown.contains("tok-123"), "{shown}");
        assert!(shown.contains("jev-latest"));
    }

    /// Live check against the real gateway. Run: `set -a; . ~/.config/strata-rag/licenses.env; set +a;
    /// cargo test --lib clients::laya::tests::live -- --ignored`. Never prints the token.
    #[tokio::test]
    #[ignore = "needs LAYA_LICENSE_TOKEN and network"]
    async fn live_choose_against_api_nonagentic_ai() {
        let token = std::env::var("LAYA_LICENSE_TOKEN").expect("LAYA_LICENSE_TOKEN");
        let client = LayaClient::new(
            "https://api.nonagentic.ai/laya",
            Secret::new(token),
            "jev-latest",
            5000,
        )
        .unwrap();
        let probs = client.choose("what is a photon", &options()).await.unwrap();
        assert!(
            probs.contains_key("physics") && probs.contains_key("music"),
            "{probs:?}"
        );
        println!("live probs: {probs:?}");
    }
}
