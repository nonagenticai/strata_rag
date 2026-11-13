//! Chat provider client, OpenAI dialect: `POST {base_url}/chat/completions` (`base_url` includes
//! `/v1`, as for embed) with `{model, messages, max_tokens}` plus, only when the provider sets
//! them, `temperature` and vLLM's `chat_template_kwargs.enable_thinking` (so the body stays
//! portable to any OpenAI-dialect server). Answered by `{choices: [{message: {content}}]}`, where
//! `content` is a string or an array of `{type: "text", text}` parts (joined). Auth, timeout,
//! error classes, the SSRF guard and the response cap: see `super::provider`. Chat is NOT retried:
//! a generation is slow and costly, and a retry after a timeout or 5xx can double it.
//!
//! Thinking models (our vLLM `local-llm`) can put leading whitespace in `message.content` and a
//! separate `message.reasoning`: the content is trimmed, `reasoning` ignored, and empty content
//! (the whole budget spent thinking) is an `invalid_response` ("empty completion"). Our vLLM's
//! `[default.chat]` sets `thinking = false`: with it on the acceptance run spent the whole
//! 800-token budget reasoning and returned nothing.
//!
//! A routed lane (`[default.chat] routed = true`: the inference-router) is given a [`RouterIdentity`]:
//! the tenant the call is billed to ([`EFFECTIVE_TENANT_HEADER`]) and the caller's attribution
//! ([`PROJECT_ID_HEADER`], [`APP_ID_HEADER`], [`ACTOR_ID_HEADER`]: the router's caller-identity
//! contract, required before it arms `enforce`). Any other call sends none of these headers: a
//! tenant's own provider is a third party.

use std::fmt;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use super::provider::{ErrorClass, ProviderHttp};
use crate::config::Secret;
use crate::error::Result;
use crate::inference::Chat;
use crate::routing::BoxFut;

/// Largest accepted chat response.
pub const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

/// The tenant the inference-router bills a routed call to (a hyphenated, non-nil uuid).
pub const EFFECTIVE_TENANT_HEADER: &str = "x-effective-tenant-id";
/// The project the call is billed to (the caller's, else the tenant-level project).
pub const PROJECT_ID_HEADER: &str = "x-project-id";
/// The calling app's licences-registry key, relayed from the caller; never strata's own.
pub const APP_ID_HEADER: &str = "x-app-id";
/// The Keycloak `sub` that acted, relayed from the caller; never defaulted.
pub const ACTOR_ID_HEADER: &str = "x-actor-id";

/// What a routed call tells the inference-router about who it is for. `app` / `actor` are `None`
/// when the caller sent none: the header is then omitted (the router counts the gap) rather than
/// filled in with strata's own identity, which would hide the caller's gap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterIdentity {
    pub tenant: Uuid,
    pub project: Uuid,
    pub app: Option<String>,
    pub actor: Option<Uuid>,
}

impl RouterIdentity {
    fn headers(&self) -> std::result::Result<HeaderMap, &'static str> {
        let mut h = HeaderMap::new();
        let mut put = |name: &'static str, value: &str| {
            HeaderValue::from_str(value)
                .map(|v| {
                    h.insert(HeaderName::from_static(name), v);
                })
                .map_err(|_| name)
        };
        put(
            EFFECTIVE_TENANT_HEADER,
            &self.tenant.hyphenated().to_string(),
        )?;
        put(PROJECT_ID_HEADER, &self.project.hyphenated().to_string())?;
        if let Some(app) = &self.app {
            put(APP_ID_HEADER, app)?;
        }
        if let Some(actor) = self.actor {
            put(ACTOR_ID_HEADER, &actor.hyphenated().to_string())?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.to_string(),
            content: content.into(),
        }
    }
}

/// Per-provider request options. `None` = the field is not sent (the server's default).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ChatOptions {
    /// `chat_template_kwargs.enable_thinking` (a vLLM extension).
    pub thinking: Option<bool>,
    pub temperature: Option<f64>,
}

pub struct ChatClient {
    http: ProviderHttp,
    model: String,
    options: ChatOptions,
}

impl fmt::Debug for ChatClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatClient")
            .field("http", &self.http)
            .field("model", &self.model)
            .field("options", &self.options)
            .finish()
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    #[serde(default)]
    content: Option<Content>,
}

/// `message.content`: a string, or (newer OpenAI-dialect servers) an array of typed parts.
#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

impl Content {
    /// The text: the string, or the `text` parts concatenated in order (other part types skipped).
    fn into_text(self) -> String {
        match self {
            Content::Text(s) => s,
            Content::Parts(parts) => parts
                .into_iter()
                .filter(|p| p.kind == "text")
                .filter_map(|p| p.text)
                .collect(),
        }
    }
}

impl ChatClient {
    pub fn new(
        base_url: &str,
        api_key: Option<Secret>,
        model: &str,
        timeout_ms: u64,
        options: ChatOptions,
    ) -> Result<Self> {
        Self::with_guard(base_url, api_key, model, timeout_ms, options, false)
    }

    /// `guarded`: see `ProviderHttp::build` (tenant-registered providers).
    pub fn with_guard(
        base_url: &str,
        api_key: Option<Secret>,
        model: &str,
        timeout_ms: u64,
        options: ChatOptions,
        guarded: bool,
    ) -> Result<Self> {
        Ok(Self {
            http: ProviderHttp::build(
                "chat",
                base_url,
                api_key,
                "the chat api_key",
                timeout_ms,
                guarded,
            )?
            .without_retries(),
            model: model.to_string(),
            options,
        })
    }

    /// The request body: `model`, `messages`, `max_tokens`, and the options that are set.
    fn body(&self, messages: &[ChatMessage], max_tokens: u32, options: ChatOptions) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        });
        if let Some(t) = options.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(thinking) = options.thinking {
            body["chat_template_kwargs"] = json!({ "enable_thinking": thinking });
        }
        body
    }

    /// The trimmed content of the first choice.
    pub async fn complete(&self, messages: &[ChatMessage], max_tokens: u32) -> Result<String> {
        self.complete_for(messages, max_tokens, None).await
    }

    /// [`Self::complete`], with `routed` (when given) sent as the router's identity headers.
    pub async fn complete_for(
        &self,
        messages: &[ChatMessage],
        max_tokens: u32,
        routed: Option<&RouterIdentity>,
    ) -> Result<String> {
        let body = self.body(messages, max_tokens, self.options);
        let headers = match routed {
            Some(id) => id.headers().map_err(|name| {
                self.http
                    .error(ErrorClass::InvalidRequest, format!("{name} header"))
            })?,
            None => HeaderMap::new(),
        };
        let resp: ChatResponse = self
            .http
            .post_json_with("/chat/completions", &body, MAX_RESPONSE_BYTES, &headers)
            .await?;
        resp.choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .map(|c| c.into_text().trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                self.http
                    .error(ErrorClass::InvalidResponse, "empty completion")
            })
    }

    /// Registration probe: one tiny completion (`max_tokens = 1`, never thinking: a provider that
    /// knows the switch gets it off, one that does not never sees it) that must decode as a chat
    /// response; its content may be empty.
    pub async fn probe(&self) -> Result<()> {
        let options = ChatOptions {
            thinking: self.options.thinking.map(|_| false),
            temperature: self.options.temperature,
        };
        let body = self.body(&[ChatMessage::new("user", "Reply with OK.")], 1, options);
        let resp: ChatResponse = self
            .http
            .post_json("/chat/completions", &body, MAX_RESPONSE_BYTES)
            .await?;
        if resp.choices.is_empty() {
            return Err(self.http.error(ErrorClass::InvalidResponse, "no choices"));
        }
        Ok(())
    }

    /// `/health`: `GET {base_url}/models`, no generation (a completion per health period would
    /// spend GPU time on probes).
    pub async fn health(&self) -> Result<()> {
        self.http.get_ok("/models").await
    }
}

impl Chat for ChatClient {
    fn complete<'a>(
        &'a self,
        messages: &'a [ChatMessage],
        max_tokens: u32,
        routed: Option<&'a RouterIdentity>,
    ) -> BoxFut<'a, Result<String>> {
        Box::pin(ChatClient::complete_for(self, messages, max_tokens, routed))
    }

    fn health(&self) -> BoxFut<'_, Result<()>> {
        Box::pin(ChatClient::health(self))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::Router;
    use axum::http::{HeaderMap, StatusCode, header};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use serde_json::{Value, json};

    use super::*;
    use crate::clients::provider::error_class;
    use crate::clients::testutil::spawn;

    type Seen = Arc<Mutex<Vec<(Option<String>, Value)>>>;

    async fn mock(reply: impl Fn() -> Response + Send + Sync + 'static) -> (String, Seen) {
        let seen: Seen = Arc::default();
        let s = seen.clone();
        let reply = Arc::new(reply);
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move |h: HeaderMap, axum::Json(b): axum::Json<Value>| {
                    let (s, reply) = (s.clone(), reply.clone());
                    async move {
                        let auth = h
                            .get(header::AUTHORIZATION)
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_owned);
                        s.lock().unwrap().push((auth, b));
                        reply()
                    }
                }),
            )
            .route("/v1/models", get(|| async { "{}" }));
        (format!("{}/v1", spawn(app).await), seen)
    }

    fn content(c: Value) -> Response {
        axum::Json(
            json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": c}}]}),
        )
        .into_response()
    }

    /// Today's vLLM settings: thinking off, temperature 0.2.
    const VLLM: ChatOptions = ChatOptions {
        thinking: Some(false),
        temperature: Some(0.2),
    };

    fn client_with(base: &str, options: ChatOptions) -> ChatClient {
        ChatClient::new(
            base,
            Some(Secret::new("sk-chat-1")),
            "local-llm",
            2000,
            options,
        )
        .unwrap()
    }

    fn client(base: &str) -> ChatClient {
        client_with(base, VLLM)
    }

    fn msgs() -> Vec<ChatMessage> {
        vec![
            ChatMessage::new("system", "be brief"),
            ChatMessage::new("user", "what is light?"),
        ]
    }

    #[tokio::test]
    async fn posts_the_openai_body_with_bearer_and_returns_trimmed_content() {
        let (base, seen) = mock(|| content(json!("\n\nLight is quantised [1]. "))).await;
        let out = client(&base).complete(&msgs(), 64).await.unwrap();
        assert_eq!(out, "Light is quantised [1].");
        let calls = seen.lock().unwrap().clone();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0.as_deref(), Some("Bearer sk-chat-1"));
        assert_eq!(
            calls[0].1,
            json!({"model": "local-llm", "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "what is light?"}],
                   "max_tokens": 64, "temperature": 0.2,
                   "chat_template_kwargs": {"enable_thinking": false}})
        );
    }

    /// Options that are not set are not sent: the body stays plain OpenAI.
    #[tokio::test]
    async fn only_the_options_that_are_set_are_sent() {
        for (options, temperature, kwargs) in [
            (ChatOptions::default(), Value::Null, Value::Null),
            (
                ChatOptions {
                    thinking: Some(false),
                    temperature: None,
                },
                Value::Null,
                json!({"enable_thinking": false}),
            ),
            (
                ChatOptions {
                    thinking: Some(true),
                    temperature: Some(0.7),
                },
                json!(0.7),
                json!({"enable_thinking": true}),
            ),
            (
                ChatOptions {
                    thinking: None,
                    temperature: Some(0.0),
                },
                json!(0.0),
                Value::Null,
            ),
        ] {
            let (base, seen) = mock(|| content(json!("ok"))).await;
            client_with(&base, options)
                .complete(&msgs(), 8)
                .await
                .unwrap();
            let body = seen.lock().unwrap()[0].1.clone();
            assert_eq!(body["max_tokens"], 8, "{options:?}");
            assert_eq!(body["temperature"], temperature, "{options:?}: {body}");
            assert_eq!(body["chat_template_kwargs"], kwargs, "{options:?}: {body}");
            let keys: Vec<&String> = body.as_object().unwrap().keys().collect();
            assert!(
                keys.len()
                    == 3 + usize::from(!temperature.is_null()) + usize::from(!kwargs.is_null()),
                "{options:?}: {body}"
            );
        }
    }

    #[tokio::test]
    async fn content_may_be_a_string_or_text_parts() {
        let parts = json!([
            {"type": "text", "text": "Light is "},
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "quantised [1]."}
        ]);
        for (reply, want) in [
            (json!(" plain [1] "), "plain [1]"),
            (parts, "Light is quantised [1]."),
        ] {
            let r = reply.clone();
            let (base, _) = mock(move || content(r.clone())).await;
            assert_eq!(client(&base).complete(&msgs(), 8).await.unwrap(), want);
        }
        let (base, _) = mock(|| content(json!([{"type": "image_url"}]))).await;
        let err = client(&base).complete(&msgs(), 8).await.unwrap_err();
        assert!(err.to_string().contains("empty completion"), "{err}");
    }

    #[tokio::test]
    async fn empty_or_missing_content_is_invalid_response() {
        for reply in [json!("  \n"), Value::Null] {
            let r = reply.clone();
            let (base, _) = mock(move || content(r.clone())).await;
            let err = client(&base).complete(&msgs(), 8).await.unwrap_err();
            assert_eq!(error_class(&err), "invalid_response", "{reply}: {err}");
            assert!(err.to_string().contains("empty completion"), "{err}");
        }
        let (base, _) = mock(|| axum::Json(json!({"choices": []})).into_response()).await;
        let err = client(&base).complete(&msgs(), 8).await.unwrap_err();
        assert_eq!(error_class(&err), "invalid_response");
    }

    #[tokio::test]
    async fn statuses_map_to_their_classes_without_leaking_key_or_url() {
        for (status, class) in [
            (StatusCode::UNAUTHORIZED, "auth"),
            (StatusCode::BAD_REQUEST, "invalid_request"),
            (StatusCode::BAD_GATEWAY, "retryable"),
        ] {
            let (base, _) = mock(move || status.into_response()).await;
            let err = client(&base).complete(&msgs(), 8).await.unwrap_err();
            assert_eq!(error_class(&err), class, "{status}");
            let m = err.to_string();
            assert!(!m.contains("sk-chat-1") && !m.contains("127.0.0.1"), "{m}");
        }
        let (base, _) = mock(|| "not json".into_response()).await;
        let err = client(&base).complete(&msgs(), 8).await.unwrap_err();
        assert_eq!(error_class(&err), "invalid_response");
    }

    /// A generation is never retried: a 502 is exactly one request.
    #[tokio::test]
    async fn a_failed_chat_call_is_not_retried() {
        let (base, seen) = mock(|| StatusCode::BAD_GATEWAY.into_response()).await;
        let err = client(&base).complete(&msgs(), 8).await.unwrap_err();
        assert_eq!(error_class(&err), "retryable");
        assert_eq!(seen.lock().unwrap().len(), 1, "exactly one request");
    }

    #[tokio::test]
    async fn a_body_over_the_cap_is_refused() {
        let (base, _) = mock(|| content(json!("x".repeat(MAX_RESPONSE_BYTES + 1)))).await;
        let err = client(&base).complete(&msgs(), 8).await.unwrap_err();
        assert_eq!(error_class(&err), "invalid_response");
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[tokio::test]
    async fn probe_and_health() {
        let (base, seen) = mock(|| content(Value::Null)).await;
        let c = client_with(
            &base,
            ChatOptions {
                thinking: Some(true),
                temperature: None,
            },
        );
        c.probe()
            .await
            .expect("an empty 1-token completion still proves the shape");
        let body = seen.lock().unwrap()[0].1.clone();
        assert_eq!(body["max_tokens"], 1);
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        c.health().await.expect("/models answers");
        // a provider without the thinking option never sees the vLLM extension, not even in the probe
        let (base, seen) = mock(|| content(json!("OK"))).await;
        client_with(&base, ChatOptions::default())
            .probe()
            .await
            .unwrap();
        assert!(
            seen.lock().unwrap()[0]
                .1
                .get("chat_template_kwargs")
                .is_none()
        );
        let (base, _) = mock(|| StatusCode::UNAUTHORIZED.into_response()).await;
        assert_eq!(
            error_class(&client(&base).probe().await.unwrap_err()),
            "auth"
        );
    }

    #[test]
    fn debug_never_prints_the_key() {
        let c = client("http://127.0.0.1:1/v1");
        assert!(!format!("{c:?}").contains("sk-chat-1"));
    }
}
