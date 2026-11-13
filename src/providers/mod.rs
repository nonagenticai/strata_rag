//! Where embed / rerank providers come from (spec §4.1):
//!
//! 1. **Store and tenant registrations** (`store_providers`, `PUT /api/stores/{id}/providers`;
//!    `tenant_providers`, `PUT /api/tenant/providers`): a store may bind its own embed / rerank /
//!    chat provider (inherited by its subtree), otherwise its tenant's applies; keys are encrypted at
//!    rest ([`crypto`]) and never returned.
//! 2. **Deployment config** (`providers.toml`, [`deployment`]): the deployment region, the policy
//!    (allowed dialects; EU requires zero data retention) and, in `mode = "dev"` only, default
//!    providers. A production deployment has none: an unregistered tenant gets
//!    `409 provider_not_configured`.
//!
//! [`validate`] holds the rules both layers share; [`registry::Providers`] resolves a request's
//! tenant to an [`crate::inference::InferenceHandle`].

pub mod crypto;
pub mod deployment;
pub mod registry;
pub mod ssrf;

use std::fmt;

use crate::config::{Secret, redact_url};
use crate::vecmath::DIM;

pub use crypto::KeyCipher;
pub use deployment::{Deployment, Mode, Policy, Region};
pub use registry::{
    CapabilityView, DefaultView, LaneLimits, PreparedStoreProviders, ProviderInput,
    ProviderTestResult, ProviderView, Providers, ProvidersView, PutProviders, PutStoreProviders,
    STORE_CAPABILITIES, StoreCapabilityView, StoreProvidersView, TestProvider,
};

/// The provider capabilities strata-rag resolves per tenant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    Embed,
    Rerank,
    Chat,
}

impl Capability {
    pub const ALL: [Capability; 3] = [Capability::Embed, Capability::Rerank, Capability::Chat];

    pub fn as_str(self) -> &'static str {
        match self {
            Capability::Embed => "embed",
            Capability::Rerank => "rerank",
            Capability::Chat => "chat",
        }
    }

    /// The one wire dialect implemented for this capability (`crate::clients::{embed, rerank,
    /// chat}`).
    pub fn implemented_dialect(self) -> &'static str {
        match self {
            Capability::Embed | Capability::Chat => "openai",
            Capability::Rerank => "cohere",
        }
    }

    pub fn default_timeout_ms(self) -> u64 {
        match self {
            Capability::Embed => 30_000,
            Capability::Rerank => 10_000,
            Capability::Chat => 60_000,
        }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Every dialect strata-rag implements; a policy naming another one is a startup error.
pub const IMPLEMENTED_DIALECTS: [&str; 2] = ["openai", "cohere"];

pub const DEFAULT_CONCURRENCY: usize = 8;
/// Upper bounds for a provider's `timeout_ms` / `concurrency`.
pub const MAX_TIMEOUT_MS: u64 = 600_000;
pub const MAX_CONCURRENCY: usize = 256;

/// One capability's provider, from a registration or a dev default. `Debug` redacts the key and
/// any URL password.
#[derive(Clone)]
pub struct ProviderSpec {
    /// No trailing `/`; the client appends `/embeddings` or `/rerank`.
    pub base_url: String,
    pub dialect: String,
    /// The `model` field sent to the provider.
    pub model: String,
    /// The store identity (`stores.embed_model`); `None` = `"{model}@{dims}"`. Embed only.
    pub model_id: Option<String>,
    /// Embed: must be 1024. Rerank / chat: must be absent.
    pub dims: Option<i32>,
    /// Where the provider processes data (`eu`, `us`); must contain the deployment region.
    pub regions: Vec<String>,
    /// Zero data retention confirmed in the DPA; required in an `eu` deployment.
    pub zdr: bool,
    pub timeout_ms: u64,
    /// Concurrent query-path calls to this provider.
    pub concurrency: usize,
    pub api_key: Option<Secret>,
    /// Chat only: `chat_template_kwargs.enable_thinking` (a vLLM extension); `None` = not sent.
    pub thinking: Option<bool>,
    /// Chat only: `temperature`; `None` = not sent (the server's default).
    pub temperature: Option<f64>,
}

impl fmt::Debug for ProviderSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderSpec")
            .field("base_url", &redact_url(&self.base_url))
            .field("dialect", &self.dialect)
            .field("model", &self.model)
            .field("model_id", &self.model_id)
            .field("dims", &self.dims)
            .field("regions", &self.regions)
            .field("zdr", &self.zdr)
            .field("timeout_ms", &self.timeout_ms)
            .field("concurrency", &self.concurrency)
            .field("api_key", &self.api_key)
            .field("thinking", &self.thinking)
            .field("temperature", &self.temperature)
            .finish()
    }
}

impl ProviderSpec {
    /// What a store records as its embedding model: `model_id`, else `"{model}@{dims}"`.
    pub fn embed_model_id(&self) -> String {
        match &self.model_id {
            Some(id) => id.clone(),
            None => format!("{}@{}", self.model, self.dims.unwrap_or(DIM as i32)),
        }
    }
}

/// Why a provider was refused: a stable code (the 422 body's `error.code`) plus a message that
/// never carries a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub code: &'static str,
    pub message: String,
}

fn reject(code: &'static str, message: String) -> Rejection {
    Rejection { code, message }
}

/// The rules every provider passes, registered or default (spec §4.1, §7): the capability's
/// implemented dialect, allowed by the policy; `regions` contains the deployment region; `zdr` in
/// an `eu` deployment; embed `dims = 1024` (rerank: none); a plain http(s) URL without
/// credentials; a model without `@` (stores record `"model@dims"`); sane timeout / concurrency.
/// The live probe is separate (registration only).
pub fn validate(
    cap: Capability,
    spec: &ProviderSpec,
    region: Region,
    policy: &Policy,
) -> Result<(), Rejection> {
    let dialect = spec.dialect.trim();
    if dialect != cap.implemented_dialect() || !policy.allowed_dialects.iter().any(|d| d == dialect)
    {
        return Err(reject(
            "dialect_not_allowed",
            format!(
                "{cap}: dialect {dialect:?} is not allowed (this deployment allows {:?} for {cap})",
                policy
                    .allowed_dialects
                    .iter()
                    .filter(|d| d.as_str() == cap.implemented_dialect())
                    .collect::<Vec<_>>()
            ),
        ));
    }
    let invalid = |m: String| reject("invalid_provider", format!("{cap}: {m}"));
    // parsed exactly as reqwest will (url::Url); see `ssrf::parse_base_url`
    if let Err(e) = ssrf::parse_base_url(&spec.base_url) {
        return Err(reject(e.code(), format!("{cap}: base_url {}", e.message())));
    }
    let model = spec.model.trim();
    if model.is_empty() || model.contains('@') {
        return Err(invalid(
            "model must be non-empty and must not contain '@' (stores record \"model@dims\")"
                .to_string(),
        ));
    }
    if let Some(id) = &spec.model_id {
        if cap != Capability::Embed {
            return Err(invalid("model_id is only valid for embed".to_string()));
        }
        if id.trim().is_empty() {
            return Err(invalid("model_id must not be empty".to_string()));
        }
    }
    if spec.regions.is_empty() {
        return Err(reject(
            "region_not_allowed",
            format!("{cap}: regions is empty; it must contain the deployment region \"{region}\""),
        ));
    }
    for r in &spec.regions {
        if Region::parse(r).is_none() {
            return Err(invalid(format!("unknown region {r:?} (eu or us)")));
        }
    }
    if !spec.regions.iter().any(|r| r == region.as_str()) {
        return Err(reject(
            "region_not_allowed",
            format!(
                "{cap}: regions {:?} do not include this deployment's region \"{region}\"",
                spec.regions
            ),
        ));
    }
    if region == Region::Eu && !spec.zdr {
        return Err(reject(
            "zdr_required",
            format!("{cap}: an eu deployment requires zdr = true (zero data retention in the DPA)"),
        ));
    }
    match cap {
        Capability::Embed if spec.dims != Some(DIM as i32) => {
            return Err(reject(
                "dims_must_be_1024",
                format!(
                    "embed: dims must be {DIM} (the schema's vector size), got {}",
                    spec.dims
                        .map_or_else(|| "none".to_string(), |d| d.to_string())
                ),
            ));
        }
        Capability::Rerank | Capability::Chat if spec.dims.is_some() => {
            return Err(invalid("dims is only valid for embed".to_string()));
        }
        _ => {}
    }
    if cap != Capability::Chat && (spec.thinking.is_some() || spec.temperature.is_some()) {
        return Err(invalid(
            "thinking and temperature are only valid for chat".to_string(),
        ));
    }
    if spec.temperature.is_some_and(|t| !(0.0..=2.0).contains(&t)) {
        return Err(invalid("temperature must be within [0, 2]".to_string()));
    }
    if spec.timeout_ms == 0 || spec.timeout_ms > MAX_TIMEOUT_MS {
        return Err(invalid(format!(
            "timeout_ms must be within 1..={MAX_TIMEOUT_MS}"
        )));
    }
    if spec.concurrency == 0 || spec.concurrency > MAX_CONCURRENCY {
        return Err(invalid(format!(
            "concurrency must be within 1..={MAX_CONCURRENCY}"
        )));
    }
    if spec
        .api_key
        .as_ref()
        .is_some_and(|k| k.expose().trim().is_empty())
    {
        return Err(invalid("api_key must not be empty".to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn embed_spec() -> ProviderSpec {
        ProviderSpec {
            base_url: "https://embed.example.test/v1".into(),
            dialect: "openai".into(),
            model: "acme/embed".into(),
            model_id: None,
            dims: Some(1024),
            regions: vec!["eu".into()],
            zdr: true,
            timeout_ms: 5000,
            concurrency: 4,
            api_key: Some(Secret::new("sk-unit-secret")),
            thinking: None,
            temperature: None,
        }
    }

    fn rerank_spec() -> ProviderSpec {
        ProviderSpec {
            dialect: "cohere".into(),
            dims: None,
            ..embed_spec()
        }
    }

    fn both() -> Policy {
        Policy {
            allowed_dialects: vec!["openai".into(), "cohere".into()],
            allow_internal_provider_hosts: false,
        }
    }

    fn code(cap: Capability, spec: &ProviderSpec, region: Region, policy: &Policy) -> &'static str {
        match validate(cap, spec, region, policy) {
            Ok(()) => "ok",
            Err(r) => r.code,
        }
    }

    #[test]
    fn a_valid_spec_passes() {
        assert_eq!(
            code(Capability::Embed, &embed_spec(), Region::Eu, &both()),
            "ok"
        );
        assert_eq!(
            code(Capability::Rerank, &rerank_spec(), Region::Eu, &both()),
            "ok"
        );
    }

    #[test]
    fn each_rule_has_its_code() {
        let e = Capability::Embed;
        let eu = Region::Eu;
        let cases: Vec<(ProviderSpec, &str)> = vec![
            (
                ProviderSpec {
                    dialect: "voyage".into(),
                    ..embed_spec()
                },
                "dialect_not_allowed",
            ),
            (
                ProviderSpec {
                    dialect: "cohere".into(),
                    ..embed_spec()
                },
                "dialect_not_allowed",
            ),
            (
                ProviderSpec {
                    regions: vec!["us".into()],
                    zdr: true,
                    ..embed_spec()
                },
                "region_not_allowed",
            ),
            (
                ProviderSpec {
                    regions: vec![],
                    ..embed_spec()
                },
                "region_not_allowed",
            ),
            (
                ProviderSpec {
                    zdr: false,
                    ..embed_spec()
                },
                "zdr_required",
            ),
            (
                ProviderSpec {
                    dims: Some(768),
                    ..embed_spec()
                },
                "dims_must_be_1024",
            ),
            (
                ProviderSpec {
                    dims: None,
                    ..embed_spec()
                },
                "dims_must_be_1024",
            ),
            (
                ProviderSpec {
                    base_url: "ftp://x".into(),
                    ..embed_spec()
                },
                "invalid_provider",
            ),
            (
                ProviderSpec {
                    base_url: "https://u:pw@host/v1".into(),
                    ..embed_spec()
                },
                "provider_host_not_allowed",
            ),
            (
                ProviderSpec {
                    model: "a@b".into(),
                    ..embed_spec()
                },
                "invalid_provider",
            ),
            (
                ProviderSpec {
                    regions: vec!["eu".into(), "asia".into()],
                    ..embed_spec()
                },
                "invalid_provider",
            ),
            (
                ProviderSpec {
                    timeout_ms: 0,
                    ..embed_spec()
                },
                "invalid_provider",
            ),
            (
                ProviderSpec {
                    concurrency: 0,
                    ..embed_spec()
                },
                "invalid_provider",
            ),
            (
                ProviderSpec {
                    api_key: Some(Secret::new("  ")),
                    ..embed_spec()
                },
                "invalid_provider",
            ),
        ];
        for (spec, want) in cases {
            assert_eq!(code(e, &spec, eu, &both()), want, "{spec:?}");
        }
        // zdr is only required in eu
        let us = ProviderSpec {
            regions: vec!["us".into()],
            zdr: false,
            ..embed_spec()
        };
        assert_eq!(code(e, &us, Region::Us, &both()), "ok");
        // rerank: dims is refused, and the policy can disallow an implemented dialect
        let r = ProviderSpec {
            dims: Some(1024),
            ..rerank_spec()
        };
        assert_eq!(
            code(Capability::Rerank, &r, eu, &both()),
            "invalid_provider"
        );
        let only_openai = Policy {
            allowed_dialects: vec!["openai".into()],
            allow_internal_provider_hosts: false,
        };
        assert_eq!(
            code(Capability::Rerank, &rerank_spec(), eu, &only_openai),
            "dialect_not_allowed"
        );
    }

    #[test]
    fn chat_takes_openai_no_dims_and_may_think() {
        let chat = ProviderSpec {
            dims: None,
            thinking: Some(true),
            temperature: Some(0.2),
            ..embed_spec()
        };
        assert_eq!(code(Capability::Chat, &chat, Region::Eu, &both()), "ok");
        let with_dims = ProviderSpec {
            dims: Some(1024),
            ..chat.clone()
        };
        assert_eq!(
            code(Capability::Chat, &with_dims, Region::Eu, &both()),
            "invalid_provider"
        );
        let cohere = ProviderSpec {
            dialect: "cohere".into(),
            ..chat.clone()
        };
        assert_eq!(
            code(Capability::Chat, &cohere, Region::Eu, &both()),
            "dialect_not_allowed"
        );
        for embed in [
            ProviderSpec {
                thinking: Some(false),
                ..embed_spec()
            },
            ProviderSpec {
                temperature: Some(0.2),
                ..embed_spec()
            },
        ] {
            assert_eq!(
                code(Capability::Embed, &embed, Region::Eu, &both()),
                "invalid_provider",
                "{embed:?}"
            );
        }
        for t in [-0.1, 2.5, f64::NAN] {
            let hot = ProviderSpec {
                temperature: Some(t),
                ..chat.clone()
            };
            assert_eq!(
                code(Capability::Chat, &hot, Region::Eu, &both()),
                "invalid_provider",
                "{t}"
            );
        }
    }

    #[test]
    fn model_identity_defaults_to_model_at_dims() {
        assert_eq!(embed_spec().embed_model_id(), "acme/embed@1024");
        let s = ProviderSpec {
            model_id: Some("BAAI/bge-m3@1024".into()),
            ..embed_spec()
        };
        assert_eq!(s.embed_model_id(), "BAAI/bge-m3@1024");
    }

    #[test]
    fn debug_redacts_the_key_and_url_password() {
        let s = ProviderSpec {
            base_url: "https://u:url-pw@h/v1".into(),
            ..embed_spec()
        };
        let dbg = format!("{s:?}");
        assert!(!dbg.contains("sk-unit-secret"), "{dbg}");
        assert!(!dbg.contains("url-pw"), "{dbg}");
        assert!(dbg.contains("Secret(***)"), "{dbg}");
    }
}
