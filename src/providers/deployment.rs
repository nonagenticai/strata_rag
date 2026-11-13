//! `providers.toml` (`STRATA_PROVIDERS_FILE`, default `/etc/strata/providers.toml`): the deployment
//! layer of provider config (spec §4.1). Same file shape as strata-ingest's (`region`, a section per
//! provider, keys as `env:NAME` references):
//!
//! ```toml
//! region = "eu"                         # this deployment's region
//! mode = "dev"                          # "dev" | "production"
//!
//! [policy]
//! allowed_dialects = ["openai", "cohere"]
//! allow_internal_provider_hosts = false  # dev only: let tenants register http / internal hosts
//!
//! # dev only: a production deployment with a [default.*] section does not start
//! [default.embed]
//! base_url = "http://127.0.0.1:9131/v1"
//! dialect = "openai"
//! model = "BAAI/bge-m3"
//! dims = 1024
//! api_key = "env:MLX_API_KEY"
//! regions = ["eu"]
//! zdr = true
//!
//! [default.rerank]
//! base_url = "http://127.0.0.1:9131/v2"
//! dialect = "cohere"
//! model = "BAAI/bge-reranker-v2-m3"
//! api_key = "env:MLX_API_KEY"
//! regions = ["eu"]
//! zdr = true
//!
//! [default.chat]                       # optional (ask + query expansion)
//! base_url = "https://api.nonagentic.ai/vllm/v1"
//! dialect = "openai"
//! model = "local-llm"
//! api_key = "env:VLLM_LICENSE_TOKEN"
//! regions = ["eu"]
//! zdr = true
//! timeout_ms = 60000
//! thinking = false                     # vLLM: chat_template_kwargs; omit = not sent
//! temperature = 0.2                    # omit = the server's default
//! routed = false                       # chat only: true = base_url is the inference-router: it
//!                                      # bills (X-Effective-Tenant-Id), strata emits no chat event
//! ```
//!
//! Every failure is a startup error (`StrataError::Config`) naming the file and the field, never a
//! key: an inline key is refused without being echoed.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use serde::Deserialize;

use super::{Capability, DEFAULT_CONCURRENCY, IMPLEMENTED_DIALECTS, ProviderSpec, validate};
use crate::config::Secret;
use crate::error::{Result, StrataError};

pub const DEFAULT_PROVIDERS_FILE: &str = "/etc/strata/providers.toml";

/// Where a deployment (and its tenants' data) lives, and where a provider processes data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Region {
    Eu,
    Us,
}

impl Region {
    pub fn as_str(self) -> &'static str {
        match self {
            Region::Eu => "eu",
            Region::Us => "us",
        }
    }

    pub fn parse(s: &str) -> Option<Region> {
        match s {
            "eu" => Some(Region::Eu),
            "us" => Some(Region::Us),
            _ => None,
        }
    }
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `dev` may carry default providers (our own `mlx_api` / `pcontract fake`); `production` may not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Dev,
    Production,
}

/// What a registration may use.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub allowed_dialects: Vec<String>,
    /// Dev only (a production deployment with `true` does not start): tenant registrations may
    /// use `http` and internal hosts (loopback, private, CGNAT, `.internal`, ...). Default `false`:
    /// the SSRF guard (`super::ssrf`) applies to every tenant registration.
    #[serde(default)]
    pub allow_internal_provider_hosts: bool,
}

/// The validated deployment layer.
#[derive(Debug, Clone)]
pub struct Deployment {
    pub region: Region,
    pub mode: Mode,
    pub policy: Policy,
    /// Dev only; always `None` in production.
    pub default_embed: Option<ProviderSpec>,
    pub default_rerank: Option<ProviderSpec>,
    pub default_chat: Option<ProviderSpec>,
    /// `[default.chat] routed = true`: the default chat provider is the cluster inference-router,
    /// which meters the call itself. strata then sends `X-Effective-Tenant-Id` and emits no
    /// `strata-rag-chat` event. Deployment-only: a tenant registration can never set it.
    pub default_chat_routed: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileRaw {
    region: Region,
    mode: Mode,
    policy: Policy,
    #[serde(default)]
    default: Option<DefaultsRaw>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DefaultsRaw {
    #[serde(default)]
    embed: Option<SpecRaw>,
    #[serde(default)]
    rerank: Option<SpecRaw>,
    /// Optional: without it (and without a tenant registration) `/api/ask` and query expansion
    /// answer 409 `provider_not_configured`; plain queries are unaffected.
    #[serde(default)]
    chat: Option<SpecRaw>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecRaw {
    base_url: String,
    dialect: String,
    model: String,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    dims: Option<i32>,
    /// `env:NAME` only.
    #[serde(default)]
    api_key: Option<String>,
    regions: Vec<String>,
    zdr: bool,
    #[serde(default)]
    timeout_ms: Option<u64>,
    #[serde(default)]
    concurrency: Option<usize>,
    /// Chat only; omitted = not sent.
    #[serde(default)]
    thinking: Option<bool>,
    /// Chat only; omitted = not sent.
    #[serde(default)]
    temperature: Option<f64>,
    /// Chat only (refused on embed / rerank); omitted = `false`. See
    /// [`Deployment::default_chat_routed`].
    #[serde(default)]
    routed: Option<bool>,
}

fn config_err(msg: String) -> StrataError {
    StrataError::Config(msg)
}

impl Deployment {
    /// Reads and validates `path`, resolving `env:` key references against `env`.
    pub fn load(path: &Path, env: &HashMap<String, String>) -> Result<Deployment> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            config_err(format!(
                "cannot read providers file {} (STRATA_PROVIDERS_FILE): {e}",
                path.display()
            ))
        })?;
        Self::parse(&text, env).map_err(|e| match e {
            StrataError::Config(m) => config_err(format!("providers file {}: {m}", path.display())),
            other => other,
        })
    }

    /// [`Self::load`] on a string (tests, tooling).
    pub fn parse(text: &str, env: &HashMap<String, String>) -> Result<Deployment> {
        let raw: FileRaw = toml::from_str(text).map_err(|e| {
            // `message()` only: the Display form quotes the offending source line, which could be
            // an inline key.
            let line = e
                .span()
                .map(|s| format!(" (line {})", text[..s.start].lines().count().max(1)))
                .unwrap_or_default();
            config_err(format!("{}{line}", e.message()))
        })?;
        for d in &raw.policy.allowed_dialects {
            if !IMPLEMENTED_DIALECTS.contains(&d.as_str()) {
                return Err(config_err(format!(
                    "policy.allowed_dialects: dialect {d:?} is not implemented (implemented: {})",
                    IMPLEMENTED_DIALECTS.join(", ")
                )));
            }
        }
        if raw.mode == Mode::Production && raw.policy.allow_internal_provider_hosts {
            return Err(config_err(
                "policy.allow_internal_provider_hosts = true is only allowed with mode = \"dev\" \
                 (it lets tenants point strata at internal hosts)"
                    .to_string(),
            ));
        }
        let (embed, rerank, chat) = match raw.default {
            None => (None, None, None),
            Some(_) if raw.mode == Mode::Production => {
                return Err(config_err(
                    "a production deployment has no default providers: remove the [default] \
                     section (tenants register their own with PUT /api/tenant/providers)"
                        .to_string(),
                ));
            }
            Some(d) => (d.embed, d.rerank, d.chat),
        };
        let mut out = Deployment {
            region: raw.region,
            mode: raw.mode,
            policy: raw.policy,
            default_embed: None,
            default_rerank: None,
            default_chat: None,
            default_chat_routed: chat.as_ref().and_then(|c| c.routed).unwrap_or(false),
        };
        out.default_embed = embed
            .map(|s| out.default_spec(Capability::Embed, s, env))
            .transpose()?;
        out.default_rerank = rerank
            .map(|s| out.default_spec(Capability::Rerank, s, env))
            .transpose()?;
        out.default_chat = chat
            .map(|s| out.default_spec(Capability::Chat, s, env))
            .transpose()?;
        Ok(out)
    }

    fn default_spec(
        &self,
        cap: Capability,
        raw: SpecRaw,
        env: &HashMap<String, String>,
    ) -> Result<ProviderSpec> {
        if cap != Capability::Chat && raw.routed.is_some() {
            return Err(config_err(format!(
                "default.{cap}.routed: only [default.chat] can be routed through the \
                 inference-router"
            )));
        }
        let api_key = raw
            .api_key
            .as_deref()
            .map(|r| resolve_key(cap, r, env))
            .transpose()?;
        let spec = ProviderSpec {
            base_url: raw.base_url.trim().trim_end_matches('/').to_string(),
            dialect: raw.dialect.trim().to_string(),
            model: raw.model.trim().to_string(),
            model_id: raw.model_id.map(|m| m.trim().to_string()),
            dims: raw.dims,
            regions: raw.regions,
            zdr: raw.zdr,
            timeout_ms: raw.timeout_ms.unwrap_or(cap.default_timeout_ms()),
            concurrency: raw.concurrency.unwrap_or(DEFAULT_CONCURRENCY),
            api_key,
            thinking: raw.thinking,
            temperature: raw.temperature,
        };
        validate(cap, &spec, self.region, &self.policy)
            .map_err(|r| config_err(format!("default.{cap}: {} ({})", r.message, r.code)))?;
        Ok(spec)
    }

    /// The dev default for `cap` (`None` in production).
    pub fn default_for(&self, cap: Capability) -> Option<&ProviderSpec> {
        match cap {
            Capability::Embed => self.default_embed.as_ref(),
            Capability::Rerank => self.default_rerank.as_ref(),
            Capability::Chat => self.default_chat.as_ref(),
        }
    }
}

/// `env:NAME` → the trimmed, non-empty value of `NAME`. Anything else is an inline key and is
/// refused without echoing it.
fn resolve_key(cap: Capability, raw: &str, env: &HashMap<String, String>) -> Result<Secret> {
    let Some(name) = raw.trim().strip_prefix("env:") else {
        return Err(config_err(format!(
            "default.{cap}.api_key must reference the key as \"env:NAME\"; inline keys are not \
             allowed"
        )));
    };
    let name = name.trim();
    env.get(name)
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .map(Secret::new)
        .ok_or_else(|| {
            config_err(format!(
                "default.{cap}.api_key: env var {name} is unset or empty"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEV: &str = r#"
region = "eu"
mode = "dev"

[policy]
allowed_dialects = ["openai", "cohere"]

[default.embed]
base_url = "http://mlx.test:9131/v1/"
dialect = "openai"
model = "BAAI/bge-m3"
dims = 1024
api_key = "env:MLX_API_KEY"
regions = ["eu"]
zdr = true

[default.rerank]
base_url = "http://mlx.test:9131/v2"
dialect = "cohere"
model = "BAAI/bge-reranker-v2-m3"
api_key = "env:MLX_API_KEY"
regions = ["eu"]
zdr = true
timeout_ms = 2500
concurrency = 3
"#;

    fn env() -> HashMap<String, String> {
        HashMap::from([("MLX_API_KEY".to_string(), " mlx-secret-1 ".to_string())])
    }

    fn err(text: &str) -> String {
        match Deployment::parse(text, &env()) {
            Ok(d) => panic!("must be rejected: {d:?}"),
            Err(e) => {
                assert!(matches!(e, StrataError::Config(_)), "{e:?}");
                e.to_string()
            }
        }
    }

    #[test]
    fn a_dev_file_with_defaults_parses_and_resolves_env_keys() {
        let d = Deployment::parse(DEV, &env()).unwrap();
        assert_eq!((d.region, d.mode), (Region::Eu, Mode::Dev));
        let e = d.default_embed.as_ref().unwrap();
        assert_eq!(e.base_url, "http://mlx.test:9131/v1");
        assert_eq!(e.api_key.as_ref().unwrap().expose(), "mlx-secret-1");
        assert_eq!(e.embed_model_id(), "BAAI/bge-m3@1024");
        assert_eq!((e.timeout_ms, e.concurrency), (30_000, DEFAULT_CONCURRENCY));
        let r = d.default_rerank.as_ref().unwrap();
        assert_eq!((r.timeout_ms, r.concurrency), (2500, 3));
        let dbg = format!("{d:?}");
        assert!(!dbg.contains("mlx-secret-1"), "{dbg}");
    }

    #[test]
    fn production_without_defaults_parses_and_with_defaults_is_refused() {
        let prod =
            "region = \"us\"\nmode = \"production\"\n[policy]\nallowed_dialects = [\"openai\"]\n";
        let d = Deployment::parse(prod, &HashMap::new()).unwrap();
        assert_eq!((d.region, d.mode), (Region::Us, Mode::Production));
        assert!(d.default_for(Capability::Embed).is_none());
        let with_default = DEV.replace("mode = \"dev\"", "mode = \"production\"");
        assert!(err(&with_default).contains("production"));
        // an empty [default] table counts too
        let empty = format!("{prod}[default]\n");
        assert!(err(&empty).contains("production"));
    }

    #[test]
    fn internal_provider_hosts_are_off_by_default_and_dev_only() {
        let d = Deployment::parse(DEV, &env()).unwrap();
        assert!(!d.policy.allow_internal_provider_hosts);
        let on = DEV.replace(
            "[policy]\n",
            "[policy]\nallow_internal_provider_hosts = true\n",
        );
        assert!(
            Deployment::parse(&on, &env())
                .unwrap()
                .policy
                .allow_internal_provider_hosts
        );
        let prod = "region = \"eu\"\nmode = \"production\"\n[policy]\n\
                    allowed_dialects = [\"openai\"]\nallow_internal_provider_hosts = true\n";
        assert!(err(prod).contains("allow_internal_provider_hosts"));
        let prod_off = prod.replace("= true", "= false");
        assert!(Deployment::parse(&prod_off, &env()).is_ok());
    }

    #[test]
    fn default_chat_is_optional_and_validated() {
        let d = Deployment::parse(DEV, &env()).unwrap();
        assert!(d.default_for(Capability::Chat).is_none(), "optional");
        let chat = format!(
            "{DEV}\n[default.chat]\nbase_url = \"https://llm.test/vllm/v1\"\ndialect = \"openai\"\n\
             model = \"local-llm\"\napi_key = \"env:MLX_API_KEY\"\nregions = [\"eu\"]\nzdr = true\n\
             thinking = true\ntemperature = 0.2\n"
        );
        let d = Deployment::parse(&chat, &env()).unwrap();
        let c = d.default_for(Capability::Chat).unwrap();
        assert_eq!(
            (c.model.as_str(), c.thinking, c.temperature, c.timeout_ms),
            ("local-llm", Some(true), Some(0.2), 60_000)
        );
        let m = err(&chat.replace(
            "model = \"local-llm\"",
            "model = \"local-llm\"\ndims = 1024",
        ));
        assert!(m.contains("default.chat"), "{m}");
        // thinking on a non-chat default is refused
        let m = err(&DEV.replacen("dims = 1024", "dims = 1024\nthinking = true", 1));
        assert!(m.contains("thinking"), "{m}");
    }

    #[test]
    fn routed_is_chat_only_and_off_by_default() {
        let chat = format!(
            "{DEV}\n[default.chat]\nbase_url = \"http://router.test:8150/api\"\n\
             dialect = \"openai\"\nmodel = \"local-llm\"\napi_key = \"env:MLX_API_KEY\"\n\
             regions = [\"eu\"]\nzdr = true\n"
        );
        assert!(
            !Deployment::parse(&chat, &env())
                .unwrap()
                .default_chat_routed
        );
        let routed = format!("{chat}routed = true\n");
        assert!(
            Deployment::parse(&routed, &env())
                .unwrap()
                .default_chat_routed
        );
        let off = format!("{chat}routed = false\n");
        assert!(!Deployment::parse(&off, &env()).unwrap().default_chat_routed);
        // set on embed or rerank (either value) it is refused, never silently ignored
        for (anchor, cap) in [
            ("dims = 1024", "default.embed.routed"),
            ("concurrency = 3", "default.rerank.routed"),
        ] {
            for v in ["true", "false"] {
                let text = DEV.replacen(anchor, &format!("{anchor}\nrouted = {v}"), 1);
                assert_ne!(text, DEV, "{anchor}");
                let m = err(&text);
                assert!(m.contains(cap), "{anchor} {v} -> {m}");
            }
        }
    }

    #[test]
    fn a_literal_key_is_refused_without_echoing_it() {
        let text = DEV.replacen("\"env:MLX_API_KEY\"", "\"sk-literal-123\"", 1);
        let m = err(&text);
        assert!(m.contains("inline keys"), "{m}");
        assert!(!m.contains("sk-literal-123"), "{m}");
    }

    #[test]
    fn an_unset_env_reference_is_refused() {
        let m = match Deployment::parse(DEV, &HashMap::new()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("unset env var must fail"),
        };
        assert!(m.contains("MLX_API_KEY"), "{m}");
    }

    #[test]
    fn defaults_follow_the_registration_rules() {
        for (from, to, want) in [
            ("dims = 1024", "dims = 768", "dims_must_be_1024"),
            (
                "dialect = \"openai\"",
                "dialect = \"cohere\"",
                "dialect_not_allowed",
            ),
            (
                "regions = [\"eu\"]\nzdr = true\n\n[default.rerank]",
                "regions = [\"us\"]\nzdr = true\n\n[default.rerank]",
                "region_not_allowed",
            ),
            (
                "zdr = true\n\n[default.rerank]",
                "zdr = false\n\n[default.rerank]",
                "zdr_required",
            ),
        ] {
            let text = DEV.replacen(from, to, 1);
            assert_ne!(text, DEV, "{from}");
            let m = err(&text);
            assert!(m.contains(want), "{from} -> {m}");
        }
    }

    #[test]
    fn policy_and_shape_errors() {
        let m = err(&DEV.replace("[\"openai\", \"cohere\"]", "[\"openai\", \"voyage\"]"));
        assert!(m.contains("voyage"), "{m}");
        // the policy may leave out an implemented dialect: then the default using it is refused
        let m = err(&DEV.replace("[\"openai\", \"cohere\"]", "[\"openai\"]"));
        assert!(m.contains("dialect_not_allowed"), "{m}");
        assert!(
            err("region = \"asia\"\nmode = \"dev\"\n[policy]\nallowed_dialects = []\n")
                .contains("line")
        );
        assert!(!err(&DEV.replace("mode = \"dev\"\n", "")).is_empty());
        assert!(
            !err(&DEV.replace(
                "[policy]\nallowed_dialects = [\"openai\", \"cohere\"]\n",
                ""
            ))
            .is_empty()
        );
        assert!(!err(&format!("{DEV}\nunknown = 1\n")).is_empty());
    }

    #[test]
    fn load_names_the_file() {
        let m = Deployment::load(Path::new("/nonexistent/strata/providers.toml"), &env())
            .unwrap_err()
            .to_string();
        assert!(m.contains("/nonexistent/strata/providers.toml"), "{m}");
        assert!(m.contains("STRATA_PROVIDERS_FILE"), "{m}");
    }
}
