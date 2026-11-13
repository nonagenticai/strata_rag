//! `POST /api/ask` (spec §6): query, then generate an answer with numbered citations through the
//! tenant's chat provider (`crate::providers`, capability `chat`).

use std::time::Instant;

use uuid::Uuid;

use crate::clients::chat::ChatMessage;
use crate::engine::Engine;
use crate::error::{Result, StrataError};
use crate::expand::ExpansionInfo;
use crate::precedence::Layer;
use crate::query::{QueryRequest, QueryResult, StageProvider, StageProviders, Timings};
use crate::search::elapsed_ms;
use crate::tenant::Tenant;

pub const MAX_ANSWER_TOKENS: u32 = 8192;

const SYSTEM_PROMPT: &str = "You answer questions using only the numbered sources provided by the user. \
Cite the source of every statement with its number in square brackets, like [1] or [2]. \
If the sources do not contain the answer, say that you do not know. Do not use outside knowledge. \
Answer in complete sentences that state the answer itself; a citation marker on its own is not an answer.";

/// Appended to `SYSTEM_PROMPT` when platform sources are among the sources (spec
/// platform-knowledge §6.4).
const PRECEDENCE_RULE: &str = " Each source is labelled tenant (the organisation's own knowledge) or platform \
(general reference knowledge). When a tenant source and a platform source conflict, the tenant source \
overrides the platform source.";

pub fn d_max_tokens() -> u32 {
    800
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AskRequest {
    #[serde(flatten)]
    pub query: QueryRequest,
    #[serde(default = "d_max_tokens")]
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct AskTimings {
    #[serde(flatten)]
    pub query: Timings,
    pub generate: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AskResponse {
    pub answer: String,
    pub citations: Vec<Uuid>,
    pub results: Vec<QueryResult>,
    pub timings_ms: AskTimings,
    /// The query's stages plus `chat`, the generation (see `QueryResponse::providers`).
    pub providers: StageProviders,
    /// Present only when the request asked for `expand` (see `QueryResponse::expansion`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansion: Option<ExpansionInfo>,
}

/// The bracketed label of one source: its store path, then the document title and pages when known
/// (`library / physics · Survey · p. 2–3`), so the answer can name where a statement comes from.
/// With `layered` (platform sources present) prefixed by its layer (spec platform-knowledge §6.4):
/// `tenant: …` or `platform <key>@<version>: …`.
fn source_label(r: &QueryResult, layered: bool) -> String {
    let mut label = match (layered, r.layer, r.pack.as_deref()) {
        (false, _, _) => String::new(),
        (true, Layer::Platform, Some(pack)) => format!("platform {pack}: "),
        (true, Layer::Platform, None) => "platform: ".to_string(),
        (true, Layer::Tenant, _) => "tenant: ".to_string(),
    };
    label.push_str(&r.store_path.join(" / "));
    if let Some(title) = r.title.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        label.push_str(" · ");
        label.push_str(title);
    }
    match (r.page_start, r.page_end) {
        (Some(a), Some(b)) if b > a => label.push_str(&format!(" · p. {a}–{b}")),
        (Some(a), _) => label.push_str(&format!(" · p. {a}")),
        _ => {}
    }
    label
}

/// System message with the citation rules; user message with `[n] (label)\ntext` blocks (see
/// `source_label`), then the question. Only when platform sources are present are the labels
/// layered and the system prompt extended with the tenant-overrides-platform rule: a tenant-only
/// prompt stays exactly as before.
pub fn build_prompt(query: &str, results: &[QueryResult]) -> Vec<ChatMessage> {
    let layered = results.iter().any(|r| r.layer == Layer::Platform);
    let mut user = String::from("Sources:\n\n");
    if results.is_empty() {
        user.push_str("(no sources found)\n\n");
    }
    for (i, r) in results.iter().enumerate() {
        user.push_str(&format!(
            "[{}] ({})\n{}\n\n",
            i + 1,
            source_label(r, layered),
            r.text
        ));
    }
    user.push_str("Question: ");
    user.push_str(query);
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: if layered {
                format!("{SYSTEM_PROMPT}{PRECEDENCE_RULE}")
            } else {
                SYSTEM_PROMPT.to_string()
            },
        },
        ChatMessage {
            role: "user".to_string(),
            content: user,
        },
    ]
}

/// `[n]` / `[n, m]` markers → `results[n-1].chunk_id`, 1-based, out-of-range ignored, deduped, first-mention order.
pub fn extract_citations(answer: &str, results: &[QueryResult]) -> Vec<Uuid> {
    let mut out: Vec<Uuid> = Vec::new();
    let mut rest = answer;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        let Some(close) = after.find(']') else { break };
        let inner = &after[..close];
        let numbers: Option<Vec<usize>> = inner
            .split(',')
            .map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let Some(numbers) = numbers else {
            rest = after; // not a citation (e.g. "[[2]]" or "[abc]"): resume just past this '['
            continue;
        };
        for n in numbers {
            if (1..=results.len()).contains(&n) {
                let id = results[n - 1].chunk_id;
                if !out.contains(&id) {
                    out.push(id);
                }
            }
        }
        rest = &after[close + 1..];
    }
    out
}

impl Engine {
    /// `query` within `tenant` (filters included), then generate.
    pub async fn ask(&self, tenant: Tenant, req: &AskRequest) -> Result<AskResponse> {
        // the start store's chat provider (its nearest store binding, else the tenant's, D92); no
        // usable one: 409 provider_not_configured (or the registration's own error) before any
        // retrieval work. An ambiguous, unknown or forbidden target fails here too, as the query would.
        let start = self.query_start(tenant, &req.query).await?;
        let inf = self.inference_for(tenant, start).await?;
        inf.require_chat()?;
        if !(1..=MAX_ANSWER_TOKENS).contains(&req.max_tokens) {
            return Err(StrataError::InvalidInput(format!(
                "max_tokens must be between 1 and {MAX_ANSWER_TOKENS}, got {}",
                req.max_tokens
            )));
        }
        let resp = self.query(tenant, &req.query).await?;
        let messages = build_prompt(req.query.query.trim(), &resp.results);
        let t = Instant::now();
        let answer = inf.chat(&messages, req.max_tokens).await?;
        let generate = elapsed_ms(t);
        let citations = extract_citations(&answer, &resp.results);
        let mut providers = resp.providers;
        providers.chat = inf.chat_provider().map(StageProvider::from);
        Ok(AskResponse {
            answer,
            citations,
            results: resp.results,
            timings_ms: AskTimings {
                query: resp.timings_ms,
                generate,
            },
            providers,
            expansion: resp.expansion,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Scores;

    fn result(text: &str, path: &[&str]) -> QueryResult {
        QueryResult {
            chunk_id: Uuid::now_v7(),
            document_id: Uuid::now_v7(),
            store_path: path.iter().map(|s| s.to_string()).collect(),
            layer: Layer::Tenant,
            pack: None,
            ord: 0,
            text: text.to_string(),
            page_start: None,
            page_end: None,
            source: "src".to_string(),
            title: None,
            scores: Scores {
                dense: None,
                lexical: None,
                rerank: Some(0.5),
            },
            source_id: None,
            source_run: None,
            metadata: serde_json::Map::new(),
        }
    }

    #[test]
    fn build_prompt_numbers_sources_with_store_paths() {
        let rs = vec![
            result("quantum photons", &["library", "physics"]),
            result("violin melody", &["library", "music"]),
        ];
        let msgs = build_prompt("what is light?", &rs);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.contains("only"));
        assert!(msgs[0].content.contains("[1]"));
        // measured 2026-10-16: without this rule the model answers a near-verbatim source with
        // just "[2]" (8/35 HotpotQA trials; 0/35 with it)
        assert!(
            msgs[0]
                .content
                .contains("a citation marker on its own is not an answer")
        );
        assert_eq!(msgs[1].role, "user");
        assert_eq!(
            msgs[1].content,
            "Sources:\n\n[1] (library / physics)\nquantum photons\n\n[2] (library / music)\nviolin melody\n\nQuestion: what is light?"
        );
    }

    #[test]
    fn build_prompt_labels_sources_with_title_and_pages() {
        let mut span = result("survey text", &["library", "harbour"]);
        span.title = Some("Quillmere Harbour Survey".to_string());
        span.page_start = Some(2);
        span.page_end = Some(3);
        let mut one = result("one page", &["library"]);
        one.title = Some("  ".to_string()); // a blank title is no title
        one.page_start = Some(4);
        one.page_end = Some(4);
        let msgs = build_prompt("q?", &[span, one]);
        assert_eq!(
            msgs[1].content,
            "Sources:\n\n[1] (library / harbour · Quillmere Harbour Survey · p. 2–3)\nsurvey text\n\n[2] (library · p. 4)\none page\n\nQuestion: q?"
        );
    }

    #[test]
    fn build_prompt_labels_layers_and_states_precedence_with_platform_sources() {
        let own = result("our leave is thirty days", &["hr", "leave"]);
        let mut shared = result("statutory leave is twenty days", &["base", "employment"]);
        shared.layer = Layer::Platform;
        shared.pack = Some("base@3".to_string());
        let msgs = build_prompt("how much leave?", &[own.clone(), shared]);
        assert!(
            msgs[0]
                .content
                .ends_with("the tenant source overrides the platform source."),
            "{}",
            msgs[0].content
        );
        assert_eq!(
            msgs[1].content,
            "Sources:\n\n[1] (tenant: hr / leave)\nour leave is thirty days\n\n[2] (platform base@3: base / employment)\nstatutory leave is twenty days\n\nQuestion: how much leave?"
        );
        // tenant sources only: no layer labels, no extra rule
        let msgs = build_prompt("q?", &[own]);
        assert_eq!(msgs[0].content, SYSTEM_PROMPT);
        assert!(msgs[1].content.starts_with("Sources:\n\n[1] (hr / leave)"));
    }

    #[test]
    fn build_prompt_without_sources_says_so() {
        let msgs = build_prompt("q?", &[]);
        assert_eq!(
            msgs[1].content,
            "Sources:\n\n(no sources found)\n\nQuestion: q?"
        );
    }

    #[test]
    fn extract_citations_maps_markers_in_first_mention_order() {
        let rs = vec![
            result("a", &["x"]),
            result("b", &["x"]),
            result("c", &["x"]),
        ];
        let (r1, r2, r3) = (rs[0].chunk_id, rs[1].chunk_id, rs[2].chunk_id);
        assert_eq!(extract_citations("see [2] and [1]", &rs), vec![r2, r1]);
        assert_eq!(
            extract_citations("[1] [1] [2] [1]", &rs),
            vec![r1, r2],
            "deduped"
        );
        assert_eq!(extract_citations("both [3, 1]", &rs), vec![r3, r1]);
        assert_eq!(extract_citations("nested [[2]]", &rs), vec![r2]);
        assert_eq!(extract_citations("spaced [ 3 ]", &rs), vec![r3]);
    }

    #[test]
    fn extract_citations_ignores_out_of_range_and_non_numbers() {
        let rs = vec![result("a", &["x"]), result("b", &["x"])];
        assert!(extract_citations("[9] [0] [abc] [] [1a] no cites", &rs).is_empty());
        assert_eq!(extract_citations("[9] then [2]", &rs), vec![rs[1].chunk_id]);
        assert!(extract_citations("unclosed [1", &rs).is_empty());
    }

    #[test]
    fn ask_request_flattens_query_params_and_defaults_max_tokens() {
        let r: AskRequest = serde_json::from_str(r#"{"query":"x","top_k":3}"#).unwrap();
        assert_eq!(r.query.query, "x");
        assert_eq!(r.query.top_k, 3);
        assert_eq!(r.max_tokens, 800);
        let r: AskRequest =
            serde_json::from_str(r#"{"query":"x","source_ids":["s"],"metadata":{"k":1}}"#).unwrap();
        assert_eq!(r.query.source_ids, Some(vec!["s".to_string()]));
        assert_eq!(r.query.metadata, Some(serde_json::json!({"k": 1})));
    }
}
