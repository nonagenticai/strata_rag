//! `POST /api/ask` (spec §6): query, then generate an answer with numbered citations through the
//! tenant's chat provider (`crate::providers`, capability `chat`).
//!
//! One numbered source is one **document**, not one retrieved chunk: a document fetched as four
//! chunks is `[1]` once, with all four passages under that number, so an answer cites the document
//! it came from rather than the retrieval split.

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

/// Most `instructions` one ask may carry: a job's persona, not a second corpus.
pub const MAX_INSTRUCTIONS_BYTES: usize = 16 * 1024;

const SYSTEM_PROMPT: &str = "You answer questions using only the numbered sources provided by the user. \
Each source is one document and may hold several passages. \
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
    /// The caller's job persona (actor, stance, preamble). Prepended to the system message, before
    /// the fixed citation rules; absent or empty leaves the prompt exactly as it was.
    #[serde(default)]
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct AskTimings {
    #[serde(flatten)]
    pub query: Timings,
    pub generate: u64,
}

/// One numbered prompt source: a whole document and the chunks of it that were retrieved.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AskDocument {
    /// 1-based position in the prompt — the number the answer cites.
    pub number: usize,
    pub document_id: Uuid,
    /// Store names from the root down to the document's store (as `QueryResult::store_path`).
    pub store_path: Vec<String>,
    pub layer: Layer,
    /// The platform hit's pack, `"<key>@<version>"`; absent for a tenant hit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pack: Option<String>,
    /// The document's `source` label and `title`, for citations.
    pub source: String,
    pub title: Option<String>,
    /// The merged page span of this document's retrieved chunks; `null` when the document has no pages.
    pub page_start: Option<i32>,
    pub page_end: Option<i32>,
    /// The retrieved chunks of this document, in document order.
    pub chunk_ids: Vec<Uuid>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AskResponse {
    pub answer: String,
    /// The chunk ids of every cited document (all its retrieved chunks, first-mention document order).
    pub citations: Vec<Uuid>,
    /// The numbered prompt sources: one entry per document, in `[n]` order.
    pub documents: Vec<AskDocument>,
    pub results: Vec<QueryResult>,
    pub timings_ms: AskTimings,
    /// The query's stages plus `chat`, the generation (see `QueryResponse::providers`).
    pub providers: StageProviders,
    /// Present only when the request asked for `expand` (see `QueryResponse::expansion`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansion: Option<ExpansionInfo>,
}

/// The retrieved chunks grouped by document: first-appearance order, each group in `ord` order.
pub fn group_documents(results: &[QueryResult]) -> Vec<Vec<usize>> {
    let mut groups: Vec<(Uuid, Vec<usize>)> = Vec::new();
    for (i, r) in results.iter().enumerate() {
        match groups.iter_mut().find(|(d, _)| *d == r.document_id) {
            Some((_, idx)) => idx.push(i),
            None => groups.push((r.document_id, vec![i])),
        }
    }
    for (_, idx) in &mut groups {
        idx.sort_by_key(|&i| results[i].ord);
    }
    groups.into_iter().map(|(_, idx)| idx).collect()
}

/// The document-level sources of `results`, numbered in prompt order.
pub fn documents(results: &[QueryResult]) -> Vec<AskDocument> {
    group_documents(results)
        .iter()
        .enumerate()
        .map(|(n, idx)| {
            let first = &results[idx[0]];
            let (page_start, page_end) = merged_pages(results, idx);
            AskDocument {
                number: n + 1,
                document_id: first.document_id,
                store_path: first.store_path.clone(),
                layer: first.layer,
                pack: first.pack.clone(),
                source: first.source.clone(),
                title: first.title.clone(),
                page_start,
                page_end,
                chunk_ids: idx.iter().map(|&i| results[i].chunk_id).collect(),
            }
        })
        .collect()
}

/// The merged page span of one document's chunks, when they carry pages.
fn merged_pages(results: &[QueryResult], idx: &[usize]) -> (Option<i32>, Option<i32>) {
    let start = idx.iter().filter_map(|&i| results[i].page_start).min();
    let end = idx.iter().filter_map(|&i| results[i].page_end).max();
    (start, end)
}

/// The bracketed label of one source: its store path, then the document title and pages when known
/// (`library / physics · Survey · p. 2–3`), so the answer can name where a statement comes from.
/// With `layered` (platform sources present) prefixed by its layer (spec platform-knowledge §6.4):
/// `tenant: …` or `platform <key>@<version>: …`.
fn source_label(
    r: &QueryResult,
    (page_start, page_end): (Option<i32>, Option<i32>),
    layered: bool,
) -> String {
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
    match (page_start, page_end) {
        (Some(a), Some(b)) if b > a => label.push_str(&format!(" · p. {a}–{b}")),
        (Some(a), _) => label.push_str(&format!(" · p. {a}")),
        _ => {}
    }
    label
}

/// The system message: the caller's instructions when given, then the fixed citation rules. Only
/// when platform sources are present is the tenant-overrides-platform rule appended, so a
/// tenant-only prompt with no instructions stays exactly as it was.
fn system_prompt(instructions: Option<&str>, layered: bool) -> String {
    let mut out = String::new();
    if let Some(i) = instructions.map(str::trim).filter(|i| !i.is_empty()) {
        out.push_str(i);
        out.push_str("\n\n");
    }
    out.push_str(SYSTEM_PROMPT);
    if layered {
        out.push_str(PRECEDENCE_RULE);
    }
    out
}

/// User message with one `[n] (label)` block per document — its chunks in document order, blank-line
/// separated — then the question.
pub fn build_prompt(
    query: &str,
    results: &[QueryResult],
    instructions: Option<&str>,
) -> Vec<ChatMessage> {
    let layered = results.iter().any(|r| r.layer == Layer::Platform);
    let mut user = String::from("Sources:\n\n");
    if results.is_empty() {
        user.push_str("(no sources found)\n\n");
    }
    for (n, idx) in group_documents(results).iter().enumerate() {
        let first = &results[idx[0]];
        user.push_str(&format!(
            "[{}] ({})\n",
            n + 1,
            source_label(first, merged_pages(results, idx), layered)
        ));
        for (k, &i) in idx.iter().enumerate() {
            if k > 0 {
                user.push_str("\n\n");
            }
            user.push_str(results[i].text.trim_end());
        }
        user.push_str("\n\n");
    }
    user.push_str("Question: ");
    user.push_str(query);
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: system_prompt(instructions, layered),
        },
        ChatMessage {
            role: "user".to_string(),
            content: user,
        },
    ]
}

/// `[n]` / `[n, m]` markers → the chunk ids of the `n`th document, 1-based, out-of-range ignored,
/// deduped, first-mention order.
pub fn extract_citations(answer: &str, documents: &[AskDocument]) -> Vec<Uuid> {
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
            if let Some(doc) = n.checked_sub(1).and_then(|i| documents.get(i)) {
                for id in &doc.chunk_ids {
                    if !out.contains(id) {
                        out.push(*id);
                    }
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
        if let Some(i) = &req.instructions
            && i.len() > MAX_INSTRUCTIONS_BYTES
        {
            return Err(StrataError::InvalidInput(format!(
                "instructions must be at most {MAX_INSTRUCTIONS_BYTES} bytes, got {}",
                i.len()
            )));
        }
        let resp = self.query(tenant, &req.query).await?;
        let documents = documents(&resp.results);
        let messages = build_prompt(
            req.query.query.trim(),
            &resp.results,
            req.instructions.as_deref(),
        );
        let t = Instant::now();
        let answer = inf.chat(&messages, req.max_tokens).await?;
        let generate = elapsed_ms(t);
        let citations = extract_citations(&answer, &documents);
        let mut providers = resp.providers;
        providers.chat = inf.chat_provider().map(StageProvider::from);
        Ok(AskResponse {
            answer,
            citations,
            documents,
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

    /// A chunk of the document `document_id`, at `ord`.
    fn chunk_of(document_id: Uuid, text: &str, ord: i32, path: &[&str]) -> QueryResult {
        QueryResult {
            document_id,
            ord,
            ..result(text, path)
        }
    }

    #[test]
    fn build_prompt_numbers_sources_with_store_paths() {
        let rs = vec![
            result("quantum photons", &["library", "physics"]),
            result("violin melody", &["library", "music"]),
        ];
        let msgs = build_prompt("what is light?", &rs, None);
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
    fn build_prompt_numbers_one_document_once_with_all_its_chunks() {
        let doc = Uuid::now_v7();
        // retrieved in rank order: the document's second chunk outranks its first
        let rs = vec![
            chunk_of(doc, "second passage", 1, &["library", "harbour"]),
            result("other document", &["library", "music"]),
            chunk_of(doc, "first passage", 0, &["library", "harbour"]),
        ];
        let msgs = build_prompt("what?", &rs, None);
        assert_eq!(
            msgs[1].content,
            "Sources:\n\n[1] (library / harbour)\nfirst passage\n\nsecond passage\n\n[2] (library / music)\nother document\n\nQuestion: what?"
        );
        let docs = documents(&rs);
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].number, 1);
        assert_eq!(docs[0].document_id, doc);
        assert_eq!(docs[0].chunk_ids, vec![rs[2].chunk_id, rs[0].chunk_id]);
        assert_eq!(docs[1].number, 2);
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
        let msgs = build_prompt("q?", &[span, one], None);
        assert_eq!(
            msgs[1].content,
            "Sources:\n\n[1] (library / harbour · Quillmere Harbour Survey · p. 2–3)\nsurvey text\n\n[2] (library · p. 4)\none page\n\nQuestion: q?"
        );
    }

    #[test]
    fn build_prompt_merges_one_documents_page_span() {
        let doc = Uuid::now_v7();
        let mut a = chunk_of(doc, "p2", 0, &["library"]);
        a.page_start = Some(2);
        a.page_end = Some(2);
        let mut b = chunk_of(doc, "p5", 1, &["library"]);
        b.page_start = Some(5);
        b.page_end = Some(5);
        let msgs = build_prompt("q?", &[a, b], None);
        assert_eq!(
            msgs[1].content,
            "Sources:\n\n[1] (library · p. 2–5)\np2\n\np5\n\nQuestion: q?"
        );
    }

    #[test]
    fn build_prompt_labels_layers_and_states_precedence_with_platform_sources() {
        let own = result("our leave is thirty days", &["hr", "leave"]);
        let mut shared = result("statutory leave is twenty days", &["base", "employment"]);
        shared.layer = Layer::Platform;
        shared.pack = Some("base@3".to_string());
        let msgs = build_prompt("how much leave?", &[own.clone(), shared], None);
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
        let msgs = build_prompt("q?", &[own], None);
        assert_eq!(msgs[0].content, SYSTEM_PROMPT);
        assert!(msgs[1].content.starts_with("Sources:\n\n[1] (hr / leave)"));
    }

    #[test]
    fn build_prompt_without_sources_says_so() {
        let msgs = build_prompt("q?", &[], None);
        assert_eq!(
            msgs[1].content,
            "Sources:\n\n(no sources found)\n\nQuestion: q?"
        );
    }

    #[test]
    fn instructions_come_before_the_rules_and_absence_is_unchanged() {
        let rs = vec![result("a", &["x"])];
        assert_eq!(build_prompt("q?", &rs, None)[0].content, SYSTEM_PROMPT);
        assert_eq!(
            build_prompt("q?", &rs, Some("  "))[0].content,
            SYSTEM_PROMPT
        );
        let msgs = build_prompt("q?", &rs, Some("You are the site host."));
        assert_eq!(
            msgs[0].content,
            format!("You are the site host.\n\n{SYSTEM_PROMPT}")
        );
    }

    #[test]
    fn extract_citations_maps_markers_in_first_mention_order() {
        let rs = vec![
            result("a", &["x"]),
            result("b", &["x"]),
            result("c", &["x"]),
        ];
        let docs = documents(&rs);
        let (r1, r2, r3) = (rs[0].chunk_id, rs[1].chunk_id, rs[2].chunk_id);
        assert_eq!(extract_citations("see [2] and [1]", &docs), vec![r2, r1]);
        assert_eq!(
            extract_citations("[1] [1] [2] [1]", &docs),
            vec![r1, r2],
            "deduped"
        );
        assert_eq!(extract_citations("both [3, 1]", &docs), vec![r3, r1]);
        assert_eq!(extract_citations("nested [[2]]", &docs), vec![r2]);
        assert_eq!(extract_citations("spaced [ 3 ]", &docs), vec![r3]);
    }

    #[test]
    fn extract_citations_expands_a_cited_document_to_its_chunks() {
        let doc = Uuid::now_v7();
        let rs = vec![
            chunk_of(doc, "first", 0, &["x"]),
            chunk_of(doc, "second", 1, &["x"]),
        ];
        let docs = documents(&rs);
        assert_eq!(docs.len(), 1);
        assert_eq!(
            extract_citations("as [1] says", &docs),
            vec![rs[0].chunk_id, rs[1].chunk_id]
        );
    }

    #[test]
    fn extract_citations_ignores_out_of_range_and_non_numbers() {
        let rs = vec![result("a", &["x"]), result("b", &["x"])];
        let docs = documents(&rs);
        assert!(extract_citations("[9] [0] [abc] [] [1a] no cites", &docs).is_empty());
        assert_eq!(
            extract_citations("[9] then [2]", &docs),
            vec![rs[1].chunk_id]
        );
        assert!(extract_citations("unclosed [1", &docs).is_empty());
    }

    #[test]
    fn ask_request_flattens_query_params_and_defaults_max_tokens() {
        let r: AskRequest = serde_json::from_str(r#"{"query":"x","top_k":3}"#).unwrap();
        assert_eq!(r.query.query, "x");
        assert_eq!(r.query.top_k, 3);
        assert_eq!(r.max_tokens, 800);
        assert_eq!(r.instructions, None);
        let r: AskRequest =
            serde_json::from_str(r#"{"query":"x","source_ids":["s"],"metadata":{"k":1}}"#).unwrap();
        assert_eq!(r.query.source_ids, Some(vec!["s".to_string()]));
        assert_eq!(r.query.metadata, Some(serde_json::json!({"k": 1})));
        let r: AskRequest =
            serde_json::from_str(r#"{"query":"x","instructions":"You are the host."}"#).unwrap();
        assert_eq!(r.instructions.as_deref(), Some("You are the host."));
    }
}
