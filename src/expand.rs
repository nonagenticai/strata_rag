//! Query expansion (opt-in `expand: true`): one chat-provider call returns a HyDE passage plus three
//! rewrites; each variant gets its own dense list, RRF-fused with the original query's list
//! weighted `ORIGINAL_QUERY_WEIGHT`. Ported from KAG's FULL-mode expansion (kag
//! `src/services/ask_pipeline.py` `EXPAND_PROMPT_TEMPLATE`, `KAG_ASK_DENSE_ORIGINAL_QUERY_WEIGHT`)
//! so the HotpotQA comparison measures the same technique. Lexical search and rerank keep the
//! user's own question.

use crate::clients::chat::ChatMessage;
use crate::error::{Result, StrataError};
use crate::inference::InferenceHandle;

/// Longest a query waits for its expansion (the chat call, slot wait included): past it the query
/// runs with the question alone, as when the chat provider fails. Sized for the worst case of the
/// slow local model: `EXPAND_MAX_TOKENS` (600) at about 7.5 tok/s is about 80 s, plus 2x headroom
/// for an even slower model. Wait time is not a concern; cutting a generation short is.
pub const EXPAND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(180);

/// RRF weight of the original query's dense list among the variants. KAG measured that at 1.0 the
/// four generated variants hold 4 of 5 votes, so a drifted HyDE passage can outvote the user's words.
pub const ORIGINAL_QUERY_WEIGHT: f32 = 2.0;

pub const MAX_REWRITES: usize = 3;

/// HyDE paragraph + 3 short rewrites, as JSON; generous so the object is never cut mid-string.
pub const EXPAND_MAX_TOKENS: u32 = 600;

/// Longest accepted variant, in chars: a runaway generation must not become a 10k-token embed.
pub const MAX_VARIANT_CHARS: usize = 2000;

const EXPAND_PROMPT: &str = "Given a user search query, output a JSON object with:
  \"hyde_passage\": a short hypothetical paragraph that would directly answer the query as if it were an excerpt from a document.
  \"rewrites\": three alternative phrasings of the query that preserve intent but vary vocabulary.

Output strict JSON. No prose outside the JSON object.

Query: ";

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Expansion {
    pub hyde_passage: String,
    pub rewrites: Vec<String>,
}

impl Expansion {
    /// The texts to embed after the original query: HyDE passage first, then the rewrites.
    pub fn variants(&self) -> Vec<String> {
        std::iter::once(self.hyde_passage.clone())
            .chain(self.rewrites.iter().cloned())
            .collect()
    }
}

/// What `/api/query` reports about expansion when it was requested.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExpansionInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyde_passage: Option<String>,
    pub rewrites: Vec<String>,
    /// Set when expansion was requested but did not run; the query then used the question alone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,
    /// Wall time of the chat call (included in `timings_ms.total`).
    pub ms: u64,
}

pub fn build_messages(query: &str) -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: format!("{EXPAND_PROMPT}{query}\n"),
    }]
}

/// Parse the model's reply: the outermost `{…}` (tolerates a ```json fence or stray prose),
/// trimmed fields, blank rewrites dropped, at most `MAX_REWRITES`. A missing or blank
/// `hyde_passage` or an over-long variant is an error — the caller falls back to the question.
pub fn parse(reply: &str) -> Result<Expansion> {
    let bad = |m: &str| StrataError::GenerationUnavailable(format!("expansion: {m}"));
    let (Some(start), Some(end)) = (reply.find('{'), reply.rfind('}')) else {
        return Err(bad("no JSON object in reply"));
    };
    if end < start {
        return Err(bad("no JSON object in reply"));
    }
    #[derive(serde::Deserialize)]
    struct Raw {
        hyde_passage: String,
        #[serde(default)]
        rewrites: Vec<String>,
    }
    let raw: Raw =
        serde_json::from_str(&reply[start..=end]).map_err(|e| bad(&format!("decode: {e}")))?;
    let hyde_passage = raw.hyde_passage.trim().to_string();
    if hyde_passage.is_empty() {
        return Err(bad("empty hyde_passage"));
    }
    let rewrites: Vec<String> = raw
        .rewrites
        .iter()
        .map(|r| r.trim().to_string())
        .filter(|r| !r.is_empty())
        .take(MAX_REWRITES)
        .collect();
    let longest = std::iter::once(&hyde_passage)
        .chain(&rewrites)
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(0);
    if longest > MAX_VARIANT_CHARS {
        return Err(bad(&format!(
            "variant of {longest} chars exceeds {MAX_VARIANT_CHARS}"
        )));
    }
    Ok(Expansion {
        hyde_passage,
        rewrites,
    })
}

pub async fn expand(inf: &InferenceHandle, query: &str) -> Result<Expansion> {
    let reply = inf.chat(&build_messages(query), EXPAND_MAX_TOKENS).await?;
    parse(&reply)
}

/// Per-list RRF weights for `n_dense` dense lists (list 0 = original query): the original gets
/// `ORIGINAL_QUERY_WEIGHT` only when variants exist, so a non-expanded query is byte-identical.
pub fn dense_weights(n_dense: usize) -> Vec<f32> {
    let mut w = vec![1.0; n_dense];
    if n_dense > 1 {
        w[0] = ORIGINAL_QUERY_WEIGHT;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_kags_expansion_prompt_with_the_query_last() {
        let m = build_messages("who wrote Dune?");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].role, "user");
        assert!(
            m[0].content
                .contains("\"hyde_passage\": a short hypothetical paragraph")
        );
        assert!(m[0].content.contains("three alternative phrasings"));
        assert!(m[0].content.ends_with("Query: who wrote Dune?\n"));
    }

    #[test]
    fn parse_plain_json() {
        let e = parse(r#"{"hyde_passage":" Frank Herbert wrote Dune. ","rewrites":["a","b","c"]}"#)
            .unwrap();
        assert_eq!(e.hyde_passage, "Frank Herbert wrote Dune.");
        assert_eq!(e.rewrites, ["a", "b", "c"]);
        assert_eq!(e.variants(), ["Frank Herbert wrote Dune.", "a", "b", "c"]);
    }

    #[test]
    fn parse_tolerates_fences_and_prose() {
        let reply = "Sure:\n```json\n{\"hyde_passage\": \"p {x}\", \"rewrites\": [\"r\"]}\n```\n";
        let e = parse(reply).unwrap();
        assert_eq!(e.hyde_passage, "p {x}");
        assert_eq!(e.rewrites, ["r"]);
    }

    #[test]
    fn parse_drops_blank_rewrites_and_caps_at_three() {
        let e = parse(r#"{"hyde_passage":"p","rewrites":["", " a ","b","  ","c","d"]}"#).unwrap();
        assert_eq!(e.rewrites, ["a", "b", "c"]);
        let none = parse(r#"{"hyde_passage":"p"}"#).unwrap();
        assert!(none.rewrites.is_empty());
    }

    #[test]
    fn parse_rejects_what_cannot_be_used() {
        for reply in [
            "no json here",
            "} backwards {",
            r#"{"rewrites":["a"]}"#,
            r#"{"hyde_passage":"   ","rewrites":["a"]}"#,
            r#"{"hyde_passage":"cut off"#,
            r#"{"hyde_passage": 3}"#,
        ] {
            assert!(
                matches!(parse(reply), Err(StrataError::GenerationUnavailable(_))),
                "{reply}"
            );
        }
        let long = format!(
            r#"{{"hyde_passage":"{}","rewrites":[]}}"#,
            "x".repeat(MAX_VARIANT_CHARS + 1)
        );
        assert!(parse(&long).is_err());
        let at_cap = format!(
            r#"{{"hyde_passage":"{}","rewrites":[]}}"#,
            "é".repeat(MAX_VARIANT_CHARS)
        );
        assert!(parse(&at_cap).is_ok(), "the cap counts chars, not bytes");
    }

    #[test]
    fn dense_weights_only_boost_the_original_when_variants_exist() {
        assert_eq!(dense_weights(1), [1.0]);
        assert_eq!(dense_weights(5), [2.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(dense_weights(0).is_empty());
    }
}
