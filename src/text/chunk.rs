//! Token-aware chunking: paragraph → sentence → token boundaries, greedy packing with
//! whole-unit overlap. Every chunk is a trimmed, contiguous slice of the input.
//!
//! Markdown structure: a chunk (and its overlap) never crosses a top-level section — a
//! paragraph opening with an ATX heading at the shallowest level the text uses (a lone
//! leading heading, i.e. Bundle's `# {title}`, does not set that level) — and a chunk does not
//! end on a heading-only paragraph while that heading's section continues; a section holding
//! only headings (a title) is merged into the next. `<!-- … -->` anchor
//! lines are transparent, so `<!-- page:N -->\n## X` opens a section. Fenced code is text.
//!
//! Pages: Bundle's `<!-- page:N -->` anchor lines give each chunk `page_start` (the page in
//! effect where it starts, else its first anchor) and `page_end` (its last anchor, else
//! `page_start`). Text without anchors has no pages. Anchors stay in the chunk text.

use std::ops::Range;
use std::path::Path;

use unicode_segmentation::UnicodeSegmentation;

use crate::error::{Result, StrataError};

pub trait TokenCounter: Send + Sync {
    fn count(&self, text: &str) -> usize;
    /// Split `text` into consecutive pieces of at most `max` tokens, preserving all text.
    fn split_by_tokens(&self, text: &str, max: usize) -> Vec<String>;
}

/// One token per whitespace-separated word. Deterministic; used by tests and the fake engine.
pub struct WhitespaceCounter;

impl TokenCounter for WhitespaceCounter {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn split_by_tokens(&self, text: &str, max: usize) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let max = max.max(1);
        let mut cuts = vec![0usize];
        let mut words = 0usize;
        let mut prev_ws = true;
        for (i, c) in text.char_indices() {
            let ws = c.is_whitespace();
            if !ws && prev_ws {
                if words > 0 && words.is_multiple_of(max) {
                    cuts.push(i);
                }
                words += 1;
            }
            prev_ws = ws;
        }
        cuts.push(text.len());
        pieces_from_cuts(text, &cuts)
    }
}

/// HuggingFace `tokenizers` counter (bge-m3's `tokenizer.json` in production).
/// Counts exclude special tokens; truncation and padding are disabled.
pub struct HfTokenCounter {
    tokenizer: tokenizers::Tokenizer,
}

impl HfTokenCounter {
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| StrataError::Model(format!("load tokenizer {}: {e}", path.display())))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| StrataError::Model(format!("disable tokenizer truncation: {e}")))?;
        tokenizer.with_padding(None);
        Ok(Self { tokenizer })
    }
}

impl TokenCounter for HfTokenCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        match self.tokenizer.encode(text, false) {
            Ok(enc) => enc.get_ids().len(),
            // Encoding plain UTF-8 does not fail in practice; a char count is a safe
            // over-estimate for bge-m3 if it ever does.
            Err(_) => text.chars().count(),
        }
    }

    fn split_by_tokens(&self, text: &str, max: usize) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let max = max.max(1);
        let enc = match self.tokenizer.encode(text, false) {
            Ok(enc) => enc,
            Err(_) => return split_chars_by(text, max),
        };
        let mut cuts = vec![0usize];
        // `encode` reports byte offsets into the original string.
        for (i, &(start, _)) in enc.get_offsets().iter().enumerate() {
            if i == 0 || !i.is_multiple_of(max) {
                continue;
            }
            let mut c = start.min(text.len());
            while !text.is_char_boundary(c) {
                c -= 1;
            }
            if cuts.last().is_some_and(|&last| c > last) {
                cuts.push(c);
            }
        }
        cuts.push(text.len());
        pieces_from_cuts(text, &cuts)
    }
}

fn pieces_from_cuts(text: &str, cuts: &[usize]) -> Vec<String> {
    cuts.windows(2)
        .filter(|w| w[1] > w[0])
        .map(|w| text[w[0]..w[1]].to_string())
        .collect()
}

fn split_chars_by(text: &str, max: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(max.max(1))
        .map(|c| c.iter().collect())
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub ord: i32,
    pub text: String,
    pub token_count: i32,
    pub page_start: Option<i32>,
    pub page_end: Option<i32>,
}

/// Every `<!-- page:N -->` anchor line, as `(byte offset of the line, N)`, in text order.
fn page_anchors(text: &str) -> Vec<(usize, i32)> {
    let mut out = Vec::new();
    let mut off = 0usize;
    for line in text.split_inclusive('\n') {
        let page = line
            .trim()
            .strip_prefix("<!-- page:")
            .and_then(|r| r.strip_suffix("-->"))
            .and_then(|n| n.trim().parse::<i32>().ok());
        if let Some(n) = page {
            out.push((off, n));
        }
        off += line.len();
    }
    out
}

fn pages_of(anchors: &[(usize, i32)], r: &Range<usize>) -> (Option<i32>, Option<i32>) {
    let before = anchors.iter().take_while(|(o, _)| *o <= r.start).last();
    let inside: Vec<i32> = anchors
        .iter()
        .filter(|(o, _)| *o >= r.start && *o < r.end)
        .map(|(_, n)| *n)
        .collect();
    let start = before.map(|(_, n)| *n).or_else(|| inside.first().copied());
    let end = inside.last().copied().or(start);
    (start, end)
}

/// A packing unit: a trimmed byte range of the input, its exact token count, the top-level
/// section it belongs to, and whether it is a heading-only paragraph.
#[derive(Debug, Clone)]
struct Unit {
    range: Range<usize>,
    tokens: usize,
    section: usize,
    heading: bool,
}

/// Chunks `text` into pieces of at most `target` tokens (as counted by `tc`), preferring
/// paragraph, then sentence, then token boundaries; consecutive chunks share whole
/// trailing units totalling at most `overlap` tokens. `target == 0` is treated as 1.
pub fn chunk_text(text: &str, tc: &dyn TokenCounter, target: usize, overlap: usize) -> Vec<Chunk> {
    let target = target.max(1);
    let units = build_units(text, tc, target);
    pack(text, tc, &units, target, overlap)
}

fn trim_range(text: &str, r: Range<usize>) -> Range<usize> {
    let s = &text[r.clone()];
    let lead = s.len() - s.trim_start().len();
    if lead == s.len() {
        return r.start..r.start;
    }
    let trail = s.len() - s.trim_end().len();
    (r.start + lead)..(r.end - trail)
}

fn paragraph_ranges(text: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0usize;
    for (i, _) in text.match_indices("\n\n") {
        out.push(start..i);
        start = i + 2;
    }
    out.push(start..text.len());
    out
}

/// ATX heading level of `line` (`#`..`######` then a space, a tab or nothing; up to 3 spaces
/// of indent), else `None`.
fn heading_level(line: &str) -> Option<usize> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > 3 {
        return None;
    }
    let rest = &line[indent..];
    let level = rest.len() - rest.trim_start_matches('#').len();
    let after = &rest[level..];
    ((1..=6).contains(&level) && (after.is_empty() || after.starts_with([' ', '\t'])))
        .then_some(level)
}

fn is_anchor(line: &str) -> bool {
    let l = line.trim();
    l.starts_with("<!--") && l.ends_with("-->")
}

fn is_fence(line: &str) -> bool {
    let l = line.trim_start();
    l.starts_with("```") || l.starts_with("~~~")
}

/// A non-empty paragraph's structure: the level of the heading it opens with (its first
/// non-anchor line), and whether it holds nothing but headings and anchors.
struct Para {
    range: Range<usize>,
    level: Option<usize>,
    heading_only: bool,
}

fn paragraphs(text: &str) -> Vec<Para> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for para in paragraph_ranges(text) {
        let para = trim_range(text, para);
        if para.is_empty() {
            continue;
        }
        let fenced_at_start = in_fence;
        let mut has_fence = false;
        let mut levels = Vec::new(); // per non-anchor line: its heading level, if any
        for line in text[para.clone()].lines() {
            if is_fence(line) {
                in_fence = !in_fence;
                has_fence = true;
                levels.push(None);
            } else if !is_anchor(line) {
                levels.push(heading_level(line));
            }
        }
        let structural = !fenced_at_start && !has_fence;
        out.push(Para {
            range: para,
            level: levels
                .first()
                .copied()
                .flatten()
                .filter(|_| !fenced_at_start),
            heading_only: structural && !levels.is_empty() && levels.iter().all(Option::is_some),
        });
    }
    out
}

/// The section level: the shallowest heading level, unless its only heading is the leading
/// paragraph (a title), in which case the shallowest level among the remaining paragraphs.
/// `None` (no headings) never opens a section.
fn top_level(paras: &[Para]) -> Option<usize> {
    let min = paras.iter().filter_map(|p| p.level).min()?;
    let at_min = paras.iter().filter(|p| p.level == Some(min)).count();
    if at_min == 1 && paras[0].level == Some(min) {
        return paras[1..].iter().filter_map(|p| p.level).min();
    }
    Some(min)
}

fn build_units(text: &str, tc: &dyn TokenCounter, target: usize) -> Vec<Unit> {
    let paras = paragraphs(text);
    let top = top_level(&paras);
    let mut units = Vec::new();
    let mut section = 0usize;
    let mut section_has_body = false; // a heading-only section (a title) joins the next one
    for p in paras {
        if section_has_body && p.level.is_some_and(|l| Some(l) <= top) {
            section += 1;
            section_has_body = false;
        }
        section_has_body |= !p.heading_only;
        let para = p.range;
        let n = tc.count(&text[para.clone()]);
        if n <= target {
            units.push(Unit {
                range: para,
                tokens: n,
                section,
                heading: p.heading_only,
            });
            continue;
        }
        let from = units.len();
        for (off, sentence) in text[para.clone()].split_sentence_bound_indices() {
            let start = para.start + off;
            let r = trim_range(text, start..start + sentence.len());
            if r.is_empty() {
                continue;
            }
            let n = tc.count(&text[r.clone()]);
            if n <= target {
                units.push(Unit {
                    range: r,
                    tokens: n,
                    section: 0,
                    heading: false,
                });
            } else {
                fit_by_tokens(text, r, tc, target, target, &mut units);
            }
        }
        for u in &mut units[from..] {
            u.section = section;
        }
    }
    units
}

/// Cuts `range` at token boundaries into units of at most `target` tokens each.
/// Re-encoding a piece alone may cost more tokens than its share of the whole
/// (sentencepiece `▁` prefix), so each piece is re-counted and, if still too big,
/// re-split with a strictly smaller `max`; at `max == 1` it falls back to characters.
fn fit_by_tokens(
    text: &str,
    range: Range<usize>,
    tc: &dyn TokenCounter,
    target: usize,
    max: usize,
    out: &mut Vec<Unit>,
) {
    let slice = &text[range.clone()];
    let pieces = tc.split_by_tokens(slice, max);
    if pieces.concat() != slice {
        // The counter broke its "preserve all text" contract; do not trust its offsets.
        split_into_char_units(text, range, tc, out);
        return;
    }
    let mut off = range.start;
    for piece in &pieces {
        let r = trim_range(text, off..off + piece.len());
        off += piece.len();
        if r.is_empty() {
            continue;
        }
        let n = tc.count(&text[r.clone()]);
        if n <= target {
            out.push(Unit {
                range: r,
                tokens: n,
                section: 0,
                heading: false,
            });
        } else if max > 1 {
            let next = max.saturating_sub(n - target).clamp(1, max - 1);
            fit_by_tokens(text, r, tc, target, next, out);
        } else {
            split_into_char_units(text, r, tc, out);
        }
    }
}

fn split_into_char_units(
    text: &str,
    range: Range<usize>,
    tc: &dyn TokenCounter,
    out: &mut Vec<Unit>,
) {
    for (i, c) in text[range.clone()].char_indices() {
        if c.is_whitespace() {
            continue;
        }
        let start = range.start + i;
        let r = start..start + c.len_utf8();
        let tokens = tc.count(&text[r.clone()]);
        out.push(Unit {
            range: r,
            tokens,
            section: 0,
            heading: false,
        });
    }
}

fn pack(
    text: &str,
    tc: &dyn TokenCounter,
    units: &[Unit],
    target: usize,
    overlap: usize,
) -> Vec<Chunk> {
    let measure = |start: usize, end: usize| -> (Range<usize>, usize) {
        let r = units[start].range.start..units[end - 1].range.end;
        let n = tc.count(&text[r.clone()]);
        (r, n)
    };

    let anchors = page_anchors(text);
    let mut chunks = Vec::new();
    let mut start = 0usize; // first unit of the next chunk (< fresh when overlapping)
    let mut fresh = 0usize; // first unit not yet in any chunk
    while fresh < units.len() {
        let mut end = fresh + 1;
        let mut est: usize = units[start..end].iter().map(|u| u.tokens).sum();
        let section = units[fresh].section;
        while end < units.len()
            && units[end].section == section
            && est + units[end].tokens <= target
        {
            est += units[end].tokens;
            end += 1;
        }
        // A heading-only paragraph goes with the body that follows it, not at a chunk's tail.
        let ends_on_heading = |end: usize| {
            end > fresh + 1
                && end < units.len()
                && units[end - 1].heading
                && units[end].section == section
        };
        while ends_on_heading(end) {
            end -= 1;
        }
        let (mut range, mut n) = measure(start, end);
        while n > target {
            if end > fresh + 1 {
                end -= 1;
            } else if start < fresh {
                start = fresh;
            } else {
                // A single unit is <= target by construction, except a lone character
                // that alone exceeds a pathological target; emit it rather than drop text.
                break;
            }
            while ends_on_heading(end) {
                end -= 1;
            }
            (range, n) = measure(start, end);
        }
        let (page_start, page_end) = pages_of(&anchors, &range);
        chunks.push(Chunk {
            ord: chunks.len() as i32,
            text: text[range].to_string(),
            token_count: n as i32,
            page_start,
            page_end,
        });

        let emitted_start = start;
        fresh = end;
        start = end;
        if fresh < units.len() {
            let mut ov = 0usize;
            while start > emitted_start + 1 {
                if units[start - 1].section != units[fresh].section {
                    break;
                }
                let t = units[start - 1].tokens;
                if ov + t > overlap || ov + t + units[fresh].tokens > target {
                    break;
                }
                ov += t;
                start -= 1;
            }
        }
    }
    chunks
}
#[cfg(test)]
mod tests {
    use super::*;

    /// One token per character (spaces included); a stand-in for CJK / minified text
    /// where a whitespace counter would see a single "word".
    struct CharCounter;

    impl TokenCounter for CharCounter {
        fn count(&self, text: &str) -> usize {
            text.chars().count()
        }
        fn split_by_tokens(&self, text: &str, max: usize) -> Vec<String> {
            let chars: Vec<char> = text.chars().collect();
            chars
                .chunks(max.max(1))
                .map(|c| c.iter().collect())
                .collect()
        }
    }

    /// Like sentencepiece: every separately-encoded text costs one extra token (`▁`),
    /// so a piece cut at exactly `max` tokens re-counts as `max + 1`.
    struct PrefixCounter;

    impl TokenCounter for PrefixCounter {
        fn count(&self, text: &str) -> usize {
            let words = text.split_whitespace().count();
            if words == 0 { 0 } else { words + 1 }
        }
        fn split_by_tokens(&self, text: &str, max: usize) -> Vec<String> {
            WhitespaceCounter.split_by_tokens(text, max)
        }
    }

    /// Non-additive the other way: joining sentences costs 3 extra tokens per ". " seam,
    /// so the summed estimate under-counts and the exact re-count must shrink the chunk.
    struct SeamPenaltyCounter;

    impl TokenCounter for SeamPenaltyCounter {
        fn count(&self, text: &str) -> usize {
            text.split_whitespace().count() + 3 * text.matches(". ").count()
        }
        fn split_by_tokens(&self, text: &str, max: usize) -> Vec<String> {
            WhitespaceCounter.split_by_tokens(text, max)
        }
    }

    /// A char-level HF tokenizer built from JSON (no model download needed): every
    /// character, including spaces and newlines, is one `[UNK]` token.
    fn char_level_hf() -> HfTokenCounter {
        let json = r#"{
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [],
          "normalizer": null,
          "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": false},
          "post_processor": null,
          "decoder": null,
          "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
        }"#;
        let path =
            std::env::temp_dir().join(format!("strata-chartok-{}.json", uuid::Uuid::now_v7()));
        std::fs::write(&path, json).expect("write tokenizer json");
        let tc = HfTokenCounter::from_file(&path).expect("load char-level tokenizer");
        std::fs::remove_file(&path).expect("remove tokenizer json");
        tc
    }

    fn assert_invariants(chunks: &[Chunk], tc: &dyn TokenCounter, target: usize) {
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.ord, i as i32, "ords must be sequential");
            assert!(!c.text.trim().is_empty(), "chunk {i} is empty");
            assert_eq!(c.text, c.text.trim(), "chunk {i} is not trimmed");
            assert_eq!(
                c.token_count as usize,
                tc.count(&c.text),
                "chunk {i} token_count is stale"
            );
            assert!(
                c.token_count as usize <= target,
                "chunk {i} has {} > {target} tokens",
                c.token_count
            );
        }
    }

    fn sentences(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("S{i} a b c d.")).collect()
    }

    // ---------- WhitespaceCounter ----------

    #[test]
    fn whitespace_counter_counts_words() {
        assert_eq!(WhitespaceCounter.count(""), 0);
        assert_eq!(WhitespaceCounter.count("  a b\t\nc  "), 3);
    }

    #[test]
    fn whitespace_split_preserves_text_and_respects_max() {
        assert_eq!(
            WhitespaceCounter.split_by_tokens("a b c d e", 2),
            vec!["a b ", "c d ", "e"]
        );
        assert_eq!(
            WhitespaceCounter.split_by_tokens("  a b", 1),
            vec!["  a ", "b"]
        );
        assert_eq!(WhitespaceCounter.split_by_tokens("   ", 3), vec!["   "]);
        assert!(WhitespaceCounter.split_by_tokens("", 3).is_empty());
        // max = 0 is treated as 1, never an infinite loop.
        assert_eq!(WhitespaceCounter.split_by_tokens("a b", 0), vec!["a ", "b"]);
        let text = "one two  three\nfour\tfive six seven";
        let pieces = WhitespaceCounter.split_by_tokens(text, 3);
        assert_eq!(pieces.concat(), text);
        assert!(pieces.iter().all(|p| WhitespaceCounter.count(p) <= 3));
    }

    // ---------- HfTokenCounter ----------

    #[test]
    fn hf_counter_loads_from_file_and_counts_without_special_tokens() {
        let tc = char_level_hf();
        assert_eq!(tc.count(""), 0);
        assert_eq!(tc.count("abc"), 3);
        assert_eq!(tc.count("漢字 ab"), 5);
    }

    #[test]
    fn hf_counter_splits_on_token_byte_offsets() {
        let tc = char_level_hf();
        assert_eq!(
            tc.split_by_tokens("漢字かな漢字", 4),
            vec!["漢字かな", "漢字"]
        );
        let text = "漢字かなカナ".repeat(50);
        let pieces = tc.split_by_tokens(&text, 7);
        assert_eq!(pieces.concat(), text);
        assert!(pieces.iter().all(|p| tc.count(p) <= 7 && !p.is_empty()));
    }

    #[test]
    fn hf_from_file_missing_path_is_model_error() {
        let err = HfTokenCounter::from_file(std::path::Path::new("/nonexistent/tokenizer.json"))
            .err()
            .expect("missing file must fail");
        assert!(matches!(err, StrataError::Model(_)), "{err:?}");
    }

    // ---------- chunk_text basics ----------

    #[test]
    fn empty_or_whitespace_text_gives_no_chunks() {
        assert!(chunk_text("", &WhitespaceCounter, 32, 4).is_empty());
        assert!(chunk_text(" \n\n \t", &WhitespaceCounter, 32, 4).is_empty());
    }

    #[test]
    fn short_text_is_one_chunk_equal_to_input() {
        let text = "para one text.\n\npara two text.";
        let chunks = chunk_text(text, &WhitespaceCounter, 32, 4);
        assert_eq!(
            chunks,
            vec![Chunk {
                ord: 0,
                text: text.to_string(),
                token_count: 6,
                page_start: None,
                page_end: None,
            }]
        );
    }

    #[test]
    fn paragraphs_are_preferred_boundaries() {
        let text = "alpha beta gamma.\n\ndelta epsilon zeta.\n\neta theta iota.";
        let chunks = chunk_text(text, &WhitespaceCounter, 4, 3);
        assert_invariants(&chunks, &WhitespaceCounter, 4);
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        // Each paragraph is 3 tokens; two do not fit in 4; overlap never repeats a
        // chunk's first (and only) unit.
        assert_eq!(
            texts,
            vec![
                "alpha beta gamma.",
                "delta epsilon zeta.",
                "eta theta iota."
            ]
        );
    }

    #[test]
    fn sentences_are_packed_greedily_with_unit_overlap() {
        let s = sentences(8);
        let text = s.join(" ");
        let chunks = chunk_text(&text, &WhitespaceCounter, 12, 5);
        assert_invariants(&chunks, &WhitespaceCounter, 12);
        assert_eq!(chunks.len(), 7);
        for (k, c) in chunks.iter().enumerate() {
            assert_eq!(c.text, format!("{} {}", s[k], s[k + 1]), "chunk {k}");
            assert_eq!(c.token_count, 10);
        }
        // Overlap: each chunk starts with the previous chunk's last sentence.
        for w in chunks.windows(2) {
            let last_of_prev = w[0]
                .text
                .rsplit_once(" S")
                .map(|(_, t)| format!("S{t}"))
                .unwrap();
            assert!(
                w[1].text.starts_with(&last_of_prev),
                "{:?} -> {:?}",
                w[0].text,
                w[1].text
            );
        }
    }

    #[test]
    fn zero_overlap_partitions_the_units() {
        let s = sentences(8);
        let text = s.join(" ");
        let chunks = chunk_text(&text, &WhitespaceCounter, 12, 0);
        assert_invariants(&chunks, &WhitespaceCounter, 12);
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        assert_eq!(
            texts,
            vec![
                format!("{} {}", s[0], s[1]),
                format!("{} {}", s[2], s[3]),
                format!("{} {}", s[4], s[5]),
                format!("{} {}", s[6], s[7]),
            ]
        );
    }

    #[test]
    fn overlap_is_skipped_when_it_would_crowd_out_the_next_unit() {
        // Units: 2 + 2 + 5 tokens, target 6, overlap 2. Chunk 0 = first two sentences (4).
        // Overlapping the 2-token sentence would make chunk 1 = 2 + 5 = 7 > 6, so no
        // overlap is taken.
        let text = "Aa bb. Cc dd. Ee ff gg hh ii.";
        let chunks = chunk_text(text, &WhitespaceCounter, 6, 2);
        assert_invariants(&chunks, &WhitespaceCounter, 6);
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["Aa bb. Cc dd.", "Ee ff gg hh ii."]);
        // With room for it (target 7), the 2-token sentence IS repeated.
        let chunks = chunk_text(text, &WhitespaceCounter, 7, 2);
        assert_invariants(&chunks, &WhitespaceCounter, 7);
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["Aa bb. Cc dd.", "Cc dd. Ee ff gg hh ii."]);
    }

    #[test]
    fn chunks_are_slices_of_the_input() {
        let text = "First para has two sentences. Here is the second.\n\nSecond para is short.";
        let chunks = chunk_text(text, &WhitespaceCounter, 6, 2);
        assert_invariants(&chunks, &WhitespaceCounter, 6);
        assert!(chunks.iter().all(|c| text.contains(&c.text)));
    }

    // ---------- Review Focus #4: one "sentence" longer than the target ----------

    #[test]
    fn overlong_unpunctuated_sentence_splits_at_token_boundaries() {
        let text = "x".repeat(5000);
        let chunks = chunk_text(&text, &CharCounter, 64, 8);
        assert_invariants(&chunks, &CharCounter, 64);
        assert_eq!(chunks.len(), 79); // ceil(5000 / 64)
        assert!(chunks[..78].iter().all(|c| c.token_count == 64));
        assert_eq!(chunks[78].token_count, 8);
        // Pieces bigger than the overlap are never repeated, so they tile the input.
        let joined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn overlong_cjk_without_spaces_splits_within_target() {
        let text = "漢字かなカナ".repeat(500); // 3000 chars, no spaces, no sentence end
        let chunks = chunk_text(&text, &CharCounter, 100, 10);
        assert_invariants(&chunks, &CharCounter, 100);
        assert_eq!(chunks.len(), 30);
        let joined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn overlong_url_list_splits_within_target() {
        let text = (0..200)
            .map(|i| format!("https://example.test/path/{i}?q=v{i}"))
            .collect::<Vec<_>>()
            .join(",");
        assert!(text.len() > 5000);
        let chunks = chunk_text(&text, &CharCounter, 384, 48);
        assert_invariants(&chunks, &CharCounter, 384);
        let total: usize = chunks.iter().map(|c| c.text.chars().count()).sum();
        assert!(total >= text.chars().count(), "text was dropped");
    }

    #[test]
    fn overlong_minified_word_run_splits_with_whitespace_counter() {
        let text = ["tok"; 1000].join(" ");
        let chunks = chunk_text(&text, &WhitespaceCounter, 32, 4);
        assert_invariants(&chunks, &WhitespaceCounter, 32);
        let words: usize = chunks
            .iter()
            .map(|c| WhitespaceCounter.count(&c.text))
            .sum();
        assert_eq!(words, 1000);
    }

    #[test]
    fn overlong_sentence_with_hf_tokenizer_stays_within_target() {
        let tc = char_level_hf();
        let text = "漢字かなカナ".repeat(500);
        let chunks = chunk_text(&text, &tc, 100, 10);
        assert_invariants(&chunks, &tc, 100);
        assert_eq!(chunks.len(), 30);
    }

    #[test]
    fn non_additive_prefix_counter_re_splits_pieces() {
        // 100 words, no punctuation; each separately-counted piece costs +1 token.
        let words: Vec<String> = (0..100).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");
        let chunks = chunk_text(&text, &PrefixCounter, 10, 0);
        assert_invariants(&chunks, &PrefixCounter, 10);
        let seen: Vec<String> = chunks
            .iter()
            .flat_map(|c| {
                c.text
                    .split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(seen, words, "every word exactly once, in order");
    }

    #[test]
    fn exact_recount_shrinks_chunks_when_estimate_undercounts() {
        let s = sentences(6);
        let text = s.join(" ");
        // Each sentence is 5 tokens; two joined = 10 + 3 seam penalty = 13 > 12.
        let chunks = chunk_text(&text, &SeamPenaltyCounter, 12, 0);
        assert_invariants(&chunks, &SeamPenaltyCounter, 12);
        assert_eq!(chunks.len(), 6);
    }

    #[test]
    fn target_zero_is_treated_as_one() {
        let chunks = chunk_text("a b c", &WhitespaceCounter, 0, 0);
        assert_invariants(&chunks, &WhitespaceCounter, 1);
        assert_eq!(chunks.len(), 3);
    }

    // ---------- real bge-m3 tokenizer (needs the model cache from Task 7) ----------

    /// Run: `STRATA_TOKENIZER_JSON=$(ls ~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*/tokenizer.json | head -1) cargo test --lib -- --ignored chunk_real_tokenizer`
    #[test]
    #[ignore = "needs STRATA_TOKENIZER_JSON pointing at the bge-m3 tokenizer.json"]
    fn chunk_real_tokenizer_overlong_inputs() {
        let path = std::env::var("STRATA_TOKENIZER_JSON")
            .expect("set STRATA_TOKENIZER_JSON to bge-m3 tokenizer.json");
        let tc = HfTokenCounter::from_file(std::path::Path::new(&path)).unwrap();
        assert!(tc.count("Hello world") >= 2);
        let minified = r#"{"k":"v","n":[1,2,3],"s":"abcdefghij"}"#.repeat(400);
        let cjk = "漢字かなカナ日本語の文章".repeat(400);
        let urls = (0..300)
            .map(|i| format!("https://example.test/{i}"))
            .collect::<Vec<_>>()
            .join(",");
        for text in [minified, cjk, urls] {
            let chunks = chunk_text(&text, &tc, 384, 48);
            assert!(chunks.len() > 1);
            assert_invariants(&chunks, &tc, 384);
        }
    }

    // ---------- structure (headings) ----------

    fn texts(chunks: &[Chunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    #[test]
    fn chunks_do_not_cross_top_level_sections() {
        // A bundle: title, page anchor glued to the first heading, two `##` sections. Everything
        // fits one chunk by budget, and overlap is generous, yet each section is its own chunk.
        let text = "# survey\n\n<!-- page:1 -->\n## Alpha\n\nalpha one.\n\nalpha two.\n\n## Beta\n\nbeta one.";
        let chunks = chunk_text(text, &WhitespaceCounter, 100, 50);
        assert_invariants(&chunks, &WhitespaceCounter, 100);
        assert_eq!(
            texts(&chunks),
            vec![
                "# survey\n\n<!-- page:1 -->\n## Alpha\n\nalpha one.\n\nalpha two.",
                "## Beta\n\nbeta one.",
            ]
        );
    }

    #[test]
    fn overlap_never_reaches_into_the_previous_section() {
        // Section Alpha needs two chunks (target 4): the second may overlap the first, but
        // Beta's chunk must start at its own heading.
        let text = "## Alpha\n\naa bb.\n\ncc dd.\n\n## Beta\n\nee ff.";
        let chunks = chunk_text(text, &WhitespaceCounter, 4, 2);
        assert_invariants(&chunks, &WhitespaceCounter, 4);
        let last = chunks.last().unwrap();
        assert_eq!(last.text, "## Beta\n\nee ff.");
        assert!(
            chunks[..chunks.len() - 1]
                .iter()
                .all(|c| !c.text.contains("Beta")),
            "{:?}",
            texts(&chunks)
        );
    }

    #[test]
    fn top_level_is_the_shallowest_heading_level() {
        // `#` chapters with `##` subsections: only chapters split.
        let text = "# One\n\nx.\n\n## Sub\n\ny.\n\n# Two\n\nz.";
        let chunks = chunk_text(text, &WhitespaceCounter, 100, 0);
        assert_eq!(
            texts(&chunks),
            vec!["# One\n\nx.\n\n## Sub\n\ny.", "# Two\n\nz."]
        );
    }

    #[test]
    fn a_lone_leading_title_does_not_set_the_top_level() {
        // One `#` title over `##` sections (Bundle's `# {title}` shape): the sections split.
        let text = "# Title\n\nintro.\n\n## A\n\na.\n\n## B\n\nb.";
        let chunks = chunk_text(text, &WhitespaceCounter, 100, 0);
        assert_eq!(
            texts(&chunks),
            vec!["# Title\n\nintro.", "## A\n\na.", "## B\n\nb."]
        );
    }

    #[test]
    fn a_chunk_does_not_end_on_a_heading() {
        // Greedy packing by budget would end chunk 0 on `### s1` (2+2+2 = 6); the heading
        // moves to the chunk holding its body instead.
        let text = "## A\n\naa bb.\n\n### s1\n\ncc.\n\n### s2\n\ndd ee ff.\n\n## B\n\ngg.";
        let chunks = chunk_text(text, &WhitespaceCounter, 6, 0);
        assert_invariants(&chunks, &WhitespaceCounter, 6);
        assert_eq!(
            texts(&chunks),
            vec![
                "## A\n\naa bb.",
                "### s1\n\ncc.",
                "### s2\n\ndd ee ff.",
                "## B\n\ngg.",
            ]
        );
    }

    #[test]
    fn headings_inside_code_fences_are_text() {
        let text = "## A\n\nx.\n\n```\n## not a heading\n```\n\ny.\n\n## B\n\nz.";
        let chunks = chunk_text(text, &WhitespaceCounter, 100, 0);
        assert_eq!(
            texts(&chunks),
            vec![
                "## A\n\nx.\n\n```\n## not a heading\n```\n\ny.",
                "## B\n\nz."
            ]
        );
    }

    #[test]
    fn hashes_without_a_space_are_not_headings() {
        let text = "#hashtag one.\n\n#2 two.\n\n####### seven.";
        let chunks = chunk_text(text, &WhitespaceCounter, 100, 0);
        assert_eq!(chunks.len(), 1);
    }

    fn pages(chunks: &[Chunk]) -> Vec<(Option<i32>, Option<i32>)> {
        chunks.iter().map(|c| (c.page_start, c.page_end)).collect()
    }

    #[test]
    fn chunks_carry_the_pages_they_span() {
        // Title before any anchor, a section spanning pages 1-2, a section on page 3.
        let text = "# survey\n\n<!-- page:1 -->\n## Alpha\n\nalpha one.\n\n<!-- page:2 -->\nalpha two.\n\n<!-- page:3 -->\n## Beta\n\nbeta one.";
        let chunks = chunk_text(text, &WhitespaceCounter, 100, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(pages(&chunks), vec![(Some(1), Some(2)), (Some(3), Some(3))]);
    }

    #[test]
    fn a_chunk_starting_mid_page_inherits_that_page() {
        let text = "<!-- page:4 -->\none two three.\n\nfour five six.\n\nseven eight nine.";
        let chunks = chunk_text(text, &WhitespaceCounter, 4, 0);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert_eq!(
                (c.page_start, c.page_end),
                (Some(4), Some(4)),
                "{:?}",
                c.text
            );
        }
    }

    #[test]
    fn text_without_page_anchors_has_no_pages() {
        let chunks = chunk_text(
            "## A\n\nbody.\n\n<!-- note -->\nmore.",
            &WhitespaceCounter,
            100,
            0,
        );
        assert_eq!(pages(&chunks), vec![(None, None)]);
    }

    #[test]
    fn escaped_or_malformed_anchors_are_not_pages() {
        let text = "&lt;!-- page:9 -->\nx.\n\n<!-- page:abc -->\ny.\n\ninline <!-- page:7 --> z.";
        let chunks = chunk_text(text, &WhitespaceCounter, 100, 0);
        assert_eq!(pages(&chunks), vec![(None, None)]);
    }
}
