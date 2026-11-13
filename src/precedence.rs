//! Tenant-over-platform precedence on a query's hits (spec platform-knowledge §6). Pure: the
//! shadow set comes from `db::documents::shadow_keys` (plus the request's `supersedes`), the hits
//! from search + rerank. Order of application, strongest first: shadowing (before rerank, by
//! `shadowed`), then the tenant margin and near-duplicate dedup (after rerank, by `apply`).

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// Which layer a hit comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layer {
    Tenant,
    Platform,
}

/// Word-5-shingle Jaccard at or above which a platform hit duplicates a tenant hit.
pub const DEFAULT_DEDUP_JACCARD: f32 = 0.8;

/// Shingle width for the near-duplicate test.
const SHINGLE_WORDS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecedenceOpts {
    /// Subtracted from every platform hit's score.
    pub platform_penalty: f32,
    /// Platform hits scoring below this (after the penalty) are dropped.
    pub platform_min_score: Option<f32>,
    /// See `DEFAULT_DEDUP_JACCARD`; above 1.0 turns dedup off.
    pub dedup_jaccard: f32,
}

/// A hit `apply` can rank.
pub trait Ranked {
    fn layer(&self) -> Layer;
    fn score(&self) -> f32;
    fn set_score(&mut self, s: f32);
    fn text(&self) -> &str;
}

/// Whether a platform document with `topic_keys` is shadowed by the caller's `shadow` set (any
/// overlap). Callers apply it to platform candidates only.
pub fn shadowed(topic_keys: &[String], shadow: &HashSet<String>) -> bool {
    !shadow.is_empty() && topic_keys.iter().any(|k| shadow.contains(k))
}

/// The tenant margin, the cut to `top_k` and the near-duplicate dedup (spec §6 steps 2-3): every
/// platform score drops by `platform_penalty`; platform hits below `platform_min_score` go; the
/// rest is re-sorted by score, descending (stable: equal scores keep their order) and cut to
/// `top_k`; then every platform hit whose text is a near-duplicate (`similarity` ≥
/// `dedup_jaccard`) of a KEPT tenant hit goes. "Kept" means in the returned top `top_k`: a tenant
/// duplicate ranked past the cut never removes a platform hit (neither would be shown). Dropping
/// a platform hit pulls the next hit into the cut, which is re-checked until nothing changes.
/// Tenant hits are never dropped or rescored.
pub fn apply<T: Ranked>(hits: Vec<T>, opts: &PrecedenceOpts, top_k: usize) -> Vec<T> {
    let mut hits: Vec<T> = hits
        .into_iter()
        .filter_map(|mut h| {
            if h.layer() == Layer::Platform {
                let s = h.score() - opts.platform_penalty;
                if let Some(min) = opts.platform_min_score
                    && s < min
                {
                    return None;
                }
                h.set_score(s);
            }
            Some(h)
        })
        .collect();
    hits.sort_by(|a, b| b.score().total_cmp(&a.score()));
    let shingles: Vec<Shingles> = hits.iter().map(|h| Shingles::of(h.text())).collect();
    let mut dropped = vec![false; hits.len()];
    // Monotone: drops only pull later hits into the cut, so tenant hits in the cut stay in it.
    loop {
        let window: Vec<usize> = (0..hits.len())
            .filter(|&i| !dropped[i])
            .take(top_k)
            .collect();
        let newly: Vec<usize> = window
            .iter()
            .copied()
            .filter(|&i| {
                hits[i].layer() == Layer::Platform
                    && window.iter().any(|&j| {
                        hits[j].layer() == Layer::Tenant
                            && shingles[i].similarity(&shingles[j]) >= opts.dedup_jaccard
                    })
            })
            .collect();
        if newly.is_empty() {
            let keep: std::collections::HashSet<usize> = window.into_iter().collect();
            return hits
                .into_iter()
                .enumerate()
                .filter(|(i, _)| keep.contains(i))
                .map(|(_, h)| h)
                .collect();
        }
        for i in newly {
            dropped[i] = true;
        }
    }
}

/// A text's lowercased whitespace tokens and, with at least `SHINGLE_WORDS` of them, its word
/// shingles.
struct Shingles {
    words: HashSet<String>,
    shingles: Option<HashSet<String>>,
}

impl Shingles {
    fn of(text: &str) -> Shingles {
        let tokens: Vec<String> = text.split_whitespace().map(str::to_lowercase).collect();
        let shingles = (tokens.len() >= SHINGLE_WORDS)
            .then(|| tokens.windows(SHINGLE_WORDS).map(|w| w.join(" ")).collect());
        Shingles {
            words: tokens.into_iter().collect(),
            shingles,
        }
    }

    /// Jaccard of the word-5-shingle sets; when either text has fewer than five words (no
    /// shingles), of the word sets instead. Two empty sets → 0.
    fn similarity(&self, other: &Shingles) -> f32 {
        match (&self.shingles, &other.shingles) {
            (Some(a), Some(b)) => jaccard(a, b),
            _ => jaccard(&self.words, &other.words),
        }
    }
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    let inter = a.intersection(b).count();
    let union = a.len() + b.len() - inter;
    if union == 0 {
        return 0.0;
    }
    inter as f32 / union as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Hit {
        id: &'static str,
        layer: Layer,
        score: f32,
        text: String,
    }

    impl Ranked for Hit {
        fn layer(&self) -> Layer {
            self.layer
        }
        fn score(&self) -> f32 {
            self.score
        }
        fn set_score(&mut self, s: f32) {
            self.score = s;
        }
        fn text(&self) -> &str {
            &self.text
        }
    }

    fn hit(id: &'static str, layer: Layer, score: f32, text: &str) -> Hit {
        Hit {
            id,
            layer,
            score,
            text: text.to_string(),
        }
    }

    fn opts(penalty: f32, min: Option<f32>) -> PrecedenceOpts {
        PrecedenceOpts {
            platform_penalty: penalty,
            platform_min_score: min,
            dedup_jaccard: DEFAULT_DEDUP_JACCARD,
        }
    }

    /// No cut.
    const ALL: usize = usize::MAX;

    fn ids(hits: &[Hit]) -> Vec<&'static str> {
        hits.iter().map(|h| h.id).collect()
    }

    #[test]
    fn layer_serialises_lowercase() {
        assert_eq!(serde_json::to_string(&Layer::Tenant).unwrap(), "\"tenant\"");
        assert_eq!(
            serde_json::from_str::<Layer>("\"platform\"").unwrap(),
            Layer::Platform
        );
    }

    #[test]
    fn shadowing_is_any_overlap() {
        let shadow: HashSet<String> = ["hr/leave".to_string()].into();
        assert!(shadowed(&["a".into(), "hr/leave".into()], &shadow));
        assert!(!shadowed(&["hr/leaves".into()], &shadow));
        assert!(!shadowed(&[], &shadow));
        assert!(!shadowed(&["hr/leave".into()], &HashSet::new()));
    }

    #[test]
    fn penalty_lowers_only_platform_scores_and_reorders() {
        let out = apply(
            vec![
                hit("p", Layer::Platform, 0.90, "alpha"),
                hit("t", Layer::Tenant, 0.87, "beta"),
            ],
            &opts(0.05, None),
            ALL,
        );
        assert_eq!(ids(&out), ["t", "p"], "tenant overtakes within the margin");
        assert!((out[0].score - 0.87).abs() < 1e-6, "tenant score untouched");
        assert!((out[1].score - 0.85).abs() < 1e-6);
        // beyond the margin the platform hit still wins
        let out = apply(
            vec![
                hit("t", Layer::Tenant, 0.5, "beta"),
                hit("p", Layer::Platform, 0.9, "alpha"),
            ],
            &opts(0.05, None),
            ALL,
        );
        assert_eq!(ids(&out), ["p", "t"]);
    }

    #[test]
    fn min_score_drops_platform_hits_after_the_penalty_but_never_tenant_hits() {
        let out = apply(
            vec![
                hit("p1", Layer::Platform, 0.62, "one"),
                hit("p2", Layer::Platform, 0.64, "two"),
                hit("t", Layer::Tenant, 0.10, "three"),
            ],
            &opts(0.05, Some(0.58)),
            ALL,
        );
        // p1: 0.57 < 0.58 dropped; p2: 0.59 kept; the low tenant hit stays
        assert_eq!(ids(&out), ["p2", "t"]);
    }

    #[test]
    fn sort_is_stable_for_equal_scores() {
        let out = apply(
            vec![
                hit("a", Layer::Tenant, 0.5, "a"),
                hit("p", Layer::Platform, 0.75, "p"),
                hit("b", Layer::Tenant, 0.5, "b"),
                hit("c", Layer::Tenant, 0.7, "c"),
            ],
            // exact in binary: p lands on 0.5 exactly
            &opts(0.25, None),
            ALL,
        );
        assert_eq!(ids(&out), ["c", "a", "p", "b"]);
    }

    #[test]
    fn near_duplicate_platform_hits_of_a_tenant_hit_are_dropped() {
        let base = "employees accrue twenty five days of paid annual leave every calendar year";
        let out = apply(
            vec![
                hit("p_dup", Layer::Platform, 0.95, &base.to_uppercase()),
                hit(
                    "p_other",
                    Layer::Platform,
                    0.9,
                    "the pension scheme matches five percent",
                ),
                hit("t", Layer::Tenant, 0.3, base),
            ],
            &opts(0.0, None),
            ALL,
        );
        assert_eq!(
            ids(&out),
            ["p_other", "t"],
            "case-insensitive duplicate dropped"
        );

        // last word changed (12 words, 8 shingles each, 7 shared): 7/9 ≈ 0.78 < 0.8: kept
        let edited = base.replace("year", "years");
        let out = apply(
            vec![
                hit("p", Layer::Platform, 0.9, &edited),
                hit("t", Layer::Tenant, 0.3, base),
            ],
            &opts(0.0, None),
            ALL,
        );
        assert_eq!(ids(&out), ["p", "t"]);
        // the same edit at a lower threshold: dropped
        let out = apply(
            vec![
                hit("p", Layer::Platform, 0.9, &edited),
                hit("t", Layer::Tenant, 0.3, base),
            ],
            &PrecedenceOpts {
                dedup_jaccard: 0.75,
                ..opts(0.0, None)
            },
            ALL,
        );
        assert_eq!(ids(&out), ["t"]);
    }

    #[test]
    fn dedup_never_drops_tenant_hits_and_needs_a_tenant_hit() {
        let text = "same text in both of these two hits here";
        let out = apply(
            vec![
                hit("t1", Layer::Tenant, 0.9, text),
                hit("t2", Layer::Tenant, 0.8, text),
            ],
            &opts(0.05, None),
            ALL,
        );
        assert_eq!(ids(&out), ["t1", "t2"]);
        let out = apply(
            vec![
                hit("p1", Layer::Platform, 0.9, text),
                hit("p2", Layer::Platform, 0.8, text),
            ],
            &opts(0.05, None),
            ALL,
        );
        assert_eq!(
            ids(&out),
            ["p1", "p2"],
            "platform vs platform is not deduplicated"
        );
    }

    #[test]
    fn short_texts_compare_word_sets() {
        // under five words: word-set Jaccard (order and case ignored)
        let out = apply(
            vec![
                hit("p", Layer::Platform, 0.9, "Leave Policy Annual"),
                hit("t", Layer::Tenant, 0.3, "annual leave policy"),
            ],
            &opts(0.0, None),
            ALL,
        );
        assert_eq!(ids(&out), ["t"]);
        let out = apply(
            vec![
                hit("p", Layer::Platform, 0.9, "annual leave"),
                hit("t", Layer::Tenant, 0.3, "annual leave policy"),
            ],
            &opts(0.0, None),
            ALL,
        );
        assert_eq!(ids(&out), ["p", "t"], "2/3 < 0.8");
        // empty texts never count as duplicates
        let out = apply(
            vec![
                hit("p", Layer::Platform, 0.9, "  "),
                hit("t", Layer::Tenant, 0.3, ""),
            ],
            &opts(0.0, None),
            ALL,
        );
        assert_eq!(ids(&out), ["p", "t"]);
    }

    #[test]
    fn dedup_only_against_tenant_hits_kept_in_the_top_k() {
        let base = "employees accrue twenty five days of paid annual leave every calendar year";
        let hits = || {
            vec![
                hit("p_dup", Layer::Platform, 0.9, base),
                hit("t_a", Layer::Tenant, 0.8, "the office opens at nine"),
                hit("t_dup", Layer::Tenant, 0.3, base),
            ]
        };
        // the tenant duplicate falls past top_k = 2: the platform hit stays (else neither shows)
        assert_eq!(ids(&apply(hits(), &opts(0.0, None), 2)), ["p_dup", "t_a"]);
        // within the cut it removes the platform hit, even ranked below it
        assert_eq!(ids(&apply(hits(), &opts(0.0, None), 3)), ["t_a", "t_dup"]);
        // a drop pulls the next hit in, which is re-checked against the cut
        let out = apply(
            vec![
                hit("p_dup", Layer::Platform, 0.9, base),
                hit("t_dup", Layer::Tenant, 0.8, base),
                hit("p_dup2", Layer::Platform, 0.7, &base.to_uppercase()),
                hit(
                    "p_other",
                    Layer::Platform,
                    0.6,
                    "the pension scheme matches five percent",
                ),
            ],
            &opts(0.0, None),
            2,
        );
        assert_eq!(ids(&out), ["t_dup", "p_other"]);
        assert!(apply(hits(), &opts(0.0, None), 0).is_empty());
    }

    #[test]
    fn shingle_jaccard_values() {
        let a = Shingles::of("a b c d e f");
        let b = Shingles::of("a b c d e g");
        // {abcde, bcdef} vs {abcde, bcdeg}: 1/3
        assert!((a.similarity(&b) - 1.0 / 3.0).abs() < 1e-6);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-6);
        // one side short: word sets {a..f} vs {a,b}: 2/6
        let short = Shingles::of("a b");
        assert!((a.similarity(&short) - 2.0 / 6.0).abs() < 1e-6);
    }
}
