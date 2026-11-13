//! Deterministic stand-ins for the real models, used by every test that needs inference.

use std::collections::HashSet;

use super::{Embedder, Reranker};
use crate::error::Result;
use crate::vecmath::{DIM, l2_normalize};

/// Bag-of-words: each lowercase alphanumeric word adds 1.0 at `fnv1a(word) % DIM`, then L2.
/// A text with no words maps to the unit vector e0, so every output has unit norm.
pub struct FakeEmbedder;

/// Score = |query words ∩ passage words| / |query words| (lowercase, unique). Empty query → 0.0.
pub struct FakeReranker;

fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
}

fn fnv1a(word: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in word.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn fake_vector(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    let mut any = false;
    for w in words(text) {
        let idx = (fnv1a(&w) % DIM as u64) as usize;
        v[idx] += 1.0;
        any = true;
    }
    if !any {
        v[0] = 1.0;
    }
    l2_normalize(&mut v);
    v
}

impl Embedder for FakeEmbedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| fake_vector(t)).collect())
    }
}

impl Reranker for FakeReranker {
    fn score(&mut self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
        let q: HashSet<String> = words(query).collect();
        Ok(passages
            .iter()
            .map(|p| {
                if q.is_empty() {
                    return 0.0;
                }
                let pw: HashSet<String> = words(p).collect();
                q.iter().filter(|w| pw.contains(*w)).count() as f32 / q.len() as f32
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vecmath::cosine;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn embedder_is_deterministic_unit_norm_and_dim() {
        let a = FakeEmbedder
            .embed(&s(&["The quick brown fox", ""]))
            .unwrap();
        let b = FakeEmbedder
            .embed(&s(&["the QUICK brown fox", ""]))
            .unwrap();
        assert_eq!(a, b, "case-insensitive and deterministic");
        for v in &a {
            assert_eq!(v.len(), DIM);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
        }
    }

    #[test]
    fn embedder_puts_shared_words_closer() {
        let v = FakeEmbedder
            .embed(&s(&[
                "postgres vector search",
                "vector search in postgres",
                "banana bread recipe",
            ]))
            .unwrap();
        assert!(cosine(&v[0], &v[1]) > cosine(&v[0], &v[2]));
    }

    #[test]
    fn reranker_scores_fraction_of_query_words() {
        let got = FakeReranker
            .score("Alpha beta", &s(&["alpha gamma", "ALPHA BETA", "delta"]))
            .unwrap();
        assert!((got[0] - 0.5).abs() < 1e-6);
        assert!((got[1] - 1.0).abs() < 1e-6);
        assert!(got[2].abs() < 1e-6);
    }

    #[test]
    fn reranker_empty_query_scores_zero() {
        let got = FakeReranker.score("  ", &s(&["anything"])).unwrap();
        assert_eq!(got, vec![0.0]);
    }
}
