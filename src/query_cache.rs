//! Query-embedding cache (spec §8.3): an LRU of query vectors, so a repeated question (a retry, a
//! paging UI, the same FAQ asked twice) costs no embed call. Keyed by
//! `(tenant, provider owner, embed lane, embed model id, hash of the normalised text)`:
//! - the model id, because a vector only means something in its own model's space;
//! - the embed lane (`EmbedLane::id`): a tenant that re-registers its provider gets a new lane, so
//!   no vector of the old provider is served after the switch, even under the same model id;
//! - the tenant, so one tenant's traffic is never served (or timed) from another's questions, and a
//!   tenant's own registered provider never shares entries with another tenant's;
//! - the provider owner: the tenant whose providers embedded the question (the tenant itself, or
//!   the platform owner for platform-knowledge leaves, spec platform-knowledge §7), so a vector
//!   embedded (and billed) as the platform's never answers the tenant's own embed or vice versa;
//! - the sha256 of the text after [`normalise_query`] (NFC, whitespace runs collapsed), so every
//!   entry has the same size whatever the query length; that normalised text is
//!   also what gets embedded on a miss, so a hit returns exactly the vector a miss would compute.
//!
//! A hit makes no provider call, so it emits no cost event (`crate::metering`). Capacity
//! `STRATA_QUERY_EMBED_CACHE` (default 1024 entries, ~4 KiB each; 0 = off).

use std::num::NonZeroUsize;
use std::sync::Mutex;

use lru::LruCache;
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

use crate::error::Result;
use crate::inference::InferenceHandle;
use crate::tenant::Tenant;

/// (tenant, provider owner, embed lane id, embed model id, sha256 of the normalised text). The text
/// is hashed so an entry costs the same whatever the query length: memory is bounded by the
/// capacity alone.
type Key = (Tenant, Tenant, u64, String, [u8; 32]);

fn digest(normalised: &str) -> [u8; 32] {
    Sha256::digest(normalised.as_bytes()).into()
}

/// NFC, whitespace runs (newlines included) collapsed to one space, trimmed.
pub fn normalise_query(text: &str) -> String {
    let nfc: String = text.nfc().collect();
    nfc.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub struct QueryEmbedCache {
    /// `None` = off (capacity 0).
    lru: Option<Mutex<LruCache<Key, Vec<f32>>>>,
}

impl std::fmt::Debug for QueryEmbedCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryEmbedCache")
            .field("capacity", &self.capacity())
            .finish()
    }
}

impl QueryEmbedCache {
    pub fn new(capacity: usize) -> Self {
        QueryEmbedCache {
            lru: NonZeroUsize::new(capacity).map(|c| Mutex::new(LruCache::new(c))),
        }
    }

    pub fn capacity(&self) -> usize {
        self.lru.as_ref().map_or(0, |m| lock(m).cap().get())
    }

    pub fn len(&self) -> usize {
        self.lru.as_ref().map_or(0, |m| lock(m).len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, key: &Key) -> Option<Vec<f32>> {
        self.lru.as_ref().and_then(|m| lock(m).get(key).cloned())
    }

    fn put(&self, key: Key, v: Vec<f32>) {
        if let Some(m) = &self.lru {
            lock(m).put(key, v);
        }
    }

    /// The query vector of `text` (normalised): from the cache, else one query-lane embed call
    /// whose vector is cached. `true` = cache hit. `inf` is `tenant`'s own.
    pub async fn embed_one(
        &self,
        tenant: Tenant,
        inf: &InferenceHandle,
        text: &str,
    ) -> Result<(Vec<f32>, bool)> {
        self.embed_one_for(tenant, tenant, inf, text).await
    }

    /// `embed_one` for `tenant` with `owner`'s providers `inf` (the platform owner's for
    /// platform-knowledge leaves).
    pub async fn embed_one_for(
        &self,
        tenant: Tenant,
        owner: Tenant,
        inf: &InferenceHandle,
        text: &str,
    ) -> Result<(Vec<f32>, bool)> {
        let text = normalise_query(text);
        let key = (
            tenant,
            owner,
            inf.embed_lane_id()?,
            inf.embed_model_id()?.to_string(),
            digest(&text),
        );
        if let Some(v) = self.get(&key) {
            return Ok((v, true));
        }
        let v = inf.embed_one(text).await?;
        self.put(key, v.clone());
        Ok((v, false))
    }

    /// Query vectors for several texts (expansion variants), in order: hits from the cache, the
    /// misses in one batched query-lane call. `inf` is `tenant`'s own.
    pub async fn embed_queries(
        &self,
        tenant: Tenant,
        inf: &InferenceHandle,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_queries_for(tenant, tenant, inf, texts).await
    }

    /// `embed_queries` for `tenant` with `owner`'s providers `inf` (see `embed_one_for`).
    pub async fn embed_queries_for(
        &self,
        tenant: Tenant,
        owner: Tenant,
        inf: &InferenceHandle,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>> {
        let lane = inf.embed_lane_id()?;
        let model = inf.embed_model_id()?.to_string();
        let normalised: Vec<String> = texts.iter().map(|t| normalise_query(t)).collect();
        let keys: Vec<Key> = normalised
            .iter()
            .map(|t| (tenant, owner, lane, model.clone(), digest(t)))
            .collect();
        let mut out: Vec<Option<Vec<f32>>> = keys.iter().map(|k| self.get(k)).collect();
        let misses: Vec<usize> = (0..keys.len()).filter(|i| out[*i].is_none()).collect();
        if !misses.is_empty() {
            let vecs = inf
                .embed_queries(misses.iter().map(|i| normalised[*i].clone()).collect())
                .await?;
            for (i, v) in misses.into_iter().zip(vecs) {
                self.put(keys[i].clone(), v.clone());
                out[i] = Some(v);
            }
        }
        Ok(out.into_iter().flatten().collect())
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // a panic while holding the lock leaves a usable LRU
    m.lock().unwrap_or_else(|p| p.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalisation_collapses_whitespace_and_composes() {
        assert_eq!(normalise_query("  white \n\t whale  "), "white whale");
        // "é" decomposed (e + combining acute) and composed are one key
        assert_eq!(
            normalise_query("caf\u{0065}\u{0301}"),
            normalise_query("caf\u{00e9}")
        );
        assert_eq!(
            normalise_query("Whale"),
            "Whale",
            "case is kept: the model sees it"
        );
    }

    #[test]
    fn zero_capacity_is_off() {
        let c = QueryEmbedCache::new(0);
        assert_eq!(c.capacity(), 0);
        let t = Tenant::new(uuid::Uuid::from_u128(1)).unwrap();
        c.put((t, t, 0, "m".into(), digest("q")), vec![1.0]);
        assert!(c.is_empty());
    }

    #[test]
    fn least_recently_used_is_evicted() {
        let c = QueryEmbedCache::new(2);
        let t = Tenant::new(uuid::Uuid::from_u128(1)).unwrap();
        let k = |q: &str| (t, t, 0, "m".to_string(), digest(q));
        c.put(k("a"), vec![1.0]);
        c.put(k("b"), vec![2.0]);
        assert_eq!(c.get(&k("a")), Some(vec![1.0])); // a is now the most recent
        c.put(k("c"), vec![3.0]);
        assert_eq!(c.get(&k("b")), None);
        assert_eq!(c.get(&k("a")), Some(vec![1.0]));
        assert_eq!(c.len(), 2);
    }

    #[tokio::test]
    async fn keys_carry_the_tenant_the_lane_and_the_model() {
        let c = QueryEmbedCache::new(8);
        let (t1, t2) = (
            Tenant::new(uuid::Uuid::from_u128(1)).unwrap(),
            Tenant::new(uuid::Uuid::from_u128(2)).unwrap(),
        );
        let a = InferenceHandle::fake();
        let b = InferenceHandle::fake().with_embed_model_id("other@1024");
        assert!(!c.embed_one(t1, &a, "whale").await.unwrap().1);
        assert!(c.embed_one(t1, &a, " whale ").await.unwrap().1);
        assert!(
            !c.embed_one(t1, &b, "whale").await.unwrap().1,
            "another model"
        );
        assert!(
            !c.embed_one(t2, &a, "whale").await.unwrap().1,
            "another tenant"
        );
        assert_eq!(c.len(), 3);
        // the same lane on behalf of another provider owner (the platform's embed for t1) misses
        assert!(
            !c.embed_one_for(t1, t2, &a, "whale").await.unwrap().1,
            "another provider owner"
        );
        assert!(c.embed_one_for(t1, t2, &a, "whale").await.unwrap().1);
        assert_eq!(c.len(), 4);
        // the same lane (a clone of the handle) hits; a new lane under the same model id (a
        // re-registered provider) misses
        assert!(c.embed_one(t1, &a.clone(), "whale").await.unwrap().1);
        let a2 = InferenceHandle::fake();
        assert_eq!(a2.embed_model_id().unwrap(), a.embed_model_id().unwrap());
        assert!(
            !c.embed_one(t1, &a2, "whale").await.unwrap().1,
            "another lane"
        );
        assert_eq!(c.len(), 5);
        // a batch: one hit, one miss, in order
        let v = c
            .embed_queries(t1, &a, vec!["whale".into(), "ship".into()])
            .await
            .unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], c.embed_one(t1, &a, "whale").await.unwrap().0);
        assert_eq!(v[1], a.embed_one("ship".into()).await.unwrap());
        assert_eq!(c.len(), 6);
    }
}
