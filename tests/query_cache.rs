//! Query-embedding cache (`strata_rag::query_cache`): a repeated question makes no embed call and
//! emits no cost event; the cache is per tenant and per model, and off at capacity 0.

mod common;

use common::{test_config, tn};

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::inference::{Embedder, FakeEmbedder, FakeReranker, InferenceHandle};
use strata_rag::metering::Metering;
use strata_rag::query::QueryRequest;
use strata_rag::text::WhitespaceCounter;

/// The fake embedder, counting the provider calls (one per embed request).
struct Counting(Arc<AtomicUsize>);

impl Embedder for Counting {
    fn embed(&mut self, texts: &[String]) -> strata_rag::Result<Vec<Vec<f32>>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        FakeEmbedder.embed(texts)
    }
}

async fn engine(cache: usize) -> (Engine, common::TestDb, Arc<AtomicUsize>) {
    let db = common::TestDb::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let mut cfg = test_config();
    cfg.query_embed_cache = cache;
    let engine = Engine::new(
        db.pool.clone(),
        InferenceHandle::in_process(Box::new(Counting(calls.clone())), Box::new(FakeReranker)),
        Arc::new(WhitespaceCounter),
        cfg,
    );
    let root = engine
        .create_store(
            tn(),
            NewStore {
                parent_id: None,
                kind: StoreKind::Router,
                name: "library".into(),
                description: "all books".into(),
            },
        )
        .await
        .unwrap()
        .id;
    let leaf = engine
        .create_store(
            tn(),
            NewStore {
                parent_id: Some(root),
                kind: StoreKind::Leaf,
                name: "sea".into(),
                description: "whales and ships".into(),
            },
        )
        .await
        .unwrap()
        .id;
    engine
        .ingest(
            tn(),
            leaf,
            "whale.txt".into(),
            None,
            "The white whale surfaced beside the ship.",
        )
        .await
        .unwrap();
    calls.store(0, Ordering::SeqCst);
    (engine, db, calls)
}

fn query(text: &str) -> QueryRequest {
    serde_json::from_value(json!({"query": text, "route_mode": "embed", "min_score": 0.0})).unwrap()
}

#[tokio::test]
async fn a_repeated_query_makes_no_embed_call_and_emits_no_embed_event() {
    let (engine, _db, calls) = engine(1024).await;
    let (m, mut rx) = Metering::capture(Arc::new(WhitespaceCounter));
    let engine = engine.with_metering(m);

    let first = engine.query(tn(), &query("white whale")).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        first.providers.embed.as_ref().map(|p| p.cached),
        Some(false)
    );
    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e.get("service_key").unwrap().to_string());
    }
    assert_eq!(events, ["strata-rag-embed", "strata-rag-rerank"]);

    // same question, other whitespace: served from the cache
    let second = engine
        .query(tn(), &query("  white \n whale "))
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1, "no second embed call");
    assert_eq!(
        second.providers.embed.as_ref().map(|p| p.cached),
        Some(true)
    );
    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e.get("service_key").unwrap().to_string());
    }
    assert_eq!(events, ["strata-rag-rerank"], "a cache hit is not metered");
    assert_eq!(
        first.results.iter().map(|r| r.chunk_id).collect::<Vec<_>>(),
        second
            .results
            .iter()
            .map(|r| r.chunk_id)
            .collect::<Vec<_>>()
    );

    // another question is a miss
    engine.query(tn(), &query("the ship")).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(engine.query_cache.len(), 2);
}

#[tokio::test]
async fn capacity_zero_turns_the_cache_off() {
    let (engine, _db, calls) = engine(0).await;
    for _ in 0..2 {
        let r = engine.query(tn(), &query("white whale")).await.unwrap();
        assert_eq!(r.providers.embed.as_ref().map(|p| p.cached), Some(false));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(engine.query_cache.is_empty());
}
