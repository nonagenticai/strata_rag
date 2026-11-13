//! `strata reembed` (`Engine::reembed_tree`): after an embed-provider switch, a tenant's tree is
//! refused with 409 `embed_model_mismatch` until every chunk, every store's `desc_vec` and the
//! content / profile rollups are recomputed with the new provider and the tree is tagged with its
//! model; then queries answer again.

mod common;

use common::{other, test_engine, tn};

use pgvector::Vector;
use serde_json::json;
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::{Engine, desc_text};
use strata_rag::error::StrataError;
use strata_rag::inference::{Embedder, FakeEmbedder, FakeReranker, InferenceHandle};
use strata_rag::providers::Providers;
use strata_rag::query::QueryRequest;
use strata_rag::vecmath::cosine;
use uuid::Uuid;

const MODEL_B: &str = "acme/reversed@1024";

/// "Model B": the fake embedder's vectors reversed. Same words still land close together, but
/// every vector differs from model A's, so a chunk that was not re-embedded is detectable.
struct Reversed;

impl Embedder for Reversed {
    fn embed(&mut self, texts: &[String]) -> strata_rag::Result<Vec<Vec<f32>>> {
        let mut v = FakeEmbedder.embed(texts)?;
        for x in &mut v {
            x.reverse();
        }
        Ok(v)
    }
}

fn model_b(text: &str) -> Vec<f32> {
    Reversed.embed(&[text.to_string()]).unwrap().remove(0)
}

fn switch_to_b(engine: &mut Engine) {
    engine.providers = Providers::fixed(
        InferenceHandle::in_process(Box::new(Reversed), Box::new(FakeReranker))
            .with_embed_model_id(MODEL_B),
    );
}

fn new_store(parent_id: Option<Uuid>, kind: StoreKind, name: &str) -> NewStore {
    NewStore {
        parent_id,
        kind,
        name: name.to_string(),
        description: format!("{name} holds documents about {name}"),
    }
}

fn query(text: &str) -> QueryRequest {
    serde_json::from_value(json!({"query": text, "route_mode": "embed", "min_score": -1.0}))
        .unwrap()
}

struct Tree {
    root: Uuid,
    deep: Uuid,
    sea: Uuid,
}

/// root router → leaf `sea`, router `deep` → leaf `abyss`; one document per leaf.
async fn tree(engine: &Engine) -> Tree {
    let root = engine
        .create_store(tn(), new_store(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    let sea = engine
        .create_store(tn(), new_store(Some(root), StoreKind::Leaf, "sea"))
        .await
        .unwrap()
        .id;
    let deep = engine
        .create_store(tn(), new_store(Some(root), StoreKind::Router, "deep"))
        .await
        .unwrap()
        .id;
    let abyss = engine
        .create_store(tn(), new_store(Some(deep), StoreKind::Leaf, "abyss"))
        .await
        .unwrap()
        .id;
    engine
        .ingest(
            tn(),
            sea,
            "sea.txt".into(),
            None,
            "The harpoon struck the white whale. The crew cheered on the deck of the ship.",
        )
        .await
        .unwrap();
    engine
        .ingest(
            tn(),
            abyss,
            "abyss.txt".into(),
            None,
            "Anglerfish hunt in the dark abyss with a glowing lure far below the surface.",
        )
        .await
        .unwrap();
    Tree { root, deep, sea }
}

async fn chunk_count(engine: &Engine) -> usize {
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM chunks")
        .fetch_one(&engine.pool)
        .await
        .unwrap();
    n as usize
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
}

#[tokio::test]
async fn a_switched_provider_is_refused_until_the_tree_is_reembedded() {
    let (mut engine, _db) = test_engine().await;
    let t = tree(&engine).await;
    let chunks = chunk_count(&engine).await;
    assert!(chunks >= 2, "both documents chunked");
    assert!(
        !engine
            .query(tn(), &query("white whale"))
            .await
            .unwrap()
            .results
            .is_empty()
    );

    switch_to_b(&mut engine);
    let err = engine.query(tn(), &query("white whale")).await.unwrap_err();
    assert_eq!((err.code(), err.status()), ("embed_model_mismatch", 409));

    let report = engine.reembed_tree(tn(), t.root, 1).await.unwrap();
    assert_eq!(report.stores, 4);
    assert_eq!(report.chunks, chunks);
    assert_eq!(report.embed_model, MODEL_B);

    // every store is tagged with model B and its desc_vec is model B's
    let stores: Vec<(String, String, String, Vector)> =
        sqlx::query_as("SELECT name, description, embed_model, desc_vec FROM stores")
            .fetch_all(&engine.pool)
            .await
            .unwrap();
    assert_eq!(stores.len(), 4);
    for (name, description, model, desc_vec) in &stores {
        assert_eq!(model, MODEL_B, "{name}");
        assert!(
            close(&desc_vec.to_vec(), &model_b(&desc_text(name, description))),
            "{name}: desc_vec re-embedded"
        );
    }
    // every chunk holds model B's vector of its own text
    let rows: Vec<(String, Vector)> = sqlx::query_as("SELECT text, embedding FROM chunks")
        .fetch_all(&engine.pool)
        .await
        .unwrap();
    assert_eq!(rows.len(), chunks);
    for (text, v) in &rows {
        assert!(close(&v.to_vec(), &model_b(text)), "chunk {text:?}");
    }
    // the rollups follow: a leaf's content_vec is the mean of its new chunk vectors, and the
    // routers above it are recomputed from their children
    let sea_content: Vector = sqlx::query_scalar("SELECT content_vec FROM stores WHERE id = $1")
        .bind(t.sea)
        .fetch_one(&engine.pool)
        .await
        .unwrap();
    let sea_chunks: Vec<Vector> =
        sqlx::query_scalar("SELECT embedding FROM chunks WHERE store_id = $1")
            .bind(t.sea)
            .fetch_all(&engine.pool)
            .await
            .unwrap();
    let mut mean = vec![0.0f32; strata_rag::vecmath::DIM];
    for c in &sea_chunks {
        for (m, x) in mean.iter_mut().zip(c.to_vec()) {
            *m += x;
        }
    }
    strata_rag::vecmath::l2_normalize(&mut mean);
    assert!(cosine(&sea_content.to_vec(), &mean) > 0.9999);
    for router in [t.root, t.deep] {
        let (content, profile): (Option<Vector>, Vector) =
            sqlx::query_as("SELECT content_vec, profile_vec FROM stores WHERE id = $1")
                .bind(router)
                .fetch_one(&engine.pool)
                .await
                .unwrap();
        let content = content.expect("router content_vec").to_vec();
        // model B's vectors are model A's reversed: the router's rollup must be in B's space
        let desc: Vector = sqlx::query_scalar("SELECT desc_vec FROM stores WHERE id = $1")
            .bind(router)
            .fetch_one(&engine.pool)
            .await
            .unwrap();
        let expected = strata_rag::vecmath::profile(
            &desc.to_vec(),
            Some(content.as_slice()),
            engine.cfg.profile_alpha,
        );
        assert!(close(&profile.to_vec(), &expected), "profile recomputed");
    }

    let resp = engine.query(tn(), &query("white whale")).await.unwrap();
    assert!(!resp.results.is_empty());
    assert!(
        resp.results[0].text.contains("whale"),
        "{:?}",
        resp.results[0].text
    );
    // and writes into the tree work again under model B
    engine
        .ingest(
            tn(),
            t.sea,
            "more.txt".into(),
            None,
            "A second whale surfaced near the ship.",
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn reembed_takes_the_tenant_root_only() {
    let (mut engine, _db) = test_engine().await;
    let t = tree(&engine).await;
    switch_to_b(&mut engine);
    // a subtree re-embedded under a model-A ancestor would mix two vector spaces in the rollups
    let err = engine.reembed_tree(tn(), t.deep, 8).await.unwrap_err();
    assert!(matches!(err, StrataError::InvalidInput(_)), "{err:?}");
    // another tenant's root is not found, exactly like a missing one
    let err = engine.reembed_tree(other(), t.root, 8).await.unwrap_err();
    assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");
    let err = engine.reembed_tree(tn(), t.root, 0).await.unwrap_err();
    assert!(matches!(err, StrataError::InvalidInput(_)), "{err:?}");
    // nothing was changed by the refusals
    let models: Vec<String> = sqlx::query_scalar("SELECT DISTINCT embed_model FROM stores")
        .fetch_all(&engine.pool)
        .await
        .unwrap();
    assert_eq!(models, vec!["BAAI/bge-m3@1024".to_string()]);
}

#[tokio::test]
async fn reembedding_with_the_same_model_is_a_noop_in_effect() {
    let (engine, _db) = test_engine().await;
    let t = tree(&engine).await;
    let before: Vec<(Uuid, Vector)> =
        sqlx::query_as("SELECT id, embedding FROM chunks ORDER BY id")
            .fetch_all(&engine.pool)
            .await
            .unwrap();
    let report = engine.reembed_tree(tn(), t.root, 3).await.unwrap();
    assert_eq!(report.chunks, before.len());
    let after: Vec<(Uuid, Vector)> = sqlx::query_as("SELECT id, embedding FROM chunks ORDER BY id")
        .fetch_all(&engine.pool)
        .await
        .unwrap();
    assert_eq!(before.len(), after.len());
    for ((a, va), (b, vb)) in before.iter().zip(&after) {
        assert_eq!(a, b);
        assert!(close(&va.to_vec(), &vb.to_vec()));
    }
    assert!(
        !engine
            .query(tn(), &query("anglerfish lure"))
            .await
            .unwrap()
            .results
            .is_empty()
    );
}

/// `strata reembed` runs in its own process: the serving process's routing cache must see the
/// re-embedded tree on its next query, without a restart. Two engines on one database stand in
/// for the server and the CLI (each has its own routing cache).
#[tokio::test]
async fn a_reembed_by_another_process_is_seen_by_the_server() {
    let (mut server, db) = test_engine().await;
    let t = tree(&server).await;
    switch_to_b(&mut server);
    // the server caches the model-A tree and refuses it
    let err = server.query(tn(), &query("white whale")).await.unwrap_err();
    assert_eq!(err.code(), "embed_model_mismatch");
    assert_eq!(server.cache.cached_tenants().await, vec![tn()]);

    let mut cli = Engine::new(
        db.pool.clone(),
        InferenceHandle::fake(),
        std::sync::Arc::new(strata_rag::text::WhitespaceCounter),
        common::test_config(),
    );
    switch_to_b(&mut cli);
    cli.reembed_tree(tn(), t.root, 8).await.unwrap();

    // the server never heard of it, and answers 200 on its next query
    let resp = server.query(tn(), &query("white whale")).await.unwrap();
    assert!(resp.results[0].text.contains("whale"));

    // any other store write by the other process is seen too: a new leaf is routed to
    let leaf = cli
        .create_store(tn(), new_store(Some(t.root), StoreKind::Leaf, "lake"))
        .await
        .unwrap()
        .id;
    cli.ingest(
        tn(),
        leaf,
        "lake.txt".into(),
        None,
        "Pike and perch swim in the cold lake.",
    )
    .await
    .unwrap();
    let tree = server.cache.get(&server.pool, tn()).await.unwrap();
    assert!(tree.nodes.contains_key(&leaf));
    let resp = server.query(tn(), &query("pike perch lake")).await.unwrap();
    assert!(
        resp.results.iter().any(|r| r.text.contains("Pike")),
        "{:?}",
        resp.results.iter().map(|r| &r.text).collect::<Vec<_>>()
    );
}

/// A tree far larger than one batch: chunks are embedded, staged and written `batch` at a time,
/// and still every chunk ends up with model B's vector of its own text, the leaves' running sums
/// (migration 0020) are rebuilt from the new vectors, and the run leaves no staged rows behind.
#[tokio::test]
async fn a_reembed_larger_than_one_batch_rewrites_every_chunk() {
    let (mut engine, _db) = test_engine().await;
    let t = tree(&engine).await;
    for i in 0..6 {
        let text = (0..120)
            .map(|w| format!("tide{i}w{w}{}", if w % 10 == 9 { "." } else { "" }))
            .collect::<Vec<_>>()
            .join(" ");
        engine
            .ingest(tn(), t.sea, format!("tide{i}.txt"), None, &text)
            .await
            .unwrap();
    }
    let chunks = chunk_count(&engine).await;
    let batch = 4;
    assert!(chunks > 5 * batch, "{chunks} chunks: several batches");

    switch_to_b(&mut engine);
    let report = engine.reembed_tree(tn(), t.root, batch).await.unwrap();
    assert_eq!(report.chunks, chunks);
    assert_eq!(report.stores, 4);

    let rows: Vec<(String, Vector)> = sqlx::query_as("SELECT text, embedding FROM chunks")
        .fetch_all(&engine.pool)
        .await
        .unwrap();
    assert_eq!(rows.len(), chunks);
    for (text, v) in &rows {
        assert!(close(&v.to_vec(), &model_b(text)), "chunk {text:?}");
    }
    let staged: i64 = sqlx::query_scalar("SELECT count(*) FROM reembed_staging")
        .fetch_one(&engine.pool)
        .await
        .unwrap();
    assert_eq!(staged, 0, "the run's staged vectors are removed");

    // the leaf's running sum is the sum of its new vectors, and its centroid their mean
    let (sum, n): (Option<Vec<f64>>, i64) =
        sqlx::query_as("SELECT content_sum, content_chunks FROM stores WHERE id = $1")
            .bind(t.sea)
            .fetch_one(&engine.pool)
            .await
            .unwrap();
    let sea: Vec<Vector> = sqlx::query_scalar("SELECT embedding FROM chunks WHERE store_id = $1")
        .bind(t.sea)
        .fetch_all(&engine.pool)
        .await
        .unwrap();
    assert_eq!(n, sea.len() as i64);
    let mut want = vec![0.0f64; strata_rag::vecmath::DIM];
    for c in &sea {
        for (w, x) in want.iter_mut().zip(c.to_vec()) {
            *w += f64::from(x);
        }
    }
    let sum = sum.expect("content_sum set");
    assert!(
        sum.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4),
        "content_sum rebuilt from model B's vectors"
    );
    let content: Vector = sqlx::query_scalar("SELECT content_vec FROM stores WHERE id = $1")
        .bind(t.sea)
        .fetch_one(&engine.pool)
        .await
        .unwrap();
    let mut mean: Vec<f32> = want.iter().map(|x| *x as f32).collect();
    strata_rag::vecmath::l2_normalize(&mut mean);
    assert!(cosine(&content.to_vec(), &mean) > 0.9999);

    let resp = engine.query(tn(), &query("tide3w42")).await.unwrap();
    assert!(!resp.results.is_empty());
}
