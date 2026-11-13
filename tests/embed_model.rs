//! Per-store embedding model guard: a store records the `"{model}@{dims}"` that made its vectors, and
//! query / ingest / re-describe refuse a store whose model differs from the configured one (409
//! `embed_model_mismatch`) instead of mixing vector spaces.

mod common;

use common::tn;

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::engine::Engine;
use strata_rag::error::StrataError;
use strata_rag::inference::InferenceHandle;
use strata_rag::providers::Providers;
use strata_rag::query::QueryRequest;
use tower::ServiceExt;
use uuid::Uuid;

fn new_store(parent_id: Option<Uuid>, kind: StoreKind, name: &str) -> NewStore {
    NewStore {
        parent_id,
        kind,
        name: name.to_string(),
        description: format!("{name} holds documents about {name}"),
    }
}

/// root router → one leaf, with one document ingested while the models still matched.
async fn tree(engine: &Engine) -> (Uuid, Uuid) {
    let root = engine
        .create_store(tn(), new_store(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    let leaf = engine
        .create_store(tn(), new_store(Some(root), StoreKind::Leaf, "sea"))
        .await
        .unwrap()
        .id;
    engine
        .ingest(
            tn(),
            leaf,
            "sea.txt".into(),
            None,
            "The harpoon struck the white whale.",
        )
        .await
        .unwrap();
    (root, leaf)
}

async fn stored_model(engine: &Engine, id: Uuid) -> String {
    sqlx::query_scalar("SELECT embed_model FROM stores WHERE id = $1")
        .bind(id)
        .fetch_one(&engine.pool)
        .await
        .unwrap()
}

/// Simulates a store embedded by another model (e.g. before a provider switch).
async fn set_model(engine: &Engine, id: Uuid, model: &str) {
    sqlx::query("UPDATE stores SET embed_model = $2 WHERE id = $1")
        .bind(id)
        .bind(model)
        .execute(&engine.pool)
        .await
        .unwrap();
    engine.cache.invalidate(tn()).await;
}

/// The fake providers, recorded under another model identity: what a provider switch looks like.
fn switch_model(engine: &mut Engine, id: &str) {
    engine.providers = Providers::fixed(InferenceHandle::fake().with_embed_model_id(id));
}

fn query(text: &str) -> QueryRequest {
    serde_json::from_value(json!({"query": text, "route_mode": "embed"})).unwrap()
}

fn assert_mismatch(err: StrataError, store: Uuid, store_model: &str) {
    match &err {
        StrataError::EmbedModelMismatch {
            store_id,
            store_model: got,
            configured,
        } => {
            assert_eq!(*store_id, store);
            assert_eq!(got, store_model);
            assert_eq!(configured, "BAAI/bge-m3@1024");
        }
        other => panic!("expected EmbedModelMismatch, got {other:?}"),
    }
    assert_eq!((err.code(), err.status()), ("embed_model_mismatch", 409));
}

#[tokio::test]
async fn a_new_store_records_the_configured_model() {
    let (mut engine, _db) = common::test_engine().await;
    let (root, leaf) = tree(&engine).await;
    assert_eq!(stored_model(&engine, root).await, "BAAI/bge-m3@1024");
    assert_eq!(stored_model(&engine, leaf).await, "BAAI/bge-m3@1024");

    // a fresh tree under another configured model records that model
    let (mut other, _db2) = common::test_engine().await;
    switch_model(&mut other, "acme/embed-v2@1024");
    let id = other
        .create_store(tn(), new_store(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    assert_eq!(stored_model(&other, id).await, "acme/embed-v2@1024");
    // ... and a child under a parent of the previous model is refused
    switch_model(&mut engine, "acme/embed-v2@1024");
    let err = engine
        .create_store(tn(), new_store(Some(root), StoreKind::Leaf, "stars"))
        .await
        .unwrap_err();
    assert_eq!(err.code(), "embed_model_mismatch", "{err:?}");
}

#[tokio::test]
async fn the_column_has_no_default_so_a_writer_must_state_the_model() {
    let (engine, _db) = common::test_engine().await;
    let err = sqlx::query(
        "INSERT INTO stores (id, tenant, parent_id, kind, name, description, desc_vec, \
                             profile_vec) \
         VALUES ($1, $2, NULL, 'router', 'r', 'd', \
                 array_fill(0.1::real, ARRAY[1024])::vector, \
                 array_fill(0.1::real, ARRAY[1024])::vector)",
    )
    .bind(Uuid::now_v7())
    .bind(tn().id())
    .execute(&engine.pool)
    .await
    .unwrap_err();
    let db = err.as_database_error().unwrap();
    assert_eq!(db.code().unwrap(), "23502", "not-null violation: {err}");
    // the missing value is the model, not the (given) tenant
    assert!(db.message().contains("embed_model"), "{err}");
}

#[tokio::test]
async fn query_refuses_a_tree_holding_a_store_of_another_model() {
    let (engine, _db) = common::test_engine().await;
    let (root, leaf) = tree(&engine).await;
    let ok = engine.query(tn(), &query("whale harpoon")).await.unwrap();
    assert!(!ok.results.is_empty());

    set_model(&engine, leaf, "old/embedder@1024").await;
    let err = engine
        .query(tn(), &query("whale harpoon"))
        .await
        .unwrap_err();
    assert_mismatch(err, leaf, "old/embedder@1024");

    // the root itself mismatching is caught the same way
    set_model(&engine, leaf, "BAAI/bge-m3@1024").await;
    set_model(&engine, root, "old/embedder@768").await;
    let err = engine
        .query(tn(), &query("whale harpoon"))
        .await
        .unwrap_err();
    assert_mismatch(err, root, "old/embedder@768");
}

#[tokio::test]
async fn ingest_and_redescribe_refuse_a_store_of_another_model() {
    let (engine, _db) = common::test_engine().await;
    let (_root, leaf) = tree(&engine).await;
    set_model(&engine, leaf, "old/embedder@1024").await;

    let err = engine
        .ingest(tn(), leaf, "more.txt".into(), None, "Another whale story.")
        .await
        .unwrap_err();
    assert_mismatch(err, leaf, "old/embedder@1024");
    let docs: i64 = sqlx::query_scalar("SELECT count(*) FROM documents WHERE store_id = $1")
        .bind(leaf)
        .fetch_one(&engine.pool)
        .await
        .unwrap();
    assert_eq!(docs, 1, "nothing written");

    let err = engine
        .update_store(tn(), leaf, None, Some("new description".into()))
        .await
        .unwrap_err();
    assert_mismatch(err, leaf, "old/embedder@1024");
}

async fn call(app: &axum::Router, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(common::with_tenant(req)).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn the_api_answers_409_embed_model_mismatch() {
    let (engine, _db) = common::test_engine().await;
    let (_root, leaf) = tree(&engine).await;
    set_model(&engine, leaf, "old/embedder@1024").await;
    let app = strata_rag::api::router(Arc::new(engine));

    let (s, v) = call(&app, "POST", "/api/query", json!({"query": "whale"})).await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "embed_model_mismatch", "{v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("old/embedder@1024"),
        "{v}"
    );

    let (s, v) = call(
        &app,
        "POST",
        &format!("/api/stores/{leaf}/documents"),
        json!({"source": "x.txt", "text": "Another whale story."}),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "embed_model_mismatch", "{v}");
}

#[tokio::test]
async fn a_new_store_under_a_router_of_another_model_is_refused() {
    let (engine, _db) = common::test_engine().await;
    let (root, _leaf) = tree(&engine).await;
    let mid = engine
        .create_store(tn(), new_store(Some(root), StoreKind::Router, "mid"))
        .await
        .unwrap()
        .id;
    // the grandparent mismatches: the new store's desc_vec would roll up into it
    set_model(&engine, root, "old/embedder@1024").await;
    let err = engine
        .create_store(tn(), new_store(Some(mid), StoreKind::Leaf, "deep"))
        .await
        .unwrap_err();
    assert_mismatch(err, root, "old/embedder@1024");
    // the direct parent mismatching is refused the same way, and through the API as 409
    set_model(&engine, root, "BAAI/bge-m3@1024").await;
    set_model(&engine, mid, "old/embedder@1024").await;
    let app = strata_rag::api::router(Arc::new(engine));
    let (s, v) = call(
        &app,
        "POST",
        "/api/stores",
        json!({"parent_id": mid, "kind": "leaf", "name": "deep", "description": "d"}),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT, "{v}");
    assert_eq!(v["error"]["code"], "embed_model_mismatch", "{v}");
}

#[tokio::test]
async fn ingest_into_a_leaf_whose_ancestor_has_another_model_is_refused() {
    let (engine, _db) = common::test_engine().await;
    let (root, leaf) = tree(&engine).await;
    set_model(&engine, root, "old/embedder@1024").await;
    let err = engine
        .ingest(tn(), leaf, "more.txt".into(), None, "Another whale story.")
        .await
        .unwrap_err();
    assert_mismatch(err, root, "old/embedder@1024");
    let err = engine
        .update_store(tn(), leaf, None, Some("new description".into()))
        .await
        .unwrap_err();
    assert_mismatch(err, root, "old/embedder@1024");
}

/// A provider's `model_id` is the store identity: the same weights served under another API name
/// (identity `bge-m3@1024` by default) keep matching stores recorded as `BAAI/bge-m3@1024` once
/// `model_id` says so.
#[tokio::test]
async fn the_model_id_override_keeps_existing_stores_matching_under_a_new_served_name() {
    let (mut engine, _db) = common::test_engine().await;
    let (_root, leaf) = tree(&engine).await;

    switch_model(&mut engine, "bge-m3@1024");
    let err = engine
        .query(tn(), &query("whale harpoon"))
        .await
        .unwrap_err();
    assert_eq!(err.code(), "embed_model_mismatch", "{err:?}");

    switch_model(&mut engine, "BAAI/bge-m3@1024");
    assert!(
        !engine
            .query(tn(), &query("whale harpoon"))
            .await
            .unwrap()
            .results
            .is_empty()
    );
    engine
        .ingest(tn(), leaf, "more.txt".into(), None, "Another whale story.")
        .await
        .unwrap();
    let id = engine
        .create_store(tn(), new_store(Some(_root), StoreKind::Leaf, "stars"))
        .await
        .unwrap()
        .id;
    assert_eq!(stored_model(&engine, id).await, "BAAI/bge-m3@1024");
}
