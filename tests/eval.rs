//! eval + bench end to end against Postgres with the fake embedder/reranker:
//! tree loading is idempotent, evaluate reports all three modes, the ingest bench leaves no
//! temporary leaf behind, and the query bench yields one sample per query.

mod common;

use common::tn;

use std::path::PathBuf;

use strata_rag::bench::{run_ingest_bench, run_query_bench};
use strata_rag::db::stores::{NewStore, StoreKind};
use strata_rag::error::StrataError;
use strata_rag::eval::{evaluate, load_tree, parse_questions, read_tree};
use strata_rag::routing::RouteMode;
use uuid::Uuid;

use common::test_engine;

const TREE: &str = r#"{"corpus_dir":"corpus","root":{"name":"lib","kind":"router","description":"books","children":[
    {"name":"sea","kind":"leaf","description":"whales harpoons ships ocean","files":["sea.txt"]},
    {"name":"stars","kind":"leaf","description":"astronomy telescope planets orbit","files":["stars.txt"]}]}}"#;

const QUESTIONS: &str = concat!(
    r#"{"query":"harpoon whale ship","expected_document":"sea.txt","expected_leaf":"sea"}"#,
    "\n\n",
    r#"{"query":"telescope planet orbit","expected_document":"stars.txt","expected_leaf":"stars"}"#,
    "\n"
);

fn fixture_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("strata-eval-{}", Uuid::now_v7()));
    std::fs::create_dir_all(dir.join("corpus")).unwrap();
    std::fs::write(dir.join("tree.json"), TREE).unwrap();
    std::fs::write(
        dir.join("corpus/sea.txt"),
        "The harpoon struck the great whale and the ship heaved on the ocean.",
    )
    .unwrap();
    std::fs::write(
        dir.join("corpus/stars.txt"),
        "Through the telescope the planet traced its orbit across the night sky.",
    )
    .unwrap();
    dir
}

#[tokio::test]
async fn load_tree_is_idempotent_and_evaluate_reports_every_mode() {
    let (engine, _db) = test_engine().await;
    let dir = fixture_dir();

    let first = load_tree(&engine, tn(), &dir.join("tree.json"))
        .await
        .unwrap();
    assert_eq!(
        (first.created, first.existing, first.failed.len()),
        (2, 0, 0)
    );
    assert!(first.chunks >= 2);
    let second = load_tree(&engine, tn(), &dir.join("tree.json"))
        .await
        .unwrap();
    assert_eq!((second.created, second.existing), (0, 2));
    assert_eq!(
        engine.store_tree(tn()).await.unwrap().len(),
        3,
        "no duplicate stores"
    );

    let spec = read_tree(&dir.join("tree.json")).unwrap();
    let qs = parse_questions(QUESTIONS).unwrap();
    let reports = evaluate(&engine, tn(), &spec, &qs).await.unwrap();
    let modes: Vec<&str> = reports.iter().map(|r| r.mode).collect();
    assert_eq!(modes, ["embed", "hybrid", "flat"]);
    assert!(reports.iter().all(|r| r.n == 2));
    let flat = &reports[2];
    assert_eq!(flat.leaf_accuracy, None);
    assert!((flat.recall_at_10 - 1.0).abs() < 1e-9, "{flat:?}");
    assert!(reports[0].leaf_accuracy.is_some());

    let bad = parse_questions(
        r#"{"query":"q","expected_document":"missing.txt","expected_leaf":"nope"}"#,
    )
    .unwrap();
    let err = evaluate(&engine, tn(), &spec, &bad)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("leaf `nope`") && err.contains("missing.txt"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn benches_run_and_clean_up_after_themselves() {
    let (engine, _db) = test_engine().await;
    let dir = fixture_dir();
    load_tree(&engine, tn(), &dir.join("tree.json"))
        .await
        .unwrap();
    let before = engine.store_tree(tn()).await.unwrap().len();

    let queries = vec!["harpoon whale".to_string(), "telescope orbit".to_string()];
    let samples = run_query_bench(&engine, tn(), &queries, 5, RouteMode::Embed)
        .await
        .unwrap();
    assert_eq!(samples.len(), 5);

    let files = vec![dir.join("corpus/sea.txt"), dir.join("corpus/stars.txt")];
    let ingest = run_ingest_bench(&engine, tn(), &files).await.unwrap();
    assert_eq!(ingest.documents, 2);
    assert!(ingest.chunks >= 2);
    assert_eq!(
        engine.store_tree(tn()).await.unwrap().len(),
        before,
        "temp leaf deleted"
    );

    assert!(
        run_query_bench(&engine, tn(), &[], 5, RouteMode::Embed)
            .await
            .is_err()
    );
    // B1: zero samples would make every percentile meaningless
    assert!(matches!(
        run_query_bench(&engine, tn(), &queries, 0, RouteMode::Embed).await,
        Err(StrataError::InvalidInput(_))
    ));
    std::fs::remove_dir_all(&dir).ok();
}

async fn store_id(engine: &strata_rag::engine::Engine, name: &str) -> Uuid {
    engine
        .store_tree(tn())
        .await
        .unwrap()
        .into_iter()
        .find(|v| v.name == name)
        .unwrap()
        .id
}

async fn desc_vec(engine: &strata_rag::engine::Engine, id: Uuid) -> String {
    sqlx::query_scalar("SELECT desc_vec::text FROM stores WHERE id = $1")
        .bind(id)
        .fetch_one(&engine.pool)
        .await
        .unwrap()
}

#[tokio::test]
async fn load_tree_applies_a_changed_description_to_an_existing_store() {
    let (engine, _db) = test_engine().await;
    let dir = fixture_dir();
    load_tree(&engine, tn(), &dir.join("tree.json"))
        .await
        .unwrap();
    let sea = store_id(&engine, "sea").await;
    let before = desc_vec(&engine, sea).await;

    let edited = TREE.replace(
        "whales harpoons ships ocean",
        "sailing vessels and sea creatures",
    );
    std::fs::write(dir.join("tree.json"), edited).unwrap();
    let again = load_tree(&engine, tn(), &dir.join("tree.json"))
        .await
        .unwrap();
    assert_eq!((again.created, again.existing, again.updated), (0, 2, 1));
    let view = engine.get_store(tn(), sea).await.unwrap();
    assert_eq!(view.description, "sailing vessels and sea creatures");
    assert_ne!(desc_vec(&engine, sea).await, before, "desc_vec re-embedded");

    let unchanged = load_tree(&engine, tn(), &dir.join("tree.json"))
        .await
        .unwrap();
    assert_eq!(
        unchanged.updated, 0,
        "an unchanged description is not rewritten"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn flat_mode_searches_only_the_leaves_of_the_evaluated_tree() {
    let (engine, _db) = test_engine().await;
    let dir = fixture_dir();
    load_tree(&engine, tn(), &dir.join("tree.json"))
        .await
        .unwrap();
    // a leftover bench leaf full of decoys that match the query better than sea.txt
    let root = store_id(&engine, "lib").await;
    let stray = engine
        .create_store(
            tn(),
            NewStore {
                parent_id: Some(root),
                kind: StoreKind::Leaf,
                name: "bench-tmp-leftover".into(),
                description: "harpoon whale ship".into(),
            },
        )
        .await
        .unwrap();
    for i in 0..12 {
        engine
            .ingest(
                tn(),
                stray.id,
                format!("decoy-{i}.txt"),
                None,
                &format!("harpoon whale ship {i}"),
            )
            .await
            .unwrap();
    }
    let spec = read_tree(&dir.join("tree.json")).unwrap();
    let qs = parse_questions(QUESTIONS).unwrap();
    let reports = evaluate(&engine, tn(), &spec, &qs).await.unwrap();
    let flat = reports.iter().find(|r| r.mode == "flat").unwrap();
    assert!(
        (flat.recall_at_5 - 1.0).abs() < 1e-9,
        "flat searched a leaf outside tree.json: {flat:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn evaluate_requires_each_leaf_to_hold_its_own_files() {
    let (engine, _db) = test_engine().await;
    let dir = fixture_dir();
    // build the tree by hand: sea.txt lands in the WRONG leaf, so it is ingested somewhere
    let lib = engine
        .create_store(
            tn(),
            NewStore {
                parent_id: None,
                kind: StoreKind::Router,
                name: "lib".into(),
                description: "books".into(),
            },
        )
        .await
        .unwrap();
    let leaf = |name: &'static str| {
        let engine = &engine;
        async move {
            engine
                .create_store(
                    tn(),
                    NewStore {
                        parent_id: Some(lib.id),
                        kind: StoreKind::Leaf,
                        name: name.into(),
                        description: "d".into(),
                    },
                )
                .await
                .unwrap()
                .id
        }
    };
    let _sea = leaf("sea").await;
    let stars = leaf("stars").await;
    for f in ["sea.txt", "stars.txt"] {
        let text = std::fs::read_to_string(dir.join("corpus").join(f)).unwrap();
        engine
            .ingest(tn(), stars, f.into(), None, &text)
            .await
            .unwrap();
    }
    let spec = read_tree(&dir.join("tree.json")).unwrap();
    let qs = parse_questions(QUESTIONS).unwrap();
    let err = evaluate(&engine, tn(), &spec, &qs)
        .await
        .expect_err("sea.txt is not in leaf sea")
        .to_string();
    assert!(
        err.contains("sea.txt") && err.contains("leaf `sea`"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn ingest_bench_counts_only_documents_it_created() {
    let (engine, _db) = test_engine().await;
    let dir = fixture_dir();
    load_tree(&engine, tn(), &dir.join("tree.json"))
        .await
        .unwrap();
    // the same file twice: the second ingest dedups to Existing and embeds nothing
    let sea = dir.join("corpus/sea.txt");
    let solo = run_ingest_bench(&engine, tn(), std::slice::from_ref(&sea))
        .await
        .unwrap();
    let twice = run_ingest_bench(&engine, tn(), &[sea.clone(), sea])
        .await
        .unwrap();
    assert_eq!(twice.documents, 1, "{twice:?}");
    assert_eq!(twice.chunks, solo.chunks, "{twice:?}");
    std::fs::remove_dir_all(&dir).ok();
}
