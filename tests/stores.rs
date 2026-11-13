mod common;

use common::tn;

use common::test_engine;
use pgvector::Vector;
use sqlx::PgPool;
use strata_rag::db::stores::{self, NewStore, StoreKind, StoreRow};
use strata_rag::engine::Engine;
use strata_rag::error::StrataError;
use strata_rag::vecmath::{self, DIM};
use uuid::Uuid;

fn new_store(parent_id: Option<Uuid>, kind: StoreKind, name: &str) -> NewStore {
    NewStore {
        parent_id,
        kind,
        name: name.to_string(),
        description: format!("{name} holds documents about {name}"),
    }
}

fn unit(i: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[i] = 1.0;
    v
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < 1e-5, "index {i}: {a} != {e}");
    }
}

/// Inserts one document with the given (token_count, embedding) chunks into `leaf`
/// with raw SQL (db::documents arrives in Task 9).
async fn insert_raw_doc(pool: &PgPool, leaf: Uuid, chunks: &[(i32, Vec<f32>)]) {
    let doc = Uuid::now_v7();
    let total: i32 = chunks.iter().map(|c| c.0).sum();
    sqlx::query(
        "INSERT INTO documents (id, store_id, source, content_hash, token_count) \
         VALUES ($1, $2, 'raw', $3, $4)",
    )
    .bind(doc)
    .bind(leaf)
    .bind(doc.as_bytes().to_vec())
    .bind(total)
    .execute(pool)
    .await
    .unwrap();
    for (ord, (tokens, emb)) in chunks.iter().enumerate() {
        sqlx::query(
            "INSERT INTO chunks (id, document_id, store_id, ord, text, token_count, embedding) \
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(Uuid::now_v7())
        .bind(doc)
        .bind(leaf)
        .bind(ord as i32)
        .bind(format!("raw chunk {ord}"))
        .bind(*tokens)
        .bind(Vector::from(emb.clone()))
        .execute(pool)
        .await
        .unwrap();
    }
}

async fn recompute(engine: &Engine, id: Uuid) {
    let mut tx = engine.pool.begin().await.unwrap();
    stores::recompute_upwards(&mut tx, tn(), id, engine.cfg.profile_alpha)
        .await
        .unwrap();
    tx.commit().await.unwrap();
}

async fn row(engine: &Engine, id: Uuid) -> StoreRow {
    let mut conn = engine.pool.acquire().await.unwrap();
    stores::get_store(&mut conn, tn(), id).await.unwrap()
}

async fn count(pool: &PgPool, sql: &'static str) -> i64 {
    sqlx::query_scalar(sql).fetch_one(pool).await.unwrap()
}

struct Tree3 {
    root: Uuid,
    mid: Uuid,
    leaf1: Uuid,
    leaf2: Uuid,
    leaf3: Uuid,
}

/// root(router) ─┬─ mid(router) ─┬─ leaf1
///               │               └─ leaf2
///               └─ leaf3
async fn build_tree(engine: &Engine) -> Tree3 {
    let root = engine
        .create_store(tn(), new_store(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    let mid = engine
        .create_store(tn(), new_store(Some(root), StoreKind::Router, "mid"))
        .await
        .unwrap()
        .id;
    let leaf1 = engine
        .create_store(tn(), new_store(Some(mid), StoreKind::Leaf, "leaf1"))
        .await
        .unwrap()
        .id;
    let leaf2 = engine
        .create_store(tn(), new_store(Some(mid), StoreKind::Leaf, "leaf2"))
        .await
        .unwrap()
        .id;
    let leaf3 = engine
        .create_store(tn(), new_store(Some(root), StoreKind::Leaf, "leaf3"))
        .await
        .unwrap()
        .id;
    Tree3 {
        root,
        mid,
        leaf1,
        leaf2,
        leaf3,
    }
}

#[tokio::test]
async fn creates_three_layer_tree() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;

    let tree = engine.store_tree(tn()).await.unwrap();
    assert_eq!(tree.len(), 5);
    let root = tree.iter().find(|s| s.id == t.root).unwrap();
    assert_eq!(root.parent_id, None);
    assert_eq!(root.kind, StoreKind::Router);
    let mut root_children = root.children.clone();
    root_children.sort();
    let mut expected = vec![t.mid, t.leaf3];
    expected.sort();
    assert_eq!(root_children, expected);

    let mid = engine.get_store(tn(), t.mid).await.unwrap();
    assert_eq!(mid.parent_id, Some(t.root));
    let mut mid_children = mid.children.clone();
    mid_children.sort();
    let mut expected = vec![t.leaf1, t.leaf2];
    expected.sort();
    assert_eq!(mid_children, expected);

    let leaf = row(&engine, t.leaf1).await;
    assert_eq!(leaf.kind, StoreKind::Leaf);
    assert_eq!(leaf.desc_vec.len(), DIM);
    assert_eq!(leaf.content_vec, None);
    assert_close(&leaf.profile_vec, &leaf.desc_vec);
    let expected_desc = engine
        .inference(tn())
        .await
        .unwrap()
        .embed_one(strata_rag::engine::desc_text(
            "leaf1",
            "leaf1 holds documents about leaf1",
        ))
        .await
        .unwrap();
    assert_close(&leaf.desc_vec, &expected_desc);
}

/// A leaf `name` under `root` holding one whale document.
async fn fill_leaf(engine: &strata_rag::engine::Engine, root: Uuid, name: &str) {
    let leaf = engine
        .create_store(tn(), new_store(Some(root), StoreKind::Leaf, name))
        .await
        .unwrap()
        .id;
    engine
        .ingest(
            tn(),
            leaf,
            format!("{name}.txt"),
            None,
            "The harpoon struck the white whale.",
        )
        .await
        .unwrap();
}

/// A tenant may hold several roots (agent D83: a tree re-embedded in another model is a tree of
/// its own). Root names stay unique within the tenant. D90: with one root a query may omit
/// `root_id`; with two it must name one (409 `ambiguous_root`, never a guessed tree), and a
/// named root routes within its own tree only.
#[tokio::test]
async fn a_second_root_is_a_tree_of_its_own() {
    let (engine, _db) = test_engine().await;
    let q = |root_id: Option<Uuid>| -> strata_rag::query::QueryRequest {
        let mut v = serde_json::json!({"query": "white whale", "route_mode": "embed",
                                       "min_score": -1.0, "beam": 4});
        if let Some(r) = root_id {
            v["root_id"] = serde_json::json!(r);
        }
        serde_json::from_value(v).unwrap()
    };
    let paths = |resp: strata_rag::query::QueryResponse| -> Vec<String> {
        resp.results
            .into_iter()
            .map(|r| r.store_path[0].clone())
            .collect()
    };
    let first = engine
        .create_store(tn(), new_store(None, StoreKind::Router, "root"))
        .await
        .unwrap()
        .id;
    fill_leaf(&engine, first, "first-leaf").await;
    // one root, omitted: routes from it
    let only = paths(engine.query(tn(), &q(None)).await.unwrap());
    assert!(!only.is_empty());
    assert!(only.iter().all(|p| p == "root"), "{only:?}");

    let second = engine
        .create_store(tn(), new_store(None, StoreKind::Router, "other-root"))
        .await
        .unwrap()
        .id;
    let err = engine
        .create_store(tn(), new_store(None, StoreKind::Router, "root"))
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::DuplicateName(_)), "{err:?}");
    let roots: Vec<Uuid> = engine
        .store_tree(tn())
        .await
        .unwrap()
        .into_iter()
        .filter(|s| s.parent_id.is_none())
        .map(|s| s.id)
        .collect();
    assert_eq!(roots, vec![first, second]);
    fill_leaf(&engine, second, "second-leaf").await;

    // two roots, omitted: 409 ambiguous_root
    let err = engine.query(tn(), &q(None)).await.unwrap_err();
    assert!(matches!(err, StrataError::AmbiguousRoot(_)), "{err:?}");
    assert_eq!((err.code(), err.status()), ("ambiguous_root", 409));
    // two roots, named: each routes within its own tree
    let one = paths(engine.query(tn(), &q(Some(first))).await.unwrap());
    assert!(!one.is_empty());
    assert!(one.iter().all(|p| p == "root"), "{one:?}");
    let other = paths(engine.query(tn(), &q(Some(second))).await.unwrap());
    assert!(!other.is_empty());
    assert!(other.iter().all(|p| p == "other-root"), "{other:?}");
}

#[tokio::test]
async fn parent_must_exist_and_be_a_router() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;

    let err = engine
        .create_store(
            tn(),
            new_store(Some(t.leaf1), StoreKind::Leaf, "under-leaf"),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::InvalidTarget(_)), "{err:?}");

    let err = engine
        .create_store(
            tn(),
            new_store(Some(Uuid::now_v7()), StoreKind::Leaf, "orphan"),
        )
        .await
        .unwrap_err();
    // 404, exactly like another tenant's parent (existence never leaks across tenants)
    assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");
}

#[tokio::test]
async fn empty_name_is_rejected() {
    let (engine, _db) = test_engine().await;
    let err = engine
        .create_store(tn(), new_store(None, StoreKind::Router, "   "))
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::InvalidInput(_)), "{err:?}");
}

#[tokio::test]
async fn duplicate_name_is_rejected() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;

    let err = engine
        .create_store(tn(), new_store(Some(t.mid), StoreKind::Leaf, "leaf1"))
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::DuplicateName(_)), "{err:?}");

    // the same name under a different parent is fine
    engine
        .create_store(tn(), new_store(Some(t.root), StoreKind::Leaf, "leaf1"))
        .await
        .unwrap();

    // renaming onto a sibling's name is a duplicate too
    let err = engine
        .update_store(tn(), t.mid, Some("leaf3".to_string()), None)
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::DuplicateName(_)), "{err:?}");
}

#[tokio::test]
async fn profile_propagates_through_three_layers() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    let alpha = engine.cfg.profile_alpha;

    insert_raw_doc(&engine.pool, t.leaf1, &[(10, unit(0)), (10, unit(0))]).await;
    insert_raw_doc(&engine.pool, t.leaf2, &[(30, unit(1))]).await;
    recompute(&engine, t.leaf1).await;
    recompute(&engine, t.leaf2).await;

    // leaf1: mean of two e0 chunks
    let leaf1 = row(&engine, t.leaf1).await;
    assert_eq!((leaf1.token_count, leaf1.chunk_count), (20, 2));
    assert_close(leaf1.content_vec.as_deref().unwrap(), &unit(0));
    assert_close(
        &leaf1.profile_vec,
        &vecmath::profile(&leaf1.desc_vec, Some(unit(0).as_slice()), alpha),
    );

    // leaf2: one e1 chunk
    let leaf2 = row(&engine, t.leaf2).await;
    assert_eq!((leaf2.token_count, leaf2.chunk_count), (30, 1));
    assert_close(leaf2.content_vec.as_deref().unwrap(), &unit(1));

    // mid: token-weighted mean 20·e0 + 30·e1, normalised
    let mut mid_content = vec![0.0_f32; DIM];
    mid_content[0] = 0.4;
    mid_content[1] = 0.6;
    vecmath::l2_normalize(&mut mid_content);
    let mid = row(&engine, t.mid).await;
    assert_eq!((mid.token_count, mid.chunk_count), (50, 3));
    assert_close(mid.content_vec.as_deref().unwrap(), &mid_content);
    assert_close(
        &mid.profile_vec,
        &vecmath::profile(&mid.desc_vec, Some(mid_content.as_slice()), alpha),
    );

    // root: children mid (content, weight 50) and leaf3 (NULL content, skipped)
    let root = row(&engine, t.root).await;
    assert_eq!((root.token_count, root.chunk_count), (50, 3));
    assert_close(root.content_vec.as_deref().unwrap(), &mid_content);
    assert_close(
        &root.profile_vec,
        &vecmath::profile(&root.desc_vec, Some(mid_content.as_slice()), alpha),
    );

    // the empty leaf keeps profile = desc
    let leaf3 = row(&engine, t.leaf3).await;
    assert_eq!(leaf3.content_vec, None);
    assert_close(&leaf3.profile_vec, &leaf3.desc_vec);
}

#[tokio::test]
async fn delete_router_with_children_requires_recursive() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;

    let err = engine.delete_store(tn(), t.mid, false).await.unwrap_err();
    assert!(
        matches!(err, StrataError::HasChildren(id) if id == t.mid),
        "{err:?}"
    );
    assert_eq!(engine.store_tree(tn()).await.unwrap().len(), 5);

    let err = engine
        .delete_store(tn(), Uuid::now_v7(), false)
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");

    // a leaf has no child stores: non-recursive delete works
    engine.delete_store(tn(), t.leaf3, false).await.unwrap();
    assert_eq!(engine.store_tree(tn()).await.unwrap().len(), 4);
}

#[tokio::test]
async fn recursive_delete_removes_subtree_and_recomputes_parent() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    insert_raw_doc(&engine.pool, t.leaf1, &[(10, unit(0))]).await;
    insert_raw_doc(&engine.pool, t.leaf2, &[(30, unit(1))]).await;
    insert_raw_doc(&engine.pool, t.leaf3, &[(5, unit(2))]).await;
    recompute(&engine, t.leaf1).await;
    recompute(&engine, t.leaf2).await;
    recompute(&engine, t.leaf3).await;
    assert_eq!(row(&engine, t.root).await.token_count, 45);

    engine.delete_store(tn(), t.mid, true).await.unwrap();

    let ids: Vec<Uuid> = engine
        .store_tree(tn())
        .await
        .unwrap()
        .iter()
        .map(|s| s.id)
        .collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&t.root) && ids.contains(&t.leaf3));
    let err = engine.get_store(tn(), t.leaf1).await.unwrap_err();
    assert!(matches!(err, StrataError::NotFound(_)), "{err:?}");

    // documents and chunks of the deleted leaves are gone (FK cascade)
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM documents").await,
        1
    );
    assert_eq!(count(&engine.pool, "SELECT count(*) FROM chunks").await, 1);

    // root now only rolls up leaf3
    let root = row(&engine, t.root).await;
    assert_eq!((root.token_count, root.chunk_count), (5, 1));
    assert_close(root.content_vec.as_deref().unwrap(), &unit(2));
    assert_eq!(
        engine.get_store(tn(), t.root).await.unwrap().children,
        vec![t.leaf3]
    );
}

#[tokio::test]
async fn update_store_reembeds_description_and_recomputes() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    let before = row(&engine, t.mid).await;

    let view = engine
        .update_store(
            tn(),
            t.mid,
            None,
            Some("finance and accounting".to_string()),
        )
        .await
        .unwrap();
    assert_eq!(view.name, "mid");
    assert_eq!(view.description, "finance and accounting");

    let after = row(&engine, t.mid).await;
    let expected = engine
        .inference(tn())
        .await
        .unwrap()
        .embed_one("mid\n\nfinance and accounting".to_string())
        .await
        .unwrap();
    assert_close(&after.desc_vec, &expected);
    assert!(after.desc_vec != before.desc_vec);
    // no content yet: profile follows the new description
    assert_close(&after.profile_vec, &expected);

    let err = engine
        .update_store(tn(), t.mid, Some("  ".to_string()), None)
        .await
        .unwrap_err();
    assert!(matches!(err, StrataError::InvalidInput(_)), "{err:?}");
}

#[tokio::test]
async fn health_reports_db_providers_and_unconfigured_clients() {
    let (engine, _db) = test_engine().await;
    let h = std::sync::Arc::new(engine).health().await;
    assert_eq!(h.db, "ok");
    assert_eq!((h.embed, h.rerank), ("ok", "ok"));
    assert_eq!((h.embed_error, h.rerank_error), (None, None));
    assert_eq!(h.laya, "unconfigured");
    assert_eq!(h.chat, "unconfigured");
}

/// Every writer locks child before parent (ingest: leaf → … → root; update_store: store → … → root).
/// A recursive delete must take its subtree in the same order — deepest first — or it holds the
/// ancestor while waiting for a leaf that an ingest holds while waiting for that ancestor (40P01).
/// A holder takes leaf1's lock, as an ingest would; once the delete is blocked behind it, `mid` must
/// still be free for the holder to take next (`FOR UPDATE NOWAIT` succeeds).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recursive_delete_locks_descendants_before_the_ancestor() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    let mid = t.mid;

    let mut holder = engine.pool.begin().await.unwrap();
    let holder_pid = common::backend_pid(&mut holder).await;
    sqlx::query("SELECT 1 FROM stores WHERE id = $1 FOR UPDATE")
        .bind(t.leaf1)
        .execute(&mut *holder)
        .await
        .unwrap();
    let pool = engine.pool.clone();
    let (rd, probe) = tokio::join!(engine.delete_store(tn(), mid, true), async move {
        common::wait_until_blocked_by(&pool, holder_pid).await;
        // what an ingest into leaf1 does next: lock the ancestor
        let probe = sqlx::query("SELECT 1 FROM stores WHERE id = $1 FOR UPDATE NOWAIT")
            .bind(mid)
            .execute(&mut *holder)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string());
        holder.commit().await.unwrap();
        probe
    });

    assert!(
        probe.is_ok(),
        "recursive delete held the ancestor while waiting for a leaf: {probe:?}"
    );
    rd.unwrap();
    let ids: Vec<Uuid> = engine
        .store_tree(tn())
        .await
        .unwrap()
        .iter()
        .map(|s| s.id)
        .collect();
    assert_eq!(ids.len(), 2, "{ids:?}");
    assert!(ids.contains(&t.root) && ids.contains(&t.leaf3));
}

/// A child committed under a descendant while a recursive delete is in flight must be deleted with
/// the subtree, not trip `parent_id ON DELETE RESTRICT` (23503 → 500). A holder locks router `sub`
/// and inserts `late` under it; once the delete of `mid` is blocked behind that lock, the holder
/// commits. The delete must then see `late` and remove it too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recursive_delete_removes_a_child_created_while_it_waits() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    let sub = engine
        .create_store(tn(), new_store(Some(t.mid), StoreKind::Router, "sub"))
        .await
        .unwrap()
        .id;

    let mut holder = engine.pool.begin().await.unwrap();
    let holder_pid = common::backend_pid(&mut holder).await;
    sqlx::query("SELECT 1 FROM stores WHERE id = $1 FOR UPDATE")
        .bind(sub)
        .execute(&mut *holder)
        .await
        .unwrap();
    let late = stores::insert_store(
        &mut holder,
        tn(),
        &new_store(Some(sub), StoreKind::Leaf, "late"),
        &unit(3),
        "BAAI/bge-m3@1024",
    )
    .await
    .unwrap()
    .id;
    let pool = engine.pool.clone();
    let (rd, ()) = tokio::join!(engine.delete_store(tn(), t.mid, true), async move {
        common::wait_until_blocked_by(&pool, holder_pid).await;
        holder.commit().await.unwrap();
    });

    rd.unwrap();
    let ids: Vec<Uuid> = engine
        .store_tree(tn())
        .await
        .unwrap()
        .iter()
        .map(|s| s.id)
        .collect();
    assert!(!ids.contains(&late), "the late child survived: {ids:?}");
    assert_eq!(ids.len(), 2, "{ids:?}");
    assert!(ids.contains(&t.root) && ids.contains(&t.leaf3));
}

/// Concurrency failures are the client's to retry, not a server fault: a foreign-key violation
/// (a parent or leaf removed under a write) and a deadlock both map to `409 conflict`, not `500`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fk_violation_and_deadlock_map_to_409_conflict() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;

    // 23503: a store whose parent does not exist (the trigger leaves a missing parent to the FK)
    let mut conn = engine.pool.acquire().await.unwrap();
    let err = stores::insert_store(
        &mut conn,
        tn(),
        &new_store(Some(Uuid::now_v7()), StoreKind::Leaf, "orphan"),
        &unit(0),
        "BAAI/bge-m3@1024",
    )
    .await
    .unwrap_err();
    assert_eq!((err.code(), err.status()), ("conflict", 409), "{err:?}");
    drop(conn);

    // 40P01: two transactions taking two rows in opposite orders; Postgres aborts one of them
    let mut a = engine.pool.begin().await.unwrap();
    let mut b = engine.pool.begin().await.unwrap();
    let lock = "SELECT 1 FROM stores WHERE id = $1 FOR UPDATE";
    sqlx::query(lock)
        .bind(t.leaf1)
        .execute(&mut *a)
        .await
        .unwrap();
    sqlx::query(lock)
        .bind(t.leaf2)
        .execute(&mut *b)
        .await
        .unwrap();
    let (ra, rb) = tokio::join!(
        sqlx::query(lock).bind(t.leaf2).execute(&mut *a),
        sqlx::query(lock).bind(t.leaf1).execute(&mut *b),
    );
    let errs: Vec<StrataError> = [ra.err(), rb.err()]
        .into_iter()
        .flatten()
        .map(StrataError::from)
        .collect();
    assert_eq!(errs.len(), 1, "exactly one deadlock victim: {errs:?}");
    assert_eq!(
        (errs[0].code(), errs[0].status()),
        ("conflict", 409),
        "{:?}",
        errs[0]
    );
}

/// update_store reads the row, embeds outside any transaction, then writes. A description committed
/// by someone else inside that embed window must not be silently overwritten with the stale value
/// update_store read. A holder updates `mid`'s description (uncommitted) while update_store renames
/// `mid`; update_store reads the old committed row, embeds, and blocks on the holder's row lock;
/// once Postgres reports it blocked, the holder commits. The rename must then refuse (409 conflict)
/// rather than write back the old description.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_store_does_not_overwrite_a_change_made_during_its_embed() {
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    let before = row(&engine, t.mid).await;

    let mut holder = engine.pool.begin().await.unwrap();
    let holder_pid = common::backend_pid(&mut holder).await;
    sqlx::query("UPDATE stores SET description = 'changed concurrently' WHERE id = $1")
        .bind(t.mid)
        .execute(&mut *holder)
        .await
        .unwrap();
    let pool = engine.pool.clone();
    let (res, ()) = tokio::join!(
        engine.update_store(tn(), t.mid, Some("renamed".to_string()), None),
        async move {
            common::wait_until_blocked_by(&pool, holder_pid).await;
            holder.commit().await.unwrap();
        }
    );

    let after = row(&engine, t.mid).await;
    assert_eq!(
        after.description, "changed concurrently",
        "the concurrent description was overwritten with the stale one"
    );
    let err = res.unwrap_err();
    assert_eq!((err.code(), err.status()), ("conflict", 409), "{err:?}");
    assert_eq!(after.name, before.name, "a refused update must not write");

    // a field the request sets itself is last-writer-wins: no conflict
    let view = engine
        .update_store(tn(), t.mid, Some("renamed".to_string()), None)
        .await
        .unwrap();
    assert_eq!(view.name, "renamed");
    assert_eq!(view.description, "changed concurrently");
    let expected = engine
        .inference(tn())
        .await
        .unwrap()
        .embed_one("renamed\n\nchanged concurrently".to_string())
        .await
        .unwrap();
    assert_close(&row(&engine, t.mid).await.desc_vec, &expected);
}

/// Makes the next `failures` writes (INSERT/UPDATE/DELETE statements) on `stores` fail with
/// SQLSTATE 40001, as a lost serialization race would. Test-DB only: a statement trigger counting
/// on a sequence (`nextval` is not rolled back with the aborted transaction). No production hook.
async fn inject_serialization_failures(pool: &PgPool, failures: i64) {
    sqlx::query("CREATE SEQUENCE fault_seq")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query(sqlx::AssertSqlSafe(format!(
        "CREATE FUNCTION fault() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN \
             IF nextval('fault_seq') <= {failures} THEN \
                 RAISE EXCEPTION 'injected serialization failure' USING ERRCODE = '40001'; \
             END IF; \
             RETURN NULL; \
         END $$"
    )))
    .execute(pool)
    .await
    .unwrap();
    sqlx::query(
        "CREATE TRIGGER fault BEFORE INSERT OR UPDATE OR DELETE ON stores \
         FOR EACH STATEMENT EXECUTE FUNCTION fault()",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// create/update/delete_store each retry their transaction once on a retryable conflict: one
/// injected 40001 is absorbed (the call succeeds), two surface as 409 conflict (exactly one retry).
#[tokio::test]
async fn store_writes_retry_a_serialization_failure_once() {
    // one failure: absorbed by the single retry
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    inject_serialization_failures(&engine.pool, 1).await;
    let created = engine
        .create_store(tn(), new_store(Some(t.mid), StoreKind::Leaf, "leaf4"))
        .await;
    assert!(created.is_ok(), "create_store did not retry: {created:?}");
    sqlx::query("ALTER SEQUENCE fault_seq RESTART")
        .execute(&engine.pool)
        .await
        .unwrap();
    let updated = engine
        .update_store(tn(), t.leaf1, None, Some("retried".to_string()))
        .await;
    assert!(updated.is_ok(), "update_store did not retry: {updated:?}");
    assert_eq!(row(&engine, t.leaf1).await.description, "retried");
    sqlx::query("ALTER SEQUENCE fault_seq RESTART")
        .execute(&engine.pool)
        .await
        .unwrap();
    let deleted = engine.delete_store(tn(), t.leaf2, false).await;
    assert!(deleted.is_ok(), "delete_store did not retry: {deleted:?}");
    assert!(engine.get_store(tn(), t.leaf2).await.is_err());

    // two failures: one retry only, then 409 conflict
    let (engine, _db) = test_engine().await;
    let t = build_tree(&engine).await;
    inject_serialization_failures(&engine.pool, 2).await;
    let restart = || async {
        sqlx::query("ALTER SEQUENCE fault_seq RESTART")
            .execute(&engine.pool)
            .await
            .unwrap();
    };
    let err = engine
        .create_store(tn(), new_store(Some(t.mid), StoreKind::Leaf, "leaf4"))
        .await
        .unwrap_err();
    assert_eq!((err.code(), err.status()), ("conflict", 409), "{err:?}");
    restart().await;
    let err = engine
        .update_store(tn(), t.leaf1, None, Some("retried".to_string()))
        .await
        .unwrap_err();
    assert_eq!((err.code(), err.status()), ("conflict", 409), "{err:?}");
    restart().await;
    let err = engine.delete_store(tn(), t.leaf2, false).await.unwrap_err();
    assert_eq!((err.code(), err.status()), ("conflict", 409), "{err:?}");
    assert_eq!(
        count(&engine.pool, "SELECT count(*) FROM stores").await,
        5,
        "nothing was written"
    );
}
