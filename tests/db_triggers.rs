//! Schema invariants enforced in Postgres (spec §4 + triggers), proven against a real pgvector.

mod common;

use common::TestDb;
use sqlx::PgPool;
use uuid::Uuid;

async fn insert_store<'e>(
    pool: impl sqlx::PgExecutor<'e>,
    parent: Option<Uuid>,
    kind: &str,
    name: &str,
) -> Result<Uuid, sqlx::Error> {
    let id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO stores (id, tenant, parent_id, kind, name, description, desc_vec, \
                             profile_vec, embed_model) \
         VALUES ($1, $5, $2, $3, $4, 'test store', \
                 array_fill(0.1::real, ARRAY[1024])::vector, \
                 array_fill(0.1::real, ARRAY[1024])::vector, 'BAAI/bge-m3@1024')",
    )
    .bind(id)
    .bind(parent)
    .bind(kind)
    .bind(name)
    .bind(common::tn().id())
    .execute(pool)
    .await?;
    Ok(id)
}

async fn insert_document(pool: &PgPool, store_id: Uuid, hash: &[u8]) -> Result<Uuid, sqlx::Error> {
    let id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO documents (id, store_id, source, content_hash, token_count) \
         VALUES ($1, $2, 'test.txt', $3, 10)",
    )
    .bind(id)
    .bind(store_id)
    .bind(hash.to_vec())
    .execute(pool)
    .await?;
    Ok(id)
}

async fn insert_chunk(
    pool: &PgPool,
    document_id: Uuid,
    store_id: Uuid,
    ord: i32,
) -> Result<Uuid, sqlx::Error> {
    let id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO chunks (id, document_id, store_id, ord, text, token_count, embedding) \
         VALUES ($1, $2, $3, $4, 'hello strata world', 3, \
                 array_fill(0.1::real, ARRAY[1024])::vector)",
    )
    .bind(id)
    .bind(document_id)
    .bind(store_id)
    .bind(ord)
    .execute(pool)
    .await?;
    Ok(id)
}

/// The error must be a plain `RAISE EXCEPTION` (SQLSTATE P0001) whose message contains `needle`.
fn assert_raised(err: sqlx::Error, needle: &str) {
    let db = err
        .as_database_error()
        .unwrap_or_else(|| panic!("expected a database error, got {err:?}"));
    assert_eq!(
        db.code().as_deref(),
        Some("P0001"),
        "unexpected error: {}",
        db.message()
    );
    assert!(
        db.message().contains(needle),
        "message {:?} does not contain {needle:?}",
        db.message()
    );
}

#[tokio::test]
async fn happy_path_inserts_router_leaf_document_chunk() {
    let db = TestDb::start().await;
    let root = insert_store(&db.pool, None, "router", "root")
        .await
        .unwrap();
    let leaf = insert_store(&db.pool, Some(root), "leaf", "docs")
        .await
        .unwrap();
    let doc = insert_document(&db.pool, leaf, b"hash-1").await.unwrap();
    let chunk = insert_chunk(&db.pool, doc, leaf, 0).await.unwrap();

    let tsv: String = sqlx::query_scalar("SELECT tsv::text FROM chunks WHERE id = $1")
        .bind(chunk)
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert!(tsv.contains("'strata'"), "generated tsv was {tsv}");

    // Deleting the document cascades to its chunks.
    sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(doc)
        .execute(&db.pool)
        .await
        .unwrap();
    let left: i64 = sqlx::query_scalar("SELECT count(*) FROM chunks")
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(left, 0);
}

#[tokio::test]
async fn leaf_cannot_be_a_parent() {
    let db = TestDb::start().await;
    let root = insert_store(&db.pool, None, "router", "root")
        .await
        .unwrap();
    let leaf = insert_store(&db.pool, Some(root), "leaf", "docs")
        .await
        .unwrap();
    let err = insert_store(&db.pool, Some(leaf), "leaf", "nested")
        .await
        .unwrap_err();
    assert_raised(err, "must be a router");
}

#[tokio::test]
async fn store_cannot_be_moved_under_a_leaf() {
    let db = TestDb::start().await;
    let root = insert_store(&db.pool, None, "router", "root")
        .await
        .unwrap();
    let leaf = insert_store(&db.pool, Some(root), "leaf", "docs")
        .await
        .unwrap();
    let other = insert_store(&db.pool, Some(root), "router", "topics")
        .await
        .unwrap();
    let err = sqlx::query("UPDATE stores SET parent_id = $1 WHERE id = $2")
        .bind(leaf)
        .bind(other)
        .execute(&db.pool)
        .await
        .unwrap_err();
    assert_raised(err, "must be a router");
}

#[tokio::test]
async fn store_kind_is_immutable() {
    let db = TestDb::start().await;
    let root = insert_store(&db.pool, None, "router", "root")
        .await
        .unwrap();
    let leaf = insert_store(&db.pool, Some(root), "leaf", "docs")
        .await
        .unwrap();
    let err = sqlx::query("UPDATE stores SET kind = 'router' WHERE id = $1")
        .bind(leaf)
        .execute(&db.pool)
        .await
        .unwrap_err();
    assert_raised(err, "kind is immutable");
}

#[tokio::test]
async fn document_must_live_in_a_leaf() {
    let db = TestDb::start().await;
    let root = insert_store(&db.pool, None, "router", "root")
        .await
        .unwrap();
    let err = insert_document(&db.pool, root, b"hash-1")
        .await
        .unwrap_err();
    assert_raised(err, "must be in a leaf");
}

#[tokio::test]
async fn document_cannot_move_between_stores() {
    let db = TestDb::start().await;
    let root = insert_store(&db.pool, None, "router", "root")
        .await
        .unwrap();
    let a = insert_store(&db.pool, Some(root), "leaf", "a")
        .await
        .unwrap();
    let b = insert_store(&db.pool, Some(root), "leaf", "b")
        .await
        .unwrap();
    let doc = insert_document(&db.pool, a, b"hash-1").await.unwrap();
    let err = sqlx::query("UPDATE documents SET store_id = $1 WHERE id = $2")
        .bind(b)
        .bind(doc)
        .execute(&db.pool)
        .await
        .unwrap_err();
    assert_raised(err, "cannot move");
}

#[tokio::test]
async fn chunk_store_must_match_its_document() {
    let db = TestDb::start().await;
    let root = insert_store(&db.pool, None, "router", "root")
        .await
        .unwrap();
    let a = insert_store(&db.pool, Some(root), "leaf", "a")
        .await
        .unwrap();
    let b = insert_store(&db.pool, Some(root), "leaf", "b")
        .await
        .unwrap();
    let doc = insert_document(&db.pool, a, b"hash-1").await.unwrap();
    let err = insert_chunk(&db.pool, doc, b, 0).await.unwrap_err();
    assert_raised(err, "does not match");
}

#[tokio::test]
async fn sibling_names_are_unique_including_roots() {
    let db = TestDb::start().await;
    insert_store(&db.pool, None, "router", "root")
        .await
        .unwrap();
    // UNIQUE NULLS NOT DISTINCT (parent_id, name): a second root with the same name collides.
    let err = insert_store(&db.pool, None, "router", "root")
        .await
        .unwrap_err();
    let code = err
        .as_database_error()
        .and_then(|d| d.code())
        .map(|c| c.into_owned());
    assert_eq!(code.as_deref(), Some("23505"));
}

#[tokio::test]
async fn migrations_are_idempotent() {
    let db = TestDb::start().await;
    // A second connect re-runs sqlx::migrate! against an already-migrated database.
    let again = strata_rag::db::connect(&db.url).await.unwrap();
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM _sqlx_migrations WHERE success")
        .fetch_one(&again)
        .await
        .unwrap();
    // One row per migration file, none applied twice (counted, not hardcoded: 0002 made a literal 1 stale).
    let files = sqlx::migrate!("./migrations").migrations.len() as i64;
    assert!(files >= 2, "{files}");
    assert_eq!(n, files);
}

/// `insert_store` with an explicit tenant (the helper above always uses `common::TENANT`).
async fn insert_store_for(
    pool: &PgPool,
    tenant: Uuid,
    parent: Option<Uuid>,
    kind: &str,
    name: &str,
) -> Result<Uuid, sqlx::Error> {
    let id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO stores (id, tenant, parent_id, kind, name, description, desc_vec, \
                             profile_vec, embed_model) \
         VALUES ($1, $2, $3, $4, $5, 'test store', \
                 array_fill(0.1::real, ARRAY[1024])::vector, \
                 array_fill(0.1::real, ARRAY[1024])::vector, 'BAAI/bge-m3@1024')",
    )
    .bind(id)
    .bind(tenant)
    .bind(parent)
    .bind(kind)
    .bind(name)
    .execute(pool)
    .await?;
    Ok(id)
}

/// Migration 0005: a child's tenant is its parent's, a tenant never changes, nil is no tenant, and
/// root names are unique per tenant (not across tenants, which would leak them).
#[tokio::test]
async fn store_tenant_invariants() {
    let db = TestDb::start().await;
    let (a, b) = (common::tn().id(), common::other().id());
    let root_a = insert_store_for(&db.pool, a, None, "router", "lib")
        .await
        .unwrap();

    // a child under another tenant's parent
    let err = insert_store_for(&db.pool, b, Some(root_a), "leaf", "x")
        .await
        .unwrap_err();
    assert_raised(err, "does not match its parent");
    let leaf_a = insert_store_for(&db.pool, a, Some(root_a), "leaf", "x")
        .await
        .unwrap();

    // a tenant UPDATE, on a root and on a child
    for id in [root_a, leaf_a] {
        let err = sqlx::query("UPDATE stores SET tenant = $2 WHERE id = $1")
            .bind(id)
            .bind(b)
            .execute(&db.pool)
            .await
            .unwrap_err();
        assert_raised(err, "tenant is immutable");
    }
    // moving a store under another tenant's router
    let root_b = insert_store_for(&db.pool, b, None, "router", "lib")
        .await
        .expect("the same root name in another tenant is allowed");
    let err = sqlx::query("UPDATE stores SET parent_id = $2 WHERE id = $1")
        .bind(leaf_a)
        .bind(root_b)
        .execute(&db.pool)
        .await
        .unwrap_err();
    assert_raised(err, "does not match its parent");
    let tenants: Vec<Uuid> =
        sqlx::query_scalar("SELECT tenant FROM stores WHERE id = ANY($1) ORDER BY id")
            .bind(vec![root_a, leaf_a])
            .fetch_all(&db.pool)
            .await
            .unwrap();
    assert_eq!(tenants, [a, a], "nothing changed");

    // nil tenant
    let err = insert_store_for(&db.pool, Uuid::nil(), None, "router", "nil")
        .await
        .unwrap_err();
    let dbe = err.as_database_error().unwrap();
    assert_eq!(dbe.code().as_deref(), Some("23514"), "{}", dbe.message());

    // a second root of the same name in the SAME tenant is still a duplicate
    let err = insert_store_for(&db.pool, a, None, "router", "lib")
        .await
        .unwrap_err();
    assert_eq!(
        err.as_database_error().unwrap().code().as_deref(),
        Some("23505")
    );
}

/// Migration 0005 adds `tenant` with no default: on a table holding stores it must FAIL (23502),
/// never invent a tenant. Replayed in a rolled-back transaction: drop the column, add a store,
/// run the migration's ADD COLUMN statement.
#[tokio::test]
async fn tenant_migration_fails_loudly_on_a_non_empty_table() {
    let db = TestDb::start().await;
    let sql = include_str!("../migrations/0005_stores_tenant.sql");
    let start = sql.find("ALTER TABLE stores\n  ADD COLUMN tenant").unwrap();
    let add_column = &sql[start..start + sql[start..].find(';').unwrap()];
    let mut tx = db.pool.begin().await.unwrap();
    sqlx::query("ALTER TABLE stores DROP COLUMN tenant CASCADE")
        .execute(&mut *tx)
        .await
        .unwrap();
    // the 0005 triggers read NEW.tenant (CASCADE drops the UPDATE OF tenant ones already); 0010's
    // deferred generation trigger did not exist before 0005 either, and its pending event would
    // make Postgres refuse the ALTER TABLE below (55006) before the column check is reached
    for trigger in [
        "stores_tenant_matches_parent",
        "stores_tenant_immutable",
        "stores_bump_generation",
    ] {
        sqlx::query(sqlx::AssertSqlSafe(format!(
            "DROP TRIGGER IF EXISTS {trigger} ON stores"
        )))
        .execute(&mut *tx)
        .await
        .unwrap();
    }
    sqlx::query(
        "INSERT INTO stores (id, parent_id, kind, name, description, desc_vec, profile_vec, \
                             embed_model) \
         VALUES ($1, NULL, 'router', 'old', 'd', array_fill(0.1::real, ARRAY[1024])::vector, \
                 array_fill(0.1::real, ARRAY[1024])::vector, 'BAAI/bge-m3@1024')",
    )
    .bind(Uuid::now_v7())
    .execute(&mut *tx)
    .await
    .unwrap();
    let err = sqlx::query(sqlx::AssertSqlSafe(add_column.to_string()))
        .execute(&mut *tx)
        .await
        .unwrap_err();
    assert_eq!(
        err.as_database_error().unwrap().code().as_deref(),
        Some("23502"),
        "{err}"
    );
    tx.rollback().await.unwrap();
}

/// The trap documented in `migrations/README.md`: 0010's `stores_bump_generation` is a DEFERRED
/// constraint trigger, so a transaction (a migration) that writes `stores` and then runs
/// `ALTER TABLE stores` is refused with 55006 while the trigger event is pending. The documented
/// workaround, `SET CONSTRAINTS stores_bump_generation IMMEDIATE` after the write, fires the
/// pending event at once and lets the ALTER through, and the generation is still bumped.
#[tokio::test]
async fn a_store_write_then_alter_needs_the_generation_trigger_set_immediate() {
    let db = TestDb::start().await;
    let alter = "ALTER TABLE stores ADD COLUMN migration_trap_probe int";

    // the trap: write, then ALTER in the same transaction
    let mut tx = db.pool.begin().await.unwrap();
    insert_store(&mut *tx, None, "router", "trap")
        .await
        .unwrap();
    let err = sqlx::query(alter).execute(&mut *tx).await.unwrap_err();
    assert_eq!(
        err.as_database_error().unwrap().code().as_deref(),
        Some("55006"),
        "{err}"
    );
    tx.rollback().await.unwrap();
    assert_eq!(
        generation(&db.pool).await,
        None,
        "the rolled-back write bumped nothing"
    );

    // the workaround: SET CONSTRAINTS ... IMMEDIATE between the write and the ALTER
    let mut tx = db.pool.begin().await.unwrap();
    insert_store(&mut *tx, None, "router", "fixed")
        .await
        .unwrap();
    sqlx::query("SET CONSTRAINTS stores_bump_generation IMMEDIATE")
        .execute(&mut *tx)
        .await
        .unwrap();
    sqlx::query(alter).execute(&mut *tx).await.unwrap();
    tx.commit().await.unwrap();
    assert_eq!(
        generation(&db.pool).await,
        Some(1),
        "the write still bumped the generation"
    );
    let probe: Option<i32> = sqlx::query_scalar("SELECT migration_trap_probe FROM stores")
        .fetch_one(&db.pool)
        .await
        .unwrap();
    assert_eq!(probe, None, "the ALTER committed");
}

async fn generation(pool: &PgPool) -> Option<i64> {
    sqlx::query_scalar("SELECT generation FROM store_generations WHERE tenant = $1")
        .bind(common::tn().id())
        .fetch_optional(pool)
        .await
        .unwrap()
}
