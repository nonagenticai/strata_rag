//! Store rows: CRUD and the bottom-up roll-up of content/profile vectors.

use std::collections::{BTreeMap, HashSet};

use pgvector::Vector;
use sqlx::postgres::PgRow;
use sqlx::{PgConnection, PgPool, Row};
use uuid::Uuid;

use crate::error::{Result, StrataError};
use crate::tenant::Tenant;
use crate::vecmath;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoreKind {
    Router,
    Leaf,
}

impl StoreKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            StoreKind::Router => "router",
            StoreKind::Leaf => "leaf",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "router" => Ok(StoreKind::Router),
            "leaf" => Ok(StoreKind::Leaf),
            other => Err(StrataError::InvalidInput(format!(
                "unknown store kind {other:?}; expected \"router\" or \"leaf\""
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StoreRow {
    pub id: Uuid,
    /// The tenant the store belongs to (migration 0005); a child's is its parent's.
    pub tenant: Uuid,
    pub parent_id: Option<Uuid>,
    pub kind: StoreKind,
    pub name: String,
    pub description: String,
    pub desc_vec: Vec<f32>,
    pub content_vec: Option<Vec<f32>>,
    pub profile_vec: Vec<f32>,
    pub token_count: i64,
    pub chunk_count: i64,
    /// `"{model}@{dims}"` that made this store's vectors (see `Config::embed_model_id`).
    pub embed_model: String,
}

pub struct NewStore {
    pub parent_id: Option<Uuid>,
    pub kind: StoreKind,
    pub name: String,
    pub description: String,
}

/// Maps write errors: unique violation → `DuplicateName(label)`, a trigger
/// `RAISE EXCEPTION` (SQLSTATE P0001) → `InvalidTarget(message)`; everything else
/// through `From<sqlx::Error>` (FK / deadlock → `Conflict`).
pub(crate) fn map_write_err(e: sqlx::Error, label: &str) -> StrataError {
    if let sqlx::Error::Database(db) = &e {
        match db.code().as_deref() {
            Some("23505") => return StrataError::DuplicateName(label.to_string()),
            Some("P0001") => return StrataError::InvalidTarget(db.message().to_string()),
            _ => {}
        }
    }
    StrataError::from(e)
}

fn row_to_store(row: &PgRow) -> Result<StoreRow> {
    let kind: String = row.try_get("kind")?;
    Ok(StoreRow {
        id: row.try_get("id")?,
        tenant: row.try_get("tenant")?,
        parent_id: row.try_get("parent_id")?,
        kind: StoreKind::parse(&kind)?,
        name: row.try_get("name")?,
        description: row.try_get("description")?,
        desc_vec: row.try_get::<Vector, _>("desc_vec")?.to_vec(),
        content_vec: row
            .try_get::<Option<Vector>, _>("content_vec")?
            .map(|v| v.to_vec()),
        profile_vec: row.try_get::<Vector, _>("profile_vec")?.to_vec(),
        token_count: row.try_get("token_count")?,
        chunk_count: row.try_get("chunk_count")?,
        embed_model: row.try_get("embed_model")?,
    })
}

/// Every column `row_to_store` reads (a macro so `concat!` builds `&'static str` SQL).
macro_rules! store_cols {
    () => {
        "id, tenant, parent_id, kind, name, description, desc_vec, content_vec, profile_vec, \
         token_count, chunk_count, embed_model"
    };
}

/// `embed_model`: the configured `"{model}@{dims}"` that produced `desc_vec`. The caller has
/// checked that a parent belongs to `tenant` (the `stores_tenant_matches_parent` trigger refuses
/// it otherwise).
pub async fn insert_store(
    conn: &mut PgConnection,
    tenant: Tenant,
    new: &NewStore,
    desc_vec: &[f32],
    embed_model: &str,
) -> Result<StoreRow> {
    let row = sqlx::query(concat!(
        "INSERT INTO stores (id, tenant, parent_id, kind, name, description, desc_vec, \
                             profile_vec, embed_model) \
         VALUES ($1, $8, $2, $3, $4, $5, $6, $6, $7) \
         RETURNING ",
        store_cols!()
    ))
    .bind(Uuid::now_v7())
    .bind(new.parent_id)
    .bind(new.kind.as_str())
    .bind(&new.name)
    .bind(&new.description)
    .bind(Vector::from(desc_vec.to_vec()))
    .bind(embed_model)
    .bind(tenant.id())
    .fetch_one(&mut *conn)
    .await
    .map_err(|e| map_write_err(e, &new.name))?;
    row_to_store(&row)
}

/// The nearest store on the path from `id` up to the root (inclusive) whose `embed_model` is not
/// `configured`, as `(id, embed_model)`. `None` when the whole lineage matches (or `id` is gone).
pub(crate) async fn lineage_embed_mismatch(
    conn: &mut PgConnection,
    id: Uuid,
    configured: &str,
) -> Result<Option<(Uuid, String)>> {
    let row: Option<(Uuid, String)> = sqlx::query_as(
        "WITH RECURSIVE up (id, parent_id, embed_model, depth) AS ( \
             SELECT id, parent_id, embed_model, 0 FROM stores WHERE id = $1 \
             UNION ALL \
             SELECT s.id, s.parent_id, s.embed_model, up.depth + 1 \
             FROM stores s JOIN up ON s.id = up.parent_id \
             WHERE up.depth < 10000 \
         ) \
         SELECT id, embed_model FROM up WHERE embed_model <> $2 ORDER BY depth LIMIT 1",
    )
    .bind(id)
    .bind(configured)
    .fetch_optional(&mut *conn)
    .await?;
    Ok(row)
}

/// `tenant`'s store `id`. Another tenant's store is `NotFound`, exactly like a missing one, so a
/// caller cannot tell the two apart.
pub async fn get_store(conn: &mut PgConnection, tenant: Tenant, id: Uuid) -> Result<StoreRow> {
    let row = sqlx::query(concat!(
        "SELECT ",
        store_cols!(),
        " FROM stores WHERE id = $1 AND tenant = $2"
    ))
    .bind(id)
    .bind(tenant.id())
    .fetch_optional(&mut *conn)
    .await?
    .ok_or_else(|| StrataError::NotFound(format!("store {id}")))?;
    row_to_store(&row)
}

/// `get_store` under a row lock.
pub async fn get_store_for_update(
    conn: &mut PgConnection,
    tenant: Tenant,
    id: Uuid,
) -> Result<StoreRow> {
    let row = sqlx::query(concat!(
        "SELECT ",
        store_cols!(),
        " FROM stores WHERE id = $1 AND tenant = $2 FOR UPDATE"
    ))
    .bind(id)
    .bind(tenant.id())
    .fetch_optional(&mut *conn)
    .await?
    .ok_or_else(|| StrataError::NotFound(format!("store {id}")))?;
    row_to_store(&row)
}

/// `tenant`'s stores, uuid v7 order.
pub async fn list_stores(pool: &PgPool, tenant: Tenant) -> Result<Vec<StoreRow>> {
    let rows = sqlx::query(concat!(
        "SELECT ",
        store_cols!(),
        " FROM stores WHERE tenant = $1 ORDER BY id"
    ))
    .bind(tenant.id())
    .fetch_all(pool)
    .await?;
    rows.iter().map(row_to_store).collect()
}

/// The columns routing reads (`crate::routing::Tree`): no `desc_vec` / `content_vec`, which
/// routing never touches, so a tree reload moves one vector per store instead of three.
#[derive(Debug, Clone)]
pub struct RoutingRow {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub kind: StoreKind,
    pub name: String,
    pub description: String,
    pub profile_vec: Vec<f32>,
    /// `"{model}@{dims}"` that made this store's vectors.
    pub embed_model: String,
}

/// `tenant`'s stores as [`RoutingRow`]s, uuid v7 order.
pub async fn list_routing_rows(pool: &PgPool, tenant: Tenant) -> Result<Vec<RoutingRow>> {
    let rows = sqlx::query(
        "SELECT id, parent_id, kind, name, description, profile_vec, embed_model \
         FROM stores WHERE tenant = $1 ORDER BY id",
    )
    .bind(tenant.id())
    .fetch_all(pool)
    .await?;
    rows.iter()
        .map(|row| -> Result<RoutingRow> {
            let kind: String = row.try_get("kind")?;
            Ok(RoutingRow {
                id: row.try_get("id")?,
                parent_id: row.try_get("parent_id")?,
                kind: StoreKind::parse(&kind)?,
                name: row.try_get("name")?,
                description: row.try_get("description")?,
                profile_vec: row.try_get::<Vector, _>("profile_vec")?.to_vec(),
                embed_model: row.try_get("embed_model")?,
            })
        })
        .collect()
}

pub(crate) async fn child_count(conn: &mut PgConnection, id: Uuid) -> Result<i64> {
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM stores WHERE parent_id = $1")
        .bind(id)
        .fetch_one(&mut *conn)
        .await?;
    Ok(n)
}

/// Child store ids, oldest first (uuid v7 order).
pub(crate) async fn child_ids(conn: &mut PgConnection, id: Uuid) -> Result<Vec<Uuid>> {
    let ids: Vec<Uuid> =
        sqlx::query_scalar("SELECT id FROM stores WHERE parent_id = $1 ORDER BY id")
            .bind(id)
            .fetch_all(&mut *conn)
            .await?;
    Ok(ids)
}

/// Updates name/description/desc_vec of `tenant`'s store only (another tenant's is `NotFound`).
/// Callers run `recompute_upwards` afterwards so `profile_vec` follows the new `desc_vec`.
pub(crate) async fn update_store_meta(
    conn: &mut PgConnection,
    tenant: Tenant,
    id: Uuid,
    name: &str,
    description: &str,
    desc_vec: &[f32],
) -> Result<()> {
    let done = sqlx::query(
        "UPDATE stores SET name = $2, description = $3, desc_vec = $4, updated_at = now() \
         WHERE id = $1 AND tenant = $5",
    )
    .bind(id)
    .bind(name)
    .bind(description)
    .bind(Vector::from(desc_vec.to_vec()))
    .bind(tenant.id())
    .execute(&mut *conn)
    .await
    .map_err(|e| map_write_err(e, name))?;
    if done.rows_affected() == 0 {
        return Err(StrataError::NotFound(format!("store {id}")));
    }
    Ok(())
}

/// Rounds of "read subtree, lock what is new" before a recursive delete gives up. The second
/// round only finds work when a child was committed under a member while we waited for its lock.
const SUBTREE_LOCK_ROUNDS: usize = 8;

/// `(id, depth)` of `id` and every descendant; depth 0 is `id`. Empty when `id` is not
/// `tenant`'s (descendants share their ancestor's tenant).
pub(crate) async fn subtree(
    conn: &mut PgConnection,
    tenant: Tenant,
    id: Uuid,
) -> Result<Vec<(Uuid, i32)>> {
    let rows = sqlx::query(
        "WITH RECURSIVE sub(id, depth) AS ( \
             SELECT id, 0 FROM stores WHERE id = $1 AND tenant = $2 \
             UNION ALL \
             SELECT s.id, sub.depth + 1 FROM stores s JOIN sub ON s.parent_id = sub.id \
         ) SELECT id, depth FROM sub",
    )
    .bind(id)
    .bind(tenant.id())
    .fetch_all(&mut *conn)
    .await?;
    rows.iter()
        .map(|r| Ok((r.try_get("id")?, r.try_get("depth")?)))
        .collect()
}

/// Deletes a store. With `recursive`, deletes every descendant store deepest
/// level first (`parent_id` is ON DELETE RESTRICT); documents and chunks go by
/// FK cascade. Returns the deleted store's `parent_id` so the caller can
/// recompute it.
///
/// Lock order: every writer locks child before parent (ingest and update_store lock a node, then
/// `recompute_upwards` walks to the root), so the subtree is locked deepest level first (then by
/// id) and the target last. Taking the target first would deadlock with an ingest that holds a
/// leaf and waits for its ancestors. Once every member is locked no child can be added under one
/// (an insert needs a key-share lock on its parent), so the subtree is re-read until it is stable.
pub(crate) async fn delete_store(
    conn: &mut PgConnection,
    tenant: Tenant,
    id: Uuid,
    recursive: bool,
) -> Result<Option<Uuid>> {
    if !recursive {
        let store = get_store_for_update(&mut *conn, tenant, id).await?;
        if child_count(&mut *conn, id).await? > 0 {
            return Err(StrataError::HasChildren(id));
        }
        sqlx::query("DELETE FROM stores WHERE id = $1")
            .bind(id)
            .execute(&mut *conn)
            .await?;
        return Ok(store.parent_id);
    }

    let mut locked: HashSet<Uuid> = HashSet::new();
    let mut members = Vec::new();
    for _ in 0..SUBTREE_LOCK_ROUNDS {
        members = subtree(&mut *conn, tenant, id).await?;
        if members.is_empty() {
            return Err(StrataError::NotFound(format!("store {id}")));
        }
        let (ids, depths): (Vec<Uuid>, Vec<i32>) = members
            .iter()
            .filter(|(sid, _)| !locked.contains(sid))
            .copied()
            .unzip();
        if ids.is_empty() {
            break;
        }
        // LockRows sits above the Sort, so rows are locked in ORDER BY order.
        let got: Vec<Uuid> = sqlx::query_scalar(
            "SELECT s.id FROM stores s \
             JOIN unnest($1::uuid[], $2::int4[]) AS t(id, depth) ON t.id = s.id \
             ORDER BY t.depth DESC, s.id \
             FOR UPDATE OF s",
        )
        .bind(ids.as_slice())
        .bind(depths.as_slice())
        .fetch_all(&mut *conn)
        .await?;
        locked.extend(got);
    }
    if members.iter().any(|(sid, _)| !locked.contains(sid)) {
        return Err(StrataError::Conflict {
            message: format!("store {id} kept gaining children during a recursive delete"),
            retryable: true,
        });
    }
    if !locked.contains(&id) {
        return Err(StrataError::NotFound(format!("store {id}")));
    }
    let parent_id = get_store(&mut *conn, tenant, id).await?.parent_id;

    let mut by_depth: BTreeMap<i32, Vec<Uuid>> = BTreeMap::new();
    for (sid, depth) in members {
        by_depth.entry(depth).or_default().push(sid);
    }
    for (_, ids) in by_depth.into_iter().rev() {
        sqlx::query("DELETE FROM stores WHERE id = ANY($1)")
            .bind(ids.as_slice())
            .execute(&mut *conn)
            .await?;
    }
    Ok(parent_id)
}

/// (content_vec, token_count, chunk_count) of a leaf from its running sums (migration 0020): the
/// mean of its chunk vectors (`content_sum / content_chunks`), normalised, and its document token
/// total. The `chunks` / `documents` triggers keep the sums, so this reads one row whatever the
/// leaf holds (it used to rescan every chunk, making a bulk load quadratic).
async fn leaf_aggregate(conn: &mut PgConnection, id: Uuid) -> Result<(Option<Vec<f32>>, i64, i64)> {
    let row =
        sqlx::query("SELECT content_sum, content_chunks, content_tokens FROM stores WHERE id = $1")
            .bind(id)
            .fetch_one(&mut *conn)
            .await?;
    let sum: Option<Vec<f64>> = row.try_get("content_sum")?;
    let chunk_count: i64 = row.try_get("content_chunks")?;
    let token_count: i64 = row.try_get("content_tokens")?;
    let content = sum.filter(|_| chunk_count > 0).map(|s| {
        let n = chunk_count as f64;
        let mut v: Vec<f32> = s.iter().map(|x| (x / n) as f32).collect();
        vecmath::l2_normalize(&mut v);
        v
    });
    Ok((content, token_count, chunk_count))
}

/// Rebuilds the running chunk-vector sums (migration 0020) of the leaves among `ids` from their
/// chunks, exactly. The triggers keep them incrementally; a re-embed that rewrites every vector
/// of a tree calls this so no float residue of the old vectors survives in the new sums.
pub(crate) async fn rebuild_content_sums(conn: &mut PgConnection, ids: &[Uuid]) -> Result<()> {
    sqlx::query(
        "UPDATE stores s SET content_sum = a.total::real[]::float8[], content_chunks = a.n \
         FROM unnest($1::uuid[]) AS t(id) \
         CROSS JOIN LATERAL (SELECT sum(c.embedding) AS total, count(*) AS n \
                             FROM chunks c WHERE c.store_id = t.id) a \
         WHERE s.id = t.id AND s.kind = 'leaf'",
    )
    .bind(ids)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

/// (content_vec, token_count, chunk_count) of a router from its children:
/// token-weighted mean of non-NULL child content_vecs, normalised; sums of counts.
async fn router_aggregate(
    conn: &mut PgConnection,
    id: Uuid,
) -> Result<(Option<Vec<f32>>, i64, i64)> {
    let rows = sqlx::query(
        "SELECT content_vec, token_count, chunk_count FROM stores WHERE parent_id = $1 ORDER BY id",
    )
    .bind(id)
    .fetch_all(&mut *conn)
    .await?;
    let mut token_count = 0_i64;
    let mut chunk_count = 0_i64;
    let mut vecs: Vec<(Vec<f32>, f64)> = Vec::new();
    for r in &rows {
        let tokens: i64 = r.try_get("token_count")?;
        token_count += tokens;
        chunk_count += r.try_get::<i64, _>("chunk_count")?;
        if let Some(v) = r.try_get::<Option<Vector>, _>("content_vec")? {
            vecs.push((v.to_vec(), tokens as f64));
        }
    }
    let items: Vec<(&[f32], f64)> = vecs.iter().map(|(v, w)| (v.as_slice(), *w)).collect();
    let content = vecmath::weighted_mean(&items).map(|mut v| {
        vecmath::l2_normalize(&mut v);
        v
    });
    Ok((content, token_count, chunk_count))
}

/// Recomputes content_vec / token_count / chunk_count / profile_vec for `start`
/// and every ancestor. Each node is locked `FOR UPDATE` before its children are
/// read, so concurrent writers under the same router serialise and the second
/// one reads the first one's committed children.
///
/// Every node is read and written within `tenant` (ancestors share it, by trigger): a `start` of
/// another tenant is `NotFound`, and nothing is written.
pub async fn recompute_upwards(
    conn: &mut PgConnection,
    tenant: Tenant,
    start: Uuid,
    alpha: f32,
) -> Result<()> {
    let mut current = Some(start);
    while let Some(id) = current {
        current = recompute_node(&mut *conn, tenant, id, alpha).await?;
    }
    Ok(())
}

/// Recomputes content_vec / token_count / chunk_count / profile_vec of the one store `id` (under
/// a row lock) from its chunks or its children's current rows; returns its `parent_id`. Callers
/// recomputing several stores go child before parent, so a router reads updated children.
pub(crate) async fn recompute_node(
    conn: &mut PgConnection,
    tenant: Tenant,
    id: Uuid,
    alpha: f32,
) -> Result<Option<Uuid>> {
    let node = get_store_for_update(&mut *conn, tenant, id).await?;
    let (content, token_count, chunk_count) = match node.kind {
        StoreKind::Leaf => leaf_aggregate(&mut *conn, id).await?,
        StoreKind::Router => router_aggregate(&mut *conn, id).await?,
    };
    let profile = vecmath::profile(&node.desc_vec, content.as_deref(), alpha);
    sqlx::query(
        "UPDATE stores SET content_vec = $2, profile_vec = $3, token_count = $4, \
                chunk_count = $5, updated_at = now() \
         WHERE id = $1 AND tenant = $6",
    )
    .bind(id)
    .bind(content.map(Vector::from))
    .bind(Vector::from(profile))
    .bind(token_count)
    .bind(chunk_count)
    .bind(tenant.id())
    .execute(&mut *conn)
    .await?;
    Ok(node.parent_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_kind_round_trips() {
        for k in [StoreKind::Router, StoreKind::Leaf] {
            assert_eq!(StoreKind::parse(k.as_str()).unwrap(), k);
        }
        assert_eq!(serde_json::to_string(&StoreKind::Leaf).unwrap(), "\"leaf\"");
    }

    #[test]
    fn unknown_store_kind_is_invalid_input() {
        let err = StoreKind::parse("branch").unwrap_err();
        assert!(matches!(err, StrataError::InvalidInput(_)), "{err:?}");
    }
}
