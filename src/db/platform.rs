//! Platform packs and exclusions (migration 0022; spec 2026-10-20-platform-knowledge-design.md
//! §3-§5). [`readable_packs`] is THE read predicate: the only place a tenant is granted another
//! owner's (the platform owner's) stores (`crate::query::Readable` applies it to a query). [`assert_writable`] is the freeze every store write
//! into a published pack hits.

use serde::{Serialize, Serializer};
use sqlx::PgConnection;
use sqlx::types::chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::db::stores::{self, StoreKind};
use crate::error::{Result, StrataError};
use crate::platform::valid_pack_key;
use crate::tenant::Tenant;

/// One pack version: a root store of the platform owner.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlatformPack {
    pub key: String,
    pub version: i32,
    pub root_id: Uuid,
    /// `None` = draft (platform owner only).
    #[serde(serialize_with = "ser_ts")]
    pub published_at: Option<DateTime<Utc>>,
    /// Set = hidden from other tenants' listings (published packs only); still readable.
    #[serde(serialize_with = "ser_ts")]
    pub retired_at: Option<DateTime<Utc>>,
}

/// A tenant's exclusion from `key_prefix` and every key below it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PlatformExclusion {
    pub key_prefix: String,
    pub reason: String,
    #[serde(serialize_with = "ser_ts_req")]
    pub created_at: DateTime<Utc>,
}

fn rfc3339(t: &DateTime<Utc>) -> String {
    t.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string()
}

fn ser_ts<S: Serializer>(t: &Option<DateTime<Utc>>, s: S) -> std::result::Result<S::Ok, S::Error> {
    match t {
        Some(t) => s.serialize_some(&rfc3339(t)),
        None => s.serialize_none(),
    }
}

fn ser_ts_req<S: Serializer>(t: &DateTime<Utc>, s: S) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_str(&rfc3339(t))
}

type PackRow = (
    String,
    i32,
    Uuid,
    Option<DateTime<Utc>>,
    Option<DateTime<Utc>>,
);

fn pack_of(r: PackRow) -> PlatformPack {
    PlatformPack {
        key: r.0,
        version: r.1,
        root_id: r.2,
        published_at: r.3,
        retired_at: r.4,
    }
}

/// `InvalidInput` unless `key` is a valid pack key / exclusion prefix (`crate::platform`).
pub fn check_key(what: &str, key: &str) -> Result<()> {
    if valid_pack_key(key) {
        Ok(())
    } else {
        Err(StrataError::InvalidInput(format!(
            "{what} {key:?} must be 1 to 6 `/`-separated segments of [a-z0-9_-]"
        )))
    }
}

/// The packs `caller` may read (spec §5): published and not excluded for `caller` (an exclusion
/// prefix covers its key and every key below it, segment-wise). A RETIRED pack stays readable
/// (spec §3: existing pins keep working, owner decision 2026-10-20); it is only left out of
/// listings ([`listed_packs`]). The platform owner reads every pack, drafts included. Ordered by
/// key, then version.
pub async fn readable_packs<'e, E: sqlx::PgExecutor<'e>>(
    db: E,
    platform_owner: Tenant,
    caller: Tenant,
) -> Result<Vec<PlatformPack>> {
    let rows: Vec<PackRow> = sqlx::query_as(
        "SELECT p.key, p.version, p.root_id, p.published_at, p.retired_at \
         FROM platform_packs p \
         WHERE $1 OR ( \
           p.published_at IS NOT NULL \
           AND NOT EXISTS ( \
             SELECT 1 FROM platform_exclusions e \
             WHERE e.tenant = $2 \
               AND (p.key = e.key_prefix OR starts_with(p.key, e.key_prefix || '/')))) \
         ORDER BY p.key, p.version",
    )
    .bind(caller == platform_owner)
    .bind(caller.id())
    .fetch_all(db)
    .await?;
    Ok(rows.into_iter().map(pack_of).collect())
}

/// `GET /api/platform/packs`: the packs `caller` may read, without retired ones unless `caller` is
/// the platform owner (a retired pack is pinned by id, never newly discovered).
pub async fn listed_packs<'e, E: sqlx::PgExecutor<'e>>(
    db: E,
    platform_owner: Tenant,
    caller: Tenant,
) -> Result<Vec<PlatformPack>> {
    let mut packs = readable_packs(db, platform_owner, caller).await?;
    if caller != platform_owner {
        packs.retain(|p| p.retired_at.is_none());
    }
    Ok(packs)
}

/// One store of a pack's subtree, as `GET /api/platform/packs/{key}/{version}/stores` lists it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PackStore {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub kind: StoreKind,
    pub name: String,
    /// Store names from the pack root (included) down to this store.
    pub path: Vec<String>,
    pub embed_model: String,
    pub chunk_count: i64,
}

type PackStoreRow = (Uuid, Option<Uuid>, String, String, Vec<String>, String, i64);

/// The subtree of the store `root_id` (a pack root), the root first, ordered by name path. Call it
/// only for a pack the caller may read ([`readable_packs`]).
pub async fn pack_stores<'e, E: sqlx::PgExecutor<'e>>(
    db: E,
    root_id: Uuid,
) -> Result<Vec<PackStore>> {
    let rows: Vec<PackStoreRow> = sqlx::query_as(
        "WITH RECURSIVE down AS ( \
           SELECT id, parent_id, kind, name, ARRAY[name::text] AS path, embed_model, \
                  chunk_count \
           FROM stores WHERE id = $1 \
           UNION ALL \
           SELECT s.id, s.parent_id, s.kind, s.name, down.path || s.name::text, \
                  s.embed_model, s.chunk_count \
           FROM stores s JOIN down ON s.parent_id = down.id) \
         SELECT id, parent_id, kind::text, name::text, path, embed_model::text, chunk_count \
         FROM down ORDER BY path",
    )
    .bind(root_id)
    .fetch_all(db)
    .await?;
    rows.into_iter()
        .map(
            |(id, parent_id, kind, name, path, embed_model, chunk_count)| {
                Ok(PackStore {
                    id,
                    parent_id,
                    kind: StoreKind::parse(&kind)?,
                    name,
                    path,
                    embed_model,
                    chunk_count,
                })
            },
        )
        .collect()
}

/// `PackPublished` (409) when `store_id` is in the subtree of a published pack's root (retired or
/// not). Takes a share lock on the pack row: called inside a write transaction it holds off a
/// concurrent publish until the write commits, and a write after the publish sees it. Call it
/// only once `store_id` is known to be the caller's (else the 409 would reveal another tenant's
/// store).
pub async fn assert_writable<'e, E: sqlx::PgExecutor<'e>>(db: E, store_id: Uuid) -> Result<()> {
    let row: Option<(String, i32, bool)> = sqlx::query_as(
        "SELECT p.key, p.version, p.published_at IS NOT NULL \
         FROM platform_packs p \
         WHERE p.root_id = ( \
           WITH RECURSIVE up AS ( \
             SELECT id, parent_id FROM stores WHERE id = $1 \
             UNION ALL \
             SELECT s.id, s.parent_id FROM stores s JOIN up ON s.id = up.parent_id) \
           SELECT id FROM up WHERE parent_id IS NULL LIMIT 1) \
         FOR SHARE OF p",
    )
    .bind(store_id)
    .fetch_optional(db)
    .await?;
    match row {
        Some((key, version, true)) => Err(StrataError::PackPublished { key, version }),
        _ => Ok(()),
    }
}

/// Store-delete guard, in the delete transaction before the stores go: deleting inside a
/// published pack is `PackPublished`, except the root store of a RETIRED pack (the curator's way
/// to drop it). Deleting the root of a draft or retired pack removes the pack row with it.
pub async fn release_for_delete(conn: &mut PgConnection, store_id: Uuid) -> Result<()> {
    let row: Option<(String, i32, Uuid, bool, bool)> = sqlx::query_as(
        "SELECT p.key, p.version, p.root_id, p.published_at IS NOT NULL, p.retired_at IS NOT NULL \
         FROM platform_packs p \
         WHERE p.root_id = ( \
           WITH RECURSIVE up AS ( \
             SELECT id, parent_id FROM stores WHERE id = $1 \
             UNION ALL \
             SELECT s.id, s.parent_id FROM stores s JOIN up ON s.id = up.parent_id) \
           SELECT id FROM up WHERE parent_id IS NULL LIMIT 1) \
         FOR UPDATE OF p",
    )
    .bind(store_id)
    .fetch_optional(&mut *conn)
    .await?;
    let Some((key, version, root_id, published, retired)) = row else {
        return Ok(());
    };
    let is_root = root_id == store_id;
    if published && !(retired && is_root) {
        return Err(StrataError::PackPublished { key, version });
    }
    if is_root {
        sqlx::query("DELETE FROM platform_packs WHERE root_id = $1")
            .bind(root_id)
            .execute(&mut *conn)
            .await?;
    }
    Ok(())
}

/// Registers `root_id` (a root store of `owner`, the platform owner) as draft `key@version`.
/// 404 when the store is not `owner`'s; 422 `invalid_target` when it has a parent; 409
/// `duplicate_name` when `key@version` exists or the store already roots a pack.
pub async fn register_pack(
    conn: &mut PgConnection,
    owner: Tenant,
    key: &str,
    version: i32,
    root_id: Uuid,
) -> Result<PlatformPack> {
    check_key("pack key", key)?;
    if version < 1 {
        return Err(StrataError::InvalidInput(format!(
            "pack version must be >= 1, got {version}"
        )));
    }
    // the root must be the platform owner's (another tenant's store is 404, as everywhere)
    let root = stores::get_store_for_update(&mut *conn, owner, root_id).await?;
    if root.parent_id.is_some() {
        return Err(StrataError::InvalidTarget(format!(
            "store {root_id} has a parent; a pack's root must be a root store"
        )));
    }
    let row: PackRow = sqlx::query_as(
        "INSERT INTO platform_packs (key, version, root_id) VALUES ($1, $2, $3) \
         RETURNING key, version, root_id, published_at, retired_at",
    )
    .bind(key)
    .bind(version)
    .bind(root_id)
    .fetch_one(&mut *conn)
    .await
    .map_err(|e| {
        if let sqlx::Error::Database(db) = &e {
            match (db.code().as_deref(), db.constraint()) {
                (Some("23505"), Some("platform_packs_root_id_key")) => {
                    return StrataError::DuplicateName(format!(
                        "store {root_id} is already the root of a pack"
                    ));
                }
                (Some("23505"), _) => {
                    return StrataError::DuplicateName(format!("pack {key}@{version}"));
                }
                (Some("P0001"), _) => return StrataError::InvalidTarget(db.message().to_string()),
                _ => {}
            }
        }
        StrataError::from(e)
    })?;
    Ok(pack_of(row))
}

/// The pack `key@version`, any state; `NotFound` otherwise.
pub async fn get_pack<'e, E: sqlx::PgExecutor<'e>>(
    db: E,
    key: &str,
    version: i32,
) -> Result<PlatformPack> {
    let row: Option<PackRow> = sqlx::query_as(
        "SELECT key, version, root_id, published_at, retired_at FROM platform_packs \
         WHERE key = $1 AND version = $2",
    )
    .bind(key)
    .bind(version)
    .fetch_optional(db)
    .await?;
    row.map(pack_of)
        .ok_or_else(|| StrataError::NotFound(format!("pack {key}@{version}")))
}

/// Publishes (freezes) `key@version`. Idempotent: an already published pack keeps its
/// `published_at`. Waits for writes holding [`assert_writable`]'s share lock to commit.
pub async fn publish_pack<'e, E: sqlx::PgExecutor<'e>>(
    db: E,
    key: &str,
    version: i32,
) -> Result<PlatformPack> {
    let row: Option<PackRow> = sqlx::query_as(
        "UPDATE platform_packs SET published_at = COALESCE(published_at, now()) \
         WHERE key = $1 AND version = $2 \
         RETURNING key, version, root_id, published_at, retired_at",
    )
    .bind(key)
    .bind(version)
    .fetch_optional(db)
    .await?;
    row.map(pack_of)
        .ok_or_else(|| StrataError::NotFound(format!("pack {key}@{version}")))
}

/// Retires published `key@version` (hidden from listings; its pins keep the root id).
/// Idempotent. A draft cannot be retired: 422 `invalid_target`.
pub async fn retire_pack(conn: &mut PgConnection, key: &str, version: i32) -> Result<PlatformPack> {
    let pack = get_pack(&mut *conn, key, version).await?;
    if pack.published_at.is_none() {
        return Err(StrataError::InvalidTarget(format!(
            "pack {key}@{version} is a draft; only a published pack can be retired"
        )));
    }
    let row: Option<PackRow> = sqlx::query_as(
        "UPDATE platform_packs SET retired_at = COALESCE(retired_at, now()) \
         WHERE key = $1 AND version = $2 AND published_at IS NOT NULL \
         RETURNING key, version, root_id, published_at, retired_at",
    )
    .bind(key)
    .bind(version)
    .fetch_optional(&mut *conn)
    .await?;
    row.map(pack_of)
        .ok_or_else(|| StrataError::NotFound(format!("pack {key}@{version}")))
}

/// `tenant`'s exclusions, by prefix.
pub async fn list_exclusions<'e, E: sqlx::PgExecutor<'e>>(
    db: E,
    tenant: Tenant,
) -> Result<Vec<PlatformExclusion>> {
    let rows: Vec<(String, String, DateTime<Utc>)> = sqlx::query_as(
        "SELECT key_prefix, reason, created_at FROM platform_exclusions \
         WHERE tenant = $1 ORDER BY key_prefix",
    )
    .bind(tenant.id())
    .fetch_all(db)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(key_prefix, reason, created_at)| PlatformExclusion {
            key_prefix,
            reason,
            created_at,
        })
        .collect())
}

/// Inserts or replaces (the reason of) `tenant`'s exclusion from `key_prefix`.
pub async fn upsert_exclusion<'e, E: sqlx::PgExecutor<'e>>(
    db: E,
    tenant: Tenant,
    key_prefix: &str,
    reason: &str,
) -> Result<PlatformExclusion> {
    check_key("key_prefix", key_prefix)?;
    let reason = reason.trim();
    if reason.is_empty() {
        return Err(StrataError::InvalidInput(
            "an exclusion needs a reason".to_string(),
        ));
    }
    let (key_prefix, reason, created_at): (String, String, DateTime<Utc>) = sqlx::query_as(
        "INSERT INTO platform_exclusions (tenant, key_prefix, reason) VALUES ($1, $2, $3) \
         ON CONFLICT (tenant, key_prefix) DO UPDATE SET reason = EXCLUDED.reason \
         RETURNING key_prefix, reason, created_at",
    )
    .bind(tenant.id())
    .bind(key_prefix)
    .bind(reason)
    .fetch_one(db)
    .await?;
    Ok(PlatformExclusion {
        key_prefix,
        reason,
        created_at,
    })
}

/// Removes `tenant`'s exclusion from exactly `key_prefix`; `NotFound` when there is none.
pub async fn delete_exclusion<'e, E: sqlx::PgExecutor<'e>>(
    db: E,
    tenant: Tenant,
    key_prefix: &str,
) -> Result<()> {
    let done = sqlx::query("DELETE FROM platform_exclusions WHERE tenant = $1 AND key_prefix = $2")
        .bind(tenant.id())
        .bind(key_prefix)
        .execute(db)
        .await?;
    if done.rows_affected() == 0 {
        return Err(StrataError::NotFound(format!("exclusion {key_prefix:?}")));
    }
    Ok(())
}
