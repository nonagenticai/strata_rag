//! `/api/platform/*`: platform knowledge packs and exclusions (spec
//! 2026-10-20-platform-knowledge-design.md §8). Every route is 404 while `STRATA_PLATFORM_OWNER`
//! is unset; pack writes are the platform owner's only (403 for any other tenant).

use axum::Json;
use axum::extract::rejection::{JsonRejection, PathRejection, QueryRejection};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::handlers::{AppState, json_body, query_params};
use crate::db::platform::{self, PackStore, PlatformExclusion, PlatformPack};
use crate::engine::Engine;
use crate::error::{Result, StrataError};
use crate::tenant::Tenant;

/// The platform owner; 404 while platform knowledge is off (the routes then do not exist).
fn owner(engine: &Engine) -> Result<Tenant> {
    engine
        .cfg
        .platform_owner
        .ok_or_else(|| StrataError::NotFound("platform knowledge is not enabled".to_string()))
}

/// The platform owner when `tenant` is it; 403 for any other tenant.
fn require_owner(engine: &Engine, tenant: Tenant) -> Result<Tenant> {
    let owner = owner(engine)?;
    if tenant != owner {
        return Err(StrataError::Forbidden(
            "only the platform owner manages platform packs".to_string(),
        ));
    }
    Ok(owner)
}

fn pack_path(p: std::result::Result<Path<(String, i32)>, PathRejection>) -> Result<(String, i32)> {
    p.map(|Path(kv)| kv).map_err(|e| {
        StrataError::InvalidInput(format!(
            "invalid pack in path (/{{key}}/{{version}}, key URL-encoded): {}",
            e.body_text()
        ))
    })
}

#[derive(Debug, Serialize)]
pub struct PackList {
    pub packs: Vec<PlatformPack>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterPackBody {
    pub key: String,
    pub version: i32,
    pub root_id: Uuid,
}

#[derive(Debug, Serialize)]
pub struct ExclusionList {
    pub exclusions: Vec<PlatformExclusion>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PutExclusionBody {
    pub key_prefix: String,
    pub reason: String,
}

#[derive(Debug, Deserialize)]
pub struct DeleteExclusionParams {
    pub key_prefix: String,
}

/// `GET /api/platform/packs`: the packs the caller may read, retired ones left out
/// (`db::platform::listed_packs`); the platform owner sees every pack, drafts and retired ones
/// included.
pub async fn list_packs(State(engine): State<AppState>, tenant: Tenant) -> Result<Json<PackList>> {
    let owner = owner(&engine)?;
    Ok(Json(PackList {
        packs: platform::listed_packs(&engine.pool, owner, tenant).await?,
    }))
}

#[derive(Debug, Serialize)]
pub struct PackStores {
    /// `"<key>@<version>"`, as query results label platform hits.
    pub pack: String,
    pub root_id: Uuid,
    pub stores: Vec<PackStore>,
}

/// `GET /api/platform/packs/{key}/{version}/stores`: the pack root's subtree, so a caller can
/// name its leaves in a query's `leaf_ids`. Allowed iff the pack is readable by the caller
/// (`db::platform::readable_packs`, the same rule as a query: a retired pack included, an
/// excluded one not; the owner sees drafts); otherwise 404, never revealing that it exists.
pub async fn list_pack_stores(
    State(engine): State<AppState>,
    tenant: Tenant,
    p: std::result::Result<Path<(String, i32)>, PathRejection>,
) -> Result<Json<PackStores>> {
    let owner = owner(&engine)?;
    let (key, version) = pack_path(p)?;
    let pack = platform::readable_packs(&engine.pool, owner, tenant)
        .await?
        .into_iter()
        .find(|p| p.key == key && p.version == version)
        .ok_or_else(|| StrataError::NotFound(format!("pack {key}@{version}")))?;
    Ok(Json(PackStores {
        pack: format!("{}@{}", pack.key, pack.version),
        root_id: pack.root_id,
        stores: platform::pack_stores(&engine.pool, pack.root_id).await?,
    }))
}

/// `POST /api/platform/packs {key, version, root_id}` → 201 draft. The root must be a root store
/// of the platform owner (404 otherwise, 422 `invalid_target` with a parent); 409
/// `duplicate_name` when `key@version` exists or the store already roots a pack.
pub async fn register_pack(
    State(engine): State<AppState>,
    tenant: Tenant,
    body: std::result::Result<Json<RegisterPackBody>, JsonRejection>,
) -> Result<(StatusCode, Json<PlatformPack>)> {
    let owner = require_owner(&engine, tenant)?;
    let b = json_body(body)?;
    let mut tx = engine.pool.begin().await?;
    let pack = platform::register_pack(&mut tx, owner, b.key.trim(), b.version, b.root_id).await?;
    tx.commit().await?;
    tracing::info!(key = %pack.key, version = pack.version, root = %pack.root_id, "platform pack registered");
    Ok((StatusCode::CREATED, Json(pack)))
}

/// `POST /api/platform/packs/{key}/{version}/publish`: freezes the pack (idempotent).
pub async fn publish_pack(
    State(engine): State<AppState>,
    tenant: Tenant,
    p: std::result::Result<Path<(String, i32)>, PathRejection>,
) -> Result<Json<PlatformPack>> {
    require_owner(&engine, tenant)?;
    let (key, version) = pack_path(p)?;
    let pack = platform::publish_pack(&engine.pool, &key, version).await?;
    tracing::info!(key = %pack.key, version = pack.version, "platform pack published");
    Ok(Json(pack))
}

/// `POST /api/platform/packs/{key}/{version}/retire`: hides a published pack from listings
/// (idempotent; a draft is 422 `invalid_target`).
pub async fn retire_pack(
    State(engine): State<AppState>,
    tenant: Tenant,
    p: std::result::Result<Path<(String, i32)>, PathRejection>,
) -> Result<Json<PlatformPack>> {
    require_owner(&engine, tenant)?;
    let (key, version) = pack_path(p)?;
    let mut tx = engine.pool.begin().await?;
    let pack = platform::retire_pack(&mut tx, &key, version).await?;
    tx.commit().await?;
    tracing::info!(key = %pack.key, version = pack.version, "platform pack retired");
    Ok(Json(pack))
}

/// `GET /api/platform/exclusions`: the caller's own exclusions.
pub async fn list_exclusions(
    State(engine): State<AppState>,
    tenant: Tenant,
) -> Result<Json<ExclusionList>> {
    owner(&engine)?;
    Ok(Json(ExclusionList {
        exclusions: platform::list_exclusions(&engine.pool, tenant).await?,
    }))
}

/// `PUT /api/platform/exclusions {key_prefix, reason}`: excludes the caller from `key_prefix`
/// and every key below it (upsert).
pub async fn put_exclusion(
    State(engine): State<AppState>,
    tenant: Tenant,
    body: std::result::Result<Json<PutExclusionBody>, JsonRejection>,
) -> Result<Json<PlatformExclusion>> {
    owner(&engine)?;
    let b = json_body(body)?;
    Ok(Json(
        platform::upsert_exclusion(&engine.pool, tenant, b.key_prefix.trim(), &b.reason).await?,
    ))
}

/// `DELETE /api/platform/exclusions?key_prefix=`: 204, or 404 when the caller has no exclusion
/// with exactly that prefix.
pub async fn delete_exclusion(
    State(engine): State<AppState>,
    tenant: Tenant,
    params: std::result::Result<Query<DeleteExclusionParams>, QueryRejection>,
) -> Result<StatusCode> {
    owner(&engine)?;
    let p = query_params(params)?;
    platform::delete_exclusion(&engine.pool, tenant, p.key_prefix.trim()).await?;
    Ok(StatusCode::NO_CONTENT)
}
