//! `tenant_providers` rows (migration 0006) and `store_providers` rows (migrations 0016, 0017). Rows are
//! always read and written for ONE tenant: a store row through its store's tenant.

use sqlx::Row;
use sqlx::types::chrono::{DateTime, Utc};

use crate::error::Result;
use crate::providers::crypto::{KEY_VERSION, Sealed};
use crate::providers::{Capability, ProviderSpec};
use crate::tenant::Tenant;

/// One stored registration. `Debug` shows whether a key is stored, never its bytes.
#[derive(Clone)]
pub struct ProviderRow {
    pub capability: String,
    pub base_url: String,
    pub dialect: String,
    pub model: String,
    pub model_id: Option<String>,
    pub dims: Option<i32>,
    pub regions: Vec<String>,
    pub zdr: bool,
    /// Chat only (migration 0008).
    pub thinking: Option<bool>,
    pub temperature: Option<f64>,
    pub timeout_ms: i32,
    pub concurrency: i32,
    pub sealed_key: Option<Sealed>,
    pub updated_at: DateTime<Utc>,
}

impl std::fmt::Debug for ProviderRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderRow")
            .field("capability", &self.capability)
            .field("model", &self.model)
            .field("has_api_key", &self.sealed_key.is_some())
            .field("updated_at", &self.updated_at)
            .finish_non_exhaustive()
    }
}

const COLUMNS: &str = "capability, base_url, dialect, model, model_id, dims, regions, zdr, thinking, temperature, \
                       timeout_ms, concurrency, api_key_ct, api_key_nonce, updated_at";

fn from_row(row: &sqlx::postgres::PgRow) -> Result<ProviderRow> {
    let ct: Option<Vec<u8>> = row.try_get("api_key_ct")?;
    let nonce: Option<Vec<u8>> = row.try_get("api_key_nonce")?;
    Ok(ProviderRow {
        capability: row.try_get("capability")?,
        base_url: row.try_get("base_url")?,
        dialect: row.try_get("dialect")?,
        model: row.try_get("model")?,
        model_id: row.try_get("model_id")?,
        dims: row.try_get("dims")?,
        regions: row.try_get("regions")?,
        zdr: row.try_get("zdr")?,
        thinking: row.try_get("thinking")?,
        temperature: row.try_get("temperature")?,
        timeout_ms: row.try_get("timeout_ms")?,
        concurrency: row.try_get("concurrency")?,
        sealed_key: match (ct, nonce) {
            (Some(ciphertext), Some(nonce)) => Some(Sealed { ciphertext, nonce }),
            _ => None,
        },
        updated_at: row.try_get("updated_at")?,
    })
}

/// All of `tenant`'s registrations.
pub async fn list<'e>(exec: impl sqlx::PgExecutor<'e>, tenant: Tenant) -> Result<Vec<ProviderRow>> {
    let rows = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {COLUMNS} FROM tenant_providers WHERE tenant = $1 ORDER BY capability"
    )))
    .bind(tenant.id())
    .fetch_all(exec)
    .await?;
    rows.iter().map(from_row).collect()
}

/// `tenant`'s registration for `cap`, if any.
pub async fn get<'e>(
    exec: impl sqlx::PgExecutor<'e>,
    tenant: Tenant,
    cap: Capability,
) -> Result<Option<ProviderRow>> {
    let row = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {COLUMNS} FROM tenant_providers WHERE tenant = $1 AND capability = $2"
    )))
    .bind(tenant.id())
    .bind(cap.as_str())
    .fetch_optional(exec)
    .await?;
    row.as_ref().map(from_row).transpose()
}

/// `(capability, updated_at)` of each of `tenant`'s registrations: the cheap per-request check
/// that a cached client is still current (also when another replica took the PUT).
pub async fn versions<'e>(
    exec: impl sqlx::PgExecutor<'e>,
    tenant: Tenant,
) -> Result<Vec<(String, DateTime<Utc>)>> {
    Ok(
        sqlx::query_as("SELECT capability, updated_at FROM tenant_providers WHERE tenant = $1")
            .bind(tenant.id())
            .fetch_all(exec)
            .await?,
    )
}

/// Inserts or replaces `tenant`'s `cap` registration; `updated_at` moves on every write.
pub async fn upsert(
    conn: &mut sqlx::PgConnection,
    tenant: Tenant,
    cap: Capability,
    spec: &ProviderSpec,
    key: Option<&Sealed>,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO tenant_providers (tenant, capability, base_url, dialect, model, model_id, \
                                       dims, regions, zdr, timeout_ms, concurrency, api_key_ct, \
                                       api_key_nonce, key_version, thinking, temperature, \
                                       updated_at) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, \
                 clock_timestamp()) \
         ON CONFLICT (tenant, capability) DO UPDATE SET \
           base_url = EXCLUDED.base_url, dialect = EXCLUDED.dialect, model = EXCLUDED.model, \
           model_id = EXCLUDED.model_id, dims = EXCLUDED.dims, regions = EXCLUDED.regions, \
           zdr = EXCLUDED.zdr, timeout_ms = EXCLUDED.timeout_ms, \
           concurrency = EXCLUDED.concurrency, api_key_ct = EXCLUDED.api_key_ct, \
           api_key_nonce = EXCLUDED.api_key_nonce, key_version = EXCLUDED.key_version, \
           thinking = EXCLUDED.thinking, temperature = EXCLUDED.temperature, updated_at = EXCLUDED.updated_at",
    )
    .bind(tenant.id())
    .bind(cap.as_str())
    .bind(&spec.base_url)
    .bind(&spec.dialect)
    .bind(&spec.model)
    .bind(&spec.model_id)
    .bind(spec.dims)
    .bind(&spec.regions)
    .bind(spec.zdr)
    .bind(i32::try_from(spec.timeout_ms).unwrap_or(i32::MAX))
    .bind(i32::try_from(spec.concurrency).unwrap_or(i32::MAX))
    .bind(key.map(|k| k.ciphertext.as_slice()))
    .bind(key.map(|k| k.nonce.as_slice()))
    .bind(key.map(|_| KEY_VERSION))
    .bind(spec.thinking)
    .bind(spec.temperature)
    .execute(conn)
    .await?;
    Ok(())
}

/// Removes `tenant`'s `cap` registration (no-op when absent).
pub async fn delete(conn: &mut sqlx::PgConnection, tenant: Tenant, cap: Capability) -> Result<()> {
    sqlx::query("DELETE FROM tenant_providers WHERE tenant = $1 AND capability = $2")
        .bind(tenant.id())
        .bind(cap.as_str())
        .execute(conn)
        .await?;
    Ok(())
}

/// True when any tenant or store has a stored (encrypted) key: the process then needs
/// `STRATA_PROVIDER_KEY` to serve them.
pub async fn any_stored_key(pool: &sqlx::PgPool) -> Result<bool> {
    Ok(sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM tenant_providers WHERE api_key_ct IS NOT NULL) \
             OR EXISTS (SELECT 1 FROM store_providers WHERE api_key_ct IS NOT NULL)",
    )
    .fetch_one(pool)
    .await?)
}

/// `store_providers` columns (the chat options since migration 0017).
const STORE_COLUMNS: &str = "p.capability, p.base_url, p.dialect, p.model, p.model_id, p.dims, \
                             p.regions, p.zdr, p.thinking, p.temperature, p.timeout_ms, \
                             p.concurrency, p.api_key_ct, p.api_key_nonce, p.updated_at";

/// `store`'s own registrations (none when `store` is not `tenant`'s).
pub async fn store_list<'e>(
    exec: impl sqlx::PgExecutor<'e>,
    tenant: Tenant,
    store: uuid::Uuid,
) -> Result<Vec<ProviderRow>> {
    let rows = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {STORE_COLUMNS} FROM store_providers p JOIN stores s ON s.id = p.store \
         WHERE p.store = $1 AND s.tenant = $2 ORDER BY p.capability"
    )))
    .bind(store)
    .bind(tenant.id())
    .fetch_all(exec)
    .await?;
    rows.iter().map(from_row).collect()
}

/// `store`'s own registration for `cap`, if any (none when `store` is not `tenant`'s).
pub async fn store_get<'e>(
    exec: impl sqlx::PgExecutor<'e>,
    tenant: Tenant,
    store: uuid::Uuid,
    cap: Capability,
) -> Result<Option<ProviderRow>> {
    let row = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {STORE_COLUMNS} FROM store_providers p JOIN stores s ON s.id = p.store \
         WHERE p.store = $1 AND s.tenant = $2 AND p.capability = $3"
    )))
    .bind(store)
    .bind(tenant.id())
    .bind(cap.as_str())
    .fetch_optional(exec)
    .await?;
    row.as_ref().map(from_row).transpose()
}

/// Per capability, the registration nearest to `store` on its path up to the root (the store's
/// own first, then its parent's, ...): `(capability, store holding the row, updated_at)`. Empty
/// when no store on the path has one, or `store` is not `tenant`'s.
pub async fn lineage_versions<'e>(
    exec: impl sqlx::PgExecutor<'e>,
    tenant: Tenant,
    store: uuid::Uuid,
) -> Result<Vec<(String, uuid::Uuid, DateTime<Utc>)>> {
    Ok(sqlx::query_as(
        "WITH RECURSIVE up (id, parent_id, depth) AS ( \
             SELECT id, parent_id, 0 FROM stores WHERE id = $1 AND tenant = $2 \
             UNION ALL \
             SELECT s.id, s.parent_id, up.depth + 1 FROM stores s JOIN up ON s.id = up.parent_id \
             WHERE up.depth < 10000 \
         ) \
         SELECT DISTINCT ON (p.capability) p.capability, p.store, p.updated_at \
         FROM up JOIN store_providers p ON p.store = up.id \
         ORDER BY p.capability, up.depth",
    )
    .bind(store)
    .bind(tenant.id())
    .fetch_all(exec)
    .await?)
}

/// Inserts or replaces `store`'s `cap` registration; `updated_at` moves on every write. The
/// caller holds `store`'s row lock and has checked its tenant.
pub async fn store_upsert(
    conn: &mut sqlx::PgConnection,
    store: uuid::Uuid,
    cap: Capability,
    spec: &ProviderSpec,
    key: Option<&Sealed>,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO store_providers (store, capability, base_url, dialect, model, model_id, \
                                      dims, regions, zdr, timeout_ms, concurrency, api_key_ct, \
                                      api_key_nonce, key_version, thinking, temperature, \
                                      updated_at) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, \
                 clock_timestamp()) \
         ON CONFLICT (store, capability) DO UPDATE SET \
           base_url = EXCLUDED.base_url, dialect = EXCLUDED.dialect, model = EXCLUDED.model, \
           model_id = EXCLUDED.model_id, dims = EXCLUDED.dims, regions = EXCLUDED.regions, \
           zdr = EXCLUDED.zdr, timeout_ms = EXCLUDED.timeout_ms, \
           concurrency = EXCLUDED.concurrency, api_key_ct = EXCLUDED.api_key_ct, \
           api_key_nonce = EXCLUDED.api_key_nonce, key_version = EXCLUDED.key_version, \
           thinking = EXCLUDED.thinking, temperature = EXCLUDED.temperature, \
           updated_at = EXCLUDED.updated_at",
    )
    .bind(store)
    .bind(cap.as_str())
    .bind(&spec.base_url)
    .bind(&spec.dialect)
    .bind(&spec.model)
    .bind(&spec.model_id)
    .bind(spec.dims)
    .bind(&spec.regions)
    .bind(spec.zdr)
    .bind(i32::try_from(spec.timeout_ms).unwrap_or(i32::MAX))
    .bind(i32::try_from(spec.concurrency).unwrap_or(i32::MAX))
    .bind(key.map(|k| k.ciphertext.as_slice()))
    .bind(key.map(|k| k.nonce.as_slice()))
    .bind(key.map(|_| KEY_VERSION))
    .bind(spec.thinking)
    .bind(spec.temperature)
    .execute(conn)
    .await?;
    Ok(())
}

/// Removes `store`'s `cap` registration (no-op when absent). The caller has checked its tenant.
pub async fn store_delete(
    conn: &mut sqlx::PgConnection,
    store: uuid::Uuid,
    cap: Capability,
) -> Result<()> {
    sqlx::query("DELETE FROM store_providers WHERE store = $1 AND capability = $2")
        .bind(store)
        .bind(cap.as_str())
        .execute(conn)
        .await?;
    Ok(())
}
