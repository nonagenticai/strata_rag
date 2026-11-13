//! Document and chunk rows.

use pgvector::Vector;
use serde_json::{Map, Value};
use sqlx::postgres::PgRow;
use sqlx::types::Json;
use sqlx::types::chrono::{DateTime, Utc};
use sqlx::{PgConnection, Postgres, QueryBuilder, Row};
use uuid::Uuid;

use crate::db::stores::map_write_err;
use crate::error::{Result, StrataError};
use crate::tenant::Tenant;
use crate::text::Chunk;

/// Rows per multi-row INSERT: 7 binds each → 7,000 binds, well under Postgres' 65,535.
const CHUNK_INSERT_BATCH: usize = 1000;

/// A document's caller-side provenance (migration 0004): the source it came from, the sync run
/// that last delivered it, and the source's mapped fields. Validated by the caller
/// (`crate::ingest::SourceTag::validate`); the schema only enforces `metadata` being an object and
/// `source_run` needing a `source_id`.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct SourceFields {
    pub source_id: Option<String>,
    pub source_run: Option<String>,
    /// `None`: not given (a new document stores `{}`, a refreshed one keeps its own).
    pub metadata: Option<Map<String, Value>>,
    /// Topic keys this document covers (platform documents; migration 0022). `None` like
    /// `metadata`: a new document stores `{}`, a refreshed one keeps its own.
    pub topic_keys: Option<Vec<String>>,
    /// Topic keys this document overrides (tenant documents): platform documents carrying one of
    /// them are shadowed in the caller's queries (spec platform-knowledge §6). `None` like
    /// `topic_keys`.
    pub supersedes: Option<Vec<String>>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentRow {
    pub id: Uuid,
    pub store_id: Uuid,
    pub source: String,
    pub title: Option<String>,
    pub token_count: i32,
    pub chunk_count: i64,
    pub source_id: Option<String>,
    pub source_run: Option<String>,
    pub metadata: Map<String, Value>,
    pub topic_keys: Vec<String>,
    pub supersedes: Vec<String>,
}

/// The `DocumentRow` columns of `documents d`, plus its chunk count. A macro so `concat!` keeps
/// the SQL a `&'static str` literal (sqlx refuses runtime-built SQL).
macro_rules! document_row_cols {
    () => {
        "d.id, d.store_id, d.source, d.title, d.token_count, d.source_id, d.source_run, \
         d.metadata, d.topic_keys, d.supersedes, \
         (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count"
    };
}

fn document_row(r: &PgRow) -> Result<DocumentRow> {
    let Json(metadata): Json<Map<String, Value>> = r.try_get("metadata")?;
    Ok(DocumentRow {
        id: r.try_get("id")?,
        store_id: r.try_get("store_id")?,
        source: r.try_get("source")?,
        title: r.try_get("title")?,
        token_count: r.try_get("token_count")?,
        chunk_count: r.try_get("chunk_count")?,
        source_id: r.try_get("source_id")?,
        source_run: r.try_get("source_run")?,
        metadata,
        topic_keys: r.try_get("topic_keys")?,
        supersedes: r.try_get("supersedes")?,
    })
}

pub(crate) async fn find_document_by_hash(
    conn: &mut PgConnection,
    store_id: Uuid,
    hash: &[u8],
) -> Result<Option<DocumentRow>> {
    let row = sqlx::query(concat!(
        "SELECT ",
        document_row_cols!(),
        " FROM documents d WHERE d.store_id = $1 AND d.content_hash = $2"
    ))
    .bind(store_id)
    .bind(hash)
    .fetch_optional(&mut *conn)
    .await?;
    row.as_ref().map(document_row).transpose()
}

/// A re-delivered document of the same source: takes the new `source` label, and the run,
/// metadata (replaced whole), topic keys and supersedes (each replaced whole) and title when
/// given; an absent one keeps the stored value (a re-delivery never erases a run). Content, chunks
/// and vectors are untouched. `None`: the document is gone (deleted concurrently) or not
/// `tenant`'s. Callers hold the leaf lock.
pub(crate) async fn update_document_source(
    conn: &mut PgConnection,
    tenant: Tenant,
    id: Uuid,
    source: &str,
    title: Option<&str>,
    origin: &SourceFields,
) -> Result<Option<DocumentRow>> {
    let row = sqlx::query(concat!(
        "UPDATE documents d SET source = $5, source_run = COALESCE($2, d.source_run), \
         metadata = COALESCE($3, d.metadata), title = COALESCE($4, d.title), \
         topic_keys = COALESCE($7::text[], d.topic_keys), \
         supersedes = COALESCE($8::text[], d.supersedes) \
         FROM stores s WHERE d.id = $1 AND s.id = d.store_id AND s.tenant = $6 RETURNING ",
        document_row_cols!()
    ))
    .bind(id)
    .bind(origin.source_run.as_deref())
    .bind(origin.metadata.as_ref().map(Json))
    .bind(title)
    .bind(source)
    .bind(tenant.id())
    .bind(origin.topic_keys.as_deref())
    .bind(origin.supersedes.as_deref())
    .fetch_optional(&mut *conn)
    .await?;
    row.as_ref().map(document_row).transpose()
}

pub(crate) async fn insert_document(
    conn: &mut PgConnection,
    store_id: Uuid,
    source: &str,
    title: Option<&str>,
    hash: &[u8],
    token_count: i32,
    origin: &SourceFields,
) -> Result<Uuid> {
    let id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO documents \
           (id, store_id, source, title, content_hash, token_count, source_id, source_run, metadata, \
            topic_keys, supersedes) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
    )
    .bind(id)
    .bind(store_id)
    .bind(source)
    .bind(title)
    .bind(hash)
    .bind(token_count)
    .bind(origin.source_id.as_deref())
    .bind(origin.source_run.as_deref())
    .bind(Json(origin.metadata.clone().unwrap_or_default()))
    .bind(origin.topic_keys.as_deref().unwrap_or_default())
    .bind(origin.supersedes.as_deref().unwrap_or_default())
    .execute(&mut *conn)
    .await
    .map_err(|e| {
        map_write_err(
            e,
            &format!("document with identical content in store {store_id}"),
        )
    })?;
    Ok(id)
}

pub(crate) async fn insert_chunks(
    conn: &mut PgConnection,
    document_id: Uuid,
    store_id: Uuid,
    chunks: &[Chunk],
    embeddings: &[Vec<f32>],
) -> Result<()> {
    if chunks.len() != embeddings.len() {
        return Err(StrataError::Model(format!(
            "{} embeddings for {} chunks",
            embeddings.len(),
            chunks.len()
        )));
    }
    for (cs, es) in chunks
        .chunks(CHUNK_INSERT_BATCH)
        .zip(embeddings.chunks(CHUNK_INSERT_BATCH))
    {
        let mut qb = QueryBuilder::<Postgres>::new(
            "INSERT INTO chunks (id, document_id, store_id, ord, text, token_count, page_start, page_end, \
             embedding) ",
        );
        qb.push_values(cs.iter().zip(es.iter()), |mut b, (c, e)| {
            b.push_bind(Uuid::now_v7())
                .push_bind(document_id)
                .push_bind(store_id)
                .push_bind(c.ord)
                .push_bind(c.text.clone())
                .push_bind(c.token_count)
                .push_bind(c.page_start)
                .push_bind(c.page_end)
                .push_bind(Vector::from(e.clone()));
        });
        qb.build().execute(&mut *conn).await?;
    }
    Ok(())
}

/// The leaf a document of `tenant` lives in (documents never move between stores). Another
/// tenant's document is `NotFound`, like a missing one.
pub(crate) async fn document_store_id(
    conn: &mut PgConnection,
    tenant: Tenant,
    id: Uuid,
) -> Result<Uuid> {
    let store_id: Option<Uuid> = sqlx::query_scalar(
        "SELECT d.store_id FROM documents d JOIN stores s ON s.id = d.store_id \
         WHERE d.id = $1 AND s.tenant = $2",
    )
    .bind(id)
    .bind(tenant.id())
    .fetch_optional(&mut *conn)
    .await?;
    store_id.ok_or_else(|| StrataError::NotFound(format!("document {id}")))
}

/// The shadow set's stored half (spec platform-knowledge §6): the distinct `supersedes` keys of
/// `tenant`'s documents in `leaf_ids`, sorted. Only `tenant`'s own stores count, so a platform leaf
/// in `leaf_ids` contributes nothing for a tenant caller.
pub async fn shadow_keys<'e, E: sqlx::PgExecutor<'e>>(
    db: E,
    tenant: Tenant,
    leaf_ids: &[Uuid],
) -> Result<Vec<String>> {
    if leaf_ids.is_empty() {
        return Ok(Vec::new());
    }
    let keys: Vec<String> = sqlx::query_scalar(
        "SELECT DISTINCT k.key FROM documents d JOIN stores s ON s.id = d.store_id \
         CROSS JOIN LATERAL unnest(d.supersedes) AS k(key) \
         WHERE d.store_id = ANY($1) AND s.tenant = $2 AND d.supersedes <> '{}' \
         ORDER BY k.key",
    )
    .bind(leaf_ids)
    .bind(tenant.id())
    .fetch_all(db)
    .await?;
    Ok(keys)
}

/// Deletes `tenant`'s document (its chunks go by FK cascade). Returns its store_id; another
/// tenant's document is `NotFound` and untouched. Callers lock the leaf first
/// (`stores::get_store_for_update`), the order ingest uses.
pub(crate) async fn delete_document(
    conn: &mut PgConnection,
    tenant: Tenant,
    id: Uuid,
) -> Result<Uuid> {
    let store_id: Option<Uuid> = sqlx::query_scalar(
        "DELETE FROM documents d USING stores s \
         WHERE d.id = $1 AND s.id = d.store_id AND s.tenant = $2 RETURNING d.store_id",
    )
    .bind(id)
    .bind(tenant.id())
    .fetch_optional(&mut *conn)
    .await?;
    store_id.ok_or_else(|| StrataError::NotFound(format!("document {id}")))
}

/// `tenant`'s documents of `source_id`, optionally in one store: (all of them, those whose run is
/// `run`). With `run = None` the second count is 0.
pub(crate) async fn source_counts(
    conn: &mut PgConnection,
    tenant: Tenant,
    source_id: &str,
    store_id: Option<Uuid>,
    run: Option<&str>,
) -> Result<(i64, i64)> {
    let counts: (i64, i64) = sqlx::query_as(
        "SELECT count(*), count(*) FILTER (WHERE d.source_run = $3) \
         FROM documents d JOIN stores s ON s.id = d.store_id \
         WHERE d.source_id = $1 AND ($2::uuid IS NULL OR d.store_id = $2) AND s.tenant = $4",
    )
    .bind(source_id)
    .bind(store_id)
    .bind(run)
    .bind(tenant.id())
    .fetch_one(&mut *conn)
    .await?;
    Ok(counts)
}

/// `tenant`'s leaves holding a document of `source_id` that a prune keeping only run `except_run`
/// (none when `None`) would delete, ordered by id.
pub(crate) async fn source_prune_leaves(
    conn: &mut PgConnection,
    tenant: Tenant,
    source_id: &str,
    store_id: Option<Uuid>,
    except_run: Option<&str>,
) -> Result<Vec<Uuid>> {
    let leaves: Vec<Uuid> = sqlx::query_scalar(
        "SELECT DISTINCT d.store_id FROM documents d JOIN stores s ON s.id = d.store_id \
         WHERE d.source_id = $1 AND ($2::uuid IS NULL OR d.store_id = $2) \
           AND ($3::text IS NULL OR d.source_run IS DISTINCT FROM $3) \
           AND s.tenant = $4 \
         ORDER BY d.store_id",
    )
    .bind(source_id)
    .bind(store_id)
    .bind(except_run)
    .bind(tenant.id())
    .fetch_all(&mut *conn)
    .await?;
    Ok(leaves)
}

/// Deletes `source_id`'s documents in one leaf, except those of run `except_run` (all of them when
/// `None`); chunks go by FK cascade. Returns how many were deleted (0 when the leaf is not
/// `tenant`'s). Callers lock the leaf first.
pub(crate) async fn delete_source_documents(
    conn: &mut PgConnection,
    tenant: Tenant,
    store_id: Uuid,
    source_id: &str,
    except_run: Option<&str>,
) -> Result<u64> {
    let done = sqlx::query(
        "DELETE FROM documents d USING stores s \
         WHERE d.store_id = $1 AND d.source_id = $2 \
           AND ($3::text IS NULL OR d.source_run IS DISTINCT FROM $3) \
           AND s.id = d.store_id AND s.tenant = $4",
    )
    .bind(store_id)
    .bind(source_id)
    .bind(except_run)
    .bind(tenant.id())
    .execute(&mut *conn)
    .await?;
    Ok(done.rows_affected())
}

/// One row of `GET /api/stores/{id}/documents`: every column of `documents` but the raw hash
/// bytes (hex instead), plus the document's chunk count.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentListItem {
    pub id: Uuid,
    pub store_id: Uuid,
    pub source: String,
    pub title: Option<String>,
    /// Hex sha256 of the normalised text (the store's dedup key).
    pub content_hash: String,
    pub token_count: i32,
    pub chunk_count: i64,
    pub source_id: Option<String>,
    pub source_run: Option<String>,
    pub metadata: Map<String, Value>,
    /// Topic keys the document covers; `[]` when unset (same as on `DocumentRow`).
    pub topic_keys: Vec<String>,
    /// Topic keys the document overrides; `[]` when unset (same as on `DocumentRow`).
    pub supersedes: Vec<String>,
    /// RFC 3339, UTC, microsecond precision.
    pub created_at: String,
    #[serde(skip)]
    pub created_at_ts: DateTime<Utc>,
}

/// Keyset position: the `(created_at, id)` of the last row of the previous page.
pub type DocumentCursor = (DateTime<Utc>, Uuid);

/// A store's documents, newest first (`created_at DESC, id DESC`), strictly after `after`; only
/// `source_id`'s when given (the cursor works the same: the filter holds for every page).
/// `WHERE store_id` holds for every page, so a cursor taken from another store only positions
/// the page, it never returns that store's rows. Chunk counts come from one grouped scan of the
/// store's chunks (`chunks_store_idx`), not a per-row subquery.
pub(crate) async fn list_documents(
    conn: &mut PgConnection,
    store_id: Uuid,
    limit: i64,
    after: Option<DocumentCursor>,
    source_id: Option<&str>,
) -> Result<Vec<DocumentListItem>> {
    let (after_ts, after_id) = after.map_or((None, None), |(t, id)| (Some(t), Some(id)));
    let rows = sqlx::query(
        "WITH page AS ( \
           SELECT d.id, d.store_id, d.source, d.title, d.content_hash, d.token_count, d.created_at, \
                  d.source_id, d.source_run, d.metadata, d.topic_keys, d.supersedes \
           FROM documents d \
           WHERE d.store_id = $1 \
             AND ($5::text IS NULL OR d.source_id = $5) \
             AND ($2::timestamptz IS NULL OR (d.created_at, d.id) < ($2::timestamptz, $3::uuid)) \
           ORDER BY d.created_at DESC, d.id DESC \
           LIMIT $4) \
         SELECT p.*, COALESCE(c.n, 0) AS chunk_count \
         FROM page p \
         LEFT JOIN (SELECT document_id, count(*) AS n FROM chunks \
                    WHERE store_id = $1 AND document_id IN (SELECT id FROM page) \
                    GROUP BY document_id) c ON c.document_id = p.id \
         ORDER BY p.created_at DESC, p.id DESC",
    )
    .bind(store_id)
    .bind(after_ts)
    .bind(after_id)
    .bind(limit)
    .bind(source_id)
    .fetch_all(&mut *conn)
    .await?;
    rows.iter()
        .map(|r| {
            let hash: Vec<u8> = r.try_get("content_hash")?;
            let created: DateTime<Utc> = r.try_get("created_at")?;
            let Json(metadata): Json<Map<String, Value>> = r.try_get("metadata")?;
            Ok(DocumentListItem {
                id: r.try_get("id")?,
                store_id: r.try_get("store_id")?,
                source: r.try_get("source")?,
                title: r.try_get("title")?,
                content_hash: hash.iter().map(|b| format!("{b:02x}")).collect(),
                token_count: r.try_get("token_count")?,
                chunk_count: r.try_get("chunk_count")?,
                source_id: r.try_get("source_id")?,
                source_run: r.try_get("source_run")?,
                metadata,
                topic_keys: r.try_get("topic_keys")?,
                supersedes: r.try_get("supersedes")?,
                created_at: created.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string(),
                created_at_ts: created,
            })
        })
        .collect()
}

/// Opaque page cursor: hex of `created_at` (µs since the epoch, big-endian i64) ‖ id (16 bytes).
pub fn encode_cursor((ts, id): DocumentCursor) -> String {
    let mut bytes = ts.timestamp_micros().to_be_bytes().to_vec();
    bytes.extend_from_slice(id.as_bytes());
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn decode_cursor(s: &str) -> Result<DocumentCursor> {
    let invalid = || StrataError::InvalidInput("invalid cursor".to_string());
    if s.len() != 48 || !s.is_ascii() {
        return Err(invalid());
    }
    let bytes = (0..24)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| invalid()))
        .collect::<Result<Vec<u8>>>()?;
    let micros = i64::from_be_bytes(bytes[..8].try_into().map_err(|_| invalid())?);
    let ts = DateTime::from_timestamp_micros(micros).ok_or_else(invalid)?;
    let id = Uuid::from_slice(&bytes[8..]).map_err(|_| invalid())?;
    Ok((ts, id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_round_trips_and_rejects_garbage() {
        let ts = DateTime::from_timestamp_micros(1_760_000_000_123_456).unwrap();
        let id = Uuid::now_v7();
        let c = encode_cursor((ts, id));
        assert_eq!(c.len(), 48);
        assert_eq!(decode_cursor(&c).unwrap(), (ts, id));
        for bad in [
            "",
            "zz",
            &c[..46],
            &format!("{c}00"),
            &c.replace('0', "g"),
            "é".repeat(24).as_str(),
        ] {
            let err = decode_cursor(bad).unwrap_err();
            assert_eq!(err.code(), "invalid_input", "{bad}");
        }
    }
}
