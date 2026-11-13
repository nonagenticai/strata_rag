//! Ingest path (spec §5): leaf check → normalise → dedupe → chunk → cap
//! pre-check → embed (no locks held) → one transaction that locks the leaf,
//! re-checks dedupe and cap, inserts, and recomputes the ancestors. Also the delete paths
//! (one document, or a source's stale documents) that share the leaf-first lock order.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use sqlx::{PgConnection, Postgres, Transaction};
use uuid::Uuid;

use crate::db::documents::{self, DocumentRow, SourceFields};
use crate::db::stores::{self, StoreKind};
use crate::engine::Engine;
use crate::error::{Result, StrataError};
use crate::tenant::Tenant;
use crate::text::{Chunk, chunk_text, normalize};

/// Longest `source_id` / `source_run`, in chars (after trimming).
pub const MAX_SOURCE_FIELD_CHARS: usize = 200;
/// Largest `metadata`, serialised as compact JSON.
pub const MAX_METADATA_BYTES: usize = 64 * 1024;
/// Longest topic key (`topic_keys`, `supersedes`), in chars.
pub const MAX_TOPIC_KEY_CHARS: usize = 128;
/// Most topic keys per document, in each of `topic_keys` and `supersedes`.
pub const MAX_TOPIC_KEYS: usize = 32;
/// Longest `provenance.source_url` / `provenance.license` / `provenance.retrieved_at`, in chars.
pub const MAX_PROVENANCE_FIELD_CHARS: usize = 2048;
/// The `metadata` key provenance is stored under.
pub const PROVENANCE_KEY: &str = "provenance";

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum IngestOutcome {
    Created(DocumentRow),
    Existing(DocumentRow),
}

/// Caller-supplied provenance of a document, as received (see `validate`).
#[derive(Debug, Clone, Default)]
pub struct SourceTag {
    pub source_id: Option<String>,
    pub source_run: Option<String>,
    /// Must be a JSON object; absent = `{}`.
    pub metadata: Option<Value>,
}

impl SourceTag {
    /// Trims the ids (1..=200 chars each), requires a `source_id` for a `source_run`, and requires
    /// `metadata` to be an object of at most 64 KiB (422, or 413 when too large). Absent metadata
    /// stays `None`: a new document stores `{}`, a refreshed one keeps its own.
    pub fn validate(self) -> Result<SourceFields> {
        let source_id = source_field("source_id", self.source_id)?;
        let source_run = source_field("source_run", self.source_run)?;
        if source_run.is_some() && source_id.is_none() {
            return Err(StrataError::InvalidInput(
                "source_run is only allowed together with source_id".to_string(),
            ));
        }
        let metadata = match self.metadata {
            None => None,
            Some(Value::Object(m)) => Some(m),
            Some(other) => {
                return Err(StrataError::InvalidInput(format!(
                    "metadata must be a JSON object, got {}",
                    json_type(&other)
                )));
            }
        };
        check_metadata_size(metadata.as_ref())?;
        Ok(SourceFields {
            source_id,
            source_run,
            metadata,
            topic_keys: None,
            supersedes: None,
        })
    }
}

fn check_metadata_size(metadata: Option<&Map<String, Value>>) -> Result<()> {
    let size = serde_json::to_vec(&metadata)
        .map_err(|e| StrataError::InvalidInput(format!("metadata: {e}")))?
        .len();
    if size > MAX_METADATA_BYTES {
        return Err(StrataError::PayloadTooLarge(format!(
            "metadata is {size} bytes as JSON; the limit is {MAX_METADATA_BYTES}"
        )));
    }
    Ok(())
}

/// Where a document's content came from (spec platform-knowledge §1 D4), stored as
/// `metadata.provenance`. Required, with an allow-listed licence, in a platform store.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Provenance {
    pub source_url: String,
    /// SPDX-style licence id (`CC-BY-4.0`, `public-domain`, ...).
    pub license: String,
    /// RFC 3339 timestamp of retrieval.
    pub retrieved_at: String,
}

impl Provenance {
    /// Trims each field; each must be non-blank and at most `MAX_PROVENANCE_FIELD_CHARS`, and
    /// `retrieved_at` must be RFC 3339 (422 otherwise).
    fn validate(self) -> Result<Provenance> {
        let field = |name: &str, v: String| -> Result<String> {
            let v = v.trim();
            if v.is_empty() {
                return Err(StrataError::InvalidInput(format!(
                    "provenance.{name} is required"
                )));
            }
            let chars = v.chars().count();
            if chars > MAX_PROVENANCE_FIELD_CHARS {
                return Err(StrataError::InvalidInput(format!(
                    "provenance.{name} must be at most {MAX_PROVENANCE_FIELD_CHARS} characters, \
                     got {chars}"
                )));
            }
            Ok(v.to_string())
        };
        let p = Provenance {
            source_url: field("source_url", self.source_url)?,
            license: field("license", self.license)?,
            retrieved_at: field("retrieved_at", self.retrieved_at)?,
        };
        if sqlx::types::chrono::DateTime::parse_from_rfc3339(&p.retrieved_at).is_err() {
            return Err(StrataError::InvalidInput(format!(
                "provenance.retrieved_at must be an RFC 3339 timestamp, got `{}`",
                p.retrieved_at
            )));
        }
        Ok(p)
    }
}

/// The platform-knowledge fields of one document (spec platform-knowledge §6, §8), as received.
/// `None` = not given: a new document stores none, a refreshed one keeps its own.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeTag {
    pub topic_keys: Option<Vec<String>>,
    pub supersedes: Option<Vec<String>>,
    pub provenance: Option<Provenance>,
}

impl KnowledgeTag {
    /// Validates the keys and provenance and folds them into `origin`: the keys as given (sorted,
    /// deduplicated), the provenance as `metadata.provenance` (overriding a caller-supplied one).
    /// Returns the validated provenance for the licence check.
    fn apply(self, origin: &mut SourceFields) -> Result<Option<Provenance>> {
        origin.topic_keys = self
            .topic_keys
            .map(|k| topic_keys("topic_keys", k))
            .transpose()?;
        origin.supersedes = self
            .supersedes
            .map(|k| topic_keys("supersedes", k))
            .transpose()?;
        let Some(p) = self.provenance else {
            return Ok(None);
        };
        let p = p.validate()?;
        let value = serde_json::to_value(&p)
            .map_err(|e| StrataError::InvalidInput(format!("provenance: {e}")))?;
        origin
            .metadata
            .get_or_insert_with(Map::new)
            .insert(PROVENANCE_KEY.to_string(), value);
        check_metadata_size(origin.metadata.as_ref())?;
        Ok(Some(p))
    }
}

/// A topic key: 1..=`MAX_TOPIC_KEY_CHARS` of `[a-z0-9_./-]` (the pack-key charset).
pub fn valid_topic_key(k: &str) -> bool {
    !k.is_empty()
        && k.len() <= MAX_TOPIC_KEY_CHARS
        && k.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'.' | b'/' | b'-')
        })
}

/// `name` (`topic_keys` / `supersedes`): at most `MAX_TOPIC_KEYS` valid keys; returned sorted and
/// deduplicated (422 otherwise).
fn topic_keys(name: &str, keys: Vec<String>) -> Result<Vec<String>> {
    if keys.len() > MAX_TOPIC_KEYS {
        return Err(StrataError::InvalidInput(format!(
            "{name} holds {} keys; the limit is {MAX_TOPIC_KEYS}",
            keys.len()
        )));
    }
    if let Some(bad) = keys.iter().find(|k| !valid_topic_key(k)) {
        return Err(StrataError::InvalidInput(format!(
            "{name}: `{bad}` is not a topic key (1-{MAX_TOPIC_KEY_CHARS} chars of [a-z0-9_./-])"
        )));
    }
    let mut keys = keys;
    keys.sort();
    keys.dedup();
    Ok(keys)
}

/// Licence enforcement (spec platform-knowledge §1 D4): a store owned by the platform owner takes
/// only documents with provenance whose licence is on `allowed` (case-insensitive). Any other
/// store: no requirement.
fn check_platform_license(
    platform_owner: Option<Tenant>,
    allowed: &[String],
    store_id: Uuid,
    store_tenant: Uuid,
    provenance: Option<&Provenance>,
) -> Result<()> {
    if platform_owner.map(Tenant::id) != Some(store_tenant) {
        return Ok(());
    }
    let Some(p) = provenance else {
        return Err(StrataError::InvalidInput(format!(
            "store {store_id} holds platform knowledge: provenance \
             {{source_url, license, retrieved_at}} is required"
        )));
    };
    if !allowed
        .iter()
        .any(|a| a.trim().eq_ignore_ascii_case(&p.license))
    {
        return Err(StrataError::InvalidInput(format!(
            "licence `{}` is not allowed for platform knowledge; allowed (STRATA_PLATFORM_LICENSES): \
             {}",
            p.license,
            allowed.join(", ")
        )));
    }
    Ok(())
}

fn json_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// An optional source identifier (`source_id`, `source_run`, `except_run`): trimmed, then
/// 1..=`MAX_SOURCE_FIELD_CHARS` chars. Absent stays absent; blank is a 422.
pub fn source_field(name: &str, value: Option<String>) -> Result<Option<String>> {
    let Some(v) = value else { return Ok(None) };
    let v = v.trim();
    if v.is_empty() {
        return Err(StrataError::InvalidInput(format!(
            "{name} must not be blank"
        )));
    }
    let chars = v.chars().count();
    if chars > MAX_SOURCE_FIELD_CHARS {
        return Err(StrataError::InvalidInput(format!(
            "{name} must be at most {MAX_SOURCE_FIELD_CHARS} characters, got {chars}"
        )));
    }
    Ok(Some(v.to_string()))
}

/// One document to ingest (`Engine::ingest_document`).
#[derive(Debug, Clone)]
pub struct IngestRequest<'a> {
    pub store_id: Uuid,
    /// File path or caller-supplied label.
    pub source: String,
    pub title: Option<String>,
    pub text: &'a str,
    pub tag: SourceTag,
}

/// `DELETE /api/sources/{source_id}/documents` (`Engine::prune_source`).
#[derive(Debug, Clone, Default)]
pub struct PruneRequest {
    pub source_id: String,
    /// Keep the documents of this run. Exactly one of `except_run` and `all` is required.
    pub except_run: Option<String>,
    /// Delete every document of the source (no run kept).
    pub all: bool,
    /// Only this leaf's documents.
    pub store_id: Option<Uuid>,
    /// Prune a leaf even when none of the source's documents there carries `except_run`.
    pub allow_empty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct PruneReport {
    pub deleted: u64,
    /// The source's documents (in scope) left after the prune.
    pub kept: i64,
    /// The leaves documents were deleted from.
    pub stores: Vec<Uuid>,
    /// The leaves left untouched because none of the source's documents there carries
    /// `except_run` (the empty-sync guard; never with `allow_empty`).
    pub refused_stores: Vec<Uuid>,
}

/// What one leaf's prune transaction did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeafOutcome {
    Deleted(u64),
    /// The empty-sync guard held: nothing deleted.
    Refused,
}

/// What one leaf-locked delete transaction removes.
#[derive(Debug, Clone, Copy)]
enum LeafDelete<'a> {
    Document(Uuid),
    Source {
        source_id: &'a str,
        except_run: Option<&'a str>,
        /// Refuse the leaf when the source has documents there but none of run `except_run`.
        guard_empty: bool,
    },
}

fn check_cap(store_id: Uuid, current: i64, adding: i64, max: i64) -> Result<()> {
    if current + adding > max {
        return Err(StrataError::LeafFull {
            store_id,
            current,
            adding,
            max,
        });
    }
    Ok(())
}

/// A dedupe hit is the caller's own document only when both carry the same `source_id`.
fn same_source(doc: &DocumentRow, origin: &SourceFields) -> bool {
    doc.source_id.is_some() && doc.source_id == origin.source_id
}

/// A dedupe hit found under the leaf lock: the same source's document takes the request's
/// `source` label, and its run, metadata and title when the request carries them (absent ones are
/// kept; no re-embed). Any other document is returned untouched.
async fn reconcile_existing(
    conn: &mut PgConnection,
    tenant: Tenant,
    doc: DocumentRow,
    source: &str,
    title: Option<&str>,
    origin: &SourceFields,
) -> Result<DocumentRow> {
    if !same_source(&doc, origin) {
        return Ok(doc);
    }
    let id = doc.id;
    documents::update_document_source(conn, tenant, id, source, title, origin)
        .await?
        // the leaf lock is held, and every delete takes it first
        .ok_or_else(|| StrataError::NotFound(format!("document {id}")))
}

impl Engine {
    /// `ingest_document` without source provenance.
    pub async fn ingest(
        &self,
        tenant: Tenant,
        store_id: Uuid,
        source: String,
        title: Option<String>,
        text: &str,
    ) -> Result<IngestOutcome> {
        self.ingest_document(
            tenant,
            IngestRequest {
                store_id,
                source,
                title,
                text,
                tag: SourceTag::default(),
            },
        )
        .await
    }

    /// `ingest_document_with` without platform-knowledge fields.
    pub async fn ingest_document(
        &self,
        tenant: Tenant,
        req: IngestRequest<'_>,
    ) -> Result<IngestOutcome> {
        self.ingest_document_with(tenant, req, KnowledgeTag::default())
            .await
    }

    /// Ingests into `tenant`'s leaf `req.store_id` (another tenant's store is 404), with the
    /// document's topic keys, supersedes and provenance. A platform store (owned by
    /// `STRATA_PLATFORM_OWNER`) requires provenance with an allow-listed licence (422), and a
    /// store inside a published pack refuses the write (409).
    pub async fn ingest_document_with(
        &self,
        tenant: Tenant,
        req: IngestRequest<'_>,
        knowledge: KnowledgeTag,
    ) -> Result<IngestOutcome> {
        let IngestRequest {
            store_id,
            source,
            title,
            text,
            tag,
        } = req;
        let mut origin = tag.validate()?;
        let provenance = knowledge.apply(&mut origin)?;

        // 1. the target must be a leaf; its providers are the nearest store binding's, else the
        //    tenant's
        let inf = self.inference_for(tenant, Some(store_id)).await?;
        let mut conn = self.pool.acquire().await?;
        let leaf = stores::get_store(&mut conn, tenant, store_id).await?;
        if leaf.kind != StoreKind::Leaf {
            return Err(StrataError::InvalidTarget(format!(
                "store {store_id} is a router; documents can only be added to a leaf"
            )));
        }
        check_platform_license(
            self.cfg.platform_owner,
            &self.cfg.platform_licenses,
            store_id,
            leaf.tenant,
            provenance.as_ref(),
        )?;
        // fail before embedding; re-checked under the leaf lock by every write below
        crate::db::platform::assert_writable(&mut *conn, store_id).await?;
        // the chunks' vectors land in the leaf and roll up into every ancestor
        self.check_lineage_embed_model(&mut conn, store_id, inf.embed_model_id()?)
            .await?;

        // 2. normalise; nothing left → 422, no row written (Review Focus #1)
        let normalized = normalize(text);
        if normalized.is_empty() {
            return Err(StrataError::InvalidInput(
                "document is empty after normalisation".to_string(),
            ));
        }

        // 3. dedupe on (store, sha256(normalised)) — nothing re-embedded. The same source's
        //    document is refreshed (label, run, metadata, title) under the leaf lock; if it vanished
        //    meanwhile (a prune), it is ingested afresh below.
        let hash = Sha256::digest(normalized.as_bytes()).to_vec();
        let found = documents::find_document_by_hash(&mut conn, store_id, &hash).await?;
        drop(conn);
        if let Some(doc) = found {
            if !same_source(&doc, &origin) {
                return Ok(IngestOutcome::Existing(doc));
            }
            let refreshed = crate::db::retry_once(|| {
                self.refresh_existing(tenant, store_id, &hash, &source, title.as_deref(), &origin)
            })
            .await?;
            if let Some(doc) = refreshed {
                return Ok(IngestOutcome::Existing(doc));
            }
        }

        // 4. chunk
        let chunks = chunk_text(
            &normalized,
            self.tc.as_ref(),
            self.cfg.chunk_target,
            self.cfg.chunk_overlap,
        );
        if chunks.is_empty() {
            return Err(StrataError::InvalidInput(
                "document produced no chunks".to_string(),
            ));
        }
        let doc_tokens = i32::try_from(self.tc.count(&normalized))
            .map_err(|_| StrataError::InvalidInput("document exceeds 2^31 tokens".to_string()))?;
        let adding = i64::from(doc_tokens);

        // 5. cheap cap pre-check, so an over-cap document is never embedded
        check_cap(store_id, leaf.token_count, adding, self.cfg.leaf_max_tokens)?;

        // 6. embed BEFORE the transaction: no row lock held during inference
        let embeddings = inf
            .embed(chunks.iter().map(|c| c.text.clone()).collect())
            .await?;

        // 7. one transaction; the leaf row lock serialises concurrent ingests
        //    into this leaf (Review Focus #2). Retried once on a deadlock.
        let written = crate::db::retry_once(|| {
            self.write_document(
                tenant,
                store_id,
                &source,
                title.as_deref(),
                &hash,
                doc_tokens,
                &chunks,
                &embeddings,
                &origin,
            )
        })
        .await?;
        let doc_id = match written {
            Ok(id) => id,
            Err(existing) => return Ok(IngestOutcome::Existing(existing)),
        };

        // 8. routing cache sees the new profile_vecs on the next query
        self.cache.invalidate(tenant).await;
        Ok(IngestOutcome::Created(DocumentRow {
            id: doc_id,
            store_id,
            source,
            title,
            token_count: doc_tokens,
            chunk_count: chunks.len() as i64,
            source_id: origin.source_id,
            source_run: origin.source_run,
            metadata: origin.metadata.unwrap_or_default(),
            topic_keys: origin.topic_keys.unwrap_or_default(),
            supersedes: origin.supersedes.unwrap_or_default(),
        }))
    }

    /// Step 3's refresh: lock the leaf (ingest's order), re-find the document and reconcile it.
    /// `None`: it was deleted since the unlocked lookup.
    async fn refresh_existing(
        &self,
        tenant: Tenant,
        store_id: Uuid,
        hash: &[u8],
        source: &str,
        title: Option<&str>,
        origin: &SourceFields,
    ) -> Result<Option<DocumentRow>> {
        let mut tx = self.pool.begin().await?;
        stores::get_store_for_update(&mut tx, tenant, store_id).await?;
        crate::db::platform::assert_writable(&mut *tx, store_id).await?;
        let Some(doc) = documents::find_document_by_hash(&mut tx, store_id, hash).await? else {
            tx.rollback().await?;
            return Ok(None);
        };
        let doc = reconcile_existing(&mut tx, tenant, doc, source, title, origin).await?;
        tx.commit().await?;
        Ok(Some(doc))
    }

    /// Step 7's transaction: lock the leaf, re-check dedupe and cap, insert, recompute ancestors.
    /// `Ok(Err(doc))` is an identical document committed while we embedded (reconciled like a
    /// step-3 hit: refreshed when it is the same source's).
    #[allow(clippy::too_many_arguments)]
    async fn write_document(
        &self,
        tenant: Tenant,
        store_id: Uuid,
        source: &str,
        title: Option<&str>,
        hash: &[u8],
        doc_tokens: i32,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
        origin: &SourceFields,
    ) -> Result<std::result::Result<Uuid, DocumentRow>> {
        let mut tx = self.pool.begin().await?;
        let locked = stores::get_store_for_update(&mut tx, tenant, store_id).await?;
        crate::db::platform::assert_writable(&mut *tx, store_id).await?;
        if let Some(doc) = documents::find_document_by_hash(&mut tx, store_id, hash).await? {
            let doc = reconcile_existing(&mut tx, tenant, doc, source, title, origin).await?;
            tx.commit().await?;
            return Ok(Err(doc));
        }
        check_cap(
            store_id,
            locked.token_count,
            i64::from(doc_tokens),
            self.cfg.leaf_max_tokens,
        )?;
        let doc_id =
            documents::insert_document(&mut tx, store_id, source, title, hash, doc_tokens, origin)
                .await?;
        documents::insert_chunks(&mut tx, doc_id, store_id, chunks, embeddings).await?;
        stores::recompute_upwards(&mut tx, tenant, store_id, self.cfg.profile_alpha).await?;
        tx.commit().await?;
        Ok(Ok(doc_id))
    }

    /// Deletes `tenant`'s document `id` (another tenant's is 404).
    pub async fn delete_document(&self, tenant: Tenant, id: Uuid) -> Result<()> {
        crate::db::retry_once(|| self.delete_document_tx(tenant, id)).await?;
        self.cache.invalidate(tenant).await;
        Ok(())
    }

    async fn delete_document_tx(&self, tenant: Tenant, id: Uuid) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        let store_id = documents::document_store_id(&mut tx, tenant, id).await?;
        self.delete_in_leaf(tx, tenant, store_id, LeafDelete::Document(id))
            .await
            .map(|_| ())
    }

    /// The one delete transaction (`delete_document`, `prune_source`). Locks the leaf BEFORE any
    /// document — ingest's order (leaf, then document) — so a delete and an ingest of the same
    /// content cannot deadlock, and ingest's locked dedupe never sees a document that a delete has
    /// already claimed. A source prune then applies the empty-sync guard under that lock, so it
    /// sees exactly the documents it would delete. Then deletes (chunks cascade) and recomputes the
    /// leaf's counts and vectors up to the root.
    async fn delete_in_leaf(
        &self,
        mut tx: Transaction<'static, Postgres>,
        tenant: Tenant,
        store_id: Uuid,
        target: LeafDelete<'_>,
    ) -> Result<LeafOutcome> {
        match stores::get_store_for_update(&mut tx, tenant, store_id).await {
            Ok(_) => {}
            // the leaf (and with it its documents, by cascade) was deleted while we waited
            Err(StrataError::NotFound(_)) => {
                return match target {
                    LeafDelete::Document(id) => {
                        Err(StrataError::NotFound(format!("document {id}")))
                    }
                    LeafDelete::Source { .. } => Ok(LeafOutcome::Deleted(0)),
                };
            }
            Err(e) => return Err(e),
        }
        // a published pack's documents are frozen (also against a source prune)
        crate::db::platform::assert_writable(&mut *tx, store_id).await?;
        let deleted = match target {
            LeafDelete::Document(id) => {
                documents::delete_document(&mut tx, tenant, id).await?;
                1
            }
            LeafDelete::Source {
                source_id,
                except_run,
                guard_empty,
            } => {
                if guard_empty && let Some(run) = except_run {
                    let (total, carrying) = documents::source_counts(
                        &mut tx,
                        tenant,
                        source_id,
                        Some(store_id),
                        Some(run),
                    )
                    .await?;
                    if total > 0 && carrying == 0 {
                        tx.rollback().await?;
                        return Ok(LeafOutcome::Refused);
                    }
                }
                documents::delete_source_documents(&mut tx, tenant, store_id, source_id, except_run)
                    .await?
            }
        };
        if deleted == 0 {
            tx.rollback().await?;
            return Ok(LeafOutcome::Deleted(0));
        }
        stores::recompute_upwards(&mut tx, tenant, store_id, self.cfg.profile_alpha).await?;
        tx.commit().await?;
        Ok(LeafOutcome::Deleted(deleted))
    }

    async fn prune_leaf_tx(
        &self,
        tenant: Tenant,
        store_id: Uuid,
        target: LeafDelete<'_>,
    ) -> Result<LeafOutcome> {
        let tx = self.pool.begin().await?;
        self.delete_in_leaf(tx, tenant, store_id, target).await
    }

    /// Deletes a source's documents except those of run `except_run` (or all of them with `all`;
    /// exactly one is required), one leaf-locked transaction per affected leaf, then invalidates
    /// the routing cache once. The empty-sync guard is per leaf and decided under its lock: a leaf
    /// where the source has documents but none of run `except_run` is left untouched and listed in
    /// `refused_stores` (the sync that should have delivered them likely failed there). Any refused
    /// leaf makes the whole call a 409 `empty_sync_run` that still carries the report, since other
    /// leaves may have been pruned. `allow_empty` turns the guard off. A source without documents
    /// → all zero.
    /// Only `tenant`'s documents of the source are counted and deleted; `store_id` of another
    /// tenant is 404. Another tenant's documents under the same `source_id` are never touched.
    pub async fn prune_source(&self, tenant: Tenant, req: PruneRequest) -> Result<PruneReport> {
        let source_id = source_field("source_id", Some(req.source_id))?
            .ok_or_else(|| StrataError::InvalidInput("source_id is required".to_string()))?;
        let except_run = source_field("except_run", req.except_run)?;
        match (&except_run, req.all) {
            (None, false) => {
                return Err(StrataError::InvalidInput(
                    "give except_run=<run> to keep that run, or all=true to delete every \
                     document of the source"
                        .to_string(),
                ));
            }
            (Some(_), true) => {
                return Err(StrataError::InvalidInput(
                    "except_run and all=true are mutually exclusive".to_string(),
                ));
            }
            _ => {}
        }
        let mut conn = self.pool.acquire().await?;
        if let Some(id) = req.store_id {
            let store = stores::get_store(&mut conn, tenant, id).await?;
            if store.kind != StoreKind::Leaf {
                return Err(StrataError::InvalidTarget(format!(
                    "store {id} is a router; documents live in leaves"
                )));
            }
        }
        // a candidate list only: each leaf is re-checked under its lock
        let leaves = documents::source_prune_leaves(
            &mut conn,
            tenant,
            &source_id,
            req.store_id,
            except_run.as_deref(),
        )
        .await?;
        drop(conn);

        let target = LeafDelete::Source {
            source_id: &source_id,
            except_run: except_run.as_deref(),
            guard_empty: !req.allow_empty,
        };
        let mut deleted = 0_u64;
        let mut stores = Vec::new();
        let mut refused_stores = Vec::new();
        let mut failure = None;
        for &leaf in &leaves {
            match crate::db::retry_once(|| self.prune_leaf_tx(tenant, leaf, target)).await {
                Ok(LeafOutcome::Deleted(0)) => {}
                Ok(LeafOutcome::Deleted(n)) => {
                    deleted += n;
                    stores.push(leaf);
                }
                Ok(LeafOutcome::Refused) => refused_stores.push(leaf),
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        // once, and also after a partial failure: earlier leaves did commit
        if !stores.is_empty() {
            self.cache.invalidate(tenant).await;
        }
        if let Some(e) = failure {
            return Err(e);
        }
        let mut conn = self.pool.acquire().await?;
        let (kept, _) =
            documents::source_counts(&mut conn, tenant, &source_id, req.store_id, None).await?;
        let report = PruneReport {
            deleted,
            kept,
            stores,
            refused_stores,
        };
        match except_run {
            Some(run) if !report.refused_stores.is_empty() => Err(StrataError::EmptySyncRun {
                source_id,
                run,
                report: Box::new(report),
            }),
            _ => Ok(report),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(id: Option<&str>, run: Option<&str>, metadata: Option<Value>) -> SourceTag {
        SourceTag {
            source_id: id.map(str::to_string),
            source_run: run.map(str::to_string),
            metadata,
        }
    }

    #[test]
    fn source_tag_trims_and_defaults_metadata_to_empty_object() {
        let f = tag(Some("  web-1 "), Some("r1\n"), None)
            .validate()
            .unwrap();
        assert_eq!(f.source_id.as_deref(), Some("web-1"));
        assert_eq!(f.source_run.as_deref(), Some("r1"));
        assert_eq!(f.metadata, None, "absent metadata stays absent");
        assert_eq!(
            SourceTag::default().validate().unwrap(),
            SourceFields::default()
        );
        // metadata without a source is allowed
        let f = tag(None, None, Some(serde_json::json!({"a": 1})))
            .validate()
            .unwrap();
        assert_eq!(f.metadata.unwrap()["a"], 1);
    }

    #[test]
    fn source_tag_rejects_bad_ids() {
        let code = |t: SourceTag| t.validate().unwrap_err().code();
        assert_eq!(code(tag(Some("  "), None, None)), "invalid_input");
        assert_eq!(code(tag(Some("a"), Some(""), None)), "invalid_input");
        assert_eq!(code(tag(None, Some("r"), None)), "invalid_input");
        let at_cap = "é".repeat(MAX_SOURCE_FIELD_CHARS);
        assert!(tag(Some(&at_cap), Some(&at_cap), None).validate().is_ok());
        let over = "é".repeat(MAX_SOURCE_FIELD_CHARS + 1);
        assert_eq!(code(tag(Some(&over), None, None)), "invalid_input");
        assert_eq!(code(tag(Some("a"), Some(&over), None)), "invalid_input");
    }

    #[test]
    fn source_tag_requires_an_object_of_at_most_64_kib() {
        for bad in [
            serde_json::json!([1, 2]),
            serde_json::json!("x"),
            serde_json::json!(3),
            serde_json::json!(null),
            serde_json::json!(true),
        ] {
            let err = tag(None, None, Some(bad.clone())).validate().unwrap_err();
            assert_eq!(err.code(), "invalid_input", "{bad}");
        }
        // {"k":"<v>"} is 8 bytes of framing
        let fits = serde_json::json!({"k": "x".repeat(MAX_METADATA_BYTES - 8)});
        assert_eq!(serde_json::to_vec(&fits).unwrap().len(), MAX_METADATA_BYTES);
        assert!(tag(None, None, Some(fits)).validate().is_ok());
        let over = serde_json::json!({"k": "x".repeat(MAX_METADATA_BYTES - 7)});
        let err = tag(None, None, Some(over)).validate().unwrap_err();
        assert_eq!((err.code(), err.status()), ("payload_too_large", 413));
    }

    fn prov(license: &str) -> Provenance {
        Provenance {
            source_url: " https://example.org/a ".to_string(),
            license: license.to_string(),
            retrieved_at: "2026-10-01T12:00:00Z".to_string(),
        }
    }

    #[test]
    fn topic_keys_are_validated_sorted_and_deduplicated() {
        let keys = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            topic_keys("topic_keys", keys(&["hr/leave", "a.b-c_9", "hr/leave"])).unwrap(),
            keys(&["a.b-c_9", "hr/leave"])
        );
        assert!(topic_keys("supersedes", Vec::new()).unwrap().is_empty());
        for bad in ["", "Upper", "sp ace", "ü", "a:b"] {
            let err = topic_keys("topic_keys", keys(&[bad])).unwrap_err();
            assert_eq!(err.code(), "invalid_input", "{bad:?}");
        }
        let at_cap = "k".repeat(MAX_TOPIC_KEY_CHARS);
        assert!(topic_keys("topic_keys", vec![at_cap.clone()]).is_ok());
        assert!(topic_keys("topic_keys", vec![format!("{at_cap}k")]).is_err());
        let many: Vec<String> = (0..=MAX_TOPIC_KEYS).map(|i| format!("k{i}")).collect();
        assert!(topic_keys("topic_keys", many[..MAX_TOPIC_KEYS].to_vec()).is_ok());
        assert!(topic_keys("topic_keys", many).is_err());
    }

    #[test]
    fn knowledge_tag_folds_provenance_into_metadata() {
        let mut origin = tag(
            None,
            None,
            Some(serde_json::json!({"a": 1, "provenance": "x"})),
        )
        .validate()
        .unwrap();
        let k = KnowledgeTag {
            topic_keys: Some(vec!["b".into(), "a".into()]),
            supersedes: None,
            provenance: Some(prov("CC-BY-4.0")),
        };
        let p = k.apply(&mut origin).unwrap().unwrap();
        assert_eq!(p.source_url, "https://example.org/a", "trimmed");
        let md = origin.metadata.unwrap();
        assert_eq!(md["a"], 1);
        assert_eq!(md[PROVENANCE_KEY]["license"], "CC-BY-4.0");
        assert_eq!(md[PROVENANCE_KEY]["retrieved_at"], "2026-10-01T12:00:00Z");
        assert_eq!(origin.topic_keys, Some(vec!["a".into(), "b".into()]));
        assert_eq!(origin.supersedes, None, "absent stays absent");

        // no metadata given: provenance alone
        let mut origin = SourceFields::default();
        KnowledgeTag {
            provenance: Some(prov("CC0-1.0")),
            ..KnowledgeTag::default()
        }
        .apply(&mut origin)
        .unwrap();
        assert_eq!(origin.metadata.unwrap().len(), 1);

        // nothing given: nothing changes
        let mut origin = SourceFields::default();
        assert!(
            KnowledgeTag::default()
                .apply(&mut origin)
                .unwrap()
                .is_none()
        );
        assert_eq!(origin, SourceFields::default());
    }

    #[test]
    fn provenance_fields_are_required_and_retrieved_at_is_rfc3339() {
        let apply = |p: Provenance| {
            KnowledgeTag {
                provenance: Some(p),
                ..KnowledgeTag::default()
            }
            .apply(&mut SourceFields::default())
        };
        assert!(apply(prov("CC0-1.0")).is_ok());
        for bad in [
            Provenance {
                source_url: "  ".into(),
                ..prov("CC0-1.0")
            },
            prov(""),
            Provenance {
                retrieved_at: "yesterday".into(),
                ..prov("CC0-1.0")
            },
            Provenance {
                source_url: "u".repeat(MAX_PROVENANCE_FIELD_CHARS + 1),
                ..prov("CC0-1.0")
            },
        ] {
            assert_eq!(
                apply(bad.clone()).unwrap_err().code(),
                "invalid_input",
                "{bad:?}"
            );
        }
    }

    #[test]
    fn platform_license_is_enforced_only_on_platform_stores() {
        let owner = Tenant::new(Uuid::from_u128(1)).unwrap();
        let allowed = vec!["CC0-1.0".to_string(), "CC-BY-4.0".to_string()];
        let check = |owner: Option<Tenant>, store_tenant: u128, p: Option<&Provenance>| {
            check_platform_license(
                owner,
                &allowed,
                Uuid::nil(),
                Uuid::from_u128(store_tenant),
                p,
            )
        };
        // tenant store: anything goes
        assert!(check(Some(owner), 2, None).is_ok());
        assert!(check(Some(owner), 2, Some(&prov("CC-BY-NC-4.0"))).is_ok());
        // platform knowledge disabled: no store is a platform store
        assert!(check(None, 1, None).is_ok());
        // platform store
        let err = check(Some(owner), 1, None).unwrap_err();
        assert_eq!(err.code(), "invalid_input");
        assert!(err.to_string().contains("provenance"), "{err}");
        let err = check(Some(owner), 1, Some(&prov("CC-BY-NC-4.0"))).unwrap_err();
        assert!(err.to_string().contains("CC-BY-NC-4.0"), "{err}");
        assert!(
            check(Some(owner), 1, Some(&prov("cc-by-4.0"))).is_ok(),
            "case-insensitive"
        );
        assert!(check(Some(owner), 1, Some(&prov("CC0-1.0"))).is_ok());
    }

    #[test]
    fn cap_allows_exactly_full() {
        assert!(check_cap(Uuid::nil(), 40, 10, 50).is_ok());
    }

    #[test]
    fn cap_rejects_one_over() {
        let err = check_cap(Uuid::nil(), 40, 11, 50).unwrap_err();
        assert!(
            matches!(
                err,
                StrataError::LeafFull {
                    current: 40,
                    adding: 11,
                    max: 50,
                    ..
                }
            ),
            "{err:?}"
        );
    }
}
