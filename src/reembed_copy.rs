//! `POST /api/stores/{id}/reembed-copy` (agent D83): re-embed a store's subtree into an EMPTY
//! target store, in the target's embedding model, while the source keeps serving in its own.
//!
//! Derived from `strata reembed` (`Engine::reembed_tree`), which re-embeds a tree in place: same
//! two phases (every vector computed BEFORE the transaction, no row lock held during inference;
//! then one transaction writes everything and recomputes the rollups child before parent), but
//! the result is a copy. The source subtree is mirrored under the target (each descendant store
//! recreated with its kind, name and description; the target itself keeps its own name), every
//! document copied with its provenance, and every chunk's text embedded again by the target's
//! providers (its store binding, else its tenant's). The copy is all-or-nothing: a query never
//! sees a half-copied target.
//!
//! The copy runs in a background task; the job row (`reembed_copy_jobs`, migration 0016) is what
//! `GET /api/stores/{id}/reembed-copy/{job}` reads, from any replica. Only a request starts one:
//! there is no schedule.
//!
//! The source keeps taking writes while the copy embeds (D91). So the copy runs in passes: the
//! first embeds the snapshot, each next one only what the source gained since (new documents, new
//! or renamed stores). After each pass the writing transaction locks the target and the source
//! subtree and reads the source again: when it holds nothing the passes have not embedded, that
//! transaction writes the copy and marks the job `done`; otherwise it rolls back and another pass
//! runs, at most [`REEMBED_COPY_MAX_PASSES`], after which the job fails `conflict` ("did not
//! converge") with nothing written. A document deleted from the source before the final
//! transaction is not copied: the target mirrors the source as it is when the copy commits.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use serde_json::{Map, Value};
use sqlx::types::Json;
use sqlx::types::chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::db::documents::{self, SourceFields};
use crate::db::stores::{self, NewStore, StoreRow};
use crate::engine::{Engine, desc_text};
use crate::error::{Result, StrataError};
use crate::reembed::{DEFAULT_REEMBED_BATCH, subtree_depths};
use crate::tenant::Tenant;
use crate::text::Chunk;

/// How often a running copy stamps `heartbeat_at` (also while one embed batch is in flight).
pub const REEMBED_COPY_BEAT_EVERY: Duration = Duration::from_secs(5);
/// A `running` job whose heartbeat is older than this lost its process (killed, crashed, rolled)
/// and reads as `failed` with error [`INTERRUPTED`]: twelve missed beats.
pub const REEMBED_COPY_STALE_AFTER: Duration = Duration::from_secs(60);
/// Passes (each followed by an attempt to commit) before a copy whose source keeps gaining
/// documents fails as not converged.
pub const REEMBED_COPY_MAX_PASSES: usize = 8;
/// Rounds of "read the source subtree, lock what is new" in the final transaction: a child
/// committed under a member while we waited for its lock is found by the next round.
const SOURCE_LOCK_ROUNDS: usize = 8;
/// A failed job's error for a 500 or a panic (the detail stays in the log), as the API answers.
const INTERNAL_MESSAGE: &str = "internal server error";
/// The error of a job whose worker stopped beating.
pub const INTERRUPTED: &str = "interrupted: the re-embed copy's process stopped; nothing was \
                               written to the target, start it again";

/// A job as `POST` (202) and `GET /api/stores/{id}/reembed-copy/{job}` answer it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReembedCopyJob {
    pub job: Uuid,
    pub source_store: Uuid,
    pub target_store: Uuid,
    /// The target's model the chunks are embedded with (`stores.embed_model` format).
    pub embed_model: String,
    /// The source's vectors are copied, not re-embedded (same model only).
    pub copy_vectors: bool,
    /// `running`, `done` or `failed`.
    pub state: String,
    pub chunks_done: i32,
    pub chunks_total: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// RFC 3339 (UTC, microseconds).
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
}

fn rfc3339(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string()
}

type JobRow = (
    Uuid,
    Uuid,
    Uuid,
    String,
    String,
    i32,
    i32,
    Option<String>,
    DateTime<Utc>,
    Option<DateTime<Utc>>,
    bool,
);

const JOB_COLS: &str = "id, source_store, target_store, embed_model, state, chunks_done, \
                        chunks_total, error, started_at, finished_at, copy_vectors";

fn job_of(r: JobRow) -> ReembedCopyJob {
    ReembedCopyJob {
        job: r.0,
        source_store: r.1,
        target_store: r.2,
        embed_model: r.3,
        state: r.4,
        chunks_done: r.5,
        chunks_total: r.6,
        error: r.7,
        started_at: rfc3339(r.8),
        finished_at: r.9.map(rfc3339),
        copy_vectors: r.10,
    }
}

fn changed_meanwhile(what: String) -> StrataError {
    StrataError::Conflict {
        message: format!(
            "{what} changed during the re-embed copy; nothing was written, run it again"
        ),
        retryable: false,
    }
}

/// One document of the source snapshot.
struct SourceDoc {
    id: Uuid,
    store_id: Uuid,
    source: String,
    title: Option<String>,
    content_hash: Vec<u8>,
    token_count: i32,
    source_id: Option<String>,
    source_run: Option<String>,
    metadata: Map<String, Value>,
    topic_keys: Vec<String>,
    supersedes: Vec<String>,
}

/// One chunk of the source snapshot (its text; the vector is the target's).
struct SourceChunk {
    document_id: Uuid,
    chunk: Chunk,
}

/// `target` holds nothing: no child store, no document.
async fn target_is_empty(conn: &mut sqlx::PgConnection, target: &StoreRow) -> Result<bool> {
    if target.chunk_count != 0 || stores::child_count(&mut *conn, target.id).await? != 0 {
        return Ok(false);
    }
    let docs: bool =
        sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM documents WHERE store_id = $1)")
            .bind(target.id)
            .fetch_one(&mut *conn)
            .await?;
    Ok(!docs)
}

fn not_empty(target: Uuid) -> StrataError {
    StrataError::InvalidTarget(format!(
        "target store {target} must be empty (no documents, no child stores): a re-embed copy \
         fills a fresh store"
    ))
}

impl Engine {
    /// Checks the request, stores the job `running` and starts the copy in a background task.
    /// 404 when `source` or `target` is not `tenant`'s; 422 `invalid_target` for the same store,
    /// a target inside the source's subtree, a kind mismatch or a target that is not empty; 409
    /// `embed_model_mismatch` when the target (or an ancestor) is not in the target's resolved
    /// model; 409 `conflict` when a copy into `target` is already running. `copy_vectors`: reuse
    /// the source's vectors instead of embedding again, only when the target's resolved model is
    /// the source's (else 409 `embed_model_mismatch`).
    pub async fn start_reembed_copy(
        self: &Arc<Self>,
        tenant: Tenant,
        source: Uuid,
        target: Uuid,
        copy_vectors: bool,
    ) -> Result<ReembedCopyJob> {
        let rows = stores::list_stores(&self.pool, tenant).await?;
        let by_id: HashMap<Uuid, &StoreRow> = rows.iter().map(|r| (r.id, r)).collect();
        let src = by_id
            .get(&source)
            .ok_or_else(|| StrataError::NotFound(format!("store {source}")))?;
        let tgt = by_id
            .get(&target)
            .ok_or_else(|| StrataError::NotFound(format!("target store {target}")))?;
        if source == target {
            return Err(StrataError::InvalidTarget(
                "target_store must differ from the source store".to_string(),
            ));
        }
        let members = subtree_depths(&rows, source);
        if members.iter().any(|(id, _)| *id == target) {
            return Err(StrataError::InvalidTarget(format!(
                "target store {target} is inside the subtree of store {source}"
            )));
        }
        if src.kind != tgt.kind {
            return Err(StrataError::InvalidTarget(format!(
                "store {source} is a {} but target store {target} is a {}; the target mirrors \
                 the source",
                src.kind.as_str(),
                tgt.kind.as_str()
            )));
        }
        // the copy writes into the target: not inside a published platform pack
        crate::db::platform::assert_writable(&self.pool, target).await?;
        let inf = self.inference_for(tenant, Some(target)).await?;
        let model = inf.embed_model_id()?.to_string();
        if copy_vectors && src.embed_model != model {
            // copied vectors are only meaningful in the model that made them
            return Err(StrataError::EmbedModelMismatch {
                store_id: source,
                store_model: src.embed_model.clone(),
                configured: model,
            });
        }
        {
            let mut conn = self.pool.acquire().await?;
            if !target_is_empty(&mut conn, tgt).await? {
                return Err(not_empty(target));
            }
            // the copy's vectors land in the target and roll up into its ancestors
            self.check_lineage_embed_model(&mut conn, target, &model)
                .await?;
        }
        let ids: Vec<Uuid> = members.iter().map(|(id, _)| *id).collect();
        let total: i64 = sqlx::query_scalar("SELECT count(*) FROM chunks WHERE store_id = ANY($1)")
            .bind(ids.as_slice())
            .fetch_one(&self.pool)
            .await?;
        let job = Uuid::now_v7();
        let row: JobRow = sqlx::query_as(sqlx::AssertSqlSafe(format!(
            "INSERT INTO reembed_copy_jobs (id, tenant, source_store, target_store, embed_model, \
                                            copy_vectors, state, chunks_total) \
             VALUES ($1, $2, $3, $4, $5, $6, 'running', $7) RETURNING {JOB_COLS}"
        )))
        .bind(job)
        .bind(tenant.id())
        .bind(source)
        .bind(target)
        .bind(&model)
        .bind(copy_vectors)
        .bind(i32::try_from(total).unwrap_or(i32::MAX))
        .fetch_one(&self.pool)
        .await
        .map_err(|e| match &e {
            sqlx::Error::Database(db) if db.code().as_deref() == Some("23505") => {
                StrataError::Conflict {
                    message: format!("a re-embed copy into store {target} is already running"),
                    retryable: false,
                }
            }
            _ => StrataError::from(e),
        })?;
        let engine = Arc::clone(self);
        let caller = crate::metering::current_caller();
        tokio::spawn(crate::metering::with_caller(caller, async move {
            engine
                .supervise_reembed_copy(tenant, job, source, target, copy_vectors)
                .await;
        }));
        tracing::info!(
            %tenant,
            %job,
            %source,
            %target,
            model = %model,
            total,
            "reembed copy started"
        );
        Ok(job_of(row))
    }

    /// `tenant`'s job `job` of source store `source`. Another tenant's, or another source's, is
    /// 404. A `running` job whose heartbeat went stale is failed `interrupted` here.
    pub async fn reembed_copy_job(
        &self,
        tenant: Tenant,
        source: Uuid,
        job: Uuid,
    ) -> Result<ReembedCopyJob> {
        sqlx::query(
            "UPDATE reembed_copy_jobs SET state = 'failed', error = $3, finished_at = now() \
             WHERE id = $1 AND tenant = $2 AND state = 'running' \
               AND heartbeat_at < now() - make_interval(secs => $4)",
        )
        .bind(job)
        .bind(tenant.id())
        .bind(INTERRUPTED)
        .bind(REEMBED_COPY_STALE_AFTER.as_secs_f64())
        .execute(&self.pool)
        .await?;
        let row: Option<JobRow> = sqlx::query_as(sqlx::AssertSqlSafe(format!(
            "SELECT {JOB_COLS} FROM reembed_copy_jobs \
             WHERE id = $1 AND tenant = $2 AND source_store = $3"
        )))
        .bind(job)
        .bind(tenant.id())
        .bind(source)
        .fetch_optional(&self.pool)
        .await?;
        row.map(job_of)
            .ok_or_else(|| StrataError::NotFound(format!("re-embed copy job {job}")))
    }

    /// Runs the copy in a task of its own while beating; finishes the job `done`, or `failed`
    /// with the error (the generic internal message for a 500 or a panic).
    async fn supervise_reembed_copy(
        self: Arc<Self>,
        tenant: Tenant,
        job: Uuid,
        source: Uuid,
        target: Uuid,
        copy_vectors: bool,
    ) {
        let beat = {
            let pool = self.pool.clone();
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(REEMBED_COPY_BEAT_EVERY);
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    tick.tick().await;
                    if let Err(e) = sqlx::query(
                        "UPDATE reembed_copy_jobs SET heartbeat_at = now() \
                         WHERE id = $1 AND state = 'running'",
                    )
                    .bind(job)
                    .execute(&pool)
                    .await
                    {
                        tracing::warn!(%job, error = %e, "reembed copy heartbeat not stored");
                    }
                }
            })
        };
        let work = {
            let engine = Arc::clone(&self);
            let caller = crate::metering::current_caller();
            tokio::spawn(crate::metering::with_caller(caller, async move {
                engine
                    .reembed_copy(tenant, job, source, target, copy_vectors)
                    .await
            }))
        };
        let outcome = work.await;
        beat.abort();
        let failed = match outcome {
            Ok(Ok(chunks)) => {
                tracing::info!(%tenant, %job, chunks, "reembed copy done");
                None
            }
            Ok(Err(e)) if e.status() == 500 => {
                tracing::error!(%job, error = %e, detail = ?e, "reembed copy failed");
                Some(INTERNAL_MESSAGE.to_string())
            }
            Ok(Err(e)) => {
                tracing::warn!(%job, error = %e, "reembed copy failed");
                Some(format!("{}: {e}", e.code()))
            }
            Err(join) => {
                tracing::error!(%job, error = %join, "reembed copy task panicked");
                Some(INTERNAL_MESSAGE.to_string())
            }
        };
        let stored = match &failed {
            None => {
                sqlx::query(
                    "UPDATE reembed_copy_jobs SET state = 'done', chunks_done = chunks_total, \
                 finished_at = now(), heartbeat_at = now() WHERE id = $1 AND state = 'running'",
                )
                .bind(job)
                .execute(&self.pool)
                .await
            }
            Some(message) => sqlx::query(
                "UPDATE reembed_copy_jobs SET state = 'failed', error = $2, finished_at = now() \
                 WHERE id = $1 AND state = 'running'",
            )
            .bind(job)
            .bind(message)
            .execute(&self.pool)
            .await,
        };
        if let Err(e) = stored {
            tracing::error!(
                %job,
                error = %e,
                "cannot store a reembed copy's outcome; it reads as interrupted once stale"
            );
        }
    }

    /// The copy itself (module docs): passes until one finds nothing new, the last one inside
    /// the writing transaction; returns the number of chunks copied.
    async fn reembed_copy(
        &self,
        tenant: Tenant,
        job: Uuid,
        source: Uuid,
        target: Uuid,
        copy_vectors: bool,
    ) -> Result<usize> {
        let target_model = {
            let mut conn = self.pool.acquire().await?;
            match stores::get_store(&mut conn, tenant, target).await {
                Ok(r) => r.embed_model,
                Err(StrataError::NotFound(_)) => {
                    return Err(changed_meanwhile(format!("target store {target}")));
                }
                Err(e) => return Err(e),
            }
        };
        let inf = self.inference_for(tenant, Some(target)).await?;
        let model = inf.embed_model_id()?.to_string();
        let mut state = CopyState {
            copy_vectors,
            ..CopyState::default()
        };
        for pass in 1..=REEMBED_COPY_MAX_PASSES {
            let found = self
                .copy_pass(tenant, job, source, &model, &inf, &mut state)
                .await?;
            match self
                .finish_copy(tenant, job, source, target, &model, &target_model, &state)
                .await
            {
                Ok(Some(chunks)) => {
                    self.cache.invalidate(tenant).await;
                    tracing::info!(%job, pass, chunks, "reembed copy converged");
                    return Ok(chunks);
                }
                // the source gained a document or store after this pass read it: another pass
                Ok(None) => {
                    tracing::info!(%job, pass, found, "reembed copy source changed; next pass");
                }
                // a deadlock with a concurrent writer of the source: nothing written, another pass
                Err(e) if e.is_retryable() => {
                    tracing::info!(%job, pass, error = %e, "reembed copy pass retried");
                }
                Err(e) => return Err(e),
            }
        }
        Err(not_converged(source))
    }

    /// One pass outside any transaction (no row lock held during inference): reads the source
    /// subtree as it is now, forgets documents deleted from it since the last pass, embeds the
    /// descriptions of stores new (or renamed) since then and the chunks of documents new since
    /// then, by the target's providers. Returns how many new stores and documents it found.
    async fn copy_pass(
        &self,
        tenant: Tenant,
        job: Uuid,
        source: Uuid,
        model: &str,
        inf: &crate::inference::InferenceHandle,
        state: &mut CopyState,
    ) -> Result<usize> {
        let rows = stores::list_stores(&self.pool, tenant).await?;
        let by_id: HashMap<Uuid, &StoreRow> = rows.iter().map(|r| (r.id, r)).collect();
        if !by_id.contains_key(&source) {
            return Err(changed_meanwhile(format!("store {source}")));
        }
        let members = subtree_depths(&rows, source);
        let member_ids: Vec<Uuid> = members.iter().map(|(id, _)| *id).collect();

        // descriptions, as create_store embeds them (query lane), in the target's model
        let stale: Vec<(Uuid, String)> = members
            .iter()
            .filter(|(id, _)| *id != source)
            .map(|(id, _)| (*id, desc_text(&by_id[id].name, &by_id[id].description)))
            .filter(|(id, text)| state.descs.get(id).is_none_or(|(t, _)| t != text))
            .collect();
        if state.copy_vectors {
            // the source's own vectors, in the target's model (checked here, the model of a store
            // with content cannot change, but a new store may have joined since the start)
            if let Some((id, _)) = members
                .iter()
                .find(|(id, _)| by_id[id].embed_model != model)
            {
                return Err(StrataError::EmbedModelMismatch {
                    store_id: *id,
                    store_model: by_id[id].embed_model.clone(),
                    configured: model.to_string(),
                });
            }
            for (id, text) in &stale {
                state
                    .descs
                    .insert(*id, (text.clone(), by_id[id].desc_vec.clone()));
            }
        } else if !stale.is_empty() {
            let vecs = inf
                .embed_queries(stale.iter().map(|(_, t)| t.clone()).collect())
                .await?;
            if vecs.len() != stale.len() {
                return Err(StrataError::Model(format!(
                    "{} embeddings for {} store descriptions",
                    vecs.len(),
                    stale.len()
                )));
            }
            for ((id, text), v) in stale.iter().cloned().zip(vecs) {
                state.descs.insert(id, (text, v));
            }
        }

        // documents: a deleted one is mirrored by not being copied, a new one is embedded
        let present: Vec<Uuid> =
            sqlx::query_scalar("SELECT id FROM documents WHERE store_id = ANY($1)")
                .bind(member_ids.as_slice())
                .fetch_all(&self.pool)
                .await?;
        let present_set: std::collections::HashSet<Uuid> = present.iter().copied().collect();
        state.docs.retain(|id, _| present_set.contains(id));
        let new_docs: Vec<Uuid> = present
            .iter()
            .copied()
            .filter(|id| !state.docs.contains_key(id))
            .collect();
        if !new_docs.is_empty() {
            type ChunkTuple = (
                Uuid,
                i32,
                String,
                i32,
                Option<i32>,
                Option<i32>,
                Option<pgvector::Vector>,
            );
            // the stored vector only when it is copied (a re-embed never reads it)
            let chunk_rows: Vec<ChunkTuple> = sqlx::query_as(
                "SELECT document_id, ord, text, token_count, page_start, page_end, \
                        CASE WHEN $2 THEN embedding END AS embedding \
                 FROM chunks WHERE document_id = ANY($1) ORDER BY document_id, ord",
            )
            .bind(new_docs.as_slice())
            .bind(state.copy_vectors)
            .fetch_all(&self.pool)
            .await?;
            let mut copied: Vec<Vec<f32>> = Vec::new();
            let mut chunks: Vec<SourceChunk> = Vec::with_capacity(chunk_rows.len());
            for c in chunk_rows {
                if state.copy_vectors {
                    let Some(v) = c.6 else {
                        return Err(StrataError::Model(format!(
                            "chunk {} of document {} has no stored vector to copy",
                            c.1, c.0
                        )));
                    };
                    copied.push(v.to_vec());
                }
                chunks.push(SourceChunk {
                    document_id: c.0,
                    chunk: Chunk {
                        ord: c.1,
                        text: c.2,
                        token_count: c.3,
                        page_start: c.4,
                        page_end: c.5,
                    },
                });
            }
            let have = state.chunk_count();
            sqlx::query(
                "UPDATE reembed_copy_jobs SET chunks_total = $2, chunks_done = $3 \
                 WHERE id = $1 AND state = 'running'",
            )
            .bind(job)
            .bind(i32::try_from(have + chunks.len()).unwrap_or(i32::MAX))
            .bind(i32::try_from(have).unwrap_or(i32::MAX))
            .execute(&self.pool)
            .await?;
            let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
            if state.copy_vectors {
                vecs = copied;
            }
            for part in chunks
                .chunks(DEFAULT_REEMBED_BATCH)
                .filter(|_| !state.copy_vectors)
            {
                vecs.extend(
                    inf.embed(part.iter().map(|c| c.chunk.text.clone()).collect())
                        .await?,
                );
                sqlx::query(
                    "UPDATE reembed_copy_jobs SET chunks_done = LEAST($2, chunks_total), \
                     heartbeat_at = now() WHERE id = $1 AND state = 'running'",
                )
                .bind(job)
                .bind(i32::try_from(have + vecs.len()).unwrap_or(i32::MAX))
                .execute(&self.pool)
                .await?;
            }
            if vecs.len() != chunks.len() {
                return Err(StrataError::Model(format!(
                    "{} embeddings for {} chunks",
                    vecs.len(),
                    chunks.len()
                )));
            }
            // a document without chunks (none is written so) is known, and skipped at the write
            for id in &new_docs {
                state.docs.insert(*id, DocCopy::default());
            }
            for (c, v) in chunks.into_iter().zip(vecs) {
                if let Some(d) = state.docs.get_mut(&c.document_id) {
                    d.chunks.push(c.chunk);
                    d.vecs.push(v);
                }
            }
        }
        Ok(stale.len() + new_docs.len())
    }

    /// The final pass and the write, in one transaction: the target locked first, then the
    /// source subtree (deepest level first, as a recursive delete locks it), so no ingest, delete
    /// or new child store can change the source until the copy commits. Re-read under those
    /// locks, the source must hold nothing `state` has not embedded: otherwise `None` (rolled
    /// back, the caller runs another pass). Documents deleted since are not copied (the target
    /// mirrors the source as of this transaction). The job is marked `done` in this transaction.
    #[allow(clippy::too_many_arguments)]
    async fn finish_copy(
        &self,
        tenant: Tenant,
        job: Uuid,
        source: Uuid,
        target: Uuid,
        model: &str,
        target_model: &str,
        state: &CopyState,
    ) -> Result<Option<usize>> {
        let mut tx = self.pool.begin().await?;
        let locked = match stores::get_store_for_update(&mut tx, tenant, target).await {
            Ok(r) => r,
            Err(StrataError::NotFound(_)) => {
                return Err(changed_meanwhile(format!("target store {target}")));
            }
            Err(e) => return Err(e),
        };
        if !target_is_empty(&mut tx, &locked).await? {
            return Err(not_empty(target));
        }
        crate::db::platform::assert_writable(&mut *tx, target).await?;
        if locked.embed_model != model || target_model != locked.embed_model {
            return Err(changed_meanwhile(format!("target store {target}")));
        }
        self.check_lineage_embed_model(&mut tx, target, model)
            .await?;

        // lock the source subtree; a child committed while we waited is locked in the next round
        let mut locked_ids: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
        let mut members: Vec<(Uuid, i32)> = Vec::new();
        for _ in 0..SOURCE_LOCK_ROUNDS {
            members = stores::subtree(&mut tx, tenant, source).await?;
            if members.is_empty() {
                return Err(changed_meanwhile(format!("store {source}")));
            }
            let (ids, depths): (Vec<Uuid>, Vec<i32>) = members
                .iter()
                .filter(|(id, _)| !locked_ids.contains(id))
                .copied()
                .unzip();
            if ids.is_empty() {
                break;
            }
            let got: Vec<Uuid> = sqlx::query_scalar(
                "SELECT s.id FROM stores s \
                 JOIN unnest($1::uuid[], $2::int4[]) AS t(id, depth) ON t.id = s.id \
                 WHERE s.tenant = $3 \
                 ORDER BY t.depth DESC, s.id \
                 FOR UPDATE OF s",
            )
            .bind(ids.as_slice())
            .bind(depths.as_slice())
            .bind(tenant.id())
            .fetch_all(&mut *tx)
            .await?;
            locked_ids.extend(got);
        }
        if members.iter().any(|(id, _)| !locked_ids.contains(id)) {
            return Ok(None);
        }
        let mut by_id: HashMap<Uuid, StoreRow> = HashMap::with_capacity(members.len());
        for (id, _) in &members {
            by_id.insert(*id, stores::get_store(&mut tx, tenant, *id).await?);
        }
        // every store embedded as it is now
        let mut copies: Vec<(Uuid, i32)> = members
            .iter()
            .filter(|(id, _)| *id != source)
            .copied()
            .collect();
        copies.sort_by_key(|(id, depth)| (*depth, *id));
        for (id, _) in &copies {
            let text = desc_text(&by_id[id].name, &by_id[id].description);
            if state.descs.get(id).is_none_or(|(t, _)| *t != text) {
                return Ok(None);
            }
        }
        // every document embedded
        type DocTuple = (
            Uuid,
            Uuid,
            String,
            Option<String>,
            Vec<u8>,
            i32,
            Option<String>,
            Option<String>,
            Json<Map<String, Value>>,
            Vec<String>,
            Vec<String>,
        );
        let member_ids: Vec<Uuid> = members.iter().map(|(id, _)| *id).collect();
        let docs: Vec<DocTuple> = sqlx::query_as(
            "SELECT id, store_id, source, title, content_hash, token_count, source_id, \
                    source_run, metadata, topic_keys, supersedes \
             FROM documents WHERE store_id = ANY($1) ORDER BY store_id, created_at, id",
        )
        .bind(member_ids.as_slice())
        .fetch_all(&mut *tx)
        .await?;
        let docs: Vec<SourceDoc> = docs
            .into_iter()
            .map(|d| SourceDoc {
                id: d.0,
                store_id: d.1,
                source: d.2,
                title: d.3,
                content_hash: d.4,
                token_count: d.5,
                source_id: d.6,
                source_run: d.7,
                metadata: (d.8).0,
                topic_keys: d.9,
                supersedes: d.10,
            })
            .collect();
        if docs.iter().any(|d| !state.docs.contains_key(&d.id)) {
            return Ok(None);
        }

        // converged: write the mirror, parents first
        let mut mirror: HashMap<Uuid, Uuid> = HashMap::from([(source, target)]);
        let mut created: Vec<(Uuid, i32)> = Vec::with_capacity(copies.len());
        for (id, depth) in &copies {
            let src = &by_id[id];
            let parent = src
                .parent_id
                .and_then(|p| mirror.get(&p).copied())
                .ok_or_else(|| changed_meanwhile(format!("store {id}")))?;
            let row = stores::insert_store(
                &mut tx,
                tenant,
                &NewStore {
                    parent_id: Some(parent),
                    kind: src.kind,
                    name: src.name.clone(),
                    description: src.description.clone(),
                },
                &state.descs[id].1,
                model,
            )
            .await?;
            mirror.insert(*id, row.id);
            created.push((row.id, *depth));
        }
        let mut written = 0usize;
        for d in &docs {
            let copy = &state.docs[&d.id];
            if copy.chunks.is_empty() {
                continue; // a document without chunks (none is written so; skip it)
            }
            let leaf = mirror
                .get(&d.store_id)
                .copied()
                .ok_or_else(|| changed_meanwhile(format!("store {}", d.store_id)))?;
            let new_doc = documents::insert_document(
                &mut tx,
                leaf,
                &d.source,
                d.title.as_deref(),
                &d.content_hash,
                d.token_count,
                &SourceFields {
                    source_id: d.source_id.clone(),
                    source_run: d.source_run.clone(),
                    metadata: Some(d.metadata.clone()),
                    topic_keys: Some(d.topic_keys.clone()),
                    supersedes: Some(d.supersedes.clone()),
                },
            )
            .await?;
            documents::insert_chunks(&mut tx, new_doc, leaf, &copy.chunks, &copy.vecs).await?;
            written += copy.chunks.len();
        }
        // rollups deepest first: a router aggregates its children's fresh content_vecs
        created.sort_by_key(|(id, depth)| (std::cmp::Reverse(*depth), *id));
        for (id, _) in &created {
            stores::recompute_node(&mut tx, tenant, *id, self.cfg.profile_alpha).await?;
        }
        stores::recompute_upwards(&mut tx, tenant, target, self.cfg.profile_alpha).await?;
        // the job finishes in the same transaction as the copy: a job failed `interrupted`
        // meanwhile (its beats did not land) rolls the copy back, so `failed` always means
        // nothing was written
        let total = i32::try_from(written).unwrap_or(i32::MAX);
        let finished = sqlx::query(
            "UPDATE reembed_copy_jobs SET state = 'done', chunks_total = $2, chunks_done = $2, \
             finished_at = now(), heartbeat_at = now() WHERE id = $1 AND state = 'running'",
        )
        .bind(job)
        .bind(total)
        .execute(&mut *tx)
        .await?
        .rows_affected();
        if finished != 1 {
            return Err(StrataError::Conflict {
                message: format!(
                    "re-embed copy job {job} is no longer running; nothing was written"
                ),
                retryable: false,
            });
        }
        tx.commit().await?;
        Ok(Some(written))
    }
}

/// What the passes have embedded so far, by the target's providers.
#[derive(Default)]
struct CopyState {
    /// Reuse the source's vectors (same model) instead of embedding (`copy_vectors`).
    copy_vectors: bool,
    /// Source store (not the source itself) → its description text and that text's vector.
    descs: HashMap<Uuid, (String, Vec<f32>)>,
    /// Source document → its chunks and their vectors.
    docs: HashMap<Uuid, DocCopy>,
}

impl CopyState {
    fn chunk_count(&self) -> usize {
        self.docs.values().map(|d| d.chunks.len()).sum()
    }
}

#[derive(Default)]
struct DocCopy {
    chunks: Vec<Chunk>,
    vecs: Vec<Vec<f32>>,
}

/// The source never held still: after [`REEMBED_COPY_MAX_PASSES`] passes the source still had
/// stores or documents the copy had not embedded. Nothing was written.
fn not_converged(source: Uuid) -> StrataError {
    StrataError::Conflict {
        message: format!(
            "re-embed copy did not converge: store {source}'s subtree was still gaining documents \
             or stores after {REEMBED_COPY_MAX_PASSES} passes; nothing was written to the target, \
             start it again once ingestion into the source pauses"
        ),
        retryable: false,
    }
}
