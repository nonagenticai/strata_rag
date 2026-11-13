//! axum handlers. Every extractor rejection is mapped to `StrataError::InvalidInput` so all
//! client errors share the `{"error":{"code","message"}}` body.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::{JsonRejection, PathRejection, QueryRejection};
use axum::extract::{Path, Query, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::ask::{AskRequest, AskResponse};
use crate::db::documents::{self, DocumentListItem, decode_cursor, encode_cursor};
use crate::db::stores::{self, NewStore, StoreKind};
use crate::engine::{Engine, HealthReport, StoreView};
use crate::error::{Result, StrataError};
use crate::eval::{
    EvalMode, EvalOptions, ModeReport, TreeSpec, evaluate_with, foreign_stores, parse_questions,
    resolve_tree_path, verdict_passes,
};
use crate::ingest::{
    IngestOutcome, IngestRequest, KnowledgeTag, Provenance, PruneReport, PruneRequest, SourceTag,
    source_field,
};
use crate::providers::{
    ProviderTestResult, ProvidersView, PutProviders, PutStoreProviders, StoreProvidersView,
    TestProvider,
};
use crate::query::{QueryRequest, QueryResponse};
use crate::reembed_copy::ReembedCopyJob;
use crate::tenant::Tenant;
use crate::tenant_eval::QuestionInput;

pub type AppState = Arc<Engine>;

#[derive(Debug, Deserialize)]
pub struct CreateStoreBody {
    #[serde(default)]
    pub parent_id: Option<Uuid>,
    pub kind: StoreKind,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateStoreBody {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StoreList {
    pub stores: Vec<StoreView>,
}

#[derive(Debug, Default, Deserialize)]
pub struct DeleteStoreParams {
    #[serde(default)]
    pub recursive: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct IngestParams {
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub source_run: Option<String>,
}

#[derive(Debug, Deserialize)]
struct IngestJson {
    text: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    source_id: Option<String>,
    #[serde(default)]
    source_run: Option<String>,
    /// Any JSON; `SourceTag::validate` requires an object (422 otherwise).
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    /// Topic keys the document covers (platform documents).
    #[serde(default)]
    topic_keys: Option<Vec<String>>,
    /// Topic keys the document overrides (tenant documents shadowing platform ones).
    #[serde(default)]
    supersedes: Option<Vec<String>>,
    /// `{source_url, license, retrieved_at}`, stored as `metadata.provenance`; required in a
    /// platform store.
    #[serde(default)]
    provenance: Option<Provenance>,
}

/// Default `source` for API ingests that name none.
pub const DEFAULT_API_SOURCE: &str = "api";

pub(crate) fn path_id(p: std::result::Result<Path<Uuid>, PathRejection>) -> Result<Uuid> {
    p.map(|Path(id)| id)
        .map_err(|e| StrataError::InvalidInput(format!("invalid id in path: {}", e.body_text())))
}

/// A body over the limit without a Content-Length (chunked) is only caught while the extractor
/// reads it; keep that 413 (rewritten to the JSON body by `json_413`) instead of calling it 422.
pub(crate) fn json_body<T>(b: std::result::Result<Json<T>, JsonRejection>) -> Result<T> {
    b.map(|Json(v)| v).map_err(|e| {
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            StrataError::PayloadTooLarge(e.body_text())
        } else {
            StrataError::InvalidInput(e.body_text())
        }
    })
}

pub(crate) fn query_params<T>(q: std::result::Result<Query<T>, QueryRejection>) -> Result<T> {
    q.map(|Query(v)| v)
        .map_err(|e| StrataError::InvalidInput(e.body_text()))
}

pub(crate) fn media_type(headers: &HeaderMap) -> String {
    headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_default()
}

pub async fn create_store(
    State(engine): State<AppState>,
    tenant: Tenant,
    body: std::result::Result<Json<CreateStoreBody>, JsonRejection>,
) -> Result<(StatusCode, Json<StoreView>)> {
    let b = json_body(body)?;
    let view = engine
        .create_store(
            tenant,
            NewStore {
                parent_id: b.parent_id,
                kind: b.kind,
                name: b.name,
                description: b.description,
            },
        )
        .await?;
    Ok((StatusCode::CREATED, Json(view)))
}

pub async fn list_stores(
    State(engine): State<AppState>,
    tenant: Tenant,
) -> Result<Json<StoreList>> {
    Ok(Json(StoreList {
        stores: engine.store_tree(tenant).await?,
    }))
}

pub async fn get_store(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
) -> Result<Json<StoreView>> {
    Ok(Json(engine.get_store(tenant, path_id(id)?).await?))
}

pub async fn update_store(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
    body: std::result::Result<Json<UpdateStoreBody>, JsonRejection>,
) -> Result<Json<StoreView>> {
    let id = path_id(id)?;
    let b = json_body(body)?;
    if b.name.is_none() && b.description.is_none() {
        return Err(StrataError::InvalidInput(
            "nothing to update: give name and/or description".into(),
        ));
    }
    Ok(Json(
        engine
            .update_store(tenant, id, b.name, b.description)
            .await?,
    ))
}

pub async fn delete_store(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
    params: std::result::Result<Query<DeleteStoreParams>, QueryRejection>,
) -> Result<StatusCode> {
    let id = path_id(id)?;
    let params = query_params(params)?;
    engine.delete_store(tenant, id, params.recursive).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// `text/plain` (or `text/markdown`, or no content type) body with
/// `?source=&title=&source_id=&source_run=`, or `application/json`
/// `{text, source?, title?, source_id?, source_run?, metadata?, topic_keys?, supersedes?,
/// provenance?: {source_url, license, retrieved_at}}` (JSON fields win over query params;
/// `metadata` and the platform-knowledge fields are JSON-only). A document already in the store
/// under the same `source_id` takes the new `source_run` / `metadata` / keys / title and is
/// answered 200 `existing`, not re-embedded. A platform store requires `provenance` with an
/// allow-listed licence (422); a store of a published pack is 409.
pub async fn ingest_document(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
    params: std::result::Result<Query<IngestParams>, QueryRejection>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, Json<IngestOutcome>)> {
    let store_id = path_id(id)?;
    let params = query_params(params)?;
    let media = media_type(&headers);
    let (text, source, title, tag, knowledge) = match media.as_str() {
        "application/json" => {
            let j: IngestJson = serde_json::from_slice(&body)
                .map_err(|e| StrataError::InvalidInput(format!("invalid JSON body: {e}")))?;
            let tag = SourceTag {
                source_id: j.source_id.or(params.source_id),
                source_run: j.source_run.or(params.source_run),
                metadata: j.metadata,
            };
            let knowledge = KnowledgeTag {
                topic_keys: j.topic_keys,
                supersedes: j.supersedes,
                provenance: j.provenance,
            };
            (
                j.text,
                j.source.or(params.source),
                j.title.or(params.title),
                tag,
                knowledge,
            )
        }
        "" | "text/plain" | "text/markdown" => {
            let text = String::from_utf8(body.to_vec())
                .map_err(|_| StrataError::InvalidInput("request body is not valid UTF-8".into()))?;
            let tag = SourceTag {
                source_id: params.source_id,
                source_run: params.source_run,
                metadata: None,
            };
            (
                text,
                params.source,
                params.title,
                tag,
                KnowledgeTag::default(),
            )
        }
        other => {
            return Err(StrataError::InvalidInput(format!(
                "unsupported content type `{other}`; use text/plain or application/json"
            )));
        }
    };
    let source = source.unwrap_or_else(|| DEFAULT_API_SOURCE.to_string());
    let outcome = engine
        .ingest_document_with(
            tenant,
            IngestRequest {
                store_id,
                source,
                title,
                text: &text,
                tag,
            },
            knowledge,
        )
        .await?;
    let status = match outcome {
        IngestOutcome::Created(_) => StatusCode::CREATED,
        IngestOutcome::Existing(_) => StatusCode::OK,
    };
    Ok((status, Json(outcome)))
}

#[derive(Debug, Default, Deserialize)]
pub struct PruneParams {
    #[serde(default)]
    pub except_run: Option<String>,
    #[serde(default)]
    pub all: bool,
    #[serde(default)]
    pub store_id: Option<Uuid>,
    #[serde(default)]
    pub allow_empty: bool,
}

/// `DELETE /api/sources/{source_id}/documents?except_run=|all=true&store_id=&allow_empty=`: removes
/// the source's documents (in one leaf with `store_id`) except those of run `except_run`, or all
/// of them with `all=true` (one of the two is required, else 422); see `Engine::prune_source`.
/// 200 `{deleted, kept, stores, refused_stores}`, also for an unknown source (all zero). A leaf
/// where none of the source's documents carries `except_run` is left untouched, and then the
/// answer is 409 `empty_sync_run` with the same four fields next to `error` (unless
/// `allow_empty=true`).
pub async fn prune_source(
    State(engine): State<AppState>,
    tenant: Tenant,
    source_id: std::result::Result<Path<String>, PathRejection>,
    params: std::result::Result<Query<PruneParams>, QueryRejection>,
) -> Result<Json<PruneReport>> {
    let Path(source_id) = source_id.map_err(|e| {
        StrataError::InvalidInput(format!("invalid source_id in path: {}", e.body_text()))
    })?;
    let p = query_params(params)?;
    Ok(Json(
        engine
            .prune_source(
                tenant,
                PruneRequest {
                    source_id,
                    except_run: p.except_run,
                    all: p.all,
                    store_id: p.store_id,
                    allow_empty: p.allow_empty,
                },
            )
            .await?,
    ))
}

pub async fn delete_document(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
) -> Result<StatusCode> {
    engine.delete_document(tenant, path_id(id)?).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn query(
    State(engine): State<AppState>,
    tenant: Tenant,
    body: std::result::Result<Json<QueryRequest>, JsonRejection>,
) -> Result<Json<QueryResponse>> {
    let req = json_body(body)?;
    let resp = engine.query(tenant, &req).await?;
    crate::metrics::record_query_timings(&resp.timings_ms);
    Ok(Json(resp))
}

pub async fn ask(
    State(engine): State<AppState>,
    tenant: Tenant,
    body: std::result::Result<Json<AskRequest>, JsonRejection>,
) -> Result<Json<AskResponse>> {
    let req = json_body(body)?;
    let resp = engine.ask(tenant, &req).await?;
    crate::metrics::record_ask_timings(&resp.timings_ms);
    Ok(Json(resp))
}

/// `GET /api/tenant/providers`: the tenant's embed / rerank providers and where each comes from
/// (`tenant` | `default` | `unconfigured`). Never a key: only `has_api_key`.
pub async fn get_tenant_providers(
    State(engine): State<AppState>,
    tenant: Tenant,
) -> Result<Json<ProvidersView>> {
    Ok(Json(engine.providers.view(tenant).await?))
}

/// `PUT /api/tenant/providers`: replaces the tenant's provider set (a capability left out or `null`
/// is removed). Validated against the deployment policy and probed live; 422 with a specific code
/// otherwise. Answers the stored view.
pub async fn put_tenant_providers(
    State(engine): State<AppState>,
    tenant: Tenant,
    body: std::result::Result<Json<PutProviders>, JsonRejection>,
) -> Result<Json<ProvidersView>> {
    let body = json_body(body)?;
    Ok(Json(engine.providers.replace(tenant, body).await?))
}

/// `POST /api/tenant/providers/test`: tries one provider without saving it (06 §3 A3): the PUT's
/// validation, SSRF guard and live probe; nothing stored, cached or metered. 200 `{ok, ...}` for
/// any provider answer (a failing provider is `ok: false`); 422 for an invalid body, a refused
/// host, or `secret_required_for_new_host` (no `api_key`, and `base_url` differs from the stored
/// registration's).
pub async fn test_tenant_provider(
    State(engine): State<AppState>,
    tenant: Tenant,
    body: std::result::Result<Json<TestProvider>, JsonRejection>,
) -> Result<Json<ProviderTestResult>> {
    let body = json_body(body)?;
    Ok(Json(engine.providers.test(tenant, body).await?))
}

/// `GET /api/stores/{id}/providers`: per capability a store can bind (embed, rerank), which
/// registration a request on this store uses (`store` | `ancestor` + `from_store` | `tenant` |
/// `default` | `unconfigured`) and its settings, plus the store's `embed_model`. Never a key.
pub async fn get_store_providers(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
) -> Result<Json<StoreProvidersView>> {
    Ok(Json(engine.store_providers(tenant, path_id(id)?).await?))
}

/// `PUT /api/stores/{id}/providers`: replaces the store's own registrations (left out or `null`
/// = removed, inherited again). Same body, validation and live probe as `PUT
/// /api/tenant/providers` minus `chat` (tenant-wide); an embed model other than the store's
/// `embed_model` is 409 `embed_model_mismatch` unless the store is empty (it is then re-tagged).
pub async fn put_store_providers(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
    body: std::result::Result<Json<PutStoreProviders>, JsonRejection>,
) -> Result<Json<StoreProvidersView>> {
    let id = path_id(id)?;
    let body = json_body(body)?;
    Ok(Json(engine.put_store_providers(tenant, id, body).await?))
}

/// `DELETE /api/stores/{id}/providers`: removes the store's own registrations; answers the view
/// (the store now inherits).
pub async fn delete_store_providers(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
) -> Result<Json<StoreProvidersView>> {
    let id = path_id(id)?;
    Ok(Json(
        engine
            .put_store_providers(tenant, id, PutStoreProviders::default())
            .await?,
    ))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReembedCopyBody {
    pub target_store: Uuid,
    /// Copy the source's vectors instead of re-embedding (the target must resolve to the
    /// source's embed model, else 409 `embed_model_mismatch`). Default false.
    #[serde(default)]
    pub copy_vectors: bool,
}

/// `POST /api/stores/{id}/reembed-copy {target_store, copy_vectors?}` → 202 with the `running`
/// job: re-embeds every document of `id`'s subtree into the empty `target_store` with the
/// target's providers (or, with `copy_vectors`, copies the source's vectors).
pub async fn start_reembed_copy(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
    body: std::result::Result<Json<ReembedCopyBody>, JsonRejection>,
) -> Result<(StatusCode, Json<ReembedCopyJob>)> {
    let id = path_id(id)?;
    let body = json_body(body)?;
    let job = engine
        .start_reembed_copy(tenant, id, body.target_store, body.copy_vectors)
        .await?;
    Ok((StatusCode::ACCEPTED, Json(job)))
}

/// `GET /api/stores/{id}/reembed-copy/{job}`: `{state, chunks_done, chunks_total, error?, ...}`.
pub async fn get_reembed_copy(
    State(engine): State<AppState>,
    tenant: Tenant,
    ids: std::result::Result<Path<(Uuid, Uuid)>, PathRejection>,
) -> Result<Json<ReembedCopyJob>> {
    let (id, job) = ids
        .map(|Path(p)| p)
        .map_err(|e| StrataError::InvalidInput(format!("invalid id in path: {}", e.body_text())))?;
    Ok(Json(engine.reembed_copy_job(tenant, id, job).await?))
}

/// Always 200; each component reports ok / down / unconfigured.
pub async fn health(State(engine): State<AppState>) -> Json<HealthReport> {
    Json(engine.health().await)
}

pub const DOCUMENTS_DEFAULT_LIMIT: i64 = 50;
pub const DOCUMENTS_MAX_LIMIT: i64 = 500;

#[derive(Debug, Default, Deserialize)]
pub struct ListDocumentsParams {
    #[serde(default)]
    pub limit: Option<i64>,
    #[serde(default)]
    pub cursor: Option<String>,
    /// Only this source's documents.
    #[serde(default)]
    pub source_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DocumentPage {
    pub items: Vec<DocumentListItem>,
    /// Opaque cursor for the next page; `null` on the last page.
    pub next: Option<String>,
}

/// `GET /api/stores/{id}/documents?limit=&cursor=&source_id=`: one store's own documents (only
/// `source_id`'s when given), newest first,
/// keyset-paged on `(created_at, id)`. Non-recursive: a router holds no documents, so it answers
/// an empty page. Unknown store → 404. `limit` defaults to 50; above 500 it is capped at 500.
pub async fn list_documents(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
    params: std::result::Result<Query<ListDocumentsParams>, QueryRejection>,
) -> Result<Json<DocumentPage>> {
    let store_id = path_id(id)?;
    let p = query_params(params)?;
    let limit = p.limit.unwrap_or(DOCUMENTS_DEFAULT_LIMIT);
    if limit < 1 {
        return Err(StrataError::InvalidInput(format!(
            "limit must be at least 1, got {limit}"
        )));
    }
    let limit = limit.min(DOCUMENTS_MAX_LIMIT);
    let after = p.cursor.as_deref().map(decode_cursor).transpose()?;
    let source_id = source_field("source_id", p.source_id)?;
    let mut conn = engine.pool.acquire().await?;
    let store = stores::get_store(&mut conn, tenant, store_id).await?;
    if store.kind != StoreKind::Leaf {
        return Ok(Json(DocumentPage {
            items: Vec::new(),
            next: None,
        }));
    }
    let mut items =
        documents::list_documents(&mut conn, store_id, limit + 1, after, source_id.as_deref())
            .await?;
    let more = items.len() as i64 > limit;
    items.truncate(limit as usize);
    let next = if more {
        items.last().map(|d| encode_cursor((d.created_at_ts, d.id)))
    } else {
        None
    };
    Ok(Json(DocumentPage { items, next }))
}

/// `POST /api/eval` body; every field optional.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalBody {
    #[serde(default)]
    pub modes: Option<Vec<EvalMode>>,
    #[serde(default)]
    pub beam: Option<usize>,
    #[serde(default)]
    pub min_score: Option<f32>,
    /// Stores outside the eval tree take beam slots and change routed scores, so by default the
    /// eval refuses (409). `true` reports anyway, marked `contaminated`, with verdict
    /// `CONTAMINATED` (never `PASS`).
    #[serde(default)]
    pub allow_foreign_stores: bool,
    /// `builtin` (the default: the server's question file) | `tenant` (the tenant's stored
    /// questions, see `/api/eval/questions`). `question_ids` or `questions` imply `tenant`.
    #[serde(default)]
    pub source: Option<EvalSource>,
    /// Only these stored questions (in this order), instead of all of them.
    #[serde(default)]
    pub question_ids: Option<Vec<Uuid>>,
    /// Questions sent with the request instead of stored ones (the run is stored, they are not).
    #[serde(default)]
    pub questions: Option<Vec<QuestionInput>>,
    /// Tenant questions only: evaluate within this store's subtree (default: the whole tree).
    #[serde(default, alias = "store_id")]
    pub root_id: Option<Uuid>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalSource {
    Builtin,
    Tenant,
}

#[derive(Debug, Serialize)]
pub struct EvalResponse {
    pub reports: Vec<ModeReport>,
    /// `PASS` | `FAIL`, or `CONTAMINATED` when foreign stores were present.
    pub verdict: &'static str,
    pub contaminated: bool,
    /// The stores outside the eval tree (empty on a clean run).
    pub foreign_stores: Vec<ForeignStoreOut>,
}

#[derive(Debug, Serialize)]
pub struct ForeignStoreOut {
    pub path: String,
    pub id: Uuid,
    pub kind: &'static str,
}

/// Reads a server-side eval file. Unreadable → 409 `eval_not_ready` with a generic message; the
/// path and the OS error are logged only (a tenant must not learn the server's filesystem).
async fn read_eval_file(path: &std::path::Path, what: &str) -> Result<String> {
    match tokio::fs::read_to_string(path).await {
        Ok(s) => Ok(s),
        Err(e) => {
            tracing::error!(path = %path.display(), error = %e, "eval {what} file cannot be read");
            Err(eval_unconfigured(what))
        }
    }
}

/// Another eval of the tenant is running (the built-in eval's refusal).
fn eval_busy() -> StrataError {
    StrataError::Conflict {
        message: "an eval is already running; retry when it finishes".into(),
        retryable: true,
    }
}

/// The server's eval `what` (questions / tree) is unusable; details are in the log only.
fn eval_unconfigured(what: &str) -> StrataError {
    StrataError::EvalNotReady(format!(
        "the server's eval {what} file is missing or invalid (see the server log)"
    ))
}

/// `POST /api/eval`: the HTTP twin of `strata eval --json` over the server's configured questions
/// (`STRATA_EVAL_QUESTIONS`) and tree (`STRATA_EVAL_TREE`, default `tree.json` next to them).
/// Never loads or ingests the tree: when the stores or documents it names are missing it answers
/// 409 `eval_not_ready` listing them; server paths and OS errors are never in a response. Slow
/// (one query per question per mode); it holds no transaction, so ingest and queries proceed while
/// it runs. One at a time per tenant: a second concurrent request of the same tenant gets 409.
/// Scores the caller's tenant's stores only.
///
/// With `source: "tenant"` (or `question_ids` / `questions`) it scores the tenant's own questions
/// instead (`crate::tenant_eval`), asynchronously: 202 with the `running` run and `Location:
/// /api/eval/runs/{id}`, which the client polls until it is `completed` or `failed`; a second
/// tenant run meanwhile is 409 `eval_running`. See `super::eval_api::start_tenant_eval`.
pub async fn eval(
    State(engine): State<AppState>,
    tenant: Tenant,
    headers: HeaderMap,
    body: std::result::Result<Json<EvalBody>, JsonRejection>,
) -> Result<Response> {
    // An empty body (no Content-Type) means all defaults.
    let b = match body {
        Err(JsonRejection::MissingJsonContentType(_)) => EvalBody::default(),
        other => json_body(other)?,
    };
    let defaults = EvalOptions::default();
    let opts = EvalOptions {
        modes: b.modes.unwrap_or(defaults.modes),
        beam: b.beam.unwrap_or(defaults.beam),
        min_score: b.min_score.unwrap_or(defaults.min_score),
    };
    if opts.modes.is_empty() {
        return Err(StrataError::InvalidInput(
            "modes must name at least one of embed, laya, hybrid, flat".into(),
        ));
    }
    if opts.beam == 0 {
        return Err(StrataError::InvalidInput("beam must be at least 1".into()));
    }
    if !opts.min_score.is_finite() {
        return Err(StrataError::InvalidInput(
            "min_score must be a finite number".into(),
        ));
    }
    let custom =
        b.source == Some(EvalSource::Tenant) || b.question_ids.is_some() || b.questions.is_some();
    if custom {
        if b.source == Some(EvalSource::Builtin) {
            return Err(StrataError::InvalidInput(
                "question_ids / questions need source \"tenant\" (or no source)".into(),
            ));
        }
        if b.allow_foreign_stores {
            return Err(StrataError::InvalidInput(
                "allow_foreign_stores applies to the built-in eval only".into(),
            ));
        }
    } else if b.root_id.is_some() {
        return Err(StrataError::InvalidInput(
            "root_id applies to tenant questions (source \"tenant\") only".into(),
        ));
    }
    // One eval at a time, refused rather than queued: each runs a query per question per mode
    // (paid laya calls in laya / hybrid), so N parallel requests would cost N times that.
    let Some(_slot) = engine.eval_slots.try_acquire(tenant) else {
        if custom {
            return Err(crate::db::eval::already_running());
        }
        return Err(eval_busy());
    };
    if custom {
        let actor = crate::metering::Caller::from_headers(&headers).actor_id;
        let run = super::eval_api::start_tenant_eval(
            &engine,
            tenant,
            actor,
            super::eval_api::TenantEvalRequest {
                question_ids: b.question_ids,
                questions: b.questions,
                root_id: b.root_id,
                opts,
            },
        )
        .await?;
        let location = format!("/api/eval/runs/{}", run.id);
        return Ok((
            StatusCode::ACCEPTED,
            [(axum::http::header::LOCATION, location)],
            Json(run),
        )
            .into_response());
    }
    // a tenant eval running in the background holds no slot; its stored row says it runs
    if crate::db::eval::has_running(&engine.pool, tenant).await? {
        return Err(eval_busy());
    }
    let questions_path = engine.cfg.eval_questions.clone();
    let tree_path = resolve_tree_path(&questions_path, engine.cfg.eval_tree.as_deref(), None);
    let questions =
        parse_questions(&read_eval_file(&questions_path, "questions").await?).map_err(|e| {
            tracing::error!(path = %questions_path.display(), error = %e, "eval questions invalid");
            eval_unconfigured("questions")
        })?;
    let spec: TreeSpec =
        serde_json::from_str(&read_eval_file(&tree_path, "tree").await?).map_err(|e| {
            tracing::error!(path = %tree_path.display(), error = %e, "eval tree invalid");
            eval_unconfigured("tree")
        })?;
    // Same guard as `strata eval`: a store outside the tree competes for beam slots, so routed
    // scores would not be the tree's.
    let foreign: Vec<ForeignStoreOut> = foreign_stores(&spec, &engine.store_tree(tenant).await?)
        .into_iter()
        .map(|f| ForeignStoreOut {
            path: f.path.join("/"),
            id: f.id,
            kind: f.kind.as_str(),
        })
        .collect();
    if !foreign.is_empty() && !b.allow_foreign_stores {
        let listing: Vec<String> = foreign
            .iter()
            .map(|f| format!("{} id={} kind={}", f.path, f.id, f.kind))
            .collect();
        return Err(StrataError::Conflict {
            message: format!(
                "refusing to evaluate: {} store(s) outside the eval tree would change routed \
                 scores (remove them, or send allow_foreign_stores=true for a CONTAMINATED report): {}",
                foreign.len(),
                listing.join("; ")
            ),
            retryable: false,
        });
    }
    // Options were validated above, so an InvalidInput here is evaluate's precondition list: tree
    // stores missing, files not ingested, or questions naming leaves the tree lacks.
    let reports = evaluate_with(&engine, tenant, &spec, &questions, &opts)
        .await
        .map_err(|e| match e {
            // `problems` names the tree's stores and files (as in tree.json), never a path
            StrataError::InvalidInput(problems) => StrataError::EvalNotReady(format!(
                "the eval tree is not loaded for this tenant (run `strata eval --tenant <uuid> \
                 --load <tree.json>`):\n{problems}"
            )),
            other => other,
        })?;
    let contaminated = !foreign.is_empty();
    let verdict = if contaminated {
        "CONTAMINATED"
    } else if verdict_passes(&reports) {
        "PASS"
    } else {
        "FAIL"
    };
    Ok(Json(EvalResponse {
        reports,
        verdict,
        contaminated,
        foreign_stores: foreign,
    })
    .into_response())
}
