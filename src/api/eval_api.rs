//! Tenant eval questions and stored runs: `/api/eval/questions[...]`, `/api/eval/runs[...]`, and
//! the tenant branch of `POST /api/eval` (`handlers::eval`). Same auth and tenant rules as every
//! route; all data is the caller's tenant's only.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::{JsonRejection, PathRejection, QueryRejection};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::error::INTERNAL_MESSAGE;
use super::handlers::{AppState, json_body, media_type, path_id, query_params};
use crate::db::documents::{decode_cursor, encode_cursor};
use crate::db::eval::{self as db, EvalRun, EvalRunSummary, NewRun};
use crate::engine::Engine;
use crate::error::{Result, StrataError};
use crate::eval::{EvalOptions, verdict_passes};
use crate::metering::Caller;
use crate::tenant::Tenant;
use crate::tenant_eval::{
    EVAL_MAX_QUESTIONS, Prepared, Question, QuestionInput, QuestionPatch, RunQuestion,
    parse_questions_csv, prepare,
};

/// Most problems an import refusal lists.
const MAX_LISTED_PROBLEMS: usize = 20;

fn actor(headers: &HeaderMap) -> Option<Uuid> {
    Caller::from_headers(headers).actor_id
}

/// A list page. `next` is the opaque cursor of the next page; `null` on the last one.
#[derive(Debug, Serialize)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub next: Option<String>,
}

/// `GET /api/eval/questions`: all the tenant's questions (at most 200), oldest first, as one page.
pub async fn list_questions(
    State(engine): State<AppState>,
    tenant: Tenant,
) -> Result<Json<Page<Question>>> {
    Ok(Json(Page {
        items: db::list_questions(&engine.pool, tenant).await?,
        next: None,
    }))
}

/// `POST /api/eval/questions`: 201 with the stored question. 422 `invalid_input` (no query, no
/// expectation, a field too long), 409 `question_exists`, 422 `too_many_questions` at 200.
pub async fn create_question(
    State(engine): State<AppState>,
    tenant: Tenant,
    headers: HeaderMap,
    body: std::result::Result<Json<QuestionInput>, JsonRejection>,
) -> Result<(StatusCode, Json<Question>)> {
    let q = json_body(body)?.validated()?;
    let row = db::create_question(&engine.pool, tenant, &q, actor(&headers)).await?;
    Ok((StatusCode::CREATED, Json(row)))
}

/// `PATCH /api/eval/questions/{id}`: absent fields kept, `null` clears an optional one.
pub async fn update_question(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
    body: std::result::Result<Json<QuestionPatch>, JsonRejection>,
) -> Result<Json<Question>> {
    let id = path_id(id)?;
    let patch = json_body(body)?;
    if patch.is_empty() {
        return Err(StrataError::InvalidInput(
            "nothing to update: give at least one field".into(),
        ));
    }
    Ok(Json(
        db::patch_question(&engine.pool, tenant, id, patch).await?,
    ))
}

/// `DELETE /api/eval/questions/{id}`: 204. Stored runs keep their snapshot of it.
pub async fn delete_question(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
) -> Result<StatusCode> {
    db::delete_question(&engine.pool, tenant, path_id(id)?).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportJson {
    questions: Vec<QuestionInput>,
}

#[derive(Debug, Serialize)]
pub struct ImportResult {
    pub added: usize,
    pub total: usize,
}

fn refuse_import(problems: Vec<String>) -> StrataError {
    let more = problems.len().saturating_sub(MAX_LISTED_PROBLEMS);
    let mut listed: Vec<String> = problems.into_iter().take(MAX_LISTED_PROBLEMS).collect();
    if more > 0 {
        listed.push(format!("(+{more} more)"));
    }
    StrataError::InvalidInput(format!("nothing imported: {}", listed.join("; ")))
}

/// `POST /api/eval/questions/import`: CSV (`text/csv`, `text/plain` or no content type; header
/// row with `query` and at least one of `expected_document`, `expected_document_ids`,
/// `expected_chunk_ids`, `expected_text`, optional `expected_leaf`; ids separated by `;`, `|` or
/// spaces) or JSON `{"questions":[...]}`. All or nothing: any bad row, a query already stored
/// (409 `question_exists`) or passing 200 questions (422 `too_many_questions`) imports none.
/// 201 `{added, total}`.
pub async fn import_questions(
    State(engine): State<AppState>,
    tenant: Tenant,
    headers: HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, Json<ImportResult>)> {
    let qs: Vec<QuestionInput> = match media_type(&headers).as_str() {
        "application/json" => {
            let j: ImportJson = serde_json::from_slice(&body)
                .map_err(|e| StrataError::InvalidInput(format!("invalid JSON body: {e}")))?;
            let mut problems = Vec::new();
            let mut seen = HashSet::new();
            let mut out = Vec::with_capacity(j.questions.len());
            for (i, q) in j.questions.into_iter().enumerate() {
                let q = q.normalized();
                let mut p = q.problems();
                if p.is_empty() && !seen.insert(q.query.to_lowercase()) {
                    p.push("the same query is in an earlier question".into());
                }
                if p.is_empty() {
                    out.push(q);
                } else {
                    problems.push(format!("question {}: {}", i + 1, p.join("; ")));
                }
            }
            if !problems.is_empty() {
                return Err(refuse_import(problems));
            }
            if out.is_empty() {
                return Err(StrataError::InvalidInput("questions is empty".into()));
            }
            out
        }
        "" | "text/csv" | "text/plain" => {
            let text = String::from_utf8(body.to_vec())
                .map_err(|_| StrataError::InvalidInput("request body is not valid UTF-8".into()))?;
            parse_questions_csv(&text).map_err(refuse_import)?
        }
        other => {
            return Err(StrataError::InvalidInput(format!(
                "unsupported content type `{other}`; use text/csv or application/json"
            )));
        }
    };
    if qs.len() > EVAL_MAX_QUESTIONS {
        return Err(StrataError::TooManyQuestions(format!(
            "the import has {} questions; a tenant holds at most {EVAL_MAX_QUESTIONS}",
            qs.len()
        )));
    }
    let (added, total) = db::import_questions(&engine.pool, tenant, &qs, actor(&headers)).await?;
    Ok((StatusCode::CREATED, Json(ImportResult { added, total })))
}

pub const RUNS_DEFAULT_LIMIT: i64 = 20;
pub const RUNS_MAX_LIMIT: i64 = 100;

#[derive(Debug, Default, Deserialize)]
pub struct ListRunsParams {
    #[serde(default)]
    pub limit: Option<i64>,
    #[serde(default)]
    pub cursor: Option<String>,
}

/// `GET /api/eval/runs?limit=&cursor=`: the tenant's stored runs, newest first, as summaries
/// (`EvalRunSummary`: reports and `question_count`, no question snapshot and no per-question
/// `results`; both are on `GET /api/eval/runs/{id}`).
/// `limit` defaults to 20, capped at 100.
pub async fn list_runs(
    State(engine): State<AppState>,
    tenant: Tenant,
    params: std::result::Result<Query<ListRunsParams>, QueryRejection>,
) -> Result<Json<Page<EvalRunSummary>>> {
    let p = query_params(params)?;
    let limit = p.limit.unwrap_or(RUNS_DEFAULT_LIMIT);
    if limit < 1 {
        return Err(StrataError::InvalidInput(format!(
            "limit must be at least 1, got {limit}"
        )));
    }
    let limit = limit.min(RUNS_MAX_LIMIT);
    let after = p.cursor.as_deref().map(decode_cursor).transpose()?;
    // a run whose worker died reads `failed`, not `running` forever
    db::fail_stale_runs(&engine.pool, Some(tenant), db::EVAL_STALE_AFTER).await?;
    let mut items = db::list_runs(&engine.pool, tenant, limit + 1, after).await?;
    let more = items.len() as i64 > limit;
    items.truncate(limit as usize);
    let next = if more {
        items.last().map(|r| encode_cursor((r.started_at_ts, r.id)))
    } else {
        None
    };
    Ok(Json(Page { items, next }))
}

/// `GET /api/eval/runs/{id}`: one run with `results` (per question and mode: `rank`, `leaf_hit`).
pub async fn get_run(
    State(engine): State<AppState>,
    tenant: Tenant,
    id: std::result::Result<Path<Uuid>, PathRejection>,
) -> Result<Json<EvalRun>> {
    let id = path_id(id)?;
    // a run whose worker died reads `failed`, not `running` forever
    db::fail_stale_runs(&engine.pool, Some(tenant), db::EVAL_STALE_AFTER).await?;
    Ok(Json(db::get_run(&engine.pool, tenant, id).await?))
}

/// The tenant branch of `POST /api/eval` (options already validated by the handler).
pub struct TenantEvalRequest {
    pub question_ids: Option<Vec<Uuid>>,
    pub questions: Option<Vec<QuestionInput>>,
    pub root_id: Option<Uuid>,
    pub opts: EvalOptions,
}

/// How often a running tenant eval beats: `heartbeat_at` and its progress (`progress.done`).
/// Far below `db::EVAL_STALE_AFTER`, so only a dead worker's run is ever swept.
const PROGRESS_EVERY: std::time::Duration = std::time::Duration::from_secs(1);

/// Starts a tenant eval: loads the questions (stored: all, or `question_ids` in order; or inline),
/// checks them against the tree, stores the run as `running` and scores it in a background task,
/// independent of the request (a 200-question hybrid run takes minutes; a proxy cuts a request off
/// long before). Answers the running run's summary. While it runs, the stored `running` row is
/// what keeps the tenant's other evals out (the `eval_runs_one_running` index, and the built-in
/// eval's check), not the request's eval slot: the row turns `completed` / `failed` in the same
/// commit that stores the outcome, so a client that polled it finished can start the next run at
/// once.
/// Refusals are immediate: 422 for an empty or oversized set or bad inline questions, 404 for a
/// `question_ids` entry or `root_id` that is not the tenant's, 409 `eval_not_ready` when the
/// tenant has no questions or one names a leaf / document / chunk that is not there, 409
/// `eval_running` when another run of the tenant is in progress (on any replica).
pub async fn start_tenant_eval(
    engine: &AppState,
    tenant: Tenant,
    started_by: Option<Uuid>,
    req: TenantEvalRequest,
) -> Result<EvalRunSummary> {
    // refused before any work (a stale run is swept first); the index on insert below is what
    // closes the race
    if db::has_running(&engine.pool, tenant).await? {
        return Err(db::already_running());
    }
    let (source, questions): (&'static str, Vec<RunQuestion>) =
        match (req.questions, req.question_ids) {
            (Some(_), Some(_)) => {
                return Err(StrataError::InvalidInput(
                    "send questions or question_ids, not both".into(),
                ));
            }
            (Some(inline), None) => {
                let mut problems = Vec::new();
                let mut out = Vec::with_capacity(inline.len());
                for (i, q) in inline.into_iter().enumerate() {
                    match q.validated() {
                        Ok(q) => out.push(RunQuestion::new(None, q)),
                        Err(e) => problems.push(format!("question {}: {e}", i + 1)),
                    }
                }
                if !problems.is_empty() {
                    return Err(StrataError::InvalidInput(problems.join("; ")));
                }
                ("inline", out)
            }
            (None, Some(ids)) => {
                let mut seen = HashSet::new();
                let ids: Vec<Uuid> = ids.into_iter().filter(|id| seen.insert(*id)).collect();
                if ids.len() > EVAL_MAX_QUESTIONS {
                    return Err(StrataError::TooManyQuestions(format!(
                        "question_ids names {}; one eval takes at most {EVAL_MAX_QUESTIONS}",
                        ids.len()
                    )));
                }
                let qs = db::questions_by_id(&engine.pool, tenant, &ids).await?;
                ("tenant", stored(qs))
            }
            (None, None) => {
                let qs = db::list_questions(&engine.pool, tenant).await?;
                if qs.is_empty() {
                    return Err(StrataError::EvalNotReady(
                        "this tenant has no eval questions; add some with POST /api/eval/questions"
                            .into(),
                    ));
                }
                ("tenant", stored(qs))
            }
        };
    if questions.is_empty() {
        return Err(StrataError::InvalidInput("no questions to evaluate".into()));
    }
    if questions.len() > EVAL_MAX_QUESTIONS {
        return Err(StrataError::TooManyQuestions(format!(
            "{} questions; one eval takes at most {EVAL_MAX_QUESTIONS}",
            questions.len()
        )));
    }
    let prepared = prepare(engine, tenant, questions, req.root_id, req.opts.clone()).await?;
    let run = db::insert_running(
        &engine.pool,
        tenant,
        NewRun {
            source,
            root_id: req.root_id,
            modes: prepared.mode_labels(),
            beam: i32::try_from(req.opts.beam).unwrap_or(i32::MAX),
            min_score: req.opts.min_score,
            started_by,
            questions: &prepared.questions,
            total: i32::try_from(prepared.total()).unwrap_or(i32::MAX),
        },
    )
    .await?;
    let scoring = execute(std::sync::Arc::clone(engine), tenant, run.id, prepared);
    spawn_supervised(engine, tenant, run.id, scoring);
    Ok(run)
}

/// Runs `work` (the scoring of the stored `running` run `run_id`) in the background, tracked in
/// `engine.eval_tasks` while it runs: a shutdown fails it `shutdown`. `work` runs in a task of its
/// own, so an error or a panic fails the run (`error`: the error's message, or the generic
/// internal one for a 500 or a panic) instead of leaving it `running`. Public for tests.
pub fn spawn_supervised<F>(
    engine: &AppState,
    tenant: Tenant,
    run_id: Uuid,
    work: F,
) -> tokio::task::JoinHandle<()>
where
    F: std::future::Future<Output = Result<()>> + Send + 'static,
{
    let tracked = engine.eval_tasks.track(tenant, run_id);
    let engine = std::sync::Arc::clone(engine);
    tokio::spawn(async move {
        let _tracked = tracked;
        let outcome = tokio::spawn(work).await;
        let message = match outcome {
            Ok(Ok(())) => return,
            Ok(Err(e)) => {
                // as an error response would put it: a 500's detail stays in the log
                if e.status() == 500 {
                    tracing::error!(run = %run_id, error = %e, detail = ?e, "tenant eval failed");
                    INTERNAL_MESSAGE.to_string()
                } else {
                    tracing::warn!(run = %run_id, error = %e, "tenant eval failed");
                    e.to_string()
                }
            }
            Err(join) => {
                tracing::error!(run = %run_id, error = %join, "tenant eval task panicked");
                INTERNAL_MESSAGE.to_string()
            }
        };
        if let Err(e) = db::fail_run(&engine.pool, tenant, run_id, &message).await {
            tracing::error!(
                run = %run_id,
                error = %e,
                "cannot store a tenant eval's failure; it is swept as interrupted once stale"
            );
        }
    })
}

/// Scores a stored running run, beating (heartbeat + progress) every [`PROGRESS_EVERY`], and
/// finishes it `completed`: verdict, reports and all per-question results at the end, in one
/// transaction. When a beat finds the run no longer `running` (swept as interrupted because its
/// beats did not land for `db::EVAL_STALE_AFTER`), scoring stops and nothing more is written: the
/// `failed` row stands.
async fn execute(engine: AppState, tenant: Tenant, run_id: Uuid, prepared: Prepared) -> Result<()> {
    let engine: &Engine = &engine;
    let done = AtomicUsize::new(0);
    let scoring = prepared.score(engine, tenant, &done);
    tokio::pin!(scoring);
    let mut tick = tokio::time::interval(PROGRESS_EVERY);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let scored = loop {
        tokio::select! {
            r = &mut scoring => break r?,
            _ = tick.tick() => {
                let n = done.load(Ordering::Relaxed);
                // every tick, progress changed or not: the beat is what keeps the run from being
                // swept. A failed write must not fail the run (the sweep threshold allows for it).
                match db::heartbeat(
                    &engine.pool,
                    tenant,
                    run_id,
                    i32::try_from(n).unwrap_or(i32::MAX),
                )
                .await
                {
                    Ok(true) => {}
                    Ok(false) => {
                        tracing::warn!(
                            run = %run_id,
                            "tenant eval was swept as interrupted; scoring stopped"
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        tracing::warn!(run = %run_id, error = %e, "eval heartbeat not stored");
                    }
                }
            }
        }
    };
    let verdict = if verdict_passes(&scored.reports) {
        "PASS"
    } else {
        "FAIL"
    };
    let reports = serde_json::to_value(&scored.reports).expect("ModeReport serialises");
    let stored = db::complete_run(
        &engine.pool,
        tenant,
        run_id,
        verdict,
        &reports,
        &scored.results,
    )
    .await?;
    if !stored {
        tracing::warn!(
            run = %run_id,
            "tenant eval finished after it was swept as interrupted; result dropped"
        );
    }
    Ok(())
}

fn stored(qs: Vec<Question>) -> Vec<RunQuestion> {
    qs.into_iter()
        .map(|q| RunQuestion::new(Some(q.id), q.input()))
        .collect()
}
