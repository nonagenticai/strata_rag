//! `eval_questions`, `eval_runs`, `eval_run_results` (migrations 0012-0015). Every statement
//! filters on the tenant: another tenant's id behaves exactly like an id that does not exist.
//! The one exception is the all-tenant sweep of stale runs, `fail_stale_runs(pool, None, ..)`.

use serde::Serialize;
use serde_json::Value;
use sqlx::postgres::PgRow;
use sqlx::types::Json;
use sqlx::types::chrono::{DateTime, Utc};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::db::documents::DocumentCursor;
use crate::error::{Result, StrataError};
use crate::tenant::Tenant;
use crate::tenant_eval::{
    EVAL_MAX_QUESTIONS, Question, QuestionInput, QuestionPatch, ResultRow, RunQuestion,
};

fn ts(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%dT%H:%M:%S%.6fZ").to_string()
}

const QUESTION_COLS: &str = "id, query, expected_document, expected_leaf, expected_document_ids, \
     expected_chunk_ids, expected_text, created_by, created_at, updated_at";

fn question(r: &PgRow) -> Result<Question> {
    Ok(Question {
        id: r.try_get("id")?,
        query: r.try_get("query")?,
        expected_document: r.try_get("expected_document")?,
        expected_leaf: r.try_get("expected_leaf")?,
        expected_document_ids: r.try_get("expected_document_ids")?,
        expected_chunk_ids: r.try_get("expected_chunk_ids")?,
        expected_text: r.try_get("expected_text")?,
        created_by: r.try_get("created_by")?,
        created_at: ts(r.try_get("created_at")?),
        updated_at: ts(r.try_get("updated_at")?),
    })
}

fn duplicate(q: &QuestionInput) -> StrataError {
    StrataError::DuplicateQuestion(format!(
        "the question `{}` is already in this tenant's set",
        q.query
    ))
}

/// Serialises a tenant's question writes (the cap check and the insert are one decision).
async fn lock_tenant(tx: &mut sqlx::PgConnection, tenant: Tenant) -> Result<()> {
    sqlx::query(
        "SELECT pg_advisory_xact_lock(hashtextextended('strata.eval_questions:' || $1::text, 0))",
    )
    .bind(tenant.id())
    .execute(&mut *tx)
    .await?;
    Ok(())
}

async fn count(tx: &mut sqlx::PgConnection, tenant: Tenant) -> Result<i64> {
    Ok(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM eval_questions WHERE tenant = $1")
            .bind(tenant.id())
            .fetch_one(&mut *tx)
            .await?,
    )
}

fn too_many(have: i64, adding: usize) -> StrataError {
    StrataError::TooManyQuestions(format!(
        "that would make {} questions; a tenant holds at most {EVAL_MAX_QUESTIONS}",
        have + adding as i64
    ))
}

/// The tenant's questions, oldest first.
pub async fn list_questions(pool: &PgPool, tenant: Tenant) -> Result<Vec<Question>> {
    let rows = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {QUESTION_COLS} FROM eval_questions WHERE tenant = $1 ORDER BY created_at, id"
    )))
    .bind(tenant.id())
    .fetch_all(pool)
    .await?;
    rows.iter().map(question).collect()
}

/// The tenant's questions with these ids, in the order asked; `NotFound` naming any id that is
/// not one of them.
pub async fn questions_by_id(pool: &PgPool, tenant: Tenant, ids: &[Uuid]) -> Result<Vec<Question>> {
    let rows = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {QUESTION_COLS} FROM eval_questions WHERE tenant = $1 AND id = ANY($2)"
    )))
    .bind(tenant.id())
    .bind(ids)
    .fetch_all(pool)
    .await?;
    let found: Vec<Question> = rows.iter().map(question).collect::<Result<_>>()?;
    let missing: Vec<String> = ids
        .iter()
        .filter(|id| !found.iter().any(|q| q.id == **id))
        .map(Uuid::to_string)
        .collect();
    if !missing.is_empty() {
        return Err(StrataError::NotFound(format!(
            "eval question(s) {}",
            missing.join(", ")
        )));
    }
    Ok(ids
        .iter()
        .filter_map(|id| found.iter().find(|q| q.id == *id).cloned())
        .collect())
}

pub async fn get_question(pool: &PgPool, tenant: Tenant, id: Uuid) -> Result<Question> {
    let row = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {QUESTION_COLS} FROM eval_questions WHERE tenant = $1 AND id = $2"
    )))
    .bind(tenant.id())
    .bind(id)
    .fetch_optional(pool)
    .await?;
    match row {
        Some(r) => question(&r),
        None => Err(StrataError::NotFound(format!("eval question {id}"))),
    }
}

async fn insert_one(
    tx: &mut sqlx::PgConnection,
    tenant: Tenant,
    q: &QuestionInput,
    actor: Option<Uuid>,
) -> Result<Option<Question>> {
    let row = sqlx::query(sqlx::AssertSqlSafe(format!(
        "INSERT INTO eval_questions (tenant, query, expected_document, expected_leaf, \
           expected_document_ids, expected_chunk_ids, expected_text, created_by) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8) \
         ON CONFLICT (tenant, lower(query)) DO NOTHING \
         RETURNING {QUESTION_COLS}"
    )))
    .bind(tenant.id())
    .bind(&q.query)
    .bind(&q.expected_document)
    .bind(&q.expected_leaf)
    .bind(&q.expected_document_ids)
    .bind(&q.expected_chunk_ids)
    .bind(&q.expected_text)
    .bind(actor)
    .fetch_optional(&mut *tx)
    .await?;
    row.as_ref().map(question).transpose()
}

/// Adds one (validated) question. `DuplicateQuestion` when the tenant already has that query
/// (case-insensitive), `TooManyQuestions` at the cap.
pub async fn create_question(
    pool: &PgPool,
    tenant: Tenant,
    q: &QuestionInput,
    actor: Option<Uuid>,
) -> Result<Question> {
    let mut tx = pool.begin().await?;
    lock_tenant(&mut tx, tenant).await?;
    let have = count(&mut tx, tenant).await?;
    if have as usize >= EVAL_MAX_QUESTIONS {
        return Err(too_many(have, 1));
    }
    let row = insert_one(&mut tx, tenant, q, actor)
        .await?
        .ok_or_else(|| duplicate(q))?;
    tx.commit().await?;
    Ok(row)
}

/// Adds every (validated) question or none. Answers `(added, total)`.
pub async fn import_questions(
    pool: &PgPool,
    tenant: Tenant,
    qs: &[QuestionInput],
    actor: Option<Uuid>,
) -> Result<(usize, usize)> {
    let mut tx = pool.begin().await?;
    lock_tenant(&mut tx, tenant).await?;
    let have = count(&mut tx, tenant).await?;
    if have as usize + qs.len() > EVAL_MAX_QUESTIONS {
        return Err(too_many(have, qs.len()));
    }
    for q in qs {
        // a duplicate rolls the whole import back (the transaction is dropped)
        insert_one(&mut tx, tenant, q, actor)
            .await?
            .ok_or_else(|| duplicate(q))?;
    }
    tx.commit().await?;
    Ok((qs.len(), have as usize + qs.len()))
}

/// Applies `patch` to the tenant's question atomically: the row is read `FOR UPDATE`, patched,
/// validated and written in one transaction, so two concurrent PATCHes of different fields both
/// land (the second waits for the first and patches its result). `NotFound` for an id that is not
/// the tenant's; `InvalidInput` when the patched question is invalid; `DuplicateQuestion` when
/// another of its questions has that query.
pub async fn patch_question(
    pool: &PgPool,
    tenant: Tenant,
    id: Uuid,
    patch: QuestionPatch,
) -> Result<Question> {
    let mut tx = pool.begin().await?;
    let cur = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {QUESTION_COLS} FROM eval_questions WHERE tenant = $1 AND id = $2 FOR UPDATE"
    )))
    .bind(tenant.id())
    .bind(id)
    .fetch_optional(&mut *tx)
    .await?
    .ok_or_else(|| StrataError::NotFound(format!("eval question {id}")))?;
    let q = patch.apply(question(&cur)?.input()).validated()?;
    let res = sqlx::query(sqlx::AssertSqlSafe(format!(
        "UPDATE eval_questions SET query = $3, expected_document = $4, expected_leaf = $5, \
           expected_document_ids = $6, expected_chunk_ids = $7, expected_text = $8, \
           updated_at = clock_timestamp() \
         WHERE tenant = $1 AND id = $2 \
         RETURNING {QUESTION_COLS}"
    )))
    .bind(tenant.id())
    .bind(id)
    .bind(&q.query)
    .bind(&q.expected_document)
    .bind(&q.expected_leaf)
    .bind(&q.expected_document_ids)
    .bind(&q.expected_chunk_ids)
    .bind(&q.expected_text)
    .fetch_one(&mut *tx)
    .await;
    let row = match res {
        Ok(r) => question(&r)?,
        Err(sqlx::Error::Database(db))
            if db.constraint() == Some("eval_questions_tenant_query_key") =>
        {
            return Err(duplicate(&q));
        }
        Err(e) => return Err(e.into()),
    };
    tx.commit().await?;
    Ok(row)
}

pub async fn delete_question(pool: &PgPool, tenant: Tenant, id: Uuid) -> Result<()> {
    let n = sqlx::query("DELETE FROM eval_questions WHERE tenant = $1 AND id = $2")
        .bind(tenant.id())
        .bind(id)
        .execute(pool)
        .await?
        .rows_affected();
    if n == 0 {
        return Err(StrataError::NotFound(format!("eval question {id}")));
    }
    Ok(())
}

// ---------- runs ----------

/// A run in progress: `running` until its background task finishes it `completed` or `failed`.
pub const RUNNING: &str = "running";
pub const COMPLETED: &str = "completed";
pub const FAILED: &str = "failed";
/// The error of a run whose worker stopped beating (its process died): see [`fail_stale_runs`].
pub const INTERRUPTED: &str = "interrupted";
/// A `running` run whose `heartbeat_at` (migration 0015) is older than this has lost its worker
/// and is failed `interrupted`. The worker beats about once a second, so this is ~60 missed beats:
/// a live worker whose beats are slow to land (a busy pool) is not swept. It is not how a stopping
/// process ends its runs: a graceful shutdown fails them `shutdown` itself
/// (`EvalTasks::fail_on_shutdown`); only a process that died without that (killed, crashed, the
/// shutdown write failed or timed out) leaves a run for this sweep.
pub const EVAL_STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(60);
/// How often every API process sweeps stale runs of all tenants (besides at start, and lazily per
/// tenant on the eval run reads and on `POST /api/eval`).
pub const EVAL_SWEEP_EVERY: std::time::Duration = std::time::Duration::from_secs(30);
/// The unique index (migration 0014) that allows one running run per tenant.
const ONE_RUNNING: &str = "eval_runs_one_running";

/// How far a run is: outcomes (question x mode) scored, out of `total`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Progress {
    pub done: i32,
    pub total: i32,
}

/// A stored eval run (`GET /api/eval/runs/{id}`). Field names follow the agent web app's `EvalRun`.
/// While `running`: `finished_at`, `took_ms`, `verdict` and `reports` are `null` and `results` is
/// `[]`; `progress` says how far it is. `failed`: `error` says why, `finished_at` / `took_ms` are
/// set, `verdict` / `reports` stay `null`.
#[derive(Debug, Clone, Serialize)]
pub struct EvalRun {
    pub id: Uuid,
    /// `tenant` (stored questions) | `inline` (questions sent in the request)
    pub source: String,
    /// The store the eval was scoped to; `null` = the tenant's whole tree.
    pub root_id: Option<Uuid>,
    pub modes: Vec<String>,
    pub beam: i32,
    pub min_score: f32,
    /// `running` | `completed` | `failed`
    pub status: String,
    /// Why a `failed` run failed; `null` otherwise.
    pub error: Option<String>,
    pub progress: Progress,
    pub verdict: Option<String>,
    pub started_by: Option<Uuid>,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub took_ms: Option<i64>,
    /// What was evaluated, as it was at run time.
    pub questions: Vec<RunQuestion>,
    /// One `ModeReport` per mode: `mode, n, recall_at_1, recall_at_5, recall_at_10,
    /// leaf_accuracy, laya_consulted, laya_fallbacks`. `null` until completed.
    pub reports: Value,
    /// Per question and mode (only on `GET /api/eval/runs/{id}`); `[]` until completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<RunResult>>,
    #[serde(skip)]
    pub started_at_ts: DateTime<Utc>,
}

/// A run as `GET /api/eval/runs` lists it (and the `202` answer of a tenant `POST /api/eval`):
/// everything but the question snapshot (only its size, `question_count`) and the per-question
/// `results`; both are on `GET /api/eval/runs/{id}`.
#[derive(Debug, Clone, Serialize)]
pub struct EvalRunSummary {
    pub id: Uuid,
    pub source: String,
    pub root_id: Option<Uuid>,
    pub modes: Vec<String>,
    pub beam: i32,
    pub min_score: f32,
    pub status: String,
    pub error: Option<String>,
    pub progress: Progress,
    pub verdict: Option<String>,
    pub started_by: Option<Uuid>,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub took_ms: Option<i64>,
    /// How many questions the run evaluated.
    pub question_count: i32,
    pub reports: Value,
    #[serde(skip)]
    pub started_at_ts: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunResult {
    pub index: i32,
    pub question_id: Option<Uuid>,
    pub mode: String,
    /// 1-based rank of the first expected result; `null` = not in the top 10.
    pub rank: Option<i32>,
    /// Routing reached `expected_leaf`; `null` for flat or without an expected leaf.
    pub leaf_hit: Option<bool>,
}

/// What a run stores when it starts. `started_at` is the insert's database time: `started_at`,
/// `finished_at` and `heartbeat_at` all come from the database clock, so `took_ms` never mixes an
/// API replica's clock with the database's.
pub struct NewRun<'a> {
    pub source: &'static str,
    pub root_id: Option<Uuid>,
    pub modes: Vec<String>,
    pub beam: i32,
    pub min_score: f32,
    pub started_by: Option<Uuid>,
    pub questions: &'a [RunQuestion],
    /// question x mode outcomes the run will score
    pub total: i32,
}

/// The columns every run read shares (the summary's `question_count` and the detail's `questions`
/// are added per query).
const RUN_BASE_COLS: &str = "id, source, root_id, modes, beam, min_score, status, error, \
     progress_done, progress_total, verdict, started_by, started_at, finished_at, reports";

struct Base {
    id: Uuid,
    source: String,
    root_id: Option<Uuid>,
    modes: Vec<String>,
    beam: i32,
    min_score: f32,
    status: String,
    error: Option<String>,
    progress: Progress,
    verdict: Option<String>,
    started_by: Option<Uuid>,
    started: DateTime<Utc>,
    finished: Option<DateTime<Utc>>,
    reports: Value,
}

fn base(r: &PgRow) -> Result<Base> {
    let reports: Option<Json<Value>> = r.try_get("reports")?;
    Ok(Base {
        id: r.try_get("id")?,
        source: r.try_get("source")?,
        root_id: r.try_get("root_id")?,
        modes: r.try_get("modes")?,
        beam: r.try_get("beam")?,
        min_score: r.try_get("min_score")?,
        status: r.try_get("status")?,
        error: r.try_get("error")?,
        progress: Progress {
            done: r.try_get("progress_done")?,
            total: r.try_get("progress_total")?,
        },
        verdict: r.try_get("verdict")?,
        started_by: r.try_get("started_by")?,
        started: r.try_get("started_at")?,
        finished: r.try_get("finished_at")?,
        reports: reports.map_or(Value::Null, |Json(v)| v),
    })
}

fn took_ms(started: DateTime<Utc>, finished: Option<DateTime<Utc>>) -> Option<i64> {
    finished.map(|f| (f - started).num_milliseconds().max(0))
}

fn run(r: &PgRow) -> Result<EvalRun> {
    let b = base(r)?;
    let Json(questions): Json<Vec<RunQuestion>> = r.try_get("questions")?;
    Ok(EvalRun {
        id: b.id,
        source: b.source,
        root_id: b.root_id,
        modes: b.modes,
        beam: b.beam,
        min_score: b.min_score,
        status: b.status,
        error: b.error,
        progress: b.progress,
        verdict: b.verdict,
        started_by: b.started_by,
        started_at: ts(b.started),
        finished_at: b.finished.map(ts),
        took_ms: took_ms(b.started, b.finished),
        questions,
        reports: b.reports,
        results: None,
        started_at_ts: b.started,
    })
}

fn summary(r: &PgRow) -> Result<EvalRunSummary> {
    let b = base(r)?;
    Ok(EvalRunSummary {
        id: b.id,
        source: b.source,
        root_id: b.root_id,
        modes: b.modes,
        beam: b.beam,
        min_score: b.min_score,
        status: b.status,
        error: b.error,
        progress: b.progress,
        verdict: b.verdict,
        started_by: b.started_by,
        started_at: ts(b.started),
        finished_at: b.finished.map(ts),
        took_ms: took_ms(b.started, b.finished),
        question_count: r.try_get("question_count")?,
        reports: b.reports,
        started_at_ts: b.started,
    })
}

async fn results_of(pool: &PgPool, tenant: Tenant, run_id: Uuid) -> Result<Vec<RunResult>> {
    let rows = sqlx::query(
        "SELECT idx, question_id, mode, rank, leaf_hit FROM eval_run_results \
         WHERE tenant = $1 AND run_id = $2 ORDER BY idx, mode",
    )
    .bind(tenant.id())
    .bind(run_id)
    .fetch_all(pool)
    .await?;
    rows.iter()
        .map(|r| {
            Ok(RunResult {
                index: r.try_get("idx")?,
                question_id: r.try_get("question_id")?,
                mode: r.try_get("mode")?,
                rank: r.try_get("rank")?,
                leaf_hit: r.try_get("leaf_hit")?,
            })
        })
        .collect()
}

/// Stores a new `running` run (progress 0 of `total`). `EvalRunning` when the tenant already has
/// a running run (the `eval_runs_one_running` index, which holds across API replicas).
pub async fn insert_running(
    pool: &PgPool,
    tenant: Tenant,
    n: NewRun<'_>,
) -> Result<EvalRunSummary> {
    let row = sqlx::query(sqlx::AssertSqlSafe(format!(
        "INSERT INTO eval_runs (tenant, source, root_id, modes, beam, min_score, status, \
           started_by, started_at, questions, progress_done, progress_total, heartbeat_at) \
         VALUES ($1, $2, $3, $4, $5, $6, '{RUNNING}', $7, now(), $8, 0, $9, now()) \
         RETURNING {RUN_BASE_COLS}, jsonb_array_length(questions) AS question_count"
    )))
    .bind(tenant.id())
    .bind(n.source)
    .bind(n.root_id)
    .bind(&n.modes)
    .bind(n.beam)
    .bind(n.min_score)
    .bind(n.started_by)
    .bind(Json(n.questions))
    .bind(n.total)
    .fetch_one(pool)
    .await
    .map_err(|e| {
        if let sqlx::Error::Database(d) = &e
            && d.constraint() == Some(ONE_RUNNING)
        {
            return already_running();
        }
        StrataError::from(e)
    })?;
    summary(&row)
}

/// The tenant has a live `running` run. A stale one (worker gone, see [`fail_stale_runs`]) is
/// failed first, so it cannot keep the tenant's next run out forever.
pub async fn has_running(pool: &PgPool, tenant: Tenant) -> Result<bool> {
    fail_stale_runs(pool, Some(tenant), EVAL_STALE_AFTER).await?;
    Ok(sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
        "SELECT EXISTS (SELECT 1 FROM eval_runs WHERE tenant = $1 AND status = '{RUNNING}')"
    )))
    .bind(tenant.id())
    .fetch_one(pool)
    .await?)
}

/// The refusal of a second run while one of the tenant's runs is in progress.
pub fn already_running() -> StrataError {
    StrataError::EvalRunning(
        "an eval of this tenant is already running; poll GET /api/eval/runs/{id} until it \
         finishes, then retry"
            .into(),
    )
}

/// The worker's heartbeat: stamps `heartbeat_at = now()` (the database clock, the one the sweep
/// compares against) and records progress (`done` outcomes scored). `false` when the run is no
/// longer `running` (swept as interrupted): nothing is written then, and the worker must stop.
pub async fn heartbeat(pool: &PgPool, tenant: Tenant, id: Uuid, done: i32) -> Result<bool> {
    let updated = sqlx::query(sqlx::AssertSqlSafe(format!(
        "UPDATE eval_runs SET progress_done = LEAST($3, progress_total), heartbeat_at = now() \
         WHERE tenant = $1 AND id = $2 AND status = '{RUNNING}'"
    )))
    .bind(tenant.id())
    .bind(id)
    .bind(done)
    .execute(pool)
    .await?
    .rows_affected();
    Ok(updated > 0)
}

/// Finishes a running run `completed` (at the database's `now()`): its verdict, reports and
/// per-question results, in one transaction. `false` when the run was no longer `running` (swept as interrupted meanwhile):
/// nothing is written then.
pub async fn complete_run(
    pool: &PgPool,
    tenant: Tenant,
    id: Uuid,
    verdict: &'static str,
    reports: &Value,
    results: &[ResultRow],
) -> Result<bool> {
    let mut tx = pool.begin().await?;
    let updated = sqlx::query(sqlx::AssertSqlSafe(format!(
        "UPDATE eval_runs SET status = '{COMPLETED}', verdict = $3, reports = $4, \
           finished_at = now(), progress_done = progress_total \
         WHERE tenant = $1 AND id = $2 AND status = '{RUNNING}'"
    )))
    .bind(tenant.id())
    .bind(id)
    .bind(verdict)
    .bind(Json(reports))
    .execute(&mut *tx)
    .await?
    .rows_affected();
    if updated == 0 {
        return Ok(false);
    }
    let idx: Vec<i32> = results.iter().map(|r| r.index as i32).collect();
    let modes: Vec<String> = results.iter().map(|r| r.mode.to_string()).collect();
    let qids: Vec<Option<Uuid>> = results.iter().map(|r| r.question_id).collect();
    let ranks: Vec<Option<i32>> = results.iter().map(|r| r.rank.map(|r| r as i32)).collect();
    let leaf: Vec<Option<bool>> = results.iter().map(|r| r.leaf_hit).collect();
    sqlx::query(
        "INSERT INTO eval_run_results (run_id, tenant, idx, mode, question_id, rank, leaf_hit) \
         SELECT $1, $2, * FROM unnest($3::int[], $4::text[], $5::uuid[], $6::int[], $7::bool[])",
    )
    .bind(id)
    .bind(tenant.id())
    .bind(&idx)
    .bind(&modes)
    .bind(&qids)
    .bind(&ranks)
    .bind(&leaf)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(true)
}

/// Finishes a running run `failed` with `error`. `false` when it was no longer running.
pub async fn fail_run(pool: &PgPool, tenant: Tenant, id: Uuid, error: &str) -> Result<bool> {
    let updated = sqlx::query(sqlx::AssertSqlSafe(format!(
        "UPDATE eval_runs SET status = '{FAILED}', error = $3, finished_at = now() \
         WHERE tenant = $1 AND id = $2 AND status = '{RUNNING}'"
    )))
    .bind(tenant.id())
    .bind(id)
    .bind(error)
    .execute(pool)
    .await?
    .rows_affected();
    Ok(updated > 0)
}

/// Fails each of `runs` (`(tenant, run id)`) still `running` with `error`; a run that finished
/// meanwhile, or an id that is not that tenant's, is left alone. Answers how many it failed.
pub async fn fail_runs(pool: &PgPool, runs: &[(Tenant, Uuid)], error: &str) -> Result<u64> {
    let tenants: Vec<Uuid> = runs.iter().map(|(t, _)| t.id()).collect();
    let ids: Vec<Uuid> = runs.iter().map(|(_, id)| *id).collect();
    Ok(sqlx::query(sqlx::AssertSqlSafe(format!(
        "UPDATE eval_runs e SET status = '{FAILED}', error = $3, finished_at = now() \
         FROM unnest($1::uuid[], $2::uuid[]) AS r(tenant, id) \
         WHERE e.tenant = r.tenant AND e.id = r.id AND e.status = '{RUNNING}'"
    )))
    .bind(&tenants)
    .bind(&ids)
    .bind(error)
    .execute(pool)
    .await?
    .rows_affected())
}

/// Fails with error `interrupted` every `running` run whose heartbeat is older than `stale_after`
/// (its worker died with its process): of `tenant`, or of every tenant with `None`. A run whose
/// worker still beats (on this or another replica) is left alone. Ages
/// are measured on the database clock only, so replicas' clock skew does not matter. Answers how
/// many runs it failed. Callers pass [`EVAL_STALE_AFTER`]; tests pass their own.
pub async fn fail_stale_runs(
    pool: &PgPool,
    tenant: Option<Tenant>,
    stale_after: std::time::Duration,
) -> Result<u64> {
    Ok(sqlx::query(sqlx::AssertSqlSafe(format!(
        "UPDATE eval_runs SET status = '{FAILED}', error = '{INTERRUPTED}', finished_at = now() \
         WHERE status = '{RUNNING}' AND ($1::uuid IS NULL OR tenant = $1) \
           AND COALESCE(heartbeat_at, started_at) < now() - make_interval(secs => $2)"
    )))
    .bind(tenant.map(|t| t.id()))
    .bind(stale_after.as_secs_f64())
    .execute(pool)
    .await?
    .rows_affected())
}

/// The tenant's run summaries, newest first (`started_at DESC, id DESC`), strictly after `after`.
/// Never reads the question snapshot itself, only its length.
pub async fn list_runs(
    pool: &PgPool,
    tenant: Tenant,
    limit: i64,
    after: Option<DocumentCursor>,
) -> Result<Vec<EvalRunSummary>> {
    let (after_ts, after_id) = after.map_or((None, None), |(t, id)| (Some(t), Some(id)));
    let rows = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {RUN_BASE_COLS}, jsonb_array_length(questions) AS question_count \
         FROM eval_runs WHERE tenant = $1 \
           AND ($2::timestamptz IS NULL OR (started_at, id) < ($2::timestamptz, $3::uuid)) \
         ORDER BY started_at DESC, id DESC LIMIT $4"
    )))
    .bind(tenant.id())
    .bind(after_ts)
    .bind(after_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    rows.iter().map(summary).collect()
}

/// One run with its per-question `results` (`[]` until it completed).
pub async fn get_run(pool: &PgPool, tenant: Tenant, id: Uuid) -> Result<EvalRun> {
    let row = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {RUN_BASE_COLS}, questions FROM eval_runs WHERE tenant = $1 AND id = $2"
    )))
    .bind(tenant.id())
    .bind(id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| StrataError::NotFound(format!("eval run {id}")))?;
    let mut out = run(&row)?;
    out.results = Some(results_of(pool, tenant, id).await?);
    Ok(out)
}
