//! Tenant eval questions: a tenant's own golden questions (stored in `eval_questions`, managed by
//! `/api/eval/questions`), scored by `POST /api/eval` with `source: "tenant"` (or inline
//! `questions`) as recall@1/5/10 per search mode, each run stored in `eval_runs` so runs can be
//! compared. The built-in file eval (`crate::eval`) is unchanged; this reuses its metrics
//! (`ModeReport`, `recall_at_k`) and its flat search.
//!
//! A question is a hit at rank r when the r-th result is the first that matches ANY of its
//! expectations: `expected_document` (title, `source`, or file name of the source),
//! `expected_document_ids`, `expected_chunk_ids`, or `expected_text` (contained in the chunk text,
//! case- and whitespace-insensitive). `expected_leaf` only feeds leaf accuracy.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

use crate::db::search::ChunkFilter;
use crate::db::stores::StoreKind;
use crate::engine::Engine;
use crate::error::{Result, StrataError};
use crate::eval::{
    EVAL_TOP_K, EvalMode, EvalOptions, ModeReport, QuestionOutcome, source_matches, store_paths,
};
use crate::query::QueryRequest;
use crate::search::search_leaves_ef;
use crate::tenant::Tenant;

/// Most questions a tenant may store, and most one eval run scores.
pub const EVAL_MAX_QUESTIONS: usize = 200;
/// Longest `query`, `expected_document`, `expected_leaf` and `expected_text`, in characters.
pub const FIELD_MAX_CHARS: usize = 2000;
/// Most ids in `expected_document_ids` / `expected_chunk_ids`.
pub const MAX_EXPECTED_IDS: usize = 100;

/// A question as a client writes it (`POST /api/eval/questions`, a CSV row, an inline
/// `POST /api/eval` question). Field names match the agent web app's `GoldenQuestionInput`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QuestionInput {
    pub query: String,
    #[serde(default)]
    pub expected_document: Option<String>,
    #[serde(default)]
    pub expected_leaf: Option<String>,
    #[serde(default)]
    pub expected_document_ids: Vec<Uuid>,
    #[serde(default)]
    pub expected_chunk_ids: Vec<Uuid>,
    #[serde(default)]
    pub expected_text: Option<String>,
}

fn clean_opt(s: Option<String>) -> Option<String> {
    s.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn dedup(ids: Vec<Uuid>) -> Vec<Uuid> {
    let mut seen = HashSet::new();
    ids.into_iter().filter(|id| seen.insert(*id)).collect()
}

impl QuestionInput {
    /// Trimmed; blank optional fields become `None`; repeated ids dropped (first kept).
    pub fn normalized(self) -> Self {
        QuestionInput {
            query: self.query.trim().to_string(),
            expected_document: clean_opt(self.expected_document),
            expected_leaf: clean_opt(self.expected_leaf),
            expected_document_ids: dedup(self.expected_document_ids),
            expected_chunk_ids: dedup(self.expected_chunk_ids),
            expected_text: clean_opt(self.expected_text),
        }
    }

    /// What is wrong with this (normalized) question; empty when it is valid.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.query.is_empty() {
            out.push("query is empty".to_string());
        }
        for (name, v) in [
            ("query", Some(&self.query)),
            ("expected_document", self.expected_document.as_ref()),
            ("expected_leaf", self.expected_leaf.as_ref()),
            ("expected_text", self.expected_text.as_ref()),
        ] {
            if v.is_some_and(|v| v.chars().count() > FIELD_MAX_CHARS) {
                out.push(format!("{name} is over {FIELD_MAX_CHARS} characters"));
            }
        }
        for (name, ids) in [
            ("expected_document_ids", &self.expected_document_ids),
            ("expected_chunk_ids", &self.expected_chunk_ids),
        ] {
            if ids.len() > MAX_EXPECTED_IDS {
                out.push(format!("{name} has more than {MAX_EXPECTED_IDS} ids"));
            }
            if ids.iter().any(Uuid::is_nil) {
                out.push(format!("{name} holds the nil uuid"));
            }
        }
        if self.expected_document.is_none()
            && self.expected_text.is_none()
            && self.expected_document_ids.is_empty()
            && self.expected_chunk_ids.is_empty()
        {
            out.push(
                "no expectation: give expected_document, expected_document_ids, \
                 expected_chunk_ids or expected_text"
                    .to_string(),
            );
        }
        out
    }

    /// Normalizes, then `InvalidInput` listing every problem.
    pub fn validated(self) -> Result<Self> {
        let q = self.normalized();
        let problems = q.problems();
        if problems.is_empty() {
            Ok(q)
        } else {
            Err(StrataError::InvalidInput(problems.join("; ")))
        }
    }
}

/// `Some(v)` for a present field (also `null` → `Some(None)`), so a PATCH can tell "absent"
/// (keep) from `null` (clear).
fn present<'de, D, T>(d: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(d).map(Some)
}

/// `PATCH /api/eval/questions/{id}`: a field left out keeps its value; `null` clears an optional
/// field (an id list is cleared with `[]`). The result must still be a valid question.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuestionPatch {
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default, deserialize_with = "present")]
    pub expected_document: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    pub expected_leaf: Option<Option<String>>,
    #[serde(default)]
    pub expected_document_ids: Option<Vec<Uuid>>,
    #[serde(default)]
    pub expected_chunk_ids: Option<Vec<Uuid>>,
    #[serde(default, deserialize_with = "present")]
    pub expected_text: Option<Option<String>>,
}

impl QuestionPatch {
    pub fn is_empty(&self) -> bool {
        self.query.is_none()
            && self.expected_document.is_none()
            && self.expected_leaf.is_none()
            && self.expected_document_ids.is_none()
            && self.expected_chunk_ids.is_none()
            && self.expected_text.is_none()
    }

    pub fn apply(self, cur: QuestionInput) -> QuestionInput {
        QuestionInput {
            query: self.query.unwrap_or(cur.query),
            expected_document: self.expected_document.unwrap_or(cur.expected_document),
            expected_leaf: self.expected_leaf.unwrap_or(cur.expected_leaf),
            expected_document_ids: self
                .expected_document_ids
                .unwrap_or(cur.expected_document_ids),
            expected_chunk_ids: self.expected_chunk_ids.unwrap_or(cur.expected_chunk_ids),
            expected_text: self.expected_text.unwrap_or(cur.expected_text),
        }
    }
}

/// A stored question (the agent web app's `GoldenQuestion`, plus the id/text expectations).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Question {
    pub id: Uuid,
    pub query: String,
    pub expected_document: Option<String>,
    pub expected_leaf: Option<String>,
    pub expected_document_ids: Vec<Uuid>,
    pub expected_chunk_ids: Vec<Uuid>,
    pub expected_text: Option<String>,
    /// The `X-Actor-Id` of the request that created it, if any.
    pub created_by: Option<Uuid>,
    /// RFC 3339, UTC, microsecond precision.
    pub created_at: String,
    pub updated_at: String,
}

impl Question {
    pub fn input(&self) -> QuestionInput {
        QuestionInput {
            query: self.query.clone(),
            expected_document: self.expected_document.clone(),
            expected_leaf: self.expected_leaf.clone(),
            expected_document_ids: self.expected_document_ids.clone(),
            expected_chunk_ids: self.expected_chunk_ids.clone(),
            expected_text: self.expected_text.clone(),
        }
    }
}

/// One evaluated question as a run keeps it: `id` is the stored question's (`None` inline).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct RunQuestion {
    pub id: Option<Uuid>,
    pub query: String,
    #[serde(default)]
    pub expected_document: Option<String>,
    #[serde(default)]
    pub expected_leaf: Option<String>,
    #[serde(default)]
    pub expected_document_ids: Vec<Uuid>,
    #[serde(default)]
    pub expected_chunk_ids: Vec<Uuid>,
    #[serde(default)]
    pub expected_text: Option<String>,
}

impl RunQuestion {
    pub fn new(id: Option<Uuid>, q: QuestionInput) -> Self {
        RunQuestion {
            id,
            query: q.query,
            expected_document: q.expected_document,
            expected_leaf: q.expected_leaf,
            expected_document_ids: q.expected_document_ids,
            expected_chunk_ids: q.expected_chunk_ids,
            expected_text: q.expected_text,
        }
    }
}

// ---------- CSV import ----------

/// RFC 4180: comma separated, `"`-quoted fields with `""` escapes and line breaks inside quotes,
/// `\n` or `\r\n` rows. Each row comes with the 1-based line it starts on.
pub fn parse_csv(text: &str) -> Vec<(usize, Vec<String>)> {
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut line = 1;
    let mut row_line = 1;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if quoted {
            match ch {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    cell.push('"');
                }
                '"' => quoted = false,
                '\n' => {
                    line += 1;
                    cell.push('\n');
                }
                _ => cell.push(ch),
            }
            continue;
        }
        match ch {
            '"' if cell.is_empty() => quoted = true,
            ',' => row.push(std::mem::take(&mut cell)),
            '\r' | '\n' => {
                if ch == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                row.push(std::mem::take(&mut cell));
                rows.push((row_line, std::mem::take(&mut row)));
                line += 1;
                row_line = line;
            }
            _ => cell.push(ch),
        }
    }
    if !cell.is_empty() || !row.is_empty() {
        row.push(cell);
        rows.push((row_line, row));
    }
    rows
}

/// The columns an import reads; any other column is ignored.
pub const CSV_COLUMNS: [&str; 6] = [
    "query",
    "expected_document",
    "expected_leaf",
    "expected_document_ids",
    "expected_chunk_ids",
    "expected_text",
];

/// Ids in one CSV cell, separated by `;`, `|` or whitespace.
fn parse_ids(cell: &str, column: &str) -> std::result::Result<Vec<Uuid>, String> {
    cell.split(|c: char| c == ';' || c == '|' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| Uuid::parse_str(s).map_err(|_| format!("{column}: `{s}` is not a uuid")))
        .collect()
}

/// Questions from CSV with a header row naming `query` and at least one expectation column (any
/// order, case-insensitive, extra columns ignored). Blank rows are skipped. Every bad row is
/// reported as `line N: ...` and then nothing is returned: an import is all or nothing.
pub fn parse_questions_csv(text: &str) -> std::result::Result<Vec<QuestionInput>, Vec<String>> {
    let rows = parse_csv(text.strip_prefix('\u{feff}').unwrap_or(text));
    let Some((_, header)) = rows.first() else {
        return Err(vec!["the file is empty".to_string()]);
    };
    let header: Vec<String> = header.iter().map(|h| h.trim().to_lowercase()).collect();
    let col: HashMap<&str, usize> = CSV_COLUMNS
        .iter()
        .filter_map(|c| header.iter().position(|h| h == c).map(|i| (*c, i)))
        .collect();
    if !col.contains_key("query") {
        return Err(vec![
            "line 1: the header needs a `query` column".to_string(),
        ]);
    }
    if !CSV_COLUMNS[1..]
        .iter()
        .any(|c| *c != "expected_leaf" && col.contains_key(c))
    {
        return Err(vec![
            "line 1: the header needs an expectation column (expected_document, \
             expected_document_ids, expected_chunk_ids or expected_text)"
                .to_string(),
        ]);
    }
    let mut out = Vec::new();
    let mut errors = Vec::new();
    let mut seen = HashSet::new();
    for (line, cells) in &rows[1..] {
        if cells.iter().all(|c| c.trim().is_empty()) {
            continue;
        }
        let get = |c: &str| col.get(c).and_then(|&i| cells.get(i)).cloned();
        let mut problems = Vec::new();
        let mut ids = |c: &str| match get(c) {
            Some(cell) => parse_ids(&cell, c).unwrap_or_else(|e| {
                problems.push(e);
                Vec::new()
            }),
            None => Vec::new(),
        };
        let expected_document_ids = ids("expected_document_ids");
        let expected_chunk_ids = ids("expected_chunk_ids");
        let q = QuestionInput {
            query: get("query").unwrap_or_default(),
            expected_document: get("expected_document"),
            expected_leaf: get("expected_leaf"),
            expected_document_ids,
            expected_chunk_ids,
            expected_text: get("expected_text"),
        }
        .normalized();
        problems.extend(q.problems());
        if problems.is_empty() && !seen.insert(q.query.to_lowercase()) {
            problems.push("the same query is on an earlier line".to_string());
        }
        if problems.is_empty() {
            out.push(q);
        } else {
            errors.push(format!("line {line}: {}", problems.join("; ")));
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    if out.is_empty() {
        return Err(vec!["the file has no question rows".to_string()]);
    }
    Ok(out)
}

// ---------- scoring ----------

/// Lowercased, whitespace runs collapsed to one space, trimmed.
pub fn normalize_text(s: &str) -> String {
    s.split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The fields of one result a question's expectations are checked against.
#[derive(Debug, Clone, Copy)]
pub struct Hit<'a> {
    pub chunk_id: Uuid,
    pub document_id: Uuid,
    pub text: &'a str,
    pub source: &'a str,
    pub title: Option<&'a str>,
}

/// `expected` names this document: its title, its `source`, or the file name of the source.
pub fn document_matches(expected: &str, source: &str, title: Option<&str>) -> bool {
    title == Some(expected) || source_matches(source, expected)
}

/// One question's expectations, prepared for matching.
#[derive(Debug, Clone)]
pub struct Expectation {
    document: Option<String>,
    document_ids: HashSet<Uuid>,
    chunk_ids: HashSet<Uuid>,
    /// normalized
    text: Option<String>,
}

impl Expectation {
    pub fn of(q: &RunQuestion) -> Self {
        Expectation {
            document: q.expected_document.clone(),
            document_ids: q.expected_document_ids.iter().copied().collect(),
            chunk_ids: q.expected_chunk_ids.iter().copied().collect(),
            text: q
                .expected_text
                .as_deref()
                .map(normalize_text)
                .filter(|t| !t.is_empty()),
        }
    }

    pub fn matches(&self, h: &Hit<'_>) -> bool {
        self.chunk_ids.contains(&h.chunk_id)
            || self.document_ids.contains(&h.document_id)
            || self
                .document
                .as_deref()
                .is_some_and(|d| document_matches(d, h.source, h.title))
            || self
                .text
                .as_deref()
                .is_some_and(|t| normalize_text(h.text).contains(t))
    }

    /// 1-based rank of the first matching hit.
    pub fn first_rank(&self, hits: &[Hit<'_>]) -> Option<usize> {
        hits.iter().position(|h| self.matches(h)).map(|i| i + 1)
    }
}

/// One question's outcome in one mode, as stored in `eval_run_results`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResultRow {
    /// Position in the run's `questions`.
    pub index: usize,
    pub mode: &'static str,
    pub question_id: Option<Uuid>,
    pub rank: Option<usize>,
    pub leaf_hit: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct Scored {
    pub reports: Vec<ModeReport>,
    pub results: Vec<ResultRow>,
}

/// Resolves `expected_leaf` among `leaves` (id, name path from the top, kind): a leaf id, a name
/// path `a/b/c`, or a bare name that is unique among them.
fn resolve_leaf(
    spec: &str,
    leaves: &[(Uuid, Vec<String>)],
) -> std::result::Result<Uuid, &'static str> {
    if let Ok(id) = Uuid::parse_str(spec) {
        return leaves
            .iter()
            .find(|(l, _)| *l == id)
            .map(|(l, _)| *l)
            .ok_or("is not a leaf store in the evaluated tree");
    }
    let matches: Vec<Uuid> = if spec.contains('/') {
        leaves
            .iter()
            .filter(|(_, p)| p.join("/") == spec)
            .map(|(l, _)| *l)
            .collect()
    } else {
        leaves
            .iter()
            .filter(|(_, p)| p.last().is_some_and(|n| n == spec))
            .map(|(l, _)| *l)
            .collect()
    };
    match matches.as_slice() {
        [one] => Ok(*one),
        [] => Err("is not a leaf store in the evaluated tree"),
        _ => Err("names several leaf stores; use its path (a/b/c) or id"),
    }
}

/// A tenant eval ready to score: its questions checked against the tree, its expectations and
/// leaves resolved. Owns everything it needs, so a background task can score it after the request
/// that prepared it has been answered.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub questions: Vec<RunQuestion>,
    expectations: Vec<Expectation>,
    expected_leaves: Vec<Option<Uuid>>,
    all_leaves: Vec<Uuid>,
    /// Deduplicated, routed modes in the given order, `flat` last: the order of `reports`.
    modes: Vec<EvalMode>,
    root_id: Option<Uuid>,
    opts: EvalOptions,
}

impl Prepared {
    /// The mode labels in report order.
    pub fn mode_labels(&self) -> Vec<String> {
        self.modes.iter().map(|m| m.label().to_string()).collect()
    }

    /// How many outcomes (question x mode) scoring produces.
    pub fn total(&self) -> usize {
        self.questions.len() * self.modes.len()
    }
}

/// Checks `questions` against `tenant`'s tree (or the subtree under `root_id`, which must be one
/// of the tenant's stores, else 404). Every expectation must name something that is there (a leaf,
/// a document, a chunk): otherwise `EvalNotReady` listing each problem, since a question that
/// cannot hit would only lower recall silently. Fast (a few reads, no query per question), so a
/// request answers these errors itself before handing the slow part, [`Prepared::score`], off.
pub async fn prepare(
    engine: &Engine,
    tenant: Tenant,
    questions: Vec<RunQuestion>,
    root_id: Option<Uuid>,
    opts: EvalOptions,
) -> Result<Prepared> {
    let views = engine.store_tree(tenant).await?;
    let paths = store_paths(&views);
    // No root_id: the tenant's only root. Several roots (D83: each a tree in its own embedding
    // model) and no root_id is 409 ambiguous_root (D90), as for a query.
    let root_id = match root_id {
        Some(id) => Some(id),
        None => {
            let roots: Vec<Uuid> = views
                .iter()
                .filter(|v| v.parent_id.is_none())
                .map(|v| v.id)
                .collect();
            if roots.len() > 1 {
                return Err(crate::query::ambiguous_root(roots.len()));
            }
            roots.first().copied()
        }
    };
    let scope: HashSet<Uuid> = match root_id {
        None => HashSet::new(),
        Some(root) => {
            if !views.iter().any(|v| v.id == root) {
                return Err(StrataError::NotFound(format!("store {root}")));
            }
            let mut children: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
            for v in &views {
                if let Some(p) = v.parent_id {
                    children.entry(p).or_default().push(v.id);
                }
            }
            let mut scope = HashSet::new();
            let mut stack = vec![root];
            while let Some(id) = stack.pop() {
                if scope.insert(id) {
                    stack.extend(children.get(&id).into_iter().flatten().copied());
                }
            }
            scope
        }
    };
    let leaves: Vec<(Uuid, Vec<String>)> = views
        .iter()
        .filter(|v| v.kind == StoreKind::Leaf && scope.contains(&v.id))
        .map(|v| (v.id, paths.get(&v.id).cloned().unwrap_or_default()))
        .collect();
    let all_leaves: Vec<Uuid> = leaves.iter().map(|(id, _)| *id).collect();

    // (document id) -> (store, source, title), the tenant's documents in scope
    let docs: Vec<(Uuid, Uuid, String, Option<String>)> = sqlx::query_as(
        "SELECT d.id, d.store_id, d.source, d.title FROM documents d \
         JOIN stores s ON s.id = d.store_id WHERE s.tenant = $1",
    )
    .bind(tenant.id())
    .fetch_all(&engine.pool)
    .await?;
    let docs: HashMap<Uuid, (String, Option<String>)> = docs
        .into_iter()
        .filter(|(_, store, _, _)| scope.contains(store))
        .map(|(id, _, source, title)| (id, (source, title)))
        .collect();
    let wanted_chunks: Vec<Uuid> = questions
        .iter()
        .flat_map(|q| q.expected_chunk_ids.iter().copied())
        .collect();
    let chunks_in_scope: HashSet<Uuid> = if wanted_chunks.is_empty() {
        HashSet::new()
    } else {
        let rows: Vec<(Uuid, Uuid)> = sqlx::query_as(
            "SELECT c.id, c.store_id FROM chunks c JOIN stores s ON s.id = c.store_id \
             WHERE s.tenant = $1 AND c.id = ANY($2)",
        )
        .bind(tenant.id())
        .bind(&wanted_chunks)
        .fetch_all(&engine.pool)
        .await?;
        rows.into_iter()
            .filter(|(_, store)| scope.contains(store))
            .map(|(id, _)| id)
            .collect()
    };

    let mut problems = Vec::new();
    if all_leaves.is_empty() {
        problems.push("the evaluated tree has no leaf stores".to_string());
    }
    let mut expected_leaves: Vec<Option<Uuid>> = Vec::with_capacity(questions.len());
    for (i, q) in questions.iter().enumerate() {
        let n = i + 1;
        expected_leaves.push(match q.expected_leaf.as_deref() {
            None => None,
            Some(spec) => match resolve_leaf(spec, &leaves) {
                Ok(id) => Some(id),
                Err(why) => {
                    problems.push(format!("question {n}: expected_leaf `{spec}` {why}"));
                    None
                }
            },
        });
        if let Some(d) = q.expected_document.as_deref()
            && !docs
                .values()
                .any(|(source, title)| document_matches(d, source, title.as_deref()))
        {
            problems.push(format!(
                "question {n}: no document `{d}` in the evaluated tree"
            ));
        }
        for id in &q.expected_document_ids {
            if !docs.contains_key(id) {
                problems.push(format!(
                    "question {n}: document {id} is not in the evaluated tree"
                ));
            }
        }
        for id in &q.expected_chunk_ids {
            if !chunks_in_scope.contains(id) {
                problems.push(format!(
                    "question {n}: chunk {id} is not in the evaluated tree"
                ));
            }
        }
    }
    if !problems.is_empty() {
        return Err(StrataError::EvalNotReady(problems.join("\n")));
    }
    let expectations: Vec<Expectation> = questions.iter().map(Expectation::of).collect();
    let mut seen = HashSet::new();
    let unique: Vec<EvalMode> = opts
        .modes
        .iter()
        .copied()
        .filter(|m| seen.insert(*m))
        .collect();
    let mut modes: Vec<EvalMode> = unique
        .iter()
        .copied()
        .filter(|m| m.route_mode().is_some())
        .collect();
    if unique.contains(&EvalMode::Flat) {
        modes.push(EvalMode::Flat);
    }
    Ok(Prepared {
        questions,
        expectations,
        expected_leaves,
        all_leaves,
        modes,
        root_id,
        opts,
    })
}

impl Prepared {
    /// Scores the questions in every mode: one query per question per mode (slow: a 200-question
    /// hybrid run takes minutes). `done` counts the outcomes scored so far, for progress. Read-only,
    /// like `evaluate_with`.
    pub async fn score(
        &self,
        engine: &Engine,
        tenant: Tenant,
        done: &AtomicUsize,
    ) -> Result<Scored> {
        let Prepared {
            questions,
            expectations,
            expected_leaves,
            all_leaves,
            modes,
            root_id,
            opts,
        } = self;
        let root_id = *root_id;
        let mut reports = Vec::new();
        let mut results = Vec::new();
        for mode in modes.iter().copied().filter(|m| *m != EvalMode::Flat) {
            let Some(route_mode) = mode.route_mode() else {
                continue;
            };
            let mut outcomes = Vec::with_capacity(questions.len());
            for (i, q) in questions.iter().enumerate() {
                let req = QueryRequest {
                    top_k: EVAL_TOP_K,
                    route_mode,
                    beam: opts.beam,
                    min_score: opts.min_score,
                    root_id,
                    ..QueryRequest::new(q.query.clone())
                };
                let resp = engine.query(tenant, &req).await?;
                let hits: Vec<Hit<'_>> = resp
                    .results
                    .iter()
                    .map(|r| Hit {
                        chunk_id: r.chunk_id,
                        document_id: r.document_id,
                        text: &r.text,
                        source: &r.source,
                        title: r.title.as_deref(),
                    })
                    .collect();
                let outcome = QuestionOutcome {
                    rank: expectations[i].first_rank(&hits),
                    leaf_hit: expected_leaves[i].map(|l| resp.routing.leaves.contains(&l)),
                    laya_consulted: resp.routing.laya_consulted,
                    laya_fallback: resp.routing.laya_fallback.is_some(),
                };
                results.push(ResultRow {
                    index: i,
                    mode: mode.label(),
                    question_id: q.id,
                    rank: outcome.rank,
                    leaf_hit: outcome.leaf_hit,
                });
                outcomes.push(outcome);
                done.fetch_add(1, Ordering::Relaxed);
            }
            reports.push(ModeReport::from_outcomes(mode.label(), &outcomes));
        }
        if modes.contains(&EvalMode::Flat) {
            // the evaluated tree's providers (a store binding wins over the tenant's)
            let inf = engine.inference_for(tenant, root_id).await?;
            let mut outcomes = Vec::with_capacity(questions.len());
            for (i, q) in questions.iter().enumerate() {
                let q_vec = inf.embed_one(q.query.clone()).await?;
                let (found, _) = search_leaves_ef(
                    &engine.pool,
                    tenant,
                    &inf,
                    all_leaves,
                    &[q_vec],
                    &q.query,
                    EVAL_TOP_K,
                    engine.cfg.rerank_candidates,
                    &ChunkFilter::default(),
                    engine.cfg.hnsw_ef_search,
                    engine.cfg.exact_scan_max_chunks,
                )
                .await?;
                let hits: Vec<Hit<'_>> = found
                    .iter()
                    .map(|h| Hit {
                        chunk_id: h.chunk.chunk_id,
                        document_id: h.chunk.document_id,
                        text: &h.chunk.text,
                        source: &h.chunk.source,
                        title: h.chunk.title.as_deref(),
                    })
                    .collect();
                let outcome = QuestionOutcome {
                    rank: expectations[i].first_rank(&hits),
                    leaf_hit: None,
                    laya_consulted: false,
                    laya_fallback: false,
                };
                results.push(ResultRow {
                    index: i,
                    mode: EvalMode::Flat.label(),
                    question_id: q.id,
                    rank: outcome.rank,
                    leaf_hit: None,
                });
                outcomes.push(outcome);
                done.fetch_add(1, Ordering::Relaxed);
            }
            reports.push(ModeReport::from_outcomes(EvalMode::Flat.label(), &outcomes));
        }
        // the order `GET /api/eval/runs/{id}` reads them back in
        results.sort_by_key(|r| (r.index, r.mode));
        Ok(Scored { reports, results })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(query: &str) -> QuestionInput {
        QuestionInput {
            query: query.into(),
            expected_document: Some("doc".into()),
            ..QuestionInput::default()
        }
    }

    #[test]
    fn validation_requires_a_query_and_an_expectation() {
        assert!(q(" what? ").validated().is_ok());
        let e = QuestionInput {
            query: "  ".into(),
            expected_leaf: Some("leaf".into()),
            expected_document: Some("   ".into()),
            ..QuestionInput::default()
        }
        .validated()
        .unwrap_err()
        .to_string();
        assert!(e.contains("query is empty"), "{e}");
        assert!(e.contains("no expectation"), "{e}");
        let ok = QuestionInput {
            query: "q".into(),
            expected_chunk_ids: vec![Uuid::from_u128(5), Uuid::from_u128(5)],
            ..QuestionInput::default()
        }
        .validated()
        .unwrap();
        assert_eq!(ok.expected_chunk_ids, [Uuid::from_u128(5)], "deduplicated");
        let nil = QuestionInput {
            query: "q".into(),
            expected_document_ids: vec![Uuid::nil()],
            ..QuestionInput::default()
        };
        assert!(nil.validated().unwrap_err().to_string().contains("nil"));
        let long = q(&"x".repeat(FIELD_MAX_CHARS + 1));
        assert!(
            long.validated()
                .unwrap_err()
                .to_string()
                .contains("query is over")
        );
    }

    #[test]
    fn patch_keeps_absent_fields_and_clears_null_ones() {
        let cur = QuestionInput {
            query: "q".into(),
            expected_document: Some("d".into()),
            expected_leaf: Some("l".into()),
            expected_text: Some("t".into()),
            ..QuestionInput::default()
        };
        let p: QuestionPatch =
            serde_json::from_str(r#"{"expected_leaf":null,"expected_text":"new"}"#).unwrap();
        assert!(!p.is_empty());
        let next = p.apply(cur.clone());
        assert_eq!(
            next.expected_document.as_deref(),
            Some("d"),
            "absent = kept"
        );
        assert_eq!(next.expected_leaf, None, "null = cleared");
        assert_eq!(next.expected_text.as_deref(), Some("new"));
        let empty: QuestionPatch = serde_json::from_str("{}").unwrap();
        assert!(empty.is_empty());
        assert!(serde_json::from_str::<QuestionPatch>(r#"{"bogus":1}"#).is_err());
    }

    #[test]
    fn csv_parses_quotes_newlines_and_crlf_with_line_numbers() {
        let rows = parse_csv("a,b\r\n\"x, \"\"y\"\"\",\"multi\nline\"\nlast,1");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], (1, vec!["a".to_string(), "b".to_string()]));
        assert_eq!(rows[1].0, 2);
        assert_eq!(rows[1].1, ["x, \"y\"", "multi\nline"]);
        assert_eq!(rows[2], (4, vec!["last".to_string(), "1".to_string()]));
    }

    #[test]
    fn csv_import_reads_any_column_order_and_ids() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let csv = format!(
            "\u{feff}Expected_Leaf,query,extra,expected_chunk_ids,expected_document\n\
             sea,How big is a whale?,ignored,{a};{b},\n\
             ,,,,\n\
             stars,\"Which planet, then?\",,,stars.txt\n"
        );
        let qs = parse_questions_csv(&csv).unwrap();
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[0].query, "How big is a whale?");
        assert_eq!(qs[0].expected_leaf.as_deref(), Some("sea"));
        assert_eq!(qs[0].expected_chunk_ids, [a, b]);
        assert_eq!(qs[0].expected_document, None, "blank cell = none");
        assert_eq!(qs[1].query, "Which planet, then?");
        assert_eq!(qs[1].expected_document.as_deref(), Some("stars.txt"));
    }

    #[test]
    fn csv_import_is_all_or_nothing_and_names_lines() {
        let csv = "query,expected_document,expected_chunk_ids\n\
                   ok,doc,\n\
                   ,doc,\n\
                   no expectation,,\n\
                   bad id,,not-a-uuid\n\
                   OK,doc,\n";
        let errs = parse_questions_csv(csv).unwrap_err();
        assert_eq!(errs.len(), 4, "{errs:?}");
        assert!(errs[0].starts_with("line 3:") && errs[0].contains("query is empty"));
        assert!(errs[1].starts_with("line 4:") && errs[1].contains("no expectation"));
        assert!(errs[2].starts_with("line 5:") && errs[2].contains("not-a-uuid"));
        assert!(errs[3].starts_with("line 6:") && errs[3].contains("earlier line"));

        let no_query = parse_questions_csv("question,expected_document\nq,d\n").unwrap_err();
        assert!(no_query[0].contains("`query`"), "{no_query:?}");
        let no_expect = parse_questions_csv("query,expected_leaf\nq,l\n").unwrap_err();
        assert!(no_expect[0].contains("expectation column"), "{no_expect:?}");
        assert!(parse_questions_csv("query,expected_text\n\n").is_err());
        assert!(parse_questions_csv("").is_err());
    }

    fn hit<'a>(chunk: u128, doc: u128, text: &'a str, source: &'a str) -> Hit<'a> {
        Hit {
            chunk_id: Uuid::from_u128(chunk),
            document_id: Uuid::from_u128(doc),
            text,
            source,
            title: None,
        }
    }

    #[test]
    fn expectations_match_any_criterion() {
        let rq = |f: &dyn Fn(&mut RunQuestion)| {
            let mut r = RunQuestion::new(
                None,
                QuestionInput {
                    query: "q".into(),
                    ..QuestionInput::default()
                },
            );
            f(&mut r);
            Expectation::of(&r)
        };
        let hits = [
            hit(1, 10, "nothing here", "a.txt"),
            hit(2, 20, "The  Great\nWHALE swam", "dir/sea.txt"),
            Hit {
                title: Some("Refund timing"),
                ..hit(3, 30, "refunds", "api")
            },
        ];
        let by_doc = rq(&|r| r.expected_document = Some("sea.txt".into()));
        assert_eq!(by_doc.first_rank(&hits), Some(2), "file name of the source");
        let by_title = rq(&|r| r.expected_document = Some("Refund timing".into()));
        assert_eq!(by_title.first_rank(&hits), Some(3));
        let by_doc_id = rq(&|r| r.expected_document_ids = vec![Uuid::from_u128(30)]);
        assert_eq!(by_doc_id.first_rank(&hits), Some(3));
        let by_chunk = rq(&|r| r.expected_chunk_ids = vec![Uuid::from_u128(2)]);
        assert_eq!(by_chunk.first_rank(&hits), Some(2));
        let by_text = rq(&|r| r.expected_text = Some("great whale".into()));
        assert_eq!(
            by_text.first_rank(&hits),
            Some(2),
            "case and whitespace folded"
        );
        let any = rq(&|r| {
            r.expected_text = Some("no such text".into());
            r.expected_chunk_ids = vec![Uuid::from_u128(1)];
        });
        assert_eq!(any.first_rank(&hits), Some(1), "any expectation counts");
        let none = rq(&|r| r.expected_document = Some("other.txt".into()));
        assert_eq!(none.first_rank(&hits), None);
    }

    #[test]
    fn leaves_resolve_by_id_path_or_unique_name() {
        let leaves = vec![
            (
                Uuid::from_u128(1),
                vec!["lib".to_string(), "sea".to_string()],
            ),
            (
                Uuid::from_u128(2),
                vec!["lib".to_string(), "a".to_string(), "x".to_string()],
            ),
            (
                Uuid::from_u128(3),
                vec!["lib".to_string(), "b".to_string(), "x".to_string()],
            ),
        ];
        assert_eq!(resolve_leaf("sea", &leaves), Ok(Uuid::from_u128(1)));
        assert_eq!(resolve_leaf("lib/b/x", &leaves), Ok(Uuid::from_u128(3)));
        assert_eq!(
            resolve_leaf(&Uuid::from_u128(2).to_string(), &leaves),
            Ok(Uuid::from_u128(2))
        );
        assert!(resolve_leaf("x", &leaves).unwrap_err().contains("several"));
        assert!(resolve_leaf("nope", &leaves).is_err());
        assert!(resolve_leaf(&Uuid::from_u128(9).to_string(), &leaves).is_err());
    }
}
