//! Retrieval evaluation: load the seed tree, then recall@1/5/10 (hit = a result chunk whose
//! document source matches `expected_document`) and leaf-routing accuracy for `embed`,
//! `hybrid` and `flat` (all leaves searched, no routing).

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::search::ChunkFilter;
use crate::db::stores::{NewStore, StoreKind};
use crate::engine::{Engine, StoreView};
use crate::error::{Result, StrataError};
use crate::ingest::IngestOutcome;
use crate::query::QueryRequest;
use crate::routing::RouteMode;
use crate::search::search_leaves_ef;
use crate::tenant::Tenant;

/// Results fetched per question (recall@10 needs 10).
pub const EVAL_TOP_K: usize = 10;

/// One evaluated mode: a routing mode, or `flat` (every leaf of the tree searched, no routing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalMode {
    Embed,
    Laya,
    Hybrid,
    Flat,
}

impl EvalMode {
    pub fn label(self) -> &'static str {
        match self {
            EvalMode::Embed => "embed",
            EvalMode::Laya => "laya",
            EvalMode::Hybrid => "hybrid",
            EvalMode::Flat => "flat",
        }
    }

    pub(crate) fn route_mode(self) -> Option<RouteMode> {
        match self {
            EvalMode::Embed => Some(RouteMode::Embed),
            EvalMode::Laya => Some(RouteMode::Laya),
            EvalMode::Hybrid => Some(RouteMode::Hybrid),
            EvalMode::Flat => None,
        }
    }
}

/// What [`evaluate_with`] scores. The default (embed, hybrid, flat at the query defaults) is what
/// `strata eval` reports. Routed modes are reported in the given order, `flat` always last.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalOptions {
    pub modes: Vec<EvalMode>,
    pub beam: usize,
    pub min_score: f32,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            modes: vec![EvalMode::Embed, EvalMode::Hybrid, EvalMode::Flat],
            beam: crate::query::d_beam(),
            min_score: crate::query::d_min_score(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct EvalQuestion {
    pub query: String,
    pub expected_document: String,
    pub expected_leaf: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TreeSpec {
    /// Relative to the directory holding tree.json.
    pub corpus_dir: String,
    pub root: NodeSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeSpec {
    pub name: String,
    pub kind: StoreKind,
    pub description: String,
    #[serde(default)]
    pub children: Vec<NodeSpec>,
    #[serde(default)]
    pub files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlatNode {
    /// Names from the root down to and including this node.
    pub path: Vec<String>,
    pub kind: StoreKind,
    pub name: String,
    pub description: String,
    pub files: Vec<String>,
    pub child_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuestionOutcome {
    /// 1-based rank of the first matching result, if any.
    pub rank: Option<usize>,
    /// Whether routing reached the expected leaf (`None` for flat).
    pub leaf_hit: Option<bool>,
    pub laya_consulted: bool,
    pub laya_fallback: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModeReport {
    pub mode: &'static str,
    pub n: usize,
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub leaf_accuracy: Option<f64>,
    pub laya_consulted: usize,
    pub laya_fallbacks: usize,
}

impl ModeReport {
    pub fn from_outcomes(mode: &'static str, outcomes: &[QuestionOutcome]) -> Self {
        let ranks: Vec<Option<usize>> = outcomes.iter().map(|o| o.rank).collect();
        let leaf_hits: Vec<bool> = outcomes.iter().filter_map(|o| o.leaf_hit).collect();
        ModeReport {
            mode,
            n: outcomes.len(),
            recall_at_1: recall_at_k(&ranks, 1),
            recall_at_5: recall_at_k(&ranks, 5),
            recall_at_10: recall_at_k(&ranks, 10),
            leaf_accuracy: if leaf_hits.is_empty() {
                None
            } else {
                Some(fraction(&leaf_hits))
            },
            laya_consulted: outcomes.iter().filter(|o| o.laya_consulted).count(),
            laya_fallbacks: outcomes.iter().filter(|o| o.laya_fallback).count(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoadSummary {
    pub created: usize,
    pub existing: usize,
    /// Existing stores whose description in tree.json differed and was re-embedded.
    pub updated: usize,
    pub chunks: i64,
    pub failed: Vec<String>,
}

pub fn parse_questions(contents: &str) -> Result<Vec<EvalQuestion>> {
    let mut out = Vec::new();
    for (i, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let q: EvalQuestion = serde_json::from_str(line)
            .map_err(|e| StrataError::InvalidInput(format!("questions line {}: {e}", i + 1)))?;
        out.push(q);
    }
    Ok(out)
}

/// Pre-order (parents before children), children in file order.
pub fn flatten_tree(spec: &TreeSpec) -> Vec<FlatNode> {
    let mut out = Vec::new();
    let mut stack: Vec<(Vec<String>, &NodeSpec)> = vec![(Vec::new(), &spec.root)];
    while let Some((mut path, node)) = stack.pop() {
        path.push(node.name.clone());
        for child in node.children.iter().rev() {
            stack.push((path.clone(), child));
        }
        out.push(FlatNode {
            path,
            kind: node.kind,
            name: node.name.clone(),
            description: node.description.clone(),
            files: node.files.clone(),
            child_count: node.children.len(),
        });
    }
    out
}

pub fn check_tree(spec: &TreeSpec) -> Vec<String> {
    let mut problems = Vec::new();
    let mut leaf_names = HashSet::new();
    for n in flatten_tree(spec) {
        match n.kind {
            StoreKind::Router => {
                if !n.files.is_empty() {
                    problems.push(format!("router `{}` lists files", n.name));
                }
            }
            StoreKind::Leaf => {
                if n.files.is_empty() {
                    problems.push(format!("leaf `{}` has no files", n.name));
                }
                if n.child_count > 0 {
                    problems.push(format!("leaf `{}` has children", n.name));
                }
                if !leaf_names.insert(n.name.clone()) {
                    problems.push(format!("duplicate leaf name `{}`", n.name));
                }
            }
        }
    }
    problems
}

pub fn check_questions_against_tree(questions: &[EvalQuestion], spec: &TreeSpec) -> Vec<String> {
    let leaves: HashMap<String, HashSet<String>> = flatten_tree(spec)
        .into_iter()
        .filter(|n| n.kind == StoreKind::Leaf)
        .map(|n| (n.name, n.files.into_iter().collect()))
        .collect();
    let mut problems = Vec::new();
    for (i, q) in questions.iter().enumerate() {
        if q.query.trim().is_empty() {
            problems.push(format!("question {}: empty query", i + 1));
        }
        match leaves.get(&q.expected_leaf) {
            None => problems.push(format!(
                "question {}: unknown leaf `{}`",
                i + 1,
                q.expected_leaf
            )),
            Some(files) if !files.contains(&q.expected_document) => problems.push(format!(
                "question {}: `{}` is not a file of leaf `{}`",
                i + 1,
                q.expected_document,
                q.expected_leaf
            )),
            Some(_) => {}
        }
    }
    problems
}

/// `source` equals `expected`, or its file name does (CLI ingests store full paths).
pub fn source_matches(source: &str, expected: &str) -> bool {
    source == expected || Path::new(source).file_name().and_then(|f| f.to_str()) == Some(expected)
}

pub fn first_hit_rank(ranked_sources: &[&str], expected: &str) -> Option<usize> {
    ranked_sources
        .iter()
        .position(|s| source_matches(s, expected))
        .map(|i| i + 1)
}

pub fn recall_at_k(ranks: &[Option<usize>], k: usize) -> f64 {
    if ranks.is_empty() {
        return 0.0;
    }
    let hits = ranks.iter().filter(|r| r.is_some_and(|r| r <= k)).count();
    hits as f64 / ranks.len() as f64
}

pub fn fraction(hits: &[bool]) -> f64 {
    if hits.is_empty() {
        return 0.0;
    }
    hits.iter().filter(|h| **h).count() as f64 / hits.len() as f64
}

/// True when every routed mode's recall@5 is at least flat's (vacuously true without flat).
pub fn verdict_passes(reports: &[ModeReport]) -> bool {
    let Some(flat) = reports.iter().find(|r| r.mode == "flat") else {
        return true;
    };
    reports
        .iter()
        .filter(|r| r.mode != "flat")
        .all(|r| r.recall_at_5 >= flat.recall_at_5)
}

/// `contaminated`: foreign stores were present (`--allow-foreign-stores`). The verdict lines then say so
/// instead of a bare PASS/FAIL, because a routed score over a contaminated tree is not the measured one.
pub fn render_report(reports: &[ModeReport], contaminated: bool) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<8} {:>4} {:>7} {:>7} {:>7} {:>9} {:>15} {:>15}",
        "mode", "n", "R@1", "R@5", "R@10", "leaf_acc", "laya_consulted", "laya_fallbacks"
    );
    for r in reports {
        let leaf = r
            .leaf_accuracy
            .map_or_else(|| "-".to_string(), |a| format!("{a:.3}"));
        let _ = writeln!(
            out,
            "{:<8} {:>4} {:>7.3} {:>7.3} {:>7.3} {:>9} {:>15} {:>15}",
            r.mode,
            r.n,
            r.recall_at_1,
            r.recall_at_5,
            r.recall_at_10,
            leaf,
            r.laya_consulted,
            r.laya_fallbacks
        );
    }
    if let Some(flat) = reports.iter().find(|r| r.mode == "flat") {
        for r in reports.iter().filter(|r| r.mode != "flat") {
            let verdict = match (r.recall_at_5 >= flat.recall_at_5, contaminated) {
                (true, false) => "PASS",
                (false, false) => "FAIL",
                (true, true) => "CONTAMINATED (would be PASS; not a verdict)",
                (false, true) => "CONTAMINATED (would be FAIL; not a verdict)",
            };
            let _ = writeln!(
                out,
                "{} recall@5 {:.3} >= flat {:.3}: {verdict}",
                r.mode, r.recall_at_5, flat.recall_at_5
            );
        }
    }
    out
}

/// Name path from the top level down to each store. A parent missing from `views` ends the
/// walk; the step bound guards against a cyclic parent chain.
pub fn store_paths(views: &[StoreView]) -> HashMap<Uuid, Vec<String>> {
    let by_id: HashMap<Uuid, &StoreView> = views.iter().map(|v| (v.id, v)).collect();
    views
        .iter()
        .map(|v| {
            let mut path = vec![v.name.clone()];
            let mut cur = v.parent_id;
            while let Some(p) = cur.and_then(|id| by_id.get(&id)) {
                if path.len() > views.len() {
                    break;
                }
                path.push(p.name.clone());
                cur = p.parent_id;
            }
            path.reverse();
            (v.id, path)
        })
        .collect()
}

/// A store in the database that `tree.json` does not describe.
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignStore {
    pub path: Vec<String>,
    pub id: Uuid,
    pub kind: StoreKind,
    pub chunk_count: i64,
}

/// Stores whose name path (top level down, i.e. name + parent, as `load_tree` matches them) is
/// not a node of `spec`: another top-level store such as a leftover `bench-tmp-*` leaf or a
/// benchmark tree, an extra store under a tree router, or a tree name under a different parent.
/// Any of these competes for beam slots in routed retrieval and silently changes embed/hybrid
/// scores, so `strata eval` refuses to report over them. Sorted by path.
pub fn foreign_stores(spec: &TreeSpec, views: &[StoreView]) -> Vec<ForeignStore> {
    let known: HashSet<Vec<String>> = flatten_tree(spec).into_iter().map(|n| n.path).collect();
    let paths = store_paths(views);
    let mut out: Vec<ForeignStore> = views
        .iter()
        .filter_map(|v| {
            let path = &paths[&v.id];
            (!known.contains(path)).then(|| ForeignStore {
                path: path.clone(),
                id: v.id,
                kind: v.kind,
                chunk_count: v.chunk_count,
            })
        })
        .collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// One line per foreign store: path, id, kind, document/chunk counts and the command that removes it.
pub fn render_foreign(foreign: &[ForeignStore], doc_counts: &HashMap<Uuid, i64>) -> String {
    let mut out = String::new();
    for f in foreign {
        let docs = doc_counts.get(&f.id).copied().unwrap_or(0);
        let recursive = if f.kind == StoreKind::Router {
            " --recursive"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "  {} id={} kind={} docs={docs} chunks={}  -> strata store rm {}{recursive}",
            f.path.join("/"),
            f.id,
            f.kind.as_str(),
            f.chunk_count,
            f.id
        );
    }
    out
}

/// Create the tree described by `tree_path` (reusing stores that already exist by parent+name)
/// and ingest every listed file; ingest dedup makes reruns cheap. Per-file failures are collected.
pub fn read_tree(tree_path: &Path) -> Result<TreeSpec> {
    serde_json::from_str(&std::fs::read_to_string(tree_path)?)
        .map_err(|e| StrataError::InvalidInput(format!("{}: {e}", tree_path.display())))
}

/// Loads into `tenant`'s stores.
pub async fn load_tree(engine: &Engine, tenant: Tenant, tree_path: &Path) -> Result<LoadSummary> {
    let spec = read_tree(tree_path)?;
    let problems = check_tree(&spec);
    if !problems.is_empty() {
        return Err(StrataError::InvalidInput(problems.join("; ")));
    }
    let corpus = tree_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(&spec.corpus_dir);
    let mut ids: HashMap<Vec<String>, Uuid> = HashMap::new();
    let mut summary = LoadSummary::default();
    for node in flatten_tree(&spec) {
        let parent_path = &node.path[..node.path.len() - 1];
        let parent_id = if parent_path.is_empty() {
            None
        } else {
            Some(*ids.get(parent_path).ok_or_else(|| {
                StrataError::InvalidInput(format!("parent of `{}` was not created", node.name))
            })?)
        };
        let views = engine.store_tree(tenant).await?;
        let mut updated = false;
        let id = match views
            .iter()
            .find(|v| v.parent_id == parent_id && v.name == node.name)
        {
            Some(v) if v.kind != node.kind => {
                return Err(StrataError::InvalidInput(format!(
                    "store `{}` exists as {} but tree.json says {}",
                    node.name,
                    v.kind.as_str(),
                    node.kind.as_str()
                )));
            }
            Some(v) => {
                let wanted = node.description.trim();
                if v.description != wanted {
                    // same path as PATCH /api/stores/{id}: re-embeds desc_vec, recomputes profiles
                    engine
                        .update_store(tenant, v.id, None, Some(wanted.to_string()))
                        .await?;
                    updated = true;
                }
                v.id
            }
            None => {
                engine
                    .create_store(
                        tenant,
                        NewStore {
                            parent_id,
                            kind: node.kind,
                            name: node.name.clone(),
                            description: node.description.clone(),
                        },
                    )
                    .await?
                    .id
            }
        };
        println!(
            "store {} [{}] {id}",
            node.path.join("/"),
            node.kind.as_str()
        );
        if updated {
            summary.updated += 1;
            println!("  updated description");
        }
        ids.insert(node.path.clone(), id);

        for file in &node.files {
            let path = corpus.join(file);
            let res: Result<IngestOutcome> = async {
                let bytes = tokio::fs::read(&path).await?;
                let text = String::from_utf8(bytes).map_err(|_| {
                    StrataError::InvalidInput(format!("{} is not valid UTF-8", path.display()))
                })?;
                engine.ingest(tenant, id, file.clone(), None, &text).await
            }
            .await;
            match res {
                Ok(IngestOutcome::Created(d)) => {
                    summary.created += 1;
                    summary.chunks += d.chunk_count;
                    println!(
                        "  created  {file} chunks={} tokens={}",
                        d.chunk_count, d.token_count
                    );
                }
                Ok(IngestOutcome::Existing(_)) => {
                    summary.existing += 1;
                    println!("  existing {file}");
                }
                Err(e) => {
                    eprintln!("  FAILED   {file}: {e}");
                    summary.failed.push(format!("{file}: {e}"));
                }
            }
        }
    }
    Ok(summary)
}

/// Score `questions` against the stores of `spec` (resolved by name path from the root).
/// Flat mode searches only `spec`'s leaves, so stray stores (e.g. a leftover `bench-tmp-*`
/// leaf) cannot leak into it; every leaf must hold every file tree.json lists for it.
/// Embed/hybrid route over the whole database, so stray stores DO change them: `run` refuses
/// via [`foreign_stores`] before calling this.
pub async fn evaluate(
    engine: &Engine,
    tenant: Tenant,
    spec: &TreeSpec,
    questions: &[EvalQuestion],
) -> Result<Vec<ModeReport>> {
    evaluate_with(engine, tenant, spec, questions, &EvalOptions::default()).await
}

/// [`evaluate`] for the chosen modes and routing parameters (`POST /api/eval`). Read-only: pool
/// queries and `Engine::query`, no transaction held across questions, so ingest is never blocked.
pub async fn evaluate_with(
    engine: &Engine,
    tenant: Tenant,
    spec: &TreeSpec,
    questions: &[EvalQuestion],
    opts: &EvalOptions,
) -> Result<Vec<ModeReport>> {
    let views = engine.store_tree(tenant).await?;
    let mut paths = store_paths(&views);
    let by_path: HashMap<Vec<String>, (Uuid, StoreKind)> = views
        .iter()
        .map(|v| (paths.remove(&v.id).unwrap_or_default(), (v.id, v.kind)))
        .collect();
    let docs: Vec<(Uuid, Uuid, String)> = sqlx::query_as(
        "SELECT d.id, d.store_id, d.source FROM documents d JOIN stores s ON s.id = d.store_id \
         WHERE s.tenant = $1",
    )
    .bind(tenant.id())
    .fetch_all(&engine.pool)
    .await?;
    let in_store = |store: Uuid, file: &str| {
        docs.iter()
            .any(|(_, s, src)| *s == store && source_matches(src, file))
    };

    let mut problems = check_questions_against_tree(questions, spec);
    let mut leaf_ids: HashMap<String, Uuid> = HashMap::new();
    for leaf in flatten_tree(spec)
        .into_iter()
        .filter(|n| n.kind == StoreKind::Leaf)
    {
        match by_path.get(&leaf.path) {
            Some(&(id, StoreKind::Leaf)) => {
                for file in &leaf.files {
                    if !in_store(id, file) {
                        problems.push(format!("`{file}` is not ingested in leaf `{}`", leaf.name));
                    }
                }
                leaf_ids.insert(leaf.name, id);
            }
            _ => problems.push(format!(
                "leaf `{}` ({}) not in the store tree",
                leaf.name,
                leaf.path.join("/")
            )),
        }
    }
    for (i, q) in questions.iter().enumerate() {
        let ingested = leaf_ids
            .get(&q.expected_leaf)
            .is_some_and(|&id| in_store(id, &q.expected_document));
        if !ingested {
            problems.push(format!(
                "question {}: `{}` is not ingested in leaf `{}`",
                i + 1,
                q.expected_document,
                q.expected_leaf
            ));
        }
    }
    if !problems.is_empty() {
        return Err(StrataError::InvalidInput(problems.join("\n")));
    }
    let all_leaves: Vec<Uuid> = leaf_ids.values().copied().collect();
    // the eval tree's own root: queries route within it, also when the tenant holds other roots
    // (a query naming none would be 409 ambiguous_root then, D90)
    let eval_root = by_path
        .get(std::slice::from_ref(&spec.root.name))
        .map(|&(id, _)| id);
    let sources: HashMap<Uuid, String> = docs.into_iter().map(|(id, _, src)| (id, src)).collect();
    let source_of = |id: &Uuid| sources.get(id).map_or("", String::as_str);

    let mut reports = Vec::new();
    let mut seen = HashSet::new();
    let routed = opts
        .modes
        .iter()
        .filter(|m| seen.insert(**m))
        .filter_map(|m| Some((m.label(), m.route_mode()?)));
    for (label, mode) in routed {
        let mut outcomes = Vec::with_capacity(questions.len());
        for q in questions {
            let req = QueryRequest {
                top_k: EVAL_TOP_K,
                route_mode: mode,
                beam: opts.beam,
                min_score: opts.min_score,
                root_id: eval_root,
                ..QueryRequest::new(q.query.clone())
            };
            let resp = engine.query(tenant, &req).await?;
            let ranked: Vec<&str> = resp
                .results
                .iter()
                .map(|r| source_of(&r.document_id))
                .collect();
            let expected_leaf = leaf_ids[q.expected_leaf.as_str()];
            outcomes.push(QuestionOutcome {
                rank: first_hit_rank(&ranked, &q.expected_document),
                leaf_hit: Some(resp.routing.leaves.contains(&expected_leaf)),
                laya_consulted: resp.routing.laya_consulted,
                laya_fallback: resp.routing.laya_fallback.is_some(),
            });
        }
        reports.push(ModeReport::from_outcomes(label, &outcomes));
    }
    if !opts.modes.contains(&EvalMode::Flat) {
        return Ok(reports);
    }

    let mut flat = Vec::with_capacity(questions.len());
    let inf = engine.inference(tenant).await?;
    for q in questions {
        let q_vec = inf.embed_one(q.query.clone()).await?;
        let (hits, _) = search_leaves_ef(
            &engine.pool,
            tenant,
            &inf,
            &all_leaves,
            &[q_vec],
            &q.query,
            EVAL_TOP_K,
            engine.cfg.rerank_candidates,
            &ChunkFilter::default(),
            engine.cfg.hnsw_ef_search,
            engine.cfg.exact_scan_max_chunks,
        )
        .await?;
        let ranked: Vec<&str> = hits
            .iter()
            .map(|h| source_of(&h.chunk.document_id))
            .collect();
        flat.push(QuestionOutcome {
            rank: first_hit_rank(&ranked, &q.expected_document),
            leaf_hit: None,
            laya_consulted: false,
            laya_fallback: false,
        });
    }
    reports.push(ModeReport::from_outcomes("flat", &flat));
    Ok(reports)
}

/// The tree a questions file is evaluated against: `--tree`, else `--load`, else the
/// `tree.json` next to the questions file.
pub fn resolve_tree_path(
    questions: &Path,
    tree: Option<&Path>,
    load: Option<&Path>,
) -> std::path::PathBuf {
    tree.or(load).map_or_else(
        || {
            questions
                .parent()
                .unwrap_or(Path::new("."))
                .join("tree.json")
        },
        Path::to_path_buf,
    )
}

pub async fn run(
    engine: &Engine,
    tenant: Tenant,
    questions: Option<&Path>,
    load: Option<&Path>,
    tree: Option<&Path>,
    json: bool,
    allow_foreign_stores: bool,
) -> Result<ExitCode> {
    if questions.is_none() && load.is_none() {
        return Err(StrataError::InvalidInput(
            "nothing to do: pass a questions file and/or --load <tree.json>".into(),
        ));
    }
    let mut code = ExitCode::SUCCESS;
    if let Some(tree) = load {
        let s = load_tree(engine, tenant, tree).await?;
        println!(
            "load: {} created, {} existing, {} descriptions updated, {} failed, {} chunks embedded",
            s.created,
            s.existing,
            s.updated,
            s.failed.len(),
            s.chunks
        );
        if !s.failed.is_empty() {
            code = ExitCode::FAILURE;
        }
    }
    if let Some(path) = questions {
        let spec = read_tree(&resolve_tree_path(path, tree, load))?;
        let qs = parse_questions(&std::fs::read_to_string(path)?)?;
        let foreign = foreign_stores(&spec, &engine.store_tree(tenant).await?);
        let warning = if foreign.is_empty() {
            None
        } else {
            let counts: Vec<(Uuid, i64)> = sqlx::query_as(
                "SELECT d.store_id, count(*) FROM documents d JOIN stores s ON s.id = d.store_id \
                 WHERE s.tenant = $1 GROUP BY d.store_id",
            )
            .bind(tenant.id())
            .fetch_all(&engine.pool)
            .await?;
            let listing = render_foreign(&foreign, &counts.into_iter().collect());
            if !allow_foreign_stores {
                eprintln!(
                    "refusing to evaluate: {} store(s) outside the evaluated tree would take beam \
                     slots in embed/hybrid routing and change their scores:\n{listing}\
                     remove them, or pass --allow-foreign-stores to report anyway (marked CONTAMINATED)",
                    foreign.len()
                );
                return Ok(ExitCode::FAILURE);
            }
            let w = format!(
                "WARNING: CONTAMINATED: {} store(s) outside the evaluated tree; embed/hybrid \
                 scores below are NOT comparable to a clean run (--allow-foreign-stores):\n{listing}",
                foreign.len()
            );
            Some(w)
        };
        let reports = evaluate(engine, tenant, &spec, &qs).await?;
        if json {
            let out = match &warning {
                // a contaminated report changes shape, so a consumer of the clean array cannot
                // mistake it for one
                Some(w) => serde_json::json!({
                    "contaminated": true,
                    "warning": w,
                    "foreign_stores": foreign.iter().map(|f| serde_json::json!({
                        "path": f.path.join("/"),
                        "id": f.id,
                        "kind": f.kind.as_str(),
                    })).collect::<Vec<_>>(),
                    "reports": reports,
                }),
                None => serde_json::to_value(&reports).expect("ModeReport serialises"),
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&out).expect("report serialises")
            );
        } else {
            if let Some(w) = &warning {
                print!("{w}");
            }
            print!("{}", render_report(&reports, warning.is_some()));
        }
        // the report is printed first; a FAIL verdict then fails the command, and so does a
        // contaminated run: its scores are not the tree's, so it can never count as a pass
        if !verdict_passes(&reports) || warning.is_some() {
            code = ExitCode::FAILURE;
        }
    }
    Ok(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tree() -> TreeSpec {
        serde_json::from_str(
            r#"{"corpus_dir":"corpus","root":{"name":"lib","kind":"router","description":"d","children":[
                {"name":"sci","kind":"router","description":"d","children":[
                    {"name":"bio","kind":"leaf","description":"d","files":["darwin.txt"]}]},
                {"name":"fic","kind":"leaf","description":"d","files":["austen.txt","melville.txt"]}]}}"#,
        )
        .unwrap()
    }

    fn view(id: u128, parent: Option<u128>, kind: StoreKind, name: &str) -> StoreView {
        StoreView {
            id: Uuid::from_u128(id),
            parent_id: parent.map(Uuid::from_u128),
            kind,
            name: name.into(),
            description: String::new(),
            token_count: 0,
            chunk_count: 7,
            children: Vec::new(),
        }
    }

    /// The stores `load_tree` creates for `tiny_tree`: lib / sci / bio, lib / fic.
    fn tiny_views() -> Vec<StoreView> {
        vec![
            view(1, None, StoreKind::Router, "lib"),
            view(2, Some(1), StoreKind::Router, "sci"),
            view(3, Some(2), StoreKind::Leaf, "bio"),
            view(4, Some(1), StoreKind::Leaf, "fic"),
        ]
    }

    #[test]
    fn foreign_stores_clean_tree_is_empty() {
        assert!(foreign_stores(&tiny_tree(), &tiny_views()).is_empty());
        // a partially loaded tree has nothing foreign either; evaluate reports what is missing
        assert!(foreign_stores(&tiny_tree(), &tiny_views()[..2]).is_empty());
    }

    #[test]
    fn foreign_stores_flags_a_top_level_bench_leaf_and_its_whole_foreign_tree() {
        let mut views = tiny_views();
        views.push(view(9, None, StoreKind::Leaf, "bench-tmp-0192"));
        views.push(view(10, None, StoreKind::Router, "hotpotqa-n50"));
        views.push(view(11, Some(10), StoreKind::Leaf, "lib"));
        let f = foreign_stores(&tiny_tree(), &views);
        let paths: Vec<String> = f.iter().map(|f| f.path.join("/")).collect();
        assert_eq!(
            paths,
            ["bench-tmp-0192", "hotpotqa-n50", "hotpotqa-n50/lib"]
        );
        assert_eq!(f[0].id, Uuid::from_u128(9));
        assert_eq!((f[0].kind, f[0].chunk_count), (StoreKind::Leaf, 7));
    }

    #[test]
    fn foreign_stores_flags_an_extra_leaf_under_a_tree_router() {
        let mut views = tiny_views();
        views.push(view(9, Some(2), StoreKind::Leaf, "bench-tmp-0192"));
        let f = foreign_stores(&tiny_tree(), &views);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!(f[0].path, ["lib", "sci", "bench-tmp-0192"]);
    }

    #[test]
    fn foreign_stores_flags_a_tree_name_under_a_different_parent() {
        let mut views = tiny_views();
        // `bio` is a tree leaf, but under `lib/sci`, not directly under `lib`
        views.push(view(9, Some(1), StoreKind::Leaf, "bio"));
        // and `fic` at the top level is not `lib/fic`
        views.push(view(10, None, StoreKind::Leaf, "fic"));
        let f = foreign_stores(&tiny_tree(), &views);
        let ids: Vec<Uuid> = f.iter().map(|f| f.id).collect();
        assert_eq!(ids, [Uuid::from_u128(10), Uuid::from_u128(9)]);
    }

    #[test]
    fn render_foreign_lists_path_id_kind_counts_and_rm_hint() {
        let mut views = tiny_views();
        views.push(view(9, None, StoreKind::Leaf, "bench-tmp-x"));
        views.push(view(10, None, StoreKind::Router, "other"));
        let f = foreign_stores(&tiny_tree(), &views);
        let out = render_foreign(&f, &HashMap::from([(Uuid::from_u128(9), 2)]));
        let id9 = Uuid::from_u128(9);
        let id10 = Uuid::from_u128(10);
        assert!(
            out.contains(&format!(
                "bench-tmp-x id={id9} kind=leaf docs=2 chunks=7  -> strata store rm {id9}\n"
            )),
            "{out}"
        );
        assert!(
            out.contains(&format!(
                "docs=0 chunks=7  -> strata store rm {id10} --recursive"
            )),
            "{out}"
        );
    }

    #[test]
    fn parse_questions_skips_blank_lines_and_reports_line_numbers() {
        let ok = "{\"query\":\"q1\",\"expected_document\":\"a.txt\",\"expected_leaf\":\"l\"}\n\n\
                  {\"query\":\"q2\",\"expected_document\":\"b.txt\",\"expected_leaf\":\"l\"}\n";
        let qs = parse_questions(ok).unwrap();
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[1].expected_document, "b.txt");
        let bad = "{\"query\":\"q1\",\"expected_document\":\"a.txt\",\"expected_leaf\":\"l\"}\n{\"query\":1}\n";
        let err = parse_questions(bad).unwrap_err().to_string();
        assert!(err.contains("line 2"), "{err}");
    }

    #[test]
    fn flatten_tree_is_preorder_parents_first() {
        let flat = flatten_tree(&tiny_tree());
        let paths: Vec<String> = flat.iter().map(|n| n.path.join("/")).collect();
        assert_eq!(paths, ["lib", "lib/sci", "lib/sci/bio", "lib/fic"]);
        assert_eq!(flat[3].files, ["austen.txt", "melville.txt"]);
        assert_eq!(flat[2].kind, StoreKind::Leaf);
    }

    #[test]
    fn check_tree_flags_structural_problems() {
        assert!(check_tree(&tiny_tree()).is_empty());
        let bad: TreeSpec = serde_json::from_str(
            r#"{"corpus_dir":"c","root":{"name":"lib","kind":"router","description":"d","files":["x.txt"],"children":[
                {"name":"a","kind":"leaf","description":"d","files":[]},
                {"name":"a","kind":"leaf","description":"d","files":["y.txt"],"children":[
                    {"name":"z","kind":"leaf","description":"d","files":["z.txt"]}]}]}}"#,
        )
        .unwrap();
        let problems = check_tree(&bad);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("router `lib` lists files")),
            "{problems:?}"
        );
        assert!(
            problems.iter().any(|p| p.contains("leaf `a` has no files")),
            "{problems:?}"
        );
        assert!(
            problems
                .iter()
                .any(|p| p.contains("duplicate leaf name `a`")),
            "{problems:?}"
        );
        assert!(
            problems.iter().any(|p| p.contains("leaf `a` has children")),
            "{problems:?}"
        );
    }

    #[test]
    fn check_questions_against_tree_flags_unknown_leaf_and_wrong_file() {
        let q = |doc: &str, leaf: &str| EvalQuestion {
            query: "q".into(),
            expected_document: doc.into(),
            expected_leaf: leaf.into(),
        };
        let tree = tiny_tree();
        assert!(check_questions_against_tree(&[q("darwin.txt", "bio")], &tree).is_empty());
        let p = check_questions_against_tree(&[q("darwin.txt", "fic"), q("x.txt", "nope")], &tree);
        assert_eq!(p.len(), 2, "{p:?}");
        assert!(p[0].contains("question 1"));
        assert!(p[1].contains("unknown leaf `nope`"));
    }

    #[test]
    fn source_matches_exact_or_file_name() {
        assert!(source_matches(
            "pg84-frankenstein.txt",
            "pg84-frankenstein.txt"
        ));
        assert!(source_matches(
            "eval/corpus/pg84-frankenstein.txt",
            "pg84-frankenstein.txt"
        ));
        assert!(!source_matches(
            "eval/corpus/pg845-x.txt",
            "pg84-frankenstein.txt"
        ));
        assert!(!source_matches("", "pg84-frankenstein.txt"));
    }

    #[test]
    fn first_hit_rank_is_one_based() {
        assert_eq!(
            first_hit_rank(&["a.txt", "b.txt", "b.txt"], "b.txt"),
            Some(2)
        );
        assert_eq!(first_hit_rank(&["a.txt"], "z.txt"), None);
        assert_eq!(first_hit_rank(&[], "z.txt"), None);
    }

    #[test]
    fn recall_at_k_counts_ranks_within_k() {
        let ranks = [Some(1), Some(3), Some(7), None];
        assert!((recall_at_k(&ranks, 1) - 0.25).abs() < 1e-9);
        assert!((recall_at_k(&ranks, 5) - 0.5).abs() < 1e-9);
        assert!((recall_at_k(&ranks, 10) - 0.75).abs() < 1e-9);
        assert_eq!(recall_at_k(&[], 5), 0.0);
    }

    #[test]
    fn fraction_of_hits() {
        assert!((fraction(&[true, false, true, true]) - 0.75).abs() < 1e-9);
        assert_eq!(fraction(&[]), 0.0);
    }

    #[test]
    fn mode_report_from_outcomes() {
        let o = |rank, leaf_hit, consulted, fallback| QuestionOutcome {
            rank,
            leaf_hit,
            laya_consulted: consulted,
            laya_fallback: fallback,
        };
        let r = ModeReport::from_outcomes(
            "hybrid",
            &[
                o(Some(1), Some(true), true, false),
                o(None, Some(false), true, true),
            ],
        );
        assert_eq!(r.n, 2);
        assert!((r.recall_at_1 - 0.5).abs() < 1e-9);
        assert_eq!(r.leaf_accuracy, Some(0.5));
        assert_eq!((r.laya_consulted, r.laya_fallbacks), (2, 1));
        let flat = ModeReport::from_outcomes("flat", &[o(Some(6), None, false, false)]);
        assert_eq!(flat.leaf_accuracy, None);
        assert!((flat.recall_at_5 - 0.0).abs() < 1e-9);
        assert!((flat.recall_at_10 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn render_report_rows_and_verdict() {
        let mk = |mode, r5, leaf| ModeReport {
            mode,
            n: 2,
            recall_at_1: 0.5,
            recall_at_5: r5,
            recall_at_10: 1.0,
            leaf_accuracy: leaf,
            laya_consulted: 0,
            laya_fallbacks: 0,
        };
        let reports = [
            mk("embed", 1.0, Some(1.0)),
            mk("hybrid", 0.5, Some(0.5)),
            mk("flat", 0.75, None),
        ];
        let out = render_report(&reports, false);
        // a contaminated run never prints a bare verdict
        let dirty = render_report(&reports, true);
        assert!(
            dirty.contains("embed recall@5 1.000 >= flat 0.750: CONTAMINATED (would be PASS"),
            "{dirty}"
        );
        assert!(
            !dirty.contains(": PASS") && !dirty.contains(": FAIL"),
            "{dirty}"
        );
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[1].split_whitespace().collect::<Vec<_>>(),
            ["embed", "2", "0.500", "1.000", "1.000", "1.000", "0", "0"]
        );
        assert_eq!(
            lines[3].split_whitespace().collect::<Vec<_>>(),
            ["flat", "2", "0.500", "0.750", "1.000", "-", "0", "0"]
        );
        assert!(
            out.contains("embed recall@5 1.000 >= flat 0.750: PASS"),
            "{out}"
        );
        assert!(
            out.contains("hybrid recall@5 0.500 >= flat 0.750: FAIL"),
            "{out}"
        );
    }

    #[test]
    fn verdict_fails_when_a_routed_mode_trails_flat() {
        let mk = |mode, r5| ModeReport {
            mode,
            n: 2,
            recall_at_1: 0.0,
            recall_at_5: r5,
            recall_at_10: 1.0,
            leaf_accuracy: None,
            laya_consulted: 0,
            laya_fallbacks: 0,
        };
        assert!(verdict_passes(&[
            mk("embed", 0.8),
            mk("hybrid", 0.75),
            mk("flat", 0.75)
        ]));
        assert!(!verdict_passes(&[
            mk("embed", 1.0),
            mk("hybrid", 0.5),
            mk("flat", 0.75)
        ]));
        assert!(
            verdict_passes(&[mk("embed", 0.1)]),
            "no flat row: no verdict"
        );
    }

    #[test]
    fn resolve_tree_path_prefers_tree_then_load_then_sibling() {
        let q = Path::new("eval/questions.jsonl");
        assert_eq!(
            resolve_tree_path(q, Some(Path::new("t.json")), Some(Path::new("l.json"))),
            Path::new("t.json")
        );
        assert_eq!(
            resolve_tree_path(q, None, Some(Path::new("l.json"))),
            Path::new("l.json")
        );
        assert_eq!(
            resolve_tree_path(q, None, None),
            Path::new("eval/tree.json")
        );
    }

    #[test]
    fn committed_eval_files_are_consistent() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let tree: TreeSpec =
            serde_json::from_str(&std::fs::read_to_string(root.join("eval/tree.json")).unwrap())
                .unwrap();
        let problems = check_tree(&tree);
        assert!(problems.is_empty(), "{problems:?}");
        let flat = flatten_tree(&tree);
        let routers = flat
            .iter()
            .filter(|n| n.kind == StoreKind::Router && n.path.len() == 2)
            .count();
        let leaves: Vec<&FlatNode> = flat.iter().filter(|n| n.kind == StoreKind::Leaf).collect();
        assert_eq!(routers, 3, "three second-layer routers");
        assert!(leaves.len() >= 7, "at least 7 leaves");

        let qs =
            parse_questions(&std::fs::read_to_string(root.join("eval/questions.jsonl")).unwrap())
                .unwrap();
        assert!(qs.len() >= 50, "need >= 50 questions, have {}", qs.len());
        let problems = check_questions_against_tree(&qs, &tree);
        assert!(problems.is_empty(), "{problems:?}");
        for leaf in &leaves {
            for file in &leaf.files {
                let n = qs.iter().filter(|q| &q.expected_document == file).count();
                assert!(n >= 3, "{file} has {n} questions, need >= 3");
            }
        }
    }
}
