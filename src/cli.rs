//! `strata` command line. Mirrors the HTTP API against the same library code (no HTTP hop).

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::Read as _;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use uuid::Uuid;

use crate::ask::AskRequest;
use crate::config::Config;
use crate::db::stores::{NewStore, StoreKind};
use crate::engine::{Engine, StoreView};
use crate::error::{Result, StrataError};
use crate::ingest::{IngestOutcome, IngestRequest, PruneRequest, SourceTag};
use crate::query::{QueryRequest, QueryResponse};
use crate::routing::RouteMode;
use crate::tenant::Tenant;

#[derive(Debug, Parser)]
#[command(
    name = "strata",
    version,
    about = "Small, fast, multi-layer RAG over a tree of stores"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// The tenant every engine command acts for (spec §8.6): it sees and changes only that tenant's
/// stores, documents and chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::Args)]
pub struct TenantArg {
    /// Tenant uuid the command acts for.
    #[arg(long, env = "STRATA_TENANT", value_parser = parse_tenant)]
    pub tenant: Tenant,
}

fn parse_tenant(s: &str) -> std::result::Result<Tenant, String> {
    Tenant::parse(s).map_err(|_| "expected a tenant uuid (not the nil uuid)".to_string())
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the HTTP API on STRATA_BIND (default 127.0.0.1:8092).
    Serve,
    /// Create, list and delete stores.
    #[command(subcommand)]
    Store(StoreCommand),
    /// Ingest a file, a directory (.txt/.md, recursive) or stdin (`-`) into a leaf.
    Ingest {
        #[command(flatten)]
        tenant: TenantArg,
        /// Leaf id (uuid) or name path from the root, e.g. `library/fiction/sea-adventure`.
        leaf: String,
        /// File, directory, or `-` for stdin.
        path: String,
        /// Document source label (single file / stdin only; directories use each file path).
        #[arg(long)]
        source: Option<String>,
        /// Document title (single file / stdin only).
        #[arg(long)]
        title: Option<String>,
        /// Caller-side source the document(s) belong to (every file of a directory).
        #[arg(long)]
        source_id: Option<String>,
        /// Sync run that delivered the document(s); needs --source-id.
        #[arg(long, requires = "source_id")]
        source_run: Option<String>,
        /// Mapped source fields, a JSON object (every file of a directory).
        #[arg(long, value_parser = parse_json)]
        metadata: Option<serde_json::Value>,
    },
    /// Manage a caller-side source's documents.
    #[command(subcommand)]
    Source(SourceCommand),
    /// Route a query down the tree and print reranked chunks.
    Query {
        #[command(flatten)]
        tenant: TenantArg,
        query: String,
        #[arg(long, default_value_t = 8)]
        top_k: usize,
        #[arg(long, value_enum, default_value_t = ModeArg::Hybrid)]
        mode: ModeArg,
        #[arg(long, default_value_t = 3)]
        beam: usize,
        #[arg(long, default_value_t = 0.35)]
        min_score: f32,
        /// Expand the query (HyDE passage + rewrites via the chat provider) for the dense search.
        #[arg(long)]
        expand: bool,
        /// Only chunks of documents from these sources (repeatable).
        #[arg(long = "source-id")]
        source_ids: Vec<String>,
        /// Only chunks of documents whose metadata contains this JSON object.
        #[arg(long, value_parser = parse_json)]
        metadata: Option<serde_json::Value>,
        /// Print the full QueryResponse as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Retrieve, then generate an answer with the chat provider (a registration or `[default.chat]`).
    Ask {
        #[command(flatten)]
        tenant: TenantArg,
        query: String,
        #[arg(long, default_value_t = 800)]
        max_tokens: u32,
    },
    /// Re-embed the tenant's whole tree with its current embed provider (after a provider
    /// switch, which leaves the tree refused with 409 embed_model_mismatch), in one transaction.
    Reembed {
        #[command(flatten)]
        tenant: TenantArg,
        /// The tenant's root store: uuid or name.
        #[arg(long)]
        store: String,
        /// Chunk texts per embed call and rows per UPDATE.
        #[arg(long, default_value_t = crate::reembed::DEFAULT_REEMBED_BATCH)]
        batch: usize,
    },
    /// Query latency p50/p95 per stage and ingest throughput (chunks/s).
    Bench {
        #[command(flatten)]
        tenant: TenantArg,
        /// JSONL file whose `query` fields are cycled through.
        #[arg(long, default_value = "eval/questions.jsonl")]
        queries: PathBuf,
        #[arg(long, default_value_t = 100)]
        n: usize,
        #[arg(long, value_enum, default_value_t = ModeArg::Embed)]
        mode: ModeArg,
        #[arg(long, default_value = "eval/corpus")]
        ingest_dir: PathBuf,
        /// How many corpus files to ingest into the temporary bench leaf.
        #[arg(long, default_value_t = 2)]
        ingest_files: usize,
        #[arg(long)]
        skip_ingest: bool,
    },
    /// Recall@1/5/10 + leaf routing accuracy for embed, hybrid and flat; `--load` builds the tree first.
    Eval {
        #[command(flatten)]
        tenant: TenantArg,
        questions: Option<PathBuf>,
        #[arg(long, value_name = "TREE_JSON")]
        load: Option<PathBuf>,
        /// Tree the questions are scored against (flat mode searches only its leaves).
        /// Default: --load, else tree.json next to the questions file.
        #[arg(long, value_name = "TREE_JSON")]
        tree: Option<PathBuf>,
        #[arg(long)]
        json: bool,
        /// Report even when the database holds stores outside the evaluated tree (they change
        /// embed/hybrid routing); the report is then marked CONTAMINATED. Default: refuse.
        #[arg(long)]
        allow_foreign_stores: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum StoreCommand {
    /// Create a store. The tenant's first store (no --parent) is its root.
    Add {
        #[command(flatten)]
        tenant: TenantArg,
        #[arg(long, value_enum)]
        kind: KindArg,
        #[arg(long)]
        name: String,
        #[arg(long)]
        description: String,
        /// Parent store: uuid or name path from the root.
        #[arg(long)]
        parent: Option<String>,
    },
    /// Print the tenant's store tree with token and chunk counts.
    Ls {
        #[command(flatten)]
        tenant: TenantArg,
    },
    /// Delete a store (uuid or name path).
    Rm {
        #[command(flatten)]
        tenant: TenantArg,
        id: String,
        /// Also delete all descendants.
        #[arg(long)]
        recursive: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum SourceCommand {
    /// Delete a source's documents, except those of `--except-run` (the re-sync step after
    /// ingesting every document with that run), or all of them with `--all`. A leaf where no
    /// document carries the run is left untouched and the command fails.
    #[command(group(clap::ArgGroup::new("keep").required(true).args(["except_run", "all"])))]
    Prune {
        #[command(flatten)]
        tenant: TenantArg,
        source_id: String,
        /// Keep the documents of this run.
        #[arg(long)]
        except_run: Option<String>,
        /// Delete every document of the source.
        #[arg(long)]
        all: bool,
        /// Only this leaf (uuid or name path).
        #[arg(long)]
        store: Option<String>,
        /// Prune even when no document carries --except-run.
        #[arg(long)]
        allow_empty: bool,
    },
}

/// `--metadata`: any JSON (the object check is the engine's, so the CLI and API agree).
fn parse_json(s: &str) -> std::result::Result<serde_json::Value, String> {
    serde_json::from_str(s).map_err(|e| format!("not valid JSON: {e}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModeArg {
    Embed,
    Laya,
    Hybrid,
}

impl From<ModeArg> for RouteMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Embed => RouteMode::Embed,
            ModeArg::Laya => RouteMode::Laya,
            ModeArg::Hybrid => RouteMode::Hybrid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum KindArg {
    Router,
    Leaf,
}

impl From<KindArg> for StoreKind {
    fn from(k: KindArg) -> Self {
        match k {
            KindArg::Router => StoreKind::Router,
            KindArg::Leaf => StoreKind::Leaf,
        }
    }
}

/// Logs go to stderr (stdout stays clean for `--json`). Filter from RUST_LOG, default `strata_rag=info`.
pub fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("strata_rag=info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

pub async fn run(cli: Cli) -> Result<ExitCode> {
    let cfg = Config::from_env()?;
    if matches!(cli.command, Command::Serve) {
        // Fail before connecting to the database or loading the tokenizer.
        cfg.check_serve_auth()?;
        check_port_free(cfg.bind)?;
    }
    let engine = Arc::new(Engine::from_config(cfg).await?);
    run_command(engine, cli.command).await
}

/// Runs one command on an already-built engine, then flushes its queued cost events to the
/// outbox. `run` is this plus config and engine construction; tests drive this so the exit flush
/// is exercised on the real path (`tests/metering_redis.rs`), not by calling `flush_metering`
/// themselves. `strata serve` takes the same path once the server has shut down gracefully.
pub async fn run_command(engine: Arc<Engine>, command: Command) -> Result<ExitCode> {
    let out = dispatch(Arc::clone(&engine), command).await;
    // success or not, the command's provider calls were metered: record them before the process
    // exits (the drainer of a serving replica delivers them)
    flush_metering(&engine).await;
    out
}

/// `strata serve`'s outbox drainer: running when metering is on (`METERING_REDIS_URL`). Only a
/// serving process takes recorded cost events to Redis; a CLI command's rows wait for one.
pub fn start_drainer(engine: &Engine) -> Result<Option<crate::metering::DrainerHandle>> {
    let (Some(_), Some(url)) = (&engine.metering, &engine.cfg.metering_redis_url) else {
        return Ok(None);
    };
    let drainer = crate::metering::Drainer::new(
        engine.pool.clone(),
        url.expose(),
        &engine.cfg.metering_stream,
        crate::metering::DrainerConfig::default(),
    )?;
    Ok(Some(drainer.spawn()))
}

/// `strata serve`'s metering shutdown, after the server stopped taking requests: FIRST flush the
/// writer's queue to the outbox (at most `METERING_FLUSH_TIMEOUT`), THEN stop the drainer, whose
/// last pass sends what the flush wrote (at most `metering::DRAINER_STOP_TIMEOUT`). Both bounds
/// together (20 s) fit the pod's 30 s `terminationGracePeriodSeconds`; whatever the last pass
/// does not send stays in the outbox for the next replica.
pub async fn stop_metering(engine: &Engine, drainer: Option<crate::metering::DrainerHandle>) {
    flush_metering(engine).await;
    if let Some(d) = drainer {
        d.stop().await;
    }
}

/// Longest a process waits on exit for its queued cost events to reach `metering_outbox`.
pub const METERING_FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Writes the engine's queued cost events to the outbox (bounded by `METERING_FLUSH_TIMEOUT`);
/// returns how many were still not written when it gave up (logged). 0 when metering is off.
pub async fn flush_metering(engine: &Engine) -> usize {
    let Some(m) = &engine.metering else {
        return 0;
    };
    let left = m.flush(METERING_FLUSH_TIMEOUT).await;
    if left > 0 {
        tracing::warn!(
            left,
            "metering: cost events not yet in the outbox at exit were lost"
        );
    }
    left
}

async fn dispatch(engine: Arc<Engine>, command: Command) -> Result<ExitCode> {
    match command {
        Command::Serve => {
            engine.spawn_laya_warmup();
            let drainer = start_drainer(&engine)?;
            let served = crate::api::serve(Arc::clone(&engine)).await;
            stop_metering(&engine, drainer).await;
            served?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Store(StoreCommand::Add {
            tenant: TenantArg { tenant },
            kind,
            name,
            description,
            parent,
        }) => {
            let parent_id = match parent {
                Some(r) => Some(resolve_store_ref(&engine.store_tree(tenant).await?, &r)?),
                None => None,
            };
            let view = engine
                .create_store(
                    tenant,
                    NewStore {
                        parent_id,
                        kind: kind.into(),
                        name,
                        description,
                    },
                )
                .await?;
            println!("created {} {} {}", view.kind.as_str(), view.name, view.id);
            Ok(ExitCode::SUCCESS)
        }
        Command::Store(StoreCommand::Ls {
            tenant: TenantArg { tenant },
        }) => {
            print!("{}", render_tree(&engine.store_tree(tenant).await?));
            Ok(ExitCode::SUCCESS)
        }
        Command::Store(StoreCommand::Rm {
            tenant: TenantArg { tenant },
            id,
            recursive,
        }) => {
            let id = resolve_store_ref(&engine.store_tree(tenant).await?, &id)?;
            engine.delete_store(tenant, id, recursive).await?;
            println!("deleted {id}");
            Ok(ExitCode::SUCCESS)
        }
        Command::Ingest {
            tenant: TenantArg { tenant },
            leaf,
            path,
            source,
            title,
            source_id,
            source_run,
            metadata,
        } => {
            let tag = SourceTag {
                source_id,
                source_run,
                metadata,
            };
            cmd_ingest(&engine, tenant, &leaf, &path, source, title, tag).await
        }
        Command::Source(SourceCommand::Prune {
            tenant: TenantArg { tenant },
            source_id,
            except_run,
            all,
            store,
            allow_empty,
        }) => {
            let store_id = match store {
                Some(r) => Some(resolve_store_ref(&engine.store_tree(tenant).await?, &r)?),
                None => None,
            };
            let report = engine
                .prune_source(
                    tenant,
                    PruneRequest {
                        source_id,
                        except_run,
                        all,
                        store_id,
                        allow_empty,
                    },
                )
                .await?;
            let stores: Vec<String> = report.stores.iter().map(Uuid::to_string).collect();
            println!(
                "deleted {} kept {} stores=[{}]",
                report.deleted,
                report.kept,
                stores.join(",")
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::Query {
            tenant: TenantArg { tenant },
            query,
            top_k,
            mode,
            beam,
            min_score,
            expand,
            source_ids,
            metadata,
            json,
        } => {
            let req = QueryRequest {
                query,
                top_k,
                route_mode: mode.into(),
                beam,
                min_score,
                root_id: None,
                root_ids: None,
                leaf_ids: None,
                expand,
                source_ids: (!source_ids.is_empty()).then_some(source_ids),
                metadata,
                supersedes: None,
                platform_penalty: None,
                platform_min_score: None,
            };
            let resp = engine.query(tenant, &req).await?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&resp).expect("QueryResponse serialises")
                );
            } else {
                print!("{}", render_query(&resp));
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Reembed {
            tenant: TenantArg { tenant },
            store,
            batch,
        } => {
            let root = resolve_store_ref(&engine.store_tree(tenant).await?, &store)?;
            let report = engine.reembed_tree(tenant, root, batch).await?;
            println!(
                "re-embedded root={} stores={} chunks={} embed_model={}",
                report.root, report.stores, report.chunks, report.embed_model
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::Bench {
            tenant: TenantArg { tenant },
            queries,
            n,
            mode,
            ingest_dir,
            ingest_files,
            skip_ingest,
        } => {
            let opts = crate::bench::BenchOptions {
                queries,
                n,
                mode: mode.into(),
                ingest_dir,
                ingest_files,
                skip_ingest,
            };
            crate::bench::run(&engine, tenant, &opts).await
        }
        Command::Eval {
            tenant: TenantArg { tenant },
            questions,
            load,
            tree,
            json,
            allow_foreign_stores,
        } => {
            crate::eval::run(
                &engine,
                tenant,
                questions.as_deref(),
                load.as_deref(),
                tree.as_deref(),
                json,
                allow_foreign_stores,
            )
            .await
        }
        Command::Ask {
            tenant: TenantArg { tenant },
            query,
            max_tokens,
        } => {
            let req = AskRequest {
                query: QueryRequest::new(query),
                max_tokens,
                instructions: None,
            };
            let resp = engine.ask(tenant, &req).await?;
            println!("{}\n", resp.answer.trim());
            println!("sources:");
            for d in &resp.documents {
                let cited = if resp.citations.iter().any(|c| d.chunk_ids.contains(c)) {
                    "*"
                } else {
                    " "
                };
                println!(
                    "{cited}[{}] {} document={} chunks={}",
                    d.number,
                    d.store_path.join("/"),
                    d.document_id,
                    d.chunk_ids.len()
                );
            }
            println!(
                "generate_ms={} total_ms={}",
                resp.timings_ms.generate, resp.timings_ms.query.total
            );
            Ok(ExitCode::SUCCESS)
        }
    }
}

async fn read_utf8(path: &Path) -> Result<String> {
    let bytes = tokio::fs::read(path).await?;
    String::from_utf8(bytes)
        .map_err(|_| StrataError::InvalidInput(format!("{} is not valid UTF-8", path.display())))
}

fn describe_outcome(out: &IngestOutcome, label: &str) -> String {
    match out {
        IngestOutcome::Created(d) => format!(
            "created  {} chunks={} tokens={} {label}",
            d.id, d.chunk_count, d.token_count
        ),
        IngestOutcome::Existing(d) => format!("existing {} {label}", d.id),
    }
}

async fn cmd_ingest(
    engine: &Engine,
    tenant: Tenant,
    leaf: &str,
    path: &str,
    source: Option<String>,
    title: Option<String>,
    tag: SourceTag,
) -> Result<ExitCode> {
    // fail before reading anything (a directory would otherwise report it once per file)
    tag.clone().validate()?;
    let leaf_id = resolve_store_ref(&engine.store_tree(tenant).await?, leaf)?;
    let one = |source: String, title: Option<String>, text: String, tag: SourceTag| async move {
        engine
            .ingest_document(
                tenant,
                IngestRequest {
                    store_id: leaf_id,
                    source,
                    title,
                    text: &text,
                    tag,
                },
            )
            .await
    };
    if path == "-" {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        let text = String::from_utf8(buf)
            .map_err(|_| StrataError::InvalidInput("stdin is not valid UTF-8".into()))?;
        let out = one(source.unwrap_or_else(|| "stdin".into()), title, text, tag).await?;
        println!("{}", describe_outcome(&out, "stdin"));
        return Ok(ExitCode::SUCCESS);
    }
    let p = Path::new(path);
    if !p.is_dir() {
        let text = read_utf8(p).await?;
        let out = one(source.unwrap_or_else(|| path.to_string()), title, text, tag).await?;
        println!("{}", describe_outcome(&out, path));
        return Ok(ExitCode::SUCCESS);
    }
    let files = collect_text_files(p)?;
    if files.is_empty() {
        return Err(StrataError::InvalidInput(format!(
            "no .txt/.md files under {path}"
        )));
    }
    let (mut ok, mut failed) = (0_usize, 0_usize);
    for f in &files {
        let label = f.display().to_string();
        let res = match read_utf8(f).await {
            Ok(text) => one(label.clone(), None, text, tag.clone()).await,
            Err(e) => Err(e),
        };
        match res {
            Ok(out) => {
                ok += 1;
                println!("{}", describe_outcome(&out, &label));
            }
            Err(e) => {
                failed += 1;
                eprintln!("FAILED {label}: {e}");
            }
        }
    }
    println!("{} files: {ok} ok, {failed} failed", files.len());
    Ok(if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

/// Indented tree, children sorted by name; stores whose parent is absent print at top level.
pub fn render_tree(views: &[StoreView]) -> String {
    if views.is_empty() {
        return "(no stores)\n".to_string();
    }
    let ids: HashSet<Uuid> = views.iter().map(|v| v.id).collect();
    let mut by_parent: HashMap<Option<Uuid>, Vec<&StoreView>> = HashMap::new();
    for v in views {
        let key = v.parent_id.filter(|p| ids.contains(p));
        by_parent.entry(key).or_default().push(v);
    }
    for list in by_parent.values_mut() {
        list.sort_by(|a, b| a.name.cmp(&b.name));
    }
    let mut out = String::new();
    render_level(&mut out, &by_parent, None, 0);
    out
}

fn render_level(
    out: &mut String,
    by_parent: &HashMap<Option<Uuid>, Vec<&StoreView>>,
    parent: Option<Uuid>,
    depth: usize,
) {
    let Some(children) = by_parent.get(&parent) else {
        return;
    };
    for v in children {
        let _ = writeln!(
            out,
            "{}{} [{}] tokens={} chunks={} id={}",
            "  ".repeat(depth),
            v.name,
            v.kind.as_str(),
            v.token_count,
            v.chunk_count,
            v.id
        );
        render_level(out, by_parent, Some(v.id), depth + 1);
    }
}

/// A uuid, or a `/`-separated name path starting at the root (`library/science/physics`).
pub fn resolve_store_ref(views: &[StoreView], reference: &str) -> Result<Uuid> {
    if let Ok(id) = Uuid::parse_str(reference) {
        return if views.iter().any(|v| v.id == id) {
            Ok(id)
        } else {
            Err(StrataError::NotFound(format!("store {id}")))
        };
    }
    let mut parent: Option<Uuid> = None;
    let mut found: Option<Uuid> = None;
    for seg in reference.split('/').filter(|s| !s.is_empty()) {
        let v = views
            .iter()
            .find(|v| v.parent_id == parent && v.name == seg)
            .ok_or_else(|| {
                StrataError::NotFound(format!("store path `{reference}`: no `{seg}`"))
            })?;
        parent = Some(v.id);
        found = Some(v.id);
    }
    found.ok_or_else(|| StrataError::InvalidInput(format!("empty store reference `{reference}`")))
}

/// Every `.txt` / `.md` (case-insensitive) file under `dir`, recursively, sorted. Symlinks skipped.
pub fn collect_text_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d)? {
            let entry = entry?;
            let path = entry.path();
            let ft = entry.file_type()?;
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() {
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_ascii_lowercase);
                if matches!(ext.as_deref(), Some("txt" | "md")) {
                    out.push(path);
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Bind-and-release probe so `strata serve` names the busy port before building the engine.
pub fn check_port_free(addr: SocketAddr) -> Result<()> {
    match std::net::TcpListener::bind(addr) {
        Ok(listener) => {
            drop(listener);
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Err(StrataError::Config(format!(
            "port {} on {} is already in use; set STRATA_BIND to a free port (never 8000)",
            addr.port(),
            addr.ip()
        ))),
        Err(e) => Err(e.into()),
    }
}

pub fn snippet(text: &str, max_chars: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        flat
    } else {
        let cut: String = flat.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

pub fn render_query(resp: &QueryResponse) -> String {
    let mut out = String::new();
    let fallback = resp
        .routing
        .laya_fallback
        .as_deref()
        .map(|r| format!(" laya_fallback={r}"))
        .unwrap_or_default();
    let _ = writeln!(
        out,
        "routing: mode={} leaves={} laya_consulted={}{fallback}",
        resp.routing
            .mode_used
            .map_or_else(|| "none".to_string(), |m| format!("{m:?}")),
        resp.routing.leaves.len(),
        resp.routing.laya_consulted
    );
    let t = resp.timings_ms;
    let _ = writeln!(
        out,
        "timings ms: embed={} route={} search={} rerank={} total={}",
        t.embed, t.route, t.search, t.rerank, t.total
    );
    if resp.results.is_empty() {
        out.push_str("no results\n");
    }
    for (i, r) in resp.results.iter().enumerate() {
        // `None` = outside the rerank budget (not reranked), printed as `-`.
        let rerank = r
            .scores
            .rerank
            .map_or_else(|| "-".to_string(), |s| format!("{s:.3}"));
        let _ = writeln!(
            out,
            "[{}] rerank={rerank} {} doc={}",
            i + 1,
            r.store_path.join("/"),
            r.document_id
        );
        let _ = writeln!(out, "    {}", snippet(&r.text, 240));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    use crate::query::{QueryResult, RoutingInfo, Timings};
    use crate::search::Scores;

    const TENANT: &str = "0192f0c8-0000-7000-8000-0000000000aa";

    /// Parses `args`, with `--tenant TENANT` appended to every command but `serve`.
    fn parse(args: &[&str]) -> Cli {
        let mut args = args.to_vec();
        if args.get(1) != Some(&"serve") {
            args.extend(["--tenant", TENANT]);
        }
        Cli::try_parse_from(&args).unwrap_or_else(|e| panic!("{args:?}: {e}"))
    }

    fn tenant() -> Tenant {
        Tenant::parse(TENANT).unwrap()
    }

    fn view(
        id: u128,
        parent: Option<u128>,
        kind: StoreKind,
        name: &str,
        tokens: i64,
        chunks: i64,
    ) -> StoreView {
        StoreView {
            id: Uuid::from_u128(id),
            parent_id: parent.map(Uuid::from_u128),
            kind,
            name: name.to_string(),
            description: format!("{name} description"),
            token_count: tokens,
            chunk_count: chunks,
            children: Vec::new(),
        }
    }

    fn sample_tree() -> Vec<StoreView> {
        vec![
            view(3, Some(2), StoreKind::Leaf, "physics", 200, 6),
            view(1, None, StoreKind::Router, "library", 300, 9),
            view(4, Some(1), StoreKind::Leaf, "art", 100, 3),
            view(2, Some(1), StoreKind::Router, "science", 200, 6),
        ]
    }

    /// Every command that reaches the engine needs a tenant; `serve` takes it per request instead.
    /// (`STRATA_TENANT` is the env twin of `--tenant`; clap reads the process env, so these
    /// parses assume it is unset, as it is in CI.)
    #[test]
    fn engine_commands_require_a_valid_tenant() {
        if std::env::var_os("STRATA_TENANT").is_some() {
            eprintln!("STRATA_TENANT is set; skipping the missing-tenant half");
        } else {
            for args in [
                &["strata", "store", "ls"][..],
                &["strata", "store", "rm", "x"],
                &[
                    "strata",
                    "store",
                    "add",
                    "--kind",
                    "leaf",
                    "--name",
                    "x",
                    "--description",
                    "y",
                ],
                &["strata", "ingest", "l", "-"],
                &["strata", "source", "prune", "s", "--all"],
                &["strata", "query", "q"],
                &["strata", "ask", "q"],
                &["strata", "bench"],
                &["strata", "reembed", "--store", "root"],
                &["strata", "eval", "--load", "t.json"],
            ] {
                let err = Cli::try_parse_from(args).expect_err(&format!("{args:?}"));
                assert!(err.to_string().contains("--tenant"), "{args:?}: {err}");
            }
        }
        for bad in ["nope", "00000000-0000-0000-0000-000000000000"] {
            assert!(
                Cli::try_parse_from(["strata", "store", "ls", "--tenant", bad]).is_err(),
                "{bad}"
            );
        }
        match parse(&["strata", "store", "ls"]).command {
            Command::Store(StoreCommand::Ls {
                tenant: TenantArg { tenant: t },
            }) => assert_eq!(t, tenant()),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn query_filter_flags_parse() {
        match parse(&[
            "strata",
            "query",
            "q",
            "--source-id",
            "a",
            "--source-id",
            "b",
            "--metadata",
            r#"{"lang":"en"}"#,
        ])
        .command
        {
            Command::Query {
                source_ids,
                metadata,
                ..
            } => {
                assert_eq!(source_ids, ["a", "b"]);
                assert_eq!(metadata, Some(serde_json::json!({"lang": "en"})));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn clap_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_serve() {
        assert!(matches!(
            parse(&["strata", "serve"]).command,
            Command::Serve
        ));
    }

    #[test]
    fn parses_store_add_with_parent_path() {
        let cli = parse(&[
            "strata",
            "store",
            "add",
            "--kind",
            "leaf",
            "--name",
            "physics",
            "--description",
            "relativity and light",
            "--parent",
            "library/science",
        ]);
        match cli.command {
            Command::Store(StoreCommand::Add {
                tenant: TenantArg { tenant: t },
                kind,
                name,
                description,
                parent,
            }) => {
                assert_eq!(t, tenant());
                assert_eq!(kind, KindArg::Leaf);
                assert_eq!(name, "physics");
                assert_eq!(description, "relativity and light");
                assert_eq!(parent.as_deref(), Some("library/science"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn store_add_requires_kind() {
        assert!(
            Cli::try_parse_from([
                "strata",
                "store",
                "add",
                "--name",
                "x",
                "--description",
                "y",
                "--tenant",
                TENANT
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_store_ls_and_rm() {
        assert!(matches!(
            parse(&["strata", "store", "ls"]).command,
            Command::Store(StoreCommand::Ls { .. })
        ));
        match parse(&["strata", "store", "rm", "library", "--recursive"]).command {
            Command::Store(StoreCommand::Rm {
                tenant: TenantArg { tenant: t },
                id,
                recursive,
            }) => {
                assert_eq!(t, tenant());
                assert_eq!(id, "library");
                assert!(recursive);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_ingest_stdin_with_source_and_title() {
        match parse(&[
            "strata",
            "ingest",
            "library/sea",
            "-",
            "--source",
            "notes",
            "--title",
            "Notes",
        ])
        .command
        {
            Command::Ingest {
                tenant: TenantArg { tenant: t },
                leaf,
                path,
                source,
                title,
                source_id,
                source_run,
                metadata,
            } => {
                assert_eq!(t, tenant());
                assert_eq!(leaf, "library/sea");
                assert_eq!(path, "-");
                assert_eq!(source.as_deref(), Some("notes"));
                assert_eq!(title.as_deref(), Some("Notes"));
                assert_eq!((source_id, source_run, metadata), (None, None, None));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_ingest_source_fields() {
        match parse(&[
            "strata",
            "ingest",
            "library/sea",
            "doc.md",
            "--source-id",
            "web-1",
            "--source-run",
            "run-7",
            "--metadata",
            r#"{"url":"https://x.test/a","lang":"en"}"#,
        ])
        .command
        {
            Command::Ingest {
                source_id,
                source_run,
                metadata,
                ..
            } => {
                assert_eq!(source_id.as_deref(), Some("web-1"));
                assert_eq!(source_run.as_deref(), Some("run-7"));
                assert_eq!(
                    metadata,
                    Some(serde_json::json!({"url":"https://x.test/a","lang":"en"}))
                );
            }
            other => panic!("unexpected {other:?}"),
        }
        // a run without a source, and metadata that is not JSON, are usage errors
        let t = ["--tenant", TENANT];
        assert!(
            Cli::try_parse_from([
                "strata",
                "ingest",
                "l",
                "-",
                "--source-run",
                "r",
                t[0],
                t[1]
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from(["strata", "ingest", "l", "-", "--metadata", "{x", t[0], t[1]])
                .is_err()
        );
    }

    #[test]
    fn parses_source_prune() {
        match parse(&["strata", "source", "prune", "web-1", "--all"]).command {
            Command::Source(SourceCommand::Prune {
                tenant: TenantArg { tenant: t },
                source_id,
                except_run,
                all,
                store,
                allow_empty,
            }) => {
                assert_eq!(t, tenant());
                assert_eq!(source_id, "web-1");
                assert!(all);
                assert_eq!((except_run, store, allow_empty), (None, None, false));
            }
            other => panic!("unexpected {other:?}"),
        }
        // a whole-source wipe must be explicit, and cannot also keep a run
        assert!(
            Cli::try_parse_from(["strata", "source", "prune", "web-1", "--tenant", TENANT])
                .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "strata",
                "source",
                "prune",
                "web-1",
                "--all",
                "--except-run",
                "r",
                "--tenant",
                TENANT
            ])
            .is_err()
        );
        match parse(&[
            "strata",
            "source",
            "prune",
            "web-1",
            "--except-run",
            "run-7",
            "--store",
            "library/sea",
            "--allow-empty",
        ])
        .command
        {
            Command::Source(SourceCommand::Prune {
                tenant: TenantArg { tenant: t },
                source_id,
                except_run,
                all,
                store,
                allow_empty,
            }) => {
                assert_eq!(t, tenant());
                assert_eq!(source_id, "web-1");
                assert!(!all);
                assert_eq!(except_run.as_deref(), Some("run-7"));
                assert_eq!(store.as_deref(), Some("library/sea"));
                assert!(allow_empty);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(Cli::try_parse_from(["strata", "source", "prune", "--tenant", TENANT]).is_err());
    }

    #[test]
    fn query_defaults_match_contract() {
        match parse(&["strata", "query", "what is light"]).command {
            Command::Query {
                tenant: TenantArg { tenant: t },
                query,
                top_k,
                mode,
                beam,
                min_score,
                expand,
                source_ids,
                metadata,
                json,
            } => {
                assert_eq!(t, tenant());
                assert!(!expand, "expansion is opt-in");
                assert!(
                    source_ids.is_empty() && metadata.is_none(),
                    "no filters by default"
                );
                assert_eq!(query, "what is light");
                assert_eq!(top_k, 8);
                assert_eq!(mode, ModeArg::Hybrid);
                assert_eq!(beam, 3);
                assert!((min_score - 0.35).abs() < f32::EPSILON);
                assert!(!json);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn query_flags_parse() {
        match parse(&[
            "strata",
            "query",
            "q",
            "--top-k",
            "5",
            "--mode",
            "embed",
            "--beam",
            "2",
            "--min-score",
            "0.5",
            "--json",
        ])
        .command
        {
            Command::Query {
                top_k,
                mode,
                beam,
                min_score,
                json,
                ..
            } => {
                assert_eq!((top_k, mode, beam, json), (5, ModeArg::Embed, 2, true));
                assert!((min_score - 0.5).abs() < f32::EPSILON);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(
            Cli::try_parse_from([
                "strata",
                "query",
                "q",
                "--mode",
                "telepathy",
                "--tenant",
                TENANT
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_ask_max_tokens() {
        match parse(&["strata", "ask", "why?", "--max-tokens", "300"]).command {
            Command::Ask {
                tenant: TenantArg { tenant: t },
                query,
                max_tokens,
            } => {
                assert_eq!(t, tenant());
                assert_eq!(query, "why?");
                assert_eq!(max_tokens, 300);
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse(&["strata", "ask", "why?"]).command {
            Command::Ask { max_tokens, .. } => assert_eq!(max_tokens, 800),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_reembed() {
        match parse(&["strata", "reembed", "--store", "library", "--batch", "64"]).command {
            Command::Reembed {
                tenant: TenantArg { tenant: t },
                store,
                batch,
            } => {
                assert_eq!(t, tenant());
                assert_eq!(store, "library");
                assert_eq!(batch, 64);
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse(&["strata", "reembed", "--store", "library"]).command {
            Command::Reembed { batch, .. } => {
                assert_eq!(batch, crate::reembed::DEFAULT_REEMBED_BATCH)
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(Cli::try_parse_from(["strata", "reembed", "--tenant", TENANT]).is_err());
    }

    #[test]
    fn mode_and_kind_args_convert() {
        assert_eq!(RouteMode::from(ModeArg::Embed), RouteMode::Embed);
        assert_eq!(RouteMode::from(ModeArg::Laya), RouteMode::Laya);
        assert_eq!(RouteMode::from(ModeArg::Hybrid), RouteMode::Hybrid);
        assert_eq!(StoreKind::from(KindArg::Router), StoreKind::Router);
        assert_eq!(StoreKind::from(KindArg::Leaf), StoreKind::Leaf);
    }

    #[test]
    fn render_tree_indents_children_sorted_by_name() {
        let expected = "\
library [router] tokens=300 chunks=9 id=00000000-0000-0000-0000-000000000001
  art [leaf] tokens=100 chunks=3 id=00000000-0000-0000-0000-000000000004
  science [router] tokens=200 chunks=6 id=00000000-0000-0000-0000-000000000002
    physics [leaf] tokens=200 chunks=6 id=00000000-0000-0000-0000-000000000003
";
        assert_eq!(render_tree(&sample_tree()), expected);
    }

    #[test]
    fn render_tree_empty() {
        assert_eq!(render_tree(&[]), "(no stores)\n");
    }

    #[test]
    fn resolve_store_ref_by_uuid_and_path() {
        let views = sample_tree();
        assert_eq!(
            resolve_store_ref(&views, "library/science/physics").unwrap(),
            Uuid::from_u128(3)
        );
        assert_eq!(
            resolve_store_ref(&views, "/library/art/").unwrap(),
            Uuid::from_u128(4)
        );
        assert_eq!(
            resolve_store_ref(&views, "00000000-0000-0000-0000-000000000002").unwrap(),
            Uuid::from_u128(2)
        );
        assert!(matches!(
            resolve_store_ref(&views, "library/nope"),
            Err(StrataError::NotFound(_))
        ));
        assert!(matches!(
            resolve_store_ref(&views, "00000000-0000-0000-0000-000000000099"),
            Err(StrataError::NotFound(_))
        ));
        assert!(matches!(
            resolve_store_ref(&views, "/"),
            Err(StrataError::InvalidInput(_))
        ));
    }

    #[test]
    fn collect_text_files_walks_recursively_and_filters() {
        let dir = std::env::temp_dir().join(format!("strata-cli-test-{}", Uuid::now_v7()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        for f in ["a.txt", "b.MD", "c.pdf", "sub/d.txt", "sub/e.bin"] {
            std::fs::write(dir.join(f), "x").unwrap();
        }
        let got = collect_text_files(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let rel: Vec<String> = got
            .iter()
            .map(|p| p.strip_prefix(&dir).unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(rel, ["a.txt", "b.MD", "sub/d.txt"]);
    }

    #[test]
    fn check_port_free_reports_the_busy_port() {
        let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = held.local_addr().unwrap();
        let err = check_port_free(addr).unwrap_err().to_string();
        assert!(err.contains(&addr.port().to_string()), "{err}");
        drop(held);
        check_port_free(addr).unwrap();
    }

    #[test]
    fn snippet_flattens_and_truncates() {
        assert_eq!(snippet("a\n\nb   c", 10), "a b c");
        assert_eq!(snippet("abcdefghij", 4), "abcd…");
    }

    fn result(rerank: Option<f32>) -> QueryResult {
        QueryResult {
            chunk_id: Uuid::from_u128(10),
            document_id: Uuid::from_u128(11),
            store_path: vec!["library".to_string(), "art".to_string()],
            layer: crate::precedence::Layer::Tenant,
            pack: None,
            ord: 0,
            text: "line one\nline two".to_string(),
            page_start: None,
            page_end: None,
            source: "src".to_string(),
            title: None,
            scores: Scores {
                dense: Some(0.5),
                lexical: None,
                rerank,
            },
            source_id: None,
            source_run: None,
            metadata: serde_json::Map::new(),
        }
    }

    #[test]
    fn render_query_prints_rerank_or_dash() {
        let resp = QueryResponse {
            results: vec![result(Some(0.98765)), result(None)],
            routing: RoutingInfo {
                mode_used: Some(RouteMode::Hybrid),
                leaves: vec![Uuid::from_u128(4)],
                laya_consulted: true,
                laya_fallback: Some("timeout".to_string()),
            },
            timings_ms: Timings {
                embed: 1,
                route: 2,
                search: 3,
                rerank: 4,
                total: 10,
            },
            providers: crate::query::StageProviders::default(),
            timings_by_model: Vec::new(),
            expansion: None,
        };
        let doc = Uuid::from_u128(11);
        let expected = format!(
            "routing: mode=Hybrid leaves=1 laya_consulted=true laya_fallback=timeout\n\
             timings ms: embed=1 route=2 search=3 rerank=4 total=10\n\
             [1] rerank=0.988 library/art doc={doc}\n    line one line two\n\
             [2] rerank=- library/art doc={doc}\n    line one line two\n"
        );
        assert_eq!(render_query(&resp), expected);
    }

    #[test]
    fn parses_eval_load_only_and_questions() {
        match parse(&["strata", "eval", "--load", "eval/tree.json"]).command {
            Command::Eval {
                tenant: TenantArg { tenant: t },
                questions,
                load,
                tree,
                json,
                allow_foreign_stores,
            } => {
                assert_eq!(t, tenant());
                assert!(questions.is_none());
                assert!(tree.is_none());
                assert!(!allow_foreign_stores, "refusing is the default");
                assert_eq!(
                    load.as_deref(),
                    Some(std::path::Path::new("eval/tree.json"))
                );
                assert!(!json);
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse(&[
            "strata",
            "eval",
            "eval/questions.jsonl",
            "--json",
            "--allow-foreign-stores",
        ])
        .command
        {
            Command::Eval {
                questions,
                json,
                allow_foreign_stores,
                ..
            } => {
                assert!(allow_foreign_stores);
                assert_eq!(
                    questions.as_deref(),
                    Some(std::path::Path::new("eval/questions.jsonl"))
                );
                assert!(json);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn bench_defaults() {
        match parse(&["strata", "bench"]).command {
            Command::Bench {
                tenant: TenantArg { tenant: t },
                queries,
                n,
                mode,
                ingest_dir,
                ingest_files,
                skip_ingest,
            } => {
                assert_eq!(t, tenant());
                assert_eq!(queries, std::path::PathBuf::from("eval/questions.jsonl"));
                assert_eq!(n, 100);
                assert_eq!(mode, ModeArg::Embed);
                assert_eq!(ingest_dir, std::path::PathBuf::from("eval/corpus"));
                assert_eq!(ingest_files, 2);
                assert!(!skip_ingest);
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
