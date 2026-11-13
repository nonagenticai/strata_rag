//! `strata bench`: query latency p50/p95 per stage over N queries, and ingest throughput
//! (chunks/s) into a temporary leaf that is deleted afterwards.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use serde::Serialize;
use uuid::Uuid;

use crate::cli::collect_text_files;
use crate::db::stores::{NewStore, StoreKind};
use crate::engine::Engine;
use crate::error::{Result, StrataError};
use crate::eval::parse_questions;
use crate::ingest::IngestOutcome;
use crate::query::{QueryRequest, Timings};
use crate::routing::RouteMode;
use crate::tenant::Tenant;

/// Spec §1 target for `/api/query` p50 without laya.
pub const P50_TARGET_MS: u64 = 250;
pub const WARMUP_QUERIES: usize = 3;

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub queries: PathBuf,
    pub n: usize,
    pub mode: RouteMode,
    pub ingest_dir: PathBuf,
    pub ingest_files: usize,
    pub skip_ingest: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QuerySample {
    pub timings: Timings,
    pub wall_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StageStats {
    pub stage: &'static str,
    pub p50: u64,
    pub p95: u64,
    pub max: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct IngestBench {
    pub documents: usize,
    pub chunks: i64,
    pub seconds: f64,
    pub chunks_per_s: f64,
}

type Getter = fn(&QuerySample) -> u64;

/// Nearest-rank percentile (`p` in 0..=100); 0 for an empty slice.
pub fn percentile(values: &[u64], p: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    let rank = ((p / 100.0) * v.len() as f64).ceil() as usize;
    v[rank.clamp(1, v.len()) - 1]
}

pub fn stage_stats(samples: &[QuerySample]) -> Vec<StageStats> {
    let stages: [(&'static str, Getter); 6] = [
        ("embed", |s| s.timings.embed),
        ("route", |s| s.timings.route),
        ("search", |s| s.timings.search),
        ("rerank", |s| s.timings.rerank),
        ("total", |s| s.timings.total),
        ("wall", |s| s.wall_ms),
    ];
    stages
        .iter()
        .map(|&(stage, get)| {
            let v: Vec<u64> = samples.iter().map(get).collect();
            StageStats {
                stage,
                p50: percentile(&v, 50.0),
                p95: percentile(&v, 95.0),
                max: v.iter().copied().max().unwrap_or(0),
            }
        })
        .collect()
}

pub fn throughput(chunks: i64, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        0.0
    } else {
        chunks as f64 / seconds
    }
}

pub fn render_bench(
    mode: &str,
    n: usize,
    stats: &[StageStats],
    ingest: Option<&IngestBench>,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "query latency, mode={mode}, n={n} (ms)");
    let _ = writeln!(out, "{:<8} {:>6} {:>6} {:>6}", "stage", "p50", "p95", "max");
    for s in stats {
        let _ = writeln!(out, "{:<8} {:>6} {:>6} {:>6}", s.stage, s.p50, s.p95, s.max);
    }
    if let Some(total) = stats.iter().find(|s| s.stage == "total") {
        let verdict = if total.p50 < P50_TARGET_MS {
            "PASS"
        } else {
            "MISS"
        };
        let _ = writeln!(
            out,
            "target: p50(total) {} ms < {P50_TARGET_MS} ms: {verdict}",
            total.p50
        );
    }
    if let Some(i) = ingest {
        let _ = writeln!(
            out,
            "ingest: {} documents, {} chunks in {:.1} s = {:.1} chunks/s",
            i.documents, i.chunks, i.seconds, i.chunks_per_s
        );
    }
    out
}

fn mode_label(mode: RouteMode) -> &'static str {
    match mode {
        RouteMode::Embed => "embed",
        RouteMode::Laya => "laya",
        RouteMode::Hybrid => "hybrid",
    }
}

pub async fn run_query_bench(
    engine: &Engine,
    tenant: Tenant,
    queries: &[String],
    n: usize,
    mode: RouteMode,
) -> Result<Vec<QuerySample>> {
    if queries.is_empty() {
        return Err(StrataError::InvalidInput(
            "bench needs at least one query".into(),
        ));
    }
    if n == 0 {
        return Err(StrataError::InvalidInput(
            "bench --n must be at least 1".into(),
        ));
    }
    let req = |q: &str| QueryRequest {
        route_mode: mode,
        ..QueryRequest::new(q)
    };
    for q in queries.iter().cycle().take(WARMUP_QUERIES) {
        engine.query(tenant, &req(q)).await?;
    }
    let mut samples = Vec::with_capacity(n);
    for q in queries.iter().cycle().take(n) {
        let started = Instant::now();
        let resp = engine.query(tenant, &req(q)).await?;
        samples.push(QuerySample {
            timings: resp.timings_ms,
            wall_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        });
    }
    Ok(samples)
}

/// Ingest `files` into a fresh `bench-tmp-<uuid>` leaf under the root router, then delete it.
pub async fn run_ingest_bench(
    engine: &Engine,
    tenant: Tenant,
    files: &[PathBuf],
) -> Result<IngestBench> {
    let views = engine.store_tree(tenant).await?;
    let root = views
        .iter()
        .find(|v| v.parent_id.is_none())
        .ok_or_else(|| {
            StrataError::InvalidInput(
                "bench needs a tree: run `strata eval --load eval/tree.json` first".into(),
            )
        })?;
    if root.kind != StoreKind::Router {
        return Err(StrataError::InvalidInput(
            "bench needs a router root".into(),
        ));
    }
    let leaf = engine
        .create_store(
            tenant,
            NewStore {
                parent_id: Some(root.id),
                kind: StoreKind::Leaf,
                name: format!("bench-tmp-{}", Uuid::now_v7()),
                description: "temporary leaf created by strata bench; deleted when the bench ends"
                    .into(),
            },
        )
        .await?;
    // An ingest error or early `?` inside `ingest_timed` still reaches the delete below. Ctrl-C
    // would not: it kills the process with the temp leaf still in the tree, where it takes a
    // routing beam slot in every later query and `strata eval`. So treat it as an error too.
    let result = tokio::select! {
        r = ingest_timed(engine, tenant, leaf.id, files) => r,
        Ok(()) = tokio::signal::ctrl_c() => Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "bench interrupted by Ctrl-C",
        )
        .into()),
    };
    let cleanup = engine.delete_store(tenant, leaf.id, true).await;
    if let Err(e) = &cleanup {
        eprintln!(
            "could not delete temporary bench leaf {id}: {e}; remove it with `strata store rm {id}`",
            id = leaf.id
        );
    }
    let bench = result?;
    cleanup?;
    Ok(bench)
}

async fn ingest_timed(
    engine: &Engine,
    tenant: Tenant,
    leaf: Uuid,
    files: &[PathBuf],
) -> Result<IngestBench> {
    let started = Instant::now();
    let (mut documents, mut chunks) = (0_usize, 0_i64);
    for f in files {
        let bytes = tokio::fs::read(f).await?;
        let text = String::from_utf8(bytes).map_err(|_| {
            StrataError::InvalidInput(format!("{} is not valid UTF-8", f.display()))
        })?;
        match engine
            .ingest(tenant, leaf, f.display().to_string(), None, &text)
            .await?
        {
            IngestOutcome::Created(d) => {
                documents += 1;
                chunks += d.chunk_count;
            }
            // a dedup hit embeds nothing; counting it would inflate chunks/s
            IngestOutcome::Existing(_) => {}
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    Ok(IngestBench {
        documents,
        chunks,
        seconds,
        chunks_per_s: throughput(chunks, seconds),
    })
}

pub async fn run(engine: &Engine, tenant: Tenant, opts: &BenchOptions) -> Result<ExitCode> {
    let questions = parse_questions(&std::fs::read_to_string(&opts.queries)?)?;
    let queries: Vec<String> = questions.into_iter().map(|q| q.query).collect();
    let samples = run_query_bench(engine, tenant, &queries, opts.n, opts.mode).await?;
    let stats = stage_stats(&samples);
    let ingest = if opts.skip_ingest {
        None
    } else {
        let files: Vec<PathBuf> = collect_text_files(&opts.ingest_dir)?
            .into_iter()
            .take(opts.ingest_files)
            .collect();
        if files.is_empty() {
            return Err(StrataError::InvalidInput(format!(
                "no .txt/.md files in {} (run scripts/fetch_corpus.sh)",
                opts.ingest_dir.display()
            )));
        }
        Some(run_ingest_bench(engine, tenant, &files).await?)
    };
    println!("rerank_candidates={}", engine.cfg.rerank_candidates);
    print!(
        "{}",
        render_bench(mode_label(opts.mode), opts.n, &stats, ingest.as_ref())
    );
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(embed: u64, total: u64, wall: u64) -> QuerySample {
        QuerySample {
            timings: Timings {
                embed,
                route: 1,
                search: 2,
                rerank: 3,
                total,
            },
            wall_ms: wall,
        }
    }

    #[test]
    fn percentile_nearest_rank() {
        let v: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile(&v, 50.0), 50);
        assert_eq!(percentile(&v, 95.0), 95);
        assert_eq!(percentile(&v, 100.0), 100);
        assert_eq!(percentile(&[40, 10, 30, 20], 50.0), 20);
        assert_eq!(percentile(&[40, 10, 30, 20], 95.0), 40);
        assert_eq!(percentile(&[7], 95.0), 7);
        assert_eq!(percentile(&[], 50.0), 0);
        assert_eq!(percentile(&[5, 9], 0.0), 5);
    }

    #[test]
    fn stage_stats_covers_every_stage() {
        let s = stage_stats(&[
            sample(10, 100, 110),
            sample(30, 300, 310),
            sample(20, 200, 210),
        ]);
        let names: Vec<&str> = s.iter().map(|x| x.stage).collect();
        assert_eq!(
            names,
            ["embed", "route", "search", "rerank", "total", "wall"]
        );
        let total = s.iter().find(|x| x.stage == "total").unwrap();
        assert_eq!((total.p50, total.p95, total.max), (200, 300, 300));
        let embed = s.iter().find(|x| x.stage == "embed").unwrap();
        assert_eq!(embed.p50, 20);
    }

    #[test]
    fn throughput_guards_zero_time() {
        assert!((throughput(500, 2.0) - 250.0).abs() < 1e-9);
        assert_eq!(throughput(500, 0.0), 0.0);
    }

    #[test]
    fn render_bench_states_target_verdict() {
        let fast = stage_stats(&[sample(5, 120, 125)]);
        assert!(render_bench("embed", 1, &fast, None).contains("p50(total) 120 ms < 250 ms: PASS"));
        let slow = stage_stats(&[sample(5, 400, 410)]);
        let out = render_bench(
            "embed",
            1,
            &slow,
            Some(&IngestBench {
                documents: 2,
                chunks: 1000,
                seconds: 4.0,
                chunks_per_s: 250.0,
            }),
        );
        assert!(out.contains("p50(total) 400 ms < 250 ms: MISS"), "{out}");
        assert!(
            out.contains("ingest: 2 documents, 1000 chunks in 4.0 s = 250.0 chunks/s"),
            "{out}"
        );
    }
}
