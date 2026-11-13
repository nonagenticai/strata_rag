//! Request telemetry: one structured `strata::request` event per HTTP request, and the Prometheus
//! metrics `GET /metrics` serves (unauthenticated, like `/health`; bind the API where only the
//! scraper and the callers reach it).
//!
//! Requests are labelled by their route template (`/api/stores/{id}`), never the raw path, and
//! never by tenant: metric cardinality stays bounded and an open endpoint names no tenant. The log
//! event carries the tenant and, for `/api/query` and `/api/ask`, the pipeline's `timings_ms`.
//!
//! [`track`] wraps the whole router and opens a per-request context (a task-local, like
//! `metering::with_caller`); [`capture_tenant`] runs inside the auth layer and copies the resolved
//! tenant into it; the query/ask handlers put their stage timings in it with
//! [`record_query_timings`] / [`record_ask_timings`].

use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;
use std::time::Instant;

use axum::extract::{MatchedPath, Request, State};
use axum::http::Method;
use axum::http::header::CONTENT_TYPE;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

use crate::ask::AskTimings;
use crate::engine::Engine;
use crate::query::Timings;
use crate::tenant::Tenant;

/// The Prometheus scrape path; open without a token (see `api::auth`).
pub const METRICS_PATH: &str = "/metrics";

/// How often `strata serve` drains recorded histogram samples when nobody scrapes.
pub const UPKEEP_EVERY: Duration = Duration::from_secs(5);

/// Bound on the outbox backlog count a scrape runs.
const OUTBOX_COUNT_TIMEOUT: Duration = Duration::from_secs(1);

/// Latency histogram buckets, seconds: from a cache-hit query to a slow generation.
const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// The route label of a request no route matched (a 404, or a 401 before routing was learned).
const UNMATCHED: &str = "unmatched";

/// The process-wide recorder's handle, installing the recorder on first use. `api::router` calls
/// it, so the recorder exists before the first request records anything.
pub fn handle() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        let recorder = PrometheusBuilder::new()
            .set_buckets(LATENCY_BUCKETS)
            .expect("LATENCY_BUCKETS is not empty")
            .build_recorder();
        let handle = recorder.handle();
        if ::metrics::set_global_recorder(recorder).is_err() {
            tracing::warn!("another metrics recorder is installed; /metrics will stay empty");
        }
        describe();
        handle
    })
}

fn describe() {
    ::metrics::describe_counter!(
        "strata_http_requests_total",
        "HTTP requests by method, route template and status"
    );
    ::metrics::describe_histogram!(
        "strata_http_request_duration_seconds",
        ::metrics::Unit::Seconds,
        "HTTP request latency by method, route template and status"
    );
    ::metrics::describe_histogram!(
        "strata_query_stage_duration_seconds",
        ::metrics::Unit::Seconds,
        "Query pipeline stage latency (embed, route, search, rerank, generate, total) by endpoint"
    );
    ::metrics::describe_counter!(
        "strata_dense_search_groups_total",
        "Dense searches per embedding-model group, by path (exact scan or hnsw index)"
    );
    ::metrics::describe_gauge!(
        "strata_db_pool_size",
        "Postgres pool connections open (idle + in use)"
    );
    ::metrics::describe_gauge!("strata_db_pool_idle", "Postgres pool connections idle");
    ::metrics::describe_gauge!(
        "strata_metering_queued",
        "Cost events queued in this process, not yet written to metering_outbox"
    );
    ::metrics::describe_counter!(
        "strata_metering_dropped_total",
        "Cost events this process lost (queue full, or refused by Postgres)"
    );
    ::metrics::describe_gauge!(
        "strata_metering_outbox_pending",
        "metering_outbox rows not yet delivered (all processes)"
    );
}

/// What one request learned on its way through, for its log event.
#[derive(Default)]
struct RequestCtx {
    tenant: OnceLock<Tenant>,
    stages: Mutex<Option<Stages>>,
}

#[derive(Clone, Copy)]
struct Stages {
    embed: u64,
    route: u64,
    search: u64,
    rerank: u64,
    total: u64,
    generate: Option<u64>,
}

tokio::task_local! {
    static CTX: Arc<RequestCtx>;
}

/// Outermost middleware: times the request, counts it, and logs one `strata::request` event.
pub async fn track(req: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = method_label(req.method());
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map_or(UNMATCHED, MatchedPath::as_str)
        .to_owned();
    let ctx = Arc::new(RequestCtx::default());
    let resp = CTX.scope(Arc::clone(&ctx), next.run(req)).await;
    let elapsed = start.elapsed();
    let status = resp.status().as_u16();

    ::metrics::counter!(
        "strata_http_requests_total",
        "method" => method,
        "route" => route.clone(),
        "status" => status.to_string(),
    )
    .increment(1);
    ::metrics::histogram!(
        "strata_http_request_duration_seconds",
        "method" => method,
        "route" => route.clone(),
        "status" => status.to_string(),
    )
    .record(elapsed.as_secs_f64());

    let ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
    let tenant = ctx.tenant.get().map(tracing::field::display);
    let stages = *ctx.stages.lock().unwrap_or_else(PoisonError::into_inner);
    macro_rules! request_event {
        ($level:expr) => {
            tracing::event!(
                target: "strata::request",
                $level,
                method,
                route = route.as_str(),
                status,
                tenant,
                ms,
                embed_ms = stages.map(|s| s.embed),
                route_ms = stages.map(|s| s.route),
                search_ms = stages.map(|s| s.search),
                rerank_ms = stages.map(|s| s.rerank),
                query_ms = stages.map(|s| s.total),
                generate_ms = stages.and_then(|s| s.generate),
                "request"
            )
        };
    }
    // probes and scrapes arrive every few seconds: kept out of the default (info) log
    if route == "/health" || route == METRICS_PATH {
        request_event!(tracing::Level::DEBUG);
    } else {
        request_event!(tracing::Level::INFO);
    }
    resp
}

/// Runs inside the auth layer: copies the tenant it resolved into the request's context.
pub async fn capture_tenant(req: Request, next: Next) -> Response {
    if let Some(t) = req.extensions().get::<Tenant>().copied() {
        let _ = CTX.try_with(|c| c.tenant.set(t));
    }
    next.run(req).await
}

/// `/api/query`'s stage timings: into the stage histograms and the request's log event.
pub fn record_query_timings(t: &Timings) {
    record_stages("query", t, None);
}

/// `/api/ask`'s stage timings (the query's stages plus `generate`).
pub fn record_ask_timings(t: &AskTimings) {
    record_stages("ask", &t.query, Some(t.generate));
}

fn record_stages(endpoint: &'static str, t: &Timings, generate: Option<u64>) {
    let mut stages = vec![
        ("embed", t.embed),
        ("route", t.route),
        ("search", t.search),
        ("rerank", t.rerank),
        ("total", t.total),
    ];
    stages.extend(generate.map(|g| ("generate", g)));
    for (stage, ms) in stages {
        ::metrics::histogram!(
            "strata_query_stage_duration_seconds",
            "endpoint" => endpoint,
            "stage" => stage,
        )
        .record(ms as f64 / 1000.0);
    }
    let _ = CTX.try_with(|c| {
        *c.stages.lock().unwrap_or_else(PoisonError::into_inner) = Some(Stages {
            embed: t.embed,
            route: t.route,
            search: t.search,
            rerank: t.rerank,
            total: t.total,
            generate,
        });
    });
}

/// One embedding-model group's dense search took `path` (`crate::search::search_groups_ef`). Label
/// values are the two paths only (bounded cardinality, no tenant).
pub fn record_dense_path(path: crate::db::search::DensePath) {
    ::metrics::counter!("strata_dense_search_groups_total", "path" => path.as_str()).increment(1);
}

/// `GET /metrics`: Prometheus text exposition. Pool and metering gauges are read at scrape time.
pub async fn render(State(engine): State<Arc<Engine>>) -> Response {
    let handle = handle();
    ::metrics::gauge!("strata_db_pool_size").set(f64::from(engine.pool.size()));
    ::metrics::gauge!("strata_db_pool_idle").set(engine.pool.num_idle() as f64);
    if let Some(m) = &engine.metering {
        ::metrics::gauge!("strata_metering_queued").set(m.queued() as f64);
        ::metrics::counter!("strata_metering_dropped_total").absolute(m.dropped());
        // the partial `metering_outbox_due_idx` covers exactly the pending rows
        let count = sqlx::query_scalar::<_, i64>(
            "SELECT count(*) FROM metering_outbox WHERE status = 'pending'",
        )
        .fetch_one(&engine.pool);
        match tokio::time::timeout(OUTBOX_COUNT_TIMEOUT, count).await {
            Ok(Ok(n)) => ::metrics::gauge!("strata_metering_outbox_pending").set(n as f64),
            Ok(Err(e)) => tracing::debug!(error = %e, "metering outbox backlog count failed"),
            Err(_) => tracing::debug!("metering outbox backlog count timed out"),
        }
    }
    // histogram samples reach the exposition only through upkeep
    handle.run_upkeep();
    (
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        handle.render(),
    )
        .into_response()
}

/// Standard methods by name; anything else is `other` (bounded label values).
fn method_label(m: &Method) -> &'static str {
    match *m {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::PATCH => "PATCH",
        Method::DELETE => "DELETE",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        _ => "other",
    }
}
