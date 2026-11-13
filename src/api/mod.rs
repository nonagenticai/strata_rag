//! HTTP API (axum 0.8). Unversioned `/api/<resource>` routes plus `/health` and `/metrics`.

pub mod auth;
mod error;
pub mod eval_api;
pub mod handlers;
pub mod licence;
pub mod platform;

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::StatusCode;
use axum::middleware::map_response;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::engine::Engine;
use crate::error::{Result, StrataError};

/// Largest accepted request body (ingest of a whole book fits comfortably).
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// The API router with auth from `engine.cfg`. Panics only if the licence HTTP client cannot be
/// built (TLS backend init); [`serve`] builds it first and reports that as a startup error.
pub fn router(engine: Arc<Engine>) -> Router {
    let tokens = auth::ApiTokens::from_config(&engine.cfg).expect("licence http client");
    router_with(engine, tokens)
}

fn router_with(engine: Arc<Engine>, tokens: auth::ApiTokens) -> Router {
    // install the Prometheus recorder before the first request records into it
    crate::metrics::handle();
    Router::new()
        .route(
            "/api/stores",
            post(handlers::create_store).get(handlers::list_stores),
        )
        .route(
            "/api/stores/{id}",
            get(handlers::get_store)
                .patch(handlers::update_store)
                .delete(handlers::delete_store),
        )
        .route(
            "/api/stores/{id}/providers",
            get(handlers::get_store_providers)
                .put(handlers::put_store_providers)
                .delete(handlers::delete_store_providers),
        )
        .route(
            "/api/stores/{id}/reembed-copy",
            post(handlers::start_reembed_copy),
        )
        .route(
            "/api/stores/{id}/reembed-copy/{job}",
            get(handlers::get_reembed_copy),
        )
        .route(
            "/api/stores/{id}/documents",
            post(handlers::ingest_document).get(handlers::list_documents),
        )
        .route("/api/documents/{id}", delete(handlers::delete_document))
        .route(
            "/api/sources/{source_id}/documents",
            delete(handlers::prune_source),
        )
        .route("/api/query", post(handlers::query))
        .route("/api/ask", post(handlers::ask))
        .route("/api/eval", post(handlers::eval))
        .route(
            "/api/eval/questions",
            get(eval_api::list_questions).post(eval_api::create_question),
        )
        .route(
            "/api/eval/questions/import",
            post(eval_api::import_questions),
        )
        .route(
            "/api/eval/questions/{id}",
            patch(eval_api::update_question).delete(eval_api::delete_question),
        )
        .route("/api/eval/runs", get(eval_api::list_runs))
        .route("/api/eval/runs/{id}", get(eval_api::get_run))
        .route(
            "/api/tenant/providers",
            get(handlers::get_tenant_providers).put(handlers::put_tenant_providers),
        )
        .route(
            "/api/tenant/providers/test",
            post(handlers::test_tenant_provider),
        )
        .route(
            "/api/platform/packs",
            get(platform::list_packs).post(platform::register_pack),
        )
        .route(
            "/api/platform/packs/{key}/{version}/publish",
            post(platform::publish_pack),
        )
        .route(
            "/api/platform/packs/{key}/{version}/retire",
            post(platform::retire_pack),
        )
        .route(
            "/api/platform/packs/{key}/{version}/stores",
            get(platform::list_pack_stores),
        )
        .route(
            "/api/platform/exclusions",
            get(platform::list_exclusions)
                .put(platform::put_exclusion)
                .delete(platform::delete_exclusion),
        )
        .route("/health", get(handlers::health))
        .route(crate::metrics::METRICS_PATH, get(crate::metrics::render))
        .with_state(engine)
        // inside the auth layer: the tenant it resolved, for the request's log event
        .layer(axum::middleware::from_fn(crate::metrics::capture_tenant))
        // Every route (and every unknown path) but `/health` and `/metrics` needs a bearer token when
        // STRATA_API_TOKENS is set (unauthenticated requests learn nothing about routes), and a
        // tenant: the token's own, or X-Tenant-Id (see `auth`).
        .layer(axum::middleware::from_fn_with_state(
            tokens,
            auth::require_token,
        ))
        // axum's own 2 MB default would win over the tower-http limit; disable it.
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(MAX_BODY_BYTES))
        // tower-http answers 413 in plain text; rewrite it into the standard JSON error body.
        .layer(map_response(json_413))
        .layer(TraceLayer::new_for_http())
        // outermost: counts and times every request, 401s and 413s included
        .layer(axum::middleware::from_fn(crate::metrics::track))
}

/// The metrics-only router: `GET` [`crate::metrics::METRICS_PATH`] and nothing else. Deliberately
/// not built from `router_with`, so no API route is reachable on `STRATA_METRICS_BIND` even if
/// network policy is wrong; every other path is axum's plain 404.
pub fn metrics_router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route(crate::metrics::METRICS_PATH, get(crate::metrics::render))
        .with_state(engine)
}

/// Binds `STRATA_METRICS_BIND`; `Ok(None)` when it is unset (no listener).
pub async fn bind_metrics(
    addr: Option<std::net::SocketAddr>,
) -> Result<Option<tokio::net::TcpListener>> {
    let Some(addr) = addr else { return Ok(None) };
    match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => Ok(Some(l)),
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Err(StrataError::Config(format!(
            "port {} on {} is already in use; set STRATA_METRICS_BIND to a free port (never 8000)",
            addr.port(),
            addr.ip()
        ))),
        Err(e) => Err(e.into()),
    }
}

/// Rewrites any 413 into `{"error":{"code":"payload_too_large",...}}`.
async fn json_413(resp: Response) -> Response {
    if resp.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            axum::Json(serde_json::json!({"error": {
                "code": StrataError::PayloadTooLarge(String::new()).code(),
                "message": "request body exceeds 16 MiB",
            }})),
        )
            .into_response();
    }
    resp
}

/// Serves on `engine.cfg.bind`, the one address `check_serve_auth` validates: there is no second
/// address argument that could differ from the checked one (an open API on a public bind).
pub async fn serve(engine: Arc<Engine>) -> Result<()> {
    engine.cfg.check_serve_auth()?;
    let tokens = auth::ApiTokens::from_config(&engine.cfg)?;
    // A tenant eval runs in a task of the process that started it and beats while it does. On a
    // shutdown signal the process fails its own running runs `shutdown` (below). A run whose beat
    // went stale lost its task without that (its process was killed or crashed) and would block
    // its tenant's next run forever: swept now and every EVAL_SWEEP_EVERY. A run still beating
    // (its worker alive on another replica) is left alone.
    let interrupted =
        crate::db::eval::fail_stale_runs(&engine.pool, None, crate::db::eval::EVAL_STALE_AFTER)
            .await?;
    if interrupted > 0 {
        tracing::warn!(interrupted, "stale tenant eval runs marked failed");
    }
    // histogram samples are drained on every scrape, and here when nobody scrapes
    tokio::spawn(async {
        let mut tick = tokio::time::interval(crate::metrics::UPKEEP_EVERY);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            crate::metrics::handle().run_upkeep();
        }
    });
    let sweeper = Arc::clone(&engine);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(crate::db::eval::EVAL_SWEEP_EVERY);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await; // the first tick is immediate; the start sweep above covered it
        loop {
            tick.tick().await;
            match crate::db::eval::fail_stale_runs(
                &sweeper.pool,
                None,
                crate::db::eval::EVAL_STALE_AFTER,
            )
            .await
            {
                Ok(0) => {}
                Ok(n) => tracing::warn!(interrupted = n, "stale tenant eval runs marked failed"),
                Err(e) => tracing::warn!(error = %e, "stale tenant eval sweep failed"),
            }
        }
    });
    let bind = engine.cfg.bind;
    if let Some(url) = &engine.cfg.license_introspect_url {
        tracing::info!(
            introspect_url = %url,
            app_keys = ?engine.cfg.license_app_keys,
            "licence-token auth on: bearers matching no STRATA_API_TOKENS entry are introspected"
        );
    }
    if !engine.cfg.auth_enabled() {
        tracing::info!(
            %bind,
            "STRATA_API_TOKENS is empty: API open on loopback, no auth (X-Tenant-Id still required)"
        );
    } else {
        tracing::info!(
            tokens = engine.cfg.api_tokens.len(),
            "bearer token auth (tenant-scoped) on for every route except /health and /metrics"
        );
    }
    let listener = match tokio::net::TcpListener::bind(bind).await {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            return Err(StrataError::Config(format!(
                "port {} on {} is already in use; set STRATA_BIND to a free port (never 8000)",
                bind.port(),
                bind.ip()
            )));
        }
        Err(e) => return Err(e.into()),
    };
    tracing::info!(%bind, "strata API listening");
    // Bound before the main server starts so a bad address fails startup, not a background task.
    let metrics_listener = bind_metrics(engine.cfg.metrics_bind).await?;
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let metrics_task = metrics_listener.map(|l| {
        tracing::info!(addr = ?l.local_addr().ok(), "strata metrics listening (GET /metrics only)");
        let app = metrics_router(Arc::clone(&engine));
        let mut stop = stop_rx.clone();
        tokio::spawn(async move {
            axum::serve(l, app)
                .with_graceful_shutdown(async move {
                    let _ = stop.wait_for(|s| *s).await;
                })
                .await
        })
    });
    let tasks = Arc::clone(&engine.eval_tasks);
    let pool = engine.pool.clone();
    // Graceful shutdown drains HTTP connections only; the eval tasks die with the process. Their
    // runs are failed `shutdown` at once on the signal, and again after the drain for any run a
    // draining request started.
    let on_signal = {
        let (tasks, pool) = (Arc::clone(&tasks), pool.clone());
        async move {
            shutdown_signal().await;
            let _ = stop_tx.send(true);
            tasks
                .fail_on_shutdown(&pool, crate::engine::EVAL_SHUTDOWN_TIMEOUT)
                .await;
        }
    };
    axum::serve(listener, router_with(engine, tokens))
        .with_graceful_shutdown(on_signal)
        .await?;
    if let Some(t) = metrics_task {
        match t.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(error = %e, "metrics listener failed"),
            Err(e) => tracing::warn!(error = %e, "metrics listener task panicked"),
        }
    }
    tasks
        .fail_on_shutdown(&pool, crate::engine::EVAL_SHUTDOWN_TIMEOUT)
        .await;
    tracing::info!("strata API stopped");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!(error = %e, "cannot listen for ctrl-c");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => {
                tracing::error!(error = %e, "cannot listen for SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutdown signal received, draining connections");
}
