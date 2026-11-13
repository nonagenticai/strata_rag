//! Shared integration-test harness: an ephemeral pgvector Postgres and a test Config.
//! Each test binary uses a subset of these helpers.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use sqlx::PgPool;
use strata_rag::config::Config;
use testcontainers_modules::postgres::Postgres;
use testcontainers_modules::testcontainers::core::IntoContainerPort;
use testcontainers_modules::testcontainers::runners::AsyncRunner;
use testcontainers_modules::testcontainers::{ContainerAsync, ImageExt};
use tokio::sync::OnceCell;

const IMAGE: &str = "pgvector/pgvector";
const TAG: &str = "0.8.7-pg18";

/// A fresh Postgres 18 + pgvector database with migrations applied, on the binary's shared server.
pub struct TestDb {
    pub pool: PgPool,
    pub url: String,
    /// The database's name on the shared server.
    _name: String,
}

/// One server per test binary: `TEST_DATABASE_URL` (a superuser URL) when set, otherwise a single
/// throwaway pgvector container. Starting a container per test made a full run spawn dozens of
/// concurrent `docker run`s and overloaded the shared Docker VM; now each binary starts exactly one
/// and every test gets its own fresh database inside it.
struct Server {
    /// Admin URL (database `postgres`) on the server.
    base: String,
    _container: Option<ContainerAsync<Postgres>>,
}

static SERVER: OnceCell<Server> = OnceCell::const_new();
static CONTAINER_ID: OnceLock<String> = OnceLock::new();
static DB_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Statics are never dropped, so the shared container is removed by an `atexit` hook instead.
extern "C" fn remove_shared_container() {
    if let Some(id) = CONTAINER_ID.get() {
        let _ = std::process::Command::new("docker")
            .args(["rm", "-f", "-v", id])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

/// An OS-assigned free port. These are ephemeral (49152+), so never 8000. A port docker has
/// published is bound on the host, so the OS will not hand it out here either.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

/// Attempts before giving up: `free_port` is check-then-use, so another process can take the port
/// between our probe and docker's bind (TOCTOU). A fresh port per attempt makes that a retry, not a flake.
const START_ATTEMPTS: usize = 3;

async fn start_container() -> (String, ContainerAsync<Postgres>) {
    let mut failures = Vec::new();
    for attempt in 1..=START_ATTEMPTS {
        // Same port inside and outside: postgres listens on P and P is published as P.
        let port = free_port();
        // A unique name, so a container docker created but could not start can be removed.
        let name = format!("strata-rag-test-{}", uuid::Uuid::now_v7());
        let started = Postgres::default()
            .with_user("strata_rag")
            .with_password("strata_rag")
            .with_name(IMAGE)
            .with_tag(TAG)
            .with_env_var("PGPORT", port.to_string())
            .with_cmd([
                "postgres".to_string(),
                "-c".to_string(),
                format!("port={port}"),
                "-c".to_string(),
                "max_connections=500".to_string(),
                "-c".to_string(),
                "fsync=off".to_string(),
            ])
            .with_mapped_port(port, port.tcp())
            .with_container_name(&name)
            .start()
            .await;
        let container = match started {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "pgvector container on port {port} failed (attempt {attempt}/{START_ATTEMPTS}): {e}"
                );
                failures.push(format!("port {port}: {e}"));
                remove_named_container(&name);
                continue;
            }
        };
        let base = format!("postgres://strata_rag:strata_rag@127.0.0.1:{port}/postgres");
        // On Docker Desktop a port taken after our probe does not fail `start()`: the publish
        // succeeds and the client reaches the squatter instead, so a connect failure retries too.
        // The publish itself can also lag the container start under load, so poll briefly.
        let mut last = String::new();
        for _ in 0..40 {
            match PgPool::connect(&base).await {
                Ok(admin) => {
                    admin.close().await;
                    return (base, container);
                }
                Err(e) => last = e.to_string(),
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
        eprintln!("connect on port {port} failed (attempt {attempt}/{START_ATTEMPTS}): {last}");
        failures.push(format!("port {port}: connect: {last}"));
        // dropping `container` removes it
    }
    panic!(
        "start pgvector container + connect (is Docker running?) failed {START_ATTEMPTS} times: {failures:?}"
    );
}

async fn server() -> &'static Server {
    SERVER
        .get_or_init(|| async {
            if let Ok(base) = std::env::var("TEST_DATABASE_URL")
                && !base.is_empty()
            {
                return Server {
                    base,
                    _container: None,
                };
            }
            let (base, c) = start_container().await;
            if CONTAINER_ID.set(c.id().to_string()).is_ok() {
                // SAFETY: registering a plain `extern "C" fn()` with atexit.
                unsafe { libc::atexit(remove_shared_container) };
            }
            Server {
                base,
                _container: Some(c),
            }
        })
        .await
}

/// Swap the database name (path segment) in a postgres URL, keeping any query string.
fn with_db_name(url: &str, name: &str) -> String {
    let (head, query) = match url.split_once('?') {
        Some((h, q)) => (h, Some(q)),
        None => (url, None),
    };
    let scheme_end = head.find("://").map(|i| i + 3).unwrap_or(0);
    let base = match head[scheme_end..].find('/') {
        Some(i) => &head[..scheme_end + i],
        None => head,
    };
    match query {
        Some(q) => format!("{base}/{name}?{q}"),
        None => format!("{base}/{name}"),
    }
}

impl TestDb {
    /// A fresh database on the binary's shared server, migrated exactly as production does
    /// (`strata_rag::db::connect`, whose first migration creates the `vector` extension).
    pub async fn start() -> TestDb {
        let srv = server().await;
        let n = DB_COUNTER.fetch_add(1, Ordering::SeqCst);
        let name = format!(
            "strata_rag_t_{}_{n}_{}",
            std::process::id(),
            uuid::Uuid::now_v7().simple()
        );
        let admin = PgPool::connect(&srv.base)
            .await
            .expect("connect to the shared test postgres server");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {name}")))
            .execute(&admin)
            .await
            .expect("create test database");
        admin.close().await;
        let url = with_db_name(&srv.base, &name);
        let pool = strata_rag::db::connect(&url)
            .await
            .unwrap_or_else(|e| panic!("connect + migrate test database {name}: {e}"));
        TestDb {
            pool,
            url,
            _name: name,
        }
    }
}

/// Best effort: remove the (created, never started) container this harness named.
fn remove_named_container(name: &str) {
    let _ = std::process::Command::new("docker")
        .args(["rm", "-f", "-v", name])
        .output();
}

/// The tenant every single-tenant test acts for.
pub const TENANT: &str = "0192f0c8-0000-7000-8000-00000000000a";
/// A second tenant, for isolation tests.
pub const OTHER_TENANT: &str = "0192f0c8-0000-7000-8000-00000000000b";

/// `TENANT` as a `Tenant`.
pub fn tn() -> strata_rag::Tenant {
    strata_rag::Tenant::parse(TENANT).expect("TENANT is a uuid")
}

/// `OTHER_TENANT` as a `Tenant`.
pub fn other() -> strata_rag::Tenant {
    strata_rag::Tenant::parse(OTHER_TENANT).expect("OTHER_TENANT is a uuid")
}

/// The platform owner (`STRATA_PLATFORM_OWNER`) in platform-knowledge tests.
pub const PLATFORM: &str = "0192f0c8-0000-7000-8000-0000000000ff";

/// `PLATFORM` as a `Tenant`.
pub fn platform() -> strata_rag::Tenant {
    strata_rag::Tenant::parse(PLATFORM).expect("PLATFORM is a uuid")
}

/// `test_config` with platform knowledge on, owned by `PLATFORM`.
pub fn platform_config() -> Config {
    let mut cfg = test_config();
    cfg.platform_owner = Some(platform());
    cfg
}

/// `req` with `X-Tenant-Id: TENANT` unless it already names a tenant: the test router runs with
/// auth off (no STRATA_API_TOKENS), where every tenanted route needs the header.
pub fn with_tenant(
    mut req: axum::http::Request<axum::body::Body>,
) -> axum::http::Request<axum::body::Body> {
    if !req.headers().contains_key("x-tenant-id") {
        req.headers_mut()
            .insert("x-tenant-id", axum::http::HeaderValue::from_static(TENANT));
    }
    req
}

/// Defaults from an empty env, shrunk so tests exercise chunking and the leaf cap cheaply.
pub fn test_config() -> Config {
    let mut cfg = Config::from_map(&HashMap::new()).expect("default config");
    cfg.chunk_target = 32;
    cfg.chunk_overlap = 4;
    cfg.leaf_max_tokens = 10_000;
    cfg
}

/// Engine wired to a fresh database on the shared ephemeral Postgres, the fake embedder/reranker and the
/// whitespace token counter. Keep the `TestDb` alive for the test's duration.
#[allow(dead_code)]
pub async fn test_engine() -> (strata_rag::engine::Engine, TestDb) {
    let db = TestDb::start().await;
    let engine = strata_rag::engine::Engine::new(
        db.pool.clone(),
        strata_rag::inference::InferenceHandle::fake(),
        std::sync::Arc::new(strata_rag::text::WhitespaceCounter),
        test_config(),
    );
    (engine, db)
}

/// The backend pid of `conn` (a transaction or a pooled connection).
pub async fn backend_pid(conn: &mut sqlx::PgConnection) -> i32 {
    sqlx::query_scalar("SELECT pg_backend_pid()")
        .fetch_one(conn)
        .await
        .expect("pg_backend_pid")
}

/// Waits until some backend is waiting on a lock held by `holder_pid` and returns its pid. This is
/// a condition, not a timing guess: lock-ordering tests hold a row lock on one connection, start the
/// operation under test, and only act once Postgres reports the operation blocked behind that lock.
/// Bounded so a test that never blocks fails instead of hanging.
pub async fn wait_until_blocked_by(pool: &PgPool, holder_pid: i32) -> i32 {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        let waiter: Option<i32> = sqlx::query_scalar(
            "SELECT pid FROM pg_stat_activity WHERE $1 = ANY (pg_blocking_pids(pid)) LIMIT 1",
        )
        .bind(holder_pid)
        .fetch_optional(pool)
        .await
        .expect("query pg_stat_activity");
        if let Some(pid) = waiter {
            return pid;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no backend blocked on a lock held by pid {holder_pid} within 15 s"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

/// A chat provider client for a mock serving `/v1/chat/completions` (and `/v1/models`) at `base`,
/// bearer `vtok`, model `local-llm`, `timeout_s` seconds, today's vLLM options (thinking off,
/// temperature 0.2).
pub fn chat_at(
    base: &str,
    timeout_s: u64,
) -> std::sync::Arc<strata_rag::clients::chat::ChatClient> {
    std::sync::Arc::new(
        strata_rag::clients::chat::ChatClient::new(
            &format!("{base}/v1"),
            Some(strata_rag::config::Secret::new("vtok")),
            "local-llm",
            timeout_s * 1000,
            strata_rag::clients::chat::ChatOptions {
                thinking: Some(false),
                temperature: Some(0.2),
            },
        )
        .expect("chat client"),
    )
}

/// Mock-HTTP-server helper shared with the unit tests (single source file).
#[path = "../../src/clients/testutil.rs"]
pub mod mock;
