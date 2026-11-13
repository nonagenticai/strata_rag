//! Postgres access. Tasks 8, 9 and 12 add `pub mod stores; pub mod documents; pub mod search;` here.

use std::future::Future;
use std::time::Duration;

use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

use crate::error::Result;

pub mod documents;
pub mod eval;
pub mod platform;
pub mod providers;
pub mod search;
pub mod stores;

/// Pool sizing and recycling (`STRATA_DB_*`, see `crate::config::Config::pool_settings`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSettings {
    pub max_connections: u32,
    /// Idle connections are closed after this. Production reaches postgres through pgbouncer in
    /// session mode, where postgres `idle_session_timeout` (600 s) kills a connection that sat
    /// idle and pgbouncer `server_idle_timeout` is 240 s; sqlx's own default (600 s) races the
    /// former and the loser reads EOF. Keep this below both.
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
}

impl Default for PoolSettings {
    fn default() -> Self {
        Self {
            max_connections: 16,
            idle_timeout: Duration::from_secs(120),
            max_lifetime: Duration::from_secs(1800),
        }
    }
}

/// [`connect_with`] at the default [`PoolSettings`].
pub async fn connect(url: &str) -> Result<PgPool> {
    connect_with(url, PoolSettings::default()).await
}

/// Open a pool and apply the embedded migrations. sqlx serialises concurrent migrators with an
/// advisory lock. If the DB is unreachable this returns `StrataError::Db`; the binary exits non-zero.
pub async fn connect_with(url: &str, settings: PoolSettings) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(settings.max_connections)
        .idle_timeout(settings.idle_timeout)
        .max_lifetime(settings.max_lifetime)
        .acquire_timeout(Duration::from_secs(5))
        .connect(url)
        .await?;
    sqlx::migrate!("./migrations").run(&pool).await?;
    Ok(pool)
}

/// Runs a whole transaction `run` and, if it lost a deadlock or serialization race
/// (`StrataError::is_retryable`), runs it once more. `run` must begin its own transaction, so the
/// aborted one is rolled back (dropped) before the retry. A second loss surfaces as 409 conflict.
pub async fn retry_once<T, F, Fut>(mut run: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    match run().await {
        Err(e) if e.is_retryable() => {
            tracing::warn!(error = %e, "transaction lost a concurrency race; retrying once");
            run().await
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::error::StrataError;

    fn conflict(retryable: bool) -> StrataError {
        StrataError::Conflict {
            message: "race".into(),
            retryable,
        }
    }

    #[tokio::test]
    async fn retries_a_retryable_failure_exactly_once() {
        let calls = AtomicUsize::new(0);
        let r = retry_once(|| async {
            match calls.fetch_add(1, Ordering::SeqCst) {
                0 => Err(conflict(true)),
                _ => Ok(7),
            }
        })
        .await;
        assert_eq!(r.unwrap(), 7);
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        let calls = AtomicUsize::new(0);
        let r: Result<()> = retry_once(|| async {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(conflict(true))
        })
        .await;
        assert!(r.unwrap_err().is_retryable());
        assert_eq!(calls.load(Ordering::SeqCst), 2, "only one retry");
    }

    #[tokio::test]
    async fn does_not_retry_other_errors() {
        let calls = AtomicUsize::new(0);
        let r: Result<()> = retry_once(|| async {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(conflict(false))
        })
        .await;
        assert_eq!(r.unwrap_err().code(), "conflict");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
