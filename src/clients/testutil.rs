//! Shared mock-HTTP-server helper (unit tests via `clients::testutil`, integration tests via
//! `tests/common`, which includes this same file with `#[path]`). One copy, not one per task.
#![allow(dead_code)]

use axum::Router;

/// Serve `app` on `127.0.0.1:0` for the rest of the process; returns `http://127.0.0.1:<port>`.
pub async fn spawn(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}
