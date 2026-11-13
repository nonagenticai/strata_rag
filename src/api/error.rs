//! `StrataError` → HTTP response. Body: `{"error":{"code","message"}}`.
//! 500s log the full error and return a generic message (no internals leak to clients).

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::error::StrataError;

pub(crate) const INTERNAL_MESSAGE: &str = "internal server error";

impl IntoResponse for StrataError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let code = self.code();
        // a partial prune still reports what it did, next to the error
        let report = match &self {
            StrataError::EmptySyncRun { report, .. } => serde_json::to_value(report.as_ref()).ok(),
            _ => None,
        };
        let message = if status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!(error = %self, detail = ?self, code, "request failed");
            INTERNAL_MESSAGE.to_string()
        } else {
            if status.is_server_error() {
                tracing::warn!(error = %self, code, "upstream failure");
            }
            self.to_string()
        };
        let mut body = json!({ "error": { "code": code, "message": message } });
        if let (Some(serde_json::Value::Object(fields)), Some(b)) = (report, body.as_object_mut()) {
            b.extend(fields);
        }
        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use serde_json::Value;
    use uuid::Uuid;

    use crate::error::StrataError;

    async fn parts(e: StrataError) -> (StatusCode, Value) {
        let resp = e.into_response();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        (status, serde_json::from_slice(&bytes).expect("json body"))
    }

    fn assert_shape(v: &Value, code: &str) {
        let err = v["error"].as_object().expect("error object");
        assert_eq!(err.len(), 2, "exactly code + message: {v}");
        assert_eq!(err["code"], code);
        assert!(err["message"].as_str().is_some_and(|m| !m.is_empty()));
    }

    #[tokio::test]
    async fn not_found_is_404_with_message() {
        let (s, v) = parts(StrataError::NotFound("store 42".into())).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_shape(&v, "not_found");
        assert!(v["error"]["message"].as_str().unwrap().contains("store 42"));
    }

    #[tokio::test]
    async fn leaf_full_is_409() {
        let e = StrataError::LeafFull {
            store_id: Uuid::nil(),
            current: 10,
            adding: 5,
            max: 12,
        };
        let (s, v) = parts(e).await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_shape(&v, "leaf_full");
    }

    #[tokio::test]
    async fn pack_published_is_409_and_forbidden_is_403() {
        let (s, v) = parts(StrataError::PackPublished {
            key: "industry/retail".into(),
            version: 3,
        })
        .await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_shape(&v, "pack_published");
        assert_eq!(v["error"]["message"], "pack industry/retail@3 is published");
        let (s, v) = parts(StrataError::Forbidden("owner only".into())).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert_shape(&v, "forbidden");
    }

    #[tokio::test]
    async fn invalid_input_is_422() {
        let (s, v) = parts(StrataError::InvalidInput("empty".into())).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
        assert_shape(&v, "invalid_input");
    }

    #[tokio::test]
    async fn upstream_is_502_and_generation_is_503() {
        let (s, v) = parts(StrataError::Upstream {
            service: "laya",
            message: "timeout".into(),
        })
        .await;
        assert_eq!(s, StatusCode::BAD_GATEWAY);
        assert_shape(&v, "upstream_unavailable");
        let (s, v) = parts(StrataError::GenerationUnavailable("vllm down".into())).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
        assert_shape(&v, "generation_unavailable");
    }

    #[tokio::test]
    async fn embed_model_mismatch_is_409() {
        let (s, v) = parts(StrataError::EmbedModelMismatch {
            store_id: Uuid::nil(),
            store_model: "old/model@1024".into(),
            configured: "BAAI/bge-m3@1024".into(),
        })
        .await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_shape(&v, "embed_model_mismatch");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("old/model@1024")
        );
    }

    #[tokio::test]
    async fn internal_errors_hide_details() {
        let (s, v) = parts(StrataError::Model(
            "tokenizer load failed: SECRET-DETAIL".into(),
        ))
        .await;
        assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
        assert_shape(&v, "internal");
        assert_eq!(v["error"]["message"], "internal server error");
        assert!(!v.to_string().contains("SECRET-DETAIL"));
        let (s, v) = parts(StrataError::Db(sqlx::Error::PoolTimedOut)).await;
        assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(v["error"]["message"], "internal server error");
    }
}
