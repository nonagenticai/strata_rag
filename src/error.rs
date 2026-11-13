//! Domain error type and its HTTP mapping (`code()` / `status()`).

/// Every fallible strata operation returns this error.
#[derive(Debug, thiserror::Error)]
pub enum StrataError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("leaf {store_id} full: {current} + {adding} > {max} tokens")]
    LeafFull {
        store_id: uuid::Uuid,
        current: i64,
        adding: i64,
        max: i64,
    },
    #[error("store {0} has children")]
    HasChildren(uuid::Uuid),
    #[error("duplicate name: {0}")]
    DuplicateName(String),
    #[error("invalid target: {0}")]
    InvalidTarget(String),
    /// A query or tenant eval named no `root_id` while the tenant holds more than one root store
    /// (D90): which tree to search is the caller's choice, never guessed.
    #[error("ambiguous root: {0}")]
    AmbiguousRoot(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("payload too large: {0}")]
    PayloadTooLarge(String),
    /// `POST /api/eval` cannot run: the server's eval files are unusable, or the caller's tenant
    /// has not loaded the eval tree. The message never carries a server path or an OS error.
    #[error("eval not ready: {0}")]
    EvalNotReady(String),
    /// The tenant already has an eval run in progress (`POST /api/eval` with tenant questions).
    #[error("eval running: {0}")]
    EvalRunning(String),
    /// The tenant already has an eval question with this query (case-insensitive).
    #[error("question exists: {0}")]
    DuplicateQuestion(String),
    /// A write would take the tenant past `tenant_eval::EVAL_MAX_QUESTIONS` questions.
    #[error("too many questions: {0}")]
    TooManyQuestions(String),
    /// A write into a store of a published platform pack's subtree: published packs are frozen
    /// (a new version is a new root).
    #[error("pack {key}@{version} is published")]
    PackPublished { key: String, version: i32 },
    /// The tenant may not do this (a platform-owner-only endpoint called by another tenant).
    #[error("forbidden: {0}")]
    Forbidden(String),
    /// The request names no tenant where one is required (`X-Tenant-Id`).
    #[error("tenant required: {0}")]
    TenantRequired(String),
    /// The tenant given is not a usable tenant id (not a uuid, the nil uuid, a `Bearer` value).
    #[error("invalid tenant: {0}")]
    InvalidTenant(String),
    /// A tenant-pinned token was sent with an `X-Tenant-Id` naming a different tenant.
    #[error("tenant mismatch: {0}")]
    TenantMismatch(String),
    /// A concurrent change won: a deadlock or serialization failure (`retryable`, the transaction
    /// is retried once before this surfaces), or a referenced row removed or still referenced
    /// under the write (foreign key 23503 / restrict 23001). Safe for the client to retry.
    #[error("conflict: {message}")]
    Conflict { message: String, retryable: bool },
    /// The store's vectors were made by a different embedding model (`stores.embed_model`) than the
    /// configured one: mixing them would silently corrupt routing and search.
    #[error(
        "store {store_id} was embedded with {store_model}, but the configured embedding model is \
         {configured}; re-embed the store or configure its model"
    )]
    EmbedModelMismatch {
        store_id: uuid::Uuid,
        store_model: String,
        configured: String,
    },
    /// The request's tenant has no provider for this capability: no registration
    /// (`PUT /api/tenant/providers`), and the deployment has no default (production never has one).
    #[error(
        "no {capability} provider is configured for this tenant; register one with \
         PUT /api/tenant/providers"
    )]
    ProviderNotConfigured { capability: &'static str },
    /// A tenant provider registration failed validation or its live probe; `code` is the specific
    /// reason (`region_not_allowed`, `zdr_required`, `dims_must_be_1024`, `dialect_not_allowed`,
    /// `probe_failed`, `api_key_required`, `invalid_provider`). The message never carries a key.
    #[error("provider rejected: {message}")]
    ProviderRejected { code: &'static str, message: String },
    /// Tenant provider keys cannot be encrypted or decrypted: `STRATA_PROVIDER_KEY` is unset.
    #[error("provider keys unavailable: {0}")]
    ProviderKeyUnavailable(String),
    /// A prune keeping only run `run` refused one or more leaves: the source has documents there
    /// but none of that run, so the sync that should have delivered them most likely failed.
    /// Other leaves may have been pruned; `report` says what happened (the HTTP body carries it).
    #[error(
        "source {source_id}: no document carries run {run} in {refused} leaf store(s) {refused_ids:?}, \
         left untouched; {deleted} document(s) deleted elsewhere (pass allow_empty=true to prune \
         anyway)",
        refused = report.refused_stores.len(),
        refused_ids = report.refused_stores,
        deleted = report.deleted
    )]
    EmptySyncRun {
        source_id: String,
        run: String,
        report: Box<crate::ingest::PruneReport>,
    },
    #[error("{service} unavailable: {message}")]
    Upstream {
        service: &'static str,
        message: String,
    },
    #[error("generation unavailable: {0}")]
    GenerationUnavailable(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("config error: {0}")]
    Config(String),
    /// Built only through `From<sqlx::Error>`, which turns concurrency SQLSTATEs into `Conflict`.
    #[error(transparent)]
    Db(sqlx::Error),
    #[error(transparent)]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl From<sqlx::Error> for StrataError {
    fn from(e: sqlx::Error) -> Self {
        if let sqlx::Error::Database(db) = &e {
            let (what, retryable) = match db.code().as_deref() {
                Some("40P01") => ("deadlock with a concurrent request", true),
                Some("40001") => ("serialization failure with a concurrent request", true),
                Some("23503") => (
                    "a referenced store or document was removed concurrently",
                    false,
                ),
                Some("23001") => ("a store gained a child concurrently", false),
                _ => return StrataError::Db(e),
            };
            return StrataError::Conflict {
                message: format!("{what}; retry the request"),
                retryable,
            };
        }
        StrataError::Db(e)
    }
}

impl StrataError {
    /// True for a deadlock / serialization failure: the whole transaction may be run again.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            StrataError::Conflict {
                retryable: true,
                ..
            }
        )
    }

    /// Stable machine-readable code used in `{"error":{"code":...}}` bodies.
    pub fn code(&self) -> &'static str {
        match self {
            StrataError::NotFound(_) => "not_found",
            StrataError::LeafFull { .. } => "leaf_full",
            StrataError::HasChildren(_) => "has_children",
            StrataError::DuplicateName(_) => "duplicate_name",
            StrataError::PackPublished { .. } => "pack_published",
            StrataError::Forbidden(_) => "forbidden",
            StrataError::Conflict { .. } => "conflict",
            StrataError::EmbedModelMismatch { .. } => "embed_model_mismatch",
            StrataError::ProviderNotConfigured { .. } => "provider_not_configured",
            StrataError::ProviderRejected { code, .. } => code,
            StrataError::ProviderKeyUnavailable(_) => "provider_key_unconfigured",
            StrataError::EmptySyncRun { .. } => "empty_sync_run",
            StrataError::InvalidTarget(_) => "invalid_target",
            StrataError::AmbiguousRoot(_) => "ambiguous_root",
            StrataError::InvalidInput(_) => "invalid_input",
            StrataError::PayloadTooLarge(_) => "payload_too_large",
            StrataError::EvalNotReady(_) => "eval_not_ready",
            StrataError::EvalRunning(_) => "eval_running",
            StrataError::DuplicateQuestion(_) => "question_exists",
            StrataError::TooManyQuestions(_) => "too_many_questions",
            StrataError::TenantRequired(_) => "tenant_required",
            StrataError::InvalidTenant(_) => "invalid_tenant",
            StrataError::TenantMismatch(_) => "tenant_mismatch",
            StrataError::Upstream { .. } => "upstream_unavailable",
            StrataError::GenerationUnavailable(_) => "generation_unavailable",
            StrataError::Model(_)
            | StrataError::Config(_)
            | StrataError::Db(_)
            | StrataError::Migrate(_)
            | StrataError::Io(_) => "internal",
        }
    }

    /// HTTP status for this error.
    pub fn status(&self) -> u16 {
        match self {
            StrataError::NotFound(_) => 404,
            StrataError::LeafFull { .. }
            | StrataError::HasChildren(_)
            | StrataError::DuplicateName(_)
            | StrataError::PackPublished { .. }
            | StrataError::AmbiguousRoot(_)
            | StrataError::Conflict { .. }
            | StrataError::EmbedModelMismatch { .. }
            | StrataError::ProviderNotConfigured { .. }
            | StrataError::EmptySyncRun { .. }
            | StrataError::EvalNotReady(_)
            | StrataError::EvalRunning(_)
            | StrataError::DuplicateQuestion(_) => 409,
            StrataError::InvalidTarget(_)
            | StrataError::InvalidInput(_)
            | StrataError::ProviderRejected { .. }
            | StrataError::TooManyQuestions(_) => 422,
            StrataError::PayloadTooLarge(_) => 413,
            StrataError::TenantRequired(_) | StrataError::InvalidTenant(_) => 400,
            StrataError::TenantMismatch(_) | StrataError::Forbidden(_) => 403,
            StrataError::Upstream { .. } => 502,
            StrataError::GenerationUnavailable(_) | StrataError::ProviderKeyUnavailable(_) => 503,
            StrataError::Model(_)
            | StrataError::Config(_)
            | StrataError::Db(_)
            | StrataError::Migrate(_)
            | StrataError::Io(_) => 500,
        }
    }
}

pub type Result<T> = std::result::Result<T, StrataError>;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_maps_to_contract_code_and_status() {
        let id = uuid::Uuid::nil();
        let cases: [(StrataError, &str, u16); 27] = [
            (StrataError::NotFound("store x".into()), "not_found", 404),
            (
                StrataError::LeafFull {
                    store_id: id,
                    current: 10,
                    adding: 5,
                    max: 12,
                },
                "leaf_full",
                409,
            ),
            (StrataError::HasChildren(id), "has_children", 409),
            (
                StrataError::PackPublished {
                    key: "industry/retail".into(),
                    version: 2,
                },
                "pack_published",
                409,
            ),
            (StrataError::Forbidden("x".into()), "forbidden", 403),
            (
                StrataError::AmbiguousRoot("2 roots".into()),
                "ambiguous_root",
                409,
            ),
            (
                StrataError::DuplicateName("a".into()),
                "duplicate_name",
                409,
            ),
            (
                StrataError::Conflict {
                    message: "deadlock".into(),
                    retryable: true,
                },
                "conflict",
                409,
            ),
            (
                StrataError::EmbedModelMismatch {
                    store_id: id,
                    store_model: "a@768".into(),
                    configured: "b@1024".into(),
                },
                "embed_model_mismatch",
                409,
            ),
            (
                StrataError::EmptySyncRun {
                    source_id: "s".into(),
                    run: "r".into(),
                    report: Box::new(crate::ingest::PruneReport {
                        deleted: 0,
                        kept: 3,
                        stores: vec![],
                        refused_stores: vec![id],
                    }),
                },
                "empty_sync_run",
                409,
            ),
            (
                StrataError::ProviderNotConfigured {
                    capability: "embed",
                },
                "provider_not_configured",
                409,
            ),
            (
                StrataError::ProviderRejected {
                    code: "zdr_required",
                    message: "x".into(),
                },
                "zdr_required",
                422,
            ),
            (
                StrataError::ProviderKeyUnavailable("x".into()),
                "provider_key_unconfigured",
                503,
            ),
            (
                StrataError::InvalidTarget("router".into()),
                "invalid_target",
                422,
            ),
            (
                StrataError::InvalidInput("empty".into()),
                "invalid_input",
                422,
            ),
            (
                StrataError::PayloadTooLarge("body".into()),
                "payload_too_large",
                413,
            ),
            (StrataError::EvalNotReady("x".into()), "eval_not_ready", 409),
            (StrataError::EvalRunning("x".into()), "eval_running", 409),
            (
                StrataError::TenantRequired("x".into()),
                "tenant_required",
                400,
            ),
            (
                StrataError::InvalidTenant("x".into()),
                "invalid_tenant",
                400,
            ),
            (
                StrataError::TenantMismatch("x".into()),
                "tenant_mismatch",
                403,
            ),
            (
                StrataError::Upstream {
                    service: "laya",
                    message: "timeout".into(),
                },
                "upstream_unavailable",
                502,
            ),
            (
                StrataError::GenerationUnavailable("down".into()),
                "generation_unavailable",
                503,
            ),
            (StrataError::Model("bad weights".into()), "internal", 500),
            (StrataError::Config("bad".into()), "internal", 500),
            (StrataError::Db(sqlx::Error::RowNotFound), "internal", 500),
            (
                StrataError::Io(std::io::Error::other("boom")),
                "internal",
                500,
            ),
        ];
        for (err, code, status) in cases {
            assert_eq!(err.code(), code, "code for {err:?}");
            assert_eq!(err.status(), status, "status for {err:?}");
        }
    }

    #[test]
    fn display_messages_are_human_readable() {
        let e = StrataError::LeafFull {
            store_id: uuid::Uuid::nil(),
            current: 1,
            adding: 2,
            max: 2,
        };
        assert_eq!(
            e.to_string(),
            "leaf 00000000-0000-0000-0000-000000000000 full: 1 + 2 > 2 tokens"
        );
        let e = StrataError::Upstream {
            service: "laya",
            message: "timeout".into(),
        };
        assert_eq!(e.to_string(), "laya unavailable: timeout");
        assert_eq!(
            StrataError::PackPublished {
                key: "base".into(),
                version: 1
            }
            .to_string(),
            "pack base@1 is published"
        );
        assert_eq!(
            StrataError::NotFound("store 7".into()).to_string(),
            "not found: store 7"
        );
    }

    #[test]
    fn io_and_sqlx_errors_convert_with_question_mark() {
        fn io() -> Result<()> {
            std::fs::read("/nonexistent/strata-rag/missing")?;
            Ok(())
        }
        fn db() -> Result<()> {
            let r: std::result::Result<(), sqlx::Error> = Err(sqlx::Error::RowNotFound);
            r?;
            Ok(())
        }
        assert!(matches!(io(), Err(StrataError::Io(_))));
        assert!(matches!(db(), Err(StrataError::Db(_))));
    }

    #[test]
    fn only_a_retryable_conflict_is_retryable() {
        let c = |retryable| StrataError::Conflict {
            message: "x".into(),
            retryable,
        };
        assert!(c(true).is_retryable());
        assert!(!c(false).is_retryable());
        assert!(!StrataError::Db(sqlx::Error::RowNotFound).is_retryable());
    }
}
