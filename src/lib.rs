#![forbid(unsafe_code)]
//! strata-rag: a tree of router/leaf stores in Postgres + pgvector, queried by routing
//! down the tree and reranking the reached leaves' chunks.

pub mod api;
pub mod ask;
pub mod bench;
pub mod cli;
pub mod clients;
pub mod config;
pub mod db;
pub mod engine;
pub mod error;
pub mod eval;
pub mod expand;
pub mod inference;
pub mod ingest;
pub mod metering;
pub mod metrics;
pub mod platform;
pub mod precedence;
pub mod providers;
pub mod query;
pub mod query_cache;
pub mod reembed;
pub mod reembed_copy;
pub mod routing;
pub mod search;
pub mod store_providers;
pub mod tenant;
pub mod tenant_eval;
pub mod text;
pub mod vecmath;

pub use error::{Result, StrataError};
pub use tenant::Tenant;
