#![forbid(unsafe_code)]
//! `strata` binary entry point.

use std::process::ExitCode;

use clap::Parser;
use strata_rag::cli::{Cli, init_tracing, run};

#[tokio::main]
async fn main() -> ExitCode {
    // Before Config::from_env (inside `run`), so the licences-file permission warning is emitted.
    init_tracing();
    let cli = Cli::parse();
    match run(cli).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
