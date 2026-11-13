use std::collections::HashMap;

use strata_rag::config::Config;
use strata_rag::engine::Engine;
use strata_rag::providers::Deployment;

const DEV: &str = r#"
region = "eu"
mode = "dev"
[policy]
allowed_dialects = ["openai", "cohere"]
[default.embed]
base_url = "http://127.0.0.1:1/v1"
dialect = "openai"
model = "BAAI/bge-m3"
dims = 1024
regions = ["eu"]
zdr = true
[default.rerank]
base_url = "http://127.0.0.1:1/v2"
dialect = "cohere"
model = "BAAI/bge-reranker-v2-m3"
regions = ["eu"]
zdr = true
"#;

fn env() -> HashMap<String, String> {
    HashMap::from([(
        "STRATA_DATABASE_URL".to_string(),
        "postgres://strata_rag:hunter2@127.0.0.1:1/strata_rag".to_string(),
    )])
}

#[tokio::test]
async fn unreachable_database_is_reported_with_redacted_url() {
    // Port 1 on loopback: nothing listens, connection is refused.
    let mut cfg = Config::from_map(&env()).expect("config");
    // the default provider clients are built but never called while the engine starts
    cfg.deployment = Some(Deployment::parse(DEV, &HashMap::new()).expect("dev providers"));
    let err = match Engine::from_config(cfg).await {
        Ok(_) => panic!("must fail without a database"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("database unreachable at postgres://strata_rag:***@127.0.0.1:1/strata_rag"),
        "{err}"
    );
    assert!(!err.contains("hunter2"), "password leaked: {err}");
}

/// Without a loaded providers.toml the engine does not start, naming the variable, before any DB
/// connect.
#[tokio::test]
async fn a_missing_providers_file_is_a_startup_error_naming_the_variable() {
    let cfg = Config::from_map(&env()).expect("config");
    let err = match Engine::from_config(cfg).await {
        Ok(_) => panic!("must fail without a providers file"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("STRATA_PROVIDERS_FILE"), "{err}");
    assert!(!err.contains("database"), "checked before the DB: {err}");
}

/// `Config::from_env`'s loader: a file that is not there fails naming it and the variable.
#[test]
fn a_providers_file_that_does_not_exist_names_the_path() {
    let err = Deployment::load(
        std::path::Path::new("/nonexistent/strata/providers.toml"),
        &HashMap::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("/nonexistent/strata/providers.toml"), "{err}");
}
