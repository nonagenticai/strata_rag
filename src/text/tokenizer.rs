//! bge-m3's `tokenizer.json` for chunk token counting (`HfTokenCounter`). Only the tokenizer file is
//! needed (no weights): embeddings come from the embed provider over HTTP.
//!
//! Resolution order:
//! 1. `STRATA_TOKENIZER_PATH`, when set (non-empty): that file is used as-is, never downloaded. The
//!    container image bakes the pinned file in and sets this, so a pod with a read-only root
//!    filesystem and no Hugging Face egress needs neither a cache volume nor the network.
//! 2. Otherwise hf-hub's blocking client, cache first (~/.cache/huggingface/hub, or `HF_HOME`),
//!    downloading only on a miss.

use std::ffi::OsString;
use std::fs::File;
use std::path::PathBuf;

use hf_hub::HFClientSync;

use crate::error::{Result, StrataError};

const OWNER: &str = "BAAI";
const NAME: &str = "bge-m3";
/// Pinned commit, so a moved ref can never change chunk boundaries under us.
const REVISION: &str = "038773d84178649586cfdf228df3e2a0838a5180";
const FILE: &str = "tokenizer.json";
/// Env var naming a local tokenizer.json that replaces the hf-hub lookup.
pub const TOKENIZER_PATH_ENV: &str = "STRATA_TOKENIZER_PATH";

/// Path of bge-m3's tokenizer.json: `STRATA_TOKENIZER_PATH` if set, else the HF cache.
/// Blocking: call it off the async executor.
pub fn bge_m3_tokenizer_path() -> Result<PathBuf> {
    match override_path(std::env::var_os(TOKENIZER_PATH_ENV))? {
        Some(path) => Ok(path),
        None => hf_hub_path(),
    }
}

/// `Ok(None)` when the override is unset or empty (empty counts as unset, like every `STRATA_*`
/// var); otherwise the path, after checking it opens as a readable regular file. A missing or
/// unreadable file is a config error naming the variable, never a silent fall back to the network.
fn override_path(value: Option<OsString>) -> Result<Option<PathBuf>> {
    let Some(value) = value.filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    let path = PathBuf::from(value);
    let unusable = |why: String| {
        StrataError::Config(format!("{TOKENIZER_PATH_ENV}={}: {why}", path.display()))
    };
    let meta = std::fs::metadata(&path).map_err(|e| unusable(e.to_string()))?;
    if !meta.is_file() {
        return Err(unusable("not a regular file".into()));
    }
    File::open(&path).map_err(|e| unusable(format!("not readable: {e}")))?;
    Ok(Some(path))
}

fn hf_hub_path() -> Result<PathBuf> {
    let hub = HFClientSync::new().map_err(|e| StrataError::Model(format!("hf-hub client: {e}")))?;
    let repo = hub.model(OWNER, NAME);
    let cached = repo
        .download_file()
        .filename(FILE)
        .revision(REVISION)
        .local_files_only(true)
        .send();
    match cached {
        Ok(path) => Ok(path),
        Err(_) => {
            tracing::info!(
                owner = OWNER,
                name = NAME,
                revision = REVISION,
                "downloading tokenizer"
            );
            repo.download_file()
                .filename(FILE)
                .revision(REVISION)
                .send()
                .map_err(|e| {
                    StrataError::Model(format!("download {OWNER}/{NAME}@{REVISION} {FILE}: {e}"))
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty per-test directory under the system temp dir.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "strata-tokenizer-{tag}-{}-{}",
            std::process::id(),
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn unset_or_empty_override_falls_through_to_hf_hub() {
        assert!(override_path(None).unwrap().is_none());
        assert!(override_path(Some(OsString::new())).unwrap().is_none());
    }

    #[test]
    fn existing_file_is_used_as_is() {
        let dir = scratch_dir("ok");
        let file = dir.join("tokenizer.json");
        std::fs::write(&file, "{}").unwrap();
        let got = override_path(Some(file.clone().into_os_string())).unwrap();
        assert_eq!(got, Some(file));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_file_is_a_config_error_naming_the_variable() {
        let dir = scratch_dir("missing");
        let file = dir.join("absent.json");
        let err = override_path(Some(file.clone().into_os_string())).unwrap_err();
        assert!(matches!(err, StrataError::Config(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains(TOKENIZER_PATH_ENV), "{msg}");
        assert!(msg.contains(&file.display().to_string()), "{msg}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn directory_is_a_config_error_naming_the_variable() {
        let dir = scratch_dir("dir");
        let err = override_path(Some(dir.clone().into_os_string())).unwrap_err();
        assert!(matches!(err, StrataError::Config(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains(TOKENIZER_PATH_ENV), "{msg}");
        assert!(msg.contains("not a regular file"), "{msg}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
