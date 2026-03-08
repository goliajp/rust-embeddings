use std::path::PathBuf;

use crate::error::{Error, Result};

/// Download a model file from HuggingFace Hub if not already cached.
/// Returns the local file path.
pub async fn ensure_model_file(hf_repo: &str, hf_filename: &str) -> Result<PathBuf> {
    let api = hf_hub::api::tokio::Api::new()
        .map_err(|e| Error::Other(format!("failed to create hf-hub api: {e}")))?;

    let repo = api.model(hf_repo.to_string());
    let path = repo
        .get(hf_filename)
        .await
        .map_err(|e| Error::Other(format!("failed to download {hf_repo}/{hf_filename}: {e}")))?;

    Ok(path)
}
