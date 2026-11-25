use std::path::PathBuf;

use hf_hub::Cache;
use hf_hub::api::tokio::ApiBuilder;

use crate::error::Result;

/// Return the local path of a model file, downloading it into the Hugging Face
/// cache (`HF_HOME`, default `~/.cache/huggingface`) on first use.
pub(crate) async fn ensure_model_file(hf_repo: &str, hf_filename: &str) -> Result<PathBuf> {
    let cache = Cache::from_env();
    if let Some(path) = cache.model(hf_repo.to_string()).get(hf_filename) {
        return Ok(path);
    }

    // hf-hub gives up on its own per-blob lock after about five seconds, so a
    // second process or client that starts while a download is running fails
    // with `LockAcquisition`. A blocking lock held across the whole download
    // makes the others wait and then find the file in the cache.
    let lock_path = cache
        .path()
        .join(format!(".embedrs-{}.lock", hf_repo.replace('/', "--")));
    let lock = tokio::task::spawn_blocking(move || -> std::io::Result<std::fs::File> {
        if let Some(dir) = lock_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = std::fs::File::create(&lock_path)?;
        file.lock()?;
        Ok(file)
    })
    .await
    .map_err(|e| super::fail(format!("model download task failed: {e}")))?
    .map_err(|e| super::fail(format!("failed to lock the model cache: {e}")))?;

    if let Some(path) = cache.model(hf_repo.to_string()).get(hf_filename) {
        return Ok(path);
    }

    let api = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(|e| super::fail(format!("failed to create hf-hub api: {e}")))?;
    let path = api
        .model(hf_repo.to_string())
        .get(hf_filename)
        .await
        .map_err(|e| super::fail(format!("failed to download {hf_repo}/{hf_filename}: {e}")))?;
    drop(lock);
    Ok(path)
}
