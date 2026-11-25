//! Local on-device embedding inference (candle + bundled model registry).

mod bert;
mod download;
mod inference;
mod model;
mod wordpiece;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};

pub(crate) use inference::InferenceEngine;
pub(crate) use model::default_model;
pub use model::{ModelDefinition, PoolingStrategy};

use crate::error::{Error, Result};
use crate::provider::{InputType, RawEmbedResponse};

/// Get a model definition by name. Returns `None` if the model is not recognized.
pub fn get_model(name: &str) -> Option<&'static ModelDefinition> {
    model::get_model(name)
}

/// List all available local model names.
pub fn list_models() -> &'static [&'static str] {
    model::list_models()
}

type Slot = Arc<tokio::sync::Mutex<Weak<InferenceEngine>>>;

// one loaded engine per model in the process, shared by every client that uses
// it and dropped when the last of them is; the per-model lock also makes
// concurrent first calls wait for a single download and load
static ENGINES: LazyLock<Mutex<HashMap<&'static str, Slot>>> = LazyLock::new(Default::default);

pub(crate) async fn shared_engine(def: &'static ModelDefinition) -> Result<Arc<InferenceEngine>> {
    let slot = ENGINES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(def.name)
        .or_default()
        .clone();
    let mut weak = slot.lock().await;
    if let Some(engine) = weak.upgrade() {
        return Ok(engine);
    }
    let engine = InferenceEngine::load(def).await?;
    *weak = Arc::downgrade(&engine);
    Ok(engine)
}

/// Embed `texts` with the local model named `model`. `client_model` is the
/// model the client was built for; its engine is kept in `cell`.
pub(crate) async fn embed(
    client_model: &'static ModelDefinition,
    cell: &tokio::sync::OnceCell<Arc<InferenceEngine>>,
    model: &str,
    texts: &[String],
    dimensions: Option<u32>,
    input_type: Option<InputType>,
) -> Result<RawEmbedResponse> {
    let def = if model == client_model.name {
        client_model
    } else {
        get_model(model).ok_or_else(|| Error::UnknownModel(model.to_string()))?
    };
    if let Some(d) = dimensions
        && d as usize != def.hidden_size
    {
        return Err(Error::Other(format!(
            "local model {} returns {}-dimensional vectors; dimensions({d}) cannot be honoured",
            def.name, def.hidden_size
        )));
    }
    if texts.is_empty() {
        return Ok(RawEmbedResponse {
            embeddings: Vec::new(),
            total_tokens: Some(0),
            model: def.name.to_string(),
        });
    }

    let engine = if std::ptr::eq(def, client_model) {
        cell.get_or_try_init(|| shared_engine(def)).await?.clone()
    } else {
        shared_engine(def).await?
    };
    let prefix = match input_type {
        Some(InputType::SearchQuery) => model::query_prefix(def),
        _ => None,
    };
    // spawn_blocking needs owned data
    let texts = texts.to_vec();
    let (embeddings, total_tokens) =
        tokio::task::spawn_blocking(move || engine.encode(&texts, prefix))
            .await
            .map_err(|e| Error::Other(format!("inference task failed: {e}")))??;
    Ok(RawEmbedResponse {
        embeddings,
        total_tokens: Some(total_tokens),
        model: def.name.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_known_model() {
        let def = get_model("all-MiniLM-L6-v2");
        assert!(def.is_some());
        let def = def.unwrap();
        assert_eq!(def.name, "all-MiniLM-L6-v2");
        assert_eq!(def.hidden_size, 384);
    }

    #[test]
    fn get_unknown_model() {
        assert!(get_model("nonexistent-model").is_none());
    }

    #[test]
    fn list_available_models() {
        let models = list_models();
        assert!(models.contains(&"all-MiniLM-L6-v2"));
    }

    #[test]
    fn engine_is_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<InferenceEngine>();
    }

    #[tokio::test]
    async fn foreign_model_is_rejected_before_loading() {
        let def = default_model();
        let cell = tokio::sync::OnceCell::new();
        let texts = vec!["hi".to_string()];
        let err = embed(def, &cell, "text-embedding-3-large", &texts, None, None)
            .await
            .err()
            .unwrap();
        assert!(matches!(err, Error::UnknownModel(ref m) if m == "text-embedding-3-large"));
        assert!(cell.get().is_none());
    }

    #[tokio::test]
    async fn unsupported_dimensions_are_rejected_before_loading() {
        let def = default_model();
        let cell = tokio::sync::OnceCell::new();
        let texts = vec!["hi".to_string()];
        let err = embed(def, &cell, def.name, &texts, Some(64), None)
            .await
            .err()
            .unwrap();
        let msg = err.to_string();
        assert!(
            msg.contains("384") && msg.contains("dimensions(64)"),
            "{msg}"
        );
        assert!(cell.get().is_none());
    }

    #[tokio::test]
    async fn native_dimensions_and_empty_input_need_no_model() {
        let def = default_model();
        let cell = tokio::sync::OnceCell::new();
        let raw = embed(def, &cell, def.name, &[], Some(384), None)
            .await
            .unwrap();
        assert!(raw.embeddings.is_empty());
        assert_eq!(raw.total_tokens, Some(0));
        assert_eq!(raw.model, def.name);
        assert!(cell.get().is_none());
    }

    #[test]
    fn only_bge_has_a_query_prefix() {
        for name in list_models() {
            let def = get_model(name).unwrap();
            assert_eq!(
                model::query_prefix(def).is_some(),
                *name == "bge-small-en-v1.5"
            );
        }
    }
}
