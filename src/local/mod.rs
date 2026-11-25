//! Local on-device embedding inference (candle + bundled model registry).

mod bert;
mod download;
mod hub;
mod inference;
mod model;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};

pub(crate) use inference::InferenceEngine;
pub(crate) use model::default_model;
pub use model::{ModelDefinition, PoolingStrategy};

use crate::error::{Error, Result};
use crate::input::{InputType, Inputs, Truncate};
use crate::request::Plan;
use crate::result::EmbedResult;
use crate::usage::Usage;

/// Get a model definition by name. Returns `None` if the model is not recognized.
pub fn get_model(name: &str) -> Option<&'static ModelDefinition> {
    model::get_model(name)
}

/// List all available local model names.
pub fn list_models() -> &'static [&'static str] {
    model::list_models()
}

/// A failure inside model loading or inference; the model name is filled in by `embed`.
pub(crate) fn fail(message: String) -> Error {
    Error::Model {
        model: String::new(),
        message,
    }
}

fn named(e: Error, name: &str) -> Error {
    match e {
        Error::Model { model, message } if model.is_empty() => Error::Model {
            model: name.to_string(),
            message,
        },
        e => e,
    }
}

/// The model a local client was built for, and its loaded engine once used.
#[derive(Clone)]
pub(crate) struct Handle {
    def: &'static ModelDefinition,
    engine: Arc<tokio::sync::OnceCell<Arc<InferenceEngine>>>,
}

impl Handle {
    pub(crate) fn new(def: &'static ModelDefinition) -> Self {
        Self {
            def,
            engine: Arc::new(tokio::sync::OnceCell::new()),
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        self.def.name
    }
}

impl crate::Client {
    /// A client for a bundled local model: `"all-MiniLM-L6-v2"`,
    /// `"all-MiniLM-L12-v2"`, `"bge-small-en-v1.5"` or `"gte-small"`.
    ///
    /// The weights are downloaded into the Hugging Face cache on first use;
    /// the tokenizer is compiled in.
    ///
    /// ```rust,no_run
    /// # async fn run() -> embedrs::Result<()> {
    /// let client = embedrs::Client::local("all-MiniLM-L6-v2")?;
    /// let result = client.embed(["hello world"]).await?;
    /// assert_eq!(result.dim(), 384);
    /// # Ok(())
    /// # }
    /// ```
    pub fn local(model: &str) -> Result<Self> {
        let def = get_model(model).ok_or_else(|| Error::UnknownModel(model.to_string()))?;
        Ok(Self::from_local_model(def))
    }

    pub(crate) fn from_local_model(def: &'static ModelDefinition) -> Self {
        Self::with_backend(crate::client::Backend::Local(Handle::new(def)), None)
    }
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

/// Embed `inputs` with the model `p.model` names (the handle's own or another bundled one).
pub(crate) async fn embed(h: &Handle, p: &Plan, inputs: &Inputs<'_>) -> Result<EmbedResult> {
    let def = if p.model == h.def.name {
        h.def
    } else {
        get_model(&p.model).ok_or_else(|| Error::UnknownModel(p.model.clone()))?
    };
    embed_with(h, def, p, inputs)
        .await
        .map_err(|e| named(e, def.name))
}

async fn embed_with(
    h: &Handle,
    def: &'static ModelDefinition,
    p: &Plan,
    inputs: &Inputs<'_>,
) -> Result<EmbedResult> {
    if let Some(d) = p.dimensions
        && d as usize != def.hidden_size
    {
        return Err(Error::unsupported(
            def.name,
            format!(
                "dimensions({d}); its vectors have {} dimensions",
                def.hidden_size
            ),
        ));
    }
    if p.truncate == Truncate::Error {
        return Err(Error::unsupported(def.name, "truncate(Error)"));
    }
    if inputs.is_empty() {
        return Ok(EmbedResult::from_flat(
            Vec::new(),
            0,
            def.name,
            Usage::new(Some(0), None),
        ));
    }
    let engine = if std::ptr::eq(def, h.def) {
        h.engine
            .get_or_try_init(|| shared_engine(def))
            .await?
            .clone()
    } else {
        shared_engine(def).await?
    };
    let prefix = match p.input_type {
        Some(InputType::SearchQuery) => model::query_prefix(def),
        _ => None,
    };
    // spawn_blocking needs owned data
    let texts: Vec<String> = inputs.iter().map(str::to_string).collect();
    let (embeddings, total_tokens) =
        tokio::task::spawn_blocking(move || engine.encode(&texts, prefix))
            .await
            .map_err(|e| fail(format!("inference task failed: {e}")))??;
    let dim = def.hidden_size;
    let mut data = Vec::with_capacity(embeddings.len() * dim);
    for v in &embeddings {
        data.extend_from_slice(v);
    }
    Ok(EmbedResult::from_flat(
        data,
        dim,
        def.name,
        Usage::new(Some(total_tokens as u64), None),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::Plan;

    fn plan(model: &str, dimensions: Option<u32>) -> Plan {
        Plan {
            model: model.into(),
            dimensions,
            input_type: None,
            truncate: Truncate::Provider,
            backoff: None,
        }
    }

    #[test]
    fn known_and_unknown_models() {
        let def = get_model("all-MiniLM-L6-v2").unwrap();
        assert_eq!(def.hidden_size, 384);
        assert!(get_model("nonexistent-model").is_none());
        assert!(list_models().contains(&"all-MiniLM-L6-v2"));
    }

    #[test]
    fn engine_is_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<InferenceEngine>();
    }

    #[tokio::test]
    async fn foreign_model_is_rejected_before_loading() {
        let h = Handle::new(default_model());
        let texts = Inputs::from(["hi"]);
        let err = embed(&h, &plan("text-embedding-3-large", None), &texts)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::UnknownModel(ref m) if m == "text-embedding-3-large"));
        assert!(h.engine.get().is_none());
    }

    #[tokio::test]
    async fn unsupported_dimensions_are_rejected_before_loading() {
        let h = Handle::new(default_model());
        let texts = Inputs::from(["hi"]);
        let err = embed(&h, &plan(h.name(), Some(64)), &texts)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("384") && msg.contains("dimensions(64)"),
            "{msg}"
        );
        assert!(h.engine.get().is_none());
    }

    #[tokio::test]
    async fn native_dimensions_and_empty_input_need_no_model() {
        let h = Handle::new(default_model());
        let empty: Vec<String> = vec![];
        let r = embed(&h, &plan(h.name(), Some(384)), &Inputs::from(empty))
            .await
            .unwrap();
        assert!(r.is_empty());
        assert_eq!(r.usage.total_tokens, Some(0));
        assert_eq!(r.model, h.name());
        assert!(h.engine.get().is_none());
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

    #[test]
    fn model_errors_get_the_model_name() {
        let e = named(fail("boom".into()), "gte-small");
        assert_eq!(e.to_string(), "local model gte-small: boom");
    }
}
