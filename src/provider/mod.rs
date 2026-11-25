mod cohere;
mod gemini;
mod http;
mod jina;
mod mistral;
mod openai;
mod openai_shape;
mod vector;
mod voyage;

pub(crate) use cohere::send_cohere;
pub(crate) use gemini::send_gemini;
pub(crate) use jina::send_jina;
pub(crate) use mistral::send_mistral;
pub(crate) use openai::{OPENAI_BASE_URL, send_openai};
pub(crate) use voyage::send_voyage;

#[cfg(feature = "local")]
use std::sync::Arc;

use crate::error::{Error, Result};

pub(crate) struct RawEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    /// `None` when the provider did not report a count
    pub total_tokens: Option<u32>,
    pub model: String,
}

/// Refuse a requested output width the model is known not to support, instead
/// of sending it and getting vectors of another width back.
pub(crate) fn check_dimensions(
    model: &str,
    dimensions: Option<u32>,
    accepted: Option<bool>,
) -> Result<()> {
    match (dimensions, accepted) {
        (Some(d), Some(false)) => Err(Error::Other(format!(
            "{model} has a fixed output width and does not accept dimensions({d})"
        ))),
        _ => Ok(()),
    }
}

/// The type of input being embedded, used by providers that support it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    /// Document for search indexing.
    SearchDocument,
    /// Query for search retrieval.
    SearchQuery,
    /// Text for classification.
    Classification,
    /// Text for clustering.
    Clustering,
}

#[derive(Clone)]
pub(crate) enum ProviderKind {
    OpenAi {
        api_key: String,
        base_url: String,
    },
    Cohere {
        api_key: String,
        base_url: String,
    },
    Gemini {
        api_key: String,
        base_url: String,
    },
    Voyage {
        api_key: String,
        base_url: String,
    },
    Jina {
        api_key: String,
        base_url: String,
    },
    Mistral {
        api_key: String,
        base_url: String,
    },
    #[cfg(feature = "local")]
    Local {
        model_def: &'static crate::local::ModelDefinition,
        engine: Arc<tokio::sync::OnceCell<Arc<crate::local::InferenceEngine>>>,
    },
}

impl ProviderKind {
    pub(crate) fn default_model(&self) -> &str {
        match self {
            Self::OpenAi { .. } => "text-embedding-3-small",
            Self::Cohere { .. } => "embed-v4.0",
            Self::Gemini { .. } => "gemini-embedding-001",
            Self::Voyage { .. } => "voyage-4-large",
            Self::Jina { .. } => "jina-embeddings-v3",
            Self::Mistral { .. } => "mistral-embed",
            #[cfg(feature = "local")]
            Self::Local { model_def, .. } => model_def.name,
        }
    }

    /// Maximum texts per single API request.
    pub(crate) fn max_batch_size(&self) -> usize {
        match self {
            Self::OpenAi { .. } => 2048,
            Self::Cohere { .. } => 96,
            Self::Gemini { .. } => 100,
            Self::Voyage { .. } => 1000,
            Self::Jina { .. } => 2048,
            Self::Mistral { .. } => 512,
            #[cfg(feature = "local")]
            Self::Local { .. } => 256,
        }
    }

    /// Texts per request when `embed_batch` splits the input. Voyage caps a
    /// request at 120k tokens for its large models, which 1000 long texts exceed.
    pub(crate) fn default_chunk_size(&self) -> usize {
        match self {
            Self::Voyage { .. } => 128,
            _ => self.max_batch_size(),
        }
    }

    /// Returns the provider name for diagnostics and tracing.
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    pub(crate) fn kind_name(&self) -> &'static str {
        match self {
            Self::OpenAi { .. } => "openai",
            Self::Cohere { .. } => "cohere",
            Self::Gemini { .. } => "gemini",
            Self::Voyage { .. } => "voyage",
            Self::Jina { .. } => "jina",
            Self::Mistral { .. } => "mistral",
            #[cfg(feature = "local")]
            Self::Local { .. } => "local",
        }
    }

    pub(crate) async fn send(
        &self,
        http: &reqwest::Client,
        model: &str,
        texts: &[String],
        dimensions: Option<u32>,
        input_type: Option<InputType>,
    ) -> Result<RawEmbedResponse> {
        match self {
            Self::OpenAi { api_key, base_url } => {
                send_openai(http, base_url, api_key, model, texts, dimensions).await
            }
            Self::Cohere { api_key, base_url } => {
                send_cohere(
                    http, base_url, api_key, model, texts, dimensions, input_type,
                )
                .await
            }
            Self::Gemini { api_key, base_url } => {
                send_gemini(
                    http, base_url, api_key, model, texts, dimensions, input_type,
                )
                .await
            }
            Self::Voyage { api_key, base_url } => {
                send_voyage(
                    http, base_url, api_key, model, texts, dimensions, input_type,
                )
                .await
            }
            Self::Jina { api_key, base_url } => {
                send_jina(
                    http, base_url, api_key, model, texts, dimensions, input_type,
                )
                .await
            }
            Self::Mistral { api_key, base_url } => {
                send_mistral(http, base_url, api_key, model, texts, dimensions).await
            }
            #[cfg(feature = "local")]
            Self::Local { model_def, engine } => {
                crate::local::embed(model_def, engine, model, texts, dimensions, input_type).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_openai() {
        let provider = ProviderKind::OpenAi {
            api_key: "key".into(),
            base_url: "url".into(),
        };
        assert_eq!(provider.default_model(), "text-embedding-3-small");
    }

    #[test]
    fn default_model_cohere() {
        let provider = ProviderKind::Cohere {
            api_key: "key".into(),
            base_url: "url".into(),
        };
        assert_eq!(provider.default_model(), "embed-v4.0");
    }

    #[test]
    fn default_model_gemini() {
        let provider = ProviderKind::Gemini {
            api_key: "key".into(),
            base_url: "url".into(),
        };
        assert_eq!(provider.default_model(), "gemini-embedding-001");
    }

    #[test]
    fn default_model_voyage() {
        let provider = ProviderKind::Voyage {
            api_key: "key".into(),
            base_url: "url".into(),
        };
        assert_eq!(provider.default_model(), "voyage-4-large");
    }

    #[test]
    fn default_model_jina() {
        let provider = ProviderKind::Jina {
            api_key: "key".into(),
            base_url: "url".into(),
        };
        assert_eq!(provider.default_model(), "jina-embeddings-v3");
    }

    #[test]
    fn default_model_mistral() {
        let provider = ProviderKind::Mistral {
            api_key: "key".into(),
            base_url: "url".into(),
        };
        assert_eq!(provider.default_model(), "mistral-embed");
        assert_eq!(provider.max_batch_size(), 512);
        assert_eq!(provider.kind_name(), "mistral");
    }

    #[test]
    fn max_batch_sizes() {
        let openai = ProviderKind::OpenAi {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(openai.max_batch_size(), 2048);

        let cohere = ProviderKind::Cohere {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(cohere.max_batch_size(), 96);

        let gemini = ProviderKind::Gemini {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(gemini.max_batch_size(), 100);

        let voyage = ProviderKind::Voyage {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(voyage.max_batch_size(), 1000);
        assert_eq!(voyage.default_chunk_size(), 128);
        assert_eq!(openai.default_chunk_size(), 2048);
    }

    #[test]
    fn known_fixed_width_rejects_dimensions() {
        assert!(check_dimensions("m", Some(256), Some(false)).is_err());
        assert!(check_dimensions("m", None, Some(false)).is_ok());
        assert!(check_dimensions("m", Some(256), None).is_ok());
        assert!(check_dimensions("m", Some(256), Some(true)).is_ok());
    }

    #[test]
    fn kind_names() {
        let openai = ProviderKind::OpenAi {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(openai.kind_name(), "openai");

        let cohere = ProviderKind::Cohere {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(cohere.kind_name(), "cohere");

        let gemini = ProviderKind::Gemini {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(gemini.kind_name(), "gemini");

        let voyage = ProviderKind::Voyage {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(voyage.kind_name(), "voyage");

        let jina = ProviderKind::Jina {
            api_key: "k".into(),
            base_url: "u".into(),
        };
        assert_eq!(jina.kind_name(), "jina");
    }

    #[test]
    fn input_type_debug_clone() {
        let it = InputType::SearchDocument;
        let cloned = it;
        assert_eq!(cloned, InputType::SearchDocument);
        let debug = format!("{it:?}");
        assert!(debug.contains("SearchDocument"));
    }

    #[test]
    fn input_type_variants() {
        assert_ne!(InputType::SearchDocument, InputType::SearchQuery);
        assert_ne!(InputType::Classification, InputType::Clustering);
    }
}
