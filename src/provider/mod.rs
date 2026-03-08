mod cohere;
mod gemini;
mod jina;
mod openai;
mod voyage;

pub(crate) use cohere::send_cohere;
pub(crate) use gemini::send_gemini;
pub(crate) use jina::send_jina;
pub(crate) use openai::send_openai;
pub(crate) use voyage::send_voyage;

use crate::error::Result;

pub(crate) struct RawEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub total_tokens: u32,
    pub model: String,
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
    OpenAi { api_key: String, base_url: String },
    Cohere { api_key: String, base_url: String },
    Gemini { api_key: String, base_url: String },
    Voyage { api_key: String, base_url: String },
    Jina { api_key: String, base_url: String },
}

impl ProviderKind {
    pub(crate) fn default_model(&self) -> &str {
        match self {
            Self::OpenAi { .. } => "text-embedding-3-small",
            Self::Cohere { .. } => "embed-v4.0",
            Self::Gemini { .. } => "gemini-embedding-001",
            Self::Voyage { .. } => "voyage-3-large",
            Self::Jina { .. } => "jina-embeddings-v3",
        }
    }

    /// Maximum texts per single API request.
    pub(crate) fn max_batch_size(&self) -> usize {
        match self {
            Self::OpenAi { .. } => 2048,
            Self::Cohere { .. } => 96,
            Self::Gemini { .. } => 100,
            Self::Voyage { .. } => 128,
            Self::Jina { .. } => 2048,
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
                send_cohere(http, base_url, api_key, model, texts, input_type).await
            }
            Self::Gemini { api_key, base_url } => {
                send_gemini(
                    http, base_url, api_key, model, texts, dimensions, input_type,
                )
                .await
            }
            Self::Voyage { api_key, base_url } => {
                send_voyage(http, base_url, api_key, model, texts, input_type).await
            }
            Self::Jina { api_key, base_url } => {
                send_jina(
                    http, base_url, api_key, model, texts, dimensions, input_type,
                )
                .await
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
        assert_eq!(provider.default_model(), "voyage-3-large");
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
