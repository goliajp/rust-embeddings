//! Unified [`Client`] facade over cloud providers and local inference.

use std::time::Duration;

use crate::backoff::BackoffConfig;
#[cfg(feature = "local")]
use crate::error::{Error, Result};
use crate::provider::{InputType, ProviderKind};
pub use crate::request::EmbedBuilder;
use crate::usage::Usage;

/// Result of a successful embedding request.
#[derive(Debug, Clone)]
pub struct EmbedResult {
    /// The embedding vectors, one per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// Token usage information.
    pub usage: Usage,
    /// The model used for embedding.
    pub model: String,
}

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Embedding client for one provider, with optional fallbacks.
///
/// Supports OpenAI, Cohere, Gemini, Voyage, Jina, Mistral, API-compatible
/// servers, and local models (feature `local`). [`crate::cloud()`] and
/// [`crate::local()`] build the default clients.
#[derive(Clone)]
pub struct Client {
    pub(crate) http: reqwest::Client,
    pub(crate) provider: ProviderKind,
    pub(crate) default_model: Option<String>,
    pub(crate) default_dimensions: Option<u32>,
    pub(crate) default_input_type: Option<InputType>,
    pub(crate) default_backoff: Option<BackoffConfig>,
    pub(crate) default_timeout: Duration,
    pub(crate) fallbacks: Vec<Client>,
}

impl Client {
    fn new_with_provider(provider: ProviderKind) -> Self {
        Self {
            http: reqwest::Client::new(),
            provider,
            default_model: None,
            default_dimensions: None,
            default_input_type: None,
            default_backoff: None,
            default_timeout: DEFAULT_TIMEOUT,
            fallbacks: Vec::new(),
        }
    }

    /// Create a client for OpenAI embedding models.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::OpenAi {
            api_key: api_key.into(),
            base_url: crate::provider::OPENAI_BASE_URL.into(),
        })
    }

    /// Create a client for any OpenAI-compatible embedding API.
    pub fn openai_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::OpenAi {
            api_key: api_key.into(),
            base_url: base_url.into(),
        })
    }

    /// Create a client for Cohere embedding models.
    pub fn cohere(api_key: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Cohere {
            api_key: api_key.into(),
            base_url: "https://api.cohere.com/v2".into(),
        })
    }

    /// Create a client for any Cohere-compatible embedding API.
    pub fn cohere_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Cohere {
            api_key: api_key.into(),
            base_url: base_url.into(),
        })
    }

    /// Create a client for Google Gemini embedding models.
    pub fn gemini(api_key: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Gemini {
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
        })
    }

    /// Create a client for any Gemini-compatible embedding API.
    pub fn gemini_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Gemini {
            api_key: api_key.into(),
            base_url: base_url.into(),
        })
    }

    /// Create a client for Voyage AI embedding models.
    pub fn voyage(api_key: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Voyage {
            api_key: api_key.into(),
            base_url: "https://api.voyageai.com/v1".into(),
        })
    }

    /// Create a client for any Voyage-compatible embedding API.
    pub fn voyage_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Voyage {
            api_key: api_key.into(),
            base_url: base_url.into(),
        })
    }

    /// Create a client for Jina AI embedding models.
    pub fn jina(api_key: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Jina {
            api_key: api_key.into(),
            base_url: "https://api.jina.ai/v1".into(),
        })
    }

    /// Create a client for any Jina-compatible embedding API.
    pub fn jina_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Jina {
            api_key: api_key.into(),
            base_url: base_url.into(),
        })
    }

    /// Create a client for Mistral embedding models (`mistral-embed`, `codestral-embed-2505`).
    pub fn mistral(api_key: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Mistral {
            api_key: api_key.into(),
            base_url: "https://api.mistral.ai/v1".into(),
        })
    }

    /// Create a client for any Mistral-compatible embedding API.
    pub fn mistral_compatible(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new_with_provider(ProviderKind::Mistral {
            api_key: api_key.into(),
            base_url: base_url.into(),
        })
    }

    /// Create a client for local model inference using candle.
    ///
    /// The model weights are downloaded from HuggingFace Hub on first use
    /// and cached locally. The tokenizer is embedded in the binary.
    ///
    /// Available models: `"all-MiniLM-L6-v2"`, `"all-MiniLM-L12-v2"`, `"bge-small-en-v1.5"`, `"gte-small"`
    ///
    /// ```rust,no_run
    /// # async fn run() -> embedrs::Result<()> {
    /// let client = embedrs::Client::local("all-MiniLM-L6-v2")?;
    /// let result = client.embed(vec!["hello world".into()]).await?;
    /// println!("dimensions: {}", result.embeddings[0].len()); // 384
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "local")]
    pub fn local(model_name: &str) -> Result<Self> {
        let model_def = crate::local::get_model(model_name)
            .ok_or_else(|| Error::UnknownModel(model_name.to_string()))?;

        Ok(Self::from_local_model(model_def))
    }

    /// create a client from a known model definition (infallible)
    #[cfg(feature = "local")]
    pub(crate) fn from_local_model(model_def: &'static crate::local::ModelDefinition) -> Self {
        Self::new_with_provider(ProviderKind::Local {
            model_def,
            engine: std::sync::Arc::new(tokio::sync::OnceCell::new()),
        })
        .with_model(model_def.name)
    }

    /// Set the default model for all embedding requests.
    pub fn with_model(self, model: impl Into<String>) -> Self {
        Self {
            default_model: Some(model.into()),
            ..self
        }
    }

    /// Set the default output dimensions.
    pub fn with_dimensions(self, dimensions: u32) -> Self {
        Self {
            default_dimensions: Some(dimensions),
            ..self
        }
    }

    /// Set the default input type for all requests.
    pub fn with_input_type(self, input_type: InputType) -> Self {
        Self {
            default_input_type: Some(input_type),
            ..self
        }
    }

    /// Retry with exponential backoff on 408, 429, 500, 502, 503, 504, 529
    /// and on connection errors.
    pub fn with_retry_backoff(self, config: BackoffConfig) -> Self {
        Self {
            default_backoff: Some(config),
            ..self
        }
    }

    /// Set the overall request timeout. Default: 60 seconds.
    pub fn with_timeout(self, timeout: Duration) -> Self {
        Self {
            default_timeout: timeout,
            ..self
        }
    }

    /// Chain a fallback client. If the primary provider fails with any error
    /// (after its retries, if backoff is set), the request goes to the
    /// fallbacks in order, and the primary's error is returned if all fail.
    ///
    /// A fallback sends its own model and dimensions (set them with
    /// [`Client::with_model`] and [`Client::with_dimensions`] on the fallback),
    /// never the primary's. The input type is shared.
    pub fn with_fallback(mut self, fallback: Client) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    /// Replace the underlying [`reqwest::Client`].
    ///
    /// Use this to share one connection pool across providers (saves
    /// keep-alive sockets when a fallback chain spans multiple hosts),
    /// or to inject custom TLS, proxies, headers, or transport timeouts.
    ///
    /// The replacement is propagated to every fallback **registered up
    /// to this point**; call this after `with_fallback(...)` if you want
    /// the share to cover them.
    ///
    /// The reqwest-level timeout still composes with [`Self::with_timeout`]:
    /// the smaller of the two wins on any given request.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        for fb in &mut self.fallbacks {
            fb.http = http.clone();
        }
        self.http = http;
        self
    }

    /// Begin an embedding request for one or more texts.
    ///
    /// ```rust,no_run
    /// # async fn run() -> embedrs::Result<()> {
    /// let client = embedrs::Client::openai("sk-...");
    /// let result = client.embed(vec!["hello world".into()]).await?;
    /// println!("dimensions: {}", result.embeddings[0].len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed(&self, texts: Vec<String>) -> EmbedBuilder<'_> {
        EmbedBuilder::new(self, texts)
    }

    /// Begin a batch embedding request that splits texts into chunks and processes
    /// them concurrently with configurable concurrency.
    ///
    /// ```rust,no_run
    /// # async fn run() -> embedrs::Result<()> {
    /// let client = embedrs::Client::openai("sk-...");
    /// let texts: Vec<String> = (0..5000).map(|i| format!("text {i}")).collect();
    /// let result = client.embed_batch(texts)
    ///     .concurrency(5)
    ///     .await?;
    /// println!("total embeddings: {}", result.embeddings.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed_batch(&self, texts: Vec<String>) -> crate::batch::BatchBuilder<'_> {
        crate::batch::BatchBuilder::new(self, texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_builder_openai() {
        let client = Client::openai("test-key")
            .with_model("text-embedding-3-large")
            .with_dimensions(256)
            .with_input_type(InputType::SearchDocument);

        assert_eq!(
            client.default_model.as_deref(),
            Some("text-embedding-3-large")
        );
        assert_eq!(client.default_dimensions, Some(256));
        assert_eq!(client.default_input_type, Some(InputType::SearchDocument));
    }

    #[test]
    fn client_openai_compatible() {
        let client = Client::openai_compatible("key", "https://api.deepseek.com/v1");
        match &client.provider {
            ProviderKind::OpenAi { base_url, .. } => {
                assert_eq!(base_url, "https://api.deepseek.com/v1");
            }
            _ => panic!("expected OpenAi provider"),
        }
    }

    #[test]
    fn client_cohere() {
        let client = Client::cohere("key");
        match &client.provider {
            ProviderKind::Cohere { base_url, .. } => {
                assert_eq!(base_url, "https://api.cohere.com/v2");
            }
            _ => panic!("expected Cohere provider"),
        }
    }

    #[test]
    fn client_cohere_compatible() {
        let client = Client::cohere_compatible("key", "https://proxy.example.com/v2");
        match &client.provider {
            ProviderKind::Cohere { base_url, .. } => {
                assert_eq!(base_url, "https://proxy.example.com/v2");
            }
            _ => panic!("expected Cohere provider"),
        }
    }

    #[test]
    fn client_gemini() {
        let client = Client::gemini("key");
        match &client.provider {
            ProviderKind::Gemini { base_url, .. } => {
                assert_eq!(base_url, "https://generativelanguage.googleapis.com/v1beta");
            }
            _ => panic!("expected Gemini provider"),
        }
    }

    #[test]
    fn client_gemini_compatible() {
        let client = Client::gemini_compatible("key", "https://proxy.example.com/v1beta");
        match &client.provider {
            ProviderKind::Gemini { base_url, .. } => {
                assert_eq!(base_url, "https://proxy.example.com/v1beta");
            }
            _ => panic!("expected Gemini provider"),
        }
    }

    #[test]
    fn client_voyage() {
        let client = Client::voyage("key");
        match &client.provider {
            ProviderKind::Voyage { base_url, .. } => {
                assert_eq!(base_url, "https://api.voyageai.com/v1");
            }
            _ => panic!("expected Voyage provider"),
        }
    }

    #[test]
    fn client_voyage_compatible() {
        let client = Client::voyage_compatible("key", "https://proxy.example.com/v1");
        match &client.provider {
            ProviderKind::Voyage { base_url, .. } => {
                assert_eq!(base_url, "https://proxy.example.com/v1");
            }
            _ => panic!("expected Voyage provider"),
        }
    }

    #[test]
    fn client_jina() {
        let client = Client::jina("key");
        match &client.provider {
            ProviderKind::Jina { base_url, .. } => {
                assert_eq!(base_url, "https://api.jina.ai/v1");
            }
            _ => panic!("expected Jina provider"),
        }
    }

    #[test]
    fn client_jina_compatible() {
        let client = Client::jina_compatible("key", "https://proxy.example.com/v1");
        match &client.provider {
            ProviderKind::Jina { base_url, .. } => {
                assert_eq!(base_url, "https://proxy.example.com/v1");
            }
            _ => panic!("expected Jina provider"),
        }
    }

    #[test]
    fn default_timeout_60s() {
        let client = Client::openai("key");
        assert_eq!(client.default_timeout, Duration::from_secs(60));
    }

    #[test]
    fn custom_timeout() {
        let client = Client::openai("key").with_timeout(Duration::from_secs(120));
        assert_eq!(client.default_timeout, Duration::from_secs(120));
    }

    #[test]
    fn client_with_retry_backoff() {
        let client = Client::openai("key").with_retry_backoff(BackoffConfig::default());
        assert!(client.default_backoff.is_some());
    }

    #[test]
    fn embed_builder_defaults() {
        let client = Client::openai("key")
            .with_model("text-embedding-3-large")
            .with_dimensions(256)
            .with_input_type(InputType::SearchQuery);

        let builder = client.embed(vec!["test".into()]);
        assert_eq!(builder.model, None);
        assert_eq!(builder.dimensions, None);
        assert_eq!(builder.input_type, None);
        assert_eq!(builder.texts.len(), 1);
    }

    #[test]
    fn embed_builder_overrides() {
        let client = Client::openai("key");
        let builder = client
            .embed(vec!["test".into()])
            .model("text-embedding-3-large")
            .dimensions(512)
            .input_type(InputType::SearchDocument)
            .timeout(Duration::from_secs(30));

        assert_eq!(builder.model.as_deref(), Some("text-embedding-3-large"));
        assert_eq!(builder.dimensions, Some(512));
        assert_eq!(builder.input_type, Some(InputType::SearchDocument));
        assert_eq!(builder.timeout, Duration::from_secs(30));
    }

    #[test]
    fn embed_builder_backoff() {
        let client = Client::openai("key")
            .with_retry_backoff(BackoffConfig::default())
            .with_timeout(Duration::from_secs(30));

        let builder = client.embed(vec!["test".into()]);
        assert!(builder.backoff.is_none());
        assert_eq!(builder.timeout, Duration::from_secs(30));

        let builder = builder
            .retry_backoff(BackoffConfig {
                max_http_retries: 5,
                ..Default::default()
            })
            .timeout(Duration::from_secs(90));
        assert_eq!(builder.backoff.as_ref().unwrap().max_http_retries, 5);
        assert_eq!(builder.timeout, Duration::from_secs(90));
    }

    #[test]
    fn client_with_fallback() {
        let client = Client::openai("key").with_fallback(Client::cohere("cohere-key"));
        assert_eq!(client.fallbacks.len(), 1);
    }

    #[test]
    fn client_with_multiple_fallbacks() {
        let client = Client::openai("key")
            .with_fallback(Client::cohere("cohere-key"))
            .with_fallback(Client::voyage("voyage-key"));
        assert_eq!(client.fallbacks.len(), 2);
    }

    #[test]
    fn embed_result_debug_clone() {
        let result = EmbedResult {
            embeddings: vec![vec![1.0, 2.0, 3.0]],
            usage: Usage {
                total_tokens: 10,
                #[cfg(feature = "cost-tracking")]
                cost: None,
            },
            model: "text-embedding-3-small".into(),
        };
        let cloned = result.clone();
        assert_eq!(cloned.embeddings.len(), 1);
        assert_eq!(cloned.usage.total_tokens, 10);
        assert_eq!(cloned.model, "text-embedding-3-small");
        let debug = format!("{result:?}");
        assert!(debug.contains("EmbedResult"));
    }
}
