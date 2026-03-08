use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::time::Duration;

use crate::backoff::BackoffConfig;
use crate::error::{Error, Result};
use crate::provider::{InputType, ProviderKind};
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

/// Unified embedding client — cloud APIs + local inference through one interface.
///
/// Supports OpenAI, Cohere, Gemini, Voyage, Jina, compatible APIs, and local models
/// (behind the `local` feature flag). Use [`crate::cloud()`] or [`crate::local()`]
/// for opinionated defaults backed by benchmark data.
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
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
            base_url: "https://api.openai.com/v1".into(),
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

    /// Enable exponential backoff for retryable HTTP errors (429, 503).
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

    /// Chain a fallback client. If the primary provider fails with a non-retryable
    /// error, the request is retried against fallback providers in order.
    pub fn with_fallback(mut self, fallback: Client) -> Self {
        self.fallbacks.push(fallback);
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
        EmbedBuilder {
            client: self,
            texts,
            model: self.default_model.clone(),
            dimensions: self.default_dimensions,
            input_type: self.default_input_type,
            backoff: self.default_backoff,
            timeout: self.default_timeout,
        }
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

/// Builder for configuring an embedding request.
///
/// Created by [`Client::embed`]. Call `.await` to execute the request.
pub struct EmbedBuilder<'a> {
    client: &'a Client,
    texts: Vec<String>,
    model: Option<String>,
    dimensions: Option<u32>,
    input_type: Option<InputType>,
    backoff: Option<BackoffConfig>,
    timeout: Duration,
}

impl EmbedBuilder<'_> {
    /// Override the model for this request.
    pub fn model(self, model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..self
        }
    }

    /// Set the output dimensions.
    pub fn dimensions(self, dimensions: u32) -> Self {
        Self {
            dimensions: Some(dimensions),
            ..self
        }
    }

    /// Set the input type for this request.
    pub fn input_type(self, input_type: InputType) -> Self {
        Self {
            input_type: Some(input_type),
            ..self
        }
    }

    /// Enable exponential backoff for retryable HTTP errors (429, 503).
    pub fn retry_backoff(self, config: BackoffConfig) -> Self {
        Self {
            backoff: Some(config),
            ..self
        }
    }

    /// Set the overall request timeout for this embedding.
    pub fn timeout(self, timeout: Duration) -> Self {
        Self { timeout, ..self }
    }

    async fn execute(self) -> Result<EmbedResult> {
        let timeout = self.timeout;
        #[cfg(feature = "tracing")]
        {
            let model = self
                .model
                .as_deref()
                .unwrap_or(self.client.provider.default_model())
                .to_string();
            let span = tracing::info_span!(
                "embedrs.embed",
                provider = self.client.provider.kind_name(),
                model = model.as_str(),
                texts = self.texts.len(),
            );
            use tracing::Instrument;
            tokio::time::timeout(timeout, self.execute_inner().instrument(span))
                .await
                .map_err(|_| Error::Timeout(timeout))?
        }
        #[cfg(not(feature = "tracing"))]
        {
            tokio::time::timeout(timeout, self.execute_inner())
                .await
                .map_err(|_| Error::Timeout(timeout))?
        }
    }

    async fn execute_inner(self) -> Result<EmbedResult> {
        // try primary provider
        let result = self.try_provider(self.client).await;

        // on failure, try fallbacks in order
        match result {
            Ok(ok) => Ok(ok),
            Err(primary_err) => {
                #[cfg(feature = "tracing")]
                if !self.client.fallbacks.is_empty() {
                    tracing::info!(
                        from_provider = self.client.provider.kind_name(),
                        error = %primary_err,
                        "primary provider failed, trying fallbacks"
                    );
                }

                for fallback in &self.client.fallbacks {
                    if let Ok(ok) = self.try_provider(fallback).await {
                        return Ok(ok);
                    }
                }
                Err(primary_err)
            }
        }
    }

    async fn try_provider(&self, client: &Client) -> Result<EmbedResult> {
        let max_batch = client.provider.max_batch_size();
        if self.texts.len() > max_batch {
            return Err(Error::InputTooLarge(self.texts.len(), max_batch));
        }

        let model = self
            .model
            .as_deref()
            .unwrap_or(client.provider.default_model());

        let max_http_retries = self
            .backoff
            .as_ref()
            .map(|b| b.max_http_retries)
            .unwrap_or(0);

        for http_attempt in 0..=max_http_retries {
            let result = client
                .provider
                .send(
                    &client.http,
                    model,
                    &self.texts,
                    self.dimensions,
                    self.input_type,
                )
                .await;

            match result {
                Ok(raw) => {
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        total_tokens = raw.total_tokens,
                        embeddings = raw.embeddings.len(),
                        "embedding succeeded"
                    );
                    #[cfg(not(feature = "cost-tracking"))]
                    let usage = Usage {
                        total_tokens: raw.total_tokens,
                    };
                    #[cfg(feature = "cost-tracking")]
                    let usage = Usage {
                        total_tokens: raw.total_tokens,
                        cost: tiktoken::pricing::estimate_cost(model, raw.total_tokens as u64, 0),
                    };
                    return Ok(EmbedResult {
                        embeddings: raw.embeddings,
                        usage,
                        model: raw.model,
                    });
                }
                Err(Error::Api {
                    status,
                    ref message,
                }) if (status == 429 || status == 503) && http_attempt < max_http_retries => {
                    if let Some(ref backoff) = self.backoff {
                        let delay = backoff.delay_for(http_attempt);
                        #[cfg(feature = "tracing")]
                        tracing::warn!(
                            status,
                            attempt = http_attempt,
                            delay_ms = delay.as_millis() as u64,
                            error = message.as_str(),
                            "retryable HTTP error, backing off"
                        );
                        tokio::time::sleep(delay).await;
                    }
                }
                Err(e) => return Err(e),
            }
        }

        unreachable!()
    }
}

impl<'a> IntoFuture for EmbedBuilder<'a> {
    type Output = Result<EmbedResult>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
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
        assert_eq!(builder.model.as_deref(), Some("text-embedding-3-large"));
        assert_eq!(builder.dimensions, Some(256));
        assert_eq!(builder.input_type, Some(InputType::SearchQuery));
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
        assert!(builder.backoff.is_some());
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
