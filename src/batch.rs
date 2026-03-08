use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::backoff::BackoffConfig;
use crate::client::{Client, EmbedResult};
use crate::error::Result;
use crate::provider::InputType;
use crate::usage::Usage;

/// Builder for concurrent batch embedding.
///
/// Created by [`Client::embed_batch`]. Splits texts into provider-appropriate
/// chunks and processes them concurrently with a semaphore-limited concurrency.
pub struct BatchBuilder<'a> {
    client: &'a Client,
    texts: Vec<String>,
    model: Option<String>,
    dimensions: Option<u32>,
    input_type: Option<InputType>,
    concurrency: usize,
    chunk_size: Option<usize>,
    backoff: Option<BackoffConfig>,
    timeout: Duration,
}

impl<'a> BatchBuilder<'a> {
    pub(crate) fn new(client: &'a Client, texts: Vec<String>) -> Self {
        Self {
            client,
            texts,
            model: client.default_model.clone(),
            dimensions: client.default_dimensions,
            input_type: client.default_input_type,
            concurrency: 5,
            chunk_size: None,
            backoff: client.default_backoff.clone(),
            timeout: client.default_timeout,
        }
    }

    /// Set the model for all embeddings in the batch.
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

    /// Set the input type for all embeddings.
    pub fn input_type(self, input_type: InputType) -> Self {
        Self {
            input_type: Some(input_type),
            ..self
        }
    }

    /// Set the maximum number of concurrent API requests.
    ///
    /// Default is 5.
    pub fn concurrency(self, n: usize) -> Self {
        Self {
            concurrency: n.max(1),
            ..self
        }
    }

    /// Override the chunk size (texts per API request).
    ///
    /// By default, this is inferred from the provider's maximum batch size.
    pub fn chunk_size(self, size: usize) -> Self {
        Self {
            chunk_size: Some(size.max(1)),
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

    /// Set the overall request timeout per chunk.
    pub fn timeout(self, timeout: Duration) -> Self {
        Self { timeout, ..self }
    }

    async fn execute(self) -> Result<EmbedResult> {
        let chunk_size = self
            .chunk_size
            .unwrap_or_else(|| self.client.provider.max_batch_size());

        if self.texts.is_empty() {
            return Ok(EmbedResult {
                embeddings: Vec::new(),
                usage: Usage::default(),
                model: self
                    .model
                    .unwrap_or_else(|| self.client.provider.default_model().to_string()),
            });
        }

        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.concurrency));
        let client = self.client;
        let model: Option<Arc<str>> = self.model.map(|s| Arc::from(s.as_str()));
        let dimensions = self.dimensions;
        let input_type = self.input_type;
        let backoff = self.backoff.map(Arc::new);
        let timeout = self.timeout;

        let chunks: Vec<Vec<String>> = {
            let mut texts = self.texts;
            let mut result = Vec::with_capacity(texts.len().div_ceil(chunk_size));
            while !texts.is_empty() {
                let at = chunk_size.min(texts.len());
                let rest = texts.split_off(at);
                result.push(texts);
                texts = rest;
            }
            result
        };

        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let sem = semaphore.clone();
                let model = model.clone();
                let backoff = backoff.clone();

                async move {
                    let _permit = sem
                        .acquire_owned()
                        .await
                        .map_err(|_| crate::error::Error::Other("semaphore closed".into()))?;
                    let mut builder = client.embed(chunk);
                    if let Some(ref m) = model {
                        builder = builder.model(m.as_ref());
                    }
                    if let Some(d) = dimensions {
                        builder = builder.dimensions(d);
                    }
                    if let Some(it) = input_type {
                        builder = builder.input_type(it);
                    }
                    if let Some(ref b) = backoff {
                        builder = builder.retry_backoff((**b).clone());
                    }
                    builder = builder.timeout(timeout);
                    builder.await
                }
            })
            .collect();

        let results = futures::future::join_all(handles).await;

        // merge results in order
        let mut all_embeddings = Vec::new();
        let mut total_usage = Usage::default();
        let mut result_model = String::new();

        for result in results {
            let embed_result = result?;
            all_embeddings.extend(embed_result.embeddings);
            total_usage.accumulate(embed_result.usage.total_tokens);
            if result_model.is_empty() {
                result_model = embed_result.model;
            }
        }

        Ok(EmbedResult {
            embeddings: all_embeddings,
            usage: total_usage,
            model: result_model,
        })
    }
}

impl<'a> IntoFuture for BatchBuilder<'a> {
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
    fn builder_inherits_client_defaults() {
        let client = Client::openai("key")
            .with_model("text-embedding-3-large")
            .with_dimensions(256)
            .with_input_type(InputType::SearchDocument);

        let builder = BatchBuilder::new(&client, vec!["a".into(), "b".into()]);
        assert_eq!(builder.concurrency, 5);
        assert_eq!(builder.model.as_deref(), Some("text-embedding-3-large"));
        assert_eq!(builder.dimensions, Some(256));
        assert_eq!(builder.input_type, Some(InputType::SearchDocument));
        assert_eq!(builder.texts.len(), 2);
    }

    #[test]
    fn builder_overrides() {
        let client = Client::openai("key");

        let builder = BatchBuilder::new(&client, vec!["a".into()])
            .model("text-embedding-3-large")
            .dimensions(512)
            .input_type(InputType::SearchQuery)
            .concurrency(10)
            .chunk_size(50);

        assert_eq!(builder.model.as_deref(), Some("text-embedding-3-large"));
        assert_eq!(builder.dimensions, Some(512));
        assert_eq!(builder.input_type, Some(InputType::SearchQuery));
        assert_eq!(builder.concurrency, 10);
        assert_eq!(builder.chunk_size, Some(50));
    }

    #[test]
    fn concurrency_minimum_one() {
        let client = Client::openai("key");
        let builder = BatchBuilder::new(&client, vec![]).concurrency(0);
        assert_eq!(builder.concurrency, 1);
    }

    #[test]
    fn chunk_size_minimum_one() {
        let client = Client::openai("key");
        let builder = BatchBuilder::new(&client, vec![]).chunk_size(0);
        assert_eq!(builder.chunk_size, Some(1));
    }

    #[test]
    fn builder_backoff_and_timeout() {
        let client = Client::openai("key")
            .with_retry_backoff(BackoffConfig::default())
            .with_timeout(Duration::from_secs(30));

        let builder = BatchBuilder::new(&client, vec!["a".into()]);
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

    #[tokio::test]
    async fn empty_batch_returns_empty() {
        let client = Client::openai("key");
        let result = client.embed_batch(vec![]).await.unwrap();
        assert!(result.embeddings.is_empty());
        assert_eq!(result.usage.total_tokens, 0);
    }
}
