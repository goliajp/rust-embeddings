//! [`EmbedBuilder`]: one embedding request with retries and fallback.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::time::Duration;

use crate::backoff::BackoffConfig;
use crate::client::{Client, EmbedResult};
use crate::error::{Error, Result};
use crate::provider::InputType;
use crate::usage::Usage;

/// Builder for configuring an embedding request.
///
/// Created by [`Client::embed`]. Call `.await` to execute the request.
pub struct EmbedBuilder<'a> {
    pub(crate) client: &'a Client,
    pub(crate) texts: Vec<String>,
    // per-request overrides; `None` means "use the client's default"
    pub(crate) model: Option<String>,
    pub(crate) dimensions: Option<u32>,
    pub(crate) input_type: Option<InputType>,
    pub(crate) backoff: Option<BackoffConfig>,
    pub(crate) timeout: Duration,
}

/// What one provider in the chain is asked for.
struct Plan<'c> {
    client: &'c Client,
    model: String,
    dimensions: Option<u32>,
    input_type: Option<InputType>,
    backoff: Option<BackoffConfig>,
}

impl<'a> EmbedBuilder<'a> {
    pub(crate) fn new(client: &'a Client, texts: Vec<String>) -> Self {
        Self {
            client,
            texts,
            model: None,
            dimensions: None,
            input_type: None,
            backoff: None,
            timeout: client.default_timeout,
        }
    }

    /// Override the model for this request.
    ///
    /// Applies to the primary provider only; each fallback uses its own model.
    pub fn model(self, model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..self
        }
    }

    /// Set the output dimensions.
    ///
    /// Applies to the primary provider only; each fallback uses its own
    /// [`Client::with_dimensions`] setting.
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

    /// Retry with exponential backoff on 408, 429, 500, 502, 503, 504, 529
    /// and on connection errors.
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

    fn plan<'c>(&self, client: &'c Client, primary: bool) -> Plan<'c> {
        let (model, dimensions) = if primary {
            (
                self.model.clone().or_else(|| client.default_model.clone()),
                self.dimensions.or(client.default_dimensions),
            )
        } else {
            (client.default_model.clone(), client.default_dimensions)
        };
        Plan {
            client,
            model: model.unwrap_or_else(|| client.provider.default_model().to_string()),
            dimensions,
            input_type: self.input_type.or(client.default_input_type),
            backoff: self.backoff.or(client.default_backoff),
        }
    }

    async fn execute(self) -> Result<EmbedResult> {
        let timeout = self.timeout;
        #[cfg(feature = "tracing")]
        {
            let model = self.plan(self.client, true).model;
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
        let primary_err = match self.try_provider(self.plan(self.client, true)).await {
            Ok(ok) => return Ok(ok),
            Err(e) => e,
        };
        #[cfg(feature = "tracing")]
        if !self.client.fallbacks.is_empty() {
            tracing::info!(
                from_provider = self.client.provider.kind_name(),
                error = %primary_err,
                "primary provider failed, trying fallbacks"
            );
        }
        for fallback in &self.client.fallbacks {
            if let Ok(ok) = self.try_provider(self.plan(fallback, false)).await {
                return Ok(ok);
            }
        }
        Err(primary_err)
    }

    async fn try_provider(&self, plan: Plan<'_>) -> Result<EmbedResult> {
        let provider = &plan.client.provider;
        let max_batch = provider.max_batch_size();
        if self.texts.len() > max_batch {
            return Err(Error::InputTooLarge(self.texts.len(), max_batch));
        }
        let max_retries = plan.backoff.map(|b| b.max_http_retries).unwrap_or(0);

        let mut attempt = 0;
        loop {
            let result = provider
                .send(
                    &plan.client.http,
                    &plan.model,
                    &self.texts,
                    plan.dimensions,
                    plan.input_type,
                )
                .await;
            let err = match result {
                Ok(raw) => {
                    if raw.embeddings.len() != self.texts.len() {
                        return Err(Error::Other(format!(
                            "{} returned {} vectors for {} texts",
                            provider.kind_name(),
                            raw.embeddings.len(),
                            self.texts.len()
                        )));
                    }
                    #[cfg(feature = "tracing")]
                    tracing::info!(
                        total_tokens = raw.total_tokens,
                        embeddings = raw.embeddings.len(),
                        "embedding succeeded"
                    );
                    return Ok(EmbedResult {
                        embeddings: raw.embeddings,
                        usage: usage_for(&plan.model, raw.total_tokens),
                        model: raw.model,
                    });
                }
                Err(e) => e,
            };
            let (Some(backoff), Some(retry_after)) = (plan.backoff, retry_hint(&err)) else {
                return Err(err);
            };
            if attempt >= max_retries {
                return Err(err);
            }
            let delay = retry_after.unwrap_or_else(|| backoff.delay_for(attempt));
            #[cfg(feature = "tracing")]
            tracing::warn!(
                attempt,
                delay_ms = delay.as_millis() as u64,
                retry_after_header = retry_after.is_some(),
                error = %err,
                "retryable error, backing off"
            );
            tokio::time::sleep(delay).await;
            attempt += 1;
        }
    }
}

/// `Some(hint)` when the error is worth retrying; the hint is the provider's
/// `Retry-After`, if it sent one.
pub(crate) fn retry_hint(err: &Error) -> Option<Option<Duration>> {
    match err {
        Error::Api {
            status: 408 | 429 | 500 | 502 | 503 | 504 | 529,
            retry_after,
            ..
        } => Some(*retry_after),
        Error::Http(e) if e.is_connect() || e.is_timeout() => Some(None),
        _ => None,
    }
}

/// Usage for a response; a count the provider did not report is 0 tokens and
/// no cost, never a cost of zero.
pub(crate) fn usage_for(model: &str, tokens: Option<u32>) -> Usage {
    #[cfg(not(feature = "cost-tracking"))]
    let _ = model;
    Usage {
        total_tokens: tokens.unwrap_or(0),
        #[cfg(feature = "cost-tracking")]
        cost: tokens.and_then(|t| tiktoken::pricing::estimate_cost(model, t as u64, 0)),
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

    fn api(status: u16, retry_after: Option<Duration>) -> Error {
        Error::Api {
            status,
            message: String::new(),
            retry_after,
        }
    }

    #[test]
    fn retryable_statuses() {
        for s in [408, 429, 500, 502, 503, 504, 529] {
            assert_eq!(retry_hint(&api(s, None)), Some(None), "{s}");
        }
        for s in [400, 401, 403, 404, 422] {
            assert_eq!(retry_hint(&api(s, None)), None, "{s}");
        }
        let hint = Some(Duration::from_secs(2));
        assert_eq!(retry_hint(&api(429, hint)), Some(hint));
        assert_eq!(retry_hint(&Error::Timeout(Duration::from_secs(1))), None);
    }

    #[test]
    fn primary_plan_uses_overrides_then_client_defaults() {
        let client = Client::openai("k")
            .with_model("text-embedding-3-large")
            .with_dimensions(256);
        let b = client.embed(vec!["t".into()]);
        let plan = b.plan(&client, true);
        assert_eq!(plan.model, "text-embedding-3-large");
        assert_eq!(plan.dimensions, Some(256));

        let b = client.embed(vec!["t".into()]).model("m2").dimensions(64);
        let plan = b.plan(&client, true);
        assert_eq!(plan.model, "m2");
        assert_eq!(plan.dimensions, Some(64));
    }

    #[test]
    fn fallback_plan_uses_its_own_model_and_dimensions() {
        let fallback = Client::cohere("k2");
        let primary = Client::openai("k")
            .with_model("text-embedding-3-large")
            .with_dimensions(256)
            .with_fallback(fallback.clone());
        let b = primary
            .embed(vec!["t".into()])
            .model("text-embedding-3-small")
            .dimensions(512)
            .input_type(InputType::SearchQuery);
        let plan = b.plan(&fallback, false);
        assert_eq!(plan.model, "embed-v4.0");
        assert_eq!(plan.dimensions, None);
        assert_eq!(plan.input_type, Some(InputType::SearchQuery));

        let fallback = Client::cohere("k2")
            .with_model("embed-v5.0-fast")
            .with_dimensions(1024);
        let plan = b.plan(&fallback, false);
        assert_eq!(plan.model, "embed-v5.0-fast");
        assert_eq!(plan.dimensions, Some(1024));
    }

    #[test]
    fn unknown_token_count_has_no_cost() {
        let u = usage_for("text-embedding-3-small", None);
        assert_eq!(u.total_tokens, 0);
        #[cfg(feature = "cost-tracking")]
        assert_eq!(u.cost, None);
        let u = usage_for("text-embedding-3-small", Some(10));
        assert_eq!(u.total_tokens, 10);
        #[cfg(feature = "cost-tracking")]
        assert!(u.cost.unwrap() > 0.0);
    }
}
