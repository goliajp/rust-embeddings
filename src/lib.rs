//! # embedrs
//!
//! Unified embedding solution — cloud APIs + local inference through one interface.
//! Opinionated defaults backed by [benchmark data](https://github.com/goliajp/airs/tree/develop/crates/benchrs).
//!
//! ## Design: 用就要好用 (if we build it, it must be great)
//!
//! - **`local()`** → all-MiniLM-L6-v2 (23MB, free, no API key)
//! - **`cloud(key)`** → OpenAI text-embedding-3-small (best discrimination, cheapest)
//! - Both produce the same `EmbedResult` — write code once, switch backends in one line
//!
//! Defaults chosen by 8-dimension benchmark across 8 models. See [benchrs](https://github.com/goliajp/airs/tree/develop/crates/benchrs).
//!
//! ## Quick start
//!
//! ```rust,no_run
//! # async fn run() -> embedrs::Result<()> {
//! // cloud — one key, done
//! let client = embedrs::cloud("sk-...");
//! let result = client.embed(vec!["hello world".into()]).await?;
//! println!("dimensions: {}", result.embeddings[0].len());
//! # Ok(())
//! # }
//! ```
//!
//! With the `local` feature enabled:
//!
//! ```rust,ignore
//! // local — zero config, free, 23MB model downloaded on first use
//! let client = embedrs::local();
//! let result = client.embed(vec!["hello world".into()]).await?;
//! ```
//!
//! ## Batch embedding
//!
//! ```rust,no_run
//! # async fn run() -> embedrs::Result<()> {
//! let client = embedrs::cloud("sk-...");
//! let texts: Vec<String> = (0..5000).map(|i| format!("text {i}")).collect();
//! let result = client.embed_batch(texts)
//!     .concurrency(5)
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Provider fallback
//!
//! Chain fallback providers for automatic failover:
//!
//! ```rust,no_run
//! # async fn run() -> embedrs::Result<()> {
//! let client = embedrs::Client::openai("sk-...")
//!     .with_fallback(embedrs::Client::cohere("cohere-key"));
//! // if OpenAI fails, automatically tries Cohere
//! let result = client.embed(vec!["hello".into()]).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Cost tracking
//!
//! Enable the `cost-tracking` feature to get estimated cost per request via `tiktoken` pricing data.
//! The `Usage::cost` field will be `Some(f64)` for models with pricing info, `None` otherwise.
//!
//! ```toml
//! embedrs = { version = "0.2", features = ["cost-tracking"] }
//! ```
//!
//! ## Error handling
//!
//! All fallible operations return [`Result<T>`]. Match on [`Error`] variants for fine-grained control:
//!
//! - [`Error::Api`] — API returned an error status (e.g., 429 rate limit, 401 unauthorized)
//! - [`Error::Timeout`] — request exceeded the configured timeout
//! - [`Error::Http`] — network-level failure
//! - [`Error::Json`] — response body could not be parsed
//! - [`Error::InputTooLarge`] — input exceeded the provider's batch size limit
//!
//! ## Similarity
//!
//! ```rust
//! let a = vec![1.0, 0.0, 0.0];
//! let b = vec![0.0, 1.0, 0.0];
//! let sim = embedrs::cosine_similarity(&a, &b);
//! assert!(sim.abs() < 1e-6);
//! ```

pub mod backoff;
pub mod batch;
pub mod client;
pub mod error;
#[cfg(feature = "local")]
pub mod local;
pub(crate) mod provider;
pub mod similarity;
pub mod usage;

pub use backoff::BackoffConfig;
pub use client::{Client, EmbedResult};
pub use error::{Error, Result};
pub use provider::InputType;
pub use similarity::{cosine_similarity, dot_product, euclidean_distance};
pub use usage::Usage;

/// Create a local embedding client with the recommended default model (all-MiniLM-L6-v2).
///
/// 23MB model, 384 dimensions, free, no API key needed.
/// Model weights downloaded from HuggingFace Hub on first use and cached locally.
///
/// Backed by benchrs experiment: best clustering separation (8.73x), 100% retrieval,
/// EN ρ=0.92, and the only model small enough for app embedding (<50MB).
///
/// ```rust,no_run
/// # async fn run() -> embedrs::Result<()> {
/// let client = embedrs::local();
/// let result = client.embed(vec!["hello world".into()]).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "local")]
pub fn local() -> Client {
    let model_def = local::default_model();
    Client::from_local_model(model_def)
}

/// Create a cloud embedding client with the recommended default provider (OpenAI text-embedding-3-small).
///
/// 1536 dimensions, best discrimination gap (0.58), 100% retrieval, balanced multilingual,
/// cheapest cloud option at $0.02/1M tokens.
///
/// Backed by benchrs experiment: best discrimination means dissimilar texts get cosine ≈ 0.09
/// (closest to zero), making similarity thresholds reliable.
///
/// ```rust,no_run
/// # async fn run() -> embedrs::Result<()> {
/// let client = embedrs::cloud("sk-...");
/// let result = client.embed(vec!["hello world".into()]).await?;
/// # Ok(())
/// # }
/// ```
pub fn cloud(api_key: impl Into<String>) -> Client {
    Client::openai(api_key)
}

#[cfg(test)]
mod tests {
    #[test]
    fn cloud_creates_openai_client() {
        let client = crate::cloud("test-key");
        match &client.provider {
            crate::provider::ProviderKind::OpenAi { base_url, .. } => {
                assert_eq!(base_url, "https://api.openai.com/v1");
            }
            _ => panic!("expected OpenAi provider"),
        }
    }

    #[cfg(feature = "local")]
    #[test]
    fn local_creates_default_client() {
        let client = crate::local();
        assert_eq!(client.default_model.as_deref(), Some("all-MiniLM-L6-v2"));
        match &client.provider {
            crate::provider::ProviderKind::Local { model_def, .. } => {
                assert_eq!(model_def.name, "all-MiniLM-L6-v2");
            }
            _ => panic!("expected Local provider"),
        }
    }

    #[cfg(feature = "local")]
    #[test]
    fn local_unknown_model_returns_error() {
        let result = crate::Client::local("nonexistent");
        assert!(result.is_err());
        match result.err().unwrap() {
            crate::Error::UnknownModel(name) => assert_eq!(name, "nonexistent"),
            other => panic!("expected UnknownModel, got {other:?}"),
        }
    }
}

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::{
        BackoffConfig, Client, EmbedResult, Error, InputType, Result, Usage, cosine_similarity,
        dot_product, euclidean_distance,
    };
}
