//! # embedrs
//!
//! One async client for text embeddings: the OpenAI, Cohere, Gemini, Voyage,
//! Jina and Mistral APIs, and four small BERT models run locally on the CPU
//! (feature `local`). Every backend returns the same [`EmbedResult`].
//!
//! ```rust,no_run
//! # async fn run() -> embedrs::Result<()> {
//! let client = embedrs::cloud("sk-...");
//! let result = client.embed(vec!["hello world".into()]).await?;
//! println!("dimensions: {}", result.embeddings[0].len());
//! # Ok(())
//! # }
//! ```
//!
//! With the `local` feature, [`local()`] runs `all-MiniLM-L6-v2` in-process;
//! the weights (91 MB) are downloaded into the Hugging Face cache on first use.
//!
//! [`Client::embed_batch`] splits large inputs into requests the provider
//! accepts and sends them concurrently. [`Client::with_fallback`] chains
//! providers, [`Client::with_retry_backoff`] retries transient failures, and
//! the `cost-tracking` feature fills [`Usage::cost`] from `tiktoken`'s price
//! table.
//!
//! ## Errors
//!
//! - [`Error::Api`]: the provider answered with a non-2xx status
//! - [`Error::Timeout`]: the request exceeded the configured timeout
//! - [`Error::Http`]: network-level failure
//! - [`Error::Json`]: the response body could not be parsed
//! - [`Error::InputTooLarge`]: more texts than the provider accepts in one request
//!
//! ## Similarity
//!
//! ```rust
//! let a = vec![1.0, 0.0, 0.0];
//! let b = vec![0.0, 1.0, 0.0];
//! let sim = embedrs::cosine_similarity(&a, &b);
//! assert!(sim.abs() < 1e-6);
//! ```

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod backoff;
pub mod batch;
pub mod client;
pub mod error;
#[cfg(feature = "local")]
pub mod local;
pub(crate) mod provider;
mod request;
pub mod similarity;
pub mod usage;

pub use backoff::BackoffConfig;
pub use client::{Client, EmbedResult};
pub use error::{Error, Result};
pub use provider::InputType;
pub use similarity::{cosine_similarity, dot_product, euclidean_distance};
pub use usage::Usage;

/// Create a local embedding client for `all-MiniLM-L6-v2` (384 dimensions,
/// texts truncated at 256 tokens).
///
/// The weights (91 MB) are downloaded from the Hugging Face Hub on first use
/// into the Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`).
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

/// Create a cloud embedding client for OpenAI `text-embedding-3-small`
/// (1536 dimensions, $0.02 per 1M tokens).
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

// compile the README examples as doctests
#[cfg(all(doctest, feature = "local", feature = "cost-tracking"))]
#[doc = include_str!("../README.md")]
struct ReadmeEn;
#[cfg(all(doctest, feature = "local", feature = "cost-tracking"))]
#[doc = include_str!("../README.ja.md")]
struct ReadmeJa;
#[cfg(all(doctest, feature = "local", feature = "cost-tracking"))]
#[doc = include_str!("../README.zh-CN.md")]
struct ReadmeZh;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::{
        BackoffConfig, Client, EmbedResult, Error, InputType, Result, Usage, cosine_similarity,
        dot_product, euclidean_distance,
    };
}
