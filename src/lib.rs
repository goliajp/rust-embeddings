//! # embedrs
//!
//! One async client for text embeddings: the OpenAI, Cohere, Gemini, Voyage,
//! Jina and Mistral APIs, and four small BERT models run locally on the CPU
//! (feature `local`). Every backend returns the same [`EmbedResult`].
//!
//! ```rust,no_run
//! # #[cfg(feature = "reqwest")]
//! # async fn run() -> embedrs::Result<()> {
//! let client = embedrs::cloud("sk-...");
//! let result = client.embed(["hello world"]).await?;
//! println!("dimensions: {}", result[0].len());
//! # Ok(())
//! # }
//! ```
//!
//! [`Client::embed_batch`] splits large inputs into requests the provider
//! accepts and sends them concurrently. [`Client::with_fallback`] chains
//! providers, [`Client::with_retry`] retries transient failures, and the
//! `cost-tracking` feature fills [`Usage::cost`] from `tiktoken`'s price
//! table.
//!
//! ## Transports
//!
//! The default `reqwest` feature sends requests with reqwest on tokio.
//! Without it embedrs has no dependencies: bring a
//! [`Transport`](transport::Transport) and build clients with
//! [`Client::new`], or build requests with [`Client::prepare`] and send them
//! with any HTTP client.
//!
//! ## Errors
//!
//! - [`Error::Api`]: the provider answered with a non-2xx status
//! - [`Error::Transport`]: no HTTP answer (connect, TLS, I/O)
//! - [`Error::Timeout`]: the call exceeded the configured timeout
//! - [`Error::Decode`]: a 2xx body that is not what the provider documents
//! - [`Error::TooManyInputs`], [`Error::InputTooLong`], [`Error::EmptyInput`],
//!   [`Error::Unsupported`]: refused before sending
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

mod batch;
mod client;
mod endpoint;
mod error;
mod input;
mod json;
#[cfg(feature = "local")]
pub mod local;
mod plan;
mod request;
mod result;
mod retry;
mod rt;
mod similarity;
pub mod transport;
mod usage;
mod wire;

pub use batch::BatchBuilder;
pub use client::{Client, Embed};
pub use endpoint::{Endpoint, Limits};
pub use error::{DecodeError, Error, Result};
pub use input::{InputType, Inputs, Truncate};
pub use request::{EmbedBuilder, Prepared};
pub use result::EmbedResult;
pub use retry::Backoff;
pub use similarity::{cosine_similarity, dot_product, euclidean_distance};
pub use usage::Usage;

/// A client for the bundled local model `all-MiniLM-L6-v2` (384
/// dimensions, texts cut at 256 tokens).
///
/// The weights (91 MB) are downloaded from the Hugging Face Hub on first use
/// into the Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`).
///
/// ```rust,no_run
/// # async fn run() -> embedrs::Result<()> {
/// let client = embedrs::local();
/// let result = client.embed(["hello world"]).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(all(feature = "local", feature = "reqwest"))]
pub fn local() -> Client {
    Client::local(local::models()[0].name).expect("the default model is bundled")
}

/// A client for OpenAI `text-embedding-3-small` (1536 dimensions) over
/// [`ReqwestTransport`](transport::ReqwestTransport).
///
/// ```rust,no_run
/// # async fn run() -> embedrs::Result<()> {
/// let client = embedrs::cloud("sk-...");
/// let result = client.embed(["hello world"]).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "reqwest")]
pub fn cloud(api_key: impl Into<String>) -> Client {
    Client::openai(api_key)
}

/// The common imports.
pub mod prelude {
    pub use crate::{
        Backoff, Client, Embed, EmbedResult, Endpoint, Error, InputType, Result, Truncate, Usage,
        cosine_similarity, dot_product, euclidean_distance,
    };
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "reqwest")]
    #[test]
    fn cloud_is_openai() {
        let c = crate::cloud("k");
        assert_eq!(c.provider(), "openai");
        assert_eq!(c.default_model(), "text-embedding-3-small");
    }

    #[cfg(all(feature = "local", feature = "reqwest"))]
    #[test]
    fn local_default_and_unknown() {
        assert_eq!(crate::local().default_model(), "all-MiniLM-L6-v2");
        assert!(matches!(
            crate::Client::local("nonexistent"),
            Err(crate::Error::UnknownModel(ref n)) if n == "nonexistent"
        ));
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
