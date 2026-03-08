//! # embedrs
//!
//! Unified cloud embedding API client for Rust.
//!
//! Supports OpenAI, Cohere, Gemini, Voyage, and Jina embedding APIs through
//! a single, consistent interface with automatic batching, retry with backoff,
//! and timeout support.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! # async fn run() -> embedrs::Result<()> {
//! let client = embedrs::Client::openai("sk-...");
//! let result = client.embed(vec!["hello world".into()]).await?;
//! println!("dimensions: {}", result.embeddings[0].len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Batch embedding
//!
//! ```rust,no_run
//! # async fn run() -> embedrs::Result<()> {
//! let client = embedrs::Client::openai("sk-...");
//! let texts: Vec<String> = (0..5000).map(|i| format!("text {i}")).collect();
//! let result = client.embed_batch(texts)
//!     .concurrency(5)
//!     .await?;
//! # Ok(())
//! # }
//! ```
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
pub(crate) mod provider;
pub mod similarity;
pub mod usage;

pub use backoff::BackoffConfig;
pub use client::{Client, EmbedResult};
pub use error::{Error, Result};
pub use provider::InputType;
pub use similarity::{cosine_similarity, dot_product, euclidean_distance};
pub use usage::Usage;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::{
        BackoffConfig, Client, EmbedResult, Error, InputType, Result, Usage, cosine_similarity,
        dot_product, euclidean_distance,
    };
}
