# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](LICENSE)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Unified cloud embedding API client for Rust. One interface for OpenAI, Cohere, Gemini, Voyage, and Jina embedding APIs -- with automatic batching, similarity functions, retry with backoff, and timeout support.

## Features

- **5 providers** -- OpenAI, Cohere, Google Gemini, Voyage AI, Jina AI (plus compatible API variants)
- **Automatic batching** -- splits large input sets into provider-appropriate chunks, processes concurrently
- **Similarity functions** -- cosine similarity, dot product, Euclidean distance
- **Input type hints** -- search document, search query, classification, clustering
- **Configurable dimensions** -- request reduced-dimension embeddings where supported
- **Exponential backoff** -- automatic retry on HTTP 429/503 with jitter
- **Request timeout** -- overall timeout covering retries and backoff
- **Builder pattern** -- ergonomic `IntoFuture`-based API (`client.embed(...).await`)
- **Client defaults** -- set model, dimensions, input type once, override per-request
- **Optional tracing** -- structured logging via `tracing` crate behind a feature flag

## Installation

```toml
[dependencies]
embedrs = "0.1"
```

Or via the command line:

```bash
cargo add embedrs
```

## Quick Start

```rust
use embedrs::prelude::*;

let client = Client::openai("sk-...");

let result = client.embed(vec!["hello world".into()]).await?;

println!("dimensions: {}", result.embeddings[0].len());
println!("tokens: {}", result.usage.total_tokens);
```

## Providers

| Provider | Constructor | Default Model | Max Batch Size |
|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 |
| Google Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 |
| Voyage AI | `Client::voyage(key)` | `voyage-3-large` | 128 |
| Jina AI | `Client::jina(key)` | `jina-embeddings-v3` | 2048 |

Each provider also has a `*_compatible` constructor for proxies or API-compatible services:

```rust
// OpenAI
let client = Client::openai("sk-...");

// Cohere
let client = Client::cohere("co-...");

// Google Gemini
let client = Client::gemini("AIza...");

// Voyage AI
let client = Client::voyage("pa-...");

// Jina AI
let client = Client::jina("jina_...");

// OpenAI-compatible (e.g., Azure, proxies)
let client = Client::openai_compatible("sk-...", "https://your-proxy.com/v1");

// Cohere-compatible
let client = Client::cohere_compatible("key", "https://proxy.example.com/v2");

// Gemini-compatible
let client = Client::gemini_compatible("key", "https://proxy.example.com/v1beta");

// Voyage-compatible
let client = Client::voyage_compatible("key", "https://proxy.example.com/v1");

// Jina-compatible
let client = Client::jina_compatible("key", "https://proxy.example.com/v1");
```

## Batch Embedding

Embed thousands of texts concurrently. Texts are automatically chunked based on the provider's maximum batch size and processed with semaphore-limited concurrency:

```rust
let client = Client::openai("sk-...");

let texts: Vec<String> = (0..5000).map(|i| format!("document {i}")).collect();

let result = client.embed_batch(texts)
    .concurrency(5)       // max concurrent API requests (default: 5)
    .chunk_size(512)       // texts per request (default: provider max)
    .model("text-embedding-3-large")
    .await?;

println!("total embeddings: {}", result.embeddings.len());
println!("total tokens: {}", result.usage.total_tokens);
```

## Input Type

Some providers use input type hints to optimize embeddings for specific use cases:

```rust
use embedrs::InputType;

// for indexing documents
let result = client.embed(docs)
    .input_type(InputType::SearchDocument)
    .await?;

// for search queries
let result = client.embed(queries)
    .input_type(InputType::SearchQuery)
    .await?;
```

Available variants: `SearchDocument`, `SearchQuery`, `Classification`, `Clustering`.

## Dimensions

Request reduced-dimension embeddings where the provider supports it:

```rust
let result = client.embed(vec!["hello".into()])
    .model("text-embedding-3-large")
    .dimensions(256)
    .await?;

assert_eq!(result.embeddings[0].len(), 256);
```

## Similarity Functions

Compute similarity and distance between embedding vectors:

```rust
use embedrs::{cosine_similarity, dot_product, euclidean_distance};

let a = vec![1.0, 0.0, 0.0];
let b = vec![0.0, 1.0, 0.0];

let cos = cosine_similarity(&a, &b);    // 0.0 (orthogonal)
let dot = dot_product(&a, &b);          // 0.0
let dist = euclidean_distance(&a, &b);  // 1.414...
```

## Backoff and Timeout

Enable exponential backoff on HTTP 429/503 errors and set an overall request timeout:

```rust
use std::time::Duration;
use embedrs::BackoffConfig;

let client = Client::openai("sk-...")
    .with_retry_backoff(BackoffConfig::default())  // 500ms base, 30s cap, 3 retries
    .with_timeout(Duration::from_secs(120));        // overall timeout (default: 60s)

// per-request override
let result = client.embed(vec!["hello".into()])
    .retry_backoff(BackoffConfig {
        base_delay: Duration::from_millis(200),
        max_delay: Duration::from_secs(10),
        jitter: true,
        max_http_retries: 5,
    })
    .timeout(Duration::from_secs(30))
    .await?;
```

Without backoff configured, HTTP 429/503 errors fail immediately.

## Client Defaults

Set defaults once, override per-request:

```rust
let client = Client::openai("sk-...")
    .with_model("text-embedding-3-large")
    .with_dimensions(256)
    .with_input_type(InputType::SearchDocument)
    .with_retry_backoff(BackoffConfig::default())
    .with_timeout(Duration::from_secs(120));

// all requests use the defaults above
let a = client.embed(vec!["doc 1".into()]).await?;
let b = client.embed(vec!["doc 2".into()]).await?;

// override for a specific request
let c = client.embed(vec!["query".into()])
    .model("text-embedding-3-small")
    .input_type(InputType::SearchQuery)
    .await?;
```

## Feature Flags

| Feature | Default | Description |
|---|---|---|
| *(none)* | yes | Core embedding client, all 5 providers |
| `tracing` | no | Structured logging via the `tracing` crate |

Enable tracing:

```toml
[dependencies]
embedrs = { version = "0.1", features = ["tracing"] }
```

## License

[MIT](LICENSE)
