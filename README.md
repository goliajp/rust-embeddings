# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/embedrs?style=flat-square)](https://crates.io/crates/embedrs)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square)](https://www.rust-lang.org)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

Unified embedding for Rust -- 6 cloud providers + local inference through one interface. Opinionated defaults backed by 8-model benchmark data.

## Design philosophy

> If we build it, it must be great -- every default backed by data.

- **`embedrs::local()?`** -- all-MiniLM-L6-v2 (23MB, free, no API key)
- **`embedrs::cloud(key)`** -- OpenAI text-embedding-3-small (best discrimination, cheapest cloud)
- Both produce the same `EmbedResult` -- write code once, switch backends in one line

Defaults chosen by 8-dimension benchmark across 8 models. See [benchrs](https://github.com/goliajp/airs/tree/develop/crates/benchrs) for full methodology.

## Quick Start

```rust
// cloud -- one key, done
let client = embedrs::cloud("sk-...");
let result = client.embed(vec!["hello world".into()]).await?;
println!("dimensions: {}", result.embeddings[0].len());
```

```rust
// local -- zero config, free, 23MB model downloaded on first use
let client = embedrs::local()?;
let result = client.embed(vec!["hello world".into()]).await?;
```

## Installation

```toml
[dependencies]
embedrs = "0.3"

# enable local inference (adds ~23MB model download on first use)
embedrs = { version = "0.3", features = ["local"] }
```

## Feature Flags

| Feature | Default | Description |
|---|---|---|
| *(none)* | yes | Core embedding client, all 5 cloud providers |
| `local` | no | Local inference via candle (all-MiniLM-L6-v2, 23MB) |
| `cost-tracking` | no | Estimated cost per request via `tiktoken` pricing data |
| `tracing` | no | Structured logging via the `tracing` crate |

```toml
[dependencies]
# cloud only
embedrs = "0.3"

# cloud + local inference
embedrs = { version = "0.3", features = ["local"] }

# with cost tracking
embedrs = { version = "0.3", features = ["cost-tracking"] }

# with tracing
embedrs = { version = "0.3", features = ["local", "tracing"] }
```

## Benchmark Results

8 dimensions, 184 unique texts. Full methodology and reproduction instructions in [benchrs](https://github.com/goliajp/airs/tree/develop/crates/benchrs).

| Metric | MiniLM-L6 | MiniLM-L12 | BGE-small | GTE-small | OpenAI | Gemini | Cohere | Voyage |
|--------|:---------:|:----------:|:---------:|:---------:|:------:|:------:|:------:|:------:|
| **Size** | **23MB** | 133MB | 133MB | 67MB | cloud | cloud | cloud | cloud |
| **Spearman ρ** | 0.81 | 0.84 | 0.71 | 0.75 | 0.91 | **0.94** | 0.91 | 0.89 |
| **Discrimination** | 0.52 | 0.52 | 0.29 | 0.14 | **0.58** | 0.30 | 0.46 | 0.45 |
| **Retrieval** | **100%** | **100%** | 89% | **100%** | **100%** | 89% | **100%** | 89% |
| **EN ρ** | 0.92 | **0.94** | 0.92 | 0.90 | 0.91 | 0.91 | 0.89 | 0.88 |
| **ZH ρ** | 0.65 | 0.74 | 0.45 | 0.40 | 0.88 | **0.99** | 0.93 | 0.89 |
| **JA ρ** | 0.60 | 0.90 | 0.20 | 0.50 | 0.90 | **1.00** | **1.00** | 0.90 |
| **Cross-lingual** | 0.25 | 0.26 | 0.66 | 0.81 | 0.71 | 0.84 | 0.68 | **0.85** |
| **Robustness** | 0.89 | 0.90 | 0.94 | **0.97** | 0.88 | 0.94 | 0.89 | 0.95 |
| **Cluster sep.** | **8.73x** | 4.38x | 1.29x | 1.09x | 2.55x | 1.11x | 1.41x | 1.30x |
| **Cost** | **$0** | **$0** | **$0** | **$0** | $0.02/1M | free tier | $0.10/1M | $0.06/1M |

### Why MiniLM-L6 for local

- 23MB -- the only model small enough for app embedding (others are 67-133MB)
- Best clustering separation at 8.73x (2nd place is 4.38x)
- 100% retrieval accuracy, EN ρ=0.92
- 12-layer models are 3-6x larger with no meaningful quality improvement
- Known weakness: poor on Chinese/Japanese (ρ=0.60-0.65) and cross-lingual (0.25)

### Why OpenAI for cloud

- Best discrimination gap at 0.58 (dissimilar texts avg cosine = 0.09, closest to zero)
- 100% retrieval accuracy, MRR=1.0
- Balanced multilingual: EN=0.91, ZH=0.88, JA=0.90 -- no weak language
- Cheapest cloud option at $0.02/1M tokens
- Gemini has higher ρ (0.94) but poor discrimination (0.30) and retrieval miss (89%)
- Cohere matches quality but costs 5x more ($0.10/1M tokens)

## Providers

| Provider | Constructor | Default Model | Max Batch Size |
|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 |
| Google Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 |
| Voyage AI | `Client::voyage(key)` | `voyage-3-large` | 128 |
| Jina AI | `Client::jina(key)` | `jina-embeddings-v3` | 2048 |
| Local | `Client::local(name)?` | `all-MiniLM-L6-v2` | 256 |

Each cloud provider also has a `*_compatible` constructor for proxies or API-compatible services:

```rust
// OpenAI-compatible (Azure, proxies, etc.)
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

Embed thousands of texts concurrently. Texts are automatically chunked based on the provider's maximum batch size:

```rust
let client = embedrs::cloud("sk-...");

let texts: Vec<String> = (0..5000).map(|i| format!("document {i}")).collect();

let result = client.embed_batch(texts)
    .concurrency(5)       // max concurrent API requests (default: 5)
    .chunk_size(512)       // texts per request (default: provider max)
    .model("text-embedding-3-large")
    .await?;

println!("total embeddings: {}", result.embeddings.len());
println!("total tokens: {}", result.usage.total_tokens);
```

## Similarity Functions

```rust
use embedrs::{cosine_similarity, dot_product, euclidean_distance};

let a = vec![1.0, 0.0, 0.0];
let b = vec![0.0, 1.0, 0.0];

let cos = cosine_similarity(&a, &b);    // 0.0 (orthogonal)
let dot = dot_product(&a, &b);          // 0.0
let dist = euclidean_distance(&a, &b);  // 1.414...
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

## Backoff and Timeout

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

## Provider Fallback

Chain fallback providers for automatic failover when the primary provider is unavailable:

```rust
let client = embedrs::Client::openai("sk-...")
    .with_fallback(embedrs::Client::cohere("cohere-key"));
// if OpenAI fails, automatically tries Cohere
let result = client.embed(vec!["hello".into()]).await?;
```

Multiple fallbacks are tried in order:

```rust
let client = embedrs::Client::openai("sk-...")
    .with_fallback(embedrs::Client::cohere("cohere-key"))
    .with_fallback(embedrs::Client::voyage("voyage-key"));
```

## Cost Tracking

Enable the `cost-tracking` feature to get estimated cost per request:

```toml
embedrs = { version = "0.3", features = ["cost-tracking"] }
```

```rust
let result = client.embed(vec!["hello".into()]).await?;
if let Some(cost) = result.usage.cost {
    println!("estimated cost: ${cost:.6}");
}
```

Cost estimation uses `tiktoken` pricing data. Returns `None` for models without pricing information.

## Error Handling

All fallible operations return `embedrs::Result<T>`. Match on `Error` variants for fine-grained control:

```rust,no_run
use embedrs::Error;

# async fn run(client: &embedrs::Client) {
match client.embed(vec!["hello".into()]).await {
    Ok(result) => println!("got {} embeddings", result.embeddings.len()),
    Err(Error::Api { status: 429, .. }) => eprintln!("rate limited"),
    Err(Error::Api { status, message }) => eprintln!("API error {status}: {message}"),
    Err(Error::Timeout(duration)) => eprintln!("timed out after {duration:?}"),
    Err(Error::Http(e)) => eprintln!("network error: {e}"),
    Err(e) => eprintln!("other error: {e}"),
}
# }
```

## Why embedrs?

| Aspect | embedrs | fastembed-rs | Raw reqwest |
|---|---|---|---|
| Cloud providers | 5 built-in (OpenAI, Cohere, Gemini, Voyage, Jina) | None | Manual per provider |
| Local inference | candle-based, 23MB default model | ONNX Runtime, multiple models | N/A |
| Unified interface | Same `EmbedResult` for cloud and local | Local only | N/A |
| Batch auto-chunking | Automatic by provider limits + concurrency | Manual | Manual |
| Provider fallback | Built-in `.with_fallback()` chain | N/A | Manual |
| Data-driven defaults | 8-dimension benchmark across 8 models ([benchrs](https://github.com/goliajp/airs/tree/develop/crates/benchrs)) | No published benchmark | N/A |
| Backoff & timeout | Built-in exponential backoff on 429/503 | N/A | Manual |

**fastembed-rs** is a solid choice if you only need local inference with ONNX Runtime and don't need cloud providers. **embedrs** is designed for applications that need both cloud and local through a single API, with opinionated defaults and production features like fallback and backoff.

## Ecosystem

embedrs is part of [airs](https://github.com/goliajp/airs) (AI in Rust Series):

| Crate | Description |
|---|---|
| [tiktoken](https://crates.io/crates/tiktoken) | High-performance BPE tokenizer for all mainstream LLMs |
| [instructors](https://crates.io/crates/instructors) | Type-safe structured output extraction from LLMs |
| [embedrs](https://crates.io/crates/embedrs) | Unified embedding -- cloud + local (this crate) |
| [chunkedrs](https://crates.io/crates/chunkedrs) | AI-native text chunking for embedding and retrieval |
| [benchrs](https://github.com/goliajp/airs/tree/develop/crates/benchrs) | Reproducible benchmark experiments for airs decisions |

## License

[MIT](LICENSE)
