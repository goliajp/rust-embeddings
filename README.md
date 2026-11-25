# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-embeddings/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-embeddings/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

> **Upgrading from 0.6.1?** No API change. Local models now read texts up to their own limit (256 tokens for the default model, was 128), so vectors of longer texts change; a fallback client now uses its own model and dimensions; options a backend cannot honour return an error instead of being ignored. Details: [CHANGELOG](CHANGELOG.md#upgrading-from-061).

One async Rust client for text embeddings. Six cloud APIs (OpenAI, Cohere, Gemini, Voyage, Jina, Mistral) and four small BERT models that run in-process on the CPU return the same `EmbedResult`, so switching backends is a one-line change. On an Apple M4 Pro, the local `all-MiniLM-L6-v2` runs 1.1–1.7x faster than fastembed (ONNX Runtime) on single texts and 1.3–12x faster on batches of 256, with the same vectors.

## Features

- **Six cloud providers** behind one `Client`, each with a `*_compatible` constructor for proxies and API-compatible servers.
- **Local inference** (feature `local`): `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, `bge-small-en-v1.5`, `gte-small` on [candle](https://github.com/huggingface/candle), pure Rust; on macOS the matrix products go to Apple's Accelerate framework. Texts are sorted by length and padded per group, and truncated at each model's own limit.
- **Batching**: `embed_batch` splits any number of texts into requests the provider accepts and runs them concurrently.
- **Retries and fallback**: exponential backoff on 408, 429, 500, 502, 503, 504, 529 and connection errors, honouring `Retry-After`; a chain of fallback providers.
- **Compact transfer**: OpenAI, Voyage and Cohere return base64 vectors, which embedrs decodes itself; a 256-vector response is 2.3x smaller than its decimal form and parses 8.6x faster.
- **Cost tracking** (feature `cost-tracking`): estimated USD per request from the [tiktoken](https://crates.io/crates/tiktoken) price table.

## Performance

Local inference, `all-MiniLM-L6-v2`, on an Apple M4 Pro Mac mini (14 cores, macOS), rustc 1.100, release build with LTO. Peers: [fastembed](https://crates.io/crates/fastembed) 7.8.0 (ONNX Runtime) on the same model, truncating at the same 256 tokens; [model2vec-rs](https://crates.io/crates/model2vec-rs) 0.5.1 with `potion-base-8M`. Vectors from embedrs and fastembed agree to a cosine of 1.000000 on 32 mixed texts. Texts are 8, 40 and 180 words (13, 54 and 232 tokens on average) and a mix of 4 to 200 words. Single texts: median of 200 calls (30 on one thread); batches: 256 texts, median of 3 runs. Load average during the runs: 1.7 to 7.2. "0.6.1" is the previous release's code on the same dependencies.

| Default threads | 0.6.1 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| one text, 8 words | 18.7 ms | **3.6 ms** | 4.3 ms | 0.0003 ms |
| one text, 40 words | 18.7 ms | **6.0 ms** | 8.5 ms | 0.0012 ms |
| one text, 180 words | 18.8 ms ¹ | **15.9 ms** | 24.6 ms | 0.0051 ms |
| 256 texts, 8 words | 155 /s | **1,872 /s** | 672 /s | 3.0 M/s |
| 256 texts, 40 words | 152 /s | **776 /s** | 184 /s | 2.1 M/s |
| 256 texts, 180 words | 145 /s ¹ | **159 /s** | 22 /s | 1.0 M/s |
| 256 texts, mixed | 152 /s ¹ | **249 /s** | 21 /s | 0.8 M/s |

| One thread | 0.6.1 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| one text, 8 words | 39.4 ms | **2.0 ms** | 3.5 ms | 0.0004 ms |
| one text, 40 words | 39.8 ms | **5.6 ms** | 7.3 ms | 0.0012 ms |
| one text, 180 words | 40.3 ms ¹ | **23.7 ms** | 26.1 ms | 0.0051 ms |
| 256 texts, 8 words | 27 /s | **798 /s** | 581 /s | 2.0 M/s |
| 256 texts, 40 words | 27 /s | **204 /s** | 156 /s | 0.57 M/s |
| 256 texts, 180 words | 27 /s ¹ | **40 /s** | 21 /s | 0.14 M/s |
| 256 texts, mixed | 27 /s ¹ | **59 /s** | 12 /s | 0.24 M/s |

¹ 0.6.1 cut every text at 128 tokens and padded every text to 128, so on 180-word texts it did about half the work of the other two.

model2vec is faster on every row because it is a different kind of model: a static embedding looks each token up in a table and averages, with no transformer layers. It is listed for scale; on quality it is not a replacement for a transformer model. Against fastembed, the gains come from padding each group of texts only to its longest member, Accelerate for the matrix products on macOS, and a forward pass that fuses the query, key and value projections. Linux was not measured for this release.

Cloud calls: an in-process mock server on loopback answers with 1536-dimensional OpenAI-shaped responses. For 256 vectors, `embed()` takes 6.85 ms on a decimal body (4.88 MB) and 1.43 ms on a base64 body (2.11 MB), which is what embedrs now asks OpenAI, Voyage and Cohere for; for 2048 vectors, 59.2 ms and 12.1 ms. The rest of the client adds about 0.05 ms per call, against network round trips of tens of milliseconds.

## Installation

```toml
[dependencies]
embedrs = "0.6"

# local inference
embedrs = { version = "0.6", features = ["local"] }
```

| Feature | Default | What it adds |
|---|---|---|
| *(none)* | yes | The client and all six cloud providers |
| `local` | no | Local inference with candle (tokenizers embedded; weights downloaded on first use) |
| `cost-tracking` | no | `Usage::cost`, from tiktoken's price table (no tokenizer vocabularies are compiled in) |
| `tracing` | no | Spans and events through the `tracing` crate |

## Quick start

```rust,no_run
async fn run() -> embedrs::Result<()> {
    // cloud: OpenAI text-embedding-3-small
    let client = embedrs::cloud("sk-...");
    let result = client.embed(vec!["hello world".into()]).await?;
    println!("{} dimensions", result.dim()); // 1536

    // local: all-MiniLM-L6-v2, no API key
    let client = embedrs::local();
    let result = client.embed(vec!["hello world".into()]).await?;
    println!("{} dimensions", result.dim()); // 384
    Ok(())
}
```

The first local call downloads the weights into the Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`); later calls and other processes reuse them.

## Providers

| Provider | Constructor | Default model | Texts per request | `dimensions` |
|---|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 | `text-embedding-3-*` |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 | `embed-v4.0`, `embed-v5.0-*`; error on v3 models |
| Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 | 128 to 3072 |
| Voyage | `Client::voyage(key)` | `voyage-4-large` | 1000 (`embed_batch` sends 128) | 4, 3.5, 3-large and code-3 families; error on older models |
| Jina | `Client::jina(key)` | `jina-embeddings-v3` | 2048 | yes |
| Mistral | `Client::mistral(key)` | `mistral-embed` | 512 | `codestral-embed`; error on `mistral-embed` |
| Local | `Client::local(name)?` | `all-MiniLM-L6-v2` | 256 | only the model's own width |

The model is a free-form string, so a new model id works without a new release: `client.embed(texts).model("text-embedding-3-large")`. Prices per 1M input tokens, as `cost-tracking` reports them (tiktoken 5.4):

| Model | USD / 1M tokens | Model | USD / 1M tokens |
|---|---|---|---|
| `text-embedding-3-small` | 0.02 | `voyage-4-large` | 0.12 |
| `text-embedding-3-large` | 0.13 | `voyage-4` | 0.06 |
| `gemini-embedding-001` | 0.15 | `voyage-4-lite` | 0.02 |
| `gemini-embedding-2` | 0.20 | `voyage-3-large` | 0.18 |
| `mistral-embed` | 0.10 | `codestral-embed` | 0.15 |

Cohere and Jina models have no entry in the table, so their `cost` is `None`.

Model ids current in November 2026: OpenAI `text-embedding-3-small`, `text-embedding-3-large`; Cohere `embed-v4.0`, `embed-v5.0-pro`, `embed-v5.0-fast`; Gemini `gemini-embedding-001`, `gemini-embedding-2`; Voyage `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-3-large`, `voyage-code-3`; Jina `jina-embeddings-v3`, `jina-embeddings-v5-text-small`, `jina-embeddings-v5-text-nano`; Mistral `mistral-embed`, `codestral-embed`.

Every cloud provider also has a `*_compatible` constructor that takes a base URL:

```rust,no_run
use embedrs::Client;

fn clients() {
    let openai_like = Client::openai_compatible("key", "https://api.deepseek.com/v1");
    let cohere_proxy = Client::cohere_compatible("key", "https://proxy.example.com/v2");
    let gemini_proxy = Client::gemini_compatible("key", "https://proxy.example.com/v1beta");
    let voyage_proxy = Client::voyage_compatible("key", "https://proxy.example.com/v1");
    let jina_proxy = Client::jina_compatible("key", "https://proxy.example.com/v1");
    let mistral_proxy = Client::mistral_compatible("key", "https://proxy.example.com/v1");
}
```

`openai_compatible` asks for decimal vectors, which every OpenAI-compatible server understands; only the official endpoint is asked for base64.

## Local models

| Model | Layers | Weights | Reads up to | Pooling |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` (default) | 6 | 91 MB | 256 tokens | mean |
| `all-MiniLM-L12-v2` | 12 | 133 MB | 128 tokens | mean |
| `bge-small-en-v1.5` | 12 | 133 MB | 512 tokens | CLS |
| `gte-small` | 12 | 67 MB | 512 tokens | mean |

All four produce 384-dimensional, L2-normalised vectors. `bge-small-en-v1.5` prepends its query instruction when the input type is `InputType::SearchQuery`. A local client accepts `.model()` with another local model id, and `.dimensions(384)`; anything else is an error. Longer texts are truncated, and `usage.total_tokens` counts the tokens the model read.

## Batch embedding

```rust,no_run
async fn run() -> embedrs::Result<()> {
    let client = embedrs::cloud("sk-...");
    let texts: Vec<String> = (0..5000).map(|i| format!("document {i}")).collect();

    let result = client
        .embed_batch(texts)
        .concurrency(5) // requests in flight (default 5)
        .chunk_size(512) // texts per request (default: provider limit)
        .model("text-embedding-3-large")
        .await?;

    println!("{} vectors, {:?} tokens", result.len(), result.usage.total_tokens);
    Ok(())
}
```

## Input type and dimensions

```rust,no_run
use embedrs::InputType;

async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let docs = client
        .embed(vec!["Rust is a systems language".into()])
        .input_type(InputType::SearchDocument)
        .await?;
    let query = client
        .embed(vec!["what is rust".into()])
        .input_type(InputType::SearchQuery)
        .dimensions(256)
        .await?;
    println!("{} / {}", docs.dim(), query.dim());
    Ok(())
}
```

`InputType` has `SearchDocument`, `SearchQuery`, `Classification` and `Clustering`. Each provider gets its own name for it (Cohere `search_query`, Gemini `RETRIEVAL_QUERY`, Jina `retrieval.query`, and so on); `gemini-embedding-2` gets it as a text prefix. OpenAI and Mistral have no such parameter and ignore it.

## Retries, timeouts and fallback

```rust,no_run
use std::time::Duration;
use embedrs::{Backoff, Client};

async fn run() -> embedrs::Result<()> {
    let client = Client::openai("sk-...")
        .with_retry(Backoff::new()) // 500 ms base, 30 s cap, 3 retries
        .with_timeout(Duration::from_secs(120)) // whole request, retries included (default 60 s)
        .with_fallback(Client::cohere("cohere-key"))
        .with_fallback(Client::voyage("voyage-key").with_model("voyage-4-lite"));

    let result = client.embed(vec!["hello".into()]).await?;
    println!("answered by {}", result.model);
    Ok(())
}
```

Without `with_retry_backoff`, a failed request is not retried. When the primary fails, each fallback is tried in order with its own model and dimensions; if all fail, the primary's error is returned. Vectors from different models are not comparable, so a fallback is for availability, not for filling one index from several models.

## Cost tracking

```rust,no_run
async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let result = client.embed(vec!["hello".into()]).await?;
    if let Some(cost) = result.usage.cost {
        println!("about ${cost:.8}");
    }
    Ok(())
}
```

`cost` is `None` when the model is not in the price table or the provider reported no token count.

## Errors

```rust,no_run
use embedrs::Error;

async fn run(client: &embedrs::Client) {
    match client.embed(vec!["hello".into()]).await {
        Ok(result) => println!("{} vectors", result.len()),
        Err(Error::Api { status: 429, .. }) => eprintln!("rate limited"),
        Err(Error::Api { status, message, .. }) => eprintln!("API error {status}: {message}"),
        Err(Error::Timeout(after)) => eprintln!("timed out after {after:?}"),
        Err(Error::Transport(e)) => eprintln!("network error: {e}"),
        Err(e) => eprintln!("{e}"),
    }
}
```

## Similarity

```rust
use embedrs::{cosine_similarity, dot_product, euclidean_distance};

let a = [1.0, 0.0, 0.0];
let b = [0.0, 1.0, 0.0];
assert_eq!(cosine_similarity(&a, &b), 0.0);
assert_eq!(dot_product(&a, &b), 0.0);
assert!((euclidean_distance(&a, &b) - 2f32.sqrt()).abs() < 1e-6);
```

## Choosing a model

[`examples/embedding_models/`](examples/embedding_models/) scores eight models on 184 English, Chinese and Japanese texts (recorded 2026-03-08; `cargo run --release --features local --example embedding_models` prints the full results from the stored vectors):

| | MiniLM-L6 | MiniLM-L12 | BGE-small | GTE-small | OpenAI 3-small | Gemini 001 | Cohere v4 | Voyage 3-large |
|---|---|---|---|---|---|---|---|---|
| Spearman ρ, all 32 pairs | 0.81 | 0.84 | 0.71 | 0.75 | 0.91 | 0.94 | 0.91 | 0.89 |
| ρ, English | 0.92 | 0.94 | 0.92 | 0.90 | 0.91 | 0.91 | 0.89 | 0.88 |
| ρ, Chinese | 0.65 | 0.74 | 0.45 | 0.40 | 0.88 | 0.99 | 0.93 | 0.89 |
| ρ, Japanese | 0.60 | 0.90 | 0.20 | 0.50 | 0.90 | 1.00 | 1.00 | 0.90 |
| Retrieval top-1 (9 queries) | 9 | 9 | 8 | 9 | 9 | 8 | 9 | 8 |
| Cross-lingual cosine | 0.25 | 0.26 | 0.66 | 0.81 | 0.71 | 0.84 | 0.68 | 0.85 |
| Similar minus dissimilar cosine | 0.52 | 0.52 | 0.29 | 0.14 | 0.58 | 0.30 | 0.46 | 0.45 |

The local models are English models: on Chinese and Japanese, and across languages, the cloud models score clearly higher. The corpus is small (32 scored pairs), so differences of a few hundredths are within noise.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
