# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-embeddings/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-embeddings/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](#license)

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

> **Upgrading from 0.6?** 0.7 changes the API: `result.embeddings[i]` is now `result[i]`, errors and backoff have new names, and a fallback runs only after a transient failure. The [migration table](CHANGELOG.md#upgrading-from-06) gives the one-line fix for each. Requests are the same as in 0.6.2; local vectors differ from 0.6.2 by at most 3.3e-7 per component.

One async Rust client for text embeddings. Six cloud APIs (OpenAI, Cohere, Gemini, Voyage, Jina, Mistral) and four small BERT models that run in-process on the CPU return the same `EmbedResult`, so switching backends is a one-line change. On an Apple M4 Pro with default threads, the local `all-MiniLM-L6-v2` runs 3.1–4.4x faster than fastembed (ONNX Runtime) on single texts and 8.9–15x faster on batches of 256, with the same vectors to within 3.3e-7.

## Features

- **Six cloud providers** behind one `Client`, each with a `*_compatible` constructor for proxies and API-compatible servers.
- **Local inference** (feature `local`): `all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, `bge-small-en-v1.5`, `gte-small` with the forward pass in the crate, no dependencies. On macOS the matrix products go to Apple's Accelerate framework; on other systems they run on the crate's own AVX2/FMA (x86_64) or NEON (aarch64) kernels. Texts run without padding and are truncated at each model's own limit.
- **Batching**: `embed_batch` splits any number of texts into requests the provider accepts, by count and by tokens, and runs them concurrently. All vectors of a result sit in one buffer.
- **Retries and fallback**: exponential backoff on 408, 429, 500, 502, 503, 504, 529 and connection errors, honouring `Retry-After` (seconds or a date); a chain of fallback providers for when the retries run out.
- **Compact transfer**: OpenAI, Voyage and Cohere return base64 vectors, which embedrs decodes itself; a 256-vector response is 2.3x smaller than its decimal form and parses 7.6x faster.
- **Cost tracking** (feature `cost-tracking`): estimated USD per request from the [tiktoken](https://crates.io/crates/tiktoken) price table.

- **Few dependencies**: the core has none, with its own JSON, base64, WordPiece and SHA-256 code. The default build adds reqwest (rustls with ring) and tokio's timer: 84 crates on macOS, down from 101 in 0.6.2. `local` adds none. With `default-features = false` embedrs is the only crate, and you bring the HTTP client.

## Performance

Everything here was measured on 2026-11-26 on an Apple M4 Pro Mac mini (14 cores, macOS 27.2.1) with rustc 1.100, release build with LTO, while nothing else was running under the machine's benchmark lock. The one-minute load average was sampled every 15 s while the cloud numbers were taken: at most 2.2. Load during the local runs is given with them.

### Local inference

`all-MiniLM-L6-v2` against 0.6.2 (candle), [fastembed](https://crates.io/crates/fastembed) 7.8.0 (ONNX Runtime) on the same model, truncating at the same 256 tokens, and [model2vec-rs](https://crates.io/crates/model2vec-rs) 0.5.1 with `potion-base-8M`, all in one process. Vectors differ from 0.6.2 by at most 3.3e-7 per component (38 texts, all four models, cosine 1 − 2e-12 or closer); 0.6.2 agreed with fastembed to a cosine of 1.000000 on 32 mixed texts. Single texts are 8, 40 and 180 words (13, 56 and 227 tokens), batches 256 texts of 8, 40 or 180 words and a mix of 4 to 200 words. Single texts: median of 200 calls (30 on one thread); batches: median of 5 runs. The one-minute load average, sampled every 10 s, was at most 4.4 (the benchmark's own threads included). One thread means `EMBEDRS_THREADS=1 VECLIB_MAXIMUM_THREADS=1` for embedrs, `RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` for 0.6.2 and one intra-op thread for fastembed.

| Default threads | 0.6.2 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| one text, 8 words | 3.13 ms | **1.34 ms** | 4.21 ms | 0.0003 ms |
| one text, 40 words | 5.86 ms | **2.47 ms** | 8.17 ms | 0.0012 ms |
| one text, 180 words | 15.7 ms | **5.48 ms** | 24.1 ms | 0.0051 ms |
| 256 texts, 8 words | 1,882 /s | **6,948 /s** | 660 /s | 2.8 M/s |
| 256 texts, 40 words | 763 /s | **1,664 /s** | 188 /s | 1.9 M/s |
| 256 texts, 180 words | 158 /s | **344 /s** | 26 /s | 1.1 M/s |
| 256 texts, mixed | 209 /s | **633 /s** | 42 /s | 1.2 M/s |

| One thread | 0.6.2 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| one text, 8 words | 2.19 ms | **1.47 ms** | 4.02 ms | 0.0004 ms |
| one text, 40 words | 5.55 ms | **2.56 ms** | 7.45 ms | 0.0012 ms |
| one text, 180 words | 23.5 ms | **9.88 ms** | 26.4 ms | 0.0051 ms |
| 256 texts, 8 words | 806 /s | **2,486 /s** | 579 /s | 2.0 M/s |
| 256 texts, 40 words | 205 /s | **520 /s** | 154 /s | 0.63 M/s |
| 256 texts, 180 words | 40 /s | **103 /s** | 26 /s | 0.15 M/s |
| 256 texts, mixed | 62 /s | **198 /s** | 26 /s | 0.25 M/s |

Peak memory (largest resident set) of a process that loads one model and embeds the 256 mixed texts: 178 MB for `all-MiniLM-L6-v2` (0.6.2: 1,715 MB), 219 to 234 MB for the 12-layer models (0.6.2: 680 to 1,883 MB). The F32 weight files are mapped and read in place, and texts run without padding in chunks of about 4,096 tokens.

The matrix products run in Accelerate in both 0.6.2 and embedrs; the rest of the forward pass is where the time went. embedrs computes GELU with a vectorised erf (0.6.2 called `erff` once per value), fuses the residual add into layer normalisation, runs attention per text and head without padding or masks, and spreads the rest over the performance cores only when a call is large enough to gain from it. model2vec is faster on every row because it is a different kind of model: a static embedding looks each token up in a table and averages, with no transformer layers. It is listed for scale; on quality it is not a replacement for a transformer model.

### Cloud calls

An in-process mock server on loopback answers with 1536-dimensional OpenAI-shaped responses; the time is the whole `embed()` call (median of 50 calls for 256 vectors, 15 for 2048), so a large part of it is moving the bytes. Peers: [async-openai](https://crates.io/crates/async-openai) 0.43.0 and [rig-core](https://crates.io/crates/rig-core) 0.47.1 (which returns `f64`), given the same bodies. A dash means not measured.

| Response | 0.6.2 | **embedrs** | async-openai | rig-core |
|---|---|---|---|---|
| 256 vectors, shortest decimals (4.9 MB) | 6.64 ms | **3.87 ms** | — | — |
| 256 vectors, 17-digit decimals (8.3 MB) | 9.48 ms | **6.40 ms** | 9.69 ms | 22.0 ms |
| 256 vectors, base64 (2.1 MB) | 1.43 ms | **0.80 ms** | 1.11 ms | — |
| 2048 vectors, shortest decimals (39 MB) | 59.6 ms | **32.7 ms** | — | — |
| 2048 vectors, 17-digit decimals (67 MB) | 80.4 ms | **53.5 ms** | 80.2 ms | 172 ms |
| 2048 vectors, base64 (17 MB) | 12.1 ms | **7.1 ms** | 9.4 ms | — |

Parsing alone takes 0.49 to 0.62 times as long as serde_json on the decimal bodies (3.0 ms against 6.1 ms for 4.9 MB). The number parser is correctly rounded: it gave the same bits as `str::parse::<f32>` on every finite `f32` printed four ways (17.1 billion strings) and on a billion random ones. For a single vector, `embed()` takes about 0.05 ms, as in 0.6.2 and async-openai; a network round trip takes tens of milliseconds.

## Installation

```toml
[dependencies]
embedrs = "0.7"

# local inference
embedrs = { version = "0.7", features = ["local"] }
```

| Feature | Default | What it adds |
|---|---|---|
| `reqwest` | yes | The HTTP transport and the constructors that use it (`embedrs::cloud`, `Client::openai`, …) |
| `local` | no | Local inference on the CPU, no dependencies (the tokenizer is compiled in; weights are downloaded on first use) |
| `cost-tracking` | no | `Usage::cost`, from tiktoken's price table (no tokenizer vocabularies are compiled in) |
| `token-count` | no | Exact OpenAI token counts for planning batches and refusing long inputs (compiles in `cl100k_base`) |
| `tracing` | no | Spans and events through the `tracing` crate |

Crates in the dependency tree on macOS, embedrs included (0.6.2 in parentheses): default 84 (101), default with `local` 84 (255), `default-features = false` 1, `local` without `reqwest` 1. On x86_64 Linux the default build has 82.

## Quick start

```rust,no_run
async fn run() -> embedrs::Result<()> {
    // cloud: OpenAI text-embedding-3-small
    let client = embedrs::cloud("sk-...");
    let result = client.embed(["hello world"]).await?;
    println!("{} dimensions", result.dim()); // 1536
    println!("{:?}", &result[0][..4]); // the first vector, a &[f32]

    // local: all-MiniLM-L6-v2, no API key
    let client = embedrs::local();
    let result = client.embed(["hello world"]).await?;
    println!("{} dimensions", result.dim()); // 384
    Ok(())
}
```

`embed` takes a `Vec<String>`, `&[String]`, `&[&str]`, an array of `&str` or one `&str`. `result[i]` is the vector of input `i`; `result.iter()` walks them, `result.as_flat()` is the whole buffer and `result.to_vecs()` copies it into a `Vec<Vec<f32>>`.

The first local call downloads the weights into the Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`); later calls and other processes reuse them. The download is pinned to a commit, resumes after an interruption, is checked against its SHA-256 and does not count against the call's timeout.

## Providers

| Provider | Constructor | Default model | Texts per request | `dimensions` |
|---|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 | `text-embedding-3-*` |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 | `embed-v4.0`, `embed-v5.0-*`; error on v3 models |
| Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 | 128 to 3072 |
| Voyage | `Client::voyage(key)` | `voyage-4-large` | 1000 | 4, 3.5, 3-large and code-3 families; error on older models |
| Jina | `Client::jina(key)` | `jina-embeddings-v3` | 2048 | yes |
| Mistral | `Client::mistral(key)` | `mistral-embed` | 512 | `codestral-embed`; error on `mistral-embed` |
| Local | `Client::local(name)?` | `all-MiniLM-L6-v2` | 256 | only the model's own width |

The model is a free-form string, so a new model id works without a new release: `client.embed(texts).model("text-embedding-3-large")`. Prices per 1M input tokens, as `cost-tracking` reports them (tiktoken 5.5):

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

Azure OpenAI and gateways that want other headers go through `Endpoint`:

```rust,no_run
use std::sync::Arc;
use embedrs::{Client, Endpoint, transport::ReqwestTransport};

fn azure() -> Client {
    // the v1 api: no api-version parameter, the deployment name goes in `model`
    let endpoint =
        Endpoint::openai_compatible("", "https://my-resource.openai.azure.com/openai/v1")
            .bearer(false)
            .header("api-key", "azure-key");
    Client::new(endpoint, Arc::new(ReqwestTransport::new())).with_model("my-embedding-deployment")
}
```

## Local models

| Model | Layers | Weights | Reads up to | Pooling |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` (default) | 6 | 91 MB | 256 tokens | mean |
| `all-MiniLM-L12-v2` | 12 | 133 MB | 128 tokens | mean |
| `bge-small-en-v1.5` | 12 | 133 MB | 512 tokens | CLS |
| `gte-small` | 12 | 67 MB | 512 tokens | mean |

All four produce 384-dimensional, L2-normalised vectors. `bge-small-en-v1.5` prepends its query instruction when the input type is `InputType::SearchQuery`. A local client accepts `.model()` with another local model id, and `.dimensions(384)`; anything else is an error. Longer texts are truncated, and `usage.total_tokens` counts the tokens the model read.

Each loaded model runs on its own thread, so local inference needs no async runtime: `LocalModel::from_files` loads weights you already have (safetensors in F32, F16 or BF16), and `embed_blocking` embeds and waits. A call with more than a few dozen tokens also uses worker threads while it runs: the performance cores on macOS, every core the process may use elsewhere. Set `EMBEDRS_THREADS` to change the number; `EMBEDRS_THREADS=1` keeps a call on the model's own thread (on macOS, Accelerate still follows `VECLIB_MAXIMUM_THREADS`).

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

A request is closed when it reaches the provider's limit of texts, or of tokens where the provider documents one (300,000 for OpenAI, 120,000 for Voyage). A text is taken to have as many tokens as it has UTF-8 bytes, which for OpenAI's tokenizer is never too few; with `token-count` the OpenAI count is exact and the requests are fuller.

## Input type and dimensions

```rust,no_run
use embedrs::InputType;

async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let docs = client
        .embed(["Rust is a systems language"])
        .input_type(InputType::SearchDocument)
        .await?;
    let query = client
        .embed(["what is rust"])
        .input_type(InputType::SearchQuery)
        .dimensions(256)
        .await?;
    println!("{} / {}", docs.dim(), query.dim());
    Ok(())
}
```

`InputType` has `SearchDocument`, `SearchQuery`, `Classification` and `Clustering`. Each provider gets its own name for it (Cohere `search_query`, Gemini `RETRIEVAL_QUERY`, Jina `retrieval.query`, and so on); `gemini-embedding-2` gets it as a text prefix. OpenAI and Mistral have no such parameter and ignore it.

## Long inputs

```rust,no_run
use embedrs::Truncate;

async fn run(client: &embedrs::Client, texts: &[String]) -> embedrs::Result<()> {
    let result = client.embed(texts).truncate(Truncate::Error).await?;
    println!("{}", result.len());
    Ok(())
}
```

`Truncate::Provider` (the default) leaves long inputs to the provider: Voyage, Cohere, Gemini and the local models cut the end, OpenAI and Jina answer with an error. `Truncate::End` asks for cutting, `Truncate::Error` for refusing; a provider without such a switch returns `Error::Unsupported`. Local models refuse with `Error::InputTooLong` before running, and so do OpenAI models when `token-count` is on.

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

    let result = client.embed(["hello"]).await?;
    println!("answered by {}", result.model);
    Ok(())
}
```

Without `with_retry`, a failed request is not retried. When the primary still fails with a transient error (a status from the list above, or no connection), each fallback is tried in order with its own model and dimensions; if all fail, the primary's error is returned. A 400, 401, 403 or 422 is returned at once, so a wrong key does not hide behind a fallback. Vectors from different models are not comparable, so a fallback is for availability, not for filling one index from several models.

## Your own HTTP client

```rust,no_run
use embedrs::{Client, transport::HttpResponse};

fn run(client: &Client) -> embedrs::Result<()> {
    let prepared = client.prepare(["hello world"])?;
    let request = prepared.request(); // method, url, headers, body
    println!("POST {} ({} bytes)", request.url, request.body.len());

    let body = br#"{"data":[{"index":0,"embedding":[0.6,0.8]}]}"#; // the answer from your HTTP client
    let result = prepared.finish(HttpResponse::new(200, vec![], body.to_vec()))?;
    assert_eq!(&result[0], &[0.6, 0.8]);
    Ok(())
}
```

`prepare` and `finish` do no I/O, so any HTTP client works, blocking ones included; [`examples/blocking.rs`](examples/blocking.rs) runs on a std `TcpStream` with no dependency at all. To keep retries, fallbacks, batching and timeouts as well, implement `Transport` (two required methods: send a request, sleep) and pass it to `Client::new`. embedrs never spawns a task and does not care which runtime polls it. For your own reqwest settings (proxy, pool, headers), build the client with `ReqwestTransport::client_builder()`, which sets up the TLS provider, and pass it through `client.with_transport(Arc::new(ReqwestTransport::from(reqwest_client)))`.

Code that should work with any embedding backend can take `&dyn embedrs::Embed`, a one-method trait that `Client` implements.

## Cost tracking

```rust,no_run
async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let result = client.embed(["hello"]).await?;
    if let Some(cost) = result.usage.cost {
        println!("about ${cost:.8}");
    }
    Ok(())
}
```

`cost` is `None` when the model is not in the price table, when the provider reported no token count, or without the `cost-tracking` feature.

## Errors

```rust,no_run
use embedrs::Error;

async fn run(client: &embedrs::Client) {
    match client.embed(["hello"]).await {
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
