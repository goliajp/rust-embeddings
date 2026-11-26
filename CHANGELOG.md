# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Upgrading from 0.6

0.7 is a breaking release. The compiler finds almost every change; the fix for
each is one line:

| 0.6 | 0.7 |
|---|---|
| `result.embeddings[i]` | `result[i]` (a `&[f32]` with `&`), `result.get(i)` |
| `for v in &result.embeddings` | `for v in result.iter()` |
| `result.embeddings` as `Vec<Vec<f32>>` | `result.to_vecs()`; without a copy: `result.as_flat()` and `result.dim()`, or `result.into_flat()` |
| `result.embeddings.len()` | `result.len()` |
| `usage.total_tokens: u32`, 0 when not reported | `Option<u64>`, `None` when not reported |
| `usage.cost` only with `cost-tracking` | always there; always `None` without `cost-tracking` |
| `Usage { .. }` or `EmbedResult { .. }` literals | `Usage::new(tokens, cost)`, `EmbedResult::from_flat(data, dim, model, usage)`; both are `#[non_exhaustive]` |
| `BackoffConfig { max_http_retries: 5, ..Default::default() }` | `Backoff::new().retries(5)`; also `.base()`, `.max()`, `.jitter()` |
| `config.delay_for(attempt)` | `backoff.delay(attempt)` |
| `client.with_retry_backoff(b)`, `.retry_backoff(b)` on a call | `client.with_retry(b)`, `.retry(b)` |
| `client.with_http_client(c)` (reqwest) | `client.with_transport(Arc::new(ReqwestTransport::from(c)))`, with `c` built from `ReqwestTransport::client_builder()` (it sets up the TLS provider) |
| `Error::Http(reqwest::Error)` | `Error::Transport(e)`, with `e.kind()` and `e.source()` |
| `Error::Json(_)` | `Error::Decode(_)` |
| `Error::InputTooLarge(count, max)` | `Error::TooManyInputs { count, max }` |
| `Error::Other(_)` | `Error::Unsupported`, `Error::Decode` or `Error::Model`, depending on the cause |
| `Error::Api { status, message, retry_after, .. }` | the same, plus `provider` |
| exhaustive `match` on `Error` or `InputType` | add `_ =>`: both are `#[non_exhaustive]`, and `Error` has the same variants under every feature set |
| `embedrs::client::EmbedResult`, `embedrs::error::Error`, `embedrs::usage::Usage`, `embedrs::similarity::…`, `embedrs::batch::…`, `embedrs::backoff::…` | the crate root: `embedrs::EmbedResult`, … |
| `embedrs::local::get_model(name)`, `list_models()` | `embedrs::local::models()`, a slice of `ModelInfo` |
| `ModelDefinition` (`hf_repo`, `hf_filename`, `hidden_size`, `max_seq_length`, `config`) | `ModelInfo` (`repo`, `file`, `dim`, `max_tokens`, `layers`), plus `revision`, `bytes` and `sha256` |
| `ModelDefinition::tokenizer_data` | removed: the tokenizer is compiled into the crate |
| `PoolingStrategy::{MeanPooling, ClsToken}` | `Pooling::{Mean, Cls}` |
| `default-features = false` | leaves the core alone: add `features = ["reqwest"]`, or pass your own transport to `Client::new` |

These compile unchanged: `embedrs::cloud`, `embedrs::local`, `Client::openai`
and the other constructors, `embed(vec![..]).await`, `embed_batch(..)` and its
`concurrency`, `chunk_size`, `model`, `dimensions`, `input_type` and `timeout`
setters, and the similarity functions.

Behaviour that changes without a compile error:

- **A fallback runs only after a transient failure** (408, 429, 500, 502, 503,
  504, 529, connection errors, timeouts inside the transport). A 400, 401, 403
  or 422 from the primary is returned at once. 0.6 fell back on every error.
- **`embed_batch` also splits by tokens** where the provider documents a cap
  per request: 300,000 for OpenAI, 120,000 for Voyage. Tokens are bounded by
  the UTF-8 length unless `token-count` is on, so long texts go out in more
  requests than in 0.6. Voyage requests now hold up to 1000 texts (128 before).
- **Local weights come from a pinned commit** and are checked against a
  SHA-256 after download. Files already in the Hugging Face cache at that
  commit are reused.
- **Downloading a local model is not under the call's timeout.**
- A batch in which one request reports no token count has
  `total_tokens: None`; 0.6 added the others up.
- **Options a model's API does not list are refused before sending**, with
  `Error::Unsupported`: `dimensions` on `text-embedding-ada-002`; for Jina,
  `Classification` and `Clustering` on `jina-embeddings-v4`, anything but
  `SearchQuery` and `SearchDocument` on `jina-clip-v2`, and any input type on
  the `jina-code-embeddings` models. A `SearchDocument` request to
  `jina-clip-v2` goes without a `task`. 0.6 sent these and got a 4xx back.
  The `jina-embeddings-v2` and `jina-clip-v1` models get no `task` at all,
  since their request schemas have none.
- **Empty strings are refused before sending** to OpenAI and Azure OpenAI
  (`Error::EmptyInput`), whose APIs do not accept them. Other
  OpenAI-compatible servers still get them.
- **Retries also follow OpenAI's `x-ratelimit-reset-requests` and
  `x-ratelimit-reset-tokens`** (the later of the two) and the `retryDelay` of
  a Gemini error when there is no `Retry-After`. A server hint up to the
  backoff's `max` is waited in full; a longer one ends the retries at once
  and the error carries it in `Error::Api::retry_after`. 0.6 waited for any
  `Retry-After`, however long.
- Token counts written with a zero fraction (`2.0`) are read; other
  fractions are a decode error.

The requests themselves do not change: for every provider, model kind, input
type and dimension setting that 0.6.2 was recorded with (510 cases), 0.7 sends
the same bytes. Vectors of local models differ from 0.6.2 by at most 3.3e-7 in
any component (38 texts, all four models; cosine 1 − 2e-12 or closer).

### Added

- `Transport`: bring your own HTTP client and async runtime. `Client::new`
  takes an `Endpoint` and any `Arc<dyn Transport>`.
- `Client::prepare` and `Prepared::finish`: build a request and read its answer
  without any I/O in embedrs, for blocking HTTP clients or wasm.
- `Endpoint`: base URL, extra headers, bearer token on or off, and `Limits`.
  Azure OpenAI and proxies work without a provider of their own.
- `Embed`, an object-safe trait for code that takes any embedding backend.
- `embed` and `embed_batch` take `&[&str]`, `&[String]`, `[&str; N]` and `&str`
  as well as `Vec<String>`.
- `Truncate`: cut or refuse inputs longer than the model reads, mapped to each
  provider's switch. Local models, and OpenAI models with `token-count`, refuse
  before sending with `Error::InputTooLong`.
- Feature `token-count`: exact OpenAI token counts for batch planning, from
  tiktoken's `cl100k_base`.
- `local::LocalModel`: load a model up front, from a file
  (`LocalModel::from_files`), or embed without an async runtime
  (`embed_blocking`).
- `Retry-After` is also read as an HTTP date.

### Changed

- The core has no dependencies. The default build adds reqwest (rustls with
  ring) and tokio's timer; `serde`, `serde_json`, `thiserror`, `rand` and
  `futures` are gone. Distinct crates on macOS, this one included: 84 by
  default (101 in 0.6.2), 1 with `default-features = false`, 84 with `local`
  (255 in 0.6.2) and 1 with `local` alone.
- `local` has no dependencies: the forward pass, the safetensors reader, the
  WordPiece tokenizer and the download code are in the crate. candle, `hf-hub`,
  `tokenizers` and rayon are gone, and nothing links onig (C) or OpenSSL. On
  macOS the matrix products go to the Accelerate framework that ships with the
  OS; elsewhere they run on in-crate AVX2/FMA (x86_64) or NEON (aarch64)
  kernels, with a portable fallback.
- Local inference runs on one worker thread per model instead of tokio's
  blocking pool, so it needs no runtime. A call with enough tokens adds
  threads for the length of the call: the performance cores on macOS, all
  available cores elsewhere, or `EMBEDRS_THREADS`.
- Local texts are no longer padded: each call's texts run unpadded in chunks
  of about 4096 tokens.
- F32 weight files are mapped and read in place; F16 and BF16 files are
  widened to f32 when loaded.
- All vectors of a result share one buffer.

### Performance

Measured on an Apple M4 Pro (conditions and peers in the README), 0.6.2 first:

- Local `all-MiniLM-L6-v2`, default threads: one text of 8 words 3.13 → 1.34
  ms, 40 words 5.86 → 2.47 ms, 180 words 15.7 → 5.48 ms; batches of 256 texts
  2.2 to 3.7 times as many texts per second (8 words 1,882 → 6,948 /s, mixed
  lengths 209 → 633 /s). With one thread: 1.5 to 3.2 times as fast.
- Peak memory of a process that loads one local model and embeds 256 texts:
  178 to 234 MB, against 680 to 1,883 MB in 0.6.2.
- `embed()` over loopback, 256 vectors of 1536 dimensions: decimal 6.64 →
  3.87 ms, base64 1.43 → 0.80 ms.
- Request bodies: 256 texts in 0.015 ms (serde_json 0.024 ms).

## [0.6.2] - 2026-11-25

### Upgrading from 0.6.1

No API change: code that builds against 0.6.1 builds against this release.
Four behaviours change, and one of them changes stored vectors:

- **Local models now read up to their real input limit.** 0.6 cut every text
  at 128 tokens. `all-MiniLM-L6-v2` now reads 256 tokens, `gte-small` and
  `bge-small-en-v1.5` 512, `all-MiniLM-L12-v2` stays at 128. Vectors of texts
  longer than 128 tokens change; shorter texts give the same vectors as
  before (padding never affected them). Re-embed long texts if you compare
  them against vectors stored by 0.6.
- **A fallback uses its own model.** `.model()`, `.dimensions()`,
  `with_model()` and `with_dimensions()` on the primary client no longer go
  to the fallbacks; set them on each fallback client.
- **Unsupported options are refused instead of ignored.** A local client
  given a model id that is not a local model returns `Error::UnknownModel`,
  and `.dimensions(n)` returns an error unless `n` is the model's width.
  Cloud models documented with a fixed width (Cohere v3, `mistral-embed`,
  older Voyage models) return an error for `.dimensions(n)` instead of
  sending a request whose answer has another width.
- **Retries cover more failures.** With `with_retry_backoff`, 408, 500,
  502, 504, 529 and connection errors are retried as well as 429 and 503.

### Fixed

- Several local clients loading the same model at the same time failed on a
  cold cache with `Lock acquisition failed: .../blobs/<sha>.lock`: hf-hub
  gives up on its download lock after about five seconds, before a 91 MB
  download ends. One process now loads each model once and shares it
  between clients, and a file lock in the cache directory makes other
  processes wait for a running download.
- Local models padded every text to 128 tokens and cut it there. Texts are
  now sorted by length and run in groups padded to the longest text in the
  group, and truncated at the model's limit. `bge-small-en-v1.5` no longer
  fails on texts over 512 tokens (`index-select invalid index 512`).
- Local token counts included padding (`"hi"` counted 128 tokens); they now
  count the tokens the model read.
- A local client given `.model("text-embedding-3-large").dimensions(64)`
  answered with 384-dimensional vectors labelled `text-embedding-3-large`.
  `.model()` now accepts another local model id and runs that model;
  `EmbedResult::model` names the model that ran.
- `bge-small-en-v1.5` gets the query instruction it was trained with when
  the input type is `SearchQuery`.
- `.dimensions()` was dropped for Cohere, Voyage and Mistral; it is now sent
  as `output_dimension` to the models that accept it.
- Gemini: the API key was sent in the URL, where an HTTP error message could
  print it; it now goes in the `x-goog-api-key` header. Token counts are read
  from `usageMetadata` instead of reported as 0, so `cost-tracking` prices
  Gemini calls. `gemini-embedding-2` gets its task as a text prefix instead
  of the `taskType` field it does not accept.
- A provider that reports no token count no longer gets a cost of 0.0 under
  `cost-tracking`; the cost is `None`.
- A fallback received the primary's model id and dimensions.
- Responses are checked: the number of vectors must match the number of
  texts, and OpenAI-shaped responses are put back in input order by `index`.
- Jina: the output format field is `embedding_type`; v5 models get
  `clustering` instead of v3's `separation` for `InputType::Clustering`.
- Voyage accepts 1000 texts per request (was limited to 128 here).
  `embed_batch` still sends 128 per request to stay under the 120k-token
  request limit of the larger models.

### Performance

- Local inference: per-group padding, Accelerate for matrix products on
  macOS (nothing to enable; the framework ships with macOS) and a forward
  pass that fuses the query, key and value projections. On an Apple M4 Pro
  with `all-MiniLM-L6-v2`, one 8-word text takes 3.6 ms instead of 18.7 ms,
  and 256 texts of 40 words run at 776 texts/s instead of 152 (default
  threads; the README has the full table and the comparison with fastembed
  and model2vec-rs).
- OpenAI (official endpoint), Voyage and Cohere are asked for base64
  vectors instead of decimal text. A 256-vector response of 1536 dimensions
  is 2.11 MB instead of 4.88 MB, and `embed()` on it takes 1.43 ms instead
  of 6.85 ms over loopback. Responses in either form are accepted from every
  provider.

### Changed

- `cost-tracking` uses tiktoken 5.4. Its price table prices `mistral-embed`,
  so the Mistral default now reports a `cost` where it reported `None`.
- Licensed under MIT or Apache-2.0, at your option (was MIT only).
- Local model weights go to the Hugging Face cache that `HF_HOME` names (the
  default is still `~/.cache/huggingface`), and no download progress bar is
  printed.

## [0.6.1] - 2026-08-13

Documentation only. No code changes.

### Fixed

- The install snippet in all three READMEs still said `embedrs = "0.5"`. 0.6.0
  shipped with its own crates.io page telling readers to depend on the
  previous minor.

## [0.6.0] - 2026-08-13

### Changed — breaking

- **The Voyage default moves from `voyage-3-large` to `voyage-4-large`.** The
  4 generation supersedes it and costs less ($0.12 vs $0.18 per 1M tokens).

  Both default to 1024 dimensions, so this does not change your vector width —
  but embeddings from different models are not interchangeable. Vectors already
  in an index were produced by `voyage-3-large`, and querying them with
  `voyage-4-large` embeddings compares points from two different spaces. Either
  re-embed the corpus, or pin the old model:

  ```rust
  let client = Client::voyage(key).with_model("voyage-3-large");
  ```

  Other providers' defaults are unchanged. Jina stays on v3 deliberately: v4
  and the v5 text models are released, but their response schema and
  task-adapter names are unverified against this crate (v5 documents four
  adapters where v3 takes `separation`), and defaulting to a model whose
  `.input_type()` mapping may be wrong is worse than staying a generation
  behind.

### Fixed

- The cost summary in `examples/benchmark.rs` quoted $0.06 / 1M tokens for
  `voyage-3-large`. The rate is $0.18; the example now names `voyage-4-large`
  at its actual $0.12.


### Changed

- **`tiktoken` floor raised to 4.1.1.** 4.1 resolves model ids the way the
  provider APIs spell them, so a caller passing the id it just sent to the API
  gets a cost instead of `None`. 4.1.1 adds the embedding models two of this
  crate's defaults name: `gemini-embedding-001` and the Voyage family
  (`voyage-3-large` is the Voyage default). Both priced at nothing before.

### Known gaps

- Cost tracking still reports nothing for Cohere, Jina and Mistral embeddings.
  The upstream price table carries no entries for them because their per-token
  rates are not published on a vendor page — a guessed rate would bill someone
  wrongly and silently. `tests/cost_tracking.rs` records where the line falls
  so it moves deliberately.

## [0.5.0] - 2026-08-13

### Changed

- **`tiktoken` floor `"3.5"` → `"4"`, with `default-features = false`.**

  The immediate reason is ecosystem coherence: `chunkedrs` 2.0 requires
  `tiktoken` 4, so a project using both and enabling `cost-tracking` would
  otherwise compile two majors of it.

  The better reason is that `default-features = false` is now the honest
  declaration. This crate uses exactly one thing from `tiktoken` — the pricing
  tables — and never encodes, counts, or constructs an encoding. Before 4.0
  there was no way to say that, so `cost-tracking` dragged in every tokenizer
  vocabulary. `tiktoken`'s rlib drops from 29.7 MB to 2.8 MB (−91%).

  **Final binaries are unchanged.** Measured both ways on a release build and
  on a debug build: identical to within 16 bytes. The linker was already
  dropping data that nothing referenced, so the win here is in build artifacts
  and caches, not in what ships. Compile time was measured too and the
  difference did not clear single-run noise, so no claim is made about it.

  `tests/cost_tracking.rs::cost_tracking_carries_no_tokenizer_vocabulary`
  asserts `list_encodings()` is empty, so the declaration cannot drift back
  from what the code actually uses.

  No token ids, pricing values, or public API changed. `estimate_cost` returns
  what it returned before.

- **`cost-tracking = ["tiktoken"]` → `["dep:tiktoken"]`.** The old spelling
  also created an implicit `tiktoken` feature; enabling it on its own compiled
  the dependency while every `#[cfg(feature = "cost-tracking")]` site stayed
  switched off, which did nothing but cost build time. `cost-tracking` remains
  the documented and only way to turn this on.

### Docs

- The install snippets in all three READMEs still said `embedrs = "0.3"` two
  releases after 0.4.0 shipped. Updated.
- The MSRV badge still read 1.94 after 0.4.0 removed the `rust-version` pin.
  Removed.

## [0.4.0] - 2026-06-07

### Added
- **Mistral provider** — `Client::mistral(key)` / `mistral_compatible(key, url)`.
  Default model `mistral-embed`; `codestral-embed-2505` also supported via
  `.model(...)`. Sixth cloud provider.
- **`Client::with_http_client(reqwest::Client)`** — share one connection pool
  across providers (helps fallback chains that span multiple hosts) or inject
  custom TLS / proxies / headers. Propagates to all fallbacks registered up to
  the call site.
- **Retry-After honored** on 429 / 503 — `Error::Api` now carries
  `retry_after: Option<Duration>`; the retry loop uses it instead of the
  configured backoff curve when present. Delta-seconds form only;
  HTTP-date falls back to backoff.
- `voyage-4-large` / `voyage-4` / `voyage-4-lite`, `gemini-embedding-2`,
  `codestral-embed-2505`, `jina-embeddings-v4` documented (model id is free-form,
  no API change needed). Voyage v4 + Gemini v2 + Codestral covered by test
  fixtures; Jina v4 single-vector schema not yet verified — try and please
  report deser failures.
- `#![warn(missing_docs)]` and `[package.metadata.docs.rs] all-features = true`
  so docs.rs builds the full surface (was previously hiding `local`,
  `cost-tracking`, `tracing`).

### Changed
- **`Error::Api` is now `#[non_exhaustive]`** with new `retry_after` field.
  External pattern matches need `..` rest. (Future field additions won't
  re-break.)
- `similarity::{cosine_similarity, dot_product, euclidean_distance}` rewritten
  with 8-lane independent accumulators so LLVM autovectorizes at
  `-C opt-level=3` (default for `--release`). Cross-platform — no `wide` /
  `std::simd` / cfg arch. Numerical drift vs. textbook scalar is small FP
  epsilon (test pins ≤1e-5 cos / ≤1e-3 rel dot at dim 1024/1536/1537).
- `tiktoken` floor `"3.1"` → `"3.5"`, `tokenizers` `"0.22"` → `"0.23"`,
  plus a `cargo update` sweep.
- Dropped pinned `rust-version = "1.94"` — MSRV floats with the toolchain.

## [0.3.3] - 2026-04-24

### Changed
- Smoke-test release via the new repo's GitHub Actions publish workflow.
  No code changes.

## [0.3.2] - 2026-04-24

### Changed
- Moved to the standalone `goliajp/rust-embeddings` repository.
  No code changes; `repository` URL updated. `tiktoken` dep switched from
  workspace path to crates.io (`tiktoken = "3.1"`, optional, `cost-tracking` feature).

### Added
- `examples/embedding_models/` — the reproducible experiment that picked the
  `local()` / `cloud()` defaults (MiniLM-L6-v2 + OpenAI). Previously kept
  in a separate crate; moved here since it only ever evaluated `embedrs`.
  Run: `cargo run --example embedding_models --features local --release`

## [0.3.1] - 2026-04

- Previous release.
