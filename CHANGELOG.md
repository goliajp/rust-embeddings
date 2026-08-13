# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Migrated from `goliajp/airs` mono-repo to standalone `goliajp/rust-embeddings`.
  No code changes; `repository` URL updated. `tiktoken` dep switched from
  workspace path to crates.io (`tiktoken = "3.1"`, optional, `cost-tracking` feature).

### Added
- `examples/embedding_models/` — the reproducible experiment that picked the
  `local()` / `cloud()` defaults (MiniLM-L6-v2 + OpenAI). Previously lived in
  the separate `benchrs` crate inside `goliajp/airs`; folded in here since
  it only ever evaluated `embedrs`.
  Run: `cargo run --example embedding_models --features local --release`

## [0.3.1] - 2026-04

- Previous release (from `goliajp/airs`).
