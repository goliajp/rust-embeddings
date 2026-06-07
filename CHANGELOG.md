# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
