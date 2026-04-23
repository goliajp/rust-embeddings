# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
