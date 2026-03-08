//! # benchrs
//!
//! Reproducible benchmark experiments for airs project decisions.
//!
//! Each experiment lives as a standalone binary under `src/bin/`.
//! Pre-recorded data ships in `data/` so anyone can reproduce analysis
//! without API keys.
//!
//! ## Running an experiment
//!
//! ```bash
//! # analyze pre-recorded data (no API keys needed)
//! cargo run --bin embedding_models
//!
//! # re-record with live API calls (requires keys in .env.local)
//! cargo run --bin embedding_models -- --record
//! ```

pub mod stats;
