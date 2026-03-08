# benchrs

Reproducible benchmark experiments for airs project decisions.

Every major technical decision in airs is backed by a recorded experiment. Each experiment is a standalone binary with pre-recorded data — anyone can reproduce the analysis without API keys or network access.

## Philosophy

> "用就要好用" — if we build it, it must be great. Opinions must be backed by data.

benchrs is the data. When documentation says "we recommend X over Y", there's an experiment here that proves it.

Results from these experiments drive the defaults in [embedrs](https://crates.io/crates/embedrs) and other airs crates.

## Experiments

| Experiment | Decision | Binary |
|------------|----------|--------|
| [Embedding Model Selection](#embedding-model-selection) | `local()` = MiniLM-L6-v2, `cloud()` = OpenAI | `embedding_models` |

## Running experiments

```bash
# analyze pre-recorded data (no API keys needed)
cargo run -p benchrs --bin embedding_models --release

# re-record with live API calls (requires keys in .env.local)
cargo run -p benchrs --bin embedding_models --release -- --record
```

## Adding a new experiment

1. Create `src/bin/<experiment_name>.rs`
2. Add `[[bin]]` entry to `Cargo.toml`
3. Define test data inline (self-contained, no external files needed for test definitions)
4. Implement `--record` mode to capture live data → `data/<experiment_name>.json.zst`
5. Implement default mode to load and analyze pre-recorded data
6. Document: methodology, environment, results, and decision in the binary's doc comment
7. Update this README

## Experiment structure

Each experiment binary follows this pattern:

```rust
/// # Experiment: <Name>
///
/// **Decision:** <What this experiment decides>
/// **Date:** <When recorded>
/// **Environment:** <Platform, Rust version, etc.>
///
/// ## Methodology
/// <How the experiment works>
///
/// ## How to run
/// <Commands>

fn main() {
    if args.contains("--record") {
        record();   // live API calls → save compressed data
    } else {
        analyze();  // load data → compute metrics → print results
    }
}
```

---

## Embedding Model Selection

**Decision:** Which embedding model should embedrs use as default for `local()` and `cloud()`?

**Date:** 2026-03-08
**Environment:** macOS Darwin 25.3.0, Apple Silicon, Rust 1.85, release mode

### Models tested

| Model | Type | Size | Dimensions |
|-------|------|------|------------|
| all-MiniLM-L6-v2 | local | 23MB | 384 |
| all-MiniLM-L12-v2 | local | 133MB | 384 |
| bge-small-en-v1.5 | local | 133MB | 384 |
| gte-small | local | 67MB | 384 |
| text-embedding-3-small | cloud (OpenAI) | — | 1536 |
| gemini-embedding-001 | cloud (Google) | — | 3072 |
| embed-v4.0 | cloud (Cohere) | — | 1536 |
| voyage-3-large | cloud (Voyage) | — | 1024 |

### Methodology

8 dimensions, 184 unique texts:

| # | Dimension | Metric | What it measures |
|---|-----------|--------|------------------|
| 1 | Graded Similarity | Spearman ρ + discrimination gap | Ranking accuracy vs 32 human-scored pairs (0-5) |
| 2 | Retrieval | Top-1 accuracy + MRR | 9 queries × 4 candidates (EN/ZH/JA) |
| 3 | Multilingual | ρ per language | English (21), Chinese (11), Japanese (5) pairs |
| 4 | Cross-lingual | Cosine similarity | 6 sentence groups × 3 languages (same meaning) |
| 5 | Length Sensitivity | ρ by text length | Short (<50), medium (50-200), long (200+) chars |
| 6 | Robustness | Cosine similarity | 15 variants: typos, casing, word order changes |
| 7 | Clustering | NN purity + separation ratio | 4 topics × 5 texts each |
| 8 | Throughput | texts/sec | 500 identical-length texts (live only) |

### Results

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

### Decision

**`local()` → all-MiniLM-L6-v2**
- 23MB — the only model small enough for app embedding (others are 67-133MB)
- Best clustering separation at 8.73x (2nd place is 4.38x) — it sees real structure in data
- 100% retrieval accuracy
- English ρ=0.92 — beats most cloud models on English text
- 12-layer models (L12, BGE, GTE) are 3-6x larger with no meaningful quality improvement
- Known weakness: poor on Chinese/Japanese (ρ=0.60-0.65) and cross-lingual (0.25)

**`cloud()` → OpenAI text-embedding-3-small**
- Best discrimination gap at 0.58 (dissimilar texts avg cosine = 0.09, closest to zero)
- 100% retrieval accuracy, MRR=1.0
- Balanced multilingual: EN=0.91, ZH=0.88, JA=0.90 — no weak language
- Cheapest cloud option at $0.02/1M tokens
- Gemini has higher ρ (0.94) but poor discrimination (0.30) and retrieval miss (89%)
- Cohere matches quality but costs 5x more ($0.10/1M tokens)
