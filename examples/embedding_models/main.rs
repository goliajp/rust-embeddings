mod corpus;
mod metrics;
mod stats;

use serde::{Deserialize, Serialize};
/// # Experiment: Embedding Model Selection
///
/// **Decision:** Which embedding model should embedrs use as default for `local()` and `cloud()`?
///
/// **Date:** 2026-03-08
/// **Environment:** macOS Darwin 25.3.0, Apple Silicon, Rust 1.85 (release mode)
///
/// ## Methodology
///
/// 8 models tested across 8 dimensions:
///
/// | # | Dimension | Metric | What it measures |
/// |---|-----------|--------|------------------|
/// | 1 | Graded Similarity | Spearman ρ | Ranking accuracy vs human scores (0-5) |
/// | 1 | Discrimination | Gap | Separation between similar/dissimilar pairs |
/// | 2 | Retrieval | Top-1, MRR | Can it find the right document? |
/// | 3 | Multilingual | ρ per language | English, Chinese, Japanese separately |
/// | 4 | Cross-lingual | Cosine | Same meaning across languages → same vector? |
/// | 5 | Length sensitivity | ρ per length | Short/medium/long text quality |
/// | 6 | Robustness | Cosine | Resilience to typos, casing, word order |
/// | 7 | Clustering | Separation ratio | Can it group related texts? |
/// | 8 | Throughput | texts/sec | Processing speed |
///
/// ## Models tested
///
/// | Model | Type | Size | Dimensions |
/// |-------|------|------|------------|
/// | all-MiniLM-L6-v2 | local | 91MB | 384 |
/// | all-MiniLM-L12-v2 | local | 133MB | 384 |
/// | bge-small-en-v1.5 | local | 133MB | 384 |
/// | gte-small | local | 67MB | 384 |
/// | text-embedding-3-small | cloud (OpenAI) | — | 1536 |
/// | gemini-embedding-001 | cloud (Google) | — | 3072 |
/// | embed-v4.0 | cloud (Cohere) | — | 1536 |
/// | voyage-3-large | cloud (Voyage) | — | 1024 |
///
/// ## Test corpus
///
/// - 184 unique texts
/// - 32 graded similarity pairs (human-scored 0.0-5.0)
/// - 9 retrieval queries (EN/ZH/JA) with 4 candidates each
/// - 6 cross-lingual groups (same sentence in EN/ZH/JA)
/// - 10 length sensitivity pairs (short/medium/long)
/// - 3 robustness sets (15 variants: typos, casing, word order)
/// - 4 topic clusters × 5 texts each
/// - 500 throughput texts
///
/// ## How to run
///
/// ```bash
/// # analyze pre-recorded data (no API keys needed)
/// cargo run --release --features local --example embedding_models
///
/// # re-record with live API calls
/// # needs OPENAI_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY
/// # in the environment or in .env.local
/// cargo run --release --features local --example embedding_models -- --record
/// ```
use std::collections::HashMap;

/// pre-recorded embeddings for all models
#[derive(Serialize, Deserialize)]
struct RecordedData {
    /// model name → { text → embedding }
    models: HashMap<String, ModelData>,
    /// metadata about the recording
    metadata: Metadata,
}

#[derive(Serialize, Deserialize)]
struct ModelData {
    dimensions: usize,
    embeddings: HashMap<String, Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    date: String,
    platform: String,
    corpus_size: usize,
    models: Vec<String>,
}

const DATA_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/embedding_models/data.json.zst"
);

#[tokio::main]
async fn main() -> embedrs::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let record_mode = args.iter().any(|a| a == "--record");

    if record_mode {
        record().await?;
    } else {
        analyze()?;
    }

    Ok(())
}

const LOCAL_MODELS: [&str; 4] = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "bge-small-en-v1.5",
    "gte-small",
];

async fn embed_all(client: &embedrs::Client, texts: &[String]) -> embedrs::Result<ModelData> {
    let result = client.embed_batch(texts.to_vec()).await?;
    let dimensions = result.embeddings[0].len();
    let embeddings = texts.iter().cloned().zip(result.embeddings).collect();
    Ok(ModelData {
        dimensions,
        embeddings,
    })
}

fn cloud_clients() -> Vec<(&'static str, Option<embedrs::Client>)> {
    let key = |name: &str| std::env::var(name).ok();
    vec![
        (
            "text-embedding-3-small",
            key("OPENAI_API_KEY").map(embedrs::Client::openai),
        ),
        (
            "gemini-embedding-001",
            key("GEMINI_API_KEY").map(embedrs::Client::gemini),
        ),
        (
            "embed-v4.0",
            key("COHERE_API_KEY").map(embedrs::Client::cohere),
        ),
        (
            "voyage-3-large",
            key("VOYAGE_API_KEY").map(|k| embedrs::Client::voyage(k).with_model("voyage-3-large")),
        ),
    ]
}

async fn record() -> embedrs::Result<()> {
    dotenvy::from_filename(".env.local").ok();
    let all_texts = corpus::collect_all_texts();
    println!("recording embeddings for {} texts...\n", all_texts.len());

    let mut data = RecordedData {
        models: HashMap::new(),
        metadata: Metadata {
            date: "2026-03-08".to_string(),
            platform: "macOS Darwin 25.3.0, Apple Silicon, Rust 1.85, release mode".to_string(),
            corpus_size: all_texts.len(),
            models: Vec::new(),
        },
    };

    for name in LOCAL_MODELS {
        let client = embedrs::Client::local(name)?;
        let model = embed_all(&client, &all_texts).await?;
        println!("  {name}: {}-dim", model.dimensions);
        data.models.insert(name.to_string(), model);
        data.metadata.models.push(name.to_string());
    }
    for (name, client) in cloud_clients() {
        let Some(client) = client else {
            println!("  {name}: skipped (no key)");
            continue;
        };
        let model = embed_all(&client, &all_texts).await?;
        println!("  {name}: {}-dim", model.dimensions);
        data.models.insert(name.to_string(), model);
        data.metadata.models.push(name.to_string());
    }

    let json = serde_json::to_vec(&data).unwrap();
    let compressed = zstd::bulk::Compressor::new(19)
        .unwrap()
        .compress(&json)
        .unwrap();
    std::fs::write(DATA_PATH, &compressed).unwrap();
    println!(
        "\nsaved: {DATA_PATH} ({:.1} MB raw, {:.1} MB compressed)",
        json.len() as f64 / 1e6,
        compressed.len() as f64 / 1e6
    );
    Ok(())
}

fn load() -> RecordedData {
    let compressed = std::fs::read(DATA_PATH)
        .unwrap_or_else(|_| panic!("data file not found: {DATA_PATH}\nrun with --record first"));
    let json = zstd::decode_all(compressed.as_slice()).expect("zstd decompression failed");
    serde_json::from_slice(&json).expect("json parse failed")
}

fn analyze() -> embedrs::Result<()> {
    let data = load();
    println!("recorded: {}", data.metadata.date);
    println!("platform: {}", data.metadata.platform);
    println!("corpus:   {} texts", data.metadata.corpus_size);
    println!("models:   {}\n", data.metadata.models.join(", "));

    let order = LOCAL_MODELS
        .into_iter()
        .chain(cloud_clients().into_iter().map(|(name, _)| name));
    for name in order {
        let Some(model) = data.models.get(name) else {
            println!("{name}: no data\n");
            continue;
        };
        println!("{name} ({}-dim)", model.dimensions);
        let lookup = |text: &str| -> &[f32] {
            model
                .embeddings
                .get(text)
                .unwrap_or_else(|| panic!("missing embedding for: {text}"))
        };
        metrics::report(&lookup);
        println!();
    }
    Ok(())
}
