//! Live comparison of the local models and the cloud providers on the
//! evaluation corpus in `embedding_models/corpus.rs`: seven quality measures,
//! then latency and throughput.
//!
//! The cloud providers run when their key is set: `OPENAI_API_KEY`,
//! `GEMINI_API_KEY`, `COHERE_API_KEY`, `VOYAGE_API_KEY` (environment or
//! `.env.local`).
//!
//! ```bash
//! cargo run --release --features local --example benchmark
//! ```

#[path = "embedding_models/corpus.rs"]
mod corpus;
#[path = "embedding_models/metrics.rs"]
mod metrics;
#[path = "embedding_models/stats.rs"]
mod stats;

use std::time::Instant;

const LOCAL_MODELS: [&str; 4] = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "bge-small-en-v1.5",
    "gte-small",
];

#[tokio::main]
async fn main() -> embedrs::Result<()> {
    dotenvy::from_filename(".env.local").ok();
    let texts = corpus::collect_all_texts();
    println!("corpus: {} unique texts\n", texts.len());

    for name in LOCAL_MODELS {
        println!("local: {name}");
        let client = embedrs::Client::local(name)?;
        let t = Instant::now();
        client.embed(vec!["warmup".into()]).await?;
        println!("  model load: {} ms", t.elapsed().as_millis());
        if let Err(e) = run(&client, &texts).await {
            println!("  error: {e}\n");
        }
    }

    let key = |name: &str| std::env::var(name).ok();
    let cloud = [
        ("openai", key("OPENAI_API_KEY").map(embedrs::Client::openai)),
        ("gemini", key("GEMINI_API_KEY").map(embedrs::Client::gemini)),
        ("cohere", key("COHERE_API_KEY").map(embedrs::Client::cohere)),
        ("voyage", key("VOYAGE_API_KEY").map(embedrs::Client::voyage)),
    ];
    for (name, client) in cloud {
        let Some(client) = client else {
            println!("{name}: skipped (no key)\n");
            continue;
        };
        println!("{name}");
        if let Err(e) = run(&client, &texts).await {
            println!("  error: {e}\n");
        }
    }
    Ok(())
}

async fn run(client: &embedrs::Client, texts: &[String]) -> embedrs::Result<()> {
    let start = Instant::now();
    let result = client.embed_batch(texts.to_vec()).await?;
    let ms = start.elapsed().as_secs_f64() * 1e3;
    println!(
        "  {} ({}-dim): {ms:.0} ms for {} texts, {} tokens",
        result.model,
        result.embeddings[0].len(),
        texts.len(),
        result.usage.total_tokens
    );

    let lookup = |text: &str| -> &[f32] {
        let i = texts.iter().position(|t| t == text).unwrap();
        &result.embeddings[i]
    };
    metrics::report(&lookup);

    let batch: Vec<String> = (0..500)
        .map(|i| format!("This is benchmark sentence number {i} for measuring embedding throughput across providers and models"))
        .collect();
    let t = Instant::now();
    client.embed_batch(batch).await?;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "  [8] throughput:    500 texts in {:.0} ms ({:.0} texts/s)\n",
        secs * 1e3,
        500.0 / secs
    );
    Ok(())
}
