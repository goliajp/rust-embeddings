/// basic: embed texts with OpenAI and display usage stats
///
/// run: cargo run -p embedrs --example basic
///
/// requires OPENAI_API_KEY in environment or .env.local file
use embedrs::{EmbedResult, cloud};

#[tokio::main]
async fn main() -> embedrs::Result<()> {
    // load API key from .env.local (project root or crate dir)
    dotenvy::from_filename(".env.local").ok();
    dotenvy::from_filename(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join(".env.local"),
    )
    .ok();

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY required");

    // create a cloud client (defaults to OpenAI text-embedding-3-small)
    let client = cloud(&api_key);

    // embed a batch of texts
    let texts: Vec<String> = vec![
        "Rust is a systems programming language focused on safety and performance.".into(),
        "The quick brown fox jumps over the lazy dog.".into(),
        "Machine learning models can generate human-like text.".into(),
        "Tokyo is the capital of Japan.".into(),
    ];

    let result: EmbedResult = client.embed(texts).await?;

    // display results
    println!("model: {}", result.model);
    println!("texts embedded: {}", result.embeddings.len());
    println!("dimensions: {}", result.embeddings[0].len());
    println!("total tokens: {}", result.usage.total_tokens);

    // show first few values of each embedding
    for (i, embedding) in result.embeddings.iter().enumerate() {
        let preview: Vec<String> = embedding
            .iter()
            .take(5)
            .map(|v| format!("{v:.4}"))
            .collect();
        println!("  [{i}] first 5 values: [{}]", preview.join(", "));
    }

    Ok(())
}
