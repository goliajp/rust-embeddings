/// local: embed texts using the local model (all-MiniLM-L6-v2) and compute similarity
///
/// no API key needed — model weights are downloaded from HuggingFace on first run (~23MB)
///
/// run: cargo run -p embedrs --features local --example local --release
use embedrs::{cosine_similarity, local};

#[tokio::main]
async fn main() -> embedrs::Result<()> {
    // create a local client (defaults to all-MiniLM-L6-v2, 384 dimensions)
    let client = local();

    let texts: Vec<String> = vec![
        "Cats are cute and playful pets.".into(),
        "Dogs are loyal and adorable companions.".into(),
        "Quantum physics explores subatomic particles.".into(),
        "The theory of relativity changed modern physics.".into(),
        "I love cooking Italian pasta dishes.".into(),
    ];

    println!("embedding {} texts with local model...", texts.len());
    let result = client.embed(texts.clone()).await?;

    println!("model: {}", result.model);
    println!("dimensions: {}", result.embeddings[0].len());
    println!("total tokens: {}\n", result.usage.total_tokens);

    // compute pairwise cosine similarity
    let n = result.embeddings.len();
    let mut best_sim = f32::NEG_INFINITY;
    let mut best_pair = (0, 0);

    println!("pairwise cosine similarity:");
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&result.embeddings[i], &result.embeddings[j]);
            println!("  [{i}] vs [{j}]: {sim:.4}");
            if sim > best_sim {
                best_sim = sim;
                best_pair = (i, j);
            }
        }
    }

    println!("\nmost similar pair:");
    println!("  \"{}\"", texts[best_pair.0]);
    println!("  \"{}\"", texts[best_pair.1]);
    println!("  cosine similarity: {best_sim:.4}");

    // demonstrate that the same API works identically to cloud
    println!("\n--- same code works with cloud too ---");
    println!("  just change: embedrs::local() → embedrs::cloud(\"sk-...\")");

    Ok(())
}
