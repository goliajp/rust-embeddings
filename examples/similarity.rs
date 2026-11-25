/// similarity: embed texts and find the most similar pair via cosine similarity
///
/// run: cargo run -p embedrs --example similarity
///
/// requires OPENAI_API_KEY in environment or .env.local file
use embedrs::{cloud, cosine_similarity};

#[tokio::main]
async fn main() -> embedrs::Result<()> {
    dotenvy::from_filename(".env.local").ok();

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY required");

    let client = cloud(&api_key);

    // texts spanning different topics — some pairs should be more similar than others
    let texts: Vec<String> = vec![
        "Cats are cute and playful pets.".into(),
        "Dogs are loyal and adorable companions.".into(),
        "Quantum physics explores subatomic particles.".into(),
        "The theory of relativity changed modern physics.".into(),
        "I love cooking Italian pasta dishes.".into(),
    ];

    let result = client.embed(texts.clone()).await?;
    println!("model: {}", result.model);
    println!(
        "embedded {} texts ({} dimensions)\n",
        result.len(),
        result[0].len()
    );

    // compute pairwise cosine similarity
    let n = result.len();
    let mut best_sim = f32::NEG_INFINITY;
    let mut best_pair = (0, 0);

    println!("pairwise cosine similarity:");
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&result[i], &result[j]);
            println!(
                "  [{i}] vs [{j}]: {sim:.4}  ({} <-> {})",
                short(&texts[i]),
                short(&texts[j])
            );
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

    Ok(())
}

// truncate text for display
fn short(s: &str) -> &str {
    if s.len() > 35 { &s[..35] } else { s }
}
