#![cfg(feature = "local")]

//! End-to-end tests for local inference pipeline.
//!
//! These tests download real models from HuggingFace Hub on first run (~23-133MB each).
//! Run with: `cargo test -p embedrs --features local --test local_e2e -- --ignored`

use embedrs::Client;

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    dot / (norm_a * norm_b)
}

#[tokio::test]
#[ignore]
async fn test_minilm_l6_embed_single() {
    let client = Client::local("all-MiniLM-L6-v2").unwrap();
    let result = client.embed(vec!["hello world".into()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 384);
    let norm = l2_norm(&result.embeddings[0]);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2 norm ≈ 1.0, got {norm}"
    );
    assert!(
        result.usage.total_tokens > 0,
        "expected total_tokens > 0, got {}",
        result.usage.total_tokens
    );
}

#[tokio::test]
#[ignore]
async fn test_minilm_l12_embed() {
    let client = Client::local("all-MiniLM-L12-v2").unwrap();
    let result = client.embed(vec!["hello world".into()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 384);
    let norm = l2_norm(&result.embeddings[0]);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2 norm ≈ 1.0, got {norm}"
    );
    assert!(
        result.usage.total_tokens > 0,
        "expected total_tokens > 0, got {}",
        result.usage.total_tokens
    );
}

#[tokio::test]
#[ignore]
async fn test_bge_small_embed() {
    let client = Client::local("bge-small-en-v1.5").unwrap();
    let result = client.embed(vec!["hello world".into()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 384);
    let norm = l2_norm(&result.embeddings[0]);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2 norm ≈ 1.0, got {norm}"
    );
    assert!(
        result.usage.total_tokens > 0,
        "expected total_tokens > 0, got {}",
        result.usage.total_tokens
    );
}

#[tokio::test]
#[ignore]
async fn test_gte_small_embed() {
    let client = Client::local("gte-small").unwrap();
    let result = client.embed(vec!["hello world".into()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 384);
    let norm = l2_norm(&result.embeddings[0]);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2 norm ≈ 1.0, got {norm}"
    );
    assert!(
        result.usage.total_tokens > 0,
        "expected total_tokens > 0, got {}",
        result.usage.total_tokens
    );
}

#[tokio::test]
#[ignore]
async fn test_semantic_similarity() {
    let client = Client::local("all-MiniLM-L6-v2").unwrap();
    let result = client
        .embed(vec![
            "dogs are great pets".into(),
            "cats are wonderful companions".into(),
            "quantum physics is complex".into(),
        ])
        .await
        .unwrap();

    assert_eq!(result.embeddings.len(), 3);

    let sim_dogs_cats = cosine_similarity(&result.embeddings[0], &result.embeddings[1]);
    let sim_dogs_quantum = cosine_similarity(&result.embeddings[0], &result.embeddings[2]);

    assert!(
        sim_dogs_cats > sim_dogs_quantum,
        "expected cosine(dogs, cats) > cosine(dogs, quantum), got {sim_dogs_cats} vs {sim_dogs_quantum}"
    );
}

#[tokio::test]
#[ignore]
async fn test_batch_local() {
    let client = Client::local("all-MiniLM-L6-v2").unwrap();
    let texts: Vec<String> = (0..20).map(|i| format!("sentence number {i}")).collect();
    let result = client.embed(texts).await.unwrap();

    assert_eq!(result.embeddings.len(), 20);
    for (i, emb) in result.embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 384, "embedding {i} has wrong dimension");
        let norm = l2_norm(emb);
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "embedding {i} L2 norm = {norm}, expected ≈ 1.0"
        );
    }
}

#[tokio::test]
#[ignore]
async fn test_deterministic() {
    let client = Client::local("all-MiniLM-L6-v2").unwrap();
    let text = vec!["determinism check".to_string()];

    let result1 = client.embed(text.clone()).await.unwrap();
    let result2 = client.embed(text).await.unwrap();

    assert_eq!(
        result1.embeddings[0], result2.embeddings[0],
        "expected identical embeddings for the same input"
    );
}

#[tokio::test]
#[ignore]
async fn test_empty_text() {
    let client = Client::local("all-MiniLM-L6-v2").unwrap();
    let result = client.embed(vec!["".to_string()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 384);
}
