#![cfg(all(feature = "local", feature = "reqwest"))]

//! End-to-end tests for local inference pipeline.
//!
//! These tests download real models from the Hugging Face Hub on first run (67 to 133 MB each).
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

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 384);
    let norm = l2_norm(&result[0]);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2 norm ≈ 1.0, got {norm}"
    );
    assert!(
        result.usage.total_tokens.unwrap() > 0,
        "expected total_tokens > 0, got {:?}",
        result.usage.total_tokens
    );
}

#[tokio::test]
#[ignore]
async fn test_minilm_l12_embed() {
    let client = Client::local("all-MiniLM-L12-v2").unwrap();
    let result = client.embed(vec!["hello world".into()]).await.unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 384);
    let norm = l2_norm(&result[0]);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2 norm ≈ 1.0, got {norm}"
    );
    assert!(
        result.usage.total_tokens.unwrap() > 0,
        "expected total_tokens > 0, got {:?}",
        result.usage.total_tokens
    );
}

#[tokio::test]
#[ignore]
async fn test_bge_small_embed() {
    let client = Client::local("bge-small-en-v1.5").unwrap();
    let result = client.embed(vec!["hello world".into()]).await.unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 384);
    let norm = l2_norm(&result[0]);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2 norm ≈ 1.0, got {norm}"
    );
    assert!(
        result.usage.total_tokens.unwrap() > 0,
        "expected total_tokens > 0, got {:?}",
        result.usage.total_tokens
    );
}

#[tokio::test]
#[ignore]
async fn test_gte_small_embed() {
    let client = Client::local("gte-small").unwrap();
    let result = client.embed(vec!["hello world".into()]).await.unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 384);
    let norm = l2_norm(&result[0]);
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2 norm ≈ 1.0, got {norm}"
    );
    assert!(
        result.usage.total_tokens.unwrap() > 0,
        "expected total_tokens > 0, got {:?}",
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

    assert_eq!(result.len(), 3);

    let sim_dogs_cats = cosine_similarity(&result[0], &result[1]);
    let sim_dogs_quantum = cosine_similarity(&result[0], &result[2]);

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

    assert_eq!(result.len(), 20);
    for (i, emb) in result.iter().enumerate() {
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
        result1[0], result2[0],
        "expected identical embeddings for the same input"
    );
}

#[tokio::test]
#[ignore]
async fn test_empty_text() {
    let client = Client::local("all-MiniLM-L6-v2").unwrap();
    let result = client.embed(vec!["".to_string()]).await.unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 384);
}

fn words(n: usize) -> String {
    let vocab = [
        "river", "stone", "light", "market", "engine", "garden", "signal", "paper",
    ];
    (0..n)
        .map(|i| vocab[(i * 7 + i / 3) % vocab.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

#[tokio::test]
#[ignore]
async fn concurrent_first_use_shares_one_download() {
    let tasks: Vec<_> = (0..4)
        .map(|_| {
            tokio::spawn(async {
                let client = Client::local("all-MiniLM-L6-v2").unwrap();
                client.embed(vec!["concurrent".into()]).await
            })
        })
        .collect();
    for task in tasks {
        let result = task.await.unwrap().unwrap();
        assert_eq!(result[0].len(), 384);
    }
}

#[tokio::test]
#[ignore]
async fn long_text_is_truncated_at_the_model_limit() {
    let long = words(700);
    for (model, limit) in [
        ("bge-small-en-v1.5", 512),
        ("all-MiniLM-L6-v2", 256),
        ("all-MiniLM-L12-v2", 128),
        ("gte-small", 512),
    ] {
        let client = Client::local(model).unwrap();
        let result = client.embed(vec![long.clone()]).await.unwrap();
        assert_eq!(result[0].len(), 384, "{model}");
        assert_eq!(result.usage.total_tokens, Some(limit as u64), "{model}");
    }
}

#[tokio::test]
#[ignore]
async fn token_count_excludes_padding() {
    let client = Client::local("all-MiniLM-L6-v2").unwrap();
    // [CLS] hi [SEP]
    let one = client.embed(vec!["hi".into()]).await.unwrap();
    assert_eq!(one.usage.total_tokens, Some(3));
    // padded to the longer text in the batch, still counted per text
    let two = client.embed(vec!["hi".into(), words(20)]).await.unwrap();
    assert_eq!(two.usage.total_tokens, Some(3 + 22));
}

#[tokio::test]
#[ignore]
async fn mixed_lengths_keep_input_order() {
    let client = Client::local("all-MiniLM-L6-v2").unwrap();
    let texts: Vec<String> = (0..150).map(|i| words(1 + (i * 37) % 120)).collect();
    let batch = client.embed(texts.clone()).await.unwrap();
    for i in [0, 1, 7, 64, 65, 149] {
        let single = client.embed(vec![texts[i].clone()]).await.unwrap();
        let sim = cosine_similarity(&batch[i], &single[0]);
        assert!(sim > 0.99999, "text {i}: cosine {sim}");
    }
}

#[tokio::test]
#[ignore]
async fn result_names_the_model_that_ran() {
    let client = Client::local("all-MiniLM-L6-v2").unwrap();
    let result = client
        .embed(vec!["hello".into()])
        .model("gte-small")
        .await
        .unwrap();
    assert_eq!(result.model, "gte-small");
    let result = client.embed(vec!["hello".into()]).await.unwrap();
    assert_eq!(result.model, "all-MiniLM-L6-v2");
}

#[tokio::test]
#[ignore]
async fn bge_query_gets_its_instruction() {
    let client = Client::local("bge-small-en-v1.5").unwrap();
    let text = vec!["how do rivers form".to_string()];
    let plain = client.embed(text.clone()).await.unwrap();
    let query = client
        .embed(text.clone())
        .input_type(embedrs::InputType::SearchQuery)
        .await
        .unwrap();
    let doc = client
        .embed(text)
        .input_type(embedrs::InputType::SearchDocument)
        .await
        .unwrap();
    assert_eq!(plain[0], doc[0]);
    assert!(query.usage.total_tokens > plain.usage.total_tokens);
    assert!(cosine_similarity(&plain[0], &query[0]) < 0.9999);
}
