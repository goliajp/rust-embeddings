use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mock_openai_response(n: usize) -> serde_json::Value {
    let data: Vec<serde_json::Value> = (0..n)
        .map(|i| {
            serde_json::json!({
                "object": "embedding",
                "index": i,
                "embedding": [0.1 * (i as f64 + 1.0), 0.2, 0.3]
            })
        })
        .collect();

    serde_json::json!({
        "object": "list",
        "data": data,
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": n * 5, "total_tokens": n * 5}
    })
}

#[tokio::test]
async fn batch_embed_splits_and_merges() {
    let server = MockServer::start().await;

    // respond to any number of texts
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response(2)))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let texts: Vec<String> = (0..4).map(|i| format!("text {i}")).collect();
    let result = client
        .embed_batch(texts)
        .chunk_size(2)
        .concurrency(2)
        .await
        .unwrap();

    // 2 chunks × 2 embeddings = 4 total
    assert_eq!(result.embeddings.len(), 4);
    assert!(result.usage.total_tokens > 0);
}

#[tokio::test]
async fn batch_embed_empty() {
    let client = embedrs::Client::openai("test-key");
    let result = client.embed_batch(vec![]).await.unwrap();
    assert!(result.embeddings.is_empty());
    assert_eq!(result.usage.total_tokens, 0);
}

#[tokio::test]
async fn batch_embed_single_chunk() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response(3)))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let texts: Vec<String> = (0..3).map(|i| format!("text {i}")).collect();
    let result = client.embed_batch(texts).chunk_size(100).await.unwrap();

    assert_eq!(result.embeddings.len(), 3);
}
