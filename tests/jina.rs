use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mock_jina_response() -> serde_json::Value {
    serde_json::json!({
        "model": "jina-embeddings-v3",
        "data": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]}
        ],
        "usage": {"total_tokens": 7}
    })
}

#[tokio::test]
async fn embed_jina_basic() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("Authorization", "Bearer jina-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_jina_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::jina_compatible("jina-key", &server.uri());
    let result = client.embed(vec!["hello".into()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(result.usage.total_tokens, 7);
    assert_eq!(result.model, "jina-embeddings-v3");
}

#[tokio::test]
async fn embed_jina_with_dimensions_and_task() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_jina_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::jina_compatible("jina-key", &server.uri());
    let result = client
        .embed(vec!["test".into()])
        .dimensions(256)
        .input_type(embedrs::InputType::Classification)
        .await
        .unwrap();
    assert!(!result.embeddings.is_empty());
}
