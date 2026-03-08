use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mock_voyage_response() -> serde_json::Value {
    serde_json::json!({
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
        ],
        "model": "voyage-3-large",
        "usage": {"total_tokens": 5}
    })
}

#[tokio::test]
async fn embed_voyage_basic() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("Authorization", "Bearer voyage-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_voyage_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::voyage_compatible("voyage-key", &server.uri());
    let result = client.embed(vec!["hello".into()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(result.usage.total_tokens, 5);
    assert_eq!(result.model, "voyage-3-large");
}

#[tokio::test]
async fn embed_voyage_with_input_type() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_voyage_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::voyage_compatible("voyage-key", &server.uri());
    let result = client
        .embed(vec!["test".into()])
        .input_type(embedrs::InputType::SearchQuery)
        .await
        .unwrap();
    assert!(!result.embeddings.is_empty());
}
