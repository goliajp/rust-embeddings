use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mock_openai_response() -> serde_json::Value {
    serde_json::json!({
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 10, "total_tokens": 10}
    })
}

#[tokio::test]
async fn embed_openai_basic() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("Authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let result = client
        .embed(vec!["hello".into(), "world".into()])
        .await
        .unwrap();

    assert_eq!(result.embeddings.len(), 2);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(result.embeddings[1], vec![0.4, 0.5, 0.6]);
    assert_eq!(result.usage.total_tokens, 10);
    assert_eq!(result.model, "text-embedding-3-small");
}

#[tokio::test]
async fn embed_openai_with_dimensions() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri()).with_dimensions(256);
    let result = client.embed(vec!["test".into()]).await.unwrap();
    assert!(!result.embeddings.is_empty());
}

#[tokio::test]
async fn embed_openai_with_model() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let result = client
        .embed(vec!["test".into()])
        .model("text-embedding-3-large")
        .await
        .unwrap();
    assert!(!result.embeddings.is_empty());
}

#[tokio::test]
async fn embed_openai_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let err = client.embed(vec!["test".into()]).await.unwrap_err();
    match err {
        embedrs::Error::Api { status, message } => {
            assert_eq!(status, 429);
            assert!(message.contains("rate limited"));
        }
        _ => panic!("expected Api error"),
    }
}
