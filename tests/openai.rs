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
async fn embed_openai_sends_correct_request_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response()))
        .expect(1)
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri()).with_dimensions(256);
    let _result = client
        .embed(vec!["hello".into(), "world".into()])
        .model("text-embedding-3-large")
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);

    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["model"], "text-embedding-3-large");
    assert_eq!(body["encoding_format"], "float");
    assert_eq!(body["dimensions"], 256);

    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input[0], "hello");
    assert_eq!(input[1], "world");
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

#[tokio::test]
async fn embed_empty_string_input() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.0, 0.0, 0.0]}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        })))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let result = client.embed(vec!["".into()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0].len(), 3);
}

#[tokio::test]
async fn embed_very_long_text() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.5, 0.5, 0.5]}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 8191, "total_tokens": 8191}
        })))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    // 100k character string
    let long_text = "a".repeat(100_000);
    let result = client.embed(vec![long_text]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.usage.total_tokens, 8191);
}
