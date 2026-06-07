use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mock_mistral_response() -> serde_json::Value {
    serde_json::json!({
        "id": "embd-aad6fc62b17349b192ef09225058bc45",
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
        ],
        "model": "mistral-embed",
        "usage": {"prompt_tokens": 9, "total_tokens": 9, "completion_tokens": 0}
    })
}

#[tokio::test]
async fn embed_mistral_basic() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("Authorization", "Bearer mistral-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_mistral_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::mistral_compatible("mistral-key", &server.uri());
    let result = client
        .embed(vec!["hello".into(), "world".into()])
        .await
        .unwrap();

    assert_eq!(result.embeddings.len(), 2);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(result.usage.total_tokens, 9);
    assert_eq!(result.model, "mistral-embed");
}

#[tokio::test]
async fn embed_mistral_codestral_model_override() {
    let server = MockServer::start().await;

    let response = serde_json::json!({
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.7, 0.8, 0.9]}],
        "model": "codestral-embed-2505",
        "usage": {"prompt_tokens": 4, "total_tokens": 4}
    });

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let client = embedrs::Client::mistral_compatible("mistral-key", &server.uri());
    let result = client
        .embed(vec!["fn main() {}".into()])
        .model("codestral-embed-2505")
        .await
        .unwrap();

    assert_eq!(result.model, "codestral-embed-2505");
    assert_eq!(result.embeddings[0], vec![0.7, 0.8, 0.9]);
}

#[tokio::test]
async fn embed_mistral_api_error_429() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limit"))
        .mount(&server)
        .await;

    let client = embedrs::Client::mistral_compatible("mistral-key", &server.uri());
    let err = client.embed(vec!["hi".into()]).await.unwrap_err();
    assert!(matches!(err, embedrs::Error::Api { status: 429, .. }));
}
