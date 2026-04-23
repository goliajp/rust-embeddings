#![cfg(feature = "cost-tracking")]

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mock_openai_response(model: &str) -> serde_json::Value {
    serde_json::json!({
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
        ],
        "model": model,
        "usage": {"prompt_tokens": 10, "total_tokens": 10}
    })
}

#[tokio::test]
async fn cost_estimated_for_known_model() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_openai_response("text-embedding-3-small")),
        )
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri())
        .with_model("text-embedding-3-small");
    let result = client.embed(vec!["hello".into()]).await.unwrap();

    assert!(
        result.usage.cost.is_some(),
        "cost should be Some for a known model"
    );
    assert!(result.usage.cost.unwrap() > 0.0, "cost should be positive");
}

#[tokio::test]
async fn cost_none_for_unknown_model() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(mock_openai_response("my-custom-embedding-v1")),
        )
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri())
        .with_model("my-custom-embedding-v1");
    let result = client.embed(vec!["hello".into()]).await.unwrap();

    assert!(
        result.usage.cost.is_none(),
        "cost should be None for an unknown model"
    );
}
