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

/// `cost-tracking` uses exactly one thing from tiktoken: the pricing tables.
/// It never encodes, counts, or constructs an encoding — so it should carry no
/// tokenizer vocabulary at all.
///
/// Since tiktoken 4 that is expressible, and this test is what keeps it true.
/// If someone later reaches for `tiktoken::get_encoding` here, or a vocabulary
/// feature gets switched on by a transitive dependency, this fails and says so
/// — rather than several megabytes quietly reappearing in every downstream
/// binary that enables cost tracking.
#[test]
fn cost_tracking_carries_no_tokenizer_vocabulary() {
    assert!(
        tiktoken::list_encodings().is_empty(),
        "embedrs needs pricing data only, but this build compiled in {:?}",
        tiktoken::list_encodings(),
    );
    assert!(
        tiktoken::get_encoding("o200k_base").is_none(),
        "no vocabulary should be constructible from an embedrs build"
    );
}

/// The pricing tables stand on their own without any vocabulary behind them.
#[test]
fn pricing_works_without_vocabularies() {
    let cost = tiktoken::pricing::estimate_cost("text-embedding-3-small", 1_000_000, 0)
        .expect("text-embedding-3-small must have pricing data");
    assert!(cost > 0.0, "cost should be positive, got {cost}");

    assert!(
        tiktoken::pricing::estimate_cost("not-a-real-model-xyz", 1000, 0).is_none(),
        "an unknown model has no pricing"
    );
}
