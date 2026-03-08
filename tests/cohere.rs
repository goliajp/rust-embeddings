use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mock_cohere_response() -> serde_json::Value {
    serde_json::json!({
        "id": "123",
        "embeddings": {
            "float": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        },
        "texts": ["hello", "world"],
        "meta": {
            "api_version": {"version": "2"},
            "billed_units": {"input_tokens": 8}
        }
    })
}

#[tokio::test]
async fn embed_cohere_basic() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embed"))
        .and(header("Authorization", "Bearer cohere-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_cohere_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::cohere_compatible("cohere-key", &server.uri());
    let result = client
        .embed(vec!["hello".into(), "world".into()])
        .await
        .unwrap();

    assert_eq!(result.embeddings.len(), 2);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(result.usage.total_tokens, 8);
}

#[tokio::test]
async fn embed_cohere_with_input_type() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_cohere_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::cohere_compatible("cohere-key", &server.uri());
    let result = client
        .embed(vec!["test".into()])
        .input_type(embedrs::InputType::SearchQuery)
        .await
        .unwrap();
    assert!(!result.embeddings.is_empty());
}

#[tokio::test]
async fn embed_cohere_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .mount(&server)
        .await;

    let client = embedrs::Client::cohere_compatible("cohere-key", &server.uri());
    let err = client.embed(vec!["test".into()]).await.unwrap_err();
    assert!(matches!(err, embedrs::Error::Api { status: 400, .. }));
}
