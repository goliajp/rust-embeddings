use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn mock_gemini_single_response() -> serde_json::Value {
    serde_json::json!({
        "embedding": {
            "values": [0.1, 0.2, 0.3]
        }
    })
}

fn mock_gemini_batch_response() -> serde_json::Value {
    serde_json::json!({
        "embeddings": [
            {"values": [0.1, 0.2, 0.3]},
            {"values": [0.4, 0.5, 0.6]}
        ]
    })
}

#[tokio::test]
async fn embed_gemini_single() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r".*:embedContent.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_gemini_single_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::gemini_compatible("gemini-key", &server.uri());
    let result = client.embed(vec!["hello".into()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
}

#[tokio::test]
async fn embed_gemini_batch() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r".*:batchEmbedContents.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_gemini_batch_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::gemini_compatible("gemini-key", &server.uri());
    let result = client
        .embed(vec!["hello".into(), "world".into()])
        .await
        .unwrap();

    assert_eq!(result.embeddings.len(), 2);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
    assert_eq!(result.embeddings[1], vec![0.4, 0.5, 0.6]);
}

#[tokio::test]
async fn embed_gemini_with_input_type() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r".*:embedContent.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_gemini_single_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::gemini_compatible("gemini-key", &server.uri());
    let result = client
        .embed(vec!["test".into()])
        .input_type(embedrs::InputType::SearchDocument)
        .await
        .unwrap();
    assert!(!result.embeddings.is_empty());
}

#[tokio::test]
async fn embed_gemini_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r".*:embedContent.*"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
        .mount(&server)
        .await;

    let client = embedrs::Client::gemini_compatible("gemini-key", &server.uri());
    let err = client.embed(vec!["test".into()]).await.unwrap_err();
    assert!(matches!(err, embedrs::Error::Api { status: 500, .. }));
}
