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

    let client = embedrs::Client::gemini_compatible("gemini-key", server.uri());
    let result = client.embed(vec!["hello".into()]).await.unwrap();

    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(result.embeddings[0], vec![0.1, 0.2, 0.3]);
}

#[tokio::test]
async fn embed_gemini_embedding_2_model_override() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path_regex(r".*models/gemini-embedding-2:embedContent.*"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_gemini_single_response()))
        .mount(&server)
        .await;

    let client = embedrs::Client::gemini_compatible("gemini-key", server.uri());
    let result = client
        .embed(vec!["hello".into()])
        .model("gemini-embedding-2")
        .await
        .unwrap();
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

    let client = embedrs::Client::gemini_compatible("gemini-key", server.uri());
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

    let client = embedrs::Client::gemini_compatible("gemini-key", server.uri());
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

    let client = embedrs::Client::gemini_compatible("gemini-key", server.uri());
    let err = client.embed(vec!["test".into()]).await.unwrap_err();
    assert!(matches!(err, embedrs::Error::Api { status: 500, .. }));
}

#[tokio::test]
async fn api_key_goes_in_a_header_not_the_url() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(r".*:embedContent$"))
        .and(wiremock::matchers::header("x-goog-api-key", "secret-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_gemini_single_response()))
        .mount(&server)
        .await;
    let client = embedrs::Client::gemini_compatible("secret-key", server.uri());
    client.embed(vec!["hello".into()]).await.unwrap();
    let requests = server.received_requests().await.unwrap();
    assert!(!requests[0].url.as_str().contains("secret-key"));
}

#[tokio::test]
async fn api_error_does_not_echo_the_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(403).set_body_string("forbidden"))
        .mount(&server)
        .await;
    let client = embedrs::Client::gemini_compatible("secret-key", server.uri());
    let err = client
        .embed(vec!["a".into(), "b".into()])
        .await
        .unwrap_err();
    assert!(!format!("{err} {err:?}").contains("secret-key"));
}

#[tokio::test]
async fn token_count_comes_from_usage_metadata() {
    let server = MockServer::start().await;
    let body = serde_json::json!({
        "embeddings": [{"values": [0.1]}, {"values": [0.2]}],
        "usageMetadata": {"promptTokenCount": 7}
    });
    Mock::given(method("POST"))
        .and(path_regex(r".*:batchEmbedContents$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    let client = embedrs::Client::gemini_compatible("k", server.uri());
    let result = client.embed(vec!["a".into(), "b".into()]).await.unwrap();
    assert_eq!(result.usage.total_tokens, 7);
}

#[tokio::test]
async fn gemini_embedding_2_sends_no_task_type() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_gemini_single_response()))
        .mount(&server)
        .await;
    let client = embedrs::Client::gemini_compatible("k", server.uri());
    client
        .embed(vec!["rivers".into()])
        .model("gemini-embedding-2")
        .input_type(embedrs::InputType::SearchQuery)
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let sent: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(sent.get("taskType").is_none());
    assert_eq!(
        sent["content"]["parts"][0]["text"],
        "task: search result | query: rivers"
    );
}
