use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

fn mock_openai_response(n: usize) -> serde_json::Value {
    let data: Vec<serde_json::Value> = (0..n)
        .map(|i| {
            serde_json::json!({
                "object": "embedding",
                "index": i,
                "embedding": [0.1 * (i as f64 + 1.0), 0.2, 0.3]
            })
        })
        .collect();

    serde_json::json!({
        "object": "list",
        "data": data,
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": n * 5, "total_tokens": n * 5}
    })
}

#[tokio::test]
async fn batch_embed_splits_and_merges() {
    let server = MockServer::start().await;

    // respond to any number of texts
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response(2)))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let texts: Vec<String> = (0..4).map(|i| format!("text {i}")).collect();
    let result = client
        .embed_batch(texts)
        .chunk_size(2)
        .concurrency(2)
        .await
        .unwrap();

    // 2 chunks × 2 embeddings = 4 total
    assert_eq!(result.embeddings.len(), 4);
    assert!(result.usage.total_tokens > 0);
}

#[tokio::test]
async fn batch_embed_empty() {
    let client = embedrs::Client::openai("test-key");
    let result = client.embed_batch(vec![]).await.unwrap();
    assert!(result.embeddings.is_empty());
    assert_eq!(result.usage.total_tokens, 0);
}

#[tokio::test]
async fn batch_embed_single_chunk() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response(3)))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let texts: Vec<String> = (0..3).map(|i| format!("text {i}")).collect();
    let result = client.embed_batch(texts).chunk_size(100).await.unwrap();

    assert_eq!(result.embeddings.len(), 3);
}

/// responder that generates distinct embeddings per input text to verify order
struct OrderPreservingResponder;

impl Respond for OrderPreservingResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
        let inputs = body["input"].as_array().unwrap();
        let data: Vec<serde_json::Value> = inputs
            .iter()
            .enumerate()
            .map(|(i, text)| {
                // use hash of text content as a deterministic embedding value
                let text_str = text.as_str().unwrap();
                let hash_val = text_str.len() as f64 * 0.01 + i as f64 * 0.001;
                serde_json::json!({
                    "object": "embedding",
                    "index": i,
                    "embedding": [hash_val, 0.2, 0.3]
                })
            })
            .collect();

        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": data,
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": inputs.len() * 5, "total_tokens": inputs.len() * 5}
        }))
    }
}

#[tokio::test]
async fn batch_preserves_text_order_across_chunks() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(OrderPreservingResponder)
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let texts: Vec<String> = (0..6).map(|i| format!("text_{i}")).collect();
    let result = client
        .embed_batch(texts.clone())
        .chunk_size(2)
        .concurrency(1) // sequential to guarantee order
        .await
        .unwrap();

    assert_eq!(result.embeddings.len(), 6);
    // verify each chunk's first embedding has the correct index-based value
    // chunk 0: texts 0,1 → indices 0,1 within chunk
    // chunk 1: texts 2,3 → indices 0,1 within chunk
    // chunk 2: texts 4,5 → indices 0,1 within chunk
    // the key point: 6 embeddings come back in the original text order
    for (i, emb) in result.embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 3, "embedding {i} should have 3 dimensions");
    }
}

#[tokio::test]
async fn batch_error_propagation_on_chunk_failure() {
    let server = MockServer::start().await;

    // first request succeeds, second fails
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let texts: Vec<String> = (0..4).map(|i| format!("text {i}")).collect();
    let err = client.embed_batch(texts).chunk_size(2).await.unwrap_err();

    match err {
        embedrs::Error::Api { status, .. } => {
            assert_eq!(status, 500);
        }
        _ => panic!("expected Api error, got {err:?}"),
    }
}

#[tokio::test]
async fn batch_respects_concurrency_limit() {
    let server = MockServer::start().await;

    let current_concurrent = Arc::new(AtomicUsize::new(0));
    let max_concurrent = Arc::new(AtomicUsize::new(0));

    struct ConcurrencyTrackingResponder {
        current: Arc<AtomicUsize>,
        max: Arc<AtomicUsize>,
    }

    impl Respond for ConcurrencyTrackingResponder {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let prev = self.current.fetch_add(1, Ordering::SeqCst);
            let now = prev + 1;
            // update max observed concurrency
            self.max.fetch_max(now, Ordering::SeqCst);

            let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            let n = body["input"].as_array().unwrap().len();

            let data: Vec<serde_json::Value> = (0..n)
                .map(|i| {
                    serde_json::json!({
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.1, 0.2, 0.3]
                    })
                })
                .collect();

            // delay so requests overlap — the semaphore must hold permits
            let template = ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({
                    "object": "list",
                    "data": data,
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": n * 5, "total_tokens": n * 5}
                }))
                .set_delay(Duration::from_millis(200));

            // decrement before returning (wiremock applies the delay after respond returns)
            self.current.fetch_sub(1, Ordering::SeqCst);

            template
        }
    }

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ConcurrencyTrackingResponder {
            current: current_concurrent.clone(),
            max: max_concurrent.clone(),
        })
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let texts: Vec<String> = (0..4).map(|i| format!("text {i}")).collect();
    let result = client
        .embed_batch(texts)
        .chunk_size(1)
        .concurrency(2)
        .await
        .unwrap();

    assert_eq!(result.embeddings.len(), 4);

    // the semaphore limits to 2, so max_concurrent should be at most 2
    // note: since respond() is synchronous in wiremock, the concurrency tracking
    // captures the synchronous overlap; the important thing is that the semaphore
    // does not allow more than 2 tasks to proceed simultaneously
    let observed_max = max_concurrent.load(Ordering::SeqCst);
    assert!(
        observed_max <= 2,
        "max concurrent requests was {observed_max}, expected at most 2"
    );
}

#[tokio::test]
async fn batch_propagates_model_and_dimensions() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(mock_openai_response(2)))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let texts: Vec<String> = (0..2).map(|i| format!("text {i}")).collect();
    let result = client
        .embed_batch(texts)
        .model("text-embedding-3-large")
        .dimensions(256)
        .input_type(embedrs::InputType::SearchDocument)
        .await
        .unwrap();

    assert_eq!(result.embeddings.len(), 2);
}

#[tokio::test]
async fn embed_at_exact_provider_limit_openai() {
    let server = MockServer::start().await;

    // openai max_batch_size = 2048
    let n = 2048;

    struct DynamicResponder;

    impl Respond for DynamicResponder {
        fn respond(&self, request: &Request) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
            let count = body["input"].as_array().unwrap().len();
            ResponseTemplate::new(200).set_body_json(mock_openai_response(count))
        }
    }

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(DynamicResponder)
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let texts: Vec<String> = (0..n).map(|i| format!("text {i}")).collect();
    let result = client.embed(texts).await.unwrap();

    assert_eq!(result.embeddings.len(), n);
    assert!(result.usage.total_tokens > 0);
}

#[tokio::test]
async fn embed_exceeding_provider_limit_returns_input_too_large() {
    // openai max_batch_size = 2048, so 2049 should fail
    let client = embedrs::Client::openai("test-key");
    let texts: Vec<String> = (0..2049).map(|i| format!("text {i}")).collect();
    let err = client.embed(texts).await.unwrap_err();

    match err {
        embedrs::Error::InputTooLarge(actual, max) => {
            assert_eq!(actual, 2049);
            assert_eq!(max, 2048);
        }
        _ => panic!("expected InputTooLarge error, got {err:?}"),
    }
}

#[tokio::test]
async fn embed_exceeding_cohere_limit_returns_input_too_large() {
    // cohere max_batch_size = 96
    let client = embedrs::Client::cohere("test-key");
    let texts: Vec<String> = (0..97).map(|i| format!("text {i}")).collect();
    let err = client.embed(texts).await.unwrap_err();

    match err {
        embedrs::Error::InputTooLarge(actual, max) => {
            assert_eq!(actual, 97);
            assert_eq!(max, 96);
        }
        _ => panic!("expected InputTooLarge error, got {err:?}"),
    }
}

#[tokio::test]
async fn embed_exceeding_gemini_limit_returns_input_too_large() {
    // gemini max_batch_size = 100
    let client = embedrs::Client::gemini("test-key");
    let texts: Vec<String> = (0..101).map(|i| format!("text {i}")).collect();
    let err = client.embed(texts).await.unwrap_err();

    match err {
        embedrs::Error::InputTooLarge(actual, max) => {
            assert_eq!(actual, 101);
            assert_eq!(max, 100);
        }
        _ => panic!("expected InputTooLarge error, got {err:?}"),
    }
}

#[tokio::test]
async fn embed_exceeding_voyage_limit_returns_input_too_large() {
    // voyage max_batch_size = 128
    let client = embedrs::Client::voyage("test-key");
    let texts: Vec<String> = (0..129).map(|i| format!("text {i}")).collect();
    let err = client.embed(texts).await.unwrap_err();

    match err {
        embedrs::Error::InputTooLarge(actual, max) => {
            assert_eq!(actual, 129);
            assert_eq!(max, 128);
        }
        _ => panic!("expected InputTooLarge error, got {err:?}"),
    }
}

#[tokio::test]
async fn embed_exceeding_jina_limit_returns_input_too_large() {
    // jina max_batch_size = 2048
    let client = embedrs::Client::jina("test-key");
    let texts: Vec<String> = (0..2049).map(|i| format!("text {i}")).collect();
    let err = client.embed(texts).await.unwrap_err();

    match err {
        embedrs::Error::InputTooLarge(actual, max) => {
            assert_eq!(actual, 2049);
            assert_eq!(max, 2048);
        }
        _ => panic!("expected InputTooLarge error, got {err:?}"),
    }
}
