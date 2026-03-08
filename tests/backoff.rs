use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Respond, ResponseTemplate};

struct RetryResponder {
    call_count: Arc<AtomicU32>,
    succeed_after: u32,
}

impl Respond for RetryResponder {
    fn respond(&self, _request: &wiremock::Request) -> ResponseTemplate {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        if count < self.succeed_after {
            ResponseTemplate::new(429).set_body_string("rate limited")
        } else {
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
            }))
        }
    }
}

#[tokio::test]
async fn backoff_retries_on_429() {
    let server = MockServer::start().await;
    let call_count = Arc::new(AtomicU32::new(0));

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(RetryResponder {
            call_count: call_count.clone(),
            succeed_after: 2,
        })
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri()).with_retry_backoff(
        embedrs::BackoffConfig {
            base_delay: std::time::Duration::from_millis(10),
            max_delay: std::time::Duration::from_millis(100),
            jitter: false,
            max_http_retries: 3,
        },
    );

    let result = client.embed(vec!["test".into()]).await.unwrap();
    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(call_count.load(Ordering::SeqCst), 3); // 2 failures + 1 success
}

#[tokio::test]
async fn no_backoff_fails_immediately() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let client = embedrs::Client::openai_compatible("test-key", &server.uri());
    let err = client.embed(vec!["test".into()]).await.unwrap_err();
    assert!(matches!(err, embedrs::Error::Api { status: 429, .. }));
}
