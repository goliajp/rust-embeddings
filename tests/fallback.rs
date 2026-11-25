use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Respond, ResponseTemplate};

fn success_body() -> serde_json::Value {
    serde_json::json!({
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 5, "total_tokens": 5}
    })
}

struct CountingResponder {
    call_count: Arc<AtomicU32>,
    status: u16,
    body: Option<serde_json::Value>,
}

impl Respond for CountingResponder {
    fn respond(&self, _request: &wiremock::Request) -> ResponseTemplate {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let resp = ResponseTemplate::new(self.status);
        match &self.body {
            Some(body) => resp.set_body_json(body),
            None => resp.set_body_string("internal server error"),
        }
    }
}

#[tokio::test]
async fn fallback_on_primary_failure() {
    let primary = MockServer::start().await;
    let fallback = MockServer::start().await;

    let primary_count = Arc::new(AtomicU32::new(0));
    let fallback_count = Arc::new(AtomicU32::new(0));

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(CountingResponder {
            call_count: primary_count.clone(),
            status: 500,
            body: None,
        })
        .mount(&primary)
        .await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(CountingResponder {
            call_count: fallback_count.clone(),
            status: 200,
            body: Some(success_body()),
        })
        .mount(&fallback)
        .await;

    let client = embedrs::Client::openai_compatible("key", primary.uri())
        .with_fallback(embedrs::Client::openai_compatible("key2", fallback.uri()));

    let result = client.embed(vec!["test".into()]).await.unwrap();
    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(primary_count.load(Ordering::SeqCst), 1);
    assert_eq!(fallback_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn fallback_skipped_on_success() {
    let primary = MockServer::start().await;
    let fallback = MockServer::start().await;

    let primary_count = Arc::new(AtomicU32::new(0));
    let fallback_count = Arc::new(AtomicU32::new(0));

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(CountingResponder {
            call_count: primary_count.clone(),
            status: 200,
            body: Some(success_body()),
        })
        .mount(&primary)
        .await;

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(CountingResponder {
            call_count: fallback_count.clone(),
            status: 200,
            body: Some(success_body()),
        })
        .mount(&fallback)
        .await;

    let client = embedrs::Client::openai_compatible("key", primary.uri())
        .with_fallback(embedrs::Client::openai_compatible("key2", fallback.uri()));

    let result = client.embed(vec!["test".into()]).await.unwrap();
    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(primary_count.load(Ordering::SeqCst), 1);
    assert_eq!(fallback_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn fallback_chain_tries_all() {
    let primary = MockServer::start().await;
    let fallback1 = MockServer::start().await;
    let fallback2 = MockServer::start().await;

    let primary_count = Arc::new(AtomicU32::new(0));
    let fallback1_count = Arc::new(AtomicU32::new(0));
    let fallback2_count = Arc::new(AtomicU32::new(0));

    // primary returns 500
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(CountingResponder {
            call_count: primary_count.clone(),
            status: 500,
            body: None,
        })
        .mount(&primary)
        .await;

    // fallback1 returns 500
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(CountingResponder {
            call_count: fallback1_count.clone(),
            status: 500,
            body: None,
        })
        .mount(&fallback1)
        .await;

    // fallback2 succeeds
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(CountingResponder {
            call_count: fallback2_count.clone(),
            status: 200,
            body: Some(success_body()),
        })
        .mount(&fallback2)
        .await;

    let client = embedrs::Client::openai_compatible("key", primary.uri())
        .with_fallback(embedrs::Client::openai_compatible("key2", fallback1.uri()))
        .with_fallback(embedrs::Client::openai_compatible("key3", fallback2.uri()));

    let result = client.embed(vec!["test".into()]).await.unwrap();
    assert_eq!(result.embeddings.len(), 1);
    assert_eq!(primary_count.load(Ordering::SeqCst), 1);
    assert_eq!(fallback1_count.load(Ordering::SeqCst), 1);
    assert_eq!(fallback2_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn fallback_does_not_receive_the_primary_model() {
    let primary = MockServer::start().await;
    let fallback = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&primary)
        .await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .mount(&fallback)
        .await;

    let client = embedrs::Client::openai_compatible("k", primary.uri())
        .with_model("text-embedding-3-large")
        .with_dimensions(256)
        .with_fallback(
            embedrs::Client::mistral_compatible("k2", fallback.uri()).with_model("mistral-embed"),
        );
    client
        .embed(vec!["hello".into()])
        .model("text-embedding-3-small")
        .await
        .unwrap();

    let sent = fallback.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["model"], "mistral-embed");
    assert!(body.get("output_dimension").is_none());
}

#[tokio::test]
async fn fallback_without_its_own_model_uses_its_provider_default() {
    let primary = MockServer::start().await;
    let fallback = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&primary)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_body()))
        .mount(&fallback)
        .await;

    let client = embedrs::Client::openai_compatible("k", primary.uri())
        .with_model("text-embedding-3-large")
        .with_fallback(embedrs::Client::voyage_compatible("k2", fallback.uri()));
    client.embed(vec!["hello".into()]).await.unwrap();

    let sent = fallback.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&sent[0].body).unwrap();
    assert_eq!(body["model"], "voyage-4-large");
}
