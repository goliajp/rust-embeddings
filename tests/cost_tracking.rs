#![cfg(feature = "cost-tracking")]

mod common;

use common::{ok, unused};
use embedrs::{Client, Endpoint};

fn openai_body(model: &str) -> String {
    format!(
        r#"{{"object":"list","data":[{{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}}],"model":"{model}","usage":{{"prompt_tokens":10,"total_tokens":10}}}}"#
    )
}

#[test]
fn cost_estimated_for_known_model() {
    let c = Client::new(Endpoint::openai_compatible("k", "http://x"), unused());
    let r = c
        .prepare(["hello"])
        .unwrap()
        .finish(ok(&openai_body("text-embedding-3-small")))
        .unwrap();
    assert_eq!(r.usage.total_tokens, Some(10));
    assert!(r.usage.cost.is_some_and(|c| c > 0.0));
}

#[test]
fn cost_none_for_unknown_or_local_model() {
    let c =
        Client::new(Endpoint::openai_compatible("k", "http://x"), unused()).with_model("my-model");
    let r = c
        .prepare(["hello"])
        .unwrap()
        .finish(ok(&openai_body("my-model")))
        .unwrap();
    assert_eq!(r.usage.total_tokens, Some(10));
    assert_eq!(r.usage.cost, None);
    assert!(tiktoken::pricing::estimate_cost("all-MiniLM-L6-v2", 1000, 0).is_none());
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
// token-count compiles in cl100k_base on purpose
#[cfg(not(feature = "token-count"))]
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

/// Which of the cloud defaults can actually be costed.
///
/// `cost` is `Option`, so a provider the price table does not carry reports
/// nothing and explains nothing — the number simply never appears. This test
/// records where that line currently falls, so it moves deliberately: the
/// table gained `gemini-embedding-001` in tiktoken 4.1, the Voyage family in
/// 4.1.1 and `mistral-embed` in 5.0. Cohere and Jina embeddings are still
/// absent upstream — their per-token rates are not published on a vendor page,
/// and a guessed rate bills someone wrongly and silently.
#[test]
fn default_models_that_the_price_table_carries() {
    for id in [
        "text-embedding-3-small",
        "gemini-embedding-001",
        "voyage-3-large",
        "voyage-4-large",
        "mistral-embed",
    ] {
        assert!(
            tiktoken::pricing::estimate_cost(id, 1_000_000, 0).is_some_and(|c| c > 0.0),
            "{id} must be priced"
        );
    }

    for id in ["embed-v4.0", "jina-embeddings-v3"] {
        assert!(
            tiktoken::pricing::estimate_cost(id, 1_000_000, 0).is_none(),
            "{id} is now priced upstream — drop it from this list and say so in the changelog"
        );
    }
}

/// Gemini reports its count in `usageMetadata`; without one the count is
/// unknown, so there is no cost rather than a cost of zero.
#[test]
fn gemini_cost_follows_reported_usage() {
    for (body, priced) in [
        (
            r#"{"embedding":{"values":[0.1]},"usageMetadata":{"promptTokenCount":1000}}"#,
            true,
        ),
        (r#"{"embedding":{"values":[0.1]}}"#, false),
    ] {
        let c = Client::new(Endpoint::gemini("k"), unused());
        let r = c.prepare(["hello"]).unwrap().finish(ok(body)).unwrap();
        assert_eq!(r.usage.cost.is_some_and(|c| c > 0.0), priced);
        if !priced {
            assert_eq!(r.usage.cost, None);
        }
    }
}
