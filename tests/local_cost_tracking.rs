//! Tests for local + cost-tracking feature combination
//!
//! These tests require both features enabled:
//! cargo test -p embedrs --features "local,cost-tracking" --test local_cost_tracking

#![cfg(all(feature = "local", feature = "cost-tracking"))]

#[test]
fn local_model_not_in_tiktoken_pricing() {
    // local model names (e.g. "all-MiniLM-L6-v2") are not OpenAI models,
    // so tiktoken pricing should return None
    let cost = tiktoken::pricing::estimate_cost("all-MiniLM-L6-v2", 1000, 0);
    assert!(
        cost.is_none(),
        "local model should have no pricing data in tiktoken"
    );
}

#[test]
fn usage_cost_none_for_local_model() {
    // simulate what the client does: build a Usage with cost from tiktoken
    let model = "all-MiniLM-L6-v2";
    let total_tokens = 100u32;
    let cost = tiktoken::pricing::estimate_cost(model, total_tokens as u64, 0);
    let usage = embedrs::Usage { total_tokens, cost };
    assert_eq!(usage.total_tokens, 100);
    assert!(
        usage.cost.is_none(),
        "local model usage.cost should be None"
    );
}

#[test]
fn usage_accumulation_with_cost_field() {
    let mut usage = embedrs::Usage::default();
    assert_eq!(usage.total_tokens, 0);
    assert!(usage.cost.is_none());

    // accumulate tokens from a "local" request (cost = None)
    usage.total_tokens += 50;
    // cost stays None since local models have no pricing

    // accumulate again
    usage.total_tokens += 75;
    assert_eq!(usage.total_tokens, 125);
    assert!(
        usage.cost.is_none(),
        "cost should remain None when no pricing data is available"
    );
}
