//! Splitting `embed_batch` input into requests the endpoint accepts.

use std::ops::Range;

use crate::endpoint::Wire;
use crate::input::Inputs;

/// Consecutive runs of inputs, each within `max_inputs` and, when a token
/// cap is known, within `max_tokens` by `count`. Order is kept. An input
/// over the cap on its own still gets a request: the provider decides.
pub(crate) fn split(
    inputs: &Inputs<'_>,
    max_inputs: usize,
    max_tokens: Option<u64>,
    count: impl Fn(&str) -> u64,
) -> Vec<Range<usize>> {
    let max_inputs = max_inputs.max(1);
    let mut out = Vec::new();
    let mut start = 0;
    let mut tokens = 0u64;
    for (i, text) in inputs.iter().enumerate() {
        let t = match max_tokens {
            Some(_) => count(text),
            None => 0,
        };
        let full = i - start == max_inputs;
        let over = max_tokens.is_some_and(|cap| i > start && tokens + t > cap);
        if full || over {
            out.push(start..i);
            start = i;
            tokens = 0;
        }
        tokens += t;
    }
    if start < inputs.len() {
        out.push(start..inputs.len());
    }
    out
}

/// An upper bound on the tokens of `text` for the planner.
///
/// Without `token-count`: the UTF-8 length. Byte-level BPE tokenizers
/// (OpenAI's) never produce more tokens than bytes, so the bound holds;
/// it is loose by the bytes-per-token ratio.
///
/// With `token-count`: OpenAI's `cl100k_base` count, exact for the
/// `text-embedding-3` models; for other providers, whose tokenizers are not
/// public, that count plus 20%.
pub(crate) fn tokens(wire: Wire, text: &str) -> u64 {
    #[cfg(feature = "token-count")]
    {
        let bpe = tiktoken::get_encoding("cl100k_base").expect("cl100k_base is compiled in");
        let n = bpe.count(text) as u64;
        if wire == Wire::OpenAi {
            n
        } else {
            n + n.div_ceil(5)
        }
    }
    #[cfg(not(feature = "token-count"))]
    {
        let _ = wire;
        text.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(texts: &[&str], max: usize, cap: Option<u64>) -> Vec<Range<usize>> {
        split(&Inputs::Strs(texts), max, cap, |t| t.len() as u64)
    }

    #[test]
    fn by_count() {
        let t = ["a"; 5];
        assert_eq!(plan(&t, 2, None), vec![0..2, 2..4, 4..5]);
        assert_eq!(plan(&t, 5, None), vec![0..5]);
        assert_eq!(plan(&t, 0, None), vec![0..1, 1..2, 2..3, 3..4, 4..5]);
        assert!(plan(&[], 3, None).is_empty());
    }

    #[test]
    fn by_tokens() {
        let t = ["aaaa", "bb", "cccc", "d", "eeeeeeeeee", "f"];
        // caps of 6: [aaaa bb] [cccc d] [eeeeeeeeee] [f]
        assert_eq!(plan(&t, 100, Some(6)), vec![0..2, 2..4, 4..5, 5..6]);
        // both caps at once
        assert_eq!(
            plan(&t, 1, Some(100)),
            (0..6).map(|i| i..i + 1).collect::<Vec<_>>()
        );
    }

    #[test]
    fn every_input_is_planned_once_and_in_order() {
        let texts: Vec<String> = (0..500).map(|i| "x".repeat(i * 7 % 23)).collect();
        let strs: Vec<&str> = texts.iter().map(String::as_str).collect();
        for (max, cap) in [
            (1, None),
            (7, Some(40)),
            (64, Some(1)),
            (1000, Some(10_000)),
        ] {
            let p = plan(&strs, max, cap);
            let mut next = 0;
            for r in &p {
                assert_eq!(r.start, next);
                assert!(r.end > r.start && r.len() <= max);
                let sum: u64 = strs[r.clone()].iter().map(|t| t.len() as u64).sum();
                if let Some(cap) = cap {
                    assert!(sum <= cap || r.len() == 1);
                }
                next = r.end;
            }
            assert_eq!(next, strs.len());
        }
    }

    #[test]
    fn byte_count_bounds_tokens() {
        #[cfg(not(feature = "token-count"))]
        assert_eq!(tokens(Wire::OpenAi, "héllo"), 6);
        #[cfg(feature = "token-count")]
        {
            assert_eq!(tokens(Wire::OpenAi, "hello world"), 2);
            assert_eq!(tokens(Wire::Voyage, "hello world"), 3);
        }
    }
}
