//! The response body OpenAI, Voyage, Jina and Mistral share:
//! `{"data": [{"index": 0, "embedding": ...}], "model": "...", "usage": {"total_tokens": n}}`.

use serde::Deserialize;

use super::RawEmbedResponse;
use super::vector::Vector;
use crate::error::{Error, Result};

#[derive(Deserialize)]
pub(crate) struct Response {
    data: Vec<Item>,
    model: Option<String>,
    usage: Option<UsageInfo>,
}

#[derive(Deserialize)]
struct Item {
    index: Option<usize>,
    embedding: Vector,
}

#[derive(Deserialize)]
struct UsageInfo {
    total_tokens: Option<u32>,
}

impl Response {
    /// Vectors in input order; `requested_model` is used when the body names none.
    pub(crate) fn into_raw(self, requested_model: &str) -> Result<RawEmbedResponse> {
        let n = self.data.len();
        let embeddings = if self.data.iter().all(|d| d.index.is_some()) {
            let mut slots: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
            for item in self.data {
                let i = item.index.unwrap_or_default();
                match slots.get_mut(i) {
                    Some(slot @ None) => *slot = Some(item.embedding.0),
                    _ => {
                        return Err(Error::Other(format!(
                            "response index {i} is out of range or repeated ({n} vectors)"
                        )));
                    }
                }
            }
            slots.into_iter().flatten().collect()
        } else {
            self.data.into_iter().map(|d| d.embedding.0).collect()
        };
        Ok(RawEmbedResponse {
            embeddings,
            total_tokens: self.usage.and_then(|u| u.total_tokens),
            model: self.model.unwrap_or_else(|| requested_model.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Result<RawEmbedResponse> {
        serde_json::from_str::<Response>(json)
            .unwrap()
            .into_raw("m")
    }

    #[test]
    fn reordered_items_come_back_in_input_order() {
        let raw = parse(
            r#"{"data":[{"index":1,"embedding":[2.0]},{"index":0,"embedding":[1.0]}],
                "model":"x","usage":{"total_tokens":4}}"#,
        )
        .unwrap();
        assert_eq!(raw.embeddings, vec![vec![1.0], vec![2.0]]);
        assert_eq!(raw.total_tokens, Some(4));
        assert_eq!(raw.model, "x");
    }

    #[test]
    fn repeated_or_out_of_range_index_is_an_error() {
        assert!(
            parse(r#"{"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]}"#).is_err()
        );
        assert!(parse(r#"{"data":[{"index":2,"embedding":[1]}]}"#).is_err());
    }

    #[test]
    fn missing_index_model_and_usage_are_tolerated() {
        let raw = parse(r#"{"data":[{"embedding":[1]},{"embedding":[2]}]}"#).unwrap();
        assert_eq!(raw.embeddings, vec![vec![1.0], vec![2.0]]);
        assert_eq!(raw.total_tokens, None);
        assert_eq!(raw.model, "m");
    }

    #[test]
    fn base64_items_are_decoded() {
        // 1.0f32 little-endian is 00 00 80 3f
        let raw =
            parse(r#"{"data":[{"index":0,"embedding":"AACAPw=="}],"usage":{"total_tokens":1}}"#)
                .unwrap();
        assert_eq!(raw.embeddings, vec![vec![1.0]]);
    }
}
