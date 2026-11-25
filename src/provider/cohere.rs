use serde::{Deserialize, Serialize};

use super::http::post_json;
use super::vector::Vector;
use super::{InputType, RawEmbedResponse};
use crate::error::{Error, Result};

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    texts: &'a [String],
    input_type: &'a str,
    embedding_types: [&'a str; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<u32>,
}

#[derive(Deserialize)]
struct Response {
    embeddings: Embeddings,
    meta: Option<Meta>,
}

#[derive(Deserialize)]
struct Embeddings {
    float: Option<Vec<Vector>>,
    base64: Option<Vec<Vector>>,
}

#[derive(Deserialize)]
struct Meta {
    billed_units: Option<BilledUnits>,
}

#[derive(Deserialize)]
struct BilledUnits {
    input_tokens: Option<u32>,
}

fn map_input_type(input_type: Option<InputType>) -> &'static str {
    match input_type {
        Some(InputType::SearchDocument) | None => "search_document",
        Some(InputType::SearchQuery) => "search_query",
        Some(InputType::Classification) => "classification",
        Some(InputType::Clustering) => "clustering",
    }
}

/// `output_dimension` exists from embed-v4 on; the v3 models have one width.
pub(crate) fn accepts_dimensions(model: &str) -> Option<bool> {
    if model.ends_with("-v3.0") {
        Some(false)
    } else if model.starts_with("embed-v") {
        Some(true)
    } else {
        None
    }
}

impl Response {
    fn into_raw(self, model: &str) -> Result<RawEmbedResponse> {
        let vectors = self
            .embeddings
            .base64
            .or(self.embeddings.float)
            .ok_or_else(|| Error::Other("cohere response has no embeddings".into()))?;
        Ok(RawEmbedResponse {
            embeddings: vectors.into_iter().map(|v| v.0).collect(),
            total_tokens: self
                .meta
                .and_then(|m| m.billed_units)
                .and_then(|b| b.input_tokens),
            model: model.to_string(),
        })
    }
}

pub(crate) async fn send_cohere(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
    dimensions: Option<u32>,
    input_type: Option<InputType>,
) -> Result<RawEmbedResponse> {
    super::check_dimensions(model, dimensions, accepts_dimensions(model))?;
    let body = Request {
        model,
        texts,
        input_type: map_input_type(input_type),
        // the float vectors' bytes, base64-encoded
        embedding_types: ["base64"],
        output_dimension: dimensions,
    };
    let req = http.post(format!("{base_url}/embed")).bearer_auth(api_key);
    post_json::<Response>(req, &body).await?.into_raw(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serialization() {
        let texts = vec!["hello".to_string(), "world".to_string()];
        let req = Request {
            model: "embed-v4.0",
            texts: &texts,
            input_type: "search_document",
            embedding_types: ["base64"],
            output_dimension: Some(512),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "embed-v4.0");
        assert_eq!(json["texts"][1], "world");
        assert_eq!(json["input_type"], "search_document");
        assert_eq!(json["embedding_types"], serde_json::json!(["base64"]));
        assert_eq!(json["output_dimension"], 512);
        let req = Request {
            output_dimension: None,
            ..req
        };
        assert!(
            serde_json::to_value(&req)
                .unwrap()
                .get("output_dimension")
                .is_none()
        );
    }

    #[test]
    fn input_type_mapping() {
        assert_eq!(
            map_input_type(Some(InputType::SearchDocument)),
            "search_document"
        );
        assert_eq!(map_input_type(Some(InputType::SearchQuery)), "search_query");
        assert_eq!(
            map_input_type(Some(InputType::Classification)),
            "classification"
        );
        assert_eq!(map_input_type(Some(InputType::Clustering)), "clustering");
        assert_eq!(map_input_type(None), "search_document");
    }

    #[test]
    fn dimension_support_by_model() {
        assert_eq!(accepts_dimensions("embed-v4.0"), Some(true));
        assert_eq!(accepts_dimensions("embed-v5.0-pro"), Some(true));
        assert_eq!(accepts_dimensions("embed-english-v3.0"), Some(false));
        assert_eq!(accepts_dimensions("my-proxy-model"), None);
    }

    fn parse(json: &str) -> Result<RawEmbedResponse> {
        serde_json::from_str::<Response>(json)
            .unwrap()
            .into_raw("embed-v4.0")
    }

    #[test]
    fn response_with_base64_or_float() {
        let raw = parse(
            r#"{"embeddings":{"base64":["AACAPw=="]},"meta":{"billed_units":{"input_tokens":15}}}"#,
        )
        .unwrap();
        assert_eq!(raw.embeddings, vec![vec![1.0]]);
        assert_eq!(raw.total_tokens, Some(15));
        let raw = parse(r#"{"embeddings":{"float":[[0.5,0.6]]},"meta":{}}"#).unwrap();
        assert_eq!(raw.embeddings, vec![vec![0.5, 0.6]]);
    }

    #[test]
    fn missing_token_count_is_unknown_not_zero() {
        assert_eq!(
            parse(r#"{"embeddings":{"float":[[1]]},"meta":{"billed_units":null}}"#)
                .unwrap()
                .total_tokens,
            None
        );
        assert_eq!(
            parse(r#"{"embeddings":{"float":[[1]]}}"#)
                .unwrap()
                .total_tokens,
            None
        );
    }

    #[test]
    fn response_without_vectors_is_an_error() {
        assert!(parse(r#"{"embeddings":{}}"#).is_err());
    }
}
