use serde::Serialize;

use super::RawEmbedResponse;
use super::http::post_json;
use super::openai_shape::Response;
use crate::error::Result;

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<u32>,
}

/// `Some(false)` for models documented with a fixed width, `None` when unknown.
pub(crate) fn accepts_dimensions(model: &str) -> Option<bool> {
    if model.starts_with("codestral-embed") {
        Some(true)
    } else if model.starts_with("mistral-embed") {
        Some(false)
    } else {
        None
    }
}

pub(crate) async fn send_mistral(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
    dimensions: Option<u32>,
) -> Result<RawEmbedResponse> {
    super::check_dimensions(model, dimensions, accepts_dimensions(model))?;
    let body = Request {
        model,
        input: texts,
        encoding_format: "float",
        output_dimension: dimensions,
    };
    let req = http
        .post(format!("{base_url}/embeddings"))
        .bearer_auth(api_key);
    post_json::<Response>(req, &body).await?.into_raw(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serialization() {
        let input = vec!["hello".to_string()];
        let req = Request {
            model: "codestral-embed",
            input: &input,
            encoding_format: "float",
            output_dimension: Some(256),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "codestral-embed");
        assert_eq!(json["input"][0], "hello");
        assert_eq!(json["output_dimension"], 256);
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
    fn dimension_support_by_model() {
        assert_eq!(accepts_dimensions("codestral-embed-2505"), Some(true));
        assert_eq!(accepts_dimensions("mistral-embed"), Some(false));
        assert_eq!(accepts_dimensions("some-new-embed"), None);
    }
}
