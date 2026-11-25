use serde::Serialize;

use super::http::post_json;
use super::openai_shape::Response;
use super::{InputType, RawEmbedResponse};
use crate::error::Result;

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<u32>,
    encoding_format: &'a str,
}

fn map_input_type(input_type: Option<InputType>) -> Option<&'static str> {
    input_type.map(|it| match it {
        InputType::SearchDocument => "document",
        InputType::SearchQuery => "query",
        InputType::Classification => "document",
        InputType::Clustering => "document",
    })
}

// models with a single output width; newer models take `output_dimension`
const FIXED_WIDTH: &[&str] = &[
    "voyage-3",
    "voyage-3-lite",
    "voyage-2",
    "voyage-large-2",
    "voyage-large-2-instruct",
    "voyage-code-2",
    "voyage-law-2",
    "voyage-finance-2",
    "voyage-multilingual-2",
];

/// `Some(false)` for models documented with a fixed width, `None` when unknown.
pub(crate) fn accepts_dimensions(model: &str) -> Option<bool> {
    if FIXED_WIDTH.contains(&model) {
        Some(false)
    } else if model.starts_with("voyage-4")
        || model.starts_with("voyage-3.5")
        || model == "voyage-3-large"
        || model == "voyage-code-3"
        || model == "voyage-code-4"
    {
        Some(true)
    } else {
        None
    }
}

pub(crate) async fn send_voyage(
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
        input: texts,
        input_type: map_input_type(input_type),
        output_dimension: dimensions,
        // voyage documents base64 as the raw float32 array
        encoding_format: "base64",
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
            model: "voyage-4-large",
            input: &input,
            input_type: Some("query"),
            output_dimension: Some(512),
            encoding_format: "base64",
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "voyage-4-large");
        assert_eq!(json["input"][0], "hello");
        assert_eq!(json["input_type"], "query");
        assert_eq!(json["output_dimension"], 512);
        assert_eq!(json["encoding_format"], "base64");

        let req = Request {
            input_type: None,
            output_dimension: None,
            ..req
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("input_type").is_none());
        assert!(json.get("output_dimension").is_none());
    }

    #[test]
    fn input_type_mapping() {
        assert_eq!(
            map_input_type(Some(InputType::SearchDocument)),
            Some("document")
        );
        assert_eq!(map_input_type(Some(InputType::SearchQuery)), Some("query"));
        assert_eq!(
            map_input_type(Some(InputType::Classification)),
            Some("document")
        );
        assert_eq!(
            map_input_type(Some(InputType::Clustering)),
            Some("document")
        );
        assert_eq!(map_input_type(None), None);
    }

    #[test]
    fn dimension_support_by_model() {
        for m in [
            "voyage-4-large",
            "voyage-4-lite",
            "voyage-3.5",
            "voyage-3-large",
            "voyage-code-3",
        ] {
            assert_eq!(accepts_dimensions(m), Some(true), "{m}");
        }
        for m in ["voyage-3", "voyage-law-2", "voyage-code-2"] {
            assert_eq!(accepts_dimensions(m), Some(false), "{m}");
        }
        assert_eq!(accepts_dimensions("voyage-9"), None);
    }
}
