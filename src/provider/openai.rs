use serde::Serialize;

use super::RawEmbedResponse;
use super::http::post_json;
use super::openai_shape::Response;
use crate::error::Result;

pub(crate) const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

// openai's own endpoint sends base64 vectors (2.3x smaller than decimal text
// and much cheaper to parse); compatible servers get "float", which every one
// of them understands
fn encoding_format(base_url: &str) -> &'static str {
    if base_url == OPENAI_BASE_URL {
        "base64"
    } else {
        "float"
    }
}

pub(crate) async fn send_openai(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
    dimensions: Option<u32>,
) -> Result<RawEmbedResponse> {
    let body = Request {
        model,
        input: texts,
        encoding_format: encoding_format(base_url),
        dimensions,
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
        let input = vec!["hello".to_string(), "world".to_string()];
        let req = Request {
            model: "text-embedding-3-small",
            input: &input,
            encoding_format: "float",
            dimensions: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "text-embedding-3-small");
        assert_eq!(json["input"][1], "world");
        assert_eq!(json["encoding_format"], "float");
        assert!(json.get("dimensions").is_none());

        let req = Request {
            dimensions: Some(256),
            ..req
        };
        assert_eq!(serde_json::to_value(&req).unwrap()["dimensions"], 256);
    }

    #[test]
    fn base64_only_for_the_official_endpoint() {
        assert_eq!(encoding_format(OPENAI_BASE_URL), "base64");
        assert_eq!(encoding_format("https://api.deepseek.com/v1"), "float");
    }
}
