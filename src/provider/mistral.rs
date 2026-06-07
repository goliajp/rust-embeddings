use serde::{Deserialize, Serialize};

use super::RawEmbedResponse;
use crate::error::{Error, Result};

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'a str,
}

#[derive(Deserialize)]
struct Response {
    data: Vec<EmbeddingData>,
    model: String,
    usage: UsageInfo,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct UsageInfo {
    total_tokens: u32,
}

pub(crate) async fn send_mistral(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
) -> Result<RawEmbedResponse> {
    let body = Request {
        model,
        input: texts,
        encoding_format: "float",
    };

    let resp = http
        .post(format!("{base_url}/embeddings"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let retry_after = super::parse_retry_after(resp.headers());
        let text = resp.text().await.unwrap_or_default();
        return Err(Error::Api {
            status: status.as_u16(),
            message: text,
            retry_after,
        });
    }

    let data: Response = resp.json().await?;
    let embeddings = data.data.into_iter().map(|d| d.embedding).collect();

    Ok(RawEmbedResponse {
        embeddings,
        total_tokens: data.usage.total_tokens,
        model: data.model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serialization_basic() {
        let input = vec!["hello".to_string()];
        let req = Request {
            model: "mistral-embed",
            input: &input,
            encoding_format: "float",
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "mistral-embed");
        assert_eq!(json["input"][0], "hello");
        assert_eq!(json["encoding_format"], "float");
    }

    #[test]
    fn response_deserialization() {
        let json = r#"{
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "mistral-embed",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.model, "mistral-embed");
        assert_eq!(resp.usage.total_tokens, 5);
    }
}
