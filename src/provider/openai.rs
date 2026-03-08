use serde::{Deserialize, Serialize};

use super::RawEmbedResponse;
use crate::error::{Error, Result};

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
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
        encoding_format: "float",
        dimensions,
    };

    let resp = http
        .post(format!("{base_url}/embeddings"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(Error::Api {
            status: status.as_u16(),
            message: text,
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
        let input = vec!["hello".to_string(), "world".to_string()];
        let req = Request {
            model: "text-embedding-3-small",
            input: &input,
            encoding_format: "float",
            dimensions: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "text-embedding-3-small");
        assert_eq!(json["input"][0], "hello");
        assert_eq!(json["input"][1], "world");
        assert_eq!(json["encoding_format"], "float");
        assert!(json.get("dimensions").is_none());
    }

    #[test]
    fn request_serialization_with_dimensions() {
        let input = vec!["test".to_string()];
        let req = Request {
            model: "text-embedding-3-large",
            input: &input,
            encoding_format: "float",
            dimensions: Some(256),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["dimensions"], 256);
    }

    #[test]
    fn response_deserialization() {
        let json = r#"{
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ],
            "model": "text-embedding-3-small",
            "usage": {"total_tokens": 10}
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(resp.data[1].embedding, vec![0.4, 0.5, 0.6]);
        assert_eq!(resp.model, "text-embedding-3-small");
        assert_eq!(resp.usage.total_tokens, 10);
    }

    #[test]
    fn response_deserialization_empty_data() {
        let json = r#"{
            "data": [],
            "model": "text-embedding-3-small",
            "usage": {"total_tokens": 0}
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert!(resp.data.is_empty());
        assert_eq!(resp.usage.total_tokens, 0);
    }
}
