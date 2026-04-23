use serde::{Deserialize, Serialize};

use super::{InputType, RawEmbedResponse};
use crate::error::{Error, Result};

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<&'a str>,
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

fn map_task(input_type: Option<InputType>) -> Option<&'static str> {
    input_type.map(|it| match it {
        InputType::SearchDocument => "retrieval.passage",
        InputType::SearchQuery => "retrieval.query",
        InputType::Classification => "classification",
        InputType::Clustering => "separation",
    })
}

pub(crate) async fn send_jina(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
    dimensions: Option<u32>,
    input_type: Option<InputType>,
) -> Result<RawEmbedResponse> {
    let body = Request {
        model,
        input: texts,
        encoding_type: "float",
        dimensions,
        task: map_task(input_type),
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
            model: "jina-embeddings-v3",
            input: &input,
            encoding_type: "float",
            dimensions: None,
            task: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "jina-embeddings-v3");
        assert_eq!(json["input"][0], "hello");
        assert_eq!(json["input"][1], "world");
        assert_eq!(json["encoding_type"], "float");
        assert!(json.get("dimensions").is_none());
        assert!(json.get("task").is_none());
    }

    #[test]
    fn request_serialization_with_all_fields() {
        let input = vec!["test".to_string()];
        let req = Request {
            model: "jina-embeddings-v3",
            input: &input,
            encoding_type: "float",
            dimensions: Some(512),
            task: Some("retrieval.query"),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["dimensions"], 512);
        assert_eq!(json["task"], "retrieval.query");
    }

    #[test]
    fn task_mapping_search_document() {
        assert_eq!(
            map_task(Some(InputType::SearchDocument)),
            Some("retrieval.passage")
        );
    }

    #[test]
    fn task_mapping_search_query() {
        assert_eq!(
            map_task(Some(InputType::SearchQuery)),
            Some("retrieval.query")
        );
    }

    #[test]
    fn task_mapping_classification() {
        assert_eq!(
            map_task(Some(InputType::Classification)),
            Some("classification")
        );
    }

    #[test]
    fn task_mapping_clustering() {
        assert_eq!(map_task(Some(InputType::Clustering)), Some("separation"));
    }

    #[test]
    fn task_mapping_none() {
        assert_eq!(map_task(None), None);
    }

    #[test]
    fn response_deserialization() {
        let json = r#"{
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]}
            ],
            "model": "jina-embeddings-v3",
            "usage": {"total_tokens": 8}
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].embedding, vec![0.1, 0.2]);
        assert_eq!(resp.data[1].embedding, vec![0.3, 0.4]);
        assert_eq!(resp.model, "jina-embeddings-v3");
        assert_eq!(resp.usage.total_tokens, 8);
    }

    #[test]
    fn response_deserialization_empty() {
        let json = r#"{
            "data": [],
            "model": "jina-embeddings-v3",
            "usage": {"total_tokens": 0}
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert!(resp.data.is_empty());
        assert_eq!(resp.usage.total_tokens, 0);
    }
}
