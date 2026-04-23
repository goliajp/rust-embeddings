use serde::{Deserialize, Serialize};

use super::{InputType, RawEmbedResponse};
use crate::error::{Error, Result};

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
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

fn map_input_type(input_type: Option<InputType>) -> Option<&'static str> {
    input_type.map(|it| match it {
        InputType::SearchDocument => "document",
        InputType::SearchQuery => "query",
        InputType::Classification => "document",
        InputType::Clustering => "document",
    })
}

pub(crate) async fn send_voyage(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
    input_type: Option<InputType>,
) -> Result<RawEmbedResponse> {
    let body = Request {
        model,
        input: texts,
        input_type: map_input_type(input_type),
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
    fn request_serialization_with_input_type() {
        let input = vec!["hello".to_string()];
        let req = Request {
            model: "voyage-3-large",
            input: &input,
            input_type: Some("query"),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "voyage-3-large");
        assert_eq!(json["input"][0], "hello");
        assert_eq!(json["input_type"], "query");
    }

    #[test]
    fn request_serialization_no_input_type() {
        let input = vec!["test".to_string()];
        let req = Request {
            model: "voyage-3-large",
            input: &input,
            input_type: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("input_type").is_none());
    }

    #[test]
    fn input_type_mapping_search_document() {
        assert_eq!(
            map_input_type(Some(InputType::SearchDocument)),
            Some("document")
        );
    }

    #[test]
    fn input_type_mapping_search_query() {
        assert_eq!(map_input_type(Some(InputType::SearchQuery)), Some("query"));
    }

    #[test]
    fn input_type_mapping_classification_falls_back_to_document() {
        assert_eq!(
            map_input_type(Some(InputType::Classification)),
            Some("document")
        );
    }

    #[test]
    fn input_type_mapping_clustering_falls_back_to_document() {
        assert_eq!(
            map_input_type(Some(InputType::Clustering)),
            Some("document")
        );
    }

    #[test]
    fn input_type_mapping_none() {
        assert_eq!(map_input_type(None), None);
    }

    #[test]
    fn response_deserialization() {
        let json = r#"{
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ],
            "model": "voyage-3-large",
            "usage": {"total_tokens": 12}
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(resp.data[1].embedding, vec![0.4, 0.5, 0.6]);
        assert_eq!(resp.model, "voyage-3-large");
        assert_eq!(resp.usage.total_tokens, 12);
    }

    #[test]
    fn response_deserialization_single_embedding() {
        let json = r#"{
            "data": [{"embedding": [1.0]}],
            "model": "voyage-3-large",
            "usage": {"total_tokens": 1}
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].embedding, vec![1.0]);
    }
}
