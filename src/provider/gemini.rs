use serde::{Deserialize, Serialize};

use super::{InputType, RawEmbedResponse};
use crate::error::{Error, Result};

// single text request
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentRequest<'a> {
    model: String,
    content: GemContent<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
}

// batch request
#[derive(Serialize)]
struct BatchRequest<'a> {
    requests: Vec<EmbedContentRequest<'a>>,
}

#[derive(Serialize)]
struct GemContent<'a> {
    parts: Vec<GemPart<'a>>,
}

#[derive(Serialize)]
struct GemPart<'a> {
    text: &'a str,
}

// single response
#[derive(Deserialize)]
struct EmbedContentResponse {
    embedding: Option<EmbeddingValues>,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

// batch response
#[derive(Deserialize)]
struct BatchResponse {
    embeddings: Vec<EmbeddingValues>,
}

fn map_task_type(input_type: Option<InputType>) -> Option<&'static str> {
    input_type.map(|it| match it {
        InputType::SearchDocument => "RETRIEVAL_DOCUMENT",
        InputType::SearchQuery => "RETRIEVAL_QUERY",
        InputType::Classification => "CLASSIFICATION",
        InputType::Clustering => "CLUSTERING",
    })
}

pub(crate) async fn send_gemini(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
    dimensions: Option<u32>,
    input_type: Option<InputType>,
) -> Result<RawEmbedResponse> {
    let task_type = map_task_type(input_type);

    if texts.len() == 1 {
        send_single(
            http, base_url, api_key, model, &texts[0], dimensions, task_type,
        )
        .await
    } else {
        send_batch(http, base_url, api_key, model, texts, dimensions, task_type).await
    }
}

async fn send_single(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    text: &str,
    dimensions: Option<u32>,
    task_type: Option<&str>,
) -> Result<RawEmbedResponse> {
    let model_path = format!("models/{model}");
    let body = EmbedContentRequest {
        model: model_path,
        content: GemContent {
            parts: vec![GemPart { text }],
        },
        task_type,
        output_dimensionality: dimensions,
    };

    let url = format!("{base_url}/models/{model}:embedContent?key={api_key}");
    let resp = http
        .post(&url)
        .header("content-type", "application/json")
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

    let data: EmbedContentResponse = resp.json().await?;
    let embedding = data
        .embedding
        .ok_or_else(|| Error::Other("no embedding in gemini response".into()))?;

    Ok(RawEmbedResponse {
        embeddings: vec![embedding.values],
        total_tokens: 0,
        model: model.to_string(),
    })
}

async fn send_batch(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
    dimensions: Option<u32>,
    task_type: Option<&str>,
) -> Result<RawEmbedResponse> {
    let model_path = format!("models/{model}");
    let requests: Vec<EmbedContentRequest> = texts
        .iter()
        .map(|t| EmbedContentRequest {
            model: model_path.clone(),
            content: GemContent {
                parts: vec![GemPart { text: t }],
            },
            task_type,
            output_dimensionality: dimensions,
        })
        .collect();

    let body = BatchRequest { requests };
    let url = format!("{base_url}/models/{model}:batchEmbedContents?key={api_key}");

    let resp = http
        .post(&url)
        .header("content-type", "application/json")
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

    let data: BatchResponse = resp.json().await?;
    let embeddings = data.embeddings.into_iter().map(|e| e.values).collect();

    Ok(RawEmbedResponse {
        embeddings,
        total_tokens: 0,
        model: model.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_request_serialization() {
        let req = EmbedContentRequest {
            model: "models/gemini-embedding-001".to_string(),
            content: GemContent {
                parts: vec![GemPart { text: "hello" }],
            },
            task_type: Some("RETRIEVAL_DOCUMENT"),
            output_dimensionality: Some(256),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "models/gemini-embedding-001");
        assert_eq!(json["content"]["parts"][0]["text"], "hello");
        assert_eq!(json["taskType"], "RETRIEVAL_DOCUMENT");
        assert_eq!(json["outputDimensionality"], 256);
    }

    #[test]
    fn single_request_serialization_no_optional_fields() {
        let req = EmbedContentRequest {
            model: "models/gemini-embedding-001".to_string(),
            content: GemContent {
                parts: vec![GemPart { text: "test" }],
            },
            task_type: None,
            output_dimensionality: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("taskType").is_none());
        assert!(json.get("outputDimensionality").is_none());
    }

    #[test]
    fn batch_request_serialization() {
        let req = BatchRequest {
            requests: vec![
                EmbedContentRequest {
                    model: "models/gemini-embedding-001".to_string(),
                    content: GemContent {
                        parts: vec![GemPart { text: "a" }],
                    },
                    task_type: None,
                    output_dimensionality: None,
                },
                EmbedContentRequest {
                    model: "models/gemini-embedding-001".to_string(),
                    content: GemContent {
                        parts: vec![GemPart { text: "b" }],
                    },
                    task_type: None,
                    output_dimensionality: None,
                },
            ],
        };
        let json = serde_json::to_value(&req).unwrap();
        let requests = json["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["content"]["parts"][0]["text"], "a");
        assert_eq!(requests[1]["content"]["parts"][0]["text"], "b");
    }

    #[test]
    fn task_type_mapping_search_document() {
        assert_eq!(
            map_task_type(Some(InputType::SearchDocument)),
            Some("RETRIEVAL_DOCUMENT")
        );
    }

    #[test]
    fn task_type_mapping_search_query() {
        assert_eq!(
            map_task_type(Some(InputType::SearchQuery)),
            Some("RETRIEVAL_QUERY")
        );
    }

    #[test]
    fn task_type_mapping_classification() {
        assert_eq!(
            map_task_type(Some(InputType::Classification)),
            Some("CLASSIFICATION")
        );
    }

    #[test]
    fn task_type_mapping_clustering() {
        assert_eq!(
            map_task_type(Some(InputType::Clustering)),
            Some("CLUSTERING")
        );
    }

    #[test]
    fn task_type_mapping_none() {
        assert_eq!(map_task_type(None), None);
    }

    #[test]
    fn single_response_deserialization() {
        let json = r#"{
            "embedding": {
                "values": [0.1, 0.2, 0.3]
            }
        }"#;
        let resp: EmbedContentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.embedding.unwrap().values, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn single_response_deserialization_no_embedding() {
        let json = r#"{"embedding": null}"#;
        let resp: EmbedContentResponse = serde_json::from_str(json).unwrap();
        assert!(resp.embedding.is_none());
    }

    #[test]
    fn batch_response_deserialization() {
        let json = r#"{
            "embeddings": [
                {"values": [0.1, 0.2]},
                {"values": [0.3, 0.4]}
            ]
        }"#;
        let resp: BatchResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.embeddings.len(), 2);
        assert_eq!(resp.embeddings[0].values, vec![0.1, 0.2]);
        assert_eq!(resp.embeddings[1].values, vec![0.3, 0.4]);
    }

    #[test]
    fn camel_case_serialization() {
        // verify serde rename_all = "camelCase" works
        let req = EmbedContentRequest {
            model: "models/test".to_string(),
            content: GemContent {
                parts: vec![GemPart { text: "x" }],
            },
            task_type: Some("RETRIEVAL_QUERY"),
            output_dimensionality: Some(128),
        };
        let json_str = serde_json::to_string(&req).unwrap();
        assert!(json_str.contains("taskType"));
        assert!(json_str.contains("outputDimensionality"));
        assert!(!json_str.contains("task_type"));
        assert!(!json_str.contains("output_dimensionality"));
    }
}
