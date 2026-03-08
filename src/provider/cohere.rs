use serde::{Deserialize, Serialize};

use super::{InputType, RawEmbedResponse};
use crate::error::{Error, Result};

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    texts: &'a [String],
    input_type: &'a str,
    embedding_types: Vec<&'a str>,
}

#[derive(Deserialize)]
struct Response {
    embeddings: Embeddings,
    meta: Meta,
}

#[derive(Deserialize)]
struct Embeddings {
    float: Vec<Vec<f32>>,
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

pub(crate) async fn send_cohere(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    texts: &[String],
    input_type: Option<InputType>,
) -> Result<RawEmbedResponse> {
    let body = Request {
        model,
        texts,
        input_type: map_input_type(input_type),
        embedding_types: vec!["float"],
    };

    let resp = http
        .post(format!("{base_url}/embed"))
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
    let total_tokens = data
        .meta
        .billed_units
        .and_then(|b| b.input_tokens)
        .unwrap_or(0);

    Ok(RawEmbedResponse {
        embeddings: data.embeddings.float,
        total_tokens,
        model: model.to_string(),
    })
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
            embedding_types: vec!["float"],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "embed-v4.0");
        assert_eq!(json["texts"][0], "hello");
        assert_eq!(json["texts"][1], "world");
        assert_eq!(json["input_type"], "search_document");
        assert_eq!(json["embedding_types"][0], "float");
    }

    #[test]
    fn embedding_types_field() {
        let texts = vec!["test".to_string()];
        let req = Request {
            model: "embed-v4.0",
            texts: &texts,
            input_type: "search_query",
            embedding_types: vec!["float", "int8"],
        };
        let json = serde_json::to_value(&req).unwrap();
        let types = json["embedding_types"].as_array().unwrap();
        assert_eq!(types.len(), 2);
        assert_eq!(types[0], "float");
        assert_eq!(types[1], "int8");
    }

    #[test]
    fn input_type_mapping_search_document() {
        assert_eq!(
            map_input_type(Some(InputType::SearchDocument)),
            "search_document"
        );
    }

    #[test]
    fn input_type_mapping_search_query() {
        assert_eq!(map_input_type(Some(InputType::SearchQuery)), "search_query");
    }

    #[test]
    fn input_type_mapping_classification() {
        assert_eq!(
            map_input_type(Some(InputType::Classification)),
            "classification"
        );
    }

    #[test]
    fn input_type_mapping_clustering() {
        assert_eq!(map_input_type(Some(InputType::Clustering)), "clustering");
    }

    #[test]
    fn input_type_mapping_none_defaults_to_search_document() {
        assert_eq!(map_input_type(None), "search_document");
    }

    #[test]
    fn response_deserialization() {
        let json = r#"{
            "embeddings": {
                "float": [[0.1, 0.2], [0.3, 0.4]]
            },
            "meta": {
                "billed_units": {
                    "input_tokens": 15
                }
            }
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert_eq!(resp.embeddings.float.len(), 2);
        assert_eq!(resp.embeddings.float[0], vec![0.1, 0.2]);
        assert_eq!(resp.meta.billed_units.unwrap().input_tokens, Some(15));
    }

    #[test]
    fn response_deserialization_no_billed_units() {
        let json = r#"{
            "embeddings": {
                "float": [[0.1]]
            },
            "meta": {
                "billed_units": null
            }
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert!(resp.meta.billed_units.is_none());
    }

    #[test]
    fn response_deserialization_missing_billed_units() {
        let json = r#"{
            "embeddings": {
                "float": [[0.5, 0.6]]
            },
            "meta": {}
        }"#;
        let resp: Response = serde_json::from_str(json).unwrap();
        assert!(resp.meta.billed_units.is_none());
    }
}
