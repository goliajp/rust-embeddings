use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use super::http::post_json;
use super::vector::Vector;
use super::{InputType, RawEmbedResponse};
use crate::error::{Error, Result};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentRequest<'a> {
    model: &'a str,
    content: GemContent<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
}

#[derive(Serialize)]
struct BatchRequest<'a> {
    requests: Vec<EmbedContentRequest<'a>>,
}

#[derive(Serialize)]
struct GemContent<'a> {
    parts: [GemPart<'a>; 1],
}

#[derive(Serialize)]
struct GemPart<'a> {
    text: Cow<'a, str>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentResponse {
    embedding: Option<EmbeddingValues>,
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchResponse {
    embeddings: Vec<EmbeddingValues>,
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vector,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    prompt_token_count: Option<u32>,
}

fn map_task_type(input_type: Option<InputType>) -> Option<&'static str> {
    input_type.map(|it| match it {
        InputType::SearchDocument => "RETRIEVAL_DOCUMENT",
        InputType::SearchQuery => "RETRIEVAL_QUERY",
        InputType::Classification => "CLASSIFICATION",
        InputType::Clustering => "CLUSTERING",
    })
}

// gemini-embedding-2 takes no task type; the task is written in front of the text
fn takes_task_prefix(model: &str) -> bool {
    model.starts_with("gemini-embedding-2")
}

fn with_task_prefix(input_type: Option<InputType>, text: &str) -> Cow<'_, str> {
    match input_type {
        None => Cow::Borrowed(text),
        Some(InputType::SearchQuery) => Cow::Owned(format!("task: search result | query: {text}")),
        Some(InputType::SearchDocument) => Cow::Owned(format!("title: none | text: {text}")),
        Some(InputType::Classification) => {
            Cow::Owned(format!("task: classification | query: {text}"))
        }
        Some(InputType::Clustering) => Cow::Owned(format!("task: clustering | query: {text}")),
    }
}

fn build_request<'a>(
    model_path: &'a str,
    model: &str,
    text: &'a str,
    dimensions: Option<u32>,
    input_type: Option<InputType>,
) -> EmbedContentRequest<'a> {
    let (text, task_type) = if takes_task_prefix(model) {
        (with_task_prefix(input_type, text), None)
    } else {
        (Cow::Borrowed(text), map_task_type(input_type))
    };
    EmbedContentRequest {
        model: model_path,
        content: GemContent {
            parts: [GemPart { text }],
        },
        task_type,
        output_dimensionality: dimensions,
    }
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
    let model_path = format!("models/{model}");
    // the key goes in a header: a url with it would end up in error messages and logs
    if let [text] = texts {
        let body = build_request(&model_path, model, text, dimensions, input_type);
        let req = http
            .post(format!("{base_url}/models/{model}:embedContent"))
            .header("x-goog-api-key", api_key);
        let data: EmbedContentResponse = post_json(req, &body).await?;
        let embedding = data
            .embedding
            .ok_or_else(|| Error::Other("no embedding in gemini response".into()))?;
        Ok(RawEmbedResponse {
            embeddings: vec![embedding.values.0],
            total_tokens: data.usage_metadata.and_then(|u| u.prompt_token_count),
            model: model.to_string(),
        })
    } else {
        let requests = texts
            .iter()
            .map(|t| build_request(&model_path, model, t, dimensions, input_type))
            .collect();
        let req = http
            .post(format!("{base_url}/models/{model}:batchEmbedContents"))
            .header("x-goog-api-key", api_key);
        let data: BatchResponse = post_json(req, &BatchRequest { requests }).await?;
        Ok(RawEmbedResponse {
            embeddings: data.embeddings.into_iter().map(|e| e.values.0).collect(),
            total_tokens: data.usage_metadata.and_then(|u| u.prompt_token_count),
            model: model.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_for_gemini_embedding_001() {
        let req = build_request(
            "models/gemini-embedding-001",
            "gemini-embedding-001",
            "hello",
            Some(256),
            Some(InputType::SearchDocument),
        );
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "models/gemini-embedding-001");
        assert_eq!(json["content"]["parts"][0]["text"], "hello");
        assert_eq!(json["taskType"], "RETRIEVAL_DOCUMENT");
        assert_eq!(json["outputDimensionality"], 256);

        let req = build_request("models/x", "gemini-embedding-001", "t", None, None);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("taskType").is_none());
        assert!(json.get("outputDimensionality").is_none());
    }

    #[test]
    fn gemini_embedding_2_gets_a_prefix_not_a_task_type() {
        let req = build_request(
            "models/gemini-embedding-2",
            "gemini-embedding-2",
            "how do rivers form",
            None,
            Some(InputType::SearchQuery),
        );
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("taskType").is_none());
        assert_eq!(
            json["content"]["parts"][0]["text"],
            "task: search result | query: how do rivers form"
        );
        let req = build_request("m", "gemini-embedding-2", "plain", None, None);
        assert_eq!(
            serde_json::to_value(&req).unwrap()["content"]["parts"][0]["text"],
            "plain"
        );
    }

    #[test]
    fn task_prefixes() {
        assert_eq!(
            with_task_prefix(Some(InputType::SearchDocument), "x"),
            "title: none | text: x"
        );
        assert_eq!(
            with_task_prefix(Some(InputType::Classification), "x"),
            "task: classification | query: x"
        );
        assert_eq!(
            with_task_prefix(Some(InputType::Clustering), "x"),
            "task: clustering | query: x"
        );
    }

    #[test]
    fn task_type_mapping() {
        assert_eq!(
            map_task_type(Some(InputType::SearchDocument)),
            Some("RETRIEVAL_DOCUMENT")
        );
        assert_eq!(
            map_task_type(Some(InputType::SearchQuery)),
            Some("RETRIEVAL_QUERY")
        );
        assert_eq!(
            map_task_type(Some(InputType::Classification)),
            Some("CLASSIFICATION")
        );
        assert_eq!(
            map_task_type(Some(InputType::Clustering)),
            Some("CLUSTERING")
        );
        assert_eq!(map_task_type(None), None);
    }

    #[test]
    fn responses_carry_token_counts() {
        let one: EmbedContentResponse = serde_json::from_str(
            r#"{"embedding":{"values":[0.1,0.2]},"usageMetadata":{"promptTokenCount":3}}"#,
        )
        .unwrap();
        assert_eq!(one.embedding.unwrap().values.0, vec![0.1, 0.2]);
        assert_eq!(one.usage_metadata.unwrap().prompt_token_count, Some(3));

        let batch: BatchResponse = serde_json::from_str(
            r#"{"embeddings":[{"values":[0.1]},{"values":[0.3]}],"usageMetadata":{"promptTokenCount":7}}"#,
        )
        .unwrap();
        assert_eq!(batch.embeddings.len(), 2);
        assert_eq!(batch.usage_metadata.unwrap().prompt_token_count, Some(7));

        let bare: BatchResponse = serde_json::from_str(r#"{"embeddings":[]}"#).unwrap();
        assert!(bare.usage_metadata.is_none());
    }
}
