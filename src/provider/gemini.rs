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
    let body = EmbedContentRequest {
        model: format!("models/{model}"),
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
    let requests: Vec<EmbedContentRequest> = texts
        .iter()
        .map(|t| EmbedContentRequest {
            model: format!("models/{model}"),
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
