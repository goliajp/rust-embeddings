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
