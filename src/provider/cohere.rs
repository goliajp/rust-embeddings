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
