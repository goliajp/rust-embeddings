use serde::Serialize;

use super::http::post_json;
use super::openai_shape::Response;
use super::{InputType, RawEmbedResponse};
use crate::error::Result;

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    embedding_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<&'a str>,
}

fn map_task(model: &str, input_type: Option<InputType>) -> Option<&'static str> {
    input_type.map(|it| match it {
        InputType::SearchDocument => "retrieval.passage",
        InputType::SearchQuery => "retrieval.query",
        InputType::Classification => "classification",
        // v5 renamed v3's `separation` adapter to `clustering`
        InputType::Clustering if model.starts_with("jina-embeddings-v5") => "clustering",
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
        embedding_type: "float",
        dimensions,
        task: map_task(model, input_type),
    };
    let req = http
        .post(format!("{base_url}/embeddings"))
        .bearer_auth(api_key);
    post_json::<Response>(req, &body).await?.into_raw(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_serialization() {
        let input = vec!["hello".to_string()];
        let req = Request {
            model: "jina-embeddings-v3",
            input: &input,
            embedding_type: "float",
            dimensions: Some(512),
            task: Some("retrieval.query"),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["embedding_type"], "float");
        assert_eq!(json["dimensions"], 512);
        assert_eq!(json["task"], "retrieval.query");
        let req = Request {
            dimensions: None,
            task: None,
            ..req
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("dimensions").is_none());
        assert!(json.get("task").is_none());
    }

    #[test]
    fn task_mapping() {
        let v3 = "jina-embeddings-v3";
        assert_eq!(
            map_task(v3, Some(InputType::SearchDocument)),
            Some("retrieval.passage")
        );
        assert_eq!(
            map_task(v3, Some(InputType::SearchQuery)),
            Some("retrieval.query")
        );
        assert_eq!(
            map_task(v3, Some(InputType::Classification)),
            Some("classification")
        );
        assert_eq!(
            map_task(v3, Some(InputType::Clustering)),
            Some("separation")
        );
        assert_eq!(
            map_task("jina-embeddings-v5-text-small", Some(InputType::Clustering)),
            Some("clustering")
        );
        assert_eq!(map_task(v3, None), None);
    }
}
