//! Cohere v2: `POST {base}/embed` with `texts`, answered by
//! `{"embeddings": {"base64": [...]}, "meta": {"billed_units": {"input_tokens": n}}}`.

use super::{Call, Parsed, check_dimensions, push_vector, str_array, writer};
use crate::error::Result;
use crate::input::{InputType, Truncate};
use crate::json::read::{Cursor, R};

pub(super) fn body(call: &Call<'_>) -> Result<(String, Vec<u8>)> {
    check_dimensions(call.model, call.dimensions, accepts_dimensions(call.model))?;
    let mut w = writer(call);
    w.raw("{");
    w.key("model");
    w.str(call.model);
    w.raw(",");
    w.key("texts");
    str_array(&mut w, call.inputs);
    w.raw(",\"input_type\":");
    w.str(input_type(call.input_type));
    // the float vectors' bytes, base64-encoded
    w.raw(",\"embedding_types\":[\"base64\"]");
    if let Some(d) = call.dimensions {
        w.raw(",\"output_dimension\":");
        w.uint(d as u64);
    }
    match call.truncate {
        Truncate::Provider => {}
        Truncate::End => w.raw(",\"truncate\":\"END\""),
        Truncate::Error => w.raw(",\"truncate\":\"NONE\""),
    }
    w.raw("}");
    Ok(("/embed".to_string(), w.finish()))
}

fn input_type(input_type: Option<InputType>) -> &'static str {
    match input_type {
        Some(InputType::SearchDocument) | None => "search_document",
        Some(InputType::SearchQuery) => "search_query",
        Some(InputType::Classification) => "classification",
        Some(InputType::Clustering) => "clustering",
    }
}

/// `output_dimension` exists from embed-v4 on; the v3 models have one width.
fn accepts_dimensions(model: &str) -> Option<bool> {
    if model.ends_with("-v3.0") {
        Some(false)
    } else if model.starts_with("embed-v") {
        Some(true)
    } else {
        None
    }
}

/// base64 vectors win over float ones when a body carries both.
pub(super) fn parse(body: &[u8]) -> R<Parsed> {
    let mut p = Parsed::default();
    let mut have_base64 = false;
    let mut have_any = false;
    let mut c = Cursor::new(body);
    c.object(|c, key| match key {
        "embeddings" => c.object(|c, key| match key {
            "base64" | "float" if !(have_base64 && key == "float") => {
                if c.null()? {
                    return Ok(());
                }
                have_any = true;
                have_base64 |= key == "base64";
                p.data.clear();
                p.dim = 0;
                c.array(|c| push_vector(c, &mut p, 0))
            }
            _ => c.skip(),
        }),
        "meta" => {
            if c.null()? {
                return Ok(());
            }
            c.object(|c, key| match key {
                "billed_units" => {
                    if c.null()? {
                        return Ok(());
                    }
                    c.object(|c, key| match key {
                        "input_tokens" => {
                            if !c.null()? {
                                p.total_tokens = Some(c.count()?);
                            }
                            Ok(())
                        }
                        _ => c.skip(),
                    })
                }
                _ => c.skip(),
            })
        }
        _ => c.skip(),
    })?;
    c.end()?;
    if !have_any {
        return c.fail("no embeddings in the response");
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_or_float() {
        let p = parse(br#"{"id":"x","embeddings":{"base64":["AACAPw=="]},"meta":{"billed_units":{"input_tokens":15}}}"#).unwrap();
        assert_eq!(p.data, vec![1.0]);
        assert_eq!(p.total_tokens, Some(15));
        let p = parse(br#"{"embeddings":{"float":[[0.5,0.6]]},"meta":{}}"#).unwrap();
        assert_eq!((p.data, p.dim), (vec![0.5, 0.6], 2));
        let p = parse(br#"{"embeddings":{"float":[[0.5]],"base64":["AACAPw=="]}}"#).unwrap();
        assert_eq!(p.data, vec![1.0]);
        let p = parse(br#"{"embeddings":{"base64":["AACAPw=="],"float":[[0.5]]}}"#).unwrap();
        assert_eq!(p.data, vec![1.0]);
    }

    #[test]
    fn missing_token_count_is_unknown_not_zero() {
        let p = parse(br#"{"embeddings":{"float":[[1]]},"meta":{"billed_units":null}}"#).unwrap();
        assert_eq!(p.total_tokens, None);
        assert_eq!(
            parse(br#"{"embeddings":{"float":[[1]]}}"#)
                .unwrap()
                .total_tokens,
            None
        );
    }

    #[test]
    fn token_count_as_a_whole_float() {
        let body = |n: &str| {
            format!(
                r#"{{"embeddings":{{"float":[[1]]}},"meta":{{"billed_units":{{"input_tokens":{n}}}}}}}"#
            )
        };
        assert_eq!(parse(body("2.0").as_bytes()).unwrap().total_tokens, Some(2));
        assert!(parse(body("2.5").as_bytes()).is_err());
    }

    #[test]
    fn response_without_vectors_is_an_error() {
        assert!(parse(br#"{"embeddings":{}}"#).is_err());
        assert!(parse(br#"{}"#).is_err());
    }

    #[test]
    fn dimension_support_by_model() {
        assert_eq!(accepts_dimensions("embed-v4.0"), Some(true));
        assert_eq!(accepts_dimensions("embed-english-v3.0"), Some(false));
        assert_eq!(accepts_dimensions("my-proxy-model"), None);
    }
}
