//! OpenAI, Voyage, Jina and Mistral: `POST {base}/embeddings` with `input`,
//! answered by `{"data": [{"index": i, "embedding": ...}], "model", "usage"}`.

use super::{Call, Parsed, check_dimensions, push_vector, str_array, writer};
use crate::endpoint::{Endpoint, OPENAI_BASE_URL, Wire};
use crate::error::{Error, Result};
use crate::input::{InputType, Truncate};
use crate::json::read::{Cursor, R};

pub(super) fn body(ep: &Endpoint, call: &Call<'_>) -> Result<(String, Vec<u8>)> {
    let model = call.model;
    let accepts = match ep.wire {
        Wire::Voyage => voyage_accepts_dimensions(model),
        Wire::Mistral => mistral_accepts_dimensions(model),
        // `dimensions` exists from text-embedding-3 on
        Wire::OpenAi if model == "text-embedding-ada-002" => Some(false),
        _ => None,
    };
    check_dimensions(model, call.dimensions, accepts)?;
    let truncate = match (ep.wire, call.truncate) {
        (_, Truncate::Provider) => None,
        (Wire::Voyage, t) => Some(("truncation", t == Truncate::End)),
        (Wire::Jina, Truncate::End) => Some(("truncate", true)),
        // openai and jina refuse long inputs by default
        (Wire::OpenAi | Wire::Jina, Truncate::Error) => None,
        (_, t) => return Err(Error::unsupported(model, format!("truncate({t:?})"))),
    };

    let mut w = writer(call);
    w.raw("{");
    w.key("model");
    w.str(model);
    w.raw(",");
    w.key("input");
    str_array(&mut w, call.inputs);
    let dims_key = match ep.wire {
        Wire::OpenAi => {
            w.raw(",\"encoding_format\":");
            // the official endpoint sends base64 (2.3x smaller and cheaper to
            // parse); compatible servers get "float", which all understand
            w.str(if ep.base_url == OPENAI_BASE_URL {
                "base64"
            } else {
                "float"
            });
            "dimensions"
        }
        Wire::Mistral => {
            w.raw(",\"encoding_format\":\"float\"");
            "output_dimension"
        }
        Wire::Jina => {
            w.raw(",\"embedding_type\":\"float\"");
            "dimensions"
        }
        _ => {
            if let Some(t) = voyage_input_type(call.input_type) {
                w.raw(",\"input_type\":");
                w.str(t);
            }
            "output_dimension"
        }
    };
    if let Some(d) = call.dimensions {
        w.raw(",");
        w.key(dims_key);
        w.uint(d as u64);
    }
    match ep.wire {
        Wire::Jina => {
            if let Some(task) = jina_task(model, call.input_type)? {
                w.raw(",\"task\":");
                w.str(task);
            }
        }
        // voyage documents base64 as the raw float32 array
        Wire::Voyage => w.raw(",\"encoding_format\":\"base64\""),
        _ => {}
    }
    if let Some((key, on)) = truncate {
        w.raw(",");
        w.key(key);
        w.raw(if on { "true" } else { "false" });
    }
    w.raw("}");
    Ok(("/embeddings".to_string(), w.finish()))
}

fn voyage_input_type(input_type: Option<InputType>) -> Option<&'static str> {
    input_type.map(|it| match it {
        InputType::SearchQuery => "query",
        _ => "document",
    })
}

/// The `task` for an input type. Each model family has its own list; an
/// input type outside it is refused rather than sent and rejected.
fn jina_task(model: &str, input_type: Option<InputType>) -> Result<Option<&'static str>> {
    let Some(it) = input_type else {
        return Ok(None);
    };
    let refuse = || Err(Error::unsupported(model, format!("input_type({it:?})")));
    if model.starts_with("jina-embeddings-v2") || model.starts_with("jina-clip-v1") {
        // their request schemas have no `task`
        return Ok(None);
    }
    if model.starts_with("jina-code-embeddings") {
        // its tasks are nl2code, qa, code2code and so on, none of the input types
        return refuse();
    }
    if model.starts_with("jina-clip-v2") {
        // documents go without a task
        return match it {
            InputType::SearchQuery => Ok(Some("retrieval.query")),
            InputType::SearchDocument => Ok(None),
            _ => refuse(),
        };
    }
    Ok(Some(match it {
        InputType::SearchDocument => "retrieval.passage",
        InputType::SearchQuery => "retrieval.query",
        // v4 has only retrieval, text-matching and code tasks
        _ if model.starts_with("jina-embeddings-v4") => return refuse(),
        InputType::Classification => "classification",
        // v5 renamed v3's `separation` adapter to `clustering`
        InputType::Clustering if model.starts_with("jina-embeddings-v5") => "clustering",
        InputType::Clustering => "separation",
    }))
}

// models with a single output width; newer models take `output_dimension`
const VOYAGE_FIXED_WIDTH: &[&str] = &[
    "voyage-3",
    "voyage-3-lite",
    "voyage-2",
    "voyage-large-2",
    "voyage-large-2-instruct",
    "voyage-code-2",
    "voyage-law-2",
    "voyage-finance-2",
    "voyage-multilingual-2",
];

/// `Some(false)` for models documented with a fixed width, `None` when unknown.
fn voyage_accepts_dimensions(model: &str) -> Option<bool> {
    if VOYAGE_FIXED_WIDTH.contains(&model) {
        Some(false)
    } else if model.starts_with("voyage-4")
        || model.starts_with("voyage-3.5")
        || model == "voyage-3-large"
        || model == "voyage-code-3"
        || model == "voyage-code-4"
    {
        Some(true)
    } else {
        None
    }
}

fn mistral_accepts_dimensions(model: &str) -> Option<bool> {
    if model.starts_with("codestral-embed") {
        Some(true)
    } else if model.starts_with("mistral-embed") {
        Some(false)
    } else {
        None
    }
}

/// Vectors land in arrival order and are put back in input order by
/// `index` when every item has one; a repeated or out-of-range index fails.
pub(super) fn parse(body: &[u8], n: usize) -> R<Parsed> {
    let mut p = Parsed::default();
    let mut index: Vec<Option<u64>> = Vec::with_capacity(n);
    let mut c = Cursor::new(body);
    let mut saw_data = false;
    c.object(|c, key| match key {
        "data" => {
            saw_data = true;
            c.array(|c| {
                let mut i = None;
                let mut got = false;
                c.object(|c, key| match key {
                    "embedding" if !got => {
                        got = true;
                        push_vector(c, &mut p, n)
                    }
                    "index" => {
                        if !c.null()? {
                            i = Some(c.u64()?);
                        }
                        Ok(())
                    }
                    _ => c.skip(),
                })?;
                if !got {
                    return c.fail("item without an embedding");
                }
                index.push(i);
                Ok(())
            })
        }
        "model" => {
            if !c.null()? {
                p.model = Some(c.str()?.into_owned());
            }
            Ok(())
        }
        "usage" => {
            if c.null()? {
                return Ok(());
            }
            c.object(|c, key| match key {
                "total_tokens" => {
                    if !c.null()? {
                        p.total_tokens = Some(c.count()?);
                    }
                    Ok(())
                }
                _ => c.skip(),
            })
        }
        _ => c.skip(),
    })?;
    c.end()?;
    if !saw_data {
        return c.fail("no data in the response");
    }
    if index.iter().all(Option::is_some) && !index.is_empty() {
        reorder(&mut p, &index).map_err(|what| crate::json::read::Fail {
            offset: body.len(),
            what,
        })?;
    }
    Ok(p)
}

fn reorder(p: &mut Parsed, index: &[Option<u64>]) -> std::result::Result<(), &'static str> {
    let rows = index.len();
    let mut seen = vec![false; rows];
    let mut in_order = true;
    for (pos, i) in index.iter().enumerate() {
        let i = i.unwrap_or(u64::MAX);
        match seen.get_mut(i as usize).filter(|_| i < rows as u64) {
            Some(s @ false) => *s = true,
            _ => return Err("response index out of range or repeated"),
        }
        in_order &= i as usize == pos;
    }
    if in_order || p.dim == 0 {
        return Ok(());
    }
    let d = p.dim;
    let mut out = vec![0f32; p.data.len()];
    for (pos, i) in index.iter().enumerate() {
        let i = i.unwrap_or_default() as usize;
        out[i * d..(i + 1) * d].copy_from_slice(&p.data[pos * d..(pos + 1) * d]);
    }
    p.data = out;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(json: &str) -> R<Parsed> {
        parse(json.as_bytes(), 2)
    }

    #[test]
    fn reordered_items_come_back_in_input_order() {
        let raw = p(
            r#"{"object":"list","data":[{"index":1,"embedding":[2.0,2.5]},
            {"object":"embedding","index":0,"embedding":[1.0,1.5]}],
            "model":"x","usage":{"prompt_tokens":4,"total_tokens":4}}"#,
        )
        .unwrap();
        assert_eq!(raw.data, vec![1.0, 1.5, 2.0, 2.5]);
        assert_eq!(raw.dim, 2);
        assert_eq!(raw.total_tokens, Some(4));
        assert_eq!(raw.model.as_deref(), Some("x"));
        let raw =
            p(r#"{"data":[{"embedding":[1]},{"embedding":[2]}],"usage":{"total_tokens":4.0}}"#);
        assert_eq!(raw.unwrap().total_tokens, Some(4));
    }

    #[test]
    fn index_after_the_embedding_also_counts() {
        let raw = p(r#"{"data":[{"embedding":[2],"index":1},{"embedding":[1],"index":0}]}"#);
        assert_eq!(raw.unwrap().data, vec![1.0, 2.0]);
    }

    #[test]
    fn repeated_or_out_of_range_index_is_an_error() {
        assert!(
            p(r#"{"data":[{"index":0,"embedding":[1]},{"index":0,"embedding":[2]}]}"#).is_err()
        );
        assert!(p(r#"{"data":[{"index":2,"embedding":[1]}]}"#).is_err());
    }

    #[test]
    fn missing_index_model_and_usage_are_tolerated() {
        let raw = p(r#"{"data":[{"embedding":[1]},{"embedding":[2]}],"model":null,"usage":null}"#)
            .unwrap();
        assert_eq!(raw.data, vec![1.0, 2.0]);
        assert_eq!(raw.total_tokens, None);
        assert_eq!(raw.model, None);
    }

    #[test]
    fn base64_items_are_decoded() {
        // 1.0f32 little-endian is 00 00 80 3f
        let raw = p(r#"{"data":[{"index":0,"embedding":"AACAPw=="}],"usage":{"total_tokens":1}}"#)
            .unwrap();
        assert_eq!(raw.data, vec![1.0]);
        // f32::from_le_bytes([255, 255, 127, 63]) encodes to "//9/Pw=="
        let raw = p(r#"{"data":[{"index":0,"embedding":"\/\/9\/Pw=="}]}"#).unwrap();
        assert_eq!(raw.data[0].to_bits(), 0x3f7f_ffff);
    }

    #[test]
    fn malformed_bodies_fail() {
        for bad in [
            "",
            "[]",
            r#"{"data":[{"index":0}]}"#,
            r#"{"data":[{"embedding":[1]},{"embedding":[1,2]}]}"#,
            r#"{"data":[{"embedding":[1]}]} x"#,
            r#"{"model":"m"}"#,
            r#"{"data":[{"embedding":[1]}],"usage":{"total_tokens":-1}}"#,
            r#"{"data":[{"embedding":[1]}],"usage":{"total_tokens":1.5}}"#,
        ] {
            assert!(p(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn jina_tasks_by_model() {
        use InputType::*;
        let t = |m: &str, it| jina_task(m, Some(it)).ok();
        assert_eq!(
            t("jina-embeddings-v3", Clustering),
            Some(Some("separation"))
        );
        assert_eq!(
            t("jina-embeddings-v5-text-small", Clustering),
            Some(Some("clustering"))
        );
        assert_eq!(
            t("jina-embeddings-v4", SearchDocument),
            Some(Some("retrieval.passage"))
        );
        assert_eq!(t("jina-embeddings-v4", Classification), None);
        assert_eq!(t("jina-embeddings-v4", Clustering), None);
        assert_eq!(
            t("jina-clip-v2", SearchQuery),
            Some(Some("retrieval.query"))
        );
        assert_eq!(t("jina-clip-v2", SearchDocument), Some(None));
        assert_eq!(t("jina-clip-v2", Clustering), None);
        assert_eq!(t("jina-code-embeddings-1.5b", SearchQuery), None);
        assert_eq!(jina_task("jina-embeddings-v4", None).ok(), Some(None));
        assert_eq!(t("jina-embeddings-v2-base-en", SearchQuery), Some(None));
        assert_eq!(t("jina-clip-v1", Clustering), Some(None));
    }

    #[test]
    fn dimension_support_by_model() {
        for m in [
            "voyage-4-large",
            "voyage-3.5",
            "voyage-3-large",
            "voyage-code-3",
        ] {
            assert_eq!(voyage_accepts_dimensions(m), Some(true), "{m}");
        }
        for m in ["voyage-3", "voyage-law-2", "voyage-code-2"] {
            assert_eq!(voyage_accepts_dimensions(m), Some(false), "{m}");
        }
        assert_eq!(voyage_accepts_dimensions("voyage-9"), None);
        assert_eq!(
            mistral_accepts_dimensions("codestral-embed-2505"),
            Some(true)
        );
        assert_eq!(mistral_accepts_dimensions("mistral-embed"), Some(false));
        assert_eq!(mistral_accepts_dimensions("other"), None);
        let ep = Endpoint::openai("k");
        let inputs = crate::input::Inputs::from(&["x"][..]);
        let call = |model| Call {
            model,
            inputs: &inputs,
            dimensions: Some(256),
            input_type: None,
            truncate: Truncate::Provider,
        };
        assert!(body(&ep, &call("text-embedding-ada-002")).is_err());
        assert!(body(&ep, &call("text-embedding-3-small")).is_ok());
    }
}
