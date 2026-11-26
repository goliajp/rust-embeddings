//! Gemini: `models/{model}:embedContent` for one text,
//! `:batchEmbedContents` for several; the key goes in `x-goog-api-key`.

use super::{Call, Parsed, push_vector, writer};
use crate::error::{Error, Result};
use crate::input::{InputType, Truncate};
use crate::json::read::{Cursor, R};
use crate::json::write::Writer;

pub(super) fn body(call: &Call<'_>) -> Result<(String, Vec<u8>)> {
    if call.truncate != Truncate::Provider {
        return Err(Error::unsupported(
            call.model,
            format!("truncate({:?})", call.truncate),
        ));
    }
    let model = call.model;
    let mut w = writer(call);
    if call.inputs.len() == 1 {
        request(&mut w, call, call.inputs.get(0));
        return Ok((format!("/models/{model}:embedContent"), w.finish()));
    }
    w.raw("{\"requests\":[");
    for (i, text) in call.inputs.iter().enumerate() {
        if i > 0 {
            w.byte(b',');
        }
        request(&mut w, call, text);
    }
    w.raw("]}");
    Ok((format!("/models/{model}:batchEmbedContents"), w.finish()))
}

fn request(w: &mut Writer, call: &Call<'_>, text: &str) {
    w.raw("{\"model\":");
    w.str(&format!("models/{}", call.model));
    w.raw(",\"content\":{\"parts\":[{\"text\":");
    // gemini-embedding-2 takes no task type; the task is written in front of the text
    let task_type = if call.model.starts_with("gemini-embedding-2") {
        match task_prefix(call.input_type) {
            Some(prefix) => w.str(&format!("{prefix}{text}")),
            None => w.str(text),
        }
        None
    } else {
        w.str(text);
        task_type(call.input_type)
    };
    w.raw("}]}");
    if let Some(t) = task_type {
        w.raw(",\"taskType\":");
        w.str(t);
    }
    if let Some(d) = call.dimensions {
        w.raw(",\"outputDimensionality\":");
        w.uint(d as u64);
    }
    w.raw("}");
}

fn task_type(input_type: Option<InputType>) -> Option<&'static str> {
    input_type.map(|it| match it {
        InputType::SearchDocument => "RETRIEVAL_DOCUMENT",
        InputType::SearchQuery => "RETRIEVAL_QUERY",
        InputType::Classification => "CLASSIFICATION",
        InputType::Clustering => "CLUSTERING",
    })
}

fn task_prefix(input_type: Option<InputType>) -> Option<&'static str> {
    input_type.map(|it| match it {
        InputType::SearchQuery => "task: search result | query: ",
        InputType::SearchDocument => "title: none | text: ",
        InputType::Classification => "task: classification | query: ",
        InputType::Clustering => "task: clustering | query: ",
    })
}

/// `{"embedding": {"values": [...]}}` for one text, `{"embeddings": [...]}`
/// for several; `usageMetadata.promptTokenCount` in both.
pub(super) fn parse(body: &[u8], n: usize) -> R<Parsed> {
    let mut p = Parsed::default();
    let mut found = false;
    let mut c = Cursor::new(body);
    c.object(|c, key| match key {
        "embedding" if n == 1 => {
            if c.null()? {
                return Ok(());
            }
            found = true;
            values(c, &mut p, n)
        }
        "embeddings" if n != 1 => {
            found = true;
            c.array(|c| values(c, &mut p, n))
        }
        "usageMetadata" => {
            if c.null()? {
                return Ok(());
            }
            c.object(|c, key| match key {
                "promptTokenCount" => {
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
    if !found {
        return c.fail("no embedding in the response");
    }
    Ok(p)
}

/// `retryDelay` of the `google.rpc.RetryInfo` in `error.details`.
pub(super) fn retry_delay(body: &[u8]) -> Option<std::time::Duration> {
    const RETRY_INFO: &str = "type.googleapis.com/google.rpc.RetryInfo";
    let mut found = None;
    let mut c = Cursor::new(body);
    let detail = |c: &mut Cursor<'_>, found: &mut Option<String>| {
        let (mut ty, mut delay) = (None, None);
        c.object(|c, key| match key {
            "@type" | "retryDelay" if c.peek() == Some(b'"') => {
                let v = c.str()?.into_owned();
                if key == "@type" {
                    ty = Some(v);
                } else {
                    delay = Some(v);
                }
                Ok(())
            }
            _ => c.skip(),
        })?;
        if ty.as_deref() == Some(RETRY_INFO) && delay.is_some() {
            *found = delay;
        }
        Ok(())
    };
    c.object(|c, key| match key {
        "error" => c.object(|c, key| match key {
            "details" => c.array(|c| detail(c, &mut found)),
            _ => c.skip(),
        }),
        _ => c.skip(),
    })
    .ok()?;
    crate::retry::parse_duration(&found?)
}

fn values(c: &mut Cursor<'_>, p: &mut Parsed, n: usize) -> R<()> {
    let mut got = false;
    c.object(|c, key| match key {
        "values" if !got => {
            got = true;
            push_vector(c, p, n)
        }
        _ => c.skip(),
    })?;
    if got {
        Ok(())
    } else {
        c.fail("embedding without values")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_and_batch_responses() {
        let p = parse(
            br#"{"embedding":{"values":[0.1,0.2]},"usageMetadata":{"promptTokenCount":3}}"#,
            1,
        )
        .unwrap();
        assert_eq!((p.data, p.total_tokens), (vec![0.1, 0.2], Some(3)));
        let p = parse(br#"{"embeddings":[{"values":[0.1]},{"values":[0.3]}],"usageMetadata":{"promptTokenCount":7}}"#, 2).unwrap();
        assert_eq!((p.data, p.total_tokens), (vec![0.1, 0.3], Some(7)));
        let p = parse(br#"{"embeddings":[]}"#, 0).unwrap();
        assert!(p.data.is_empty() && p.total_tokens.is_none());
    }

    #[test]
    fn retry_delay_from_retry_info() {
        let body = br#"{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED",
            "details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":"x"}]},
            {"retryDelay":"13s","@type":"type.googleapis.com/google.rpc.RetryInfo"}]}}"#;
        assert_eq!(retry_delay(body), Some(std::time::Duration::from_secs(13)));
        let half = br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1.5s"}]}}"#;
        assert_eq!(
            retry_delay(half),
            Some(std::time::Duration::from_millis(1500))
        );
        for bad in [
            &br#"{"error":{"details":[{"@type":"other","retryDelay":"13s"}]}}"#[..],
            br#"{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"soon"}]}}"#,
            br#"{"error":{"message":"no details"}}"#,
            b"not json",
        ] {
            assert_eq!(retry_delay(bad), None);
        }
    }

    #[test]
    fn missing_embedding_is_an_error() {
        assert!(parse(br#"{"usageMetadata":{}}"#, 1).is_err());
        assert!(parse(br#"{"embedding":null}"#, 1).is_err());
        assert!(parse(br#"{"embeddings":[{"x":1}]}"#, 2).is_err());
    }
}
