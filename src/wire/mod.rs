//! Provider request builders and response parsers. Every function here is
//! synchronous and does no I/O.

mod cohere;
mod gemini;
mod openai;

use crate::endpoint::{Endpoint, Wire};
use crate::error::{DecodeError, Error, Result};
use crate::input::{InputType, Inputs, Truncate};
use crate::json::read::{Cursor, Fail};
use crate::json::write::Writer;
use crate::transport::{HttpRequest, HttpResponse, Method};

/// One request's worth of input and options.
pub(crate) struct Call<'a> {
    pub(crate) model: &'a str,
    pub(crate) inputs: &'a Inputs<'a>,
    pub(crate) dimensions: Option<u32>,
    pub(crate) input_type: Option<InputType>,
    pub(crate) truncate: Truncate,
}

/// The vectors of one response, row-major, and what it reported.
#[derive(Debug, Default)]
pub(crate) struct Parsed {
    pub(crate) data: Vec<f32>,
    pub(crate) dim: usize,
    pub(crate) total_tokens: Option<u64>,
    pub(crate) model: Option<String>,
}

pub(crate) fn build(ep: &Endpoint, call: &Call<'_>) -> Result<HttpRequest> {
    let (path, body) = match ep.wire {
        Wire::OpenAi | Wire::Voyage | Wire::Jina | Wire::Mistral => openai::body(ep, call)?,
        Wire::Cohere => cohere::body(call)?,
        Wire::Gemini => gemini::body(call)?,
    };
    let mut headers = Vec::with_capacity(2 + ep.headers.len());
    headers.push(("content-type", "application/json".to_string()));
    if ep.wire == Wire::Gemini {
        // in a header: a url with the key would end up in error messages and logs
        headers.push(("x-goog-api-key", ep.key.clone()));
    } else if ep.bearer {
        headers.push(("authorization", format!("Bearer {}", ep.key)));
    }
    headers.extend(ep.headers.iter().cloned());
    Ok(HttpRequest {
        method: Method::Post,
        url: format!("{}{path}", ep.base_url),
        headers,
        body,
    })
}

/// Check the status and read the vectors of a response to a request for `n` inputs.
pub(crate) fn parse(wire: Wire, n: usize, resp: &HttpResponse) -> Result<Parsed> {
    if !(200..300).contains(&resp.status) {
        return Err(api_error(wire, resp));
    }
    let decode = |f: Fail| DecodeError::new(wire.name(), f.offset, f.what);
    let parsed = match wire {
        Wire::OpenAi | Wire::Voyage | Wire::Jina | Wire::Mistral => {
            openai::parse(&resp.body, n).map_err(decode)?
        }
        Wire::Cohere => cohere::parse(&resp.body).map_err(decode)?,
        Wire::Gemini => gemini::parse(&resp.body, n).map_err(decode)?,
    };
    let got = parsed.data.len().checked_div(parsed.dim).unwrap_or(0);
    if got != n {
        return Err(DecodeError::new(
            wire.name(),
            resp.body.len(),
            format!("{got} vectors for {n} inputs"),
        )
        .into());
    }
    Ok(parsed)
}

/// Appends one vector to `p.data`; the first one fixes the width.
pub(crate) fn push_vector(
    c: &mut Cursor<'_>,
    p: &mut Parsed,
    n: usize,
) -> crate::json::read::R<()> {
    let len = c.vector(&mut p.data)?;
    if p.dim == 0 && len > 0 && p.data.len() == len {
        p.dim = len;
        p.data.reserve(len * n.saturating_sub(1));
    } else if len != p.dim {
        return c.fail("vectors of different lengths in one response");
    }
    Ok(())
}

/// Refuse a requested width the model is known not to support, instead of
/// sending it and getting vectors of another width back.
pub(crate) fn check_dimensions(
    model: &str,
    dims: Option<u32>,
    accepted: Option<bool>,
) -> Result<()> {
    match (dims, accepted) {
        (Some(d), Some(false)) => Err(Error::unsupported(model, format!("dimensions({d})"))),
        _ => Ok(()),
    }
}

pub(crate) fn writer(call: &Call<'_>) -> Writer {
    let text: usize = call.inputs.iter().map(str::len).sum();
    Writer::with_capacity(text + 16 * call.inputs.len() + 256)
}

/// `[ "a", "b" ]`
pub(crate) fn str_array(w: &mut Writer, inputs: &Inputs<'_>) {
    w.byte(b'[');
    for (i, t) in inputs.iter().enumerate() {
        if i > 0 {
            w.byte(b',');
        }
        w.str(t);
    }
    w.byte(b']');
}

const MAX_MESSAGE: usize = 2048;

fn api_error(wire: Wire, resp: &HttpResponse) -> Error {
    Error::Api {
        status: resp.status,
        message: error_message(&resp.body),
        retry_after: resp
            .header("retry-after")
            .and_then(|v| crate::retry::parse_retry_after(v, std::time::SystemTime::now())),
        provider: wire.name(),
    }
}

/// `error.message`, `error` (a string), `message` or `detail` of a JSON
/// error body; otherwise the start of the body as text.
pub(crate) fn error_message(body: &[u8]) -> String {
    if let Some(m) = json_message(body) {
        return m;
    }
    let text = String::from_utf8_lossy(body);
    let text = text.trim();
    let mut end = text.len().min(MAX_MESSAGE);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

fn json_message(body: &[u8]) -> Option<String> {
    let mut found: Option<String> = None;
    let mut c = Cursor::new(body);
    c.object(|c, key| {
        match key {
            "error" => match c.peek() {
                Some(b'"') => found = Some(c.str()?.into_owned()),
                Some(b'{') => c.object(|c, key| {
                    if key == "message" && c.peek() == Some(b'"') {
                        found = Some(c.str()?.into_owned());
                        Ok(())
                    } else {
                        c.skip()
                    }
                })?,
                _ => c.skip()?,
            },
            "message" | "detail" if found.is_none() && c.peek() == Some(b'"') => {
                found = Some(c.str()?.into_owned());
            }
            _ => c.skip()?,
        }
        Ok(())
    })
    .ok()?;
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages() {
        for (body, want) in [
            (r#"{"error":{"message":"bad key","type":"x"}}"#, "bad key"),
            (r#"{"error":"plain"}"#, "plain"),
            (r#"{"message":"cohere says no"}"#, "cohere says no"),
            (r#"{"detail":"jina says no"}"#, "jina says no"),
            (
                r#"{"error":{"code":400,"message":"gemini","status":"INVALID_ARGUMENT"}}"#,
                "gemini",
            ),
            ("  upstream timeout\n", "upstream timeout"),
            (r#"{"other":1}"#, r#"{"other":1}"#),
        ] {
            assert_eq!(error_message(body.as_bytes()), want, "{body}");
        }
        let long = "é".repeat(2000);
        let m = error_message(long.as_bytes());
        assert!(m.len() <= MAX_MESSAGE && m.chars().all(|c| c == 'é'));
    }

    #[test]
    fn non_2xx_is_an_api_error_with_retry_after() {
        let resp = HttpResponse::new(
            429,
            vec![("Retry-After".into(), "7".into())],
            br#"{"error":{"message":"slow down"}}"#.to_vec(),
        );
        match parse(Wire::Voyage, 1, &resp) {
            Err(Error::Api {
                status: 429,
                message,
                retry_after,
                provider: "voyage",
            }) => {
                assert_eq!(message, "slow down");
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(7)));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn count_mismatch_is_a_decode_error() {
        let resp = HttpResponse::new(200, vec![], br#"{"data":[{"embedding":[1]}]}"#.to_vec());
        assert!(matches!(
            parse(Wire::OpenAi, 2, &resp),
            Err(Error::Decode(_))
        ));
    }

    #[test]
    fn dimension_check() {
        assert!(check_dimensions("m", Some(256), Some(false)).is_err());
        assert!(check_dimensions("m", None, Some(false)).is_ok());
        assert!(check_dimensions("m", Some(256), None).is_ok());
        assert!(check_dimensions("m", Some(256), Some(true)).is_ok());
    }
}
