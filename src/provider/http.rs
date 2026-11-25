use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{Error, Result};

/// POST `body` as JSON and parse a 2xx response as `T`. A non-2xx status
/// becomes [`Error::Api`] with the response text and any `Retry-After` hint.
pub(crate) async fn post_json<T: DeserializeOwned>(
    req: reqwest::RequestBuilder,
    body: &impl Serialize,
) -> Result<T> {
    let resp = req.json(body).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let retry_after = parse_retry_after(resp.headers());
        let message = resp.text().await.unwrap_or_default();
        return Err(Error::Api {
            status: status.as_u16(),
            message,
            retry_after,
        });
    }
    let bytes = resp.bytes().await?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Parse `Retry-After` (RFC 9110 section 10.2.3) in its delay-seconds form.
///
/// The HTTP-date form is not parsed; callers fall back to configured backoff
/// when this returns `None`.
pub(crate) fn parse_retry_after(
    headers: &reqwest::header::HeaderMap,
) -> Option<std::time::Duration> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let secs: u64 = raw.trim().parse().ok()?;
    Some(std::time::Duration::from_secs(secs))
}
