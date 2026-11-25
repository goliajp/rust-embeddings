//! The HTTP seam: embedrs builds requests and parses responses, a
//! [`Transport`] moves the bytes.
//!
//! With the default `reqwest` feature, [`ReqwestTransport`] is used by every
//! named constructor. Without it, implement [`Transport`] over any HTTP
//! client and runtime and pass it to [`Client::new`](crate::Client::new), or
//! drive requests yourself with [`Client::prepare`](crate::Client::prepare).

use std::error::Error as StdError;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

#[cfg(feature = "reqwest")]
mod reqwest;
#[cfg(feature = "reqwest")]
pub use self::reqwest::ReqwestTransport;

/// A boxed, sendable future, the return type of [`Transport`] methods.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Moves HTTP requests for a [`Client`](crate::Client).
///
/// embedrs never spawns tasks and never calls a runtime: it only polls the
/// futures returned here, so the client works on whatever executor these
/// futures work on.
pub trait Transport: Send + Sync + 'static {
    /// Send one request and return the status, headers and the whole body.
    ///
    /// Redirects should not be followed: embedrs handles the ones it needs.
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>>;

    /// Resolve after `d`. Used for retry backoff and timeouts.
    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()>;

    /// Send one request and write the body to `sink` as it arrives. Only a
    /// 2xx body is written; for other statuses only the head is returned.
    ///
    /// The default buffers the body through [`Transport::send`]; implement it
    /// to stream large downloads (local model weights) instead of holding
    /// them in memory.
    fn download<'a>(
        &'a self,
        req: HttpRequest,
        sink: &'a mut (dyn std::io::Write + Send),
    ) -> BoxFuture<'a, Result<ResponseHead, TransportError>> {
        Box::pin(async move {
            let resp = self.send(req).await?;
            if (200..300).contains(&resp.status) {
                sink.write_all(&resp.body)
                    .map_err(|e| TransportError::new(TransportErrorKind::Io, e))?;
            }
            Ok(ResponseHead {
                status: resp.status,
                headers: resp.headers,
            })
        })
    }
}

/// HTTP method of a request embedrs sends.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    /// `GET`
    Get,
    /// `POST`
    Post,
}

impl Method {
    /// The method name, upper case.
    pub fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
        }
    }
}

/// One request, fully built.
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq)]
pub struct HttpRequest {
    /// Method.
    pub method: Method,
    /// Absolute URL.
    pub url: String,
    /// Header names are lower case.
    pub headers: Vec<(&'static str, String)>,
    /// Body bytes (JSON for embedding calls, empty for downloads).
    pub body: Vec<u8>,
}

const SECRET_HEADERS: [&str; 3] = ["authorization", "x-goog-api-key", "api-key"];

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let headers: Vec<(&str, &str)> = self
            .headers
            .iter()
            .map(|(k, v)| {
                let v = if SECRET_HEADERS.contains(k) {
                    "<redacted>"
                } else {
                    v.as_str()
                };
                (*k, v)
            })
            .collect();
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("url", &self.url)
            .field("headers", &headers)
            .field("body", &format_args!("<{} bytes>", self.body.len()))
            .finish()
    }
}

/// Status and headers of a response.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseHead {
    /// HTTP status code.
    pub status: u16,
    /// Header names in lower case.
    pub headers: Vec<(String, String)>,
}

impl ResponseHead {
    /// A response head; header names are lower-cased.
    pub fn new(status: u16, headers: Vec<(String, String)>) -> Self {
        Self {
            status,
            headers: lower(headers),
        }
    }

    /// First value of a header, by lower-case name.
    pub fn header(&self, name: &str) -> Option<&str> {
        find(&self.headers, name)
    }
}

/// A complete response.
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status: u16,
    /// Header names in lower case.
    pub headers: Vec<(String, String)>,
    /// The whole body.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// A response; header names are lower-cased.
    pub fn new(status: u16, headers: Vec<(String, String)>, body: Vec<u8>) -> Self {
        Self {
            status,
            headers: lower(headers),
            body,
        }
    }

    /// First value of a header, by lower-case name.
    pub fn header(&self, name: &str) -> Option<&str> {
        find(&self.headers, name)
    }
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .field("body", &format_args!("<{} bytes>", self.body.len()))
            .finish()
    }
}

fn lower(mut headers: Vec<(String, String)>) -> Vec<(String, String)> {
    for (k, _) in &mut headers {
        k.make_ascii_lowercase();
    }
    headers
}

fn find<'h>(headers: &'h [(String, String)], name: &str) -> Option<&'h str> {
    headers
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

/// What went wrong below HTTP.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransportErrorKind {
    /// The connection could not be made (DNS, refused, TLS handshake).
    Connect,
    /// The transport's own timeout fired.
    Timeout,
    /// Reading or writing failed after the connection was made.
    Io,
    /// Anything else.
    Other,
}

/// A failure to exchange a request and response.
#[derive(Debug)]
pub struct TransportError {
    kind: TransportErrorKind,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl TransportError {
    /// An error of `kind` caused by `source`.
    pub fn new(
        kind: TransportErrorKind,
        source: impl Into<Box<dyn StdError + Send + Sync>>,
    ) -> Self {
        Self {
            kind,
            source: Some(source.into()),
        }
    }

    /// An error of `kind` with no underlying cause.
    pub fn from_kind(kind: TransportErrorKind) -> Self {
        Self { kind, source: None }
    }

    /// The kind, which decides whether a request is retried.
    pub fn kind(&self) -> TransportErrorKind {
        self.kind
    }

    /// Connect errors and transport timeouts are worth another attempt.
    pub(crate) fn is_retryable(&self) -> bool {
        matches!(
            self.kind,
            TransportErrorKind::Connect | TransportErrorKind::Timeout
        )
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            TransportErrorKind::Connect => "connect error",
            TransportErrorKind::Timeout => "transport timeout",
            TransportErrorKind::Io => "i/o error",
            TransportErrorKind::Other => "transport error",
        };
        match &self.source {
            Some(s) => write!(f, "{kind}: {s}"),
            None => f.write_str(kind),
        }
    }
}

impl StdError for TransportError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|s| s as &(dyn StdError + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_hides_credentials() {
        let req = HttpRequest {
            method: Method::Post,
            url: "https://x/v1/embeddings".into(),
            headers: vec![
                ("authorization", "Bearer sk-secret".into()),
                ("x-goog-api-key", "g-secret".into()),
                ("api-key", "azure-secret".into()),
                ("content-type", "application/json".into()),
            ],
            body: b"{}".to_vec(),
        };
        let s = format!("{req:?}");
        assert!(!s.contains("secret"), "{s}");
        assert!(s.contains("application/json"));
        assert!(s.contains("<2 bytes>"));
    }

    #[test]
    fn headers_are_found_case_insensitively() {
        let r = HttpResponse::new(200, vec![("Retry-After".into(), "3".into())], vec![]);
        assert_eq!(r.header("retry-after"), Some("3"));
        assert_eq!(r.header("x"), None);
    }

    #[test]
    fn error_display_and_source() {
        let e = TransportError::new(
            TransportErrorKind::Connect,
            std::io::Error::other("refused"),
        );
        assert_eq!(e.to_string(), "connect error: refused");
        assert!(e.source().is_some());
        assert!(e.is_retryable());
        let e = TransportError::from_kind(TransportErrorKind::Io);
        assert!(e.source().is_none());
        assert!(!e.is_retryable());
    }
}
