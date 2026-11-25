use std::sync::OnceLock;
use std::time::Duration;

use super::{
    BoxFuture, HttpRequest, HttpResponse, Method, ResponseHead, Transport, TransportError,
    TransportErrorKind,
};

/// [`Transport`] over a [`reqwest::Client`] on tokio (feature `reqwest`).
///
/// The named constructors ([`Client::openai`](crate::Client::openai) and the
/// others) share one process-wide instance. Build your own from a configured
/// `reqwest::Client` for proxies, custom TLS roots or a shared pool:
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use embedrs::transport::ReqwestTransport;
///
/// let http = ReqwestTransport::client_builder()
///     .proxy(reqwest::Proxy::all("http://proxy.internal:3128").unwrap())
///     .build()
///     .unwrap();
/// let client = embedrs::Client::openai("sk-...")
///     .with_transport(Arc::new(ReqwestTransport::from(http)));
/// ```
#[derive(Debug, Clone)]
pub struct ReqwestTransport {
    http: reqwest::Client,
}

impl ReqwestTransport {
    /// A transport with its own connection pool, no redirects followed.
    pub fn new() -> Self {
        let http = Self::client_builder()
            .build()
            .expect("a reqwest client with default settings builds");
        Self { http }
    }

    /// A `reqwest::ClientBuilder` set up the way embedrs needs: TLS through
    /// rustls with the ring provider, no redirects followed. Configure it
    /// and pass the built client to [`ReqwestTransport::from`].
    ///
    /// Build reqwest clients through this (or install a rustls crypto
    /// provider yourself first): embedrs enables reqwest without a bundled
    /// provider, and reqwest panics when it finds none.
    pub fn client_builder() -> reqwest::ClientBuilder {
        // a provider the application installed first wins
        let _ = rustls::crypto::ring::default_provider().install_default();
        reqwest::Client::builder().redirect(reqwest::redirect::Policy::none())
    }

    /// The process-wide instance used by the named constructors.
    pub(crate) fn shared() -> std::sync::Arc<dyn Transport> {
        static SHARED: OnceLock<std::sync::Arc<ReqwestTransport>> = OnceLock::new();
        SHARED
            .get_or_init(|| std::sync::Arc::new(ReqwestTransport::new()))
            .clone()
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl From<reqwest::Client> for ReqwestTransport {
    fn from(http: reqwest::Client) -> Self {
        Self { http }
    }
}

fn kind(e: &reqwest::Error) -> TransportErrorKind {
    if e.is_connect() {
        TransportErrorKind::Connect
    } else if e.is_timeout() {
        TransportErrorKind::Timeout
    } else if e.is_body() || e.is_request() {
        TransportErrorKind::Io
    } else {
        TransportErrorKind::Other
    }
}

fn error(e: reqwest::Error) -> TransportError {
    TransportError::new(kind(&e), e)
}

impl ReqwestTransport {
    fn request(&self, req: HttpRequest) -> reqwest::RequestBuilder {
        let method = match req.method {
            Method::Get => reqwest::Method::GET,
            Method::Post => reqwest::Method::POST,
        };
        let mut b = self.http.request(method, req.url);
        for (k, v) in req.headers {
            b = b.header(k, v);
        }
        if !req.body.is_empty() {
            b = b.body(req.body);
        }
        b
    }
}

fn head(resp: &reqwest::Response) -> (u16, Vec<(String, String)>) {
    let headers = resp
        .headers()
        .iter()
        .filter_map(|(k, v)| Some((k.as_str().to_string(), v.to_str().ok()?.to_string())))
        .collect();
    (resp.status().as_u16(), headers)
}

impl Transport for ReqwestTransport {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        let fut = self.request(req).send();
        Box::pin(async move {
            let resp = fut.await.map_err(error)?;
            let (status, headers) = head(&resp);
            let body = resp.bytes().await.map_err(error)?.into();
            Ok(HttpResponse::new(status, headers, body))
        })
    }

    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        Box::pin(tokio::time::sleep(d))
    }

    fn download<'a>(
        &'a self,
        req: HttpRequest,
        sink: &'a mut (dyn std::io::Write + Send),
    ) -> BoxFuture<'a, Result<ResponseHead, TransportError>> {
        let fut = self.request(req).send();
        Box::pin(async move {
            let mut resp = fut.await.map_err(error)?;
            let (status, headers) = head(&resp);
            if (200..300).contains(&status) {
                while let Some(chunk) = resp.chunk().await.map_err(error)? {
                    sink.write_all(&chunk)
                        .map_err(|e| TransportError::new(TransportErrorKind::Io, e))?;
                }
            }
            Ok(ResponseHead::new(status, headers))
        })
    }
}
