//! [`Client`]: one provider (or local model), its defaults and fallbacks.

use std::sync::Arc;
use std::time::Duration;

use crate::batch::BatchBuilder;
use crate::endpoint::Endpoint;
use crate::error::Result;
use crate::input::{InputType, Inputs, Truncate};
use crate::request::{EmbedBuilder, Prepared};
use crate::result::EmbedResult;
use crate::retry::Backoff;
use crate::transport::{BoxFuture, Transport};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub(crate) enum Backend {
    Cloud(Endpoint),
    #[cfg(feature = "local")]
    Local(crate::local::Handle),
}

impl Backend {
    pub(crate) fn endpoint(&self) -> Option<&Endpoint> {
        match self {
            Backend::Cloud(ep) => Some(ep),
            #[cfg(feature = "local")]
            Backend::Local(_) => None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Defaults {
    pub(crate) model: Option<String>,
    pub(crate) dimensions: Option<u32>,
    pub(crate) input_type: Option<InputType>,
    pub(crate) truncate: Option<Truncate>,
    pub(crate) backoff: Option<Backoff>,
    pub(crate) timeout: Duration,
}

/// An embedding client for one provider or local model, with optional
/// fallbacks. Cheap to clone.
///
/// ```rust,no_run
/// # async fn run() -> embedrs::Result<()> {
/// let client = embedrs::Client::openai("sk-...");
/// let result = client.embed(["hello world"]).await?;
/// println!("dimensions: {}", result.dim());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Client {
    pub(crate) backend: Backend,
    pub(crate) transport: Option<Arc<dyn Transport>>,
    pub(crate) defaults: Defaults,
    pub(crate) fallbacks: Vec<Client>,
}

#[cfg(feature = "reqwest")]
macro_rules! constructors {
    ($($name:ident, $compat:ident, $doc:literal;)*) => {$(
        #[doc = concat!("A client for the ", $doc, " API over [`ReqwestTransport`](crate::transport::ReqwestTransport) (feature `reqwest`).")]
        pub fn $name(key: impl Into<String>) -> Self {
            Self::new(Endpoint::$name(key), crate::transport::ReqwestTransport::shared())
        }

        #[doc = concat!("A client for a server that speaks the ", $doc, " API at `base_url` (feature `reqwest`).")]
        pub fn $compat(key: impl Into<String>, base_url: impl Into<String>) -> Self {
            Self::new(Endpoint::$compat(key, base_url), crate::transport::ReqwestTransport::shared())
        }
    )*};
}

impl Client {
    /// A client for `endpoint` that sends requests through `transport`.
    pub fn new(endpoint: Endpoint, transport: Arc<dyn Transport>) -> Self {
        Self::with_backend(Backend::Cloud(endpoint), Some(transport))
    }

    pub(crate) fn with_backend(backend: Backend, transport: Option<Arc<dyn Transport>>) -> Self {
        Self {
            backend,
            transport,
            defaults: Defaults {
                model: None,
                dimensions: None,
                input_type: None,
                truncate: None,
                backoff: None,
                timeout: DEFAULT_TIMEOUT,
            },
            fallbacks: Vec::new(),
        }
    }

    #[cfg(feature = "reqwest")]
    constructors! {
        openai, openai_compatible, "OpenAI embeddings";
        cohere, cohere_compatible, "Cohere v2 embed";
        gemini, gemini_compatible, "Gemini embedContent";
        voyage, voyage_compatible, "Voyage AI embeddings";
        jina, jina_compatible, "Jina AI embeddings";
        mistral, mistral_compatible, "Mistral embeddings";
    }

    /// Default model for every call.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.defaults.model = Some(model.into());
        self
    }

    /// Default output width, for models that can shorten their vectors.
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.defaults.dimensions = Some(dimensions);
        self
    }

    /// Default input type.
    pub fn with_input_type(mut self, input_type: InputType) -> Self {
        self.defaults.input_type = Some(input_type);
        self
    }

    /// Default handling of inputs longer than the model reads.
    pub fn with_truncate(mut self, truncate: Truncate) -> Self {
        self.defaults.truncate = Some(truncate);
        self
    }

    /// Retry transient failures (408, 429, 500, 502, 503, 504, 529,
    /// connection errors) with this backoff. Without it nothing is retried.
    pub fn with_retry(mut self, backoff: Backoff) -> Self {
        self.defaults.backoff = Some(backoff);
        self
    }

    /// Time limit of one call, retries and fallbacks included. Default 60 s.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.defaults.timeout = timeout;
        self
    }

    /// Chain a fallback. When this client fails with a transient error
    /// (after its retries), the call goes to the fallbacks in order; the
    /// first error is returned if all fail. Client errors (400, 401, 403,
    /// 422, unsupported options) are returned at once.
    ///
    /// A fallback uses its own model and dimensions, never this client's;
    /// the input type is shared.
    pub fn with_fallback(mut self, fallback: Client) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    /// Send requests through `transport`, here and in every fallback
    /// registered so far.
    pub fn with_transport(mut self, transport: Arc<dyn Transport>) -> Self {
        for fb in &mut self.fallbacks {
            fb.transport = Some(transport.clone());
        }
        self.transport = Some(transport);
        self
    }

    /// Embed the texts in one request (one model run for local models).
    ///
    /// Takes `Vec<String>`, `&[&str]`, `&[String]`, `[&str; N]` and `&str`.
    pub fn embed<'t>(&self, texts: impl Into<Inputs<'t>>) -> EmbedBuilder<'_, 't> {
        EmbedBuilder::new(self, texts.into())
    }

    /// Embed any number of texts: split into requests the endpoint accepts
    /// (by count, and by tokens where the endpoint has a cap) and sent
    /// concurrently.
    pub fn embed_batch<'t>(&self, texts: impl Into<Inputs<'t>>) -> BatchBuilder<'_, 't> {
        BatchBuilder::new(self, texts.into())
    }

    /// Build the HTTP request for `texts` without sending it, for code that
    /// brings its own HTTP client: blocking code, other runtimes, wasm.
    /// Send [`Prepared::request`] and pass the answer to [`Prepared::finish`].
    /// No retries, no fallbacks, no timeout.
    ///
    /// ```rust
    /// # fn run() -> embedrs::Result<()> {
    /// use std::sync::Arc;
    /// # struct Unused;
    /// # impl embedrs::transport::Transport for Unused {
    /// #   fn send(&self, _: embedrs::transport::HttpRequest) -> embedrs::transport::BoxFuture<'_, Result<embedrs::transport::HttpResponse, embedrs::transport::TransportError>> { unimplemented!() }
    /// #   fn sleep(&self, _: std::time::Duration) -> embedrs::transport::BoxFuture<'static, ()> { unimplemented!() }
    /// # }
    /// let client = embedrs::Client::new(
    ///     embedrs::Endpoint::openai_compatible("sk-...", "http://localhost:8080/v1"),
    ///     Arc::new(Unused),
    /// );
    /// let prepared = client.prepare(["hello"])?;
    /// assert_eq!(prepared.request().url, "http://localhost:8080/v1/embeddings");
    /// // send it with any HTTP client, then:
    /// let answer = embedrs::transport::HttpResponse::new(
    ///     200,
    ///     vec![],
    ///     br#"{"data":[{"index":0,"embedding":[0.5,0.5]}]}"#.to_vec(),
    /// );
    /// let result = prepared.finish(answer)?;
    /// assert_eq!(&result[0], &[0.5, 0.5]);
    /// # Ok(())
    /// # }
    /// # run().unwrap();
    /// ```
    pub fn prepare<'t>(&self, texts: impl Into<Inputs<'t>>) -> Result<Prepared> {
        self.embed(texts).prepare()
    }

    pub(crate) fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        match &self.transport {
            Some(t) => t.sleep(d),
            None => Box::pin(crate::rt::timer::sleep(d)),
        }
    }

    pub(crate) fn provider(&self) -> &'static str {
        match &self.backend {
            Backend::Cloud(ep) => ep.provider(),
            #[cfg(feature = "local")]
            Backend::Local(_) => "local",
        }
    }

    pub(crate) fn default_model(&self) -> &str {
        match &self.backend {
            Backend::Cloud(ep) => ep.default_model(),
            #[cfg(feature = "local")]
            Backend::Local(h) => h.name(),
        }
    }
}

/// Anything that turns texts into vectors: [`Client`], or your own backend.
///
/// This is the seam for crates built on embedrs (such as chunkers) and for
/// test doubles: take `&dyn Embed` and any backend plugs in.
pub trait Embed: Send + Sync {
    /// Embed `texts` in input order.
    fn embed_texts<'a>(
        &'a self,
        texts: &'a [&'a str],
        input_type: Option<InputType>,
    ) -> BoxFuture<'a, Result<EmbedResult>>;
}

impl Embed for Client {
    /// [`Client::embed_batch`] with the client's defaults.
    fn embed_texts<'a>(
        &'a self,
        texts: &'a [&'a str],
        input_type: Option<InputType>,
    ) -> BoxFuture<'a, Result<EmbedResult>> {
        let mut b = self.embed_batch(texts);
        if let Some(t) = input_type {
            b = b.input_type(t);
        }
        Box::pin(std::future::IntoFuture::into_future(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_is_object_safe() {
        fn take(_: &dyn Embed) {}
        let c = Client::new(Endpoint::openai("k"), crate::request::tests::fake(vec![]));
        take(&c);
    }

    #[test]
    fn defaults_and_fallbacks() {
        let t = crate::request::tests::fake(vec![]);
        let c = Client::new(Endpoint::cohere("k"), t.clone())
            .with_model("embed-v4.0")
            .with_dimensions(256)
            .with_input_type(InputType::SearchDocument)
            .with_truncate(Truncate::End)
            .with_retry(Backoff::new())
            .with_timeout(Duration::from_secs(5))
            .with_fallback(Client::new(Endpoint::voyage("k"), t.clone()));
        assert_eq!(c.defaults.model.as_deref(), Some("embed-v4.0"));
        assert_eq!(c.defaults.dimensions, Some(256));
        assert_eq!(c.defaults.truncate, Some(Truncate::End));
        assert_eq!(c.defaults.timeout, Duration::from_secs(5));
        assert_eq!(c.fallbacks.len(), 1);
        assert_eq!(c.provider(), "cohere");
        assert_eq!(c.fallbacks[0].default_model(), "voyage-4-large");
        let other = crate::request::tests::fake(vec![]);
        let c = c.with_transport(other.clone());
        assert!(Arc::ptr_eq(
            c.fallbacks[0].transport.as_ref().unwrap(),
            &other
        ));
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn named_constructors_share_one_transport() {
        let a = Client::openai("k");
        let b = Client::gemini_compatible("k", "http://x");
        assert!(Arc::ptr_eq(
            a.transport.as_ref().unwrap(),
            b.transport.as_ref().unwrap()
        ));
        assert_eq!(b.provider(), "gemini");
        assert_eq!(a.default_model(), "text-embedding-3-small");
    }
}
