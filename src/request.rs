//! [`EmbedBuilder`]: one call, with retries, fallbacks and a timeout.

use std::future::IntoFuture;
use std::time::Duration;

use crate::client::{Backend, Client};
use crate::endpoint::Wire;
use crate::error::{Error, Result};
use crate::input::{InputType, Inputs, Truncate};
use crate::result::EmbedResult;
use crate::retry::Backoff;
use crate::transport::{BoxFuture, HttpRequest, HttpResponse};
use crate::usage::Usage;
use crate::wire::{self, Call, Parsed};

/// Per-call settings; `None` means "the client's default".
#[derive(Clone, Default)]
pub(crate) struct Overrides {
    pub(crate) model: Option<String>,
    pub(crate) dimensions: Option<u32>,
    pub(crate) input_type: Option<InputType>,
    pub(crate) truncate: Option<Truncate>,
    pub(crate) backoff: Option<Backoff>,
    pub(crate) timeout: Option<Duration>,
}

macro_rules! setters {
    () => {
        /// Model for this call (the primary client only; each fallback uses its own).
        pub fn model(mut self, model: impl Into<String>) -> Self {
            self.o.model = Some(model.into());
            self
        }

        /// Output width (the primary client only; each fallback uses its own).
        pub fn dimensions(mut self, dimensions: u32) -> Self {
            self.o.dimensions = Some(dimensions);
            self
        }

        /// Input type.
        pub fn input_type(mut self, input_type: InputType) -> Self {
            self.o.input_type = Some(input_type);
            self
        }

        /// Handling of inputs longer than the model reads.
        pub fn truncate(mut self, truncate: Truncate) -> Self {
            self.o.truncate = Some(truncate);
            self
        }

        /// Retry transient failures with this backoff.
        pub fn retry(mut self, backoff: Backoff) -> Self {
            self.o.backoff = Some(backoff);
            self
        }

        /// Time limit, retries and fallbacks included.
        pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
            self.o.timeout = Some(timeout);
            self
        }
    };
}
pub(crate) use setters;

/// One embedding call. Created by [`Client::embed`]; `.await` it.
#[must_use = "an embedding call does nothing until awaited"]
pub struct EmbedBuilder<'c, 't> {
    client: &'c Client,
    inputs: Inputs<'t>,
    o: Overrides,
}

/// What one client in the chain is asked for.
pub(crate) struct Plan {
    pub(crate) model: String,
    pub(crate) dimensions: Option<u32>,
    pub(crate) input_type: Option<InputType>,
    pub(crate) truncate: Truncate,
    pub(crate) backoff: Option<Backoff>,
}

pub(crate) fn plan(client: &Client, o: &Overrides, primary: bool) -> Plan {
    let d = &client.defaults;
    let (model, dimensions) = if primary {
        (
            o.model.clone().or_else(|| d.model.clone()),
            o.dimensions.or(d.dimensions),
        )
    } else {
        (d.model.clone(), d.dimensions)
    };
    Plan {
        model: model.unwrap_or_else(|| client.default_model().to_string()),
        dimensions,
        input_type: o.input_type.or(d.input_type),
        truncate: o.truncate.or(d.truncate).unwrap_or_default(),
        backoff: o.backoff.or(d.backoff),
    }
}

impl<'c, 't> EmbedBuilder<'c, 't> {
    pub(crate) fn new(client: &'c Client, inputs: Inputs<'t>) -> Self {
        Self {
            client,
            inputs,
            o: Overrides::default(),
        }
    }

    setters!();

    /// The HTTP request this call would send, for sending it yourself; see
    /// [`Client::prepare`].
    pub fn prepare(self) -> Result<Prepared> {
        let p = plan(self.client, &self.o, true);
        let ep = match &self.client.backend {
            Backend::Cloud(ep) => ep,
            #[cfg(feature = "local")]
            Backend::Local(_) => {
                return Err(Error::unsupported(
                    &p.model,
                    "prepare (local models run in-process)",
                ));
            }
        };
        check_count(self.inputs.len(), ep.limits.max_inputs)?;
        let request = wire::build(ep, &call(&p, &self.inputs))?;
        Ok(Prepared {
            request,
            wire: ep.wire,
            n: self.inputs.len(),
            model: p.model,
        })
    }
}

impl<'c, 't: 'c> IntoFuture for EmbedBuilder<'c, 't> {
    type Output = Result<EmbedResult>;
    type IntoFuture = BoxFuture<'c, Result<EmbedResult>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { run(self.client, &self.inputs, &self.o).await })
    }
}

/// A built request waiting for its response. See [`Client::prepare`].
#[derive(Debug)]
pub struct Prepared {
    request: HttpRequest,
    wire: Wire,
    n: usize,
    model: String,
}

impl Prepared {
    /// The request to send.
    pub fn request(&self) -> &HttpRequest {
        &self.request
    }

    /// Check the response's status and read its vectors.
    pub fn finish(self, response: HttpResponse) -> Result<EmbedResult> {
        let parsed = wire::parse(self.wire, self.n, &response)?;
        Ok(result(parsed, &self.model))
    }
}

fn call<'a>(p: &'a Plan, inputs: &'a Inputs<'a>) -> Call<'a> {
    Call {
        model: &p.model,
        inputs,
        dimensions: p.dimensions,
        input_type: p.input_type,
        truncate: p.truncate,
    }
}

fn check_count(count: usize, max: usize) -> Result<()> {
    if count > max {
        return Err(Error::TooManyInputs { count, max });
    }
    Ok(())
}

pub(crate) fn result(p: Parsed, requested: &str) -> EmbedResult {
    let usage = Usage::priced(requested, p.total_tokens);
    let model = p.model.unwrap_or_else(|| requested.to_string());
    EmbedResult::from_flat(p.data, p.dim, model, usage)
}

/// One call through the whole chain, under the timeout.
pub(crate) async fn run(
    client: &Client,
    inputs: &Inputs<'_>,
    o: &Overrides,
) -> Result<EmbedResult> {
    let timeout = o.timeout.unwrap_or(client.defaults.timeout);
    let work = chain(client, inputs, o);
    #[cfg(feature = "tracing")]
    let work = {
        use tracing::Instrument;
        let span = tracing::info_span!(
            "embedrs.embed",
            provider = client.provider(),
            model = plan(client, o, true).model.as_str(),
            texts = inputs.len(),
        );
        work.instrument(span)
    };
    crate::rt::race::race(work, client.sleep(timeout))
        .await
        .unwrap_or(Err(Error::Timeout(timeout)))
}

async fn chain(client: &Client, inputs: &Inputs<'_>, o: &Overrides) -> Result<EmbedResult> {
    let err = match attempts(client, &plan(client, o, true), inputs).await {
        Ok(r) => return Ok(r),
        Err(e) => e,
    };
    if !err.is_retryable() || client.fallbacks.is_empty() {
        return Err(err);
    }
    #[cfg(feature = "tracing")]
    tracing::info!(from_provider = client.provider(), error = %err, "trying fallbacks");
    for fb in &client.fallbacks {
        if let Ok(r) = attempts(fb, &plan(fb, o, false), inputs).await {
            return Ok(r);
        }
    }
    Err(err)
}

/// The first attempt and its retries against one client.
async fn attempts(client: &Client, p: &Plan, inputs: &Inputs<'_>) -> Result<EmbedResult> {
    let ep = match &client.backend {
        Backend::Cloud(ep) => ep,
        #[cfg(feature = "local")]
        Backend::Local(h) => return crate::local::embed(h, p, inputs).await,
    };
    check_count(inputs.len(), ep.limits.max_inputs)?;
    let req = wire::build(ep, &call(p, inputs))?;
    let transport = client
        .transport
        .as_ref()
        .expect("cloud clients are built with a transport");
    let retries = p.backoff.map_or(0, |b| b.max_retries());
    let mut attempt = 0;
    let mut req = Some(req);
    loop {
        let this = if attempt < retries {
            req.clone()
        } else {
            req.take()
        };
        let this = this.expect("one request per attempt");
        let outcome = match transport.send(this).await {
            Ok(resp) => wire::parse(ep.wire, inputs.len(), &resp),
            Err(e) => Err(e.into()),
        };
        let err = match outcome {
            Ok(parsed) => {
                #[cfg(feature = "tracing")]
                tracing::info!(total_tokens = parsed.total_tokens, "embedding succeeded");
                return Ok(result(parsed, &p.model));
            }
            Err(e) => e,
        };
        let Some(backoff) = p
            .backoff
            .filter(|_| err.is_retryable() && attempt < retries)
        else {
            return Err(err);
        };
        let hint = match &err {
            Error::Api { retry_after, .. } => *retry_after,
            _ => None,
        };
        let delay = hint.unwrap_or_else(|| backoff.delay(attempt));
        #[cfg(feature = "tracing")]
        tracing::warn!(attempt, delay_ms = delay.as_millis() as u64, error = %err, "retrying");
        client.sleep(delay).await;
        attempt += 1;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::endpoint::Endpoint;
    use crate::rt::join::tests::block_on;
    use crate::transport::{Transport, TransportError, TransportErrorKind};

    pub(crate) type Reply = std::result::Result<HttpResponse, TransportErrorKind>;

    /// Answers from a script, records requests and sleeps; sleeping takes no time.
    pub(crate) struct Fake {
        pub(crate) replies: Mutex<VecDeque<Reply>>,
        pub(crate) sent: Mutex<Vec<HttpRequest>>,
        pub(crate) slept: Mutex<Vec<Duration>>,
    }

    impl Transport for Fake {
        fn send(
            &self,
            req: HttpRequest,
        ) -> BoxFuture<'_, std::result::Result<HttpResponse, TransportError>> {
            self.sent.lock().unwrap().push(req);
            let r = self
                .replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("a scripted reply");
            Box::pin(std::future::ready(r.map_err(TransportError::from_kind)))
        }
        fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
            self.slept.lock().unwrap().push(d);
            if d >= Duration::from_secs(1000) {
                // stands for the call timeout: never fires
                return Box::pin(std::future::pending());
            }
            Box::pin(std::future::ready(()))
        }
    }

    pub(crate) fn fake_full(replies: Vec<Reply>) -> Arc<Fake> {
        Arc::new(Fake {
            replies: Mutex::new(replies.into()),
            sent: Mutex::new(vec![]),
            slept: Mutex::new(vec![]),
        })
    }

    pub(crate) fn fake(replies: Vec<Reply>) -> Arc<dyn Transport> {
        fake_full(replies)
    }

    pub(crate) fn ok(body: &str) -> Reply {
        Ok(HttpResponse::new(200, vec![], body.as_bytes().to_vec()))
    }

    pub(crate) fn status(code: u16, headers: &[(&str, &str)]) -> Reply {
        let h = headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Ok(HttpResponse::new(
            code,
            h,
            br#"{"error":{"message":"nope"}}"#.to_vec(),
        ))
    }

    const ONE: &str =
        r#"{"data":[{"index":0,"embedding":[0.5,1.5]}],"model":"m","usage":{"total_tokens":3}}"#;

    fn client(t: &Arc<Fake>) -> Client {
        Client::new(Endpoint::openai_compatible("k", "http://x"), t.clone())
            .with_timeout(Duration::from_secs(1000))
    }

    #[test]
    fn retries_follow_retry_after_then_backoff() {
        let t = fake_full(vec![
            status(429, &[("retry-after", "7")]),
            status(503, &[]),
            ok(ONE),
        ]);
        let c = client(&t).with_retry(Backoff::new().jitter(false));
        let r = block_on(c.embed(["a"]).into_future()).unwrap();
        assert_eq!(&r[0], &[0.5, 1.5]);
        assert_eq!(r.usage.total_tokens, Some(3));
        assert_eq!(t.sent.lock().unwrap().len(), 3);
        let slept = t.slept.lock().unwrap();
        // the timeout timer first, then the two retry delays
        assert_eq!(
            slept[1..],
            [Duration::from_secs(7), Duration::from_millis(1000)]
        );
    }

    #[test]
    fn no_backoff_means_no_retry_and_client_errors_are_not_retried() {
        let t = fake_full(vec![status(503, &[])]);
        assert!(block_on(client(&t).embed(["a"]).into_future()).is_err());
        let t = fake_full(vec![status(400, &[])]);
        let c = client(&t).with_retry(Backoff::new());
        let e = block_on(c.embed(["a"]).into_future()).unwrap_err();
        assert!(matches!(e, Error::Api { status: 400, .. }));
        assert_eq!(t.sent.lock().unwrap().len(), 1);
    }

    #[test]
    fn retries_stop_after_the_limit() {
        let t = fake_full(vec![Err(TransportErrorKind::Connect); 3]);
        let c = client(&t).with_retry(Backoff::new().retries(2));
        let e = block_on(c.embed(["a"]).into_future()).unwrap_err();
        assert!(matches!(e, Error::Transport(_)));
        assert_eq!(t.sent.lock().unwrap().len(), 3);
    }

    #[test]
    fn fallback_only_on_transient_errors() {
        let primary = fake_full(vec![status(500, &[]), status(401, &[])]);
        let backup = fake_full(vec![ok(ONE)]);
        let c = client(&primary).with_fallback(client(&backup));
        let r = block_on(c.embed(["a"]).into_future()).unwrap();
        assert_eq!(r.model, "m");
        let e = block_on(c.embed(["a"]).into_future()).unwrap_err();
        assert!(matches!(e, Error::Api { status: 401, .. }));
        assert_eq!(backup.sent.lock().unwrap().len(), 1);
    }

    #[test]
    fn fallback_failure_returns_the_primary_error() {
        let primary = fake_full(vec![status(502, &[])]);
        let backup = fake_full(vec![status(500, &[])]);
        let c = client(&primary).with_fallback(client(&backup));
        let e = block_on(c.embed(["a"]).into_future()).unwrap_err();
        assert!(matches!(e, Error::Api { status: 502, .. }));
    }

    #[test]
    fn timeout() {
        struct Never;
        impl Transport for Never {
            fn send(
                &self,
                _: HttpRequest,
            ) -> BoxFuture<'_, std::result::Result<HttpResponse, TransportError>> {
                Box::pin(std::future::pending())
            }
            fn sleep(&self, _: Duration) -> BoxFuture<'static, ()> {
                Box::pin(std::future::ready(()))
            }
        }
        let c = Client::new(Endpoint::openai("k"), Arc::new(Never))
            .with_timeout(Duration::from_secs(2));
        let e = block_on(c.embed(["a"]).into_future()).unwrap_err();
        assert!(matches!(e, Error::Timeout(d) if d == Duration::from_secs(2)));
    }

    #[test]
    fn too_many_inputs_are_refused_before_sending() {
        let t = fake_full(vec![]);
        let texts = vec!["x"; 97];
        let c =
            Client::new(Endpoint::cohere("k"), t.clone()).with_timeout(Duration::from_secs(1000));
        let e = block_on(c.embed(&texts).into_future()).unwrap_err();
        assert!(matches!(e, Error::TooManyInputs { count: 97, max: 96 }));
        assert!(t.sent.lock().unwrap().is_empty());
    }

    #[test]
    fn plans_for_primary_and_fallback() {
        let fb = Client::new(Endpoint::cohere("k"), fake(vec![]));
        let primary = Client::new(Endpoint::openai("k"), fake(vec![]))
            .with_model("text-embedding-3-large")
            .with_dimensions(256)
            .with_fallback(fb.clone());
        let o = Overrides {
            model: Some("m2".into()),
            input_type: Some(InputType::SearchQuery),
            ..Default::default()
        };
        let p = plan(&primary, &o, true);
        assert_eq!((p.model.as_str(), p.dimensions), ("m2", Some(256)));
        let p = plan(&fb, &o, false);
        assert_eq!((p.model.as_str(), p.dimensions), ("embed-v4.0", None));
        assert_eq!(p.input_type, Some(InputType::SearchQuery));
        assert_eq!(p.truncate, Truncate::Provider);
    }
}
