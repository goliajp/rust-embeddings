//! [`BatchBuilder`]: any number of texts, split into requests and sent
//! with bounded concurrency.

use std::future::IntoFuture;

use crate::client::{Backend, Client};
use crate::error::{DecodeError, Error, Result};
use crate::input::{InputType, Inputs, Truncate};
use crate::plan;
use crate::request::{Overrides, run, setters};
use crate::result::EmbedResult;
use crate::retry::Backoff;
use crate::transport::BoxFuture;
use crate::usage::Usage;

const MAX_CONCURRENCY: usize = 64;
#[cfg(feature = "local")]
const LOCAL_CHUNK: usize = 256;

/// A batch call. Created by [`Client::embed_batch`]; `.await` it.
///
/// The input is cut into consecutive requests within the endpoint's
/// [`Limits`](crate::Limits): inputs per request, and tokens per request
/// where the provider has a cap (counted as UTF-8 bytes, which is never
/// fewer than the tokens of a byte-level BPE; exact OpenAI token counts with
/// the `token-count` feature). Each request gets the timeout, retries and
/// fallbacks of a single call.
#[must_use = "an embedding call does nothing until awaited"]
pub struct BatchBuilder<'c, 't> {
    client: &'c Client,
    inputs: Inputs<'t>,
    o: Overrides,
    concurrency: usize,
    chunk_size: Option<usize>,
}

impl<'c, 't> BatchBuilder<'c, 't> {
    pub(crate) fn new(client: &'c Client, inputs: Inputs<'t>) -> Self {
        Self {
            client,
            inputs,
            o: Overrides::default(),
            concurrency: 5,
            chunk_size: None,
        }
    }

    setters!();

    /// Requests in flight at once. Default 5, at most 64.
    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.clamp(1, MAX_CONCURRENCY);
        self
    }

    /// At most this many inputs per request (default: the endpoint's limit).
    pub fn chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = Some(n.max(1));
        self
    }

    fn ranges(&self) -> Vec<std::ops::Range<usize>> {
        match &self.client.backend {
            Backend::Cloud(ep) => {
                let max = self.chunk_size.unwrap_or(ep.limits.max_inputs);
                let wire = ep.wire;
                let cap = ep.limits.max_request_tokens.map(u64::from);
                plan::split(&self.inputs, max, cap, |t| plan::tokens(wire, t))
            }
            #[cfg(feature = "local")]
            Backend::Local(_) => {
                let max = self.chunk_size.unwrap_or(LOCAL_CHUNK);
                plan::split(&self.inputs, max, None, |_| 0)
            }
        }
    }

    async fn execute(self) -> Result<EmbedResult> {
        if self.inputs.is_empty() {
            let model = crate::request::plan(self.client, &self.o, true).model;
            return Ok(EmbedResult::from_flat(
                Vec::new(),
                0,
                model,
                Usage::new(Some(0), None),
            ));
        }
        let parts: Vec<(usize, Inputs<'_>)> = self
            .ranges()
            .into_iter()
            .map(|r| (r.start, self.inputs.slice(r)))
            .collect();
        let (client, o) = (self.client, &self.o);
        let futs: Vec<BoxFuture<'_, Result<EmbedResult>>> = parts
            .iter()
            .map(|(start, part)| {
                Box::pin(async move {
                    run(client, part, o).await.map_err(|e| match e {
                        Error::InputTooLong { index, tokens, max } => Error::InputTooLong {
                            index: start + index,
                            tokens,
                            max,
                        },
                        Error::EmptyInput { index } => Error::EmptyInput {
                            index: start + index,
                        },
                        e => e,
                    })
                }) as BoxFuture<'_, _>
            })
            .collect();
        let results = crate::rt::join::ordered(futs, self.concurrency).await?;
        merge(results, client.provider())
    }
}

fn merge(mut results: Vec<EmbedResult>, provider: &'static str) -> Result<EmbedResult> {
    if results.len() == 1 {
        return Ok(results.pop().expect("one result"));
    }
    let dim = results[0].dim();
    let total = results.iter().map(|r| r.as_flat().len()).sum();
    let mut data = Vec::with_capacity(total);
    let mut usage: Option<Usage> = None;
    let model = results[0].model.clone();
    for r in results {
        if r.dim() != dim && !r.is_empty() {
            return Err(DecodeError::new(
                provider,
                0,
                format!(
                    "parts of one batch came back with widths {dim} and {} (a fallback served part of it)",
                    r.dim()
                ),
            )
            .into());
        }
        usage = Some(match usage {
            None => r.usage.clone(),
            Some(u) => u.add(&r.usage),
        });
        data.extend_from_slice(r.as_flat());
    }
    Ok(EmbedResult::from_flat(
        data,
        dim,
        model,
        usage.unwrap_or_default(),
    ))
}

impl<'c, 't: 'c> IntoFuture for BatchBuilder<'c, 't> {
    type Output = Result<EmbedResult>;
    type IntoFuture = BoxFuture<'c, Result<EmbedResult>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::endpoint::{Endpoint, Limits};
    use crate::request::tests::{fake_full, ok, status};
    use crate::rt::join::tests::block_on;

    fn body(values: &[f32], tokens: Option<u64>) -> String {
        let data: Vec<String> = values
            .iter()
            .enumerate()
            .map(|(i, v)| format!(r#"{{"index":{i},"embedding":[{v},{v}]}}"#))
            .collect();
        let usage = tokens.map_or("null".into(), |t| format!(r#"{{"total_tokens":{t}}}"#));
        format!(
            r#"{{"data":[{}],"model":"m","usage":{usage}}}"#,
            data.join(",")
        )
    }

    #[test]
    fn splits_by_count_and_keeps_order() {
        let t = fake_full(vec![
            ok(&body(&[1.0, 2.0], Some(2))),
            ok(&body(&[3.0], Some(1))),
        ]);
        let c = Client::new(Endpoint::openai_compatible("k", "http://x"), t.clone())
            .with_timeout(Duration::from_secs(1000));
        let r = block_on(
            c.embed_batch(["a", "b", "c"])
                .chunk_size(2)
                .concurrency(1)
                .into_future(),
        )
        .unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(&r[2], &[3.0, 3.0]);
        assert_eq!(r.usage.total_tokens, Some(3));
        let sent = t.sent.lock().unwrap();
        assert_eq!(sent.len(), 2);
        assert!(String::from_utf8_lossy(&sent[1].body).contains(r#""input":["c"]"#));
    }

    #[test]
    fn splits_by_token_cap() {
        let t = fake_full(vec![ok(&body(&[1.0], None)), ok(&body(&[2.0], Some(9)))]);
        let ep =
            Endpoint::openai_compatible("k", "http://x").limits(Limits::new(100).request_tokens(1));
        let c = Client::new(ep, t.clone()).with_timeout(Duration::from_secs(1000));
        let r = block_on(c.embed_batch(["aaaaa", "bbbbb"]).into_future()).unwrap();
        assert_eq!(r.len(), 2);
        // one part without a count makes the total unknown
        assert_eq!(r.usage.total_tokens, None);
        assert_eq!(t.sent.lock().unwrap().len(), 2);
    }

    #[test]
    fn first_error_fails_the_batch() {
        let t = fake_full(vec![ok(&body(&[1.0], Some(1))), status(400, &[])]);
        let c = Client::new(Endpoint::openai_compatible("k", "http://x"), t)
            .with_timeout(Duration::from_secs(1000));
        let e = block_on(
            c.embed_batch(["a", "b"])
                .chunk_size(1)
                .concurrency(1)
                .into_future(),
        );
        assert!(e.is_err());
    }

    #[test]
    fn empty_batch_sends_nothing() {
        let t = fake_full(vec![]);
        let c = Client::new(Endpoint::voyage("k"), t.clone());
        let empty: Vec<String> = vec![];
        let r = block_on(c.embed_batch(empty).into_future()).unwrap();
        assert!(r.is_empty());
        assert_eq!(r.model, "voyage-4-large");
        assert_eq!(r.usage.total_tokens, Some(0));
        assert!(t.sent.lock().unwrap().is_empty());
    }

    #[test]
    fn mixed_widths_are_an_error() {
        let a = EmbedResult::from_flat(vec![1.0, 2.0], 2, "a", Usage::default());
        let b = EmbedResult::from_flat(vec![1.0], 1, "b", Usage::default());
        assert!(merge(vec![a, b], "openai").is_err());
    }

    #[test]
    fn concurrency_is_clamped() {
        let c = Client::new(Endpoint::openai("k"), fake_full(vec![]));
        assert_eq!(c.embed_batch(["a"]).concurrency(0).concurrency, 1);
        assert_eq!(c.embed_batch(["a"]).concurrency(1000).concurrency, 64);
        assert_eq!(c.embed_batch(["a"]).chunk_size(0).chunk_size, Some(1));
    }
}
