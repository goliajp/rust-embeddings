//! A loaded model: tokenize, run texts in chunks of similar length, pool,
//! normalise.

use std::path::Path;
use std::sync::Mutex;

use super::forward::{Buffers, forward};
use super::model::ModelInfo;
use super::pool;
use super::weights::Weights;
use super::wordpiece;
use crate::error::{Error, Result};

// texts are sorted by token count and run in chunks of about this many
// tokens; a chunk has no padding, so its size only trades matrix-product
// efficiency against activation memory
const CHUNK_TOKENS: usize = 4096;

// below this many tokens in a call, one thread does all of it: on macOS the
// matrix products run in Accelerate either way and the rest is too small
// to pay for starting threads
#[cfg(target_os = "macos")]
const POOL_MIN_TOKENS: usize = 48;
#[cfg(not(target_os = "macos"))]
const POOL_MIN_TOKENS: usize = 1;

pub(crate) struct Engine {
    weights: Weights,
    info: &'static ModelInfo,
    buffers: Mutex<Buffers>,
}

impl Engine {
    /// Map the weights file and lay out the model.
    pub(crate) fn load(info: &'static ModelInfo, weights: &Path) -> Result<Self> {
        Ok(Self {
            weights: Weights::load(weights, &info.shape())?,
            info,
            buffers: Mutex::new(Buffers::new()),
        })
    }

    /// Token ids of every text, cut at the model's limit, or `InputTooLong`
    /// for the first text over it when `refuse_long`.
    fn tokenize(
        &self,
        texts: &[String],
        prefix: Option<&str>,
        refuse_long: bool,
    ) -> Result<Vec<Vec<u32>>> {
        let wp = wordpiece::get();
        let max = self.info.max_tokens;
        let mut joined = String::new();
        let mut all = Vec::with_capacity(texts.len());
        for (index, text) in texts.iter().enumerate() {
            let text = match prefix {
                Some(p) => {
                    joined.clear();
                    joined.push_str(p);
                    joined.push_str(text);
                    joined.as_str()
                }
                None => text.as_str(),
            };
            let mut ids = Vec::new();
            if refuse_long {
                wp.encode(text, usize::MAX, &mut ids);
                if ids.len() > max {
                    return Err(Error::InputTooLong {
                        index,
                        tokens: u32::try_from(ids.len()).unwrap_or(u32::MAX),
                        max: max as u32,
                    });
                }
            } else {
                wp.encode(text, max, &mut ids);
            }
            all.push(ids);
        }
        Ok(all)
    }

    /// The vectors, row-major in input order, and the tokens the model read.
    pub(crate) fn encode(
        &self,
        texts: &[String],
        prefix: Option<&str>,
        refuse_long: bool,
        threads: Option<usize>,
    ) -> Result<(Vec<f32>, u64)> {
        let ids = self.tokenize(texts, prefix, refuse_long)?;
        let total: usize = ids.iter().map(Vec::len).sum();
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by_key(|&i| ids[i].len());
        let dim = self.info.dim;
        let mut out = vec![0f32; ids.len() * dim];
        let mut chunks = Vec::new();
        let (mut start, mut tokens) = (0, 0);
        for (k, &i) in order.iter().enumerate() {
            if tokens > 0 && tokens + ids[i].len() > CHUNK_TOKENS {
                chunks.push(start..k);
                (start, tokens) = (k, 0);
            }
            tokens += ids[i].len();
        }
        chunks.push(start..order.len());
        let threads = if total >= POOL_MIN_TOKENS {
            threads.unwrap_or_else(pool::threads)
        } else {
            1
        };
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let mut rows = Vec::new();
        pool::scoped(threads, |par| {
            for chunk in chunks {
                let group = &order[chunk];
                let batch: Vec<&[u32]> = group.iter().map(|&i| ids[i].as_slice()).collect();
                rows.resize(group.len() * dim, 0.0);
                forward(
                    &self.weights,
                    &batch,
                    self.info.pooling,
                    &mut rows,
                    &mut buffers,
                    par,
                );
                for (k, &i) in group.iter().enumerate() {
                    out[i * dim..(i + 1) * dim].copy_from_slice(&rows[k * dim..(k + 1) * dim]);
                }
            }
        });
        Ok((out, total as u64))
    }
}
