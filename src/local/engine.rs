//! A loaded model: tokenize, run texts in groups of similar length, pool,
//! normalise.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::bert::Bert;
use super::fail;
use super::model::{ModelInfo, Pooling};
use super::wordpiece;
use crate::error::{Error, Result};

// texts are sorted by token count and run in groups of this many, so each
// group is padded only to its own longest text
const GROUP: usize = 64;

// up to this many positions (texts x padded length) the softmax and
// layer-norm kernels are faster on one thread than handed to rayon's pool:
// measured with accelerate on an m4 pro, where the matrix products do not
// use rayon; elsewhere they do, and one thread would slow them down
#[cfg(target_os = "macos")]
const SERIAL_POSITIONS: usize = 224;

pub(crate) struct Engine {
    model: Bert,
    info: &'static ModelInfo,
    #[cfg(target_os = "macos")]
    serial: rayon::ThreadPool,
}

fn tensor_err(what: &'static str) -> impl Fn(candle_core::Error) -> Error {
    move |e| fail(format!("{what}: {e}"))
}

impl Engine {
    /// Map the weights file and build the model.
    pub(crate) fn load(info: &'static ModelInfo, weights: &Path) -> Result<Self> {
        let device = Device::Cpu;
        // safety: the file is a read-only blob in the model cache (or a file
        // the caller handed over); it is not modified while mapped
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device) }
            .map_err(tensor_err("cannot load the weights"))?;
        let model = Bert::load(vb, &info.shape()).map_err(tensor_err("cannot build the model"))?;
        #[cfg(target_os = "macos")]
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(move |_| format!("embedrs-{}-short", info.name))
            .build()
            .map_err(|e| fail(format!("cannot start the short-text thread: {e}")))?;
        Ok(Self {
            model,
            info,
            #[cfg(target_os = "macos")]
            serial,
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

    /// The vectors, row-major in input order, and the tokens the model read
    /// (padding excluded).
    pub(crate) fn encode(
        &self,
        texts: &[String],
        prefix: Option<&str>,
        refuse_long: bool,
    ) -> Result<(Vec<f32>, u64)> {
        let ids = self.tokenize(texts, prefix, refuse_long)?;
        let total = ids.iter().map(|v| v.len() as u64).sum();
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by_key(|&i| ids[i].len());
        let dim = self.info.dim;
        let mut out = vec![0f32; ids.len() * dim];
        for group in order.chunks(GROUP) {
            let batch: Vec<&[u32]> = group.iter().map(|&i| ids[i].as_slice()).collect();
            let rows = self.run(&batch)?;
            for (k, &i) in group.iter().enumerate() {
                out[i * dim..(i + 1) * dim].copy_from_slice(&rows[k * dim..(k + 1) * dim]);
            }
        }
        Ok((out, total))
    }

    fn run(&self, batch: &[&[u32]]) -> Result<Vec<f32>> {
        #[cfg(target_os = "macos")]
        {
            let longest = batch.iter().map(|e| e.len()).max().unwrap_or(0);
            if batch.len() * longest <= SERIAL_POSITIONS {
                return self.serial.install(|| self.forward(batch));
            }
        }
        self.forward(batch)
    }

    fn forward(&self, batch: &[&[u32]]) -> Result<Vec<f32>> {
        let device = Device::Cpu;
        let len = batch.iter().map(|e| e.len()).max().unwrap_or(0);
        let mut ids = vec![0u32; batch.len() * len];
        let mut mask = vec![0u32; batch.len() * len];
        for (k, e) in batch.iter().enumerate() {
            ids[k * len..k * len + e.len()].copy_from_slice(e);
            mask[k * len..k * len + e.len()].fill(1);
        }
        let padded = batch.iter().any(|e| e.len() < len);
        let shape = (batch.len(), len);
        let ids =
            Tensor::from_vec(ids, shape, &device).map_err(tensor_err("tensor creation failed"))?;
        let mask =
            Tensor::from_vec(mask, shape, &device).map_err(tensor_err("tensor creation failed"))?;
        let states = self
            .model
            .forward(&ids, padded.then_some(&mask))
            .map_err(tensor_err("forward pass failed"))?;
        let pooled = match self.info.pooling {
            Pooling::Mean => mean_pool(&states, &mask),
            Pooling::Cls => states.narrow(1, 0, 1).and_then(|t| t.squeeze(1)),
        }
        .map_err(tensor_err("pooling failed"))?;
        l2_normalize(&pooled)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(tensor_err("normalization failed"))
    }
}

fn mean_pool(states: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
    let mask = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let summed = states.broadcast_mul(&mask)?.sum(1)?;
    let counts = mask.sum(1)?;
    summed.broadcast_div(&counts)
}

fn l2_normalize(t: &Tensor) -> candle_core::Result<Tensor> {
    let norms = t.sqr()?.sum_keepdim(1)?.sqrt()?;
    t.broadcast_div(&norms)
}
