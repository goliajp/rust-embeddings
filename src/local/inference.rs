use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::bert::Bert;
use super::download;
use super::model::{ModelDefinition, PoolingStrategy};
use super::wordpiece;
use crate::error::{Error, Result};

// texts are sorted by token count and run in groups of this many, so each
// group is padded only to its own longest text
const GROUP: usize = 64;

/// A loaded model.
pub(crate) struct InferenceEngine {
    model: Bert,
    max_seq_length: usize,
    pooling: PoolingStrategy,
}

impl InferenceEngine {
    /// Load the weights, downloading them on first use.
    pub async fn load(def: &ModelDefinition) -> Result<Arc<Self>> {
        let weights_path = download::ensure_model_file(def.hf_repo, def.hf_filename).await?;

        let config = def.config.clone();
        let pooling = def.pooling;
        let max_seq_length = def.max_seq_length;
        let engine = tokio::task::spawn_blocking(move || -> Result<InferenceEngine> {
            let device = Device::Cpu;
            // safety: the file is a read-only blob in the model cache; it is not
            // modified while mapped
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| Error::Other(format!("failed to load weights: {e}")))?
            };
            let model = Bert::load(vb, &config)
                .map_err(|e| Error::Other(format!("failed to build model: {e}")))?;
            Ok(InferenceEngine {
                model,
                max_seq_length,
                pooling,
            })
        })
        .await
        .map_err(|e| Error::Other(format!("model loading task failed: {e}")))??;

        Ok(Arc::new(engine))
    }

    /// Embed `texts`, each prefixed with `prefix` if given. Returns the vectors in
    /// input order and the number of tokens the model read, padding excluded.
    pub fn encode(&self, texts: &[String], prefix: Option<&str>) -> Result<(Vec<Vec<f32>>, u32)> {
        if texts.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let inputs: Vec<String> = match prefix {
            Some(p) => texts.iter().map(|t| format!("{p}{t}")).collect(),
            None => texts.to_vec(),
        };
        let tokenizer = wordpiece::get();
        let encodings: Vec<Vec<u32>> = inputs
            .iter()
            .map(|text| {
                let mut ids = Vec::new();
                tokenizer.encode(text, self.max_seq_length, &mut ids);
                ids
            })
            .collect();
        let total_tokens: u32 = encodings.iter().map(|e| e.len() as u32).sum();

        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|&i| encodings[i].len());

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); encodings.len()];
        for group in order.chunks(GROUP) {
            let batch: Vec<&[u32]> = group.iter().map(|&i| encodings[i].as_slice()).collect();
            let rows = self.forward(&batch)?;
            for (&i, row) in group.iter().zip(rows) {
                out[i] = row;
            }
        }
        Ok((out, total_tokens))
    }

    fn forward(&self, batch: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let device = Device::Cpu;
        let max_len = batch.iter().map(|e| e.len()).max().unwrap_or(0);
        let token_ids =
            padded_tensor(batch, max_len, &device, |ids, row| row.copy_from_slice(ids))?;
        // without padding every real token is attended to
        let attention_mask = padded_tensor(batch, max_len, &device, |_, row| row.fill(1))?;
        let padded = batch.iter().any(|e| e.len() < max_len);

        let output = self
            .model
            .forward(&token_ids, padded.then_some(&attention_mask))
            .map_err(|e| Error::Other(format!("forward pass failed: {e}")))?;
        let pooled = match self.pooling {
            PoolingStrategy::MeanPooling => mean_pool(&output, &attention_mask)?,
            PoolingStrategy::ClsToken => cls_pool(&output)?,
        };
        l2_normalize(&pooled)?
            .to_vec2::<f32>()
            .map_err(|e| Error::Other(format!("tensor extraction failed: {e}")))
    }
}

// one row per text, zero beyond its length; `fill` writes the text's part
fn padded_tensor(
    batch: &[&[u32]],
    max_len: usize,
    device: &Device,
    fill: impl Fn(&[u32], &mut [u32]),
) -> Result<Tensor> {
    let mut flat = vec![0u32; batch.len() * max_len];
    for (row, ids) in flat.chunks_mut(max_len).zip(batch) {
        fill(ids, &mut row[..ids.len()]);
    }
    Tensor::from_vec(flat, (batch.len(), max_len), device)
        .map_err(|e| Error::Other(format!("tensor creation failed: {e}")))
}

fn cls_pool(output: &Tensor) -> Result<Tensor> {
    // take the first token ([CLS]) from each sequence
    output
        .narrow(1, 0, 1)
        .and_then(|t| t.squeeze(1))
        .map_err(|e| Error::Other(format!("cls pooling failed: {e}")))
}

fn mean_pool(output: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let (_batch, _seq, hidden) = output
        .dims3()
        .map_err(|e| Error::Other(format!("unexpected tensor shape: {e}")))?;

    let mask = attention_mask
        .unsqueeze(2)
        .and_then(|m| m.to_dtype(DType::F32))
        .and_then(|m| {
            m.broadcast_as((
                attention_mask.dim(0).unwrap(),
                attention_mask.dim(1).unwrap(),
                hidden,
            ))
        })
        .map_err(|e| Error::Other(format!("mask broadcast failed: {e}")))?;

    let masked = output
        .mul(&mask)
        .map_err(|e| Error::Other(format!("masked mul failed: {e}")))?;

    let summed = masked
        .sum(1)
        .map_err(|e| Error::Other(format!("sum failed: {e}")))?;

    let counts = mask
        .sum(1)
        .map_err(|e| Error::Other(format!("count sum failed: {e}")))?;

    summed
        .div(&counts)
        .map_err(|e| Error::Other(format!("division failed: {e}")))
}

fn l2_normalize(tensor: &Tensor) -> Result<Tensor> {
    let norms = tensor
        .sqr()
        .and_then(|t| t.sum_keepdim(1))
        .and_then(|t| t.sqrt())
        .map_err(|e| Error::Other(format!("norm calculation failed: {e}")))?;

    tensor
        .broadcast_div(&norms)
        .map_err(|e| Error::Other(format!("normalization failed: {e}")))
}

#[cfg(test)]
mod tests {
    #[test]
    fn bert_config_values_are_reasonable() {
        for name in super::super::model::list_models() {
            let def = super::super::model::get_model(name).unwrap();
            let config = &def.config;
            assert!(
                config.hidden_size > 0,
                "{name} hidden_size should be positive"
            );
            assert!(
                config.num_hidden_layers > 0,
                "{name} num_hidden_layers should be positive"
            );
            assert!(
                config.num_attention_heads > 0,
                "{name} num_attention_heads should be positive"
            );
            assert!(
                config.vocab_size > 0,
                "{name} vocab_size should be positive"
            );
            assert!(
                config.intermediate_size > config.hidden_size,
                "{name} intermediate_size should be larger than hidden_size"
            );
            assert!(
                config.max_position_embeddings > 0,
                "{name} max_position_embeddings should be positive"
            );
            // hidden_size must be divisible by num_attention_heads
            assert_eq!(
                config.hidden_size % config.num_attention_heads,
                0,
                "{name} hidden_size must be divisible by num_attention_heads"
            );
            // hidden_size should match model definition
            assert_eq!(
                config.hidden_size, def.hidden_size,
                "{name} config.hidden_size should match def.hidden_size"
            );
        }
    }
}
