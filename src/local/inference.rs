use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::{Tokenizer, TruncationParams};

use super::bert::Bert;
use super::download;
use super::model::{ModelDefinition, PoolingStrategy};
use crate::error::Result;

// texts are sorted by token count and run in groups of this many, so each
// group is padded only to its own longest text
const GROUP: usize = 64;

/// A loaded model and its tokenizer.
pub(crate) struct InferenceEngine {
    model: Bert,
    tokenizer: Tokenizer,
    pooling: PoolingStrategy,
}

impl InferenceEngine {
    /// Load the weights (downloading them on first use) and the embedded tokenizer.
    pub async fn load(def: &ModelDefinition) -> Result<Arc<Self>> {
        let tokenizer = load_tokenizer(def)?;
        let weights_path = download::ensure_model_file(def.hf_repo, def.hf_filename).await?;

        let config = def.config.clone();
        let pooling = def.pooling;
        let engine = tokio::task::spawn_blocking(move || -> Result<InferenceEngine> {
            let device = Device::Cpu;
            // safety: the file is a read-only blob in the model cache; it is not
            // modified while mapped
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| super::fail(format!("failed to load weights: {e}")))?
            };
            let model = Bert::load(vb, &config)
                .map_err(|e| super::fail(format!("failed to build model: {e}")))?;
            Ok(InferenceEngine {
                model,
                tokenizer,
                pooling,
            })
        })
        .await
        .map_err(|e| super::fail(format!("model loading task failed: {e}")))??;

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
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| super::fail(format!("tokenization failed: {e}")))?;
        let total_tokens: u32 = encodings.iter().map(|e| e.len() as u32).sum();

        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|&i| encodings[i].len());

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); encodings.len()];
        for group in order.chunks(GROUP) {
            let batch: Vec<&tokenizers::Encoding> = group.iter().map(|&i| &encodings[i]).collect();
            let rows = self.forward(&batch)?;
            for (&i, row) in group.iter().zip(rows) {
                out[i] = row;
            }
        }
        Ok((out, total_tokens))
    }

    fn forward(&self, batch: &[&tokenizers::Encoding]) -> Result<Vec<Vec<f32>>> {
        let device = Device::Cpu;
        let max_len = batch.iter().map(|e| e.len()).max().unwrap_or(0);
        let token_ids = padded_tensor(batch, max_len, &device, |e| e.get_ids())?;
        let attention_mask = padded_tensor(batch, max_len, &device, |e| e.get_attention_mask())?;
        let padded = batch.iter().any(|e| e.len() < max_len);

        let output = self
            .model
            .forward(&token_ids, padded.then_some(&attention_mask))
            .map_err(|e| super::fail(format!("forward pass failed: {e}")))?;
        let pooled = match self.pooling {
            PoolingStrategy::MeanPooling => mean_pool(&output, &attention_mask)?,
            PoolingStrategy::ClsToken => cls_pool(&output)?,
        };
        l2_normalize(&pooled)?
            .to_vec2::<f32>()
            .map_err(|e| super::fail(format!("tensor extraction failed: {e}")))
    }
}

pub(super) fn load_tokenizer(def: &ModelDefinition) -> Result<Tokenizer> {
    let bytes = decompress_zstd(def.tokenizer_data)?;
    let mut tokenizer = Tokenizer::from_bytes(&bytes)
        .map_err(|e| super::fail(format!("failed to load tokenizer: {e}")))?;
    // the embedded tokenizer files pad every text to a fixed length; padding is
    // done per group instead, and truncation follows the model's own limit
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: def.max_seq_length,
            ..Default::default()
        }))
        .map_err(|e| super::fail(format!("failed to set truncation: {e}")))?;
    Ok(tokenizer)
}

fn padded_tensor(
    batch: &[&tokenizers::Encoding],
    max_len: usize,
    device: &Device,
    extract: impl Fn(&tokenizers::Encoding) -> &[u32],
) -> Result<Tensor> {
    let mut flat = vec![0u32; batch.len() * max_len];
    for (row, e) in flat.chunks_mut(max_len).zip(batch) {
        let ids = extract(e);
        row[..ids.len()].copy_from_slice(ids);
    }
    Tensor::from_vec(flat, (batch.len(), max_len), device)
        .map_err(|e| super::fail(format!("tensor creation failed: {e}")))
}

fn cls_pool(output: &Tensor) -> Result<Tensor> {
    // take the first token ([CLS]) from each sequence
    output
        .narrow(1, 0, 1)
        .and_then(|t| t.squeeze(1))
        .map_err(|e| super::fail(format!("cls pooling failed: {e}")))
}

fn mean_pool(output: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let (_batch, _seq, hidden) = output
        .dims3()
        .map_err(|e| super::fail(format!("unexpected tensor shape: {e}")))?;

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
        .map_err(|e| super::fail(format!("mask broadcast failed: {e}")))?;

    let masked = output
        .mul(&mask)
        .map_err(|e| super::fail(format!("masked mul failed: {e}")))?;

    let summed = masked
        .sum(1)
        .map_err(|e| super::fail(format!("sum failed: {e}")))?;

    let counts = mask
        .sum(1)
        .map_err(|e| super::fail(format!("count sum failed: {e}")))?;

    summed
        .div(&counts)
        .map_err(|e| super::fail(format!("division failed: {e}")))
}

fn l2_normalize(tensor: &Tensor) -> Result<Tensor> {
    let norms = tensor
        .sqr()
        .and_then(|t| t.sum_keepdim(1))
        .and_then(|t| t.sqrt())
        .map_err(|e| super::fail(format!("norm calculation failed: {e}")))?;

    tensor
        .broadcast_div(&norms)
        .map_err(|e| super::fail(format!("normalization failed: {e}")))
}

fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(data)
        .map_err(|e| super::fail(format!("zstd decoder init failed: {e}")))?;
    std::io::Read::read_to_end(&mut decoder, &mut output)
        .map_err(|e| super::fail(format!("zstd decompression failed: {e}")))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompress_tokenizer() {
        let def = super::super::model::get_model("all-MiniLM-L6-v2").unwrap();
        let bytes = decompress_zstd(def.tokenizer_data).unwrap();
        // should be valid JSON
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(parsed.is_object());
        // should contain "model" key (wordpiece tokenizer)
        assert!(parsed.get("model").is_some());
    }

    #[test]
    fn decompress_round_trip() {
        let def = super::super::model::get_model("all-MiniLM-L6-v2").unwrap();
        let bytes = decompress_zstd(def.tokenizer_data).unwrap();
        // original tokenizer.json is ~466KB
        assert!(bytes.len() > 400_000);
        assert!(bytes.len() < 600_000);
    }

    #[test]
    fn decompress_tokenizer_all_minilm_l12_v2() {
        let def = super::super::model::get_model("all-MiniLM-L12-v2").unwrap();
        let bytes = decompress_zstd(def.tokenizer_data).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(parsed.is_object());
        assert!(parsed.get("model").is_some());
    }

    #[test]
    fn decompress_tokenizer_bge_small_en_v1_5() {
        let def = super::super::model::get_model("bge-small-en-v1.5").unwrap();
        let bytes = decompress_zstd(def.tokenizer_data).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(parsed.is_object());
        assert!(parsed.get("model").is_some());
    }

    #[test]
    fn decompress_tokenizer_gte_small() {
        let def = super::super::model::get_model("gte-small").unwrap();
        let bytes = decompress_zstd(def.tokenizer_data).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(parsed.is_object());
        assert!(parsed.get("model").is_some());
    }

    #[test]
    fn all_models_produce_valid_tokenizer() {
        for name in super::super::model::list_models() {
            let def = super::super::model::get_model(name).unwrap();
            let bytes = decompress_zstd(def.tokenizer_data)
                .unwrap_or_else(|e| panic!("{name} decompression failed: {e}"));
            let tokenizer = Tokenizer::from_bytes(&bytes)
                .unwrap_or_else(|e| panic!("{name} tokenizer load failed: {e}"));
            // tokenizer should be able to encode a simple string
            let encoding = tokenizer.encode("hello world", true).unwrap();
            assert!(
                !encoding.get_ids().is_empty(),
                "{name} tokenizer produced no tokens"
            );
        }
    }

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

    #[test]
    fn decompress_invalid_data_returns_error() {
        let invalid = b"not zstd data";
        let result = decompress_zstd(invalid);
        assert!(result.is_err());
    }
}
