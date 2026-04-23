use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::BertModel;
use tokenizers::Tokenizer;

use super::download;
use super::model::{ModelDefinition, PoolingStrategy};
use crate::error::{Error, Result};

/// Cached inference engine holding a loaded model and tokenizer.
pub(crate) struct InferenceEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    pooling: PoolingStrategy,
    #[allow(dead_code)]
    model_name: String,
    #[allow(dead_code)]
    hidden_size: usize,
}

// candle tensors are Send + Sync on CPU
unsafe impl Send for InferenceEngine {}
unsafe impl Sync for InferenceEngine {}

impl InferenceEngine {
    /// Load the model from HuggingFace Hub (or cache) and decompress the tokenizer.
    pub async fn load(def: &ModelDefinition) -> Result<Arc<Self>> {
        // decompress tokenizer
        let tokenizer_bytes = decompress_zstd(def.tokenizer_data)?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|e| Error::Other(format!("failed to load tokenizer: {e}")))?;

        // download weights
        let weights_path = download::ensure_model_file(def.hf_repo, def.hf_filename).await?;

        // load model on blocking thread
        let config = def.config.clone();
        let model_name = def.name.to_string();
        let hidden_size = def.hidden_size;
        let pooling = def.pooling;

        let engine = tokio::task::spawn_blocking(move || -> Result<InferenceEngine> {
            let device = Device::Cpu;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| Error::Other(format!("failed to load weights: {e}")))?
            };
            let model = BertModel::load(vb, &config)
                .map_err(|e| Error::Other(format!("failed to build model: {e}")))?;

            Ok(InferenceEngine {
                model,
                tokenizer,
                pooling,
                model_name,
                hidden_size,
            })
        })
        .await
        .map_err(|e| Error::Other(format!("model loading task failed: {e}")))??;

        Ok(Arc::new(engine))
    }

    /// Encode texts into embeddings. Returns (embeddings, total_tokens).
    pub fn encode(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u32)> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Other(format!("tokenization failed: {e}")))?;

        let total_tokens: u32 = encodings.iter().map(|e| e.get_ids().len() as u32).sum();

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        let device = Device::Cpu;

        // build padded tensors
        let token_ids = build_padded_tensor(&encodings, max_len, &device, |e| e.get_ids())?;
        let type_ids = build_padded_tensor(&encodings, max_len, &device, |e| e.get_type_ids())?;
        let attention_mask =
            build_padded_tensor(&encodings, max_len, &device, |e| e.get_attention_mask())?;

        // forward pass
        let output = self
            .model
            .forward(&token_ids, &type_ids, Some(&attention_mask))
            .map_err(|e| Error::Other(format!("forward pass failed: {e}")))?;

        // pooling
        let pooled = match self.pooling {
            PoolingStrategy::MeanPooling => mean_pool(&output, &attention_mask)?,
            PoolingStrategy::ClsToken => cls_pool(&output)?,
        };

        // L2 normalize
        let normalized = l2_normalize(&pooled)?;

        let embeddings = normalized
            .to_vec2::<f32>()
            .map_err(|e| Error::Other(format!("tensor extraction failed: {e}")))?;

        Ok((embeddings, total_tokens))
    }

    #[allow(dead_code)]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

fn build_padded_tensor(
    encodings: &[tokenizers::Encoding],
    max_len: usize,
    device: &Device,
    extract: impl Fn(&tokenizers::Encoding) -> &[u32],
) -> Result<Tensor> {
    let batch: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| {
            let ids = extract(e);
            let mut padded = ids.to_vec();
            padded.resize(max_len, 0);
            padded
        })
        .collect();

    Tensor::new(batch, device).map_err(|e| Error::Other(format!("tensor creation failed: {e}")))
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

fn decompress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(data)
        .map_err(|e| Error::Other(format!("zstd decoder init failed: {e}")))?;
    std::io::Read::read_to_end(&mut decoder, &mut output)
        .map_err(|e| Error::Other(format!("zstd decompression failed: {e}")))?;
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
