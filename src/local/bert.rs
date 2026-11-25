//! BERT encoder forward pass for the bundled sentence-embedding models.
//!
//! Same weights and arithmetic as candle-transformers' BERT, with
//! the query, key and value projections fused into one matrix product, the
//! attention scale folded into the query weights, and the fused softmax and
//! layer-norm kernels.

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use super::model::BertShape;

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    fn load(vb: VarBuilder, size: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            bias: vb.get(size, "bias")?,
            eps: eps as f32,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm(x, &self.weight, &self.bias, self.eps)
    }
}

/// `x · w + b` with `w` stored as (in, out), contiguous.
struct Dense {
    weight: Tensor,
    bias: Tensor,
}

impl Dense {
    fn load(vb: VarBuilder, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            weight: vb.get((output, input), "weight")?.t()?.contiguous()?,
            bias: vb.get(output, "bias")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight)?.broadcast_add(&self.bias)
    }
}

struct Layer {
    qkv: Dense,
    attn_out: Dense,
    attn_norm: LayerNorm,
    ff_in: Dense,
    ff_out: Dense,
    ff_norm: LayerNorm,
}

impl Layer {
    fn load(vb: VarBuilder, c: &BertShape) -> Result<Self> {
        let h = c.hidden;
        let attn = vb.pp("attention");
        let sa = attn.pp("self");
        let scale = 1.0 / ((h / c.heads) as f64).sqrt();
        let q = Dense::load(sa.pp("query"), h, h)?;
        let k = Dense::load(sa.pp("key"), h, h)?;
        let v = Dense::load(sa.pp("value"), h, h)?;
        let qkv = Dense {
            weight: Tensor::cat(&[(q.weight * scale)?, k.weight, v.weight], 1)?.contiguous()?,
            bias: Tensor::cat(&[(q.bias * scale)?, k.bias, v.bias], 0)?,
        };
        Ok(Self {
            qkv,
            attn_out: Dense::load(attn.pp("output").pp("dense"), h, h)?,
            attn_norm: LayerNorm::load(attn.pp("output").pp("LayerNorm"), h, c.eps)?,
            ff_in: Dense::load(vb.pp("intermediate").pp("dense"), h, c.intermediate)?,
            ff_out: Dense::load(vb.pp("output").pp("dense"), c.intermediate, h)?,
            ff_norm: LayerNorm::load(vb.pp("output").pp("LayerNorm"), h, c.eps)?,
        })
    }

    /// `x` is (batch * seq, hidden); `mask_bias` is (batch, 1, 1, seq) or `None`
    /// when no position is padding.
    fn forward(
        &self,
        x: &Tensor,
        shape: (usize, usize, usize, usize),
        mask_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, s, heads, d) = shape;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b, s, 3, heads, d))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let mut scores = q.matmul(&k.t()?)?;
        if let Some(bias) = mask_bias {
            scores = scores.broadcast_add(bias)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * s, heads * d))?;
        let a = self
            .attn_norm
            .forward(&(self.attn_out.forward(&ctx)? + x)?)?;
        let f = self.ff_in.forward(&a)?.gelu_erf()?;
        self.ff_norm.forward(&(self.ff_out.forward(&f)? + a)?)
    }
}

pub(crate) struct Bert {
    word: Tensor,
    position: Tensor,
    token_type: Tensor,
    norm: LayerNorm,
    layers: Vec<Layer>,
    heads: usize,
    head_dim: usize,
}

impl Bert {
    pub(crate) fn load(vb: VarBuilder, c: &BertShape) -> Result<Self> {
        let vb = if vb.contains_tensor("embeddings.word_embeddings.weight") {
            vb
        } else {
            vb.pp("bert")
        };
        let h = c.hidden;
        let e = vb.pp("embeddings");
        let layers = (0..c.layers)
            .map(|i| Layer::load(vb.pp(format!("encoder.layer.{i}")), c))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            word: e.get((c.vocab, h), "word_embeddings.weight")?,
            position: e.get((c.max_positions, h), "position_embeddings.weight")?,
            token_type: e.get((c.type_vocab, h), "token_type_embeddings.weight")?,
            norm: LayerNorm::load(e.pp("LayerNorm"), h, c.eps)?,
            layers,
            heads: c.heads,
            head_dim: h / c.heads,
        })
    }

    /// Token states, (batch, seq, hidden). Every text is a single segment, so
    /// all token type ids are 0. `mask` is (batch, seq) of 0/1, `None` when no
    /// position is padding.
    pub(crate) fn forward(&self, ids: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, s) = ids.dims2()?;
        let h = self.heads * self.head_dim;
        let tokens = self
            .word
            .index_select(&ids.flatten_all()?, 0)?
            .reshape((b, s, h))?;
        let fixed = self
            .position
            .narrow(0, 0, s)?
            .broadcast_add(&self.token_type.narrow(0, 0, 1)?)?;
        let mut x = self
            .norm
            .forward(&tokens.broadcast_add(&fixed)?)?
            .reshape((b * s, h))?;
        let mask_bias = match mask {
            // padding gets the smallest f32 added to its score, as in the reference
            Some(m) => Some(
                m.to_dtype(DType::F32)?
                    .affine(-(f32::MIN as f64), f32::MIN as f64)?
                    .reshape((b, 1, 1, s))?,
            ),
            None => None,
        };
        let shape = (b, s, self.heads, self.head_dim);
        for layer in &self.layers {
            x = layer.forward(&x, shape, mask_bias.as_ref())?;
        }
        x.reshape((b, s, h))
    }
}

#[cfg(all(test, feature = "reqwest"))]
mod tests {
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config, HiddenAct, PositionEmbeddingType};

    use super::Bert;
    use crate::local::{hub, models, wordpiece};

    fn reference_config(s: &crate::local::model::BertShape) -> Config {
        Config {
            vocab_size: s.vocab,
            hidden_size: s.hidden,
            num_hidden_layers: s.layers,
            num_attention_heads: s.heads,
            intermediate_size: s.intermediate,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: s.max_positions,
            type_vocab_size: s.type_vocab,
            initializer_range: 0.02,
            layer_norm_eps: s.eps,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }

    // downloads every bundled model (67 to 133 MB each)
    #[test]
    #[ignore]
    fn matches_the_candle_reference() {
        let texts = ["hi", "a somewhat longer sentence about rivers and stones", ""];
        let transport = crate::transport::ReqwestTransport::shared();
        let rt = tokio::runtime::Runtime::new().unwrap();
        for m in models() {
            let path = rt
                .block_on(hub::Hub::from_env().ensure(&m.pinned(), transport.as_ref()))
                .unwrap();
            // safety: read-only cache blob
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, &Device::Cpu).unwrap()
            };
            let shape = m.shape();
            let reference = BertModel::load(vb.clone(), &reference_config(&shape)).unwrap();
            let ours = Bert::load(vb, &shape).unwrap();

            let enc: Vec<Vec<u32>> = texts
                .iter()
                .map(|t| {
                    let mut ids = Vec::new();
                    wordpiece::get().encode(t, m.max_tokens, &mut ids);
                    ids
                })
                .collect();
            let len = enc.iter().map(Vec::len).max().unwrap();
            let mut ids = vec![0u32; texts.len() * len];
            let mut mask = vec![0u32; texts.len() * len];
            for (i, e) in enc.iter().enumerate() {
                ids[i * len..i * len + e.len()].copy_from_slice(e);
                mask[i * len..i * len + e.len()].fill(1);
            }
            let ids = Tensor::from_vec(ids, (texts.len(), len), &Device::Cpu).unwrap();
            let mask = Tensor::from_vec(mask, (texts.len(), len), &Device::Cpu).unwrap();
            let types = ids.zeros_like().unwrap();

            let want = reference.forward(&ids, &types, Some(&mask)).unwrap().to_vec3::<f32>().unwrap();
            let got = ours.forward(&ids, Some(&mask)).unwrap().to_vec3::<f32>().unwrap();
            let mut worst = 0f32;
            for (i, e) in enc.iter().enumerate() {
                for t in 0..e.len() {
                    for (a, b) in want[i][t].iter().zip(&got[i][t]) {
                        worst = worst.max((a - b).abs());
                    }
                }
            }
            assert!(worst < 1e-4, "{}: max abs difference {worst}", m.name);
        }
    }
}
