//! A BERT encoder's weights, laid out for the forward pass. Query, key and
//! value are fused into one matrix with the attention scale folded into the
//! query rows; position and token-type embeddings are pre-added. Everything
//! else is used as stored: on macOS the f32 matrices are read in place from
//! the mapped file.

use std::path::Path;

use super::gemm::Weight;
use super::model::BertShape;
use super::safetensors::{F32s, Safetensors};
use crate::error::Result;

pub(crate) struct Linear {
    pub(crate) w: Weight,
    pub(crate) b: Vec<f32>,
}

pub(crate) struct Norm {
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
}

pub(crate) struct Layer {
    pub(crate) qkv: Linear,
    pub(crate) out: Linear,
    pub(crate) norm1: Norm,
    pub(crate) ff_in: Linear,
    pub(crate) ff_out: Linear,
    pub(crate) norm2: Norm,
}

pub(crate) struct Weights {
    pub(crate) shape: BertShape,
    /// (vocab, hidden).
    pub(crate) word: F32s,
    /// (max_positions, hidden): position embedding plus token type 0.
    pub(crate) pos_type: Vec<f32>,
    pub(crate) norm: Norm,
    pub(crate) layers: Vec<Layer>,
}

struct Reader<'a> {
    st: &'a Safetensors,
    prefix: &'static str,
}

impl Reader<'_> {
    fn get(&self, name: &str, shape: &[usize]) -> Result<F32s> {
        self.st.f32s(&format!("{}{name}", self.prefix), shape)
    }

    fn vec(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        Ok(self.get(name, &[n])?.to_vec())
    }

    fn norm(&self, name: &str, n: usize) -> Result<Norm> {
        Ok(Norm {
            w: self.vec(&format!("{name}.weight"), n)?,
            b: self.vec(&format!("{name}.bias"), n)?,
        })
    }

    fn linear(&self, name: &str, n: usize, k: usize) -> Result<Linear> {
        Ok(Linear {
            w: Weight::new(self.get(&format!("{name}.weight"), &[n, k])?, n, k),
            b: self.vec(&format!("{name}.bias"), n)?,
        })
    }

    fn layer(&self, i: usize, c: &BertShape) -> Result<Layer> {
        let (h, f) = (c.hidden, c.intermediate);
        let p = format!("encoder.layer.{i}.");
        let scale = (1.0 / ((h / c.heads) as f64).sqrt()) as f32;
        let mut w = Vec::with_capacity(3 * h * h);
        let mut b = Vec::with_capacity(3 * h);
        for (part, s) in [("query", scale), ("key", 1.0), ("value", 1.0)] {
            let name = format!("{p}attention.self.{part}");
            w.extend(
                self.get(&format!("{name}.weight"), &[h, h])?
                    .iter()
                    .map(|v| v * s),
            );
            b.extend(
                self.get(&format!("{name}.bias"), &[h])?
                    .iter()
                    .map(|v| v * s),
            );
        }
        Ok(Layer {
            qkv: Linear {
                w: Weight::new(F32s::Owned(w), 3 * h, h),
                b,
            },
            out: self.linear(&format!("{p}attention.output.dense"), h, h)?,
            norm1: self.norm(&format!("{p}attention.output.LayerNorm"), h)?,
            ff_in: self.linear(&format!("{p}intermediate.dense"), f, h)?,
            ff_out: self.linear(&format!("{p}output.dense"), h, f)?,
            norm2: self.norm(&format!("{p}output.LayerNorm"), h)?,
        })
    }
}

impl Weights {
    pub(crate) fn load(path: &Path, c: &BertShape) -> Result<Self> {
        let st = Safetensors::open(path)?;
        let prefix = if st.contains("embeddings.word_embeddings.weight") {
            ""
        } else {
            "bert."
        };
        let r = Reader { st: &st, prefix };
        let h = c.hidden;
        let pos = r.get(
            "embeddings.position_embeddings.weight",
            &[c.max_positions, h],
        )?;
        let ty = r.get(
            "embeddings.token_type_embeddings.weight",
            &[c.type_vocab, h],
        )?;
        let pos_type = pos
            .chunks_exact(h)
            .flat_map(|row| row.iter().zip(&ty[..h]).map(|(p, t)| p + t))
            .collect();
        Ok(Self {
            shape: *c,
            word: r.get("embeddings.word_embeddings.weight", &[c.vocab, h])?,
            pos_type,
            norm: r.norm("embeddings.LayerNorm", h)?,
            layers: (0..c.layers)
                .map(|i| r.layer(i, c))
                .collect::<Result<_>>()?,
        })
    }
}
