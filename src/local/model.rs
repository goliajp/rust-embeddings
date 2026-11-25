//! The bundled models: shape, pooling, limits, and the pinned weights file.

use super::hub::Pinned;

/// How token states become one vector.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pooling {
    /// Mean of the token states, padding excluded.
    Mean,
    /// The state of the `[CLS]` token.
    Cls,
}

/// A bundled local model.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInfo {
    /// The name `Client::local` takes.
    pub name: &'static str,
    /// Vector width.
    pub dim: usize,
    /// Transformer layers.
    pub layers: usize,
    /// Tokens read per text, `[CLS]` and `[SEP]` included; longer texts are cut.
    pub max_tokens: usize,
    /// Pooling.
    pub pooling: Pooling,
    /// Instruction put in front of texts with `InputType::SearchQuery`.
    pub query_prefix: Option<&'static str>,
    /// Hugging Face repository.
    pub repo: &'static str,
    /// The commit the weights are pinned to.
    pub revision: &'static str,
    /// The weights file in the repository.
    pub file: &'static str,
    /// Its size in bytes.
    pub bytes: u64,
    /// Its SHA-256, checked after every download.
    pub sha256: &'static str,
}

impl ModelInfo {
    pub(crate) fn pinned(&self) -> Pinned {
        Pinned {
            repo: self.repo,
            revision: self.revision,
            file: self.file,
            size: self.bytes,
            sha256: self.sha256,
        }
    }

    pub(crate) fn shape(&self) -> BertShape {
        BertShape {
            vocab: 30522,
            hidden: self.dim,
            layers: self.layers,
            heads: 12,
            intermediate: 1536,
            max_positions: 512,
            type_vocab: 2,
            eps: 1e-12,
        }
    }
}

/// Sizes of a BERT encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BertShape {
    pub(crate) vocab: usize,
    pub(crate) hidden: usize,
    pub(crate) layers: usize,
    pub(crate) heads: usize,
    pub(crate) intermediate: usize,
    pub(crate) max_positions: usize,
    pub(crate) type_vocab: usize,
    pub(crate) eps: f64,
}

const MODELS: [ModelInfo; 4] = [
    ModelInfo {
        name: "all-MiniLM-L6-v2",
        dim: 384,
        layers: 6,
        max_tokens: 256,
        pooling: Pooling::Mean,
        query_prefix: None,
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        revision: "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        file: "model.safetensors",
        bytes: 90_868_376,
        sha256: "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    },
    ModelInfo {
        name: "all-MiniLM-L12-v2",
        dim: 384,
        layers: 12,
        max_tokens: 128,
        pooling: Pooling::Mean,
        query_prefix: None,
        repo: "sentence-transformers/all-MiniLM-L12-v2",
        revision: "a50ef00143b4d5391434df20ae11632588ac25be",
        file: "model.safetensors",
        bytes: 133_466_304,
        sha256: "d2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542",
    },
    ModelInfo {
        name: "bge-small-en-v1.5",
        dim: 384,
        layers: 12,
        max_tokens: 512,
        pooling: Pooling::Cls,
        query_prefix: Some("Represent this sentence for searching relevant passages: "),
        repo: "BAAI/bge-small-en-v1.5",
        revision: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        file: "model.safetensors",
        bytes: 133_466_304,
        sha256: "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
    },
    ModelInfo {
        name: "gte-small",
        dim: 384,
        layers: 12,
        max_tokens: 512,
        pooling: Pooling::Mean,
        query_prefix: None,
        repo: "thenlper/gte-small",
        revision: "17e1f347d17fe144873b1201da91788898c639cd",
        file: "model.safetensors",
        bytes: 66_746_168,
        sha256: "9a1eb90bbac323ea08aa5629b624fe6ae75db121b904799c2266a1e2c2de22d2",
    },
];

/// Every bundled model; the first is the default.
pub fn models() -> &'static [ModelInfo] {
    &MODELS
}

pub(crate) fn find(name: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table() {
        assert_eq!(models()[0].name, "all-MiniLM-L6-v2");
        for m in models() {
            assert_eq!(find(m.name), Some(m));
            assert_eq!(m.dim, 384);
            assert_eq!(m.revision.len(), 40);
            assert_eq!(m.sha256.len(), 64);
            assert_eq!(m.shape().hidden % m.shape().heads, 0);
            assert_eq!(m.query_prefix.is_some(), m.name == "bge-small-en-v1.5");
        }
        assert_eq!(find("gpt-4"), None);
        let lim: Vec<usize> = models().iter().map(|m| m.max_tokens).collect();
        assert_eq!(lim, [256, 128, 512, 512]);
    }
}
