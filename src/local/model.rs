use candle_transformers::models::bert::{Config as BertConfig, HiddenAct, PositionEmbeddingType};

/// Strategy for pooling token-level embeddings into a single sentence embedding.
#[derive(Debug, Clone, Copy)]
pub enum PoolingStrategy {
    /// Average all token embeddings weighted by the attention mask.
    MeanPooling,
    /// Use the [CLS] token embedding.
    #[allow(dead_code)]
    ClsToken,
}

/// Definition of a local embedding model.
#[derive(Debug, Clone)]
pub struct ModelDefinition {
    /// Human-readable model name.
    pub name: &'static str,
    /// HuggingFace Hub repository ID.
    pub hf_repo: &'static str,
    /// Filename of the safetensors weights in the repo.
    pub hf_filename: &'static str,
    /// BERT model configuration.
    pub config: BertConfig,
    /// Output embedding dimension.
    pub hidden_size: usize,
    /// Maximum sequence length.
    pub max_seq_length: usize,
    /// Compressed tokenizer data (zstd).
    pub tokenizer_data: &'static [u8],
    /// Pooling strategy.
    pub pooling: PoolingStrategy,
}

const TOKENIZER_MINILM: &[u8] = include_bytes!("tokenizer_data/all_minilm_l6_v2.json.zst");
const TOKENIZER_MINILM_L12: &[u8] = include_bytes!("tokenizer_data/all_minilm_l12_v2.json.zst");
const TOKENIZER_BGE_SMALL: &[u8] = include_bytes!("tokenizer_data/bge_small_en_v1_5.json.zst");
const TOKENIZER_GTE_SMALL: &[u8] = include_bytes!("tokenizer_data/gte_small.json.zst");

/// shared BERT-small config (384 hidden, 12 heads, vocab 30522)
fn bert_small_config(num_hidden_layers: usize) -> BertConfig {
    BertConfig {
        vocab_size: 30522,
        hidden_size: 384,
        num_hidden_layers,
        num_attention_heads: 12,
        intermediate_size: 1536,
        hidden_act: HiddenAct::Gelu,
        hidden_dropout_prob: 0.1,
        max_position_embeddings: 512,
        type_vocab_size: 2,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        pad_token_id: 0,
        position_embedding_type: PositionEmbeddingType::Absolute,
        use_cache: true,
        classifier_dropout: None,
        model_type: Some("bert".to_string()),
    }
}

static ALL_MINILM_L6_V2: std::sync::LazyLock<ModelDefinition> =
    std::sync::LazyLock::new(|| ModelDefinition {
        name: "all-MiniLM-L6-v2",
        hf_repo: "sentence-transformers/all-MiniLM-L6-v2",
        hf_filename: "model.safetensors",
        config: bert_small_config(6),
        hidden_size: 384,
        max_seq_length: 256,
        tokenizer_data: TOKENIZER_MINILM,
        pooling: PoolingStrategy::MeanPooling,
    });

static ALL_MINILM_L12_V2: std::sync::LazyLock<ModelDefinition> =
    std::sync::LazyLock::new(|| ModelDefinition {
        name: "all-MiniLM-L12-v2",
        hf_repo: "sentence-transformers/all-MiniLM-L12-v2",
        hf_filename: "model.safetensors",
        config: bert_small_config(12),
        hidden_size: 384,
        max_seq_length: 128,
        tokenizer_data: TOKENIZER_MINILM_L12,
        pooling: PoolingStrategy::MeanPooling,
    });

static BGE_SMALL_EN_V1_5: std::sync::LazyLock<ModelDefinition> =
    std::sync::LazyLock::new(|| ModelDefinition {
        name: "bge-small-en-v1.5",
        hf_repo: "BAAI/bge-small-en-v1.5",
        hf_filename: "model.safetensors",
        config: bert_small_config(12),
        hidden_size: 384,
        max_seq_length: 512,
        tokenizer_data: TOKENIZER_BGE_SMALL,
        pooling: PoolingStrategy::ClsToken,
    });

static GTE_SMALL: std::sync::LazyLock<ModelDefinition> =
    std::sync::LazyLock::new(|| ModelDefinition {
        name: "gte-small",
        hf_repo: "thenlper/gte-small",
        hf_filename: "model.safetensors",
        config: bert_small_config(12),
        hidden_size: 384,
        max_seq_length: 512,
        tokenizer_data: TOKENIZER_GTE_SMALL,
        pooling: PoolingStrategy::MeanPooling,
    });

/// return the default model definition (all-MiniLM-L6-v2)
pub(crate) fn default_model() -> &'static ModelDefinition {
    &ALL_MINILM_L6_V2
}

pub fn get_model(name: &str) -> Option<&'static ModelDefinition> {
    match name {
        "all-MiniLM-L6-v2" => Some(&ALL_MINILM_L6_V2),
        "all-MiniLM-L12-v2" => Some(&ALL_MINILM_L12_V2),
        "bge-small-en-v1.5" => Some(&BGE_SMALL_EN_V1_5),
        "gte-small" => Some(&GTE_SMALL),
        _ => None,
    }
}

/// list all available local model names
pub fn list_models() -> &'static [&'static str] {
    &[
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "bge-small-en-v1.5",
        "gte-small",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minilm_l6_definition() {
        let def = get_model("all-MiniLM-L6-v2").unwrap();
        assert_eq!(def.name, "all-MiniLM-L6-v2");
        assert_eq!(def.hidden_size, 384);
        assert_eq!(def.max_seq_length, 256);
        assert_eq!(def.hf_repo, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(def.hf_filename, "model.safetensors");
        assert!(!def.tokenizer_data.is_empty());
        assert!(matches!(def.pooling, PoolingStrategy::MeanPooling));
    }

    #[test]
    fn minilm_l12_definition() {
        let def = get_model("all-MiniLM-L12-v2").unwrap();
        assert_eq!(def.name, "all-MiniLM-L12-v2");
        assert_eq!(def.hidden_size, 384);
        assert_eq!(def.max_seq_length, 128);
        assert_eq!(def.hf_repo, "sentence-transformers/all-MiniLM-L12-v2");
        assert!(matches!(def.pooling, PoolingStrategy::MeanPooling));
        assert_eq!(def.config.num_hidden_layers, 12);
    }

    #[test]
    fn bge_small_definition() {
        let def = get_model("bge-small-en-v1.5").unwrap();
        assert_eq!(def.name, "bge-small-en-v1.5");
        assert_eq!(def.hidden_size, 384);
        assert_eq!(def.max_seq_length, 512);
        assert_eq!(def.hf_repo, "BAAI/bge-small-en-v1.5");
        assert!(matches!(def.pooling, PoolingStrategy::ClsToken));
    }

    #[test]
    fn gte_small_definition() {
        let def = get_model("gte-small").unwrap();
        assert_eq!(def.name, "gte-small");
        assert_eq!(def.hidden_size, 384);
        assert_eq!(def.max_seq_length, 512);
        assert_eq!(def.hf_repo, "thenlper/gte-small");
        assert!(matches!(def.pooling, PoolingStrategy::MeanPooling));
    }

    #[test]
    fn all_models_have_nonempty_tokenizer_data() {
        for name in list_models() {
            let def = get_model(name).expect(name);
            assert!(
                !def.tokenizer_data.is_empty(),
                "{name} has empty tokenizer data"
            );
        }
    }

    #[test]
    fn all_tokenizer_data_is_zstd() {
        // zstd magic number: 0x28B52FFD (little-endian)
        for name in list_models() {
            let data = get_model(name).unwrap().tokenizer_data;
            assert!(data.len() > 4, "{name} tokenizer data too short");
            assert_eq!(data[0], 0x28, "{name} zstd magic[0]");
            assert_eq!(data[1], 0xB5, "{name} zstd magic[1]");
            assert_eq!(data[2], 0x2F, "{name} zstd magic[2]");
            assert_eq!(data[3], 0xFD, "{name} zstd magic[3]");
        }
    }

    #[test]
    fn bert_small_config_values() {
        let config = bert_small_config(6);
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_hidden_layers, 6);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.intermediate_size, 1536);
    }

    #[test]
    fn list_models_returns_all_four() {
        let models = list_models();
        assert_eq!(models.len(), 4);
        assert!(models.contains(&"all-MiniLM-L6-v2"));
        assert!(models.contains(&"all-MiniLM-L12-v2"));
        assert!(models.contains(&"bge-small-en-v1.5"));
        assert!(models.contains(&"gte-small"));
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(get_model("gpt-4").is_none());
    }
}
