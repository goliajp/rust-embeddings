mod download;
mod inference;
mod model;

pub(crate) use inference::InferenceEngine;
pub(crate) use model::default_model;
pub use model::{ModelDefinition, PoolingStrategy};

/// Get a model definition by name. Returns `None` if the model is not recognized.
pub fn get_model(name: &str) -> Option<&'static ModelDefinition> {
    model::get_model(name)
}

/// List all available local model names.
pub fn list_models() -> &'static [&'static str] {
    model::list_models()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_known_model() {
        let def = get_model("all-MiniLM-L6-v2");
        assert!(def.is_some());
        let def = def.unwrap();
        assert_eq!(def.name, "all-MiniLM-L6-v2");
        assert_eq!(def.hidden_size, 384);
    }

    #[test]
    fn get_unknown_model() {
        assert!(get_model("nonexistent-model").is_none());
    }

    #[test]
    fn list_available_models() {
        let models = list_models();
        assert!(models.contains(&"all-MiniLM-L6-v2"));
    }
}
