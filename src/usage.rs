/// Token usage information from an embedding request.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// Number of tokens consumed by the input texts.
    pub total_tokens: u32,
}

impl Usage {
    pub(crate) fn accumulate(&mut self, tokens: u32) {
        self.total_tokens += tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_usage() {
        let usage = Usage::default();
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn accumulate_single() {
        let mut usage = Usage::default();
        usage.accumulate(100);
        assert_eq!(usage.total_tokens, 100);
    }

    #[test]
    fn accumulate_multiple() {
        let mut usage = Usage::default();
        usage.accumulate(100);
        usage.accumulate(200);
        assert_eq!(usage.total_tokens, 300);
    }

    #[test]
    fn clone_usage() {
        let mut usage = Usage::default();
        usage.accumulate(10);
        let cloned = usage.clone();
        assert_eq!(cloned.total_tokens, 10);
    }

    #[test]
    fn debug_format() {
        let usage = Usage::default();
        let debug = format!("{usage:?}");
        assert!(debug.contains("Usage"));
        assert!(debug.contains("total_tokens"));
    }
}
