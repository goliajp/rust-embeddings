//! Token usage and cost of a call.

/// Tokens a call consumed and what they cost.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    /// Tokens the provider reported; `None` when it reported no count.
    pub total_tokens: Option<u64>,
    /// Estimated cost in USD from `tiktoken`'s price table (feature
    /// `cost-tracking`); `None` without the feature, for a model with no
    /// price, or when the token count is unknown.
    pub cost: Option<f64>,
}

impl Usage {
    /// A usage record, for [`Embed`](crate::Embed) implementations outside embedrs.
    pub fn new(total_tokens: Option<u64>, cost: Option<f64>) -> Self {
        Self { total_tokens, cost }
    }

    /// Usage of one response; an unknown count has no cost, never a cost of zero.
    pub(crate) fn priced(model: &str, tokens: Option<u64>) -> Self {
        #[cfg(feature = "cost-tracking")]
        let cost = tokens.and_then(|t| tiktoken::pricing::estimate_cost(model, t, 0));
        #[cfg(not(feature = "cost-tracking"))]
        let cost = {
            let _ = model;
            None
        };
        Self {
            total_tokens: tokens,
            cost,
        }
    }

    /// Sum of two parts of one batch. An unknown part makes the sum unknown.
    pub(crate) fn add(&self, other: &Usage) -> Usage {
        Usage {
            total_tokens: self
                .total_tokens
                .zip(other.total_tokens)
                .map(|(a, b)| a + b),
            cost: self.cost.zip(other.cost).map(|(a, b)| a + b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_parts_poison_the_sum() {
        let a = Usage::new(Some(3), Some(0.5));
        let b = Usage::new(Some(4), Some(0.25));
        assert_eq!(a.add(&b), Usage::new(Some(7), Some(0.75)));
        assert_eq!(a.add(&Usage::new(None, None)), Usage::new(None, None));
        assert_eq!(a.add(&Usage::new(Some(1), None)), Usage::new(Some(4), None));
    }

    #[test]
    fn unknown_count_has_no_cost() {
        let u = Usage::priced("text-embedding-3-small", None);
        assert_eq!(u, Usage::default());
        let u = Usage::priced("text-embedding-3-small", Some(10));
        assert_eq!(u.total_tokens, Some(10));
        #[cfg(feature = "cost-tracking")]
        assert!(u.cost.unwrap() > 0.0);
        #[cfg(not(feature = "cost-tracking"))]
        assert_eq!(u.cost, None);
    }
}
