//! What goes into a call: the texts and per-call options.

/// The type of input being embedded, for providers and models that use it.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputType {
    /// Document for search indexing.
    SearchDocument,
    /// Query for search retrieval.
    SearchQuery,
    /// Text for classification.
    Classification,
    /// Text for clustering.
    Clustering,
}

/// What happens to an input longer than the model reads.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Truncate {
    /// Whatever the provider does by default: Voyage, Cohere, Gemini and
    /// local models cut the end, OpenAI and Jina answer with an error.
    #[default]
    Provider,
    /// Cut the end. Sent as Voyage `truncation: true`, Cohere
    /// `truncate: "END"`, Jina `truncate: true`, Gemini
    /// `embedContentConfig.autoTruncate: true`; what local models do.
    /// [`Error::Unsupported`](crate::Error::Unsupported) for providers
    /// without such a switch.
    End,
    /// Refuse long inputs. Sent as Voyage `truncation: false`, Cohere
    /// `truncate: "NONE"`, Gemini `embedContentConfig.autoTruncate: false`; OpenAI and Jina refuse by default; local models,
    /// and OpenAI models with the `token-count` feature, return
    /// [`Error::InputTooLong`](crate::Error::InputTooLong) before
    /// running. [`Error::Unsupported`](crate::Error::Unsupported) for
    /// providers that cannot be asked to refuse.
    Error,
}

/// The texts of one call, owned or borrowed.
///
/// Built with `From` from `Vec<String>`, `&[String]`, `&Vec<String>`,
/// `&[&str]`, `&Vec<&str>`, `[&str; N]` and `&str`, so `embed` takes any of
/// them.
#[derive(Debug, Clone)]
pub enum Inputs<'t> {
    /// Owned strings.
    Owned(Vec<String>),
    /// Borrowed strings.
    Strings(&'t [String]),
    /// Borrowed string slices.
    Strs(&'t [&'t str]),
    /// Owned list of borrowed string slices.
    StrVec(Vec<&'t str>),
}

impl<'t> Inputs<'t> {
    /// Number of texts.
    pub fn len(&self) -> usize {
        match self {
            Inputs::Owned(v) => v.len(),
            Inputs::Strings(v) => v.len(),
            Inputs::Strs(v) => v.len(),
            Inputs::StrVec(v) => v.len(),
        }
    }

    /// True when there are no texts.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Text `i`.
    ///
    /// # Panics
    ///
    /// If `i` is out of range.
    pub fn get(&self, i: usize) -> &str {
        match self {
            Inputs::Owned(v) => &v[i],
            Inputs::Strings(v) => &v[i],
            Inputs::Strs(v) => v[i],
            Inputs::StrVec(v) => v[i],
        }
    }

    /// The texts in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }

    /// A borrowed view of `range`.
    pub(crate) fn slice(&self, range: std::ops::Range<usize>) -> Inputs<'_> {
        match self {
            Inputs::Owned(v) => Inputs::Strings(&v[range]),
            Inputs::Strings(v) => Inputs::Strings(&v[range]),
            Inputs::Strs(v) => Inputs::Strs(&v[range]),
            Inputs::StrVec(v) => Inputs::Strs(&v[range]),
        }
    }
}

impl From<Vec<String>> for Inputs<'_> {
    fn from(v: Vec<String>) -> Self {
        Inputs::Owned(v)
    }
}

impl<'t> From<&'t [String]> for Inputs<'t> {
    fn from(v: &'t [String]) -> Self {
        Inputs::Strings(v)
    }
}

impl<'t> From<&'t Vec<String>> for Inputs<'t> {
    fn from(v: &'t Vec<String>) -> Self {
        Inputs::Strings(v)
    }
}

impl<'t> From<&'t [&'t str]> for Inputs<'t> {
    fn from(v: &'t [&'t str]) -> Self {
        Inputs::Strs(v)
    }
}

impl<'t> From<&'t Vec<&'t str>> for Inputs<'t> {
    fn from(v: &'t Vec<&'t str>) -> Self {
        Inputs::Strs(v)
    }
}

impl<'t, const N: usize> From<[&'t str; N]> for Inputs<'t> {
    fn from(v: [&'t str; N]) -> Self {
        Inputs::StrVec(v.to_vec())
    }
}

impl<'t> From<&'t str> for Inputs<'t> {
    fn from(v: &'t str) -> Self {
        Inputs::StrVec(vec![v])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts<'t>(i: impl Into<Inputs<'t>>) -> Vec<String> {
        i.into().iter().map(str::to_string).collect()
    }

    #[test]
    fn every_source_gives_the_same_texts() {
        let owned = vec!["a".to_string(), "b".to_string()];
        let strs = vec!["a", "b"];
        let want = vec!["a".to_string(), "b".to_string()];
        assert_eq!(texts(owned.clone()), want);
        assert_eq!(texts(&owned), want);
        assert_eq!(texts(owned.as_slice()), want);
        assert_eq!(texts(&strs), want);
        assert_eq!(texts(strs.as_slice()), want);
        assert_eq!(texts(["a", "b"]), want);
        assert_eq!(texts("a"), vec!["a".to_string()]);
        // the 0.6 call shape still infers Vec<String>
        assert_eq!(texts(vec!["a".into(), "b".into()]), want);
    }

    #[test]
    fn slices() {
        let i = Inputs::from(vec!["a".to_string(), "b".into(), "c".into()]);
        let s = i.slice(1..3);
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0), "b");
        let strs = ["x", "y"];
        assert_eq!(Inputs::from(strs).slice(1..2).get(0), "y");
    }
}
