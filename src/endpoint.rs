//! [`Endpoint`]: where requests go and how they are shaped. Pure data.

/// Request and response format of a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Wire {
    OpenAi,
    Cohere,
    Gemini,
    Voyage,
    Jina,
    Mistral,
}

impl Wire {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Wire::OpenAi => "openai",
            Wire::Cohere => "cohere",
            Wire::Gemini => "gemini",
            Wire::Voyage => "voyage",
            Wire::Jina => "jina",
            Wire::Mistral => "mistral",
        }
    }

    pub(crate) fn default_model(self) -> &'static str {
        match self {
            Wire::OpenAi => "text-embedding-3-small",
            Wire::Cohere => "embed-v4.0",
            Wire::Gemini => "gemini-embedding-001",
            Wire::Voyage => "voyage-4-large",
            Wire::Jina => "jina-embeddings-v3",
            Wire::Mistral => "mistral-embed",
        }
    }

    fn base_url(self) -> &'static str {
        match self {
            Wire::OpenAi => OPENAI_BASE_URL,
            Wire::Cohere => "https://api.cohere.com/v2",
            Wire::Gemini => "https://generativelanguage.googleapis.com/v1beta",
            Wire::Voyage => "https://api.voyageai.com/v1",
            Wire::Jina => "https://api.jina.ai/v1",
            Wire::Mistral => "https://api.mistral.ai/v1",
        }
    }

    fn limits(self) -> Limits {
        let (max_inputs, max_request_tokens, max_input_tokens) = match self {
            Wire::OpenAi => (2048, Some(300_000), Some(8192)),
            // 120k tokens is the request cap of the larger models
            Wire::Voyage => (1000, Some(120_000), None),
            Wire::Cohere => (96, None, None),
            // the next three input caps are observed, not documented
            Wire::Gemini => (100, None, None),
            Wire::Jina => (2048, None, None),
            Wire::Mistral => (512, None, None),
        };
        Limits {
            max_inputs,
            max_request_tokens,
            max_input_tokens,
        }
    }
}

pub(crate) const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// Per-request caps of an endpoint, used to refuse oversized `embed` calls
/// and to split `embed_batch` input into requests.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Inputs per request.
    pub max_inputs: usize,
    /// Tokens per request, when the provider documents a cap.
    pub max_request_tokens: Option<u32>,
    /// Tokens per input, when the provider documents a cap. Checked before
    /// sending for OpenAI models with [`Truncate::Error`](crate::Truncate::Error)
    /// and the `token-count` feature.
    pub max_input_tokens: Option<u32>,
}

impl Limits {
    /// Caps with no token limits.
    pub fn new(max_inputs: usize) -> Self {
        Self {
            max_inputs: max_inputs.max(1),
            max_request_tokens: None,
            max_input_tokens: None,
        }
    }

    /// Set the per-request token cap.
    pub fn request_tokens(self, max: u32) -> Self {
        Self {
            max_request_tokens: Some(max),
            ..self
        }
    }

    /// Set the per-input token cap.
    pub fn input_tokens(self, max: u32) -> Self {
        Self {
            max_input_tokens: Some(max),
            ..self
        }
    }
}

/// A provider endpoint: request format, base URL, credentials, extra
/// headers and limits.
///
/// ```rust
/// // Azure OpenAI v1 api: key in an `api-key` header instead of a bearer
/// // token, no api-version parameter, and the deployment name as the model
/// let azure = embedrs::Endpoint::openai_compatible(
///     "",
///     "https://my-resource.openai.azure.com/openai/v1",
/// )
/// .bearer(false)
/// .header("api-key", "azure-key");
/// ```
#[non_exhaustive]
#[derive(Clone)]
pub struct Endpoint {
    pub(crate) wire: Wire,
    pub(crate) base_url: String,
    pub(crate) key: String,
    pub(crate) bearer: bool,
    pub(crate) headers: Vec<(&'static str, String)>,
    pub(crate) limits: Limits,
}

impl std::fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let headers: Vec<&str> = self.headers.iter().map(|(k, _)| *k).collect();
        f.debug_struct("Endpoint")
            .field("provider", &self.wire.name())
            .field("base_url", &self.base_url)
            .field("key", &"<redacted>")
            .field("bearer", &self.bearer)
            .field("headers", &headers)
            .field("limits", &self.limits)
            .finish()
    }
}

macro_rules! constructors {
    ($($wire:ident: $name:ident, $compat:ident, $doc:literal;)*) => {$(
        #[doc = concat!("The ", $doc, " API.")]
        pub fn $name(key: impl Into<String>) -> Self {
            Self::with_wire(Wire::$wire, key.into(), Wire::$wire.base_url().into())
        }

        #[doc = concat!("A server that speaks the ", $doc, " API at `base_url`.")]
        pub fn $compat(key: impl Into<String>, base_url: impl Into<String>) -> Self {
            Self::with_wire(Wire::$wire, key.into(), base_url.into())
        }
    )*};
}

impl Endpoint {
    fn with_wire(wire: Wire, key: String, base_url: String) -> Self {
        Self {
            wire,
            base_url,
            key,
            bearer: true,
            headers: Vec::new(),
            limits: wire.limits(),
        }
    }

    constructors! {
        OpenAi: openai, openai_compatible, "OpenAI embeddings";
        Cohere: cohere, cohere_compatible, "Cohere v2 embed";
        Gemini: gemini, gemini_compatible, "Gemini embedContent";
        Voyage: voyage, voyage_compatible, "Voyage AI embeddings";
        Jina: jina, jina_compatible, "Jina AI embeddings";
        Mistral: mistral, mistral_compatible, "Mistral embeddings";
    }

    /// Replace the base URL.
    pub fn base_url(self, url: impl Into<String>) -> Self {
        Self {
            base_url: url.into(),
            ..self
        }
    }

    /// Add a header to every request (proxies, Azure's `api-key`).
    pub fn header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    /// Send the key as `Authorization: Bearer <key>` (default `true`; Gemini
    /// sends it as `x-goog-api-key` instead). Turn it off when the key goes
    /// in a [`header`](Self::header).
    pub fn bearer(self, on: bool) -> Self {
        Self { bearer: on, ..self }
    }

    /// Replace the limits.
    pub fn limits(self, limits: Limits) -> Self {
        Self { limits, ..self }
    }

    /// The provider name used in errors and traces.
    pub fn provider(&self) -> &'static str {
        self.wire.name()
    }

    /// The model used when none is set.
    pub fn default_model(&self) -> &'static str {
        self.wire.default_model()
    }

    /// OpenAI and Azure OpenAI document that an input cannot be an empty
    /// string; other OpenAI-compatible servers often accept one.
    pub(crate) fn refuses_empty_input(&self) -> bool {
        if self.wire != Wire::OpenAi {
            return false;
        }
        let host = self
            .base_url
            .split_once("://")
            .map_or(self.base_url.as_str(), |(_, rest)| rest)
            .split(['/', ':'])
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        host == "api.openai.com"
            || [
                ".openai.azure.com",
                ".services.ai.azure.com",
                ".cognitiveservices.azure.com",
            ]
            .iter()
            .any(|suffix| host.ends_with(suffix))
    }

    /// The current limits.
    pub fn current_limits(&self) -> Limits {
        self.limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_per_provider() {
        let cases = [
            (Endpoint::openai("k"), "https://api.openai.com/v1", 2048),
            (Endpoint::cohere("k"), "https://api.cohere.com/v2", 96),
            (
                Endpoint::gemini("k"),
                "https://generativelanguage.googleapis.com/v1beta",
                100,
            ),
            (Endpoint::voyage("k"), "https://api.voyageai.com/v1", 1000),
            (Endpoint::jina("k"), "https://api.jina.ai/v1", 2048),
            (Endpoint::mistral("k"), "https://api.mistral.ai/v1", 512),
        ];
        for (e, url, max) in cases {
            assert_eq!(e.base_url, url);
            assert_eq!(e.limits.max_inputs, max);
            assert!(e.bearer);
        }
        assert_eq!(
            Endpoint::voyage("k").limits.max_request_tokens,
            Some(120_000)
        );
        assert_eq!(Endpoint::openai("k").limits.max_input_tokens, Some(8192));
    }

    #[test]
    fn empty_input_refused_by_openai_and_azure_only() {
        let compat = |url| Endpoint::openai_compatible("k", url).refuses_empty_input();
        assert!(Endpoint::openai("k").refuses_empty_input());
        assert!(compat("https://my-resource.openai.azure.com/openai/v1"));
        assert!(compat(
            "https://My-Resource.services.ai.azure.com/openai/v1/"
        ));
        assert!(!compat("http://localhost:11434/v1"));
        assert!(!compat("https://openai.azure.com.example.org/v1"));
        assert!(!Endpoint::mistral("k").refuses_empty_input());
        assert!(!Endpoint::jina("k").refuses_empty_input());
    }

    #[test]
    fn options() {
        let e = Endpoint::openai_compatible("k", "http://x")
            .bearer(false)
            .header("api-key", "secret")
            .limits(Limits::new(0).request_tokens(10).input_tokens(5));
        assert!(!e.bearer);
        assert_eq!(e.limits.max_inputs, 1);
        assert_eq!(e.limits.max_request_tokens, Some(10));
        assert_eq!(e.base_url("http://y").base_url, "http://y");
        let dbg = format!(
            "{:?}",
            Endpoint::openai("sk-secret").header("api-key", "secret2")
        );
        assert!(!dbg.contains("secret"), "{dbg}");
    }
}
