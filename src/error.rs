//! Crate-wide error type and `Result` alias.

use std::fmt;
use std::time::Duration;

use crate::transport::TransportError;

/// All errors produced by embedrs. The variants are the same with every
/// feature set.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// The request did not get an HTTP answer: connect, TLS or I/O failure.
    Transport(TransportError),

    /// The provider answered with a non-2xx status.
    #[non_exhaustive]
    Api {
        /// HTTP status code.
        status: u16,
        /// The provider's error message, or the start of the body when it has none.
        message: String,
        /// When the provider says to retry: the `Retry-After` header (seconds
        /// or an HTTP date); without it, the later of OpenAI's
        /// `x-ratelimit-reset-requests` and `x-ratelimit-reset-tokens`, or
        /// the `retryDelay` of a Gemini error. Retries wait exactly this long;
        /// a hint longer than the backoff's `max` ends the retries, and the
        /// error is returned with the hint so the caller can schedule the
        /// next attempt.
        retry_after: Option<Duration>,
        /// Which provider answered (`"openai"`, `"cohere"`, ...).
        provider: &'static str,
    },

    /// A 2xx answer whose body is not what the provider documents.
    Decode(DecodeError),

    /// The call did not finish within the configured timeout.
    Timeout(Duration),

    /// More inputs in one `embed` call than the endpoint accepts.
    TooManyInputs {
        /// Inputs given.
        count: usize,
        /// Inputs the endpoint accepts per request.
        max: usize,
    },

    /// An input longer than the model reads, with [`Truncate::Error`](crate::Truncate::Error).
    InputTooLong {
        /// Position of the input.
        index: usize,
        /// Its length in tokens.
        tokens: u32,
        /// The model's limit.
        max: u32,
    },

    /// An empty string, for an endpoint whose API refuses one (OpenAI, Azure OpenAI).
    EmptyInput {
        /// Position of the input.
        index: usize,
    },

    /// A request option the model or provider cannot honour.
    Unsupported {
        /// The model the request was for.
        model: String,
        /// What was asked for, e.g. `"dimensions(256)"`.
        option: String,
    },

    /// A local model name that is not bundled.
    UnknownModel(String),

    /// A local model could not be downloaded or loaded.
    Model {
        /// The model name.
        model: String,
        /// What went wrong.
        message: String,
    },
}

/// Where and why a response body could not be read.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError {
    /// The provider whose answer it was.
    pub provider: &'static str,
    /// Byte offset in the body where reading stopped.
    pub offset: usize,
    /// What was expected or found.
    pub what: String,
}

impl DecodeError {
    pub(crate) fn new(provider: &'static str, offset: usize, what: impl Into<String>) -> Self {
        Self {
            provider,
            offset,
            what: what.into(),
        }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "could not read the {} response at byte {}: {}",
            self.provider, self.offset, self.what
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Transport(e) => write!(f, "transport: {e}"),
            Error::Api {
                status,
                message,
                provider,
                ..
            } => write!(f, "{provider} api error ({status}): {message}"),
            Error::Decode(e) => e.fmt(f),
            Error::Timeout(d) => write!(f, "request timed out after {d:?}"),
            Error::TooManyInputs { count, max } => write!(
                f,
                "{count} inputs in one request; the endpoint accepts at most {max}"
            ),
            Error::InputTooLong { index, tokens, max } => write!(
                f,
                "input {index} is {tokens} tokens; the model reads at most {max}"
            ),
            Error::EmptyInput { index } => {
                write!(
                    f,
                    "input {index} is empty; the endpoint refuses empty strings"
                )
            }
            Error::Unsupported { model, option } => {
                write!(f, "{model} does not support {option}")
            }
            Error::UnknownModel(m) => write!(f, "unknown local model: {m}"),
            Error::Model { model, message } => write!(f, "local model {model}: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Transport(e) => Some(e),
            _ => None,
        }
    }
}

impl From<TransportError> for Error {
    fn from(e: TransportError) -> Self {
        Error::Transport(e)
    }
}

impl From<DecodeError> for Error {
    fn from(e: DecodeError) -> Self {
        Error::Decode(e)
    }
}

impl Error {
    /// Worth another attempt, and worth a fallback provider: transport
    /// failures that may be transient and 408, 429, 500, 502, 503, 504, 529.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Api { status, .. } => {
                matches!(status, 408 | 429 | 500 | 502 | 503 | 504 | 529)
            }
            Error::Transport(e) => e.is_retryable(),
            _ => false,
        }
    }

    pub(crate) fn unsupported(model: &str, option: impl Into<String>) -> Self {
        Error::Unsupported {
            model: model.to_string(),
            option: option.into(),
        }
    }
}

/// `Result<T, embedrs::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::TransportErrorKind;

    fn api(status: u16) -> Error {
        Error::Api {
            status,
            message: "m".into(),
            retry_after: None,
            provider: "openai",
        }
    }

    #[test]
    fn display() {
        assert_eq!(api(429).to_string(), "openai api error (429): m");
        assert!(
            Error::Timeout(Duration::from_secs(60))
                .to_string()
                .contains("60s")
        );
        let e = Error::TooManyInputs {
            count: 3000,
            max: 2048,
        };
        assert!(e.to_string().contains("3000") && e.to_string().contains("2048"));
        assert_eq!(
            Error::unsupported("mistral-embed", "dimensions(256)").to_string(),
            "mistral-embed does not support dimensions(256)"
        );
        let d = DecodeError::new("cohere", 7, "expected an object");
        assert_eq!(
            Error::from(d).to_string(),
            "could not read the cohere response at byte 7: expected an object"
        );
    }

    #[test]
    fn retryable_statuses() {
        for s in [408, 429, 500, 502, 503, 504, 529] {
            assert!(api(s).is_retryable(), "{s}");
        }
        for s in [400, 401, 403, 404, 422] {
            assert!(!api(s).is_retryable(), "{s}");
        }
        assert!(!Error::Timeout(Duration::from_secs(1)).is_retryable());
        let t = TransportError::from_kind(TransportErrorKind::Connect);
        assert!(Error::from(t).is_retryable());
    }

    #[test]
    fn source_is_the_transport_error() {
        use std::error::Error as _;
        let t = TransportError::new(TransportErrorKind::Io, std::io::Error::other("reset"));
        let e = Error::from(t);
        assert!(e.source().unwrap().to_string().contains("reset"));
        assert!(api(500).source().is_none());
    }

    #[test]
    fn error_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>() {}
        check::<Error>();
    }
}
