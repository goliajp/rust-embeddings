//! Crate-wide error type and `Result` alias.

use std::time::Duration;

use thiserror::Error;

/// All errors produced by embedrs.
#[derive(Debug, Error)]
pub enum Error {
    /// Transport-level failure from the HTTP client (connect, TLS, decode, etc.).
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// Response body could not be parsed as JSON, or did not match the expected schema.
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),

    /// Provider returned a non-2xx HTTP status with an error payload.
    #[error("api error ({status}): {message}")]
    #[non_exhaustive]
    Api {
        /// HTTP status code returned by the provider.
        status: u16,
        /// Human-readable error message extracted from the response body.
        message: String,
        /// Value of the `Retry-After` response header (delta-seconds form only),
        /// if present. Used by the retry loop to respect the provider's hint
        /// instead of the configured backoff.
        retry_after: Option<Duration>,
    },

    /// Request did not complete within the configured timeout.
    #[error("request timed out after {0:?}")]
    Timeout(Duration),

    /// Caller submitted more texts in one call than the provider accepts.
    #[error("input too large: {0} texts exceeds provider maximum of {1}")]
    InputTooLarge(usize, usize),

    /// Catch-all for errors that do not fit the categories above.
    #[error("{0}")]
    Other(String),

    /// Local-inference model id is not in the bundled registry.
    #[cfg(feature = "local")]
    #[error("unknown local model: {0}")]
    UnknownModel(String),
}

/// Convenience alias: `Result<T, embedrs::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_api_error() {
        let err = Error::Api {
            status: 429,
            message: "rate limited".into(),
            retry_after: None,
        };
        let s = err.to_string();
        assert!(s.contains("429"));
        assert!(s.contains("rate limited"));
    }

    #[test]
    fn display_timeout() {
        let err = Error::Timeout(Duration::from_secs(60));
        let s = err.to_string();
        assert!(s.contains("timed out"));
        assert!(s.contains("60"));
    }

    #[test]
    fn display_input_too_large() {
        let err = Error::InputTooLarge(3000, 2048);
        let s = err.to_string();
        assert!(s.contains("3000"));
        assert!(s.contains("2048"));
    }

    #[cfg(feature = "local")]
    #[test]
    fn display_unknown_model() {
        let err = Error::UnknownModel("nonexistent".into());
        let s = err.to_string();
        assert!(s.contains("nonexistent"));
    }

    #[test]
    fn display_other() {
        let err = Error::Other("something went wrong".into());
        assert_eq!(err.to_string(), "something went wrong");
    }

    #[test]
    fn json_error_conversion() {
        let json_err = serde_json::from_str::<String>("invalid").unwrap_err();
        let err: Error = json_err.into();
        assert!(matches!(err, Error::Json(_)));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn debug_format() {
        let err = Error::Api {
            status: 500,
            message: "internal".into(),
            retry_after: None,
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("Api"));
        assert!(debug.contains("500"));
    }
}
