use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("api error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("request timed out after {0:?}")]
    Timeout(Duration),

    #[error("input too large: {0} texts exceeds provider maximum of {1}")]
    InputTooLarge(usize, usize),

    #[error("{0}")]
    Other(String),

    #[cfg(feature = "local")]
    #[error("unknown local model: {0}")]
    UnknownModel(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_api_error() {
        let err = Error::Api {
            status: 429,
            message: "rate limited".into(),
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
        };
        let debug = format!("{err:?}");
        assert!(debug.contains("Api"));
        assert!(debug.contains("500"));
    }
}
