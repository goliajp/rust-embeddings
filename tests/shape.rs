//! Public types have the same fields and variants with every feature set:
//! this file builds under each of them and names all of them.

use std::time::Duration;

use embedrs::{DecodeError, EmbedResult, Error, Usage};

#[test]
fn usage_and_result_fields() {
    let u = Usage::new(Some(1), None);
    let Usage {
        total_tokens, cost, ..
    } = u.clone();
    assert_eq!((total_tokens, cost), (Some(1), None));
    let r = EmbedResult::from_flat(vec![1.0, 2.0], 2, "m", u);
    let EmbedResult { usage, model, .. } = &r;
    assert_eq!((usage.total_tokens, model.as_str()), (Some(1), "m"));
    assert_eq!((r.len(), r.dim(), &r[0]), (1, 2, &[1.0, 2.0][..]));
}

#[test]
#[expect(
    clippy::unneeded_wildcard_pattern,
    reason = "naming every field is the point"
)]
fn every_error_variant() {
    fn name(e: &Error) -> &'static str {
        match e {
            Error::Transport(_) => "transport",
            Error::Api {
                status: _,
                message: _,
                retry_after: _,
                provider: _,
                ..
            } => "api",
            Error::Decode(DecodeError {
                provider: _,
                offset: _,
                what: _,
                ..
            }) => "decode",
            Error::Timeout(_) => "timeout",
            Error::TooManyInputs { count: _, max: _ } => "too many",
            Error::InputTooLong {
                index: _,
                tokens: _,
                max: _,
            } => "too long",
            Error::EmptyInput { index: _ } => "empty",
            Error::Unsupported {
                model: _,
                option: _,
            } => "unsupported",
            Error::UnknownModel(_) => "unknown model",
            Error::Model {
                model: _,
                message: _,
            } => "model",
            _ => "newer",
        }
    }
    assert_eq!(name(&Error::Timeout(Duration::from_secs(1))), "timeout");
    assert_eq!(name(&Error::UnknownModel("x".into())), "unknown model");
}

#[test]
fn error_is_send_sync_static() {
    fn check<T: std::error::Error + Send + Sync + 'static>() {}
    check::<Error>();
}
