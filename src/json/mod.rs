//! JSON for the request bodies and responses embedrs handles: a writer for
//! flat request objects and a cursor that reads responses in one pass.

pub(crate) mod base64;
pub(crate) mod float;
pub(crate) mod read;
pub(crate) mod write;
