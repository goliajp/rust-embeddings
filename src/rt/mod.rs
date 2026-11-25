//! Async helpers on std only; embedrs needs no async runtime.

pub(crate) mod join;
#[cfg(feature = "local")]
pub(crate) mod oneshot;
pub(crate) mod race;
pub(crate) mod timer;
