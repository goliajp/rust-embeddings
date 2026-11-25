//! Async helpers on std only; embedrs needs no async runtime.

pub(crate) mod join;
pub(crate) mod oneshot;
pub(crate) mod race;
pub(crate) mod timer;
