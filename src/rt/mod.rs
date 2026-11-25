//! Async helpers on std only; embedrs needs no async runtime.

pub(crate) mod join;
#[cfg(feature = "local")]
pub(crate) mod oneshot;
pub(crate) mod race;
pub(crate) mod timer;

/// Poll `f` to completion on the calling thread, parking between wake-ups.
#[cfg(any(test, feature = "local"))]
pub(crate) fn block_on<F: std::future::Future>(f: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct Unpark(std::thread::Thread);
    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::park();
    }
}
