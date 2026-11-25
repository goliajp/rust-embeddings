//! A single value handed from a std thread to a future.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

struct Slot<T> {
    value: Option<T>,
    waker: Option<Waker>,
    closed: bool,
}

pub(crate) struct Sender<T>(Arc<Mutex<Slot<T>>>);
pub(crate) struct Receiver<T>(Arc<Mutex<Slot<T>>>);

pub(crate) fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let slot = Arc::new(Mutex::new(Slot {
        value: None,
        waker: None,
        closed: false,
    }));
    (Sender(slot.clone()), Receiver(slot))
}

impl<T> Sender<T> {
    pub(crate) fn send(self, value: T) {
        let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner());
        s.value = Some(value);
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let waker = {
            let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner());
            s.closed = true;
            s.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

/// Resolves to `None` when the sender was dropped without sending (the
/// thread panicked).
impl<T> Future for Receiver<T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
        let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = s.value.take() {
            return Poll::Ready(Some(v));
        }
        if s.closed {
            return Poll::Ready(None);
        }
        s.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// Run `f` on a new std thread and await its result.
pub(crate) async fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (tx, rx) = channel();
    std::thread::spawn(move || tx.send(f()));
    rx.await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::join::tests::block_on;

    #[test]
    fn value_crosses_threads() {
        assert_eq!(block_on(spawn(|| 40 + 2)), Some(42));
    }

    #[test]
    fn dropped_sender_resolves_to_none() {
        let (tx, rx) = channel::<u8>();
        std::thread::spawn(move || drop(tx));
        assert_eq!(block_on(rx), None);
        let r = block_on(spawn(|| -> u8 { panic!("worker failed") }));
        assert_eq!(r, None);
    }
}
