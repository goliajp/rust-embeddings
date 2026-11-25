//! A timer on one std thread, for clients without a transport to provide one.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

type Slot = Arc<Mutex<Option<Waker>>>;

struct Entry(Instant, Slot);

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

struct Timer {
    heap: Mutex<BinaryHeap<Reverse<Entry>>>,
    cv: Condvar,
}

fn timer() -> &'static Timer {
    static T: OnceLock<&'static Timer> = OnceLock::new();
    T.get_or_init(|| {
        let t: &'static Timer = Box::leak(Box::new(Timer {
            heap: Mutex::new(BinaryHeap::new()),
            cv: Condvar::new(),
        }));
        std::thread::Builder::new()
            .name("embedrs-timer".into())
            .spawn(move || run(t))
            .expect("spawning the timer thread");
        t
    })
}

fn run(t: &Timer) {
    let mut heap = t.heap.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        let now = Instant::now();
        while heap.peek().is_some_and(|Reverse(e)| e.0 <= now) {
            let Some(Reverse(Entry(_, slot))) = heap.pop() else {
                break;
            };
            if let Some(w) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
                w.wake();
            }
        }
        heap = match heap.peek() {
            Some(Reverse(e)) => {
                let wait = e.0.saturating_duration_since(now);
                t.cv.wait_timeout(heap, wait)
                    .unwrap_or_else(|e| e.into_inner())
                    .0
            }
            None => t.cv.wait(heap).unwrap_or_else(|e| e.into_inner()),
        };
    }
}

/// Resolves once `d` has passed.
pub(crate) fn sleep(d: Duration) -> Sleep {
    Sleep {
        deadline: Instant::now() + d,
        slot: None,
    }
}

pub(crate) struct Sleep {
    deadline: Instant,
    slot: Option<Slot>,
}

impl Future for Sleep {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.deadline {
            return Poll::Ready(());
        }
        match &self.slot {
            Some(slot) => {
                *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(cx.waker().clone());
            }
            None => {
                let slot: Slot = Arc::new(Mutex::new(Some(cx.waker().clone())));
                let t = timer();
                t.heap
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(Reverse(Entry(self.deadline, slot.clone())));
                t.cv.notify_one();
                self.slot = Some(slot);
            }
        }
        Poll::Pending
    }
}

impl Drop for Sleep {
    fn drop(&mut self) {
        // the heap keeps the entry until its deadline; do not keep the task alive with it
        if let Some(slot) = &self.slot {
            slot.lock().unwrap_or_else(|e| e.into_inner()).take();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::join::tests::block_on;

    #[test]
    fn sleeps_at_least_the_duration() {
        let start = Instant::now();
        block_on(sleep(Duration::from_millis(30)));
        assert!(start.elapsed() >= Duration::from_millis(30));
        let start = Instant::now();
        block_on(async {
            let long = sleep(Duration::from_secs(60));
            let short = sleep(Duration::from_millis(10));
            assert_eq!(crate::rt::race::race(long, short).await, None);
        });
        assert!(start.elapsed() < Duration::from_secs(5));
    }
}
