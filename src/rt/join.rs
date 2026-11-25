use std::future::poll_fn;
use std::task::Poll;

use crate::error::Result;
use crate::transport::BoxFuture;

/// Run `futs` with at most `limit` in flight and return their outputs in
/// input order. The first error is returned and the futures still pending
/// are dropped.
///
/// Every in-flight future is polled on each wake-up: O(limit) per wake,
/// and `limit` is small.
pub(crate) async fn ordered<'a, T>(
    futs: Vec<BoxFuture<'a, Result<T>>>,
    limit: usize,
) -> Result<Vec<T>> {
    let limit = limit.max(1);
    let mut out: Vec<Option<T>> = futs.iter().map(|_| None).collect();
    let mut queue = futs.into_iter().enumerate();
    let mut flight: Vec<(usize, BoxFuture<'a, Result<T>>)> = Vec::with_capacity(limit);
    poll_fn(|cx| {
        loop {
            while flight.len() < limit {
                match queue.next() {
                    Some(f) => flight.push(f),
                    None => break,
                }
            }
            if flight.is_empty() {
                return Poll::Ready(Ok(()));
            }
            let before = flight.len();
            let mut i = 0;
            while i < flight.len() {
                match flight[i].1.as_mut().poll(cx) {
                    Poll::Ready(Ok(v)) => {
                        out[flight[i].0] = Some(v);
                        drop(flight.swap_remove(i));
                    }
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                    Poll::Pending => i += 1,
                }
            }
            // started futures need a first poll; otherwise wait for a wake-up
            if flight.len() == before {
                return Poll::Pending;
            }
        }
    })
    .await?;
    Ok(out.into_iter().flatten().collect())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::future::Future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Wake, Waker};

    use crate::error::Error;

    struct Flag(Mutex<bool>, std::sync::Condvar);
    impl Wake for Flag {
        fn wake(self: Arc<Self>) {
            *self.0.lock().unwrap() = true;
            self.1.notify_one();
        }
    }

    /// A minimal executor for tests: polls on the calling thread.
    pub(crate) fn block_on<F: Future>(f: F) -> F::Output {
        let flag = Arc::new(Flag(Mutex::new(false), std::sync::Condvar::new()));
        let waker = Waker::from(flag.clone());
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);
        loop {
            if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
            let mut woken = flag.0.lock().unwrap();
            while !*woken {
                woken = flag.1.wait(woken).unwrap();
            }
            *woken = false;
        }
    }

    // ready after `polls` polls, waking itself each time
    struct After {
        polls: usize,
        value: Result<usize>,
        live: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        started: bool,
    }

    impl Future for After {
        type Output = Result<usize>;
        fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if !self.started {
                self.started = true;
                let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(now, Ordering::SeqCst);
            }
            if self.polls == 0 {
                self.live.fetch_sub(1, Ordering::SeqCst);
                let v = std::mem::replace(&mut self.value, Ok(0));
                return Poll::Ready(v);
            }
            self.polls -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }

    fn futs(
        delays: &[usize],
        fail: Option<usize>,
        peak: &Arc<AtomicUsize>,
    ) -> Vec<BoxFuture<'static, Result<usize>>> {
        let live = Arc::new(AtomicUsize::new(0));
        delays
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                let value = if Some(i) == fail {
                    Err(Error::UnknownModel(format!("{i}")))
                } else {
                    Ok(i)
                };
                Box::pin(After {
                    polls: d,
                    value,
                    live: live.clone(),
                    peak: peak.clone(),
                    started: false,
                }) as BoxFuture<'static, Result<usize>>
            })
            .collect()
    }

    #[test]
    fn results_keep_input_order_under_out_of_order_completion() {
        let peak = Arc::new(AtomicUsize::new(0));
        let delays = [9, 1, 5, 0, 7, 3, 2, 8, 4, 6];
        let got = block_on(ordered(futs(&delays, None, &peak), 3)).unwrap();
        assert_eq!(got, (0..10).collect::<Vec<_>>());
        assert_eq!(peak.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn first_error_is_returned() {
        let peak = Arc::new(AtomicUsize::new(0));
        let err = block_on(ordered(futs(&[5, 0, 5], Some(1), &peak), 2)).unwrap_err();
        assert!(matches!(err, Error::UnknownModel(ref s) if s == "1"));
    }

    #[test]
    fn empty_and_limit_zero() {
        let peak = Arc::new(AtomicUsize::new(0));
        assert!(
            block_on(ordered(futs(&[], None, &peak), 4))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            block_on(ordered(futs(&[1, 0], None, &peak), 0)).unwrap(),
            vec![0, 1]
        );
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }
}
