use std::future::{Future, poll_fn};
use std::pin::pin;
use std::task::Poll;

/// Run `work` until it finishes or `timer` fires; `None` when the timer won.
/// `work` is polled first, so a result ready at the deadline is kept.
pub(crate) async fn race<T>(
    work: impl Future<Output = T>,
    timer: impl Future<Output = ()>,
) -> Option<T> {
    let mut work = pin!(work);
    let mut timer = pin!(timer);
    poll_fn(|cx| {
        if let Poll::Ready(v) = work.as_mut().poll(cx) {
            return Poll::Ready(Some(v));
        }
        if timer.as_mut().poll(cx).is_ready() {
            return Poll::Ready(None);
        }
        Poll::Pending
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt::join::tests::block_on;
    use std::future::{pending, ready};

    #[test]
    fn first_to_finish_wins() {
        assert_eq!(block_on(race(ready(1), pending())), Some(1));
        assert_eq!(block_on(race(pending::<i32>(), ready(()))), None);
        // a result ready together with the deadline is kept
        assert_eq!(block_on(race(ready(2), ready(()))), Some(2));
    }
}
