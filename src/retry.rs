//! Retry policy: exponential backoff with jitter, and `Retry-After`.

use std::hash::BuildHasher;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Exponential backoff for retryable failures: 408, 429, 500, 502, 503,
/// 504, 529 and connection errors.
///
/// ```rust
/// use std::time::Duration;
/// let b = embedrs::Backoff::new().retries(5).base(Duration::from_millis(200));
/// assert_eq!(b.jitter(false).delay(1), Duration::from_millis(400));
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backoff {
    base: Duration,
    max: Duration,
    jitter: bool,
    retries: u32,
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

impl Backoff {
    /// 3 retries, 500 ms doubling up to 30 s, with jitter.
    pub fn new() -> Self {
        Self {
            base: Duration::from_millis(500),
            max: Duration::from_secs(30),
            jitter: true,
            retries: 3,
        }
    }

    /// Delay before the first retry.
    pub fn base(self, d: Duration) -> Self {
        Self { base: d, ..self }
    }

    /// Upper bound of any delay.
    pub fn max(self, d: Duration) -> Self {
        Self { max: d, ..self }
    }

    /// Multiply each delay by a random factor in [0.5, 1.0).
    pub fn jitter(self, on: bool) -> Self {
        Self { jitter: on, ..self }
    }

    /// Retries after the first attempt.
    pub fn retries(self, n: u32) -> Self {
        Self { retries: n, ..self }
    }

    /// Number of retries.
    pub fn max_retries(&self) -> u32 {
        self.retries
    }

    /// Delay before retry number `attempt` (0-based):
    /// `min(base * 2^attempt, max)`, times the jitter factor.
    pub fn delay(&self, attempt: u32) -> Duration {
        let capped = self
            .base
            .saturating_mul(1u32 << attempt.min(16))
            .min(self.max);
        if self.jitter {
            capped.mul_f64(0.5 + unit() * 0.5)
        } else {
            capped
        }
    }
}

// splitmix64 over a process-wide counter, seeded once; not cryptographic
fn unit() -> f64 {
    static STATE: OnceLock<AtomicU64> = OnceLock::new();
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
    let state = STATE.get_or_init(|| {
        let seed = std::hash::RandomState::new().hash_one(std::time::Instant::now());
        AtomicU64::new(seed)
    });
    let mut z = state
        .fetch_add(GAMMA, Ordering::Relaxed)
        .wrapping_add(GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// `Retry-After` as delta-seconds or an HTTP date (IMF-fixdate, RFC 850 or
/// asctime). A date in the past means now; anything else is `None`.
pub(crate) fn parse_retry_after(v: &str, now: SystemTime) -> Option<Duration> {
    let v = v.trim();
    if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) {
        return v.parse().ok().map(Duration::from_secs);
    }
    let at = UNIX_EPOCH + Duration::from_secs(http_date(v)?);
    Some(at.duration_since(now).unwrap_or(Duration::ZERO))
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Seconds since the epoch of an HTTP date in GMT.
fn http_date(v: &str) -> Option<u64> {
    let parts: Vec<&str> = v.split_ascii_whitespace().collect();
    let (day, month, year, time) = match parts.as_slice() {
        // Sun, 06 Nov 1994 08:49:37 GMT
        [_, d, m, y, t, "GMT"] => (*d, *m, y.parse::<i64>().ok()?, *t),
        // Sunday, 06-Nov-94 08:49:37 GMT
        [_, dmy, t, "GMT"] => {
            let mut it = dmy.split('-');
            let (d, m, y) = (it.next()?, it.next()?, it.next()?);
            if it.next().is_some() || y.len() != 2 {
                return None;
            }
            // two-digit years more than 50 years ahead are in the past
            let y: i64 = y.parse().ok()?;
            let year = if y < 70 { 2000 + y } else { 1900 + y };
            (d, m, year, *t)
        }
        // Sun Nov  6 08:49:37 1994
        [_, m, d, t, y] => (*d, *m, y.parse::<i64>().ok()?, *t),
        _ => return None,
    };
    let day: i64 = day.parse().ok().filter(|d| (1..=31).contains(d))?;
    let month = MONTHS.iter().position(|m| *m == month)? as i64 + 1;
    let mut hms = time.split(':').map(|x| x.parse::<i64>().ok());
    let (h, mi, s) = (hms.next()??, hms.next()??, hms.next()??);
    if hms.next().is_some() || h > 23 || mi > 59 || s > 60 || year < 1970 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    u64::try_from(days * 86_400 + h * 3600 + mi * 60 + s).ok()
}

// days since 1970-01-01 of a proleptic gregorian date
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_and_capped() {
        let b = Backoff::new().jitter(false);
        let ms = |n| Duration::from_millis(n);
        assert_eq!(b.delay(0), ms(500));
        assert_eq!(b.delay(1), ms(1000));
        assert_eq!(b.delay(3), ms(4000));
        assert_eq!(b.delay(10), Duration::from_secs(30));
        assert_eq!(b.delay(u32::MAX), Duration::from_secs(30));
        assert_eq!(b.max_retries(), 3);
        assert_eq!(Backoff::default(), Backoff::new());
    }

    #[test]
    fn jitter_stays_in_range_and_varies() {
        let b = Backoff::new();
        let delays: Vec<Duration> = (0..200).map(|_| b.delay(0)).collect();
        for d in &delays {
            assert!(*d >= Duration::from_millis(250) && *d < Duration::from_millis(500));
        }
        let distinct: std::collections::HashSet<_> = delays.iter().collect();
        assert!(distinct.len() > 150);
        for _ in 0..10_000 {
            let u = unit();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn retry_after_seconds_and_dates() {
        let now = UNIX_EPOCH + Duration::from_secs(784_111_777); // 1994-11-06 08:49:37
        let s = |n| Some(Duration::from_secs(n));
        assert_eq!(parse_retry_after("120", now), s(120));
        assert_eq!(parse_retry_after(" 0 ", now), s(0));
        for v in [
            "Sun, 06 Nov 1994 08:49:47 GMT",
            "Sunday, 06-Nov-94 08:49:47 GMT",
            "Sun Nov  6 08:49:47 1994",
        ] {
            assert_eq!(parse_retry_after(v, now), s(10), "{v}");
        }
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:00:00 GMT", now),
            s(0)
        );
        for bad in [
            "",
            "-1",
            "1.5",
            "soon",
            "Sun, 06 Foo 1994 08:49:37 GMT",
            "Sun, 06 Nov 1994 25:00:00 GMT",
        ] {
            assert_eq!(parse_retry_after(bad, now), None, "{bad}");
        }
    }

    #[test]
    fn civil_days() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        assert_eq!(days_from_civil(2026, 11, 26), 20_783);
    }
}
