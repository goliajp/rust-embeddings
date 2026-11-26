//! Worker threads for one forward pass, started in a `std::thread::scope`.
//! Work is a count of items that the caller and the workers claim one at a
//! time, so a slow core (an efficiency core, a busy neighbour) only takes
//! fewer items. Between jobs the workers spin briefly, then park.

use std::cell::UnsafeCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use std::thread::Thread;
use std::time::{Duration, Instant};

/// An item runner: `None` runs everything on the calling thread.
pub(crate) type Par<'a> = Option<&'a Pool>;

type Job = *const (dyn Fn(usize) + Sync);

pub(crate) struct Pool {
    threads: usize,
    generation: AtomicUsize,
    done: AtomicUsize,
    sleepers: AtomicUsize,
    stop: AtomicBool,
    panicked: AtomicBool,
    job: UnsafeCell<Option<Job>>,
    workers: OnceLock<Vec<Thread>>,
}

// safety: `job` is written only by the owner between jobs, while every
// worker is idle, and read by workers only during a job
unsafe impl Sync for Pool {}
unsafe impl Send for Pool {}

// how long an idle worker spins before parking
const SPIN: Duration = Duration::from_micros(50);

/// Threads for one pass: `EMBEDRS_THREADS` if set, else the performance
/// cores on macOS, else what the OS gives this process.
pub(crate) fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if let Some(n) = std::env::var("EMBEDRS_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            return n.max(1);
        }
        #[cfg(target_os = "macos")]
        if let Some(n) = performance_cores() {
            return n;
        }
        std::thread::available_parallelism().map_or(1, |n| n.get())
    })
}

#[cfg(target_os = "macos")]
fn performance_cores() -> Option<usize> {
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const u8,
            old: *mut u8,
            oldlen: *mut usize,
            new: *const u8,
            newlen: usize,
        ) -> i32;
    }
    let mut n: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    // safety: a nul-terminated name and a buffer of the length passed
    let r = unsafe {
        sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr() as *const u8,
            &mut n as *mut i32 as *mut u8,
            &mut len,
            std::ptr::null(),
            0,
        )
    };
    (r == 0 && n > 0).then_some(n as usize)
}

/// Run `f` with `threads` threads (the caller included) available as a pool.
pub(crate) fn scoped<R>(threads: usize, f: impl FnOnce(Par) -> R) -> R {
    if threads <= 1 {
        return f(None);
    }
    let pool = Pool {
        threads,
        generation: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        sleepers: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
        panicked: AtomicBool::new(false),
        job: UnsafeCell::new(None),
        workers: OnceLock::new(),
    };
    std::thread::scope(|s| {
        let handles: Vec<Thread> = (1..threads)
            .map(|i| {
                let p = &pool;
                s.spawn(move || p.work(i)).thread().clone()
            })
            .collect();
        pool.workers.set(handles).expect("set once");
        let _stop = Stop(&pool);
        f(Some(&pool))
    })
}

// ends the workers when the pass is over, also when it unwinds, so the scope can join them
struct Stop<'a>(&'a Pool);

impl Drop for Stop<'_> {
    fn drop(&mut self) {
        self.0.stop.store(true, Release);
        self.0.generation.fetch_add(1, SeqCst);
        for t in self.0.workers.get().into_iter().flatten() {
            t.unpark();
        }
    }
}

// counts a worker's job as done even when it panicked, so the owner never waits forever
struct Done<'a>(&'a Pool);

impl Drop for Done<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.panicked.store(true, Release);
        }
        self.0.done.fetch_add(1, Release);
    }
}

impl Pool {
    pub(crate) fn threads(&self) -> usize {
        self.threads
    }

    fn work(&self, index: usize) {
        let mut seen = 0;
        loop {
            let start = Instant::now();
            let mut spins = 0u32;
            while self.generation.load(Acquire) == seen {
                spins = spins.wrapping_add(1);
                if !spins.is_multiple_of(64) || start.elapsed() < SPIN {
                    std::hint::spin_loop();
                    continue;
                }
                self.sleepers.fetch_add(1, SeqCst);
                if self.generation.load(SeqCst) == seen {
                    std::thread::park();
                }
                self.sleepers.fetch_sub(1, SeqCst);
            }
            seen = self.generation.load(Acquire);
            if self.stop.load(Acquire) {
                return;
            }
            let _done = Done(self);
            // safety: set by `run` before the generation changed, and kept
            // alive until every worker has counted itself done
            let job = unsafe { (*self.job.get()).expect("a job") };
            unsafe { (*job)(index) };
        }
    }

    /// Call `f(thread index)` on every thread of the pool, the caller as 0.
    fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        // safety: no worker reads `job` until the generation changes below;
        // the pointer is not used after every worker reported done
        unsafe { *self.job.get() = Some(std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f)) };
        self.done.store(0, Relaxed);
        self.generation.fetch_add(1, SeqCst);
        if self.sleepers.load(SeqCst) > 0 {
            for t in self.workers.get().into_iter().flatten() {
                t.unpark();
            }
        }
        f(0);
        while self.done.load(Acquire) != self.threads - 1 {
            std::hint::spin_loop();
        }
        assert!(!self.panicked.load(Acquire), "a worker thread panicked");
    }
}

/// `f(thread, item)` for every item in `0..n`, items claimed in order.
pub(crate) fn for_each(par: Par, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match par {
        Some(pool) if n > 1 => {
            let next = AtomicUsize::new(0);
            pool.run(&|t| {
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= n {
                        break;
                    }
                    f(t, i);
                }
            });
        }
        _ => (0..n).for_each(|i| f(0, i)),
    }
}

/// Number of threads `par` can use.
pub(crate) fn width(par: Par) -> usize {
    par.map_or(1, Pool::threads)
}

/// A raw pointer that may cross threads; users write disjoint ranges.
#[derive(Clone, Copy)]
pub(crate) struct Shared(pub(crate) *mut f32);
// safety: every user writes ranges no other item touches
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    // a method, so closures capture the whole Shared rather than its raw field
    pub(crate) fn get(self) -> *mut f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_item_runs_once() {
        for threads in [1, 2, 5] {
            scoped(threads, |par| {
                for n in [0, 1, 7, 1000] {
                    let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                    for_each(par, n, &|t, i| {
                        assert!(t < threads);
                        hits[i].fetch_add(1, Relaxed);
                    });
                    assert!(hits.iter().all(|h| h.load(Relaxed) == 1));
                }
                // long enough between jobs for the workers to park
                std::thread::sleep(Duration::from_millis(2));
                let sum = AtomicUsize::new(0);
                for_each(par, 100, &|_, i| {
                    sum.fetch_add(i, Relaxed);
                });
                assert_eq!(sum.load(Relaxed), 4950);
            });
        }
    }

    #[test]
    #[should_panic(expected = "a worker thread panicked")]
    fn a_panicking_item_fails_the_call() {
        scoped(3, |par| {
            for_each(par, 64, &|t, _| {
                if t == 1 {
                    panic!("boom");
                }
                std::thread::sleep(Duration::from_millis(1));
            })
        });
    }
}
