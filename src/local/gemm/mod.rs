//! Single-precision matrix products. The weight matrices of a linear layer go
//! to Accelerate's `cblas_sgemm` on macOS and to our own register-blocked
//! kernels elsewhere; attention uses the own kernels everywhere for short
//! sequences.
//!
//! Own kernels: B is packed into panels of NR columns, each (k, NR)
//! contiguous and zero-padded; A is read in place with broadcast loads, so it
//! needs no packing. C is accumulated into, never overwritten.

#[cfg(target_os = "macos")]
pub(crate) mod accelerate;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
mod portable;

use super::pool::{self, Par, Shared};
use super::safetensors::F32s;

/// The kernel family a packed matrix was laid out for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Isa {
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Portable,
}

impl Isa {
    pub(crate) fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        return Self::Neon;
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Self::Avx2;
        }
        #[allow(unreachable_code)]
        Self::Portable
    }

    fn nr(self) -> usize {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => neon::NR,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => avx2::NR,
            Self::Portable => portable::NR,
        }
    }

    fn mr(self) -> usize {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => neon::MR,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => avx2::MR,
            Self::Portable => portable::MR,
        }
    }
}

/// B (k, n) in panels for one kernel family; reusable as a buffer.
pub(crate) struct Packed {
    isa: Isa,
    data: Vec<f32>,
    k: usize,
    n: usize,
}

impl Packed {
    pub(crate) fn new(isa: Isa) -> Self {
        Self {
            isa,
            data: Vec::new(),
            k: 0,
            n: 0,
        }
    }

    fn fill(&mut self, k: usize, n: usize, at: impl Fn(usize, usize) -> f32) {
        let nr = self.isa.nr();
        let panels = n.div_ceil(nr);
        self.data.clear();
        self.data.reserve(panels * k * nr);
        for p in 0..panels {
            for kk in 0..k {
                for j in p * nr..p * nr + nr {
                    self.data.push(if j < n { at(kk, j) } else { 0.0 });
                }
            }
        }
        self.k = k;
        self.n = n;
    }

    /// B[kk][j] = src[kk * ld + j].
    pub(crate) fn pack(&mut self, src: &[f32], ld: usize, k: usize, n: usize) {
        assert!(k > 0 && n > 0 && src.len() >= (k - 1) * ld + n);
        self.fill(k, n, |kk, j| src[kk * ld + j]);
    }

    /// B[kk][j] = src[j * ld + kk]: B given as its transpose, n rows of k.
    pub(crate) fn pack_t(&mut self, src: &[f32], ld: usize, n: usize, k: usize) {
        assert!(k > 0 && n > 0 && src.len() >= (n - 1) * ld + k);
        self.fill(k, n, |kk, j| src[j * ld + kk]);
    }
}

// products under this many multiply-adds are not worth splitting across threads
const PARALLEL_MIN: usize = 1 << 18;

/// C (m, n; row stride ldc) += A (m, k; row stride lda) · B.
pub(crate) fn gemm(
    m: usize,
    a: &[f32],
    lda: usize,
    b: &Packed,
    c: &mut [f32],
    ldc: usize,
    par: Par,
) {
    if m == 0 {
        return;
    }
    assert!(ldc >= b.n && c.len() >= (m - 1) * ldc + b.n);
    // safety: c covers the rows and columns just checked
    unsafe { gemm_ptr(m, a, lda, b, c.as_mut_ptr(), ldc, par) }
}

/// [`gemm`] into C at a raw pointer, for callers that write disjoint column
/// ranges of the same rows from several threads.
///
/// # Safety
/// `c` must be valid for writes at `i * ldc + j` for every `i < m`, `j < n`,
/// and no other thread may touch those elements during the call.
pub(crate) unsafe fn gemm_ptr(
    m: usize,
    a: &[f32],
    lda: usize,
    b: &Packed,
    c: *mut f32,
    ldc: usize,
    par: Par,
) {
    let (k, n) = (b.k, b.n);
    if m == 0 {
        return;
    }
    assert!(lda >= k && ldc >= n && a.len() >= (m - 1) * lda + k);
    let (a, cs) = (a.as_ptr() as usize, Shared(c));
    let (mr, np) = (b.isa.mr(), n.div_ceil(b.isa.nr()));
    // safety: the asserts above bound every row and column the tiles touch
    let tile =
        |r0, r1, p0, p1| unsafe { tile(b, a as *const f32, lda, cs.get(), ldc, r0, r1, p0, p1) };
    let threads = pool::width(par);
    if threads == 1 || m * n * k < PARALLEL_MIN {
        return tile(0, m, 0, np);
    }
    let mc = mr * 8;
    let blocks = m.div_ceil(mc);
    let per = (np * blocks).div_ceil(threads * 4).clamp(1, np);
    let groups = np.div_ceil(per);
    pool::for_each(par, blocks * groups, &|_, i| {
        let (r0, p0) = (i / groups * mc, i % groups * per);
        tile(r0, (r0 + mc).min(m), p0, (p0 + per).min(np));
    });
}

/// Rows r0..r1 of C, column panels p0..p1.
#[allow(clippy::too_many_arguments)]
unsafe fn tile(
    b: &Packed,
    a: *const f32,
    lda: usize,
    c: *mut f32,
    ldc: usize,
    r0: usize,
    r1: usize,
    p0: usize,
    p1: usize,
) {
    // safety: forwarded from `gemm`
    unsafe {
        match b.isa {
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => neon::tile(&b.data, b.k, b.n, a, lda, c, ldc, r0, r1, p0, p1),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => avx2::tile(&b.data, b.k, b.n, a, lda, c, ldc, r0, r1, p0, p1),
            Isa::Portable => portable::tile(&b.data, b.k, b.n, a, lda, c, ldc, r0, r1, p0, p1),
        }
    }
}

/// The blocked loop over one tile, shared by every kernel family.
/// `$kernel::<R>` computes R rows by NR columns of C in place, so a partial
/// row block costs only its rows; a partial column panel goes through a
/// scratch tile.
macro_rules! tile_loop {
    ($kernel:ident, $mr:expr, $nr:expr, $kc:expr $(, $feature:literal)?) => {
        #[allow(clippy::too_many_arguments)]
        $(#[target_feature(enable = $feature)])?
        pub(super) unsafe fn tile(data: &[f32], k: usize, n: usize, a: *const f32, lda: usize, c: *mut f32, ldc: usize, r0: usize, r1: usize, p0: usize, p1: usize) {
            // safety (whole body): the caller bounds rows r0..r1 of a and c
            // and columns of c up to n; a kernel touches only `rows` rows
            let run = |rows: usize, kc: usize, ap: &[*const f32; 8], bp: *const f32, c: *mut f32, ldc: usize| unsafe {
                match rows {
                    1 => $kernel::<1>(kc, ap, bp, c, ldc),
                    2 => $kernel::<2>(kc, ap, bp, c, ldc),
                    3 => $kernel::<3>(kc, ap, bp, c, ldc),
                    4 => $kernel::<4>(kc, ap, bp, c, ldc),
                    5 if $mr >= 5 => $kernel::<5>(kc, ap, bp, c, ldc),
                    6 if $mr >= 6 => $kernel::<6>(kc, ap, bp, c, ldc),
                    7 if $mr >= 7 => $kernel::<7>(kc, ap, bp, c, ldc),
                    8 if $mr >= 8 => $kernel::<8>(kc, ap, bp, c, ldc),
                    _ => unreachable!("row block larger than the kernel"),
                }
            };
            let mut kc0 = 0;
            while kc0 < k {
                let kc = ($kc).min(k - kc0);
                for p in p0..p1 {
                    let cols = ($nr).min(n - p * $nr);
                    let bp = unsafe { data.as_ptr().add(p * k * $nr + kc0 * $nr) };
                    let mut r = r0;
                    while r < r1 {
                        let rows = ($mr).min(r1 - r);
                        let mut ap = [std::ptr::null(); 8];
                        for (i, q) in ap.iter_mut().take(rows).enumerate() {
                            *q = unsafe { a.add((r + i) * lda + kc0) };
                        }
                        let cp = unsafe { c.add(r * ldc + p * $nr) };
                        if cols == $nr {
                            run(rows, kc, &ap, bp, cp, ldc);
                        } else {
                            let mut t = [0f32; $mr * $nr];
                            for i in 0..rows {
                                for j in 0..cols {
                                    t[i * $nr + j] = unsafe { *cp.add(i * ldc + j) };
                                }
                            }
                            run(rows, kc, &ap, bp, t.as_mut_ptr(), $nr);
                            for i in 0..rows {
                                for j in 0..cols {
                                    unsafe { *cp.add(i * ldc + j) = t[i * $nr + j] };
                                }
                            }
                        }
                        r += $mr;
                    }
                }
                kc0 += $kc;
            }
        }
    };
}
pub(crate) use tile_loop;

/// A linear layer's weight, (n outputs, k inputs) as stored by PyTorch.
pub(crate) struct Weight {
    n: usize,
    k: usize,
    #[cfg(target_os = "macos")]
    w: F32s,
    #[cfg(not(target_os = "macos"))]
    w: Packed,
}

impl Weight {
    pub(crate) fn new(w: F32s, n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k);
        #[cfg(target_os = "macos")]
        return Self { n, k, w };
        #[cfg(not(target_os = "macos"))]
        {
            let mut p = Packed::new(Isa::detect());
            p.pack_t(&w, k, n, k);
            Self { n, k, w: p }
        }
    }
}

/// C (m, n) += A (m, k) · Wᵀ.
pub(crate) fn linear(a: &[f32], m: usize, w: &Weight, c: &mut [f32], par: Par) {
    assert!(a.len() >= m * w.k && c.len() >= m * w.n);
    #[cfg(target_os = "macos")]
    {
        let _ = par;
        accelerate::sgemm(true, m, w.n, w.k, a, w.k, &w.w, w.k, 1.0, c, w.n);
    }
    #[cfg(not(target_os = "macos"))]
    gemm(m, a, w.k, &w.w, c, w.n, par);
}

#[cfg(test)]
mod tests;
