//! Single-precision matrix products. The weight matrices of a linear layer go
//! to Accelerate's `cblas_sgemm` on macOS and to our own register-blocked
//! kernels elsewhere; attention uses the own kernels everywhere for short
//! sequences.
//!
//! Own kernels: B is packed into panels of NR columns, each (k, NR)
//! contiguous and zero-padded; A is read in place with broadcast loads, so it
//! needs no packing. C gets the product, plus a bias if asked; it is read
//! back only between k blocks.

#[cfg(target_os = "macos")]
pub(crate) mod accelerate;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
mod packed;
mod portable;
mod tile;

pub(crate) use packed::Packed;

use super::pool::{self, Par, Shared};

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

    fn kc(self) -> usize {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => neon::KC,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => avx2::KC,
            Self::Portable => portable::KC,
        }
    }
}

/// What the product starts from.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Init<'a> {
    /// Zero: C = A · B.
    Zero,
    /// A bias row, n values: C = bias + A · B.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    Bias(&'a [f32]),
}

/// How a kernel starts: from C (every k block after the first), from zero,
/// or from the bias.
pub(crate) const ADD: u8 = 0;
pub(crate) const ZERO: u8 = 1;
pub(crate) const BIAS: u8 = 2;

// products under this many multiply-adds are not worth splitting across threads
const PARALLEL_MIN: usize = 1 << 18;

/// C (m, n; row stride ldc) = A (m, k; row stride lda) · B, plus a bias if [`Init`] says so.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    m: usize,
    a: &[f32],
    lda: usize,
    b: &Packed,
    c: &mut [f32],
    ldc: usize,
    init: Init<'_>,
    par: Par,
) {
    if m == 0 {
        return;
    }
    assert!(ldc >= b.n && c.len() >= (m - 1) * ldc + b.n);
    // safety: c covers the rows and columns just checked
    unsafe { gemm_ptr(m, a, lda, b, c.as_mut_ptr(), ldc, init, par) }
}

/// [`gemm`] into C at a raw pointer, for callers that write disjoint column
/// ranges of the same rows from several threads.
///
/// # Safety
/// `c` must be valid for writes at `i * ldc + j` for every `i < m`, `j < n`,
/// and no other thread may touch those elements during the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_ptr(
    m: usize,
    a: &[f32],
    lda: usize,
    b: &Packed,
    c: *mut f32,
    ldc: usize,
    init: Init<'_>,
    par: Par,
) {
    let (k, n) = (b.k, b.n);
    if m == 0 {
        return;
    }
    assert!(lda >= k && ldc >= n && a.len() >= (m - 1) * lda + k);
    let (a, cs) = (a.as_ptr() as usize, Shared(c));
    let (mr, np) = (b.isa.mr(), n.div_ceil(b.isa.nr()));
    let (start, bias) = match init {
        Init::Zero => (ZERO, 0),
        Init::Bias(v) => {
            assert!(v.len() >= n);
            (BIAS, v.as_ptr() as usize)
        }
    };
    // safety: the asserts above bound every row and column the tiles touch
    let tile = |r0, r1, p0, p1| unsafe {
        let s = (start, bias as *const f32);
        tile(b, a as *const f32, lda, cs.get(), ldc, s, r0, r1, p0, p1)
    };
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
    start: (u8, *const f32),
    r0: usize,
    r1: usize,
    p0: usize,
    p1: usize,
) {
    let (d, k, n) = (&b.data, b.k, b.n);
    // safety: forwarded from `gemm`
    unsafe {
        match b.isa {
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => neon::tile(d, k, n, a, lda, c, ldc, start, r0, r1, p0, p1),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => avx2::tile(d, k, n, a, lda, c, ldc, start, r0, r1, p0, p1),
            Isa::Portable => portable::tile(d, k, n, a, lda, c, ldc, start, r0, r1, p0, p1),
        }
    }
}

/// A linear layer's weight, given as (n outputs, k inputs) the way PyTorch
/// stores it, kept as Wᵀ (k, n) for Accelerate on macOS and in panels for the
/// own kernels elsewhere. Accelerate reads a transposed operand several times
/// slower for the short inputs of one text, so the copy is worth its memory.
pub(crate) struct Weight {
    n: usize,
    k: usize,
    #[cfg(target_os = "macos")]
    w: Vec<f32>,
    #[cfg(not(target_os = "macos"))]
    w: Packed,
}

impl Weight {
    pub(crate) fn new(w: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k);
        #[cfg(target_os = "macos")]
        {
            let mut t = vec![0f32; n * k];
            for (o, row) in w.chunks_exact(k).enumerate() {
                for (i, &v) in row.iter().enumerate() {
                    t[i * n + o] = v;
                }
            }
            Self { n, k, w: t }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let mut p = Packed::new(Isa::detect());
            p.pack_t(w, k, n, k);
            Self { n, k, w: p }
        }
    }
}

/// C (m, n) = bias + A (m, k) · Wᵀ.
pub(crate) fn linear(a: &[f32], m: usize, w: &Weight, bias: &[f32], c: &mut [f32], par: Par) {
    assert!(a.len() >= m * w.k && c.len() >= m * w.n && bias.len() == w.n);
    #[cfg(target_os = "macos")]
    {
        super::ops::prefill(par, &mut c[..m * w.n], bias);
        accelerate::sgemm(false, m, w.n, w.k, a, w.k, &w.w, w.n, 1.0, c, w.n);
    }
    #[cfg(not(target_os = "macos"))]
    gemm(m, a, w.k, &w.w, c, w.n, Init::Bias(bias), par);
}

#[cfg(test)]
mod tests;
