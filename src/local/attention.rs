//! Self-attention per (sequence, head) on unpadded sequences: scores = Q·Kᵀ
//! (the scale is already in Q), a softmax per row, then scores · V into the
//! head's columns of the context. Queries, keys and values are read in place
//! from the fused (tokens, 3 · hidden) projection.

use std::sync::Mutex;

use super::gemm::{self, Isa, Packed};
use super::ops;
use super::pool::{self, Par, Shared};

/// Per-thread buffers.
pub(crate) struct Scratch {
    scores: Vec<f32>,
    inv: Vec<f32>,
    kt: Packed,
    v: Packed,
}

impl Scratch {
    pub(crate) fn new() -> Self {
        let isa = Isa::detect();
        Self {
            scores: Vec::new(),
            inv: Vec::new(),
            kt: Packed::new(isa),
            v: Packed::new(isa),
        }
    }
}

// from this length on, a head's two products go to Accelerate
#[cfg(target_os = "macos")]
const ACCELERATE_MIN: usize = 64;

/// Which queries attend.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Queries {
    /// Every token; its output goes to the token's own row of `ctx`.
    All,
    /// The first token of each sequence; sequence k's output goes to row k.
    First,
}

/// `ctx` (rows, hidden) gets each head's output; it must be zero where the
/// own kernels accumulate. `seqs` holds (first token, length) per sequence;
/// `h` is the hidden size.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    qkv: &[f32],
    seqs: &[(usize, usize)],
    h: usize,
    heads: usize,
    queries: Queries,
    ctx: &mut [f32],
    par: Par,
    scratch: &[Mutex<Scratch>],
) {
    let (w, d) = (3 * h, h / heads);
    let total: usize = seqs.iter().map(|s| s.1).sum();
    let rows = match queries {
        Queries::All => total,
        Queries::First => seqs.len(),
    };
    assert!(seqs.iter().all(|&(o, s)| s > 0 && o + s <= total));
    assert!(qkv.len() >= total * w && ctx.len() >= rows * h && scratch.len() >= pool::width(par));
    let out = Shared(ctx.as_mut_ptr());
    pool::for_each(par, seqs.len() * heads, &|t, i| {
        let ((o, s), hd) = (seqs[i / heads], i % heads);
        // query rows of this sequence, and the first output row
        let (m, row) = match queries {
            Queries::All => (s, o),
            Queries::First => (1, i / heads),
        };
        let mut guard = scratch[t].lock().unwrap_or_else(|e| e.into_inner());
        let sc = &mut *guard;
        let q = &qkv[o * w + hd * d..];
        let k = &qkv[o * w + h + hd * d..];
        let v = &qkv[o * w + 2 * h + hd * d..];
        sc.scores.clear();
        sc.scores.resize(m * s, 0.0);
        sc.inv.resize(m, 0.0);
        // safety: this item alone writes rows row..row+m, columns hd*d..hd*d+d
        let c = unsafe { out.get().add(row * h + hd * d) };
        #[cfg(target_os = "macos")]
        if s >= ACCELERATE_MIN {
            use gemm::accelerate::{sgemm, sgemm_ptr};
            sgemm(true, m, s, d, q, w, k, w, 0.0, &mut sc.scores, s);
            ops::softmax_rows(&mut sc.scores, s, &mut sc.inv);
            unsafe {
                sgemm_ptr(false, m, d, s, &sc.scores, s, v, w, 0.0, c, h);
                scale_rows(c, h, d, &sc.inv);
            }
            return;
        }
        sc.kt.pack_t(k, w, s, d);
        gemm::gemm(m, q, w, &sc.kt, &mut sc.scores, s, None);
        ops::softmax_rows(&mut sc.scores, s, &mut sc.inv);
        sc.v.pack(v, w, s, d);
        unsafe {
            gemm::gemm_ptr(m, &sc.scores, s, &sc.v, c, h, None);
            scale_rows(c, h, d, &sc.inv);
        }
    });
}

/// Row i of the (rows, d) block at `c` (row stride `ld`) times `inv[i]`.
///
/// # Safety
/// `c` must be valid for `inv.len()` rows of `d` values at stride `ld`, and
/// no other thread may touch them.
unsafe fn scale_rows(c: *mut f32, ld: usize, d: usize, inv: &[f32]) {
    for (i, &f) in inv.iter().enumerate() {
        // safety: forwarded from the caller
        let row = unsafe { std::slice::from_raw_parts_mut(c.add(i * ld), d) };
        row.iter_mut().for_each(|v| *v *= f);
    }
}
