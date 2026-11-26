//! Elementwise and row kernels in plain Rust that the compiler vectorises;
//! on x86 each also gets a copy compiled for AVX2 + FMA, picked at run time.

use super::pool::{self, Par, Shared};

/// erf on [-4, 4] as x·P(x²)/Q(x²) (Eigen's single-precision form); outside
/// that range erf is ±1 in f32. Max error 4e-7.
#[inline(always)]
pub(crate) fn erf(x: f32) -> f32 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    let mut p = -2.726_142_4e-10_f32;
    p = p.mul_add(x2, 2.770_681_5e-8);
    p = p.mul_add(x2, -2.101_024e-6);
    p = p.mul_add(x2, -5.692_506_2e-5);
    p = p.mul_add(x2, -7.349_906_4e-4);
    p = p.mul_add(x2, -2.954_6e-3);
    p = p.mul_add(x2, -1.609_603_3e-2);
    let mut q = -1.456_607_15e-5_f32;
    q = q.mul_add(x2, -2.133_740_5e-4);
    q = q.mul_add(x2, -1.682_827e-3);
    q = q.mul_add(x2, -7.373_329_3e-3);
    q = q.mul_add(x2, -1.426_473_9e-2);
    x * p / q
}

/// GELU in its exact form, x·Φ(x); the four models were trained with it.
#[inline(always)]
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// e^x for x <= 88.7 (Cephes' single-precision form); max relative error 1e-7.
#[inline(always)]
pub(crate) fn exp(x: f32) -> f32 {
    let x = x.clamp(-87.3, 88.7);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = n.mul_add(-0.693_359_4, x);
    let r = n.mul_add(2.121_944_4e-4, r);
    let mut p = 1.987_569_1e-4_f32;
    p = p.mul_add(r, 1.398_199_9e-3);
    p = p.mul_add(r, 8.333_452e-3);
    p = p.mul_add(r, 4.166_579_6e-2);
    p = p.mul_add(r, 1.666_666_6e-1);
    p = p.mul_add(r, 0.5);
    let y = (p * r).mul_add(r, r) + 1.0;
    // safety: n is in [-126, 128] after the clamp
    y * f32::from_bits(((unsafe { n.to_int_unchecked::<i32>() } + 127) << 23) as u32)
}

#[inline(always)]
fn gelu_body(x: &mut [f32]) {
    for v in x {
        *v = gelu(*v);
    }
}

// v = e^(v - max) over a slice. The compiler leaves `exp` scalar in a
// loop, so the vector versions spell out the same operations in the same
// order, and give the same bits
fn exp_minus(x: &mut [f32], max: f32) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // safety: the features were just detected
        return unsafe { exp_minus_avx2(x, max) };
    }
    #[cfg(target_arch = "aarch64")]
    return exp_minus_neon(x, max);
    #[allow(unreachable_code)]
    for v in x.iter_mut() {
        *v = exp(*v - max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn exp_minus_avx2(x: &mut [f32], max: f32) {
    use std::arch::x86_64::*;
    let (chunks, rest) = x.as_chunks_mut::<8>();
    let k = _mm256_set1_ps;
    let m = k(max);
    for c in chunks {
        // safety: c is 8 values
        let v = _mm256_sub_ps(unsafe { _mm256_loadu_ps(c.as_ptr()) }, m);
        let v = _mm256_min_ps(_mm256_max_ps(v, k(-87.3)), k(88.7));
        let n = _mm256_floor_ps(_mm256_add_ps(
            _mm256_mul_ps(v, k(std::f32::consts::LOG2_E)),
            k(0.5),
        ));
        let r = _mm256_fmadd_ps(n, k(-0.693_359_4), v);
        let r = _mm256_fmadd_ps(n, k(2.121_944_4e-4), r);
        let mut p = k(1.987_569_1e-4);
        for c in [
            1.398_199_9e-3,
            8.333_452e-3,
            4.166_579_6e-2,
            1.666_666_6e-1,
            0.5,
        ] {
            p = _mm256_fmadd_ps(p, r, k(c));
        }
        let y = _mm256_add_ps(_mm256_fmadd_ps(_mm256_mul_ps(p, r), r, r), k(1.0));
        let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(
            _mm256_cvttps_epi32(n),
            _mm256_set1_epi32(127),
        ));
        // safety: c is 8 values
        unsafe { _mm256_storeu_ps(c.as_mut_ptr(), _mm256_mul_ps(y, _mm256_castsi256_ps(e))) };
    }
    for v in rest {
        *v = exp(*v - max);
    }
}

#[cfg(target_arch = "aarch64")]
fn exp_minus_neon(x: &mut [f32], max: f32) {
    use std::arch::aarch64::*;
    let (chunks, rest) = x.as_chunks_mut::<4>();
    // safety (whole loop): NEON is part of aarch64, and c is 4 values
    unsafe {
        let k = vdupq_n_f32;
        let m = k(max);
        for c in chunks {
            let v = vsubq_f32(vld1q_f32(c.as_ptr()), m);
            let v = vminq_f32(vmaxq_f32(v, k(-87.3)), k(88.7));
            let n = vrndmq_f32(vaddq_f32(vmulq_f32(v, k(std::f32::consts::LOG2_E)), k(0.5)));
            let r = vfmaq_f32(v, n, k(-0.693_359_4));
            let r = vfmaq_f32(r, n, k(2.121_944_4e-4));
            let mut p = k(1.987_569_1e-4);
            for c in [
                1.398_199_9e-3,
                8.333_452e-3,
                4.166_579_6e-2,
                1.666_666_6e-1,
                0.5,
            ] {
                p = vfmaq_f32(k(c), p, r);
            }
            let y = vaddq_f32(vfmaq_f32(r, vmulq_f32(p, r), r), k(1.0));
            let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
            vst1q_f32(c.as_mut_ptr(), vmulq_f32(y, vreinterpretq_f32_s32(e)));
        }
    }
    for v in rest {
        *v = exp(*v - max);
    }
}

/// Each row of `x` (row length `n`) becomes e^(v - max of the row); the
/// row's 1 / sum goes to `inv`. The caller scales whatever it computes from
/// a row instead of the row itself, which saves a pass over it.
#[inline(always)]
fn softmax_rows_body(x: &mut [f32], n: usize, inv: &mut [f32]) {
    for (row, inv) in x.chunks_exact_mut(n).zip(inv.iter_mut()) {
        let (chunks, rest) = row.as_chunks::<8>();
        let mut m = [f32::NEG_INFINITY; 8];
        for c in chunks {
            for i in 0..8 {
                m[i] = if c[i] > m[i] { c[i] } else { m[i] };
            }
        }
        let mut max = rest
            .iter()
            .fold(f32::NEG_INFINITY, |a, &v| if v > a { v } else { a });
        for v in m {
            max = if v > max { v } else { max };
        }
        exp_minus(row, max);
        let mut s = [0f32; 8];
        let (chunks, rest) = row.as_chunks::<8>();
        for c in chunks {
            for i in 0..8 {
                s[i] += c[i];
            }
        }
        let sum: f32 = rest.iter().sum();
        *inv = 1.0 / (sum + s.iter().sum::<f32>());
    }
}

#[inline(always)]
fn layernorm_body(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    layernorm_rows(x, None, w, b, eps)
}

#[inline(always)]
fn add_layernorm_body(x: &mut [f32], residual: &[f32], w: &[f32], b: &[f32], eps: f32) {
    layernorm_rows(x, Some(residual), w, b, eps)
}

/// Rows of `x` normalised, `residual` added to them first.
#[inline(always)]
fn layernorm_rows(x: &mut [f32], residual: Option<&[f32]>, w: &[f32], b: &[f32], eps: f32) {
    for (k, row) in x.chunks_exact_mut(w.len()).enumerate() {
        if let Some(r) = residual {
            for (c, r) in row.iter_mut().zip(&r[k * w.len()..(k + 1) * w.len()]) {
                *c += r;
            }
        }
        let mut s = [0f32; 8];
        for c in row.as_chunks::<8>().0 {
            for i in 0..8 {
                s[i] += c[i];
            }
        }
        let mean = s.iter().sum::<f32>() / w.len() as f32;
        let mut v = [0f32; 8];
        for c in row.as_chunks::<8>().0 {
            for i in 0..8 {
                let d = c[i] - mean;
                v[i] = d.mul_add(d, v[i]);
            }
        }
        let inv = 1.0 / (v.iter().sum::<f32>() / w.len() as f32 + eps).sqrt();
        for ((c, w), b) in row
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip(w.as_chunks::<8>().0)
            .zip(b.as_chunks::<8>().0)
        {
            for i in 0..8 {
                c[i] = ((c[i] - mean) * inv).mul_add(w[i], b[i]);
            }
        }
    }
}

macro_rules! dispatch {
    ($name:ident, $body:ident, $avx:ident, ($($arg:ident: $ty:ty),*)) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        fn $avx($($arg: $ty),*) {
            $body($($arg),*)
        }

        pub(crate) fn $name($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // safety: the features were just detected
                return unsafe { $avx($($arg),*) };
            }
            $body($($arg),*)
        }
    };
}

dispatch!(gelu_slice, gelu_body, gelu_avx2, (x: &mut [f32]));
dispatch!(softmax_rows, softmax_rows_body, softmax_rows_avx2, (x: &mut [f32], n: usize, inv: &mut [f32]));
dispatch!(layernorm, layernorm_body, layernorm_avx2, (x: &mut [f32], w: &[f32], b: &[f32], eps: f32));
dispatch!(add_layernorm, add_layernorm_body, add_layernorm_avx2, (x: &mut [f32], residual: &[f32], w: &[f32], b: &[f32], eps: f32));

// rows per parallel item: about 32 K values
fn block_rows(width: usize) -> usize {
    (32_768 / width).max(1)
}

/// `f(block, first row)` over blocks of whole rows of `x` (row length `width`), in parallel.
pub(crate) fn rows(par: Par, x: &mut [f32], width: usize, f: &(dyn Fn(&mut [f32], usize) + Sync)) {
    let m = x.len() / width;
    let per = block_rows(width);
    let blocks = m.div_ceil(per);
    if pool::width(par) == 1 || blocks == 1 {
        return f(x, 0);
    }
    let base = Shared(x.as_mut_ptr());
    pool::for_each(par, blocks, &|_, i| {
        let r0 = i * per;
        let r1 = (r0 + per).min(m);
        // safety: blocks are disjoint row ranges of x
        let block = unsafe {
            std::slice::from_raw_parts_mut(base.get().add(r0 * width), (r1 - r0) * width)
        };
        f(block, r0);
    });
}

/// Every row of `c` (m, n) set to `bias`.
pub(crate) fn prefill(par: Par, c: &mut [f32], bias: &[f32]) {
    let n = bias.len();
    rows(par, c, n, &|block, _| {
        block
            .chunks_exact_mut(n)
            .for_each(|row| row.copy_from_slice(bias))
    });
}

#[cfg(test)]
#[path = "ops_tests.rs"]
mod tests;
