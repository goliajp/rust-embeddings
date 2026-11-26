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

#[inline(always)]
fn softmax_body(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut s = [0f32; 8];
    let (chunks, rest) = x.as_chunks_mut::<8>();
    for c in chunks {
        for i in 0..8 {
            c[i] = exp(c[i] - max);
            s[i] += c[i];
        }
    }
    let mut sum: f32 = 0.0;
    for v in rest {
        *v = exp(*v - max);
        sum += *v;
    }
    let inv = 1.0 / (sum + s.iter().sum::<f32>());
    for v in x {
        *v *= inv;
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
dispatch!(softmax, softmax_body, softmax_avx2, (x: &mut [f32]));
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
