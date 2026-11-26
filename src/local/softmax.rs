//! e^x and the attention softmax. The vector versions spell out the scalar
//! `exp` operation for operation, so every path gives the same bits.

// 1.5 · 2^23 + 127: adding it rounds to an integer, which lands in the low
// mantissa bits already biased as an exponent
const ROUND: f32 = 12_583_039.0;
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
const LO: f32 = -87.3;
#[cfg(test)]
const HI: f32 = 88.7;
// Cephes' single-precision polynomial, highest power first; it is
// evaluated in pairs (Estrin), which halves the chain of dependent
// operations a value waits on
const P: [f32; 6] = [
    1.987_569_1e-4,
    1.398_199_9e-3,
    8.333_452e-3,
    4.166_579_6e-2,
    1.666_666_6e-1,
    0.5,
];

/// e^x for x <= 88.7; max relative error 2e-7.
#[cfg(test)]
pub(crate) fn exp(x: f32) -> f32 {
    exp_clamped(x.clamp(LO, HI))
}

#[inline(always)]
fn exp_clamped(x: f32) -> f32 {
    let t = x.mul_add(std::f32::consts::LOG2_E, ROUND);
    let n = t - ROUND;
    let r = n.mul_add(-LN2_HI, x);
    let r = n.mul_add(-LN2_LO, r);
    let r2 = r * r;
    let (a, b, c) = (
        P[0].mul_add(r, P[1]),
        P[2].mul_add(r, P[3]),
        P[4].mul_add(r, P[5]),
    );
    let p = a.mul_add(r2, b).mul_add(r2, c);
    let y = p.mul_add(r2, r + 1.0);
    y * f32::from_bits(t.to_bits() << 23)
}

/// Each row of `x` (rows `stride` apart, the first `n` values counted)
/// becomes e^(v - max of the row); the row's 1 / sum goes to `inv`. The
/// caller scales whatever it computes from a row instead of the row itself,
/// which saves a pass over it. Values past `n` in a row are left unspecified.
pub(crate) fn softmax_rows(x: &mut [f32], n: usize, stride: usize, inv: &mut [f32]) {
    assert!(n > 0 && stride >= n && x.len() >= (inv.len().max(1) - 1) * stride + n);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // safety: the features were just detected, and the assert bounds every row
        return unsafe { avx2::rows(x, n, stride, inv) };
    }
    #[cfg(target_arch = "aarch64")]
    return neon::rows(x, n, stride, inv);
    #[allow(unreachable_code)]
    for (i, inv) in inv.iter_mut().enumerate() {
        let row = &mut x[i * stride..i * stride + n];
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = exp_clamped((*v - max).max(LO));
            sum += *v;
        }
        *inv = 1.0 / sum;
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    // lanes 8 - k.. of an unaligned load at MASK[k] are off
    static MASK: [i32; 16] = [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0];

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp8(v: __m256) -> __m256 {
        let k = _mm256_set1_ps;
        let t = _mm256_fmadd_ps(v, k(std::f32::consts::LOG2_E), k(ROUND));
        let n = _mm256_sub_ps(t, k(ROUND));
        let r = _mm256_fmadd_ps(n, k(-LN2_HI), v);
        let r = _mm256_fmadd_ps(n, k(-LN2_LO), r);
        let r2 = _mm256_mul_ps(r, r);
        let a = _mm256_fmadd_ps(k(P[0]), r, k(P[1]));
        let b = _mm256_fmadd_ps(k(P[2]), r, k(P[3]));
        let c = _mm256_fmadd_ps(k(P[4]), r, k(P[5]));
        let p = _mm256_fmadd_ps(_mm256_fmadd_ps(a, r2, b), r2, c);
        let y = _mm256_fmadd_ps(p, r2, _mm256_add_ps(r, k(1.0)));
        _mm256_mul_ps(
            y,
            _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_castps_si256(t))),
        )
    }

    /// # Safety
    /// AVX2 and FMA must be present; `x` must hold `inv.len()` rows of `n`
    /// values `stride` apart.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn rows(x: &mut [f32], n: usize, stride: usize, inv: &mut [f32]) {
        // safety (whole body): every access is below n in a row, or masked
        unsafe {
            let (full, tail) = (n / 8 * 8, n % 8);
            let mask = _mm256_loadu_si256(MASK.as_ptr().add(8 - tail) as *const __m256i);
            let lo = _mm256_set1_ps(LO);
            for (i, inv) in inv.iter_mut().enumerate() {
                let p = x.as_mut_ptr().add(i * stride);
                let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
                let mut m = [ninf; 4];
                let mut j = 0;
                while j + 32 <= full {
                    for (q, m) in m.iter_mut().enumerate() {
                        *m = _mm256_max_ps(*m, _mm256_loadu_ps(p.add(j + 8 * q)));
                    }
                    j += 32;
                }
                while j < full {
                    m[0] = _mm256_max_ps(m[0], _mm256_loadu_ps(p.add(j)));
                    j += 8;
                }
                let t = _mm256_blendv_ps(
                    ninf,
                    _mm256_maskload_ps(p.add(full), mask),
                    _mm256_castsi256_ps(mask),
                );
                let m = _mm256_max_ps(
                    _mm256_max_ps(m[0], m[1]),
                    _mm256_max_ps(m[2], _mm256_max_ps(m[3], t)),
                );
                let m = _mm256_max_ps(m, _mm256_permute2f128_ps::<1>(m, m));
                let m = _mm256_max_ps(m, _mm256_shuffle_ps::<0b01_00_11_10>(m, m));
                let m = _mm256_max_ps(m, _mm256_shuffle_ps::<0b10_11_00_01>(m, m));
                let e = |j: usize| {
                    exp8(_mm256_max_ps(
                        _mm256_sub_ps(_mm256_loadu_ps(p.add(j)), m),
                        lo,
                    ))
                };
                let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
                let mut j = 0;
                while j + 16 <= full {
                    let (e0, e1) = (e(j), e(j + 8));
                    _mm256_storeu_ps(p.add(j), e0);
                    _mm256_storeu_ps(p.add(j + 8), e1);
                    s0 = _mm256_add_ps(s0, e0);
                    s1 = _mm256_add_ps(s1, e1);
                    j += 16;
                }
                if j < full {
                    let e0 = e(j);
                    _mm256_storeu_ps(p.add(j), e0);
                    s0 = _mm256_add_ps(s0, e0);
                }
                if tail > 0 {
                    let v = _mm256_maskload_ps(p.add(full), mask);
                    let e = exp8(_mm256_max_ps(_mm256_sub_ps(v, m), lo));
                    _mm256_maskstore_ps(p.add(full), mask, e);
                    s1 = _mm256_add_ps(s1, _mm256_and_ps(e, _mm256_castsi256_ps(mask)));
                }
                let s = _mm256_add_ps(s0, s1);
                let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
                let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
                let s = _mm_add_ss(s, _mm_movehdup_ps(s));
                *inv = 1.0 / _mm_cvtss_f32(s);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    #[inline(always)]
    fn exp4(v: float32x4_t) -> float32x4_t {
        // safety: NEON is part of aarch64
        unsafe {
            let k = vdupq_n_f32;
            let t = vfmaq_f32(k(ROUND), v, k(std::f32::consts::LOG2_E));
            let n = vsubq_f32(t, k(ROUND));
            let r = vfmaq_f32(v, n, k(-LN2_HI));
            let r = vfmaq_f32(r, n, k(-LN2_LO));
            let r2 = vmulq_f32(r, r);
            let a = vfmaq_f32(k(P[1]), k(P[0]), r);
            let b = vfmaq_f32(k(P[3]), k(P[2]), r);
            let c = vfmaq_f32(k(P[5]), k(P[4]), r);
            let p = vfmaq_f32(c, vfmaq_f32(b, a, r2), r2);
            let y = vfmaq_f32(vaddq_f32(r, k(1.0)), p, r2);
            vmulq_f32(
                y,
                vreinterpretq_f32_u32(vshlq_n_u32::<23>(vreinterpretq_u32_f32(t))),
            )
        }
    }

    pub(super) fn rows(x: &mut [f32], n: usize, stride: usize, inv: &mut [f32]) {
        for (i, inv) in inv.iter_mut().enumerate() {
            let row = &mut x[i * stride..i * stride + n];
            let (chunks, rest) = row.as_chunks_mut::<4>();
            // safety (whole block): NEON is part of aarch64, and c is 4 values
            unsafe {
                let mut m = [vdupq_n_f32(f32::NEG_INFINITY); 4];
                for (q, c) in chunks.iter().enumerate() {
                    m[q % 4] = vmaxq_f32(m[q % 4], vld1q_f32(c.as_ptr()));
                }
                let m4 = vmaxq_f32(vmaxq_f32(m[0], m[1]), vmaxq_f32(m[2], m[3]));
                let max = rest.iter().fold(vmaxvq_f32(m4), |a, &v| a.max(v));
                let (mv, lo) = (vdupq_n_f32(max), vdupq_n_f32(LO));
                let mut s = [vdupq_n_f32(0.0); 2];
                for (q, c) in chunks.iter_mut().enumerate() {
                    let e = exp4(vmaxq_f32(vsubq_f32(vld1q_f32(c.as_ptr()), mv), lo));
                    vst1q_f32(c.as_mut_ptr(), e);
                    s[q % 2] = vaddq_f32(s[q % 2], e);
                }
                let mut sum = 0.0;
                for v in rest.iter_mut() {
                    *v = exp_clamped((*v - max).max(LO));
                    sum += *v;
                }
                *inv = 1.0 / (vaddvq_f32(vaddq_f32(s[0], s[1])) + sum);
            }
        }
    }
}

#[cfg(test)]
#[path = "softmax_tests.rs"]
mod tests;
