//! 6 x 16 kernel: 12 accumulators of 8 lanes, A broadcast one value per row.

use std::arch::x86_64::*;

pub(super) const MR: usize = 6;
pub(super) const NR: usize = 16;
// keeps a kc x 16 panel block (16 KB) in a 32 KB L1
pub(super) const KC: usize = 256;
// panel rows ahead to prefetch (one 64-byte line each)
const PREFETCH: usize = 16;

/// C[0..R][0..16] += A[0..R][0..kc] · B panel, or = (bias +) as S says; `a` holds R row pointers (R <= MR).
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel<const R: usize, const S: u8>(
    kc: usize,
    a: &[*const f32; 8],
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    bias: *const f32,
) {
    // safety: the tile loop passes kc readable values per row pointer, a
    // panel of kc x 16 values at b, R rows of 16 writable values at c, and
    // 16 values at bias when S is BIAS
    unsafe {
        let mut acc = [[_mm256_setzero_ps(); 2]; R];
        match S {
            super::ADD => {
                for (r, row) in acc.iter_mut().enumerate() {
                    row[0] = _mm256_loadu_ps(c.add(r * ldc));
                    row[1] = _mm256_loadu_ps(c.add(r * ldc + 8));
                }
            }
            super::BIAS => {
                let (b0, b1) = (_mm256_loadu_ps(bias), _mm256_loadu_ps(bias.add(8)));
                acc.iter_mut().for_each(|row| *row = [b0, b1]);
            }
            _ => {}
        }
        let mut step = |k: usize| {
            let bp = b.add(k * NR);
            // the packed data is read front to back; fetching ahead keeps the
            // weights streaming when they come from memory, not cache. The
            // address may run past the panel, which a prefetch allows
            _mm_prefetch::<_MM_HINT_T0>(bp.wrapping_add(PREFETCH * NR) as *const i8);
            let (b0, b1) = (_mm256_loadu_ps(bp), _mm256_loadu_ps(bp.add(8)));
            for (r, row) in acc.iter_mut().enumerate() {
                let x = _mm256_broadcast_ss(&*a[r].add(k));
                row[0] = _mm256_fmadd_ps(x, b0, row[0]);
                row[1] = _mm256_fmadd_ps(x, b1, row[1]);
            }
        };
        // four steps per trip: one step alone is as many instructions as the
        // core issues while its multiply-adds run, so loop overhead showed
        let mut k = 0;
        while k + 4 <= kc {
            step(k);
            step(k + 1);
            step(k + 2);
            step(k + 3);
            k += 4;
        }
        while k < kc {
            step(k);
            k += 1;
        }
        for (r, row) in acc.iter().enumerate() {
            _mm256_storeu_ps(c.add(r * ldc), row[0]);
            _mm256_storeu_ps(c.add(r * ldc + 8), row[1]);
        }
    }
}

/// One panel from its transpose: out[kk * 16 + j] = src[j * ld + kk] for
/// j < rows (<= 16) and kk < len, 8 x 8 blocks at a time.
///
/// # Safety
/// AVX2 must be present; `src` must hold `rows` rows of `len` values `ld`
/// apart, and `out` len x 16 values.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn panel_t(src: *const f32, ld: usize, rows: usize, len: usize, out: *mut f32) {
    // safety (whole body): every index stays below rows and len
    unsafe {
        let (full_j, full_k) = (rows / 8 * 8, len / 8 * 8);
        for j0 in (0..full_j).step_by(8) {
            for k0 in (0..full_k).step_by(8) {
                let r: [__m256; 8] =
                    std::array::from_fn(|i| _mm256_loadu_ps(src.add((j0 + i) * ld + k0)));
                let t: [__m256; 8] = std::array::from_fn(|i| {
                    let (x, y) = (r[i / 2 * 2], r[i / 2 * 2 + 1]);
                    if i % 2 == 0 {
                        _mm256_unpacklo_ps(x, y)
                    } else {
                        _mm256_unpackhi_ps(x, y)
                    }
                });
                // u[q] holds columns q and q + 4 of four rows (q < 4), for rows 0..4 and 4..8
                let u = |t0: __m256, t1: __m256, t2: __m256, t3: __m256| {
                    [
                        _mm256_shuffle_ps::<0x44>(t0, t2),
                        _mm256_shuffle_ps::<0xee>(t0, t2),
                        _mm256_shuffle_ps::<0x44>(t1, t3),
                        _mm256_shuffle_ps::<0xee>(t1, t3),
                    ]
                };
                let (lo, hi) = (u(t[0], t[1], t[2], t[3]), u(t[4], t[5], t[6], t[7]));
                for q in 0..4 {
                    _mm256_storeu_ps(
                        out.add((k0 + q) * NR + j0),
                        _mm256_permute2f128_ps::<0x20>(lo[q], hi[q]),
                    );
                    _mm256_storeu_ps(
                        out.add((k0 + q + 4) * NR + j0),
                        _mm256_permute2f128_ps::<0x31>(lo[q], hi[q]),
                    );
                }
            }
        }
        for j in 0..rows {
            let ks = if j < full_j { full_k..len } else { 0..len };
            for kk in ks {
                *out.add(kk * NR + j) = *src.add(j * ld + kk);
            }
        }
    }
}

super::tile::tile_loop!(kernel, MR, NR, KC, 96, "avx2,fma");
