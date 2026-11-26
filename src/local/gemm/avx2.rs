//! 6 x 16 kernel: 12 accumulators of 8 lanes, A broadcast one value per row.

use std::arch::x86_64::*;

pub(super) const MR: usize = 6;
pub(super) const NR: usize = 16;
// keeps a kc x 16 panel block (16 KB) in a 32 KB L1
const KC: usize = 256;

/// C[0..R][0..16] += A[0..R][0..kc] · B panel; `a` holds R row pointers (R <= MR).
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel<const R: usize>(
    kc: usize,
    a: &[*const f32; 8],
    b: *const f32,
    c: *mut f32,
    ldc: usize,
) {
    // safety: the tile loop passes kc readable values per row pointer, a
    // panel of kc x 16 values at b, and R rows of 16 writable values at c
    unsafe {
        let mut acc = [[_mm256_setzero_ps(); 2]; R];
        for (r, row) in acc.iter_mut().enumerate() {
            row[0] = _mm256_loadu_ps(c.add(r * ldc));
            row[1] = _mm256_loadu_ps(c.add(r * ldc + 8));
        }
        for k in 0..kc {
            let bp = b.add(k * NR);
            let (b0, b1) = (_mm256_loadu_ps(bp), _mm256_loadu_ps(bp.add(8)));
            for (r, row) in acc.iter_mut().enumerate() {
                let x = _mm256_broadcast_ss(&*a[r].add(k));
                row[0] = _mm256_fmadd_ps(x, b0, row[0]);
                row[1] = _mm256_fmadd_ps(x, b1, row[1]);
            }
        }
        for (r, row) in acc.iter().enumerate() {
            _mm256_storeu_ps(c.add(r * ldc), row[0]);
            _mm256_storeu_ps(c.add(r * ldc + 8), row[1]);
        }
    }
}

super::tile_loop!(kernel, MR, NR, KC, "avx2,fma");
