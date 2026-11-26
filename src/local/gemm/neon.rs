//! 8 x 12 kernel: 24 accumulators of 4 lanes, A broadcast one value per row.

use std::arch::aarch64::*;

pub(super) const MR: usize = 8;
pub(super) const NR: usize = 12;
const KC: usize = 512;

/// C[0..8][0..12] += A[0..8][0..kc] · B panel; `a` holds 8 row pointers.
#[inline(always)]
unsafe fn kernel(kc: usize, a: &[*const f32; 8], b: *const f32, c: *mut f32, ldc: usize) {
    // safety: the tile loop passes kc readable values per row pointer, a
    // panel of kc x 12 values at b, and 8 rows of 12 writable values at c
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 3]; MR];
        for (r, row) in acc.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = vld1q_f32(c.add(r * ldc + 4 * j));
            }
        }
        for k in 0..kc {
            let bp = b.add(k * NR);
            let (b0, b1, b2) = (vld1q_f32(bp), vld1q_f32(bp.add(4)), vld1q_f32(bp.add(8)));
            for (r, row) in acc.iter_mut().enumerate() {
                let x = vld1q_dup_f32(a[r].add(k));
                row[0] = vfmaq_f32(row[0], b0, x);
                row[1] = vfmaq_f32(row[1], b1, x);
                row[2] = vfmaq_f32(row[2], b2, x);
            }
        }
        for (r, row) in acc.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                vst1q_f32(c.add(r * ldc + 4 * j), *v);
            }
        }
    }
}

super::tile_loop!(kernel, MR, NR, KC);
