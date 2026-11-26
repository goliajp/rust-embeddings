//! 4 x 16 kernel in plain Rust, for targets without a dedicated one.

pub(super) const MR: usize = 4;
pub(super) const NR: usize = 16;
const KC: usize = 256;

/// C[0..4][0..16] += A[0..4][0..kc] · B panel; `a` holds 4 row pointers.
#[inline(always)]
unsafe fn kernel(kc: usize, a: &[*const f32; 8], b: *const f32, c: *mut f32, ldc: usize) {
    // safety: as for the other kernels, with 4 rows of 16
    unsafe {
        let mut acc = [[0f32; NR]; MR];
        for (r, row) in acc.iter_mut().enumerate() {
            row.copy_from_slice(std::slice::from_raw_parts(c.add(r * ldc), NR));
        }
        for k in 0..kc {
            let bk = std::slice::from_raw_parts(b.add(k * NR), NR);
            for (r, row) in acc.iter_mut().enumerate() {
                let x = *a[r].add(k);
                for (v, &w) in row.iter_mut().zip(bk) {
                    *v += x * w;
                }
            }
        }
        for (r, row) in acc.iter().enumerate() {
            std::slice::from_raw_parts_mut(c.add(r * ldc), NR).copy_from_slice(row);
        }
    }
}

super::tile_loop!(kernel, MR, NR, KC);
