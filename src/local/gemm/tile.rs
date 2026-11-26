//! The loop over row tiles, panels and k blocks around a kernel.

/// The blocked loop over one tile, shared by every kernel family.
/// `$kernel::<R, S>` computes R rows by NR columns of C in place, starting
/// from C, zero or the bias as S says, so a partial row block costs only its
/// rows; a partial column panel goes through a scratch tile.
macro_rules! tile_loop {
    ($kernel:ident, $mr:expr, $nr:expr, $kc:expr, $mc:expr $(, $feature:literal)?) => {
        #[allow(clippy::too_many_arguments)]
        $(#[target_feature(enable = $feature)])?
        pub(super) unsafe fn tile(data: &[f32], k: usize, n: usize, a: *const f32, lda: usize, c: *mut f32, ldc: usize, start: (u8, *const f32), r0: usize, r1: usize, p0: usize, p1: usize) {
            // safety (whole body): the caller bounds rows r0..r1 of a and c
            // and columns of c up to n; a kernel touches only `rows` rows
            use super::{ADD, BIAS, ZERO};
            let run = |rows: usize, s: u8, kc: usize, ap: &[*const f32; 8], bp: *const f32, c: *mut f32, ldc: usize, bias: *const f32| unsafe {
                macro_rules! rows {
                    ($s:expr) => {
                        match rows {
                            1 => $kernel::<1, { $s }>(kc, ap, bp, c, ldc, bias),
                            2 => $kernel::<2, { $s }>(kc, ap, bp, c, ldc, bias),
                            3 => $kernel::<3, { $s }>(kc, ap, bp, c, ldc, bias),
                            4 => $kernel::<4, { $s }>(kc, ap, bp, c, ldc, bias),
                            5 if $mr >= 5 => $kernel::<5, { $s }>(kc, ap, bp, c, ldc, bias),
                            6 if $mr >= 6 => $kernel::<6, { $s }>(kc, ap, bp, c, ldc, bias),
                            7 if $mr >= 7 => $kernel::<7, { $s }>(kc, ap, bp, c, ldc, bias),
                            8 if $mr >= 8 => $kernel::<8, { $s }>(kc, ap, bp, c, ldc, bias),
                            _ => unreachable!("row block larger than the kernel"),
                        }
                    };
                }
                match s {
                    ZERO => rows!(ZERO),
                    BIAS => rows!(BIAS),
                    _ => rows!(ADD),
                }
            };
            let np = n.div_ceil($nr);
            let mut kc0 = 0;
            while kc0 < k {
                let kc = ($kc).min(k - kc0);
                let s = if kc0 == 0 { start.0 } else { ADD };
                // one row tile by one panel
                let one = |r: usize, rows: usize, ap: &[*const f32; 8], p: usize| {
                    let cols = ($nr).min(n - p * $nr);
                    let bp = unsafe { data.as_ptr().add(kc0 * np * $nr + p * kc * $nr) };
                    let cp = unsafe { c.add(r * ldc + p * $nr) };
                    let bias = start.1.wrapping_add(p * $nr);
                    if cols == $nr {
                        return run(rows, s, kc, ap, bp, cp, ldc, bias);
                    }
                    let mut bt = [0f32; $nr];
                    if s == BIAS {
                        for j in 0..cols {
                            bt[j] = unsafe { *bias.add(j) };
                        }
                    }
                    let mut t = [0f32; $mr * $nr];
                    for i in 0..rows {
                        for j in 0..cols {
                            t[i * $nr + j] = unsafe { *cp.add(i * ldc + j) };
                        }
                    }
                    run(rows, s, kc, ap, bp, t.as_mut_ptr(), $nr, bt.as_ptr());
                    for i in 0..rows {
                        for j in 0..cols {
                            unsafe { *cp.add(i * ldc + j) = t[i * $nr + j] };
                        }
                    }
                };
                let pointers = |r: usize, rows: usize| {
                    let mut ap = [std::ptr::null(); 8];
                    for (i, q) in ap.iter_mut().take(rows).enumerate() {
                        *q = unsafe { a.add((r + i) * lda + kc0) };
                    }
                    ap
                };
                if kc * (p1 - p0) * $nr <= 8192 {
                    // the panels of this k block fit in L1 together (attention's
                    // short products): each row tile goes across all of them
                    let mut r = r0;
                    while r < r1 {
                        let rows = ($mr).min(r1 - r);
                        let ap = pointers(r, rows);
                        for p in p0..p1 {
                            one(r, rows, &ap, p);
                        }
                        r += $mr;
                    }
                } else {
                    // rows in blocks of MC, so the A rows of a block stay in
                    // L2 while every panel of this k block passes over them
                    let mut m0 = r0;
                    while m0 < r1 {
                        let m1 = (m0 + $mc).min(r1);
                        for p in p0..p1 {
                            let mut r = m0;
                            while r < m1 {
                                let rows = ($mr).min(m1 - r);
                                one(r, rows, &pointers(r, rows), p);
                                r += $mr;
                            }
                        }
                        m0 = m1;
                    }
                }
                kc0 += $kc;
            }
        }
    };
}
pub(crate) use tile_loop;
