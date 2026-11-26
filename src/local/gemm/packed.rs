//! B packed into panels for the own kernels.

use super::Isa;
#[cfg(target_arch = "x86_64")]
use super::avx2;

/// B (k, n) in panels for one kernel family; reusable as a buffer. The k
/// rows are cut into blocks of KC, and within a block the panels follow one
/// another, each (block rows, NR); a product walks the data front to back.
pub(crate) struct Packed {
    pub(super) isa: Isa,
    pub(super) data: Vec<f32>,
    pub(super) k: usize,
    pub(super) n: usize,
}

impl Packed {
    pub(crate) fn new(isa: Isa) -> Self {
        Self {
            isa,
            data: Vec::new(),
            k: 0,
            n: 0,
        }
    }

    /// Storage for (k, n), zero in the columns that pad the last panel and
    /// unspecified elsewhere; returns (NR, KC, panels).
    fn shape(&mut self, k: usize, n: usize) -> (usize, usize, usize) {
        let (nr, kc) = (self.isa.nr(), self.isa.kc());
        let panels = n.div_ceil(nr);
        self.data.resize(panels * k * nr, 0.0);
        self.k = k;
        self.n = n;
        let (p, cols) = (panels - 1, n - (panels - 1) * nr);
        if cols < nr {
            for k0 in (0..k).step_by(kc) {
                let len = kc.min(k - k0);
                let last = &mut self.data[(k0 * panels + p * len) * nr..][..len * nr];
                last.chunks_exact_mut(nr)
                    .for_each(|row| row[cols..].fill(0.0));
            }
        }
        (nr, kc, panels)
    }

    /// B[kk][j] = src[kk * ld + j].
    pub(crate) fn pack(&mut self, src: &[f32], ld: usize, k: usize, n: usize) {
        assert!(k > 0 && n > 0 && src.len() >= (k - 1) * ld + n);
        let (nr, kc, panels) = self.shape(k, n);
        for k0 in (0..k).step_by(kc) {
            let len = kc.min(k - k0);
            let block = &mut self.data[k0 * panels * nr..(k0 + len) * panels * nr];
            for kk in 0..len {
                let row = &src[(k0 + kk) * ld..(k0 + kk) * ld + n];
                for (p, part) in row.chunks(nr).enumerate() {
                    let o = (p * len + kk) * nr;
                    copy(&mut block[o..o + part.len()], part);
                }
            }
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Counts the zero columns that pad the last panel as part of B, so a
    /// product writes whole panels; C then needs room for them.
    pub(crate) fn pad(&mut self) {
        self.n = self.n.next_multiple_of(self.isa.nr());
    }

    /// B[kk][j] = src[j * ld + kk]: B given as its transpose, n rows of k.
    pub(crate) fn pack_t(&mut self, src: &[f32], ld: usize, n: usize, k: usize) {
        assert!(k > 0 && n > 0 && src.len() >= (n - 1) * ld + k);
        let (nr, kc, panels) = self.shape(k, n);
        for k0 in (0..k).step_by(kc) {
            let len = kc.min(k - k0);
            let block = &mut self.data[k0 * panels * nr..(k0 + len) * panels * nr];
            for (p, panel) in block.chunks_exact_mut(len * nr).enumerate() {
                let j0 = p * nr;
                let rows = &src[j0 * ld + k0..];
                let cols = nr.min(n - j0);
                #[cfg(target_arch = "x86_64")]
                if self.isa == Isa::Avx2 {
                    // safety: the isa was detected; rows holds `cols` rows of
                    // `len` values `ld` apart, and the panel len x 16 values
                    unsafe { avx2::panel_t(rows.as_ptr(), ld, cols, len, panel.as_mut_ptr()) };
                    continue;
                }
                for jj in 0..cols {
                    for (kk, &v) in rows[jj * ld..jj * ld + len].iter().enumerate() {
                        panel[kk * nr + jj] = v;
                    }
                }
            }
        }
    }
}

// a panel row is 12 or 16 values, which a fixed-size copy keeps inline
#[inline(always)]
fn copy(dst: &mut [f32], src: &[f32]) {
    if let (Ok(d), Ok(s)) = (
        <&mut [f32; 16]>::try_from(&mut *dst),
        <&[f32; 16]>::try_from(src),
    ) {
        *d = *s;
    } else if let (Ok(d), Ok(s)) = (
        <&mut [f32; 12]>::try_from(&mut *dst),
        <&[f32; 12]>::try_from(src),
    ) {
        *d = *s;
    } else {
        dst.copy_from_slice(src);
    }
}
