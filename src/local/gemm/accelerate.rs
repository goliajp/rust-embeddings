//! `cblas_sgemm` from Apple's Accelerate framework, which ships with macOS
//! and runs single-precision products on the matrix unit.

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

fn int(v: usize) -> i32 {
    i32::try_from(v).expect("matrix dimension fits in i32")
}

/// C (m, n) = A (m, k) · op(B) + beta · C, row-major; op(B) is Bᵀ when
/// `trans_b`, B then being (n, k).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm(
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(ldc >= n && c.len() >= (m - 1) * ldc + n);
    // safety: c covers the rows and columns just checked
    unsafe { sgemm_ptr(trans_b, m, n, k, a, lda, b, ldb, beta, c.as_mut_ptr(), ldc) }
}

/// [`sgemm`] into C at a raw pointer.
///
/// # Safety
/// `c` must be valid for writes at `i * ldc + j` for every `i < m`, `j < n`,
/// and no other thread may touch those elements during the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sgemm_ptr(
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: *mut f32,
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let (b_rows, b_cols) = if trans_b { (n, k) } else { (k, n) };
    assert!(lda >= k && ldb >= b_cols && ldc >= n);
    assert!(a.len() >= (m - 1) * lda + k && b.len() >= (b_rows - 1) * ldb + b_cols);
    // safety: the asserts and the caller bound every element the call reads and writes
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            if trans_b { TRANS } else { NO_TRANS },
            int(m),
            int(n),
            int(k),
            1.0,
            a.as_ptr(),
            int(lda),
            b.as_ptr(),
            int(ldb),
            beta,
            c,
            int(ldc),
        )
    }
}
