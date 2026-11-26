use super::*;

fn rnd(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn isas() -> Vec<Isa> {
    let mut v = vec![Isa::Portable];
    if Isa::detect() != Isa::Portable {
        v.push(Isa::detect());
    }
    v
}

// C += A·B on strided operands, against an f64 product
fn check(isa: Isa, m: usize, k: usize, n: usize, par: Par, seed: &mut u64) {
    let (lda, ldb, ldc) = (k + 3, n + 5, n + 2);
    let a = rnd(m * lda, seed);
    let b = rnd(k * ldb, seed);
    let c0 = rnd(m * ldc, seed);
    let mut want = c0.clone();
    for i in 0..m {
        for j in 0..n {
            let mut s = c0[i * ldc + j] as f64;
            for p in 0..k {
                s += a[i * lda + p] as f64 * b[p * ldb + j] as f64;
            }
            want[i * ldc + j] = s as f32;
        }
    }
    let mut packed = Packed::new(isa);
    packed.pack(&b, ldb, k, n);
    let mut c = c0.clone();
    gemm(m, &a, lda, &packed, &mut c, ldc, par);
    let tol = 1e-6 * (k as f32).sqrt() * 4.0;
    for i in 0..m {
        for j in 0..ldc {
            let (g, w) = (c[i * ldc + j], want[i * ldc + j]);
            assert!(
                (g - w).abs() <= tol,
                "{isa:?} m={m} k={k} n={n} at ({i},{j}): {g} vs {w}"
            );
        }
    }
    // the transposed packing gives the same product
    let bt: Vec<f32> = (0..n * k).map(|x| b[(x % k) * ldb + x / k]).collect();
    packed.pack_t(&bt, k, n, k);
    let mut c2 = c0.clone();
    gemm(m, &a, lda, &packed, &mut c2, ldc, par);
    assert_eq!(c, c2);
}

#[test]
fn matches_f64_on_edge_shapes() {
    let mut seed = 1;
    for isa in isas() {
        for m in [1, 3, 4, 5, 6, 7, 8, 9, 13, 25] {
            for k in [1, 31, 255, 256, 257, 513] {
                for n in [1, 11, 12, 13, 16, 17, 32, 40] {
                    check(isa, m, k, n, None, &mut seed);
                }
            }
        }
    }
}

#[test]
fn threaded_matches_serial() {
    let mut seed = 2;
    pool::scoped(4, |par| {
        for isa in isas() {
            for (m, k, n) in [
                (13, 384, 1152),
                (70, 384, 384),
                (200, 1536, 384),
                (1, 384, 1536),
                (129, 32, 200),
            ] {
                check(isa, m, k, n, par, &mut seed);
            }
        }
    });
}

#[cfg(target_os = "macos")]
#[test]
fn accelerate_matches_own() {
    let mut seed = 3;
    let (m, k, n) = (37, 384, 72);
    let a = rnd(m * k, &mut seed);
    let w = rnd(n * k, &mut seed);
    let mut c = rnd(m * n, &mut seed);
    let mut d = c.clone();
    accelerate::sgemm(true, m, n, k, &a, k, &w, k, 1.0, &mut c, n);
    let mut p = Packed::new(Isa::detect());
    p.pack_t(&w, k, n, k);
    gemm(m, &a, k, &p, &mut d, n, None);
    for (x, y) in c.iter().zip(&d) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}
