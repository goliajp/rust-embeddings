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

// C = bias + A·B on strided operands, against an f64 product
fn check(isa: Isa, m: usize, k: usize, n: usize, par: Par, seed: &mut u64) {
    let (lda, ldb, ldc) = (k + 3, n + 5, n + 2);
    let a = rnd(m * lda, seed);
    let b = rnd(k * ldb, seed);
    let bias = rnd(n, seed);
    let mut want = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                want[i * n + j] += a[i * lda + p] as f64 * b[p * ldb + j] as f64;
            }
        }
    }
    // a reused buffer holds junk from a larger matrix
    let mut packed = Packed::new(isa);
    packed.pack(&vec![1e3; (k + 1) * (n + 20)], n + 20, k + 1, n + 20);
    packed.pack(&b, ldb, k, n);
    let tol = 1e-6 * (k as f32).sqrt() * 4.0;
    // C is never read, and columns past n are never written
    let mut c = vec![f32::NAN; m * ldc];
    gemm(m, &a, lda, &packed, &mut c, ldc, Init::Bias(&bias), par);
    for i in 0..m {
        for j in 0..ldc {
            let g = c[i * ldc + j];
            if j >= n {
                assert!(g.is_nan());
                continue;
            }
            let w = (want[i * n + j] + bias[j] as f64) as f32;
            assert!(
                (g - w).abs() <= tol,
                "{isa:?} m={m} k={k} n={n} at ({i},{j}): {g} vs {w}"
            );
        }
    }
    // the transposed packing gives the same product
    let bt: Vec<f32> = (0..n * k).map(|x| b[(x % k) * ldb + x / k]).collect();
    packed.pack_t(&bt, k, n, k);
    let mut c2 = vec![f32::NAN; m * ldc];
    gemm(m, &a, lda, &packed, &mut c2, ldc, Init::Bias(&bias), par);
    assert_eq!(
        c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        c2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // from zero, and a padded B writes zeros into its padding columns
    packed.pad();
    let np = packed.n();
    let ldz = np + 1;
    let mut z = vec![f32::NAN; m * ldz];
    gemm(m, &a, lda, &packed, &mut z, ldz, Init::Zero, par);
    for i in 0..m {
        for j in 0..ldz {
            let g = z[i * ldz + j];
            if j >= np {
                assert!(g.is_nan());
            } else if j >= n {
                assert_eq!(g, 0.0, "{isa:?} padding ({i},{j})");
            } else {
                let w = want[i * n + j] as f32;
                assert!(
                    (g - w).abs() <= tol,
                    "{isa:?} zero m={m} k={k} n={n} ({i},{j}): {g} vs {w}"
                );
            }
        }
    }
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
    accelerate::sgemm(true, m, n, k, &a, k, &w, k, 0.0, &mut c, n);
    let mut p = Packed::new(Isa::detect());
    p.pack_t(&w, k, n, k);
    gemm(m, &a, k, &p, &mut d, n, Init::Zero, None);
    for (x, y) in c.iter().zip(&d) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}
