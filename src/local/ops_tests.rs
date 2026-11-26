use super::*;
use crate::local::reference::erf as erf64;

fn sweep(lo: f32, hi: f32, n: usize) -> impl Iterator<Item = f32> {
    (0..n).map(move |i| lo + (hi - lo) * i as f32 / (n - 1) as f32)
}

#[test]
fn erf_and_gelu_are_close_to_f64() {
    let mut worst_erf = 0f64;
    let mut worst_gelu = 0f64;
    let xs: Vec<f32> = sweep(-10.0, 10.0, 400_001).collect();
    let mut ys = xs.clone();
    gelu_slice(&mut ys);
    for (&x, &y) in xs.iter().zip(&ys) {
        let x64 = x as f64;
        worst_erf = worst_erf.max((erf(x) as f64 - erf64(x64)).abs());
        let exact = 0.5 * x64 * (1.0 + erf64(x64 / std::f64::consts::SQRT_2));
        // relative to the value's size, floored at 1 so near zero it is absolute
        worst_gelu = worst_gelu.max((y as f64 - exact).abs() / exact.abs().max(1.0));
        assert_eq!(y, gelu(x));
    }
    assert!(worst_erf < 5e-7, "erf {worst_erf:e}");
    assert!(worst_gelu < 1.2e-6, "gelu {worst_gelu:e}");
}

#[test]
fn layernorm_matches_f64() {
    let h = 384;
    let mut x: Vec<f32> = (0..3 * h)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.3 + 4.0)
        .collect();
    let w: Vec<f32> = (0..h).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..h).map(|i| (i % 5) as f32 * 0.01).collect();
    let src = x.clone();
    layernorm(&mut x, &w, &b, 1e-12);
    for (row, out) in src.chunks(h).zip(x.chunks(h)) {
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / h as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / h as f64;
        for i in 0..h {
            let want = (row[i] as f64 - mean) / (var + 1e-12).sqrt() * w[i] as f64 + b[i] as f64;
            assert!((out[i] as f64 - want).abs() < 2e-6, "{} vs {want}", out[i]);
        }
    }
}

#[test]
fn rows_and_prefill_cover_every_row() {
    pool::scoped(3, |par| {
        for m in [1, 2, 85, 86, 300] {
            let n = 384;
            let bias: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
            let mut c = vec![f32::NAN; m * n];
            prefill(par, &mut c, &bias);
            assert!(c.chunks(n).all(|r| r == &bias[..]));
            rows(par, &mut c, n, &|block, r0| {
                for (i, row) in block.chunks_exact_mut(n).enumerate() {
                    row.fill((r0 + i) as f32);
                }
            });
            assert!(
                c.chunks(n)
                    .enumerate()
                    .all(|(i, r)| r.iter().all(|&v| v == i as f32))
            );
        }
    });
}

#[test]
fn add_layernorm_adds_first() {
    let h = 16;
    let x: Vec<f32> = (0..2 * h).map(|i| i as f32 * 0.25).collect();
    let r: Vec<f32> = (0..2 * h).map(|i| (i % 3) as f32).collect();
    let (w, b) = (vec![1.5; h], vec![0.25; h]);
    let mut sum: Vec<f32> = x.iter().zip(&r).map(|(a, b)| a + b).collect();
    layernorm(&mut sum, &w, &b, 1e-12);
    let mut got = x.clone();
    add_layernorm(&mut got, &r, &w, &b, 1e-12);
    assert_eq!(got, sum);
}
