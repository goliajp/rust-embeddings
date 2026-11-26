use super::*;

#[test]
fn exp_is_close_to_f64() {
    let mut worst = 0f64;
    for i in 0..400_001 {
        let x = -87.0 + 175.0 * i as f32 / 400_000.0;
        let e = (x as f64).exp();
        worst = worst.max(((exp(x) as f64) - e).abs() / e);
    }
    assert!(worst < 2e-7, "{worst:e}");
}

// rows of every tail length, `stride` apart, with junk past n that must
// not reach the result
fn rows(n: usize, stride: usize) -> Vec<f32> {
    (0..3 * stride)
        .map(|i| {
            if i % stride < n {
                (i as f32 * 0.7).sin() * 9.0 - 30.0 * (i % 5 == 0) as u8 as f32
            } else {
                1e4
            }
        })
        .collect()
}

#[test]
fn vector_rows_give_the_scalar_bits() {
    for n in [1, 7, 8, 9, 31, 32, 33, 227] {
        let stride = n + 9;
        let src = rows(n, stride);
        let mut x = src.clone();
        let mut inv = [0f32; 3];
        softmax_rows(&mut x, n, stride, &mut inv);
        for r in 0..3 {
            let row = &src[r * stride..r * stride + n];
            let max = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            for (j, v) in row.iter().enumerate() {
                assert_eq!(
                    x[r * stride + j].to_bits(),
                    exp(v - max).to_bits(),
                    "{n} {r} {j}"
                );
            }
        }
    }
}

#[test]
fn softmax_rows_match_f64() {
    for n in [1usize, 7, 8, 13, 233] {
        for stride in [n, n + 3, n.next_multiple_of(16)] {
            let src = rows(n, stride);
            let mut x = src.clone();
            let mut inv = vec![0f32; 3];
            softmax_rows(&mut x, n, stride, &mut inv);
            for r in 0..3 {
                let row = &src[r * stride..r * stride + n];
                let max = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
                let denom: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
                for (j, v) in row.iter().enumerate() {
                    let want = (*v as f64 - max).exp() / denom;
                    let got = x[r * stride + j] as f64 * inv[r] as f64;
                    assert!((got - want).abs() < 1e-6, "{n} {stride}: {got} vs {want}");
                }
            }
        }
    }
}
