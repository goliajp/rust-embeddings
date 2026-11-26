//! Test-only f64 BERT forward pass, written for clarity: the oracle the fast
//! path is measured against.

use super::model::{BertShape, Pooling};
use super::safetensors::Safetensors;

/// erf in f64 to about 1e-15: Taylor series below 2.5, the continued
/// fraction of erfc above.
pub(crate) fn erf(x: f64) -> f64 {
    let a = x.abs();
    let v = if a < 2.5 {
        let (mut term, mut sum, x2) = (a, a, a * a);
        for n in 1..200 {
            term *= -x2 / n as f64;
            let t = term / (2 * n + 1) as f64;
            sum += t;
            if t.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        let mut t = a;
        for n in (1..=80).rev() {
            t = a + (n as f64 / 2.0) / t;
        }
        1.0 - (-a * a).exp() / (std::f64::consts::PI.sqrt() * t)
    };
    v.copysign(x)
}

fn get(st: &Safetensors, prefix: &str, name: &str, shape: &[usize]) -> Vec<f64> {
    st.f32s(&format!("{prefix}{name}"), shape)
        .unwrap()
        .iter()
        .map(|&v| v as f64)
        .collect()
}

fn linear(x: &[f64], t: usize, w: &[f64], b: &[f64]) -> Vec<f64> {
    let (n, k) = (b.len(), x.len() / t);
    let mut y = vec![0f64; t * n];
    for i in 0..t {
        for o in 0..n {
            let row = &w[o * k..(o + 1) * k];
            y[i * n + o] = b[o]
                + row
                    .iter()
                    .zip(&x[i * k..(i + 1) * k])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
        }
    }
    y
}

fn layernorm(x: &mut [f64], w: &[f64], b: &[f64], eps: f64) {
    for row in x.chunks_mut(w.len()) {
        let mean = row.iter().sum::<f64>() / w.len() as f64;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / w.len() as f64;
        for i in 0..w.len() {
            row[i] = (row[i] - mean) / (var + eps).sqrt() * w[i] + b[i];
        }
    }
}

/// The pooled, normalised vector of one token sequence.
pub(crate) fn embed(st: &Safetensors, c: &BertShape, pooling: Pooling, ids: &[u32]) -> Vec<f64> {
    let p = if st.contains("embeddings.word_embeddings.weight") {
        ""
    } else {
        "bert."
    };
    let (h, t, heads) = (c.hidden, ids.len(), c.heads);
    let d = h / heads;
    let word = get(st, p, "embeddings.word_embeddings.weight", &[c.vocab, h]);
    let pos = get(
        st,
        p,
        "embeddings.position_embeddings.weight",
        &[c.max_positions, h],
    );
    let ty = get(
        st,
        p,
        "embeddings.token_type_embeddings.weight",
        &[c.type_vocab, h],
    );
    let mut x = vec![0f64; t * h];
    for (i, &id) in ids.iter().enumerate() {
        for j in 0..h {
            x[i * h + j] = word[id as usize * h + j] + pos[i * h + j] + ty[j];
        }
    }
    let ln = |name: &str| {
        (
            get(st, p, &format!("{name}.weight"), &[h]),
            get(st, p, &format!("{name}.bias"), &[h]),
        )
    };
    let (w, b) = ln("embeddings.LayerNorm");
    layernorm(&mut x, &w, &b, c.eps);
    for l in 0..c.layers {
        let pre = format!("encoder.layer.{l}.");
        let lin = |name: &str, n: usize, k: usize, x: &[f64]| {
            let w = get(st, p, &format!("{pre}{name}.weight"), &[n, k]);
            let b = get(st, p, &format!("{pre}{name}.bias"), &[n]);
            linear(x, t, &w, &b)
        };
        let q = lin("attention.self.query", h, h, &x);
        let k = lin("attention.self.key", h, h, &x);
        let v = lin("attention.self.value", h, h, &x);
        let mut ctx = vec![0f64; t * h];
        for hd in 0..heads {
            for i in 0..t {
                let s: Vec<f64> = (0..t)
                    .map(|j| {
                        (0..d)
                            .map(|e| q[i * h + hd * d + e] * k[j * h + hd * d + e])
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let max = s.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - max).exp()).collect();
                let sum: f64 = e.iter().sum();
                for j in 0..t {
                    for f in 0..d {
                        ctx[i * h + hd * d + f] += e[j] / sum * v[j * h + hd * d + f];
                    }
                }
            }
        }
        let mut a = lin("attention.output.dense", h, h, &ctx);
        a.iter_mut().zip(&x).for_each(|(a, x)| *a += x);
        let (w, b) = ln(&format!("{pre}attention.output.LayerNorm"));
        layernorm(&mut a, &w, &b, c.eps);
        let mut f = lin("intermediate.dense", c.intermediate, h, &a);
        f.iter_mut()
            .for_each(|v| *v = 0.5 * *v * (1.0 + erf(*v / std::f64::consts::SQRT_2)));
        x = lin("output.dense", h, c.intermediate, &f);
        x.iter_mut().zip(&a).for_each(|(x, a)| *x += a);
        let (w, b) = ln(&format!("{pre}output.LayerNorm"));
        layernorm(&mut x, &w, &b, c.eps);
    }
    let mut out = match pooling {
        Pooling::Cls => x[..h].to_vec(),
        Pooling::Mean => (0..h)
            .map(|j| (0..t).map(|i| x[i * h + j]).sum::<f64>() / t as f64)
            .collect(),
    };
    let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
    out.iter_mut().for_each(|v| *v /= norm);
    out
}

#[test]
fn erf_reference_is_accurate() {
    // published values of erf
    for (x, want) in [
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (2.0, 0.995_322_265_018_952_7),
        (2.5, 0.999_593_047_982_555),
        (3.0, 0.999_977_909_503_001_4),
        (5.0, 0.999_999_999_998_462_5),
    ] {
        assert!((erf(x) - want).abs() < 1e-14, "{x}: {}", erf(x));
        assert_eq!(erf(-x), -erf(x));
    }
}
