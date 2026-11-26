//! The encoder forward pass over a chunk of unpadded token sequences, then
//! pooling and L2 normalisation. Every step but attention works per token,
//! so the tokens of all sequences sit in one (tokens, hidden) matrix.

use std::sync::Mutex;

use super::attention::{self, Queries, Scratch};
use super::gemm::linear;
use super::model::Pooling;
use super::ops;
use super::pool::{self, Par};
use super::weights::Weights;

/// Activations, kept between calls so repeated calls do not fault in fresh pages.
pub(crate) struct Buffers {
    ids: Vec<u32>,
    positions: Vec<u32>,
    seqs: Vec<(usize, usize)>,
    x: Vec<f32>,
    // residual rows of the last layer when only the [CLS] rows go on
    xc: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    a: Vec<f32>,
    f: Vec<f32>,
    scratch: Vec<Mutex<Scratch>>,
}

impl Buffers {
    pub(crate) fn new() -> Self {
        Self {
            ids: Vec::new(),
            positions: Vec::new(),
            seqs: Vec::new(),
            x: Vec::new(),
            xc: Vec::new(),
            qkv: Vec::new(),
            ctx: Vec::new(),
            a: Vec::new(),
            f: Vec::new(),
            scratch: Vec::new(),
        }
    }
}

fn fit(v: &mut Vec<f32>, n: usize) {
    v.resize(n, 0.0);
}

/// One vector per sequence into `out` (sequences x hidden), in order.
pub(crate) fn forward(
    w: &Weights,
    batch: &[&[u32]],
    pooling: Pooling,
    out: &mut [f32],
    b: &mut Buffers,
    par: Par,
) {
    let c = &w.shape;
    let (h, f) = (c.hidden, c.intermediate);
    let eps = c.eps as f32;
    b.ids.clear();
    b.positions.clear();
    b.seqs.clear();
    for seq in batch {
        assert!(!seq.is_empty() && seq.len() <= c.max_positions);
        b.seqs.push((b.ids.len(), seq.len()));
        b.ids.extend_from_slice(seq);
        b.positions.extend(0..seq.len() as u32);
    }
    let t = b.ids.len();
    fit(&mut b.x, t * h);
    fit(&mut b.qkv, t * 3 * h);
    fit(&mut b.ctx, t * h);
    fit(&mut b.a, t * h);
    fit(&mut b.f, t * f);
    while b.scratch.len() < pool::width(par) {
        b.scratch.push(Mutex::new(Scratch::new()));
    }

    let (ids, positions) = (&b.ids, &b.positions);
    ops::rows(par, &mut b.x, h, &|block, r0| {
        for (i, row) in block.chunks_exact_mut(h).enumerate() {
            let (id, p) = (ids[r0 + i] as usize, positions[r0 + i] as usize);
            let word = &w.word[id * h..(id + 1) * h];
            let fixed = &w.pos_type[p * h..(p + 1) * h];
            for ((o, a), b) in row.iter_mut().zip(word).zip(fixed) {
                *o = a + b;
            }
        }
        ops::layernorm(block, &w.norm.w, &w.norm.b, eps);
    });

    let last = w.layers.len().saturating_sub(1);
    for (li, l) in w.layers.iter().enumerate() {
        ops::prefill(par, &mut b.qkv, &l.qkv.b);
        linear(&b.x, t, &l.qkv.w, &mut b.qkv, par);
        // with [CLS] pooling, the last layer's output is read at the first
        // token only: every token still gives keys and values, but the
        // rest of the layer runs on one row per sequence
        let cls = pooling == Pooling::Cls && li == last;
        let (rows, queries) = if cls {
            (b.seqs.len(), Queries::First)
        } else {
            (t, Queries::All)
        };
        let ctx = &mut b.ctx[..rows * h];
        ops::rows(par, ctx, h, &|block, _| block.fill(0.0));
        attention::attention(&b.qkv, &b.seqs, h, c.heads, queries, ctx, par, &b.scratch);
        if cls {
            b.xc.clear();
            for &(o, _) in &b.seqs {
                b.xc.extend_from_slice(&b.x[o * h..(o + 1) * h]);
            }
        }
        let residual = if cls { &b.xc } else { &b.x };

        // the residual is added after the product, not preloaded into it: (x·W + b) + r
        // stays closer to an f64 pass
        let a = &mut b.a[..rows * h];
        ops::prefill(par, a, &l.out.b);
        linear(&b.ctx[..rows * h], rows, &l.out.w, a, par);
        ops::rows(par, a, h, &|block, r0| {
            ops::add_layernorm(block, &residual[r0 * h..], &l.norm1.w, &l.norm1.b, eps)
        });

        let ff = &mut b.f[..rows * f];
        ops::prefill(par, ff, &l.ff_in.b);
        linear(&b.a[..rows * h], rows, &l.ff_in.w, ff, par);
        ops::rows(par, ff, f, &|block, _| ops::gelu_slice(block));

        let x = &mut b.x[..rows * h];
        ops::prefill(par, x, &l.ff_out.b);
        linear(&b.f[..rows * f], rows, &l.ff_out.w, x, par);
        let a = &b.a;
        ops::rows(par, x, h, &|block, r0| {
            ops::add_layernorm(block, &a[r0 * h..], &l.norm2.w, &l.norm2.b, eps)
        });
    }

    for (k, &(o, s)) in b.seqs.iter().enumerate() {
        let v = &mut out[k * h..(k + 1) * h];
        match pooling {
            // the last layer left sequence k's [CLS] state in row k
            Pooling::Cls => v.copy_from_slice(&b.x[k * h..(k + 1) * h]),
            Pooling::Mean => {
                v.fill(0.0);
                for row in b.x[o * h..(o + s) * h].chunks_exact(h) {
                    v.iter_mut().zip(row).for_each(|(a, b)| *a += b);
                }
                v.iter_mut().for_each(|a| *a /= s as f32);
            }
        }
        let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        v.iter_mut().for_each(|a| *a /= norm);
    }
}
