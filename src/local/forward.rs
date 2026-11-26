//! The encoder forward pass over a chunk of unpadded token sequences, then
//! pooling and L2 normalisation. Every step but attention works per token,
//! so the tokens of all sequences sit in one (tokens, hidden) matrix.

use std::sync::Mutex;

use super::attention::{self, Scratch};
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

    for l in &w.layers {
        ops::prefill(par, &mut b.qkv, &l.qkv.b);
        linear(&b.x, t, &l.qkv.w, &mut b.qkv, par);
        ops::rows(par, &mut b.ctx, h, &|block, _| block.fill(0.0));
        attention::attention(&b.qkv, &b.seqs, h, c.heads, &mut b.ctx, par, &b.scratch);

        // the residual is added after the product, not preloaded into it: (x·W + b) + r
        // stays closer to an f64 pass
        ops::prefill(par, &mut b.a, &l.out.b);
        linear(&b.ctx, t, &l.out.w, &mut b.a, par);
        let x = &b.x;
        ops::rows(par, &mut b.a, h, &|block, r0| {
            ops::add_layernorm(block, &x[r0 * h..], &l.norm1.w, &l.norm1.b, eps)
        });

        ops::prefill(par, &mut b.f, &l.ff_in.b);
        linear(&b.a, t, &l.ff_in.w, &mut b.f, par);
        ops::rows(par, &mut b.f, f, &|block, _| ops::gelu_slice(block));

        ops::prefill(par, &mut b.x, &l.ff_out.b);
        linear(&b.f, t, &l.ff_out.w, &mut b.x, par);
        let a = &b.a;
        ops::rows(par, &mut b.x, h, &|block, r0| {
            ops::add_layernorm(block, &a[r0 * h..], &l.norm2.w, &l.norm2.b, eps)
        });
    }

    for (k, &(o, s)) in b.seqs.iter().enumerate() {
        let v = &mut out[k * h..(k + 1) * h];
        match pooling {
            Pooling::Cls => v.copy_from_slice(&b.x[o * h..(o + 1) * h]),
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
