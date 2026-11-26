//! The forward pass against vectors recorded from the previous (candle)
//! implementation and against an f64 reference. Each test skips a model whose
//! weights are not in the Hugging Face cache, and skips entirely in debug
//! builds, where the forward pass is too slow: run with `--release`.

use std::path::PathBuf;

use super::engine::Engine;
use super::model::{ModelInfo, models};
use super::safetensors::Safetensors;
use super::{hub, reference, wordpiece};

fn cached(m: &ModelInfo) -> Option<PathBuf> {
    let p = hub::Hub::from_env().snapshot(&m.pinned());
    if !p.exists() {
        eprintln!("skipped {}: weights not in the cache", m.name);
    }
    p.exists().then_some(p)
}

fn release() -> bool {
    if cfg!(debug_assertions) {
        eprintln!("skipped: needs --release");
    }
    !cfg!(debug_assertions)
}

/// The 38 texts, and per model and input type (none, query) the token count
/// and the vectors.
struct Recorded {
    texts: Vec<String>,
    runs: Vec<(u64, Vec<f32>)>,
}

fn recorded() -> Recorded {
    let raw = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/local_reference.bin"
    ))
    .unwrap();
    assert_eq!(&raw[..8], b"EMB07REF");
    let mut at = 8;
    let mut take = |n: usize| {
        at += n;
        &raw[at - n..at]
    };
    let count = u32::from_le_bytes(take(4).try_into().unwrap()) as usize;
    let texts: Vec<String> = (0..count)
        .map(|_| {
            let len = u32::from_le_bytes(take(4).try_into().unwrap()) as usize;
            String::from_utf8(take(len).to_vec()).unwrap()
        })
        .collect();
    let runs = (0..8)
        .map(|_| {
            let tokens = u64::from_le_bytes(take(8).try_into().unwrap());
            let v = take(count * 384 * 4)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            (tokens, v)
        })
        .collect();
    Recorded { texts, runs }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

#[test]
fn matches_the_recorded_vectors() {
    if !release() {
        return;
    }
    let rec = recorded();
    for (mi, m) in models().iter().enumerate() {
        let Some(path) = cached(m) else { continue };
        let engine = Engine::load(m, &path).unwrap();
        for (qi, query) in [false, true].into_iter().enumerate() {
            let prefix = m.query_prefix.filter(|_| query);
            let (got, tokens) = engine.encode(&rec.texts, prefix, false, None).unwrap();
            let (want_tokens, want) = &rec.runs[mi * 2 + qi];
            assert_eq!(tokens, *want_tokens, "{} tokens", m.name);
            let (mut worst, mut cos) = (0f32, 1f64);
            for (g, w) in got.chunks(384).zip(want.chunks(384)) {
                worst = g
                    .iter()
                    .zip(w)
                    .map(|(a, b)| (a - b).abs())
                    .fold(worst, f32::max);
                cos = cos.min(cosine(g, w));
            }
            eprintln!(
                "{} query={query}: max abs {worst:.2e}, min cosine 1 - {:.2e}",
                m.name,
                1.0 - cos
            );
            assert!(
                worst <= 2e-5 && cos >= 1.0 - 1e-6,
                "{} query={query}: {worst:e} / {cos}",
                m.name
            );
        }
    }
}

// on texts up to 64 tokens: our error to the f64 pass is at most 1.5 times
// the recorded implementation's
#[test]
fn as_close_to_f64_as_the_recorded_vectors() {
    if !release() {
        return;
    }
    let rec = recorded();
    for (mi, m) in models().iter().enumerate() {
        let Some(path) = cached(m) else { continue };
        let engine = Engine::load(m, &path).unwrap();
        let st = Safetensors::open(&path).unwrap();
        let (got, _) = engine.encode(&rec.texts, None, false, None).unwrap();
        let (mut ours, mut theirs) = (0f64, 0f64);
        let mut checked = 0;
        for (i, text) in rec.texts.iter().enumerate() {
            let mut ids = Vec::new();
            wordpiece::get().encode(text, m.max_tokens, &mut ids);
            if ids.len() > 64 {
                continue;
            }
            let exact = reference::embed(&st, &m.shape(), m.pooling, &ids);
            for (j, e) in exact.iter().enumerate() {
                ours = ours.max((got[i * 384 + j] as f64 - e).abs());
                theirs = theirs.max((rec.runs[mi * 2].1[i * 384 + j] as f64 - e).abs());
            }
            checked += 1;
        }
        eprintln!(
            "{}: {checked} texts, max abs to f64: ours {ours:.2e}, recorded {theirs:.2e}",
            m.name
        );
        assert!(
            checked >= 8 && ours <= 1.5 * theirs,
            "{}: {ours:e} vs {theirs:e}",
            m.name
        );
    }
}

// a text's vector does not depend on what else is in the call, chunk
// boundaries included, and the length limit is exact
#[test]
fn batches_and_limits() {
    if !release() {
        return;
    }
    let m = &models()[0];
    let Some(path) = cached(m) else { return };
    let engine = Engine::load(m, &path).unwrap();
    let rec = recorded();
    let mut texts: Vec<String> = (0..300)
        .map(|i| rec.texts[i % rec.texts.len()].clone())
        .collect();
    texts.push(String::new());
    let (all, _) = engine.encode(&texts, None, false, None).unwrap();
    // one thread gives the same vectors as the default pool
    let (one, _) = engine.encode(&texts, None, false, Some(1)).unwrap();
    let d = one
        .iter()
        .zip(&all)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(d <= 1e-5, "one thread: {d:e}");
    for (i, t) in texts.iter().enumerate().step_by(7) {
        let (one, _) = engine
            .encode(std::slice::from_ref(t), None, false, None)
            .unwrap();
        let d = one
            .iter()
            .zip(&all[i * 384..])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(d <= 1e-5, "text {i}: {d:e}");
    }
    // exactly the limit, and one token more cut back to it
    let word = |n: usize| vec!["stone"; n].join(" ");
    let at = word(m.max_tokens - 2);
    let over = word(m.max_tokens - 1);
    let (a, ta) = engine
        .encode(std::slice::from_ref(&at), None, false, None)
        .unwrap();
    let (b, tb) = engine
        .encode(std::slice::from_ref(&over), None, false, None)
        .unwrap();
    assert_eq!((ta, tb), (m.max_tokens as u64, m.max_tokens as u64));
    assert_eq!(a, b);
    assert!(engine.encode(&[at], None, true, None).is_ok());
    assert!(engine.encode(&[over], None, true, None).is_err());
}
