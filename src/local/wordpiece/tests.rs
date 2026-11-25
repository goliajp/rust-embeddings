// differential tests against the `tokenizers` crate, configured as the
// bundled models used it: no padding, truncation at the model's limit
use std::sync::LazyLock;

use proptest::prelude::*;
use proptest::test_runner::{TestRng, TestRunner};
use tokenizers::{Tokenizer, TruncationParams};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

// the four bundled tokenizer files and the model limits they ran with
const CONFIGS: [(&str, usize); 4] = [
    ("all_minilm_l6_v2", 256),
    ("all_minilm_l12_v2", 128),
    ("bge_small_en_v1_5", 512),
    ("gte_small", 512),
];

fn reference(name: &str, max_len: usize) -> Tokenizer {
    let raw = std::fs::read(format!("{FIXTURES}/tokenizers/{name}.json.zst")).unwrap();
    let json = zstd::decode_all(&raw[..]).unwrap();
    let mut tok = Tokenizer::from_bytes(&json).unwrap();
    tok.with_padding(None);
    tok.with_truncation(Some(TruncationParams {
        max_length: max_len,
        ..Default::default()
    }))
    .unwrap();
    tok
}

static REFERENCES: LazyLock<Vec<(String, Tokenizer)>> = LazyLock::new(|| {
    CONFIGS
        .iter()
        .map(|&(name, len)| (name.to_owned(), reference(name, len)))
        .collect()
});

fn env_or(key: &str, default: usize) -> usize {
    std::env::var(key).map_or(default, |v| v.parse().unwrap())
}

/// Number of texts whose ids differ from `tok`'s; prints the first few.
fn mismatches(tok: &Tokenizer, label: &str, texts: &[String]) -> usize {
    let max_len = tok.get_truncation().unwrap().max_length;
    let want = match texts {
        [one] => vec![tok.encode(one.as_str(), true).unwrap()],
        _ => tok.encode_batch(texts.to_vec(), true).unwrap(),
    };
    let wp = super::get();
    let mut bad = 0;
    let mut got = Vec::new();
    for (text, want) in texts.iter().zip(&want) {
        got.clear();
        wp.encode(text, max_len, &mut got);
        if got != want.get_ids() {
            bad += 1;
            if bad <= 5 {
                eprintln!(
                    "{label} max_len {max_len}: {text:?}\n  want {:?}\n  got  {got:?}",
                    want.get_ids()
                );
            }
        }
    }
    bad
}

fn check_all(texts: &[String]) -> usize {
    REFERENCES
        .iter()
        .map(|(name, tok)| mismatches(tok, name, texts))
        .sum()
}

// every scalar value alone, inside "a{c}b" and doubled; EMBEDRS_WP_STRIDE=1
// runs all of them
#[test]
fn every_scalar_value() {
    let stride = env_or("EMBEDRS_WP_STRIDE", 61);
    let chars: Vec<char> = (0..=0x10FFFFu32)
        .step_by(stride)
        .filter_map(char::from_u32)
        .collect();
    let mut bad = 0;
    for chunk in chars.chunks(4096) {
        let mut texts = Vec::with_capacity(chunk.len() * 3);
        for c in chunk {
            texts.push(c.to_string());
            texts.push(format!("a{c}b"));
            texts.push(format!("{c}{c}"));
        }
        bad += check_all(&texts);
    }
    eprintln!(
        "scalar values checked: {} (stride {stride}), mismatches: {bad}",
        chars.len()
    );
    assert_eq!(bad, 0);
}

// records are "<byte length>\n<text>\n"
fn corpus() -> Vec<String> {
    let raw = std::fs::read(format!("{FIXTURES}/wordpiece-corpus.zst")).unwrap();
    let data = String::from_utf8(zstd::decode_all(&raw[..]).unwrap()).unwrap();
    let mut texts = Vec::new();
    let mut rest = data.as_str();
    while let Some(nl) = rest.find('\n') {
        let len: usize = rest[..nl].parse().unwrap();
        texts.push(rest[nl + 1..nl + 1 + len].to_owned());
        rest = &rest[nl + 2 + len..];
    }
    texts
}

#[test]
fn fixture_corpus() {
    let texts = corpus();
    assert!(texts.len() > 1000);
    let mut bad = 0;
    for (name, _) in REFERENCES.iter() {
        for max_len in [128, 256, 512] {
            bad += mismatches(&reference(name, max_len), name, &texts);
        }
    }
    eprintln!("corpus texts: {}, mismatches: {bad}", texts.len());
    assert_eq!(bad, 0);
}

#[test]
fn truncation_edges() {
    let wp = super::get();
    let long = "word ".repeat(600);
    let tok = reference("all_minilm_l6_v2", 2);
    for max_len in [2, 3, 4, 127, 128, 129, 511, 512] {
        let tok = {
            let mut t = tok.clone();
            t.with_truncation(Some(TruncationParams {
                max_length: max_len,
                ..Default::default()
            }))
            .unwrap();
            t
        };
        let mut got = Vec::new();
        wp.encode(&long, max_len, &mut got);
        assert_eq!(got, tok.encode(long.as_str(), true).unwrap().get_ids());
    }
    // encode appends to what is already there
    let mut out = vec![7];
    wp.encode("hello", 8, &mut out);
    assert_eq!(out[0], 7);
    assert_eq!(out.len(), 4);
}

// pieces random texts are built from: scripts, marks, specials, separators
fn fragment() -> impl Strategy<Value = String> {
    let word = |lo: u32, hi: u32| {
        proptest::collection::vec(
            proptest::char::range(char::from_u32(lo).unwrap(), char::from_u32(hi).unwrap()),
            1..12,
        )
        .prop_map(|v| v.into_iter().collect::<String>())
    };
    let pick =
        |items: &'static [&'static str]| proptest::sample::select(items).prop_map(str::to_owned);
    prop_oneof![
        4 => word(0x61, 0x7A),
        2 => word(0x41, 0x5A),
        2 => word(0xC0, 0x24F),
        1 => word(0x370, 0x3FF),
        1 => word(0x400, 0x4FF),
        1 => word(0x600, 0x6FF),
        1 => word(0x900, 0x97F),
        1 => word(0xE00, 0xE7F),
        1 => word(0x3040, 0x30FF),
        1 => word(0x4E00, 0x4F00),
        1 => word(0xAC00, 0xD7A3),
        1 => word(0x300, 0x36F),
        1 => word(0x1D15E, 0x1D172),
        1 => word(0x1E000, 0x1E02A),
        1 => word(0xF900, 0xFAFF),
        1 => word(0x1F300, 0x1FAFF),
        1 => any::<char>().prop_map(String::from),
        1 => (0u32..0x20).prop_map(|c| char::from_u32(c).unwrap().to_string()),
        2 => pick(&[" ", "  ", "\t", "\n", "\r\n", "\u{a0}", "\u{3000}", "\u{2028}", "\u{85}", "\u{b}"]),
        2 => pick(&[".", ",", "!", "?", "'", "\"", "-", "(", ")", "#", "##", "¿", "«", "»", "、", "。"]),
        1 => pick(&["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[mask]", "[MASK", "[[CLS]]"]),
        1 => pick(&["\u{1F469}\u{200D}\u{1F4BB}", "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}", "\u{FE0F}", "\u{200D}", "\u{FFFD}", "\u{0}"]),
    ]
}

fn random_text() -> impl Strategy<Value = String> {
    proptest::collection::vec(fragment(), 0..40).prop_map(|v| v.concat())
}

// EMBEDRS_WP_CASES sets the number of random texts
#[test]
fn random_mixed_script() {
    let cases = env_or("EMBEDRS_WP_CASES", 10_000);
    let config = ProptestConfig {
        cases: cases as u32,
        failure_persistence: None,
        ..ProptestConfig::default()
    };
    let checked = std::sync::atomic::AtomicUsize::new(0);
    let rng = TestRng::deterministic_rng(config.rng_algorithm);
    TestRunner::new_with_rng(config, rng)
        .run(&random_text(), |text| {
            checked.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            prop_assert_eq!(check_all(&[text]), 0);
            Ok(())
        })
        .unwrap();
    eprintln!(
        "random texts checked: {}, mismatches: 0",
        checked.into_inner()
    );
}
