//! Property tests of the JSON writer and reader through the public API:
//! request bodies against serde_json's bytes, response parsing against the
//! values the body was printed from, and mutated bodies that must fail
//! cleanly. The cases per run are set by PROPTEST_CASES.

mod common;

use common::unused;
use embedrs::transport::HttpResponse;
use embedrs::{Client, Endpoint, Error};
use proptest::prelude::*;

fn any_text() -> impl Strategy<Value = String> {
    prop_oneof![
        any::<String>(),
        "[\\x00-\\x7f]{0,40}",
        proptest::collection::vec(
            prop_oneof![
                Just('"'),
                Just('\\'),
                Just('\n'),
                Just('\u{1}'),
                Just('é'),
                Just('😀'),
                any::<char>()
            ],
            0..30
        )
        .prop_map(|v| v.into_iter().collect()),
    ]
}

proptest! {
    #[test]
    fn request_body_matches_serde_json(texts in proptest::collection::vec(any_text(), 1..20), model in any_text()) {
        let c = Client::new(Endpoint::openai_compatible("k", "http://x"), unused()).with_model(model.clone());
        let p = c.prepare(&texts).unwrap();
        let want = format!(
            "{{\"model\":{},\"input\":{},\"encoding_format\":\"float\"}}",
            serde_json::to_string(&model).unwrap(),
            serde_json::to_string(&texts).unwrap()
        );
        prop_assert_eq!(String::from_utf8(p.request().body.clone()).unwrap(), want);
    }
}

#[derive(Debug, Clone)]
enum Fmt {
    Short,
    Exp,
    Wide,
    Nine,
    Base64,
}

fn print(v: f32, f: &Fmt) -> String {
    match f {
        Fmt::Short => format!("{v}"),
        Fmt::Exp => format!("{v:e}"),
        Fmt::Wide => format!("{}", v as f64),
        Fmt::Nine => format!("{v:.8e}"),
        Fmt::Base64 => unreachable!(),
    }
}

fn b64(values: &[f32]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut s = String::new();
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            s.push(if i <= c.len() {
                A[(n >> (18 - 6 * i) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    s
}

fn finite() -> impl Strategy<Value = f32> {
    // clearing the lowest exponent bit makes infinities and nans finite
    any::<u32>().prop_map(|b| match f32::from_bits(b) {
        v if v.is_finite() => v,
        _ => f32::from_bits(b & !(1 << 23)),
    })
}

fn ws() -> impl Strategy<Value = &'static str> {
    prop_oneof![Just(""), Just(" "), Just("\n  "), Just("\t"), Just("\r\n")]
}

/// A body with `rows` vectors in a shuffled order, printed with `fmt`, plus
/// fields the parser must skip; returns the body and the expected flat values.
fn body() -> impl Strategy<Value = (String, Vec<f32>, usize, usize)> {
    (1usize..6, 1usize..40, prop_oneof![Just(Fmt::Short), Just(Fmt::Exp), Just(Fmt::Wide), Just(Fmt::Nine), Just(Fmt::Base64)])
        .prop_flat_map(|(rows, dim, fmt)| {
            (
                proptest::collection::vec(finite(), rows * dim),
                Just(Vec::from_iter(0..rows)).prop_shuffle(),
                Just(fmt),
                Just(dim),
                ws(),
                any::<bool>(),
            )
        })
        .prop_map(|(values, order, fmt, dim, sp, extra)| {
            let rows = order.len();
            let mut items = Vec::new();
            let mut want = vec![0f32; rows * dim];
            for &i in &order {
                let row = &values[i * dim..(i + 1) * dim];
                let vector = match fmt {
                    Fmt::Base64 => format!("\"{}\"", b64(row)),
                    ref f => {
                        let parts: Vec<String> = row.iter().map(|v| print(*v, f)).collect();
                        format!("[{sp}{}{sp}]", parts.join(&format!(",{sp}")))
                    }
                };
                for (k, v) in row.iter().enumerate() {
                    // the reference is std's correctly rounded parse of the printed text
                    want[i * dim + k] = match fmt {
                        Fmt::Base64 => *v,
                        ref f => print(*v, f).parse().unwrap(),
                    };
                }
                let skip = if extra { r#""object":"embedding","meta":{"a":[1,2,{"b":null}],"c":"\"}"},"# } else { "" };
                items.push(format!("{{{sp}{skip}\"index\"{sp}:{sp}{i},{sp}\"embedding\":{sp}{vector}{sp}}}"));
            }
            let body = format!(
                "{sp}{{\"object\":\"list\",\"data\":[{}],{sp}\"model\":\"m\",\"usage\":{{\"prompt_tokens\":3,\"total_tokens\":{rows}}}}}{sp}",
                items.join(",")
            );
            (body, want, rows, dim)
        })
}

proptest! {
    #[test]
    fn response_parse_matches_the_printed_values((body, want, rows, dim) in body()) {
        let c = Client::new(Endpoint::openai_compatible("k", "http://x"), unused());
        let texts = vec!["t"; rows];
        let r = c.prepare(&texts).unwrap()
            .finish(HttpResponse::new(200, vec![], body.into_bytes()))
            .unwrap();
        prop_assert_eq!(r.dim(), dim);
        let got: Vec<u32> = r.as_flat().iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
        prop_assert_eq!(got, want);
        prop_assert_eq!(r.usage.total_tokens, Some(rows as u64));
    }

    #[test]
    fn mutated_bodies_fail_cleanly(
        (body, _, rows, _) in body(),
        cuts in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>(), 0u8..3), 1..6),
    ) {
        let mut bytes = body.into_bytes();
        for (at, byte, op) in cuts {
            if bytes.is_empty() {
                break;
            }
            let i = at.index(bytes.len());
            match op {
                0 => bytes[i] = byte,
                1 => { bytes.remove(i); }
                _ => bytes.insert(i, byte),
            }
        }
        let c = Client::new(Endpoint::openai_compatible("k", "http://x"), unused());
        let texts = vec!["t"; rows];
        match c.prepare(&texts).unwrap().finish(HttpResponse::new(200, vec![], bytes)) {
            Ok(r) => prop_assert_eq!(r.len(), rows),
            Err(Error::Decode(_)) => {}
            Err(e) => prop_assert!(false, "unexpected error {e:?}"),
        }
    }

    #[test]
    fn arbitrary_bytes_fail_cleanly(bytes in proptest::collection::vec(any::<u8>(), 0..300)) {
        for ep in [Endpoint::openai("k"), Endpoint::cohere("k"), Endpoint::gemini("k")] {
            let c = Client::new(ep, unused());
            let r = c.prepare(["a", "b"]).unwrap().finish(HttpResponse::new(200, vec![], bytes.clone()));
            prop_assert!(matches!(r, Ok(_) | Err(Error::Decode(_))));
        }
    }
}
