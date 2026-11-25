//! Requests are byte-identical to the ones embedrs 0.6.2 sent. The fixture
//! was recorded from 0.6.2 against a local server: method, path, the
//! content-type and credential headers, and the exact body, for every
//! provider, model kind, dimensions, input type and a set of awkward texts.

mod common;

use embedrs::{Client, Endpoint, Error, InputType};
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/requests-0.6.2.json");
const BASE: &str = "http://127.0.0.1:PORT/v9";

fn endpoint(provider: &str) -> Endpoint {
    match provider {
        "openai" => Endpoint::openai_compatible("key-1", BASE),
        "cohere" => Endpoint::cohere_compatible("key-2", BASE),
        "gemini" => Endpoint::gemini_compatible("key-3", BASE),
        "voyage" => Endpoint::voyage_compatible("key-4", BASE),
        "jina" => Endpoint::jina_compatible("key-5", BASE),
        "mistral" => Endpoint::mistral_compatible("key-6", BASE),
        p => panic!("unknown provider {p}"),
    }
}

fn input_type(s: &str) -> InputType {
    match s {
        "SearchDocument" => InputType::SearchDocument,
        "SearchQuery" => InputType::SearchQuery,
        "Classification" => InputType::Classification,
        "Clustering" => InputType::Clustering,
        s => panic!("unknown input type {s}"),
    }
}

#[test]
fn every_request_matches_0_6_2() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 510);
    let (mut sent, mut refused) = (0, 0);
    for (i, case) in cases.iter().enumerate() {
        let client = Client::new(
            endpoint(case["provider"].as_str().unwrap()),
            common::unused(),
        );
        let texts: Vec<&str> = case["texts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();
        let mut b = client.embed(&texts);
        if let Some(m) = case["model"].as_str() {
            b = b.model(m);
        }
        if let Some(d) = case["dimensions"].as_u64() {
            b = b.dimensions(d as u32);
        }
        if let Some(t) = case["input_type"].as_str() {
            b = b.input_type(input_type(t));
        }
        let got = b.prepare();
        if case["error"].is_string() {
            assert!(
                matches!(got, Err(Error::Unsupported { .. })),
                "case {i}: {got:?}"
            );
            refused += 1;
            continue;
        }
        let prepared = got.unwrap_or_else(|e| panic!("case {i}: {e}"));
        let req = prepared.request();
        let want = &case["request"];
        assert_eq!(req.method.as_str(), want["method"], "case {i}");
        assert_eq!(
            req.url.strip_prefix("http://127.0.0.1:PORT").unwrap(),
            want["path"].as_str().unwrap(),
            "case {i}"
        );
        let headers: serde_json::Map<String, Value> = req
            .headers
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.clone())))
            .collect();
        assert_eq!(&Value::Object(headers), &want["headers"], "case {i}");
        assert_eq!(
            String::from_utf8(req.body.clone()).unwrap(),
            want["body"].as_str().unwrap(),
            "case {i}"
        );
        sent += 1;
    }
    assert_eq!((sent, refused), (450, 60));
}

#[test]
fn official_openai_endpoint_asks_for_base64() {
    let client = Client::new(Endpoint::openai("k"), common::unused());
    let p = client.prepare(["hi"]).unwrap();
    assert_eq!(p.request().url, "https://api.openai.com/v1/embeddings");
    assert_eq!(
        p.request().body,
        br#"{"model":"text-embedding-3-small","input":["hi"],"encoding_format":"base64"}"#
    );
}
