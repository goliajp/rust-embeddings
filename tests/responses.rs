//! Response parsing per provider, through `prepare` and `finish`: no sockets.

mod common;

use common::{ok, unused};
use embedrs::transport::HttpResponse;
use embedrs::{Client, Endpoint, Error, InputType};

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

fn finish(
    ep: Endpoint,
    texts: &[&str],
    resp: HttpResponse,
) -> embedrs::Result<embedrs::EmbedResult> {
    Client::new(ep, unused()).prepare(texts)?.finish(resp)
}

#[test]
fn openai_shape_decimal_and_base64() {
    for ep in [
        Endpoint::openai("k"),
        Endpoint::mistral("k"),
        Endpoint::jina("k"),
        Endpoint::voyage("k"),
    ] {
        let body = r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]},
            {"object":"embedding","index":1,"embedding":[0.4,0.5,0.6]}],
            "model":"served-model","usage":{"prompt_tokens":10,"total_tokens":10}}"#;
        let r = finish(ep.clone(), &["hello", "world"], ok(body)).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(&r[0], &[0.1, 0.2, 0.3]);
        assert_eq!(&r[1], &[0.4, 0.5, 0.6]);
        assert_eq!(r.usage.total_tokens, Some(10));
        assert_eq!(r.model, "served-model");

        let v = [1.5f32, -2.25, 3.0e-7];
        let body = format!(r#"{{"data":[{{"index":0,"embedding":"{}"}}]}}"#, b64(&v));
        let r = finish(ep, &["a"], ok(&body)).unwrap();
        assert_eq!(&r[0], &v);
        assert_eq!(r.usage.total_tokens, None);
    }
}

#[test]
fn openai_model_falls_back_to_the_requested_one() {
    let r = finish(
        Endpoint::openai("k"),
        &["a"],
        ok(r#"{"data":[{"embedding":[1]}]}"#),
    )
    .unwrap();
    assert_eq!(r.model, "text-embedding-3-small");
}

#[test]
fn cohere() {
    let body = format!(
        r#"{{"id":"x","embeddings":{{"base64":["{}","{}"]}},"texts":["a","b"],"meta":{{"api_version":{{"version":"2"}},"billed_units":{{"input_tokens":15}}}}}}"#,
        b64(&[1.0, 2.0]),
        b64(&[3.0, 4.0])
    );
    let r = finish(Endpoint::cohere("k"), &["a", "b"], ok(&body)).unwrap();
    assert_eq!(r.to_vecs(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert_eq!(r.usage.total_tokens, Some(15));
    assert_eq!(r.model, "embed-v4.0");
}

#[test]
fn gemini_single_and_batch() {
    let c = Client::new(Endpoint::gemini("k"), unused()).with_input_type(InputType::SearchQuery);
    let one = c.prepare(["a"]).unwrap();
    assert!(
        one.request()
            .url
            .ends_with("/models/gemini-embedding-001:embedContent")
    );
    let r = one
        .finish(ok(
            r#"{"embedding":{"values":[0.5,0.25]},"usageMetadata":{"promptTokenCount":3}}"#,
        ))
        .unwrap();
    assert_eq!((&r[0], r.usage.total_tokens), (&[0.5, 0.25][..], Some(3)));
    let two = c.prepare(["a", "b"]).unwrap();
    assert!(two.request().url.ends_with(":batchEmbedContents"));
    let r = two
        .finish(ok(r#"{"embeddings":[{"values":[1]},{"values":[2]}]}"#))
        .unwrap();
    assert_eq!(r.as_flat(), &[1.0, 2.0]);
    assert_eq!(r.usage.total_tokens, None);
}

#[test]
fn error_bodies_become_api_errors() {
    let cases = [
        (
            Endpoint::openai("k"),
            401,
            r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error"}}"#,
            "Incorrect API key",
        ),
        (
            Endpoint::cohere("k"),
            400,
            r#"{"message":"invalid request: texts must not be empty"}"#,
            "invalid request: texts must not be empty",
        ),
        (
            Endpoint::gemini("k"),
            400,
            r#"{"error":{"code":400,"message":"API key not valid","status":"INVALID_ARGUMENT"}}"#,
            "API key not valid",
        ),
        (
            Endpoint::voyage("k"),
            429,
            r#"{"detail":"rate limited"}"#,
            "rate limited",
        ),
        (
            Endpoint::jina("k"),
            500,
            "upstream connect error",
            "upstream connect error",
        ),
    ];
    for (ep, status, body, want) in cases {
        let provider = ep.provider();
        let resp = HttpResponse::new(
            status,
            vec![("Retry-After".into(), "2".into())],
            body.as_bytes().to_vec(),
        );
        match finish(ep, &["a"], resp) {
            Err(Error::Api {
                status: s,
                message,
                retry_after,
                provider: p,
                ..
            }) => {
                assert_eq!((s, message.as_str(), p), (status, want, provider));
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(2)));
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn malformed_2xx_bodies_are_decode_errors() {
    for body in [
        "",
        "{",
        "not json",
        r#"{"data":[{"embedding":[1,"x"]}]}"#,
        r#"{"data":[]}"#,
    ] {
        let e = finish(Endpoint::openai("k"), &["a"], ok(body)).unwrap_err();
        match e {
            Error::Decode(d) => assert_eq!(d.provider, "openai"),
            other => panic!("{body}: {other:?}"),
        }
    }
}

#[test]
fn unsupported_options_are_refused_before_sending() {
    let c = Client::new(Endpoint::mistral("k"), unused());
    assert!(matches!(
        c.embed(["a"]).dimensions(64).prepare(),
        Err(Error::Unsupported { ref option, .. }) if option == "dimensions(64)"
    ));
    let c = Client::new(Endpoint::gemini("k"), unused()).with_truncate(embedrs::Truncate::End);
    assert!(matches!(c.prepare(["a"]), Err(Error::Unsupported { .. })));
}

#[test]
fn truncate_switches() {
    use embedrs::Truncate;
    let body = |ep: Endpoint, t: Truncate| {
        let p = Client::new(ep, unused())
            .with_truncate(t)
            .prepare(["a"])
            .unwrap();
        String::from_utf8(p.request().body.clone()).unwrap()
    };
    assert!(body(Endpoint::voyage("k"), Truncate::End).ends_with(r#","truncation":true}"#));
    assert!(body(Endpoint::voyage("k"), Truncate::Error).ends_with(r#","truncation":false}"#));
    assert!(body(Endpoint::cohere("k"), Truncate::End).ends_with(r#","truncate":"END"}"#));
    assert!(body(Endpoint::cohere("k"), Truncate::Error).ends_with(r#","truncate":"NONE"}"#));
    assert!(body(Endpoint::jina("k"), Truncate::End).ends_with(r#","truncate":true}"#));
    assert!(!body(Endpoint::openai("k"), Truncate::Error).contains("trunc"));
    for ep in [Endpoint::openai("k"), Endpoint::mistral("k")] {
        let c = Client::new(ep, unused()).with_truncate(Truncate::End);
        assert!(matches!(c.prepare(["a"]), Err(Error::Unsupported { .. })));
    }
}

#[test]
fn extra_headers_and_bearer_off() {
    let ep = Endpoint::openai_compatible("", "https://res.openai.azure.com/openai/deployments/d")
        .bearer(false)
        .header("api-key", "azure-key");
    let p = Client::new(ep, unused()).prepare(["a"]).unwrap();
    let h = &p.request().headers;
    assert!(h.iter().all(|(k, _)| *k != "authorization"));
    assert!(h.contains(&("api-key", "azure-key".to_string())));
    assert!(format!("{:?}", p.request()).contains("<redacted>"));
}
