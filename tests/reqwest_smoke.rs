//! The default transport against a real socket.
#![cfg(feature = "reqwest")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;

use embedrs::transport::TransportErrorKind;
use embedrs::{Backoff, Client, Error};

struct Seen {
    head: String,
    body: Vec<u8>,
}

/// Serves the scripted (status line, extra headers, body, delay) replies, one per connection.
fn serve(
    replies: Vec<(&'static str, &'static str, &'static str, u64)>,
) -> (String, mpsc::Receiver<Seen>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for (status, headers, body, delay) in replies {
            let (mut s, _) = l.accept().unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut head = String::new();
            let mut len = 0;
            loop {
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
                if line == "\r\n" {
                    break;
                }
                head.push_str(&line);
            }
            let mut b = vec![0; len];
            r.read_exact(&mut b).unwrap();
            tx.send(Seen { head, body: b }).unwrap();
            std::thread::sleep(Duration::from_millis(delay));
            let _ = write!(
                s,
                "HTTP/1.1 {status}\r\n{headers}content-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (url, rx)
}

const OK: &str =
    r#"{"data":[{"index":0,"embedding":[0.25,0.5]}],"model":"m","usage":{"total_tokens":2}}"#;

#[tokio::test]
async fn sends_headers_and_body_and_retries_on_429() {
    let (url, seen) = serve(vec![
        (
            "429 Too Many Requests",
            "retry-after: 0\r\n",
            r#"{"error":{"message":"slow"}}"#,
            0,
        ),
        ("200 OK", "", OK, 0),
    ]);
    let client = Client::openai_compatible("sk-test", &url)
        .with_retry(Backoff::new().base(Duration::from_millis(1)));
    let r = client.embed(["hello"]).await.unwrap();
    assert_eq!(&r[0], &[0.25, 0.5]);
    assert_eq!(r.usage.total_tokens, Some(2));
    for _ in 0..2 {
        let s = seen.recv().unwrap();
        let head = s.head.to_ascii_lowercase();
        assert!(head.starts_with("post /embeddings http/1.1"), "{head}");
        assert!(head.contains("authorization: bearer sk-test"));
        assert!(head.contains("content-type: application/json"));
        assert_eq!(
            s.body,
            br#"{"model":"text-embedding-3-small","input":["hello"],"encoding_format":"float"}"#
        );
    }
}

#[tokio::test]
async fn api_error_carries_status_and_message() {
    let (url, _seen) = serve(vec![(
        "401 Unauthorized",
        "",
        r#"{"error":{"message":"bad key"}}"#,
        0,
    )]);
    let e = Client::openai_compatible("k", &url)
        .embed(["a"])
        .await
        .unwrap_err();
    assert!(
        matches!(e, Error::Api { status: 401, ref message, .. } if message == "bad key"),
        "{e:?}"
    );
}

#[tokio::test]
async fn refused_connection_is_a_connect_error() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let e = Client::openai_compatible("k", format!("http://127.0.0.1:{port}"))
        .embed(["a"])
        .await
        .unwrap_err();
    match e {
        Error::Transport(t) => assert_eq!(t.kind(), TransportErrorKind::Connect),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn slow_server_hits_the_timeout() {
    let (url, _seen) = serve(vec![("200 OK", "", OK, 2000)]);
    let e = Client::openai_compatible("k", &url)
        .with_timeout(Duration::from_millis(200))
        .embed(["a"])
        .await
        .unwrap_err();
    assert!(matches!(e, Error::Timeout(_)), "{e:?}");
}
