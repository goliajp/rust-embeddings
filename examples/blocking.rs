//! Embed without an async runtime and without any dependency: embedrs builds
//! the request and reads the answer, a std `TcpStream` carries it.
//!
//! ```bash
//! ollama pull nomic-embed-text
//! cargo run --example blocking --no-default-features
//! ```
//!
//! `EMBED_URL` (default `http://localhost:11434/v1`) can be any plain-HTTP
//! OpenAI-compatible server, `EMBED_MODEL` any model it serves.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use embedrs::transport::{
    BoxFuture, HttpRequest, HttpResponse, Transport, TransportError, TransportErrorKind,
};
use embedrs::{Client, Endpoint, cosine_similarity};

/// One HTTP/1.0 exchange over a std socket.
fn exchange(req: &HttpRequest) -> std::io::Result<HttpResponse> {
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
    let rest = req
        .url
        .strip_prefix("http://")
        .ok_or_else(|| bad("only http:// urls"))?;
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    let mut head = format!(
        "{} /{path} HTTP/1.0\r\nhost: {host}\r\n",
        req.method.as_str()
    );
    for (name, value) in &req.headers {
        head += &format!("{name}: {value}\r\n");
    }
    head += &format!("content-length: {}\r\n\r\n", req.body.len());

    let mut socket = TcpStream::connect(host)?;
    socket.write_all(head.as_bytes())?;
    socket.write_all(&req.body)?;
    let mut answer = Vec::new();
    socket.read_to_end(&mut answer)?;

    let split = answer
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| bad("no header end"))?;
    let head = String::from_utf8_lossy(&answer[..split]).into_owned();
    let mut lines = head.lines();
    let status = lines
        .next()
        .and_then(|l| l.split(' ').nth(1)?.parse().ok())
        .ok_or_else(|| bad("no status"))?;
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    Ok(HttpResponse::new(
        status,
        headers,
        answer[split + 4..].to_vec(),
    ))
}

/// The same exchange behind the async trait; it blocks the polling thread.
struct Std;

impl Transport for Std {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        let answer = exchange(&req).map_err(|e| TransportError::new(TransportErrorKind::Io, e));
        Box::pin(std::future::ready(answer))
    }

    fn sleep(&self, d: Duration) -> BoxFuture<'static, ()> {
        Box::pin(async move { std::thread::sleep(d) })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    let endpoint = Endpoint::openai_compatible(
        var("EMBED_KEY", "none"),
        var("EMBED_URL", "http://localhost:11434/v1"),
    );
    let client =
        Client::new(endpoint, Arc::new(Std)).with_model(var("EMBED_MODEL", "nomic-embed-text"));

    let prepared = client.prepare(["a cat on a sofa", "a kitten on a couch"])?;
    let answer = exchange(prepared.request())?;
    let result = prepared.finish(answer)?;
    println!(
        "{}: {} vectors of {} dimensions, cosine {:.3}",
        result.model,
        result.len(),
        result.dim(),
        cosine_similarity(&result[0], &result[1])
    );
    Ok(())
}
