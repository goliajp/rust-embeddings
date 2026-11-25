use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::rt::join::tests::block_on;
use crate::transport::{BoxFuture, HttpResponse, ResponseHead, TransportError, TransportErrorKind};

/// Blocking HTTP/1.1 over std sockets: enough for these tests.
struct StdTransport;

fn exchange(req: &HttpRequest) -> io::Result<(u16, Vec<(String, String)>, BufReader<TcpStream>, Option<u64>)> {
    let rest = req.url.strip_prefix("http://").expect("http url");
    let (host, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
    let mut s = TcpStream::connect(host)?;
    write!(s, "{} {path} HTTP/1.1\r\nhost: {host}\r\nconnection: close\r\n", req.method.as_str())?;
    for (k, v) in &req.headers {
        write!(s, "{k}: {v}\r\n")?;
    }
    write!(s, "\r\n")?;
    let mut r = BufReader::new(s);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let status = line.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let mut headers = Vec::new();
    let mut len = None;
    loop {
        let mut h = String::new();
        r.read_line(&mut h)?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':').expect("header line");
        if k.eq_ignore_ascii_case("content-length") {
            len = v.trim().parse().ok();
        }
        headers.push((k.to_ascii_lowercase(), v.trim().to_string()));
    }
    Ok((status, headers, r, len))
}

impl Transport for StdTransport {
    fn send(&self, req: HttpRequest) -> BoxFuture<'_, std::result::Result<HttpResponse, TransportError>> {
        let r = exchange(&req).and_then(|(status, headers, mut r, _)| {
            let mut body = Vec::new();
            r.read_to_end(&mut body)?;
            Ok(HttpResponse::new(status, headers, body))
        });
        Box::pin(std::future::ready(r.map_err(|e| TransportError::new(TransportErrorKind::Io, e))))
    }

    fn sleep(&self, _: Duration) -> BoxFuture<'static, ()> {
        Box::pin(std::future::ready(()))
    }

    fn download<'a>(
        &'a self,
        req: HttpRequest,
        sink: &'a mut (dyn Write + Send),
    ) -> BoxFuture<'a, std::result::Result<ResponseHead, TransportError>> {
        let r = exchange(&req).and_then(|(status, headers, mut r, len)| {
            if (200..300).contains(&status) {
                let n = io::copy(&mut r, sink)?;
                if len.is_some_and(|l| l != n) {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "body cut short"));
                }
            }
            Ok(ResponseHead::new(status, headers))
        });
        Box::pin(std::future::ready(r.map_err(|e| TransportError::new(TransportErrorKind::Io, e))))
    }
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    // close the connection after this many body bytes
    cut: Option<usize>,
}

type Seen = Arc<Mutex<Vec<String>>>;

/// Serves `handler(request head)` on every connection; returns the base URL
/// and the request heads it saw.
fn server(handler: impl Fn(&str) -> Reply + Send + Sync + 'static) -> (String, Seen) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let seen: Seen = Default::default();
    let log = seen.clone();
    let handler = Arc::new(handler);
    std::thread::spawn(move || {
        for s in l.incoming() {
            let mut s = s.unwrap();
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut head = String::new();
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                head.push_str(&line.to_ascii_lowercase());
            }
            log.lock().unwrap().push(head.clone());
            let reply = handler(&head);
            let mut out = format!("HTTP/1.1 {} X\r\ncontent-length: {}\r\n", reply.status, reply.body.len());
            for (k, v) in &reply.headers {
                out.push_str(&format!("{k}: {v}\r\n"));
            }
            out.push_str("connection: close\r\n\r\n");
            let _ = s.write_all(out.as_bytes());
            let body = &reply.body[..reply.cut.unwrap_or(reply.body.len())];
            let _ = s.write_all(body);
        }
    });
    (base, seen)
}

fn blob() -> Vec<u8> {
    (0..3_000_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect()
}

fn pinned(data: &[u8]) -> Pinned {
    let mut h = Sha256::new();
    h.update(data);
    Pinned {
        repo: "org/model",
        revision: "0123456789abcdef0123456789abcdef01234567",
        file: "model.safetensors",
        size: data.len() as u64,
        sha256: Box::leak(hex(&h.finish()).into_boxed_str()),
    }
}

fn scratch(name: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "embedrs-hub-{name}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn hub(cache: &Path, endpoint: &str, token: Option<&str>) -> Hub {
    Hub {
        cache: cache.to_path_buf(),
        endpoint: endpoint.to_string(),
        token: token.map(str::to_string),
    }
}

fn range_start(head: &str) -> Option<usize> {
    head.lines()
        .find_map(|l| l.strip_prefix("range: bytes="))
        .and_then(|v| v.trim().trim_end_matches('-').parse().ok())
}

/// Serves `data`, honouring `range` when `ranges` is true.
fn serving(data: Vec<u8>, ranges: bool, cut: Option<usize>) -> impl Fn(&str) -> Reply {
    move |head| match range_start(head).filter(|_| ranges) {
        Some(start) => Reply {
            status: 206,
            headers: vec![("content-range".into(), format!("bytes {start}-{}/{}", data.len() - 1, data.len()))],
            body: data[start..].to_vec(),
            cut: None,
        },
        None => Reply {
            status: 200,
            headers: vec![],
            body: data.clone(),
            cut,
        },
    }
}

#[test]
fn clean_download_is_published_and_found_again() {
    let data = blob();
    let p = pinned(&data);
    let (base, seen) = server(serving(data.clone(), true, None));
    let dir = scratch("clean");
    let h = hub(&dir, &base, None);
    let path = block_on(h.ensure(&p, &StdTransport)).unwrap();
    assert_eq!(path, dir.join("models--org--model/snapshots").join(p.revision).join(p.file));
    assert_eq!(fs::read(&path).unwrap(), data);
    #[cfg(unix)]
    assert!(fs::symlink_metadata(&path).unwrap().file_type().is_symlink());
    assert!(seen.lock().unwrap()[0].starts_with("get /org/model/resolve/0123456789abcdef"));
    assert_eq!(block_on(h.ensure(&p, &StdTransport)).unwrap(), path);
    assert_eq!(seen.lock().unwrap().len(), 1);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn interrupted_download_resumes_with_a_range() {
    let data = blob();
    let p = pinned(&data);
    let dir = scratch("resume");
    let (base, _) = server(serving(data.clone(), true, Some(1_234_567)));
    assert!(block_on(hub(&dir, &base, None).ensure(&p, &StdTransport)).is_err());
    let part = dir.join("models--org--model/blobs").join(format!("{}.incomplete", p.sha256));
    assert_eq!(fs::metadata(&part).unwrap().len(), 1_234_567);
    let (base, seen) = server(serving(data.clone(), true, None));
    let path = block_on(hub(&dir, &base, None).ensure(&p, &StdTransport)).unwrap();
    assert_eq!(fs::read(path).unwrap(), data);
    assert_eq!(range_start(&seen.lock().unwrap()[0]), Some(1_234_567));
    assert!(!part.exists());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn server_ignoring_the_range_restarts_cleanly() {
    let data = blob();
    let p = pinned(&data);
    let dir = scratch("norange");
    let (base, _) = server(serving(data.clone(), false, Some(999_999)));
    assert!(block_on(hub(&dir, &base, None).ensure(&p, &StdTransport)).is_err());
    let (base, seen) = server(serving(data.clone(), false, None));
    let path = block_on(hub(&dir, &base, None).ensure(&p, &StdTransport)).unwrap();
    assert_eq!(fs::read(path).unwrap(), data);
    assert_eq!(range_start(&seen.lock().unwrap()[0]), Some(999_999));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn wrong_content_is_rejected_and_removed() {
    let data = blob();
    let p = pinned(&data);
    let mut bad = data.clone();
    bad[1_000_000] ^= 1;
    let dir = scratch("badhash");
    let (base, _) = server(serving(bad, true, None));
    let e = block_on(hub(&dir, &base, None).ensure(&p, &StdTransport)).unwrap_err();
    assert!(e.to_string().contains("failed verification"), "{e}");
    let blobs = dir.join("models--org--model/blobs");
    assert_eq!(fs::read_dir(&blobs).unwrap().count(), 0);
    assert!(!h_snapshot_exists(&dir, &p));
    fs::remove_dir_all(&dir).unwrap();
}

fn h_snapshot_exists(dir: &Path, p: &Pinned) -> bool {
    hub(dir, "", None).snapshot(p).exists()
}

#[test]
fn redirect_to_another_host_drops_the_token() {
    let data = blob();
    let p = pinned(&data);
    let (cdn, cdn_seen) = server(serving(data.clone(), true, None));
    let (base, seen) = server(move |_| Reply {
        status: 302,
        headers: vec![("location".into(), format!("{cdn}/blob?sig=x"))],
        body: vec![],
        cut: None,
    });
    let dir = scratch("redirect");
    let path = block_on(hub(&dir, &base, Some("hf_secret")).ensure(&p, &StdTransport)).unwrap();
    assert_eq!(fs::read(path).unwrap(), data);
    assert!(seen.lock().unwrap()[0].contains("authorization: bearer hf_secret"));
    let cdn_head = &cdn_seen.lock().unwrap()[0];
    assert!(cdn_head.starts_with("get /blob?sig=x"), "{cdn_head}");
    assert!(!cdn_head.contains("authorization"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn concurrent_callers_download_once() {
    let data = blob();
    let p = pinned(&data);
    let (base, seen) = server(serving(data.clone(), true, None));
    let dir = scratch("race");
    let h = hub(&dir, &base, None);
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let h = h.clone();
            std::thread::spawn(move || block_on(h.ensure(&p, &StdTransport)).unwrap())
        })
        .collect();
    for t in threads {
        assert_eq!(fs::read(t.join().unwrap()).unwrap(), data);
    }
    assert_eq!(seen.lock().unwrap().len(), 1);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn urls() {
    assert_eq!(origin("https://cdn.example/a/b?c"), "https://cdn.example");
    assert_eq!(resolve("https://h/a/b", "https://x/y"), "https://x/y");
    assert_eq!(resolve("https://h/a/b", "/z"), "https://h/z");
    assert_eq!(resolve("https://h/a/b", "c"), "https://h/a/c");
}
