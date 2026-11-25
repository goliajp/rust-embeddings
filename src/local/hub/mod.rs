//! Model files from the Hugging Face Hub, through the client's transport,
//! into the cache layout `huggingface_hub` uses, so files downloaded by
//! either are found by both.
//!
//! Every file is pinned to a commit and a SHA-256: the download needs no
//! metadata request and is verified before it is published.

mod sha256;

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use self::sha256::{Sha256, hex};
use super::fail;
use crate::error::Result;
use crate::transport::{HttpRequest, Method, Transport};

const MAX_REDIRECTS: usize = 5;

/// One file of one repository at one commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pinned {
    pub(crate) repo: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) file: &'static str,
    pub(crate) size: u64,
    pub(crate) sha256: &'static str,
}

/// Where files come from and go.
#[derive(Debug, Clone)]
pub(crate) struct Hub {
    pub(crate) cache: PathBuf,
    pub(crate) endpoint: String,
    pub(crate) token: Option<String>,
}

/// The cache directory: `HF_HUB_CACHE`, else `HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`.
pub(crate) fn cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return p.into();
    }
    if let Some(p) = std::env::var_os("HF_HOME") {
        return PathBuf::from(p).join("hub");
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

impl Hub {
    /// Settings from the environment: the cache above, `HF_ENDPOINT`, `HF_TOKEN`.
    pub(crate) fn from_env() -> Self {
        Self {
            cache: cache_dir(),
            endpoint: std::env::var("HF_ENDPOINT")
                .unwrap_or_else(|_| "https://huggingface.co".into()),
            token: std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()),
        }
    }

    fn repo_dir(&self, p: &Pinned) -> PathBuf {
        self.cache
            .join(format!("models--{}", p.repo.replace('/', "--")))
    }

    /// The path the file has once downloaded.
    pub(crate) fn snapshot(&self, p: &Pinned) -> PathBuf {
        self.repo_dir(p)
            .join("snapshots")
            .join(p.revision)
            .join(p.file)
    }

    /// The file's path, downloading and verifying it first if it is not cached.
    pub(crate) async fn ensure(&self, p: &Pinned, transport: &dyn Transport) -> Result<PathBuf> {
        let path = self.snapshot(p);
        if path.exists() {
            return Ok(path);
        }
        // the lock file huggingface_hub uses, so its processes and ours wait for each other
        let lock = self
            .cache
            .join(".locks")
            .join(format!("models--{}", p.repo.replace('/', "--")))
            .join(format!("{}.lock", p.sha256));
        let guard = crate::rt::oneshot::spawn(move || -> io::Result<File> {
            fs::create_dir_all(lock.parent().expect("lock path has a parent"))?;
            let f = OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(&lock)?;
            f.lock()?;
            Ok(f)
        })
        .await
        .ok_or_else(|| fail("the cache lock thread panicked".into()))?
        .map_err(|e| fail(format!("cannot lock the model cache: {e}")))?;
        if !path.exists() {
            self.fetch(p, transport).await?;
            publish(&self.repo_dir(p), p)
                .map_err(|e| fail(format!("cannot publish {}: {e}", p.file)))?;
        }
        drop(guard);
        Ok(path)
    }

    /// Download into `blobs/<sha>.incomplete`, resuming a partial file, and
    /// rename it to `blobs/<sha>` once size and hash match.
    async fn fetch(&self, p: &Pinned, transport: &dyn Transport) -> Result<()> {
        let blobs = self.repo_dir(p).join("blobs");
        let blob = blobs.join(p.sha256);
        if blob.exists() {
            return Ok(());
        }
        let part = blobs.join(format!("{}.incomplete", p.sha256));
        let io = |what: &str, e: io::Error| fail(format!("{what} {}: {e}", part.display()));
        fs::create_dir_all(&blobs).map_err(|e| io("cannot create", e))?;
        let mut sink = Sink::open(&part, p.size).map_err(|e| io("cannot open", e))?;
        if sink.offset < p.size {
            self.download(p, transport, &mut sink).await?;
        }
        let digest = sink.finish().map_err(|e| io("cannot write", e))?;
        let len = fs::metadata(&part).map_err(|e| io("cannot read", e))?.len();
        if len != p.size || digest != p.sha256 {
            let _ = fs::remove_file(&part);
            return Err(fail(format!(
                "{} from {} failed verification: {len} bytes with sha256 {digest}, expected {} bytes with sha256 {}",
                p.file, p.repo, p.size, p.sha256
            )));
        }
        fs::rename(&part, &blob).map_err(|e| io("cannot rename", e))
    }

    async fn download(&self, p: &Pinned, transport: &dyn Transport, sink: &mut Sink) -> Result<()> {
        let mut url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint, p.repo, p.revision, p.file
        );
        let mut auth = self.token.as_ref().map(|t| format!("Bearer {t}"));
        for _ in 0..=MAX_REDIRECTS {
            let mut headers = Vec::new();
            if sink.offset > 0 {
                headers.push(("range", format!("bytes={}-", sink.offset)));
            }
            if let Some(a) = &auth {
                headers.push(("authorization", a.clone()));
            }
            let req = HttpRequest {
                method: Method::Get,
                url: url.clone(),
                headers,
                body: Vec::new(),
            };
            let head = transport.download(req, sink).await?;
            match head.status {
                200 => {
                    return sink
                        .restarted()
                        .map_err(|e| fail(format!("cannot rewrite {}: {e}", p.file)));
                }
                206 => {
                    let start = head
                        .header("content-range")
                        .and_then(|v| v.strip_prefix("bytes "))
                        .and_then(|v| v.split('-').next())
                        .and_then(|v| v.trim().parse::<u64>().ok());
                    if start != Some(sink.resumed_at) {
                        return Err(fail(format!(
                            "unexpected content-range {:?} for {}",
                            head.header("content-range"),
                            p.file
                        )));
                    }
                    return Ok(());
                }
                301 | 302 | 303 | 307 | 308 => {
                    let next = head.header("location").ok_or_else(|| {
                        fail(format!("redirect without a location for {}", p.file))
                    })?;
                    let next = resolve(&url, next);
                    if origin(&next) != origin(&url) {
                        auth = None;
                    }
                    url = next;
                }
                s => {
                    return Err(fail(format!(
                        "downloading {} from {} answered {s}",
                        p.file, p.repo
                    )));
                }
            }
        }
        Err(fail(format!(
            "more than {MAX_REDIRECTS} redirects for {}",
            p.file
        )))
    }
}

fn origin(url: &str) -> &str {
    let rest = url.find("://").map_or(0, |i| i + 3);
    let end = url[rest..].find('/').map_or(url.len(), |i| rest + i);
    &url[..end]
}

fn resolve(base: &str, location: &str) -> String {
    if location.starts_with("http://") || location.starts_with("https://") {
        location.to_string()
    } else if location.starts_with('/') {
        format!("{}{location}", origin(base))
    } else {
        let dir = base.rfind('/').map_or(base, |i| &base[..=i]);
        format!("{dir}{location}")
    }
}

/// `snapshots/<rev>/<file>` pointing at `blobs/<sha>`: a relative symlink, or
/// a copy where symlinks are not available.
fn publish(repo_dir: &Path, p: &Pinned) -> io::Result<()> {
    let dir = repo_dir.join("snapshots").join(p.revision);
    fs::create_dir_all(&dir)?;
    let link = dir.join(p.file);
    let target = Path::new("../../blobs").join(p.sha256);
    #[cfg(unix)]
    if std::os::unix::fs::symlink(&target, &link).is_ok() {
        return Ok(());
    }
    fs::copy(repo_dir.join("blobs").join(p.sha256), &link).map(drop)
}

/// The partial file and two running hashes: one over the whole file
/// (for a resumed download) and one over this response's body alone (for a
/// server that ignored the range and sent everything again).
struct Sink {
    file: File,
    path: PathBuf,
    offset: u64,
    resumed_at: u64,
    whole: Sha256,
    body: Sha256,
}

impl Sink {
    fn open(path: &Path, size: u64) -> io::Result<Self> {
        let mut whole = Sha256::new();
        let mut offset = 0;
        if let Ok(mut f) = File::open(path) {
            let len = f.metadata()?.len();
            if len <= size {
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    let n = f.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    whole.update(&buf[..n]);
                    offset += n as u64;
                }
            }
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        if offset == 0 {
            file.set_len(0)?;
        }
        Ok(Self {
            file,
            path: path.to_path_buf(),
            offset,
            resumed_at: offset,
            whole,
            body: Sha256::new(),
        })
    }

    /// After a 200: the body starts at byte 0 of the file, so drop the
    /// prefix written before it.
    fn restarted(&mut self) -> io::Result<()> {
        if self.resumed_at == 0 {
            return Ok(());
        }
        let tmp = self.path.with_extension("restart");
        let mut src = File::open(&self.path)?;
        io::Seek::seek(&mut src, io::SeekFrom::Start(self.resumed_at))?;
        io::copy(&mut src, &mut File::create(&tmp)?)?;
        fs::rename(&tmp, &self.path)?;
        self.file = OpenOptions::new().append(true).open(&self.path)?;
        self.whole = std::mem::replace(&mut self.body, Sha256::new());
        self.resumed_at = 0;
        Ok(())
    }

    fn finish(mut self) -> io::Result<String> {
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(hex(&self.whole.finish()))
    }
}

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.file.write(buf)?;
        self.whole.update(&buf[..n]);
        self.body.update(&buf[..n]);
        self.offset += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests;
