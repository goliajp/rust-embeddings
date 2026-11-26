//! Local inference (feature `local`): four small BERT models on the CPU,
//! with an in-crate forward pass and WordPiece tokenizer. On macOS the
//! matrix products go to the Accelerate framework that ships with the OS.
//!
//! Each loaded model runs on its own worker thread, one batch at a time,
//! so inference never blocks the caller's async threads and needs no
//! runtime. Weights are downloaded once into the Hugging Face cache,
//! pinned to a commit and checked against their SHA-256.

mod attention;
mod engine;
mod forward;
mod gemm;
mod hub;
mod model;
mod ops;
#[cfg(test)]
mod parity;
mod pool;
#[cfg(test)]
mod reference;
mod safetensors;
mod softmax;
mod weights;
mod wordpiece;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};

pub use model::{ModelInfo, Pooling, models};

use self::engine::Engine;
use crate::error::{Error, Result};
use crate::input::{InputType, Inputs, Truncate};
use crate::request::Plan;
use crate::result::EmbedResult;
use crate::rt::oneshot;
use crate::transport::Transport;
use crate::usage::Usage;

/// The Hugging Face cache embedrs reads and writes: `HF_HUB_CACHE`, else
/// `HF_HOME/hub`, else `~/.cache/huggingface/hub`.
pub fn cache_dir() -> PathBuf {
    hub::cache_dir()
}

/// A failure inside model loading or inference; the model name is filled in later.
pub(crate) fn fail(message: String) -> Error {
    Error::Model {
        model: String::new(),
        message,
    }
}

fn named(e: Error, name: &str) -> Error {
    match e {
        Error::Model { model, message } if model.is_empty() => Error::Model {
            model: name.to_string(),
            message,
        },
        e => e,
    }
}

type Reply = Result<(Vec<f32>, u64)>;

struct Job {
    texts: Vec<String>,
    prefix: Option<&'static str>,
    refuse_long: bool,
    threads: Option<usize>,
    reply: oneshot::Sender<Reply>,
}

struct Worker {
    info: &'static ModelInfo,
    jobs: mpsc::Sender<Job>,
}

/// A loaded model and the thread that runs it. Cheap to clone; the thread
/// ends when the last clone is dropped.
#[derive(Clone)]
pub struct LocalModel {
    worker: Arc<Worker>,
    threads: Option<usize>,
}

impl std::fmt::Debug for LocalModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalModel")
            .field("model", &self.worker.info.name)
            .finish()
    }
}

// one loaded copy of each bundled model per process, shared by every client
static LOADED: LazyLock<Mutex<HashMap<&'static str, Weak<Worker>>>> =
    LazyLock::new(Default::default);

fn info(name: &str) -> Result<&'static ModelInfo> {
    model::find(name).ok_or_else(|| Error::UnknownModel(name.to_string()))
}

impl LocalModel {
    /// Load a bundled model, downloading its weights through `transport` on
    /// first use. Concurrent loads of one model, in this process or others,
    /// download it once.
    pub async fn load(name: &str, transport: Arc<dyn Transport>) -> Result<Self> {
        let info = info(name)?;
        if let Some(m) = Self::loaded(info) {
            return Ok(m);
        }
        let path = hub::Hub::from_env()
            .ensure(&info.pinned(), transport.as_ref())
            .await
            .map_err(|e| named(e, info.name))?;
        let model = oneshot::spawn(move || Self::start(info, &path))
            .await
            .ok_or_else(|| named(fail("the loading thread panicked".into()), info.name))??;
        let mut loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(w) = loaded.get(info.name).and_then(Weak::upgrade) {
            return Ok(Self {
                worker: w,
                threads: None,
            });
        }
        loaded.insert(info.name, Arc::downgrade(&model.worker));
        Ok(model)
    }

    /// Load a bundled model's architecture from a weights file you provide
    /// (`model.safetensors` from the model's repository). Nothing is
    /// downloaded or verified.
    pub fn from_files(name: &str, weights: &Path) -> Result<Self> {
        Self::start(info(name)?, weights)
    }

    fn loaded(info: &'static ModelInfo) -> Option<Self> {
        let loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
        let worker = loaded.get(info.name)?.upgrade()?;
        Some(Self {
            worker,
            threads: None,
        })
    }

    fn start(info: &'static ModelInfo, weights: &Path) -> Result<Self> {
        let engine = Engine::load(info, weights).map_err(|e| named(e, info.name))?;
        let (jobs, rx) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name(format!("embedrs-{}", info.name))
            .spawn(move || {
                for job in rx {
                    job.reply.send(engine.encode(
                        &job.texts,
                        job.prefix,
                        job.refuse_long,
                        job.threads,
                    ));
                }
            })
            .map_err(|e| {
                named(
                    fail(format!("cannot start the worker thread: {e}")),
                    info.name,
                )
            })?;
        Ok(Self {
            worker: Arc::new(Worker { info, jobs }),
            threads: None,
        })
    }

    /// The model this is.
    pub fn info(&self) -> &'static ModelInfo {
        self.worker.info
    }

    /// Use at most `threads` threads for each call made through this handle,
    /// the model's own thread included; 1 keeps every call on that thread.
    /// Without it, a call with more than a few dozen tokens uses the
    /// performance cores on macOS and every core the process may use
    /// elsewhere. On macOS the matrix products run in Accelerate, whose own
    /// threads follow `VECLIB_MAXIMUM_THREADS`. Clones made afterwards keep
    /// the setting; the model itself is shared and stays loaded once.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads.max(1));
        self
    }

    async fn run(
        &self,
        texts: Vec<String>,
        input_type: Option<InputType>,
        truncate: Truncate,
    ) -> Result<EmbedResult> {
        let info = self.worker.info;
        if texts.is_empty() {
            return Ok(empty(info));
        }
        let (reply, rx) = oneshot::channel();
        let job = Job {
            texts,
            prefix: info
                .query_prefix
                .filter(|_| input_type == Some(InputType::SearchQuery)),
            refuse_long: truncate == Truncate::Error,
            threads: self.threads,
            reply,
        };
        let stopped = || named(fail("the worker thread stopped".into()), info.name);
        self.worker.jobs.send(job).map_err(|_| stopped())?;
        let (data, tokens) = rx
            .await
            .ok_or_else(stopped)?
            .map_err(|e| named(e, info.name))?;
        Ok(EmbedResult::from_flat(
            data,
            info.dim,
            info.name,
            Usage::new(Some(tokens), None),
        ))
    }

    /// Embed and block until done, for code without an async runtime.
    pub fn embed_blocking(
        &self,
        texts: &[&str],
        input_type: Option<InputType>,
    ) -> Result<EmbedResult> {
        let texts = texts.iter().map(|t| t.to_string()).collect();
        crate::rt::block_on(self.run(texts, input_type, Truncate::Provider))
    }
}

fn empty(info: &ModelInfo) -> EmbedResult {
    EmbedResult::from_flat(Vec::new(), 0, info.name, Usage::new(Some(0), None))
}

/// The local model of a client: loaded on first use, or handed over.
#[derive(Clone)]
pub(crate) struct Handle {
    info: &'static ModelInfo,
    model: Arc<OnceLock<LocalModel>>,
}

impl Handle {
    pub(crate) fn name(&self) -> &'static str {
        self.info.name
    }

    /// The model `name` names: this handle's (loaded once), or another
    /// bundled one (shared through the process-wide table).
    async fn model(
        &self,
        name: &str,
        transport: Option<&Arc<dyn Transport>>,
    ) -> Result<LocalModel> {
        let mine = name == self.info.name;
        if mine && let Some(m) = self.model.get() {
            return Ok(m.clone());
        }
        let Some(t) = transport else {
            return Err(Error::unsupported(
                self.info.name,
                format!("model({name}) on a client built from a loaded model, without a transport"),
            ));
        };
        let m = LocalModel::load(name, t.clone()).await?;
        if mine {
            let _ = self.model.set(m.clone());
        }
        Ok(m)
    }
}

impl crate::Client {
    /// A client for a bundled local model: `"all-MiniLM-L6-v2"`,
    /// `"all-MiniLM-L12-v2"`, `"bge-small-en-v1.5"` or `"gte-small"`.
    /// The weights are downloaded on the first call (not under its timeout).
    ///
    /// ```rust,no_run
    /// # async fn run() -> embedrs::Result<()> {
    /// let client = embedrs::Client::local("all-MiniLM-L6-v2")?;
    /// let result = client.embed(["hello world"]).await?;
    /// assert_eq!(result.dim(), 384);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "reqwest")]
    pub fn local(model: &str) -> Result<Self> {
        let handle = Handle {
            info: info(model)?,
            model: Arc::new(OnceLock::new()),
        };
        Ok(Self::with_backend(
            crate::client::Backend::Local(handle),
            Some(crate::transport::ReqwestTransport::shared()),
        ))
    }

    /// A client for a model loaded with [`LocalModel::load`] or
    /// [`LocalModel::from_files`].
    pub fn from_local(model: LocalModel) -> Self {
        let handle = Handle {
            info: model.info(),
            model: Arc::new(OnceLock::from(model)),
        };
        Self::with_backend(crate::client::Backend::Local(handle), None)
    }
}

/// Load the model a call will use, outside its timeout.
pub(crate) async fn prepare(client: &crate::Client, h: &Handle, model: &str) -> Result<()> {
    info(model)?;
    h.model(model, client.transport.as_ref()).await.map(drop)
}

/// Embed `inputs` with the model `p.model` names.
pub(crate) async fn embed(
    client: &crate::Client,
    h: &Handle,
    p: &Plan,
    inputs: &Inputs<'_>,
) -> Result<EmbedResult> {
    let info = info(&p.model)?;
    if let Some(d) = p.dimensions
        && d as usize != info.dim
    {
        return Err(Error::unsupported(
            info.name,
            format!("dimensions({d}); its vectors have {} dimensions", info.dim),
        ));
    }
    if inputs.is_empty() {
        return Ok(empty(info));
    }
    let model = h.model(info.name, client.transport.as_ref()).await?;
    let texts = inputs.iter().map(str::to_string).collect();
    model.run(texts, p.input_type, p.truncate).await
}

#[cfg(test)]
mod tests;
