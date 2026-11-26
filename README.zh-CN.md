# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-embeddings/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-embeddings/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

> **从 0.6 升级**：0.7 的 API 有改动：`result.embeddings[i]` 改成 `result[i]`，错误类型和退避配置换了名字，备用服务只在临时故障时启用。[迁移表](CHANGELOG.md#upgrading-from-06)列出了每一处的改法，都只要改一行。发出的请求与 0.6.2 相同；本地模型的向量与 0.6.2 的差别每个分量不超过 3.3e-7。

embedrs 是一个异步的 Rust 文本向量化（embedding）客户端。六家云端 API（OpenAI、Cohere、Gemini、Voyage、Jina、Mistral）和四个在本进程 CPU 上运行的小型 BERT 模型返回同一个 `EmbedResult`，换后端只要改一行。在 Apple M4 Pro 上用默认线程数，本地的 `all-MiniLM-L6-v2` 处理单条文本比 fastembed（ONNX Runtime）快 3.1～4.4 倍，处理 256 条一批的文本快 8.9～15 倍，两者输出的向量相差不到 3.3e-7。

## 功能

- **六家云端服务**共用一个 `Client`，每家都有 `*_compatible` 构造函数，可以接代理或兼容同一 API 的服务
- **本地推理**（feature `local`）：`all-MiniLM-L6-v2`、`all-MiniLM-L12-v2`、`bge-small-en-v1.5`、`gte-small`，前向计算由 embedrs 自己实现，不依赖其他 crate。在 macOS 上矩阵乘法交给系统自带的 Accelerate 框架，在其他系统上用 embedrs 自带的 AVX2/FMA（x86_64）或 NEON（aarch64）内核。文本不做补齐，按各模型自己的上限截断
- **批量**：`embed_batch` 把任意数量的文本按条数和 token 数拆成服务商能接受的请求，并发发送。一次调用的所有向量放在同一块内存里
- **重试和备用**：遇到 408、429、500、502、503、504、529 和连接错误时按指数退避重试，遵守 `Retry-After`（秒数或日期）；重试用完后依次改用备用服务
- **紧凑传输**：OpenAI、Voyage 和 Cohere 以 base64 返回向量，由 embedrs 自己解码。256 个向量的响应比十进制文本小 2.3 倍，解析快 7.6 倍
- **费用估算**（feature `cost-tracking`）：按 [tiktoken](https://crates.io/crates/tiktoken) 的价格表估算每次请求的美元费用

- **依赖少**：核心部分不依赖任何第三方库，JSON、base64、WordPiece 和 SHA-256 都是自己实现的。默认构建只加上 reqwest（rustls + ring）和 tokio 的定时器，在 macOS 上共 84 个 crate，0.6.2 是 101 个。开启 `local` 不增加 crate。设置 `default-features = false` 时只有 embedrs 一个 crate，HTTP 客户端由你提供

## 性能

以下数字都是 2026-11-26 在一台 Apple M4 Pro Mac mini（14 核，macOS 27.2.1）上测的，rustc 1.100，release 构建并开启 LTO，测量期间占着这台机器的基准锁，没有别的任务同时跑。测云端调用时每 15 秒采一次 1 分钟平均负载，最高 2.2；本地推理的负载写在对应的小节里。

### 本地推理

模型 `all-MiniLM-L6-v2`，在同一个进程里对比 0.6.2（基于 candle）、[fastembed](https://crates.io/crates/fastembed) 7.8.0（ONNX Runtime，同一个模型，同样在 256 个 token 处截断）和 [model2vec-rs](https://crates.io/crates/model2vec-rs) 0.5.1（模型 `potion-base-8M`）。输出的向量与 0.6.2 的差别每个分量不超过 3.3e-7（38 条文本，四个模型都比过，余弦相似度与 1 相差不到 2e-12）；0.6.2 与 fastembed 在 32 条长度不一的文本上，余弦相似度最低为 1.000000。单条文本长 8、40、180 个英文单词（13、56、227 个 token），批量是 256 条 8、40 或 180 个词的文本，另有一组 4 到 200 个词混合的文本。单条文本取 200 次调用的中位数（单线程时取 30 次），批量取 5 轮的中位数。每 10 秒采一次 1 分钟平均负载，最高 4.4（包含测试程序自己的线程）。「单线程」指 embedrs 设置 `EMBEDRS_THREADS=1 VECLIB_MAXIMUM_THREADS=1`，0.6.2 设置 `RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`，fastembed 只用一个算子内线程。

| 默认线程数 | 0.6.2 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| 单条，8 词 | 3.13 ms | **1.34 ms** | 4.21 ms | 0.0003 ms |
| 单条，40 词 | 5.86 ms | **2.47 ms** | 8.17 ms | 0.0012 ms |
| 单条，180 词 | 15.7 ms | **5.48 ms** | 24.1 ms | 0.0051 ms |
| 256 条，8 词 | 1,882 条/秒 | **6,948 条/秒** | 660 条/秒 | 280 万条/秒 |
| 256 条，40 词 | 763 条/秒 | **1,664 条/秒** | 188 条/秒 | 190 万条/秒 |
| 256 条，180 词 | 158 条/秒 | **344 条/秒** | 26 条/秒 | 110 万条/秒 |
| 256 条，长短混合 | 209 条/秒 | **633 条/秒** | 42 条/秒 | 120 万条/秒 |

| 单线程 | 0.6.2 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| 单条，8 词 | 2.19 ms | **1.47 ms** | 4.02 ms | 0.0004 ms |
| 单条，40 词 | 5.55 ms | **2.56 ms** | 7.45 ms | 0.0012 ms |
| 单条，180 词 | 23.5 ms | **9.88 ms** | 26.4 ms | 0.0051 ms |
| 256 条，8 词 | 806 条/秒 | **2,486 条/秒** | 579 条/秒 | 200 万条/秒 |
| 256 条，40 词 | 205 条/秒 | **520 条/秒** | 154 条/秒 | 63 万条/秒 |
| 256 条，180 词 | 40 条/秒 | **103 条/秒** | 26 条/秒 | 15 万条/秒 |
| 256 条，长短混合 | 62 条/秒 | **198 条/秒** | 26 条/秒 | 25 万条/秒 |

加载一个模型、再计算那 256 条长短混合文本，整个进程的内存峰值（最大常驻内存）：`all-MiniLM-L6-v2` 为 178 MB（0.6.2 为 1,715 MB），三个 12 层的模型为 219～234 MB（0.6.2 为 680～1,883 MB）。F32 的权重文件直接映射进内存读取，文本不做补齐，每次最多约 4,096 个 token 一起算。

0.6.2 和 embedrs 的矩阵乘法都在 Accelerate 里算，时间差在前向计算的其他部分：embedrs 的 GELU 用向量化的 erf（0.6.2 对每个值调用一次 `erff`），残差加法和层归一化合在一遍里做，注意力按每条文本、每个头分别计算，不补齐也不用掩码，只有调用的计算量够大时才把这些计算分给性能核。model2vec 在每一行都更快，因为它是另一类模型：静态向量模型只是在表里查出每个 token 的向量再求平均，没有 Transformer 层。这里列出它只是作为参照，在质量上它不能替代 Transformer 模型。

### 云端调用

本进程内的模拟服务器通过回环地址返回 1536 维、OpenAI 格式的响应。表里是整个 `embed()` 调用的时间（256 个向量取 50 次的中位数，2048 个取 15 次），其中很大一部分花在传输字节上。对比对象是 [async-openai](https://crates.io/crates/async-openai) 0.43.0 和 [rig-core](https://crates.io/crates/rig-core) 0.47.1（它返回 `f64`），收到的响应完全相同。“—”表示没有测。

| 响应 | 0.6.2 | **embedrs** | async-openai | rig-core |
|---|---|---|---|---|
| 256 个向量，最短十进制（4.9 MB）| 6.64 ms | **3.87 ms** | — | — |
| 256 个向量，17 位十进制（8.3 MB）| 9.48 ms | **6.40 ms** | 9.69 ms | 22.0 ms |
| 256 个向量，base64（2.1 MB）| 1.43 ms | **0.80 ms** | 1.11 ms | — |
| 2048 个向量，最短十进制（39 MB）| 59.6 ms | **32.7 ms** | — | — |
| 2048 个向量，17 位十进制（67 MB）| 80.4 ms | **53.5 ms** | 80.2 ms | 172 ms |
| 2048 个向量，base64（17 MB）| 12.1 ms | **7.1 ms** | 9.4 ms | — |

只看解析，十进制响应的用时是 serde_json 的 0.49～0.62 倍（4.9 MB 用 3.0 ms，serde_json 用 6.1 ms）。数字解析的舍入是正确的：所有有限的 `f32` 各用四种写法打印（共 171 亿个字符串），再加 10 亿个随机字符串，结果都和 `str::parse::<f32>` 逐位相同。只有一个向量时，`embed()` 约用 0.05 ms，和 0.6.2、async-openai 一样，而一次网络往返通常要几十毫秒。

## 安装

```toml
[dependencies]
embedrs = "0.7"

# 本地推理
embedrs = { version = "0.7", features = ["local"] }
```

| feature | 默认 | 作用 |
|---|---|---|
| `reqwest` | 开 | HTTP 传输层，以及用到它的构造函数（`embedrs::cloud`、`Client::openai` 等）|
| `local` | 关 | 在本机 CPU 上推理，不引入依赖（分词器编译在库里，权重首次使用时下载）|
| `cost-tracking` | 关 | `Usage::cost`，取自 tiktoken 的价格表（不会编译进任何分词器词表）|
| `token-count` | 关 | 精确计算 OpenAI 的 token 数，用于拆分批量请求和拒绝超长文本（会编译进 `cl100k_base` 词表）|
| `tracing` | 关 | 通过 `tracing` crate 输出 span 和事件 |

macOS 上依赖树里的 crate 数（含 embedrs 自己，括号里是 0.6.2）：默认 84（101），默认加 `local` 也是 84（255），`default-features = false` 为 1，只开 `local` 不开 `reqwest` 为 1。x86_64 Linux 上默认构建是 82 个。

## 快速开始

```rust,no_run
async fn run() -> embedrs::Result<()> {
    // 云端：OpenAI text-embedding-3-small
    let client = embedrs::cloud("sk-...");
    let result = client.embed(["hello world"]).await?;
    println!("{} dimensions", result.dim()); // 1536
    println!("{:?}", &result[0][..4]); // 第一个向量，类型是 &[f32]

    // 本地：all-MiniLM-L6-v2，不需要 API key
    let client = embedrs::local();
    let result = client.embed(["hello world"]).await?;
    println!("{} dimensions", result.dim()); // 384
    Ok(())
}
```

`embed` 接受 `Vec<String>`、`&[String]`、`&[&str]`、`&str` 数组或单个 `&str`。`result[i]` 是第 `i` 条文本的向量；`result.iter()` 逐个遍历，`result.as_flat()` 是整块数据，`result.to_vecs()` 复制成 `Vec<Vec<f32>>`。

第一次本地调用会把权重下载到 Hugging Face 缓存目录（`HF_HOME`，默认 `~/.cache/huggingface`），之后的调用和其他进程直接复用。下载的是固定 commit 的文件，中断后可以续传，下完会校验 SHA-256，下载时间不计入调用的超时。

## 云端服务

| 服务商 | 构造函数 | 默认模型 | 每次请求的文本数 | `dimensions` |
|---|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 | `text-embedding-3-*` |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 | `embed-v4.0`、`embed-v5.0-*`；v3 模型报错 |
| Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 | 128 到 3072 |
| Voyage | `Client::voyage(key)` | `voyage-4-large` | 1000 | 4、3.5、3-large、code-3 系列；更早的模型报错 |
| Jina | `Client::jina(key)` | `jina-embeddings-v3` | 2048 | 支持 |
| Mistral | `Client::mistral(key)` | `mistral-embed` | 512 | `codestral-embed`；`mistral-embed` 报错 |
| 本地 | `Client::local(name)?` | `all-MiniLM-L6-v2` | 256 | 只能是模型本身的维度 |

模型名是任意字符串，新模型不用等新版本就能用：`client.embed(texts).model("text-embedding-3-large")`。下表是 `cost-tracking` 采用的每百万输入 token 价格（tiktoken 5.5）：

| 模型 | 美元／百万 token | 模型 | 美元／百万 token |
|---|---|---|---|
| `text-embedding-3-small` | 0.02 | `voyage-4-large` | 0.12 |
| `text-embedding-3-large` | 0.13 | `voyage-4` | 0.06 |
| `gemini-embedding-001` | 0.15 | `voyage-4-lite` | 0.02 |
| `gemini-embedding-2` | 0.20 | `voyage-3-large` | 0.18 |
| `mistral-embed` | 0.10 | `codestral-embed` | 0.15 |

价格表里没有 Cohere 和 Jina 的模型，它们的 `cost` 是 `None`。

2026 年 11 月可用的模型名：OpenAI `text-embedding-3-small`、`text-embedding-3-large`；Cohere `embed-v4.0`、`embed-v5.0-pro`、`embed-v5.0-fast`；Gemini `gemini-embedding-001`、`gemini-embedding-2`；Voyage `voyage-4-large`、`voyage-4`、`voyage-4-lite`、`voyage-code-4`、`voyage-3.5`、`voyage-3.5-lite`、`voyage-3-large`、`voyage-code-3`；Jina `jina-embeddings-v3`、`jina-embeddings-v5-text-small`、`jina-embeddings-v5-text-nano`；Mistral `mistral-embed`、`codestral-embed`。

每家云端服务都有一个接收 base URL 的 `*_compatible` 构造函数：

```rust,no_run
use embedrs::Client;

fn clients() {
    let openai_like = Client::openai_compatible("key", "https://api.deepseek.com/v1");
    let cohere_proxy = Client::cohere_compatible("key", "https://proxy.example.com/v2");
    let gemini_proxy = Client::gemini_compatible("key", "https://proxy.example.com/v1beta");
    let voyage_proxy = Client::voyage_compatible("key", "https://proxy.example.com/v1");
    let jina_proxy = Client::jina_compatible("key", "https://proxy.example.com/v1");
    let mistral_proxy = Client::mistral_compatible("key", "https://proxy.example.com/v1");
}
```

`openai_compatible` 请求十进制向量，所有兼容 OpenAI 的服务都支持这种格式；只有官方地址才请求 base64。

Azure OpenAI 以及需要其他请求头的网关，用 `Endpoint` 配置：

```rust,no_run
use std::sync::Arc;
use embedrs::{Client, Endpoint, transport::ReqwestTransport};

fn azure() -> Client {
    // the v1 api: no api-version parameter, the deployment name goes in `model`
    let endpoint =
        Endpoint::openai_compatible("", "https://my-resource.openai.azure.com/openai/v1")
            .bearer(false)
            .header("api-key", "azure-key");
    Client::new(endpoint, Arc::new(ReqwestTransport::new())).with_model("my-embedding-deployment")
}
```

## 本地模型

| 模型 | 层数 | 权重大小 | 最多读取 | 池化方式 |
|---|---|---|---|---|
| `all-MiniLM-L6-v2`（默认）| 6 | 91 MB | 256 个 token | 平均 |
| `all-MiniLM-L12-v2` | 12 | 133 MB | 128 个 token | 平均 |
| `bge-small-en-v1.5` | 12 | 133 MB | 512 个 token | CLS |
| `gte-small` | 12 | 67 MB | 512 个 token | 平均 |

四个模型都输出 384 维、经过 L2 归一化的向量。输入类型为 `InputType::SearchQuery` 时，`bge-small-en-v1.5` 会在文本前加上它训练时用的查询指令。本地客户端的 `.model()` 可以换成另一个本地模型，`.dimensions()` 只接受 384，其他值都会报错。超长文本会被截断，`usage.total_tokens` 统计的是模型实际读入的 token 数。

每个加载的模型有自己的线程，所以本地推理不需要异步运行时：`LocalModel::from_files` 加载你已有的权重文件（F32、F16 或 BF16 的 safetensors），`embed_blocking` 计算完再返回。一次调用的 token 超过几十个时，计算期间还会用上工作线程：macOS 上是性能核的个数，其他系统上是进程能用的全部核。线程数可以用 `EMBEDRS_THREADS` 设置；`EMBEDRS_THREADS=1` 让整个调用都在模型自己的线程上算（macOS 上 Accelerate 仍按 `VECLIB_MAXIMUM_THREADS` 决定线程数）。

## 批量向量化

```rust,no_run
async fn run() -> embedrs::Result<()> {
    let client = embedrs::cloud("sk-...");
    let texts: Vec<String> = (0..5000).map(|i| format!("document {i}")).collect();

    let result = client
        .embed_batch(texts)
        .concurrency(5) // 同时进行的请求数（默认 5）
        .chunk_size(512) // 每次请求的文本数（默认取服务商上限）
        .model("text-embedding-3-large")
        .await?;

    println!("{} vectors, {:?} tokens", result.len(), result.usage.total_tokens);
    Ok(())
}
```

一个请求装到服务商的条数上限就发出；服务商写明了 token 上限的，到上限也会发出（OpenAI 30 万，Voyage 12 万）。每条文本的 token 数按 UTF-8 字节数计，对 OpenAI 的分词器来说不会少算；开启 `token-count` 后 OpenAI 的计数是精确的，每个请求能装得更满。

## 输入类型和维度

```rust,no_run
use embedrs::InputType;

async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let docs = client
        .embed(["Rust is a systems language"])
        .input_type(InputType::SearchDocument)
        .await?;
    let query = client
        .embed(["what is rust"])
        .input_type(InputType::SearchQuery)
        .dimensions(256)
        .await?;
    println!("{} / {}", docs.dim(), query.dim());
    Ok(())
}
```

`InputType` 有 `SearchDocument`、`SearchQuery`、`Classification`、`Clustering` 四种，发给各服务商时换成各自的写法（Cohere 是 `search_query`，Gemini 是 `RETRIEVAL_QUERY`，Jina 是 `retrieval.query`，依此类推）；`gemini-embedding-2` 则以文本前缀的形式传入。OpenAI 和 Mistral 没有这个参数，会忽略它。

## 超长文本

```rust,no_run
use embedrs::Truncate;

async fn run(client: &embedrs::Client, texts: &[String]) -> embedrs::Result<()> {
    let result = client.embed(texts).truncate(Truncate::Error).await?;
    println!("{}", result.len());
    Ok(())
}
```

默认的 `Truncate::Provider` 由服务商决定：Voyage、Cohere、Gemini 和本地模型会截掉末尾，OpenAI 和 Jina 会返回错误。`Truncate::End` 要求截断，`Truncate::Error` 要求拒绝；服务商没有这种开关时返回 `Error::Unsupported`。本地模型在计算前就以 `Error::InputTooLong` 拒绝，OpenAI 模型在开启 `token-count` 时也是如此。

## 重试、超时和备用服务

```rust,no_run
use std::time::Duration;
use embedrs::{Backoff, Client};

async fn run() -> embedrs::Result<()> {
    let client = Client::openai("sk-...")
        .with_retry(Backoff::new()) // 起始 500 ms，上限 30 s，重试 3 次
        .with_timeout(Duration::from_secs(120)) // 整个请求的时限，含重试（默认 60 s）
        .with_fallback(Client::cohere("cohere-key"))
        .with_fallback(Client::voyage("voyage-key").with_model("voyage-4-lite"));

    let result = client.embed(["hello"]).await?;
    println!("answered by {}", result.model);
    Ok(())
}
```

不调用 `with_retry` 时，请求失败不会重试。主服务重试后仍然是临时故障（上面列出的状态码，或者连不上），就按顺序尝试各个备用服务，每个都用自己的模型和维度；全部失败时返回主服务的错误。400、401、403、422 会立即返回，所以密钥写错不会被备用服务掩盖。不同模型的向量不能互相比较，所以备用服务只用来保证可用，不要用它往同一个索引里写入不同模型的向量。

## 使用自己的 HTTP 客户端

```rust,no_run
use embedrs::{Client, transport::HttpResponse};

fn run(client: &Client) -> embedrs::Result<()> {
    let prepared = client.prepare(["hello world"])?;
    let request = prepared.request(); // method, url, headers, body
    println!("POST {} ({} bytes)", request.url, request.body.len());

    let body = br#"{"data":[{"index":0,"embedding":[0.6,0.8]}]}"#; // 你的 HTTP 客户端收到的响应
    let result = prepared.finish(HttpResponse::new(200, vec![], body.to_vec()))?;
    assert_eq!(&result[0], &[0.6, 0.8]);
    Ok(())
}
```

`prepare` 和 `finish` 不做任何 I/O，所以什么 HTTP 客户端都能用，包括阻塞式的；[`examples/blocking.rs`](examples/blocking.rs) 直接用标准库的 `TcpStream`，没有任何依赖。如果还要用重试、备用服务、批量和超时，就实现 `Transport`（必须实现的方法只有两个：发送请求和等待一段时间），传给 `Client::new`。embedrs 不会启动任务，也不限定用哪个运行时。要自己配置 reqwest（代理、连接池、请求头），先用 `ReqwestTransport::client_builder()` 构建客户端（它会设置好 TLS provider），再写 `client.with_transport(Arc::new(ReqwestTransport::from(reqwest_client)))`。

想让代码支持任意向量化后端，可以接收 `&dyn embedrs::Embed`，这个 trait 只有一个方法，`Client` 已经实现了它。

## 费用估算

```rust,no_run
async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let result = client.embed(["hello"]).await?;
    if let Some(cost) = result.usage.cost {
        println!("about ${cost:.8}");
    }
    Ok(())
}
```

模型不在价格表里、服务商没有返回 token 数，或者没有开启 `cost-tracking` 时，`cost` 为 `None`。

## 错误处理

```rust,no_run
use embedrs::Error;

async fn run(client: &embedrs::Client) {
    match client.embed(["hello"]).await {
        Ok(result) => println!("{} vectors", result.len()),
        Err(Error::Api { status: 429, .. }) => eprintln!("rate limited"),
        Err(Error::Api { status, message, .. }) => eprintln!("API error {status}: {message}"),
        Err(Error::Timeout(after)) => eprintln!("timed out after {after:?}"),
        Err(Error::Transport(e)) => eprintln!("network error: {e}"),
        Err(e) => eprintln!("{e}"),
    }
}
```

## 相似度函数

```rust
use embedrs::{cosine_similarity, dot_product, euclidean_distance};

let a = [1.0, 0.0, 0.0];
let b = [0.0, 1.0, 0.0];
assert_eq!(cosine_similarity(&a, &b), 0.0);
assert_eq!(dot_product(&a, &b), 0.0);
assert!((euclidean_distance(&a, &b) - 2f32.sqrt()).abs() < 1e-6);
```

## 怎么选模型

[`examples/embedding_models/`](examples/embedding_models/) 用 184 条英文、中文、日文文本给八个模型打分（2026-03-08 录制；运行 `cargo run --release --features local --example embedding_models` 会从保存的向量重新算出全部结果）：

| | MiniLM-L6 | MiniLM-L12 | BGE-small | GTE-small | OpenAI 3-small | Gemini 001 | Cohere v4 | Voyage 3-large |
|---|---|---|---|---|---|---|---|---|
| Spearman ρ，全部 32 对 | 0.81 | 0.84 | 0.71 | 0.75 | 0.91 | 0.94 | 0.91 | 0.89 |
| ρ，英文 | 0.92 | 0.94 | 0.92 | 0.90 | 0.91 | 0.91 | 0.89 | 0.88 |
| ρ，中文 | 0.65 | 0.74 | 0.45 | 0.40 | 0.88 | 0.99 | 0.93 | 0.89 |
| ρ，日文 | 0.60 | 0.90 | 0.20 | 0.50 | 0.90 | 1.00 | 1.00 | 0.90 |
| 检索 top-1（9 个查询）| 9 | 9 | 8 | 9 | 9 | 8 | 9 | 8 |
| 跨语言余弦相似度 | 0.25 | 0.26 | 0.66 | 0.81 | 0.71 | 0.84 | 0.68 | 0.85 |
| 相似对与不相似对的余弦差 | 0.52 | 0.52 | 0.29 | 0.14 | 0.58 | 0.30 | 0.46 | 0.45 |

四个本地模型都是英文模型：中文、日文和跨语言的得分明显低于云端模型。语料不大（打分的文本对只有 32 对），相差零点零几的结果在误差范围内。

## 许可证

可任选 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 许可证。
