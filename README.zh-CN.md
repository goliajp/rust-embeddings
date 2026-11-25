# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-embeddings/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-embeddings/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](#许可证)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

> **从 0.6.1 升级**：API 没变。本地模型现在按各自的上限截断文本（默认模型从 128 个 token 改为 256 个），所以较长文本的向量会变；备用客户端改用它自己的模型和维度；后端做不到的选项现在会报错，不再被悄悄忽略。详见 [CHANGELOG](CHANGELOG.md#upgrading-from-061)。

embedrs 是一个异步的 Rust 文本向量化（embedding）客户端。六家云端 API（OpenAI、Cohere、Gemini、Voyage、Jina、Mistral）和四个在本进程 CPU 上运行的小型 BERT 模型返回同一个 `EmbedResult`，换后端只要改一行。在 Apple M4 Pro 上，本地的 `all-MiniLM-L6-v2` 处理单条文本比 fastembed（ONNX Runtime）快 1.1～1.7 倍，处理 256 条一批的文本快 1.3～12 倍，两者输出的向量相同。

## 功能

- **六家云端服务**共用一个 `Client`，每家都有 `*_compatible` 构造函数，可以接代理或兼容同一 API 的服务
- **本地推理**（feature `local`）：`all-MiniLM-L6-v2`、`all-MiniLM-L12-v2`、`bge-small-en-v1.5`、`gte-small`，基于纯 Rust 的 [candle](https://github.com/huggingface/candle)；在 macOS 上矩阵乘法交给系统自带的 Accelerate 框架。文本先按长度排序、分组补齐，再按各模型自己的上限截断
- **批量**：`embed_batch` 把任意数量的文本拆成服务商能接受的请求，并发发送
- **重试和备用**：遇到 408、429、500、502、503、504、529 和连接错误时按指数退避重试，遵守 `Retry-After`；主服务失败后依次改用备用服务
- **紧凑传输**：OpenAI、Voyage 和 Cohere 以 base64 返回向量，由 embedrs 自己解码。256 个向量的响应比十进制文本小 2.3 倍，解析快 8.6 倍
- **费用估算**（feature `cost-tracking`）：按 [tiktoken](https://crates.io/crates/tiktoken) 的价格表估算每次请求的美元费用

## 性能

本地推理，模型 `all-MiniLM-L6-v2`，机器是 Apple M4 Pro Mac mini（14 核，macOS），rustc 1.100，release 构建并开启 LTO。对比对象：[fastembed](https://crates.io/crates/fastembed) 7.8.0（ONNX Runtime），同一个模型，同样在 256 个 token 处截断；[model2vec-rs](https://crates.io/crates/model2vec-rs) 0.5.1，模型 `potion-base-8M`。在 32 条长度不一的文本上，embedrs 和 fastembed 输出向量的余弦相似度最低为 1.000000。测试文本长 8、40、180 个英文单词（平均 13、54、232 个 token），另有一组 4 到 200 个词混合的文本。单条文本取 200 次调用的中位数（单线程时取 30 次），批量取 256 条文本 3 轮的中位数。测试期间的系统负载在 1.7 到 7.2 之间。“0.6.1”列是上一个版本的代码，依赖版本与本版相同。

| 默认线程数 | 0.6.1 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| 一条文本，8 个词 | 18.7 ms | **3.6 ms** | 4.3 ms | 0.0003 ms |
| 一条文本，40 个词 | 18.7 ms | **6.0 ms** | 8.5 ms | 0.0012 ms |
| 一条文本，180 个词 | 18.8 ms ¹ | **15.9 ms** | 24.6 ms | 0.0051 ms |
| 256 条，8 个词 | 155 条/s | **1,872 条/s** | 672 条/s | 300 万条/s |
| 256 条，40 个词 | 152 条/s | **776 条/s** | 184 条/s | 210 万条/s |
| 256 条，180 个词 | 145 条/s ¹ | **159 条/s** | 22 条/s | 100 万条/s |
| 256 条，长度混合 | 152 条/s ¹ | **249 条/s** | 21 条/s | 80 万条/s |

| 单线程 | 0.6.1 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| 一条文本，8 个词 | 39.4 ms | **2.0 ms** | 3.5 ms | 0.0004 ms |
| 一条文本，40 个词 | 39.8 ms | **5.6 ms** | 7.3 ms | 0.0012 ms |
| 一条文本，180 个词 | 40.3 ms ¹ | **23.7 ms** | 26.1 ms | 0.0051 ms |
| 256 条，8 个词 | 27 条/s | **798 条/s** | 581 条/s | 200 万条/s |
| 256 条，40 个词 | 27 条/s | **204 条/s** | 156 条/s | 57 万条/s |
| 256 条，180 个词 | 27 条/s ¹ | **40 条/s** | 21 条/s | 14 万条/s |
| 256 条，长度混合 | 27 条/s ¹ | **59 条/s** | 12 条/s | 24 万条/s |

¹ 0.6.1 把每条文本都截断并补齐到 128 个 token，所以在 180 个词的文本上，它的计算量只有另外两者的一半左右。

model2vec 在每一行都更快，因为它是另一类模型：静态向量模型只是在表里查出每个 token 的向量再求平均，没有 Transformer 层。这里列出它只是作为参照，在质量上它不能替代 Transformer 模型。相比 fastembed 的提速来自三处：每组文本只补齐到组内最长的一条；在 macOS 上用 Accelerate 计算矩阵乘法；前向计算把 query、key、value 三个投影合并成一次矩阵乘法。本版没有在 Linux 上测量。

云端调用：本进程内的模拟服务器通过回环地址返回 1536 维、OpenAI 格式的响应。256 个向量时，十进制响应体（4.88 MB）上 `embed()` 用时 6.85 ms，base64 响应体（2.11 MB）上用时 1.43 ms，embedrs 现在向 OpenAI、Voyage 和 Cohere 请求的就是 base64；2048 个向量时分别是 59.2 ms 和 12.1 ms。客户端其余部分每次调用约增加 0.05 ms，而一次网络往返通常要几十毫秒。

## 安装

```toml
[dependencies]
embedrs = "0.6"

# 本地推理
embedrs = { version = "0.6", features = ["local"] }
```

| feature | 默认开启 | 增加的内容 |
|---|---|---|
| *（无）* | 是 | 客户端和全部六家云端服务 |
| `local` | 否 | 基于 candle 的本地推理（分词器已内置，权重首次使用时下载）|
| `cost-tracking` | 否 | `Usage::cost`，数据来自 tiktoken 的价格表（不编译任何分词词表）|
| `tracing` | 否 | 通过 `tracing` crate 输出 span 和事件 |

## 快速开始

```rust,no_run
async fn run() -> embedrs::Result<()> {
    // 云端：OpenAI text-embedding-3-small
    let client = embedrs::cloud("sk-...");
    let result = client.embed(vec!["hello world".into()]).await?;
    println!("{} dimensions", result.embeddings[0].len()); // 1536

    // 本地：all-MiniLM-L6-v2，不需要 API key
    let client = embedrs::local();
    let result = client.embed(vec!["hello world".into()]).await?;
    println!("{} dimensions", result.embeddings[0].len()); // 384
    Ok(())
}
```

第一次本地调用会把权重下载到 Hugging Face 缓存目录（`HF_HOME`，默认 `~/.cache/huggingface`），之后的调用和其他进程直接复用。

## 云端服务

| 服务商 | 构造函数 | 默认模型 | 每次请求的文本数 | `dimensions` |
|---|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 | `text-embedding-3-*` |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 | `embed-v4.0`、`embed-v5.0-*`；v3 模型报错 |
| Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 | 128 到 3072 |
| Voyage | `Client::voyage(key)` | `voyage-4-large` | 1000（`embed_batch` 每次发 128）| 4、3.5、3-large、code-3 系列；更早的模型报错 |
| Jina | `Client::jina(key)` | `jina-embeddings-v3` | 2048 | 支持 |
| Mistral | `Client::mistral(key)` | `mistral-embed` | 512 | `codestral-embed`；`mistral-embed` 报错 |
| 本地 | `Client::local(name)?` | `all-MiniLM-L6-v2` | 256 | 只能是模型本身的维度 |

模型名是任意字符串，新模型不用等新版本就能用：`client.embed(texts).model("text-embedding-3-large")`。下表是 `cost-tracking` 采用的每百万输入 token 价格（tiktoken 5.4）：

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

## 本地模型

| 模型 | 层数 | 权重大小 | 最多读取 | 池化方式 |
|---|---|---|---|---|
| `all-MiniLM-L6-v2`（默认）| 6 | 91 MB | 256 个 token | 平均 |
| `all-MiniLM-L12-v2` | 12 | 133 MB | 128 个 token | 平均 |
| `bge-small-en-v1.5` | 12 | 133 MB | 512 个 token | CLS |
| `gte-small` | 12 | 67 MB | 512 个 token | 平均 |

四个模型都输出 384 维、经过 L2 归一化的向量。输入类型为 `InputType::SearchQuery` 时，`bge-small-en-v1.5` 会在文本前加上它训练时用的查询指令。本地客户端的 `.model()` 可以换成另一个本地模型，`.dimensions()` 只接受 384，其他值都会报错。超长文本会被截断，`usage.total_tokens` 统计的是模型实际读入的 token 数。

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

    println!("{} vectors, {} tokens", result.embeddings.len(), result.usage.total_tokens);
    Ok(())
}
```

## 输入类型和维度

```rust,no_run
use embedrs::InputType;

async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let docs = client
        .embed(vec!["Rust is a systems language".into()])
        .input_type(InputType::SearchDocument)
        .await?;
    let query = client
        .embed(vec!["what is rust".into()])
        .input_type(InputType::SearchQuery)
        .dimensions(256)
        .await?;
    println!("{} / {}", docs.embeddings[0].len(), query.embeddings[0].len());
    Ok(())
}
```

`InputType` 有 `SearchDocument`、`SearchQuery`、`Classification`、`Clustering` 四种，发给各服务商时换成各自的写法（Cohere 是 `search_query`，Gemini 是 `RETRIEVAL_QUERY`，Jina 是 `retrieval.query`，依此类推）；`gemini-embedding-2` 则以文本前缀的形式传入。OpenAI 和 Mistral 没有这个参数，会忽略它。

## 重试、超时和备用服务

```rust,no_run
use std::time::Duration;
use embedrs::{BackoffConfig, Client};

async fn run() -> embedrs::Result<()> {
    let client = Client::openai("sk-...")
        .with_retry_backoff(BackoffConfig::default()) // 起始 500 ms，上限 30 s，重试 3 次
        .with_timeout(Duration::from_secs(120)) // 整个请求的时限，含重试（默认 60 s）
        .with_fallback(Client::cohere("cohere-key"))
        .with_fallback(Client::voyage("voyage-key").with_model("voyage-4-lite"));

    let result = client.embed(vec!["hello".into()]).await?;
    println!("answered by {}", result.model);
    Ok(())
}
```

不调用 `with_retry_backoff` 时，请求失败不会重试。主服务失败后，按顺序尝试各个备用服务，每个都用自己的模型和维度；全部失败时返回主服务的错误。不同模型的向量不能互相比较，所以备用服务只用来保证可用，不要用它往同一个索引里写入不同模型的向量。

## 费用估算

```rust,no_run
async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let result = client.embed(vec!["hello".into()]).await?;
    if let Some(cost) = result.usage.cost {
        println!("about ${cost:.8}");
    }
    Ok(())
}
```

模型不在价格表里，或服务商没有返回 token 数时，`cost` 为 `None`。

## 错误处理

```rust,no_run
use embedrs::Error;

async fn run(client: &embedrs::Client) {
    match client.embed(vec!["hello".into()]).await {
        Ok(result) => println!("{} vectors", result.embeddings.len()),
        Err(Error::Api { status: 429, .. }) => eprintln!("rate limited"),
        Err(Error::Api { status, message, .. }) => eprintln!("API error {status}: {message}"),
        Err(Error::Timeout(after)) => eprintln!("timed out after {after:?}"),
        Err(Error::Http(e)) => eprintln!("network error: {e}"),
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
