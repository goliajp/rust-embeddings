# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/embedrs?style=flat-square)](https://crates.io/crates/embedrs)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square)](https://www.rust-lang.org)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

Rust 统一 Embedding 方案 -- 6 家云端提供商 + 本地推理，一套接口搞定。默认值基于 8 模型基准测试数据。

## 设计理念

> "用就要好用" -- 要做就做到极致，每个默认值都有数据支撑。

- **`embedrs::local()?`** -- all-MiniLM-L6-v2（23MB，免费，无需 API key）
- **`embedrs::cloud(key)`** -- OpenAI text-embedding-3-small（区分度最佳，价格最低）
- 两者返回相同的 `EmbedResult` -- 写一次代码，一行切换后端

默认模型经 8 维度 8 模型对比测试选出，完整方法论见 [`examples/embedding_models/`](examples/embedding_models/)。

## 快速开始

```rust
// 云端 -- 一个 key 搞定
let client = embedrs::cloud("sk-...");
let result = client.embed(vec!["hello world".into()]).await?;
println!("dimensions: {}", result.embeddings[0].len());
```

```rust
// 本地 -- 零配置，免费，首次使用自动下载 23MB 模型
let client = embedrs::local()?;
let result = client.embed(vec!["hello world".into()]).await?;
```

## 安装

```toml
[dependencies]
embedrs = "0.3"

# 启用本地推理（首次使用下载约 23MB 模型）
embedrs = { version = "0.3", features = ["local"] }
```

## Feature Flags

| Feature | 默认 | 说明 |
|---|---|---|
| *(none)* | 是 | 核心功能，5 个云端提供商 |
| `local` | 否 | 本地推理，基于 candle（all-MiniLM-L6-v2，23MB） |
| `cost-tracking` | 否 | 通过 `tiktoken` 定价数据估算每次请求的费用 |
| `tracing` | 否 | 通过 `tracing` crate 输出结构化日志 |

```toml
[dependencies]
# 仅云端
embedrs = "0.3"

# 云端 + 本地推理
embedrs = { version = "0.3", features = ["local"] }

# 启用费用追踪
embedrs = { version = "0.3", features = ["cost-tracking"] }

# 启用 tracing
embedrs = { version = "0.3", features = ["local", "tracing"] }
```

## 基准测试结果

8 个评测维度，184 条测试文本。完整方法论与复现方式见 [`examples/embedding_models/`](examples/embedding_models/) —— 运行 `cargo run --example embedding_models --features local --release`。

| 指标 | MiniLM-L6 | MiniLM-L12 | BGE-small | GTE-small | OpenAI | Gemini | Cohere | Voyage |
|--------|:---------:|:----------:|:---------:|:---------:|:------:|:------:|:------:|:------:|
| **模型大小** | **23MB** | 133MB | 133MB | 67MB | 云端 | 云端 | 云端 | 云端 |
| **Spearman ρ** | 0.81 | 0.84 | 0.71 | 0.75 | 0.91 | **0.94** | 0.91 | 0.89 |
| **区分度** | 0.52 | 0.52 | 0.29 | 0.14 | **0.58** | 0.30 | 0.46 | 0.45 |
| **检索准确率** | **100%** | **100%** | 89% | **100%** | **100%** | 89% | **100%** | 89% |
| **英文 ρ** | 0.92 | **0.94** | 0.92 | 0.90 | 0.91 | 0.91 | 0.89 | 0.88 |
| **中文 ρ** | 0.65 | 0.74 | 0.45 | 0.40 | 0.88 | **0.99** | 0.93 | 0.89 |
| **日文 ρ** | 0.60 | 0.90 | 0.20 | 0.50 | 0.90 | **1.00** | **1.00** | 0.90 |
| **跨语言** | 0.25 | 0.26 | 0.66 | 0.81 | 0.71 | 0.84 | 0.68 | **0.85** |
| **鲁棒性** | 0.89 | 0.90 | 0.94 | **0.97** | 0.88 | 0.94 | 0.89 | 0.95 |
| **聚类分离度** | **8.73x** | 4.38x | 1.29x | 1.09x | 2.55x | 1.11x | 1.41x | 1.30x |
| **费用** | **$0** | **$0** | **$0** | **$0** | $0.02/1M | 免费额度 | $0.10/1M | $0.06/1M |

### 为什么本地选 MiniLM-L6

- 23MB -- 唯一适合嵌入应用的小模型（其余 67-133MB）
- 聚类分离度 8.73x 碾压第二名 4.38x，能看到数据真实结构
- 检索 100%，英文 ρ=0.92，超过大部分云端模型
- 12 层模型大了 3-6 倍，质量没有明显提升
- 已知短板：中日文较弱（ρ=0.60-0.65），跨语言差（0.25）

### 为什么云端选 OpenAI

- 区分度 0.58 最佳（不相似文本余弦 ≈ 0.09，最接近零）
- 检索 100%，MRR=1.0
- 多语言均衡：EN=0.91、ZH=0.88、JA=0.90，没有短板语言
- 价格最低 $0.02/1M tokens
- Gemini 相关性更高（0.94）但区分度差（0.30）且检索只有 89%
- Cohere 质量接近但价格贵 5 倍（$0.10/1M）

## 服务提供商

| 提供商 | 构造方法 | 默认模型 | 单次批量上限 |
|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 |
| Google Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 |
| Voyage AI | `Client::voyage(key)` | `voyage-3-large` | 128 |
| Jina AI | `Client::jina(key)` | `jina-embeddings-v3` | 2048 |
| 本地 | `Client::local(name)?` | `all-MiniLM-L6-v2` | 256 |

每个云端提供商都有 `*_compatible` 变体，用于代理或私有部署：

```rust
// OpenAI 兼容（Azure、代理等）
let client = Client::openai_compatible("sk-...", "https://your-proxy.com/v1");

// Cohere 兼容
let client = Client::cohere_compatible("key", "https://proxy.example.com/v2");

// Gemini 兼容
let client = Client::gemini_compatible("key", "https://proxy.example.com/v1beta");

// Voyage 兼容
let client = Client::voyage_compatible("key", "https://proxy.example.com/v1");

// Jina 兼容
let client = Client::jina_compatible("key", "https://proxy.example.com/v1");
```

## 批量 Embedding

自动按提供商限制分块，并发处理大规模文本：

```rust
let client = embedrs::cloud("sk-...");

let texts: Vec<String> = (0..5000).map(|i| format!("document {i}")).collect();

let result = client.embed_batch(texts)
    .concurrency(5)       // 最大并发请求数（默认 5）
    .chunk_size(512)       // 每次请求的文本数（默认按提供商上限）
    .model("text-embedding-3-large")
    .await?;

println!("total embeddings: {}", result.embeddings.len());
println!("total tokens: {}", result.usage.total_tokens);
```

## 相似度计算

```rust
use embedrs::{cosine_similarity, dot_product, euclidean_distance};

let a = vec![1.0, 0.0, 0.0];
let b = vec![0.0, 1.0, 0.0];

let cos = cosine_similarity(&a, &b);    // 0.0（正交）
let dot = dot_product(&a, &b);          // 0.0
let dist = euclidean_distance(&a, &b);  // 1.414...
```

## 输入类型

部分提供商支持指定输入类型以优化 Embedding 质量：

```rust
use embedrs::InputType;

// 索引文档
let result = client.embed(docs)
    .input_type(InputType::SearchDocument)
    .await?;

// 搜索查询
let result = client.embed(queries)
    .input_type(InputType::SearchQuery)
    .await?;
```

可选类型：`SearchDocument`、`SearchQuery`、`Classification`、`Clustering`。

## 输出维度

指定输出向量维度（适用于支持该参数的提供商）：

```rust
let result = client.embed(vec!["hello".into()])
    .model("text-embedding-3-large")
    .dimensions(256)
    .await?;

assert_eq!(result.embeddings[0].len(), 256);
```

## 重试与超时

```rust
use std::time::Duration;
use embedrs::BackoffConfig;

let client = Client::openai("sk-...")
    .with_retry_backoff(BackoffConfig::default())  // 500ms 基础延迟, 30s 上限, 3 次重试
    .with_timeout(Duration::from_secs(120));        // 整体超时（默认 60s）

// 按请求覆盖
let result = client.embed(vec!["hello".into()])
    .retry_backoff(BackoffConfig {
        base_delay: Duration::from_millis(200),
        max_delay: Duration::from_secs(10),
        jitter: true,
        max_http_retries: 5,
    })
    .timeout(Duration::from_secs(30))
    .await?;
```

未配置退避时，HTTP 429/503 错误立即失败。

## 客户端默认值

设置一次默认值，按需覆盖：

```rust
let client = Client::openai("sk-...")
    .with_model("text-embedding-3-large")
    .with_dimensions(256)
    .with_input_type(InputType::SearchDocument)
    .with_retry_backoff(BackoffConfig::default())
    .with_timeout(Duration::from_secs(120));

// 所有请求使用上述默认值
let a = client.embed(vec!["doc 1".into()]).await?;
let b = client.embed(vec!["doc 2".into()]).await?;

// 针对特定请求覆盖
let c = client.embed(vec!["query".into()])
    .model("text-embedding-3-small")
    .input_type(InputType::SearchQuery)
    .await?;
```

## 提供商回退

链式配置回退提供商，主提供商不可用时自动切换：

```rust
let client = embedrs::Client::openai("sk-...")
    .with_fallback(embedrs::Client::cohere("cohere-key"));
// OpenAI 失败时自动尝试 Cohere
let result = client.embed(vec!["hello".into()]).await?;
```

支持多个回退，按顺序尝试：

```rust
let client = embedrs::Client::openai("sk-...")
    .with_fallback(embedrs::Client::cohere("cohere-key"))
    .with_fallback(embedrs::Client::voyage("voyage-key"));
```

## 费用追踪

启用 `cost-tracking` feature 可获取每次请求的费用估算：

```toml
embedrs = { version = "0.3", features = ["cost-tracking"] }
```

```rust
let result = client.embed(vec!["hello".into()]).await?;
if let Some(cost) = result.usage.cost {
    println!("estimated cost: ${cost:.6}");
}
```

费用估算基于 `tiktoken` 定价数据。无定价信息的模型返回 `None`。

## 错误处理

所有可能失败的操作返回 `embedrs::Result<T>`。通过匹配 `Error` 变体进行细粒度控制：

```rust,no_run
use embedrs::Error;

# async fn run(client: &embedrs::Client) {
match client.embed(vec!["hello".into()]).await {
    Ok(result) => println!("got {} embeddings", result.embeddings.len()),
    Err(Error::Api { status: 429, .. }) => eprintln!("rate limited"),
    Err(Error::Api { status, message }) => eprintln!("API error {status}: {message}"),
    Err(Error::Timeout(duration)) => eprintln!("timed out after {duration:?}"),
    Err(Error::Http(e)) => eprintln!("network error: {e}"),
    Err(e) => eprintln!("other error: {e}"),
}
# }
```

## 为什么选 embedrs？

| 对比维度 | embedrs | fastembed-rs | 裸写 reqwest |
|---|---|---|---|
| 云端提供商 | 内置 5 家（OpenAI、Cohere、Gemini、Voyage、Jina） | 无 | 每家手动对接 |
| 本地推理 | 基于 candle，默认 23MB 模型 | ONNX Runtime，多种模型 | 不适用 |
| 统一接口 | 云端和本地返回相同 `EmbedResult` | 仅本地 | 不适用 |
| 批量自动分块 | 按提供商限制自动分块 + 并发 | 手动 | 手动 |
| 提供商回退 | 内置 `.with_fallback()` 链式调用 | 不适用 | 手动 |
| 数据驱动默认值 | 8 维度 8 模型基准测试（[`examples/embedding_models/`](examples/embedding_models/)） | 无公开基准测试 | 不适用 |
| 退避与超时 | 内置指数退避，自动处理 429/503 | 不适用 | 手动 |

**fastembed-rs** 如果只需本地 ONNX Runtime 推理且不需要云端提供商，是不错的选择。**embedrs** 面向需要云端 + 本地统一 API 的场景，提供开箱即用的默认值和生产级特性（回退、退避等）。

## 生态系统

GOLIA 出品的一系列独立 AI 基础设施 crate，每个各自独立仓：

| Crate | 仓库 | 说明 |
|---|---|---|
| [tiktoken](https://crates.io/crates/tiktoken) | [rust-tiktoken](https://github.com/goliajp/rust-tiktoken) | 高性能 BPE 分词器，支持所有主流 LLM |
| [instructors](https://crates.io/crates/instructors) | [rust-instructor](https://github.com/goliajp/rust-instructor) | 类型安全的 LLM 结构化输出提取 |
| **embedrs** | [rust-embeddings](https://github.com/goliajp/rust-embeddings) | 统一 Embedding —— 云端 + 本地（本 crate） |
| [chunkedrs](https://crates.io/crates/chunkedrs) | [rust-chunker](https://github.com/goliajp/rust-chunker) | AI 原生文本分块，用于 Embedding 和检索 |

## 许可证

[MIT](LICENSE)
