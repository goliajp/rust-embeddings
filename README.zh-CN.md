# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](LICENSE)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

Rust 统一 Embedding 方案 -- 云端 API + 本地推理，一套接口，开箱即用。

## 设计理念

> "好用就好用" -- 每个默认值都有数据支撑，拿来就能用。

- **`embedrs::local()?`** -- all-MiniLM-L6-v2（23MB，免费，无需 API key）
- **`embedrs::cloud(key)`** -- OpenAI text-embedding-3-small（区分度最佳，价格最低）
- 两者返回相同的 `EmbedResult` -- 写一次代码，一行切换后端

默认模型经 8 维度 8 模型对比测试选出，完整方法论见 [benchrs](https://github.com/goliajp/airs/tree/develop/crates/benchrs)。

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
embedrs = "0.2"

# 启用本地推理（首次使用下载约 23MB 模型）
embedrs = { version = "0.2", features = ["local"] }
```

## 基准测试结果

8 个评测维度，184 条测试文本。完整方法论与复现方式见 [benchrs](https://github.com/goliajp/airs/tree/develop/crates/benchrs)。

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
embedrs = "0.2"

# 云端 + 本地推理
embedrs = { version = "0.2", features = ["local"] }

# 启用费用追踪
embedrs = { version = "0.2", features = ["cost-tracking"] }

# 启用 tracing
embedrs = { version = "0.2", features = ["local", "tracing"] }
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
embedrs = { version = "0.2", features = ["cost-tracking"] }
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

## 许可证

[MIT](LICENSE)
