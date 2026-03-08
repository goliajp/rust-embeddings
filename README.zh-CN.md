# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](LICENSE)

[English](README.md) | **简体中文** | [日本語](README.ja.md)

统一的云端 Embedding API 客户端，支持 OpenAI、Cohere、Gemini、Voyage 和 Jina 五大提供商。

通过一致的接口调用不同提供商的 Embedding API，内置自动分批、指数退避重试和超时控制。

## 特性

- 统一接口 -- 一套 API 适配 5 个提供商，切换无需改代码
- 自动分批 -- 根据提供商限制自动分块，并发处理大规模文本
- 指数退避 -- HTTP 429/503 错误自动重试，支持抖动
- 超时控制 -- 可配置的请求级和客户端级超时
- 输入类型 -- 支持 `SearchDocument`、`SearchQuery`、`Classification`、`Clustering`
- 维度控制 -- 可指定输出向量维度（提供商支持时）
- 相似度计算 -- 内置余弦相似度、点积、欧氏距离
- 兼容 API -- 每个提供商均支持自定义 `base_url`，兼容代理和私有部署
- 可选 tracing -- 通过 feature flag 集成结构化日志

## 安装

```toml
[dependencies]
embedrs = "0.1"
```

## 快速开始

```rust
use embedrs::prelude::*;

let client = Client::openai("sk-...");
let result = client.embed(vec!["hello world".into()]).await?;
println!("dimensions: {}", result.embeddings[0].len());
println!("tokens: {}", result.usage.total_tokens);
```

## 服务提供商

| 提供商 | 构造方法 | 默认模型 | 单次批量上限 |
|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 |
| Google Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 |
| Voyage | `Client::voyage(key)` | `voyage-3-large` | 128 |
| Jina | `Client::jina(key)` | `jina-embeddings-v3` | 2048 |

每个提供商均有对应的 `_compatible` 变体，用于自定义 API 地址：

```rust
// OpenAI
let client = Client::openai("sk-...");

// Cohere
let client = Client::cohere("co-...");

// Google Gemini
let client = Client::gemini("AIza...");

// Voyage
let client = Client::voyage("pa-...");

// Jina
let client = Client::jina("jina_...");

// OpenAI 兼容 API（自定义地址）
let client = Client::openai_compatible("sk-...", "https://api.deepseek.com/v1");

// Cohere 兼容代理
let client = Client::cohere_compatible("co-...", "https://proxy.example.com/v2");

// Gemini 兼容代理
let client = Client::gemini_compatible("AIza...", "https://proxy.example.com/v1beta");

// Voyage 兼容代理
let client = Client::voyage_compatible("pa-...", "https://proxy.example.com/v1");

// Jina 兼容代理
let client = Client::jina_compatible("jina_...", "https://proxy.example.com/v1");
```

## 批量 Embedding

通过 `embed_batch` 自动分块并发处理大规模文本，分块大小自动适配提供商限制：

```rust
let client = Client::openai("sk-...");

let texts: Vec<String> = (0..5000).map(|i| format!("text {i}")).collect();
let result = client.embed_batch(texts)
    .concurrency(5)       // 最大并发请求数（默认 5）
    .chunk_size(100)      // 覆盖默认分块大小（可选）
    .await?;

println!("total embeddings: {}", result.embeddings.len());
println!("total tokens: {}", result.usage.total_tokens);
```

## 输入类型

部分提供商支持指定输入类型以优化 Embedding 质量：

```rust
use embedrs::InputType;

// 索引文档时
let result = client.embed(docs)
    .input_type(InputType::SearchDocument)
    .await?;

// 搜索查询时
let result = client.embed(queries)
    .input_type(InputType::SearchQuery)
    .await?;
```

支持四种输入类型：`SearchDocument`、`SearchQuery`、`Classification`、`Clustering`。

## 输出维度

指定输出向量维度（适用于支持该参数的提供商）：

```rust
let client = Client::openai("sk-...")
    .with_dimensions(256);

// 或按请求覆盖
let result = client.embed(vec!["hello".into()])
    .dimensions(512)
    .await?;
```

## 相似度计算

内置三种向量相似度/距离函数：

```rust
use embedrs::{cosine_similarity, dot_product, euclidean_distance};

let a = vec![1.0, 0.0, 0.0];
let b = vec![0.0, 1.0, 0.0];

let cos = cosine_similarity(&a, &b);   // 0.0（正交）
let dot = dot_product(&a, &b);         // 0.0
let dist = euclidean_distance(&a, &b); // 1.414...
```

## 重试与超时

启用 HTTP 429/503 错误的指数退避重试，并设置整体请求超时：

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

未配置退避时，HTTP 429/503 错误将立即失败（默认行为不变）。

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
let r1 = client.embed(vec!["text 1".into()]).await?;
let r2 = client.embed(vec!["text 2".into()]).await?;

// 针对特定请求覆盖默认值
let r3 = client.embed(vec!["query".into()])
    .model("text-embedding-3-small")
    .input_type(InputType::SearchQuery)
    .await?;
```

## Feature Flags

| Feature | 默认启用 | 说明 |
|---|---|---|
| (default) | -- | 核心 Embedding 功能，无额外依赖 |
| `tracing` | 否 | 集成 `tracing` crate，输出结构化日志（provider、model、token 数等） |

启用 tracing：

```toml
[dependencies]
embedrs = { version = "0.1", features = ["tracing"] }
```

## 许可证

[MIT](LICENSE)
