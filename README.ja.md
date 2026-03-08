# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](LICENSE)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

Rust 向けの統一クラウド Embedding API クライアント。OpenAI、Cohere、Gemini、Voyage、Jina の 5 つのプロバイダーを単一のインターフェースでサポート。自動バッチ分割、指数バックオフリトライ、タイムアウト機能付き。

## 特徴

- 5 プロバイダー統一 API (OpenAI, Cohere, Gemini, Voyage, Jina)
- 各プロバイダーの互換 API もサポート (OpenAI 互換、Cohere 互換など)
- 自動バッチ分割と並行処理 (`embed_batch`)
- 入力タイプ指定 (検索ドキュメント、検索クエリ、分類、クラスタリング)
- 出力次元数の指定
- 類似度関数 (コサイン類似度、ドット積、ユークリッド距離)
- HTTP 429/503 に対する指数バックオフリトライ
- リクエスト全体のタイムアウト
- オプションの `tracing` 統合
- `IntoFuture` によるエルゴノミックな `.await` 構文

## インストール

```toml
[dependencies]
embedrs = "0.1"
```

または:

```bash
cargo add embedrs
```

## クイックスタート

```rust
use embedrs::prelude::*;

let client = Client::openai("sk-...");

let result = client.embed(vec!["hello world".into()]).await?;
println!("dimensions: {}", result.embeddings[0].len());
println!("tokens: {}", result.usage.total_tokens);
```

## プロバイダー

| プロバイダー | コンストラクタ | デフォルトモデル | バッチ上限 |
|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 |
| Google Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 |
| Voyage AI | `Client::voyage(key)` | `voyage-3-large` | 128 |
| Jina AI | `Client::jina(key)` | `jina-embeddings-v3` | 2048 |

各プロバイダーには `*_compatible` コンストラクタもあり、カスタムエンドポイントを指定できます。

```rust
// OpenAI
let client = Client::openai("sk-...");

// Cohere
let client = Client::cohere("co-...");

// Google Gemini
let client = Client::gemini("AIza...");

// Voyage AI
let client = Client::voyage("pa-...");

// Jina AI
let client = Client::jina("jina_...");

// OpenAI 互換 API (カスタムエンドポイント)
let client = Client::openai_compatible("sk-...", "https://api.example.com/v1");

// Cohere 互換 API
let client = Client::cohere_compatible("key", "https://proxy.example.com/v2");

// Gemini 互換 API
let client = Client::gemini_compatible("key", "https://proxy.example.com/v1beta");

// Voyage 互換 API
let client = Client::voyage_compatible("key", "https://proxy.example.com/v1");

// Jina 互換 API
let client = Client::jina_compatible("key", "https://proxy.example.com/v1");
```

## バッチ Embedding

大量のテキストをプロバイダーの上限に合わせて自動分割し、並行処理します:

```rust
let client = Client::openai("sk-...");

let texts: Vec<String> = (0..5000).map(|i| format!("text {i}")).collect();

let result = client.embed_batch(texts)
    .concurrency(5)           // 最大同時リクエスト数
    .chunk_size(100)          // 1 リクエストあたりのテキスト数 (省略時はプロバイダーのデフォルト)
    .model("text-embedding-3-large")
    .await?;

println!("total embeddings: {}", result.embeddings.len());
println!("total tokens: {}", result.usage.total_tokens);
```

## 入力タイプ

一部のプロバイダー (Cohere, Gemini, Voyage, Jina) は入力タイプの指定をサポートしています:

```rust
use embedrs::InputType;

// 検索インデックス用ドキュメントの Embedding
let result = client.embed(docs)
    .input_type(InputType::SearchDocument)
    .await?;

// 検索クエリの Embedding
let result = client.embed(queries)
    .input_type(InputType::SearchQuery)
    .await?;
```

利用可能な入力タイプ: `SearchDocument`、`SearchQuery`、`Classification`、`Clustering`

## 出力次元数

出力ベクトルの次元数を指定 (OpenAI, Gemini, Jina で対応):

```rust
let client = Client::openai("sk-...")
    .with_dimensions(256);   // クライアントレベルのデフォルト

// またはリクエストごとに指定
let result = client.embed(vec!["hello".into()])
    .dimensions(512)
    .await?;
```

## 類似度関数

Embedding ベクトル間の類似度を計算するユーティリティ関数:

```rust
use embedrs::{cosine_similarity, dot_product, euclidean_distance};

let a = vec![1.0, 0.0, 0.0];
let b = vec![0.0, 1.0, 0.0];

let cos = cosine_similarity(&a, &b);   // 0.0 (直交)
let dot = dot_product(&a, &b);          // 0.0
let dist = euclidean_distance(&a, &b);  // 1.414...
```

## バックオフとタイムアウト

HTTP 429/503 エラーに対する指数バックオフと、リクエスト全体のタイムアウトを設定:

```rust
use std::time::Duration;
use embedrs::BackoffConfig;

let client = Client::openai("sk-...")
    .with_retry_backoff(BackoffConfig::default())  // 500ms ベース, 30s 上限, 3 回リトライ
    .with_timeout(Duration::from_secs(120));        // 全体タイムアウト (デフォルト: 60s)

// リクエストごとに上書き
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

バックオフ未設定時、HTTP 429/503 エラーは即座に失敗します。

## クライアントデフォルト

デフォルトを一度設定し、リクエストごとに上書き可能:

```rust
let client = Client::openai("sk-...")
    .with_model("text-embedding-3-large")
    .with_dimensions(256)
    .with_input_type(InputType::SearchDocument)
    .with_retry_backoff(BackoffConfig::default())
    .with_timeout(Duration::from_secs(120));

// すべてのリクエストで上記のデフォルトが使用されます
let result = client.embed(vec!["hello".into()]).await?;

// 特定のリクエストでのみ上書き
let result = client.embed(vec!["query".into()])
    .model("text-embedding-3-small")
    .input_type(InputType::SearchQuery)
    .await?;
```

## Feature フラグ

| Feature | デフォルト | 説明 |
|---|---|---|
| (なし) | 有効 | コア機能: 5 プロバイダー、バッチ処理、類似度関数 |
| `tracing` | 無効 | `tracing` クレートによる構造化ログ出力 |

```toml
# tracing を有効にする場合
[dependencies]
embedrs = { version = "0.1", features = ["tracing"] }
```

## ライセンス

[MIT](LICENSE)
