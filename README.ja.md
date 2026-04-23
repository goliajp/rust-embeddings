# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](LICENSE)
[![Downloads](https://img.shields.io/crates/d/embedrs?style=flat-square)](https://crates.io/crates/embedrs)
[![MSRV](https://img.shields.io/badge/MSRV-1.94-blue?style=flat-square)](https://www.rust-lang.org)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

Rust 向け統一 Embedding ソリューション -- 6 つのクラウドプロバイダー + ローカル推論を単一インターフェースで。8 モデルベンチマークデータに基づくデフォルト選定。

## 設計思想

> 作るからには最高のものを -- すべてのデフォルトはデータに裏付けられています。

- **`embedrs::local()?`** -- all-MiniLM-L6-v2（23MB、無料、API キー不要）
- **`embedrs::cloud(key)`** -- OpenAI text-embedding-3-small（識別性能最高、最安値）
- どちらも同じ `EmbedResult` を返す -- コードは一度書くだけ、バックエンドは一行で切り替え

デフォルトは 8 次元 8 モデルのベンチマークで選定。詳細は [`examples/embedding_models/`](examples/embedding_models/) を参照。

## クイックスタート

```rust
// クラウド -- API キーひとつで完了
let client = embedrs::cloud("sk-...");
let result = client.embed(vec!["hello world".into()]).await?;
println!("dimensions: {}", result.embeddings[0].len());
```

```rust
// ローカル -- 設定不要、無料、初回使用時に 23MB モデルを自動ダウンロード
let client = embedrs::local()?;
let result = client.embed(vec!["hello world".into()]).await?;
```

## インストール

```toml
[dependencies]
embedrs = "0.3"

# ローカル推論を有効化（初回使用時に約 23MB のモデルをダウンロード）
embedrs = { version = "0.3", features = ["local"] }
```

## Feature フラグ

| Feature | デフォルト | 説明 |
|---|---|---|
| *(なし)* | 有効 | コア機能、5 クラウドプロバイダー |
| `local` | 無効 | candle によるローカル推論（all-MiniLM-L6-v2、23MB） |
| `cost-tracking` | 無効 | `tiktoken` の価格データによるリクエストごとのコスト推定 |
| `tracing` | 無効 | `tracing` クレートによる構造化ログ |

```toml
[dependencies]
# クラウドのみ
embedrs = "0.3"

# クラウド + ローカル推論
embedrs = { version = "0.3", features = ["local"] }

# コスト追跡付き
embedrs = { version = "0.3", features = ["cost-tracking"] }

# tracing 付き
embedrs = { version = "0.3", features = ["local", "tracing"] }
```

## ベンチマーク結果

8 つの評価次元、184 テキスト。詳細な方法論と再現手順は [`examples/embedding_models/`](examples/embedding_models/) にあります — `cargo run --example embedding_models --features local --release` で実行。

| 指標 | MiniLM-L6 | MiniLM-L12 | BGE-small | GTE-small | OpenAI | Gemini | Cohere | Voyage |
|--------|:---------:|:----------:|:---------:|:---------:|:------:|:------:|:------:|:------:|
| **サイズ** | **23MB** | 133MB | 133MB | 67MB | クラウド | クラウド | クラウド | クラウド |
| **Spearman ρ** | 0.81 | 0.84 | 0.71 | 0.75 | 0.91 | **0.94** | 0.91 | 0.89 |
| **識別性** | 0.52 | 0.52 | 0.29 | 0.14 | **0.58** | 0.30 | 0.46 | 0.45 |
| **検索精度** | **100%** | **100%** | 89% | **100%** | **100%** | 89% | **100%** | 89% |
| **英語 ρ** | 0.92 | **0.94** | 0.92 | 0.90 | 0.91 | 0.91 | 0.89 | 0.88 |
| **中国語 ρ** | 0.65 | 0.74 | 0.45 | 0.40 | 0.88 | **0.99** | 0.93 | 0.89 |
| **日本語 ρ** | 0.60 | 0.90 | 0.20 | 0.50 | 0.90 | **1.00** | **1.00** | 0.90 |
| **言語横断** | 0.25 | 0.26 | 0.66 | 0.81 | 0.71 | 0.84 | 0.68 | **0.85** |
| **頑健性** | 0.89 | 0.90 | 0.94 | **0.97** | 0.88 | 0.94 | 0.89 | 0.95 |
| **クラスタ分離度** | **8.73x** | 4.38x | 1.29x | 1.09x | 2.55x | 1.11x | 1.41x | 1.30x |
| **コスト** | **$0** | **$0** | **$0** | **$0** | $0.02/1M | 無料枠 | $0.10/1M | $0.06/1M |

### ローカルに MiniLM-L6 を選んだ理由

- 23MB -- アプリ組込みに適した唯一のサイズ（他モデルは 67-133MB）
- クラスタ分離度 8.73x で圧倒的（2位は 4.38x）、データの構造を正確に捉える
- 検索精度 100%、英語 ρ=0.92 で大半のクラウドモデルを上回る
- 12 層モデルは 3-6 倍大きいが、品質は大差なし
- 既知の弱点：中国語・日本語が弱い（ρ=0.60-0.65）、言語横断も低い（0.25）

### クラウドに OpenAI を選んだ理由

- 識別性 0.58 で最高（非類似テキストの平均コサイン = 0.09、ゼロに最も近い）
- 検索精度 100%、MRR=1.0
- 多言語バランスが良い：EN=0.91、ZH=0.88、JA=0.90、弱い言語なし
- 最安値 $0.02/1M トークン
- Gemini は ρ が高い（0.94）が識別性が低く（0.30）、検索 89% で取りこぼしあり
- Cohere は品質同等だが 5 倍高い（$0.10/1M）

## プロバイダー

| プロバイダー | コンストラクタ | デフォルトモデル | バッチ上限 |
|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 |
| Google Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 |
| Voyage AI | `Client::voyage(key)` | `voyage-3-large` | 128 |
| Jina AI | `Client::jina(key)` | `jina-embeddings-v3` | 2048 |
| ローカル | `Client::local(name)?` | `all-MiniLM-L6-v2` | 256 |

各クラウドプロバイダーには `*_compatible` コンストラクタがあり、プロキシや互換 API を指定可能:

```rust
// OpenAI 互換（Azure、プロキシなど）
let client = Client::openai_compatible("sk-...", "https://your-proxy.com/v1");

// Cohere 互換
let client = Client::cohere_compatible("key", "https://proxy.example.com/v2");

// Gemini 互換
let client = Client::gemini_compatible("key", "https://proxy.example.com/v1beta");

// Voyage 互換
let client = Client::voyage_compatible("key", "https://proxy.example.com/v1");

// Jina 互換
let client = Client::jina_compatible("key", "https://proxy.example.com/v1");
```

## バッチ Embedding

大量テキストをプロバイダーの上限に合わせて自動分割し、並行処理:

```rust
let client = embedrs::cloud("sk-...");

let texts: Vec<String> = (0..5000).map(|i| format!("document {i}")).collect();

let result = client.embed_batch(texts)
    .concurrency(5)       // 最大同時リクエスト数（デフォルト: 5）
    .chunk_size(512)       // 1リクエストあたりのテキスト数（デフォルト: プロバイダー上限）
    .model("text-embedding-3-large")
    .await?;

println!("total embeddings: {}", result.embeddings.len());
println!("total tokens: {}", result.usage.total_tokens);
```

## 類似度関数

```rust
use embedrs::{cosine_similarity, dot_product, euclidean_distance};

let a = vec![1.0, 0.0, 0.0];
let b = vec![0.0, 1.0, 0.0];

let cos = cosine_similarity(&a, &b);    // 0.0（直交）
let dot = dot_product(&a, &b);          // 0.0
let dist = euclidean_distance(&a, &b);  // 1.414...
```

## 入力タイプ

一部のプロバイダーは入力タイプの指定による最適化をサポート:

```rust
use embedrs::InputType;

// ドキュメントのインデックス用
let result = client.embed(docs)
    .input_type(InputType::SearchDocument)
    .await?;

// 検索クエリ用
let result = client.embed(queries)
    .input_type(InputType::SearchQuery)
    .await?;
```

利用可能な入力タイプ: `SearchDocument`、`SearchQuery`、`Classification`、`Clustering`

## 出力次元数

対応プロバイダーで出力ベクトルの次元数を指定:

```rust
let result = client.embed(vec!["hello".into()])
    .model("text-embedding-3-large")
    .dimensions(256)
    .await?;

assert_eq!(result.embeddings[0].len(), 256);
```

## バックオフとタイムアウト

```rust
use std::time::Duration;
use embedrs::BackoffConfig;

let client = Client::openai("sk-...")
    .with_retry_backoff(BackoffConfig::default())  // 500ms ベース, 30s 上限, 3 回リトライ
    .with_timeout(Duration::from_secs(120));        // 全体タイムアウト（デフォルト: 60s）

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

// すべてのリクエストで上記デフォルトを使用
let a = client.embed(vec!["doc 1".into()]).await?;
let b = client.embed(vec!["doc 2".into()]).await?;

// 特定リクエストのみ上書き
let c = client.embed(vec!["query".into()])
    .model("text-embedding-3-small")
    .input_type(InputType::SearchQuery)
    .await?;
```

## プロバイダーフォールバック

フォールバックプロバイダーをチェーンして、プライマリが利用不可の場合に自動切り替え:

```rust
let client = embedrs::Client::openai("sk-...")
    .with_fallback(embedrs::Client::cohere("cohere-key"));
// OpenAI が失敗した場合、自動的に Cohere を試行
let result = client.embed(vec!["hello".into()]).await?;
```

複数のフォールバックを順番に試行:

```rust
let client = embedrs::Client::openai("sk-...")
    .with_fallback(embedrs::Client::cohere("cohere-key"))
    .with_fallback(embedrs::Client::voyage("voyage-key"));
```

## コスト追跡

`cost-tracking` フィーチャーを有効にすると、リクエストごとの推定コストを取得可能:

```toml
embedrs = { version = "0.3", features = ["cost-tracking"] }
```

```rust
let result = client.embed(vec!["hello".into()]).await?;
if let Some(cost) = result.usage.cost {
    println!("estimated cost: ${cost:.6}");
}
```

コスト推定は `tiktoken` の価格データに基づきます。価格情報のないモデルは `None` を返します。

## エラーハンドリング

すべての失敗可能な操作は `embedrs::Result<T>` を返します。`Error` のバリアントをマッチして細かく制御:

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

## なぜ embedrs？

| 比較項目 | embedrs | fastembed-rs | 素の reqwest |
|---|---|---|---|
| クラウドプロバイダー | 5 社内蔵（OpenAI、Cohere、Gemini、Voyage、Jina） | なし | プロバイダーごとに手動実装 |
| ローカル推論 | candle ベース、デフォルト 23MB モデル | ONNX Runtime、複数モデル | 非対応 |
| 統一インターフェース | クラウドとローカルで同じ `EmbedResult` | ローカルのみ | 非対応 |
| バッチ自動分割 | プロバイダー上限に応じた自動分割 + 並行処理 | 手動 | 手動 |
| プロバイダーフォールバック | 内蔵 `.with_fallback()` チェーン | 非対応 | 手動 |
| データ駆動デフォルト | 8 次元 8 モデルベンチマーク（[`examples/embedding_models/`](examples/embedding_models/)） | 公開ベンチマークなし | 非対応 |
| バックオフとタイムアウト | 指数バックオフ内蔵、429/503 自動処理 | 非対応 | 手動 |

**fastembed-rs** はローカル ONNX Runtime 推論のみで十分な場合には優れた選択肢です。**embedrs** はクラウド + ローカルを単一 API で統合し、データに裏付けられたデフォルトとフォールバック・バックオフなどの本番向け機能を提供します。

## エコシステム

GOLIA の独立した AI インフラ crate ファミリーの一員、各々が独自のリポジトリ:

| クレート | リポジトリ | 説明 |
|---|---|---|
| [tiktoken](https://crates.io/crates/tiktoken) | [rust-tiktoken](https://github.com/goliajp/rust-tiktoken) | 主要 LLM 全対応の高性能 BPE トークナイザー |
| [instructors](https://crates.io/crates/instructors) | [rust-instructor](https://github.com/goliajp/rust-instructor) | LLM からの型安全な構造化出力抽出 |
| **embedrs** | [rust-embeddings](https://github.com/goliajp/rust-embeddings) | 統一 Embedding -- クラウド + ローカル（本クレート） |
| [chunkedrs](https://crates.io/crates/chunkedrs) | [rust-chunker](https://github.com/goliajp/rust-chunker) | Embedding・検索向け AI ネイティブテキストチャンキング |

## ライセンス

[MIT](LICENSE)
