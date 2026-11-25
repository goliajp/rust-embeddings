# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-embeddings/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-embeddings/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

> **0.6.1 からの更新**：API に変更はありません。ローカルモデルは各モデル本来の上限までテキストを読むようになりました（既定のモデルは 128 token から 256 token に変更）。そのため長いテキストのベクトルは変わります。フォールバック先のクライアントは自身のモデルと次元数を使うようになりました。バックエンドが対応していない指定は、無視せずにエラーを返します。詳細は [CHANGELOG](CHANGELOG.md#upgrading-from-061) を参照してください。

embedrs はテキスト埋め込み（embedding）用の非同期 Rust クライアントです。6 社のクラウド API（OpenAI、Cohere、Gemini、Voyage、Jina、Mistral）と、同じプロセスの CPU で動く 4 つの小型 BERT モデルが同じ `EmbedResult` を返すので、バックエンドは 1 行の変更で切り替えられます。Apple M4 Pro での測定では、ローカルの `all-MiniLM-L6-v2` は fastembed（ONNX Runtime）と比べて、1 件ずつの処理で 1.1〜1.7 倍、256 件のバッチで 1.3〜12 倍速く、出力するベクトルは同じです。

## 特長

- **6 社のクラウド API** を 1 つの `Client` で扱う。各社に `*_compatible` コンストラクタがあり、プロキシや互換 API にも接続できる
- **ローカル推論**（feature `local`）：`all-MiniLM-L6-v2`、`all-MiniLM-L12-v2`、`bge-small-en-v1.5`、`gte-small` を純 Rust の [candle](https://github.com/huggingface/candle) で実行する。macOS では行列積を OS 標準の Accelerate フレームワークで計算する。テキストは長さ順に並べてグループごとにパディングし、モデルごとの上限で切り詰める
- **バッチ処理**：`embed_batch` は任意の件数のテキストを各社が受け付ける単位のリクエストに分け、並行して送る
- **リトライとフォールバック**：408、429、500、502、503、504、529 と接続エラーを指数バックオフで再試行し、`Retry-After` に従う。失敗したときは登録順に別のプロバイダーを試す
- **転送量の削減**：OpenAI、Voyage、Cohere からはベクトルを base64 で受け取り、embedrs 自身でデコードする。256 本のベクトルを返す応答のサイズは 10 進数のテキストで受け取る場合の 2.3 分の 1 になり、解析は 8.6 倍速い
- **料金の見積もり**（feature `cost-tracking`）：[tiktoken](https://crates.io/crates/tiktoken) の料金表をもとに、リクエストごとの料金を米ドルで見積もる

## 性能

ローカル推論、モデルは `all-MiniLM-L6-v2`。測定環境は Apple M4 Pro Mac mini（14 コア、macOS）、rustc 1.100、LTO を有効にした release ビルドです。比較対象は、同じモデルを同じ 256 token で切り詰める [fastembed](https://crates.io/crates/fastembed) 7.8.0（ONNX Runtime）と、`potion-base-8M` を使う [model2vec-rs](https://crates.io/crates/model2vec-rs) 0.5.1 です。長さの異なる 32 件のテキストで、embedrs と fastembed のベクトルのコサイン類似度は最小でも 1.000000 でした。テキストは英単語 8、40、180 語（平均 13、54、232 token）と、4〜200 語が混在するものです。1 件ずつの処理は 200 回の呼び出しの中央値（1 スレッドでは 30 回）、バッチは 256 件を 3 回処理した中央値です。測定中のロードアベレージは 1.7〜7.2 でした。「0.6.1」の列は、依存関係を同じにした前リリースのコードです。

| 既定のスレッド数 | 0.6.1 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| 1 件、8 語 | 18.7 ms | **3.6 ms** | 4.3 ms | 0.0003 ms |
| 1 件、40 語 | 18.7 ms | **6.0 ms** | 8.5 ms | 0.0012 ms |
| 1 件、180 語 | 18.8 ms ¹ | **15.9 ms** | 24.6 ms | 0.0051 ms |
| 256 件、8 語 | 155 件/s | **1,872 件/s** | 672 件/s | 300 万件/s |
| 256 件、40 語 | 152 件/s | **776 件/s** | 184 件/s | 210 万件/s |
| 256 件、180 語 | 145 件/s ¹ | **159 件/s** | 22 件/s | 100 万件/s |
| 256 件、長さ混在 | 152 件/s ¹ | **249 件/s** | 21 件/s | 80 万件/s |

| 1 スレッド | 0.6.1 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| 1 件、8 語 | 39.4 ms | **2.0 ms** | 3.5 ms | 0.0004 ms |
| 1 件、40 語 | 39.8 ms | **5.6 ms** | 7.3 ms | 0.0012 ms |
| 1 件、180 語 | 40.3 ms ¹ | **23.7 ms** | 26.1 ms | 0.0051 ms |
| 256 件、8 語 | 27 件/s | **798 件/s** | 581 件/s | 200 万件/s |
| 256 件、40 語 | 27 件/s | **204 件/s** | 156 件/s | 57 万件/s |
| 256 件、180 語 | 27 件/s ¹ | **40 件/s** | 21 件/s | 14 万件/s |
| 256 件、長さ混在 | 27 件/s ¹ | **59 件/s** | 12 件/s | 24 万件/s |

¹ 0.6.1 はすべてのテキストを 128 token で切り詰め、128 token までパディングしていたため、180 語のテキストでは他の 2 つの半分程度の計算しかしていません。

model2vec がどの行でも速いのは、モデルの種類が違うためです。静的埋め込みモデルは token ごとのベクトルを表から引いて平均するだけで、Transformer の層を持ちません。ここでは目安として載せており、品質の面では Transformer モデルの代わりにはなりません。fastembed との差は、テキストのグループごとに最長のテキストまでしかパディングしないこと、macOS では行列積を Accelerate で計算すること、query、key、value の射影を 1 回の行列積にまとめたことによるものです。このリリースでは Linux での測定はしていません。

クラウド呼び出し：同じプロセス内のモックサーバーが、ループバック経由で 1536 次元の OpenAI 形式の応答を返します。256 本のベクトルでは、10 進数の応答（4.88 MB）に対して `embed()` は 6.85 ms、base64 の応答（2.11 MB）に対して 1.43 ms でした。embedrs が OpenAI、Voyage、Cohere に要求するのは base64 です。2048 本では 59.2 ms と 12.1 ms です。クライアントのその他の処理は 1 回あたり約 0.05 ms で、数十ミリ秒かかるネットワークの往復と比べればわずかです。

## インストール

```toml
[dependencies]
embedrs = "0.6"

# ローカル推論を使う場合
embedrs = { version = "0.6", features = ["local"] }
```

| feature | 既定 | 追加される機能 |
|---|---|---|
| *（なし）* | 有効 | クライアントと 6 社すべてのクラウド API |
| `local` | 無効 | candle によるローカル推論（トークナイザーは同梱、重みは初回使用時にダウンロード）|
| `cost-tracking` | 無効 | `Usage::cost`。tiktoken の料金表を使う（トークナイザーの語彙はビルドに含まれない）|
| `tracing` | 無効 | `tracing` crate による span とイベントの出力 |

## 使い方

```rust,no_run
async fn run() -> embedrs::Result<()> {
    // クラウド：OpenAI text-embedding-3-small
    let client = embedrs::cloud("sk-...");
    let result = client.embed(vec!["hello world".into()]).await?;
    println!("{} dimensions", result.embeddings[0].len()); // 1536

    // ローカル：all-MiniLM-L6-v2、API キーは不要
    let client = embedrs::local();
    let result = client.embed(vec!["hello world".into()]).await?;
    println!("{} dimensions", result.embeddings[0].len()); // 384
    Ok(())
}
```

ローカルモデルを初めて使うときは、重みを Hugging Face のキャッシュ（`HF_HOME`、既定は `~/.cache/huggingface`）にダウンロードします。2 回目以降の呼び出しや他のプロセスは、ダウンロード済みの重みを使います。

## クラウド API

| プロバイダー | コンストラクタ | 既定のモデル | 1 リクエストのテキスト数 | `dimensions` |
|---|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 | `text-embedding-3-*` |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 | `embed-v4.0`、`embed-v5.0-*`。v3 のモデルはエラー |
| Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 | 128〜3072 |
| Voyage | `Client::voyage(key)` | `voyage-4-large` | 1000（`embed_batch` は 128 ずつ送信）| 4、3.5、3-large、code-3 系列。それより古いモデルはエラー |
| Jina | `Client::jina(key)` | `jina-embeddings-v3` | 2048 | 対応 |
| Mistral | `Client::mistral(key)` | `mistral-embed` | 512 | `codestral-embed`。`mistral-embed` はエラー |
| ローカル | `Client::local(name)?` | `all-MiniLM-L6-v2` | 256 | モデル本来の次元数のみ |

モデル名は任意の文字列なので、新しいモデルも embedrs の更新を待たずに使えます（`client.embed(texts).model("text-embedding-3-large")`）。`cost-tracking` が使う入力 100 万 token あたりの料金は次のとおりです（tiktoken 5.4）。

| モデル | 米ドル／100 万 token | モデル | 米ドル／100 万 token |
|---|---|---|---|
| `text-embedding-3-small` | 0.02 | `voyage-4-large` | 0.12 |
| `text-embedding-3-large` | 0.13 | `voyage-4` | 0.06 |
| `gemini-embedding-001` | 0.15 | `voyage-4-lite` | 0.02 |
| `gemini-embedding-2` | 0.20 | `voyage-3-large` | 0.18 |
| `mistral-embed` | 0.10 | `codestral-embed` | 0.15 |

Cohere と Jina のモデルは料金表に載っていないため、`cost` は `None` になります。

2026 年 11 月時点のモデル名は次のとおりです。OpenAI `text-embedding-3-small`、`text-embedding-3-large`；Cohere `embed-v4.0`、`embed-v5.0-pro`、`embed-v5.0-fast`；Gemini `gemini-embedding-001`、`gemini-embedding-2`；Voyage `voyage-4-large`、`voyage-4`、`voyage-4-lite`、`voyage-code-4`、`voyage-3.5`、`voyage-3.5-lite`、`voyage-3-large`、`voyage-code-3`；Jina `jina-embeddings-v3`、`jina-embeddings-v5-text-small`、`jina-embeddings-v5-text-nano`；Mistral `mistral-embed`、`codestral-embed`。

各クラウド API には、ベース URL を指定できる `*_compatible` コンストラクタもあります。

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

`openai_compatible` は、OpenAI 互換のサーバーならどれでも扱える 10 進数のベクトルを要求します。base64 を要求するのは OpenAI 公式のエンドポイントだけです。

## ローカルモデル

| モデル | 層数 | 重みのサイズ | 読み込む上限 | プーリング |
|---|---|---|---|---|
| `all-MiniLM-L6-v2`（既定）| 6 | 91 MB | 256 token | 平均 |
| `all-MiniLM-L12-v2` | 12 | 133 MB | 128 token | 平均 |
| `bge-small-en-v1.5` | 12 | 133 MB | 512 token | CLS |
| `gte-small` | 12 | 67 MB | 512 token | 平均 |

どのモデルも L2 正規化した 384 次元のベクトルを返します。入力の種類が `InputType::SearchQuery` のとき、`bge-small-en-v1.5` には学習時と同じクエリ用の指示文を先頭に付けます。ローカルのクライアントでは `.model()` で別のローカルモデルを指定でき、`.dimensions()` は 384 だけを受け付けます。それ以外の値はエラーになります。上限を超えるテキストは切り詰め、`usage.total_tokens` にはモデルが実際に読んだ token 数が入ります。

## バッチ処理

```rust,no_run
async fn run() -> embedrs::Result<()> {
    let client = embedrs::cloud("sk-...");
    let texts: Vec<String> = (0..5000).map(|i| format!("document {i}")).collect();

    let result = client
        .embed_batch(texts)
        .concurrency(5) // 同時に送るリクエスト数（既定 5）
        .chunk_size(512) // 1 リクエストのテキスト数（既定はプロバイダーの上限）
        .model("text-embedding-3-large")
        .await?;

    println!("{} vectors, {} tokens", result.embeddings.len(), result.usage.total_tokens);
    Ok(())
}
```

## 入力の種類と次元数

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

`InputType` には `SearchDocument`、`SearchQuery`、`Classification`、`Clustering` があり、各社の API にはそれぞれの名前で渡します（Cohere は `search_query`、Gemini は `RETRIEVAL_QUERY`、Jina は `retrieval.query` など）。`gemini-embedding-2` にはテキストの接頭辞として渡します。OpenAI と Mistral にはこの指定がないため無視されます。

## リトライ、タイムアウト、フォールバック

```rust,no_run
use std::time::Duration;
use embedrs::{BackoffConfig, Client};

async fn run() -> embedrs::Result<()> {
    let client = Client::openai("sk-...")
        .with_retry_backoff(BackoffConfig::default()) // 初回 500 ms、上限 30 s、3 回まで
        .with_timeout(Duration::from_secs(120)) // リトライを含めたリクエスト全体の時間（既定 60 s）
        .with_fallback(Client::cohere("cohere-key"))
        .with_fallback(Client::voyage("voyage-key").with_model("voyage-4-lite"));

    let result = client.embed(vec!["hello".into()]).await?;
    println!("answered by {}", result.model);
    Ok(())
}
```

`with_retry_backoff` を指定しない場合、失敗したリクエストは再試行しません。主となるプロバイダーが失敗すると、登録順にフォールバック先を試します。フォールバック先はそれぞれ自身のモデルと次元数を使います。すべて失敗した場合は、主となるプロバイダーのエラーを返します。モデルが違うとベクトルは比較できないため、フォールバックは可用性を保つためのものです。1 つのインデックスに複数のモデルのベクトルを混ぜる用途には使えません。

## 料金の見積もり

```rust,no_run
async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let result = client.embed(vec!["hello".into()]).await?;
    if let Some(cost) = result.usage.cost {
        println!("about ${cost:.8}");
    }
    Ok(())
}
```

モデルが料金表にない場合や、プロバイダーが token 数を返さなかった場合、`cost` は `None` です。

## エラー処理

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

## 類似度の計算

```rust
use embedrs::{cosine_similarity, dot_product, euclidean_distance};

let a = [1.0, 0.0, 0.0];
let b = [0.0, 1.0, 0.0];
assert_eq!(cosine_similarity(&a, &b), 0.0);
assert_eq!(dot_product(&a, &b), 0.0);
assert!((euclidean_distance(&a, &b) - 2f32.sqrt()).abs() < 1e-6);
```

## モデルの選び方

[`examples/embedding_models/`](examples/embedding_models/) では、英語、中国語、日本語の 184 件のテキストで 8 つのモデルを評価しています（2026-03-08 に記録。`cargo run --release --features local --example embedding_models` を実行すると、保存したベクトルから全項目を再計算します）。

| | MiniLM-L6 | MiniLM-L12 | BGE-small | GTE-small | OpenAI 3-small | Gemini 001 | Cohere v4 | Voyage 3-large |
|---|---|---|---|---|---|---|---|---|
| Spearman ρ（全 32 組）| 0.81 | 0.84 | 0.71 | 0.75 | 0.91 | 0.94 | 0.91 | 0.89 |
| ρ（英語）| 0.92 | 0.94 | 0.92 | 0.90 | 0.91 | 0.91 | 0.89 | 0.88 |
| ρ（中国語）| 0.65 | 0.74 | 0.45 | 0.40 | 0.88 | 0.99 | 0.93 | 0.89 |
| ρ（日本語）| 0.60 | 0.90 | 0.20 | 0.50 | 0.90 | 1.00 | 1.00 | 0.90 |
| 検索の top-1（9 クエリ）| 9 | 9 | 8 | 9 | 9 | 8 | 9 | 8 |
| 言語間のコサイン類似度 | 0.25 | 0.26 | 0.66 | 0.81 | 0.71 | 0.84 | 0.68 | 0.85 |
| 類似ペアと非類似ペアのコサイン差 | 0.52 | 0.52 | 0.29 | 0.14 | 0.58 | 0.30 | 0.46 | 0.45 |

ローカルの 4 モデルは英語向けのモデルで、中国語、日本語、言語をまたぐ比較ではクラウドのモデルより明らかに低いスコアです。評価データは小さい（スコア付きのペアは 32 組）ため、0.0x 程度の差は誤差の範囲です。

## ライセンス

[MIT](LICENSE-MIT) と [Apache-2.0](LICENSE-APACHE) のどちらかを選んで利用できます。
