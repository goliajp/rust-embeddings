# embedrs

[![Crates.io](https://img.shields.io/crates/v/embedrs?style=flat-square&logo=rust)](https://crates.io/crates/embedrs)
[![docs.rs](https://img.shields.io/docsrs/embedrs?style=flat-square&logo=docs.rs)](https://docs.rs/embedrs)
[![CI](https://img.shields.io/github/actions/workflow/status/goliajp/rust-embeddings/ci.yml?branch=develop&style=flat-square&logo=github&label=ci)](https://github.com/goliajp/rust-embeddings/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/embedrs?style=flat-square)](#ライセンス)

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

> **0.6 からの移行**：0.7 では API が変わりました。`result.embeddings[i]` は `result[i]` になり、エラー型とバックオフ設定の名前が変わり、フォールバックは一時的な失敗のときだけ動きます。変更箇所ごとの直し方は[移行表](CHANGELOG.md#upgrading-from-06)にあり、どれも 1 行の修正で済みます。送信するリクエストは 0.6.2 と同じです。ローカルモデルのベクトルと 0.6.2 との差は、各成分で最大 3.3e-7 です。

embedrs はテキスト埋め込み（embedding）用の非同期 Rust クライアントです。6 社のクラウド API（OpenAI、Cohere、Gemini、Voyage、Jina、Mistral）と、同じプロセスの CPU で動く 4 つの小型 BERT モデルが同じ `EmbedResult` を返すので、バックエンドは 1 行の変更で切り替えられます。Apple M4 Pro での測定（既定のスレッド数）では、ローカルの `all-MiniLM-L6-v2` は fastembed（ONNX Runtime）と比べて、1 件ずつの処理で 3.1〜4.4 倍、256 件のバッチで 8.9〜15 倍速く、出力するベクトルの差は 3.3e-7 未満です。

## 特長

- **6 社のクラウド API** を 1 つの `Client` で扱う。各社に `*_compatible` コンストラクタがあり、プロキシや互換 API にも接続できる
- **ローカル推論**（feature `local`）：`all-MiniLM-L6-v2`、`all-MiniLM-L12-v2`、`bge-small-en-v1.5`、`gte-small` を実行する。順伝播は embedrs 自身の実装で、依存クレートはない。macOS では行列積を OS 標準の Accelerate フレームワークで計算し、それ以外では embedrs 内の AVX2/FMA（x86_64）または NEON（aarch64）カーネルで計算する。テキストはパディングせずに処理し、モデルごとの上限で切り詰める
- **バッチ処理**：`embed_batch` は任意の件数のテキストを、件数と token 数に応じて各社が受け付ける大きさのリクエストに分け、並行して送る。結果のベクトルはすべて 1 つのバッファに入る
- **リトライとフォールバック**：408、429、500、502、503、504、529 と接続エラーを指数バックオフで再試行し、`Retry-After`（秒数または日時）に従う。再試行しても失敗したときは、登録順に別のプロバイダーを試す
- **転送量の削減**：OpenAI、Voyage、Cohere からはベクトルを base64 で受け取り、embedrs 自身でデコードする。256 本のベクトルを返す応答のサイズは 10 進数のテキストで受け取る場合の 2.3 分の 1 になり、解析は 7.6 倍速い
- **料金の見積もり**（feature `cost-tracking`）：[tiktoken](https://crates.io/crates/tiktoken) の料金表をもとに、リクエストごとの料金を米ドルで見積もる

- **少ない依存**：コアは依存クレートを持たず、JSON、base64、WordPiece、SHA-256 を自前で実装している。既定のビルドで加わるのは reqwest（rustls と ring）と tokio のタイマーだけで、macOS では全部で 84 クレート（0.6.2 は 101）。`local` を有効にしても増えない。`default-features = false` なら embedrs だけになり、HTTP クライアントは利用者が用意する

## 性能

以下の数値はすべて 2026-11-26 に Apple M4 Pro Mac mini（14 コア、macOS 27.2.1）で測定しました。rustc 1.100、LTO を有効にした release ビルドで、測定中はこのマシンのベンチマーク用ロックを取得し、ほかの作業は動かしていません。クラウド呼び出しの測定中に 1 分間のロードアベレージを 15 秒ごとに記録し、最大 2.2 でした。ローカル推論の測定中の負荷はその節に書いています。

### ローカル推論

モデルは `all-MiniLM-L6-v2` です。同じプロセスの中で、0.6.2（candle を使う版）、同じモデルを同じ 256 token で切り詰める [fastembed](https://crates.io/crates/fastembed) 7.8.0（ONNX Runtime）、`potion-base-8M` を使う [model2vec-rs](https://crates.io/crates/model2vec-rs) 0.5.1 と比べました。ベクトルと 0.6.2 との差は各成分で最大 3.3e-7 です（38 件のテキスト、4 モデルすべて。コサイン類似度と 1 との差は 2e-12 以下）。0.6.2 と fastembed のコサイン類似度は、長さの異なる 32 件のテキストで最小でも 1.000000 でした。1 件ずつの処理は英単語 8、40、180 語（13、56、227 token）のテキスト、バッチは 8、40、180 語のテキスト 256 件と、4〜200 語が混在する 256 件です。1 件ずつの処理は 200 回の呼び出しの中央値（1 スレッドでは 30 回）、バッチは 5 回の中央値です。1 分間のロードアベレージを 10 秒ごとに記録し、最大 4.4 でした（ベンチマーク自身のスレッドを含む）。1 スレッドとは、embedrs では `EMBEDRS_THREADS=1 VECLIB_MAXIMUM_THREADS=1`、0.6.2 では `RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`、fastembed では演算内スレッド 1 本の設定です。

| 既定のスレッド数 | 0.6.2 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| 1 件、8 語 | 3.13 ms | **1.34 ms** | 4.21 ms | 0.0003 ms |
| 1 件、40 語 | 5.86 ms | **2.47 ms** | 8.17 ms | 0.0012 ms |
| 1 件、180 語 | 15.7 ms | **5.48 ms** | 24.1 ms | 0.0051 ms |
| 256 件、8 語 | 1,882 件/秒 | **6,948 件/秒** | 660 件/秒 | 280 万件/秒 |
| 256 件、40 語 | 763 件/秒 | **1,664 件/秒** | 188 件/秒 | 190 万件/秒 |
| 256 件、180 語 | 158 件/秒 | **344 件/秒** | 26 件/秒 | 110 万件/秒 |
| 256 件、長さ混在 | 209 件/秒 | **633 件/秒** | 42 件/秒 | 120 万件/秒 |

| 1 スレッド | 0.6.2 | **embedrs** | fastembed | model2vec |
|---|---|---|---|---|
| 1 件、8 語 | 2.19 ms | **1.47 ms** | 4.02 ms | 0.0004 ms |
| 1 件、40 語 | 5.55 ms | **2.56 ms** | 7.45 ms | 0.0012 ms |
| 1 件、180 語 | 23.5 ms | **9.88 ms** | 26.4 ms | 0.0051 ms |
| 256 件、8 語 | 806 件/秒 | **2,486 件/秒** | 579 件/秒 | 200 万件/秒 |
| 256 件、40 語 | 205 件/秒 | **520 件/秒** | 154 件/秒 | 63 万件/秒 |
| 256 件、180 語 | 40 件/秒 | **103 件/秒** | 26 件/秒 | 15 万件/秒 |
| 256 件、長さ混在 | 62 件/秒 | **198 件/秒** | 26 件/秒 | 25 万件/秒 |

Linux（Intel i7-10700K、8 コア、AVX2 対応の x86_64。ビルド設定とテキストは同じで、同じくベンチマーク用ロックの下で測定。ロードアベレージは最大 12.7）では、行列積を embedrs 内の AVX2/FMA カーネルで計算します。既定のスレッド数では embedrs がすべての行で fastembed より速く、1 スレッドでは 1 件ずつの処理と 180 語のバッチで ONNX Runtime のほうが速くなります。

| Intel i7-10700K、Linux | 0.6.2 | **embedrs** | fastembed | 0.6.2 1 スレッド | **embedrs 1 スレッド** | fastembed 1 スレッド |
|---|---|---|---|---|---|---|
| 1 件、8 語 | 11.9 ms | **1.92 ms** | 1.96 ms | 9.32 ms | **4.58 ms** | 3.89 ms |
| 1 件、40 語 | 22.5 ms | **3.03 ms** | 3.67 ms | 21.2 ms | **11.9 ms** | 11.6 ms |
| 1 件、180 語 | 46.3 ms | **8.74 ms** | 10.3 ms | 85.2 ms | **51.6 ms** | 46.2 ms |
| 256 件、8 語 | 640 件/秒 | **2,208 件/秒** | 1,196 件/秒 | 224 件/秒 | **341 件/秒** | 258 件/秒 |
| 256 件、40 語 | 171 件/秒 | **529 件/秒** | 289 件/秒 | 53 件/秒 | **80 件/秒** | 70 件/秒 |
| 256 件、180 語 | 33 件/秒 | **110 件/秒** | 62 件/秒 | 10 件/秒 | **16 件/秒** | 17 件/秒 |
| 256 件、長さ混在 | 49 件/秒 | **201 件/秒** | 61 件/秒 | 15 件/秒 | **30 件/秒** | 17 件/秒 |

モデルを 1 つ読み込んで長さ混在の 256 件を処理したプロセスのメモリ使用量のピーク（最大常駐セットサイズ）は、`all-MiniLM-L6-v2` で 178 MB（0.6.2 は 1,715 MB）、12 層の 3 モデルで 219〜234 MB（0.6.2 は 680〜1,883 MB）です。Linux のマシンでは `all-MiniLM-L6-v2` が 185 MB（0.6.2 は 1,009 MB）、`all-MiniLM-L12-v2` が 264 MB（0.6.2 は 572 MB）でした。F32 の重みファイルはメモリにマップしてそのまま読み、テキストはパディングせず、最大で約 4,096 token ずつまとめて計算します。

行列積は 0.6.2 でも embedrs でも Accelerate で計算しており、差は順伝播のそれ以外の部分にあります。embedrs は GELU をベクトル化した erf で計算し（0.6.2 は値ごとに `erff` を呼んでいた）、残差の加算をレイヤー正規化と同じ 1 回の走査で行い、アテンションはテキストとヘッドごとにパディングもマスクもなしで計算し、呼び出しの計算量が十分に大きいときだけ性能コアに分担させます。model2vec がどの行でも速いのは、モデルの種類が違うためです。静的埋め込みモデルは token ごとのベクトルを表から引いて平均するだけで、Transformer の層を持ちません。ここでは目安として載せており、品質の面では Transformer モデルの代わりにはなりません。

### クラウド呼び出し

同じプロセス内のモックサーバーが、ループバック経由で 1536 次元の OpenAI 形式の応答を返します。時間は `embed()` の呼び出し全体のもので（256 本は 50 回、2048 本は 15 回の中央値）、その多くはデータの転送にかかっています。比較対象は [async-openai](https://crates.io/crates/async-openai) 0.43.0 と [rig-core](https://crates.io/crates/rig-core) 0.47.1（`f64` を返す）で、同じ応答を渡しています。「—」は測定していない項目です。

| 応答 | 0.6.2 | **embedrs** | async-openai | rig-core |
|---|---|---|---|---|
| 256 本、最短の 10 進数（4.9 MB）| 6.64 ms | **3.87 ms** | — | — |
| 256 本、17 桁の 10 進数（8.3 MB）| 9.48 ms | **6.40 ms** | 9.69 ms | 22.0 ms |
| 256 本、base64（2.1 MB）| 1.43 ms | **0.80 ms** | 1.11 ms | — |
| 2048 本、最短の 10 進数（39 MB）| 59.6 ms | **32.7 ms** | — | — |
| 2048 本、17 桁の 10 進数（67 MB）| 80.4 ms | **53.5 ms** | 80.2 ms | 172 ms |
| 2048 本、base64（17 MB）| 12.1 ms | **7.1 ms** | 9.4 ms | — |

解析だけを比べると、10 進数の応答にかかる時間は serde_json の 0.49〜0.62 倍です（4.9 MB で 3.0 ms、serde_json は 6.1 ms）。数値の丸めは正確で、有限の `f32` をすべて 4 通りの書式で出力した 171 億個の文字列と、10 億個のランダムな文字列のすべてで、`str::parse::<f32>` とビット単位で一致しました。ベクトルが 1 本のとき `embed()` は約 0.05 ms で、0.6.2 や async-openai と同じです。ネットワークの往復には数十ミリ秒かかります。

## インストール

```toml
[dependencies]
embedrs = "0.7"

# ローカル推論を使う場合
embedrs = { version = "0.7", features = ["local"] }
```

| feature | 既定 | 追加される機能 |
|---|---|---|
| `reqwest` | 有効 | HTTP の送受信と、それを使うコンストラクタ（`embedrs::cloud`、`Client::openai` など）|
| `local` | 無効 | CPU でのローカル推論、依存クレートなし（トークナイザーは同梱、重みは初回使用時にダウンロード）|
| `cost-tracking` | 無効 | `Usage::cost`。tiktoken の料金表を使う（トークナイザーの語彙はビルドに含まれない）|
| `token-count` | 無効 | OpenAI の token 数を正確に数え、バッチの分割と長すぎる入力の検出に使う（`cl100k_base` の語彙をビルドに含む）|
| `tracing` | 無効 | `tracing` crate による span とイベントの出力 |

macOS での依存ツリーのクレート数（embedrs 自身を含む。括弧内は 0.6.2）は、既定で 84（101）、既定に `local` を加えても 84（255）、`default-features = false` で 1、`reqwest` なしで `local` だけでも 1 です。x86_64 Linux の既定のビルドは 82 です。

## 使い方

```rust,no_run
async fn run() -> embedrs::Result<()> {
    // クラウド：OpenAI text-embedding-3-small
    let client = embedrs::cloud("sk-...");
    let result = client.embed(["hello world"]).await?;
    println!("{} dimensions", result.dim()); // 1536
    println!("{:?}", &result[0][..4]); // 最初のベクトル（&[f32]）

    // ローカル：all-MiniLM-L6-v2、API キーは不要
    let client = embedrs::local();
    let result = client.embed(["hello world"]).await?;
    println!("{} dimensions", result.dim()); // 384
    Ok(())
}
```

`embed` には `Vec<String>`、`&[String]`、`&[&str]`、`&str` の配列、単独の `&str` を渡せます。`result[i]` は `i` 番目のテキストのベクトルです。`result.iter()` で順に取り出せ、`result.as_flat()` はバッファ全体、`result.to_vecs()` は `Vec<Vec<f32>>` へのコピーです。

ローカルモデルを初めて使うときは、重みを Hugging Face のキャッシュ（`HF_HOME`、既定は `~/.cache/huggingface`）にダウンロードします。2 回目以降の呼び出しや他のプロセスは、ダウンロード済みの重みを使います。ダウンロードするのは固定したコミットのファイルで、中断しても続きから再開し、完了後に SHA-256 を検証します。ダウンロードの時間は呼び出しのタイムアウトに含まれません。

## クラウド API

| プロバイダー | コンストラクタ | 既定のモデル | 1 リクエストのテキスト数 | `dimensions` |
|---|---|---|---|---|
| OpenAI | `Client::openai(key)` | `text-embedding-3-small` | 2048 | `text-embedding-3-*` |
| Cohere | `Client::cohere(key)` | `embed-v4.0` | 96 | `embed-v4.0`、`embed-v5.0-*`。v3 のモデルはエラー |
| Gemini | `Client::gemini(key)` | `gemini-embedding-001` | 100 | 128〜3072 |
| Voyage | `Client::voyage(key)` | `voyage-4-large` | 1000 | 4、3.5、3-large、code-3 系列。それより古いモデルはエラー |
| Jina | `Client::jina(key)` | `jina-embeddings-v3` | 2048 | 対応 |
| Mistral | `Client::mistral(key)` | `mistral-embed` | 512 | `codestral-embed`。`mistral-embed` はエラー |
| ローカル | `Client::local(name)?` | `all-MiniLM-L6-v2` | 256 | モデル本来の次元数のみ |

モデル名は任意の文字列なので、新しいモデルも embedrs の更新を待たずに使えます（`client.embed(texts).model("text-embedding-3-large")`）。`cost-tracking` が使う入力 100 万 token あたりの料金は次のとおりです（tiktoken 5.5）。

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

Azure OpenAI や、別のヘッダーが必要なゲートウェイには `Endpoint` を使います。

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

## ローカルモデル

| モデル | 層数 | 重みのサイズ | 読み込む上限 | プーリング |
|---|---|---|---|---|
| `all-MiniLM-L6-v2`（既定）| 6 | 91 MB | 256 token | 平均 |
| `all-MiniLM-L12-v2` | 12 | 133 MB | 128 token | 平均 |
| `bge-small-en-v1.5` | 12 | 133 MB | 512 token | CLS |
| `gte-small` | 12 | 67 MB | 512 token | 平均 |

どのモデルも L2 正規化した 384 次元のベクトルを返します。入力の種類が `InputType::SearchQuery` のとき、`bge-small-en-v1.5` には学習時と同じクエリ用の指示文を先頭に付けます。ローカルのクライアントでは `.model()` で別のローカルモデルを指定でき、`.dimensions()` は 384 だけを受け付けます。それ以外の値はエラーになります。上限を超えるテキストは切り詰め、`usage.total_tokens` にはモデルが実際に読んだ token 数が入ります。

読み込んだモデルはそれぞれ専用のスレッドで動くため、ローカル推論に非同期ランタイムは要りません。`LocalModel::from_files` は手元の重みファイル（F32、F16、BF16 の safetensors）を読み込み、`embed_blocking` は計算が終わるまで待って結果を返します。1 回の呼び出しの token が数十を超えると、計算中はワーカースレッドも使います。数は macOS では性能コアの数、それ以外ではプロセスが使えるすべてのコアです。`EMBEDRS_THREADS` で変えられ、`EMBEDRS_THREADS=1` なら呼び出し全体をモデル専用のスレッドで計算します（macOS の Accelerate のスレッド数は引き続き `VECLIB_MAXIMUM_THREADS` に従います）。

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

    println!("{} vectors, {:?} tokens", result.len(), result.usage.total_tokens);
    Ok(())
}
```

1 つのリクエストには、プロバイダーの件数の上限まで、token 数の上限が公開されている場合はその上限まで（OpenAI は 30 万、Voyage は 12 万）テキストを詰めます。テキストの token 数は UTF-8 のバイト数とみなします。OpenAI のトークナイザーでは、これが実際より少なくなることはありません。`token-count` を有効にすると OpenAI の token 数を正確に数えるので、1 リクエストにより多く詰められます。

## 入力の種類と次元数

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

`InputType` には `SearchDocument`、`SearchQuery`、`Classification`、`Clustering` があり、各社の API にはそれぞれの名前で渡します（Cohere は `search_query`、Gemini は `RETRIEVAL_QUERY`、Jina は `retrieval.query` など）。`gemini-embedding-2` にはテキストの接頭辞として渡します。OpenAI と Mistral にはこの指定がないため無視されます。

## 長い入力

```rust,no_run
use embedrs::Truncate;

async fn run(client: &embedrs::Client, texts: &[String]) -> embedrs::Result<()> {
    let result = client.embed(texts).truncate(Truncate::Error).await?;
    println!("{}", result.len());
    Ok(())
}
```

既定の `Truncate::Provider` では、長い入力の扱いはプロバイダーに任せます。Voyage、Cohere、Gemini、ローカルモデルは末尾を切り詰め、OpenAI と Jina はエラーを返します。`Truncate::End` は切り詰めを、`Truncate::Error` は拒否を指定します。プロバイダーにその指定がない場合は `Error::Unsupported` になります。ローカルモデルは計算の前に `Error::InputTooLong` を返し、`token-count` を有効にしていれば OpenAI のモデルも送信前に同じエラーを返します。

## リトライ、タイムアウト、フォールバック

```rust,no_run
use std::time::Duration;
use embedrs::{Backoff, Client};

async fn run() -> embedrs::Result<()> {
    let client = Client::openai("sk-...")
        .with_retry(Backoff::new()) // 初回 500 ms、上限 30 s、3 回まで
        .with_timeout(Duration::from_secs(120)) // リトライを含めたリクエスト全体の時間（既定 60 s）
        .with_fallback(Client::cohere("cohere-key"))
        .with_fallback(Client::voyage("voyage-key").with_model("voyage-4-lite"));

    let result = client.embed(["hello"]).await?;
    println!("answered by {}", result.model);
    Ok(())
}
```

`with_retry` を指定しない場合、失敗したリクエストは再試行しません。主となるプロバイダーが再試行後も一時的なエラー（上に挙げたステータスコード、または接続できない）で失敗すると、登録順にフォールバック先を試します。フォールバック先はそれぞれ自身のモデルと次元数を使います。すべて失敗した場合は、主となるプロバイダーのエラーを返します。400、401、403、422 はすぐに返すので、API キーの誤りがフォールバックに隠れることはありません。モデルが違うとベクトルは比較できないため、フォールバックは可用性を保つためのものです。1 つのインデックスに複数のモデルのベクトルを混ぜる用途には使えません。

## 自分の HTTP クライアントを使う

```rust,no_run
use embedrs::{Client, transport::HttpResponse};

fn run(client: &Client) -> embedrs::Result<()> {
    let prepared = client.prepare(["hello world"])?;
    let request = prepared.request(); // method, url, headers, body
    println!("POST {} ({} bytes)", request.url, request.body.len());

    let body = br#"{"data":[{"index":0,"embedding":[0.6,0.8]}]}"#; // HTTP クライアントが受け取った応答
    let result = prepared.finish(HttpResponse::new(200, vec![], body.to_vec()))?;
    assert_eq!(&result[0], &[0.6, 0.8]);
    Ok(())
}
```

`prepare` と `finish` は I/O をしないので、同期式のものも含めてどの HTTP クライアントとも組み合わせられます。[`examples/blocking.rs`](examples/blocking.rs) は標準ライブラリの `TcpStream` だけを使い、依存クレートがありません。再試行、フォールバック、バッチ、タイムアウトも使いたい場合は、`Transport`（必須のメソッドはリクエストの送信と待機の 2 つ）を実装して `Client::new` に渡します。embedrs はタスクを起動せず、どのランタイムでも動きます。reqwest の設定（プロキシ、接続プール、ヘッダー）を変えるには、TLS プロバイダーの設定を済ませた `ReqwestTransport::client_builder()` でクライアントを作り、`client.with_transport(Arc::new(ReqwestTransport::from(reqwest_client)))` と書きます。

どの埋め込みバックエンドでも動くコードを書くには、`&dyn embedrs::Embed` を受け取ります。メソッドが 1 つだけのトレイトで、`Client` に実装されています。

## 料金の見積もり

```rust,no_run
async fn run(client: &embedrs::Client) -> embedrs::Result<()> {
    let result = client.embed(["hello"]).await?;
    if let Some(cost) = result.usage.cost {
        println!("about ${cost:.8}");
    }
    Ok(())
}
```

モデルが料金表にない場合、プロバイダーが token 数を返さなかった場合、`cost-tracking` を有効にしていない場合、`cost` は `None` です。

## エラー処理

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
