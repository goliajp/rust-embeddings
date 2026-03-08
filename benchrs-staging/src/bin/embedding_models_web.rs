/// web interface for the embedding model selection experiment
///
/// ```bash
/// cargo run -p benchrs --bin embedding_models_web --release
/// # open http://localhost:3000
/// ```
use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::services::{ServeDir, ServeFile};

use benchrs::stats;

const DATA_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/embedding_models.json.zst"
);
const WEB_DIST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/web/dist");

// ═══════════════════════════════════════════
//  data types
// ═══════════════════════════════════════════

#[derive(Serialize, Deserialize, Clone)]
struct RecordedData {
    models: HashMap<String, ModelData>,
    metadata: Metadata,
}

#[derive(Serialize, Deserialize, Clone)]
struct ModelData {
    dimensions: usize,
    embeddings: HashMap<String, Vec<f32>>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Metadata {
    date: String,
    platform: String,
    corpus_size: usize,
    models: Vec<String>,
}

#[derive(Serialize, Clone)]
struct ModelResult {
    name: String,
    dimensions: usize,
    similarity_rho: f64,
    discrimination_gap: f64,
    similar_avg: f64,
    dissimilar_avg: f64,
    retrieval_top1: usize,
    retrieval_total: usize,
    mrr: f64,
    en_rho: f64,
    zh_rho: f64,
    ja_rho: f64,
    crosslingual_avg: f64,
    crosslingual_min: f64,
    robustness_avg: f64,
    robustness_min: f64,
    cluster_purity: usize,
    cluster_total: usize,
    cluster_separation: f64,
}

#[derive(Deserialize)]
struct RunRequest {
    openai_key: Option<String>,
    gemini_key: Option<String>,
    cohere_key: Option<String>,
    voyage_key: Option<String>,
}

struct AppState {
    prerecorded: Vec<ModelResult>,
}

// ═══════════════════════════════════════════
//  test data (same as CLI version)
// ═══════════════════════════════════════════

fn graded_similarity_pairs() -> Vec<(&'static str, &'static str, f32)> {
    vec![
        (
            "A dog is running through the grass",
            "A dog runs across a green field",
            4.8,
        ),
        (
            "The stock market crashed today",
            "Financial markets experienced a sharp decline",
            4.5,
        ),
        (
            "She plays the piano beautifully",
            "She is a talented pianist",
            4.3,
        ),
        (
            "The children are playing in the park",
            "Kids are having fun at the playground",
            4.2,
        ),
        (
            "He is cooking dinner in the kitchen",
            "A man prepares a meal at home",
            4.0,
        ),
        (
            "Scientists discovered a new species of fish",
            "Researchers found a previously unknown fish species",
            4.7,
        ),
        (
            "The movie received excellent reviews",
            "Critics praised the film highly",
            4.4,
        ),
        (
            "She graduated from university last year",
            "She completed her college degree recently",
            4.1,
        ),
        (
            "The cat sat on the windowsill",
            "A bird was perched on the fence",
            2.5,
        ),
        (
            "He drives to work every morning",
            "She takes the bus to school",
            2.8,
        ),
        (
            "The restaurant serves Italian food",
            "The cafe has a French menu",
            3.0,
        ),
        (
            "I enjoy reading science fiction novels",
            "She likes watching fantasy movies",
            2.7,
        ),
        (
            "The temperature dropped below freezing",
            "It was a very cold winter day",
            3.5,
        ),
        (
            "He is a software engineer at Google",
            "She works as a data scientist at Meta",
            3.0,
        ),
        (
            "The sun rises in the east",
            "Quantum entanglement violates local realism",
            0.2,
        ),
        (
            "She ordered a cappuccino at Starbucks",
            "The Pythagorean theorem relates triangle sides",
            0.1,
        ),
        (
            "The train arrived at platform 3",
            "Photosynthesis converts light into energy",
            0.3,
        ),
        (
            "He adopted a golden retriever puppy",
            "The GDP of Japan decreased in Q3",
            0.2,
        ),
        (
            "The concert tickets sold out quickly",
            "Mitochondria are the powerhouse of the cell",
            0.1,
        ),
        (
            "We went hiking in the mountains",
            "TCP/IP is the backbone of the internet",
            0.3,
        ),
        (
            "The baby started crawling last week",
            "The Fourier transform decomposes signals",
            0.1,
        ),
        (
            "今天天气真好，适合出去走走",
            "天气晴朗，是个散步的好日子",
            4.5,
        ),
        (
            "机器学习需要大量的训练数据",
            "人工智能系统依赖海量数据集进行学习",
            4.3,
        ),
        ("他在大学学习计算机科学", "她是一名软件工程专业的学生", 3.2),
        ("这家餐厅的菜很好吃", "今天的股票市场表现不佳", 0.3),
        ("春天来了，花都开了", "量子计算机使用量子比特", 0.1),
        ("我喜欢在周末看电影", "她每天早上跑步锻炼身体", 1.5),
        ("東京は日本の首都です", "日本の首都は東京である", 5.0),
        (
            "彼は毎日電車で通勤しています",
            "彼女はバスで学校に通っています",
            2.8,
        ),
        (
            "桜の季節は本当に美しい",
            "プログラミング言語の構文解析",
            0.2,
        ),
        (
            "このレストランのラーメンは最高だ",
            "あのカフェのコーヒーは美味しい",
            2.5,
        ),
        (
            "人工知能の研究が進んでいる",
            "AI技術は急速に発展している",
            4.6,
        ),
    ]
}

fn retrieval_test_sets() -> Vec<(&'static str, &'static [&'static str], usize)> {
    vec![
        (
            "How to sort an array in Python?",
            &[
                "Python's sorted() function returns a new sorted list from an iterable",
                "JavaScript uses Array.prototype.sort() for sorting",
                "The quick brown fox jumps over the lazy dog",
                "Machine learning algorithms can classify images",
            ],
            0,
        ),
        (
            "What causes climate change?",
            &[
                "The recipe calls for two cups of flour",
                "Greenhouse gas emissions from burning fossil fuels trap heat in the atmosphere",
                "The stock market saw gains in the tech sector",
                "Regular exercise improves cardiovascular health",
            ],
            1,
        ),
        (
            "Best practices for database indexing",
            &[
                "Yoga and meditation can reduce stress levels",
                "The history of ancient Rome spans over a thousand years",
                "Create indexes on columns frequently used in WHERE clauses and JOIN conditions",
                "Modern art galleries feature contemporary paintings",
            ],
            2,
        ),
        (
            "How does photosynthesis work?",
            &[
                "Investment portfolios should be diversified across asset classes",
                "The Eiffel Tower was completed in 1889",
                "Blockchain technology uses distributed ledgers for transactions",
                "Plants convert sunlight, water, and CO2 into glucose and oxygen using chlorophyll",
            ],
            3,
        ),
        (
            "Symptoms of vitamin D deficiency",
            &[
                "Low vitamin D levels can cause fatigue, bone pain, muscle weakness, and mood changes",
                "The latest smartphone features a triple camera system",
                "Renaissance art flourished in 15th century Italy",
                "Rust's ownership system prevents memory safety bugs at compile time",
            ],
            0,
        ),
        (
            "如何学习一门新的编程语言？",
            &[
                "通过阅读官方文档、做练习项目和参与开源社区来学习编程语言",
                "今天的天气预报说下午会下雨",
                "这部电影获得了奥斯卡最佳影片奖",
                "健康饮食应该包含足够的蔬菜和水果",
            ],
            0,
        ),
        (
            "为什么要使用版本控制系统？",
            &[
                "瑜伽有助于放松身心，减轻压力",
                "中国的长城是世界七大奇迹之一",
                "Git等版本控制系统可以追踪代码变更、支持多人协作、方便回滚",
                "今年的樱花比往年开得早",
            ],
            2,
        ),
        (
            "Rustプログラミング言語の特徴は？",
            &[
                "今日のランチはカレーライスにしよう",
                "Rustは所有権システムによりメモリ安全性をコンパイル時に保証する言語です",
                "東京オリンピックは2021年に開催された",
                "この映画の評価はとても高い",
            ],
            1,
        ),
        (
            "健康的な食生活とは？",
            &[
                "バランスの取れた栄養摂取と規則正しい食事時間が健康的な食生活の基本です",
                "新幹線は時速300キロで走行する",
                "プログラミングの基礎を学ぶにはPythonがおすすめ",
                "来週の会議は水曜日に変更になった",
            ],
            0,
        ),
    ]
}

fn crosslingual_groups() -> Vec<&'static [&'static str]> {
    vec![
        &[
            "Artificial intelligence is transforming the world",
            "人工智能正在改变世界",
            "人工知能は世界を変えている",
        ],
        &[
            "The weather is beautiful today",
            "今天天气很好",
            "今日は天気がいい",
        ],
        &[
            "I like to read books in my free time",
            "我喜欢在空闲时间读书",
            "暇な時に本を読むのが好きです",
        ],
        &[
            "Programming is a valuable skill to learn",
            "编程是一项值得学习的技能",
            "プログラミングは学ぶ価値のあるスキルだ",
        ],
        &[
            "Climate change is a global challenge",
            "气候变化是一个全球性挑战",
            "気候変動は地球規模の課題である",
        ],
        &[
            "Music can improve your mood",
            "音乐可以改善你的心情",
            "音楽は気分を良くしてくれる",
        ],
    ]
}

fn robustness_pairs() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "The quick brown fox jumps over the lazy dog",
            &[
                "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
                "the quick brown fox jumps over the lazy dog",
                "The quikc brown fox jumsp over the layz dog",
                "The lazy dog was jumped over by the quick brown fox",
                "The  quick  brown  fox  jumps  over  the  lazy  dog.",
            ],
        ),
        (
            "Machine learning is a subset of artificial intelligence",
            &[
                "MACHINE LEARNING IS A SUBSET OF ARTIFICIAL INTELLIGENCE",
                "machine learning is a subset of artificial intelligence",
                "Machin lerning is a subst of artifical inteligence",
                "Artificial intelligence includes machine learning as a subset",
                "Machine  learning  is  a  subset  of  artificial  intelligence!",
            ],
        ),
        (
            "She bought three red apples from the grocery store",
            &[
                "SHE BOUGHT THREE RED APPLES FROM THE GROCERY STORE",
                "she bought three red apples from the grocery store",
                "She bougt three red aples from the grocey store",
                "From the grocery store, three red apples were bought by her",
                "She bought  three  red  apples  from the  grocery  store.",
            ],
        ),
    ]
}

fn cluster_groups() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "technology",
            &[
                "Python is a popular programming language for data science",
                "The new GPU delivers 50% better performance than its predecessor",
                "Kubernetes orchestrates containerized applications at scale",
                "Version control with Git is essential for software development",
                "Cloud computing enables on-demand access to computing resources",
            ],
        ),
        (
            "cooking",
            &[
                "Preheat the oven to 350 degrees before baking the cake",
                "Fresh herbs add depth of flavor to any dish",
                "The secret to a good risotto is constant stirring and warm broth",
                "Sear the steak on high heat for a perfect crust",
                "Homemade pasta requires only flour, eggs, and a bit of salt",
            ],
        ),
        (
            "sports",
            &[
                "The marathon runner completed the race in under three hours",
                "Basketball requires both individual skill and team coordination",
                "Swimming is an excellent low-impact cardiovascular exercise",
                "The tennis match went to five sets before the champion prevailed",
                "Training for a triathlon involves swimming, cycling, and running",
            ],
        ),
        (
            "finance",
            &[
                "Diversifying your investment portfolio reduces overall risk",
                "The Federal Reserve raised interest rates by 25 basis points",
                "Compound interest is the most powerful force in wealth building",
                "ETFs offer broad market exposure with lower fees than mutual funds",
                "A strong balance sheet indicates financial stability and low debt",
            ],
        ),
    ]
}

fn collect_all_texts() -> Vec<String> {
    let mut all: Vec<String> = Vec::new();
    let mut add = |s: &str| {
        let owned = s.to_string();
        if !all.contains(&owned) {
            all.push(owned);
        }
    };
    for (a, b, _) in graded_similarity_pairs() {
        add(a);
        add(b);
    }
    for (q, cs, _) in retrieval_test_sets() {
        add(q);
        for c in cs {
            add(c);
        }
    }
    for group in crosslingual_groups() {
        for s in group {
            add(s);
        }
    }
    for (orig, variants) in robustness_pairs() {
        add(orig);
        for v in variants {
            add(v);
        }
    }
    for (_, texts) in cluster_groups() {
        for t in texts {
            add(t);
        }
    }
    all
}

// ═══════════════════════════════════════════
//  analysis
// ═══════════════════════════════════════════

fn analyze_model(name: &str, model_data: &ModelData) -> ModelResult {
    let lookup =
        |text: &str| -> &Vec<f32> { model_data.embeddings.get(text).expect("missing embedding") };

    let similarity_pairs = graded_similarity_pairs();
    let retrieval_sets = retrieval_test_sets();
    let crosslingual = crosslingual_groups();
    let robust_pairs = robustness_pairs();
    let clusters = cluster_groups();

    // [1] similarity
    let mut model_scores: Vec<f64> = Vec::new();
    let mut human_scores: Vec<f64> = Vec::new();
    let mut similar_cos: Vec<f64> = Vec::new();
    let mut dissimilar_cos: Vec<f64> = Vec::new();
    for (a, b, score) in &similarity_pairs {
        let cos = embedrs::cosine_similarity(lookup(a), lookup(b));
        model_scores.push(cos as f64);
        human_scores.push(*score as f64);
        if *score >= 3.5 {
            similar_cos.push(cos as f64);
        } else if *score <= 1.5 {
            dissimilar_cos.push(cos as f64);
        }
    }
    let rho = stats::spearman(&human_scores, &model_scores);
    let sim_avg = stats::mean(&similar_cos);
    let dis_avg = stats::mean(&dissimilar_cos);

    // [2] retrieval
    let mut top1 = 0;
    let mut mrr_sum = 0.0;
    let total = retrieval_sets.len();
    for (query, candidates, correct) in &retrieval_sets {
        let q = lookup(query);
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, embedrs::cosine_similarity(q, lookup(c))))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let rank = scored.iter().position(|(i, _)| *i == *correct).unwrap() + 1;
        if rank == 1 {
            top1 += 1;
        }
        mrr_sum += 1.0 / rank as f64;
    }

    // [3] multilingual
    let calc_rho = |pairs: Vec<&(&str, &str, f32)>| -> f64 {
        let h: Vec<f64> = pairs.iter().map(|(_, _, s)| *s as f64).collect();
        let m: Vec<f64> = pairs
            .iter()
            .map(|(a, b, _)| embedrs::cosine_similarity(lookup(a), lookup(b)) as f64)
            .collect();
        stats::spearman(&h, &m)
    };
    let en: Vec<_> = similarity_pairs
        .iter()
        .filter(|(a, _, _)| a.is_ascii())
        .collect();
    let zh: Vec<_> = similarity_pairs
        .iter()
        .filter(|(a, _, _)| {
            !a.is_ascii() && a.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
        })
        .collect();
    let ja: Vec<_> = similarity_pairs
        .iter()
        .filter(|(a, _, _)| {
            !a.is_ascii() && a.chars().any(|c| ('\u{3040}'..='\u{30ff}').contains(&c))
        })
        .collect();

    // [4] cross-lingual
    let mut cross: Vec<f64> = Vec::new();
    for group in &crosslingual {
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                cross.push(embedrs::cosine_similarity(lookup(group[i]), lookup(group[j])) as f64);
            }
        }
    }

    // [6] robustness
    let mut rob: Vec<f64> = Vec::new();
    for (orig, variants) in &robust_pairs {
        let o = lookup(orig);
        for v in *variants {
            rob.push(embedrs::cosine_similarity(o, lookup(v)) as f64);
        }
    }

    // [7] clustering
    let mut all_ct: Vec<(&str, &str)> = Vec::new();
    for (label, texts) in &clusters {
        for t in *texts {
            all_ct.push((t, label));
        }
    }
    let mut nn_hits = 0;
    for i in 0..all_ct.len() {
        let (ti, li) = all_ct[i];
        let ei = lookup(ti);
        let mut best = f64::NEG_INFINITY;
        let mut bl = "";
        for (j, (tj, lj)) in all_ct.iter().enumerate() {
            if i == j {
                continue;
            }
            let cos = embedrs::cosine_similarity(ei, lookup(tj)) as f64;
            if cos > best {
                best = cos;
                bl = lj;
            }
        }
        if bl == li {
            nn_hits += 1;
        }
    }
    let mut intra: Vec<f64> = Vec::new();
    let mut inter: Vec<f64> = Vec::new();
    for i in 0..all_ct.len() {
        for j in (i + 1)..all_ct.len() {
            let cos = embedrs::cosine_similarity(lookup(all_ct[i].0), lookup(all_ct[j].0)) as f64;
            if all_ct[i].1 == all_ct[j].1 {
                intra.push(cos);
            } else {
                inter.push(cos);
            }
        }
    }
    let intra_avg = stats::mean(&intra);
    let inter_avg = stats::mean(&inter);

    ModelResult {
        name: name.to_string(),
        dimensions: model_data.dimensions,
        similarity_rho: rho,
        discrimination_gap: sim_avg - dis_avg,
        similar_avg: sim_avg,
        dissimilar_avg: dis_avg,
        retrieval_top1: top1,
        retrieval_total: total,
        mrr: mrr_sum / total as f64,
        en_rho: calc_rho(en),
        zh_rho: calc_rho(zh),
        ja_rho: calc_rho(ja),
        crosslingual_avg: stats::mean(&cross),
        crosslingual_min: cross.iter().cloned().fold(f64::INFINITY, f64::min),
        robustness_avg: stats::mean(&rob),
        robustness_min: rob.iter().cloned().fold(f64::INFINITY, f64::min),
        cluster_purity: nn_hits,
        cluster_total: all_ct.len(),
        cluster_separation: intra_avg / inter_avg.max(0.001),
    }
}

fn load_prerecorded() -> Vec<ModelResult> {
    let compressed = std::fs::read(DATA_PATH).expect("data file not found");
    let mut output = Vec::new();
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(compressed.as_slice()).unwrap();
    std::io::Read::read_to_end(&mut decoder, &mut output).unwrap();
    let data: RecordedData = serde_json::from_slice(&output).unwrap();

    let order = [
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "bge-small-en-v1.5",
        "gte-small",
        "text-embedding-3-small",
        "gemini-embedding-001",
        "embed-v4.0",
        "voyage-3-large",
    ];

    order
        .iter()
        .filter_map(|name| data.models.get(*name).map(|d| analyze_model(name, d)))
        .collect()
}

// ═══════════════════════════════════════════
//  handlers
// ═══════════════════════════════════════════

async fn results_handler(State(state): State<Arc<AppState>>) -> Json<Vec<ModelResult>> {
    Json(state.prerecorded.clone())
}

async fn run_handler(
    Json(req): Json<RunRequest>,
) -> Sse<ReceiverStream<Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

    tokio::spawn(async move {
        let all_texts = collect_all_texts();

        let send = |msg: &str| {
            let tx = tx.clone();
            let msg = msg.to_string();
            async move {
                let _ = tx
                    .send(Ok(Event::default().event("progress").data(msg)))
                    .await;
            }
        };

        // local models
        let local_models = [
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "bge-small-en-v1.5",
            "gte-small",
        ];
        let mut results: Vec<ModelResult> = Vec::new();

        for model_name in &local_models {
            send(&format!("running {model_name}...")).await;
            match run_single_model_local(model_name, &all_texts).await {
                Ok(r) => {
                    send(&format!("{model_name} done")).await;
                    results.push(r);
                }
                Err(e) => {
                    send(&format!("{model_name} error: {e}")).await;
                }
            }
        }

        // cloud models
        let cloud_configs: Vec<(&str, &str, Option<&str>)> = vec![
            (
                "openai",
                "text-embedding-3-small",
                req.openai_key.as_deref(),
            ),
            ("gemini", "gemini-embedding-001", req.gemini_key.as_deref()),
            ("cohere", "embed-v4.0", req.cohere_key.as_deref()),
            ("voyage", "voyage-3-large", req.voyage_key.as_deref()),
        ];

        for (provider, label, key) in &cloud_configs {
            let Some(api_key) = key else {
                send(&format!("{label}: skipped (no key)")).await;
                continue;
            };
            send(&format!("running {label}...")).await;
            match run_single_model_cloud(provider, label, api_key, &all_texts).await {
                Ok(r) => {
                    send(&format!("{label} done")).await;
                    results.push(r);
                }
                Err(e) => {
                    send(&format!("{label} error: {e}")).await;
                }
            }
        }

        // send final results
        let json = serde_json::to_string(&results).unwrap();
        let _ = tx
            .send(Ok(Event::default().event("results").data(json)))
            .await;
        let _ = tx
            .send(Ok(Event::default().event("done").data("complete")))
            .await;
    });

    Sse::new(ReceiverStream::new(rx))
}

async fn run_single_model_local(
    model_name: &str,
    all_texts: &[String],
) -> embedrs::Result<ModelResult> {
    let client = embedrs::Client::local(model_name)?;
    let _ = client.embed(vec!["warmup".into()]).await?;
    let result = client.embed_batch(all_texts.to_vec()).await?;

    let mut embeddings = HashMap::new();
    for (text, emb) in all_texts.iter().zip(result.embeddings.iter()) {
        embeddings.insert(text.clone(), emb.clone());
    }

    let model_data = ModelData {
        dimensions: result.embeddings[0].len(),
        embeddings,
    };

    Ok(analyze_model(model_name, &model_data))
}

async fn run_single_model_cloud(
    provider: &str,
    label: &str,
    api_key: &str,
    all_texts: &[String],
) -> embedrs::Result<ModelResult> {
    let client = match provider {
        "openai" => embedrs::Client::openai(api_key),
        "gemini" => embedrs::Client::gemini(api_key),
        "cohere" => embedrs::Client::cohere(api_key),
        "voyage" => embedrs::Client::voyage(api_key),
        _ => unreachable!(),
    };

    let result = client.embed_batch(all_texts.to_vec()).await?;

    let mut embeddings = HashMap::new();
    for (text, emb) in all_texts.iter().zip(result.embeddings.iter()) {
        embeddings.insert(text.clone(), emb.clone());
    }

    let model_data = ModelData {
        dimensions: result.embeddings[0].len(),
        embeddings,
    };

    Ok(analyze_model(label, &model_data))
}

// ═══════════════════════════════════════════
//  main
// ═══════════════════════════════════════════

#[tokio::main]
async fn main() {
    println!("loading pre-recorded data...");
    let prerecorded = load_prerecorded();
    println!("loaded {} model results", prerecorded.len());

    let state = Arc::new(AppState { prerecorded });

    let index_file = format!("{WEB_DIST}/index.html");
    let spa_fallback = ServeDir::new(WEB_DIST).not_found_service(ServeFile::new(&index_file));

    let app = axum::Router::new()
        .route("/api/results", get(results_handler))
        .route("/api/run", post(run_handler))
        .with_state(state)
        .fallback_service(spa_fallback)
        .layer(tower_http::cors::CorsLayer::permissive());

    let addr = "0.0.0.0:3000";
    println!("server running at http://localhost:3000");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
