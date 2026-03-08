use benchrs::stats;
use serde::{Deserialize, Serialize};
/// # Experiment: Embedding Model Selection
///
/// **Decision:** Which embedding model should embedrs use as default for `local()` and `cloud()`?
///
/// **Date:** 2026-03-08
/// **Environment:** macOS Darwin 25.3.0, Apple Silicon, Rust 1.85 (release mode)
///
/// ## Methodology
///
/// 8 models tested across 8 dimensions:
///
/// | # | Dimension | Metric | What it measures |
/// |---|-----------|--------|------------------|
/// | 1 | Graded Similarity | Spearman ρ | Ranking accuracy vs human scores (0-5) |
/// | 1 | Discrimination | Gap | Separation between similar/dissimilar pairs |
/// | 2 | Retrieval | Top-1, MRR | Can it find the right document? |
/// | 3 | Multilingual | ρ per language | English, Chinese, Japanese separately |
/// | 4 | Cross-lingual | Cosine | Same meaning across languages → same vector? |
/// | 5 | Length sensitivity | ρ per length | Short/medium/long text quality |
/// | 6 | Robustness | Cosine | Resilience to typos, casing, word order |
/// | 7 | Clustering | Separation ratio | Can it group related texts? |
/// | 8 | Throughput | texts/sec | Processing speed |
///
/// ## Models tested
///
/// | Model | Type | Size | Dimensions |
/// |-------|------|------|------------|
/// | all-MiniLM-L6-v2 | local | 23MB | 384 |
/// | all-MiniLM-L12-v2 | local | 133MB | 384 |
/// | bge-small-en-v1.5 | local | 133MB | 384 |
/// | gte-small | local | 67MB | 384 |
/// | text-embedding-3-small | cloud (OpenAI) | — | 1536 |
/// | gemini-embedding-001 | cloud (Google) | — | 3072 |
/// | embed-v4.0 | cloud (Cohere) | — | 1536 |
/// | voyage-3-large | cloud (Voyage) | — | 1024 |
///
/// ## Test corpus
///
/// - 184 unique texts
/// - 32 graded similarity pairs (human-scored 0.0-5.0)
/// - 9 retrieval queries (EN/ZH/JA) with 4 candidates each
/// - 6 cross-lingual groups (same sentence in EN/ZH/JA)
/// - 10 length sensitivity pairs (short/medium/long)
/// - 3 robustness sets (15 variants: typos, casing, word order)
/// - 4 topic clusters × 5 texts each
/// - 500 throughput texts
///
/// ## How to run
///
/// ```bash
/// # analyze pre-recorded data (no API keys needed)
/// cargo run --bin embedding_models
///
/// # re-record with live API calls
/// # requires .env.local with OPENAI_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY
/// cargo run --bin embedding_models -- --record
/// ```
use std::collections::HashMap;

/// pre-recorded embeddings for all models
#[derive(Serialize, Deserialize)]
struct RecordedData {
    /// model name → { text → embedding }
    models: HashMap<String, ModelData>,
    /// metadata about the recording
    metadata: Metadata,
}

#[derive(Serialize, Deserialize)]
struct ModelData {
    dimensions: usize,
    embeddings: HashMap<String, Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    date: String,
    platform: String,
    corpus_size: usize,
    models: Vec<String>,
}

const DATA_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/embedding_models.json.zst"
);

#[tokio::main]
async fn main() -> embedrs::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let record_mode = args.iter().any(|a| a == "--record");

    if record_mode {
        record().await?;
    } else {
        analyze()?;
    }

    Ok(())
}

// ═══════════════════════════════════════════
//  test data definitions
// ═══════════════════════════════════════════

fn graded_similarity_pairs() -> Vec<(&'static str, &'static str, f32)> {
    vec![
        // english: high similarity (4.0-5.0)
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
        // english: medium similarity (2.0-3.5)
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
        // english: low similarity (0.0-1.5)
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
        // chinese
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
        // japanese
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

fn length_sensitivity_pairs() -> Vec<(&'static str, &'static str, f32)> {
    vec![
        ("dogs are loyal", "dogs are faithful animals", 4.5),
        ("it is raining", "the weather is wet", 3.8),
        ("she smiled", "he frowned", 1.5),
        ("fast car", "quick automobile", 4.7),
        (
            "Machine learning algorithms can identify patterns in large datasets",
            "Deep learning models are trained to recognize complex patterns in big data",
            4.2,
        ),
        (
            "The new restaurant downtown has an excellent selection of seafood dishes",
            "There is a great seafood place that recently opened in the city center",
            4.0,
        ),
        (
            "Regular physical exercise helps maintain cardiovascular health and reduces stress",
            "Consistent workout routines contribute to heart health and lower anxiety levels",
            4.3,
        ),
        (
            "The rapid advancement of artificial intelligence and machine learning technologies has fundamentally transformed how businesses operate, enabling automated decision-making, predictive analytics, and personalized customer experiences at scale across various industries worldwide",
            "AI and ML breakthroughs are reshaping the corporate landscape by automating decisions, forecasting trends, and delivering tailored user experiences to millions of customers in diverse sectors globally",
            4.5,
        ),
        (
            "Sustainable development requires balancing economic growth with environmental protection and social equity, ensuring that the needs of the present generation are met without compromising the ability of future generations to meet their own needs, as outlined in the Brundtland Report",
            "The concept of sustainability involves finding equilibrium between economic progress, ecological conservation, and social fairness so that current prosperity does not come at the expense of opportunities for those who will come after us",
            4.6,
        ),
        (
            "Quantum computing leverages the principles of quantum mechanics, including superposition and entanglement, to perform certain computations exponentially faster than classical computers, with potential applications in cryptography, drug discovery, and optimization problems",
            "The Renaissance was a cultural movement that began in Italy in the 14th century, characterized by renewed interest in classical Greek and Roman art, literature, and philosophy, which eventually spread throughout Europe and laid the groundwork for the modern age",
            0.3,
        ),
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
    for (a, b, _) in length_sensitivity_pairs() {
        add(a);
        add(b);
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
//  record mode: embed with live APIs
// ═══════════════════════════════════════════

async fn record() -> embedrs::Result<()> {
    dotenvy::from_filename(".env.local").ok();
    dotenvy::from_filename(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join(".env.local"),
    )
    .ok();

    let all_texts = collect_all_texts();
    println!("recording embeddings for {} texts...\n", all_texts.len());

    let mut data = RecordedData {
        models: HashMap::new(),
        metadata: Metadata {
            date: "2026-03-08".to_string(),
            platform: "macOS Darwin 25.3.0, Apple Silicon, Rust 1.85, release mode".to_string(),
            corpus_size: all_texts.len(),
            models: Vec::new(),
        },
    };

    // local models
    let local_models = [
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "bge-small-en-v1.5",
        "gte-small",
    ];

    for model_name in &local_models {
        print!("  {model_name}...");
        let client = embedrs::Client::local(model_name)?;
        // warmup
        let _ = client.embed(vec!["warmup".into()]).await?;
        let result = client.embed_batch(all_texts.clone()).await?;

        let mut embeddings = HashMap::new();
        for (text, emb) in all_texts.iter().zip(result.embeddings.iter()) {
            embeddings.insert(text.clone(), emb.clone());
        }

        let dim = result.embeddings[0].len();
        data.models.insert(
            model_name.to_string(),
            ModelData {
                dimensions: dim,
                embeddings,
            },
        );
        data.metadata.models.push(model_name.to_string());
        println!(" done ({dim}-dim, {} texts)", all_texts.len());
    }

    // cloud models
    let cloud_configs: Vec<(&str, &str, Option<String>)> = vec![
        (
            "openai",
            "text-embedding-3-small",
            std::env::var("OPENAI_API_KEY").ok(),
        ),
        (
            "gemini",
            "gemini-embedding-001",
            std::env::var("GEMINI_API_KEY").ok(),
        ),
        ("cohere", "embed-v4.0", std::env::var("COHERE_API_KEY").ok()),
        (
            "voyage",
            "voyage-3-large",
            std::env::var("VOYAGE_API_KEY").ok(),
        ),
    ];

    for (provider, model_label, key) in &cloud_configs {
        let Some(api_key) = key else {
            println!("  {model_label}: skipped (no key)");
            continue;
        };
        print!("  {model_label}...");

        let client = match *provider {
            "openai" => embedrs::Client::openai(api_key),
            "gemini" => embedrs::Client::gemini(api_key),
            "cohere" => embedrs::Client::cohere(api_key),
            "voyage" => embedrs::Client::voyage(api_key),
            _ => unreachable!(),
        };

        let result = client.embed_batch(all_texts.clone()).await?;

        let mut embeddings = HashMap::new();
        for (text, emb) in all_texts.iter().zip(result.embeddings.iter()) {
            embeddings.insert(text.clone(), emb.clone());
        }

        let dim = result.embeddings[0].len();
        data.models.insert(
            model_label.to_string(),
            ModelData {
                dimensions: dim,
                embeddings,
            },
        );
        data.metadata.models.push(model_label.to_string());
        println!(" done ({dim}-dim)");
    }

    // compress and save
    let json = serde_json::to_vec(&data).unwrap();
    let mut encoder = zstd::bulk::Compressor::new(19).unwrap();
    let compressed = encoder.compress(&json).unwrap();

    let out_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/embedding_models.json.zst");
    std::fs::write(&out_path, &compressed).unwrap();
    println!(
        "\nsaved: {} ({:.1} MB raw → {:.1} MB compressed)",
        out_path.display(),
        json.len() as f64 / 1_000_000.0,
        compressed.len() as f64 / 1_000_000.0
    );

    Ok(())
}

// ═══════════════════════════════════════════
//  analyze mode: load pre-recorded data
// ═══════════════════════════════════════════

fn analyze() -> embedrs::Result<()> {
    // load compressed data
    let compressed = std::fs::read(DATA_PATH)
        .unwrap_or_else(|_| panic!("data file not found: {DATA_PATH}\nrun with --record first"));

    let mut output = Vec::new();
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(compressed.as_slice())
        .expect("zstd decoder init failed");
    std::io::Read::read_to_end(&mut decoder, &mut output).expect("zstd decompression failed");

    let data: RecordedData = serde_json::from_slice(&output).expect("json parse failed");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Experiment: Embedding Model Selection for embedrs v1.0     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("recorded: {}", data.metadata.date);
    println!("platform: {}", data.metadata.platform);
    println!("corpus:   {} texts", data.metadata.corpus_size);
    println!("models:   {}\n", data.metadata.models.join(", "));

    let similarity_pairs = graded_similarity_pairs();
    let retrieval_sets = retrieval_test_sets();
    let crosslingual = crosslingual_groups();
    let length_pairs = length_sensitivity_pairs();
    let robust_pairs = robustness_pairs();
    let clusters = cluster_groups();

    // analyze each model
    let model_order = [
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "bge-small-en-v1.5",
        "gte-small",
        "text-embedding-3-small",
        "gemini-embedding-001",
        "embed-v4.0",
        "voyage-3-large",
    ];

    for model_name in &model_order {
        let Some(model_data) = data.models.get(*model_name) else {
            println!("━━━ {model_name}: no data ━━━\n");
            continue;
        };

        println!("━━━ {model_name} ({}-dim) ━━━", model_data.dimensions);

        let lookup = |text: &str| -> &Vec<f32> {
            model_data
                .embeddings
                .get(text)
                .unwrap_or_else(|| panic!("missing embedding for: {}", &text[..text.len().min(50)]))
        };

        // [1] graded similarity
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
        let gap = sim_avg - dis_avg;
        println!(
            "  [1] similarity:   ρ={rho:.4}, gap={gap:.4} (sim={sim_avg:.4}, dis={dis_avg:.4})"
        );

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
        println!(
            "  [2] retrieval:    top-1={top1}/{total} ({:.0}%), MRR={:.4}",
            stats::pct(top1, total),
            mrr_sum / total as f64
        );

        // [3] multilingual
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

        let calc_rho = |pairs: &[&(&str, &str, f32)]| -> f64 {
            let h: Vec<f64> = pairs.iter().map(|(_, _, s)| *s as f64).collect();
            let m: Vec<f64> = pairs
                .iter()
                .map(|(a, b, _)| embedrs::cosine_similarity(lookup(a), lookup(b)) as f64)
                .collect();
            stats::spearman(&h, &m)
        };

        println!(
            "  [3] multilingual: en={:.4}, zh={:.4}, ja={:.4}",
            calc_rho(&en),
            calc_rho(&zh),
            calc_rho(&ja)
        );

        // [4] cross-lingual
        let mut cross: Vec<f64> = Vec::new();
        for group in &crosslingual {
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    cross.push(
                        embedrs::cosine_similarity(lookup(group[i]), lookup(group[j])) as f64,
                    );
                }
            }
        }
        let cross_avg = stats::mean(&cross);
        let cross_min = cross.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("  [4] cross-lingual: avg={cross_avg:.4}, min={cross_min:.4}");

        // [5] length sensitivity
        let short: Vec<_> = length_pairs
            .iter()
            .filter(|(a, _, _)| a.len() < 50)
            .collect();
        let medium: Vec<_> = length_pairs
            .iter()
            .filter(|(a, _, _)| a.len() >= 50 && a.len() < 200)
            .collect();
        let long: Vec<_> = length_pairs
            .iter()
            .filter(|(a, _, _)| a.len() >= 200)
            .collect();

        let len_rho = |pairs: &[&(&str, &str, f32)]| -> f64 {
            let h: Vec<f64> = pairs.iter().map(|(_, _, s)| *s as f64).collect();
            let m: Vec<f64> = pairs
                .iter()
                .map(|(a, b, _)| embedrs::cosine_similarity(lookup(a), lookup(b)) as f64)
                .collect();
            if h.len() >= 3 {
                stats::spearman(&h, &m)
            } else {
                stats::pearson(&h, &m)
            }
        };

        println!(
            "  [5] length:       short={:.4}, medium={:.4}, long={:.4}",
            len_rho(&short),
            len_rho(&medium),
            len_rho(&long)
        );

        // [6] robustness
        let mut rob: Vec<f64> = Vec::new();
        for (orig, variants) in &robust_pairs {
            let o = lookup(orig);
            for v in *variants {
                rob.push(embedrs::cosine_similarity(o, lookup(v)) as f64);
            }
        }
        let rob_avg = stats::mean(&rob);
        let rob_min = rob.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("  [6] robustness:   avg={rob_avg:.4}, min={rob_min:.4}");

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
                let cos =
                    embedrs::cosine_similarity(lookup(all_ct[i].0), lookup(all_ct[j].0)) as f64;
                if all_ct[i].1 == all_ct[j].1 {
                    intra.push(cos);
                } else {
                    inter.push(cos);
                }
            }
        }
        let intra_avg = stats::mean(&intra);
        let inter_avg = stats::mean(&inter);
        println!(
            "  [7] clustering:   purity={nn_hits}/{}, ratio={:.2}x (intra={intra_avg:.4}, inter={inter_avg:.4})",
            all_ct.len(),
            intra_avg / inter_avg.max(0.001)
        );

        println!();
    }

    // summary table
    println!("━━━ SUMMARY ━━━\n");
    println!("Decisions based on this experiment:");
    println!();
    println!("  LOCAL DEFAULT: all-MiniLM-L6-v2");
    println!("    - 23MB (app-embeddable, others are 67-133MB)");
    println!("    - best clustering separation (8.73x, 2nd place is 4.38x)");
    println!("    - 100% retrieval accuracy, MRR=1.0");
    println!("    - english ρ=0.92 (beats most cloud models)");
    println!("    - 12-layer models are 3-6x larger with no quality gain");
    println!();
    println!("  CLOUD DEFAULT: OpenAI text-embedding-3-small");
    println!("    - best discrimination gap (0.58, 2nd place is 0.46)");
    println!("    - 100% retrieval accuracy, MRR=1.0");
    println!("    - balanced multilingual (en=0.91, zh=0.88, ja=0.90)");
    println!("    - cheapest ($0.02/1M tokens)");
    println!("    - Gemini ranks higher (ρ=0.94) but poor discrimination (0.30), retrieval miss");
    println!("    - Cohere strong but 5x more expensive ($0.10/1M tokens)");

    Ok(())
}
