/// benchmark: rigorous comparison of local vs OpenAI vs Gemini
///
/// methodology:
///   1. graded similarity — 32 pairs with human scores (0-5), spearman correlation
///   2. retrieval accuracy — query→document matching, top-1/top-3 accuracy
///   3. multilingual — english, chinese, japanese breakdown
///   4. cross-lingual alignment — same sentence across languages
///   5. text length sensitivity — short/medium/long text quality
///   6. robustness — typos, casing, word order variations
///   7. clustering quality — topic grouping accuracy
///   8. throughput — 500 texts batch processing speed
///
/// run: cargo run -p embedrs --features local --example benchmark --release
use std::time::Instant;

#[tokio::main]
async fn main() -> embedrs::Result<()> {
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

    let openai_key = std::env::var("OPENAI_API_KEY").ok();
    let gemini_key = std::env::var("GEMINI_API_KEY").ok();
    let cohere_key = std::env::var("COHERE_API_KEY").ok();
    let voyage_key = std::env::var("VOYAGE_API_KEY").ok();

    println!("╔══════════════════════════════════════════════╗");
    println!("║       embedrs benchmark suite v1.0           ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // collect all texts for tests 1-7
    let similarity_pairs = graded_similarity_pairs();
    let retrieval_sets = retrieval_test_sets();
    let crosslingual_groups = crosslingual_groups();
    let length_pairs = length_sensitivity_pairs();
    let robustness_pairs = robustness_pairs();
    let cluster_groups = cluster_groups();

    let mut all_texts: Vec<String> = Vec::new();
    let mut add = |s: &str| {
        let owned = s.to_string();
        if !all_texts.contains(&owned) {
            all_texts.push(owned);
        }
    };

    for (a, b, _) in &similarity_pairs {
        add(a);
        add(b);
    }
    for (query, candidates, _) in &retrieval_sets {
        add(query);
        for c in *candidates {
            add(c);
        }
    }
    for group in &crosslingual_groups {
        for s in *group {
            add(s);
        }
    }
    for (a, b, _) in &length_pairs {
        add(a);
        add(b);
    }
    for (orig, variants) in &robustness_pairs {
        add(orig);
        for v in *variants {
            add(v);
        }
    }
    for (_, texts) in &cluster_groups {
        for t in *texts {
            add(t);
        }
    }

    println!("corpus: {} unique texts (tests 1-7)", all_texts.len());
    println!(
        "  {} similarity pairs, {} retrieval queries, {} crosslingual groups",
        similarity_pairs.len(),
        retrieval_sets.len(),
        crosslingual_groups.len()
    );
    println!(
        "  {} length pairs, {} robustness sets, {} clusters\n",
        length_pairs.len(),
        robustness_pairs.len(),
        cluster_groups.len()
    );

    // --- run each provider ---
    // local models
    let local_models = [
        (
            "all-MiniLM-L6-v2",
            "LOCAL: all-MiniLM-L6-v2 (6L, 23MB, 384-dim)",
        ),
        (
            "all-MiniLM-L12-v2",
            "LOCAL: all-MiniLM-L12-v2 (12L, 133MB, 384-dim)",
        ),
        (
            "bge-small-en-v1.5",
            "LOCAL: bge-small-en-v1.5 (12L, 133MB, 384-dim)",
        ),
        ("gte-small", "LOCAL: gte-small (12L, 67MB, 384-dim)"),
    ];

    for (model_name, label) in &local_models {
        println!("━━━ {label} ━━━");
        if let Err(e) = run_provider(
            &format!("local:{model_name}"),
            None,
            &all_texts,
            &similarity_pairs,
            &retrieval_sets,
            &crosslingual_groups,
            &length_pairs,
            &robustness_pairs,
            &cluster_groups,
        )
        .await
        {
            println!("  error: {e}\n");
        }
    }

    // cloud providers
    let cloud_providers: Vec<(&str, &str, Option<&str>)> = {
        let mut v = Vec::new();
        if let Some(k) = openai_key.as_deref() {
            v.push((
                "openai",
                "OPENAI: text-embedding-3-small (1536-dim)",
                Some(k),
            ));
        }
        if let Some(k) = gemini_key.as_deref() {
            v.push(("gemini", "GEMINI: gemini-embedding-001 (3072-dim)", Some(k)));
        }
        if let Some(k) = cohere_key.as_deref() {
            v.push(("cohere", "COHERE: embed-v4.0 (1024-dim)", Some(k)));
        }
        if let Some(k) = voyage_key.as_deref() {
            v.push(("voyage", "VOYAGE: voyage-3-large (1024-dim)", Some(k)));
        }
        v
    };

    for (name, label, key) in &cloud_providers {
        println!("━━━ {label} ━━━");
        if let Err(e) = run_provider(
            name,
            *key,
            &all_texts,
            &similarity_pairs,
            &retrieval_sets,
            &crosslingual_groups,
            &length_pairs,
            &robustness_pairs,
            &cluster_groups,
        )
        .await
        {
            println!("  error: {e}\n");
        }
    }

    if cloud_providers.is_empty() {
        println!("━━━ CLOUD: skipped (no API keys found) ━━━\n");
    }

    println!("━━━ COST SUMMARY ━━━");
    println!("  local:   $0 / unlimited (CPU inference)");
    println!("  openai:  $0.02 / 1M tokens (text-embedding-3-small)");
    println!("  gemini:  free tier 1500 RPM, then usage-based");
    println!("  cohere:  $0.10 / 1M tokens (embed-v4.0)");
    println!("  voyage:  $0.06 / 1M tokens (voyage-3-large)\n");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_provider(
    name: &str,
    api_key: Option<&str>,
    all_texts: &[String],
    similarity_pairs: &[(&str, &str, f32)],
    retrieval_sets: &[(&str, &[&str], usize)],
    crosslingual_groups: &[&[&str]],
    length_pairs: &[(&str, &str, f32)],
    robustness_pairs: &[(&str, &[&str])],
    cluster_groups: &[(&str, &[&str])],
) -> embedrs::Result<()> {
    let client = if let Some(model_name) = name.strip_prefix("local:") {
        embedrs::Client::local(model_name)?
    } else {
        match name {
            "openai" => embedrs::Client::openai(api_key.unwrap()),
            "gemini" => embedrs::Client::gemini(api_key.unwrap()),
            "cohere" => embedrs::Client::cohere(api_key.unwrap()),
            "voyage" => embedrs::Client::voyage(api_key.unwrap()),
            _ => unreachable!(),
        }
    };

    // warmup for local (includes model download + load)
    if name.starts_with("local:") {
        let t = Instant::now();
        let _ = client.embed(vec!["warmup".into()]).await?;
        println!("  model load: {}ms", t.elapsed().as_millis());
    }

    // embed all texts
    let start = Instant::now();
    let result = client.embed_batch(all_texts.to_vec()).await?;
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0;

    println!("  dimensions: {}", result.embeddings[0].len());
    println!(
        "  latency: {ms:.0}ms total, {:.1}ms/text ({} texts)",
        ms / all_texts.len() as f64,
        all_texts.len()
    );
    if result.usage.total_tokens > 0 {
        println!("  tokens: {}", result.usage.total_tokens);
    }

    let lookup = |text: &str| -> &Vec<f32> {
        let idx = all_texts
            .iter()
            .position(|t| t == text)
            .unwrap_or_else(|| panic!("text not found in corpus: {}", &text[..text.len().min(50)]));
        &result.embeddings[idx]
    };

    // ── test 1: graded similarity ──
    println!("\n  [1] GRADED SIMILARITY (spearman ρ)");
    let mut model_scores: Vec<f64> = Vec::new();
    let mut human_scores: Vec<f64> = Vec::new();
    let mut similar_cosines: Vec<f64> = Vec::new();
    let mut dissimilar_cosines: Vec<f64> = Vec::new();

    for (a, b, human_score) in similarity_pairs {
        let cos = embedrs::cosine_similarity(lookup(a), lookup(b));
        model_scores.push(cos as f64);
        human_scores.push(*human_score as f64);
        if *human_score >= 3.5 {
            similar_cosines.push(cos as f64);
        } else if *human_score <= 1.5 {
            dissimilar_cosines.push(cos as f64);
        }
    }

    let rho = spearman_correlation(&human_scores, &model_scores);
    let sim_avg = mean(&similar_cosines);
    let dis_avg = mean(&dissimilar_cosines);
    let gap = sim_avg - dis_avg;
    println!("      ρ = {rho:.4}");
    println!(
        "      similar avg cosine   = {sim_avg:.4} ({} pairs)",
        similar_cosines.len()
    );
    println!(
        "      dissimilar avg cosine = {dis_avg:.4} ({} pairs)",
        dissimilar_cosines.len()
    );
    println!("      discrimination gap    = {gap:.4}");

    // ── test 2: retrieval accuracy ──
    println!("\n  [2] RETRIEVAL ACCURACY");
    let mut top1_hits = 0;
    let mut top3_hits = 0;
    let mut mrr_sum = 0.0;
    let total = retrieval_sets.len();

    for (query, candidates, correct_idx) in retrieval_sets {
        let q_emb = lookup(query);
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, embedrs::cosine_similarity(q_emb, lookup(c))))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let rank = scored.iter().position(|(i, _)| *i == *correct_idx).unwrap() + 1;
        mrr_sum += 1.0 / rank as f64;

        if rank == 1 {
            top1_hits += 1;
        }
        if rank <= 3 {
            top3_hits += 1;
        }
    }

    let mrr = mrr_sum / total as f64;
    println!(
        "      top-1:  {top1_hits}/{total} ({:.0}%)",
        pct(top1_hits, total)
    );
    println!(
        "      top-3:  {top3_hits}/{total} ({:.0}%)",
        pct(top3_hits, total)
    );
    println!("      MRR:    {mrr:.4}");

    // ── test 3: multilingual breakdown ──
    println!("\n  [3] MULTILINGUAL BREAKDOWN");
    let en_pairs: Vec<_> = similarity_pairs
        .iter()
        .filter(|(a, _, _)| a.is_ascii())
        .collect();
    let zh_pairs: Vec<_> = similarity_pairs
        .iter()
        .filter(|(a, _, _)| {
            !a.is_ascii() && a.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
        })
        .collect();
    let ja_pairs: Vec<_> = similarity_pairs
        .iter()
        .filter(|(a, _, _)| {
            !a.is_ascii() && a.chars().any(|c| ('\u{3040}'..='\u{30ff}').contains(&c))
        })
        .collect();

    for (label, pairs) in [
        ("english", &en_pairs),
        ("chinese", &zh_pairs),
        ("japanese", &ja_pairs),
    ] {
        if pairs.is_empty() {
            continue;
        }
        let h: Vec<f64> = pairs.iter().map(|(_, _, s)| *s as f64).collect();
        let m: Vec<f64> = pairs
            .iter()
            .map(|(a, b, _)| embedrs::cosine_similarity(lookup(a), lookup(b)) as f64)
            .collect();
        let r = spearman_correlation(&h, &m);
        println!("      {label:8} ρ = {r:.4} ({} pairs)", pairs.len());
    }

    // ── test 4: cross-lingual alignment ──
    println!("\n  [4] CROSS-LINGUAL ALIGNMENT");
    let mut cross_sims: Vec<f64> = Vec::new();
    for group in crosslingual_groups {
        // all pairs within group should be highly similar
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let cos = embedrs::cosine_similarity(lookup(group[i]), lookup(group[j]));
                cross_sims.push(cos as f64);
            }
        }
    }
    let cross_avg = mean(&cross_sims);
    let cross_min = cross_sims.iter().cloned().fold(f64::INFINITY, f64::min);
    let cross_max = cross_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "      avg cosine = {cross_avg:.4} (min={cross_min:.4}, max={cross_max:.4}, {} pairs)",
        cross_sims.len()
    );
    println!("      ideal: avg → 1.0 (same meaning across en/zh/ja)");

    // ── test 5: text length sensitivity ──
    println!("\n  [5] TEXT LENGTH SENSITIVITY");
    let short_pairs: Vec<_> = length_pairs
        .iter()
        .filter(|(a, _, _)| a.len() < 50)
        .collect();
    let medium_pairs: Vec<_> = length_pairs
        .iter()
        .filter(|(a, _, _)| a.len() >= 50 && a.len() < 200)
        .collect();
    let long_pairs: Vec<_> = length_pairs
        .iter()
        .filter(|(a, _, _)| a.len() >= 200)
        .collect();

    for (label, pairs) in [
        ("short", &short_pairs),
        ("medium", &medium_pairs),
        ("long", &long_pairs),
    ] {
        if pairs.is_empty() {
            continue;
        }
        let h: Vec<f64> = pairs.iter().map(|(_, _, s)| *s as f64).collect();
        let m: Vec<f64> = pairs
            .iter()
            .map(|(a, b, _)| embedrs::cosine_similarity(lookup(a), lookup(b)) as f64)
            .collect();
        let r = if h.len() >= 3 {
            spearman_correlation(&h, &m)
        } else {
            // too few pairs for meaningful spearman, use pearson
            pearson_correlation(&h, &m)
        };
        let avg_cos = mean(&m);
        println!(
            "      {label:6} ρ = {r:.4}, avg cosine = {avg_cos:.4} ({} pairs)",
            pairs.len()
        );
    }

    // ── test 6: robustness ──
    println!("\n  [6] ROBUSTNESS (typos, casing, word order)");
    let mut robust_scores: Vec<f64> = Vec::new();
    for (original, variants) in robustness_pairs {
        let orig_emb = lookup(original);
        for variant in *variants {
            let cos = embedrs::cosine_similarity(orig_emb, lookup(variant));
            robust_scores.push(cos as f64);
        }
    }
    let robust_avg = mean(&robust_scores);
    let robust_min = robust_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "      avg cosine = {robust_avg:.4} (min={robust_min:.4}, {} variants)",
        robust_scores.len()
    );
    println!("      ideal: avg → 1.0 (meaning unchanged despite surface variation)");

    // ── test 7: clustering quality ──
    println!("\n  [7] CLUSTERING QUALITY");
    // for each text, find its nearest neighbor; if same cluster → hit
    let mut all_cluster_texts: Vec<(&str, &str)> = Vec::new(); // (text, label)
    for (label, texts) in cluster_groups {
        for t in *texts {
            all_cluster_texts.push((t, label));
        }
    }

    let mut nn_hits = 0;
    let total_cluster = all_cluster_texts.len();
    for i in 0..total_cluster {
        let (text_i, label_i) = all_cluster_texts[i];
        let emb_i = lookup(text_i);
        let mut best_sim = f64::NEG_INFINITY;
        let mut best_label = "";
        for j in 0..total_cluster {
            if i == j {
                continue;
            }
            let (text_j, label_j) = all_cluster_texts[j];
            let cos = embedrs::cosine_similarity(emb_i, lookup(text_j)) as f64;
            if cos > best_sim {
                best_sim = cos;
                best_label = label_j;
            }
        }
        if best_label == label_i {
            nn_hits += 1;
        }
    }
    let nn_purity = pct(nn_hits, total_cluster);
    println!("      nearest-neighbor purity = {nn_hits}/{total_cluster} ({nn_purity:.0}%)");

    // also compute inter/intra cluster ratio
    let mut intra_sims: Vec<f64> = Vec::new();
    let mut inter_sims: Vec<f64> = Vec::new();
    for i in 0..total_cluster {
        for j in (i + 1)..total_cluster {
            let cos = embedrs::cosine_similarity(
                lookup(all_cluster_texts[i].0),
                lookup(all_cluster_texts[j].0),
            ) as f64;
            if all_cluster_texts[i].1 == all_cluster_texts[j].1 {
                intra_sims.push(cos);
            } else {
                inter_sims.push(cos);
            }
        }
    }
    let intra_avg = mean(&intra_sims);
    let inter_avg = mean(&inter_sims);
    println!("      intra-cluster avg = {intra_avg:.4} (same topic)");
    println!("      inter-cluster avg = {inter_avg:.4} (different topics)");
    println!(
        "      separation ratio  = {:.2}x",
        intra_avg / inter_avg.max(0.001)
    );

    // ── test 8: throughput ──
    println!("\n  [8] THROUGHPUT (500 texts)");
    let throughput_texts: Vec<String> = (0..500)
        .map(|i| {
            format!(
                "This is benchmark sentence number {} for measuring embedding throughput performance across different providers and models",
                i
            )
        })
        .collect();

    let t = Instant::now();
    let tp_result = client.embed_batch(throughput_texts.clone()).await?;
    let tp_elapsed = t.elapsed();
    let tp_ms = tp_elapsed.as_secs_f64() * 1000.0;
    let texts_per_sec = 500.0 / tp_elapsed.as_secs_f64();
    println!("      500 texts in {tp_ms:.0}ms ({texts_per_sec:.0} texts/sec)");
    println!("      {:.1}ms/text", tp_ms / 500.0);
    if tp_result.usage.total_tokens > 0 {
        println!("      tokens: {}", tp_result.usage.total_tokens);
    }

    println!();
    Ok(())
}

// ═══════════════════════════════════════════
//  test data
// ═══════════════════════════════════════════

/// graded similarity pairs: (text_a, text_b, human_score 0.0-5.0)
fn graded_similarity_pairs() -> Vec<(&'static str, &'static str, f32)> {
    vec![
        // === english: high similarity (4.0-5.0) ===
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
        // === english: medium similarity (2.0-3.5) ===
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
        // === english: low similarity (0.0-1.5) ===
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
        // === chinese ===
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
        // === japanese ===
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

/// retrieval test: (query, candidates, index_of_correct_answer)
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

/// cross-lingual: same sentence in en/zh/ja — embeddings should be close
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

/// text length sensitivity: short (<50 chars), medium (50-200), long (200+)
fn length_sensitivity_pairs() -> Vec<(&'static str, &'static str, f32)> {
    vec![
        // short pairs (<50 chars)
        ("dogs are loyal", "dogs are faithful animals", 4.5),
        ("it is raining", "the weather is wet", 3.8),
        ("she smiled", "he frowned", 1.5),
        ("fast car", "quick automobile", 4.7),
        // medium pairs (50-200 chars)
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
        // long pairs (200+ chars)
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

/// robustness: original → variations (typos, casing, word order)
fn robustness_pairs() -> Vec<(&'static str, &'static [&'static str])> {
    vec![
        (
            "The quick brown fox jumps over the lazy dog",
            &[
                // all uppercase
                "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
                // all lowercase
                "the quick brown fox jumps over the lazy dog",
                // typos
                "The quikc brown fox jumsp over the layz dog",
                // word order change (passive)
                "The lazy dog was jumped over by the quick brown fox",
                // extra whitespace / punctuation
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

/// cluster groups: texts grouped by topic for clustering quality test
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

// ═══════════════════════════════════════════
//  math utilities
// ═══════════════════════════════════════════

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn pct(num: usize, den: usize) -> f64 {
    num as f64 / den as f64 * 100.0
}

fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let rank_x = ranks(x);
    let rank_y = ranks(y);
    let d_sq_sum: f64 = rank_x
        .iter()
        .zip(rank_y.iter())
        .map(|(rx, ry)| (rx - ry).powi(2))
        .sum();
    1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1.0))
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let mean_x = mean(x);
    let mean_y = mean(y);
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let den = (den_x * den_y).sqrt();
    if den < 1e-12 {
        return 0.0;
    }
    num / den
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut result = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1 && (indexed[j + 1].1 - indexed[j].1).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            result[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    result
}
