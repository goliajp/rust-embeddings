//! The evaluation corpus: 184 texts in English, Chinese and Japanese with
//! human similarity scores, retrieval queries, translations, typo variants and
//! topic clusters. The recorded embeddings in `data.json.zst` are keyed by
//! these exact strings, so editing a text means re-recording.

pub fn graded_similarity_pairs() -> Vec<(&'static str, &'static str, f32)> {
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

pub fn retrieval_test_sets() -> Vec<(&'static str, &'static [&'static str], usize)> {
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

pub fn crosslingual_groups() -> Vec<&'static [&'static str]> {
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

pub fn length_sensitivity_pairs() -> Vec<(&'static str, &'static str, f32)> {
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

pub fn robustness_pairs() -> Vec<(&'static str, &'static [&'static str])> {
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

pub fn cluster_groups() -> Vec<(&'static str, &'static [&'static str])> {
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

pub fn collect_all_texts() -> Vec<String> {
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
