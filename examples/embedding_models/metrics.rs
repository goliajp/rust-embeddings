//! The seven quality measures, computed from any text -> vector lookup.

use crate::corpus;
use crate::stats;

type Pair = (&'static str, &'static str, f32);

fn cos(a: &[f32], b: &[f32]) -> f64 {
    embedrs::cosine_similarity(a, b) as f64
}

/// Score one model and print one line per measure.
pub fn report<'a>(lookup: &dyn Fn(&str) -> &'a [f32]) {
    similarity(lookup);
    retrieval(lookup);
    multilingual(lookup);
    crosslingual(lookup);
    length(lookup);
    robustness(lookup);
    clustering(lookup);
}

fn similarity<'a>(lookup: &dyn Fn(&str) -> &'a [f32]) {
    let mut model = Vec::new();
    let mut human = Vec::new();
    let mut similar = Vec::new();
    let mut dissimilar = Vec::new();
    for (a, b, score) in corpus::graded_similarity_pairs() {
        let c = cos(lookup(a), lookup(b));
        model.push(c);
        human.push(score as f64);
        if score >= 3.5 {
            similar.push(c);
        } else if score <= 1.5 {
            dissimilar.push(c);
        }
    }
    let rho = stats::spearman(&human, &model);
    let (sim, dis) = (stats::mean(&similar), stats::mean(&dissimilar));
    println!(
        "  [1] similarity:    rho={rho:.4}, gap={:.4} (similar={sim:.4}, dissimilar={dis:.4})",
        sim - dis
    );
}

fn retrieval<'a>(lookup: &dyn Fn(&str) -> &'a [f32]) {
    let sets = corpus::retrieval_test_sets();
    let mut top1 = 0;
    let mut mrr = 0.0;
    for (query, candidates, correct) in &sets {
        let q = lookup(query);
        let mut scored: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cos(q, lookup(c))))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let rank = scored.iter().position(|(i, _)| i == correct).unwrap() + 1;
        top1 += usize::from(rank == 1);
        mrr += 1.0 / rank as f64;
    }
    let n = sets.len();
    println!(
        "  [2] retrieval:     top-1={top1}/{n} ({:.0}%), MRR={:.4}",
        stats::pct(top1, n),
        mrr / n as f64
    );
}

fn rho_of<'a>(lookup: &dyn Fn(&str) -> &'a [f32], pairs: &[&Pair]) -> f64 {
    let human: Vec<f64> = pairs.iter().map(|p| p.2 as f64).collect();
    let model: Vec<f64> = pairs
        .iter()
        .map(|p| cos(lookup(p.0), lookup(p.1)))
        .collect();
    if human.len() >= 3 {
        stats::spearman(&human, &model)
    } else {
        stats::pearson(&human, &model)
    }
}

fn multilingual<'a>(lookup: &dyn Fn(&str) -> &'a [f32]) {
    let pairs = corpus::graded_similarity_pairs();
    let has = |s: &str, lo: char, hi: char| s.chars().any(|c| (lo..=hi).contains(&c));
    let en: Vec<_> = pairs.iter().filter(|p| p.0.is_ascii()).collect();
    let zh: Vec<_> = pairs
        .iter()
        .filter(|p| !p.0.is_ascii() && has(p.0, '\u{4e00}', '\u{9fff}'))
        .collect();
    let ja: Vec<_> = pairs
        .iter()
        .filter(|p| !p.0.is_ascii() && has(p.0, '\u{3040}', '\u{30ff}'))
        .collect();
    println!(
        "  [3] multilingual:  en={:.4}, zh={:.4}, ja={:.4}",
        rho_of(lookup, &en),
        rho_of(lookup, &zh),
        rho_of(lookup, &ja)
    );
}

fn crosslingual<'a>(lookup: &dyn Fn(&str) -> &'a [f32]) {
    let mut sims = Vec::new();
    for group in corpus::crosslingual_groups() {
        for i in 0..group.len() {
            for j in i + 1..group.len() {
                sims.push(cos(lookup(group[i]), lookup(group[j])));
            }
        }
    }
    let min = sims.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "  [4] cross-lingual: avg={:.4}, min={min:.4}",
        stats::mean(&sims)
    );
}

fn length<'a>(lookup: &dyn Fn(&str) -> &'a [f32]) {
    let pairs = corpus::length_sensitivity_pairs();
    let short: Vec<_> = pairs.iter().filter(|p| p.0.len() < 50).collect();
    let medium: Vec<_> = pairs
        .iter()
        .filter(|p| (50..200).contains(&p.0.len()))
        .collect();
    let long: Vec<_> = pairs.iter().filter(|p| p.0.len() >= 200).collect();
    println!(
        "  [5] length:        short={:.4}, medium={:.4}, long={:.4}",
        rho_of(lookup, &short),
        rho_of(lookup, &medium),
        rho_of(lookup, &long)
    );
}

fn robustness<'a>(lookup: &dyn Fn(&str) -> &'a [f32]) {
    let mut sims = Vec::new();
    for (original, variants) in corpus::robustness_pairs() {
        for v in variants {
            sims.push(cos(lookup(original), lookup(v)));
        }
    }
    let min = sims.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "  [6] robustness:    avg={:.4}, min={min:.4}",
        stats::mean(&sims)
    );
}

fn clustering<'a>(lookup: &dyn Fn(&str) -> &'a [f32]) {
    let items: Vec<(&str, &str)> = corpus::cluster_groups()
        .into_iter()
        .flat_map(|(label, texts)| texts.iter().map(move |t| (*t, label)))
        .collect();
    let mut hits = 0;
    let mut intra = Vec::new();
    let mut inter = Vec::new();
    for (i, (ti, li)) in items.iter().enumerate() {
        let nearest = items
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .max_by(|a, b| {
                cos(lookup(ti), lookup(a.1.0)).total_cmp(&cos(lookup(ti), lookup(b.1.0)))
            })
            .map(|(_, (_, l))| *l);
        hits += usize::from(nearest == Some(*li));
        for (tj, lj) in &items[i + 1..] {
            let c = cos(lookup(ti), lookup(tj));
            if li == lj {
                intra.push(c);
            } else {
                inter.push(c);
            }
        }
    }
    let (a, b) = (stats::mean(&intra), stats::mean(&inter));
    println!(
        "  [7] clustering:    purity={hits}/{}, ratio={:.2}x (intra={a:.4}, inter={b:.4})",
        items.len(),
        a / b.max(0.001)
    );
}
