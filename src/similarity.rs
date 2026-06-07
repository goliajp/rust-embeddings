//! Vector similarity primitives (cosine, dot product, Euclidean distance).
//!
//! All routines run scalar code, but the hot loops are written
//! autovectorization-friendly: 8 independent accumulators with no inter-iter
//! dependency, so LLVM emits AVX2 / NEON SIMD at `-C opt-level=3` (the default
//! for `--release`) without us pulling in `wide` or `std::simd`. Tail elements
//! fold through a scalar epilogue.

const LANES: usize = 8;

/// Reduce the partial-sum array to a single f32.
#[inline]
fn hsum(acc: [f32; LANES]) -> f32 {
    acc.iter().sum()
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in `[-1.0, 1.0]`. Returns `0.0` if either vector has zero magnitude.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);

    let mut dot = [0.0f32; LANES];
    let mut na = [0.0f32; LANES];
    let mut nb = [0.0f32; LANES];

    let chunks = len / LANES;
    for i in 0..chunks {
        let base = i * LANES;
        for j in 0..LANES {
            let x = a[base + j];
            let y = b[base + j];
            dot[j] += x * y;
            na[j] += x * x;
            nb[j] += y * y;
        }
    }

    let mut dot_s = hsum(dot);
    let mut na_s = hsum(na);
    let mut nb_s = hsum(nb);
    for i in (chunks * LANES)..len {
        let x = a[i];
        let y = b[i];
        dot_s += x * y;
        na_s += x * x;
        nb_s += y * y;
    }

    let denom = (na_s * nb_s).sqrt();
    if denom == 0.0 { 0.0 } else { dot_s / denom }
}

/// Compute dot product between two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);

    let mut acc = [0.0f32; LANES];
    let chunks = len / LANES;
    for i in 0..chunks {
        let base = i * LANES;
        for j in 0..LANES {
            acc[j] += a[base + j] * b[base + j];
        }
    }
    let mut s = hsum(acc);
    for i in (chunks * LANES)..len {
        s += a[i] * b[i];
    }
    s
}

/// Compute Euclidean distance between two vectors.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);

    let mut acc = [0.0f32; LANES];
    let chunks = len / LANES;
    for i in 0..chunks {
        let base = i * LANES;
        for j in 0..LANES {
            let d = a[base + j] - b[base + j];
            acc[j] += d * d;
        }
    }
    let mut s = hsum(acc);
    for i in (chunks * LANES)..len {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dp = dot_product(&a, &b);
        assert!((dp - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(dot_product(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn euclidean_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let dist = euclidean_distance(&v, &v);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn euclidean_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_unit() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(dot_product(&a, &b), 0.0);
        assert!(euclidean_distance(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn single_element_vectors() {
        let a = vec![3.0];
        let b = vec![4.0];
        // cosine of same-direction scalars is 1.0
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!((dot_product(&a, &b) - 12.0).abs() < 1e-6);
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn high_dim_matches_naive_scalar() {
        // Confirm the chunked-accumulator path agrees with the textbook scalar
        // implementation for realistic embedding dimensions (1536 = OpenAI
        // text-embedding-3-small, plus a non-multiple-of-LANES case at 1537).
        fn naive_cos(a: &[f32], b: &[f32]) -> f32 {
            let mut d = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                d += x * y;
                na += x * x;
                nb += y * y;
            }
            let den = (na * nb).sqrt();
            if den == 0.0 { 0.0 } else { d / den }
        }
        // deterministic pseudo-random — no rand dep, no flake
        let a: Vec<f32> = (0..1537).map(|i| ((i as f32 * 1.7).sin())).collect();
        let b: Vec<f32> = (0..1537).map(|i| ((i as f32 * 0.3).cos())).collect();

        for n in [1024, 1536, 1537] {
            let na = &a[..n];
            let nb = &b[..n];
            let fast = cosine_similarity(na, nb);
            let slow = naive_cos(na, nb);
            // chunked sums reorder additions → expect tiny FP drift, not bit-equal
            assert!(
                (fast - slow).abs() < 1e-5,
                "dim={n}: fast={fast} slow={slow}"
            );

            let fast_dot = dot_product(na, nb);
            let slow_dot: f32 = na.iter().zip(nb.iter()).map(|(x, y)| x * y).sum();
            assert!((fast_dot - slow_dot).abs() < 1e-3 * slow_dot.abs().max(1.0));

            let fast_eu = euclidean_distance(na, nb);
            let slow_eu: f32 = na
                .iter()
                .zip(nb.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt();
            assert!((fast_eu - slow_eu).abs() < 1e-4 * slow_eu.abs().max(1.0));
        }
    }

    #[test]
    fn mismatched_lengths_truncates_via_zip() {
        // zip stops at the shorter vector — this documents the behavior
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        // only first 2 elements are compared
        let expected_dot = 1.0 * 1.0 + 2.0 * 2.0; // 5.0
        assert!((dot_product(&a, &b) - expected_dot).abs() < 1e-6);

        let expected_dist = 0.0f32; // (1-1)^2 + (2-2)^2 = 0
        assert!((euclidean_distance(&a, &b) - expected_dist).abs() < 1e-6);

        // cosine: dot / (norm_a * norm_b) where norms only cover first 2 elements
        let norm_a = (1.0f32 * 1.0 + 2.0 * 2.0).sqrt();
        let norm_b = (1.0f32 * 1.0 + 2.0 * 2.0).sqrt();
        let expected_cos = expected_dot / (norm_a * norm_b);
        assert!((cosine_similarity(&a, &b) - expected_cos).abs() < 1e-6);
    }
}
