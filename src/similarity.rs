/// Compute cosine similarity between two vectors.
///
/// Returns a value in `[-1.0, 1.0]`. Returns `0.0` if either vector has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Compute dot product between two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute Euclidean distance between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
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
