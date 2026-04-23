use embedrs::{cosine_similarity, dot_product, euclidean_distance};

#[test]
fn cosine_similarity_identical() {
    let v = vec![1.0, 2.0, 3.0];
    assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert!(cosine_similarity(&a, &b).abs() < 1e-6);
}

#[test]
fn dot_product_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
}

#[test]
fn euclidean_distance_basic() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
}

#[test]
fn similarity_with_real_embeddings() {
    // simulate normalized embeddings
    let a = vec![0.5, 0.5, 0.5, 0.5];
    let b = vec![0.5, 0.5, -0.5, -0.5];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-6); // orthogonal normalized vectors
}
