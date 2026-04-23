/// compute spearman rank correlation coefficient
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
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

/// compute pearson correlation coefficient
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
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

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

pub fn pct(num: usize, den: usize) -> f64 {
    num as f64 / den as f64 * 100.0
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spearman_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((spearman(&x, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn spearman_inverse() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((spearman(&x, &y) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_perfect() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mean_basic() {
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn mean_empty() {
        assert_eq!(mean(&[]), 0.0);
    }
}
