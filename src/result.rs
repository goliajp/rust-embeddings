//! [`EmbedResult`]: every vector of a call in one buffer.

use std::ops::Index;
use std::slice::ChunksExact;

use crate::usage::Usage;

/// The vectors of a call, one per input, in input order, stored row-major
/// in one buffer. `result[i]` is the vector of input `i`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedResult {
    /// Tokens and cost.
    pub usage: Usage,
    /// The model that produced the vectors.
    pub model: String,
    data: Vec<f32>,
    dim: usize,
}

impl EmbedResult {
    /// A result from a flat row-major buffer of vectors of width `dim`, for
    /// [`Embed`](crate::Embed) implementations outside embedrs.
    ///
    /// # Panics
    ///
    /// If `data` is not empty and its length is not a multiple of `dim`.
    pub fn from_flat(data: Vec<f32>, dim: usize, model: impl Into<String>, usage: Usage) -> Self {
        assert!(
            data.is_empty() || (dim > 0 && data.len().is_multiple_of(dim)),
            "{} values do not split into vectors of width {dim}",
            data.len()
        );
        Self {
            usage,
            model: model.into(),
            data,
            dim,
        }
        .normalized()
    }

    fn normalized(mut self) -> Self {
        if self.data.is_empty() {
            self.dim = 0;
        }
        self
    }

    /// Number of vectors.
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    /// True when there are no vectors.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Width of every vector; 0 when there are none.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The vector of input `i`.
    pub fn get(&self, i: usize) -> Option<&[f32]> {
        let start = i.checked_mul(self.dim)?;
        self.data
            .get(start..start + self.dim)
            .filter(|_| i < self.len())
    }

    /// The vectors in input order.
    pub fn iter(&self) -> ChunksExact<'_, f32> {
        self.data.chunks_exact(self.dim.max(1))
    }

    /// All values, row-major: vector `i` is `[i * dim .. (i + 1) * dim]`.
    pub fn as_flat(&self) -> &[f32] {
        &self.data
    }

    /// The buffer and the width, without copying.
    pub fn into_flat(self) -> (Vec<f32>, usize) {
        (self.data, self.dim)
    }

    /// One `Vec` per input (copies every value).
    pub fn to_vecs(&self) -> Vec<Vec<f32>> {
        self.iter().map(<[f32]>::to_vec).collect()
    }
}

impl Index<usize> for EmbedResult {
    type Output = [f32];

    fn index(&self, i: usize) -> &[f32] {
        match self.get(i) {
            Some(v) => v,
            None => panic!("vector index {i} out of range for {} vectors", self.len()),
        }
    }
}

impl<'a> IntoIterator for &'a EmbedResult {
    type Item = &'a [f32];
    type IntoIter = ChunksExact<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(data: Vec<f32>, dim: usize) -> EmbedResult {
        EmbedResult::from_flat(data, dim, "m", Usage::default())
    }

    #[test]
    fn rows() {
        let res = r(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        assert_eq!(res.len(), 2);
        assert_eq!(res.dim(), 3);
        assert_eq!(&res[1], &[4.0, 5.0, 6.0]);
        assert_eq!(res.get(2), None);
        assert_eq!(res.iter().count(), 2);
        assert_eq!(
            res.to_vecs(),
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
        );
        let rows: Vec<&[f32]> = (&res).into_iter().collect();
        assert_eq!(rows[0], &[1.0, 2.0, 3.0]);
        assert_eq!(res.clone().into_flat().1, 3);
    }

    #[test]
    fn empty() {
        for res in [r(vec![], 0), r(vec![], 384)] {
            assert!(res.is_empty());
            assert_eq!(res.len(), 0);
            assert_eq!(res.dim(), 0);
            assert_eq!(res.iter().count(), 0);
            assert_eq!(res.get(0), None);
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn index_out_of_range_panics() {
        let _ = &r(vec![1.0], 1)[1];
    }

    #[test]
    #[should_panic(expected = "do not split")]
    fn ragged_buffer_panics() {
        r(vec![1.0, 2.0, 3.0], 2);
    }
}
