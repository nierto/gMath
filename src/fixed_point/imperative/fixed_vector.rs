//! FixedVector — dynamically-sized vector of FixedPoint values.
//!
//! All operations delegate to FixedPoint arithmetic.

use std::ops::{Index, IndexMut};
use super::FixedPoint;

/// A dynamically-sized vector of fixed-point values.
///
/// Backed by `Vec<FixedPoint>`. Not `Copy` due to heap allocation.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedVector {
    data: Vec<FixedPoint>,
}

impl FixedVector {
    /// Create a zero-filled vector of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            data: vec![FixedPoint::ZERO; dim],
        }
    }

    /// Create from a slice of f32 values.
    pub fn from_f32_slice(values: &[f32]) -> Self {
        Self {
            data: values.iter().map(|&v| FixedPoint::from_f32(v)).collect(),
        }
    }

    /// Create from a slice of FixedPoint values.
    pub fn from_slice(values: &[FixedPoint]) -> Self {
        Self {
            data: values.to_vec(),
        }
    }

    /// Number of components.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Alias for `len()`.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Whether the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Dot product of two vectors.
    ///
    /// Panics if dimensions differ.
    pub fn dot(&self, other: &FixedVector) -> FixedPoint {
        assert_eq!(self.len(), other.len(), "FixedVector::dot: dimension mismatch");
        let mut sum = FixedPoint::ZERO;
        for i in 0..self.len() {
            sum += self.data[i] * other.data[i];
        }
        sum
    }

    /// Squared length (self . self).
    pub fn length_squared(&self) -> FixedPoint {
        self.dot(self)
    }

    /// Length (Euclidean norm).
    pub fn length(&self) -> FixedPoint {
        self.length_squared().sqrt()
    }

    /// Normalize in place (divide each component by length).
    ///
    /// Panics if length is zero.
    pub fn normalize(&mut self) {
        let len = self.length();
        for v in &mut self.data {
            *v = *v / len;
        }
    }

    /// Return a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }

    /// Apply a function to each component, returning a new vector.
    pub fn map(&self, f: impl Fn(FixedPoint) -> FixedPoint) -> Self {
        Self {
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    /// Iterator over components.
    pub fn iter(&self) -> std::slice::Iter<'_, FixedPoint> {
        self.data.iter()
    }

    /// Mutable iterator over components.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, FixedPoint> {
        self.data.iter_mut()
    }

    /// Safe metric distance between two vectors (Euclidean).
    pub fn metric_distance_safe(&self, other: &FixedVector) -> FixedPoint {
        assert_eq!(self.len(), other.len(), "FixedVector::metric_distance_safe: dimension mismatch");
        let mut sum = FixedPoint::ZERO;
        for i in 0..self.len() {
            let diff = self.data[i] - other.data[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

impl Index<usize> for FixedVector {
    type Output = FixedPoint;
    #[inline]
    fn index(&self, idx: usize) -> &FixedPoint {
        &self.data[idx]
    }
}

impl IndexMut<usize> for FixedVector {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut FixedPoint {
        &mut self.data[idx]
    }
}

impl Default for FixedVector {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}
