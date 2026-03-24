//! FixedVector — dynamically-sized vector of FixedPoint values.
//!
//! All operations delegate to FixedPoint arithmetic.

use std::ops::{Add, Sub, Neg, Mul, Index, IndexMut};
use super::FixedPoint;
use super::linalg::{compute_tier_dot, compute_tier_dot_raw};
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;

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

    /// Dot product of two vectors at compute tier (tier N+1).
    ///
    /// Accumulates at double width, single downscale at end → 1 ULP.
    /// Panics if dimensions differ.
    pub fn dot(&self, other: &FixedVector) -> FixedPoint {
        assert_eq!(self.len(), other.len(), "FixedVector::dot: dimension mismatch");
        compute_tier_dot(&self.data, &other.data)
    }

    /// Squared length (self . self).
    pub fn length_squared(&self) -> FixedPoint {
        self.dot(self)
    }

    /// Length (Euclidean norm).
    pub fn length(&self) -> FixedPoint {
        self.length_squared().sqrt()
    }

    /// Fused length — sqrt(Σ x_i²) entirely at compute tier.
    ///
    /// More precise than `length()` which materializes the dot product
    /// before taking sqrt. This version keeps the accumulated sum at
    /// tier N+1 and takes sqrt there — single downscale at the end.
    pub fn length_fused(&self) -> FixedPoint {
        super::fused::sqrt_sum_sq(&self.data)
    }

    /// Fused Euclidean distance to another vector — sqrt(Σ (a_i - b_i)²)
    /// entirely at compute tier.
    ///
    /// Saves 2 materializations vs `(self - other).length()`.
    pub fn distance_to(&self, other: &FixedVector) -> FixedPoint {
        assert_eq!(self.len(), other.len(), "FixedVector::distance_to: dimension mismatch");
        super::fused::euclidean_distance(&self.data, &other.data)
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

    /// Metric distance between two vectors (Euclidean) at compute tier.
    ///
    /// sum-of-squared-differences accumulated at tier N+1, single downscale → 1 ULP.
    pub fn metric_distance_safe(&self, other: &FixedVector) -> FixedPoint {
        assert_eq!(self.len(), other.len(), "FixedVector::metric_distance_safe: dimension mismatch");
        let diff_raw: Vec<BinaryStorage> = (0..self.len())
            .map(|i| (self.data[i] - other.data[i]).raw())
            .collect();
        let sum_sq = FixedPoint::from_raw(compute_tier_dot_raw(&diff_raw, &diff_raw));
        sum_sq.sqrt()
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

// ============================================================================
// L1A: Arithmetic operators
// ============================================================================

impl Add for FixedVector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.len(), rhs.len(), "FixedVector::add: dimension mismatch");
        Self {
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a + b).collect(),
        }
    }
}

impl<'a, 'b> Add<&'b FixedVector> for &'a FixedVector {
    type Output = FixedVector;
    fn add(self, rhs: &'b FixedVector) -> FixedVector {
        assert_eq!(self.len(), rhs.len(), "FixedVector::add: dimension mismatch");
        FixedVector {
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a + b).collect(),
        }
    }
}

impl Sub for FixedVector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        assert_eq!(self.len(), rhs.len(), "FixedVector::sub: dimension mismatch");
        Self {
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a - b).collect(),
        }
    }
}

impl<'a, 'b> Sub<&'b FixedVector> for &'a FixedVector {
    type Output = FixedVector;
    fn sub(self, rhs: &'b FixedVector) -> FixedVector {
        assert_eq!(self.len(), rhs.len(), "FixedVector::sub: dimension mismatch");
        FixedVector {
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a - b).collect(),
        }
    }
}

impl Neg for FixedVector {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            data: self.data.iter().map(|&v| -v).collect(),
        }
    }
}

impl Neg for &FixedVector {
    type Output = FixedVector;
    fn neg(self) -> FixedVector {
        FixedVector {
            data: self.data.iter().map(|&v| -v).collect(),
        }
    }
}

/// Scalar * Vector
impl Mul<FixedVector> for FixedPoint {
    type Output = FixedVector;
    fn mul(self, rhs: FixedVector) -> FixedVector {
        FixedVector {
            data: rhs.data.iter().map(|&v| self * v).collect(),
        }
    }
}

/// Vector * Scalar
impl Mul<FixedPoint> for FixedVector {
    type Output = Self;
    fn mul(self, rhs: FixedPoint) -> Self {
        Self {
            data: self.data.iter().map(|&v| v * rhs).collect(),
        }
    }
}

/// &Vector * Scalar
impl Mul<FixedPoint> for &FixedVector {
    type Output = FixedVector;
    fn mul(self, rhs: FixedPoint) -> FixedVector {
        FixedVector {
            data: self.data.iter().map(|&v| v * rhs).collect(),
        }
    }
}

// ============================================================================
// L1A: Additional vector operations
// ============================================================================

impl FixedVector {
    /// Compute-tier precise dot product.
    ///
    /// Accumulates at tier N+1 (double width) and rounds once at the end.
    /// For an n-element vector, this gives 1 ULP of rounding error instead
    /// of the n ULP that the standard `dot()` method may accumulate.
    pub fn dot_precise(&self, other: &FixedVector) -> FixedPoint {
        assert_eq!(self.len(), other.len(), "FixedVector::dot_precise: dimension mismatch");
        compute_tier_dot(&self.data, &other.data)
    }

    /// Cross product (3D vectors only).
    ///
    /// Panics if either vector is not 3-dimensional.
    pub fn cross(&self, other: &FixedVector) -> FixedVector {
        assert_eq!(self.len(), 3, "FixedVector::cross: self must be 3D");
        assert_eq!(other.len(), 3, "FixedVector::cross: other must be 3D");
        FixedVector::from_slice(&[
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        ])
    }

    /// Outer product: u ⊗ v → Matrix where M[i][j] = u[i] * v[j].
    pub fn outer_product(&self, other: &FixedVector) -> super::FixedMatrix {
        let mut m = super::FixedMatrix::new(self.len(), other.len());
        for i in 0..self.len() {
            for j in 0..other.len() {
                m.set(i, j, self.data[i] * other.data[j]);
            }
        }
        m
    }

    /// Access the underlying data slice (for compute-tier operations).
    #[inline]
    pub(crate) fn as_slice(&self) -> &[FixedPoint] {
        &self.data
    }
}
