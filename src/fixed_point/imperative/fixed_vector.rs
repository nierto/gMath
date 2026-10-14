//! FixedVector: dynamically-sized vector of fixed-point values of one format.
//!
//! All operations delegate to the element type's arithmetic. The element
//! type defaults to `FixedPoint`, the build profile's format.

use std::ops::{Add, Sub, Neg, Mul, Index, IndexMut};
use super::kernels;
use super::FixedPoint;
use crate::fixed_point::fixed_num::{Fixed, FixedNum};
use crate::fixed_point::width::Storage;

/// `a * b - c * d` (`subtract`) or `a * b + c * d`, rounded once from the
/// exact value. Panics when the result leaves the storage range.
#[inline]
fn rotated<S: Storage, const F: u32>(a: Fixed<S, F>, b: Fixed<S, F>, c: Fixed<S, F>, d: Fixed<S, F>, subtract: bool) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let (p, q) = (a.raw().mul_wide(b.raw()), c.raw().mul_wide(d.raw()));
    let sum = if subtract { Fixed::<S, F>::compute_checked_subtract(p, q) } else { Fixed::<S, F>::compute_checked_add(p, q) };
    Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(sum.expect("FixedVector::rotate_pairs: overflow")))
}

/// A dynamically-sized vector of fixed-point values.
///
/// Backed by `Vec<T>`. Not `Copy` due to heap allocation.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedVector<T = FixedPoint> {
    data: Vec<T>,
}

impl<S: Storage, const F: u32> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Create a zero-filled vector of the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            data: vec![Fixed::ZERO; dim],
        }
    }

    /// Create from a slice of f32 values.
    pub fn from_f32_slice(values: &[f32]) -> Self {
        Self {
            data: values.iter().map(|&v| Fixed::from_f32(v)).collect(),
        }
    }

    /// Create from a slice of FixedPoint values.
    pub fn from_slice(values: &[Fixed<S, F>]) -> Self {
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
    pub fn dot(&self, other: &Self) -> Fixed<S, F> {
        assert_eq!(self.len(), other.len(), "FixedVector::dot: dimension mismatch");
        kernels::dot(&self.data, &other.data)
    }

    /// Squared length (self . self).
    pub fn length_squared(&self) -> Fixed<S, F> {
        self.dot(self)
    }

    /// Length (Euclidean norm): the sum of squares and the root at the
    /// compute tier, one rounding (was `length_squared().sqrt()`: two roundings,
    /// and an overflow once the squared length left storage).
    pub fn length(&self) -> Fixed<S, F> {
        let raw = Fixed::raw_slice(&self.data);
        Fixed::from_raw(kernels::sqrt_dot::<Fixed<S, F>>(raw, raw).expect("FixedVector::length exceeds storage"))
    }

    /// Normalize in place (divide each component by length).
    ///
    /// The sum of squares is exact and its root is taken at the compute tier;
    /// each component is divided by that compute-tier length and rounded to
    /// storage once (the length is never rounded to storage, so a short
    /// nonzero vector normalizes instead of dividing by zero).
    ///
    /// Panics if the vector is zero or its squared length exceeds the compute
    /// tier.
    pub fn normalize(&mut self) {
        let raw = Fixed::raw_slice(&self.data);
        let sum_sq = kernels::exact_dot::<Fixed<S, F>>(raw, raw).expect("FixedVector::normalize: squared length exceeds the compute tier");
        assert!(!Fixed::<S, F>::compute_is_zero(&sum_sq), "FixedVector::normalize: zero vector");
        let len = Fixed::<S, F>::sqrt_at_compute_tier(sum_sq);
        let one = Fixed::<S, F>::one().raw().widen();
        for v in &mut self.data {
            // v at 3F over the length at 2F: |v / len| <= 1, never overflows
            let numer = S::widen_product(Fixed::<S, F>::upscale_to_compute(v.raw()), one);
            *v = Fixed::from_raw(kernels::divide_to_storage_nearest::<Fixed<S, F>>(numer, len).expect("FixedVector::normalize: zero length"));
        }
    }

    /// Return a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }

    /// Apply a function to each component, returning a new vector.
    pub fn map(&self, f: impl Fn(Fixed<S, F>) -> Fixed<S, F>) -> Self {
        Self {
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    /// Iterator over components.
    pub fn iter(&self) -> std::slice::Iter<'_, Fixed<S, F>> {
        self.data.iter()
    }

    /// Mutable iterator over components.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Fixed<S, F>> {
        self.data.iter_mut()
    }

    /// Metric distance between two vectors (Euclidean) at compute tier.
    ///
    /// sum-of-squared-differences accumulated at tier N+1, single downscale → 1 ULP.
    pub fn metric_distance_safe(&self, other: &Self) -> Fixed<S, F> {
        assert_eq!(self.len(), other.len(), "FixedVector::metric_distance_safe: dimension mismatch");
        let diff_raw: Vec<S> = (0..self.len())
            .map(|i| (self.data[i] - other.data[i]).raw())
            .collect();
        Fixed::from_raw(kernels::sqrt_dot::<Fixed<S, F>>(&diff_raw, &diff_raw).expect("metric distance exceeds storage"))
    }
}

impl<S: Storage, const F: u32> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Fused length: sqrt(Σ x_i²) entirely at compute tier.
    ///
    /// More precise than `length()` which materializes the dot product
    /// before taking sqrt. This version keeps the accumulated sum at
    /// tier N+1 and takes sqrt there: single downscale at the end.
    pub fn length_fused(&self) -> Fixed<S, F> {
        super::fused::sqrt_sum_sq(&self.data)
    }

    /// Fused Euclidean distance to another vector: sqrt(Σ (a_i - b_i)²)
    /// entirely at compute tier.
    ///
    /// Saves 2 materializations vs `(self - other).length()`.
    pub fn distance_to(&self, other: &Self) -> Fixed<S, F> {
        assert_eq!(self.len(), other.len(), "FixedVector::distance_to: dimension mismatch");
        super::fused::euclidean_distance(&self.data, &other.data)
    }
}

impl<T> Index<usize> for FixedVector<T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: usize) -> &T {
        &self.data[idx]
    }
}

impl<T> IndexMut<usize> for FixedVector<T> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.data[idx]
    }
}

impl<T> Default for FixedVector<T> {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

// ============================================================================
// L1A: Arithmetic operators
// ============================================================================

impl<S: Storage, const F: u32> Add for FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.len(), rhs.len(), "FixedVector::add: dimension mismatch");
        Self {
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a + b).collect(),
        }
    }
}

impl<'a, 'b, S: Storage, const F: u32> Add<&'b FixedVector<Fixed<S, F>>> for &'a FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedVector<Fixed<S, F>>;
    fn add(self, rhs: &'b FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        assert_eq!(self.len(), rhs.len(), "FixedVector::add: dimension mismatch");
        FixedVector {
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a + b).collect(),
        }
    }
}

impl<S: Storage, const F: u32> Sub for FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        assert_eq!(self.len(), rhs.len(), "FixedVector::sub: dimension mismatch");
        Self {
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a - b).collect(),
        }
    }
}

impl<'a, 'b, S: Storage, const F: u32> Sub<&'b FixedVector<Fixed<S, F>>> for &'a FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedVector<Fixed<S, F>>;
    fn sub(self, rhs: &'b FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        assert_eq!(self.len(), rhs.len(), "FixedVector::sub: dimension mismatch");
        FixedVector {
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a - b).collect(),
        }
    }
}

impl<S: Storage, const F: u32> Neg for FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            data: self.data.iter().map(|&v| -v).collect(),
        }
    }
}

impl<S: Storage, const F: u32> Neg for &FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedVector<Fixed<S, F>>;
    fn neg(self) -> FixedVector<Fixed<S, F>> {
        FixedVector {
            data: self.data.iter().map(|&v| -v).collect(),
        }
    }
}

/// Scalar * Vector
impl<S: Storage, const F: u32> Mul<FixedVector<Fixed<S, F>>> for Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedVector<Fixed<S, F>>;
    fn mul(self, rhs: FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        FixedVector {
            data: rhs.data.iter().map(|&v| self * v).collect(),
        }
    }
}

/// Vector * Scalar
impl<S: Storage, const F: u32> Mul<Fixed<S, F>> for FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    fn mul(self, rhs: Fixed<S, F>) -> Self {
        Self {
            data: self.data.iter().map(|&v| v * rhs).collect(),
        }
    }
}

/// &Vector * Scalar
impl<S: Storage, const F: u32> Mul<Fixed<S, F>> for &FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedVector<Fixed<S, F>>;
    fn mul(self, rhs: Fixed<S, F>) -> FixedVector<Fixed<S, F>> {
        FixedVector {
            data: self.data.iter().map(|&v| v * rhs).collect(),
        }
    }
}

// ============================================================================
// L1A: Additional vector operations
// ============================================================================

impl<S: Storage, const F: u32> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Compute-tier precise dot product.
    ///
    /// Accumulates at tier N+1 (double width) and rounds once at the end.
    /// For an n-element vector, this gives 1 ULP of rounding error instead
    /// of the n ULP that the standard `dot()` method may accumulate.
    pub fn dot_precise(&self, other: &Self) -> Fixed<S, F> {
        assert_eq!(self.len(), other.len(), "FixedVector::dot_precise: dimension mismatch");
        kernels::dot(&self.data, &other.data)
    }

    /// Cross product (3D vectors only).
    ///
    /// Each component `a b - c d` is the exact difference of two exact
    /// products at the compute tier, rounded to storage once.
    ///
    /// Panics if either vector is not 3-dimensional or a component exceeds
    /// storage.
    pub fn cross(&self, other: &Self) -> Self {
        assert_eq!(self.len(), 3, "FixedVector::cross: self must be 3D");
        assert_eq!(other.len(), 3, "FixedVector::cross: other must be 3D");
        let (u, v) = (&self.data, &other.data);
        let component = |i: usize, j: usize| {
            // u_i v_j - u_j v_i; each product of two storage values is exact at 2F
            let first = kernels::exact_dot::<Fixed<S, F>>(&[u[i].raw()], &[v[j].raw()]).expect("FixedVector::cross: product");
            let second = kernels::exact_dot::<Fixed<S, F>>(&[u[j].raw()], &[v[i].raw()]).expect("FixedVector::cross: product");
            Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(Fixed::<S, F>::compute_subtract(first, second)))
        };
        Self::from_slice(&[component(1, 2), component(2, 0), component(0, 1)])
    }

    /// Outer product: u ⊗ v → Matrix where M[i][j] = u[i] * v[j].
    pub fn outer_product(&self, other: &Self) -> super::FixedMatrix<Fixed<S, F>> {
        let mut m = super::FixedMatrix::<Fixed<S, F>>::new(self.len(), other.len());
        for i in 0..self.len() {
            for j in 0..other.len() {
                m.set(i, j, self.data[i] * other.data[j]);
            }
        }
        m
    }

    /// Access the underlying data slice (for compute-tier operations).
    #[inline]
    pub fn as_slice(&self) -> &[Fixed<S, F>] {
        &self.data
    }

    /// Mutable access to the underlying data slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Fixed<S, F>] {
        &mut self.data
    }

    /// Rotate the pairs `(v[i], v[i + half])`, `half = rotary_dim / 2`, by
    /// the angles whose sine and cosine are given:
    ///
    /// ```text
    /// v[i]        = v[i] * cos[i] - v[i + half] * sin[i]
    /// v[i + half] = v[i] * sin[i] + v[i + half] * cos[i]
    /// ```
    ///
    /// Elements from `rotary_dim` on are left unchanged (a partial rotary
    /// embedding). Each output is rounded once: the two products are exact
    /// at the compute tier, their sum or difference is exact, and that value
    /// is rounded to storage (nearest, ties toward positive infinity). The
    /// result is the correctly rounded value of the expression, within half
    /// a unit; writing the expression with the operators rounds each product
    /// first and can be a unit off.
    ///
    /// # Panics
    /// Panics if `rotary_dim` exceeds the vector length, if `sin` or `cos`
    /// holds fewer than `rotary_dim / 2` values, or on overflow (as the
    /// operators do).
    pub fn rotate_pairs(&mut self, sin: &[Fixed<S, F>], cos: &[Fixed<S, F>], rotary_dim: usize) {
        let half = rotary_dim / 2;
        assert!(rotary_dim <= self.data.len(), "FixedVector::rotate_pairs: rotary_dim exceeds the vector length");
        assert!(sin.len() >= half && cos.len() >= half, "FixedVector::rotate_pairs: sin/cos shorter than rotary_dim / 2");
        let (lo, hi) = self.data.split_at_mut(half);
        for i in 0..half {
            let (x0, x1) = (lo[i], hi[i]);
            lo[i] = rotated(x0, cos[i], x1, sin[i], true);
            hi[i] = rotated(x0, sin[i], x1, cos[i], false);
        }
    }
}
