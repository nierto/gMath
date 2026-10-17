//! FixedMatrix: row-major matrix of `Fixed<S, F>` values.
//!
//! All matrix arithmetic uses compute-tier accumulation for precision:
//! matrix multiply, matrix-vector multiply accumulate at tier N+1 and
//! round once at the end.

use std::ops::{Add, Sub, Neg, Mul};
use super::kernels;
use super::FixedVector;
use crate::fixed_point::fixed_num::{Fixed, FixedNum};
use crate::fixed_point::width::Storage;

/// A row-major matrix of fixed-point values of one format.
///
/// Backed by a flat `Vec<T>`. Not `Copy` due to heap allocation. The element
/// type is the caller's format, usually a `Fixed<S, F>`.
#[derive(Clone, Debug)]
pub struct FixedMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

// ============================================================================
// Core constructors and accessors (existing API preserved exactly)
// ============================================================================

impl<S: Storage, const F: u32> FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Create a zero-filled matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Fixed::ZERO; rows * cols],
        }
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get element at (row, col).
    ///
    /// Panics if out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Fixed<S, F> {
        assert!(row < self.rows && col < self.cols, "FixedMatrix<Fixed<S, F>>: index out of bounds");
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    ///
    /// Panics if out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: Fixed<S, F>) {
        assert!(row < self.rows && col < self.cols, "FixedMatrix<Fixed<S, F>>: index out of bounds");
        self.data[row * self.cols + col] = value;
    }
}

// ============================================================================
// L1A: New constructors
// ============================================================================

impl<S: Storage, const F: u32> FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Create from a flat slice of values in row-major order.
    ///
    /// Panics if `values.len() != rows * cols`.
    pub fn from_slice(rows: usize, cols: usize, values: &[Fixed<S, F>]) -> Self {
        assert_eq!(values.len(), rows * cols,
            "FixedMatrix::from_slice: expected {} elements, got {}", rows * cols, values.len());
        Self {
            rows,
            cols,
            data: values.to_vec(),
        }
    }

    /// Create from a function: M\[i\]\[j\] = f(i, j).
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> Fixed<S, F>) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self { rows, cols, data }
    }

    /// Identity matrix of size n×n.
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |r, c| {
            if r == c { Fixed::one() } else { Fixed::ZERO }
        })
    }

    /// Diagonal matrix from a vector.
    pub fn diagonal(v: &FixedVector<Fixed<S, F>>) -> Self {
        let n = v.len();
        Self::from_fn(n, n, |r, c| {
            if r == c { v[r] } else { Fixed::ZERO }
        })
    }
}

// ============================================================================
// L1A: Matrix properties
// ============================================================================

impl<S: Storage, const F: u32> FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Transpose: Aᵀ\[i\]\[j\] = A\[j\]\[i\].
    pub fn transpose(&self) -> Self {
        let mut data = Vec::with_capacity(self.rows * self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                data.push(self.data[r * self.cols + c]);
            }
        }
        Self { rows: self.cols, cols: self.rows, data }
    }

    /// Trace: sum of diagonal elements.
    ///
    /// Panics if matrix is not square.
    pub fn trace(&self) -> Fixed<S, F> {
        assert_eq!(self.rows, self.cols, "FixedMatrix::trace: matrix must be square");
        let mut sum = Fixed::ZERO;
        for i in 0..self.rows {
            sum += self.get(i, i);
        }
        sum
    }

    /// Whether this is a square matrix.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Extract a row as a FixedVector<Fixed<S, F>>.
    pub fn row(&self, r: usize) -> FixedVector<Fixed<S, F>> {
        assert!(r < self.rows, "FixedMatrix::row: index out of bounds");
        let start = r * self.cols;
        FixedVector::<Fixed<S, F>>::from_slice(&self.data[start..start + self.cols])
    }

    /// Extract a column as a FixedVector<Fixed<S, F>>.
    pub fn col(&self, c: usize) -> FixedVector<Fixed<S, F>> {
        assert!(c < self.cols, "FixedMatrix::col: index out of bounds");
        let mut v = FixedVector::<Fixed<S, F>>::new(self.rows);
        for r in 0..self.rows {
            v[r] = self.get(r, c);
        }
        v
    }
}

// ============================================================================
// L1A: Matrix-vector multiply (compute-tier accumulated)
// ============================================================================

impl<S: Storage, const F: u32> FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Matrix-vector multiply: y = A * x.
    ///
    /// Uses compute-tier accumulation: each row dot product is computed
    /// at tier N+1 precision and rounded once.
    ///
    /// Panics if `self.cols() != v.len()`.
    pub fn mul_vector(&self, v: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        assert_eq!(self.cols, v.len(),
            "FixedMatrix::mul_vector: cols ({}) != vector len ({})", self.cols, v.len());
        let v_raw: Vec<S> = v.as_slice().iter().map(|fp| fp.raw()).collect();
        let mut result = FixedVector::<Fixed<S, F>>::new(self.rows);
        for r in 0..self.rows {
            let row_start = r * self.cols;
            let row_raw: Vec<S> = self.data[row_start..row_start + self.cols]
                .iter().map(|fp| fp.raw()).collect();
            result[r] = Fixed::from_raw(kernels::dot_raw::<Fixed<S, F>>(&row_raw, &v_raw));
        }
        result
    }
}

// ============================================================================
// L1A: Matrix-matrix multiply (compute-tier accumulated)
// ============================================================================

impl<S: Storage, const F: u32> Mul for FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    /// Matrix multiply: C = A * B.
    ///
    /// Uses compute-tier accumulation for each output element.
    /// Panics if `self.cols() != rhs.rows()`.
    fn mul(self, rhs: Self) -> Self {
        mat_mul(&self, &rhs)
    }
}

impl<S: Storage, const F: u32> Mul for &FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedMatrix<Fixed<S, F>>;
    fn mul(self, rhs: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> {
        mat_mul(self, rhs)
    }
}


fn mat_mul<S: Storage, const F: u32>(a: &FixedMatrix<Fixed<S, F>>, b: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(a.cols, b.rows,
        "FixedMatrix::mul: A.cols ({}) != B.rows ({})", a.cols, b.rows);
    let k = a.cols;
    // Pre-extract B columns as raw storage for efficient repeated access
    let b_cols_raw: Vec<Vec<S>> = (0..b.cols)
        .map(|c| (0..b.rows).map(|r| b.data[r * b.cols + c].raw()).collect())
        .collect();

    let mut result = FixedMatrix::<Fixed<S, F>>::new(a.rows, b.cols);
    for r in 0..a.rows {
        let row_start = r * k;
        let a_row_raw: Vec<S> = a.data[row_start..row_start + k]
            .iter().map(|fp| fp.raw()).collect();
        for c in 0..b.cols {
            let dot = kernels::dot_raw::<Fixed<S, F>>(&a_row_raw, &b_cols_raw[c]);
            result.data[r * b.cols + c] = Fixed::from_raw(dot);
        }
    }
    result
}

// ============================================================================
// L1A: Element-wise arithmetic
// ============================================================================

impl<S: Storage, const F: u32> Add for FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols),
            "FixedMatrix::add: dimension mismatch");
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a + b).collect(),
        }
    }
}

impl<'a, 'b, S: Storage, const F: u32> Add<&'b FixedMatrix<Fixed<S, F>>> for &'a FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedMatrix<Fixed<S, F>>;
    fn add(self, rhs: &'b FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols),
            "FixedMatrix::add: dimension mismatch");
        FixedMatrix::<Fixed<S, F>> {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a + b).collect(),
        }
    }
}

impl<S: Storage, const F: u32> Sub for FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols),
            "FixedMatrix::sub: dimension mismatch");
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a - b).collect(),
        }
    }
}

impl<'a, 'b, S: Storage, const F: u32> Sub<&'b FixedMatrix<Fixed<S, F>>> for &'a FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedMatrix<Fixed<S, F>>;
    fn sub(self, rhs: &'b FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols),
            "FixedMatrix::sub: dimension mismatch");
        FixedMatrix::<Fixed<S, F>> {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a - b).collect(),
        }
    }
}

impl<S: Storage, const F: u32> Neg for FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| -v).collect(),
        }
    }
}

impl<S: Storage, const F: u32> Neg for &FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedMatrix<Fixed<S, F>>;
    fn neg(self) -> FixedMatrix<Fixed<S, F>> {
        FixedMatrix::<Fixed<S, F>> {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| -v).collect(),
        }
    }
}

/// Scalar * Matrix
impl<S: Storage, const F: u32> Mul<FixedMatrix<Fixed<S, F>>> for Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedMatrix<Fixed<S, F>>;
    fn mul(self, rhs: FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> {
        FixedMatrix::<Fixed<S, F>> {
            rows: rhs.rows,
            cols: rhs.cols,
            data: rhs.data.iter().map(|&v| self * v).collect(),
        }
    }
}

/// Matrix * Scalar
impl<S: Storage, const F: u32> Mul<Fixed<S, F>> for FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    fn mul(self, rhs: Fixed<S, F>) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| v * rhs).collect(),
        }
    }
}

/// &Matrix * Scalar
impl<S: Storage, const F: u32> Mul<Fixed<S, F>> for &FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = FixedMatrix<Fixed<S, F>>;
    fn mul(self, rhs: Fixed<S, F>) -> FixedMatrix<Fixed<S, F>> {
        FixedMatrix::<Fixed<S, F>> {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| v * rhs).collect(),
        }
    }
}

// ============================================================================
// L1A: Block operations
// ============================================================================

impl<S: Storage, const F: u32> FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Extract a submatrix starting at (row, col) with given dimensions.
    ///
    /// Panics if the submatrix extends beyond the matrix bounds.
    pub fn submatrix(&self, row: usize, col: usize, rows: usize, cols: usize) -> Self {
        assert!(row + rows <= self.rows && col + cols <= self.cols,
            "FixedMatrix::submatrix: extends beyond bounds");
        Self::from_fn(rows, cols, |r, c| self.get(row + r, col + c))
    }

    /// Insert a submatrix at position (row, col).
    ///
    /// Panics if the submatrix extends beyond the matrix bounds.
    pub fn set_submatrix(&mut self, row: usize, col: usize, sub: &FixedMatrix<Fixed<S, F>>) {
        assert!(row + sub.rows <= self.rows && col + sub.cols <= self.cols,
            "FixedMatrix::set_submatrix: extends beyond bounds");
        for r in 0..sub.rows {
            for c in 0..sub.cols {
                self.set(row + r, col + c, sub.get(r, c));
            }
        }
    }

    /// Kronecker product: A ⊗ B.
    ///
    /// Result is (A.rows * B.rows) × (A.cols * B.cols).
    pub fn kronecker(&self, other: &FixedMatrix<Fixed<S, F>>) -> Self {
        let out_rows = self.rows * other.rows;
        let out_cols = self.cols * other.cols;
        Self::from_fn(out_rows, out_cols, |r, c| {
            let ai = r / other.rows;
            let bi = r % other.rows;
            let aj = c / other.cols;
            let bj = c % other.cols;
            self.get(ai, aj) * other.get(bi, bj)
        })
    }

    /// Access the underlying data slice.
    #[inline]
    pub(crate) fn data_slice(&self) -> &[Fixed<S, F>] {
        &self.data
    }

    /// Swap rows `i` and `j` in place. O(cols) time, zero allocation.
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j { return; }
        assert!(i < self.rows && j < self.rows, "FixedMatrix::swap_rows: index out of bounds");
        for c in 0..self.cols {
            self.data.swap(i * self.cols + c, j * self.cols + c);
        }
    }

    /// Extract raw S values for row `r`, columns `col_start..col_end`.
    pub(crate) fn row_raw_range(&self, r: usize, col_start: usize, col_end: usize) -> Vec<S> {
        self.data[r * self.cols + col_start..r * self.cols + col_end]
            .iter().map(|fp| fp.raw()).collect()
    }
}

// ============================================================================
// PartialEq (not in original — needed for testing and correctness)
// ============================================================================

impl<T: PartialEq> PartialEq for FixedMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl<T> Default for FixedMatrix<T> {
    fn default() -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }
}
