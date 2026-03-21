//! FixedMatrix — row-major matrix of FixedPoint values.
//!
//! All matrix arithmetic uses compute-tier accumulation for precision:
//! matrix multiply, matrix-vector multiply accumulate at tier N+1 and
//! round once at the end.

use std::ops::{Add, Sub, Neg, Mul};
use super::FixedPoint;
use super::FixedVector;
use super::linalg::compute_tier_dot_raw;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;

/// A row-major matrix of fixed-point values.
///
/// Backed by a flat `Vec<FixedPoint>`. Not `Copy` due to heap allocation.
#[derive(Clone, Debug)]
pub struct FixedMatrix {
    rows: usize,
    cols: usize,
    data: Vec<FixedPoint>,
}

// ============================================================================
// Core constructors and accessors (existing API preserved exactly)
// ============================================================================

impl FixedMatrix {
    /// Create a zero-filled matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![FixedPoint::ZERO; rows * cols],
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
    pub fn get(&self, row: usize, col: usize) -> FixedPoint {
        assert!(row < self.rows && col < self.cols, "FixedMatrix: index out of bounds");
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    ///
    /// Panics if out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: FixedPoint) {
        assert!(row < self.rows && col < self.cols, "FixedMatrix: index out of bounds");
        self.data[row * self.cols + col] = value;
    }
}

// ============================================================================
// L1A: New constructors
// ============================================================================

impl FixedMatrix {
    /// Create from a flat slice of values in row-major order.
    ///
    /// Panics if `values.len() != rows * cols`.
    pub fn from_slice(rows: usize, cols: usize, values: &[FixedPoint]) -> Self {
        assert_eq!(values.len(), rows * cols,
            "FixedMatrix::from_slice: expected {} elements, got {}", rows * cols, values.len());
        Self {
            rows,
            cols,
            data: values.to_vec(),
        }
    }

    /// Create from a function: M[i][j] = f(i, j).
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> FixedPoint) -> Self {
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
            if r == c { FixedPoint::one() } else { FixedPoint::ZERO }
        })
    }

    /// Diagonal matrix from a vector.
    pub fn diagonal(v: &FixedVector) -> Self {
        let n = v.len();
        Self::from_fn(n, n, |r, c| {
            if r == c { v[r] } else { FixedPoint::ZERO }
        })
    }
}

// ============================================================================
// L1A: Matrix properties
// ============================================================================

impl FixedMatrix {
    /// Transpose: Aᵀ[i][j] = A[j][i].
    pub fn transpose(&self) -> Self {
        Self::from_fn(self.cols, self.rows, |r, c| self.get(c, r))
    }

    /// Trace: sum of diagonal elements.
    ///
    /// Panics if matrix is not square.
    pub fn trace(&self) -> FixedPoint {
        assert_eq!(self.rows, self.cols, "FixedMatrix::trace: matrix must be square");
        let mut sum = FixedPoint::ZERO;
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

    /// Extract a row as a FixedVector.
    pub fn row(&self, r: usize) -> FixedVector {
        assert!(r < self.rows, "FixedMatrix::row: index out of bounds");
        let start = r * self.cols;
        FixedVector::from_slice(&self.data[start..start + self.cols])
    }

    /// Extract a column as a FixedVector.
    pub fn col(&self, c: usize) -> FixedVector {
        assert!(c < self.cols, "FixedMatrix::col: index out of bounds");
        let mut v = FixedVector::new(self.rows);
        for r in 0..self.rows {
            v[r] = self.get(r, c);
        }
        v
    }
}

// ============================================================================
// L1A: Matrix-vector multiply (compute-tier accumulated)
// ============================================================================

impl FixedMatrix {
    /// Matrix-vector multiply: y = A * x.
    ///
    /// Uses compute-tier accumulation: each row dot product is computed
    /// at tier N+1 precision and rounded once.
    ///
    /// Panics if `self.cols() != v.len()`.
    pub fn mul_vector(&self, v: &FixedVector) -> FixedVector {
        assert_eq!(self.cols, v.len(),
            "FixedMatrix::mul_vector: cols ({}) != vector len ({})", self.cols, v.len());
        let v_raw: Vec<BinaryStorage> = v.as_slice().iter().map(|fp| fp.raw()).collect();
        let mut result = FixedVector::new(self.rows);
        for r in 0..self.rows {
            let row_start = r * self.cols;
            let row_raw: Vec<BinaryStorage> = self.data[row_start..row_start + self.cols]
                .iter().map(|fp| fp.raw()).collect();
            result[r] = FixedPoint::from_raw(compute_tier_dot_raw(&row_raw, &v_raw));
        }
        result
    }
}

// ============================================================================
// L1A: Matrix-matrix multiply (compute-tier accumulated)
// ============================================================================

impl Mul for FixedMatrix {
    type Output = Self;
    /// Matrix multiply: C = A * B.
    ///
    /// Uses compute-tier accumulation for each output element.
    /// Panics if `self.cols() != rhs.rows()`.
    fn mul(self, rhs: Self) -> Self {
        mat_mul(&self, &rhs)
    }
}

impl Mul for &FixedMatrix {
    type Output = FixedMatrix;
    fn mul(self, rhs: &FixedMatrix) -> FixedMatrix {
        mat_mul(self, rhs)
    }
}


fn mat_mul(a: &FixedMatrix, b: &FixedMatrix) -> FixedMatrix {
    assert_eq!(a.cols, b.rows,
        "FixedMatrix::mul: A.cols ({}) != B.rows ({})", a.cols, b.rows);
    let k = a.cols;
    // Pre-extract B columns as raw storage for efficient repeated access
    let b_cols_raw: Vec<Vec<BinaryStorage>> = (0..b.cols)
        .map(|c| (0..b.rows).map(|r| b.data[r * b.cols + c].raw()).collect())
        .collect();

    let mut result = FixedMatrix::new(a.rows, b.cols);
    for r in 0..a.rows {
        let row_start = r * k;
        let a_row_raw: Vec<BinaryStorage> = a.data[row_start..row_start + k]
            .iter().map(|fp| fp.raw()).collect();
        for c in 0..b.cols {
            let dot = compute_tier_dot_raw(&a_row_raw, &b_cols_raw[c]);
            result.data[r * b.cols + c] = FixedPoint::from_raw(dot);
        }
    }
    result
}

// ============================================================================
// L1A: Element-wise arithmetic
// ============================================================================

impl Add for FixedMatrix {
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

impl<'a, 'b> Add<&'b FixedMatrix> for &'a FixedMatrix {
    type Output = FixedMatrix;
    fn add(self, rhs: &'b FixedMatrix) -> FixedMatrix {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols),
            "FixedMatrix::add: dimension mismatch");
        FixedMatrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a + b).collect(),
        }
    }
}

impl Sub for FixedMatrix {
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

impl<'a, 'b> Sub<&'b FixedMatrix> for &'a FixedMatrix {
    type Output = FixedMatrix;
    fn sub(self, rhs: &'b FixedMatrix) -> FixedMatrix {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols),
            "FixedMatrix::sub: dimension mismatch");
        FixedMatrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().zip(rhs.data.iter())
                .map(|(&a, &b)| a - b).collect(),
        }
    }
}

impl Neg for FixedMatrix {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| -v).collect(),
        }
    }
}

impl Neg for &FixedMatrix {
    type Output = FixedMatrix;
    fn neg(self) -> FixedMatrix {
        FixedMatrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| -v).collect(),
        }
    }
}

/// Scalar * Matrix
impl Mul<FixedMatrix> for FixedPoint {
    type Output = FixedMatrix;
    fn mul(self, rhs: FixedMatrix) -> FixedMatrix {
        FixedMatrix {
            rows: rhs.rows,
            cols: rhs.cols,
            data: rhs.data.iter().map(|&v| self * v).collect(),
        }
    }
}

/// Matrix * Scalar
impl Mul<FixedPoint> for FixedMatrix {
    type Output = Self;
    fn mul(self, rhs: FixedPoint) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| v * rhs).collect(),
        }
    }
}

/// &Matrix * Scalar
impl Mul<FixedPoint> for &FixedMatrix {
    type Output = FixedMatrix;
    fn mul(self, rhs: FixedPoint) -> FixedMatrix {
        FixedMatrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| v * rhs).collect(),
        }
    }
}

// ============================================================================
// L1A: Block operations
// ============================================================================

impl FixedMatrix {
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
    pub fn set_submatrix(&mut self, row: usize, col: usize, sub: &FixedMatrix) {
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
    pub fn kronecker(&self, other: &FixedMatrix) -> Self {
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
    pub(crate) fn data_slice(&self) -> &[FixedPoint] {
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

    /// Extract raw BinaryStorage values for row `r`, columns `col_start..col_end`.
    pub(crate) fn row_raw_range(&self, r: usize, col_start: usize, col_end: usize) -> Vec<BinaryStorage> {
        self.data[r * self.cols + col_start..r * self.cols + col_end]
            .iter().map(|fp| fp.raw()).collect()
    }

    /// Extract raw BinaryStorage values for column `c`, rows `row_start..row_end`.
    pub(crate) fn col_raw_range(&self, c: usize, row_start: usize, row_end: usize) -> Vec<BinaryStorage> {
        (row_start..row_end).map(|r| self.data[r * self.cols + c].raw()).collect()
    }
}

// ============================================================================
// PartialEq (not in original — needed for testing and correctness)
// ============================================================================

impl PartialEq for FixedMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl Default for FixedMatrix {
    fn default() -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }
}
