//! FixedMatrix — row-major matrix of FixedPoint values.

use super::FixedPoint;

/// A row-major matrix of fixed-point values.
///
/// Backed by a flat `Vec<FixedPoint>`. Not `Copy` due to heap allocation.
#[derive(Clone, Debug)]
pub struct FixedMatrix {
    rows: usize,
    cols: usize,
    data: Vec<FixedPoint>,
}

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

impl Default for FixedMatrix {
    fn default() -> Self {
        Self {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }
}
