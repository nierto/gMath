//! ComputeMatrix — matrix type operating entirely at compute tier (tier N+1).
//!
//! All operations (multiply, add, LU solve) stay at compute-tier precision.
//! Only downscaled to FixedMatrix via `to_fixed_matrix()` at the very end.
//! This gives 0-1 ULP for arbitrarily long matrix operation chains — the
//! same principle as FASC's BinaryCompute chain persistence for scalars.

use super::FixedPoint;
use super::FixedMatrix;
use super::linalg::{ComputeStorage, upscale_to_compute, round_to_storage};
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_add, compute_subtract, compute_negate, compute_multiply, compute_divide,
    compute_halve, compute_is_zero, compute_is_negative, sqrt_at_compute_tier,
};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// ComputeMatrix
// ============================================================================

/// Row-major matrix of compute-tier values (tier N+1 precision).
///
/// Used internally by matrix functions (exp, sqrt, log) to keep all
/// intermediate operations at double width. Downscale once at the end
/// via `to_fixed_matrix()` for 0-1 ULP final precision.
pub(crate) struct ComputeMatrix {
    rows: usize,
    cols: usize,
    data: Vec<ComputeStorage>,
}

fn compute_zero() -> ComputeStorage {
    upscale_to_compute(FixedPoint::ZERO.raw())
}

fn compute_one() -> ComputeStorage {
    upscale_to_compute(FixedPoint::one().raw())
}

impl ComputeMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let z = compute_zero();
        Self { rows, cols, data: vec![z; rows * cols] }
    }

    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> ComputeStorage) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self { rows, cols, data }
    }

    pub fn identity(n: usize) -> Self {
        let z = compute_zero();
        let o = compute_one();
        Self::from_fn(n, n, |r, c| if r == c { o } else { z })
    }

    /// Upscale a FixedMatrix to compute tier.
    pub fn from_fixed_matrix(m: &FixedMatrix) -> Self {
        Self::from_fn(m.rows(), m.cols(), |r, c| upscale_to_compute(m.get(r, c).raw()))
    }

    /// Downscale to FixedMatrix — single rounding per element (0-1 ULP).
    pub fn to_fixed_matrix(&self) -> FixedMatrix {
        FixedMatrix::from_fn(self.rows, self.cols, |r, c| {
            FixedPoint::from_raw(round_to_storage(self.get(r, c)))
        })
    }

    #[inline]
    pub fn rows(&self) -> usize { self.rows }
    #[inline]
    #[allow(dead_code)]
    pub fn cols(&self) -> usize { self.cols }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> ComputeStorage {
        self.data[row * self.cols + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: ComputeStorage) {
        self.data[row * self.cols + col] = val;
    }

    pub fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j { return; }
        for c in 0..self.cols {
            self.data.swap(i * self.cols + c, j * self.cols + c);
        }
    }

    // ── Arithmetic ──

    pub fn add(&self, other: &Self) -> Self {
        Self::from_fn(self.rows, self.cols, |r, c|
            compute_add(self.get(r, c), other.get(r, c)))
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::from_fn(self.rows, self.cols, |r, c|
            compute_subtract(self.get(r, c), other.get(r, c)))
    }

    pub fn neg(&self) -> Self {
        Self::from_fn(self.rows, self.cols, |r, c|
            compute_negate(self.get(r, c)))
    }

    pub fn scalar_mul(&self, s: ComputeStorage) -> Self {
        Self::from_fn(self.rows, self.cols, |r, c|
            compute_multiply(self.get(r, c), s))
    }

    pub fn halve(&self) -> Self {
        Self::from_fn(self.rows, self.cols, |r, c|
            compute_halve(self.get(r, c)))
    }

    /// Matrix multiply at compute tier.
    pub fn mat_mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let k = self.cols;
        Self::from_fn(self.rows, other.cols, |r, c| {
            let mut acc = compute_zero();
            for m in 0..k {
                acc = compute_add(acc, compute_multiply(self.get(r, m), other.get(m, c)));
            }
            acc
        })
    }

    /// Extract column as Vec<ComputeStorage>.
    pub fn col_vec(&self, c: usize) -> Vec<ComputeStorage> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    /// Matrix-vector multiply at compute tier. Returns compute-tier result vector.
    /// Both matrix and vector stay at tier N+1 — no mid-chain downscale.
    pub fn mul_vector_compute(&self, v: &[ComputeStorage]) -> Vec<ComputeStorage> {
        assert_eq!(self.cols, v.len());
        (0..self.rows).map(|r| {
            let mut acc = compute_zero();
            for c in 0..self.cols {
                acc = compute_add(acc, compute_multiply(self.get(r, c), v[c]));
            }
            acc
        }).collect()
    }

    /// Frobenius norm computed at tier N+1: sqrt(sum(a_ij²)).
    /// Sum-of-squares and sqrt at compute tier, single downscale at end.
    pub fn frobenius_norm_compute(&self) -> FixedPoint {
        let mut sum_sq = compute_zero();
        for r in 0..self.rows {
            for c in 0..self.cols {
                let v = self.get(r, c);
                sum_sq = compute_add(sum_sq, compute_multiply(v, v));
            }
        }
        FixedPoint::from_raw(round_to_storage(sqrt_at_compute_tier(sum_sq)))
    }

    /// 1-norm computed at tier N+1: max absolute column sum.
    /// Single downscale at end for comparison.
    pub fn norm_1_compute(&self) -> FixedPoint {
        let mut max_col_sum = compute_zero();
        for c in 0..self.cols {
            let mut col_sum = compute_zero();
            for r in 0..self.rows {
                col_sum = compute_add(col_sum, compute_abs(self.get(r, c)));
            }
            if col_sum > max_col_sum {
                max_col_sum = col_sum;
            }
        }
        FixedPoint::from_raw(round_to_storage(max_col_sum))
    }

    /// Copy this matrix (all data at compute tier).
    pub fn copy(&self) -> Self {
        Self::from_fn(self.rows, self.cols, |r, c| self.get(r, c))
    }

    /// Transpose at compute tier — no downscale.
    pub fn transpose(&self) -> Self {
        Self::from_fn(self.cols, self.rows, |r, c| self.get(c, r))
    }

    /// Trace at compute tier: sum of diagonal elements, single downscale at end.
    pub fn trace_compute(&self) -> FixedPoint {
        assert_eq!(self.rows, self.cols, "trace: matrix must be square");
        let mut acc = compute_zero();
        for i in 0..self.rows {
            acc = compute_add(acc, self.get(i, i));
        }
        FixedPoint::from_raw(round_to_storage(acc))
    }
}

// ============================================================================
// Compute-tier LU decomposition
// ============================================================================

pub(crate) struct ComputeLU {
    l: ComputeMatrix,
    u: ComputeMatrix,
    perm: Vec<usize>,
}

/// Compute-tier sub-dot: init - sum(a[i] * b[i])
fn compute_sub_dot(init: ComputeStorage, a: &[ComputeStorage], b: &[ComputeStorage]) -> ComputeStorage {
    let mut acc = init;
    for i in 0..a.len() {
        acc = compute_subtract(acc, compute_multiply(a[i], b[i]));
    }
    acc
}

fn compute_abs(a: ComputeStorage) -> ComputeStorage {
    if compute_is_negative(&a) { compute_negate(a) } else { a }
}

pub(crate) fn compute_lu_decompose(a: &ComputeMatrix) -> Result<ComputeLU, OverflowDetected> {
    let n = a.rows();
    let mut pa = ComputeMatrix::from_fn(n, n, |r, c| a.get(r, c));
    let mut l = ComputeMatrix::new(n, n);
    let mut u = ComputeMatrix::new(n, n);
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Pivoting: find max |candidate| for rows k..n
        let mut max_abs = compute_zero();
        let mut max_row = k;
        for i in k..n {
            let candidate = if k == 0 {
                pa.get(i, k)
            } else {
                let l_row: Vec<ComputeStorage> = (0..k).map(|j| l.get(i, j)).collect();
                let u_col: Vec<ComputeStorage> = (0..k).map(|j| u.get(j, k)).collect();
                compute_sub_dot(pa.get(i, k), &l_row, &u_col)
            };
            let abs_c = compute_abs(candidate);
            if abs_c > max_abs {
                max_abs = abs_c;
                max_row = i;
            }
        }

        if compute_is_zero(&max_abs) {
            return Err(OverflowDetected::DivisionByZero);
        }

        if max_row != k {
            pa.swap_rows(k, max_row);
            perm.swap(k, max_row);
            for j in 0..k {
                let tmp = l.get(k, j);
                l.set(k, j, l.get(max_row, j));
                l.set(max_row, j, tmp);
            }
        }

        // U row k
        for j in k..n {
            let val = if k == 0 {
                pa.get(k, j)
            } else {
                let l_row: Vec<ComputeStorage> = (0..k).map(|m| l.get(k, m)).collect();
                let u_col: Vec<ComputeStorage> = (0..k).map(|m| u.get(m, j)).collect();
                compute_sub_dot(pa.get(k, j), &l_row, &u_col)
            };
            u.set(k, j, val);
        }

        // L column k
        let pivot = u.get(k, k);
        l.set(k, k, compute_one());
        for i in (k + 1)..n {
            let numerator = if k == 0 {
                pa.get(i, k)
            } else {
                let l_row: Vec<ComputeStorage> = (0..k).map(|m| l.get(i, m)).collect();
                let u_col: Vec<ComputeStorage> = (0..k).map(|m| u.get(m, k)).collect();
                compute_sub_dot(pa.get(i, k), &l_row, &u_col)
            };
            l.set(i, k, compute_divide(numerator, pivot)?);
        }
    }

    Ok(ComputeLU { l, u, perm })
}

impl ComputeLU {
    /// Solve Ax = b at compute tier. Returns compute-tier solution vector.
    pub fn solve(&self, b: &[ComputeStorage]) -> Result<Vec<ComputeStorage>, OverflowDetected> {
        let n = self.l.rows();
        let pb: Vec<ComputeStorage> = (0..n).map(|i| b[self.perm[i]]).collect();

        // Forward: Ly = pb
        let mut y = vec![compute_zero(); n];
        for i in 0..n {
            if i == 0 {
                y[0] = pb[0];
            } else {
                let l_row: Vec<ComputeStorage> = (0..i).map(|j| self.l.get(i, j)).collect();
                let y_prev: Vec<ComputeStorage> = y[0..i].to_vec();
                y[i] = compute_sub_dot(pb[i], &l_row, &y_prev);
            }
        }

        // Back: Ux = y
        let mut x = vec![compute_zero(); n];
        for i in (0..n).rev() {
            let diag = self.u.get(i, i);
            if compute_is_zero(&diag) {
                return Err(OverflowDetected::DivisionByZero);
            }
            if i == n - 1 {
                x[n - 1] = compute_divide(y[n - 1], diag)?;
            } else {
                let u_row: Vec<ComputeStorage> = (i + 1..n).map(|j| self.u.get(i, j)).collect();
                let x_tail: Vec<ComputeStorage> = x[i + 1..n].to_vec();
                let numerator = compute_sub_dot(y[i], &u_row, &x_tail);
                x[i] = compute_divide(numerator, diag)?;
            }
        }

        Ok(x)
    }

    /// Solve for full inverse at compute tier.
    pub fn inverse(&self) -> Result<ComputeMatrix, OverflowDetected> {
        let n = self.l.rows();
        let mut inv = ComputeMatrix::new(n, n);
        for j in 0..n {
            let mut e_j = vec![compute_zero(); n];
            e_j[j] = compute_one();
            let col = self.solve(&e_j)?;
            for i in 0..n {
                inv.set(i, j, col[i]);
            }
        }
        Ok(inv)
    }
}
