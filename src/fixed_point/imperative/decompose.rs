//! Matrix decompositions: LU, QR, Cholesky, Eigenvalue (Jacobi), SVD, Schur.
//!
//! All decompositions use compute-tier accumulation (tier N+1) for
//! precision-critical inner sums (elimination, substitution, Cholesky diag).
//!
//! Non-iterative: LU, QR, Cholesky.
//! Iterative: Jacobi symmetric eigenvalue, Golub-Kahan SVD, Francis QR Schur.

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::interval::exact_product;
use super::linalg::{
    compute_tier_dot_raw, compute_tier_sub_dot_raw, compute_tier_sub_dot_compute,
    upscale_to_compute, round_to_storage, compute_abs, deflation_threshold, exact_dot,
    householder_vector, noise_floor, reflect, scale_by, stagnation_threshold, ComputeStorage,
    Rotation, STAGNATION_SWEEPS,
};
use super::wide_acc::widen_storage;
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    sqrt_at_compute_tier, compute_divide, downscale_to_storage,
    compute_multiply, compute_add, compute_negate,
    compute_is_negative, compute_is_zero,
    compute_checked_add, compute_checked_divide, compute_halve, make_compute_int,
};
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// LU Decomposition with Partial Pivoting
// ============================================================================

/// Result of LU decomposition with partial pivoting: PA = LU.
///
/// - `l` is unit lower triangular (diagonal = 1.0, stored explicitly)
/// - `u` is upper triangular
/// - `perm` is the permutation vector: row `i` of PA came from row `perm[i]` of A
/// - `num_swaps` tracks parity for determinant sign
#[derive(Clone, Debug)]
pub struct LUDecomposition {
    pub l: FixedMatrix,
    pub u: FixedMatrix,
    pub perm: Vec<usize>,
    pub num_swaps: usize,
}

/// LU decomposition with partial pivoting (Doolittle, compute-tier).
///
/// For an n×n matrix A, computes PA = LU where P is a permutation,
/// L is unit lower triangular, and U is upper triangular.
///
/// **Precision strategy:** Uses Doolittle direct formulas where each entry is
/// computed via `compute_tier_sub_dot_raw`: the entire inner sum accumulates
/// at tier N+1, rounding once. This gives 1 ULP per entry regardless of matrix
/// size, instead of the O(n) ULP that incremental elimination produces.
///
/// Returns `Err(DivisionByZero)` if the matrix is singular.
pub fn lu_decompose(a: &FixedMatrix) -> Result<LUDecomposition, OverflowDetected> {
    assert!(a.is_square(), "lu_decompose: matrix must be square");
    let n = a.rows();

    // Work on permuted copy of A
    let mut pa = a.clone();
    let mut l = FixedMatrix::new(n, n);
    let mut u = FixedMatrix::new(n, n);
    let mut perm: Vec<usize> = (0..n).collect();
    let mut num_swaps: usize = 0;

    for k in 0..n {
        // ── Partial pivoting ──
        // Compute candidate U[k][k] for each remaining row to find best pivot.
        // U[k][k] = PA[k][k] - SUM(L[k][m] * U[m][k], m=0..k-1)
        let mut max_abs = FixedPoint::ZERO;
        let mut max_row = k;
        for i in k..n {
            let candidate = if k == 0 {
                pa.get(i, k)
            } else {
                let l_row = l.row_raw_range(i, 0, k);
                let u_col = u.col_raw_range(k, 0, k);
                FixedPoint::from_raw(compute_tier_sub_dot_raw(pa.get(i, k).raw(), &l_row, &u_col))
            };
            if candidate.abs() > max_abs {
                max_abs = candidate.abs();
                max_row = i;
            }
        }

        if max_abs.is_zero() {
            return Err(OverflowDetected::DivisionByZero);
        }

        // Row swap in PA and L (already-computed columns)
        if max_row != k {
            pa.swap_rows(k, max_row);
            perm.swap(k, max_row);
            num_swaps += 1;
            for j in 0..k {
                let tmp = l.get(k, j);
                l.set(k, j, l.get(max_row, j));
                l.set(max_row, j, tmp);
            }
        }

        // ── U row k: U[k][j] = PA[k][j] - SUM(L[k][m] * U[m][j], m=0..k-1) ──
        // Each entry computed via compute_tier_sub_dot_raw → 1 ULP
        for j in k..n {
            if k == 0 {
                u.set(k, j, pa.get(k, j));
            } else {
                let l_row = l.row_raw_range(k, 0, k);
                let u_col = u.col_raw_range(j, 0, k);
                u.set(k, j, FixedPoint::from_raw(
                    compute_tier_sub_dot_raw(pa.get(k, j).raw(), &l_row, &u_col)
                ));
            }
        }

        // ── L column k: L[i][k] = (PA[i][k] - SUM(L[i][m] * U[m][k], m=0..k-1)) / U[k][k] ──
        // Each entry: compute_tier_sub_dot_raw (1 ULP) + division (1 ULP) = 2 ULP max
        let pivot = u.get(k, k);
        l.set(k, k, FixedPoint::one()); // Unit diagonal
        for i in (k + 1)..n {
            let numerator = if k == 0 {
                pa.get(i, k)
            } else {
                let l_row = l.row_raw_range(i, 0, k);
                let u_col = u.col_raw_range(k, 0, k);
                FixedPoint::from_raw(compute_tier_sub_dot_raw(pa.get(i, k).raw(), &l_row, &u_col))
            };
            l.set(i, k, numerator / pivot);
        }
    }

    Ok(LUDecomposition { l, u, perm, num_swaps })
}

impl LUDecomposition {
    /// Solve Ax = b using forward then back substitution.
    ///
    /// Inner sums use compute-tier accumulation for maximum precision.
    pub fn solve(&self, b: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let n = self.l.rows();
        assert_eq!(b.len(), n, "LU solve: dimension mismatch");

        // Apply permutation
        let mut pb = FixedVector::new(n);
        for i in 0..n {
            pb[i] = b[self.perm[i]];
        }

        // Forward substitution: Ly = pb (L is unit lower triangular)
        let mut y = FixedVector::new(n);
        for i in 0..n {
            if i == 0 {
                y[0] = pb[0];
            } else {
                let l_row = self.l.row_raw_range(i, 0, i);
                let y_raw: Vec<BinaryStorage> = (0..i).map(|j| y[j].raw()).collect();
                y[i] = FixedPoint::from_raw(
                    compute_tier_sub_dot_raw(pb[i].raw(), &l_row, &y_raw)
                );
            }
        }

        // Back substitution: Ux = y
        let mut x = FixedVector::new(n);
        for i in (0..n).rev() {
            let diag = self.u.get(i, i);
            if diag.is_zero() {
                return Err(OverflowDetected::DivisionByZero);
            }
            if i == n - 1 {
                x[n - 1] = y[n - 1] / diag;
            } else {
                let u_row = self.u.row_raw_range(i, i + 1, n);
                let x_raw: Vec<BinaryStorage> = (i + 1..n).map(|j| x[j].raw()).collect();
                let numerator = FixedPoint::from_raw(
                    compute_tier_sub_dot_raw(y[i].raw(), &u_row, &x_raw)
                );
                x[i] = numerator / diag;
            }
        }

        Ok(x)
    }

    /// Determinant: det(A) = (-1)^num_swaps * product(U diagonal).
    ///
    /// Product accumulated at compute tier: single downscale at the end.
    pub fn determinant(&self) -> FixedPoint {
        let n = self.u.rows();
        // Multiply all diagonal values at compute tier, downscale once
        use crate::fixed_point::universal::fasc::stack_evaluator::compute::compute_multiply;
        let mut acc = upscale_to_compute(self.u.get(0, 0).raw());
        for i in 1..n {
            acc = compute_multiply(acc, upscale_to_compute(self.u.get(i, i).raw()));
        }
        let det_raw = round_to_storage(acc);
        let det = FixedPoint::from_raw(det_raw);
        if self.num_swaps % 2 == 1 { -det } else { det }
    }

    /// Iterative refinement: improve solution accuracy by computing residual
    /// at compute tier and correcting.
    ///
    /// One step typically reduces error from O(κ) ULP to O(1) ULP.
    /// For ill-conditioned systems (Hilbert etc.), this is the difference
    /// between millions of ULP and single-digit ULP.
    pub fn refine(&self, a: &FixedMatrix, b: &FixedVector, x: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let n = a.rows();
        // Compute residual r = b - Ax at compute tier (high precision)
        let mut r = FixedVector::new(n);
        for i in 0..n {
            let a_row = a.row_raw_range(i, 0, n);
            let x_raw: Vec<BinaryStorage> = (0..n).map(|j| x[j].raw()).collect();
            r[i] = FixedPoint::from_raw(
                compute_tier_sub_dot_raw(b[i].raw(), &a_row, &x_raw)
            );
        }
        // Solve A*dx = r using existing factorization
        let dx = self.solve(&r)?;
        // x_refined = x + dx
        let mut x_refined = FixedVector::new(n);
        for i in 0..n {
            x_refined[i] = x[i] + dx[i];
        }
        Ok(x_refined)
    }

    /// Compute A^{-1} by solving AX = I column by column.
    pub fn inverse(&self) -> Result<FixedMatrix, OverflowDetected> {
        let n = self.l.rows();
        let mut inv = FixedMatrix::new(n, n);
        for j in 0..n {
            let mut e_j = FixedVector::new(n);
            e_j[j] = FixedPoint::one();
            let col = self.solve(&e_j)?;
            for i in 0..n {
                inv.set(i, j, col[i]);
            }
        }
        Ok(inv)
    }
}

// ============================================================================
// QR Decomposition via Householder Reflections
// ============================================================================

/// Result of QR decomposition via Householder reflections: A = QR.
#[derive(Clone, Debug)]
pub struct QRDecomposition {
    pub q: FixedMatrix,
    pub r: FixedMatrix,
}

/// QR decomposition via Householder reflections.
///
/// For an m×n matrix A (m >= n), computes A = QR where Q is m×m orthogonal
/// and R is m×n upper triangular.
///
/// All column norms and reflection dot products use compute-tier accumulation.
pub fn qr_decompose(a: &FixedMatrix) -> Result<QRDecomposition, OverflowDetected> {
    let m = a.rows();
    let n = a.cols();
    assert!(m >= n, "qr_decompose: requires m >= n");

    let mut r = a.clone();
    let mut q = FixedMatrix::identity(m);
    let two = FixedPoint::from_int(2);

    for k in 0..n {
        let col_len = m - k;

        // Extract column x = R[k..m, k] as raw storage
        let x_raw: Vec<BinaryStorage> = (k..m).map(|i| r.get(i, k).raw()).collect();

        // ||x||^2 via compute-tier dot
        let norm_sq = FixedPoint::from_raw(compute_tier_dot_raw(&x_raw, &x_raw));
        if norm_sq.is_zero() {
            continue;
        }
        let norm_x = norm_sq.try_sqrt()?;

        // Sign choice: alpha = -sign(x_0) * ||x|| (avoids cancellation in v[0])
        let x_0 = r.get(k, k);
        let alpha = if x_0.is_negative() { norm_x } else { -norm_x };

        // Householder vector: v = x - alpha*e_1 → v[0] = x_0 - alpha, v[i] = x[i]
        let mut v = Vec::<FixedPoint>::with_capacity(col_len);
        v.push(x_0 - alpha);
        for i in 1..col_len {
            v.push(FixedPoint::from_raw(x_raw[i]));
        }
        let v_raw: Vec<BinaryStorage> = v.iter().map(|fp| fp.raw()).collect();

        // v^T v via compute-tier
        let vtv = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &v_raw));
        if vtv.is_zero() {
            continue;
        }

        // Apply H to R: R[k..m, k..n] -= 2 * v * (v^T * R[k..m, j]) / vtv
        for j in k..n {
            let col_j_raw: Vec<BinaryStorage> = (k..m).map(|i| r.get(i, j).raw()).collect();
            let vt_rj = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &col_j_raw));
            let scale = two * vt_rj / vtv;
            for i in k..m {
                let r_ij = r.get(i, j);
                r.set(i, j, r_ij - scale * v[i - k]);
            }
        }

        // Apply H to Q: Q[:, k..m] *= H → Q[i, j] -= scale_i * v[j-k]
        for i in 0..m {
            let q_row_raw: Vec<BinaryStorage> = (k..m).map(|j| q.get(i, j).raw()).collect();
            let qi_dot_v = FixedPoint::from_raw(compute_tier_dot_raw(&q_row_raw, &v_raw));
            let scale = two * qi_dot_v / vtv;
            for j in k..m {
                let q_ij = q.get(i, j);
                q.set(i, j, q_ij - scale * v[j - k]);
            }
        }
    }

    Ok(QRDecomposition { q, r })
}

impl QRDecomposition {
    /// Solve Ax = b via R^{-1} Q^T b (back substitution).
    pub fn solve(&self, b: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let m = self.q.rows();
        let n = self.r.cols();
        assert_eq!(b.len(), m, "QR solve: dimension mismatch");

        // Compute Q^T b via compute-tier dot products
        let mut qtb = FixedVector::new(m);
        for i in 0..m {
            let q_col_raw: Vec<BinaryStorage> = (0..m).map(|j| self.q.get(j, i).raw()).collect();
            let b_raw: Vec<BinaryStorage> = (0..m).map(|j| b[j].raw()).collect();
            qtb[i] = FixedPoint::from_raw(compute_tier_dot_raw(&q_col_raw, &b_raw));
        }

        // Back substitution on R (n×n upper triangular part)
        let mut x = FixedVector::new(n);
        for i in (0..n).rev() {
            let diag = self.r.get(i, i);
            if diag.is_zero() {
                return Err(OverflowDetected::DivisionByZero);
            }
            if i == n - 1 {
                x[n - 1] = qtb[n - 1] / diag;
            } else {
                let r_row = self.r.row_raw_range(i, i + 1, n);
                let x_raw: Vec<BinaryStorage> = (i + 1..n).map(|j| x[j].raw()).collect();
                let numerator = FixedPoint::from_raw(
                    compute_tier_sub_dot_raw(qtb[i].raw(), &r_row, &x_raw)
                );
                x[i] = numerator / diag;
            }
        }

        Ok(x)
    }
}

// ============================================================================
// Cholesky Decomposition (A = LL^T for SPD matrices)
// ============================================================================

/// Result of Cholesky decomposition: A = LL^T.
///
/// `l` is lower triangular with positive diagonal entries.
#[derive(Clone, Debug)]
pub struct CholeskyDecomposition {
    pub l: FixedMatrix,
}

/// Cholesky decomposition for symmetric positive-definite matrices.
///
/// Returns `Err(DomainError)` if the matrix is not positive-definite.
///
/// **Precision strategy:** Uses fused compute-tier operations throughout:
/// - Diagonal: `sqrt(A[i][i] - dot(L_row, L_row))` computed entirely at tier N+1,
///   single downscale at the end → 0-1 ULP per entry.
/// - Off-diagonal: `(A[j][i] - dot(L_j, L_i)) / L[i][i]` with the sub_dot at
///   tier N+1 fed directly into compute_divide, single downscale → 0-1 ULP.
pub fn cholesky_decompose(a: &FixedMatrix) -> Result<CholeskyDecomposition, OverflowDetected> {
    assert!(a.is_square(), "cholesky_decompose: matrix must be square");
    let n = a.rows();
    let mut l = FixedMatrix::new(n, n);

    for i in 0..n {
        // Diagonal: L[i][i] = sqrt(A[i][i] - SUM L[i][k]^2)
        // FUSED at compute tier: sub_dot → sqrt → downscale (single rounding)
        let diag_compute = if i == 0 {
            upscale_to_compute(a.get(0, 0).raw())
        } else {
            let l_row = l.row_raw_range(i, 0, i);
            compute_tier_sub_dot_compute(a.get(i, i).raw(), &l_row, &l_row)
        };

        // Check positive-definiteness at compute tier (before sqrt)
        if compute_is_negative(&diag_compute) || compute_is_zero(&diag_compute) {
            return Err(OverflowDetected::DomainError);
        }

        // sqrt at compute tier, then single downscale → 0-1 ULP
        let sqrt_compute = sqrt_at_compute_tier(diag_compute);
        let l_ii_raw = downscale_to_storage(sqrt_compute)
            .map_err(|_| OverflowDetected::TierOverflow)?;
        let l_ii = FixedPoint::from_raw(l_ii_raw);
        l.set(i, i, l_ii);

        // Off-diagonal: L[j][i] = (A[j][i] - SUM L[j][k]*L[i][k]) / L[i][i]
        // FUSED: sub_dot at compute tier → divide at compute tier → downscale
        let l_ii_compute = upscale_to_compute(l_ii.raw());
        for j in (i + 1)..n {
            let numerator_compute = if i == 0 {
                upscale_to_compute(a.get(j, i).raw())
            } else {
                let l_j_row = l.row_raw_range(j, 0, i);
                let l_i_row = l.row_raw_range(i, 0, i);
                compute_tier_sub_dot_compute(a.get(j, i).raw(), &l_j_row, &l_i_row)
            };
            let quotient_compute = compute_divide(numerator_compute, l_ii_compute)
                .map_err(|_| OverflowDetected::DivisionByZero)?;
            let l_ji_raw = downscale_to_storage(quotient_compute)
                .map_err(|_| OverflowDetected::TierOverflow)?;
            l.set(j, i, FixedPoint::from_raw(l_ji_raw));
        }
    }

    Ok(CholeskyDecomposition { l })
}

impl CholeskyDecomposition {
    /// Solve Ax = b: forward (Ly = b), then back (L^T x = y).
    pub fn solve(&self, b: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let n = self.l.rows();
        assert_eq!(b.len(), n, "Cholesky solve: dimension mismatch");

        // Forward: Ly = b
        let mut y = FixedVector::new(n);
        for i in 0..n {
            let diag = self.l.get(i, i);
            if i == 0 {
                y[0] = b[0] / diag;
            } else {
                let l_row = self.l.row_raw_range(i, 0, i);
                let y_raw: Vec<BinaryStorage> = (0..i).map(|j| y[j].raw()).collect();
                let numerator = FixedPoint::from_raw(
                    compute_tier_sub_dot_raw(b[i].raw(), &l_row, &y_raw)
                );
                y[i] = numerator / diag;
            }
        }

        // Back: L^T x = y (L^T[i][j] = L[j][i])
        let mut x = FixedVector::new(n);
        for i in (0..n).rev() {
            let diag = self.l.get(i, i);
            if i == n - 1 {
                x[n - 1] = y[n - 1] / diag;
            } else {
                let lt_row = self.l.col_raw_range(i, i + 1, n);
                let x_raw: Vec<BinaryStorage> = (i + 1..n).map(|j| x[j].raw()).collect();
                let numerator = FixedPoint::from_raw(
                    compute_tier_sub_dot_raw(y[i].raw(), &lt_row, &x_raw)
                );
                x[i] = numerator / diag;
            }
        }

        Ok(x)
    }

    /// Determinant: det(A) = product(L[i][i])^2.
    pub fn determinant(&self) -> FixedPoint {
        let n = self.l.rows();
        let mut det_l = FixedPoint::one();
        for i in 0..n {
            det_l = det_l * self.l.get(i, i);
        }
        det_l * det_l
    }
}

// ============================================================================
// Shared kernels of the iterative decompositions
// ============================================================================
//
// Jacobi, Golub-Kahan and Francis converge only if the orthogonal transforms
// they apply inject less rounding noise than their convergence tests resolve.
// A coefficient rounded to storage precision injects about |x| ulp into every
// entry it touches, so every coefficient here stays at the compute tier and
// every transformed entry is narrowed once, from an exact accumulator
// (`Rotation`, `householder_vector` and `reflect` in `linalg`). Every step is
// checked: leaving the storage range is a `TierOverflow`, never a wrap.

/// Iteration budget of the QR-type iterations: this many steps per n².
const ITERATIONS_PER_N_SQUARED: usize = 30;

/// Sweep budget of the Jacobi iteration.
const JACOBI_MAX_SWEEPS: usize = 100;

/// Francis iterations on one block, after its two exceptional shifts, before
/// the block is taken to sit at the precision floor.
const SCHUR_FLOOR_ITERATIONS: usize = 30;

/// Reflect column `col` of `mat`, rows `start..start + v.len()`, in the
/// hyperplane orthogonal to `v`.
fn reflect_column(
    mat: &mut FixedMatrix, col: usize, start: usize, v: &[BinaryStorage], v_dot_v: ComputeStorage,
) -> Result<(), OverflowDetected> {
    let mut w: Vec<BinaryStorage> = (start..start + v.len()).map(|i| mat.get(i, col).raw()).collect();
    reflect(&mut w, v, v_dot_v)?;
    for (k, value) in w.into_iter().enumerate() {
        mat.set(start + k, col, FixedPoint::from_raw(value));
    }
    Ok(())
}

/// Reflect row `row` of `mat`, columns `start..start + v.len()`, in the
/// hyperplane orthogonal to `v`.
fn reflect_row(
    mat: &mut FixedMatrix, row: usize, start: usize, v: &[BinaryStorage], v_dot_v: ComputeStorage,
) -> Result<(), OverflowDetected> {
    let mut w: Vec<BinaryStorage> = (start..start + v.len()).map(|c| mat.get(row, c).raw()).collect();
    reflect(&mut w, v, v_dot_v)?;
    for (k, value) in w.into_iter().enumerate() {
        mat.set(row, start + k, FixedPoint::from_raw(value));
    }
    Ok(())
}

/// Rotate columns `a` and `b` of every row:
/// `(M_a, M_b) <- (cs M_a + sn M_b, -sn M_a + cs M_b)`.
fn rotate_columns(mat: &mut FixedMatrix, a: usize, b: usize, rot: &Rotation) -> Result<(), OverflowDetected> {
    for r in 0..mat.rows() {
        let (new_a, new_b) = rot.apply(mat.get(r, a), mat.get(r, b))?;
        mat.set(r, a, new_a);
        mat.set(r, b, new_b);
    }
    Ok(())
}

fn checked_add_fp(a: FixedPoint, b: FixedPoint) -> Result<FixedPoint, OverflowDetected> {
    a.raw().checked_add(b.raw()).map(FixedPoint::from_raw).ok_or(OverflowDetected::TierOverflow)
}

fn checked_sub_fp(a: FixedPoint, b: FixedPoint) -> Result<FixedPoint, OverflowDetected> {
    a.raw().checked_sub(b.raw()).map(FixedPoint::from_raw).ok_or(OverflowDetected::TierOverflow)
}

fn compute_sum(terms: &[ComputeStorage]) -> Result<ComputeStorage, OverflowDetected> {
    let mut acc = make_compute_int(0);
    for term in terms {
        acc = compute_checked_add(acc, *term)?;
    }
    Ok(acc)
}

/// `sqrt(a^2 + b^2)` of two compute raws in ratio form: neither is squared.
fn compute_hypot(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    let (x, y) = (compute_abs(a), compute_abs(b));
    let (big, small) = if x >= y { (x, y) } else { (y, x) };
    if compute_is_zero(&big) {
        return Ok(big);
    }
    let one = make_compute_int(1);
    let ratio = compute_divide(small, big)?;
    Ok(compute_multiply(big, sqrt_at_compute_tier(compute_add(one, compute_multiply(ratio, ratio)))))
}

// ============================================================================
// Symmetric Eigenvalue Decomposition (Jacobi Method)
// ============================================================================

/// Result of symmetric eigenvalue decomposition: A = Q Λ Qᵀ.
///
/// - `values` contains eigenvalues (diagonal of Λ), sorted descending by absolute value
/// - `vectors` is orthogonal (Qᵀ Q = I), columns are eigenvectors
#[derive(Clone, Debug)]
pub struct EigenDecomposition {
    pub values: FixedVector,
    pub vectors: FixedMatrix,
}

/// Symmetric eigenvalue decomposition via the classical Jacobi method.
///
/// **Why Jacobi for fixed-point:** each rotation zeroes one off-diagonal pair
/// outright and later rotations absorb the rounding earlier ones left behind,
/// so the method reaches the rounding floor without a shift strategy.
///
/// **Algorithm:**
/// 1. Cyclic-by-row sweeps: every (p,q) with p<q whose entry exceeds its bound
///    is zeroed by a rotation (and A[q][p] by symmetry).
/// 2. The rotation angle comes from the quadratic formula, no trig: `t = tan θ`
///    is the smaller root, formed without squaring τ when |τ| > 1. The diagonal
///    is updated in Rutishauser's form `a_pp + t a_pq`, `a_qq - t a_pq`.
/// 3. Converged when a whole sweep finds every off-diagonal entry within the
///    tight relative bound of its two diagonal entries, floored at four quanta.
///    The absolute floor matters: an exact zero eigenvalue pair is computed as
///    rounding noise, which a purely relative test never passes. A run whose
///    largest off-diagonal entry has not decreased for five sweeps has reached
///    the precision floor and is accepted only if every entry is within the
///    looser sqrt(quantum) relative bound.
///
/// **Precision:** rotation coefficients stay at the compute tier; every updated
/// entry is narrowed once from an exact accumulator. No entry is squared at
/// storage precision.
///
/// **Errors:** `Err(PrecisionLimit)` if neither criterion is met within 100
/// sweeps (never a partially converged result); `Err(TierOverflow)` if an entry
/// leaves the storage range. Panics if the matrix is not square. The matrix
/// must be symmetric; that is not checked.
pub fn eigen_symmetric(a: &FixedMatrix) -> Result<EigenDecomposition, OverflowDetected> {
    eigen_symmetric_within(a, JACOBI_MAX_SWEEPS)
}

fn eigen_symmetric_within(a: &FixedMatrix, max_sweeps: usize) -> Result<EigenDecomposition, OverflowDetected> {
    assert!(a.is_square(), "eigen_symmetric: matrix must be square");
    let n = a.rows();

    if n == 0 {
        return Ok(EigenDecomposition {
            values: FixedVector::new(0),
            vectors: FixedMatrix::new(0, 0),
        });
    }

    if n == 1 {
        return Ok(EigenDecomposition {
            values: FixedVector::from_slice(&[a.get(0, 0)]),
            vectors: FixedMatrix::identity(1),
        });
    }

    // Work on a mutable copy; accumulate eigenvectors in V (starts as I)
    let mut s = a.clone();
    let mut v = FixedMatrix::identity(n);

    let mut converged = false;
    let mut best_off: Option<FixedPoint> = None;
    let mut stagnant = 0usize;
    for _sweep in 0..max_sweeps {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let bound = deflation_threshold(s.get(p, p).abs().max(s.get(q, q).abs()));
                if s.get(p, q).abs() > bound {
                    jacobi_rotate(&mut s, &mut v, p, q)?;
                    rotated = true;
                }
            }
        }
        if !rotated {
            converged = true;
            break;
        }

        let (off, _, _) = largest_off_diagonal(&s);
        if best_off.map_or(true, |best| off < best) {
            best_off = Some(off);
            stagnant = 0;
        } else {
            stagnant += 1;
        }
        if stagnant >= STAGNATION_SWEEPS && off_diagonal_within_stagnation_bound(&s) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(OverflowDetected::PrecisionLimit);
    }

    // One rotation on the largest remaining off-diagonal entry: it is within
    // the bound, but still contributes to the nearest eigenvalues.
    let (largest, p, q) = largest_off_diagonal(&s);
    if !largest.is_zero() {
        jacobi_rotate(&mut s, &mut v, p, q)?;
    }

    // Extract eigenvalues from diagonal
    let mut eigen_pairs: Vec<(FixedPoint, usize)> = (0..n)
        .map(|i| (s.get(i, i), i))
        .collect();

    // Sort descending by absolute value
    eigen_pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));

    let mut values = FixedVector::new(n);
    let mut vectors = FixedMatrix::new(n, n);
    for (k, (val, orig_idx)) in eigen_pairs.iter().enumerate() {
        values[k] = *val;
        for r in 0..n {
            vectors.set(r, k, v.get(r, *orig_idx));
        }
    }

    Ok(EigenDecomposition { values, vectors })
}

/// Largest |s[p][q]| over p < q, with its position (the first on ties).
fn largest_off_diagonal(s: &FixedMatrix) -> (FixedPoint, usize, usize) {
    let n = s.rows();
    let (mut largest, mut at_p, mut at_q) = (FixedPoint::ZERO, 0, 1);
    for p in 0..n {
        for q in (p + 1)..n {
            let value = s.get(p, q).abs();
            if value > largest {
                largest = value;
                at_p = p;
                at_q = q;
            }
        }
    }
    (largest, at_p, at_q)
}

fn off_diagonal_within_stagnation_bound(s: &FixedMatrix) -> bool {
    let n = s.rows();
    (0..n).all(|p| {
        ((p + 1)..n).all(|q| s.get(p, q).abs() <= stagnation_threshold(s.get(p, p).abs().max(s.get(q, q).abs())))
    })
}

/// Zero `s[p][q]` (and `s[q][p]`) by a Jacobi rotation, accumulating it into `v`.
///
/// With `τ = (a_pp - a_qq) / (2 a_pq)`, `t = sign(τ) / (|τ| + sqrt(1 + τ²))`,
/// `cs = 1 / sqrt(1 + t²)`, `sn = t cs`, all at the compute tier. The
/// off-diagonal rows rotate by `(cs, sn)`; the diagonal moves by `± t a_pq`.
fn jacobi_rotate(s: &mut FixedMatrix, v: &mut FixedMatrix, p: usize, q: usize) -> Result<(), OverflowDetected> {
    let n = s.rows();
    let (a_pp, a_qq, a_pq) = (s.get(p, p), s.get(q, q), s.get(p, q));
    if a_pq.is_zero() {
        return Ok(());
    }
    let one = make_compute_int(1);
    let num = compute_checked_add(upscale_to_compute(a_pp.raw()), compute_negate(upscale_to_compute(a_qq.raw())))?;
    let den = compute_checked_add(upscale_to_compute(a_pq.raw()), upscale_to_compute(a_pq.raw()))?;
    let negative = !compute_is_zero(&num) && (compute_is_negative(&num) != compute_is_negative(&den));
    let (num_abs, den_abs) = (compute_abs(num), compute_abs(den));
    let t_abs = if num_abs <= den_abs {
        // |τ| <= 1
        let tau = compute_divide(num_abs, den_abs)?;
        let root = sqrt_at_compute_tier(compute_add(one, compute_multiply(tau, tau)));
        compute_divide(one, compute_add(tau, root))?
    } else {
        // |τ| > 1: with r = 1/|τ|, t = r / (1 + sqrt(1 + r²))
        let r = compute_divide(den_abs, num_abs)?;
        let root = sqrt_at_compute_tier(compute_add(one, compute_multiply(r, r)));
        compute_divide(r, compute_add(one, root))?
    };
    let t = if negative { compute_negate(t_abs) } else { t_abs };
    let cs = compute_divide(one, sqrt_at_compute_tier(compute_add(one, compute_multiply(t, t))))?;
    let rot = Rotation::from_parts(cs, compute_multiply(t, cs));

    for r in 0..n {
        if r == p || r == q {
            continue;
        }
        let (new_rp, new_rq) = rot.apply(s.get(r, p), s.get(r, q))?;
        s.set(r, p, new_rp);
        s.set(p, r, new_rp);
        s.set(r, q, new_rq);
        s.set(q, r, new_rq);
    }

    let shift = scale_by(t, a_pq)?;
    s.set(p, p, checked_add_fp(a_pp, shift)?);
    s.set(q, q, checked_sub_fp(a_qq, shift)?);
    s.set(p, q, FixedPoint::ZERO);
    s.set(q, p, FixedPoint::ZERO);

    rotate_columns(v, p, q, &rot)
}

// ============================================================================
// Singular Value Decomposition (Golub-Kahan Bidiagonalization + QR Iteration)
// ============================================================================

/// Result of SVD: A = U Σ Vᵀ.
///
/// - `u` is m×m orthogonal
/// - `sigma` contains singular values (non-negative, sorted descending)
/// - `vt` is n×n orthogonal (Vᵀ, not V)
#[derive(Clone, Debug)]
pub struct SVDDecomposition {
    pub u: FixedMatrix,
    pub sigma: FixedVector,
    pub vt: FixedMatrix,
}

/// SVD via Golub-Kahan bidiagonalization + implicit QR iteration.
///
/// For an m×n matrix A (m >= n), computes A = U Σ Vᵀ where:
/// - U is m×m orthogonal
/// - Σ is m×n with non-negative diagonal entries (singular values)
/// - Vᵀ is n×n orthogonal
///
/// **Algorithm:**
/// 1. Householder bidiagonalization: A = U₀ B V₀ᵀ (B upper bidiagonal)
/// 2. Golub-Kahan implicit QR iteration with Wilkinson shift on B; a zero
///    diagonal entry is chased out of the active block by rotations, with U
///    (row rotations) or V (column rotations) taking the matching transpose
/// 3. Singular values extracted from converged B diagonal
///
/// **Precision:** Householder factors `2 (v.w)/(v.v)` and rotation
/// coefficients stay at the compute tier, and every transformed entry is
/// narrowed once from an exact accumulator. The Wilkinson shift is formed at
/// the compute tier from exact products.
///
/// **Convergence:** a superdiagonal entry is negligible within the tight
/// relative bound of its diagonal neighbours, floored at four quanta, and a
/// diagonal entry of at most four quanta is set to zero and deflated. An exact
/// zero singular value is computed as a block of rounding noise that a purely
/// relative test never passes. A block whose largest superdiagonal entry has
/// not decreased for five iterations has reached the precision floor: its entry
/// with the smallest backward error is deflated if it lies within the looser
/// sqrt(quantum) relative bound.
///
/// **Errors:** `Err(PrecisionLimit)` if the iteration budget (30 n² steps) runs
/// out: the unconverged diagonal is never returned. `Err(TierOverflow)` if a
/// column or row norm, or a transformed entry, leaves the storage range.
///
/// Returns singular values sorted descending. For m < n, transposes
/// internally and adjusts U/V accordingly.
pub fn svd_decompose(a: &FixedMatrix) -> Result<SVDDecomposition, OverflowDetected> {
    svd_decompose_within(a, ITERATIONS_PER_N_SQUARED)
}

fn svd_decompose_within(a: &FixedMatrix, iterations_per_n_squared: usize) -> Result<SVDDecomposition, OverflowDetected> {
    let (m, n) = (a.rows(), a.cols());

    if m == 0 || n == 0 {
        return Ok(SVDDecomposition {
            u: FixedMatrix::identity(m),
            sigma: FixedVector::new(0),
            vt: FixedMatrix::identity(n),
        });
    }

    // If m < n, compute SVD of Aᵀ then swap U and V:
    // if Aᵀ = U' Σ' V'ᵀ then A = V' Σ'ᵀ U'ᵀ, so U_A = V' and Vᵀ_A = U'ᵀ.
    if m < n {
        let at = a.transpose();
        let mut result = svd_decompose_within(&at, iterations_per_n_squared)?;
        let u_new = result.vt.transpose();
        let vt_new = result.u.transpose();
        result.u = u_new;
        result.vt = vt_new;
        return Ok(result);
    }

    // ── Phase 1: Householder Bidiagonalization ──
    // Transform A into upper bidiagonal B via left and right Householder reflections:
    // U₀ᵀ A V₀ = B
    let mut b = a.clone();
    let mut u_acc = FixedMatrix::identity(m);
    let mut v_acc = FixedMatrix::identity(n);

    for j in 0..n {
        // ── Left Householder: zero out B[j+1..m, j] ──
        let column: Vec<BinaryStorage> = (j..m).map(|i| b.get(i, j).raw()).collect();
        if let Some((v_hh, vtv)) = householder_vector(&column)? {
            for c in j..n {
                reflect_column(&mut b, c, j, &v_hh, vtv)?;
            }
            for r in 0..m {
                reflect_row(&mut u_acc, r, j, &v_hh, vtv)?;
            }
        }

        // ── Right Householder: zero out B[j, j+2..n] ──
        if j + 1 < n {
            let row: Vec<BinaryStorage> = (j + 1..n).map(|c| b.get(j, c).raw()).collect();
            if let Some((v_hh, vtv)) = householder_vector(&row)? {
                for r in j..m {
                    reflect_row(&mut b, r, j + 1, &v_hh, vtv)?;
                }
                for r in 0..n {
                    reflect_row(&mut v_acc, r, j + 1, &v_hh, vtv)?;
                }
            }
        }
    }

    // ── Phase 2: Golub-Kahan Implicit QR Iteration ──
    // Bidiagonal elements: diagonal d[0..n], superdiagonal e[0..n-1]
    let mut d: Vec<FixedPoint> = (0..n).map(|i| b.get(i, i)).collect();
    let mut e: Vec<FixedPoint> = (0..n.saturating_sub(1)).map(|i| b.get(i, i + 1)).collect();

    let floor = noise_floor();
    let max_iter = iterations_per_n_squared * n * n;
    let mut iter_count = 0usize;
    let mut q_end = n; // exclusive end of the unconverged part

    // Stagnation state of the active block (p, q)
    let mut stall_block = (usize::MAX, usize::MAX);
    let mut stall_best = FixedPoint::ZERO;
    let mut stall_count = 0usize;

    loop {
        // Peel converged superdiagonal entries off the bottom
        while q_end > 1
            && e[q_end - 2].abs() <= deflation_threshold(d[q_end - 1].abs().max(d[q_end - 2].abs()))
        {
            q_end -= 1;
        }
        if q_end <= 1 {
            break;
        }
        if iter_count >= max_iter {
            return Err(OverflowDetected::PrecisionLimit);
        }

        // Active block: d[p..=q], e[p..q]
        let q = q_end - 1;
        let mut p = q;
        while p > 0 && e[p - 1].abs() > deflation_threshold(d[p].abs().max(d[p - 1].abs())) {
            p -= 1;
        }

        // ── Stagnation fallback ──
        let largest = (p..q).map(|k| e[k].abs()).max().expect("active block has a superdiagonal entry");
        let mut forced_zero: Option<usize> = None;
        if stall_block != (p, q) || largest < stall_best {
            stall_block = (p, q);
            stall_best = largest;
            stall_count = 0;
        } else {
            stall_count += 1;
        }
        if stall_count >= STAGNATION_SWEEPS {
            stall_count = 0;
            let ie = (p..q).min_by_key(|&k| e[k].abs()).expect("active block has a superdiagonal entry");
            let id = (p..=q).min_by_key(|&k| d[k].abs()).expect("active block has a diagonal entry");
            let e_ok = e[ie].abs() <= stagnation_threshold(d[ie].abs().max(d[ie + 1].abs()));
            let mut neighbour = FixedPoint::ZERO;
            if id > 0 {
                neighbour = neighbour.max(d[id - 1].abs()).max(e[id - 1].abs());
            }
            if id < q {
                neighbour = neighbour.max(e[id].abs());
            }
            let d_ok = d[id].abs() <= stagnation_threshold(neighbour);
            if e_ok && (!d_ok || e[ie].abs() <= d[id].abs()) {
                e[ie] = FixedPoint::ZERO;
                iter_count += 1;
                continue;
            }
            if d_ok {
                forced_zero = Some(id);
            }
        }

        // ── Zero diagonal at the bottom of the block ──
        // Chase e[q-1] upward with column rotations (columns j and q), which V takes.
        if d[q].abs() <= floor || forced_zero == Some(q) {
            d[q] = FixedPoint::ZERO;
            let mut bulge = e[q - 1];
            e[q - 1] = FixedPoint::ZERO;
            for j in (p..q).rev() {
                let rot = Rotation::zeroing(d[j], bulge)?;
                d[j] = rot.combine(d[j], bulge)?;
                if j > p {
                    bulge = rot.neg_sin_times(e[j - 1])?;
                    e[j - 1] = rot.cos_times(e[j - 1])?;
                }
                rotate_columns(&mut v_acc, j, q, &rot)?;
            }
            iter_count += 1;
            continue;
        }

        // ── Zero diagonal inside the block ──
        // Chase e[i] downward with row rotations (rows j and i): the rows move by
        // G, so U takes Gᵀ on columns (j, i).
        if let Some(i) = (p..q).find(|&i| d[i].abs() <= floor || forced_zero == Some(i)) {
            d[i] = FixedPoint::ZERO;
            let mut bulge = e[i];
            e[i] = FixedPoint::ZERO;
            for j in (i + 1)..=q {
                let rot = Rotation::zeroing(d[j], bulge)?;
                d[j] = rot.combine(d[j], bulge)?;
                if j < q {
                    bulge = rot.neg_sin_times(e[j])?;
                    e[j] = rot.cos_times(e[j])?;
                }
                rotate_columns(&mut u_acc, j, i, &rot)?;
            }
            iter_count += 1;
            continue;
        }

        // ── Implicit QR step (Golub-Kahan), Wilkinson shift ──
        let shift = wilkinson_shift(d[q - 1], e[q - 1], d[q], if q >= 2 { Some(e[q - 2]) } else { None })?;
        let mut x = compute_checked_add(exact_product(d[p].raw(), d[p].raw()), compute_negate(shift))?;
        let mut z = exact_product(d[p].raw(), e[p].raw());
        let mut z_value = FixedPoint::ZERO;

        for i in p..q {
            // Right rotation on columns i, i+1
            let rot = Rotation::zeroing_compute(x, z)?;
            if i > p {
                e[i - 1] = rot.combine(e[i - 1], z_value)?;
            }
            let (new_di, new_ei) = rot.apply(d[i], e[i])?;
            d[i] = new_di;
            e[i] = new_ei;
            let bulge = rot.sin_times(d[i + 1])?;
            d[i + 1] = rot.cos_times(d[i + 1])?;
            rotate_columns(&mut v_acc, i, i + 1, &rot)?;

            // Left rotation on rows i, i+1
            let rot2 = Rotation::zeroing(d[i], bulge)?;
            d[i] = rot2.combine(d[i], bulge)?;
            let (new_ei, new_di1) = rot2.apply(e[i], d[i + 1])?;
            e[i] = new_ei;
            d[i + 1] = new_di1;
            rotate_columns(&mut u_acc, i, i + 1, &rot2)?;

            // Set up for next iteration of the chase
            if i + 1 < q {
                x = upscale_to_compute(e[i].raw());
                z_value = rot2.sin_times(e[i + 1])?;
                z = upscale_to_compute(z_value.raw());
                e[i + 1] = rot2.cos_times(e[i + 1])?;
            }
        }

        iter_count += 1;
    }

    // ── Phase 3: Make singular values non-negative and sort descending ──
    for i in 0..n {
        if d[i].is_negative() {
            d[i] = -d[i];
            // Flip sign of corresponding V column (row of Vᵀ)
            for r in 0..n {
                v_acc.set(r, i, -v_acc.get(r, i));
            }
        }
    }

    // Sort by descending singular value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut sigma = FixedVector::new(n);
    let mut u_sorted = FixedMatrix::new(m, m);
    let mut vt_sorted = FixedMatrix::new(n, n);

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sigma[new_idx] = d[old_idx];
        for r in 0..m {
            u_sorted.set(r, new_idx, u_acc.get(r, old_idx));
        }
        // Vᵀ[new_idx, r] = V[r, old_idx] = v_acc[r, old_idx]
        for r in 0..n {
            vt_sorted.set(new_idx, r, v_acc.get(r, old_idx));
        }
    }

    // Copy remaining U columns (m > n case)
    for new_idx in n..m {
        for r in 0..m {
            u_sorted.set(r, new_idx, u_acc.get(r, new_idx));
        }
    }

    Ok(SVDDecomposition {
        u: u_sorted,
        sigma,
        vt: vt_sorted,
    })
}

/// Eigenvalue of the trailing 2×2 block of BᵀB closest to its last diagonal
/// entry, as a compute raw:
///   [d[q-1]² + e[q-2]²,  d[q-1] e[q-1]]
///   [d[q-1] e[q-1],      d[q]² + e[q-1]²]
/// The block's entries are exact products; the discriminant is formed in
/// ratio form, without squaring them.
fn wilkinson_shift(
    d_prev: FixedPoint, e_last: FixedPoint, d_last: FixedPoint, e_prev: Option<FixedPoint>,
) -> Result<ComputeStorage, OverflowDetected> {
    let e_prev_sq = match e_prev {
        Some(ep) => exact_product(ep.raw(), ep.raw()),
        None => make_compute_int(0),
    };
    let f = compute_checked_add(exact_product(d_prev.raw(), d_prev.raw()), e_prev_sq)?;
    let g = compute_checked_add(exact_product(d_last.raw(), d_last.raw()), exact_product(e_last.raw(), e_last.raw()))?;
    let h = exact_product(d_prev.raw(), e_last.raw());
    let diff = compute_halve(compute_checked_add(f, compute_negate(g))?);
    if compute_is_zero(&diff) && compute_is_zero(&h) {
        return Ok(g);
    }
    let disc = compute_hypot(diff, h)?;
    let denom = if compute_is_negative(&diff) {
        compute_checked_add(diff, compute_negate(disc))?
    } else {
        compute_checked_add(diff, disc)?
    };
    let ratio = compute_checked_divide(h, denom)?;
    compute_checked_add(g, compute_negate(compute_multiply(h, ratio)))
}

// ============================================================================
// Schur Decomposition (Hessenberg Reduction + Francis QR Iteration)
// ============================================================================

/// Result of real Schur decomposition: A = Q T Qᵀ.
///
/// - `q` is orthogonal (Qᵀ Q = I)
/// - `t` is quasi-upper-triangular (upper triangular with possible 2×2 diagonal
///   blocks for complex eigenvalue pairs)
#[derive(Clone, Debug)]
pub struct SchurDecomposition {
    pub q: FixedMatrix,
    pub t: FixedMatrix,
}

/// Real Schur decomposition via Hessenberg reduction + Francis implicit double-shift QR.
///
/// For an n×n matrix A, computes A = Q T Qᵀ where Q is orthogonal and T is
/// quasi-upper-triangular (real Schur form): every entry below the
/// subdiagonal is exactly zero, and a nonzero subdiagonal entry only opens a
/// 2×2 block whose eigenvalues are a complex pair. A 2×2 block with real
/// eigenvalues is split by a rotation, so the diagonal of T carries every real
/// eigenvalue.
///
/// **Algorithm:**
/// 1. Reduce A to upper Hessenberg form H via Householder reflections, setting
///    the annihilated entries to zero
/// 2. Apply Francis double-shift QR steps to the bottom unreduced block, the
///    bulge chased to the last row (a final 2×2 rotation) and the chased
///    entries set to zero; exceptional shifts after 10 and 20 iterations
///    without deflation break the cycles an ordinary shift can sit in
/// 3. Deflate when a subdiagonal entry is within the tight relative bound of
///    its diagonal neighbours, floored at four quanta (set to zero); split
///    converged 2×2 blocks with real eigenvalues
///
/// **Precision:** Householder factors, rotation coefficients and shifts stay at
/// the compute tier; every transformed entry is narrowed once from an exact
/// accumulator. A block that has not deflated after 30 iterations has reached
/// the precision floor: its smallest subdiagonal entry is deflated if it lies
/// within the looser sqrt(quantum) relative bound.
///
/// **Errors:** `Err(PrecisionLimit)` if the iteration budget (30 n² steps) runs
/// out: an unconverged T is never returned. `Err(TierOverflow)` if a norm or a
/// transformed entry leaves the storage range. Panics if the matrix is not
/// square.
pub fn schur_decompose(a: &FixedMatrix) -> Result<SchurDecomposition, OverflowDetected> {
    schur_decompose_within(a, ITERATIONS_PER_N_SQUARED)
}

fn schur_decompose_within(a: &FixedMatrix, iterations_per_n_squared: usize) -> Result<SchurDecomposition, OverflowDetected> {
    assert!(a.is_square(), "schur_decompose: matrix must be square");
    let n = a.rows();

    if n <= 1 {
        return Ok(SchurDecomposition {
            q: FixedMatrix::identity(n),
            t: a.clone(),
        });
    }

    // ── Phase 1: Hessenberg Reduction ──
    // Reduce A to upper Hessenberg form H via Householder: Qᵀ A Q = H
    let mut h = a.clone();
    let mut q_acc = FixedMatrix::identity(n);

    for k in 0..n.saturating_sub(2) {
        let start = k + 1;
        let column: Vec<BinaryStorage> = (start..n).map(|i| h.get(i, k).raw()).collect();
        let (v_hh, vtv) = match householder_vector(&column)? {
            Some(reflector) => reflector,
            None => continue,
        };
        // Left: H[start..n, :] ; columns before k are already zero in these rows
        for c in k..n {
            reflect_column(&mut h, c, start, &v_hh, vtv)?;
        }
        // Right: H[:, start..n]
        for r in 0..n {
            reflect_row(&mut h, r, start, &v_hh, vtv)?;
        }
        // Accumulate into Q: Q[:, start..n]
        for r in 0..n {
            reflect_row(&mut q_acc, r, start, &v_hh, vtv)?;
        }
        for i in (start + 1)..n {
            h.set(i, k, FixedPoint::ZERO);
        }
    }

    // ── Phase 2: Francis Implicit Double-Shift QR Iteration ──
    let max_iter = iterations_per_n_squared * n * n;
    let mut iter_count = 0usize;
    let mut its = 0usize; // iterations since the last deflation at the bottom
    let mut nn = n; // h[0..nn, 0..nn] holds the unconverged part

    while nn > 0 {
        // Start l of the bottom unreduced block; a negligible subdiagonal entry
        // is set to zero (its backward error is the entry itself)
        let mut l = nn - 1;
        while l > 0 {
            let bound = deflation_threshold(h.get(l, l).abs().max(h.get(l - 1, l - 1).abs()));
            if h.get(l, l - 1).abs() <= bound {
                h.set(l, l - 1, FixedPoint::ZERO);
                break;
            }
            l -= 1;
        }

        match nn - l {
            1 => {
                nn -= 1;
                its = 0;
                continue;
            }
            2 => {
                split_real_block(&mut h, &mut q_acc, l)?;
                nn -= 2;
                its = 0;
                continue;
            }
            _ => {}
        }

        if iter_count >= max_iter {
            return Err(OverflowDetected::PrecisionLimit);
        }

        // Precision floor: deflate the smallest subdiagonal entry if it is
        // within the loose bound
        if its >= SCHUR_FLOOR_ITERATIONS {
            let i = ((l + 1)..nn).min_by_key(|&i| h.get(i, i - 1).abs()).expect("block of size >= 3");
            if h.get(i, i - 1).abs() <= stagnation_threshold(h.get(i, i).abs().max(h.get(i - 1, i - 1).abs())) {
                h.set(i, i - 1, FixedPoint::ZERO);
                its = 0;
                iter_count += 1;
                continue;
            }
        }

        let (trace, det) = if its == 10 || its == 20 {
            exceptional_shifts(&h, l, nn, its)?
        } else {
            trailing_shifts(&h, nn)?
        };
        francis_step(&mut h, &mut q_acc, l, nn, trace, det)?;
        its += 1;
        iter_count += 1;
    }

    Ok(SchurDecomposition { q: q_acc, t: h })
}

/// Trace and determinant of the trailing 2×2 block (the double-shift pair),
/// as compute raws.
fn trailing_shifts(h: &FixedMatrix, nn: usize) -> Result<(ComputeStorage, ComputeStorage), OverflowDetected> {
    let (a, b) = (h.get(nn - 2, nn - 2), h.get(nn - 2, nn - 1));
    let (c, d) = (h.get(nn - 1, nn - 2), h.get(nn - 1, nn - 1));
    let trace = compute_checked_add(upscale_to_compute(a.raw()), upscale_to_compute(d.raw()))?;
    let det = compute_checked_add(exact_product(a.raw(), d.raw()), compute_negate(exact_product(b.raw(), c.raw())))?;
    Ok((trace, det))
}

/// Exceptional shift pair after 10 and 20 iterations without deflation
/// (LAPACK `dlahqr`): the 2×2 block `[[w, -7/16 s], [s, w]]` with
/// `w = h + 3/4 s`, where `s` sums two subdiagonal magnitudes (at the top of
/// the block after 10 iterations, at the bottom after 20). It breaks the
/// cycles an ordinary Francis step sits in, such as permutation matrices.
fn exceptional_shifts(h: &FixedMatrix, l: usize, nn: usize, its: usize) -> Result<(ComputeStorage, ComputeStorage), OverflowDetected> {
    let (s, anchor) = if its == 10 {
        (checked_add_fp(h.get(l + 1, l).abs(), h.get(l + 2, l + 1).abs())?, h.get(l, l))
    } else {
        (checked_add_fp(h.get(nn - 1, nn - 2).abs(), h.get(nn - 2, nn - 3).abs())?, h.get(nn - 1, nn - 1))
    };
    let s_c = upscale_to_compute(s.raw());
    let three_quarters = compute_divide(make_compute_int(3), make_compute_int(4))?;
    let seven_sixteenths = compute_divide(make_compute_int(7), make_compute_int(16))?;
    let w = compute_checked_add(upscale_to_compute(anchor.raw()), compute_multiply(three_quarters, s_c))?;
    let trace = compute_checked_add(w, w)?;
    let det = compute_checked_add(
        compute_multiply(w, w),
        compute_multiply(seven_sixteenths, compute_multiply(s_c, s_c)),
    )?;
    Ok((trace, det))
}

/// One Francis double-shift step on the unreduced block `h[l..nn, l..nn]`
/// (size >= 3), applied to all of H (real Schur form) and accumulated into Q.
/// The shifts only steer convergence; every transform applied is orthogonal.
fn francis_step(
    h: &mut FixedMatrix, q_acc: &mut FixedMatrix, l: usize, nn: usize,
    trace: ComputeStorage, det: ComputeStorage,
) -> Result<(), OverflowDetected> {
    let n = h.rows();

    // First column of (H - s1 I)(H - s2 I) = H² - trace H + det I
    let (h11, h12, h21) = (h.get(l, l), h.get(l, l + 1), h.get(l + 1, l));
    let (h22, h32) = (h.get(l + 1, l + 1), h.get(l + 2, l + 1));
    let h11_c = upscale_to_compute(h11.raw());
    let x = compute_sum(&[
        exact_product(h11.raw(), h11.raw()),
        exact_product(h12.raw(), h21.raw()),
        compute_negate(compute_multiply(trace, h11_c)),
        det,
    ])?;
    let y = compute_multiply(
        upscale_to_compute(h21.raw()),
        compute_sum(&[h11_c, upscale_to_compute(h22.raw()), compute_negate(trace)])?,
    );
    let z = exact_product(h21.raw(), h32.raw());

    // Reflector introducing the bulge at rows l..l+3
    let norm = compute_hypot(compute_hypot(x, y)?, z)?;
    if compute_is_zero(&norm) {
        return Ok(());
    }
    let lead = if compute_is_negative(&x) {
        compute_checked_add(x, compute_negate(norm))?
    } else {
        compute_checked_add(x, norm)?
    };
    let v_hh = direction_to_storage(&[lead, y, z])?;
    let vtv = exact_dot(&v_hh, &v_hh)?;
    if !compute_is_zero(&vtv) {
        for c in l..n {
            reflect_column(h, c, l, &v_hh, vtv)?;
        }
        for r in 0..nn.min(l + 4) {
            reflect_row(h, r, l, &v_hh, vtv)?;
        }
        for r in 0..n {
            reflect_row(q_acc, r, l, &v_hh, vtv)?;
        }
    }

    // ── Bulge chase down to the last row of the block ──
    for k in (l + 1)..(nn - 1) {
        if k + 2 < nn {
            // 3-element reflector on rows k..k+3 zeroes h[k+1, k-1], h[k+2, k-1]
            let column: Vec<BinaryStorage> = (k..k + 3).map(|i| h.get(i, k - 1).raw()).collect();
            if let Some((v_hh, vtv)) = householder_vector(&column)? {
                for c in (k - 1)..n {
                    reflect_column(h, c, k, &v_hh, vtv)?;
                }
                for r in 0..nn.min(k + 4) {
                    reflect_row(h, r, k, &v_hh, vtv)?;
                }
                for r in 0..n {
                    reflect_row(q_acc, r, k, &v_hh, vtv)?;
                }
            }
            h.set(k + 1, k - 1, FixedPoint::ZERO);
            h.set(k + 2, k - 1, FixedPoint::ZERO);
        } else {
            // Last step: a rotation on rows k, k+1 zeroes h[k+1, k-1]
            let rot = Rotation::zeroing(h.get(k, k - 1), h.get(k + 1, k - 1))?;
            for c in (k - 1)..n {
                let (top, bottom) = rot.apply(h.get(k, c), h.get(k + 1, c))?;
                h.set(k, c, top);
                h.set(k + 1, c, bottom);
            }
            for r in 0..nn {
                let (left, right) = rot.apply(h.get(r, k), h.get(r, k + 1))?;
                h.set(r, k, left);
                h.set(r, k + 1, right);
            }
            rotate_columns(q_acc, k, k + 1, &rot)?;
            h.set(k + 1, k - 1, FixedPoint::ZERO);
        }
    }

    Ok(())
}

/// Split the converged 2×2 block at rows and columns (i, i+1) into two 1×1
/// blocks when its eigenvalues are real: rotate by the eigenvector
/// `(λ - d, c)` of `λ = (a+d)/2 + sign(p) sqrt(p² + bc)`, `p = (a-d)/2`
/// (no cancellation in `λ - d = p + sign(p) sqrt(..)`). A complex pair keeps
/// its block.
fn split_real_block(h: &mut FixedMatrix, q_acc: &mut FixedMatrix, i: usize) -> Result<(), OverflowDetected> {
    let n = h.rows();
    let (a, b) = (h.get(i, i), h.get(i, i + 1));
    let (c, d) = (h.get(i + 1, i), h.get(i + 1, i + 1));
    if c.is_zero() {
        return Ok(());
    }
    let p = compute_halve(compute_checked_add(upscale_to_compute(a.raw()), compute_negate(upscale_to_compute(d.raw())))?);
    let disc = compute_checked_add(compute_multiply(p, p), exact_product(b.raw(), c.raw()))?;
    if compute_is_negative(&disc) {
        return Ok(());
    }
    let root = sqrt_at_compute_tier(disc);
    let lead = if compute_is_negative(&p) {
        compute_checked_add(p, compute_negate(root))?
    } else {
        compute_checked_add(p, root)?
    };
    let rot = Rotation::zeroing_compute(lead, upscale_to_compute(c.raw()))?;
    // Gᵀ H on rows i, i+1 (entries left of column i are zero in both rows)
    for col in i..n {
        let (top, bottom) = rot.apply(h.get(i, col), h.get(i + 1, col))?;
        h.set(i, col, top);
        h.set(i + 1, col, bottom);
    }
    // H G on columns i, i+1 (entries below row i+1 are zero in both columns)
    for row in 0..(i + 2) {
        let (left, right) = rot.apply(h.get(row, i), h.get(row, i + 1))?;
        h.set(row, i, left);
        h.set(row, i + 1, right);
    }
    rotate_columns(q_acc, i, i + 1, &rot)?;
    h.set(i + 1, i, FixedPoint::ZERO);
    Ok(())
}

/// An integer vector parallel to `values` (compute raws) whose largest entry
/// lies in `(M/2, M]`, `M = 2^min(2F, W-3)` raw. All entries are halved or
/// doubled together, so the direction is kept to one part in `M/2`; a
/// reflector depends only on the direction of its vector. The size matters for
/// `reflect`: its factor `2 (v.w)/(v.v)` is rounded at `2F` fractional bits,
/// which moves an output entry by up to `|v_k| / 2^(2F+1)` ulp, so a direction
/// much larger than `2^(2F)` would cost precision; and `M <= 2^(W-3)` keeps a
/// three-entry `v.v` inside the compute tier.
fn direction_to_storage(values: &[ComputeStorage]) -> Result<Vec<BinaryStorage>, OverflowDetected> {
    let largest = |vs: &[ComputeStorage]| {
        let mut big = make_compute_int(0);
        for value in vs {
            let magnitude = compute_abs(*value);
            if magnitude > big {
                big = magnitude;
            }
        }
        big
    };
    let mut scaled = values.to_vec();
    if compute_is_zero(&largest(&scaled)) {
        return Ok(vec![FixedPoint::ZERO.raw(); values.len()]);
    }
    let limit = widen_storage(direction_magnitude());
    while largest(&scaled) > limit {
        for value in scaled.iter_mut() {
            *value = compute_halve(*value);
        }
    }
    let half_limit = compute_halve(limit);
    while largest(&scaled) <= half_limit {
        for value in scaled.iter_mut() {
            *value = compute_checked_add(*value, *value)?;
        }
    }
    Ok(scaled.into_iter().map(compute_to_storage_exact).collect())
}

/// `2^min(2F, W-3)` as a storage raw: the largest entry of a reflector
/// direction built by `direction_to_storage`.
#[inline]
fn direction_magnitude() -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    { 1i32 << (2 * crate::fixed_point::frac_config::FRAC_BITS).min(29) }
    #[cfg(table_format = "q32_32")]
    { 1i64 << 61 }
    #[cfg(table_format = "q64_64")]
    { 1i128 << 125 }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::I256::from_i128(1) << 253usize }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::I512::from_i128(1) << 509usize }
}

/// A compute-width integer known to fit storage, as a storage integer.
#[inline]
fn compute_to_storage_exact(v: ComputeStorage) -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    { v as i32 }
    #[cfg(table_format = "q32_32")]
    { v as i64 }
    #[cfg(table_format = "q64_64")]
    { v.as_i128() }
    #[cfg(table_format = "q128_128")]
    { v.as_i256() }
    #[cfg(table_format = "q256_256")]
    { v.as_i512() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_matrix(rows: &[&[i32]]) -> FixedMatrix {
        FixedMatrix::from_fn(rows.len(), rows[0].len(), |i, j| FixedPoint::from_int(rows[i][j]))
    }

    /// Running out of iterations is an error, never a partially converged Ok.
    #[test]
    fn iteration_budget_exhaustion_is_an_error() {
        let a = int_matrix(&[&[4, 1, 2], &[1, 3, 1], &[2, 1, 5]]);
        assert_eq!(eigen_symmetric_within(&a, 0).unwrap_err(), OverflowDetected::PrecisionLimit);
        assert_eq!(svd_decompose_within(&a, 0).unwrap_err(), OverflowDetected::PrecisionLimit);
        assert_eq!(schur_decompose_within(&a, 0).unwrap_err(), OverflowDetected::PrecisionLimit);
        assert!(eigen_symmetric(&a).is_ok());
        assert!(svd_decompose(&a).is_ok());
        assert!(schur_decompose(&a).is_ok());
    }
}
