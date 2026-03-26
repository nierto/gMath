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
use super::linalg::{
    compute_tier_dot_raw, compute_tier_sub_dot_raw, compute_tier_sub_dot_compute,
    upscale_to_compute, round_to_storage, givens, convergence_threshold,
    convergence_threshold_tight, apply_givens_compute,
};
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    sqrt_at_compute_tier, compute_divide, downscale_to_storage,
    compute_multiply, compute_add, compute_negate,
    compute_is_negative, compute_is_zero,
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
/// computed via `compute_tier_sub_dot_raw` — the entire inner sum accumulates
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
    /// Product accumulated at compute tier — single downscale at the end.
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
/// **Why Jacobi for fixed-point:** The method is inherently self-correcting.
/// Each Givens rotation introduces ~1 ULP of rounding error, but subsequent
/// rotations targeting the same off-diagonal element correct it. This makes
/// Jacobi far more robust than QR iteration for fixed-point arithmetic.
///
/// **Algorithm:**
/// 1. Cyclic-by-row sweeps: for each (i,j) with i<j, apply a Givens rotation
///    to zero A[i][j] (and A[j][i] by symmetry).
/// 2. Rotation angle computed via quadratic formula — NO trig functions.
///    When a_ii == a_jj, use exact 45° rotation (cs = sn = √2/2).
/// 3. Convergence: off-diagonal Frobenius norm drops below threshold, or
///    stagnation detected (5 sweeps with no improvement).
///
/// **Precision:** All rotation parameters (tau, t, cs, sn) computed at
/// compute tier via upscale → compute_multiply/divide → downscale.
///
/// Returns `Err(DomainError)` if matrix is not square.
/// The input matrix should be symmetric; only the lower triangle is read.
pub fn eigen_symmetric(a: &FixedMatrix) -> Result<EigenDecomposition, OverflowDetected> {
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

    let one = FixedPoint::one();
    let two = FixedPoint::from_int(2);
    let half = one / two;

    // Tight convergence threshold: all rotations at compute tier → 1 ULP/step
    // so we can converge to 2*FRAC_BITS/3 instead of FRAC_BITS/2
    let diag_max = {
        let mut m = FixedPoint::ZERO;
        for i in 0..n {
            let d = s.get(i, i).abs();
            if d > m { m = d; }
        }
        m
    };
    let threshold = convergence_threshold_tight(diag_max);

    // off-diagonal Frobenius norm squared (symmetric: count each pair once, multiply by 2)
    let off_diag_norm_sq = |mat: &FixedMatrix| -> FixedPoint {
        let mut sum = FixedPoint::ZERO;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = mat.get(i, j);
                sum += v * v;
            }
        }
        two * sum
    };

    let max_sweeps = 100;
    let mut prev_off = off_diag_norm_sq(&s);
    let mut stagnation_count = 0usize;

    for _sweep in 0..max_sweeps {
        // Check convergence: all off-diagonal elements negligible
        let off = off_diag_norm_sq(&s);
        if off <= threshold * threshold {
            break;
        }

        // Stagnation detection
        if off >= prev_off {
            stagnation_count += 1;
            if stagnation_count >= 5 {
                break;
            }
        } else {
            stagnation_count = 0;
        }
        prev_off = off;

        // Cyclic-by-row sweep
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = s.get(p, q);
                if a_pq.abs() <= threshold {
                    continue; // Skip negligible elements
                }

                let a_pp = s.get(p, p);
                let a_qq = s.get(q, q);
                let diff = a_pp - a_qq;

                // Compute rotation: tan(2θ) = 2*a_pq / (a_pp - a_qq)
                // Solve quadratic: t² + 2τt - 1 = 0, where τ = (a_pp - a_qq) / (2*a_pq)
                // Take the smaller root for numerical stability: t = sign(τ) / (|τ| + √(1+τ²))
                let (cs, sn) = if diff.abs() <= threshold {
                    // a_pp ≈ a_qq → θ = π/4, exact 45° rotation
                    // cs = sn = 1/√2, but compute precisely
                    let sqrt2_inv = (one + one).try_sqrt()
                        .map(|s| one / s)
                        .unwrap_or(half); // fallback: ~0.5 if sqrt fails
                    let sn_val = if a_pq.is_negative() { -sqrt2_inv } else { sqrt2_inv };
                    (sqrt2_inv, sn_val)
                } else {
                    // τ = (a_pp - a_qq) / (2 * a_pq)
                    // Compute at compute tier for maximum precision
                    let tau_compute = {
                        let num = upscale_to_compute(diff.raw());
                        let den = upscale_to_compute((two * a_pq).raw());
                        compute_divide(num, den)
                            .unwrap_or(upscale_to_compute(diff.raw())) // fallback
                    };

                    // t = sign(τ) / (|τ| + √(1 + τ²))
                    // Compute 1 + τ² at compute tier
                    let one_compute = upscale_to_compute(one.raw());
                    let tau_sq = compute_multiply(tau_compute, tau_compute);
                    let disc = compute_add(one_compute, tau_sq);
                    let sqrt_disc = sqrt_at_compute_tier(disc);

                    let abs_tau = if compute_is_negative(&tau_compute) {
                        compute_negate(tau_compute)
                    } else {
                        tau_compute
                    };
                    let denom = compute_add(abs_tau, sqrt_disc);
                    let t_compute = compute_divide(one_compute, denom)
                        .unwrap_or(one_compute);

                    // Apply sign of τ
                    let t_compute = if compute_is_negative(&tau_compute) {
                        compute_negate(t_compute)
                    } else {
                        t_compute
                    };

                    // cs = 1 / √(1 + t²)
                    let t_sq = compute_multiply(t_compute, t_compute);
                    let one_plus_tsq = compute_add(one_compute, t_sq);
                    let sqrt_1pt = sqrt_at_compute_tier(one_plus_tsq);
                    let cs_compute = compute_divide(one_compute, sqrt_1pt)
                        .unwrap_or(one_compute);

                    // sn = t * cs
                    let sn_compute = compute_multiply(t_compute, cs_compute);

                    // Downscale to storage
                    let cs_val = FixedPoint::from_raw(round_to_storage(cs_compute));
                    let sn_val = FixedPoint::from_raw(round_to_storage(sn_compute));
                    (cs_val, sn_val)
                };

                // Apply Jacobi rotation: S' = Jᵀ S J
                // All rotation applications at compute tier (1 ULP per element)
                // Only rows/cols p and q change
                for r in 0..n {
                    if r == p || r == q { continue; }
                    let s_rp = s.get(r, p);
                    let s_rq = s.get(r, q);
                    let (new_rp, new_rq) = apply_givens_compute(cs, sn, s_rp, s_rq);
                    s.set(r, p, new_rp);
                    s.set(p, r, new_rp); // symmetric
                    s.set(r, q, new_rq);
                    s.set(q, r, new_rq); // symmetric
                }

                // Update diagonal block at compute tier
                // new_pp = cs²*a_pp + 2*cs*sn*a_pq + sn²*a_qq  (3-element dot at compute tier)
                // new_qq = sn²*a_pp - 2*cs*sn*a_pq + cs²*a_qq
                let a_pp = s.get(p, p);
                let a_qq = s.get(q, q);
                let cs_sq = cs * cs;
                let sn_sq = sn * sn;
                let cs_sn_2 = two * cs * sn;
                let new_pp = FixedPoint::from_raw(compute_tier_dot_raw(
                    &[cs_sq.raw(), cs_sn_2.raw(), sn_sq.raw()],
                    &[a_pp.raw(), a_pq.raw(), a_qq.raw()],
                ));
                let new_qq = FixedPoint::from_raw(compute_tier_dot_raw(
                    &[sn_sq.raw(), (-cs_sn_2).raw(), cs_sq.raw()],
                    &[a_pp.raw(), a_pq.raw(), a_qq.raw()],
                ));
                s.set(p, p, new_pp);
                s.set(q, q, new_qq);
                s.set(p, q, FixedPoint::ZERO);
                s.set(q, p, FixedPoint::ZERO);

                // Accumulate rotation into V: V' = V * J (compute tier)
                for r in 0..n {
                    let v_rp = v.get(r, p);
                    let v_rq = v.get(r, q);
                    let (new_vp, new_vq) = apply_givens_compute(cs, sn, v_rp, v_rq);
                    v.set(r, p, new_vp);
                    v.set(r, q, new_vq);
                }
            }
        }
    }

    // Post-convergence refinement: one targeted rotation on the largest
    // remaining off-diagonal element. The last sweep may have exited with
    // a residual off-diagonal that's below threshold but still contributes
    // 1-2 ULP to the nearest eigenvalue. One extra rotation recovers this.
    {
        let mut max_abs = FixedPoint::ZERO;
        let mut max_p = 0;
        let mut max_q = 1;
        for p in 0..n {
            for q in (p + 1)..n {
                let val = s.get(p, q).abs();
                if val > max_abs {
                    max_abs = val;
                    max_p = p;
                    max_q = q;
                }
            }
        }
        if !max_abs.is_zero() {
            let p = max_p;
            let q = max_q;
            let a_pq = s.get(p, q);
            let a_pp = s.get(p, p);
            let a_qq = s.get(q, q);
            let diff = a_pp - a_qq;

            let (cs, sn) = if diff.abs().is_zero() {
                let sqrt2_inv = (one + one).try_sqrt()
                    .map(|s| one / s)
                    .unwrap_or(half);
                let sn_val = if a_pq.is_negative() { -sqrt2_inv } else { sqrt2_inv };
                (sqrt2_inv, sn_val)
            } else {
                let tau_compute = {
                    let num = upscale_to_compute(diff.raw());
                    let den = upscale_to_compute((two * a_pq).raw());
                    compute_divide(num, den).unwrap_or(upscale_to_compute(diff.raw()))
                };
                let one_compute = upscale_to_compute(one.raw());
                let tau_sq = compute_multiply(tau_compute, tau_compute);
                let disc = compute_add(one_compute, tau_sq);
                let sqrt_disc = sqrt_at_compute_tier(disc);
                let abs_tau = if compute_is_negative(&tau_compute) { compute_negate(tau_compute) } else { tau_compute };
                let denom = compute_add(abs_tau, sqrt_disc);
                let t_compute = compute_divide(one_compute, denom).unwrap_or(one_compute);
                let t_compute = if compute_is_negative(&tau_compute) { compute_negate(t_compute) } else { t_compute };
                let t_sq = compute_multiply(t_compute, t_compute);
                let sqrt_1pt = sqrt_at_compute_tier(compute_add(one_compute, t_sq));
                let cs_compute = compute_divide(one_compute, sqrt_1pt).unwrap_or(one_compute);
                let sn_compute = compute_multiply(t_compute, cs_compute);
                (FixedPoint::from_raw(round_to_storage(cs_compute)),
                 FixedPoint::from_raw(round_to_storage(sn_compute)))
            };

            for r in 0..n {
                if r == p || r == q { continue; }
                let (new_rp, new_rq) = apply_givens_compute(cs, sn, s.get(r, p), s.get(r, q));
                s.set(r, p, new_rp); s.set(p, r, new_rp);
                s.set(r, q, new_rq); s.set(q, r, new_rq);
            }
            let a_pp = s.get(p, p);
            let a_qq = s.get(q, q);
            let cs_sq = cs * cs;
            let sn_sq = sn * sn;
            let cs_sn_2 = two * cs * sn;
            s.set(p, p, FixedPoint::from_raw(compute_tier_dot_raw(
                &[cs_sq.raw(), cs_sn_2.raw(), sn_sq.raw()],
                &[a_pp.raw(), a_pq.raw(), a_qq.raw()],
            )));
            s.set(q, q, FixedPoint::from_raw(compute_tier_dot_raw(
                &[sn_sq.raw(), (-cs_sn_2).raw(), cs_sq.raw()],
                &[a_pp.raw(), a_pq.raw(), a_qq.raw()],
            )));
            s.set(p, q, FixedPoint::ZERO);
            s.set(q, p, FixedPoint::ZERO);
            for r in 0..n {
                let (new_vp, new_vq) = apply_givens_compute(cs, sn, v.get(r, p), v.get(r, q));
                v.set(r, p, new_vp);
                v.set(r, q, new_vq);
            }
        }
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
/// 2. Golub-Kahan implicit QR iteration with Wilkinson shift on B
/// 3. Singular values extracted from converged B diagonal
///
/// **FASC-UGOD strategy:** Householder reflections reuse the QR
/// infrastructure (compute-tier dot products). The Wilkinson shift
/// computation (2×2 trailing block eigenvalue) is done entirely at
/// compute tier to avoid overflow in the intermediate a²+b² term.
/// Givens rotations use the existing ratio-based `givens()` helper.
///
/// Returns singular values sorted descending. For m < n, transposes
/// internally and adjusts U/V accordingly.
pub fn svd_decompose(a: &FixedMatrix) -> Result<SVDDecomposition, OverflowDetected> {
    let (m, n) = (a.rows(), a.cols());

    if m == 0 || n == 0 {
        return Ok(SVDDecomposition {
            u: FixedMatrix::identity(m),
            sigma: FixedVector::new(0),
            vt: FixedMatrix::identity(n),
        });
    }

    // If m < n, compute SVD of Aᵀ then swap U and V
    if m < n {
        let at = a.transpose();
        let mut result = svd_decompose(&at)?;
        // A = U Σ Vᵀ  ↔  Aᵀ = V Σᵀ Uᵀ
        // So SVD(Aᵀ) gives (U', Σ', V'ᵀ) → A's SVD is (V'ᵀ)ᵀ, Σ', U'ᵀ...
        // Actually: if Aᵀ = U' Σ' V'ᵀ then A = V' Σ'ᵀ U'ᵀ
        // So U_A = V' (which is (V'ᵀ)ᵀ = result.vt.transpose())
        // and V_A = U' (so Vᵀ_A = U'ᵀ = result.u.transpose())
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
    let two = FixedPoint::from_int(2);
    let k = n.min(m); // number of bidiagonalization steps

    for j in 0..k {
        // ── Left Householder: zero out B[j+1..m, j] ──
        if j < m {
            let col_len = m - j;
            let x_raw: Vec<BinaryStorage> = (j..m).map(|i| b.get(i, j).raw()).collect();
            let norm_sq = FixedPoint::from_raw(compute_tier_dot_raw(&x_raw, &x_raw));
            if !norm_sq.is_zero() {
                let norm_x = norm_sq.try_sqrt()?;
                let x_0 = b.get(j, j);
                let alpha = if x_0.is_negative() { norm_x } else { -norm_x };

                let mut v_hh = Vec::<FixedPoint>::with_capacity(col_len);
                v_hh.push(x_0 - alpha);
                for i in 1..col_len {
                    v_hh.push(FixedPoint::from_raw(x_raw[i]));
                }
                let v_raw: Vec<BinaryStorage> = v_hh.iter().map(|fp| fp.raw()).collect();
                let vtv = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &v_raw));

                if !vtv.is_zero() {
                    // Apply to B: B[j..m, j..n] -= 2 v (vᵀ B) / vᵀv
                    for c in j..n {
                        let col_raw: Vec<BinaryStorage> = (j..m).map(|i| b.get(i, c).raw()).collect();
                        let vt_col = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &col_raw));
                        let scale = two * vt_col / vtv;
                        for i in j..m {
                            b.set(i, c, b.get(i, c) - scale * v_hh[i - j]);
                        }
                    }
                    // Accumulate into U: U[:, j..m] -= 2 (U v) vᵀ / vᵀv
                    for r in 0..m {
                        let u_row_raw: Vec<BinaryStorage> = (j..m).map(|c| u_acc.get(r, c).raw()).collect();
                        let dot = FixedPoint::from_raw(compute_tier_dot_raw(&u_row_raw, &v_raw));
                        let scale = two * dot / vtv;
                        for c in j..m {
                            u_acc.set(r, c, u_acc.get(r, c) - scale * v_hh[c - j]);
                        }
                    }
                }
            }
        }

        // ── Right Householder: zero out B[j, j+2..n] ──
        if j + 1 < n {
            let row_start = j + 1;
            let row_len = n - row_start;
            if row_len > 0 {
                let x_raw: Vec<BinaryStorage> = (row_start..n).map(|c| b.get(j, c).raw()).collect();
                let norm_sq = FixedPoint::from_raw(compute_tier_dot_raw(&x_raw, &x_raw));
                if !norm_sq.is_zero() {
                    let norm_x = norm_sq.try_sqrt()?;
                    let x_0 = b.get(j, row_start);
                    let alpha = if x_0.is_negative() { norm_x } else { -norm_x };

                    let mut v_hh = Vec::<FixedPoint>::with_capacity(row_len);
                    v_hh.push(x_0 - alpha);
                    for i in 1..row_len {
                        v_hh.push(FixedPoint::from_raw(x_raw[i]));
                    }
                    let v_raw: Vec<BinaryStorage> = v_hh.iter().map(|fp| fp.raw()).collect();
                    let vtv = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &v_raw));

                    if !vtv.is_zero() {
                        // Apply to B: B[j..m, row_start..n] -= 2 (B v) vᵀ / vᵀv
                        for r in j..m {
                            let row_raw: Vec<BinaryStorage> = (row_start..n).map(|c| b.get(r, c).raw()).collect();
                            let dot = FixedPoint::from_raw(compute_tier_dot_raw(&row_raw, &v_raw));
                            let scale = two * dot / vtv;
                            for c in row_start..n {
                                b.set(r, c, b.get(r, c) - scale * v_hh[c - row_start]);
                            }
                        }
                        // Accumulate into V: V[:, row_start..n] -= 2 (V v) vᵀ / vᵀv
                        for r in 0..n {
                            let v_row_raw: Vec<BinaryStorage> = (row_start..n).map(|c| v_acc.get(r, c).raw()).collect();
                            let dot = FixedPoint::from_raw(compute_tier_dot_raw(&v_row_raw, &v_raw));
                            let scale = two * dot / vtv;
                            for c in row_start..n {
                                v_acc.set(r, c, v_acc.get(r, c) - scale * v_hh[c - row_start]);
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Phase 2: Golub-Kahan Implicit QR Iteration ──
    // Extract bidiagonal elements: diagonal d[0..n], superdiagonal e[0..n-1]
    let mut d: Vec<FixedPoint> = (0..n).map(|i| b.get(i, i)).collect();
    let mut e: Vec<FixedPoint> = (0..n.saturating_sub(1)).map(|i| b.get(i, i + 1)).collect();

    let max_iter = 30 * n * n; // generous iteration budget
    let mut iter_count = 0usize;

    // Work on the active submatrix d[p..=q], e[p..q-1]
    // Find converged superdiagonals from the bottom
    let mut q_end = n; // exclusive upper bound

    while q_end > 1 && iter_count < max_iter {
        // Find the largest q such that e[q-1] is negligible
        let mut found_active = false;
        for idx in (1..q_end).rev() {
            let thresh_val = convergence_threshold(d[idx].abs().max(d[idx - 1].abs()));
            if e[idx - 1].abs() <= thresh_val {
                // e[idx-1] has converged — check if this splits the problem
                if idx == q_end - 1 {
                    q_end -= 1; // peel off converged singular value
                } else {
                    // Split point found but not at the boundary; continue
                    found_active = true;
                    break;
                }
            } else {
                found_active = true;
                break;
            }
        }
        if !found_active || q_end <= 1 {
            break;
        }

        // Find the start of the active block (first non-negligible e from bottom)
        let q = q_end - 1; // last index in active block
        let mut p = q;
        while p > 0 {
            let thresh_val = convergence_threshold(d[p].abs().max(d[p - 1].abs()));
            if e[p - 1].abs() <= thresh_val {
                break;
            }
            p -= 1;
        }

        // ── Zero-diagonal deflation (LAPACK dbdsqr approach) ──
        // When d[i] ≈ 0 in the active block, the Wilkinson shift degenerates.
        // Fix: eliminate e[i-1] or e[i] via Givens rotations, then re-check.
        //
        // For upper bidiagonal B:
        //   Row i has: e[i-1] (from column i-1) and d[i] on diagonal.
        //   If d[i] = 0, row i is [0, ..., e[i-1], 0, e[i], ...].
        //
        // Case d[q] ≈ 0 (bottom of active block):
        //   Zero e[q-1] via RIGHT Givens on columns q-1 and q.
        //   This affects V, not U. Chase the bulge upward through e[q-2], e[q-3], ...
        //
        // Case d[i] ≈ 0 for i < q (interior):
        //   Zero e[i] via LEFT Givens on rows i and i+1.
        //   Chase the bulge downward.
        {
            let mut deflated = false;

            // Check d[q] first (most common case for rank-deficient)
            {
                let d_thresh = convergence_threshold(
                    if q > 0 { d[q - 1].abs().max(e[q - 1].abs()) }
                    else { e[0].abs().max(FixedPoint::one()) }
                );
                if d[q].abs() <= d_thresh {
                    // d[q] ≈ 0: chase e[q-1] to zero via RIGHT Givens (columns)
                    // Rotate columns q-1 and q to zero B[q-1, q] = e[q-1]
                    // while B[q, q] = d[q] ≈ 0.
                    let mut bulge = e[q - 1];
                    e[q - 1] = FixedPoint::ZERO;
                    for j in (p..q).rev() {
                        // Givens to zero bulge against d[j]
                        let (cs, sn) = givens(d[j], bulge);
                        d[j] = cs * d[j] + sn * bulge;
                        if j > p {
                            // Bulge moves to e[j-1]
                            bulge = -sn * e[j - 1];
                            e[j - 1] = cs * e[j - 1];
                        }
                        // RIGHT rotation → update V columns j and q
                        for r in 0..n {
                            let v_rj = v_acc.get(r, j);
                            let v_rq = v_acc.get(r, q);
                            v_acc.set(r, j, cs * v_rj + sn * v_rq);
                            v_acc.set(r, q, -sn * v_rj + cs * v_rq);
                        }
                    }
                    deflated = true;
                }
            }

            // Check interior d[i] for i in p..q
            if !deflated {
                for i in p..q {
                    let d_thresh = convergence_threshold(
                        e[i].abs().max(
                            if i > 0 && i - 1 < e.len() { e[i.saturating_sub(1)].abs() }
                            else { FixedPoint::one() }
                        )
                    );
                    if d[i].abs() <= d_thresh {
                        // d[i] ≈ 0: chase e[i] to zero via LEFT Givens (rows)
                        let mut bulge = e[i];
                        e[i] = FixedPoint::ZERO;
                        for j in (i + 1)..=q {
                            let (cs, sn) = givens(d[j], bulge);
                            d[j] = cs * d[j] + sn * bulge;
                            if j < q {
                                bulge = -sn * e[j];
                                e[j] = cs * e[j];
                            }
                            // LEFT rotation → update U columns i and j
                            for r in 0..m {
                                let u_ri = u_acc.get(r, i);
                                let u_rj = u_acc.get(r, j);
                                u_acc.set(r, i, cs * u_ri + sn * u_rj);
                                u_acc.set(r, j, -sn * u_ri + cs * u_rj);
                            }
                        }
                        deflated = true;
                        break;
                    }
                }
            }

            if deflated {
                iter_count += 1;
                continue;
            }
        }

        // Wilkinson shift: eigenvalue of trailing 2×2 of BᵀB closest to d[q]²
        // The 2×2 block of BᵀB at bottom-right is:
        //   [d[q-1]² + e[q-2]²,   d[q-1]*e[q-1]  ]
        //   [d[q-1]*e[q-1],        d[q]² + e[q-1]² ]
        // (simplified when e[q-2] doesn't exist for the first row)
        let shift = {
            let dq = d[q];
            let eq_1 = e[q - 1];
            let dq_1 = d[q - 1];

            // Compute at storage tier — these are products of individual values,
            // not long sums, so compute-tier isn't critical here
            let f = dq_1 * dq_1 + if q >= 2 { e[q - 2] * e[q - 2] } else { FixedPoint::ZERO };
            let g = dq * dq + eq_1 * eq_1;
            let h = dq_1 * eq_1;

            // Eigenvalue of [[f, h], [h, g]] closer to g (Wilkinson)
            let half = FixedPoint::one() / FixedPoint::from_int(2);
            let diff = (f - g) * half;
            if diff.is_zero() && h.is_zero() {
                g
            } else {
                let disc_sq = diff * diff + h * h;
                let disc = disc_sq.try_sqrt().unwrap_or(diff.abs());
                let signed_disc = if diff.is_negative() { -disc } else { disc };
                g - h * h / (diff + signed_disc)
            }
        };

        // ── Implicit QR step (Golub-Kahan) ──
        // Chase the bulge from (p, p+1) to (q-1, q)
        let mut x = d[p] * d[p] - shift;
        let mut z = d[p] * e[p];

        for i in p..q {
            // ── Right Givens: all 2-element sums at compute tier ──
            let (cs, sn) = givens(x, z);

            if i > p {
                // e[i-1] = cs*e[i-1] + sn*z (2-element dot)
                e[i - 1] = FixedPoint::from_raw(compute_tier_dot_raw(
                    &[cs.raw(), sn.raw()], &[e[i - 1].raw(), z.raw()]
                ));
            }
            let old_di = d[i];
            let old_ei = e[i];
            let (new_di, new_ei) = apply_givens_compute(cs, sn, old_di, old_ei);
            d[i] = new_di;
            e[i] = new_ei;
            let bulge = sn * d[i + 1]; // single multiply — no accumulation needed
            d[i + 1] = cs * d[i + 1];

            // V accumulation at compute tier
            for r in 0..n {
                let (new_v0, new_v1) = apply_givens_compute(
                    cs, sn, v_acc.get(r, i), v_acc.get(r, i + 1));
                v_acc.set(r, i, new_v0);
                v_acc.set(r, i + 1, new_v1);
            }

            // ── Left Givens: all 2-element sums at compute tier ──
            x = d[i];
            z = bulge;
            let (cs2, sn2) = givens(x, z);

            // d[i] = cs2*d[i] + sn2*bulge (2-element dot)
            d[i] = FixedPoint::from_raw(compute_tier_dot_raw(
                &[cs2.raw(), sn2.raw()], &[d[i].raw(), bulge.raw()]
            ));
            let old_ei = e[i];
            let old_di1 = d[i + 1];
            let (new_ei, new_di1) = apply_givens_compute(cs2, sn2, old_ei, old_di1);
            e[i] = new_ei;
            d[i + 1] = new_di1;

            // U accumulation at compute tier
            for r in 0..m {
                let (new_u0, new_u1) = apply_givens_compute(
                    cs2, sn2, u_acc.get(r, i), u_acc.get(r, i + 1));
                u_acc.set(r, i, new_u0);
                u_acc.set(r, i + 1, new_u1);
            }

            // Set up for next iteration of the chase
            if i + 1 < q {
                x = e[i];
                z = sn2 * e[i + 1];
                e[i + 1] = cs2 * e[i + 1];
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

    // Copy remaining U columns (m > n case) — they stay as-is from identity
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
/// quasi-upper-triangular (real Schur form).
///
/// **Algorithm:**
/// 1. Reduce A to upper Hessenberg form H via Householder reflections
/// 2. Apply Francis implicit double-shift QR iteration to H
/// 3. Converged T has eigenvalues on diagonal (real) or in 2×2 blocks (complex pairs)
///
/// **FASC-UGOD strategy:** Householder reflections use compute-tier dot products
/// (same as QR decomposition). Francis shifts are computed from the trailing 2×2
/// block at storage tier — no transcendentals needed (just trace and determinant
/// of a 2×2, which are additions and multiplications).
pub fn schur_decompose(a: &FixedMatrix) -> Result<SchurDecomposition, OverflowDetected> {
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
    let two = FixedPoint::from_int(2);

    for k in 0..n.saturating_sub(2) {
        let col_len = n - k - 1;
        let start = k + 1;

        let x_raw: Vec<BinaryStorage> = (start..n).map(|i| h.get(i, k).raw()).collect();
        let norm_sq = FixedPoint::from_raw(compute_tier_dot_raw(&x_raw, &x_raw));
        if norm_sq.is_zero() {
            continue;
        }
        let norm_x = norm_sq.try_sqrt()?;
        let x_0 = h.get(start, k);
        let alpha = if x_0.is_negative() { norm_x } else { -norm_x };

        let mut v_hh = Vec::<FixedPoint>::with_capacity(col_len);
        v_hh.push(x_0 - alpha);
        for i in 1..col_len {
            v_hh.push(FixedPoint::from_raw(x_raw[i]));
        }
        let v_raw: Vec<BinaryStorage> = v_hh.iter().map(|fp| fp.raw()).collect();
        let vtv = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &v_raw));
        if vtv.is_zero() {
            continue;
        }

        // Left multiply: H[start..n, :] -= 2 v (vᵀ H[start..n, :]) / vᵀv
        for c in 0..n {
            let col_raw: Vec<BinaryStorage> = (start..n).map(|i| h.get(i, c).raw()).collect();
            let dot = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &col_raw));
            let scale = two * dot / vtv;
            for i in start..n {
                h.set(i, c, h.get(i, c) - scale * v_hh[i - start]);
            }
        }

        // Right multiply: H[:, start..n] -= 2 (H[:, start..n] v) vᵀ / vᵀv
        for r in 0..n {
            let row_raw: Vec<BinaryStorage> = (start..n).map(|c| h.get(r, c).raw()).collect();
            let dot = FixedPoint::from_raw(compute_tier_dot_raw(&row_raw, &v_raw));
            let scale = two * dot / vtv;
            for c in start..n {
                h.set(r, c, h.get(r, c) - scale * v_hh[c - start]);
            }
        }

        // Accumulate into Q: Q[:, start..n] -= 2 (Q v) vᵀ / vᵀv
        for r in 0..n {
            let q_row_raw: Vec<BinaryStorage> = (start..n).map(|c| q_acc.get(r, c).raw()).collect();
            let dot = FixedPoint::from_raw(compute_tier_dot_raw(&q_row_raw, &v_raw));
            let scale = two * dot / vtv;
            for c in start..n {
                q_acc.set(r, c, q_acc.get(r, c) - scale * v_hh[c - start]);
            }
        }
    }

    // ── Phase 2: Francis Implicit Double-Shift QR Iteration ──
    let max_iter = 30 * n * n;
    let mut iter_count = 0usize;
    let mut nn = n; // active submatrix is h[0..nn, 0..nn]

    while nn > 2 && iter_count < max_iter {
        // Find the lowest converged subdiagonal
        let thresh = convergence_threshold(
            h.get(nn - 1, nn - 1).abs().max(h.get(nn - 2, nn - 2).abs())
        );

        if h.get(nn - 1, nn - 2).abs() <= thresh {
            // 1×1 block converged at position nn-1
            nn -= 1;
            continue;
        }

        // Check for 2×2 block convergence
        if nn >= 3 {
            let thresh2 = convergence_threshold(
                h.get(nn - 2, nn - 2).abs().max(h.get(nn - 3, nn - 3).abs())
            );
            if h.get(nn - 2, nn - 3).abs() <= thresh2 {
                // 2×2 block at [nn-2..nn, nn-2..nn] has converged
                nn -= 2;
                continue;
            }
        }

        // Find the start of the active unreduced Hessenberg block
        let mut l = nn - 2;
        while l > 0 {
            let thresh_l = convergence_threshold(
                h.get(l, l).abs().max(h.get(l - 1, l - 1).abs())
            );
            if h.get(l, l - 1).abs() <= thresh_l {
                break;
            }
            l -= 1;
        }

        // Special case: 2×2 active block — apply single Givens rotation
        if l == nn - 2 {
            // Wilkinson single shift from the 2×2 block
            let a11 = h.get(l, l);
            let a12 = h.get(l, l + 1);
            let a21 = h.get(l + 1, l);
            let a22 = h.get(l + 1, l + 1);
            let half = FixedPoint::one() / two;
            let d_val = (a11 - a22) * half;
            let mu = if d_val.is_zero() && (a12 * a21).is_zero() {
                a22
            } else {
                let disc_sq = d_val * d_val + a12 * a21;
                let disc = disc_sq.abs().try_sqrt().unwrap_or(d_val.abs());
                let signed_disc = if d_val.is_negative() { -disc } else { disc };
                a22 - a21 * a12 / (d_val + signed_disc)
            };

            let x_val = h.get(l, l) - mu;
            let y_val = h.get(l + 1, l);
            let (cs, sn) = givens(x_val, y_val);

            // Apply from left: H[l..l+2, :] — compute tier
            for c in 0..n {
                let (new0, new1) = apply_givens_compute(
                    cs, sn, h.get(l, c), h.get(l + 1, c));
                h.set(l, c, new0);
                h.set(l + 1, c, new1);
            }
            // Apply from right: H[:, l..l+2] — compute tier
            for r in 0..nn {
                let (new0, new1) = apply_givens_compute(
                    cs, sn, h.get(r, l), h.get(r, l + 1));
                h.set(r, l, new0);
                h.set(r, l + 1, new1);
            }
            // Accumulate into Q — compute tier
            for r in 0..n {
                let (new0, new1) = apply_givens_compute(
                    cs, sn, q_acc.get(r, l), q_acc.get(r, l + 1));
                q_acc.set(r, l, new0);
                q_acc.set(r, l + 1, new1);
            }
            iter_count += 1;
            continue;
        }

        // Compute double shift from trailing 2×2 block
        // Eigenvalues are roots of x² - sx + p = 0
        let s = h.get(nn - 2, nn - 2) + h.get(nn - 1, nn - 1);     // trace
        let p = h.get(nn - 2, nn - 2) * h.get(nn - 1, nn - 1)
              - h.get(nn - 2, nn - 1) * h.get(nn - 1, nn - 2);     // determinant

        // First column of M = H² - sH + pI
        // m₁ = H[l,l]² + H[l,l+1]*H[l+1,l] - s*H[l,l] + p
        // m₂ = H[l+1,l] * (H[l,l] + H[l+1,l+1] - s)
        // m₃ = H[l+1,l] * H[l+2,l+1]  (only if l+2 < nn)
        let h_ll = h.get(l, l);
        let h_l1l = h.get(l + 1, l);
        let h_ll1 = h.get(l, l + 1);
        let h_l1l1 = h.get(l + 1, l + 1);

        let mut x = h_ll * h_ll + h_ll1 * h_l1l - s * h_ll + p;
        let mut y = h_l1l * (h_ll + h_l1l1 - s);
        let mut z = if l + 2 < nn { h_l1l * h.get(l + 2, l + 1) } else { FixedPoint::ZERO };

        // ── Bulge chase ──
        for k in l..nn.saturating_sub(2) {
            // Compute Householder to zero [y; z] in [x; y; z]
            let col_size = if k + 2 < nn { 3 } else { 2 };

            if col_size == 3 {
                let vec_raw = [x.raw(), y.raw(), z.raw()];
                let norm_sq = FixedPoint::from_raw(compute_tier_dot_raw(&vec_raw, &vec_raw));
                if norm_sq.is_zero() { break; }
                let norm_v = norm_sq.try_sqrt()?;
                let alpha = if x.is_negative() { norm_v } else { -norm_v };
                let v0 = x - alpha;
                let v1 = y;
                let v2 = z;
                let v_raw = [v0.raw(), v1.raw(), v2.raw()];
                let vtv = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &v_raw));
                if vtv.is_zero() { break; }

                // Apply from left: H[k..k+3, :] -= 2 v (vᵀ H) / vᵀv
                // All dot products at compute tier
                for c in 0..n {
                    let col_raw = [h.get(k, c).raw(), h.get(k + 1, c).raw(), h.get(k + 2, c).raw()];
                    let dot_val = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &col_raw));
                    let scale = two * dot_val / vtv;
                    h.set(k, c, h.get(k, c) - scale * v0);
                    h.set(k + 1, c, h.get(k + 1, c) - scale * v1);
                    h.set(k + 2, c, h.get(k + 2, c) - scale * v2);
                }

                // Apply from right: H[:, k..k+3] -= 2 (H v) vᵀ / vᵀv
                let c_end = nn.min(k + 4);
                for r in 0..c_end {
                    let row_raw = [h.get(r, k).raw(), h.get(r, k + 1).raw(), h.get(r, k + 2).raw()];
                    let dot_val = FixedPoint::from_raw(compute_tier_dot_raw(&row_raw, &v_raw));
                    let scale = two * dot_val / vtv;
                    h.set(r, k, h.get(r, k) - scale * v0);
                    h.set(r, k + 1, h.get(r, k + 1) - scale * v1);
                    h.set(r, k + 2, h.get(r, k + 2) - scale * v2);
                }

                // Accumulate into Q
                for r in 0..n {
                    let q_raw = [q_acc.get(r, k).raw(), q_acc.get(r, k + 1).raw(), q_acc.get(r, k + 2).raw()];
                    let dot_val = FixedPoint::from_raw(compute_tier_dot_raw(&q_raw, &v_raw));
                    let scale = two * dot_val / vtv;
                    q_acc.set(r, k, q_acc.get(r, k) - scale * v0);
                    q_acc.set(r, k + 1, q_acc.get(r, k + 1) - scale * v1);
                    q_acc.set(r, k + 2, q_acc.get(r, k + 2) - scale * v2);
                }
            } else {
                // 2×2 Givens rotation at the bottom of the chase — compute tier
                let (cs, sn) = givens(x, y);

                for c in 0..n {
                    let (new0, new1) = apply_givens_compute(
                        cs, sn, h.get(k, c), h.get(k + 1, c));
                    h.set(k, c, new0);
                    h.set(k + 1, c, new1);
                }
                for r in 0..nn {
                    let (new0, new1) = apply_givens_compute(
                        cs, sn, h.get(r, k), h.get(r, k + 1));
                    h.set(r, k, new0);
                    h.set(r, k + 1, new1);
                }
                for r in 0..n {
                    let (new0, new1) = apply_givens_compute(
                        cs, sn, q_acc.get(r, k), q_acc.get(r, k + 1));
                    q_acc.set(r, k, new0);
                    q_acc.set(r, k + 1, new1);
                }
            }

            // Prepare next bulge chase values
            if k + 3 < nn {
                x = h.get(k + 1, k);
                y = h.get(k + 2, k);
                z = if k + 3 < nn { h.get(k + 3, k) } else { FixedPoint::ZERO };
            } else if k + 2 < nn {
                x = h.get(k + 1, k);
                y = h.get(k + 2, k);
            }
        }

        iter_count += 1;
    }

    // Clean up near-zero subdiagonal entries
    for i in 1..n {
        let thresh = convergence_threshold(
            h.get(i, i).abs().max(h.get(i - 1, i - 1).abs())
        );
        if h.get(i, i - 1).abs() <= thresh {
            h.set(i, i - 1, FixedPoint::ZERO);
        }
    }

    Ok(SchurDecomposition { q: q_acc, t: h })
}
