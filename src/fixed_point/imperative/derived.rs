//! L1C: Derived matrix operations built on top of L1B decompositions.
//!
//! Norms, least-squares, SPD inverse, condition number, pseudoinverse, rank, nullspace.

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::linalg::compute_tier_dot_raw;
use super::decompose::{lu_decompose, qr_decompose, cholesky_decompose, svd_decompose};
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// Matrix norms
// ============================================================================

/// Frobenius norm: ||A||_F = sqrt(sum of squares of all entries).
///
/// Uses compute-tier accumulation for the sum of squares.
pub fn frobenius_norm(a: &FixedMatrix) -> FixedPoint {
    let data = a.data_slice();
    let raw: Vec<BinaryStorage> = data.iter().map(|fp| fp.raw()).collect();
    let sum_sq = FixedPoint::from_raw(compute_tier_dot_raw(&raw, &raw));
    sum_sq.sqrt()
}

/// 1-norm: max absolute column sum, accumulated at compute tier.
pub fn norm_1(a: &FixedMatrix) -> FixedPoint {
    let ones: Vec<BinaryStorage> = vec![FixedPoint::one().raw(); a.rows()];
    let mut max_sum = FixedPoint::ZERO;
    for j in 0..a.cols() {
        let abs_col: Vec<BinaryStorage> = (0..a.rows()).map(|i| a.get(i, j).abs().raw()).collect();
        let col_sum = FixedPoint::from_raw(compute_tier_dot_raw(&abs_col, &ones));
        if col_sum > max_sum {
            max_sum = col_sum;
        }
    }
    max_sum
}

/// Infinity-norm: max absolute row sum, accumulated at compute tier.
pub fn norm_inf(a: &FixedMatrix) -> FixedPoint {
    let ones: Vec<BinaryStorage> = vec![FixedPoint::one().raw(); a.cols()];
    let mut max_sum = FixedPoint::ZERO;
    for i in 0..a.rows() {
        let abs_row: Vec<BinaryStorage> = (0..a.cols()).map(|j| a.get(i, j).abs().raw()).collect();
        let row_sum = FixedPoint::from_raw(compute_tier_dot_raw(&abs_row, &ones));
        if row_sum > max_sum {
            max_sum = row_sum;
        }
    }
    max_sum
}

// ============================================================================
// Least squares
// ============================================================================

/// Least-squares solve: min ||Ax - b||_2 via QR decomposition.
///
/// Works for m >= n (overdetermined or square systems).
pub fn least_squares(a: &FixedMatrix, b: &FixedVector) -> Result<FixedVector, OverflowDetected> {
    let qr = qr_decompose(a)?;
    qr.solve(b)
}

// ============================================================================
// SPD inverse
// ============================================================================

/// Inverse of a symmetric positive-definite matrix via Cholesky.
///
/// Approximately 2× faster than LU-based inverse for SPD matrices.
pub fn inverse_spd(a: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> {
    let chol = cholesky_decompose(a)?;
    let n = a.rows();
    let mut inv = FixedMatrix::new(n, n);
    for j in 0..n {
        let mut e_j = FixedVector::new(n);
        e_j[j] = FixedPoint::one();
        let col = chol.solve(&e_j)?;
        for i in 0..n {
            inv.set(i, j, col[i]);
        }
    }
    Ok(inv)
}

// ============================================================================
// Condition number estimate (1-norm based, no SVD required)
// ============================================================================

/// Condition number estimate: κ_1(A) = ||A||_1 * ||A^{-1}||_1.
///
/// Uses LU-based inverse. This gives the exact 1-norm condition number,
/// not an estimate. The 2-norm condition number (σ_max / σ_min) requires SVD.
pub fn condition_number_1(a: &FixedMatrix) -> Result<FixedPoint, OverflowDetected> {
    let lu = lu_decompose(a)?;
    let inv = lu.inverse()?;
    Ok(norm_1(a) * norm_1(&inv))
}

// ============================================================================
// Solve variants (convenience wrappers)
// ============================================================================

/// Solve Ax = b using LU decomposition.
pub fn solve(a: &FixedMatrix, b: &FixedVector) -> Result<FixedVector, OverflowDetected> {
    let lu = lu_decompose(a)?;
    lu.solve(b)
}

/// Solve Ax = b for SPD matrix using Cholesky.
pub fn solve_spd(a: &FixedMatrix, b: &FixedVector) -> Result<FixedVector, OverflowDetected> {
    let chol = cholesky_decompose(a)?;
    chol.solve(b)
}

/// Determinant via LU decomposition.
pub fn determinant(a: &FixedMatrix) -> Result<FixedPoint, OverflowDetected> {
    let lu = lu_decompose(a)?;
    Ok(lu.determinant())
}

/// Matrix inverse via LU decomposition.
pub fn inverse(a: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> {
    let lu = lu_decompose(a)?;
    lu.inverse()
}

// ============================================================================
// SVD-based operations
// ============================================================================

/// Default singular value threshold factor.
///
/// Singular values σ_i with σ_i < threshold * σ_max are treated as zero.
/// The threshold is `max(m, n) * quantum_relative`, where quantum_relative
/// is approximately the storage tier quantum relative to σ_max.
///
/// For manual control, use `pseudoinverse_with_threshold`.
fn default_sv_threshold(sigma: &FixedVector, m: usize, n: usize) -> FixedPoint {
    if sigma.len() == 0 {
        return FixedPoint::one();
    }
    let sigma_max = sigma[0]; // already sorted descending
    if sigma_max.is_zero() {
        return FixedPoint::one();
    }
    // threshold ≈ max(m,n) * σ_max * ε, where ε ≈ 2^(-FRAC_BITS/2)
    // Using convergence_threshold gives us σ_max >> (FRAC_BITS/2), then multiply by max(m,n)
    use super::linalg::convergence_threshold;
    let base = convergence_threshold(sigma_max);
    let factor = FixedPoint::from_int(m.max(n) as i32);
    factor * base
}

/// Moore-Penrose pseudoinverse: A⁺ = V Σ⁺ Uᵀ.
///
/// Singular values below the default threshold (based on matrix dimensions
/// and storage precision) are treated as zero.
///
/// Works for any m×n matrix — not just square or full-rank.
pub fn pseudoinverse(a: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> {
    let svd = svd_decompose(a)?;
    let (m, n) = (a.rows(), a.cols());
    let k = svd.sigma.len();
    let thresh = default_sv_threshold(&svd.sigma, m, n);

    // A⁺ = V Σ⁺ Uᵀ = Vᵀᵀ Σ⁺ Uᵀ
    // Σ⁺ is n×m diagonal with 1/σ_i for non-negligible σ_i
    let mut result = FixedMatrix::new(n, m);
    for i in 0..k {
        if svd.sigma[i] <= thresh {
            break; // remaining are smaller (sorted descending)
        }
        let inv_sigma = FixedPoint::one() / svd.sigma[i];
        // Rank-1 update: result += (1/σ_i) * V[:,i] * U[:,i]ᵀ
        // V[:,i] = Vᵀ[i,:] transposed, U[:,i] from U
        for r in 0..n {
            for c in 0..m {
                let v_ri = svd.vt.get(i, r); // V = Vᵀ transposed: V[r,i] = Vᵀ[i,r]
                let u_ci = svd.u.get(c, i);
                result.set(r, c, result.get(r, c) + inv_sigma * v_ri * u_ci);
            }
        }
    }

    Ok(result)
}

/// Pseudoinverse with a user-specified threshold.
///
/// Singular values σ_i with σ_i < threshold are treated as zero.
pub fn pseudoinverse_with_threshold(
    a: &FixedMatrix,
    threshold: FixedPoint,
) -> Result<FixedMatrix, OverflowDetected> {
    let svd = svd_decompose(a)?;
    let (m, n) = (a.rows(), a.cols());
    let k = svd.sigma.len();

    let mut result = FixedMatrix::new(n, m);
    for i in 0..k {
        if svd.sigma[i] <= threshold {
            break;
        }
        let inv_sigma = FixedPoint::one() / svd.sigma[i];
        for r in 0..n {
            for c in 0..m {
                let v_ri = svd.vt.get(i, r);
                let u_ci = svd.u.get(c, i);
                result.set(r, c, result.get(r, c) + inv_sigma * v_ri * u_ci);
            }
        }
    }

    Ok(result)
}

/// Numerical rank: count of singular values above threshold.
///
/// Uses the default threshold (based on matrix dimensions and precision).
pub fn rank(a: &FixedMatrix) -> Result<usize, OverflowDetected> {
    let svd = svd_decompose(a)?;
    let thresh = default_sv_threshold(&svd.sigma, a.rows(), a.cols());
    let mut r = 0;
    for i in 0..svd.sigma.len() {
        if svd.sigma[i] > thresh {
            r += 1;
        } else {
            break; // sorted descending
        }
    }
    Ok(r)
}

/// 2-norm condition number: κ₂(A) = σ_max / σ_min.
///
/// Returns `Err(DivisionByZero)` if the matrix is rank-deficient
/// (smallest singular value is zero or below threshold).
pub fn condition_number_2(a: &FixedMatrix) -> Result<FixedPoint, OverflowDetected> {
    let svd = svd_decompose(a)?;
    let k = svd.sigma.len();
    if k == 0 {
        return Err(OverflowDetected::DivisionByZero);
    }
    let sigma_max = svd.sigma[0];
    let sigma_min = svd.sigma[k - 1];
    if sigma_min.is_zero() {
        return Err(OverflowDetected::DivisionByZero);
    }
    Ok(sigma_max / sigma_min)
}

/// Nullspace basis: columns of V corresponding to near-zero singular values.
///
/// Returns an n×k matrix whose columns span the nullspace, where k = n - rank.
/// Uses the default threshold for determining which singular values are "zero".
pub fn nullspace(a: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> {
    let svd = svd_decompose(a)?;
    let n = a.cols();
    let thresh = default_sv_threshold(&svd.sigma, a.rows(), n);

    // Find rank
    let mut r = 0;
    for i in 0..svd.sigma.len() {
        if svd.sigma[i] > thresh {
            r += 1;
        } else {
            break;
        }
    }

    let null_dim = n - r;
    if null_dim == 0 {
        return Ok(FixedMatrix::new(n, 0));
    }

    // Nullspace columns are V[:,r..n] = rows r..n of Vᵀ, transposed
    let mut basis = FixedMatrix::new(n, null_dim);
    for j in 0..null_dim {
        for i in 0..n {
            basis.set(i, j, svd.vt.get(r + j, i)); // V[i, r+j] = Vᵀ[r+j, i]
        }
    }

    Ok(basis)
}
