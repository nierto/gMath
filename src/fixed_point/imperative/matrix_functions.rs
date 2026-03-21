//! L1D: Matrix functions — exp, log, sqrt, pow.
//!
//! All operations use ComputeMatrix internally (tier N+1 precision throughout).
//! Only one downscale per output element at the very end → 0-1 ULP.
//!
//! Internal `_compute` variants accept and return ComputeMatrix, enabling
//! chains like `matrix_pow` (log → scalar_mul → exp) to stay at compute
//! tier throughout — the matrix analog of FASC's BinaryCompute chain persistence.

use super::FixedPoint;
use super::FixedMatrix;
use super::decompose::lu_decompose;
use super::linalg::{convergence_threshold, upscale_to_compute, ComputeStorage};
use super::compute_matrix::{ComputeMatrix, compute_lu_decompose};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// Padé [6/6] coefficients for matrix exponential
// ============================================================================

const PADE_B: [&str; 7] = [
    "1",
    "0.5",
    "0.113636363636363636363",    // 5/44
    "0.015151515151515151515",    // 1/66
    "0.001262626262626262626",    // 1/792
    "0.000063131313131313131",    // 1/15840
    "0.000001503126503126503",    // 1/665280
];

fn pade_coeff_compute(k: usize) -> ComputeStorage {
    upscale_to_compute(FixedPoint::from_str(PADE_B[k]).raw())
}

// ============================================================================
// Matrix exponential: exp(A) via Padé [6/6] at compute tier
// ============================================================================

/// Compute-tier internal: exp(A) where A is already a ComputeMatrix.
/// Returns ComputeMatrix — no downscale. Used by matrix_pow for chaining.
pub(crate) fn matrix_exp_compute(a: &ComputeMatrix) -> Result<ComputeMatrix, OverflowDetected> {
    let n = a.rows();

    if a.frobenius_norm_compute().is_zero() {
        return Ok(ComputeMatrix::identity(n));
    }

    // Scaling: find s such that ||A||_1 / 2^s < 0.5
    // norm_1 downscales for comparison — only the integer count s matters
    let a_norm = a.norm_1_compute();
    let mut s = 0u32;
    let mut scale = a_norm;
    let one = FixedPoint::one();
    let half = one / FixedPoint::from_int(2);
    while scale >= half {
        scale = scale / (one + one);
        s += 1;
    }

    // B = A / 2^s at compute tier
    let mut b = a.copy();
    for _ in 0..s {
        b = b.halve();
    }

    // Powers at compute tier
    let b2 = b.mat_mul(&b);
    let b4 = b2.mat_mul(&b2);
    let b6 = b2.mat_mul(&b4);

    let id = ComputeMatrix::identity(n);

    // Padé coefficients at compute tier
    let c0 = pade_coeff_compute(0);
    let c1 = pade_coeff_compute(1);
    let c2 = pade_coeff_compute(2);
    let c3 = pade_coeff_compute(3);
    let c4 = pade_coeff_compute(4);
    let c5 = pade_coeff_compute(5);
    let c6 = pade_coeff_compute(6);

    // V = c0*I + c2*B² + c4*B⁴ + c6*B⁶  (even terms)
    let v = id.scalar_mul(c0)
        .add(&b2.scalar_mul(c2))
        .add(&b4.scalar_mul(c4))
        .add(&b6.scalar_mul(c6));

    // P_odd = c1*I + c3*B² + c5*B⁴
    let p_odd = ComputeMatrix::identity(n).scalar_mul(c1)
        .add(&b2.scalar_mul(c3))
        .add(&b4.scalar_mul(c5));

    // U = B * P_odd
    let u = b.mat_mul(&p_odd);

    // N = V + U, D = V - U
    let n_mat = v.add(&u);
    let d_mat = v.sub(&u);

    // Solve D * R = N via compute-tier LU
    let lu_d = compute_lu_decompose(&d_mat)?;
    let mut exp_b = ComputeMatrix::new(n, n);
    for j in 0..n {
        let n_col = n_mat.col_vec(j);
        let r_col = lu_d.solve(&n_col)?;
        for i in 0..n {
            exp_b.set(i, j, r_col[i]);
        }
    }

    // Unsquare at compute tier
    let mut result = exp_b;
    for _ in 0..s {
        result = result.mat_mul(&result);
    }

    Ok(result)
}

/// Matrix exponential: exp(A) via Padé [6/6] with scaling-and-squaring.
///
/// **Precision:** Entire Padé evaluation, LU solve, and squaring performed
/// at compute tier (tier N+1). Single downscale at the end → 0-1 ULP.
pub fn matrix_exp(a: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> {
    assert!(a.is_square(), "matrix_exp: matrix must be square");
    let a_c = ComputeMatrix::from_fixed_matrix(a);
    Ok(matrix_exp_compute(&a_c)?.to_fixed_matrix())
}

// ============================================================================
// Matrix square root: Denman-Beavers at compute tier
// ============================================================================

/// Compute-tier internal: sqrt(A) where A is already a ComputeMatrix.
/// Returns ComputeMatrix — no downscale. Used by matrix_log_compute for chaining.
pub(crate) fn matrix_sqrt_compute(a: &ComputeMatrix) -> Result<ComputeMatrix, OverflowDetected> {
    let n = a.rows();
    let max_iter = 50;
    let threshold = convergence_threshold(a.frobenius_norm_compute());

    let mut y = a.copy();
    let mut z = ComputeMatrix::identity(n);

    for _ in 0..max_iter {
        // Snapshot Y for convergence check (compute-tier copy)
        let y_prev = y.copy();

        let z_inv = compute_lu_decompose(&z)?.inverse()?;
        let y_inv = compute_lu_decompose(&y)?.inverse()?;

        y = y.add(&z_inv).halve();
        z = z.add(&y_inv).halve();

        // Convergence: ||Y - Y_prev||_F at compute tier, single downscale for comparison
        let diff = y.sub(&y_prev);
        let diff_norm = diff.frobenius_norm_compute();
        if diff_norm < threshold {
            return Ok(y);
        }
    }

    Ok(y)
}

/// Matrix square root: A^{1/2} via Denman-Beavers iteration at compute tier.
///
/// **Precision:** Entire iteration at tier N+1. Single downscale at the end.
pub fn matrix_sqrt(a: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> {
    assert!(a.is_square(), "matrix_sqrt: matrix must be square");
    let a_c = ComputeMatrix::from_fixed_matrix(a);
    Ok(matrix_sqrt_compute(&a_c)?.to_fixed_matrix())
}

// ============================================================================
// Matrix logarithm: inverse scaling-and-squaring at compute tier
// ============================================================================

/// Compute-tier internal: log(A) where A is already a ComputeMatrix.
/// Returns ComputeMatrix — no downscale. Used by matrix_pow for chaining.
///
/// The sqrt loop now stays entirely at compute tier via matrix_sqrt_compute.
/// Previously: N downscale-upscale cycles (one per sqrt). Now: zero.
pub(crate) fn matrix_log_compute(a: &ComputeMatrix) -> Result<ComputeMatrix, OverflowDetected> {
    let n = a.rows();
    let id = ComputeMatrix::identity(n);
    let quarter = FixedPoint::one() / FixedPoint::from_int(4);

    // Phase 1: Repeatedly take sqrt at compute tier until ||A_s - I|| < 0.25
    let mut a_s = a.copy();
    let mut s = 0u32;
    for _ in 0..30 {
        // Convergence check: downscale difference for comparison only
        let diff = a_s.sub(&id);
        let diff_norm = diff.frobenius_norm_compute();
        if diff_norm < quarter {
            break;
        }
        a_s = matrix_sqrt_compute(&a_s)?;
        s += 1;
    }

    // Phase 2: Horner Taylor of log(I + X) at compute tier
    let x = a_s.sub(&id);

    // Horner: log(I+X) = X * (I - X/2 * (I - 2X/3 * (I - 3X/4 * ...)))
    let num_terms = 22;
    let mut horner = ComputeMatrix::identity(n);
    for k in (1..num_terms).rev() {
        let coeff = FixedPoint::from_int(k as i32) / FixedPoint::from_int((k + 1) as i32);
        let coeff_compute = upscale_to_compute(coeff.raw());
        let x_scaled = x.scalar_mul(coeff_compute);
        horner = id.sub(&x_scaled.mat_mul(&horner));
    }
    let mut log_approx = x.mat_mul(&horner);

    // Phase 3: Unscale: log(A) = 2^s * log(A_s)
    for _ in 0..s {
        log_approx = log_approx.add(&log_approx);
    }

    Ok(log_approx)
}

/// Matrix logarithm: log(A) via inverse scaling-and-squaring at compute tier.
///
/// **Precision:** Square roots and Horner Taylor evaluation at tier N+1.
/// Single downscale at the end.
pub fn matrix_log(a: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> {
    assert!(a.is_square(), "matrix_log: matrix must be square");
    let a_c = ComputeMatrix::from_fixed_matrix(a);
    Ok(matrix_log_compute(&a_c)?.to_fixed_matrix())
}

// ============================================================================
// Matrix power: A^p for real scalar p
// ============================================================================

/// Matrix power: A^p = exp(p * log(A)) for real scalar p.
///
/// **Precision:** Entire log → scalar_mul → exp chain at compute tier.
/// Previously: 2 materialization boundaries (log→storage, storage→exp).
/// Now: zero. Single downscale at the end.
pub fn matrix_pow(a: &FixedMatrix, p: FixedPoint) -> Result<FixedMatrix, OverflowDetected> {
    assert!(a.is_square(), "matrix_pow: matrix must be square");
    let n = a.rows();

    if p.is_zero() {
        return Ok(FixedMatrix::identity(n));
    }
    if p == FixedPoint::one() {
        return Ok(a.clone());
    }
    if p == -FixedPoint::one() {
        return lu_decompose(a)?.inverse();
    }

    // Entire chain at compute tier: log → scalar_mul → exp
    let a_c = ComputeMatrix::from_fixed_matrix(a);
    let log_a = matrix_log_compute(&a_c)?;
    let p_c = upscale_to_compute(p.raw());
    let p_log_a = log_a.scalar_mul(p_c);
    let result = matrix_exp_compute(&p_log_a)?;
    Ok(result.to_fixed_matrix())
}
