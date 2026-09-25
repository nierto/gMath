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
use super::compute_matrix::{compute_lu_decompose, ComputeLU, ComputeMatrix};
use super::linalg::{
    compute_tier_sub_dot_compute, upscale_to_compute, round_to_storage, compute_abs, compute_product,
    householder_vector_compute, reflect_compute, ComputeStorage,
    compute_deflation_threshold, compute_noise_floor, compute_stagnation_threshold,
    compute_scale_up, downscale_shifted_to_storage, scale_up_exponent,
    Rotation, STAGNATION_SWEEPS,
};
use super::wide_acc::{exact_dot_compute, exact_sub_dot_compute, narrow_product_to_compute};
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
///
/// `l` and `u` are the compute-tier factors rounded once to storage, for
/// inspection. `solve`, `inverse`, `refine` and `determinant` use the
/// compute-tier factors themselves, kept alongside.
#[derive(Clone, Debug)]
pub struct LUDecomposition {
    pub l: FixedMatrix,
    pub u: FixedMatrix,
    pub perm: Vec<usize>,
    pub num_swaps: usize,
    compute: ComputeLU,
}

/// LU decomposition with partial pivoting (Doolittle, compute-tier).
///
/// For an n×n matrix A, computes PA = LU where P is a permutation,
/// L is unit lower triangular, and U is upper triangular.
///
/// **Precision strategy:** the factorization runs at the compute tier
/// throughout: every entry is an exact sum of products rounded once to the
/// compute tier, and later entries are formed from those compute-tier
/// entries, never from storage-rounded ones. The factors are rounded to
/// storage once, for the public `l` and `u`. Before 0.6.4 every entry was
/// rounded to storage and reused (up to 5 units in L and U, 114 units in
/// `solve`, 70 in `determinant` on well-conditioned 4 x 4 systems).
///
/// Returns `Err(DivisionByZero)` if the matrix is singular.
pub fn lu_decompose(a: &FixedMatrix) -> Result<LUDecomposition, OverflowDetected> {
    assert!(a.is_square(), "lu_decompose: matrix must be square");
    let compute = compute_lu_decompose(&ComputeMatrix::from_fixed_matrix(a))?;
    Ok(LUDecomposition {
        l: narrow_matrix(compute.l())?,
        u: narrow_matrix(compute.u())?,
        perm: compute.perm().to_vec(),
        num_swaps: compute.num_swaps(),
        compute,
    })
}

impl LUDecomposition {
    /// Solve Ax = b: forward then back substitution on the compute-tier
    /// factors, every sum exact, the solution rounded once.
    pub fn solve(&self, b: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let n = self.l.rows();
        assert_eq!(b.len(), n, "LU solve: dimension mismatch");
        let bc: Vec<ComputeStorage> = (0..n).map(|i| upscale_to_compute(b[i].raw())).collect();
        narrow_vector(&self.compute.solve(&bc)?)
    }

    /// Determinant: det(A) = (-1)^num_swaps * product(U diagonal), formed at
    /// the compute tier from the compute-tier factor and rounded once.
    pub fn determinant(&self) -> FixedPoint {
        FixedPoint::from_raw(round_to_storage(self.compute.determinant()))
    }

    /// Iterative refinement: the residual `b - Ax` exact at the compute tier,
    /// the correction solved at the compute tier, `x + dx` rounded once.
    pub fn refine(&self, a: &FixedMatrix, b: &FixedVector, x: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let n = a.rows();
        let x_raw: Vec<BinaryStorage> = (0..n).map(|j| x[j].raw()).collect();
        let r: Vec<ComputeStorage> = (0..n)
            .map(|i| compute_tier_sub_dot_compute(b[i].raw(), &a.row_raw_range(i, 0, n), &x_raw))
            .collect();
        let dx = self.compute.solve(&r)?;
        let refined: Result<Vec<ComputeStorage>, OverflowDetected> = (0..n)
            .map(|i| compute_checked_add(upscale_to_compute(x[i].raw()), dx[i]))
            .collect();
        narrow_vector(&refined?)
    }

    /// Compute A^{-1} by solving AX = I column by column at the compute tier,
    /// every entry rounded once.
    pub fn inverse(&self) -> Result<FixedMatrix, OverflowDetected> {
        narrow_matrix(&self.compute.inverse()?)
    }
}

/// A compute-tier matrix rounded to storage, entry by entry (checked).
fn narrow_matrix(c: &ComputeMatrix) -> Result<FixedMatrix, OverflowDetected> {
    let mut out = FixedMatrix::new(c.rows(), c.cols());
    for i in 0..c.rows() {
        for j in 0..c.cols() {
            out.set(i, j, FixedPoint::from_raw(downscale_to_storage(c.get(i, j))?));
        }
    }
    Ok(out)
}

/// A compute-tier vector rounded to storage (checked).
fn narrow_vector(c: &[ComputeStorage]) -> Result<FixedVector, OverflowDetected> {
    let values: Result<Vec<FixedPoint>, OverflowDetected> =
        c.iter().map(|v| downscale_to_storage(*v).map(FixedPoint::from_raw)).collect();
    Ok(FixedVector::from_slice(&values?))
}

// ============================================================================
// QR Decomposition via Householder Reflections
// ============================================================================

/// Result of QR decomposition via Householder reflections: A = QR.
///
/// `q` and `r` are the compute-tier factors rounded once to storage;
/// `solve` uses the compute-tier factors, kept alongside.
#[derive(Clone, Debug)]
pub struct QRDecomposition {
    pub q: FixedMatrix,
    pub r: FixedMatrix,
    compute_q: ComputeMatrix,
    compute_r: ComputeMatrix,
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

    // R and Q stay at the compute tier through every reflection and are
    // rounded to storage once at the end: exact sums, each update one exact
    // quotient (compute-tier Householder kernels). Before 0.6.4 R and Q were
    // rounded to storage after every reflection, so an entry touched by k
    // reflections carried k roundings (4 units on a 3 x 3 at Q16.16).
    let mut r = ComputeMatrix::from_fixed_matrix(a);
    let mut q = ComputeMatrix::identity(m);

    for k in 0..n {
        let x: Vec<ComputeStorage> = (k..m).map(|i| r.get(i, k)).collect();
        let Some((v, vv)) = householder_vector_compute(&x)? else { continue };

        // R <- H R, column by column
        for j in k..n {
            let mut col: Vec<ComputeStorage> = (k..m).map(|i| r.get(i, j)).collect();
            reflect_compute(&mut col, &v, vv)?;
            for (i, c) in (k..m).zip(col) {
                r.set(i, j, c);
            }
        }
        // the reflected column is (alpha, 0, ..., 0) exactly
        for i in (k + 1)..m {
            r.set(i, k, make_compute_int(0));
        }

        // Q <- Q H, row by row
        for i in 0..m {
            let mut row: Vec<ComputeStorage> = (k..m).map(|j| q.get(i, j)).collect();
            reflect_compute(&mut row, &v, vv)?;
            for (j, c) in (k..m).zip(row) {
                q.set(i, j, c);
            }
        }
    }

    Ok(QRDecomposition { q: narrow_matrix(&q)?, r: narrow_matrix(&r)?, compute_q: q, compute_r: r })
}

impl QRDecomposition {
    /// Solve Ax = b via R^{-1} Q^T b on the compute-tier factors: `Q^T b`
    /// exact sums, back substitution at the compute tier, the solution
    /// rounded once (before 0.6.4 on the storage factors: up to 27 units on
    /// well-conditioned 4 x 4 systems).
    pub fn solve(&self, b: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let m = self.q.rows();
        let n = self.r.cols();
        assert_eq!(b.len(), m, "QR solve: dimension mismatch");

        let bc: Vec<ComputeStorage> = (0..m).map(|j| upscale_to_compute(b[j].raw())).collect();
        let mut qtb = Vec::with_capacity(m);
        for i in 0..m {
            let q_col: Vec<ComputeStorage> = (0..m).map(|j| self.compute_q.get(j, i)).collect();
            qtb.push(narrow_product_to_compute(exact_dot_compute(&q_col, &bc)?)?);
        }

        let mut x = vec![make_compute_int(0); n];
        for i in (0..n).rev() {
            let diag = self.compute_r.get(i, i);
            if compute_is_zero(&diag) {
                return Err(OverflowDetected::DivisionByZero);
            }
            let r_row: Vec<ComputeStorage> = (i + 1..n).map(|j| self.compute_r.get(i, j)).collect();
            let numerator = exact_sub_dot_compute(qtb[i], &r_row, &x[i + 1..n])?;
            x[i] = compute_divide(numerator, diag)?;
        }
        narrow_vector(&x)
    }
}

// ============================================================================
// Cholesky Decomposition (A = LL^T for SPD matrices)
// ============================================================================

/// Result of Cholesky decomposition: A = LL^T.
///
/// `l` is lower triangular with positive diagonal entries: the compute-tier
/// factor rounded once to storage. `solve` and `determinant` use the
/// compute-tier factor, kept alongside.
#[derive(Clone, Debug)]
pub struct CholeskyDecomposition {
    pub l: FixedMatrix,
    compute: ComputeMatrix,
}

/// Cholesky decomposition for symmetric positive-definite matrices.
///
/// Returns `Err(DomainError)` if the matrix is not positive-definite.
///
/// **Precision strategy:** the factor is built at the compute tier: each
/// diagonal entry `sqrt(A[i][i] - sum L[i][k]^2)` and each off-diagonal
/// `(A[j][i] - sum L[j][k] L[i][k]) / L[i][i]` from exact sums of the
/// compute-tier entries before it, rounded once to storage for `l`. Before
/// 0.6.4 each entry was rounded to storage and reused.
pub fn cholesky_decompose(a: &FixedMatrix) -> Result<CholeskyDecomposition, OverflowDetected> {
    assert!(a.is_square(), "cholesky_decompose: matrix must be square");
    let n = a.rows();
    let mut lc = ComputeMatrix::new(n, n);

    for i in 0..n {
        let row_i: Vec<ComputeStorage> = (0..i).map(|k| lc.get(i, k)).collect();
        let diag = exact_sub_dot_compute(upscale_to_compute(a.get(i, i).raw()), &row_i, &row_i)?;
        // positive-definiteness, decided at the compute tier (before sqrt)
        if compute_is_negative(&diag) || compute_is_zero(&diag) {
            return Err(OverflowDetected::DomainError);
        }
        let l_ii = sqrt_at_compute_tier(diag);
        lc.set(i, i, l_ii);
        for j in (i + 1)..n {
            let row_j: Vec<ComputeStorage> = (0..i).map(|k| lc.get(j, k)).collect();
            let numerator = exact_sub_dot_compute(upscale_to_compute(a.get(j, i).raw()), &row_j, &row_i)?;
            lc.set(j, i, compute_divide(numerator, l_ii)?);
        }
    }

    Ok(CholeskyDecomposition { l: narrow_matrix(&lc)?, compute: lc })
}

impl CholeskyDecomposition {
    /// Solve Ax = b: forward (Ly = b), then back (L^T x = y), on the
    /// compute-tier factor with exact sums; the solution rounded once.
    pub fn solve(&self, b: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let n = self.l.rows();
        assert_eq!(b.len(), n, "Cholesky solve: dimension mismatch");
        let l = &self.compute;

        let mut y = vec![make_compute_int(0); n];
        for i in 0..n {
            let l_row: Vec<ComputeStorage> = (0..i).map(|k| l.get(i, k)).collect();
            let numerator = exact_sub_dot_compute(upscale_to_compute(b[i].raw()), &l_row, &y[..i])?;
            y[i] = compute_divide(numerator, l.get(i, i))?;
        }

        let mut x = vec![make_compute_int(0); n];
        for i in (0..n).rev() {
            let lt_row: Vec<ComputeStorage> = (i + 1..n).map(|k| l.get(k, i)).collect();
            let numerator = exact_sub_dot_compute(y[i], &lt_row, &x[i + 1..n])?;
            x[i] = compute_divide(numerator, l.get(i, i))?;
        }
        narrow_vector(&x)
    }

    /// Determinant: det(A) = product(L[i][i])^2, formed at the compute tier
    /// and rounded once (before 0.6.4 a chain of storage products).
    pub fn determinant(&self) -> FixedPoint {
        let n = self.l.rows();
        let mut det_l = make_compute_int(1);
        for i in 0..n {
            det_l = compute_multiply(det_l, self.compute.get(i, i));
        }
        FixedPoint::from_raw(round_to_storage(compute_multiply(det_l, det_l)))
    }
}

// ============================================================================
// Shared kernels of the iterative decompositions
// ============================================================================
//
// Jacobi, Golub-Kahan and Francis converge only if the orthogonal transforms
// they apply inject less rounding noise than their convergence tests resolve.
// Their state (the matrix being reduced and the accumulated transforms) is
// carried at the compute tier, every transformed entry rounded once there
// from an exact accumulator (`Rotation::apply_compute`,
// `householder_vector_compute` and `reflect_compute` in `linalg`), and the
// convergence tests work at the compute scale; the results are rounded to
// storage once. Every step is checked: leaving the range is a `TierOverflow`,
// never a wrap.

/// Iteration budget of the QR-type iterations: this many steps per n².
const ITERATIONS_PER_N_SQUARED: usize = 30;

/// Sweep budget of the Jacobi iteration.
const JACOBI_MAX_SWEEPS: usize = 100;

/// Francis iterations on one block, after its two exceptional shifts, before
/// a stagnant block may be taken to sit at the precision floor.
const SCHUR_FLOOR_ITERATIONS: usize = 30;

/// Reflect column `col` of a compute-tier matrix, rows `start..start + v.len()`.
fn reflect_compute_column(
    mat: &mut ComputeMatrix, col: usize, start: usize, v: &[ComputeStorage], v_dot_v: ComputeStorage,
) -> Result<(), OverflowDetected> {
    let mut w: Vec<ComputeStorage> = (start..start + v.len()).map(|i| mat.get(i, col)).collect();
    reflect_compute(&mut w, v, v_dot_v)?;
    for (k, value) in w.into_iter().enumerate() {
        mat.set(start + k, col, value);
    }
    Ok(())
}

/// Reflect row `row` of a compute-tier matrix, columns `start..start + v.len()`.
fn reflect_compute_row(
    mat: &mut ComputeMatrix, row: usize, start: usize, v: &[ComputeStorage], v_dot_v: ComputeStorage,
) -> Result<(), OverflowDetected> {
    let mut w: Vec<ComputeStorage> = (start..start + v.len()).map(|c| mat.get(row, c)).collect();
    reflect_compute(&mut w, v, v_dot_v)?;
    for (k, value) in w.into_iter().enumerate() {
        mat.set(row, start + k, value);
    }
    Ok(())
}

/// Rotate columns `a` and `b` of every row of a compute-tier matrix, each
/// entry rounded once at the compute tier.
fn rotate_compute_columns(mat: &mut ComputeMatrix, a: usize, b: usize, rot: &Rotation) -> Result<(), OverflowDetected> {
    for r in 0..mat.rows() {
        let (new_a, new_b) = rot.apply_compute(mat.get(r, a), mat.get(r, b))?;
        mat.set(r, a, new_a);
        mat.set(r, b, new_b);
    }
    Ok(())
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
/// 3. Converged when a whole sweep finds every off-diagonal entry within
///    `2^-(3F/2)` of its two diagonal entries (relative), floored at
///    `2^-(3F/2)` absolute. The absolute floor matters: an exact zero
///    eigenvalue pair is computed as rounding noise, which a purely relative
///    test never passes. A run whose largest off-diagonal entry has not
///    decreased for five sweeps has reached the precision floor and is
///    accepted only if every entry is within one storage unit relative.
///
/// **Precision:** the matrix and the eigenvector accumulator are carried at
/// the compute tier through every rotation (coefficients too), each updated
/// entry rounded once at the compute tier from its exact value, and rounded
/// to storage once at the end. Measured against mpmath on well-separated
/// spectra: eigenvalues and eigenvectors within one unit on every profile and
/// split (0.6.3 carried them at storage between rotations).
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

    // S and the eigenvector accumulator V stay at the compute tier through
    // every rotation and are rounded to storage once at the end.
    let mut s = ComputeMatrix::from_fixed_matrix(a);
    let mut v = ComputeMatrix::identity(n);

    let mut converged = false;
    let mut best_off: Option<ComputeStorage> = None;
    let mut stagnant = 0usize;
    for _sweep in 0..max_sweeps {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let bound = compute_deflation_threshold(compute_abs(s.get(p, p)).max(compute_abs(s.get(q, q))));
                if compute_abs(s.get(p, q)) > bound {
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
    if !compute_is_zero(&largest) {
        jacobi_rotate(&mut s, &mut v, p, q)?;
    }

    // Eigenvalues from the diagonal, each rounded once
    let mut eigen_pairs: Vec<(FixedPoint, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        eigen_pairs.push((FixedPoint::from_raw(downscale_to_storage(s.get(i, i))?), i));
    }

    // Sort descending by absolute value
    eigen_pairs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap_or(std::cmp::Ordering::Equal));

    let vectors_all = narrow_matrix(&v)?;
    let mut values = FixedVector::new(n);
    let mut vectors = FixedMatrix::new(n, n);
    for (k, (val, orig_idx)) in eigen_pairs.iter().enumerate() {
        values[k] = *val;
        for r in 0..n {
            vectors.set(r, k, vectors_all.get(r, *orig_idx));
        }
    }

    Ok(EigenDecomposition { values, vectors })
}

/// Largest |s[p][q]| over p < q, with its position (the first on ties).
fn largest_off_diagonal(s: &ComputeMatrix) -> (ComputeStorage, usize, usize) {
    let n = s.rows();
    let (mut largest, mut at_p, mut at_q) = (make_compute_int(0), 0, 1);
    for p in 0..n {
        for q in (p + 1)..n {
            let value = compute_abs(s.get(p, q));
            if value > largest {
                largest = value;
                at_p = p;
                at_q = q;
            }
        }
    }
    (largest, at_p, at_q)
}

fn off_diagonal_within_stagnation_bound(s: &ComputeMatrix) -> bool {
    let n = s.rows();
    (0..n).all(|p| {
        ((p + 1)..n).all(|q| {
            compute_abs(s.get(p, q)) <= compute_stagnation_threshold(compute_abs(s.get(p, p)).max(compute_abs(s.get(q, q))))
        })
    })
}

/// Zero `s[p][q]` (and `s[q][p]`) by a Jacobi rotation, accumulating it into `v`.
///
/// With `τ = (a_pp - a_qq) / (2 a_pq)`, `t = sign(τ) / (|τ| + sqrt(1 + τ²))`,
/// `cs = 1 / sqrt(1 + t²)`, `sn = t cs`, all at the compute tier. The
/// off-diagonal rows rotate by `(cs, sn)`; the diagonal moves by `± t a_pq`.
/// Every entry is a compute raw, each updated entry rounded once at the
/// compute tier from its exact value.
fn jacobi_rotate(s: &mut ComputeMatrix, v: &mut ComputeMatrix, p: usize, q: usize) -> Result<(), OverflowDetected> {
    let n = s.rows();
    let (a_pp, a_qq, a_pq) = (s.get(p, p), s.get(q, q), s.get(p, q));
    if compute_is_zero(&a_pq) {
        return Ok(());
    }
    let one = make_compute_int(1);
    let num = compute_checked_add(a_pp, compute_negate(a_qq))?;
    let den = compute_checked_add(a_pq, a_pq)?;
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
        let (new_rp, new_rq) = rot.apply_compute(s.get(r, p), s.get(r, q))?;
        s.set(r, p, new_rp);
        s.set(p, r, new_rp);
        s.set(r, q, new_rq);
        s.set(q, r, new_rq);
    }

    let shift = compute_product(t, a_pq)?;
    s.set(p, p, compute_checked_add(a_pp, shift)?);
    s.set(q, q, compute_checked_add(a_qq, compute_negate(shift))?);
    s.set(p, q, make_compute_int(0));
    s.set(q, p, make_compute_int(0));

    for r in 0..v.rows() {
        let (new_a, new_b) = rot.apply_compute(v.get(r, p), v.get(r, q))?;
        v.set(r, p, new_a);
        v.set(r, q, new_b);
    }
    Ok(())
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
/// **Precision:** B, U and V are carried at the compute tier through the
/// bidiagonalization and the whole iteration (Householder factors, rotation
/// coefficients and the Wilkinson shift too), each transformed entry rounded
/// once at the compute tier from its exact value, and rounded to storage once
/// at the end. Measured against mpmath on well-separated spectra: singular
/// values and vectors within one unit on every profile and split (0.6.3
/// carried them at storage and converged to `2^-(2F/3)` relative).
///
/// **Convergence:** a superdiagonal entry is negligible within `2^-(3F/2)`
/// of its diagonal neighbours (relative), floored at `2^-(3F/2)` absolute,
/// and a diagonal entry at or below that floor is set to zero and deflated.
/// An exact zero singular value is computed as a block of rounding noise that
/// a purely relative test never passes. A block in which no diagonal or
/// superdiagonal entry has reached a new smallest magnitude for five
/// iterations has reached the precision floor: its entry with the smallest
/// backward error is deflated if it lies within one storage unit relative.
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
    // B, U and V stay at the compute tier through the bidiagonalization and
    // the QR iteration, and are rounded to storage once at the end.
    let mut b = ComputeMatrix::from_fixed_matrix(a);
    let mut u_acc = ComputeMatrix::identity(m);
    let mut v_acc = ComputeMatrix::identity(n);

    for j in 0..n {
        // ── Left Householder: zero out B[j+1..m, j] ──
        let column: Vec<ComputeStorage> = (j..m).map(|i| b.get(i, j)).collect();
        if let Some((v_hh, vtv)) = householder_vector_compute(&column)? {
            for c in j..n {
                reflect_compute_column(&mut b, c, j, &v_hh, vtv)?;
            }
            for r in 0..m {
                reflect_compute_row(&mut u_acc, r, j, &v_hh, vtv)?;
            }
            // the reflected column is (alpha, 0, ..., 0) exactly
            for i in (j + 1)..m {
                b.set(i, j, make_compute_int(0));
            }
        }

        // ── Right Householder: zero out B[j, j+2..n] ──
        if j + 1 < n {
            let row: Vec<ComputeStorage> = (j + 1..n).map(|c| b.get(j, c)).collect();
            if let Some((v_hh, vtv)) = householder_vector_compute(&row)? {
                for r in j..m {
                    reflect_compute_row(&mut b, r, j + 1, &v_hh, vtv)?;
                }
                for r in 0..n {
                    reflect_compute_row(&mut v_acc, r, j + 1, &v_hh, vtv)?;
                }
                for c in (j + 2)..n {
                    b.set(j, c, make_compute_int(0));
                }
            }
        }
    }

    // ── Phase 2: Golub-Kahan Implicit QR Iteration ──
    // Bidiagonal elements: diagonal d[0..n], superdiagonal e[0..n-1], carried at
    // the compute tier for the whole iteration and narrowed once at the end. A
    // chase rounded to storage loses a bulge of less than one quantum, and with
    // it the shift: on entries of a few hundred quanta the step then repeats
    // itself exactly and the block never converges.
    let mut d: Vec<ComputeStorage> = (0..n).map(|i| b.get(i, i)).collect();
    let mut e: Vec<ComputeStorage> = (0..n.saturating_sub(1)).map(|i| b.get(i, i + 1)).collect();
    let zero = make_compute_int(0);
    // Power-of-two exponent by which each d[i] (and e[i]) has been scaled up.
    // An active block whose entries are all below 1/2 is scaled up (exactly)
    // until its largest lies in [1/2, 1): singular values scale with the
    // block and the rotations do not depend on the scale, while the shift and
    // the chase keep their relative precision only at that size (at F = 10 a
    // block near 2^-8 left the shift a few significant bits and the iteration
    // froze). Undone in the final rounding.
    let mut exponent = vec![0u32; n];

    let floor = compute_noise_floor();
    let max_iter = iterations_per_n_squared * n * n;
    let mut iter_count = 0usize;
    let mut q_end = n; // exclusive end of the unconverged part

    // Stagnation state of the active block (p, q): the smallest magnitude each
    // superdiagonal and diagonal entry has reached, and the iterations since any
    // entry last reached a new one
    let mut stall_block = (usize::MAX, usize::MAX);
    let mut stall_best: Vec<ComputeStorage> = Vec::new();
    let mut stall_count = 0usize;

    loop {
        // Peel converged superdiagonal entries off the bottom
        while q_end > 1
            && compute_abs(e[q_end - 2]) <= compute_deflation_threshold(compute_abs(d[q_end - 1]).max(compute_abs(d[q_end - 2])))
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
        while p > 0 && compute_abs(e[p - 1]) > compute_deflation_threshold(compute_abs(d[p]).max(compute_abs(d[p - 1]))) {
            p -= 1;
        }

        let k = scale_up_exponent(&d[p..=q].iter().chain(&e[p..q]).copied().collect::<Vec<_>>());
        if k > 0 {
            for i in p..=q {
                d[i] = compute_scale_up(d[i], k);
                exponent[i] += k;
            }
            for i in p..q {
                e[i] = compute_scale_up(e[i], k);
            }
            if stall_block == (p, q) {
                for best in stall_best.iter_mut() {
                    *best = compute_scale_up(*best, k);
                }
            }
        }

        // ── Stagnation fallback ──
        // A block in which no entry has reached a new smallest magnitude for
        // STAGNATION_SWEEPS iterations sits at the precision floor. The largest
        // entry alone is no such evidence: it often holds while the bottom of
        // the block converges.
        let magnitudes: Vec<ComputeStorage> =
            (p..q).map(|k| compute_abs(e[k])).chain((p..=q).map(|k| compute_abs(d[k]))).collect();
        let mut forced_zero: Option<usize> = None;
        if stall_block != (p, q) {
            stall_block = (p, q);
            stall_best = magnitudes;
            stall_count = 0;
        } else {
            let mut improved = false;
            for (best, magnitude) in stall_best.iter_mut().zip(magnitudes) {
                if magnitude < *best {
                    *best = magnitude;
                    improved = true;
                }
            }
            stall_count = if improved { 0 } else { stall_count + 1 };
        }
        if stall_count >= STAGNATION_SWEEPS {
            stall_count = 0;
            let ie = (p..q).min_by_key(|&k| compute_abs(e[k])).expect("active block has a superdiagonal entry");
            let id = (p..=q).min_by_key(|&k| compute_abs(d[k])).expect("active block has a diagonal entry");
            let e_ok = compute_abs(e[ie]) <= compute_stagnation_threshold(compute_abs(d[ie]).max(compute_abs(d[ie + 1])));
            let mut neighbour = zero;
            if id > 0 {
                neighbour = neighbour.max(compute_abs(d[id - 1])).max(compute_abs(e[id - 1]));
            }
            if id < q {
                neighbour = neighbour.max(compute_abs(e[id]));
            }
            let d_ok = compute_abs(d[id]) <= compute_stagnation_threshold(neighbour);
            if e_ok && (!d_ok || compute_abs(e[ie]) <= compute_abs(d[id])) {
                e[ie] = zero;
                iter_count += 1;
                continue;
            }
            if d_ok {
                forced_zero = Some(id);
            }
        }

        // ── Zero diagonal at the bottom of the block ──
        // Chase e[q-1] upward with column rotations (columns j and q), which V takes.
        if compute_abs(d[q]) <= floor || forced_zero == Some(q) {
            d[q] = zero;
            let mut bulge = e[q - 1];
            e[q - 1] = zero;
            for j in (p..q).rev() {
                let rot = Rotation::zeroing_compute(d[j], bulge)?;
                d[j] = rot.combine_compute(d[j], bulge)?;
                if j > p {
                    bulge = rot.neg_sin_times_compute(e[j - 1])?;
                    e[j - 1] = rot.cos_times_compute(e[j - 1])?;
                }
                rotate_compute_columns(&mut v_acc, j, q, &rot)?;
            }
            iter_count += 1;
            continue;
        }

        // ── Zero diagonal inside the block ──
        // Chase e[i] downward with row rotations (rows j and i): the rows move by
        // G, so U takes Gᵀ on columns (j, i).
        if let Some(i) = (p..q).find(|&i| compute_abs(d[i]) <= floor || forced_zero == Some(i)) {
            d[i] = zero;
            let mut bulge = e[i];
            e[i] = zero;
            for j in (i + 1)..=q {
                let rot = Rotation::zeroing_compute(d[j], bulge)?;
                d[j] = rot.combine_compute(d[j], bulge)?;
                if j < q {
                    bulge = rot.neg_sin_times_compute(e[j])?;
                    e[j] = rot.cos_times_compute(e[j])?;
                }
                rotate_compute_columns(&mut u_acc, j, i, &rot)?;
            }
            iter_count += 1;
            continue;
        }

        // ── Implicit QR step (Golub-Kahan), Wilkinson shift ──
        let shift = wilkinson_shift(d[q - 1], e[q - 1], d[q], if q >= 2 { Some(e[q - 2]) } else { None })?;
        let mut x = compute_checked_add(compute_product(d[p], d[p])?, compute_negate(shift))?;
        let mut z = compute_product(d[p], e[p])?;

        for i in p..q {
            // Right rotation on columns i, i+1
            let rot = Rotation::zeroing_compute(x, z)?;
            if i > p {
                e[i - 1] = rot.combine_compute(e[i - 1], z)?;
            }
            (d[i], e[i]) = rot.apply_compute(d[i], e[i])?;
            let bulge = rot.sin_times_compute(d[i + 1])?;
            d[i + 1] = rot.cos_times_compute(d[i + 1])?;
            rotate_compute_columns(&mut v_acc, i, i + 1, &rot)?;

            // Left rotation on rows i, i+1
            let rot2 = Rotation::zeroing_compute(d[i], bulge)?;
            d[i] = rot2.combine_compute(d[i], bulge)?;
            (e[i], d[i + 1]) = rot2.apply_compute(e[i], d[i + 1])?;
            rotate_compute_columns(&mut u_acc, i, i + 1, &rot2)?;

            // Set up for next iteration of the chase
            if i + 1 < q {
                x = e[i];
                z = rot2.sin_times_compute(e[i + 1])?;
                e[i + 1] = rot2.cos_times_compute(e[i + 1])?;
            }
        }

        iter_count += 1;
    }

    // ── Phase 3: Make singular values non-negative and sort descending ──
    let mut values: Vec<FixedPoint> = Vec::with_capacity(n);
    for i in 0..n {
        values.push(FixedPoint::from_raw(downscale_shifted_to_storage(compute_abs(d[i]), exponent[i])?));
        if compute_is_negative(&d[i]) {
            // Flip sign of corresponding V column (row of Vᵀ)
            for r in 0..n {
                v_acc.set(r, i, compute_negate(v_acc.get(r, i)));
            }
        }
    }
    let (u_acc, v_acc) = (narrow_matrix(&u_acc)?, narrow_matrix(&v_acc)?);

    // Sort by descending singular value
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| values[b].cmp(&values[a]));

    let mut sigma = FixedVector::new(n);
    let mut u_sorted = FixedMatrix::new(m, m);
    let mut vt_sorted = FixedMatrix::new(n, n);

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sigma[new_idx] = values[old_idx];
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
/// The entries are compute raws; each product is rounded once at the compute
/// tier. The discriminant is formed in ratio form, without squaring the block.
fn wilkinson_shift(
    d_prev: ComputeStorage, e_last: ComputeStorage, d_last: ComputeStorage, e_prev: Option<ComputeStorage>,
) -> Result<ComputeStorage, OverflowDetected> {
    let e_prev_sq = match e_prev {
        Some(ep) => compute_product(ep, ep)?,
        None => make_compute_int(0),
    };
    let f = compute_checked_add(compute_product(d_prev, d_prev)?, e_prev_sq)?;
    let g = compute_checked_add(compute_product(d_last, d_last)?, compute_product(e_last, e_last)?)?;
    let h = compute_product(d_prev, e_last)?;
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
///    entries set to zero; exceptional shifts every 10 iterations without
///    deflation break the cycles an ordinary shift can sit in
/// 3. Deflate when a subdiagonal entry is within `2^-(3F/2)` of its diagonal
///    neighbours (relative), floored at `2^-(3F/2)` absolute (set to zero);
///    split converged 2×2 blocks with real eigenvalues
///
/// **Precision:** H and Q are carried at the compute tier through the
/// Hessenberg reduction and the whole iteration (Householder factors, rotation
/// coefficients and shifts too), each transformed entry rounded once at the
/// compute tier from its exact value, and rounded to storage once at the end.
/// Measured against exact eigenvalues: within one unit on every profile and
/// split (0.6.3 carried H at storage and converged to `2^-(2F/3)` relative).
/// A block that has not deflated after 30 iterations and in which no
/// subdiagonal entry has reached a new minimum for five iterations has
/// reached the precision floor: its smallest subdiagonal entry is deflated if
/// it lies within one storage unit relative.
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
    // Reduce A to upper Hessenberg form H via Householder: Qᵀ A Q = H. H and
    // Q stay at the compute tier through the reduction and the Francis
    // iteration, and are rounded to storage once at the end.
    let mut h = ComputeMatrix::from_fixed_matrix(a);
    let mut q_acc = ComputeMatrix::identity(n);
    let zero = make_compute_int(0);

    for k in 0..n.saturating_sub(2) {
        let start = k + 1;
        let column: Vec<ComputeStorage> = (start..n).map(|i| h.get(i, k)).collect();
        let (v_hh, vtv) = match householder_vector_compute(&column)? {
            Some(reflector) => reflector,
            None => continue,
        };
        // Left: H[start..n, :] ; columns before k are already zero in these rows
        for c in k..n {
            reflect_compute_column(&mut h, c, start, &v_hh, vtv)?;
        }
        // Right: H[:, start..n]
        for r in 0..n {
            reflect_compute_row(&mut h, r, start, &v_hh, vtv)?;
        }
        // Accumulate into Q: Q[:, start..n]
        for r in 0..n {
            reflect_compute_row(&mut q_acc, r, start, &v_hh, vtv)?;
        }
        for i in (start + 1)..n {
            h.set(i, k, zero);
        }
    }

    // ── Phase 2: Francis Implicit Double-Shift QR Iteration ──
    let max_iter = iterations_per_n_squared * n * n;
    let mut iter_count = 0usize;
    let mut its = 0usize; // iterations since the last deflation at the bottom
    let mut nn = n; // h[0..nn, 0..nn] holds the unconverged part

    // Stagnation state of the active block (l, nn): the smallest magnitude
    // each subdiagonal entry has reached, and the iterations since any entry
    // last reached a new one
    let mut stall_block = (usize::MAX, usize::MAX);
    let mut stall_best: Vec<ComputeStorage> = Vec::new();
    let mut stall_count = 0usize;

    while nn > 0 {
        // Start l of the bottom unreduced block; a negligible subdiagonal entry
        // is set to zero (its backward error is the entry itself)
        let mut l = nn - 1;
        while l > 0 {
            let bound = compute_deflation_threshold(compute_abs(h.get(l, l)).max(compute_abs(h.get(l - 1, l - 1))));
            if compute_abs(h.get(l, l - 1)) <= bound {
                h.set(l, l - 1, zero);
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

        // Precision floor: once the block has run its exceptional shifts and no
        // subdiagonal entry has reached a new minimum for five iterations,
        // deflate the smallest entry if it is within the loose bound. A block
        // that is still converging, however slowly, keeps iterating.
        let subdiagonal: Vec<ComputeStorage> = ((l + 1)..nn).map(|i| compute_abs(h.get(i, i - 1))).collect();
        if stall_block != (l, nn) {
            stall_block = (l, nn);
            stall_best = subdiagonal.clone();
            stall_count = 0;
        } else {
            let mut improved = false;
            for (best, entry) in stall_best.iter_mut().zip(&subdiagonal) {
                if *entry < *best {
                    *best = *entry;
                    improved = true;
                }
            }
            stall_count = if improved { 0 } else { stall_count + 1 };
        }
        if its >= SCHUR_FLOOR_ITERATIONS && stall_count >= STAGNATION_SWEEPS {
            let i = ((l + 1)..nn).min_by_key(|&i| compute_abs(h.get(i, i - 1))).expect("block of size >= 3");
            if compute_abs(h.get(i, i - 1))
                <= compute_stagnation_threshold(compute_abs(h.get(i, i)).max(compute_abs(h.get(i - 1, i - 1))))
            {
                h.set(i, i - 1, zero);
                its = 0;
                stall_count = 0;
                iter_count += 1;
                continue;
            }
        }

        // The shifts and the first column of the step are homogeneous in H, so
        // they are formed from the active block scaled up by a power of two
        // (exact) until its largest entry lies in [1/2, 1): a small block
        // keeps its relative precision there, and the step's direction is the
        // same.
        let block: Vec<ComputeStorage> = (l..nn).flat_map(|i| (l..nn).map(move |j| (i, j))).map(|(i, j)| h.get(i, j)).collect();
        let k = scale_up_exponent(&block);
        let (trace, det) = if its > 0 && its % 10 == 0 {
            exceptional_shifts(&h, l, nn, its, k)?
        } else {
            trailing_shifts(&h, nn, k)?
        };
        francis_step(&mut h, &mut q_acc, l, nn, trace, det, k)?;
        its += 1;
        iter_count += 1;
    }

    Ok(SchurDecomposition { q: narrow_matrix(&q_acc)?, t: narrow_matrix(&h)? })
}

/// Trace and determinant of the trailing 2×2 block (the double-shift pair),
/// as compute raws (the determinant from exact products, rounded once), of H
/// scaled by `2^k`.
fn trailing_shifts(h: &ComputeMatrix, nn: usize, k: u32) -> Result<(ComputeStorage, ComputeStorage), OverflowDetected> {
    let hs = |i: usize, j: usize| compute_scale_up(h.get(i, j), k);
    let (a, b) = (hs(nn - 2, nn - 2), hs(nn - 2, nn - 1));
    let (c, d) = (hs(nn - 1, nn - 2), hs(nn - 1, nn - 1));
    let trace = compute_checked_add(a, d)?;
    let det = exact_sub_dot_compute(make_compute_int(0), &[b, compute_negate(a)], &[c, d])?;
    Ok((trace, det))
}

/// Exceptional shift pair every 10 iterations without deflation (LAPACK
/// `dlahqr`): the 2×2 block `[[w, -7/16 s], [s, w]]` with `w = h + 3/4 s`,
/// where `s` sums two subdiagonal magnitudes, at the top of the block after
/// 10, 30, 50, ... iterations and at the bottom after 20, 40, .... It breaks
/// the cycles an ordinary Francis step sits in, such as permutation matrices;
/// with shifts at 10 and 20 only, a cycle entered later persisted. Of H
/// scaled by `2^k`.
fn exceptional_shifts(h: &ComputeMatrix, l: usize, nn: usize, its: usize, k: u32) -> Result<(ComputeStorage, ComputeStorage), OverflowDetected> {
    let hs = |i: usize, j: usize| compute_scale_up(h.get(i, j), k);
    let (s_c, anchor) = if its % 20 == 10 {
        (compute_checked_add(compute_abs(hs(l + 1, l)), compute_abs(hs(l + 2, l + 1)))?, hs(l, l))
    } else {
        (compute_checked_add(compute_abs(hs(nn - 1, nn - 2)), compute_abs(hs(nn - 2, nn - 3)))?, hs(nn - 1, nn - 1))
    };
    let three_quarters = compute_divide(make_compute_int(3), make_compute_int(4))?;
    let seven_sixteenths = compute_divide(make_compute_int(7), make_compute_int(16))?;
    let w = compute_checked_add(anchor, compute_multiply(three_quarters, s_c))?;
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
/// `trace` and `det` are of H scaled by `2^k`; the first column is formed at
/// that scale (its direction does not depend on it).
fn francis_step(
    h: &mut ComputeMatrix, q_acc: &mut ComputeMatrix, l: usize, nn: usize,
    trace: ComputeStorage, det: ComputeStorage, k: u32,
) -> Result<(), OverflowDetected> {
    let n = h.rows();
    let zero = make_compute_int(0);

    // First column of (H - s1 I)(H - s2 I) = H² - trace H + det I
    let hs = |i: usize, j: usize| compute_scale_up(h.get(i, j), k);
    let (h11, h12, h21) = (hs(l, l), hs(l, l + 1), hs(l + 1, l));
    let (h22, h32) = (hs(l + 1, l + 1), hs(l + 2, l + 1));
    let x = exact_sub_dot_compute(det, &[compute_negate(h11), compute_negate(h12), trace], &[h11, h21, h11])?;
    let y = compute_product(h21, compute_sum(&[h11, h22, compute_negate(trace)])?)?;
    let z = compute_product(h21, h32)?;

    // Reflector introducing the bulge at rows l..l+3
    if let Some((v_hh, vtv)) = householder_vector_compute(&[x, y, z])? {
        for c in l..n {
            reflect_compute_column(h, c, l, &v_hh, vtv)?;
        }
        for r in 0..nn.min(l + 4) {
            reflect_compute_row(h, r, l, &v_hh, vtv)?;
        }
        for r in 0..n {
            reflect_compute_row(q_acc, r, l, &v_hh, vtv)?;
        }
    }

    // ── Bulge chase down to the last row of the block ──
    for k in (l + 1)..(nn - 1) {
        if k + 2 < nn {
            // 3-element reflector on rows k..k+3 zeroes h[k+1, k-1], h[k+2, k-1]
            let column: Vec<ComputeStorage> = (k..k + 3).map(|i| h.get(i, k - 1)).collect();
            if let Some((v_hh, vtv)) = householder_vector_compute(&column)? {
                for c in (k - 1)..n {
                    reflect_compute_column(h, c, k, &v_hh, vtv)?;
                }
                for r in 0..nn.min(k + 4) {
                    reflect_compute_row(h, r, k, &v_hh, vtv)?;
                }
                for r in 0..n {
                    reflect_compute_row(q_acc, r, k, &v_hh, vtv)?;
                }
            }
            h.set(k + 1, k - 1, zero);
            h.set(k + 2, k - 1, zero);
        } else {
            // Last step: a rotation on rows k, k+1 zeroes h[k+1, k-1]
            let rot = Rotation::zeroing_compute(h.get(k, k - 1), h.get(k + 1, k - 1))?;
            for c in (k - 1)..n {
                let (top, bottom) = rot.apply_compute(h.get(k, c), h.get(k + 1, c))?;
                h.set(k, c, top);
                h.set(k + 1, c, bottom);
            }
            for r in 0..nn {
                let (left, right) = rot.apply_compute(h.get(r, k), h.get(r, k + 1))?;
                h.set(r, k, left);
                h.set(r, k + 1, right);
            }
            rotate_compute_columns(q_acc, k, k + 1, &rot)?;
            h.set(k + 1, k - 1, zero);
        }
    }

    Ok(())
}

/// Split the converged 2×2 block at rows and columns (i, i+1) into two 1×1
/// blocks when its eigenvalues are real: rotate by the eigenvector
/// `(λ - d, c)` of `λ = (a+d)/2 + sign(p) sqrt(p² + bc)`, `p = (a-d)/2`
/// (no cancellation in `λ - d = p + sign(p) sqrt(..)`). A complex pair keeps
/// its block.
fn split_real_block(h: &mut ComputeMatrix, q_acc: &mut ComputeMatrix, i: usize) -> Result<(), OverflowDetected> {
    let n = h.rows();
    if compute_is_zero(&h.get(i + 1, i)) {
        return Ok(());
    }
    // the rotation's direction is homogeneous in the block: form it from the
    // block scaled up to [1/2, 1) (exact), where a small block keeps its
    // relative precision
    let k = scale_up_exponent(&[h.get(i, i), h.get(i, i + 1), h.get(i + 1, i), h.get(i + 1, i + 1)]);
    let hs = |r: usize, c: usize| compute_scale_up(h.get(r, c), k);
    let (a, b) = (hs(i, i), hs(i, i + 1));
    let (c, d) = (hs(i + 1, i), hs(i + 1, i + 1));
    let p = compute_halve(compute_checked_add(a, compute_negate(d))?);
    let disc = exact_sub_dot_compute(make_compute_int(0), &[compute_negate(p), compute_negate(b)], &[p, c])?;
    if compute_is_negative(&disc) {
        return Ok(());
    }
    let root = sqrt_at_compute_tier(disc);
    let lead = if compute_is_negative(&p) {
        compute_checked_add(p, compute_negate(root))?
    } else {
        compute_checked_add(p, root)?
    };
    let rot = Rotation::zeroing_compute(lead, c)?;
    // Gᵀ H on rows i, i+1 (entries left of column i are zero in both rows)
    for col in i..n {
        let (top, bottom) = rot.apply_compute(h.get(i, col), h.get(i + 1, col))?;
        h.set(i, col, top);
        h.set(i + 1, col, bottom);
    }
    // H G on columns i, i+1 (entries below row i+1 are zero in both columns)
    for row in 0..(i + 2) {
        let (left, right) = rot.apply_compute(h.get(row, i), h.get(row, i + 1))?;
        h.set(row, i, left);
        h.set(row, i + 1, right);
    }
    rotate_compute_columns(q_acc, i, i + 1, &rot)?;
    h.set(i + 1, i, make_compute_int(0));
    Ok(())
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
