//! Validation tests for L1B matrix decompositions (LU, QR, Cholesky).
//!
//! Tests verify:
//! 1. Reconstruction: PA = LU, A = QR, A = LL^T
//! 2. Structural properties: L lower triangular, U upper triangular, Q orthogonal
//! 3. Solver correctness: Ax = b round-trip
//! 4. Known-value tests against hand-computed results
//! 5. Determinant consistency
//! 6. Error detection: singular matrices, non-SPD inputs

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::OverflowDetected;
use g_math::fixed_point::imperative::decompose::{
    lu_decompose, qr_decompose, cholesky_decompose,
};

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') {
        -FixedPoint::from_str(&s[1..])
    } else {
        FixedPoint::from_str(s)
    }
}

/// Check that two matrices are approximately equal (within tolerance).
fn matrices_approx_eq(a: &FixedMatrix, b: &FixedMatrix, tol: FixedPoint) -> bool {
    if a.rows() != b.rows() || a.cols() != b.cols() { return false; }
    for r in 0..a.rows() {
        for c in 0..a.cols() {
            if (a.get(r, c) - b.get(r, c)).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Check that two vectors are approximately equal.
fn vectors_approx_eq(a: &FixedVector, b: &FixedVector, tol: FixedPoint) -> bool {
    if a.len() != b.len() { return false; }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > tol { return false; }
    }
    true
}

/// Reconstruct PA from LU decomposition.
fn reconstruct_lu(lu: &g_math::fixed_point::imperative::decompose::LUDecomposition) -> FixedMatrix {
    &lu.l * &lu.u
}

/// Build permutation of A from LU perm vector.
fn permute_matrix(a: &FixedMatrix, perm: &[usize]) -> FixedMatrix {
    let n = a.rows();
    FixedMatrix::from_fn(n, a.cols(), |r, c| a.get(perm[r], c))
}

// Tolerance for reconstruction tests
fn tol() -> FixedPoint { fp("0.000000001") }
// Tighter tolerance for exact-integer tests
fn tight_tol() -> FixedPoint { fp("0.0000000000000001") }

// ============================================================================
// LU Decomposition Tests
// ============================================================================

#[test]
fn test_lu_2x2() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("4"), fp("3"), fp("6"), fp("3")]);
    let lu = lu_decompose(&a).unwrap();
    let pa = permute_matrix(&a, &lu.perm);
    let reconstructed = reconstruct_lu(&lu);
    assert!(matrices_approx_eq(&pa, &reconstructed, tol()),
        "PA != LU for 2x2 matrix");
}

#[test]
fn test_lu_3x3() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("2"), fp("1"), fp("1"),
        fp("4"), fp("3"), fp("3"),
        fp("8"), fp("7"), fp("9"),
    ]);
    let lu = lu_decompose(&a).unwrap();
    let pa = permute_matrix(&a, &lu.perm);
    let reconstructed = reconstruct_lu(&lu);
    assert!(matrices_approx_eq(&pa, &reconstructed, tol()),
        "PA != LU for 3x3 matrix");
}

#[test]
fn test_lu_identity() {
    let a = FixedMatrix::identity(4);
    let lu = lu_decompose(&a).unwrap();
    assert!(matrices_approx_eq(&lu.l, &FixedMatrix::identity(4), tight_tol()));
    assert!(matrices_approx_eq(&lu.u, &FixedMatrix::identity(4), tight_tol()));
}

#[test]
fn test_lu_singular_matrix() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("2"), fp("4")]);
    let result = lu_decompose(&a);
    assert!(result.is_err(), "singular matrix should fail");
}

#[test]
fn test_lu_solve() {
    // A = [[2, 1], [5, 3]], b = [4, 7] → x = [5, -6]
    let a = FixedMatrix::from_slice(2, 2, &[fp("2"), fp("1"), fp("5"), fp("3")]);
    let b = FixedVector::from_slice(&[fp("4"), fp("7")]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    assert!(vectors_approx_eq(&x, &FixedVector::from_slice(&[fp("5"), fp("-6")]), tol()),
        "LU solve incorrect: got {:?}", x);
}

#[test]
fn test_lu_solve_roundtrip() {
    // Solve Ax = b, then verify A*x ≈ b
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("10"),
    ]);
    let b = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    let ax = a.mul_vector(&x);
    assert!(vectors_approx_eq(&ax, &b, tol()),
        "A*x != b: Ax={:?}, b={:?}", ax, b);
}

#[test]
fn test_lu_determinant() {
    // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let lu = lu_decompose(&a).unwrap();
    let det = lu.determinant();
    assert!((det - fp("-2")).abs() < tol(), "det should be -2, got {}", det);
}

#[test]
fn test_lu_inverse() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("4"), fp("7"), fp("2"), fp("6")]);
    let lu = lu_decompose(&a).unwrap();
    let inv = lu.inverse().unwrap();
    let product = &a * &inv;
    assert!(matrices_approx_eq(&product, &FixedMatrix::identity(2), tol()),
        "A * A^-1 != I");
}

#[test]
fn test_lu_inverse_3x3() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("0"), fp("1"), fp("4"),
        fp("5"), fp("6"), fp("0"),
    ]);
    let lu = lu_decompose(&a).unwrap();
    let inv = lu.inverse().unwrap();
    let product = &a * &inv;
    assert!(matrices_approx_eq(&product, &FixedMatrix::identity(3), tol()),
        "A * A^-1 != I for 3x3");
}

// ============================================================================
// QR Decomposition Tests
// ============================================================================

#[test]
fn test_qr_2x2() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let qr = qr_decompose(&a).unwrap();
    let reconstructed = &qr.q * &qr.r;
    assert!(matrices_approx_eq(&a, &reconstructed, tol()),
        "A != QR for 2x2");
}

#[test]
fn test_qr_3x3() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("12"), fp("-51"), fp("4"),
        fp("6"), fp("167"), fp("-68"),
        fp("-4"), fp("24"), fp("-41"),
    ]);
    let qr = qr_decompose(&a).unwrap();
    let reconstructed = &qr.q * &qr.r;
    assert!(matrices_approx_eq(&a, &reconstructed, tol()),
        "A != QR for 3x3");
}

#[test]
fn test_qr_orthogonality() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("10"),
    ]);
    let qr = qr_decompose(&a).unwrap();
    // Q^T * Q should be I
    let qtq = &qr.q.transpose() * &qr.q;
    assert!(matrices_approx_eq(&qtq, &FixedMatrix::identity(3), tol()),
        "Q^T Q != I: {:?}", qtq);
}

#[test]
fn test_qr_identity() {
    let a = FixedMatrix::identity(3);
    let qr = qr_decompose(&a).unwrap();
    // QR of I: Q and R may have flipped signs (QR is not sign-unique),
    // but Q*R must reconstruct I exactly and Q must be orthogonal.
    let reconstructed = &qr.q * &qr.r;
    assert!(matrices_approx_eq(&reconstructed, &a, tight_tol()),
        "QR of identity should reconstruct to identity");
    let qtq = &qr.q.transpose() * &qr.q;
    assert!(matrices_approx_eq(&qtq, &FixedMatrix::identity(3), tight_tol()),
        "Q should be orthogonal");
}

#[test]
fn test_qr_solve() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("2"), fp("1"), fp("5"), fp("3")]);
    let b = FixedVector::from_slice(&[fp("4"), fp("7")]);
    let qr = qr_decompose(&a).unwrap();
    let x = qr.solve(&b).unwrap();
    let ax = a.mul_vector(&x);
    assert!(vectors_approx_eq(&ax, &b, tol()),
        "QR solve: Ax != b");
}

#[test]
fn test_qr_rectangular() {
    // 3×2 matrix (overdetermined)
    let a = FixedMatrix::from_slice(3, 2, &[
        fp("1"), fp("2"),
        fp("3"), fp("4"),
        fp("5"), fp("6"),
    ]);
    let qr = qr_decompose(&a).unwrap();
    assert_eq!(qr.q.rows(), 3);
    assert_eq!(qr.q.cols(), 3);
    assert_eq!(qr.r.rows(), 3);
    assert_eq!(qr.r.cols(), 2);
    let reconstructed = &qr.q * &qr.r;
    assert!(matrices_approx_eq(&a, &reconstructed, tol()),
        "A != QR for 3x2");
}

// ============================================================================
// Cholesky Decomposition Tests
// ============================================================================

#[test]
fn test_cholesky_2x2() {
    // A = [[4, 2], [2, 3]] (SPD)
    let a = FixedMatrix::from_slice(2, 2, &[fp("4"), fp("2"), fp("2"), fp("3")]);
    let chol = cholesky_decompose(&a).unwrap();
    let reconstructed = &chol.l * &chol.l.transpose();
    assert!(matrices_approx_eq(&a, &reconstructed, tol()),
        "A != LL^T for 2x2");
}

#[test]
fn test_cholesky_3x3() {
    // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] (SPD, classic example)
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("12"), fp("-16"),
        fp("12"), fp("37"), fp("-43"),
        fp("-16"), fp("-43"), fp("98"),
    ]);
    let chol = cholesky_decompose(&a).unwrap();
    let reconstructed = &chol.l * &chol.l.transpose();
    assert!(matrices_approx_eq(&a, &reconstructed, tol()),
        "A != LL^T for 3x3 SPD");
    // Known L: [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    assert!((chol.l.get(0, 0) - fp("2")).abs() < tol());
    assert!((chol.l.get(1, 0) - fp("6")).abs() < tol());
    assert!((chol.l.get(1, 1) - fp("1")).abs() < tol());
    assert!((chol.l.get(2, 0) - fp("-8")).abs() < tol());
    assert!((chol.l.get(2, 1) - fp("5")).abs() < tol());
    assert!((chol.l.get(2, 2) - fp("3")).abs() < tol());
}

#[test]
fn test_cholesky_identity() {
    let a = FixedMatrix::identity(3);
    let chol = cholesky_decompose(&a).unwrap();
    assert!(matrices_approx_eq(&chol.l, &FixedMatrix::identity(3), tight_tol()),
        "Cholesky of I should be I");
}

#[test]
fn test_cholesky_non_spd_fails() {
    // Not positive definite (has negative eigenvalue)
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("2"), fp("1")]);
    let result = cholesky_decompose(&a);
    assert!(result.is_err(), "non-SPD matrix should fail Cholesky");
}

#[test]
fn test_cholesky_solve() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("4"), fp("2"), fp("2"), fp("3")]);
    let b = FixedVector::from_slice(&[fp("1"), fp("2")]);
    let chol = cholesky_decompose(&a).unwrap();
    let x = chol.solve(&b).unwrap();
    let ax = a.mul_vector(&x);
    assert!(vectors_approx_eq(&ax, &b, tol()),
        "Cholesky solve: Ax != b");
}

#[test]
fn test_cholesky_determinant() {
    // det([[4, 2], [2, 3]]) = 4*3 - 2*2 = 8
    let a = FixedMatrix::from_slice(2, 2, &[fp("4"), fp("2"), fp("2"), fp("3")]);
    let chol = cholesky_decompose(&a).unwrap();
    let det = chol.determinant();
    assert!((det - fp("8")).abs() < tol(), "det should be 8, got {}", det);
}

// ============================================================================
// Cross-decomposition consistency tests
// ============================================================================

#[test]
fn test_lu_and_qr_solve_agree() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("3"), fp("1"), fp("2"),
        fp("1"), fp("4"), fp("1"),
        fp("2"), fp("1"), fp("5"),
    ]);
    let b = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);

    let lu = lu_decompose(&a).unwrap();
    let qr = qr_decompose(&a).unwrap();
    let x_lu = lu.solve(&b).unwrap();
    let x_qr = qr.solve(&b).unwrap();

    assert!(vectors_approx_eq(&x_lu, &x_qr, tol()),
        "LU and QR solvers should agree");
}

#[test]
fn test_all_three_solve_spd() {
    // SPD matrix: all three decompositions should produce the same solution
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("2"), fp("1"),
        fp("2"), fp("5"), fp("3"),
        fp("1"), fp("3"), fp("6"),
    ]);
    let b = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);

    let lu = lu_decompose(&a).unwrap();
    let qr = qr_decompose(&a).unwrap();
    let chol = cholesky_decompose(&a).unwrap();

    let x_lu = lu.solve(&b).unwrap();
    let x_qr = qr.solve(&b).unwrap();
    let x_chol = chol.solve(&b).unwrap();

    assert!(vectors_approx_eq(&x_lu, &x_qr, tol()), "LU != QR");
    assert!(vectors_approx_eq(&x_lu, &x_chol, tol()), "LU != Cholesky");
}

#[test]
fn test_lu_determinant_consistency() {
    // det(A) from LU should match det(A) from Cholesky for SPD matrices
    let a = FixedMatrix::from_slice(2, 2, &[fp("4"), fp("2"), fp("2"), fp("3")]);
    let lu = lu_decompose(&a).unwrap();
    let chol = cholesky_decompose(&a).unwrap();
    let det_lu = lu.determinant();
    let det_chol = chol.determinant();
    assert!((det_lu - det_chol).abs() < tol(),
        "LU det ({}) != Cholesky det ({})", det_lu, det_chol);
}
