//! Decomposition validation against mpmath 50-digit reference values.
//!
//! All reference values computed by mpmath 1.3.0 at 50 decimal digits of
//! precision. These are the mathematical ground truth for solver correctness,
//! determinant accuracy, inverse accuracy, and Cholesky factor values.
//!
//! This complements decompose_validation.rs (algebraic identity tests) with
//! concrete numerical verification against an independent high-precision source.

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::decompose::{
    lu_decompose, qr_decompose, cholesky_decompose,
};

/// Parse a decimal string to FixedPoint, handling negative values correctly.
///
/// FixedPoint::from_str routes through gmath_parse which may not handle
/// leading '-' for runtime strings. This helper negates explicitly.
fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') {
        -FixedPoint::from_str(&s[1..])
    } else {
        FixedPoint::from_str(s)
    }
}

/// Tolerance: 1e-12 (conservative for ill-conditioned systems like Wilson)
fn tol() -> FixedPoint { fp("0.000000000001") }

fn assert_fp_eq(got: FixedPoint, expected: FixedPoint, name: &str) {
    let diff = (got - expected).abs();
    assert!(diff < tol(),
        "mpmath validation failed for {}: got {}, expected {}, diff={}",
        name, got, expected, diff);
}

fn assert_vec_eq(got: &FixedVector, expected: &[&str], name: &str) {
    assert_eq!(got.len(), expected.len(), "{}: dimension mismatch", name);
    for i in 0..got.len() {
        assert_fp_eq(got[i], fp(expected[i]),
            &format!("{}[{}]", name, i));
    }
}

// ============================================================================
// mpmath reference: System 1 (3x3 SPD)
// A=[[3,1,2],[1,4,1],[2,1,5]], b=[1,2,3]
// x = [-0.2, 0.4, 0.6], det(A) = 40
// ============================================================================

fn system1_a() -> FixedMatrix {
    FixedMatrix::from_slice(3, 3, &[
        fp("3"), fp("1"), fp("2"),
        fp("1"), fp("4"), fp("1"),
        fp("2"), fp("1"), fp("5"),
    ])
}
fn system1_b() -> FixedVector {
    FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")])
}

#[test]
fn test_mpmath_lu_solve_system1() {
    let a = system1_a();
    let b = system1_b();
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    assert_vec_eq(&x, &["-0.2", "0.4", "0.6"], "LU solve system1");
}

#[test]
fn test_mpmath_qr_solve_system1() {
    let a = system1_a();
    let b = system1_b();
    let qr = qr_decompose(&a).unwrap();
    let x = qr.solve(&b).unwrap();
    assert_vec_eq(&x, &["-0.2", "0.4", "0.6"], "QR solve system1");
}

#[test]
fn test_mpmath_cholesky_solve_system1() {
    // System 1 is SPD: all eigenvalues positive
    let a = system1_a();
    let b = system1_b();
    let chol = cholesky_decompose(&a).unwrap();
    let x = chol.solve(&b).unwrap();
    assert_vec_eq(&x, &["-0.2", "0.4", "0.6"], "Cholesky solve system1");
}

#[test]
fn test_mpmath_det_system1() {
    let a = system1_a();
    let lu = lu_decompose(&a).unwrap();
    assert_fp_eq(lu.determinant(), fp("40"), "det(system1)");
}

// ============================================================================
// mpmath reference: System 2 (2x2)
// A=[[2,1],[5,3]], b=[4,7]
// x = [5, -6], det(A) = 1
// ============================================================================

#[test]
fn test_mpmath_lu_solve_system2() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("2"), fp("1"), fp("5"), fp("3")]);
    let b = FixedVector::from_slice(&[fp("4"), fp("7")]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    assert_vec_eq(&x, &["5", "-6"], "LU solve system2");
}

#[test]
fn test_mpmath_det_system2() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("2"), fp("1"), fp("5"), fp("3")]);
    let lu = lu_decompose(&a).unwrap();
    assert_fp_eq(lu.determinant(), fp("1"), "det(system2)");
}

// ============================================================================
// mpmath reference: System 3 (3x3 inverse)
// A=[[1,2,3],[0,1,4],[5,6,0]]
// A^-1 = [[-24,18,5],[20,-15,-4],[-5,4,1]]
// det(A) = 1
// ============================================================================

#[test]
fn test_mpmath_inverse_system3() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("0"), fp("1"), fp("4"),
        fp("5"), fp("6"), fp("0"),
    ]);
    let lu = lu_decompose(&a).unwrap();
    let inv = lu.inverse().unwrap();

    // mpmath reference: A^-1
    assert_fp_eq(inv.get(0, 0), fp("-24"), "inv[0,0]");
    assert_fp_eq(inv.get(0, 1), fp("18"), "inv[0,1]");
    assert_fp_eq(inv.get(0, 2), fp("5"), "inv[0,2]");
    assert_fp_eq(inv.get(1, 0), fp("20"), "inv[1,0]");
    assert_fp_eq(inv.get(1, 1), fp("-15"), "inv[1,1]");
    assert_fp_eq(inv.get(1, 2), fp("-4"), "inv[1,2]");
    assert_fp_eq(inv.get(2, 0), fp("-5"), "inv[2,0]");
    assert_fp_eq(inv.get(2, 1), fp("4"), "inv[2,1]");
    assert_fp_eq(inv.get(2, 2), fp("1"), "inv[2,2]");
}

#[test]
fn test_mpmath_det_system3() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("0"), fp("1"), fp("4"),
        fp("5"), fp("6"), fp("0"),
    ]);
    let lu = lu_decompose(&a).unwrap();
    assert_fp_eq(lu.determinant(), fp("1"), "det(system3)");
}

// ============================================================================
// mpmath reference: Cholesky factors
// A=[[4,12,-16],[12,37,-43],[-16,-43,98]]
// L = [[2,0,0],[6,1,0],[-8,5,3]]
// ============================================================================

#[test]
fn test_mpmath_cholesky_factors() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("12"), fp("-16"),
        fp("12"), fp("37"), fp("-43"),
        fp("-16"), fp("-43"), fp("98"),
    ]);
    let chol = cholesky_decompose(&a).unwrap();

    // mpmath reference: exact L values
    assert_fp_eq(chol.l.get(0, 0), fp("2"), "L[0,0]");
    assert_fp_eq(chol.l.get(1, 0), fp("6"), "L[1,0]");
    assert_fp_eq(chol.l.get(1, 1), fp("1"), "L[1,1]");
    assert_fp_eq(chol.l.get(2, 0), fp("-8"), "L[2,0]");
    assert_fp_eq(chol.l.get(2, 1), fp("5"), "L[2,1]");
    assert_fp_eq(chol.l.get(2, 2), fp("3"), "L[2,2]");

    // Off-diagonal zeros
    assert_fp_eq(chol.l.get(0, 1), fp("0"), "L[0,1]");
    assert_fp_eq(chol.l.get(0, 2), fp("0"), "L[0,2]");
    assert_fp_eq(chol.l.get(1, 2), fp("0"), "L[1,2]");
}

// ============================================================================
// mpmath reference: Cholesky solve
// A=[[4,2],[2,3]], b=[1,2]
// x = [-0.125, 0.75]
// ============================================================================

#[test]
fn test_mpmath_cholesky_solve() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("4"), fp("2"), fp("2"), fp("3")]);
    let b = FixedVector::from_slice(&[fp("1"), fp("2")]);
    let chol = cholesky_decompose(&a).unwrap();
    let x = chol.solve(&b).unwrap();
    assert_vec_eq(&x, &["-0.125", "0.75"], "Cholesky solve");
}

// ============================================================================
// mpmath reference: det([[1,2],[3,4]]) = -2
// ============================================================================

#[test]
fn test_mpmath_det_2x2() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let lu = lu_decompose(&a).unwrap();
    assert_fp_eq(lu.determinant(), fp("-2"), "det([[1,2],[3,4]])");
}

// ============================================================================
// mpmath reference: Wilson matrix (4x4 near-singular, ill-conditioned)
// A = [[10,7,8,7],[7,5,6,5],[8,6,10,9],[7,5,9,10]]
// b = [32,23,33,31] → x = [1,1,1,1] (exact)
// det(A) = 1
// ============================================================================

#[test]
fn test_mpmath_wilson_matrix_solve() {
    let a = FixedMatrix::from_slice(4, 4, &[
        fp("10"), fp("7"), fp("8"), fp("7"),
        fp("7"), fp("5"), fp("6"), fp("5"),
        fp("8"), fp("6"), fp("10"), fp("9"),
        fp("7"), fp("5"), fp("9"), fp("10"),
    ]);
    let b = FixedVector::from_slice(&[fp("32"), fp("23"), fp("33"), fp("31")]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    // mpmath: x = [1, 1, 1, 1] exactly
    assert_vec_eq(&x, &["1", "1", "1", "1"], "Wilson LU solve");
}

#[test]
fn test_mpmath_wilson_matrix_det() {
    let a = FixedMatrix::from_slice(4, 4, &[
        fp("10"), fp("7"), fp("8"), fp("7"),
        fp("7"), fp("5"), fp("6"), fp("5"),
        fp("8"), fp("6"), fp("10"), fp("9"),
        fp("7"), fp("5"), fp("9"), fp("10"),
    ]);
    let lu = lu_decompose(&a).unwrap();
    assert_fp_eq(lu.determinant(), fp("1"), "Wilson det");
}

#[test]
fn test_mpmath_wilson_qr_solve() {
    let a = FixedMatrix::from_slice(4, 4, &[
        fp("10"), fp("7"), fp("8"), fp("7"),
        fp("7"), fp("5"), fp("6"), fp("5"),
        fp("8"), fp("6"), fp("10"), fp("9"),
        fp("7"), fp("5"), fp("9"), fp("10"),
    ]);
    let b = FixedVector::from_slice(&[fp("32"), fp("23"), fp("33"), fp("31")]);
    let qr = qr_decompose(&a).unwrap();
    let x = qr.solve(&b).unwrap();
    assert_vec_eq(&x, &["1", "1", "1", "1"], "Wilson QR solve");
}

// ============================================================================
// mpmath reference: Wilson matrix is SPD → Cholesky should work too
// ============================================================================

#[test]
fn test_mpmath_wilson_cholesky_solve() {
    let a = FixedMatrix::from_slice(4, 4, &[
        fp("10"), fp("7"), fp("8"), fp("7"),
        fp("7"), fp("5"), fp("6"), fp("5"),
        fp("8"), fp("6"), fp("10"), fp("9"),
        fp("7"), fp("5"), fp("9"), fp("10"),
    ]);
    let b = FixedVector::from_slice(&[fp("32"), fp("23"), fp("33"), fp("31")]);
    let chol = cholesky_decompose(&a).unwrap();
    let x = chol.solve(&b).unwrap();
    assert_vec_eq(&x, &["1", "1", "1", "1"], "Wilson Cholesky solve");
}
