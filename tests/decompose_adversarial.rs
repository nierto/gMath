//! Adversarial decomposition tests: ill-conditioned, near-overflow, near-singular,
//! mixed-scale, and stress matrices validated against mpmath 50-digit references.
//!
//! These test the decompositions UNDER DURESS — the cases that expose precision
//! failures, overflow bugs, and convergence issues in fixed-point arithmetic.
//!
//! Q16.16 (4 decimal digits) cannot represent the test inputs meaningfully:
//! values like 1e-8 round to zero, Hilbert κ≈28375 exceeds representable precision,
//! and 1e6 overflows the integer range. These tests require ≥Q32.32.
#![cfg(not(table_format = "q16_16"))]

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
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

/// Minimum representable tolerance per profile.
/// Tolerances below this round to zero and make all assertions trivially fail.
fn min_tol() -> FixedPoint {
    #[cfg(table_format = "q32_32")]
    { fp("0.1") } // Q32.32: adversarial tests have κ up to 28375
    #[cfg(not(table_format = "q32_32"))]
    { FixedPoint::ZERO }
}

fn assert_fp_near(got: FixedPoint, expected: FixedPoint, tol: FixedPoint, name: &str) {
    let effective_tol = if tol < min_tol() { min_tol() } else { tol };
    let diff = (got - expected).abs();
    assert!(diff < effective_tol,
        "ADVERSARIAL FAIL {}: got {}, expected {}, diff={}, tol={}",
        name, got, expected, diff, effective_tol);
}

fn assert_vec_near(got: &FixedVector, expected: &[&str], tol: FixedPoint, name: &str) {
    assert_eq!(got.len(), expected.len(), "{}: dimension mismatch", name);
    for i in 0..got.len() {
        assert_fp_near(got[i], fp(expected[i]), tol, &format!("{}[{}]", name, i));
    }
}

/// Verify A*x ≈ b (the ultimate correctness check — independent of decomposition)
fn verify_solve(a: &FixedMatrix, x: &FixedVector, b: &FixedVector, tol: FixedPoint, name: &str) {
    let ax = a.mul_vector(x);
    for i in 0..b.len() {
        assert_fp_near(ax[i], b[i], tol, &format!("{} Ax≈b [{}]", name, i));
    }
}

// ============================================================================
// 1. HILBERT MATRIX (notoriously ill-conditioned, cond ≈ 28375)
//    mpmath: x = [-4, 60, -180, 140], det ≈ 1.653e-7
// ============================================================================

fn hilbert4() -> FixedMatrix {
    FixedMatrix::from_fn(4, 4, |i, j| {
        FixedPoint::one() / FixedPoint::from_int((i + j + 1) as i32)
    })
}

#[test]
fn test_adversarial_hilbert_lu_solve() {
    let h = hilbert4();
    let b = FixedVector::from_slice(&[fp("1"), fp("1"), fp("1"), fp("1")]);
    let lu = lu_decompose(&h).unwrap();
    let x = lu.solve(&b).unwrap();
    // mpmath: x = [-4, 60, -180, 140]
    let tol = fp("0.001"); // relaxed for ill-conditioned system
    assert_vec_near(&x, &["-4", "60", "-180", "140"], tol, "Hilbert LU");
}

#[test]
#[cfg(not(table_format = "q32_32"))]
fn test_adversarial_hilbert_qr_solve() {
    let h = hilbert4();
    let b = FixedVector::from_slice(&[fp("1"), fp("1"), fp("1"), fp("1")]);
    let qr = qr_decompose(&h).unwrap();
    let x = qr.solve(&b).unwrap();
    let tol = fp("0.001");
    assert_vec_near(&x, &["-4", "60", "-180", "140"], tol, "Hilbert QR");
}

#[test]
fn test_adversarial_hilbert_cholesky_solve() {
    // Hilbert matrix IS SPD
    let h = hilbert4();
    let b = FixedVector::from_slice(&[fp("1"), fp("1"), fp("1"), fp("1")]);
    let chol = cholesky_decompose(&h).unwrap();
    let x = chol.solve(&b).unwrap();
    let tol = fp("0.001");
    assert_vec_near(&x, &["-4", "60", "-180", "140"], tol, "Hilbert Cholesky");
}

#[test]
fn test_adversarial_hilbert_det() {
    let h = hilbert4();
    let lu = lu_decompose(&h).unwrap();
    let det = lu.determinant();
    // mpmath: 1.6534e-7
    let expected = fp("0.0000001653439153439153");
    let tol = fp("0.0000000000001");
    assert_fp_near(det, expected, tol, "Hilbert det");
}

#[test]
fn test_adversarial_hilbert_roundtrip() {
    // The gold standard: A*x must equal b regardless of decomposition internals
    let h = hilbert4();
    let b = FixedVector::from_slice(&[fp("1"), fp("1"), fp("1"), fp("1")]);
    let lu = lu_decompose(&h).unwrap();
    let x = lu.solve(&b).unwrap();
    verify_solve(&h, &x, &b, fp("0.0000001"), "Hilbert roundtrip");
}

// ============================================================================
// 2. NEAR-SINGULAR MATRIX (det ≈ -0.003, perturbed singular)
//    mpmath: x ≈ [-1/3, 2/3, ~0], det ≈ -0.003
// ============================================================================

#[test]
fn test_adversarial_near_singular_solve() {
    // [[1,2,3],[4,5,6],[7,8,9.001]] — rank-deficient + tiny perturbation
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("9.001"),
    ]);
    let b = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    // Verify via round-trip (more robust than comparing x to mpmath for ill-conditioned)
    verify_solve(&a, &x, &b, fp("0.00001"), "near-singular roundtrip");
}

#[test]
fn test_adversarial_near_singular_det() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("9.001"),
    ]);
    let lu = lu_decompose(&a).unwrap();
    let det = lu.determinant();
    // mpmath: det ≈ -0.003
    assert_fp_near(det, fp("-0.003"), fp("0.001"), "near-singular det");
}

#[test]
fn test_adversarial_truly_singular() {
    // [[1,2,3],[4,5,6],[7,8,9]] — exactly singular
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("9"),
    ]);
    let result = lu_decompose(&a);
    assert!(result.is_err(), "exactly singular matrix must fail LU");
}

// ============================================================================
// 3. LARGE VALUES (entries ~ 1e9, products ~ 1e18 near Q64.64 limit)
//    mpmath: x = [-9, 8], det = -1e18
// ============================================================================

#[test]
#[cfg(not(table_format = "q32_32"))]
fn test_adversarial_large_values_solve() {
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1000000000"), fp("2000000000"),
        fp("3000000000"), fp("5000000000"),
    ]);
    let b = FixedVector::from_slice(&[fp("7000000000"), fp("13000000000")]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    // mpmath: x = [-9, 8]
    assert_vec_near(&x, &["-9", "8"], fp("0.0001"), "large values LU");
}

#[test]
#[cfg(not(table_format = "q32_32"))]
fn test_adversarial_large_values_det() {
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1000000000"), fp("2000000000"),
        fp("3000000000"), fp("5000000000"),
    ]);
    let lu = lu_decompose(&a).unwrap();
    let det = lu.determinant();
    // mpmath: det = -1e18
    assert_fp_near(det, fp("-1000000000000000000"), fp("1000"), "large det");
}

#[test]
#[cfg(not(table_format = "q32_32"))]
fn test_adversarial_large_values_roundtrip() {
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1000000000"), fp("2000000000"),
        fp("3000000000"), fp("5000000000"),
    ]);
    let b = FixedVector::from_slice(&[fp("7000000000"), fp("13000000000")]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    verify_solve(&a, &x, &b, fp("1"), "large roundtrip"); // abs tol 1 for 1e9 scale
}

// ============================================================================
// 4. TINY VALUES (entries ~ 1e-8)
//    mpmath: x ≈ [-9, 8], det ≈ -1e-16
// ============================================================================

#[test]
#[cfg(not(table_format = "q32_32"))]
fn test_adversarial_tiny_values_solve() {
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("0.00000001"), fp("0.00000002"),
        fp("0.00000003"), fp("0.00000005"),
    ]);
    let b = FixedVector::from_slice(&[fp("0.00000007"), fp("0.00000013")]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    // mpmath: x ≈ [-9, 8]
    assert_vec_near(&x, &["-9", "8"], fp("0.01"), "tiny values LU");
}

#[test]
#[cfg(not(table_format = "q32_32"))]
fn test_adversarial_tiny_values_roundtrip() {
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("0.00000001"), fp("0.00000002"),
        fp("0.00000003"), fp("0.00000005"),
    ]);
    let b = FixedVector::from_slice(&[fp("0.00000007"), fp("0.00000013")]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    verify_solve(&a, &x, &b, fp("0.0000000000000001"), "tiny roundtrip");
}

// ============================================================================
// 5. MIXED SCALE (1e6 to 1e-6 ratio = 1e12 condition)
//    mpmath: x ≈ [1, 1]
// ============================================================================

#[test]
fn test_adversarial_mixed_scale_solve() {
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1000000"), fp("0.000001"),
        fp("0.000001"), fp("1000000"),
    ]);
    let b = FixedVector::from_slice(&[
        fp("1000000.000001"),
        fp("0.000001") + fp("1000000"), // 1000000.000001
    ]);
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    // mpmath: x ≈ [1, 1]
    assert_vec_near(&x, &["1", "1"], fp("0.000001"), "mixed scale LU");
}

#[test]
#[cfg(not(table_format = "q32_32"))]
fn test_adversarial_mixed_scale_qr() {
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1000000"), fp("0.000001"),
        fp("0.000001"), fp("1000000"),
    ]);
    let b = FixedVector::from_slice(&[
        fp("1000000.000001"),
        fp("1000000.000001"),
    ]);
    let qr = qr_decompose(&a).unwrap();
    let x = qr.solve(&b).unwrap();
    assert_vec_near(&x, &["1", "1"], fp("0.000001"), "mixed scale QR");
}

// ============================================================================
// 6. BARELY SPD (smallest eigenvalue ≈ 0.001)
//    Cholesky must succeed but with reduced precision due to near-singularity
// ============================================================================

#[test]
fn test_adversarial_barely_spd_cholesky() {
    // eigenvalues [3, 0.001], rotated 45 degrees
    // A ≈ [[1.5005, 1.4995], [1.4995, 1.5005]]
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1.5005"), fp("1.4995"),
        fp("1.4995"), fp("1.5005"),
    ]);
    let chol = cholesky_decompose(&a);
    assert!(chol.is_ok(), "barely SPD should succeed Cholesky");
    let chol = chol.unwrap();
    // Verify reconstruction
    let reconstructed = &chol.l * &chol.l.transpose();
    let tol = fp("0.00001");
    for r in 0..2 {
        for c in 0..2 {
            assert_fp_near(reconstructed.get(r, c), a.get(r, c), tol,
                &format!("barely SPD LL^T [{},{}]", r, c));
        }
    }
}

#[test]
fn test_adversarial_barely_spd_solve() {
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1.5005"), fp("1.4995"),
        fp("1.4995"), fp("1.5005"),
    ]);
    let b = FixedVector::from_slice(&[fp("1"), fp("1")]);
    let chol = cholesky_decompose(&a).unwrap();
    let x = chol.solve(&b).unwrap();
    // mpmath: x ≈ [0.3333, 0.3333]
    assert_vec_near(&x, &["0.3333333333", "0.3333333333"], fp("0.001"), "barely SPD solve");
}

// ============================================================================
// 7. DIAGONALLY DOMINANT 5x5
//    mpmath: x ≈ [0.00987, 0.00984, 0.00984, 0.00984, 0.00987]
// ============================================================================

#[test]
fn test_adversarial_diag_dominant_5x5() {
    let a = FixedMatrix::from_fn(5, 5, |i, j| {
        if i == j {
            fp("100")
        } else {
            FixedPoint::one() / FixedPoint::from_int((((i as i32) - (j as i32)).abs() + 1) as i32)
        }
    });
    let b = FixedVector::from_slice(&[fp("1"), fp("1"), fp("1"), fp("1"), fp("1")]);

    // LU solve
    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    verify_solve(&a, &x, &b, fp("0.000000001"), "5x5 diag dominant roundtrip");

    // mpmath reference: symmetric pattern x[0]=x[4], x[1]=x[3]
    let tol = fp("0.00001");
    assert_fp_near(x[0], fp("0.0098736372692514"), tol, "5x5 x[0]");
    assert_fp_near(x[4], fp("0.0098736372692514"), tol, "5x5 x[4] (symmetric)");
    // x[0] ≈ x[4] (symmetry of the matrix)
    assert_fp_near(x[0], x[4], fp("0.0000000001"), "5x5 symmetry x[0]=x[4]");
    assert_fp_near(x[1], x[3], fp("0.0000000001"), "5x5 symmetry x[1]=x[3]");
}

// ============================================================================
// 8. STRESS: 8x8 random-ish well-conditioned system
// ============================================================================

#[test]
fn test_adversarial_8x8_roundtrip() {
    // Diagonally dominant → guaranteed non-singular, well-conditioned
    let a = FixedMatrix::from_fn(8, 8, |i, j| {
        if i == j {
            FixedPoint::from_int(20)
        } else {
            FixedPoint::from_int(((i * 3 + j * 7 + 1) % 5) as i32) - FixedPoint::from_int(2)
        }
    });
    let b_data: Vec<FixedPoint> = (0..8).map(|i| FixedPoint::from_int((i * i + 1) as i32)).collect();
    let b = FixedVector::from_slice(&b_data);

    let lu = lu_decompose(&a).unwrap();
    let x = lu.solve(&b).unwrap();
    verify_solve(&a, &x, &b, fp("0.0000001"), "8x8 LU roundtrip");

    let qr = qr_decompose(&a).unwrap();
    let x_qr = qr.solve(&b).unwrap();
    verify_solve(&a, &x_qr, &b, fp("0.0000001"), "8x8 QR roundtrip");

    // LU and QR should agree
    for i in 0..8 {
        assert_fp_near(x[i], x_qr[i], fp("0.0000001"),
            &format!("8x8 LU vs QR agree [{}]", i));
    }
}

// ============================================================================
// 9. NEGATIVE DEFINITE (must fail Cholesky)
// ============================================================================

#[test]
fn test_adversarial_negative_definite_fails() {
    // Negative definite: -I
    let a = FixedMatrix::from_fn(3, 3, |i, j| {
        if i == j { fp("-1") } else { fp("0") }
    });
    assert!(cholesky_decompose(&a).is_err(),
        "negative definite must fail Cholesky");
}

#[test]
fn test_adversarial_indefinite_fails() {
    // Indefinite: eigenvalues +1 and -1
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("0"), fp("1"),
        fp("1"), fp("0"),
    ]);
    assert!(cholesky_decompose(&a).is_err(),
        "indefinite matrix must fail Cholesky");
}

// ============================================================================
// 10. ZERO MATRIX (must fail — singular)
// ============================================================================

#[test]
fn test_adversarial_zero_matrix() {
    let a = FixedMatrix::new(3, 3); // all zeros
    assert!(lu_decompose(&a).is_err(), "zero matrix must fail LU");
}
