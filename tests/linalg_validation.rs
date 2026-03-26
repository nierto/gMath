//! Linear algebra validation tests for L1A (Basic Matrix/Vector Arithmetic).
//!
//! Tests are structured in three categories:
//! 1. Algebraic identity tests (exact properties that must hold)
//! 2. Known-value tests (manually computed reference values)
//! 3. Compute-tier precision tests (verify accumulation advantage)

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::OverflowDetected;

// ============================================================================
// S1: Serialization round-trip tests
// ============================================================================

#[test]
fn test_fixedpoint_serialization_roundtrip() {
    let values = [
        fp("0"), fp("1"), fp("-1"), fp("3.14159265358979323846"),
        fp("0.5"), fp("-0.333333333333333"), fp("12345.6789"),
    ];
    for &val in &values {
        let bytes = val.to_bytes();
        let recovered = FixedPoint::from_bytes(&bytes).unwrap();
        assert_eq!(val, recovered, "round-trip failed for {}", val);
    }
}

#[test]
fn test_fixedpoint_raw_bytes_roundtrip() {
    let val = fp("2.718281828459045235360287471352662");
    let bytes = val.to_raw_bytes();
    let recovered = FixedPoint::from_raw_bytes(&bytes).unwrap();
    assert_eq!(val, recovered);
}

#[test]
fn test_fixedpoint_profile_tag() {
    let val = fp("1");
    let bytes = val.to_bytes();
    let tag = bytes[0];
    assert!(tag >= 0x01 && tag <= 0x05, "unexpected profile tag: {:#x}", tag);
    assert_eq!(tag, FixedPoint::profile_tag());
}

#[test]
fn test_fixedpoint_wrong_profile_tag() {
    let val = fp("1");
    let mut bytes = val.to_bytes();
    bytes[0] = 0xFF; // wrong tag
    assert!(FixedPoint::from_bytes(&bytes).is_err());
}

#[test]
fn test_fixedpoint_truncated_bytes() {
    assert!(FixedPoint::from_bytes(&[0x01]).is_err());
    assert!(FixedPoint::from_bytes(&[]).is_err());
}

#[test]
fn test_fixedvector_serialization_roundtrip() {
    let v = FixedVector::from_slice(&[fp("1.5"), fp("-2.7"), fp("0"), fp("3.14")]);
    let bytes = v.to_bytes();
    let recovered = FixedVector::from_bytes(&bytes).unwrap();
    assert_eq!(v, recovered);
}

#[test]
fn test_fixedvector_empty_roundtrip() {
    let v = FixedVector::new(0);
    let bytes = v.to_bytes();
    let recovered = FixedVector::from_bytes(&bytes).unwrap();
    assert_eq!(v.len(), recovered.len());
}

#[test]
fn test_fixedmatrix_serialization_roundtrip() {
    let m = FixedMatrix::from_slice(2, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
    ]);
    let bytes = m.to_bytes();
    let recovered = FixedMatrix::from_bytes(&bytes).unwrap();
    assert_eq!(m, recovered);
}

#[test]
fn test_fixedmatrix_identity_roundtrip() {
    let m = FixedMatrix::identity(4);
    let bytes = m.to_bytes();
    let recovered = FixedMatrix::from_bytes(&bytes).unwrap();
    assert_eq!(m, recovered);
}

#[test]
fn test_serialization_byte_sizes() {
    let raw_len = FixedPoint::raw_byte_len();
    // Profile-dependent: 4 (Q16.16), 8 (Q32.32), 16 (Q64.64), 32 (Q128.128), or 64 (Q256.256)
    assert!(raw_len == 4 || raw_len == 8 || raw_len == 16 || raw_len == 32 || raw_len == 64,
        "unexpected raw_byte_len: {}", raw_len);

    let val = fp("1");
    let bytes = val.to_bytes();
    assert_eq!(bytes.len(), 1 + raw_len); // 1 tag + raw

    let v = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let vbytes = v.to_bytes();
    assert_eq!(vbytes.len(), 4 + 3 * raw_len); // u32 + 3 elements

    let m = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let mbytes = m.to_bytes();
    assert_eq!(mbytes.len(), 8 + 4 * raw_len); // 2×u32 + 4 elements
}

// ============================================================================
// Helper: construct FixedPoint from string for maximum precision
// ============================================================================

fn fp(s: &str) -> FixedPoint {
    FixedPoint::from_str(s)
}

// ============================================================================
// 1. FixedVector operator tests
// ============================================================================

#[test]
fn test_vector_add() {
    let a = FixedVector::from_slice(&[fp("1.5"), fp("2.5"), fp("3.0")]);
    let b = FixedVector::from_slice(&[fp("0.5"), fp("1.0"), fp("-1.0")]);
    let c = &a + &b;
    assert_eq!(c[0], fp("2.0"));
    assert_eq!(c[1], fp("3.5"));
    assert_eq!(c[2], fp("2.0"));
}

#[test]
fn test_vector_sub() {
    let a = FixedVector::from_slice(&[fp("5.0"), fp("3.0")]);
    let b = FixedVector::from_slice(&[fp("2.0"), fp("1.0")]);
    let c = &a - &b;
    assert_eq!(c[0], fp("3.0"));
    assert_eq!(c[1], fp("2.0"));
}

#[test]
fn test_vector_neg() {
    let a = FixedVector::from_slice(&[fp("1.0"), fp("-2.0"), fp("3.0")]);
    let b = -&a;
    assert_eq!(b[0], fp("-1.0"));
    assert_eq!(b[1], fp("2.0"));
    assert_eq!(b[2], fp("-3.0"));
}

#[test]
fn test_vector_scalar_mul() {
    let v = FixedVector::from_slice(&[fp("2.0"), fp("3.0"), fp("4.0")]);
    let s = fp("0.5");
    let r = &v * s;
    assert_eq!(r[0], fp("1.0"));
    assert_eq!(r[1], fp("1.5"));
    assert_eq!(r[2], fp("2.0"));
}

#[test]
fn test_vector_add_sub_identity() {
    // (a + b) - b = a
    let a = FixedVector::from_slice(&[fp("1.23"), fp("4.56"), fp("7.89")]);
    let b = FixedVector::from_slice(&[fp("9.87"), fp("6.54"), fp("3.21")]);
    let result = &(&a + &b) - &b;
    for i in 0..a.len() {
        assert_eq!(result[i], a[i], "add-sub identity failed at index {}", i);
    }
}

#[test]
fn test_vector_neg_double_identity() {
    // -(-a) = a
    let a = FixedVector::from_slice(&[fp("1.5"), fp("-2.7"), fp("0.0")]);
    let result = -(-a.clone());
    assert_eq!(result, a);
}

// ============================================================================
// 2. Cross product tests
// ============================================================================

#[test]
fn test_cross_product_basic() {
    // i × j = k
    let i = FixedVector::from_slice(&[fp("1"), fp("0"), fp("0")]);
    let j = FixedVector::from_slice(&[fp("0"), fp("1"), fp("0")]);
    let k = i.cross(&j);
    assert_eq!(k[0], fp("0"));
    assert_eq!(k[1], fp("0"));
    assert_eq!(k[2], fp("1"));
}

#[test]
fn test_cross_product_anticommutative() {
    // a × b = -(b × a)
    let a = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let b = FixedVector::from_slice(&[fp("4"), fp("5"), fp("6")]);
    let ab = a.cross(&b);
    let ba = b.cross(&a);
    assert_eq!(ab, -ba);
}

#[test]
fn test_cross_product_self_is_zero() {
    // a × a = 0
    let a = FixedVector::from_slice(&[fp("3.5"), fp("-1.2"), fp("7.8")]);
    let result = a.cross(&a);
    let zero = FixedVector::from_slice(&[fp("0"), fp("0"), fp("0")]);
    assert_eq!(result, zero);
}

#[test]
fn test_cross_product_perpendicular() {
    // (a × b) · a = 0 and (a × b) · b = 0
    let a = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let b = FixedVector::from_slice(&[fp("4"), fp("5"), fp("6")]);
    let cross = a.cross(&b);
    assert_eq!(cross.dot(&a), fp("0"));
    assert_eq!(cross.dot(&b), fp("0"));
}

// ============================================================================
// 3. Outer product tests
// ============================================================================

#[test]
fn test_outer_product() {
    let u = FixedVector::from_slice(&[fp("1"), fp("2")]);
    let v = FixedVector::from_slice(&[fp("3"), fp("4"), fp("5")]);
    let m = u.outer_product(&v);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    assert_eq!(m.get(0, 0), fp("3"));
    assert_eq!(m.get(0, 1), fp("4"));
    assert_eq!(m.get(0, 2), fp("5"));
    assert_eq!(m.get(1, 0), fp("6"));
    assert_eq!(m.get(1, 1), fp("8"));
    assert_eq!(m.get(1, 2), fp("10"));
}

// ============================================================================
// 4. FixedMatrix constructor tests
// ============================================================================

#[test]
fn test_matrix_identity() {
    let i3 = FixedMatrix::identity(3);
    assert_eq!(i3.rows(), 3);
    assert_eq!(i3.cols(), 3);
    for r in 0..3 {
        for c in 0..3 {
            if r == c {
                assert_eq!(i3.get(r, c), fp("1"));
            } else {
                assert_eq!(i3.get(r, c), fp("0"));
            }
        }
    }
}

#[test]
fn test_matrix_diagonal() {
    let v = FixedVector::from_slice(&[fp("2"), fp("3"), fp("4")]);
    let d = FixedMatrix::diagonal(&v);
    assert_eq!(d.get(0, 0), fp("2"));
    assert_eq!(d.get(1, 1), fp("3"));
    assert_eq!(d.get(2, 2), fp("4"));
    assert_eq!(d.get(0, 1), fp("0"));
    assert_eq!(d.get(1, 0), fp("0"));
}

#[test]
fn test_matrix_from_fn() {
    // Hilbert-like matrix: M[i][j] = 1/(i+j+1)
    let m = FixedMatrix::from_fn(2, 2, |r, c| {
        FixedPoint::one() / FixedPoint::from_int((r + c + 1) as i32)
    });
    assert_eq!(m.get(0, 0), fp("1") / fp("1")); // 1/1
    assert_eq!(m.get(0, 1), fp("1") / fp("2")); // 1/2
    assert_eq!(m.get(1, 0), fp("1") / fp("2")); // 1/2
    assert_eq!(m.get(1, 1), fp("1") / fp("3")); // 1/3
}

// ============================================================================
// 5. Matrix arithmetic tests
// ============================================================================

#[test]
fn test_matrix_transpose() {
    let m = FixedMatrix::from_slice(2, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
    ]);
    let t = m.transpose();
    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.get(0, 0), fp("1"));
    assert_eq!(t.get(0, 1), fp("4"));
    assert_eq!(t.get(1, 0), fp("2"));
    assert_eq!(t.get(1, 1), fp("5"));
    assert_eq!(t.get(2, 0), fp("3"));
    assert_eq!(t.get(2, 1), fp("6"));
}

#[test]
fn test_matrix_transpose_transpose_identity() {
    // (Aᵀ)ᵀ = A
    let m = FixedMatrix::from_slice(2, 3, &[
        fp("1.5"), fp("2.7"), fp("3.1"),
        fp("4.9"), fp("5.3"), fp("6.8"),
    ]);
    assert_eq!(m.transpose().transpose(), m);
}

#[test]
fn test_matrix_trace() {
    let m = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("9"),
    ]);
    assert_eq!(m.trace(), fp("15")); // 1 + 5 + 9
}

#[test]
fn test_matrix_add_sub() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let b = FixedMatrix::from_slice(2, 2, &[fp("5"), fp("6"), fp("7"), fp("8")]);
    let sum = &a + &b;
    assert_eq!(sum.get(0, 0), fp("6"));
    assert_eq!(sum.get(1, 1), fp("12"));
    let diff = &a - &b;
    assert_eq!(diff.get(0, 0), fp("-4"));
    assert_eq!(diff.get(1, 1), fp("-4"));
}

#[test]
fn test_matrix_scalar_mul() {
    let m = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let s = fp("3");
    let r = &m * s;
    assert_eq!(r.get(0, 0), fp("3"));
    assert_eq!(r.get(0, 1), fp("6"));
    assert_eq!(r.get(1, 0), fp("9"));
    assert_eq!(r.get(1, 1), fp("12"));
}

// ============================================================================
// 6. Matrix-vector multiply tests
// ============================================================================

#[test]
fn test_matrix_vector_multiply() {
    // [1 2] * [5] = [1*5+2*6]   = [17]
    // [3 4]   [6]   [3*5+4*6]     [39]
    let m = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let v = FixedVector::from_slice(&[fp("5"), fp("6")]);
    let r = m.mul_vector(&v);
    assert_eq!(r[0], fp("17"));
    assert_eq!(r[1], fp("39"));
}

#[test]
fn test_identity_mul_vector() {
    // I * v = v
    let id = FixedMatrix::identity(3);
    let v = FixedVector::from_slice(&[fp("1.5"), fp("2.5"), fp("3.5")]);
    let r = id.mul_vector(&v);
    assert_eq!(r, v);
}

// ============================================================================
// 7. Matrix-matrix multiply tests
// ============================================================================

#[test]
fn test_matrix_multiply_2x2() {
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8]   = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]     [43 50]
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let b = FixedMatrix::from_slice(2, 2, &[fp("5"), fp("6"), fp("7"), fp("8")]);
    let c = &a * &b;
    assert_eq!(c.get(0, 0), fp("19"));
    assert_eq!(c.get(0, 1), fp("22"));
    assert_eq!(c.get(1, 0), fp("43"));
    assert_eq!(c.get(1, 1), fp("50"));
}

#[test]
fn test_matrix_multiply_identity() {
    // A * I = A
    let a = FixedMatrix::from_slice(2, 2, &[fp("1.5"), fp("2.7"), fp("3.1"), fp("4.9")]);
    let id = FixedMatrix::identity(2);
    let r = &a * &id;
    assert_eq!(r, a);
}

#[test]
fn test_matrix_multiply_identity_left() {
    // I * A = A
    let a = FixedMatrix::from_slice(2, 2, &[fp("1.5"), fp("2.7"), fp("3.1"), fp("4.9")]);
    let id = FixedMatrix::identity(2);
    let r = &id * &a;
    assert_eq!(r, a);
}

#[test]
fn test_matrix_multiply_non_square() {
    // [1 2 3] * [7  8 ]   = [1*7+2*9+3*11  1*8+2*10+3*12]   = [58  64]
    // [4 5 6]   [9  10]     [4*7+5*9+6*11  4*8+5*10+6*12]     [139 154]
    //           [11 12]
    let a = FixedMatrix::from_slice(2, 3, &[fp("1"), fp("2"), fp("3"), fp("4"), fp("5"), fp("6")]);
    let b = FixedMatrix::from_slice(3, 2, &[fp("7"), fp("8"), fp("9"), fp("10"), fp("11"), fp("12")]);
    let c = &a * &b;
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.get(0, 0), fp("58"));
    assert_eq!(c.get(0, 1), fp("64"));
    assert_eq!(c.get(1, 0), fp("139"));
    assert_eq!(c.get(1, 1), fp("154"));
}

#[test]
fn test_matrix_transpose_multiply_property() {
    // (AB)ᵀ = BᵀAᵀ
    let a = FixedMatrix::from_slice(2, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
    ]);
    let b = FixedMatrix::from_slice(3, 2, &[
        fp("7"), fp("8"),
        fp("9"), fp("10"),
        fp("11"), fp("12"),
    ]);
    let ab_t = (&a * &b).transpose();
    let bt_at = &b.transpose() * &a.transpose();
    assert_eq!(ab_t, bt_at);
}

// ============================================================================
// 8. Block operations tests
// ============================================================================

#[test]
fn test_submatrix() {
    let m = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("9"),
    ]);
    let sub = m.submatrix(1, 1, 2, 2);
    assert_eq!(sub.get(0, 0), fp("5"));
    assert_eq!(sub.get(0, 1), fp("6"));
    assert_eq!(sub.get(1, 0), fp("8"));
    assert_eq!(sub.get(1, 1), fp("9"));
}

#[test]
fn test_set_submatrix() {
    let mut m = FixedMatrix::new(3, 3);
    let sub = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    m.set_submatrix(1, 1, &sub);
    assert_eq!(m.get(0, 0), fp("0"));
    assert_eq!(m.get(1, 1), fp("1"));
    assert_eq!(m.get(1, 2), fp("2"));
    assert_eq!(m.get(2, 1), fp("3"));
    assert_eq!(m.get(2, 2), fp("4"));
}

#[test]
fn test_kronecker_product() {
    // [1 2] ⊗ [0 5] = [1*0 1*5 2*0 2*5]   = [0  5  0  10]
    // [3 4]   [6 7]   [1*6 1*7 2*6 2*7]     [6  7  12 14]
    //                  [3*0 3*5 4*0 4*5]     [0  15 0  20]
    //                  [3*6 3*7 4*6 4*7]     [18 21 24 28]
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let b = FixedMatrix::from_slice(2, 2, &[fp("0"), fp("5"), fp("6"), fp("7")]);
    let k = a.kronecker(&b);
    assert_eq!(k.rows(), 4);
    assert_eq!(k.cols(), 4);
    assert_eq!(k.get(0, 0), fp("0"));
    assert_eq!(k.get(0, 1), fp("5"));
    assert_eq!(k.get(0, 2), fp("0"));
    assert_eq!(k.get(0, 3), fp("10"));
    assert_eq!(k.get(1, 0), fp("6"));
    assert_eq!(k.get(1, 1), fp("7"));
    assert_eq!(k.get(1, 2), fp("12"));
    assert_eq!(k.get(1, 3), fp("14"));
    assert_eq!(k.get(2, 0), fp("0"));
    assert_eq!(k.get(2, 1), fp("15"));
    assert_eq!(k.get(2, 2), fp("0"));
    assert_eq!(k.get(2, 3), fp("20"));
    assert_eq!(k.get(3, 0), fp("18"));
    assert_eq!(k.get(3, 1), fp("21"));
    assert_eq!(k.get(3, 2), fp("24"));
    assert_eq!(k.get(3, 3), fp("28"));
}

// ============================================================================
// 9. Compute-tier precision validation
// ============================================================================

#[test]
fn test_compute_tier_dot_vs_standard_dot() {
    // Both should agree for small vectors, but compute_tier is more precise.
    // For integer-valued vectors, they should be identical.
    let a = FixedVector::from_slice(&[fp("3"), fp("4")]);
    let b = FixedVector::from_slice(&[fp("5"), fp("6")]);
    assert_eq!(a.dot(&b), fp("39"));           // 3*5 + 4*6
    assert_eq!(a.dot_precise(&b), fp("39"));   // same via compute tier
}

#[test]
fn test_compute_tier_dot_fractional() {
    // 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 = 0.01 + 0.04 + 0.09 = 0.14
    let a = FixedVector::from_slice(&[fp("0.1"), fp("0.2"), fp("0.3")]);
    let b = a.clone();
    let standard = a.dot(&b);
    let precise = a.dot_precise(&b);
    let expected = fp("0.14");
    // Both should be extremely close to 0.14 (within a few ULP)
    #[cfg(table_format = "q16_16")]
    let tolerance = fp("0.01");
    #[cfg(not(table_format = "q16_16"))]
    let tolerance = fp("0.0000001");
    let diff_standard = (standard - expected).abs();
    let diff_precise = (precise - expected).abs();
    assert!(diff_standard < tolerance,
        "standard dot too far: diff={}", diff_standard);
    assert!(diff_precise < tolerance,
        "compute-tier dot too far: diff={}", diff_precise);
}

// ============================================================================
// 10. Matrix row/col extraction
// ============================================================================

#[test]
fn test_matrix_row_col() {
    let m = FixedMatrix::from_slice(2, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
    ]);
    let r0 = m.row(0);
    assert_eq!(r0, FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]));
    let c1 = m.col(1);
    assert_eq!(c1, FixedVector::from_slice(&[fp("2"), fp("5")]));
}

// ============================================================================
// 11. Large matrix multiply (stress test for compute-tier accumulation)
// ============================================================================

#[test]
fn test_large_identity_multiply() {
    // 16×16 identity * 16×16 identity = 16×16 identity
    let id = FixedMatrix::identity(16);
    let result = &id * &id;
    assert_eq!(result, id);
}

#[test]
fn test_matrix_multiply_accumulation_16x16() {
    // Create a 16×16 matrix where each element = 0.1
    // A * A should have each element = 16 * 0.1 * 0.1 = 0.16
    let val = fp("0.1");
    let a = FixedMatrix::from_fn(16, 16, |_, _| val);
    let result = &a * &a;
    let expected = fp("0.16"); // 16 * (0.1)^2

    // Check a few elements — compute-tier accumulation should give good precision
    for r in 0..4 {
        for c in 0..4 {
            let diff = (result.get(r, c) - expected).abs();
            // Allow small tolerance (a few ULP at most)
            #[cfg(table_format = "q16_16")]
            let tolerance = fp("0.01");
            #[cfg(not(table_format = "q16_16"))]
            let tolerance = fp("0.0000001");
            assert!(diff < tolerance,
                "16×16 multiply error at ({},{}): got {}, expected {}, diff={}",
                r, c, result.get(r, c), expected, diff);
        }
    }
}

// ============================================================================
// 12. UGOD overflow detection tests (try_* pattern)
// ============================================================================

#[test]
fn test_try_sqrt_positive() {
    let val = fp("4");
    let result = val.try_sqrt();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), fp("2"));
}

#[test]
fn test_try_sqrt_negative_returns_domain_error() {
    let val = fp("-1");
    let result = val.try_sqrt();
    assert!(result.is_err());
    match result.unwrap_err() {
        OverflowDetected::DomainError => {} // expected
        other => panic!("expected DomainError, got {:?}", other),
    }
}

#[test]
fn test_try_exp_normal_value() {
    let val = fp("1");
    let result = val.try_exp();
    assert!(result.is_ok());
    // e^1 ≈ 2.718...
    let e = result.unwrap();
    let diff = (e - fp("2.718281828")).abs();
    #[cfg(table_format = "q16_16")]
    let exp_tol = fp("0.01");
    #[cfg(not(table_format = "q16_16"))]
    let exp_tol = fp("0.000000001");
    assert!(diff < exp_tol, "exp(1) too far from e: {}", e);
}

/// exp(44) ≈ 1.28e19 exceeds Q64.64 range (~9.2e18) but fits in Q128.128+.
/// Q16.16: exp(44) also overflows.
#[test]
#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn test_try_exp_overflow_returns_tier_overflow() {
    let val = fp("44");
    let result = val.try_exp();
    assert!(result.is_err(), "exp(44) should overflow in this profile, got {:?}", result);
    match result.unwrap_err() {
        OverflowDetected::TierOverflow => {} // expected — UGOD working correctly
        other => panic!("expected TierOverflow, got {:?}", other),
    }
}

/// Q16.16/Q32.32: exp(43) ≈ 4.73e18 far exceeds both profiles' range.
#[test]
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
fn test_try_exp_near_boundary_succeeds() {
    // exp(43) ≈ 4.73 × 10^18, fits within Q64.64 range
    let val = fp("43");
    let result = val.try_exp();
    assert!(result.is_ok(), "exp(43) should fit in Q64.64, got {:?}", result);
}

#[test]
fn test_try_ln_domain_error() {
    let val = fp("-5");
    let result = val.try_ln();
    assert!(result.is_err());
    match result.unwrap_err() {
        OverflowDetected::DomainError => {}
        other => panic!("expected DomainError for ln(-5), got {:?}", other),
    }
}

#[test]
fn test_try_sin_cos_always_succeed() {
    // sin and cos never overflow — bounded to [-1, 1]
    let val = fp("100");
    assert!(val.try_sin().is_ok());
    assert!(val.try_cos().is_ok());
}

#[test]
fn test_try_asin_domain_error() {
    let val = fp("2"); // |x| > 1
    let result = val.try_asin();
    assert!(result.is_err(), "asin(2) should be domain error");
}
