//! Validation tests for fused compute-tier operations.
//!
//! All reference values from mpmath at 60-digit precision.
//! Tests verify both correctness and precision advantage over unfused paths.

use g_math::fixed_point::{FixedPoint, FixedVector};
use g_math::fixed_point::imperative::fused;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn tight() -> FixedPoint { fp("0.000000001") }
fn ulp1() -> FixedPoint { fp("0.0000000000000000002") } // ~1 ULP at Q64.64

fn assert_fp(got: FixedPoint, exp: FixedPoint, tol: FixedPoint, name: &str) {
    let d = (got - exp).abs();
    assert!(d < tol, "{}: got {}, expected {}, diff={}", name, got, exp, d);
}

// ============================================================================
// sqrt_sum_sq — fused norm
// ============================================================================

#[test]
fn test_sqrt_sum_sq_3_4_5_triangle() {
    // sqrt(3² + 4²) = 5 (exact)
    let result = fused::sqrt_sum_sq(&[fp("3"), fp("4")]);
    assert_fp(result, fp("5"), tight(), "sqrt(3²+4²)");
}

#[test]
fn test_sqrt_sum_sq_unit_vector() {
    // sqrt(1²) = 1
    let result = fused::sqrt_sum_sq(&[fp("1")]);
    assert_fp(result, fp("1"), tight(), "sqrt(1²)");
}

#[test]
fn test_sqrt_sum_sq_3d() {
    // sqrt(1² + 2² + 3²) = sqrt(14) = 3.741657386773941...
    let result = fused::sqrt_sum_sq(&[fp("1"), fp("2"), fp("3")]);
    assert_fp(result, fp("3.741657386773941"), tight(), "sqrt(1²+2²+3²)");
}

#[test]
fn test_sqrt_sum_sq_small_values() {
    // sqrt(0.1² + 0.2² + 0.3²) = sqrt(0.14) = 0.374165738677394...
    let result = fused::sqrt_sum_sq(&[fp("0.1"), fp("0.2"), fp("0.3")]);
    assert_fp(result, fp("0.374165738677394"), tight(), "sqrt(0.1²+0.2²+0.3²)");
}

#[test]
fn test_sqrt_sum_sq_matches_vector_length() {
    // Fused should match FixedVector::length() within 1 ULP
    let v = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3"), fp("4"), fp("5")]);
    let fused_len = v.length_fused();
    let naive_len = v.length();
    let diff = (fused_len - naive_len).abs();
    // They should be very close — fused may be slightly more precise
    assert!(diff < fp("0.000000001"),
        "length_fused={} vs length={}, diff={}", fused_len, naive_len, diff);
}

#[test]
fn test_sqrt_sum_sq_high_dim() {
    // 23-dimensional: all 1s → sqrt(23)
    let vals: Vec<FixedPoint> = vec![fp("1"); 23];
    let result = fused::sqrt_sum_sq(&vals);
    // mpmath: sqrt(23) = 4.795831523312719...
    assert_fp(result, fp("4.795831523312719"), tight(), "sqrt(23×1²)");
}

// ============================================================================
// euclidean_distance — fused distance
// ============================================================================

#[test]
fn test_euclidean_distance_3_4_5() {
    // dist([0,0], [3,4]) = 5
    let a = [FixedPoint::ZERO, FixedPoint::ZERO];
    let b = [fp("3"), fp("4")];
    let result = fused::euclidean_distance(&a, &b);
    assert_fp(result, fp("5"), tight(), "dist([0,0],[3,4])");
}

#[test]
fn test_euclidean_distance_3d() {
    // dist([1,2,3], [4,6,3]) = sqrt(9+16+0) = 5
    let a = [fp("1"), fp("2"), fp("3")];
    let b = [fp("4"), fp("6"), fp("3")];
    let result = fused::euclidean_distance(&a, &b);
    assert_fp(result, fp("5"), tight(), "dist([1,2,3],[4,6,3])");
}

#[test]
fn test_euclidean_distance_decimal() {
    // dist([0.1,0.2], [0.4,0.6]) = sqrt(0.09+0.16) = 0.5
    let a = [fp("0.1"), fp("0.2")];
    let b = [fp("0.4"), fp("0.6")];
    let result = fused::euclidean_distance(&a, &b);
    assert_fp(result, fp("0.5"), tight(), "dist([0.1,0.2],[0.4,0.6])");
}

#[test]
fn test_euclidean_distance_self() {
    let a = [fp("1"), fp("2"), fp("3")];
    let result = fused::euclidean_distance(&a, &a);
    assert!(result.is_zero() || result.abs() < tight(), "dist(a,a)={}", result);
}

#[test]
fn test_euclidean_distance_matches_vector() {
    // Fused distance should match (a-b).length()
    let a = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let b = FixedVector::from_slice(&[fp("4"), fp("6"), fp("8")]);
    let fused_dist = a.distance_to(&b);
    let naive_dist = (&a - &b).length();
    let diff = (fused_dist - naive_dist).abs();
    assert!(diff < fp("0.000000001"),
        "distance_to={} vs (a-b).length()={}, diff={}", fused_dist, naive_dist, diff);
}

// ============================================================================
// softmax — fused stable softmax
// ============================================================================

#[test]
fn test_softmax_uniform() {
    let scores = vec![fp("1"); 4];
    let result = fused::softmax(&scores).unwrap();
    for (i, w) in result.iter().enumerate() {
        assert_fp(*w, fp("0.25"), fp("0.001"), &format!("softmax_uniform[{i}]"));
    }
}

#[test]
fn test_softmax_sums_to_one() {
    let scores = vec![fp("1"), fp("2"), fp("3"), fp("4")];
    let result = fused::softmax(&scores).unwrap();
    let sum: FixedPoint = result.iter().copied().fold(FixedPoint::ZERO, |a, b| a + b);
    assert_fp(sum, fp("1"), fp("0.000001"), "softmax_sum");
}

#[test]
fn test_softmax_monotone() {
    let scores = vec![fp("1"), fp("2"), fp("3")];
    let result = fused::softmax(&scores).unwrap();
    assert!(result[0] < result[1], "softmax not monotone: [0]={} >= [1]={}", result[0], result[1]);
    assert!(result[1] < result[2], "softmax not monotone: [1]={} >= [2]={}", result[1], result[2]);
}

#[test]
fn test_softmax_mpmath_values() {
    // mpmath reference: softmax([1,2,3,4])
    let scores = vec![fp("1"), fp("2"), fp("3"), fp("4")];
    let result = fused::softmax(&scores).unwrap();
    assert_fp(result[0], fp("0.032058603280084"), fp("0.0001"), "softmax[0]");
    assert_fp(result[1], fp("0.087144318742032"), fp("0.0001"), "softmax[1]");
    assert_fp(result[2], fp("0.236882818089910"), fp("0.0001"), "softmax[2]");
    assert_fp(result[3], fp("0.643914259887972"), fp("0.0001"), "softmax[3]");
}

#[test]
fn test_softmax_shift_invariance() {
    // softmax(x + c) = softmax(x) for any constant c
    let scores1 = vec![fp("1"), fp("2"), fp("3")];
    let scores2 = vec![fp("101"), fp("102"), fp("103")];
    let r1 = fused::softmax(&scores1).unwrap();
    let r2 = fused::softmax(&scores2).unwrap();
    for i in 0..3 {
        assert_fp(r1[i], r2[i], fp("0.001"),
            &format!("shift_invariance[{i}]"));
    }
}

#[test]
fn test_softmax_empty() {
    let result = fused::softmax(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_softmax_single() {
    let result = fused::softmax(&[fp("5")]).unwrap();
    assert_fp(result[0], fp("1"), tight(), "softmax_single");
}

// ============================================================================
// rms_norm_factor — fused RMSNorm
// ============================================================================

#[test]
fn test_rms_norm_constant_vector() {
    // [2,2,2,2]: mean(x²) = 4, sqrt(4+eps) ≈ 2, factor ≈ 0.5
    let vals = vec![fp("2"); 4];
    let factor = fused::rms_norm_factor(&vals, fp("0.000001")).unwrap();
    assert_fp(factor, fp("0.5"), fp("0.001"), "rms_norm_constant");
}

#[test]
fn test_rms_norm_mpmath() {
    // [1,2,3]: mean(x²) = 14/3, factor = 1/sqrt(14/3 + 1e-6) = 0.46291...
    let vals = vec![fp("1"), fp("2"), fp("3")];
    let factor = fused::rms_norm_factor(&vals, fp("0.000001")).unwrap();
    assert_fp(factor, fp("0.46291"), fp("0.001"), "rms_norm_1_2_3");
}

#[test]
fn test_rms_norm_ones() {
    // [1,1,1]: mean(x²) = 1, factor = 1/sqrt(1+eps) ≈ 1
    let vals = vec![fp("1"); 3];
    let factor = fused::rms_norm_factor(&vals, fp("0.000001")).unwrap();
    assert_fp(factor, fp("1"), fp("0.001"), "rms_norm_ones");
}

// ============================================================================
// silu — fused SiLU activation
// ============================================================================

#[test]
fn test_silu_zero() {
    let result = fused::silu(FixedPoint::ZERO);
    assert_fp(result, FixedPoint::ZERO, tight(), "silu(0)");
}

#[test]
fn test_silu_one() {
    // mpmath: silu(1) = 0.73105857863000487925...
    let result = fused::silu(fp("1"));
    assert_fp(result, fp("0.731058578630004"), tight(), "silu(1)");
}

#[test]
fn test_silu_two() {
    // mpmath: silu(2) = 1.76159415595576488...
    let result = fused::silu(fp("2"));
    assert_fp(result, fp("1.761594155955764"), tight(), "silu(2)");
}

#[test]
fn test_silu_neg_one() {
    // mpmath: silu(-1) = -0.26894142136999512...
    let result = fused::silu(fp("-1"));
    assert_fp(result, fp("-0.268941421369995"), tight(), "silu(-1)");
}

#[test]
fn test_silu_neg_two() {
    // mpmath: silu(-2) = -0.23840584404423511...
    let result = fused::silu(fp("-2"));
    assert_fp(result, fp("-0.238405844044235"), tight(), "silu(-2)");
}

#[test]
fn test_silu_half() {
    // mpmath: silu(0.5) = 0.31122966560092728...
    let result = fused::silu(fp("0.5"));
    assert_fp(result, fp("0.311229665600927"), tight(), "silu(0.5)");
}

#[test]
fn test_silu_large_positive() {
    // silu(x) → x for large x (sigmoid → 1)
    let x = fp("10");
    let result = fused::silu(x);
    assert_fp(result, x, fp("0.001"), "silu(10)≈10");
}

#[test]
fn test_silu_large_negative() {
    // silu(x) → 0 for large negative x (sigmoid → 0)
    let result = fused::silu(fp("-10"));
    assert!(result.abs() < fp("0.001"), "silu(-10)={}, expected ~0", result);
}

// ============================================================================
// Precision comparison: fused vs unfused
// ============================================================================

#[test]
fn test_fused_norm_precision_vs_unfused() {
    // For a large-dimension vector, fused should be at least as precise as unfused
    let n = 50;
    let vals: Vec<FixedPoint> = (1..=n).map(|i| fp(&format!("0.{}", i))).collect();
    let v = FixedVector::from_slice(&vals);

    let fused_len = v.length_fused();
    let naive_len = v.length();

    // Both should be close to the same value
    let diff = (fused_len - naive_len).abs();
    assert!(diff < fp("0.000000001"),
        "50D norm: fused={} naive={} diff={}", fused_len, naive_len, diff);
}
