//! Validation tests for fused transcendental paths.
//!
//! Tests:
//! 1. evaluate_sincos() — fused sin+cos with single range reduction
//! 2. Identity short-circuits: exp(ln(x)) = x, ln(exp(x)) = x
//!
//! Reference values from mpmath at 50+ decimal digits.

use g_math::canonical::{gmath, evaluate, evaluate_sincos, StackValue};
use g_math::fixed_point::FixedPoint;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

/// Convert StackValue to FixedPoint, handling any domain.
fn sv_to_fp(sv: &StackValue) -> FixedPoint {
    match sv {
        StackValue::Error(e) => panic!("sv_to_fp: error value {:?}", e),
        _ => {
            // Use to_decimal_string() for the most general conversion path
            let s = format!("{}", sv);
            // Parse the formatted output as FixedPoint
            fp(&s)
        }
    }
}

fn tight() -> FixedPoint { fp("0.000000001") }

fn assert_fp(got: FixedPoint, exp: FixedPoint, tol: FixedPoint, name: &str) {
    let d = (got - exp).abs();
    assert!(d < tol, "{}: got {}, expected {}, diff={}", name, got, exp, d);
}

// ============================================================================
// evaluate_sincos — fused sin+cos
// ============================================================================

#[test]
fn test_sincos_zero() {
    let (sin_val, cos_val) = evaluate_sincos(&gmath("0")).unwrap();
    let sin_fp = sv_to_fp(&sin_val);
    let cos_fp = sv_to_fp(&cos_val);
    assert_fp(sin_fp, fp("0"), tight(), "sin(0)");
    assert_fp(cos_fp, fp("1"), tight(), "cos(0)");
}

#[test]
fn test_sincos_half() {
    // mpmath: sin(0.5) = 0.47942553860420300027328793521557138808...
    // mpmath: cos(0.5) = 0.87758256189037271611726095893830865404...
    let (sin_val, cos_val) = evaluate_sincos(&gmath("0.5")).unwrap();
    let sin_fp = sv_to_fp(&sin_val);
    let cos_fp = sv_to_fp(&cos_val);
    assert_fp(sin_fp, fp("0.479425538604203"), tight(), "sin(0.5)");
    assert_fp(cos_fp, fp("0.877582561890372"), tight(), "cos(0.5)");
}

#[test]
fn test_sincos_one() {
    // mpmath: sin(1) = 0.84147098480789650665250232163029899962...
    // mpmath: cos(1) = 0.54030230586813971740093660744297660373...
    let (sin_val, cos_val) = evaluate_sincos(&gmath("1")).unwrap();
    let sin_fp = sv_to_fp(&sin_val);
    let cos_fp = sv_to_fp(&cos_val);
    assert_fp(sin_fp, fp("0.841470984807896"), tight(), "sin(1)");
    assert_fp(cos_fp, fp("0.540302305868139"), tight(), "cos(1)");
}

#[test]
fn test_sincos_negative() {
    // sin(-x) = -sin(x), cos(-x) = cos(x)
    let (sin_pos, cos_pos) = evaluate_sincos(&gmath("1")).unwrap();
    let (sin_neg, cos_neg) = evaluate_sincos(&(-gmath("1"))).unwrap();
    let sin_pos_fp = sv_to_fp(&sin_pos);
    let cos_pos_fp = sv_to_fp(&cos_pos);
    let sin_neg_fp = sv_to_fp(&sin_neg);
    let cos_neg_fp = sv_to_fp(&cos_neg);
    assert_fp(sin_neg_fp, -sin_pos_fp, tight(), "sin(-x) = -sin(x)");
    assert_fp(cos_neg_fp, cos_pos_fp, tight(), "cos(-x) = cos(x)");
}

#[test]
fn test_sincos_pythagorean_identity() {
    // sin²(x) + cos²(x) = 1 for various x values
    for x_str in &["0.5", "1", "1.5", "2", "3", "0.1", "0.01"] {
        let (sin_val, cos_val) = evaluate_sincos(&gmath(x_str)).unwrap();
        let sin_fp = sv_to_fp(&sin_val);
        let cos_fp = sv_to_fp(&cos_val);
        let sum_sq = sin_fp * sin_fp + cos_fp * cos_fp;
        assert_fp(sum_sq, fp("1"), tight(), &format!("sin²({x_str}) + cos²({x_str}) = 1"));
    }
}

#[test]
fn test_sincos_matches_separate_sin_cos() {
    // Fused sincos should produce identical results to separate sin() and cos()
    for x_str in &["0.5", "1.5", "2.718", "0.001"] {
        let (sin_fused, cos_fused) = evaluate_sincos(&gmath(x_str)).unwrap();
        let sin_separate = evaluate(&gmath(x_str).sin()).unwrap();
        let cos_separate = evaluate(&gmath(x_str).cos()).unwrap();

        let sf = sv_to_fp(&sin_fused);
        let cf = sv_to_fp(&cos_fused);
        let ss = sv_to_fp(&sin_separate);
        let cs = sv_to_fp(&cos_separate);

        assert_fp(sf, ss, tight(), &format!("sincos vs sin({x_str})"));
        assert_fp(cf, cs, tight(), &format!("sincos vs cos({x_str})"));
    }
}

#[test]
fn test_sincos_large_angle() {
    // Large angle: range reduction should handle this
    let (sin_val, cos_val) = evaluate_sincos(&gmath("100")).unwrap();
    let sin_fp = sv_to_fp(&sin_val);
    let cos_fp = sv_to_fp(&cos_val);
    // Pythagorean identity should still hold
    let sum_sq = sin_fp * sin_fp + cos_fp * cos_fp;
    assert_fp(sum_sq, fp("1"), tight(), "sin²(100) + cos²(100) = 1");
}

#[test]
fn test_sincos_with_expression() {
    // evaluate_sincos of a compound expression: sin(1+2), cos(1+2)
    let expr = gmath("1") + gmath("2");
    let (sin_val, cos_val) = evaluate_sincos(&expr).unwrap();
    let sin3 = evaluate(&gmath("3").sin()).unwrap();
    let cos3 = evaluate(&gmath("3").cos()).unwrap();
    assert_fp(sv_to_fp(&sin_val), sv_to_fp(&sin3), tight(), "sincos(1+2) vs sin(3)");
    assert_fp(sv_to_fp(&cos_val), sv_to_fp(&cos3), tight(), "sincos(1+2) vs cos(3)");
}

// ============================================================================
// Identity short-circuits
// ============================================================================

#[test]
fn test_exp_ln_identity() {
    // exp(ln(x)) = x for positive x
    for x_str in &["0.5", "1", "2", "3", "10", "0.1", "0.01", "100"] {
        let expr = gmath(x_str).ln().exp();
        let result = evaluate(&expr).unwrap();
        let expected = evaluate(&gmath(x_str)).unwrap();
        let r = sv_to_fp(&result);
        let e = sv_to_fp(&expected);
        assert_fp(r, e, tight(), &format!("exp(ln({x_str})) = {x_str}"));
    }
}

#[test]
fn test_ln_exp_identity() {
    // ln(exp(x)) = x
    for x_str in &["0.5", "1", "2", "3", "0.1", "0.01", "-1", "-2"] {
        let expr = gmath(x_str).exp().ln();
        let result = evaluate(&expr).unwrap();
        let expected = evaluate(&gmath(x_str)).unwrap();
        let r = sv_to_fp(&result);
        let e = sv_to_fp(&expected);
        assert_fp(r, e, tight(), &format!("ln(exp({x_str})) = {x_str}"));
    }
}

#[test]
fn test_exp_ln_preserves_chain() {
    // exp(ln(x)) + 1 — the short-circuit should work inside larger expressions
    let expr = gmath("2").ln().exp() + gmath("1");
    let result = evaluate(&expr).unwrap();
    let expected = evaluate(&(gmath("2") + gmath("1"))).unwrap();
    let r = sv_to_fp(&result);
    let e = sv_to_fp(&expected);
    assert_fp(r, e, tight(), "exp(ln(2)) + 1 = 3");
}

#[test]
fn test_identity_does_not_fire_on_different_inner() {
    // exp(ln(x) + y) should NOT short-circuit to x
    // This should compute normally: exp(ln(2) + ln(3)) = exp(ln(6)) = 6
    let expr = (gmath("2").ln() + gmath("3").ln()).exp();
    let result = evaluate(&expr).unwrap();
    let r = sv_to_fp(&result);
    // exp(ln(2) + ln(3)) = exp(ln(6)) = 6
    assert_fp(r, fp("6"), fp("0.001"), "exp(ln(2)+ln(3)) = 6");
}
