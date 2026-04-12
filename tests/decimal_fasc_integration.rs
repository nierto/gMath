//! Decimal FASC integration tests — verify `gmath("0.1").exp()` routes to decimal engine.
//!
//! These tests exercise the full FASC pipeline:
//!   `gmath("0.1") → LazyExpr::Literal → StackEvaluator → parse_literal → Decimal →
//!    evaluate_exp → try_decimal_compute → decimal_exp → DecimalCompute → Display`
//!
//! The goal: prove that decimal-domain inputs flow through the native decimal
//! transcendental engines and emerge with the decimal precision guarantee.
//!
//! # Running
//!
//! ```bash
//! GMATH_PROFILE=embedded cargo test --test decimal_fasc_integration -- --nocapture
//! ```

#![cfg(table_format = "q64_64")]

use g_math::fixed_point::canonical::{gmath, evaluate};
use g_math::fixed_point::universal::fasc::stack_evaluator::StackValue;

// ════════════════════════════════════════════════════════════════════
// Decimal dispatch verification
// ════════════════════════════════════════════════════════════════════

/// Verify that `gmath("0.1").exp()` returns a `DecimalCompute` variant.
///
/// Before FASC integration, decimal inputs went through binary conversion first,
/// losing their decimal identity. After integration, the decimal engine handles
/// them natively and preserves decimal-domain semantics through chain persistence.
#[test]
fn exp_of_decimal_returns_decimal_compute() {
    let result = evaluate(&gmath("0.1").exp()).expect("exp(0.1) should not error");
    match &result {
        StackValue::DecimalCompute(..) => {}
        other => panic!("Expected DecimalCompute variant, got {:?}", other.domain_type()),
    }
}

#[test]
fn ln_of_decimal_returns_decimal_compute() {
    let result = evaluate(&gmath("2.0").ln()).expect("ln(2.0) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "Expected DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn sqrt_of_decimal_returns_decimal_compute() {
    let result = evaluate(&gmath("2.0").sqrt()).expect("sqrt(2.0) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "Expected DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn sin_of_decimal_returns_decimal_compute() {
    let result = evaluate(&gmath("0.5").sin()).expect("sin(0.5) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "Expected DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn cos_of_decimal_returns_decimal_compute() {
    let result = evaluate(&gmath("0.5").cos()).expect("cos(0.5) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "Expected DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn atan_of_decimal_returns_decimal_compute() {
    let result = evaluate(&gmath("1.0").atan()).expect("atan(1.0) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "Expected DecimalCompute, got {:?}", result.domain_type());
}

// ════════════════════════════════════════════════════════════════════
// Composed transcendentals — should work for free via FASC composition
// ════════════════════════════════════════════════════════════════════

#[test]
fn tan_of_decimal_returns_decimal_compute() {
    // tan(x) = sin(x) / cos(x) — composed. Both sides decimal → divide stays decimal.
    let result = evaluate(&gmath("0.5").tan()).expect("tan(0.5) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "tan composition should return DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn sinh_of_decimal_returns_decimal_compute() {
    // sinh(x) = (exp(x) - exp(-x)) / 2 — composed from exp.
    let result = evaluate(&gmath("0.5").sinh()).expect("sinh(0.5) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "sinh composition should return DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn cosh_of_decimal_returns_decimal_compute() {
    let result = evaluate(&gmath("0.5").cosh()).expect("cosh(0.5) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "cosh composition should return DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn tanh_of_decimal_returns_decimal_compute() {
    let result = evaluate(&gmath("0.5").tanh()).expect("tanh(0.5) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "tanh composition should return DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn asinh_of_decimal_returns_decimal_compute() {
    let result = evaluate(&gmath("1.0").asinh()).expect("asinh(1.0) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "asinh composition should return DecimalCompute, got {:?}", result.domain_type());
}

// ════════════════════════════════════════════════════════════════════
// Chain persistence — multi-transcendental chains stay decimal throughout
// ════════════════════════════════════════════════════════════════════

#[test]
fn sin_of_ln_of_decimal_chain() {
    // sin(ln(2.0)) — two transcendentals chained. Should stay DecimalCompute.
    let result = evaluate(&gmath("2.0").ln().sin()).expect("sin(ln(2)) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "sin(ln(x)) chain should stay DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn exp_of_sin_of_decimal_chain() {
    let result = evaluate(&gmath("0.5").sin().exp()).expect("exp(sin(0.5)) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "exp(sin(x)) chain should stay DecimalCompute, got {:?}", result.domain_type());
}

#[test]
fn triple_chain_stays_decimal() {
    // sqrt(exp(ln(0.5))) — three chained transcendentals
    let result = evaluate(&gmath("0.5").ln().exp().sqrt())
        .expect("sqrt(exp(ln(0.5))) should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "3-chain should stay DecimalCompute, got {:?}", result.domain_type());
}

// ════════════════════════════════════════════════════════════════════
// Arithmetic mixing with DecimalCompute
// ════════════════════════════════════════════════════════════════════

#[test]
fn decimal_plus_decimal_compute_stays_decimal() {
    // exp(0.5) is DecimalCompute, 0.1 is Decimal. Add → DecimalCompute.
    let a = gmath("0.5").exp();
    let b = gmath("0.1");
    let result = evaluate(&(a + b)).expect("addition should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "DecimalCompute + Decimal should stay DecimalCompute");
}

#[test]
fn decimal_compute_times_decimal_stays_decimal() {
    let a = gmath("0.5").sin();
    let b = gmath("2.0");
    let result = evaluate(&(a * b)).expect("multiply should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "DecimalCompute * Decimal should stay DecimalCompute");
}

#[test]
fn decimal_compute_divided_by_decimal_stays_decimal() {
    let a = gmath("0.5").cos();
    let b = gmath("2.0");
    let result = evaluate(&(a / b)).expect("divide should not error");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "DecimalCompute / Decimal should stay DecimalCompute");
}

// ════════════════════════════════════════════════════════════════════
// Value correctness — results match mpmath within storage ULP
// ════════════════════════════════════════════════════════════════════

/// Compare Display output of decimal transcendental against expected string prefix.
///
/// We check up to 15 significant digits (well within the compute-tier precision
/// absorbed into storage dp).
fn assert_decimal_result_matches(result: StackValue, expected_prefix: &str, label: &str) {
    let s = format!("{:.20}", result);
    // Strip trailing zeros/garbage for comparison; check that expected_prefix is contained
    let clean = s.trim_end_matches('0').trim_end_matches('.');
    assert!(
        clean.starts_with(expected_prefix) || s.starts_with(expected_prefix),
        "{}: expected result starting with '{}', got '{}'",
        label, expected_prefix, s
    );
}

#[test]
fn exp_1_matches_e_to_15_digits() {
    // e = 2.71828182845904523536...
    let result = evaluate(&gmath("1.0").exp()).expect("exp(1) failed");
    assert_decimal_result_matches(result, "2.718281828459045", "exp(1)");
}

#[test]
fn ln_2_matches_mpmath_to_15_digits() {
    // ln(2) = 0.69314718055994530941...
    let result = evaluate(&gmath("2.0").ln()).expect("ln(2) failed");
    assert_decimal_result_matches(result, "0.693147180559945", "ln(2)");
}

#[test]
fn sqrt_2_matches_mpmath_to_15_digits() {
    // sqrt(2) = 1.41421356237309504880...
    let result = evaluate(&gmath("2.0").sqrt()).expect("sqrt(2) failed");
    assert_decimal_result_matches(result, "1.414213562373095", "sqrt(2)");
}

#[test]
fn sin_1_matches_mpmath_to_15_digits() {
    // sin(1) = 0.84147098480789650665...
    let result = evaluate(&gmath("1.0").sin()).expect("sin(1) failed");
    assert_decimal_result_matches(result, "0.841470984807896", "sin(1)");
}

#[test]
fn cos_1_matches_mpmath_to_15_digits() {
    // cos(1) = 0.54030230586813971740...
    let result = evaluate(&gmath("1.0").cos()).expect("cos(1) failed");
    assert_decimal_result_matches(result, "0.540302305868139", "cos(1)");
}

#[test]
fn atan_1_matches_pi_over_4_to_15_digits() {
    // atan(1) = π/4 = 0.78539816339744830961...
    let result = evaluate(&gmath("1.0").atan()).expect("atan(1) failed");
    assert_decimal_result_matches(result, "0.785398163397448", "atan(1)");
}

// ════════════════════════════════════════════════════════════════════
// Financial formula — compound interest at exact decimal rates
// ════════════════════════════════════════════════════════════════════

#[test]
fn compound_interest_monthly_matches() {
    // A = P × (1 + r/n)^(nt)
    // P=1000, r=0.05, n=12, t=1 → A = 1000 × (1.00416666...)^12
    // Expected: A ≈ 1051.161897881733...
    //
    // Using ln/exp: A = P × exp(nt × ln(1 + r/n))
    let _p = gmath("1000.0");
    let r_over_n = gmath("0.00416666666666666666");  // 0.05/12 truncated
    let one_plus_r_over_n = gmath("1.0") + r_over_n;
    // Use "12.0" (decimal) so nt stays in decimal domain, preserving DecimalCompute chain
    let nt = gmath("12.0");
    let ln_factor = one_plus_r_over_n.ln();
    let nt_times_ln = nt * ln_factor;
    let growth = nt_times_ln.exp();
    // growth ≈ 1.051161897881733...
    let result = evaluate(&growth).expect("compound interest failed");
    assert!(matches!(result, StackValue::DecimalCompute(..)),
        "compound interest chain should stay DecimalCompute");

    let s = format!("{:.15}", result);
    // Allow small error in the last few digits since 0.05/12 was truncated at dp=20
    assert!(
        s.starts_with("1.05116189788") || s.starts_with("1.0511618978817"),
        "compound interest growth: got '{}'", s
    );
}
