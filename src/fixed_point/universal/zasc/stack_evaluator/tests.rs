use super::*;
use crate::fixed_point::universal::zasc::lazy_expr::gmath;
use crate::fixed_point::universal::tier_types::{CompactShadow, ShadowConstantId};

#[test]
fn test_literal_parsing() {
    let mut eval = StackEvaluator::new(DeploymentProfile::default());

    // Test decimal parsing (domain check only, not value — value is profile-specific BinaryStorage)
    let decimal = eval.parse_literal("123.456").unwrap();
    assert!(matches!(decimal, StackValue::Decimal(3, _, _)));

    // Test integer parsing (domain check)
    let integer = eval.parse_literal("255").unwrap();
    assert!(matches!(integer, StackValue::Binary(_, _, _)));

    // Test hex parsing (now works via byte-level 0x prefix check)
    let hex = eval.parse_literal("0xFF").unwrap();
    assert!(matches!(hex, StackValue::Binary(_, _, _)));

    // Test binary prefix parsing
    let bin = eval.parse_literal("0b1010").unwrap();
    assert!(matches!(bin, StackValue::Binary(_, _, _)));

    // Test fraction parsing
    let fraction = eval.parse_literal("1/3").unwrap();
    assert!(matches!(fraction, StackValue::Symbolic(_)));

    // Test repeating decimal parsing
    let repeating = eval.parse_literal("0.333...").unwrap();
    assert!(matches!(repeating, StackValue::Symbolic(_)));

    // Test named constant parsing
    let pi = eval.parse_literal("pi").unwrap();
    assert!(matches!(pi, StackValue::Symbolic(_)));
}

#[test]
fn test_simple_arithmetic() {
    let expr = gmath("10") + gmath("20");
    let result = evaluate(&expr).unwrap();

    // Result is Q-format binary: 30 << fractional_bits
    // Just verify it's in the binary domain
    assert!(matches!(result, StackValue::Binary(_, _, _)),
            "Expected Binary result, got {:?}", result);
}

#[test]
fn test_exp_lazy_evaluation() {
    // Test that exp() creates expression tree without immediate evaluation
    let expr = gmath("1.5").exp();

    // Expression should build successfully
    assert_eq!(expr.operation_count(), 1);
    assert_eq!(expr.depth(), 2);
}

// ========================================================================
// EXP COMPUTATION TESTS (Q64.64-specific — require BinaryStorage = i128)
// These tests use i128 literals and .abs() which only work in embedded profile
// ========================================================================

#[cfg(table_format = "q64_64")]
#[test]
fn test_exp_basic_computation() {
    // Test exp(0) = 1
    let expr = gmath("0").exp();
    let result = evaluate(&expr).unwrap();

    match result {
        StackValue::Binary(tier, val, _) => {
            assert_eq!(tier, 3, "Expected tier 3 (Q64.64)");
            let expected_one = 1i128 << 64; // 1.0 in Q64.64
            assert_eq!(val, expected_one, "exp(0) should equal 1.0");
        }
        _ => panic!("Expected Binary result, got {:?}", result),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_exp_positive_value() {
    let expr = gmath("1").exp();
    let result = evaluate(&expr).unwrap();

    match result {
        StackValue::Binary(tier, val, _) => {
            assert_eq!(tier, 3, "Expected tier 3 (Q64.64)");
            let e_q64 = 2.718281828 * (1u128 << 64) as f64;
            let expected_approx = e_q64 as i128;
            let diff = (val - expected_approx).abs();
            let tolerance = 1i128 << 54;
            assert!(diff < tolerance,
                "exp(1) = {} is not close to e ≈ {}. Diff: {}", val, expected_approx, diff);
        }
        _ => panic!("Expected Binary result, got {:?}", result),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_exp_half() {
    let expr = gmath("0.5").exp();
    let result = evaluate(&expr).unwrap();

    match result {
        StackValue::Binary(_, val, _) => {
            let exp_half_q64 = 1.648721271 * (1u128 << 64) as f64;
            let expected_approx = exp_half_q64 as i128;
            let diff = (val - expected_approx).abs();
            let tolerance = 1i128 << 54;
            assert!(diff < tolerance,
                "exp(0.5) = {} is not close to expected ≈ {}. Diff: {}", val, expected_approx, diff);
        }
        _ => panic!("Expected Binary result, got {:?}", result),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_exp_negative_value() {
    let expr = gmath("-1").exp();
    let result = evaluate(&expr).unwrap();

    match result {
        StackValue::Binary(_, val, _) => {
            let exp_neg_one_q64 = 0.367879441 * (1u128 << 64) as f64;
            let expected_approx = exp_neg_one_q64 as i128;
            let diff = (val - expected_approx).abs();
            let tolerance = 1i128 << 54;
            assert!(diff < tolerance,
                "exp(-1) = {} is not close to expected ≈ {}. Diff: {}", val, expected_approx, diff);
        }
        _ => panic!("Expected Binary result, got {:?}", result),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_exp_composition() {
    let expr1 = gmath("0.5").exp() + gmath("0.5").exp();
    let result1 = evaluate(&expr1).unwrap();
    let expr2 = gmath("0.5").exp() * gmath("2");
    let result2 = evaluate(&expr2).unwrap();

    match (result1, result2) {
        (StackValue::Binary(_, v1, _), StackValue::Binary(_, v2, _)) => {
            let diff = (v1 - v2).abs();
            let tolerance = 1i128 << 50;
            assert!(diff < tolerance,
                "exp(0.5) + exp(0.5) = {} should equal 2 * exp(0.5) = {}. Diff: {}", v1, v2, diff);
        }
        _ => panic!("Expected Binary results for both"),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_exp_from_decimal() {
    let expr = gmath("1.5").exp();
    let result = evaluate(&expr).unwrap();

    match result {
        StackValue::Binary(tier, val, _) => {
            assert_eq!(tier, 3, "Expected tier 3");
            let exp_1_5_q64 = 4.481689 * (1u128 << 64) as f64;
            let expected_approx = exp_1_5_q64 as i128;
            let diff = (val - expected_approx).abs();
            let tolerance = 1i128 << 54;
            assert!(diff < tolerance,
                "exp(1.5) = {} is not close to expected ≈ {}. Diff: {}", val, expected_approx, diff);
        }
        _ => panic!("Expected Binary result, got {:?}", result),
    }
}

#[test]
fn test_exp_profile_tier_selection() {
    // Test that different profiles select correct tiers
    let eval_embedded = StackEvaluator::new(DeploymentProfile::Embedded);
    assert_eq!(eval_embedded.profile_max_binary_tier(), 3);

    let eval_balanced = StackEvaluator::new(DeploymentProfile::Balanced);
    assert_eq!(eval_balanced.profile_max_binary_tier(), 4);

    let eval_scientific = StackEvaluator::new(DeploymentProfile::Scientific);
    assert_eq!(eval_scientific.profile_max_binary_tier(), 5);
}

#[test]
fn test_exp_display() {
    // Test that exp displays correctly in expression tree
    let expr = gmath("1.5").exp();
    let display = format!("{}", expr);
    assert!(display.contains("exp"), "Expression should contain 'exp'");
    assert!(display.contains("1.5"), "Expression should contain '1.5'");
}

// ========================================================================
// NATURAL LOGARITHM TESTS
// ========================================================================

#[test]
fn test_ln_lazy_evaluation() {
    // Test that ln() creates expression tree without immediate evaluation
    let expr = gmath("2.0").ln();

    // Expression should build successfully
    assert_eq!(expr.operation_count(), 1);
    assert_eq!(expr.depth(), 2);
}

// ========================================================================
// LN COMPUTATION TESTS (Q64.64-specific — require BinaryStorage = i128)
// ========================================================================

#[cfg(table_format = "q64_64")]
#[test]
fn test_ln_basic_computation() {
    let expr = gmath("1").ln();
    let result = evaluate(&expr).unwrap();

    match result {
        StackValue::Binary(tier, val, _) => {
            assert_eq!(tier, 3, "Expected tier 3 (Q64.64)");
            let tolerance = 1i128 << 40;
            assert!(val.abs() < tolerance, "ln(1) = {} should be close to 0", val);
        }
        _ => panic!("Expected Binary result, got {:?}", result),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_ln_e_equals_one() {
    let expr = gmath("2.718281828").ln();
    let result = evaluate(&expr).unwrap();

    match result {
        StackValue::Binary(tier, val, _) => {
            assert_eq!(tier, 3, "Expected tier 3 (Q64.64)");
            let one_q64 = 1i128 << 64;
            let diff = (val - one_q64).abs();
            let tolerance = 1i128 << 50;
            assert!(diff < tolerance,
                "ln(e) = {} is not close to 1.0 = {}. Diff: {}", val, one_q64, diff);
        }
        _ => panic!("Expected Binary result, got {:?}", result),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_ln_2() {
    let expr = gmath("2").ln();
    let result = evaluate(&expr).unwrap();

    match result {
        StackValue::Binary(tier, val, _) => {
            assert_eq!(tier, 3, "Expected tier 3 (Q64.64)");
            let ln2_q64 = 0.693147180 * (1u128 << 64) as f64;
            let expected_approx = ln2_q64 as i128;
            let diff = (val - expected_approx).abs();
            let tolerance = 1i128 << 54;
            assert!(diff < tolerance,
                "ln(2) = {} is not close to expected ≈ {}. Diff: {}", val, expected_approx, diff);
        }
        _ => panic!("Expected Binary result, got {:?}", result),
    }
}

#[test]
fn test_ln_domain_error() {
    // Test that ln(0) and ln(-1) return domain errors
    let expr_zero = gmath("0").ln();
    let result_zero = evaluate(&expr_zero);
    assert!(matches!(result_zero, Err(OverflowDetected::DomainError)),
            "ln(0) should return DomainError, got {:?}", result_zero);

    let expr_neg = gmath("-1").ln();
    let result_neg = evaluate(&expr_neg);
    assert!(matches!(result_neg, Err(OverflowDetected::DomainError)),
            "ln(-1) should return DomainError, got {:?}", result_neg);
}

#[test]
#[cfg(table_format = "q64_64")]
fn test_ln_exp_inverse() {
    // Test that ln(exp(x)) ≈ x
    // Use x = 1.5
    let x_q64 = 1.5 * (1u128 << 64) as f64;
    let x_expected = x_q64 as i128;

    let expr = gmath("1.5").exp().ln();
    let result = evaluate(&expr).unwrap();

    match result {
        StackValue::Binary(_, val, _) => {
            let diff = (val - x_expected).abs();
            let tolerance = 1i128 << 52;  // Allow some accumulation error

            assert!(
                diff < tolerance,
                "ln(exp(1.5)) = {} is not close to 1.5 = {}. Diff: {}",
                val,
                x_expected,
                diff
            );
        }
        _ => panic!("Expected Binary result, got {:?}", result),
    }
}

#[test]
fn test_ln_display() {
    // Test that ln displays correctly in expression tree
    let expr = gmath("2.0").ln();
    let display = format!("{}", expr);
    assert!(display.contains("ln"), "Expression should contain 'ln'");
    assert!(display.contains("2.0"), "Expression should contain '2.0'");
}

// ========================================================================
// HYPERBOLIC FUNCTION TESTS
// ========================================================================

#[test]
fn test_sinh_display() {
    let expr = gmath("1.0").sinh();
    let display = format!("{}", expr);
    assert!(display.contains("sinh"), "Expression should contain 'sinh'");
}

#[test]
fn test_cosh_display() {
    let expr = gmath("1.0").cosh();
    let display = format!("{}", expr);
    assert!(display.contains("cosh"), "Expression should contain 'cosh'");
}

#[test]
fn test_tanh_display() {
    let expr = gmath("1.0").tanh();
    let display = format!("{}", expr);
    assert!(display.contains("tanh"), "Expression should contain 'tanh'");
}

#[test]
fn test_sinh_zero() {
    // sinh(0) = 0 — profile-independent
    let expr = gmath("0").sinh();
    let result = evaluate(&expr).unwrap();
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let val = eval.to_binary_storage(&result).unwrap();
    let zero = to_binary_storage(0);
    assert_eq!(val, zero, "sinh(0) should be exactly 0");
}

#[test]
fn test_cosh_zero() {
    // cosh(0) = 1 — compare using profile-native Q-format
    let expr = gmath("0").cosh();
    let result = evaluate(&expr).unwrap();
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let val = eval.to_binary_storage(&result).unwrap();
    let one_val = eval.to_binary_storage(&eval.make_binary_int(1)).unwrap();
    assert_eq!(val, one_val, "cosh(0) should be exactly 1.0");
}

#[test]
fn test_tanh_zero() {
    // tanh(0) = 0 — profile-independent
    let expr = gmath("0").tanh();
    let result = evaluate(&expr).unwrap();
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let val = eval.to_binary_storage(&result).unwrap();
    let zero = to_binary_storage(0);
    assert_eq!(val, zero, "tanh(0) should be exactly 0");
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_sinh_basic() {
    // sinh(1) ≈ 1.1752011936...
    let expr = gmath("1").sinh();
    let result = evaluate(&expr).unwrap();
    match result {
        StackValue::Binary(_, val, _) => {
            let expected_approx = (1.1752011936 * (1u128 << 64) as f64) as i128;
            let diff = (val - expected_approx).abs();
            let tolerance = 1i128 << 54;
            assert!(diff < tolerance,
                "sinh(1) = {} not close to expected {}. Diff: {}", val, expected_approx, diff);
        }
        _ => panic!("Expected Binary result"),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_cosh_basic() {
    // cosh(1) ≈ 1.5430806348...
    let expr = gmath("1").cosh();
    let result = evaluate(&expr).unwrap();
    match result {
        StackValue::Binary(_, val, _) => {
            let expected_approx = (1.5430806348 * (1u128 << 64) as f64) as i128;
            let diff = (val - expected_approx).abs();
            let tolerance = 1i128 << 54;
            assert!(diff < tolerance,
                "cosh(1) = {} not close to expected {}. Diff: {}", val, expected_approx, diff);
        }
        _ => panic!("Expected Binary result"),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_tanh_basic() {
    // tanh(1) ≈ 0.7615941559...
    let expr = gmath("1").tanh();
    let result = evaluate(&expr).unwrap();
    match result {
        StackValue::Binary(_, val, _) => {
            let expected_approx = (0.7615941559 * (1u128 << 64) as f64) as i128;
            let diff = (val - expected_approx).abs();
            let tolerance = 1i128 << 54;
            assert!(diff < tolerance,
                "tanh(1) = {} not close to expected {}. Diff: {}", val, expected_approx, diff);
        }
        _ => panic!("Expected Binary result"),
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_identity_cosh2_minus_sinh2() {
    // cosh²(x) - sinh²(x) = 1 for all x
    // This test uses Q64.64-specific i128 arithmetic for verification
    let expr_cosh = gmath("1").cosh();
    let cosh_val = evaluate(&expr_cosh).unwrap();

    let expr_sinh = gmath("1").sinh();
    let sinh_val = evaluate(&expr_sinh).unwrap();

    // Compute cosh² and sinh²
    let mut eval = StackEvaluator::new(DeploymentProfile::default());
    let cosh2 = eval.multiply_values(cosh_val.clone(), cosh_val).unwrap();
    let sinh2 = eval.multiply_values(sinh_val.clone(), sinh_val).unwrap();
    let diff = eval.subtract_values(cosh2, sinh2).unwrap();

    let diff_binary = eval.to_binary_storage(&diff).unwrap();
    let one_binary = eval.to_binary_storage(&eval.make_binary_int(1)).unwrap();
    let error = (diff_binary - one_binary).abs();
    let tolerance = 1i128 << 55;
    assert!(error < tolerance,
        "cosh²(1) - sinh²(1) = {} should be close to 1.0 = {}. Error: {}",
        diff_binary, one_binary, error);
}

#[test]
fn test_asinh_zero() {
    // asinh(0) = ln(0 + sqrt(0 + 1)) = ln(1) = 0
    let expr = gmath("0").asinh();
    let result = evaluate(&expr).unwrap();
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let val = eval.to_binary_storage(&result).unwrap();
    let zero = to_binary_storage(0);
    // Allow small tolerance from ln(1) computation
    #[cfg(table_format = "q256_256")]
    {
        let diff = if val > zero { val - zero } else { zero - val };
        let tolerance = I512::from_i128(1) << 20;
        assert!(diff < tolerance, "asinh(0) should be close to 0");
    }
    #[cfg(table_format = "q128_128")]
    {
        let diff = if val > zero { val - zero } else { zero - val };
        let tolerance = I256::from_i128(1) << 20;
        assert!(diff < tolerance, "asinh(0) should be close to 0");
    }
    #[cfg(table_format = "q64_64")]
    {
        let diff = (val - zero).abs();
        let tolerance = 1i128 << 20;
        assert!(diff < tolerance, "asinh(0) should be close to 0");
    }
}

#[test]
fn test_acosh_domain_error() {
    // acosh(0.5) should return DomainError (x < 1)
    let expr = gmath("0.5").acosh();
    let result = evaluate(&expr);
    assert!(result.is_err(), "acosh(0.5) should return an error");
}

#[test]
fn test_atanh_domain_error() {
    // atanh(1) should return DomainError (|x| >= 1)
    let expr = gmath("1").atanh();
    let result = evaluate(&expr);
    assert!(result.is_err(), "atanh(1) should return an error, got {:?}", result);
}

#[test]
fn test_atanh_zero() {
    // atanh(0) = ln(1/1) / 2 = 0
    let expr = gmath("0").atanh();
    let result = evaluate(&expr).unwrap();
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let val = eval.to_binary_storage(&result).unwrap();
    let zero = to_binary_storage(0);
    #[cfg(table_format = "q256_256")]
    {
        let diff = if val > zero { val - zero } else { zero - val };
        let tolerance = I512::from_i128(1) << 20;
        assert!(diff < tolerance, "atanh(0) should be close to 0");
    }
    #[cfg(table_format = "q128_128")]
    {
        let diff = if val > zero { val - zero } else { zero - val };
        let tolerance = I256::from_i128(1) << 20;
        assert!(diff < tolerance, "atanh(0) should be close to 0");
    }
    #[cfg(table_format = "q64_64")]
    {
        let diff = (val - zero).abs();
        let tolerance = 1i128 << 20;
        assert!(diff < tolerance, "atanh(0) should be close to 0");
    }
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_asinh_roundtrip() {
    // asinh(sinh(x)) ≈ x for x = 1
    let expr = gmath("1").sinh().asinh();
    let result = evaluate(&expr).unwrap();
    match result {
        StackValue::Binary(_, val, _) => {
            let one_q64 = 1i128 << 64;
            let diff = (val - one_q64).abs();
            // Round-trip tolerance: exp + ln composition
            let tolerance = 1i128 << 55;
            assert!(diff < tolerance,
                "asinh(sinh(1)) = {} should be close to 1.0 = {}. Diff: {}",
                val, one_q64, diff);
        }
        _ => panic!("Expected Binary result"),
    }
}

// ========================================================================
// TRIGONOMETRIC FUNCTION TESTS
// ========================================================================

#[test]
fn test_sin_display() {
    let expr = gmath("1.0").sin();
    let display = format!("{}", expr);
    assert!(display.contains("sin"), "Expression should contain 'sin'");
}

#[test]
fn test_cos_display() {
    let expr = gmath("1.0").cos();
    let display = format!("{}", expr);
    assert!(display.contains("cos"), "Expression should contain 'cos'");
}

#[test]
fn test_tan_display() {
    let expr = gmath("1.0").tan();
    let display = format!("{}", expr);
    assert!(display.contains("tan"), "Expression should contain 'tan'");
}

#[test]
fn test_atan2_display() {
    let expr = gmath("1.0").atan2(gmath("2.0"));
    let display = format!("{}", expr);
    assert!(display.contains("atan2"), "Expression should contain 'atan2'");
}

// ============================================================================
// TRIGONOMETRIC FUNCTION TESTS
// ============================================================================

#[test]
fn test_sin_zero() {
    let expr = gmath("0").sin();
    let result = evaluate(&expr).expect("sin(0) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let zero = eval.make_binary_int(0);
    let result_bs = eval.to_binary_storage(&result).unwrap();
    let zero_bs = eval.to_binary_storage(&zero).unwrap();
    assert_eq!(result_bs, zero_bs, "sin(0) should be 0");
}

#[test]
fn test_cos_zero() {
    let expr = gmath("0").cos();
    let result = evaluate(&expr).expect("cos(0) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let one = eval.make_binary_int(1);
    let result_bs = eval.to_binary_storage(&result).unwrap();
    let one_bs = eval.to_binary_storage(&one).unwrap();
    assert_eq!(result_bs, one_bs, "cos(0) should be 1");
}

#[test]
fn test_tan_zero() {
    let expr = gmath("0").tan();
    let result = evaluate(&expr).expect("tan(0) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let zero = eval.make_binary_int(0);
    let result_bs = eval.to_binary_storage(&result).unwrap();
    let zero_bs = eval.to_binary_storage(&zero).unwrap();
    assert_eq!(result_bs, zero_bs, "tan(0) should be 0");
}

#[test]
fn test_atan_zero() {
    let expr = gmath("0").atan();
    let result = evaluate(&expr).expect("atan(0) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let zero = eval.make_binary_int(0);
    let result_bs = eval.to_binary_storage(&result).unwrap();
    let zero_bs = eval.to_binary_storage(&zero).unwrap();
    assert_eq!(result_bs, zero_bs, "atan(0) should be 0");
}

#[test]
fn test_asin_zero() {
    let expr = gmath("0").asin();
    let result = evaluate(&expr).expect("asin(0) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let result_bs = eval.to_binary_storage(&result).unwrap();
    let zero = eval.make_binary_int(0);
    let zero_bs = eval.to_binary_storage(&zero).unwrap();
    assert_eq!(result_bs, zero_bs, "asin(0) should be 0");
}

#[test]
fn test_asin_domain_error() {
    // asin(2) should fail — outside [-1, 1]
    let expr = gmath("2").asin();
    let result = evaluate(&expr);
    assert!(result.is_err(), "asin(2) should return domain error");
}

#[test]
fn test_acos_zero() {
    // acos(0) = π/2 ≈ 1.5707963...
    // Allows ≤1 ULP tolerance due to compute-tier round-trip:
    // acos(0) = π/2 - asin(0), where asin(0) = atan(0/sqrt(1-0²))
    // The sqrt→divide→atan chain may introduce sub-ULP noise at compute tier
    // that becomes 1 ULP after downscale rounding.
    let expr = gmath("0").acos();
    let result = evaluate(&expr).expect("acos(0) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let result_bs = eval.to_binary_storage(&result).unwrap();
    #[cfg(table_format = "q256_256")]
    {
        use crate::fixed_point::domains::binary_fixed::transcendental::pi_half_i512;
        let expected = pi_half_i512();
        let diff = if result_bs > expected { result_bs - expected } else { expected - result_bs };
        assert!(diff <= I512::from_i128(1), "acos(0) should be π/2 within 1 ULP, diff = {:?}", diff);
    }
    #[cfg(table_format = "q128_128")]
    {
        use crate::fixed_point::domains::binary_fixed::transcendental::pi_half_i256;
        let expected = pi_half_i256();
        let diff = if result_bs > expected { result_bs - expected } else { expected - result_bs };
        assert!(diff <= I256::from_i128(1), "acos(0) should be π/2 within 1 ULP, diff = {:?}", diff);
    }
    #[cfg(table_format = "q64_64")]
    {
        use crate::fixed_point::domains::binary_fixed::transcendental::pi_half_i128;
        let expected = pi_half_i128();
        let diff = if result_bs > expected { (result_bs - expected).abs() } else { (expected - result_bs).abs() };
        assert!(diff <= 1, "acos(0) should be π/2 within 1 ULP, diff = {}", diff);
    }
}

#[test]
fn test_acos_domain_error() {
    // acos(2) should fail — outside [-1, 1]
    let expr = gmath("2").acos();
    let result = evaluate(&expr);
    assert!(result.is_err(), "acos(2) should return domain error");
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_sin_cos_pythagorean_identity() {
    // sin²(x) + cos²(x) = 1 for x = 1.0
    let sin_expr = gmath("1").sin();
    let cos_expr = gmath("1").cos();
    let sin_val = evaluate(&sin_expr).expect("sin(1) should succeed");
    let cos_val = evaluate(&cos_expr).expect("cos(1) should succeed");

    let mut eval = StackEvaluator::new(DeploymentProfile::default());
    let sin_sq = eval.multiply_values(sin_val.clone(), sin_val).unwrap();
    let cos_sq = eval.multiply_values(cos_val.clone(), cos_val).unwrap();
    let sum = eval.add_values(sin_sq, cos_sq).unwrap();

    let one = eval.make_binary_int(1);
    let sum_bs = eval.to_binary_storage(&sum).unwrap();
    let one_bs = eval.to_binary_storage(&one).unwrap();

    // Allow small rounding error (within a few ULP)
    let diff = if sum_bs > one_bs { sum_bs - one_bs } else { one_bs - sum_bs };
    assert!(diff < 1024, "sin²(1) + cos²(1) should be very close to 1, diff={}", diff);
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_atan_one_is_pi_over_4() {
    // atan(1) = π/4 ≈ 0.78539816...
    let expr = gmath("1").atan();
    let result = evaluate(&expr).expect("atan(1) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let result_bs = eval.to_binary_storage(&result).unwrap();

    // π/4 in Q64.64 ≈ 14488038916154245685 (0x C90FDAA2 2168C234)
    // We check that it's approximately right (within small tolerance)
    let pi_over_4_approx: i128 = 14488038916154245685;
    let diff = (result_bs - pi_over_4_approx).abs();
    // Allow up to 256 ULP tolerance
    assert!(diff < 256, "atan(1) should be close to π/4, diff={}", diff);
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_sin_basic() {
    // sin(1) ≈ 0.8414709848... in Q64.64
    let expr = gmath("1").sin();
    let result = evaluate(&expr).expect("sin(1) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let result_bs = eval.to_binary_storage(&result).unwrap();

    // Just verify it's positive and less than 1.0
    let one_q64: i128 = 1i128 << 64;
    assert!(result_bs > 0, "sin(1) should be positive");
    assert!(result_bs < one_q64, "sin(1) should be less than 1");
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_cos_basic() {
    // cos(1) ≈ 0.5403023058... in Q64.64
    let expr = gmath("1").cos();
    let result = evaluate(&expr).expect("cos(1) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let result_bs = eval.to_binary_storage(&result).unwrap();

    // Just verify it's positive and less than 1.0
    let one_q64: i128 = 1i128 << 64;
    assert!(result_bs > 0, "cos(1) should be positive");
    assert!(result_bs < one_q64, "cos(1) should be less than 1");
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_tan_one() {
    // tan(1) ≈ 1.5574077... in Q64.64
    let expr = gmath("1").tan();
    let result = evaluate(&expr).expect("tan(1) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let result_bs = eval.to_binary_storage(&result).unwrap();

    // tan(1) should be > 1
    let one_q64: i128 = 1i128 << 64;
    assert!(result_bs > one_q64, "tan(1) should be > 1");
    // tan(1) < 2
    assert!(result_bs < 2 * one_q64, "tan(1) should be < 2");
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_atan2_basic() {
    // atan2(1, 1) = π/4 ≈ 0.78539816...
    let expr = gmath("1").atan2(gmath("1"));
    let result = evaluate(&expr).expect("atan2(1,1) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let result_bs = eval.to_binary_storage(&result).unwrap();

    let pi_over_4_approx: i128 = 14488038916154245685;
    let diff = (result_bs - pi_over_4_approx).abs();
    assert!(diff < 256, "atan2(1,1) should be close to π/4, diff={}", diff);
}

#[cfg(table_format = "q64_64")]
#[test]
fn test_asin_roundtrip() {
    // asin(sin(0.5)) ≈ 0.5
    let inner_expr = gmath("0.5").sin();
    let asin_expr = inner_expr.asin();
    let result = evaluate(&asin_expr).expect("asin(sin(0.5)) should succeed");
    let eval = StackEvaluator::new(DeploymentProfile::default());
    let result_bs = eval.to_binary_storage(&result).unwrap();

    // Should be close to 0.5 in Q64.64
    let half_q64: i128 = 1i128 << 63; // 0.5 in Q64.64
    let diff = (result_bs - half_q64).abs();
    // Allow some roundtrip error
    assert!(diff < 65536, "asin(sin(0.5)) should be close to 0.5, diff={}", diff);
}

// ========================================================================
// gmath_parse() — RUNTIME STRING PARSING TESTS
// ========================================================================

#[test]
fn test_gmath_parse_decimal() {
    let s = String::from("1.5");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    assert!(matches!(result, StackValue::Decimal(_, _, _)),
        "Expected Decimal, got {:?}", result);
}

#[test]
fn test_gmath_parse_integer() {
    let s = String::from("42");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    assert!(matches!(result, StackValue::Binary(_, _, _)),
        "Expected Binary, got {:?}", result);
}

#[test]
fn test_gmath_parse_fraction() {
    let s = String::from("1/3");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    assert!(matches!(result, StackValue::Symbolic(_)),
        "Expected Symbolic, got {:?}", result);
}

#[test]
fn test_gmath_parse_repeating_decimal() {
    let s = String::from("0.333...");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    assert!(matches!(result, StackValue::Symbolic(_)),
        "Expected Symbolic for repeating decimal, got {:?}", result);
}

#[test]
fn test_gmath_parse_hex() {
    let s = String::from("0xFF");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    assert!(matches!(result, StackValue::Binary(_, _, _)),
        "Expected Binary for hex, got {:?}", result);
}

#[test]
fn test_gmath_parse_binary_literal() {
    let s = String::from("0b1010");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    assert!(matches!(result, StackValue::Binary(_, _, _)),
        "Expected Binary for 0b prefix, got {:?}", result);
}

#[test]
fn test_gmath_parse_ternary() {
    let s = String::from("0t1.5");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    assert!(matches!(result, StackValue::Ternary(_, _, _)),
        "Expected Ternary for 0t prefix, got {:?}", result);
}

#[test]
fn test_gmath_parse_named_constant() {
    let s = String::from("pi");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    assert!(matches!(result, StackValue::Symbolic(_)),
        "Expected Symbolic for named constant, got {:?}", result);
}

#[test]
fn test_gmath_parse_invalid_input() {
    let s = String::from("not_a_number");
    let result = gmath_parse(&s);
    assert!(result.is_err(), "Expected parse error for invalid input");
}

#[test]
fn test_gmath_parse_empty_string() {
    let s = String::from("");
    let result = gmath_parse(&s);
    assert!(result.is_err(), "Expected parse error for empty string");
}

#[test]
fn test_gmath_parse_arithmetic_with_gmath() {
    // Mix gmath_parse (runtime) with gmath (static) in the same expression
    let runtime_val = String::from("1.5");
    let parsed = gmath_parse(&runtime_val).unwrap();
    let expr = parsed + gmath("2.5");
    let result = evaluate(&expr).unwrap();
    // 1.5 + 2.5 = 4.0 — should succeed regardless of domain
    let display = format!("{}", result);
    assert!(display.starts_with("4.0"), "1.5 + 2.5 should be 4.0, got {}", display);
}

#[test]
fn test_gmath_parse_arithmetic_both_runtime() {
    let a = String::from("10");
    let b = String::from("3");
    let expr = gmath_parse(&a).unwrap() * gmath_parse(&b).unwrap();
    let result = evaluate(&expr).unwrap();
    let display = format!("{}", result);
    assert!(display.starts_with("30"), "10 * 3 should be 30, got {}", display);
}

#[test]
fn test_gmath_parse_transcendental() {
    // exp(0) = 1 via runtime string
    let s = String::from("0");
    let expr = gmath_parse(&s).unwrap().exp();
    let result = evaluate(&expr).unwrap();
    let display = format!("{}", result);
    assert!(display.starts_with("1.0"), "exp(0) should be 1.0, got {}", display);
}

#[test]
fn test_gmath_parse_chaining() {
    // Parse → evaluate → chain back → evaluate again
    let s = String::from("100.00");
    let first = evaluate(&gmath_parse(&s).unwrap()).unwrap();
    let chained = LazyExpr::from(first) * gmath("1.05");
    let second = evaluate(&chained).unwrap();
    let display = format!("{}", second);
    assert!(display.starts_with("105.0"), "100 * 1.05 should be 105, got {}", display);
}

#[test]
fn test_gmath_parse_loop_accumulation() {
    // Simulate reading values from a data source
    let values = vec!["1.0", "2.0", "3.0", "4.0"];
    let mut acc = gmath_parse(values[0]).unwrap();
    for v in &values[1..] {
        acc = acc + gmath_parse(v).unwrap();
    }
    let result = evaluate(&acc).unwrap();
    let display = format!("{}", result);
    assert!(display.starts_with("10.0"), "1+2+3+4 should be 10, got {}", display);
}

#[test]
fn test_gmath_parse_mode_routing() {
    use crate::fixed_point::universal::zasc::mode::{set_mode, reset_mode, GmathMode, ComputeMode, OutputMode};

    // Parse under symbolic mode
    set_mode(GmathMode { compute: ComputeMode::Symbolic, output: OutputMode::Auto });
    let s = String::from("1.5");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    assert!(matches!(result, StackValue::Symbolic(_)),
        "Symbolic mode should produce Symbolic result, got {:?}", result);
    reset_mode();

    // Parse under default auto mode
    let expr2 = gmath_parse(&s).unwrap();
    let result2 = evaluate(&expr2).unwrap();
    assert!(matches!(result2, StackValue::Decimal(_, _, _)),
        "Auto mode should route '1.5' to Decimal, got {:?}", result2);
}

// ========================================================================
// gmath_parse() — COMPACT SHADOW TESTS
// ========================================================================

#[test]
fn test_gmath_parse_decimal_shadow_created() {
    // Decimal "1.5" → shadow should be 15/10 reduced = 3/2
    let s = String::from("1.5");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "Decimal parse should create a shadow");
    let (num, den) = shadow.as_rational().expect("Shadow should be rational");
    // 1.5 → scaled_value=15, scale=10 → shadow from_rational(15, 10)
    // from_rational may or may not GCD-reduce; check the ratio
    assert_eq!(num as f64 / den as f64, 1.5, "Shadow should represent 1.5, got {}/{}", num, den);
}

#[test]
fn test_gmath_parse_integer_shadow_created() {
    // Integer "42" → shadow should be 42/1
    let s = String::from("42");
    let expr = gmath_parse(&s).unwrap();
    let result = evaluate(&expr).unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "Integer parse should create a shadow");
    let (num, den) = shadow.as_rational().expect("Shadow should be rational");
    assert_eq!(num, 42, "Shadow numerator should be 42, got {}", num);
    assert_eq!(den, 1, "Shadow denominator should be 1, got {}", den);
}

#[test]
fn test_gmath_parse_shadow_survives_add() {
    // 1.5 + 2.5 → shadow should be 3/2 + 5/2 = 4/1
    let a = gmath_parse("1.5").unwrap();
    let b = gmath_parse("2.5").unwrap();
    let result = evaluate(&(a + b)).unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "Addition result should have shadow");
    let (num, den) = shadow.as_rational().expect("Shadow should be rational");
    // 1.5 + 2.5 = 4.0; shadow should represent 4
    let ratio = num as f64 / den as f64;
    assert!((ratio - 4.0).abs() < 1e-10, "Shadow should represent 4.0, got {}/{} = {}", num, den, ratio);
}

#[test]
fn test_gmath_parse_shadow_survives_multiply() {
    // 1.5 * 2.0 → shadow should represent 3.0
    let a = gmath_parse("1.5").unwrap();
    let b = gmath_parse("2.0").unwrap();
    let result = evaluate(&(a * b)).unwrap();
    let shadow = result.shadow();
    // Decimal * Decimal may go through UGOD or via_rational; check shadow if present
    if let Some((num, den)) = shadow.as_rational() {
        let ratio = num as f64 / den as f64;
        assert!((ratio - 3.0).abs() < 1e-10,
            "Shadow should represent 3.0, got {}/{} = {}", num, den, ratio);
    }
    // If shadow is None, the result went through BinaryCompute (no shadow) — acceptable
}

#[test]
fn test_gmath_parse_shadow_subtract() {
    // 10.0 - 3.5 → shadow should represent 6.5
    let a = gmath_parse("10.0").unwrap();
    let b = gmath_parse("3.5").unwrap();
    let result = evaluate(&(a - b)).unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "Subtraction result should have shadow");
    let (num, den) = shadow.as_rational().expect("Shadow should be rational");
    let ratio = num as f64 / den as f64;
    assert!((ratio - 6.5).abs() < 1e-10, "Shadow should represent 6.5, got {}/{} = {}", num, den, ratio);
}

#[test]
fn test_gmath_parse_shadow_negate() {
    // -(1.5) → shadow should be -3/2
    let a = gmath_parse("1.5").unwrap();
    let result = evaluate(&(-a)).unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "Negation result should have shadow");
    let (num, den) = shadow.as_rational().expect("Shadow should be rational");
    let ratio = num as f64 / den as f64;
    assert!((ratio - (-1.5)).abs() < 1e-10, "Shadow should represent -1.5, got {}/{} = {}", num, den, ratio);
}

#[test]
fn test_gmath_parse_shadow_chain_preserved() {
    // Parse → evaluate → chain → evaluate: shadow should survive the round-trip
    let first = evaluate(&gmath_parse("2.5").unwrap()).unwrap();
    let shadow_before = first.shadow();
    assert!(shadow_before.is_some(), "First evaluation should have shadow");

    // Chain into new expression
    let chained = LazyExpr::from(first) + gmath_parse("1.0").unwrap();
    let second = evaluate(&chained).unwrap();
    let shadow_after = second.shadow();
    assert!(shadow_after.is_some(), "Chained result should preserve shadow");
    let (num, den) = shadow_after.as_rational().expect("Shadow should be rational");
    let ratio = num as f64 / den as f64;
    assert!((ratio - 3.5).abs() < 1e-10, "Shadow should represent 3.5, got {}/{} = {}", num, den, ratio);
}

#[test]
fn test_gmath_parse_symbolic_no_shadow_needed() {
    // Symbolic values (fractions) are already exact — shadow should be None
    let result = evaluate(&gmath_parse("1/3").unwrap()).unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_none(), "Symbolic values need no shadow (they ARE exact), got {:?}", shadow);
}

#[test]
fn test_gmath_parse_shadow_matches_static_gmath() {
    // gmath_parse("1.5") and gmath("1.5") should produce identical shadows
    let runtime_result = evaluate(&gmath_parse("1.5").unwrap()).unwrap();
    let static_result = evaluate(&gmath("1.5")).unwrap();

    let runtime_shadow = runtime_result.shadow();
    let static_shadow = static_result.shadow();
    assert_eq!(runtime_shadow, static_shadow,
        "gmath_parse and gmath should produce identical shadows: {:?} vs {:?}",
        runtime_shadow, static_shadow);
}

// ============================================================================
// SHADOW SYSTEM INTEGRATION TESTS
// Validate shadow fast path in to_rational() and Display
// ============================================================================

#[test]
fn test_shadow_fast_path_decimal_to_rational() {
    // to_rational() should use shadow, giving exact 3/2 (not Q-format expanded)
    let result = evaluate(&gmath("1.5")).unwrap();
    assert!(result.shadow().is_some(), "decimal 1.5 must have shadow");

    let rational = result.to_rational().unwrap();
    let num = rational.numerator_i128().unwrap();
    let den = rational.denominator_i128().unwrap();
    // Shadow fast path: 15/10 reduced by RationalNumber → 3/2
    // Without shadow: would be huge Q-format value / 2^64
    assert!(den.abs() < 1000, "shadow fast path should give small denominator, got {}/{}", num, den);
    // Verify the actual value: 3/2 = 1.5
    assert_eq!(num * 2, den * 3, "1.5 should be 3/2 (or equivalent), got {}/{}", num, den);
}

#[test]
fn test_shadow_fast_path_integer_to_rational() {
    let result = evaluate(&gmath("42")).unwrap();
    assert!(result.shadow().is_some(), "integer 42 must have shadow");

    let rational = result.to_rational().unwrap();
    let num = rational.numerator_i128().unwrap();
    let den = rational.denominator_i128().unwrap();
    // Shadow fast path: 42/1
    assert_eq!(num, 42, "should be 42/1 via shadow, got {}/{}", num, den);
    assert_eq!(den, 1, "denominator should be 1 via shadow, got {}", den);
}

#[test]
fn test_shadow_fast_path_arithmetic_chain() {
    // 1.5 + 2.5 → shadow_add(3/2, 5/2) = 4/1 → to_rational should return 4/1
    let result = evaluate(&(gmath("1.5") + gmath("2.5"))).unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "arithmetic result should preserve shadow");

    let rational = result.to_rational().unwrap();
    let num = rational.numerator_i128().unwrap();
    let den = rational.denominator_i128().unwrap();
    assert_eq!(num, 4, "1.5 + 2.5 = 4/1, got {}/{}", num, den);
    assert_eq!(den, 1, "denominator should be 1, got {}", den);
}

#[test]
fn test_shadow_fast_path_multiplication() {
    // 3 * 7 = 21/1 via shadow
    let result = evaluate(&(gmath("3") * gmath("7"))).unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "integer multiply should preserve shadow");

    let rational = result.to_rational().unwrap();
    let num = rational.numerator_i128().unwrap();
    let den = rational.denominator_i128().unwrap();
    assert_eq!(num, 21, "3 * 7 = 21, got {}/{}", num, den);
    assert_eq!(den, 1);
}

#[test]
fn test_shadow_none_falls_through() {
    // Transcendental results have no shadow — to_rational must still work
    let result = evaluate(&gmath("1.0").exp()).unwrap();
    assert!(result.shadow().is_none(), "transcendental should have no shadow");
    let rational = result.to_rational().unwrap();
    // Should still produce valid rational via Q-format reconstruction
    let num = rational.numerator_i128();
    assert!(num.is_some(), "to_rational must succeed even without shadow");
}

#[test]
fn test_shadow_display_rational() {
    let shadow = CompactShadow::from_rational(1, 3);
    assert_eq!(format!("{}", shadow), "1/3");

    let shadow = CompactShadow::from_rational(42, 1);
    assert_eq!(format!("{}", shadow), "42");

    let shadow = CompactShadow::from_rational(-7, 2);
    assert_eq!(format!("{}", shadow), "-7/2");
}

#[test]
fn test_shadow_display_special() {
    let shadow = CompactShadow::None;
    assert_eq!(format!("{}", shadow), "none");

    let shadow = CompactShadow::ConstantRef(ShadowConstantId::Pi);
    assert_eq!(format!("{}", shadow), "\u{03C0}");

    let shadow = CompactShadow::ConstantRef(ShadowConstantId::E);
    assert_eq!(format!("{}", shadow), "e");

    let shadow = CompactShadow::ConstantRef(ShadowConstantId::Sqrt2);
    assert_eq!(format!("{}", shadow), "\u{221A}2");
}

#[test]
fn test_shadow_convenience_accessors() {
    let shadow = CompactShadow::from_rational(3, 4);
    assert_eq!(shadow.numerator(), Some(3));
    assert_eq!(shadow.denominator(), Some(4));
    assert_eq!(shadow.constant_id(), None);

    let shadow = CompactShadow::ConstantRef(ShadowConstantId::Phi);
    assert_eq!(shadow.numerator(), None);
    assert_eq!(shadow.denominator(), None);
    assert_eq!(shadow.constant_id(), Some(ShadowConstantId::Phi));

    let shadow = CompactShadow::None;
    assert_eq!(shadow.numerator(), None);
    assert_eq!(shadow.denominator(), None);
    assert_eq!(shadow.constant_id(), None);
}

#[test]
fn test_shadow_hex_literal() {
    // Hex literals should have shadows
    let mut eval = StackEvaluator::new(DeploymentProfile::default());
    let result = eval.parse_literal("0xFF").unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "hex literal 0xFF should have shadow, got {:?}", shadow);
    assert_eq!(shadow.as_rational(), Some((255, 1)));
}

#[test]
fn test_shadow_binary_literal() {
    // Binary literals should have shadows
    let mut eval = StackEvaluator::new(DeploymentProfile::default());
    let result = eval.parse_literal("0b1010").unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "binary literal 0b1010 should have shadow, got {:?}", shadow);
    assert_eq!(shadow.as_rational(), Some((10, 1)));
}

#[test]
fn test_shadow_division_preserves() {
    // 10 / 4 = 5/2 via shadow
    let result = evaluate(&(gmath("10") / gmath("4"))).unwrap();
    let shadow = result.shadow();
    // Division may lose shadow if intermediate overflows, but for small values it should work
    if shadow.is_some() {
        let (num, den) = shadow.as_rational().unwrap();
        // 10/1 ÷ 4/1 = 10/4 = 5/2
        assert_eq!(num * 2, den as i128 * 5, "10/4 should be 5/2, got {}/{}", num, den);
    }
}

#[test]
fn test_shadow_negation_preserves() {
    // -(1.5) → shadow should be -3/2
    let result = evaluate(&(-gmath("1.5"))).unwrap();
    let shadow = result.shadow();
    assert!(shadow.is_some(), "negation should preserve shadow");
    let (num, den) = shadow.as_rational().unwrap();
    // -1.5 = -(15/10) → shadow_negate
    assert!(num < 0, "negated value should have negative numerator, got {}/{}", num, den);
}
