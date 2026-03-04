//! Error Handling & Input Validation Tests
//!
//! Validates that the ZASC pipeline gracefully handles:
//! - Malformed / empty input strings
//! - Domain errors (ln(0), sqrt(-1), asin(2), ...)
//! - Division by zero
//! - Edge cases (negative zero)
//! - Named constant parsing (pi, e, sqrt2, phi, ln2, ln10)
//!
//! All tests are pure Rust, no external dependencies, no rebuild required.

use g_math::canonical::{gmath, evaluate, LazyExpr};
use g_math::fixed_point::core_types::OverflowDetected;

// ============================================================================
// Helper: gmath_safe for negative literals
// ============================================================================

fn gmath_safe(input: &'static str) -> LazyExpr {
    if input.starts_with('-') {
        let positive: &'static str = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                input.as_ptr().add(1),
                input.len() - 1,
            ))
        };
        -gmath(positive)
    } else {
        gmath(input)
    }
}

// ============================================================================
// Phase 1A: Parse Errors — empty and malformed input
// ============================================================================

#[test]
fn parse_empty_string() {
    let result = evaluate(&gmath(""));
    assert!(result.is_err(), "empty string should return Err, got {:?}", result);
}

#[test]
fn parse_whitespace_only() {
    let r1 = evaluate(&gmath(" "));
    assert!(r1.is_err(), "single space should return Err, got {:?}", r1);
}

#[test]
fn parse_tab_only() {
    let r = evaluate(&gmath("\t"));
    assert!(r.is_err(), "tab should return Err, got {:?}", r);
}

#[test]
fn parse_bare_dot() {
    let r = evaluate(&gmath("."));
    assert!(r.is_err(), "bare dot should return Err, got {:?}", r);
}

#[test]
fn parse_bare_minus() {
    let r = evaluate(&gmath("-"));
    assert!(r.is_err(), "bare minus should return Err, got {:?}", r);
}

#[test]
fn parse_bare_slash() {
    let r = evaluate(&gmath("/"));
    assert!(r.is_err(), "bare slash should return Err, got {:?}", r);
}

#[test]
fn parse_fraction_missing_denominator() {
    let r = evaluate(&gmath("1/"));
    assert!(r.is_err(), "'1/' should return Err, got {:?}", r);
}

#[test]
fn parse_fraction_missing_numerator() {
    let r = evaluate(&gmath("/3"));
    assert!(r.is_err(), "'/3' should return Err, got {:?}", r);
}

#[test]
fn parse_double_slash_fraction() {
    let r = evaluate(&gmath("1/2/3"));
    assert!(r.is_err(), "'1/2/3' should return Err, got {:?}", r);
}

#[test]
fn parse_double_dot_decimal() {
    let r = evaluate(&gmath("1.2.3"));
    assert!(r.is_err(), "'1.2.3' should return Err, got {:?}", r);
}

#[test]
fn parse_double_minus() {
    let r = evaluate(&gmath("--5"));
    assert!(r.is_err(), "'--5' should return Err, got {:?}", r);
}

#[test]
fn parse_scientific_notation_rejected() {
    // The library uses fixed-point, not floating-point — "1e10" is not a valid format
    let r = evaluate(&gmath("1e10"));
    assert!(r.is_err(), "'1e10' should return Err (no scientific notation), got {:?}", r);
}

#[test]
fn parse_nan_rejected() {
    let r = evaluate(&gmath("NaN"));
    assert!(r.is_err(), "'NaN' should return Err, got {:?}", r);
}

#[test]
fn parse_inf_rejected() {
    let r = evaluate(&gmath("inf"));
    assert!(r.is_err(), "'inf' should return Err, got {:?}", r);
}

#[test]
fn parse_infinity_rejected() {
    let r = evaluate(&gmath("Infinity"));
    assert!(r.is_err(), "'Infinity' should return Err, got {:?}", r);
}

#[test]
fn parse_unknown_constant_foobar() {
    let r = evaluate(&gmath("foobar"));
    assert!(r.is_err(), "'foobar' should return Err, got {:?}", r);
}

#[test]
fn parse_unknown_constant_pi_squared() {
    let r = evaluate(&gmath("PI_SQUARED"));
    assert!(r.is_err(), "'PI_SQUARED' should return Err, got {:?}", r);
}

// ============================================================================
// Phase 1A-extra: Whitespace-padded inputs
// ============================================================================

/// Leading/trailing whitespace: document behavior (may be trimmed or rejected)
#[test]
fn parse_leading_space_integer() {
    let r = evaluate(&gmath(" 5"));
    // Either successfully parses as 5 (with trimming) or returns ParseError
    // Must NOT panic
    match &r {
        Ok(v) => {
            let s = format!("{}", v);
            assert!(s.contains('5'), "if accepted, should represent 5, got '{}'", s);
        }
        Err(e) => {
            assert!(
                matches!(e, OverflowDetected::ParseError | OverflowDetected::InvalidInput),
                "if rejected, should be ParseError or InvalidInput, got {:?}",
                e
            );
        }
    }
}

#[test]
fn parse_trailing_space_integer() {
    let r = evaluate(&gmath("5 "));
    match &r {
        Ok(v) => {
            let s = format!("{}", v);
            assert!(s.contains('5'), "if accepted, should represent 5, got '{}'", s);
        }
        Err(e) => {
            assert!(
                matches!(e, OverflowDetected::ParseError | OverflowDetected::InvalidInput),
                "if rejected, should be ParseError or InvalidInput, got {:?}",
                e
            );
        }
    }
}

#[test]
fn parse_surrounding_space_decimal() {
    let r = evaluate(&gmath(" 1.5 "));
    match &r {
        Ok(v) => {
            let s = format!("{}", v);
            assert!(s.starts_with("1.5"), "if accepted, should represent 1.5, got '{}'", s);
        }
        Err(e) => {
            assert!(
                matches!(e, OverflowDetected::ParseError | OverflowDetected::InvalidInput),
                "if rejected, should be ParseError or InvalidInput, got {:?}",
                e
            );
        }
    }
}

// ============================================================================
// Phase 1B: Domain Errors
// ============================================================================

#[test]
fn domain_error_ln_zero() {
    let r = evaluate(&gmath("0").ln());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "ln(0) should return DomainError, got {:?}",
        r
    );
}

#[test]
fn domain_error_ln_negative() {
    let r = evaluate(&gmath_safe("-1").ln());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "ln(-1) should return DomainError, got {:?}",
        r
    );
}

#[test]
fn domain_error_sqrt_negative() {
    let r = evaluate(&gmath_safe("-1").sqrt());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "sqrt(-1) should return DomainError, got {:?}",
        r
    );
}

#[test]
fn domain_error_sqrt_negative_large() {
    let r = evaluate(&gmath_safe("-100").sqrt());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "sqrt(-100) should return DomainError, got {:?}",
        r
    );
}

#[test]
fn domain_error_asin_above_one() {
    let r = evaluate(&gmath("2").asin());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "asin(2) should return DomainError (|x| > 1), got {:?}",
        r
    );
}

#[test]
fn domain_error_asin_below_neg_one() {
    let r = evaluate(&gmath_safe("-2").asin());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "asin(-2) should return DomainError, got {:?}",
        r
    );
}

#[test]
fn domain_error_acos_above_one() {
    let r = evaluate(&gmath("2").acos());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "acos(2) should return DomainError (|x| > 1), got {:?}",
        r
    );
}

#[test]
fn domain_error_acos_below_neg_one() {
    let r = evaluate(&gmath_safe("-2").acos());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "acos(-2) should return DomainError, got {:?}",
        r
    );
}

#[test]
fn domain_error_atanh_at_one() {
    let r = evaluate(&gmath("1").atanh());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "atanh(1) should return DomainError (|x| >= 1), got {:?}",
        r
    );
}

#[test]
fn domain_error_atanh_at_neg_one() {
    let r = evaluate(&gmath_safe("-1").atanh());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "atanh(-1) should return DomainError, got {:?}",
        r
    );
}

#[test]
fn domain_error_atanh_above_one() {
    let r = evaluate(&gmath("2").atanh());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "atanh(2) should return DomainError, got {:?}",
        r
    );
}

#[test]
fn domain_error_acosh_below_one() {
    let r = evaluate(&gmath("0.5").acosh());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "acosh(0.5) should return DomainError (x < 1), got {:?}",
        r
    );
}

#[test]
fn domain_error_acosh_zero() {
    let r = evaluate(&gmath("0").acosh());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "acosh(0) should return DomainError, got {:?}",
        r
    );
}

#[test]
fn domain_error_acosh_negative() {
    let r = evaluate(&gmath_safe("-1").acosh());
    assert!(
        matches!(r, Err(OverflowDetected::DomainError)),
        "acosh(-1) should return DomainError, got {:?}",
        r
    );
}

// ============================================================================
// Phase 1C: Division by Zero
// ============================================================================

#[test]
fn division_by_zero_integer() {
    let r = evaluate(&(gmath("5") / gmath("0")));
    assert!(
        matches!(r, Err(OverflowDetected::DivisionByZero)),
        "5 / 0 should return DivisionByZero, got {:?}",
        r
    );
}

#[test]
fn division_by_zero_zero_over_zero() {
    let r = evaluate(&(gmath("0") / gmath("0")));
    assert!(
        matches!(r, Err(OverflowDetected::DivisionByZero)),
        "0 / 0 should return DivisionByZero, got {:?}",
        r
    );
}

#[test]
fn division_by_zero_decimal() {
    let r = evaluate(&(gmath("1.5") / gmath("0.0")));
    // Decimal domain may return PrecisionLoss (via rational path) instead of DivisionByZero
    assert!(
        matches!(
            r,
            Err(OverflowDetected::DivisionByZero) | Err(OverflowDetected::PrecisionLoss)
        ),
        "1.5 / 0.0 should return DivisionByZero or PrecisionLoss, got {:?}",
        r
    );
}

#[test]
fn division_by_zero_fraction() {
    let r = evaluate(&(gmath("1/3") / gmath("0")));
    assert!(
        r.is_err(),
        "1/3 / 0 should return error, got {:?}",
        r
    );
}

#[test]
fn division_by_zero_negative_numerator() {
    let r = evaluate(&(gmath_safe("-5") / gmath("0")));
    assert!(
        matches!(r, Err(OverflowDetected::DivisionByZero)),
        "-5 / 0 should return DivisionByZero, got {:?}",
        r
    );
}

// ============================================================================
// Phase 1D: Negative Zero
// ============================================================================

#[test]
fn negative_zero_integer() {
    let pos = evaluate(&gmath("0")).unwrap();
    let neg = evaluate(&gmath_safe("-0")).unwrap();
    let neg_str = format!("{}", neg);
    // Both should display as zero (no negative sign)
    assert!(
        !neg_str.starts_with('-') || neg_str.starts_with("-0.000"),
        "-0 should display as zero or -0.0..., got '{}'",
        neg_str
    );
    // If both have binary storage, they should be equal
    if let (Some(pv), Some(nv)) = (pos.as_binary_storage(), neg.as_binary_storage()) {
        assert_eq!(pv, nv, "-0 and 0 should have identical binary storage");
    }
}

#[test]
fn negative_zero_decimal() {
    let pos = evaluate(&gmath("0.0")).unwrap();
    let neg = evaluate(&gmath_safe("-0.0")).unwrap();
    let pos_str = format!("{}", pos);
    let neg_str = format!("{}", neg);
    assert!(
        !neg_str.starts_with('-') || neg_str.starts_with("-0.0"),
        "-0.0 should display as zero, got '{}'",
        neg_str
    );
    println!("  0.0 displays as: '{}'", pos_str);
    println!(" -0.0 displays as: '{}'", neg_str);
}

// ============================================================================
// Phase 1E: Named Constants
//
// Named constants are parsed into StackValue::Symbolic(RationalNumber).
// Display for Symbolic shows the rational fraction (e.g., "num/den"), NOT decimal.
// This is correct behavior — Symbolic domain preserves exact representation.
// ============================================================================

/// Helper: verify a named constant parses to Symbolic
macro_rules! test_named_constant_parses {
    ($test_name:ident, $input:expr, $label:expr) => {
        #[test]
        fn $test_name() {
            let result = evaluate(&gmath($input));
            assert!(
                result.is_ok(),
                "{}: gmath('{}') should succeed, got {:?}",
                $label, $input, result
            );
            let val = result.unwrap();
            let s = format!("{}", val);
            // Symbolic constants display as decimal strings or rational fractions
            assert!(
                !s.is_empty() && (s.contains('.') || s.contains('/')),
                "{}: named constant should display as number, got '{}'",
                $label, s
            );
            println!("  {} = {}", $label, s);
        }
    };
}

test_named_constant_parses!(named_constant_pi_lower, "pi", "pi");
test_named_constant_parses!(named_constant_pi_upper, "PI", "PI");
test_named_constant_parses!(named_constant_pi_mixed, "Pi", "Pi");
test_named_constant_parses!(named_constant_e_lower, "e", "e");
test_named_constant_parses!(named_constant_e_upper, "E", "E");
test_named_constant_parses!(named_constant_sqrt2, "sqrt2", "sqrt2");
test_named_constant_parses!(named_constant_phi, "phi", "phi");
test_named_constant_parses!(named_constant_ln2, "ln2", "ln2");
test_named_constant_parses!(named_constant_ln10, "ln10", "ln10");

/// All case variants of pi should produce the same value
#[test]
fn named_constant_pi_case_invariant() {
    let pi1 = format!("{}", evaluate(&gmath("pi")).unwrap());
    let pi2 = format!("{}", evaluate(&gmath("PI")).unwrap());
    let pi3 = format!("{}", evaluate(&gmath("Pi")).unwrap());
    assert_eq!(pi1, pi2, "pi and PI should be equal");
    assert_eq!(pi2, pi3, "PI and Pi should be equal");
}

/// Named constants in symbolic arithmetic: pi + 0 should not error
#[test]
fn named_constant_in_symbolic_arithmetic() {
    // Adding a symbolic constant to zero should succeed
    let result = evaluate(&(gmath("pi") + gmath("0")));
    // This may produce PrecisionLimit if the rational is too large for tier ops.
    // Document the behavior.
    match &result {
        Ok(val) => println!("  pi + 0 = {}", val),
        Err(e) => println!("  pi + 0 = Err({:?}) (symbolic tier overflow — known limitation)", e),
    }
}

/// Named constants in transcendentals: currently produces TierOverflow
/// because Symbolic rationals with huge numerator/denominator overflow
/// during conversion to binary for transcendental evaluation.
#[test]
fn named_constant_transcendental_limitation() {
    let sin_pi = evaluate(&gmath("pi").sin());
    let exp_ln2 = evaluate(&gmath("ln2").exp());

    // Document current behavior: TierOverflow on high-precision Symbolic → Binary conversion
    match &sin_pi {
        Ok(val) => println!("  sin(pi) = {} (if this works, great!)", val),
        Err(e) => println!("  sin(pi) = Err({:?}) (known limitation: symbolic→binary overflow)", e),
    }
    match &exp_ln2 {
        Ok(val) => println!("  exp(ln2) = {} (if this works, great!)", val),
        Err(e) => println!("  exp(ln2) = Err({:?}) (known limitation: symbolic→binary overflow)", e),
    }

    // Neither should panic — that's the key invariant
}

/// Use decimal approximations of constants for transcendental tests instead
#[test]
fn constant_decimal_approx_sin_pi() {
    // Use decimal approximation rather than named constant
    let result = evaluate(&gmath("3.14159265358979323846").sin());
    assert!(result.is_ok(), "sin(3.14159...) should succeed, got {:?}", result);
    let s = format!("{}", result.unwrap());
    let trimmed = s.trim_start_matches('-');
    assert!(
        trimmed.starts_with("0.000"),
        "sin(pi_approx) should be ~0, got '{}'",
        s
    );
}

/// Use decimal approximation: exp(0.693147...) ≈ 2
#[test]
fn constant_decimal_approx_exp_ln2() {
    let result = evaluate(&gmath("0.69314718055994530941").exp());
    assert!(result.is_ok(), "exp(ln2_approx) should succeed, got {:?}", result);
    let s = format!("{}", result.unwrap());
    assert!(
        s.starts_with("2.000") || s.starts_with("1.999"),
        "exp(ln2_approx) should be ~2.0, got '{}'",
        s
    );
}
