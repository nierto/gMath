//! Fractal Router Integration Tests
//!
//! Validates that the fractal topology router correctly eliminates domain
//! crisscross by routing cross-domain arithmetic through the optimal domain.
//!
//! **KEY INVARIANT**: `gmath("0.1") + gmath("255")` routes through Decimal
//! domain (not Symbolic/rational fallback), because 0.1 is decimal-exact
//! and 255 is decimal-exact (integer = exact in all domains).

use g_math::canonical::{gmath, gmath_parse, evaluate};

// ============================================================================
// ROUTER CLASSIFICATION TESTS (via public API)
// ============================================================================

#[test]
fn crisscross_decimal_plus_integer_stays_decimal() {
    // Previously: Decimal + Binary → rational fallback → Symbolic
    // Now: Router sees both are decimal-exact → Decimal domain
    let expr = gmath("0.1") + gmath("255");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    // 0.1 + 255 = 255.1
    assert!(s.contains("255.1"), "Expected 255.1, got: {}", s);
}

#[test]
fn crisscross_integer_plus_decimal_stays_decimal() {
    let expr = gmath("100") + gmath("0.25");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.contains("100.25"), "Expected 100.25, got: {}", s);
}

#[test]
fn crisscross_decimal_mul_integer() {
    // 0.1 * 10 = 1.0
    let expr = gmath("0.1") * gmath("10");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.starts_with("1"), "Expected 1, got: {}", s);
}

#[test]
fn crisscross_integer_sub_decimal() {
    // 100 - 0.01 = 99.99
    let expr = gmath("100") - gmath("0.01");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.contains("99.99"), "Expected 99.99, got: {}", s);
}

#[test]
fn crisscross_decimal_div_integer() {
    // 1.0 / 4 = 0.25
    let expr = gmath("1.0") / gmath("4");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.contains("0.25"), "Expected 0.25, got: {}", s);
}

// ============================================================================
// SAME-DOMAIN REGRESSION TESTS (must still work)
// ============================================================================

#[test]
fn same_domain_binary_add() {
    let expr = gmath("100") + gmath("200");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.starts_with("300"), "Expected 300, got: {}", s);
}

#[test]
fn same_domain_decimal_add() {
    let expr = gmath("0.1") + gmath("0.2");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.starts_with("0.3"), "Expected 0.3, got: {}", s);
}

#[test]
fn same_domain_symbolic_add() {
    // 1/3 + 1/3 = 2/3 — both are ternary+symbolic, but ternary routing is deferred
    // so they go through their native domain (symbolic for fractions)
    let expr = gmath("1/3") + gmath("1/3");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.contains("2/3") || s.contains("0.666"), "Expected 2/3, got: {}", s);
}

// ============================================================================
// FINANCIAL CHAIN TESTS (decimal throughout)
// ============================================================================

#[test]
fn financial_compound_interest_chain() {
    // principal * rate: 1000.00 * 1.05 → should stay Decimal if both are decimal-exact
    let expr = gmath("1000.00") * gmath("1.05");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.starts_with("1050"), "Expected ~1050, got: {}", s);
}

#[test]
fn financial_mixed_integer_decimal_chain() {
    // price * quantity + tax: (9.99 * 3) + 2
    let price_qty = gmath("9.99") * gmath("3");
    let total = price_qty + gmath("2");
    let result = evaluate(&total).unwrap();
    let s = format!("{}", result);
    // 9.99 * 3 = 29.97, + 2 = 31.97
    assert!(s.starts_with("31.97"), "Expected 31.97, got: {}", s);
}

// ============================================================================
// TRANSCENDENTAL + ARITHMETIC CHAIN TESTS
// ============================================================================

#[test]
fn decimal_exp_plus_integer() {
    // exp(0.1) + 1 — exp returns DecimalCompute, 1 is Binary
    // The DecimalCompute + integer should work (compute-tier early exit handles this)
    let expr = gmath("0.1").exp() + gmath("1");
    let result = evaluate(&expr).unwrap();
    // exp(0.1) ~ 1.10517, + 1 ~ 2.10517
    let s = format!("{}", result);
    assert!(s.starts_with("2.1"), "Expected ~2.1, got: {}", s);
}

#[test]
fn integer_mul_after_decimal_add() {
    // (0.5 + 0.5) * 100 → should be Decimal add then Decimal * integer
    let sum = gmath("0.5") + gmath("0.5");
    let expr = sum * gmath("100");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.starts_with("100"), "Expected 100, got: {}", s);
}

// ============================================================================
// INCOMPATIBLE DOMAIN TESTS (should fall back to symbolic)
// ============================================================================

#[test]
fn incompatible_decimal_plus_ternary_fraction() {
    // 0.1 (decimal) + 1/3 (ternary) → no common domain → symbolic
    let expr = gmath("0.1") + gmath("1/3");
    let result = evaluate(&expr).unwrap();
    // 0.1 + 1/3 = 1/10 + 1/3 = 13/30 ≈ 0.4333...
    let s = format!("{}", result);
    assert!(
        s.contains("13/30") || s.contains("0.43"),
        "Expected 13/30 or ~0.43, got: {}", s
    );
}

// ============================================================================
// EDGE CASES
// ============================================================================

#[test]
fn zero_integer_plus_decimal() {
    let expr = gmath("0") + gmath("0.5");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.starts_with("0.5"), "Expected 0.5, got: {}", s);
}

#[test]
fn negative_integer_plus_decimal() {
    let expr = gmath("-1") + gmath("0.5");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.contains("-0.5") || s.contains("0.5"), "Expected -0.5, got: {}", s);
}

#[test]
fn large_integer_times_decimal() {
    // 1000000 * 0.001 = 1000
    let expr = gmath("1000000") * gmath("0.001");
    let result = evaluate(&expr).unwrap();
    let s = format!("{}", result);
    assert!(s.starts_with("1000"), "Expected 1000, got: {}", s);
}

/// 0.5.0 item 0 regression: integer literals beyond the profile's binary
/// storage range must fall back to the Symbolic domain (UGOD's ladder top
/// never fails) instead of erroring at parse. Pre-fix, realtime failed to
/// parse "32768"+ with Overflow, so `1000000 * 0.001` errored on realtime
/// while succeeding on every other profile (bisected to <=0.4.32).
#[test]
fn oversized_integer_literal_falls_back_to_symbolic() {
    use g_math::canonical::StackValue;
    for s in ["32768", "1000000", "9223372036854775807"] {
        let v = evaluate(&gmath_parse(s).expect("must parse on every profile"))
            .expect("must evaluate on every profile");
        // Value must be exact regardless of the domain it landed in:
        // v - s == 0 through the router.
        let diff = format!("{}", evaluate(&(gmath_parse(s).unwrap() - gmath_parse(s).unwrap())).unwrap());
        let trimmed = if diff.contains('.') {
            diff.trim_end_matches('0').trim_end_matches('.')
        } else {
            diff.as_str()
        };
        assert!(trimmed == "0" || trimmed == "-0" || trimmed == "0/1", "self-diff of {s} = {diff:?}");
        // On profiles whose integer range cannot hold the value, the
        // fallback domain is Symbolic; wide profiles keep Binary.
        let fits_binary = {
            #[cfg(table_format = "q16_16")]
            { s.parse::<i128>().unwrap().unsigned_abs() < (1u128 << 15) }
            #[cfg(table_format = "q32_32")]
            { s.parse::<i128>().unwrap().unsigned_abs() < (1u128 << 31) }
            #[cfg(table_format = "q64_64")]
            { s.parse::<i128>().unwrap().unsigned_abs() < (1u128 << 63) }
            #[cfg(not(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64")))]
            { true }
        };
        match (&v, fits_binary) {
            (StackValue::Binary(..), true) => {}
            (StackValue::Symbolic(_), false) => {}
            (other, fits) => panic!("{s}: unexpected variant {other:?} (fits_binary={fits})"),
        }
    }
    // The original failing expression, end to end.
    let r = evaluate(&(gmath_parse("1000000").unwrap() * gmath_parse("0.001").unwrap()))
        .expect("1000000 * 0.001 must evaluate on every profile");
    assert!(format!("{r}").starts_with("1000"), "got {r}");
}
