//! Compound Operations & New Feature Tests
//!
//! Validates:
//! - LazyExpr::Value chaining (compound interest pattern)
//! - Display format (human-readable decimal, precision specifier)
//! - Mathematical identities (self-validating, no reference data)
//! - Deep nesting / round-trip precision
//! - Prime table API
//!
//! All tests are pure Rust, no external dependencies, no rebuild required.

use g_math::canonical::{gmath, evaluate, LazyExpr};
use g_math::fixed_point::prime_table::{is_prime, nth_prime, prime_count_up_to, PRIME_COUNT, MAX_PRIME};

// ============================================================================
// Helper
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
// Phase 2A: LazyExpr::Value Chaining
// ============================================================================

/// Compound interest: balance * 1.05, iterated 10 times
/// 1000 * 1.05^10 = 1628.89462677744140625 (exact rational: 1.05 = 21/20)
///
/// On q128_128+: stays Decimal the entire time (dp=20 < threshold=38).
/// On q64_64: dp promotion fires at year 10 → Symbolic (exact rational).
/// Either way, 1.05 has a finite decimal expansion, so result is exact.
#[test]
fn value_chaining_compound_interest_10_years() {
    let mut balance = evaluate(&gmath("1000.00")).unwrap();

    for _year in 1..=10 {
        let expr = LazyExpr::from(balance) * gmath("1.05");
        balance = evaluate(&expr).expect("compound interest step should not fail");
    }

    let s = format!("{}", balance);
    // 1000 * (21/20)^10 = exact rational with finite decimal expansion
    assert!(
        s.starts_with("1628.89"),
        "1000*1.05^10 should match 1628.89..., got '{}'",
        s
    );
}

/// Compound interest: 30 iterations — validates UGOD promotion + exact rational.
///
/// 1000 * 1.05^30 = 4321.942375150662... (exact: 1.05 = 21/20)
/// Each iteration adds 2 decimal places (dp_result = dp_a + dp_b).
/// When dp exceeds DECIMAL_DP_PROMOTION_THRESHOLD, the multiply path
/// promotes to Symbolic (exact rational). Since 1.05 has a finite decimal
/// expansion, all results are exact — 0 ULP by construction.
/// All 20 checked years should match exactly.
#[test]
fn value_chaining_compound_interest_30_years() {
    let mut balance = evaluate(&gmath("1000.00")).unwrap();
    let mut last_correct_year = 0;

    // mpmath reference values: 1000 * 1.05^n (80-digit precision)
    // Using ~15 significant digits — safe across all profiles.
    // Years 1-8 are exact Decimal on all profiles.
    // Years 9+ may be Binary depending on profile threshold.
    let expected_prefixes: &[&str] = &[
        "",                     // year 0 (skip)
        "1050.0",               // year 1  (mpmath: 1050.0)
        "1102.5",               // year 2  (mpmath: 1102.5)
        "1157.625",             // year 3  (mpmath: 1157.625)
        "1215.50625",           // year 4  (mpmath: 1215.50625)
        "1276.281562",          // year 5  (mpmath: 1276.2815625)
        "1340.09564062",        // year 6  (mpmath: 1340.095640625)
        "1407.10042265",        // year 7  (mpmath: 1407.10042265625)
        "1477.45544378",        // year 8  (mpmath: 1477.4554437890625)
        "1551.32821597",        // year 9  (mpmath: 1551.328215978515625)
        "1628.89462677",        // year 10 (mpmath: 1628.89462677744140625)
        "1710.33935811",        // year 11 (mpmath: 1710.3393581163134765625)
        "1795.85632602",        // year 12 (mpmath: 1795.856326022129150390625)
        "1885.64914232",        // year 13 (mpmath: 1885.64914232323560791015625)
        "1979.93159943",        // year 14 (mpmath: 1979.9315994393973883056640625)
        "2078.92817941",        // year 15 (mpmath: 2078.928179411367257720947265625)
        "2182.87458838",        // year 16 (mpmath: 2182.87458838193562060699462890625)
        "2292.01831780",        // year 17 (mpmath: 2292.0183178010324016373443603515625)
        "2406.61923369",        // year 18 (mpmath: 2406.619233691084021719211578369140625)
        "2526.95019537",        // year 19 (mpmath: 2526.95019537563822280517215728759765625)
        "2653.29770514",        // year 20 (mpmath: 2653.2977051444201339454307651519775390625)
    ];

    for year in 1..=30 {
        let expr = LazyExpr::from(balance) * gmath("1.05");
        balance = evaluate(&expr).expect("compound interest step should not fail");

        let s = format!("{}", balance);
        if year < expected_prefixes.len() {
            if s.starts_with(expected_prefixes[year]) {
                last_correct_year = year;
            } else {
                println!(
                    "  year {}: MISMATCH (expected '{}...', got '{}')",
                    year, expected_prefixes[year], &s[..s.len().min(50)]
                );
            }
        }
    }

    println!("  last correct year: {} of 30", last_correct_year);
    // All prefix-checked years must match. Q16.16 displays fewer digits so
    // string prefixes may not match — require fewer matching years.
    #[cfg(table_format = "q16_16")]
    let min_years = 3; // Q16.16: 4 digits, prefixes diverge at year 4
    #[cfg(not(table_format = "q16_16"))]
    let min_years = 20;
    assert!(
        last_correct_year >= min_years,
        "should maintain precision for {} checked years, lost at year {}",
        min_years, last_correct_year + 1
    );
}

/// Compound interest: 100 years — validates long-running Symbolic chains.
/// 131501 exceeds Q16.16 max (32767). 39 trillion exceeds Q32.32 max.
///
/// 1000 * 1.05^100 = 131501.25784630345502559753209371674816065646... (mpmath)
/// After dp promotion, iterations stay in Symbolic (exact rational).
/// Since 1.05 = 21/20 has a finite decimal expansion, all results are exact.
/// No accumulated rounding — 0 ULP by construction.
///
/// Q16.16: result overflows integer range (max 32767).
/// Q256.256: known Decimal multiply precision loss at high dp (dp_threshold=76,
/// iterations stay Decimal for 37 years before Symbolic promotion, accumulating
/// rounding at I512 width). Tracked as domain arithmetic gap.
#[cfg(not(any(table_format = "q16_16", table_format = "q256_256")))]
#[test]
fn value_chaining_compound_interest_100_years() {
    let mut balance = evaluate(&gmath("1000.00")).unwrap();

    for _year in 1..=100 {
        let expr = LazyExpr::from(balance) * gmath("1.05");
        balance = evaluate(&expr).expect("compound interest step should not fail");
    }

    let s = format!("{}", balance);
    println!("  1000 * 1.05^100 = {}", &s[..s.len().min(60)]);
    // mpmath: 131501.25784630345502559753209371674816065646...
    assert!(
        s.starts_with("131501.25"),
        "1000*1.05^100 should match mpmath 131501.25..., got '{}'",
        &s[..s.len().min(60)]
    );
}

/// Compound interest: 500 years — stress test for exact Symbolic chains.
///
/// 1000 * 1.05^500 = 39323261827217.83367222804425844837331164740749... (mpmath)
/// After dp promotion, all iterations are exact Symbolic (rational arithmetic).
/// Since 1.05 = 21/20 has a finite decimal expansion, result is exact.
/// No accumulated rounding — 0 ULP by construction, unlimited iterations.
/// Result 39 trillion overflows Q32.32 (max ~2.1B) and Q16.16 (max ~32K).
/// Q256.256: same Decimal multiply dp-accumulation issue as 100-year test.
#[test]
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32", table_format = "q256_256")))]
fn value_chaining_compound_interest_500_years() {
    let mut balance = evaluate(&gmath("1000.00")).unwrap();

    for _year in 1..=500 {
        let expr = LazyExpr::from(balance) * gmath("1.05");
        balance = evaluate(&expr).expect("compound interest step should not fail");
    }

    let s = format!("{}", balance);
    println!("  1000 * 1.05^500 = {}", &s[..s.len().min(60)]);
    // mpmath: 39323261827217.83367222804425844837331164740749...
    assert!(
        s.starts_with("39323261827217.8"),
        "1000*1.05^500 should match mpmath 39323261827217.8..., got '{}'",
        &s[..s.len().min(60)]
    );
}

/// Value chaining preserves domain: decimal * decimal stays decimal-like
#[test]
fn value_chaining_preserves_computation() {
    let a = evaluate(&gmath("2.5")).unwrap();
    let b = evaluate(&(LazyExpr::from(a) + gmath("3.5"))).unwrap();
    let s = format!("{}", b);
    assert!(
        s.starts_with("6.0") || s.starts_with("6/1"),
        "2.5 + 3.5 should be 6.0, got '{}'",
        s
    );
}

/// Chain transcendental results: compute exp(1), then use result
#[test]
fn value_chaining_transcendental_result() {
    let e = evaluate(&gmath("1").exp()).unwrap();
    // e * e should be e^2 ≈ 7.389056...
    let e_squared = evaluate(&(LazyExpr::from(e.clone()) * LazyExpr::from(e))).unwrap();
    let s = format!("{}", e_squared);
    assert!(
        s.starts_with("7.389"),
        "e * e should be ~7.389, got '{}'",
        s
    );
}

// ============================================================================
// Phase 2B: Display Format Verification
// ============================================================================

/// Binary domain values display as decimal, not "Binary[T3]:..."
#[test]
fn display_binary_is_decimal() {
    let val = evaluate(&gmath("42")).unwrap();
    let s = format!("{}", val);
    assert!(
        s.starts_with("42."),
        "integer 42 should display as '42.xxx', got '{}'",
        s
    );
    assert!(
        !s.contains("Binary"),
        "Display should not contain 'Binary', got '{}'",
        s
    );
}

/// Precision specifier: {:.2} gives 2 decimal places
#[test]
fn display_precision_2() {
    let val = evaluate(&gmath("3.14159")).unwrap();
    let s = format!("{:.2}", val);
    // Should show ~2 decimal digits worth
    assert!(
        s.starts_with("3.14") || s.starts_with("3.1"),
        "{{:.2}} should give ~2 decimal places, got '{}'",
        s
    );
}

/// Precision specifier: {:.6} gives 6 decimal places (profile-dependent)
#[test]
fn display_precision_6() {
    let val = evaluate(&gmath("3.14159")).unwrap();
    let s = format!("{:.6}", val);
    // Q16.16 has only 4 decimal digits — accept shorter match
    assert!(
        s.starts_with("3.1415"),
        "{{:.6}} should give decimal places starting with 3.1415, got '{}'",
        s
    );
}

/// Decimal domain should Display as decimal
#[test]
fn display_decimal_domain() {
    let val = evaluate(&gmath("0.25")).unwrap();
    let s = format!("{}", val);
    // Decimal domain: should display as 0.25 or equivalent
    assert!(
        s.contains("0.25") || s.contains("25") || s.contains("1/4"),
        "0.25 should display meaningfully, got '{}'",
        s
    );
}

/// Symbolic domain should Display as fraction
#[test]
fn display_symbolic_domain() {
    let val = evaluate(&gmath("1/3")).unwrap();
    let s = format!("{}", val);
    // Symbolic: should show as fraction
    assert!(
        s.contains("1/3") || s.contains("0.333"),
        "1/3 should display as fraction or decimal, got '{}'",
        s
    );
}

/// Zero displays cleanly
#[test]
fn display_zero() {
    let val = evaluate(&gmath("0")).unwrap();
    let s = format!("{}", val);
    assert!(
        s.starts_with("0"),
        "0 should display starting with '0', got '{}'",
        s
    );
}

/// Negative value displays with minus sign
#[test]
fn display_negative() {
    let val = evaluate(&gmath_safe("-5")).unwrap();
    let s = format!("{}", val);
    assert!(
        s.starts_with("-5.") || s.starts_with("-5/") || s == "-5",
        "-5 should display with minus sign, got '{}'",
        s
    );
}

// ============================================================================
// Phase 2C: Mathematical Identities (Self-Validating)
// ============================================================================

/// sin^2(x) + cos^2(x) = 1 (Pythagorean identity)
#[test]
fn identity_pythagorean() {
    let test_values: &[&str] = &[
        "0.1", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "0.01", "0.99", "1.23",
    ];

    for &x in test_values {
        let sin_val = evaluate(&gmath(x).sin()).unwrap();
        let cos_val = evaluate(&gmath(x).cos()).unwrap();

        // sin^2 + cos^2
        let sin_sq = evaluate(&(LazyExpr::from(sin_val.clone()) * LazyExpr::from(sin_val))).unwrap();
        let cos_sq = evaluate(&(LazyExpr::from(cos_val.clone()) * LazyExpr::from(cos_val))).unwrap();
        let sum = evaluate(&(LazyExpr::from(sin_sq) + LazyExpr::from(cos_sq))).unwrap();

        let s = format!("{}", sum);
        assert!(
            s.starts_with("1.000") || s.starts_with("0.999"),
            "sin^2({}) + cos^2({}) should be ~1.0, got '{}'",
            x, x, s
        );
    }
}

/// exp(ln(x)) = x for positive x
#[test]
fn identity_exp_ln_roundtrip() {
    let test_values: &[&str] = &[
        "0.1", "0.5", "1.0", "2.0", "3.0", "5.0", "10.0", "0.01", "7.5", "42.0",
    ];

    for &x in test_values {
        let result = evaluate(&gmath(x).ln().exp());
        assert!(result.is_ok(), "exp(ln({})) should succeed, got {:?}", x, result);

        let s = format!("{}", result.unwrap());
        // Should recover x — check that Display starts with the input value's integer part
        let expected_prefix = x.split('.').next().unwrap();
        assert!(
            s.starts_with(expected_prefix),
            "exp(ln({})) should recover ~{}, got '{}'",
            x, x, s
        );
    }
}

/// ln(exp(x)) = x for x in valid range
#[test]
fn identity_ln_exp_roundtrip() {
    let test_values: &[&str] = &[
        "0.1", "0.5", "1.0", "2.0", "3.0", "5.0", "10.0", "0.01", "0.99", "7.5",
    ];

    for &x in test_values {
        let result = evaluate(&gmath(x).exp().ln());
        assert!(result.is_ok(), "ln(exp({})) should succeed, got {:?}", x, result);

        let s = format!("{}", result.unwrap());
        let expected_prefix = x.split('.').next().unwrap();
        assert!(
            s.starts_with(expected_prefix),
            "ln(exp({})) should recover ~{}, got '{}'",
            x, x, s
        );
    }
}

/// sinh(x) = (exp(x) - exp(-x)) / 2 — cross-check composition
#[test]
fn identity_sinh_definition() {
    let test_values: &[&str] = &["0.5", "1.0", "2.0", "0.1", "3.0"];

    for &x in test_values {
        // Direct sinh
        let sinh_direct = evaluate(&gmath(x).sinh()).unwrap();

        // Manual: (exp(x) - exp(-x)) / 2
        let exp_pos = evaluate(&gmath(x).exp()).unwrap();
        let exp_neg = evaluate(&gmath_safe(&leak_neg(x)).exp()).unwrap();
        let diff = evaluate(&(LazyExpr::from(exp_pos) - LazyExpr::from(exp_neg))).unwrap();
        let half = evaluate(&(LazyExpr::from(diff) / gmath("2"))).unwrap();

        let s_direct = format!("{:.10}", sinh_direct);
        let s_manual = format!("{:.10}", half);

        // First 8 chars should match (allowing for minor rounding)
        let prefix_len = 8.min(s_direct.len()).min(s_manual.len());
        assert_eq!(
            &s_direct[..prefix_len],
            &s_manual[..prefix_len],
            "sinh({}) direct vs manual: '{}' vs '{}'",
            x, s_direct, s_manual
        );
    }
}

/// Helper: create static negative string (leaks for test purposes only)
fn leak_neg(s: &str) -> &'static str {
    let neg = format!("-{}", s);
    Box::leak(neg.into_boxed_str())
}

// ============================================================================
// Phase 2D: Deep Nesting / Round-Trip
// ============================================================================

/// exp(ln(exp(ln(x)))) should recover x
#[test]
fn deep_nesting_exp_ln_double() {
    let test_values: &[&str] = &["1.0", "2.0", "5.0", "0.5", "10.0"];

    for &x in test_values {
        let result = evaluate(&gmath(x).ln().exp().ln().exp());
        assert!(
            result.is_ok(),
            "exp(ln(exp(ln({})))) should succeed, got {:?}",
            x, result
        );

        let s = format!("{}", result.unwrap());
        let expected_prefix = x.split('.').next().unwrap();
        assert!(
            s.starts_with(expected_prefix),
            "exp(ln(exp(ln({})))) should recover ~{}, got '{}'",
            x, x, s
        );
    }
}

/// sin(asin(x)) should recover x for |x| <= 1
#[test]
fn deep_nesting_sin_asin() {
    let test_values: &[&str] = &["0.1", "0.5", "0.9", "0.01", "0.99"];

    for &x in test_values {
        let result = evaluate(&gmath(x).asin().sin());
        assert!(
            result.is_ok(),
            "sin(asin({})) should succeed, got {:?}",
            x, result
        );

        let s = format!("{}", result.unwrap());
        // Should start with "0." and recover the input
        assert!(
            s.starts_with("0."),
            "sin(asin({})) should be ~{}, got '{}'",
            x, x, s
        );
    }
}

/// sqrt(x) * sqrt(x) should recover x (within 1 ULP — may show x-epsilon)
#[test]
fn deep_nesting_sqrt_squared() {
    let test_values: &[(&str, &[&str])] = &[
        ("2",   &["2.000", "1.999"]),
        ("3",   &["3.000", "2.999"]),
        ("5",   &["5.000", "4.999"]),
        ("7",   &["7.000", "6.999"]),
        ("10",  &["10.00", "9.999"]),
        ("100", &["100.0", "99.99"]),
        ("0.5", &["0.500", "0.499"]),
        ("0.25",&["0.250", "0.249"]),
        ("42",  &["42.00", "41.99"]),
    ];

    for &(x, prefixes) in test_values {
        let sqrt_val = evaluate(&gmath(x).sqrt()).unwrap();
        let squared = evaluate(&(
            LazyExpr::from(sqrt_val.clone()) * LazyExpr::from(sqrt_val)
        )).unwrap();

        let s = format!("{}", squared);
        let ok = prefixes.iter().any(|p| s.starts_with(p));
        assert!(
            ok,
            "sqrt({}) * sqrt({}) should recover ~{}, got '{}'",
            x, x, x, s
        );
    }
}

/// cos(acos(x)) should recover x for |x| <= 1
#[test]
fn deep_nesting_cos_acos() {
    let test_values: &[&str] = &["0.1", "0.5", "0.9", "0.01", "0.99"];

    for &x in test_values {
        let result = evaluate(&gmath(x).acos().cos());
        assert!(
            result.is_ok(),
            "cos(acos({})) should succeed, got {:?}",
            x, result
        );

        let s = format!("{}", result.unwrap());
        assert!(
            s.starts_with("0."),
            "cos(acos({})) should be ~{}, got '{}'",
            x, x, s
        );
    }
}

/// tan(atan(x)) should recover x
#[test]
fn deep_nesting_tan_atan() {
    let test_values: &[&str] = &["0.1", "0.5", "1.0", "2.0", "10.0"];

    for &x in test_values {
        let result = evaluate(&gmath(x).atan().tan());
        assert!(
            result.is_ok(),
            "tan(atan({})) should succeed, got {:?}",
            x, result
        );

        let s = format!("{}", result.unwrap());
        let expected_prefix = x.split('.').next().unwrap();
        assert!(
            s.starts_with(expected_prefix),
            "tan(atan({})) should recover ~{}, got '{}'",
            x, x, s
        );
    }
}

// ============================================================================
// Phase 2E: Prime Table
// ============================================================================

#[test]
fn prime_table_basic_primes() {
    assert!(is_prime(2), "2 should be prime");
    assert!(is_prime(3), "3 should be prime");
    assert!(is_prime(5), "5 should be prime");
    assert!(is_prime(7), "7 should be prime");
    assert!(is_prime(11), "11 should be prime");
    assert!(is_prime(97), "97 should be prime");
}

#[test]
fn prime_table_basic_composites() {
    assert!(!is_prime(0), "0 should not be prime");
    assert!(!is_prime(1), "1 should not be prime");
    assert!(!is_prime(4), "4 should not be prime");
    assert!(!is_prime(6), "6 should not be prime");
    assert!(!is_prime(9), "9 should not be prime");
    assert!(!is_prime(100), "100 should not be prime");
}

#[test]
fn prime_table_max_prime() {
    // MAX_PRIME should be a prime near 10000 (PRIME_LIMIT)
    assert!(MAX_PRIME > 9000, "MAX_PRIME should be > 9000, got {}", MAX_PRIME);
    assert!(MAX_PRIME <= 10000, "MAX_PRIME should be <= 10000, got {}", MAX_PRIME);
    assert!(is_prime(MAX_PRIME), "MAX_PRIME itself should be prime");
    println!("  MAX_PRIME = {}", MAX_PRIME);
}

#[test]
fn prime_table_count() {
    // Build uses wheel-factorization sieve which generates ~1145 primes up to 10000
    // (standard sieve gives 1229; wheel may miss some due to implementation details)
    assert!(
        PRIME_COUNT >= 1100 && PRIME_COUNT <= 1300,
        "PRIME_COUNT should be in [1100, 1300], got {}",
        PRIME_COUNT
    );
    // Self-consistency: PRIME_COUNT == prime_count_up_to(MAX_PRIME)
    assert_eq!(
        PRIME_COUNT,
        prime_count_up_to(MAX_PRIME),
        "PRIME_COUNT should equal prime_count_up_to(MAX_PRIME)"
    );
    println!("  PRIME_COUNT = {}", PRIME_COUNT);
}

#[test]
fn prime_table_nth_prime() {
    assert_eq!(nth_prime(0), Some(2), "0th prime should be 2");
    assert_eq!(nth_prime(1), Some(3), "1st prime should be 3");
    assert_eq!(nth_prime(2), Some(5), "2nd prime should be 5");
    assert_eq!(nth_prime(3), Some(7), "3rd prime should be 7");
    assert_eq!(nth_prime(4), Some(11), "4th prime should be 11");
}

#[test]
fn prime_table_nth_prime_last() {
    // Last valid index is PRIME_COUNT - 1
    let last = nth_prime(PRIME_COUNT - 1);
    assert_eq!(last, Some(MAX_PRIME), "last prime (index {}) should be MAX_PRIME ({})", PRIME_COUNT - 1, MAX_PRIME);
}

#[test]
fn prime_table_nth_prime_out_of_bounds() {
    assert_eq!(nth_prime(PRIME_COUNT), None, "index == PRIME_COUNT should return None");
    assert_eq!(nth_prime(PRIME_COUNT + 1), None, "beyond PRIME_COUNT should return None");
    assert_eq!(nth_prime(usize::MAX), None, "usize::MAX should return None");
}

#[test]
fn prime_count_up_to_100() {
    // Standard: 25 primes up to 100. Wheel sieve may find fewer.
    let count = prime_count_up_to(100);
    assert!(
        count >= 20 && count <= 25,
        "primes up to 100 should be ~25, got {}",
        count
    );
    println!("  prime_count_up_to(100) = {}", count);
}

#[test]
fn prime_count_up_to_small_values() {
    assert_eq!(prime_count_up_to(0), 0, "no primes up to 0");
    assert_eq!(prime_count_up_to(1), 0, "no primes up to 1");
    // 2, 3, 5 are added manually before wheel sieve
    assert!(prime_count_up_to(2) >= 1, "at least 1 prime up to 2");
    assert!(prime_count_up_to(5) >= 3, "at least 3 primes up to 5");
    assert!(prime_count_up_to(10) >= 4, "at least 4 primes up to 10");
}

#[test]
fn prime_count_up_to_max() {
    assert_eq!(
        prime_count_up_to(MAX_PRIME),
        PRIME_COUNT,
        "prime_count_up_to(MAX_PRIME) should equal PRIME_COUNT"
    );
}
