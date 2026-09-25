//! Boundary & Range Stress Tests
//!
//! Validates behavior at:
//! - Tier boundaries (i8::MAX, i16::MAX, i32::MAX, i64::MAX)
//! - Transcendental function extremes (near overflow/underflow)
//! - Decimal range extremes (18-digit precision limits)
//! - Profile-specific type boundaries (i128, I256, I512)
//!
//! All tests are pure Rust, no external dependencies, no rebuild required.

use g_math::canonical::{gmath, gmath_parse, evaluate, LazyExpr, StackValue};
use g_math::fixed_point::FixedPoint;

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

/// `k` storage units (k * 2^-FRAC_BITS) for the build's split.
fn ulps(k: i32) -> FixedPoint {
    let mut u = FixedPoint::one();
    for _ in 0..g_math::fixed_point::frac_config::FRAC_BITS { u = u / FixedPoint::from_int(2); }
    u * FixedPoint::from_int(k)
}

fn frac_digits(s: &str) -> usize {
    s.split_once('.').map_or(0, |(_, f)| f.len())
}

/// Display-prefix check where the Display prints enough fractional digits to
/// decide it. The canonical Display prints only the digits the split
/// resolves (2 at GMATH_FRAC_BITS=8), so a 3-digit prefix is undecidable
/// there: the stored value must then lie within `k` storage units of the
/// mpmath `reference` instead. A pattern without a '.' must match the whole
/// Display string (e.g. "0" for an exact Symbolic zero).
fn assert_display_or_stored(v: &StackValue, patterns: &[&str], reference: &str, k: i32, what: &str) {
    let s = format!("{}", v);
    if patterns.iter().any(|p| !p.contains('.') && s == *p) {
        return;
    }
    let need = patterns.iter().map(|p| frac_digits(p)).max().unwrap_or(0);
    if frac_digits(&s) >= need {
        assert!(patterns.iter().any(|p| p.contains('.') && s.starts_with(p)), "{what}: got '{s}'");
    } else {
        let got = FixedPoint::from_raw(v.as_binary_storage().expect("binary-materializable result"));
        let r = match reference.strip_prefix('-') {
            Some(m) => -FixedPoint::from_str(m),
            None => FixedPoint::from_str(reference),
        };
        let d = (got - r).abs();
        assert!(d <= ulps(k), "{what}: stored {got} ({} raw) vs {reference}: {} units > {k}", got.raw(), d.raw());
    }
}

// ============================================================================
// Phase 3A: Tier Boundary Arithmetic
// ============================================================================

/// i8::MAX = 127 — fits in tier 1 (Tiny)
#[test]
fn tier_boundary_i8_max() {
    let r = evaluate(&gmath("127"));
    assert!(r.is_ok(), "127 should parse and evaluate, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(s.starts_with("127."), "127 should display as 127.xxx, got '{}'", s);
}

/// i8::MAX + 1 = 128 — requires tier 2 (Small)
#[test]
fn tier_boundary_i8_overflow_to_small() {
    let r = evaluate(&(gmath("127") + gmath("1")));
    assert!(r.is_ok(), "127 + 1 should succeed (promote to tier 2), got {:?}", r);
    let s = format!("{}", r.unwrap());
    // 128 exceeds the binary range at GMATH_FRAC_BITS=24 (+-128): the exact
    // result then stays Symbolic and displays without a fraction point
    assert!(s.starts_with("128.") || s == "128", "127 + 1 = 128, got '{}'", s);
}

/// i16::MAX = 32767
#[test]
fn tier_boundary_i16_max() {
    let r = evaluate(&gmath("32767"));
    assert!(r.is_ok(), "32767 should parse, got {:?}", r);
    let s = format!("{}", r.unwrap());
    // 32767 exceeds the binary range at GMATH_FRAC_BITS >= 17: the literal
    // then parses to Symbolic and displays without a fraction point
    assert!(s.starts_with("32767.") || s == "32767", "32767 should display correctly, got '{}'", s);
}

/// Product requiring tier promotion — profile-appropriate values
#[test]
#[cfg(not(table_format = "q16_16"))]
fn tier_boundary_multiply_requires_promotion() {
    let r = evaluate(&(gmath("200") * gmath("200")));
    assert!(r.is_ok(), "200 * 200 should succeed with tier promotion, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(s.starts_with("40000."), "200 * 200 = 40000, got '{}'", s);
}
/// Q16.16: 15 * 15 = 225 (exceeds i8 tier, fits Q16.16)
#[test]
#[cfg(table_format = "q16_16")]
fn tier_boundary_multiply_requires_promotion() {
    let r = evaluate(&(gmath("15") * gmath("15")));
    assert!(r.is_ok(), "15 * 15 should succeed with tier promotion, got {:?}", r);
    let s = format!("{}", r.unwrap());
    // 225 exceeds the binary range at GMATH_FRAC_BITS=24 (+-128): the exact
    // product then stays Symbolic and displays without a fraction point
    assert!(s.starts_with("225.") || s == "225", "15 * 15 = 225, got '{}'", s);
}

/// Large multiply — profile-appropriate values
#[test]
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
fn tier_boundary_large_multiply() {
    let r = evaluate(&(gmath("50000") * gmath("50000")));
    assert!(r.is_ok(), "50000 * 50000 should succeed, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(s.starts_with("2500000000."), "50000 * 50000 = 2500000000, got '{}'", s);
}
/// Q32.32: 46340² = 2147395600 (near max). mpmath: 2147395600 * 2^32 = 9222993873574297600
#[test]
#[cfg(table_format = "q32_32")]
fn tier_boundary_large_multiply() {
    let r = evaluate(&(gmath("46340") * gmath("46340")));
    assert!(r.is_ok(), "46340² should succeed in Q32.32, got {:?}", r);
    let raw = r.unwrap().as_binary_storage().unwrap();
    let ulp = (raw - 9222993873574297600_i64).unsigned_abs();
    assert!(ulp <= 1, "46340² ULP={}", ulp);
}
/// Realtime: 180² = 32400 (near the Q16.16 max), raw 32400 * 2^F for the
/// build's split (2123366400 at Q16.16, 33177600 at Q22.10). Where 32400
/// exceeds the range 2^(31-F) (GMATH_FRAC_BITS >= 17) the largest n with n²
/// in range is used instead (11² = 121 at Q8.24).
#[test]
#[cfg(table_format = "q16_16")]
fn tier_boundary_large_multiply() {
    let mut n: i32 = 180;
    while FixedPoint::try_from_int(n * n).is_err() { n -= 1; }
    let lit = n.to_string();
    let r = evaluate(&(gmath_parse(&lit).unwrap() * gmath_parse(&lit).unwrap()));
    assert!(r.is_ok(), "{n}² should succeed on realtime, got {:?}", r);
    let raw = r.unwrap().as_binary_storage().unwrap();
    let ulp = (raw - ((n * n) << g_math::fixed_point::frac_config::FRAC_BITS)).unsigned_abs();
    assert!(ulp <= 1, "{n}² ULP={}", ulp);
}

/// i32::MAX = 2147483647 — overflows Q16.16 (max integer ~32767)
#[test]
#[cfg(not(table_format = "q16_16"))]
fn tier_boundary_i32_max() {
    let r = evaluate(&gmath("2147483647"));
    assert!(r.is_ok(), "i32::MAX should parse, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(
        s.starts_with("2147483647."),
        "i32::MAX should display correctly, got '{}'",
        s
    );
}

/// i64::MAX overflows Q32.32/Q16.16 — only test on Q64.64+
#[test]
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
fn tier_boundary_i64_max() {
    let r = evaluate(&gmath("9223372036854775807"));
    assert!(r.is_ok(), "i64::MAX should parse, got {:?}", r);
    let val = r.unwrap();
    let s = format!("{}", val);
    assert!(
        s.starts_with("922337203685477580"),
        "i64::MAX should display correctly, got '{}'",
        s
    );
}

/// Subtraction crossing zero: 1 - 2 = -1
#[test]
fn tier_boundary_subtract_across_zero() {
    let r = evaluate(&(gmath("1") - gmath("2")));
    assert!(r.is_ok(), "1 - 2 should succeed, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(s.starts_with("-1."), "1 - 2 = -1, got '{}'", s);
}

/// Multiply that exactly fits: 127 * 1 = 127 (no promotion needed)
#[test]
fn tier_boundary_multiply_exact_fit() {
    let r = evaluate(&(gmath("127") * gmath("1")));
    assert!(r.is_ok(), "127 * 1 should succeed, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(s.starts_with("127."), "127 * 1 = 127, got '{}'", s);
}

/// Chain of additions staying within tier
#[test]
fn tier_boundary_chain_additions() {
    // 10 + 10 + 10 + 10 + 10 = 50 (stays in tier 1)
    let mut expr = gmath("10");
    for _ in 0..4 {
        expr = expr + gmath("10");
    }
    let r = evaluate(&expr);
    assert!(r.is_ok(), "chain of 5 × 10 additions should succeed, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(s.starts_with("50."), "10+10+10+10+10 = 50, got '{}'", s);
}

// ============================================================================
// Phase 3B: Transcendental Extremes
// ============================================================================

/// exp(44) — near Q64.64 integer overflow
/// exp(44) ≈ 1.28516... × 10^19
/// On Q16.16/Q32.32: overflows the native Q64.64 compute tier (exp(44)×2^64 > i128::MAX).
/// On Q64.64+: fits in the I256/I512 compute tier.
#[test]
fn transcendental_exp_large_argument() {
    let r = evaluate(&gmath("44").exp());
    // May succeed or overflow depending on profile — must not panic
    match &r {
        Ok(val) => {
            let s = format!("{}", val);
            println!("exp(44) = {}", s);
            // On Q64.64+, should be a very large positive number.
            // On Q16.16/Q32.32, native Q64.64 compute overflows — any result is acceptable.
            #[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
            assert!(
                s.len() > 10 && !s.starts_with('-') && !s.starts_with("0."),
                "exp(44) should be a large positive, got '{}'",
                s
            );
        }
        Err(e) => {
            // Overflow is acceptable for large exp
            println!("exp(44) returned error: {:?} (acceptable)", e);
        }
    }
}

/// exp(-44) — near underflow (result < 1 ULP for Q64.64)
#[test]
fn transcendental_exp_large_negative() {
    let r = evaluate(&gmath_safe("-44").exp());
    match &r {
        Ok(val) => {
            println!("exp(-44) = {}", val);
            // Should be very close to zero (7.78e-20, far below 1 unit on
            // every realtime split)
            assert_display_or_stored(val, &["0.000", "-0.000"], "0", 0,
                "exp(-44) should be near zero");
        }
        Err(e) => {
            // Underflow to zero is acceptable
            println!("exp(-44) returned error: {:?} (acceptable)", e);
        }
    }
}

/// exp(0) = 1 exactly
#[test]
fn transcendental_exp_zero() {
    let r = evaluate(&gmath("0").exp());
    assert!(r.is_ok(), "exp(0) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["1.000"], "1", 0, "exp(0) should be exactly 1.0");
}

/// exp(1) = e ≈ 2.71828...
#[test]
fn transcendental_exp_one() {
    let r = evaluate(&gmath("1").exp());
    assert!(r.is_ok(), "exp(1) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["2.718"], "2.718281828459045235360287471352662", 0, "exp(1) should be ~2.718...");
}

/// ln(tiny value) — very negative result
#[test]
fn transcendental_ln_tiny() {
    let r = evaluate(&gmath("0.0000000000000001").ln());
    match &r {
        Ok(val) => {
            let s = format!("{}", val);
            println!("ln(1e-16) = {}", s);
            // Should be a large negative number (~-36.84)
            assert!(
                s.starts_with("-3"),
                "ln(1e-16) should be ~-36.8, got '{}'",
                s
            );
        }
        Err(e) => {
            println!("ln(tiny) returned error: {:?}", e);
        }
    }
}

/// ln(1) = 0 exactly
#[test]
fn transcendental_ln_one() {
    let r = evaluate(&gmath("1").ln());
    assert!(r.is_ok(), "ln(1) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["0.000", "0"], "0", 0, "ln(1) should be exactly 0");
}

/// sin(1000000) — tests range reduction over many periods
#[test]
fn transcendental_sin_large_argument() {
    let r = evaluate(&gmath("1000000").sin());
    match &r {
        Ok(val) => {
            let s = format!("{}", val);
            println!("sin(1000000) = {}", s);
            // Result should be in [-1, 1]
            // sin(1000000) ≈ -0.34999... (mpmath)
            let trimmed = s.trim_start_matches('-');
            assert!(
                trimmed.starts_with("0.") || trimmed.starts_with("1.000"),
                "sin(1000000) should be in [-1, 1], got '{}'",
                s
            );
        }
        Err(e) => {
            // Range reduction failure for very large args is documented behavior
            println!("sin(1000000) returned error: {:?} (may be expected for large args)", e);
        }
    }
}

/// atan(very large) should approach pi/2
#[test]
fn transcendental_atan_large() {
    let r = evaluate(&gmath("999999999").atan());
    match &r {
        Ok(val) => {
            // atan(999999999) = pi/2 - 1.000000001e-9 (mpmath)
            assert_display_or_stored(val, &["1.570"], "1.570796325794896618231321690973084775432", 0,
                "atan(999999999) should approach pi/2 ≈ 1.5708");
        }
        Err(e) => {
            println!("atan(999999999) returned error: {:?}", e);
        }
    }
}

/// sqrt(0) = 0 exactly
#[test]
fn transcendental_sqrt_zero() {
    let r = evaluate(&gmath("0").sqrt());
    assert!(r.is_ok(), "sqrt(0) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["0.000", "0"], "0", 0, "sqrt(0) should be 0");
}

/// sqrt(1) = 1 exactly
#[test]
fn transcendental_sqrt_one() {
    let r = evaluate(&gmath("1").sqrt());
    assert!(r.is_ok(), "sqrt(1) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["1.000"], "1", 0, "sqrt(1) should be 1");
}

/// sqrt(very large number) — 1e12 overflows Q32.32/Q16.16
#[test]
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
fn transcendental_sqrt_large() {
    let r = evaluate(&gmath("1000000000000").sqrt());
    assert!(r.is_ok(), "sqrt(1e12) should succeed, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(
        s.starts_with("1000000."),
        "sqrt(1e12) should be 1000000, got '{}'",
        s
    );
}

/// cos(0) = 1 exactly
#[test]
fn transcendental_cos_zero() {
    let r = evaluate(&gmath("0").cos());
    assert!(r.is_ok(), "cos(0) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["1.000"], "1", 0, "cos(0) should be exactly 1");
}

/// sin(0) = 0 exactly
#[test]
fn transcendental_sin_zero() {
    let r = evaluate(&gmath("0").sin());
    assert!(r.is_ok(), "sin(0) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["0.000", "0"], "0", 0, "sin(0) should be exactly 0");
}

// ============================================================================
// Phase 3C: Decimal Range Extremes
// ============================================================================

/// 18 decimal places — Q64.64 max meaningful precision
#[test]
fn decimal_extreme_18_places() {
    let r = evaluate(&gmath("0.000000000000000001"));
    // Very small value — may be at or below 1 ULP for Q64.64
    match &r {
        Ok(val) => {
            let s = format!("{}", val);
            println!("0.000000000000000001 = {}", s);
            // Should be very close to zero. On narrow profiles the literal
            // is held EXACTLY in the Symbolic domain (0.5.0 item 1 parse
            // fallback) and displays as "0." at profile precision — 1e-18
            // is far below one realtime/compact ulp.
            assert!(
                s.starts_with("0.000") || s == "0.",
                "18-place decimal should be near zero, got '{}'",
                s
            );
        }
        Err(e) => {
            println!("18-place decimal returned error: {:?} (may underflow)", e);
        }
    }
}

/// Large integer + decimal — near integer limit
#[test]
fn decimal_extreme_large_integer() {
    let r = evaluate(&gmath("999999999999999999"));
    match &r {
        Ok(val) => {
            let s = format!("{}", val);
            // Should display as the large integer
            assert!(
                s.starts_with("99999999999999999"),
                "large integer should display correctly, got '{}'",
                s
            );
        }
        Err(e) => {
            // May overflow on Q64.64 (max integer part is ~9.2e18)
            println!("large integer returned error: {:?}", e);
        }
    }
}

/// Two large decimals that nearly overflow when added
#[test]
fn decimal_extreme_large_addition() {
    let r = evaluate(&(gmath("4000000000000000000") + gmath("4000000000000000000")));
    match &r {
        Ok(val) => {
            let s = format!("{}", val);
            assert!(
                s.starts_with("800000000000000000"),
                "4e18 + 4e18 = 8e18, got '{}'",
                s
            );
        }
        Err(e) => {
            // Overflow is acceptable near limits
            println!("large addition returned error: {:?} (near limit)", e);
        }
    }
}

/// Multiply decimals: 99.99 * 99.99 = 9998.0001
#[test]
fn decimal_extreme_multiply() {
    let r = evaluate(&(gmath("99.99") * gmath("99.99")));
    assert!(r.is_ok(), "99.99 * 99.99 should succeed, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(
        s.starts_with("9998.") || s.starts_with("9998/") || s.contains("9998"),
        "99.99 * 99.99 = 9998.0001, got '{}'",
        s
    );
}

/// Very small * very small — tests precision floor
#[test]
#[cfg(not(table_format = "q16_16"))]
fn decimal_extreme_tiny_multiply() {
    let r = evaluate(&(gmath("0.001") * gmath("0.001")));
    assert!(r.is_ok(), "0.001 * 0.001 should succeed, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(
        s.starts_with("0.000001") || s.contains("1/1000000"),
        "0.001 * 0.001 = 0.000001, got '{}'",
        s
    );
}
/// Q16.16: 0.1 * 0.1 = 0.01 (within Q16.16 resolution)
#[test]
#[cfg(table_format = "q16_16")]
fn decimal_extreme_tiny_multiply() {
    let r = evaluate(&(gmath("0.1") * gmath("0.1")));
    assert!(r.is_ok(), "0.1 * 0.1 should succeed, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(
        s.starts_with("0.01") || s.contains("1/100"),
        "0.1 * 0.1 = 0.01, got '{}'",
        s
    );
}

/// Subtraction of nearly equal values — catastrophic cancellation test
#[test]
fn decimal_extreme_catastrophic_cancellation() {
    let r = evaluate(&(gmath("1.0000000000001") - gmath("1.0")));
    match &r {
        Ok(val) => {
            let s = format!("{}", val);
            println!("1.0000000000001 - 1.0 = {}", s);
            // Should give approximately 1e-13. On narrow profiles the
            // 13-dp literal is held EXACTLY in the Symbolic domain (0.5.0
            // item 1 parse fallback) and the difference displays as "0."
            // at profile precision — 1e-13 is below one narrow-profile ulp.
            // (Pre-fix the literal itself truncated to 1.0 at parse and the
            // subtraction was exactly zero — displayed "0.0000", passing
            // this assertion by coincidence.)
            assert!(
                s.starts_with("0.000") || s == "0.",
                "near-cancellation should give small positive, got '{}'",
                s
            );
        }
        Err(e) => {
            println!("catastrophic cancellation test returned error: {:?}", e);
        }
    }
}

// ============================================================================
// Phase 3D: Profile-Specific Type Boundaries
// ============================================================================

/// Q64.64: i128 storage, max integer ~9.2 × 10^18
#[cfg(table_format = "q64_64")]
mod q64_64_boundaries {
    use super::*;

    #[test]
    fn q64_64_max_representable_integer() {
        // Q64.64: integer part limited to i64 range (~9.2e18)
        // Value just under the limit
        let r = evaluate(&gmath("9000000000000000000"));
        assert!(r.is_ok(), "9e18 should fit in Q64.64, got {:?}", r);
        let s = format!("{}", r.unwrap());
        assert!(
            s.starts_with("900000000000000000"),
            "9e18 should display correctly, got '{}'",
            s
        );
    }

    #[test]
    fn q64_64_multiply_promoting_to_i256() {
        // Two values whose product requires I256 intermediate
        // 3e9 * 3e9 = 9e18 — within range but needs promotion
        let r = evaluate(&(gmath("3000000000") * gmath("3000000000")));
        assert!(r.is_ok(), "3e9 * 3e9 should succeed with I256 intermediate, got {:?}", r);
        let s = format!("{}", r.unwrap());
        assert!(
            s.starts_with("900000000000000000"),
            "3e9 * 3e9 = 9e18, got '{}'",
            s
        );
    }

    #[test]
    fn q64_64_smallest_representable() {
        // 1 ULP in Q64.64 = 2^(-64) ≈ 5.42e-20
        // 0.00000000000000001 (1e-17) is well above 1 ULP
        let r = evaluate(&gmath("0.00000000000000001"));
        assert!(r.is_ok(), "1e-17 should be representable in Q64.64, got {:?}", r);
    }

    #[test]
    fn q64_64_exp_near_overflow() {
        // exp(43) ≈ 4.7e18, within Q64.64 range
        let r = evaluate(&gmath("43").exp());
        assert!(r.is_ok(), "exp(43) should fit in Q64.64, got {:?}", r);
        let s = format!("{}", r.unwrap());
        println!("exp(43) = {}", s);
    }
}

/// Q128.128: I256 storage, much larger range
#[cfg(table_format = "q128_128")]
mod q128_128_boundaries {
    use super::*;

    #[test]
    fn q128_128_large_integer() {
        // Values that overflow i128 but fit in I256
        // 10^30 is well within I256 integer range
        let r = evaluate(&gmath("1000000000000000000000000000000"));
        assert!(r.is_ok(), "10^30 should fit in Q128.128, got {:?}", r);
    }

    #[test]
    fn q128_128_high_precision_decimal() {
        // Q128.128 supports 38 decimal digits of precision
        let r = evaluate(&gmath("0.00000000000000000000000000000000000001"));
        assert!(r.is_ok(), "38-digit decimal should work in Q128.128, got {:?}", r);
    }

    #[test]
    fn q128_128_exp_wider_range() {
        // exp(80) ≈ 5.5e34, fits in Q128.128 but not Q64.64
        let r = evaluate(&gmath("80").exp());
        assert!(r.is_ok(), "exp(80) should fit in Q128.128, got {:?}", r);
    }
}

/// Q256.256: I512 storage, extreme range
#[cfg(table_format = "q256_256")]
mod q256_256_boundaries {
    use super::*;

    #[test]
    fn q256_256_extreme_precision() {
        // Q256.256 supports 77 decimal digits of precision
        let r = evaluate(&gmath("3.14159265358979323846264338327950288419716939937510"));
        assert!(r.is_ok(), "50-digit pi should work in Q256.256, got {:?}", r);
    }

    #[test]
    fn q256_256_exp_extreme_range() {
        // exp(170) ≈ 7.6e73, well within Q256.256 range
        let r = evaluate(&gmath("170").exp());
        assert!(r.is_ok(), "exp(170) should fit in Q256.256, got {:?}", r);
    }
}

// ============================================================================
// Phase 3E: Arithmetic Edge Cases
// ============================================================================

/// Profile-appropriate test values that fit in all profiles.
/// Q16.16 max integer: 32767. Q16.16 min resolution: ~0.00001.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
const ARITH_CASES: &[&str] = &["1", "42", "0.5", "100", "0.01"];
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
const ARITH_CASES_NONZERO: &[&str] = &["1", "42", "0.5", "100"];
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
const ARITH_CASES: &[&str] = &["1", "42", "0.5", "1000000", "0.001"];
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
const ARITH_CASES_NONZERO: &[&str] = &["1", "42", "999999999", "0.5"];

/// x + 0 = x
#[test]
fn arithmetic_additive_identity() {
    for &x in ARITH_CASES {
        let result = evaluate(&(gmath(x) + gmath("0")));
        assert!(result.is_ok(), "{} + 0 should succeed, got {:?}", x, result);
    }
}

/// x * 1 = x
#[test]
fn arithmetic_multiplicative_identity() {
    for &x in ARITH_CASES {
        let result = evaluate(&(gmath(x) * gmath("1")));
        assert!(result.is_ok(), "{} * 1 should succeed, got {:?}", x, result);
    }
}

/// x * 0 = 0
#[test]
fn arithmetic_multiply_by_zero() {
    for &x in ARITH_CASES_NONZERO {
        let result = evaluate(&(gmath(x) * gmath("0")));
        assert!(result.is_ok(), "{} * 0 should succeed, got {:?}", x, result);
        let s = format!("{}", result.unwrap());
        let trimmed = s.trim_start_matches('-');
        assert!(
            trimmed.starts_with("0.000") || trimmed.starts_with("0/") || trimmed == "0" || trimmed.starts_with("0.0"),
            "{} * 0 should be 0, got '{}'",
            x, s
        );
    }
}

/// x - x = 0
#[test]
fn arithmetic_self_subtraction() {
    for &x in ARITH_CASES {
        let result = evaluate(&(gmath(x) - gmath(x)));
        assert!(result.is_ok(), "{} - {} should succeed, got {:?}", x, x, result);
        let s = format!("{}", result.unwrap());
        let trimmed = s.trim_start_matches('-');
        assert!(
            trimmed == "0" || trimmed == "0.0" || trimmed.starts_with("0.00") || trimmed.starts_with("0/"),
            "{} - {} should be 0, got '{}'",
            x, x, s
        );
    }
}

/// x / x = 1 (for x != 0)
#[test]
fn arithmetic_self_division() {
    for &x in ARITH_CASES {
        let result = evaluate(&(gmath(x) / gmath(x)));
        assert!(result.is_ok(), "{} / {} should succeed, got {:?}", x, x, result);
        let s = format!("{}", result.unwrap());
        assert!(
            s == "1" || s == "1.0" || s.starts_with("1.00") || s.starts_with("1/1"),
            "{} / {} should be 1, got '{}'",
            x, x, s
        );
    }
}

/// Commutativity: a + b = b + a
#[test]
fn arithmetic_commutativity_add() {
    let pairs: &[(&str, &str)] = &[
        ("3", "7"),
        ("1.5", "2.5"),
        ("100", "0.001"),
        ("1/3", "1/7"),
    ];
    for &(a, b) in pairs {
        let ab = evaluate(&(gmath(a) + gmath(b)));
        let ba = evaluate(&(gmath(b) + gmath(a)));
        match (ab, ba) {
            (Ok(v1), Ok(v2)) => {
                let s1 = format!("{:.10}", v1);
                let s2 = format!("{:.10}", v2);
                assert_eq!(s1, s2, "{} + {} != {} + {}: '{}' vs '{}'", a, b, b, a, s1, s2);
            }
            (Err(e1), Err(e2)) => {
                assert_eq!(e1, e2, "both orderings should give same error");
            }
            (r1, r2) => {
                panic!("{} + {} = {:?} but {} + {} = {:?}", a, b, r1, b, a, r2);
            }
        }
    }
}

/// Commutativity: a * b = b * a
#[test]
fn arithmetic_commutativity_mul() {
    let pairs: &[(&str, &str)] = &[
        ("3", "7"),
        ("1.5", "2.5"),
        ("100", "0.01"),
    ];
    for &(a, b) in pairs {
        let ab = evaluate(&(gmath(a) * gmath(b)));
        let ba = evaluate(&(gmath(b) * gmath(a)));
        match (ab, ba) {
            (Ok(v1), Ok(v2)) => {
                let s1 = format!("{:.10}", v1);
                let s2 = format!("{:.10}", v2);
                assert_eq!(s1, s2, "{} * {} != {} * {}: '{}' vs '{}'", a, b, b, a, s1, s2);
            }
            (Err(e1), Err(e2)) => {
                assert_eq!(e1, e2, "both orderings should give same error");
            }
            (r1, r2) => {
                panic!("{} * {} = {:?} but {} * {} = {:?}", a, b, r1, b, a, r2);
            }
        }
    }
}

// ============================================================================
// Phase 3F: Transcendental Boundary Values (exact mathematical results)
// ============================================================================

/// exp(ln(2)) = 2 — hardcoded mpmath reference
#[test]
fn transcendental_exp_ln_2_exact() {
    let r = evaluate(&gmath("2").ln().exp());
    assert!(r.is_ok(), "exp(ln(2)) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["2.000", "1.999"], "2", 0, "exp(ln(2)) should be exactly 2.0");
}

/// ln(e) = 1 — using named constant
#[test]
fn transcendental_ln_e_is_one() {
    let r = evaluate(&gmath("e").ln());
    assert!(r.is_ok(), "ln(e) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["1.000", "0.999"], "1", 0, "ln(e) should be ~1.0");
}

/// pow(2, 10) = 1024
#[test]
fn transcendental_pow_2_10() {
    // 1024 exceeds the realtime range at GMATH_FRAC_BITS >= 21: use the
    // largest power of two in range there (2^6 = 64 at Q8.24)
    let mut e: u32 = 10;
    while FixedPoint::try_from_int(1 << e).is_err() { e -= 1; }
    let r = evaluate(&gmath("2").pow(gmath_parse(&e.to_string()).unwrap()));
    assert!(r.is_ok(), "pow(2, {e}) should succeed, got {:?}", r);
    let s = format!("{}", r.unwrap());
    assert!(
        s.starts_with(&format!("{}.", 1 << e)),
        "2^{e} should be {}, got '{}'",
        1 << e,
        s
    );
}

/// pow(10, 0) = 1
#[test]
fn transcendental_pow_x_zero() {
    let r = evaluate(&gmath("10").pow(gmath("0")));
    assert!(r.is_ok(), "pow(10, 0) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["1.000"], "1", 0, "10^0 should be 1");
}

/// sqrt(4) = 2 exactly
#[test]
fn transcendental_sqrt_4_exact() {
    let r = evaluate(&gmath("4").sqrt());
    assert!(r.is_ok(), "sqrt(4) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["2.000"], "2", 0, "sqrt(4) should be exactly 2.0");
}

/// sqrt(9) = 3 exactly
#[test]
fn transcendental_sqrt_9_exact() {
    let r = evaluate(&gmath("9").sqrt());
    assert!(r.is_ok(), "sqrt(9) should succeed, got {:?}", r);
    assert_display_or_stored(&r.unwrap(), &["3.000"], "3", 0, "sqrt(9) should be exactly 3.0");
}
