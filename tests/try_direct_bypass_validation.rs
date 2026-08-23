//! 0.5.0 item 2 — fallible composed transcendentals bypass FASC (ROADMAP).
//!
//! The `try_*` variants of the composed transcendentals (tan, asin, acos,
//! sinh, cosh, tanh, asinh, acosh, atanh, plus atan) previously routed
//! through the FASC pipeline (`try_apply_unary(LazyExpr)`); they are now
//! direct compute-tier compositions mirroring the infallible methods.
//!
//! Contract pinned here:
//! 1. On in-domain inputs, `try_x(v)` is BIT-IDENTICAL to the infallible
//!    `x(v)` (which is 0-ULP gated in fasc_ulp_validation) — same engines,
//!    same composition, same single downscale.
//! 2. Domain violations return `Err(DomainError)` (0.4.27 contract), never
//!    panic: asin/acos beyond |1|, acosh below 1, atanh at/beyond |1|.
//! 3. Storage overflow returns `Err(TierOverflow)`, never a panic and never
//!    a silent wrap (0.5.0 item 1 contract).
//! 4. Boundary values: asin(±1) = ±π/2, acos(1) = 0 — no division by
//!    sqrt(0) on the way.

use g_math::fixed_point::{FixedPoint, OverflowDetected};

fn fp(s: &str) -> FixedPoint {
    FixedPoint::from_str(s)
}

/// In-domain inputs valid for EVERY function under test on EVERY profile
/// (|x| < 1 for the asin/acos/atanh subset; small enough for Q8.24).
const UNIT_INTERVAL: &[&str] = &["0.5", "-0.5", "0.25", "-0.75", "0"];
/// Inputs for the unrestricted functions (tan/atan/sinh/cosh/tanh/asinh).
const GENERAL: &[&str] = &["0.5", "-0.5", "1.5", "-1.25", "2", "0"];

#[test]
fn try_variants_bit_identical_to_infallible() {
    for s in GENERAL {
        let v = fp(s);
        assert_eq!(v.try_tan().unwrap(), v.tan(), "tan({s})");
        assert_eq!(v.try_atan().unwrap(), v.atan(), "atan({s})");
        assert_eq!(v.try_sinh().unwrap(), v.sinh(), "sinh({s})");
        assert_eq!(v.try_cosh().unwrap(), v.cosh(), "cosh({s})");
        assert_eq!(v.try_tanh().unwrap(), v.tanh(), "tanh({s})");
        assert_eq!(v.try_asinh().unwrap(), v.asinh(), "asinh({s})");
    }
    for s in UNIT_INTERVAL {
        let v = fp(s);
        assert_eq!(v.try_asin().unwrap(), v.asin(), "asin({s})");
        assert_eq!(v.try_acos().unwrap(), v.acos(), "acos({s})");
        assert_eq!(v.try_atanh().unwrap(), v.atanh(), "atanh({s})");
    }
    for s in &["1", "1.5", "2", "10"] {
        let v = fp(s);
        assert_eq!(v.try_acosh().unwrap(), v.acosh(), "acosh({s})");
    }
    // tanh deep in saturation territory still matches the infallible twin
    // (wide profiles resolve 1 - 2e-26; narrow ones round to exactly 1).
    let v = fp("30");
    assert_eq!(v.try_tanh().unwrap(), v.tanh(), "tanh(30)");
}

#[test]
fn domain_violations_are_domain_errors() {
    for (name, r) in [
        ("asin(1.5)", fp("1.5").try_asin()),
        ("asin(-1.5)", fp("-1.5").try_asin()),
        ("acos(1.5)", fp("1.5").try_acos()),
        ("acos(-1.5)", fp("-1.5").try_acos()),
        ("acosh(0.5)", fp("0.5").try_acosh()),
        ("acosh(-2)", fp("-2").try_acosh()),
        ("atanh(1)", fp("1").try_atanh()),
        ("atanh(-1)", fp("-1").try_atanh()),
        ("atanh(2)", fp("2").try_atanh()),
    ] {
        match r {
            Err(OverflowDetected::DomainError) => {}
            other => panic!("{name}: expected Err(DomainError), got {other:?}"),
        }
    }
}

#[test]
fn asin_acos_boundaries_are_exact() {
    // asin(±1) = ±π/2 via the boundary shortcut (no division by sqrt(0)).
    //
    // No decimal literal can serve as a sub-ulp π/2 reference on every
    // profile (the parser caps at 76 fractional digits; scientific needs
    // 77+), so the numeric grounding is structural: acos(0) computes
    // π/2 − atan(0) through the GENERAL branch — the same pi-half constant
    // and the same single downscale — and must agree with the asin(1)
    // shortcut bit-for-bit. The constant itself is 0-ULP mpmath-gated in
    // fasc_ulp_validation via the infallible asin/acos.
    let asin_one = fp("1").try_asin().unwrap();
    assert_eq!(
        asin_one,
        fp("0").try_acos().unwrap(),
        "asin(1) shortcut must equal acos(0) general path"
    );
    assert!(asin_one > fp("1.57"), "asin(1) magnitude sanity");
    let asin_neg_one = fp("-1").try_asin().unwrap();
    assert_eq!(
        asin_one + asin_neg_one,
        FixedPoint::ZERO,
        "asin(1) + asin(-1) must cancel exactly"
    );
    // acos(1) = π/2 − π/2 = exactly 0.
    assert_eq!(fp("1").try_acos().unwrap(), FixedPoint::ZERO, "acos(1)");
}

#[test]
fn overflow_is_loud_never_wrapped() {
    // cosh(180) ≈ 3.7e77 exceeds even scientific Q256.256 (max ~5.8e76);
    // every narrower profile overflows earlier still. The contract is a
    // typed error — never a panic, never a wrapped value.
    match fp("180").try_cosh() {
        Err(OverflowDetected::TierOverflow) => {}
        other => panic!("try_cosh(180): expected Err(TierOverflow), got {other:?}"),
    }
    match fp("180").try_sinh() {
        Err(OverflowDetected::TierOverflow) => {}
        other => panic!("try_sinh(180): expected Err(TierOverflow), got {other:?}"),
    }
}
