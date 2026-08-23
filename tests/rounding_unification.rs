//! 0.5.0 rounding unification (ROADMAP item 0c) — the permanent gate.
//!
//! One rule per domain, on every path (docs/design/ROUNDING_CENSUS.md):
//! - binary: round-to-nearest, ties toward +∞
//! - decimal: banker's where rounding occurs (canonical mul is exact,
//!   tiers 1–5 div are exact-or-rational-fallback)
//! - ternary: nearest — tie-free for mul (odd scale), ties toward +∞
//!   where ties exist (div, conversion)
//!
//! The measured pre-unification divergence (48.7% of sampled products,
//! imperative vs canonical, compact) must be ZERO now — asserted here on
//! every profile, including constructed exact-tie inputs.

use g_math::canonical::{evaluate, CompactShadow, LazyExpr, StackValue};
use g_math::fixed_point::FixedPoint;

fn binary_tier() -> u8 {
    match evaluate(&g_math::canonical::gmath("1")).unwrap() {
        StackValue::Binary(t, _, _) => t,
        v => panic!("expected binary literal, got {v:?}"),
    }
}

fn fasc_mul_raw(a: i128, b: i128, tier: u8) -> i128 {
    let sa = StackValue::Binary(tier, a as _, CompactShadow::None);
    let sb = StackValue::Binary(tier, b as _, CompactShadow::None);
    match evaluate(&(LazyExpr::from(sa) * LazyExpr::from(sb))).unwrap() {
        StackValue::Binary(_, r, _) => r as i128,
        v => panic!("mul left binary domain: {v:?}"),
    }
}

fn fasc_div_raw(a: i128, b: i128, tier: u8) -> i128 {
    let sa = StackValue::Binary(tier, a as _, CompactShadow::None);
    let sb = StackValue::Binary(tier, b as _, CompactShadow::None);
    match evaluate(&(LazyExpr::from(sa) / LazyExpr::from(sb))).unwrap() {
        StackValue::Binary(_, r, _) => r as i128,
        v => panic!("div left binary domain: {v:?}"),
    }
}

#[cfg(table_format = "q16_16")]
const FB: u32 = 16; // default FRAC_BITS in CI; custom splits share the kernel
#[cfg(table_format = "q32_32")]
const FB: u32 = 32;
#[cfg(table_format = "q64_64")]
const FB: u32 = 64;
#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
const FB: u32 = 0; // wide raws exceed i128; sweep skipped, tie pins below use FixedPoint

/// Nearest, ties toward +∞ reference model at storage scale.
fn model_mul(a: i128, b: i128, fb: u32) -> i128 {
    let wide = a * b; // callers keep |a·b| within i128
    let half = 1i128 << (fb - 1);
    let rem = wide & ((1i128 << fb) - 1);
    let floor = wide >> fb;
    if rem >= half { floor + 1 } else { floor }
}

fn model_div(a: i128, b: i128, fb: u32) -> i128 {
    let num = a << fb;
    let q = num / b;
    let r2 = 2 * (num - q * b).abs();
    let babs = b.abs();
    let positive = (num < 0) == (b < 0);
    if if positive { r2 >= babs } else { r2 > babs } {
        q + if positive { 1 } else { -1 }
    } else {
        q
    }
}

#[test]
#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn binary_mul_div_path_independent_and_nearest() {
    // q16_16 CI runs FRAC_BITS=16; guard against custom splits locally.
    #[cfg(table_format = "q16_16")]
    if g_math::fixed_point::frac_config::FRAC_BITS != 16 {
        return;
    }
    let tier = binary_tier();
    let mut checked = 0u32;
    for i in 0..1500i128 {
        for j in 0..40i128 {
            let a = i * 331 + 17 + ((i & 1) * -2 + 1) * (j * 7919);
            let b = j * 257 + 29 + if i % 3 == 0 { -5000 } else { 4321 };
            if b == 0 { continue; }
            // Keep products within i128 for the model on q64_64.
            if FB == 64 && (a.abs() > 1 << 30 || b.abs() > 1 << 30) { continue; }
            let imp_mul = (FixedPoint::from_raw(a as _) * FixedPoint::from_raw(b as _)).raw() as i128;
            let can_mul = fasc_mul_raw(a, b, tier);
            let want_mul = model_mul(a, b, FB);
            assert_eq!(imp_mul, want_mul, "imperative mul vs model at ({a},{b})");
            assert_eq!(can_mul, want_mul, "canonical mul vs model at ({a},{b})");
            let imp_div = (FixedPoint::from_raw(a as _) / FixedPoint::from_raw(b as _)).raw() as i128;
            let can_div = fasc_div_raw(a, b, tier);
            let want_div = model_div(a, b, FB);
            assert_eq!(imp_div, want_div, "imperative div vs model at ({a},{b})");
            assert_eq!(can_div, want_div, "canonical div vs model at ({a},{b})");
            checked += 1;
        }
    }
    assert!(checked > 10_000, "sweep too small: {checked}");
}

#[test]
#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn binary_constructed_exact_ties_agree_and_round_up() {
    #[cfg(table_format = "q16_16")]
    if g_math::fixed_point::frac_config::FRAC_BITS != 16 {
        return;
    }
    let tier = binary_tier();
    // Multiply tie: raw(1) * raw(2^(FB-1)) — product is exactly half an ulp.
    let half_raw = 1i128 << (FB - 1);
    let imp = (FixedPoint::from_raw(1 as _) * FixedPoint::from_raw(half_raw as _)).raw() as i128;
    let can = fasc_mul_raw(1, half_raw, tier);
    assert_eq!(imp, 1, "positive mul tie must round up (ties toward +∞)");
    assert_eq!(can, 1, "canonical must agree on the tie");
    // Negative tie: −(half ulp) → toward +∞ is 0.
    let imp_n = (FixedPoint::from_raw(-1i128 as _) * FixedPoint::from_raw(half_raw as _)).raw() as i128;
    let can_n = fasc_mul_raw(-1, half_raw, tier);
    assert_eq!(imp_n, 0, "negative mul tie must round toward +∞ (to 0)");
    assert_eq!(can_n, 0, "canonical must agree on the negative tie");
    // Divide tie: raw(3) / raw(2·2^FB) = 1.5 raw → 2 (up); negative → −1.
    let two = 2i128 << FB;
    assert_eq!((FixedPoint::from_raw(3 as _) / FixedPoint::from_raw(two as _)).raw() as i128, 2);
    assert_eq!(fasc_div_raw(3, two, tier), 2);
    assert_eq!((FixedPoint::from_raw(-3i128 as _) / FixedPoint::from_raw(two as _)).raw() as i128, -1);
    assert_eq!(fasc_div_raw(-3, two, tier), -1);
    // UGOD sub-ulp regression (pre-fix: wrong-direction bump): exact
    // quotient −0.75 raw units must round to −1, never +1.
    let a = -3i128;
    let b = 4i128 << FB;
    assert_eq!((FixedPoint::from_raw(a as _) / FixedPoint::from_raw(b as _)).raw() as i128, -1);
    assert_eq!(fasc_div_raw(a, b, tier), -1);
}

#[test]
#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
fn binary_wide_profiles_nearest_spot_checks() {
    // Wide raws exceed i128; pin via values: 1.5 * 1.5 = 2.25 exactly
    // (representable), and a sub-ulp product rounds to nearest.
    let fp = |s: &str| FixedPoint::from_str(s);
    assert_eq!((fp("1.5") * fp("1.5")).raw(), fp("2.25").raw());
    // smallest positive raw times 0.5 = half ulp → ties up to 1 raw.
    let tiny = FixedPoint::from_raw({
        #[cfg(table_format = "q128_128")]
        { g_math::fixed_point::I256::from_i128(1) }
        #[cfg(table_format = "q256_256")]
        { g_math::fixed_point::I512::from_i128(1) }
    });
    let half = fp("0.5");
    let prod = (tiny * half).raw();
    let one_raw = {
        #[cfg(table_format = "q128_128")]
        { g_math::fixed_point::I256::from_i128(1) }
        #[cfg(table_format = "q256_256")]
        { g_math::fixed_point::I512::from_i128(1) }
    };
    assert_eq!(prod, one_raw, "positive half-ulp product must tie up");
    let nprod = ((-tiny) * half).raw();
    let zero_raw = {
        #[cfg(table_format = "q128_128")]
        { g_math::fixed_point::I256::from_i128(0) }
        #[cfg(table_format = "q256_256")]
        { g_math::fixed_point::I512::from_i128(0) }
    };
    assert_eq!(nprod, zero_raw, "negative half-ulp product must tie toward +∞ (0)");
}
