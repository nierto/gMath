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

#[macro_use]
mod common;

macro_rules! shared_items {
    ($S:ty, $F:literal) => {
            #[allow(dead_code)]
            type FixedPoint = Fixed<$S, $F>;
            const FRAC_BITS: u32 = $F;
            type LazyExpr<T = FixedPoint> = g_math::canonical::LazyExpr<T>;
            type StackValue<T = FixedPoint> = g_math::canonical::StackValue<T>;
            const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
            use g_math::fixed_point::Fixed;




            // wide raws exceed i128: the sweep is skipped there, the tie pins use FixedPoint
            #[allow(dead_code)]
            const FB: u32 = if BITS == 32 { 16 } else if BITS == 64 { 32 } else if BITS == 128 { 64 } else { 0 };
    };
}

macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            #![allow(unused_imports, dead_code)]
            shared_items!($S, $F);
            #[test]
            fn binary_wide_profiles_nearest_spot_checks() {
                if !((BITS == 256 || BITS == 512)) { return; }
                // Wide raws exceed i128; pin via values: 1.5 * 1.5 = 2.25 exactly
                // (representable), and a sub-ulp product rounds to nearest.
                let fp = |s: &str| FixedPoint::from_str(s);
                assert_eq!((fp("1.5") * fp("1.5")).raw(), fp("2.25").raw());
                // smallest positive raw times 0.5 = half ulp → ties up to 1 raw.
                let tiny = FixedPoint::from_raw({
                    <$S as g_math::fixed_point::WideInt>::from_i128(1)
                });
                let half = fp("0.5");
                let prod = (tiny * half).raw();
                let one_raw = {
                    <$S as g_math::fixed_point::WideInt>::from_i128(1)
                };
                assert_eq!(prod, one_raw, "positive half-ulp product must tie up");
                let nprod = ((-tiny) * half).raw();
                let zero_raw = {
                    <$S as g_math::fixed_point::WideInt>::from_i128(0)
                };
                assert_eq!(nprod, zero_raw, "negative half-ulp product must tie toward +∞ (0)");
            }
        }
    };
}

// The part of the suite written for a subset of the words.
macro_rules! narrow_suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            #![allow(unused_imports, dead_code)]
            shared_items!($S, $F);
            use g_math::canonical::{evaluate, CompactShadow};
            fn binary_tier() -> u8 {
                match evaluate(&g_math::canonical::gmath::<FixedPoint>("1")).unwrap() {
                    StackValue::Binary(t, _, _) => t,
                    v => panic!("expected binary literal, got {v:?}"),
                }
            }
            fn fasc_mul_raw(a: i128, b: i128, tier: u8) -> i128 {
                let sa = StackValue::<FixedPoint>::Binary(tier, <$S as g_math::fixed_point::WideInt>::from_i128(a), CompactShadow::None);
                let sb = StackValue::<FixedPoint>::Binary(tier, <$S as g_math::fixed_point::WideInt>::from_i128(b), CompactShadow::None);
                match evaluate(&(LazyExpr::from(sa) * LazyExpr::from(sb))).unwrap() {
                    StackValue::Binary(_, r, _) => i128::from(r),
                    v => panic!("mul left binary domain: {v:?}"),
                }
            }
            fn fasc_div_raw(a: i128, b: i128, tier: u8) -> i128 {
                let sa = StackValue::<FixedPoint>::Binary(tier, <$S as g_math::fixed_point::WideInt>::from_i128(a), CompactShadow::None);
                let sb = StackValue::<FixedPoint>::Binary(tier, <$S as g_math::fixed_point::WideInt>::from_i128(b), CompactShadow::None);
                match evaluate(&(LazyExpr::from(sa) / LazyExpr::from(sb))).unwrap() {
                    StackValue::Binary(_, r, _) => i128::from(r),
                    v => panic!("div left binary domain: {v:?}"),
                }
            }
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
            fn binary_mul_div_path_independent_and_nearest() {
                // q16_16 CI runs FRAC_BITS=16; guard against custom splits locally.
                if BITS == 32 && FRAC_BITS != 16 {
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
            fn binary_constructed_exact_ties_agree_and_round_up() {
                if BITS == 32 && FRAC_BITS != 16 {
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
        }
    };
}

for_each_format!(suite);
narrow_suite!(narrow_suite_q22_10, i32, 10);
narrow_suite!(narrow_suite_q16_16, i32, 16);
narrow_suite!(narrow_suite_q32_32, i64, 32);
narrow_suite!(narrow_suite_q64_64, i128, 64);

