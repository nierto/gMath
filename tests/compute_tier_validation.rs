//! Validation for the public `compute_tier` module (0.4.32).
//!
//! mpmath 60-digit references (tests/data/compute_tier_refs.rs, generated;
//! see the header of that file) at the two profiles the inference surface
//! targets, matching the RowScaledTQ19/matvec_q2f precedent:
//! realtime (one table per GMATH_FRAC_BITS from 2 to 30, COMPUTE_FRAC_BITS =
//! 2 x FRAC_BITS; the tables were Q16.16 only until 0.6.5, so this suite
//! failed at every other split) and q32_32 (FRAC_BITS=32, COMPUTE_FRAC_BITS=64).
//!
//! Gates:
//! - compute-tier results vs mpmath, in compute-tier ULP
//! - storage-level results after `to_fixed` vs mpmath, exact (0 LSB) for all functions
//! - path independence vs the imperative `FixedPoint` methods (bit-identical)
//! - domain-violation panics and saturation behavior
//!
//! Requires: `cargo test --features inference` under GMATH_PROFILE=realtime|compact

#![cfg(feature = "inference")]
#![cfg(any(table_format = "q16_16", table_format = "q32_32"))]

use g_math::fixed_point::compute_tier as ct;
use g_math::fixed_point::compute_tier::ComputeStorage;
use g_math::fixed_point::FixedPoint;

#[allow(dead_code)]
mod data {
    include!("data/compute_tier_refs.rs");
}
use data::refs;

// ============================================================================
// Helpers
// ============================================================================

fn cs(x: i128) -> ComputeStorage {
    ComputeStorage::try_from(x).expect("reference input fits ComputeStorage")
}

fn cs_to_i128(x: ComputeStorage) -> i128 {
    x as i128
}

/// The value of `s` where the build's storage range holds it (every profile
/// but realtime at a high GMATH_FRAC_BITS holds all of this file's inputs).
fn fits(s: &str) -> Option<FixedPoint> {
    FixedPoint::try_from_str(s).ok()
}

fn fp(s: &str) -> FixedPoint {
    if let Some(rest) = s.strip_prefix('-') {
        -FixedPoint::from_str(rest)
    } else {
        FixedPoint::from_str(s)
    }
}

/// Assert compute-tier closeness in compute ULP and storage-level value after
/// the single `to_fixed` rounding.
fn check(
    label: &str,
    f: impl Fn(ComputeStorage) -> ComputeStorage,
    table: &[(i128, i128, i64)],
    max_compute_ulp: i128,
    max_storage_ulp: i64,
) {
    for &(x_raw, want_compute, want_storage) in table {
        let got = f(cs(x_raw));
        let cdiff = (cs_to_i128(got) - want_compute).abs();
        assert!(
            cdiff <= max_compute_ulp,
            "{label}(x_raw={x_raw}): compute-tier diff {cdiff} ULP (got {}, want {want_compute})",
            cs_to_i128(got)
        );
        let got_fixed = ct::to_fixed(got);
        let sdiff = (got_fixed.raw() as i64 - want_storage).abs();
        assert!(
            sdiff <= max_storage_ulp,
            "{label}(x_raw={x_raw}): storage diff {sdiff} LSB (got {}, want {want_storage})",
            got_fixed.raw() as i64
        );
    }
}

// ============================================================================
// mpmath references — compute tier + storage level
// ============================================================================
//
// Tolerances are pinned to MEASURED maxima over this corpus (2026-08-01),
// not guessed. The compute-tier value is the raw tier-N+1 kernel output
// *before* the storage rounding that the 0-ULP contract applies to:
// - realtime: at every split (2 to 30 fractional bits, measured 2026-10-05)
//   primitives 0 compute-ULP and composed forms at most 1.
// - q16_16: primitives measured 0 compute-ULP (the kernel output is itself
//   a correctly rounded Q64.64→Q32.32 downscale); composed forms ≤1.
// - q32_32: the raw Q64.64 kernel output is exposed undownscaled —
//   measured exp 4, ln 3, sqrt 1; composed ≤5 (softplus, which stacks
//   two kernels).
// Storage level after the final `to_fixed` rounding measured EXACT (0 LSB)
// for every function at both profiles, composed forms included.

#[cfg(table_format = "q16_16")]
const PRIM_CULP: i128 = 0;
#[cfg(table_format = "q32_32")]
const PRIM_CULP: i128 = 4;

#[cfg(table_format = "q16_16")]
const COMP_CULP: i128 = 1;
#[cfg(table_format = "q32_32")]
const COMP_CULP: i128 = 5;

// Storage level: the compute value (correctly rounded at 2 x FRAC_BITS) is
// rounded once more to storage. Measured 2026-10-05 over the corpus at every
// realtime split: 0 LSB from 5 fractional bits up; at 2, 3 and 4 the second
// rounding can land one unit from the directly rounded reference (a compute
// tier of 4 to 8 bits leaves the value within reach of a storage tie).
#[cfg(table_format = "q16_16")]
const STORAGE_LSB: i64 = if g_math::fixed_point::frac_config::FRAC_BITS <= 4 { 1 } else { 0 };
#[cfg(table_format = "q32_32")]
const STORAGE_LSB: i64 = 0;

#[test]
fn mpmath_exp() {
    check("exp", ct::exp, refs::EXP, PRIM_CULP, STORAGE_LSB);
}

#[test]
fn mpmath_ln() {
    check("ln", ct::ln, refs::LN, PRIM_CULP, STORAGE_LSB);
}

#[test]
fn mpmath_sqrt() {
    check("sqrt", ct::sqrt, refs::SQRT, PRIM_CULP, STORAGE_LSB);
}

#[test]
fn mpmath_sigmoid() {
    check("sigmoid", ct::sigmoid, refs::SIGMOID, COMP_CULP, STORAGE_LSB);
}

#[test]
fn mpmath_softplus() {
    check("softplus", ct::softplus, refs::SOFTPLUS, COMP_CULP, STORAGE_LSB);
}

#[test]
fn mpmath_ln1p() {
    check("ln1p", ct::ln1p, refs::LN1P, COMP_CULP, STORAGE_LSB);
}

#[test]
fn mpmath_sinh() {
    check("sinh", |x| ct::sinhcosh(x).0, refs::SINH, COMP_CULP, STORAGE_LSB);
}

#[test]
fn mpmath_cosh() {
    check("cosh", |x| ct::sinhcosh(x).1, refs::COSH, COMP_CULP, STORAGE_LSB);
}

// ============================================================================
// Path independence vs the imperative surface
// ============================================================================

#[test]
fn path_independence_exp_ln_sqrt() {
    // An input is used where the split can hold it and the result (the
    // realtime range runs from +-2^29 at 2 fractional bits to +-2 at 30).
    let mut compared = 0;
    for s in ["0.0625", "0.5", "1", "1.25", "2", "3.5", "7.75", "-3.5", "-0.75", "-0.0625"] {
        let Some(x) = fits(s) else { continue };
        if let Ok(want) = x.try_exp() {
            assert_eq!(
                ct::to_fixed(ct::exp(ct::from_fixed(x))).raw(),
                want.raw(),
                "exp({s}) diverges between compute_tier and FixedPoint"
            );
            compared += 1;
        }
        if let Ok(want) = x.try_ln() {
            assert_eq!(
                ct::to_fixed(ct::ln(ct::from_fixed(x))).raw(),
                want.raw(),
                "ln({s}) diverges between compute_tier and FixedPoint"
            );
            compared += 1;
        }
        if let Ok(want) = x.try_sqrt() {
            assert_eq!(
                ct::to_fixed(ct::sqrt(ct::from_fixed(x))).raw(),
                want.raw(),
                "sqrt({s}) diverges between compute_tier and FixedPoint"
            );
            compared += 1;
        }
    }
    assert!(compared >= 6, "only {compared} comparisons ran at this split");
}

#[test]
fn roundtrip_from_to_fixed_is_identity() {
    for s in ["0", "0.0625", "1", "7.75", "-2.5", "-0.0625"] {
        let Some(x) = fits(s) else { continue };
        assert_eq!(ct::to_fixed(ct::from_fixed(x)).raw(), x.raw());
    }
}

// ============================================================================
// Guards: domain panics, saturation, identities
// ============================================================================

#[test]
#[should_panic(expected = "outside the domain")]
fn ln_zero_panics() {
    let _ = ct::ln(cs(0));
}

#[test]
#[should_panic(expected = "outside the domain")]
fn ln_negative_panics() {
    let _ = ct::ln(ct::from_fixed(fp("-1")));
}

#[test]
#[should_panic(expected = "outside the domain")]
fn sqrt_negative_panics() {
    let _ = ct::sqrt(ct::from_fixed(fp("-0.5")));
}

#[test]
#[should_panic(expected = "outside the domain")]
fn ln1p_at_minus_one_panics() {
    let _ = ct::ln1p(ct::from_fixed(fp("-1")));
}

#[test]
fn exp_saturates_at_ceiling_never_wraps() {
    // exp of the largest storage value exceeds the compute tier: e^(2^(W-1-F))
    // against a compute range of 2^(2W-1-2F). The one exception is realtime
    // at 30 fractional bits, where e^2 = 7.39 is below the compute range of 8
    // (it still exceeds the storage range of 2).
    let big = ct::from_fixed(FixedPoint::from_raw(g_math::fixed_point::imperative::BinaryStorage::MAX));
    let sat = ct::exp(big);
    #[cfg(table_format = "q16_16")]
    let saturates = g_math::fixed_point::frac_config::FRAC_BITS < 30;
    #[cfg(not(table_format = "q16_16"))]
    let saturates = true;
    if saturates {
        assert_eq!(cs_to_i128(sat), cs_to_i128(ct::ceiling()), "exp must saturate at ceiling()");
    }
    // The saturated value must NOT silently convert to storage.
    assert!(ct::try_to_fixed(sat).is_none(), "saturated exp must not fit storage");
}

#[test]
fn sigmoid_extremes_and_symmetry() {
    // Extremes pin to exactly 0 and 1 at the compute tier.
    if let Some(fifty) = fits("50") {
        let big = ct::from_fixed(fifty);
        assert_eq!(cs_to_i128(ct::sigmoid(big)), cs_to_i128(ct::one()));
        assert_eq!(cs_to_i128(ct::sigmoid(-big)), 0);
    }
    // sigmoid(x) + sigmoid(-x) = 1, within 2 compute ULP.
    for s in ["0", "0.5", "2.5", "7"] {
        let Some(x) = fits(s) else { continue };
        let x = ct::from_fixed(x);
        let sum = cs_to_i128(ct::sigmoid(x)) + cs_to_i128(ct::sigmoid(-x));
        let diff = (sum - cs_to_i128(ct::one())).abs();
        assert!(diff <= 2, "sigmoid symmetry off by {diff} compute ULP at x={s}");
    }
}

#[test]
fn softplus_difference_identity() {
    // softplus(x) - softplus(-x) = x exactly (mathematically); allow 2 compute ULP.
    for s in ["0", "0.5", "2.5", "7"] {
        let Some(x) = fits(s) else { continue };
        let x = ct::from_fixed(x);
        let d = cs_to_i128(ct::softplus(x)) - cs_to_i128(ct::softplus(-x));
        let diff = (d - cs_to_i128(x)).abs();
        assert!(diff <= 2, "softplus identity off by {diff} compute ULP at x={s}");
    }
}

