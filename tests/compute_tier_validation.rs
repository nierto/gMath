//! Validation for the public `compute_tier` module (0.4.32).
//!
//! mpmath 60-digit references (tests/data/compute_tier_refs.rs, generated —
//! see the header of that file) at the two profiles the inference surface
//! targets, matching the RowScaledTQ19/matvec_q2f precedent:
//! q16_16 (FRAC_BITS=16, COMPUTE_FRAC_BITS=32) and
//! q32_32 (FRAC_BITS=32, COMPUTE_FRAC_BITS=64).
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

#[test]
fn mpmath_exp() {
    check("exp", ct::exp, refs::EXP, PRIM_CULP, 0);
}

#[test]
fn mpmath_ln() {
    check("ln", ct::ln, refs::LN, PRIM_CULP, 0);
}

#[test]
fn mpmath_sqrt() {
    check("sqrt", ct::sqrt, refs::SQRT, PRIM_CULP, 0);
}

#[test]
fn mpmath_sigmoid() {
    check("sigmoid", ct::sigmoid, refs::SIGMOID, COMP_CULP, 0);
}

#[test]
fn mpmath_softplus() {
    check("softplus", ct::softplus, refs::SOFTPLUS, COMP_CULP, 0);
}

#[test]
fn mpmath_ln1p() {
    check("ln1p", ct::ln1p, refs::LN1P, COMP_CULP, 0);
}

#[test]
fn mpmath_sinh() {
    check("sinh", |x| ct::sinhcosh(x).0, refs::SINH, COMP_CULP, 0);
}

#[test]
fn mpmath_cosh() {
    check("cosh", |x| ct::sinhcosh(x).1, refs::COSH, COMP_CULP, 0);
}

// ============================================================================
// Path independence vs the imperative surface
// ============================================================================

#[test]
fn path_independence_exp_ln_sqrt() {
    for s in ["0.0625", "0.5", "1", "1.25", "2", "3.5", "7.75"] {
        let x = fp(s);
        assert_eq!(
            ct::to_fixed(ct::exp(ct::from_fixed(x))).raw(),
            x.exp().raw(),
            "exp({s}) diverges between compute_tier and FixedPoint"
        );
        assert_eq!(
            ct::to_fixed(ct::ln(ct::from_fixed(x))).raw(),
            x.ln().raw(),
            "ln({s}) diverges between compute_tier and FixedPoint"
        );
        assert_eq!(
            ct::to_fixed(ct::sqrt(ct::from_fixed(x))).raw(),
            x.sqrt().raw(),
            "sqrt({s}) diverges between compute_tier and FixedPoint"
        );
    }
    for s in ["-3.5", "-0.75", "-0.0625"] {
        let x = fp(s);
        assert_eq!(
            ct::to_fixed(ct::exp(ct::from_fixed(x))).raw(),
            x.exp().raw(),
            "exp({s}) diverges between compute_tier and FixedPoint"
        );
    }
}

#[test]
fn roundtrip_from_to_fixed_is_identity() {
    for s in ["0", "0.0625", "1", "7.75", "-2.5", "-0.0625"] {
        let x = fp(s);
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
    // exp of the largest storage value vastly exceeds the compute tier.
    let big = ct::from_fixed(FixedPoint::from_raw(
        g_math::fixed_point::imperative::BinaryStorage::MAX / 2,
    ));
    let sat = ct::exp(big);
    assert_eq!(cs_to_i128(sat), cs_to_i128(ct::ceiling()), "exp must saturate at ceiling()");
    // The saturated value must NOT silently convert to storage.
    assert!(ct::try_to_fixed(sat).is_none(), "saturated exp must not fit storage");
}

#[test]
fn sigmoid_extremes_and_symmetry() {
    // Extremes pin to exactly 0 and 1 at the compute tier.
    let big = ct::from_fixed(fp("50"));
    assert_eq!(cs_to_i128(ct::sigmoid(big)), cs_to_i128(ct::one()));
    assert_eq!(cs_to_i128(ct::sigmoid(-big)), 0);
    // sigmoid(x) + sigmoid(-x) = 1, within 2 compute ULP.
    for s in ["0", "0.5", "2.5", "7"] {
        let x = ct::from_fixed(fp(s));
        let sum = cs_to_i128(ct::sigmoid(x)) + cs_to_i128(ct::sigmoid(-x));
        let diff = (sum - cs_to_i128(ct::one())).abs();
        assert!(diff <= 2, "sigmoid symmetry off by {diff} compute ULP at x={s}");
    }
}

#[test]
fn softplus_difference_identity() {
    // softplus(x) - softplus(-x) = x exactly (mathematically); allow 2 compute ULP.
    for s in ["0", "0.5", "2.5", "7"] {
        let x = ct::from_fixed(fp(s));
        let d = cs_to_i128(ct::softplus(x)) - cs_to_i128(ct::softplus(-x));
        let diff = (d - cs_to_i128(x)).abs();
        assert!(diff <= 2, "softplus identity off by {diff} compute ULP at x={s}");
    }
}

