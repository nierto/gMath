//! Decimal exp over its full representable range, against mpmath.
//!
//! Gates `DecimalFixed<D>::{exp, sinh, cosh, tanh}` (plus the large-argument
//! `asinh`/`acosh` path and the `sin`/`cos` range reduction up to the largest
//! argument the compute tier holds) for every D the decimal tests use on the
//! profile,
//! and the engine `decimal_exp` materialised at the canonical storage dp,
//! from far below zero (results that round to 0) up to the largest argument
//! whose result is representable. Expected values are f(x) * 10^D rounded
//! half to even (the decimal rule), from
//! `scripts/generate_decimal_exp_range_refs.py` (mpmath, 500 digits).
//!
//! Gate: 0 units everywhere (correctly rounded). The engine rounds once to
//! the compute dp and again to D, so a result whose compute-dp value lands
//! exactly on a D tie could double-round; that has probability about
//! 10^-(compute dp - D) per input and did not occur on this corpus.
//!
//! Beyond the range: `decimal_exp` returns `Err(TierOverflow)` (the value is
//! outside the compute tier), the storage narrowing returns
//! `Err(TierOverflow)`, and the infallible `DecimalFixed` methods panic.
//! Before 0.6.4 realtime `DecimalFixed::<4>::exp(22)` was 446565 units high,
//! `exp(22.9451)` returned -9222509832.1468 (a wrapped table entry), and
//! embedded `exp(44)` at 19 decimals was 198 units low.
//!
//! ```bash
//! GMATH_PROFILE=embedded cargo test --test decimal_exp_range_validation -- --nocapture
//! ```

use g_math::fixed_point::domains::decimal_fixed::transcendental::decimal_compute::BinaryStorage;
use g_math::fixed_point::domains::decimal_fixed::transcendental::{
    decimal_compute_from_int, decimal_compute_is_zero, decimal_downscale_to_storage, decimal_exp,
    decimal_upscale_to_compute, DECIMAL_STORAGE_MAX_DP,
};
use g_math::fixed_point::{DecimalFixed, OverflowDetected};
use std::panic::{catch_unwind, AssertUnwindSafe};

include!("data/decimal_exp_range_refs.rs");

/// Compute-tier integer width in bits (the exp argument range is about this).
#[cfg(table_format = "q16_16")]
const COMPUTE_BITS: i64 = 64;
#[cfg(table_format = "q32_32")]
const COMPUTE_BITS: i64 = 128;
#[cfg(table_format = "q64_64")]
const COMPUTE_BITS: i64 = 256;
#[cfg(table_format = "q128_128")]
const COMPUTE_BITS: i64 = 512;
#[cfg(table_format = "q256_256")]
const COMPUTE_BITS: i64 = 1024;

fn eval<const D: u8>(f: &str, x: i128) -> i128 {
    let v = DecimalFixed::<D>::from_raw(x);
    let r = match f {
        "exp" => v.exp(),
        "sinh" => v.sinh(),
        "cosh" => v.cosh(),
        "tanh" => v.tanh(),
        "asinh" => v.asinh(),
        "acosh" => v.acosh(),
        "sin" => v.sin(),
        "cos" => v.cos(),
        other => panic!("unknown function {other}"),
    };
    r.raw_value()
}

fn eval_at(f: &str, d: u8, x: i128) -> i128 {
    match d {
        0 => eval::<0>(f, x),
        2 => eval::<2>(f, x),
        4 => eval::<4>(f, x),
        9 => eval::<9>(f, x),
        19 => eval::<19>(f, x),
        28 => eval::<28>(f, x),
        38 => eval::<38>(f, x),
        other => panic!("no dispatch for D = {other}"),
    }
}

thread_local! {
    static QUIET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f`, turning a panic into `None` without printing it (only on this
/// thread: the other tests keep their panic messages).
fn quiet<T>(f: impl FnOnce() -> T) -> Option<T> {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !QUIET.with(|q| q.get()) {
                default(info);
            }
        }));
    });
    QUIET.with(|q| q.set(true));
    let r = catch_unwind(AssertUnwindSafe(f)).ok();
    QUIET.with(|q| q.set(false));
    r
}

/// Parse a decimal integer string into the profile's BinaryStorage.
fn storage(s: &str) -> BinaryStorage {
    let (neg, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
    {
        let v: BinaryStorage = digits.parse().expect("storage literal");
        if neg { -v } else { v }
    }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    {
        let ten = BinaryStorage::from_i128(10);
        let mut v = BinaryStorage::from_i128(0);
        for ch in digits.chars() {
            v = v * ten + BinaryStorage::from_i128(ch.to_digit(10).expect("digit") as i128);
        }
        if neg { -v } else { v }
    }
}

fn one_unit() -> BinaryStorage {
    storage("1")
}

/// |a - b| in storage units, `u128::MAX` when it does not fit.
fn units(a: BinaryStorage, b: BinaryStorage) -> u128 {
    #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
    {
        (a as i128).checked_sub(b as i128).map_or(u128::MAX, |d| d.unsigned_abs())
    }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    {
        let d = a - b;
        if d.fits_in_i128() { d.as_i128().unsigned_abs() } else { u128::MAX }
    }
}

#[test]
fn decimal_fixed_exp_family_is_correctly_rounded() {
    let mut failures = Vec::new();
    // (function, D) -> (points, max units, mismatches, panics)
    let mut stats: Vec<((&str, u8), (usize, u128, usize, usize))> = Vec::new();
    for &(f, d, x, expected) in FIXED_REFS {
        let got = quiet(|| eval_at(f, d, x));
        let idx = match stats.iter().position(|(k, _)| *k == (f, d)) {
            Some(i) => i,
            None => {
                stats.push(((f, d), (0, 0, 0, 0)));
                stats.len() - 1
            }
        };
        let s = &mut stats[idx].1;
        s.0 += 1;
        match got {
            Some(v) => {
                let units = (v - expected).unsigned_abs();
                if units > 0 {
                    s.2 += 1;
                    if failures.len() < 20 {
                        failures.push(format!("{f}<{d}>({x}): got {v}, expected {expected} ({units} units)"));
                    }
                }
                s.1 = s.1.max(units);
            }
            None => {
                s.3 += 1;
                if failures.len() < 20 {
                    failures.push(format!("{f}<{d}>({x}): panicked, expected {expected}"));
                }
            }
        }
    }
    for ((f, d), (n, max, bad, panics)) in &stats {
        eprintln!("DecimalFixed<{d:2}>::{f:5}: points={n:4} max_units={max} mismatches={bad} panics={panics}");
    }
    assert!(failures.is_empty(), "{} failures, first:\n{}", failures.len(), failures.join("\n"));
}

/// Just past the largest representable argument the infallible methods
/// panic (never a wrapped or saturated value).
#[test]
fn decimal_fixed_exp_beyond_range_panics() {
    for f in ["exp", "sinh", "cosh"] {
        let mut tops: Vec<(u8, i128)> = Vec::new();
        for &(g, d, x, _) in FIXED_REFS {
            if g != f {
                continue;
            }
            match tops.iter_mut().find(|(e, _)| *e == d) {
                Some(t) => t.1 = t.1.max(x),
                None => tops.push((d, x)),
            }
        }
        for (d, top) in tops {
            let beyond = quiet(|| eval_at(f, d, top + 1));
            assert!(beyond.is_none(), "{f}<{d}>({}) should panic, got {:?}", top + 1, beyond);
        }
    }
}

#[test]
fn engine_exp_at_canonical_storage_is_correctly_rounded() {
    let dp = DECIMAL_STORAGE_MAX_DP;
    let mut failures = Vec::new();
    let (mut errors, mut mismatches, mut max_units) = (0usize, 0usize, 0u128);
    let start = std::time::Instant::now();
    for &(x, expected) in ENGINE_EXP_REFS {
        let xc = decimal_upscale_to_compute(storage(x), dp).expect("input upscale");
        match decimal_exp(xc).and_then(|r| decimal_downscale_to_storage(r, dp)) {
            Ok(v) => {
                let u = units(v, storage(expected));
                max_units = max_units.max(u);
                if u > 0 {
                    mismatches += 1;
                    if failures.len() < 20 {
                        failures.push(format!("exp({x} at dp {dp}): {u} units off, expected {expected}"));
                    }
                }
            }
            Err(e) => {
                errors += 1;
                if failures.len() < 20 {
                    failures.push(format!("exp({x} at dp {dp}): {e:?}, expected {expected}"));
                }
            }
        }
    }
    let per_call = start.elapsed().as_nanos() / ENGINE_EXP_REFS.len().max(1) as u128;
    eprintln!(
        "decimal_exp at dp {dp}: points={} max_units={max_units} mismatches={mismatches} errors={errors} ({per_call} ns/call incl. conversions)",
        ENGINE_EXP_REFS.len()
    );
    assert!(failures.is_empty(), "first failures:\n{}", failures.join("\n"));
}

#[test]
fn engine_exp_reports_overflow() {
    let dp = DECIMAL_STORAGE_MAX_DP;
    // one storage unit past the largest argument whose exp fits BinaryStorage:
    // the engine value exists, the narrowing reports it
    let past = storage(ENGINE_EXP_TOP) + one_unit();
    let xc = decimal_upscale_to_compute(past, dp).expect("input upscale");
    let r = decimal_exp(xc).and_then(|r| decimal_downscale_to_storage(r, dp));
    assert_eq!(r, Err(OverflowDetected::TierOverflow));
    // far past the compute tier: exp itself reports it
    for k in [COMPUTE_BITS, COMPUTE_BITS + 1, 2 * COMPUTE_BITS] {
        assert_eq!(decimal_exp(decimal_compute_from_int(k)), Err(OverflowDetected::TierOverflow), "exp({k})");
    }
    // and far below zero the result is 0, not an error
    for k in [COMPUTE_BITS, COMPUTE_BITS + 1, 2 * COMPUTE_BITS] {
        let r = decimal_exp(decimal_compute_from_int(-k)).expect("exp of a large negative");
        assert!(decimal_compute_is_zero(&r), "exp(-{k}) should round to 0");
    }
}

/// The inputs from the defect report, pinned by value.
#[test]
fn reported_defects() {
    #[cfg(table_format = "q16_16")]
    {
        // mpmath: e^22 = 3584912846.131591561681159...
        assert_eq!(DecimalFixed::<4>::from_integer(22).exp().raw_value(), 35849128461316);
        // e^23 = 9744803446.2 exceeds the i64 compute tier (9223372036.9 at 9 dp)
        assert!(quiet(|| DecimalFixed::<4>::from_integer(23).exp()).is_none());
        // tanh doubled x at the compute tier and wrapped for |x| > 4.6e9
        let big = DecimalFixed::<4>::from_raw(90_000_000_000_000);
        assert_eq!(big.tanh().raw_value(), 10_000);
        assert_eq!((-big).tanh().raw_value(), -10_000);
    }
    #[cfg(table_format = "q64_64")]
    {
        // mpmath: e^40 = 235385266837019985.4078999107490348045...
        assert_eq!(
            DecimalFixed::<19>::from_integer(40).exp().raw_value(),
            2353852668370199854078999107490348045
        );
        // asinh(1e30) = 69.7727... (x^2 wrapped the compute tier)
        assert_eq!(DecimalFixed::<0>::from_raw(1_000_000_000_000_000_000_000_000_000_000).asinh().raw_value(), 70);
    }
}
