//! Fallible `DecimalFixed<D>` transcendentals: the `try_` twins of the 18
//! methods (exp ln sqrt sin cos sincos tan atan atan2 asin acos sinh cosh
//! sinhcosh tanh asinh acosh atanh).
//!
//! Contract pinned here (the one `FixedPoint`'s `try_` methods follow, see
//! `tests/try_direct_bypass_validation.rs`):
//! 1. Whenever the infallible method returns, `try_` returns `Ok` of the
//!    BIT-IDENTICAL value; whenever it panics, `try_` returns `Err`. Both run
//!    the same compute-tier core; only the boundary conversions differ.
//! 2. Domain violations at storage are `Err(DomainError)`: ln of x <= 0,
//!    sqrt of x < 0, asin/acos beyond |1|, acosh below 1, atanh at or beyond
//!    |1|, atan2(0, 0), and tan where cos x is exactly 0 at the compute dp.
//! 3. Results beyond i128 at D, and arguments or intermediates beyond the
//!    decimal compute tier, are `Err(TierOverflow)`, never a panic or a
//!    wrapped value. An in-domain argument that only reaches a domain
//!    boundary because the compute dp is coarser than D (realtime D > 9,
//!    compact D > 19) is `Err(PrecisionLimit)`.
//! 4. `try_` never panics, for any raw at any D: swept below over the i128
//!    and compute-tier extremes and every power of ten, at D in
//!    {0, 2, 4, 9, 18, 19, 20, 28, 38} on every profile.
//!
//! Correctness of the values: the `try_` forms of exp/sinh/cosh/tanh/asinh/
//! acosh/sin/cos are checked against the mpmath corpus of
//! `tests/decimal_exp_range_validation.rs` (0 units); boundary values
//! against mpmath pi (`mp.dps = 80`, digits embedded below).
//!
//! ```bash
//! GMATH_PROFILE=embedded cargo test --test decimal_try_transcendentals_validation
//! ```

use g_math::fixed_point::domains::decimal_fixed::transcendental::DECIMAL_COMPUTE_DP;
use g_math::fixed_point::{DecimalFixed, OverflowDetected};
use std::panic::{catch_unwind, AssertUnwindSafe};

mod refs {
    #![allow(dead_code)]
    include!("data/decimal_exp_range_refs.rs");
}

/// mpmath `pi / 2` and `pi` at 80 digits.
const PI_HALF: &str = "1.5707963267948966192313216916397514420985846996875529104874722961539082031431045";
const PI: &str = "3.141592653589793238462643383279502884197169399375105820974944592307816406286209";

type R<T> = Result<T, OverflowDetected>;

const UNARY: &[&str] = &[
    "exp", "ln", "sqrt", "sin", "cos", "sincos", "tan", "atan", "asin", "acos", "sinh", "cosh",
    "sinhcosh", "tanh", "asinh", "acosh", "atanh",
];

/// Every D the sweep covers (the type allows 0..=38 on every profile).
const SWEEP_D: &[u8] = &[0, 2, 4, 9, 18, 19, 20, 28, 38];

/// The D values the decimal tests use on this profile.
#[cfg(table_format = "q16_16")]
const PROFILE_D: &[u8] = &[4, 2, 0];
#[cfg(table_format = "q32_32")]
const PROFILE_D: &[u8] = &[9, 4, 0];
#[cfg(table_format = "q64_64")]
const PROFILE_D: &[u8] = &[19, 9, 0];
#[cfg(table_format = "q128_128")]
const PROFILE_D: &[u8] = &[38, 28, 19, 0];
#[cfg(table_format = "q256_256")]
const PROFILE_D: &[u8] = &[38, 19, 0];

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

fn infallible<const D: u8>(f: &str, x: DecimalFixed<D>) -> Vec<i128> {
    let one = |v: DecimalFixed<D>| vec![v.raw_value()];
    let two = |(a, b): (DecimalFixed<D>, DecimalFixed<D>)| vec![a.raw_value(), b.raw_value()];
    match f {
        "exp" => one(x.exp()),
        "ln" => one(x.ln()),
        "sqrt" => one(x.sqrt()),
        "sin" => one(x.sin()),
        "cos" => one(x.cos()),
        "sincos" => two(x.sincos()),
        "tan" => one(x.tan()),
        "atan" => one(x.atan()),
        "asin" => one(x.asin()),
        "acos" => one(x.acos()),
        "sinh" => one(x.sinh()),
        "cosh" => one(x.cosh()),
        "sinhcosh" => two(x.sinhcosh()),
        "tanh" => one(x.tanh()),
        "asinh" => one(x.asinh()),
        "acosh" => one(x.acosh()),
        "atanh" => one(x.atanh()),
        other => unreachable!("{other}"),
    }
}

fn fallible<const D: u8>(f: &str, x: DecimalFixed<D>) -> R<Vec<i128>> {
    let one = |v: R<DecimalFixed<D>>| v.map(|v| vec![v.raw_value()]);
    let two = |v: R<(DecimalFixed<D>, DecimalFixed<D>)>| v.map(|(a, b)| vec![a.raw_value(), b.raw_value()]);
    match f {
        "exp" => one(x.try_exp()),
        "ln" => one(x.try_ln()),
        "sqrt" => one(x.try_sqrt()),
        "sin" => one(x.try_sin()),
        "cos" => one(x.try_cos()),
        "sincos" => two(x.try_sincos()),
        "tan" => one(x.try_tan()),
        "atan" => one(x.try_atan()),
        "asin" => one(x.try_asin()),
        "acos" => one(x.try_acos()),
        "sinh" => one(x.try_sinh()),
        "cosh" => one(x.try_cosh()),
        "sinhcosh" => two(x.try_sinhcosh()),
        "tanh" => one(x.try_tanh()),
        "asinh" => one(x.try_asinh()),
        "acosh" => one(x.try_acosh()),
        "atanh" => one(x.try_atanh()),
        other => unreachable!("{other}"),
    }
}

/// Contract 1 + 4 for one call: `try_` does not panic, and it is `Ok` of the
/// infallible result exactly when the infallible method returns.
fn agree(label: &str, inf: impl FnOnce() -> Vec<i128>, tr: impl FnOnce() -> R<Vec<i128>>) -> Result<R<Vec<i128>>, String> {
    let tried = match quiet(tr) {
        Some(t) => t,
        None => return Err(format!("{label}: try_ PANICKED")),
    };
    match (quiet(inf), &tried) {
        (Some(v), Ok(w)) if v == *w => Ok(tried),
        (None, Err(_)) => Ok(tried),
        (inf, _) => Err(format!("{label}: infallible {inf:?} vs try {tried:?}")),
    }
}

fn agree_unary<const D: u8>(f: &str, raw: i128) -> Result<R<Vec<i128>>, String> {
    let x = DecimalFixed::<D>::from_raw(raw);
    agree(&format!("{f}<{D}>(raw {raw})"), || infallible(f, x), || fallible(f, x))
}

fn agree_atan2<const D: u8>(y: i128, x: i128) -> Result<R<Vec<i128>>, String> {
    let (yv, xv) = (DecimalFixed::<D>::from_raw(y), DecimalFixed::<D>::from_raw(x));
    agree(
        &format!("atan2<{D}>(raw {y}, raw {x})"),
        || vec![yv.atan2(xv).raw_value()],
        || yv.try_atan2(xv).map(|v| vec![v.raw_value()]),
    )
}

macro_rules! at_d {
    ($d:expr, $f:ident ( $($arg:expr),* )) => {
        match $d {
            0 => $f::<0>($($arg),*),
            2 => $f::<2>($($arg),*),
            4 => $f::<4>($($arg),*),
            9 => $f::<9>($($arg),*),
            18 => $f::<18>($($arg),*),
            19 => $f::<19>($($arg),*),
            20 => $f::<20>($($arg),*),
            28 => $f::<28>($($arg),*),
            38 => $f::<38>($($arg),*),
            other => panic!("no dispatch for D = {other}"),
        }
    };
}

fn scale(d: u8) -> i128 {
    10i128.pow(d as u32)
}

/// `n / 10^k` as a raw at `d` decimals (truncated), `None` if it does not fit.
fn raw_of(n: i128, k: u32, d: u8) -> Option<i128> {
    let d = d as u32;
    if d >= k { n.checked_mul(10i128.checked_pow(d - k)?) } else { Some(n / 10i128.pow(k - d)) }
}

/// `s` (a decimal string "i.fff...") rounded to `d` fraction digits, as a raw
/// (no ties: the constants are irrational), `None` beyond i128.
fn try_round_str(s: &str, d: u8) -> Option<i128> {
    let (int, frac) = s.split_once('.').unwrap();
    let digits = format!("{int}{}", &frac[..d as usize]);
    let v: i128 = digits.parse().ok()?;
    if frac.as_bytes()[d as usize] >= b'5' { v.checked_add(1) } else { Some(v) }
}

fn round_str(s: &str, d: u8) -> i128 {
    try_round_str(s, d).expect("constant beyond i128 at this D")
}

/// Raws the no-panic sweep feeds every method: the i128 and i64 (realtime
/// compute tier) extremes, 0, +-1, +-SCALE and its neighbours, +-10^k.
fn sweep_raws(d: u8) -> Vec<i128> {
    let s = scale(d);
    let mut v = vec![
        i128::MIN, i128::MIN + 1, i128::MAX, i128::MAX - 1,
        i64::MIN as i128, i64::MAX as i128, i64::MIN as i128 + 1,
        0, 1, -1, s, -s, s - 1, -s + 1, s + 1, -s - 1, s.saturating_mul(2), s.saturating_mul(-2),
    ];
    for k in [1u32, 2, 4, 6, 9, 10, 12, 15, 18, 19, 20, 24, 28, 30, 34, 36, 37, 38] {
        let p = 10i128.pow(k);
        v.extend([p, -p, 5 * (p / 10), -5 * (p / 10)]);
    }
    v
}

fn run_d(d: u8, f: &str, raw: i128) -> Result<R<Vec<i128>>, String> {
    at_d!(d, agree_unary(f, raw))
}

// ============================================================================
// Contract 1 + 4: no panic anywhere, try_ == infallible
// ============================================================================

#[test]
fn try_never_panics_and_matches_infallible_on_extreme_raws() {
    let mut failures = Vec::new();
    let mut calls = 0usize;
    for &d in SWEEP_D {
        for raw in sweep_raws(d) {
            for f in UNARY {
                calls += 1;
                if let Err(e) = run_d(d, f, raw) {
                    failures.push(e);
                }
            }
        }
        let pair_raws = {
            let s = scale(d);
            [i128::MIN, i128::MAX, i64::MIN as i128, i64::MAX as i128, 0, 1, -1, s, -s, 10i128.pow(20), -10i128.pow(38)]
        };
        for &y in &pair_raws {
            for &x in &pair_raws {
                calls += 1;
                if let Err(e) = at_d!(d, agree_atan2(y, x)) {
                    failures.push(e);
                }
            }
        }
    }
    eprintln!("sweep: {calls} calls, {} failures", failures.len());
    assert!(failures.is_empty(), "{} failures, first:\n{}", failures.len(), failures[..failures.len().min(25)].join("\n"));
}

/// In-domain values n / 10^k: bit-identity on ordinary inputs, per function,
/// at every D the decimal tests use on the profile.
#[test]
fn try_bit_identical_on_in_domain_grid() {
    const GRID: &[(i128, u32)] = &[
        (0, 0), (5, 1), (-5, 1), (25, 2), (-25, 2), (75, 2), (-75, 2), (9999, 4), (-9999, 4),
        (1, 3), (-1, 3), (1, 0), (-1, 0), (10001, 4), (15, 1), (-15, 1), (2, 0), (-225, 2),
        (3, 0), (73, 1), (-73, 1), (10, 0), (-10, 0), (123456, 3), (-123456, 3), (1_000_000, 0),
    ];
    let mut failures = Vec::new();
    let mut ok = 0usize;
    for &d in PROFILE_D {
        for &(n, k) in GRID {
            let Some(raw) = raw_of(n, k, d) else { continue };
            for f in UNARY {
                match run_d(d, f, raw) {
                    Ok(Ok(_)) => ok += 1,
                    Ok(Err(_)) => {}
                    Err(e) => failures.push(e),
                }
            }
            for &(m, j) in GRID {
                let Some(xr) = raw_of(m, j, d) else { continue };
                match at_d!(d, agree_atan2(raw, xr)) {
                    Ok(Ok(_)) => ok += 1,
                    Ok(Err(_)) => {}
                    Err(e) => failures.push(e),
                }
            }
        }
    }
    eprintln!("grid: {ok} Ok results bit-identical");
    assert!(ok > 1000, "grid produced too few Ok results ({ok})");
    assert!(failures.is_empty(), "{} failures, first:\n{}", failures.len(), failures[..failures.len().min(25)].join("\n"));
}

/// The try_ forms reproduce the mpmath corpus of the exp-range gate exactly
/// (0 units), and error where the infallible method panics past the range.
#[test]
fn try_matches_mpmath_corpus() {
    let mut failures = Vec::new();
    for &(f, d, x, expected) in refs::FIXED_REFS {
        match run_d(d, f, x) {
            Ok(Ok(v)) if v == vec![expected] => {}
            other => {
                if failures.len() < 20 {
                    failures.push(format!("{f}<{d}>({x}): {other:?}, expected Ok([{expected}])"));
                }
            }
        }
    }
    for f in ["exp", "sinh", "cosh"] {
        let mut tops: Vec<(u8, i128)> = Vec::new();
        for &(g, d, x, _) in refs::FIXED_REFS {
            if g == f {
                match tops.iter_mut().find(|(e, _)| *e == d) {
                    Some(t) => t.1 = t.1.max(x),
                    None => tops.push((d, x)),
                }
            }
        }
        for (d, top) in tops {
            match run_d(d, f, top + 1) {
                Ok(Err(OverflowDetected::TierOverflow)) => {}
                other => failures.push(format!("{f}<{d}>({}) past the top: {other:?}", top + 1)),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ============================================================================
// Contract 2: domain errors
// ============================================================================

fn expect_err(label: &str, r: Result<R<Vec<i128>>, String>, want: OverflowDetected, failures: &mut Vec<String>) {
    match r {
        Ok(Err(e)) if e == want => {}
        other => failures.push(format!("{label}: expected Err({want:?}), got {other:?}")),
    }
}

fn domain_cases<const D: u8>(failures: &mut Vec<String>) {
    let s = DecimalFixed::<D>::SCALE;
    let dom = OverflowDetected::DomainError;
    for (f, raw) in [
        ("ln", 0), ("ln", -1), ("ln", -s), ("ln", i128::MIN),
        ("sqrt", -1), ("sqrt", -s), ("sqrt", i128::MIN),
        ("asin", s + 1), ("asin", -s - 1), ("asin", i128::MAX), ("asin", i128::MIN),
        ("acos", s + 1), ("acos", -s - 1), ("acos", i128::MIN),
        ("acosh", s - 1), ("acosh", 0), ("acosh", s.saturating_mul(-2)), ("acosh", i128::MIN),
        ("atanh", s), ("atanh", -s), ("atanh", s + 1), ("atanh", s.saturating_mul(2)), ("atanh", i128::MIN),
    ] {
        expect_err(&format!("{f}<{D}>(raw {raw})"), agree_unary::<D>(f, raw), dom, failures);
    }
    expect_err(&format!("atan2<{D}>(0, 0)"), agree_atan2::<D>(0, 0), dom, failures);
}

#[test]
fn domain_violations_are_domain_errors() {
    let mut failures = Vec::new();
    for &d in SWEEP_D {
        at_d!(d, domain_cases(&mut failures));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// tan at pi/2 rounded to D, where D >= the compute dp: the reduced argument
/// rounds to 0 there, so cos x is exactly 0 and tan is `DomainError` (as
/// `FixedPoint::try_tan`); the infallible `tan` panics. |x - pi/2| is
/// 2.05e-10 at D = 9, 3.1e-20 at D = 19, 1.4e-39 at D = 38 (mpmath).
#[test]
fn tan_at_a_reachable_pole_is_a_domain_error() {
    let mut failures = Vec::new();
    let mut checked = 0;
    for &d in &[9u8, 19, 38] {
        if d < DECIMAL_COMPUTE_DP {
            continue;
        }
        checked += 1;
        let raw = round_str(PI_HALF, d);
        expect_err(&format!("tan<{d}>(pi/2)"), run_d(d, "tan", raw), OverflowDetected::DomainError, &mut failures);
    }
    // below the compute dp the pole is not reachable: tan(pi/2 at D) is a
    // large finite value
    let d = PROFILE_D[PROFILE_D.len() - 2];
    if d < DECIMAL_COMPUTE_DP {
        match run_d(d, "tan", round_str(PI_HALF, d)) {
            Ok(Ok(_)) | Ok(Err(OverflowDetected::TierOverflow)) => {}
            other => failures.push(format!("tan<{d}>(pi/2): {other:?}")),
        }
    }
    eprintln!("pole cases checked: {checked}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ============================================================================
// Contract 3: overflow and precision limits
// ============================================================================

fn overflow_cases<const D: u8>(failures: &mut Vec<String>) {
    let s = DecimalFixed::<D>::SCALE;
    let tier = OverflowDetected::TierOverflow;
    // e^200 = 7.2e86 is beyond i128 at every D; 200 fits every compute tier
    if let Some(big) = 200i128.checked_mul(s) {
        for f in ["exp", "sinh", "cosh", "sinhcosh"] {
            expect_err(&format!("{f}<{D}>(200)"), agree_unary::<D>(f, big), tier, failures);
        }
        for f in ["sinh", "sinhcosh"] {
            expect_err(&format!("{f}<{D}>(-200)"), agree_unary::<D>(f, -big), tier, failures);
        }
        expect_err(&format!("cosh<{D}>(-200)"), agree_unary::<D>("cosh", -big), tier, failures);
        // tanh never overflows: exactly +-1
        for (raw, want) in [(big, s), (-big, -s)] {
            match agree_unary::<D>("tanh", raw) {
                Ok(Ok(v)) if v == vec![want] => {}
                other => failures.push(format!("tanh<{D}>({raw}): {other:?}, expected Ok([{want}])")),
            }
        }
        // exp(-200) rounds to 0
        match agree_unary::<D>("exp", -big) {
            Ok(Ok(v)) if v == vec![0] => {}
            other => failures.push(format!("exp<{D}>(-200): {other:?}")),
        }
    }
    // e^1.5 = 4.48 is beyond i128 at D = 38 (max 1.7)
    if D == 38 {
        expect_err("exp<38>(1.5)", agree_unary::<D>("exp", s + s / 2), tier, failures);
        expect_err("cosh<38>(1.5)", agree_unary::<D>("cosh", s + s / 2), tier, failures);
    }
}

#[test]
fn overflow_is_tier_overflow_never_wrapped() {
    let mut failures = Vec::new();
    for &d in SWEEP_D {
        at_d!(d, overflow_cases(&mut failures));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Arguments outside the decimal compute tier (realtime: i64 at 9 dp,
/// compact: i128 at 19 dp) are `TierOverflow` for every function whose
/// storage domain admits them.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[test]
fn argument_beyond_the_compute_tier_is_tier_overflow() {
    let mut failures = Vec::new();
    let raw = 10i128.pow(30); // 10^30 at D = 0
    for f in UNARY {
        let r = agree_unary::<0>(f, raw);
        let want = match *f {
            "asin" | "acos" | "atanh" => OverflowDetected::DomainError,
            _ => OverflowDetected::TierOverflow,
        };
        expect_err(&format!("{f}<0>(1e30)"), r, want, &mut failures);
    }
    expect_err("atan2<0>(1e30, 1)", agree_atan2::<0>(raw, 1), OverflowDetected::TierOverflow, &mut failures);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Results beyond i128 at D = 38 from in-range arguments: ln(1e-38) = -87.5.
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
#[test]
fn ln_result_beyond_storage_is_tier_overflow() {
    let mut failures = Vec::new();
    expect_err("ln<38>(1e-38)", agree_unary::<38>("ln", 1), OverflowDetected::TierOverflow, &mut failures);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Where D exceeds the compute dp, an in-domain argument can round onto a
/// domain boundary at the compute tier: `PrecisionLimit`, not
/// `DomainError` (the argument is in the domain) and not a wrong value.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[test]
fn boundary_collapse_below_the_compute_dp_is_precision_limit() {
    let mut failures = Vec::new();
    let s = DecimalFixed::<38>::SCALE;
    let lim = OverflowDetected::PrecisionLimit;
    expect_err("ln<38>(1e-38)", agree_unary::<38>("ln", 1), lim, &mut failures);
    expect_err("asin<38>(1 - 1e-38)", agree_unary::<38>("asin", s - 1), lim, &mut failures);
    expect_err("asin<38>(-1 + 1e-38)", agree_unary::<38>("asin", -s + 1), lim, &mut failures);
    expect_err("acos<38>(1 - 1e-38)", agree_unary::<38>("acos", s - 1), lim, &mut failures);
    expect_err("atanh<38>(1 - 1e-38)", agree_unary::<38>("atanh", s - 1), lim, &mut failures);
    expect_err("atanh<38>(-1 + 1e-38)", agree_unary::<38>("atanh", -s + 1), lim, &mut failures);
    expect_err("atan2<38>(1e-38, 1e-38)", agree_atan2::<38>(1, 1), lim, &mut failures);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// compact at D = 19: (1+x)/(1-x) for x = 1 - 1e-19 is 2e19, beyond the
/// i128 compute tier at 19 dp, so atanh is `TierOverflow` (an intermediate
/// limit; the result 22.1 itself would fit).
#[cfg(table_format = "q32_32")]
#[test]
fn compact_atanh_next_to_one_is_tier_overflow() {
    let mut failures = Vec::new();
    let s = DecimalFixed::<19>::SCALE;
    expect_err("atanh<19>(1 - 1e-19)", agree_unary::<19>("atanh", s - 1), OverflowDetected::TierOverflow, &mut failures);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// |y/x| beyond the compute tier: atan2 = sign(y) pi/2 - atan(x/y). mpmath:
/// atan2(1e6, 1e-4) = 1.57079632669..., atan2(1e15, 1e-9) = 1.5707963267948966192313206...
/// Reachable only where the compute tier is narrow (realtime, compact).
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[test]
fn atan2_with_a_huge_ratio_is_correctly_rounded() {
    #[cfg(table_format = "q16_16")]
    let (y, x, expected) = (1_000_000 * 10_000i128, 1i128, 15708i128);
    #[cfg(table_format = "q32_32")]
    let (y, x, expected) = (10i128.pow(15) * 10i128.pow(9), 1i128, 1_570_796_327i128);
    #[cfg(table_format = "q16_16")]
    type T = DecimalFixed<4>;
    #[cfg(table_format = "q32_32")]
    type T = DecimalFixed<9>;
    for (yr, xr, e) in [(y, x, expected), (-y, x, -expected), (y, -x, expected), (-y, -x, -expected)] {
        let (yv, xv) = (T::from_raw(yr), T::from_raw(xr));
        assert_eq!(yv.try_atan2(xv).map(|v| v.raw_value()), Ok(e), "try_atan2({yr}, {xr})");
        assert_eq!(yv.atan2(xv).raw_value(), e, "atan2({yr}, {xr})");
    }
}

// ============================================================================
// Boundary values (as FixedPoint pins them)
// ============================================================================

fn boundary_cases<const D: u8>(failures: &mut Vec<String>) {
    let s = DecimalFixed::<D>::SCALE;
    let pi_half = round_str(PI_HALF, D);
    // pi is beyond i128 at D = 38 (3.14e38 > 1.7e38): TierOverflow there
    let pi = try_round_str(PI, D);
    let pi_want = |v: Result<R<Vec<i128>>, String>| match pi {
        Some(p) => matches!(v, Ok(Ok(ref w)) if *w == vec![p]),
        None => matches!(v, Ok(Err(OverflowDetected::TierOverflow))),
    };
    if !pi_want(agree_unary::<D>("acos", -s)) {
        failures.push(format!("acos<{D}>(-1): {:?}, expected pi {pi:?}", agree_unary::<D>("acos", -s)));
    }
    if !pi_want(agree_atan2::<D>(0, -s)) {
        failures.push(format!("atan2<{D}>(0, -1): {:?}, expected pi {pi:?}", agree_atan2::<D>(0, -s)));
    }
    let cases: &[(&str, i128, Vec<i128>)] = &[
        ("asin", s, vec![pi_half]),
        ("asin", -s, vec![-pi_half]),
        ("acos", s, vec![0]),
        ("acos", 0, vec![pi_half]),
        ("atanh", 0, vec![0]),
        ("asinh", 0, vec![0]),
        ("acosh", s, vec![0]),
        ("tanh", 0, vec![0]),
        ("sinh", 0, vec![0]),
        ("cosh", 0, vec![s]),
        ("sinhcosh", 0, vec![0, s]),
        ("sin", 0, vec![0]),
        ("cos", 0, vec![s]),
        ("sincos", 0, vec![0, s]),
        ("tan", 0, vec![0]),
        ("atan", 0, vec![0]),
        ("exp", 0, vec![s]),
        ("ln", s, vec![0]),
        ("sqrt", 0, vec![0]),
        ("sqrt", s, vec![s]),
        ("sqrt", s / 25, vec![s / 5]), // sqrt(0.04) = 0.2
    ];
    for (f, raw, want) in cases {
        match agree_unary::<D>(f, *raw) {
            Ok(Ok(v)) if v == *want => {}
            other => failures.push(format!("{f}<{D}>(raw {raw}): {other:?}, expected Ok({want:?})")),
        }
    }
    for (y, x, want) in [(0, s, 0), (s, 0, pi_half), (-s, 0, -pi_half)] {
        match agree_atan2::<D>(y, x) {
            Ok(Ok(v)) if v == vec![want] => {}
            other => failures.push(format!("atan2<{D}>({y}, {x}): {other:?}, expected Ok([{want}])")),
        }
    }
}

/// asin(+-1) = +-pi/2 and acos(-1) = pi correctly rounded to D, acos(1) = 0,
/// and the zeros and ones of every function exact, at every D the decimal
/// tests use on the profile (all at or below the compute dp).
#[test]
fn boundary_values_are_exact() {
    let mut failures = Vec::new();
    for &d in PROFILE_D {
        at_d!(d, boundary_cases(&mut failures));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
