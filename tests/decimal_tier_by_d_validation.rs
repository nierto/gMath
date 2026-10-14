//! `DecimalFixed<D>` transcendentals against mpmath at several `D` in one
//! build (phase 6a: the compute word follows `D`, not the build).
//!
//! Part 1 characterises every compute word at every `D`: each engine is run
//! on each word whose dp covers `D`, and the distance from the correctly
//! rounded value (half to even at `D`) is tallied per (D, word). Part 2 is
//! the gate on the public type: every function at every `D` within the
//! bound the characterisation justifies.
//!
//! References: `tests/data/decimal_tier_refs.rs` from
//! `scripts/generate_decimal_tier_refs.py` (mpmath, 200 digits).

use g_math::fixed_point::domains::decimal_fixed::transcendental::decimal_compute::{
    try_decimal_compute_to_i128, try_i128_upscale_to_compute,
};
use g_math::fixed_point::domains::decimal_fixed::transcendental::{
    decimal_atan, decimal_atan2, decimal_cos, decimal_exp, decimal_ln, decimal_sin, decimal_sinhcosh, decimal_sqrt,
    DecimalWord,
};
use g_math::fixed_point::{DecimalFixed, DecimalTier, Dp, I1024, I256, I512};
use std::collections::BTreeMap;

include!("data/decimal_tier_refs.rs");

type R = Result<i128, g_math::fixed_point::OverflowDetected>;

/// The engine `f` on the word `C` at `D`: upscale, run, downscale.
fn engine<C: DecimalWord>(f: &str, x: i128, d: u8) -> Option<R> {
    let up = |v: i128| try_i128_upscale_to_compute::<C>(v, d);
    let down = |v: C| try_decimal_compute_to_i128::<C>(v, d);
    let xc = match up(x) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let r = match f {
        "exp" => decimal_exp(xc),
        "ln" => decimal_ln(xc),
        "sqrt" => decimal_sqrt(xc),
        "sin" => decimal_sin(xc),
        "cos" => decimal_cos(xc),
        "atan" => decimal_atan(xc),
        "sinh" => decimal_sinhcosh(xc).map(|(s, _)| s),
        "cosh" => decimal_sinhcosh(xc).map(|(_, c)| c),
        _ => return None,
    };
    Some(r.and_then(down))
}

fn engine_atan2<C: DecimalWord>(y: i128, x: i128, d: u8) -> R {
    let yc = try_i128_upscale_to_compute::<C>(y, d)?;
    let xc = try_i128_upscale_to_compute::<C>(x, d)?;
    try_decimal_compute_to_i128::<C>(decimal_atan2(yc, xc)?, d)
}

#[derive(Default)]
struct Cell {
    cases: usize,
    exact: usize,
    max_units: u128,
    errors: usize,
}

fn tally(cells: &mut BTreeMap<(u8, &'static str), Cell>, d: u8, word: &'static str, got: R, want: i128) {
    let c = cells.entry((d, word)).or_default();
    c.cases += 1;
    match got {
        Ok(v) => {
            let u = (v - want).unsigned_abs();
            if u == 0 {
                c.exact += 1;
            }
            c.max_units = c.max_units.max(u);
        }
        Err(_) => c.errors += 1,
    }
}

/// Every word at every D: how far each engine lands from the correctly
/// rounded value. Printed, not asserted; the gate below pins the chosen word.
#[test]
fn every_word_at_every_d_characterised() {
    let mut cells: BTreeMap<(u8, &'static str), Cell> = BTreeMap::new();
    macro_rules! on_words {
        ($d:expr, $want:expr, $run:ident $(, $arg:expr)*) => {{
            if $d <= 9 { if let Some(r) = $run::<i64>($($arg,)* $d) { tally(&mut cells, $d, "i64", r, $want); } }
            if $d <= 19 { if let Some(r) = $run::<i128>($($arg,)* $d) { tally(&mut cells, $d, "i128", r, $want); } }
            if $d <= 38 { if let Some(r) = $run::<I256>($($arg,)* $d) { tally(&mut cells, $d, "I256", r, $want); } }
            if let Some(r) = $run::<I512>($($arg,)* $d) { tally(&mut cells, $d, "I512", r, $want); }
            if let Some(r) = $run::<I1024>($($arg,)* $d) { tally(&mut cells, $d, "I1024", r, $want); }
        }};
    }
    for &(d, f, x, want) in DECIMAL_TIER_REFS {
        on_words!(d, want, engine, f, x);
    }
    fn a2<C: DecimalWord>(y: i128, x: i128, d: u8) -> Option<R> {
        Some(engine_atan2::<C>(y, x, d))
    }
    for &(d, y, x, want) in DECIMAL_TIER_ATAN2_REFS {
        on_words!(d, want, a2, y, x);
    }
    println!("{:>3} {:<6} {:>6} {:>6} {:>6} {:>10}", "D", "word", "cases", "exact", "errors", "max units");
    for ((d, w), c) in &cells {
        println!("{:>3} {:<6} {:>6} {:>6} {:>6} {:>10}", d, w, c.cases, c.exact, c.errors, c.max_units);
    }
}

/// The public type at every D: the chosen word gives the correctly rounded
/// value on every case.
fn public<const D: u8>(f: &str, x: i128) -> R
where
    Dp<D>: DecimalTier,
{
    let v = DecimalFixed::<D>::from_raw(x);
    let r = |r: Result<DecimalFixed<D>, _>| r.map(|v| v.raw_value());
    match f {
        "exp" => r(v.try_exp()),
        "ln" => r(v.try_ln()),
        "sqrt" => r(v.try_sqrt()),
        "sin" => r(v.try_sin()),
        "cos" => r(v.try_cos()),
        "tan" => r(v.try_tan()),
        "atan" => r(v.try_atan()),
        "asin" => r(v.try_asin()),
        "acos" => r(v.try_acos()),
        "sinh" => r(v.try_sinh()),
        "cosh" => r(v.try_cosh()),
        "tanh" => r(v.try_tanh()),
        "asinh" => r(v.try_asinh()),
        "acosh" => r(v.try_acosh()),
        "atanh" => r(v.try_atanh()),
        _ => panic!("unknown function {f}"),
    }
}

fn public_atan2<const D: u8>(y: i128, x: i128) -> R
where
    Dp<D>: DecimalTier,
{
    DecimalFixed::<D>::from_raw(y).try_atan2(DecimalFixed::<D>::from_raw(x)).map(|v| v.raw_value())
}

macro_rules! at_d {
    ($d:expr, $f:ident $(, $arg:expr)*) => {
        match $d {
            2 => $f::<2>($($arg),*), 4 => $f::<4>($($arg),*), 6 => $f::<6>($($arg),*), 9 => $f::<9>($($arg),*),
            12 => $f::<12>($($arg),*), 19 => $f::<19>($($arg),*), 28 => $f::<28>($($arg),*), 38 => $f::<38>($($arg),*),
            other => panic!("no arm for D = {other}"),
        }
    };
}

#[test]
fn decimal_fixed_is_correctly_rounded_at_every_d() {
    let mut worst: BTreeMap<(u8, &str), u128> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut n = 0;
    for &(d, f, x, want) in DECIMAL_TIER_REFS {
        n += 1;
        match at_d!(d, public, f, x) {
            Ok(v) => {
                let u = (v - want).unsigned_abs();
                let w = worst.entry((d, f)).or_default();
                *w = (*w).max(u);
                if u > 0 {
                    failures.push(format!("{f}<{d}>({x}): got {v}, mpmath {want}, {u} unit(s)"));
                }
            }
            Err(e) => failures.push(format!("{f}<{d}>({x}): {e:?}, mpmath {want}")),
        }
    }
    for &(d, y, x, want) in DECIMAL_TIER_ATAN2_REFS {
        n += 1;
        match at_d!(d, public_atan2, y, x) {
            Ok(v) => {
                let u = (v - want).unsigned_abs();
                if u > 0 {
                    failures.push(format!("atan2<{d}>({y}, {x}): got {v}, mpmath {want}, {u} unit(s)"));
                }
            }
            Err(e) => failures.push(format!("atan2<{d}>({y}, {x}): {e:?}, mpmath {want}")),
        }
    }
    // the hard corpus (next to singularities and range limits): reported per
    // function, and gated like the rest
    let mut hard_off: BTreeMap<&str, (usize, usize, u128)> = BTreeMap::new();
    for &(d, f, x, want) in DECIMAL_TIER_HARD_REFS {
        n += 1;
        let e = hard_off.entry(f).or_default();
        e.0 += 1;
        match at_d!(d, public, f, x) {
            Ok(v) => {
                let u = (v - want).unsigned_abs();
                if u > 0 {
                    e.1 += 1;
                    e.2 = e.2.max(u);
                    failures.push(format!("HARD {f}<{d}>({x}): got {v}, mpmath {want}, {u} unit(s)"));
                }
            }
            Err(err) => failures.push(format!("HARD {f}<{d}>({x}): {err:?}, mpmath {want}")),
        }
    }
    for (f, (c, off, u)) in &hard_off {
        println!("  hard {f:<6} {c:>3} cases, {off:>2} off, max {u} unit(s)");
    }
    println!("{n} cases, {} off", failures.len());
    for ((d, f), u) in &worst {
        if *u > 0 {
            println!("  worst {f}<{d}>: {u} unit(s)");
        }
    }
    assert!(failures.is_empty(), "{} of {n} cases off:\n{}", failures.len(), failures[..failures.len().min(40)].join("\n"));
}
