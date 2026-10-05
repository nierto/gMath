//! Gate for the 0.6.7 wide-tier functions.
//!
//! `g_math::wide`: sigmoid, softplus, silu and sqrt at Q64.64 against mpmath
//! (140 digits; sqrt against exact integer arithmetic), measured in Q64.64
//! units and, narrowed once, required to equal the nearest value at 40 and
//! at 20 fractional bits. Full-width `mul_div` and exact decimal ratios
//! against Python integers and `Fraction`.
//!
//! `fused::sigmoid_mul` and `fused::entropy` against 120-digit references
//! parsed at the build's split, in storage units.
//!
//! References: `scripts/generate_gate_refs.py`. Integer arithmetic only.

use g_math::fixed_point::imperative::fused::{entropy, sigmoid_mul, sigmoid_mul_in_place, sigmoid_mul_slice};
use g_math::fixed_point::wide::{
    mul_div_floor, mul_div_floor_u128, mul_div_nearest, narrow_q64, sigmoid_q64, silu_q64, softplus_q64, sqrt_q64, sqrt_q64_to,
    try_ratio_from_str, ONE_Q64,
};
use g_math::fixed_point::{FixedPoint, OverflowDetected};

#[allow(dead_code)]
mod refs {
    include!("data/gate_refs.rs");
}

fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }

/// |got - want| in storage units, saturating at 2^30.
fn units(got: FixedPoint, want: FixedPoint) -> i32 {
    let half = fp("0.5");
    let mut unit = FixedPoint::one();
    for _ in 0..g_math::fixed_point::frac_config::FRAC_BITS { unit = unit * half; }
    let cap = FixedPoint::try_from_int(1 << 30).ok();
    match got.try_sub(want).and_then(|d| d.abs().try_div(unit)) {
        Ok(q) if cap.map_or(true, |c| q < c) => q.to_int(),
        _ => 1 << 30,
    }
}

/// One Q64.64 function over its table: the worst error in Q64.64 units and
/// how many narrowed results differ from the nearest value at Q.40 and Q.20.
fn q64_table(name: &str, table: &[(i128, i128, i128, i128)], f: impl Fn(i128) -> i128) -> (u128, usize, usize) {
    let (mut worst, mut off40, mut off20) = (0u128, 0, 0);
    for &(x_q40, want64, want40, want20) in table {
        let got = f(x_q40 << 24);
        worst = worst.max(got.abs_diff(want64));
        off40 += (narrow_q64(got, 40) != want40) as usize;
        off20 += (narrow_q64(got, 20) != want20) as usize;
    }
    println!("{name}: {} inputs, worst {worst} units of 2^-64, {off40} not nearest at Q.40, {off20} not nearest at Q.20", table.len());
    (worst, off40, off20)
}

#[test]
fn sigmoid_softplus_silu_at_q64() {
    // Measured bounds, in units of 2^-64. The functions are built on the
    // Q64.64 exp and ln engines, which are a few units off at that
    // precision; narrowing leaves 24 or 44 guard bits, so the narrowed
    // value is the nearest one unless the exact value lies within the
    // error of a rounding boundary, which no input here does.
    let (worst, off40, off20) = q64_table("sigmoid", refs::SIGMOID, sigmoid_q64);
    assert!(worst <= SIGMOID_BOUND, "sigmoid {worst}");
    assert_eq!((off40, off20), (0, 0));
    let (worst, off40, off20) = q64_table("softplus", refs::SOFTPLUS, softplus_q64);
    assert!(worst <= SOFTPLUS_BOUND, "softplus {worst}");
    assert_eq!((off40, off20), (0, 0));
    let (worst, off40, off20) = q64_table("silu", refs::SILU, silu_q64);
    assert!(worst <= SILU_BOUND, "silu {worst}");
    assert_eq!((off40, off20), (0, 0));

    // exact points and the saturated ends
    assert_eq!(sigmoid_q64(0), ONE_Q64 / 2);
    assert_eq!(silu_q64(0), 0);
    assert_eq!(sigmoid_q64(i128::MIN), 0);
    assert_eq!(sigmoid_q64(i128::MAX), ONE_Q64);
    assert_eq!(softplus_q64(i128::MIN), 0);
    assert_eq!(softplus_q64(i128::MAX), i128::MAX);
    assert_eq!(silu_q64(i128::MAX), i128::MAX);
    assert_eq!(silu_q64(i128::MIN), 0);
    // sigmoid(-x) = 1 - sigmoid(x) within the measured error
    for &(x, ..) in refs::SIGMOID {
        let (a, b) = (sigmoid_q64(x << 24), sigmoid_q64(-(x << 24)));
        assert!((a + b).abs_diff(ONE_Q64) <= 2 * SIGMOID_BOUND, "symmetry at {x}");
    }
}

/// Measured 3 (2026-10-05).
const SIGMOID_BOUND: u128 = 4;
/// Measured 9.
const SOFTPLUS_BOUND: u128 = 12;
/// Measured 79 at |x| near 40: the sigmoid error times |x|, plus the product's rounding.
const SILU_BOUND: u128 = 4 * 40 + 1;

#[test]
fn sqrt_q64_is_correctly_rounded() {
    let (worst, off40, off20) = q64_table("sqrt", refs::SQRT, |x| sqrt_q64(x).unwrap());
    // exact integer arithmetic: the nearest Q64.64 value, always
    assert_eq!(worst, 0);
    // Narrowing that value rounds a second time. The table holds one input
    // built to show it (1 + 2^-40, whose root lies just below a midpoint at
    // 40 bits); sqrt_q64_to rounds once and is right there too.
    println!("sqrt, narrowed afterwards: {off40} not nearest at Q.40, {off20} at Q.20");
    assert_eq!((off40, off20), (1, 0));
    for &(x_q40, want64, want40, want20) in refs::SQRT {
        let x = x_q40 << 24;
        assert_eq!(sqrt_q64_to(x, 64), Some(want64), "sqrt at 64 bits, x = {x_q40}");
        assert_eq!(sqrt_q64_to(x, 40), Some(want40), "sqrt at 40 bits, x = {x_q40}");
        assert_eq!(sqrt_q64_to(x, 20), Some(want20), "sqrt at 20 bits, x = {x_q40}");
    }
    assert_eq!(sqrt_q64(-1), None);
    assert_eq!(sqrt_q64(0), Some(0));
    assert_eq!(sqrt_q64(4 * ONE_Q64), Some(2 * ONE_Q64));
    assert_eq!(sqrt_q64(ONE_Q64 / 4), Some(ONE_Q64 / 2));
    // down to no fractional bits: the nearest integer to the root
    assert_eq!(sqrt_q64_to(2 * ONE_Q64, 0), Some(1)); // 1.41
    assert_eq!(sqrt_q64_to(3 * ONE_Q64, 0), Some(2)); // 1.73
    assert_eq!(sqrt_q64_to(ONE_Q64 / 4, 0), Some(1)); // 0.5 rounds up
    assert_eq!(sqrt_q64_to(ONE_Q64 / 4 - 1, 0), Some(0));
    assert_eq!(sqrt_q64_to(1, 0), Some(0));
    assert_eq!(sqrt_q64_to(1, 32), Some(1)); // sqrt(2^-64) = 2^-32 exactly
    // the largest input: sqrt(2^63 - 2^-64) is just below 2^31.5
    let top = sqrt_q64(i128::MAX).unwrap();
    assert!(top > 3_037_000_499 * ONE_Q64 && top < 3_037_000_500 * ONE_Q64);
}

#[test]
fn narrow_q64_rounds_to_nearest_ties_up() {
    assert_eq!(narrow_q64(ONE_Q64, 20), 1 << 20);
    assert_eq!(narrow_q64(3 << 43, 20), 2); // 1.5 units -> 2
    assert_eq!(narrow_q64(-(3 << 43), 20), -1); // -1.5 units -> -1
    assert_eq!(narrow_q64(1 << 43, 20), 1); // 0.5 -> 1
    assert_eq!(narrow_q64(-(1 << 43), 20), 0); // -0.5 -> 0
    assert_eq!(narrow_q64((1 << 43) - 1, 20), 0);
    assert_eq!(narrow_q64(12345, 64), 12345);
    assert_eq!(narrow_q64(i128::MAX, 0), 1 << 63);
    assert_eq!(narrow_q64(i128::MIN, 0), -(1 << 63));
}

#[test]
fn mul_div_matches_exact_integers() {
    for &(a, b, d, floor, nearest) in refs::MUL_DIV {
        assert_eq!(mul_div_floor(a, b, d).ok(), floor, "floor({a} * {b} / {d})");
        assert_eq!(mul_div_nearest(a, b, d).ok(), nearest, "nearest({a} * {b} / {d})");
    }
    for &(a, b, d, floor) in refs::MUL_DIV_U128 {
        assert_eq!(mul_div_floor_u128(a, b, d).ok(), floor, "floor({a} * {b} / {d}) unsigned");
    }
    assert_eq!(mul_div_floor(1, 1, 0), Err(OverflowDetected::DivisionByZero));
    assert_eq!(mul_div_nearest(1, 1, 0), Err(OverflowDetected::DivisionByZero));
    assert_eq!(mul_div_floor_u128(1, 1, 0), Err(OverflowDetected::DivisionByZero));
    assert_eq!(mul_div_floor(i128::MAX, 2, 1), Err(OverflowDetected::TierOverflow));
    assert_eq!(mul_div_floor_u128(u128::MAX, 2, 1), Err(OverflowDetected::TierOverflow));
    let fits = refs::MUL_DIV.iter().filter(|c| c.3.is_some()).count();
    assert!(fits * 2 > refs::MUL_DIV.len(), "too few in-range cases: {fits}");
}

#[test]
fn ratios_are_exact() {
    for &(s, want) in refs::RATIO {
        assert_eq!(try_ratio_from_str(s).ok(), want, "{s:?}");
    }
    for &s in refs::RATIO_BAD {
        assert_eq!(try_ratio_from_str(s), Err(OverflowDetected::ParseError), "{s:?}");
    }
    assert_eq!(try_ratio_from_str("1e40"), Err(OverflowDetected::TierOverflow));
    assert_eq!(try_ratio_from_str("1e-40"), Err(OverflowDetected::TierOverflow));
    assert_eq!(try_ratio_from_str("1e999999999999"), Err(OverflowDetected::TierOverflow));
    // an integer literal is one whose denominator is 1; a ratio times a count is one mul_div
    assert_eq!(try_ratio_from_str("10000000.0"), Ok((10_000_000, 1)));
    let (num, den) = try_ratio_from_str("0.25").unwrap();
    assert_eq!((num, den), (1, 4));
    assert_eq!(mul_div_floor(num, 4096, den), Ok(1024));
    assert_eq!(mul_div_nearest(num, 4095, den), Ok(1024)); // 1023.75
}

#[test]
fn sigmoid_mul_rounds_once() {
    let mut worst = 0;
    for &(x, gate, want) in refs::SIGMOID_MUL {
        let got = sigmoid_mul(fp(x), fp(gate));
        worst = worst.max(units(got, fp(want)));
    }
    println!("sigmoid_mul: worst {worst} units over {} cases", refs::SIGMOID_MUL.len());
    assert_eq!(worst, 0, "the correctly rounded product on every case");

    // the slice forms are the scalar function element by element
    let xs: Vec<FixedPoint> = refs::SIGMOID_MUL.iter().map(|c| fp(c.0)).collect();
    let gs: Vec<FixedPoint> = refs::SIGMOID_MUL.iter().map(|c| fp(c.1)).collect();
    let each: Vec<FixedPoint> = xs.iter().zip(&gs).map(|(&x, &g)| sigmoid_mul(x, g)).collect();
    assert_eq!(sigmoid_mul_slice(&xs, &gs), each);
    let mut in_place = xs.clone();
    sigmoid_mul_in_place(&mut in_place, &gs);
    assert_eq!(in_place, each);

}

#[test]
fn entropy_rounds_once() {
    let mut worst = 0;
    for &(ws, want) in refs::ENTROPY {
        let w: Vec<FixedPoint> = ws.iter().map(|s| fp(s)).collect();
        worst = worst.max(units(entropy(&w).unwrap(), fp(want)));
    }
    println!("entropy: worst {worst} units over {} cases", refs::ENTROPY.len());
    assert!(worst <= ENTROPY_BOUND, "entropy {worst}");
    assert_eq!(entropy(&[]), Ok(FixedPoint::ZERO));
    assert_eq!(entropy(&[FixedPoint::one()]), Ok(FixedPoint::ZERO));
    assert_eq!(entropy(&[FixedPoint::ZERO, FixedPoint::ZERO]), Ok(FixedPoint::ZERO));
    assert_eq!(entropy(&[fp("0.5"), fp("-0.5")]), Err(OverflowDetected::DomainError));
}

const ENTROPY_BOUND: i32 = 0;
