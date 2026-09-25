//! # Wide Tier: Q64.64 Transcendentals and Constants
//!
//! Integer-only `exp`, `ln`, `sin`, `cos` and `π` in Q64.64 format (`i128`,
//! value = `raw / 2^64`), for quantities outside every storage tier, such as
//! RoPE inverse frequencies `theta^(-2i/d)` with `theta` up to `1e7`.
//!
//! These are the Q64.64 engines that the realtime and compact profiles run at
//! their compute tier. They are the same functions with the same results on
//! every profile. Before 0.6.4 they were reachable only through the
//! undocumented `domains` path (`exp_q64_64_native`, `ln_q64_64_native`),
//! which they match bit for bit.
//!
//! ## Contract
//!
//! Errors are in Q64.64 units (`2^-64`) against a 120-digit mpmath reference,
//! measured on the inputs of `tests/wide_q64_validation.rs`; they are
//! measured bounds, not proven ones.
//!
//! | fn | domain | result | measured error |
//! |----|--------|--------|----------------|
//! | [`exp_q64`] | any `i128` | saturates to `i128::MAX` for `x >= 41`, flushes to 0 for `x < -40` | 4 units of `2^-64` relative to the result (absolute below 1) |
//! | [`ln_q64`] | `x > 0` | `None` for `x <= 0` | 55 |
//! | [`sin_q64`], [`cos_q64`], [`sincos_q64`] | any `i128` | in `[-1, 1]` | 3 for `|x| <= 2π`, then growing as `0.34 |x|` |
//!
//! The sine and cosine error grows with the angle because range reduction
//! subtracts multiples of the truncated `PI_HALF_Q64`, which is 0.537 units
//! below π/2. Measured worst cases: 1390 units for `|x| < 2^12`, 3.6e5 below
//! `2^20`, 1.4e9 below `2^32` (8e-11 absolute), 1.5e18 below `2^62` (0.08
//! absolute). `exp_q64(ln_q64(x))` returns `x` within 33 units relative for
//! `x` in `[2^40, 2^100]`.
//!
//! Exact points: `exp_q64(0) == ONE_Q64`, `ln_q64(ONE_Q64) == Some(0)`,
//! `sincos_q64(0) == (0, ONE_Q64)`. Rounding direction and monotonicity are
//! not guaranteed.
//!
//! The constants are `floor(c * 2^f)` of the exact value, each truncated
//! independently (so `TWO_PI_Q64 == 2 * PI_Q64 == 4 * PI_HALF_Q64 + 2`).
//! `PI_Q64` and `PI_HALF_Q64` are the constants the sine and cosine engines
//! use for range reduction.

use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::domains::binary_fixed::transcendental::{
    exp_tier_n_plus_1::exp_q64_64_native,
    ln_tier_n_plus_1::ln_q64_64_native,
    sin_cos_tier_n_plus_1::{self as sincos, sincos_q64_64},
};
use crate::fixed_point::imperative::decimal_literal;

/// `1.0` in Q64.64.
pub const ONE_Q64: i128 = 1 << 64;

/// `floor(π * 2^64)`.
pub const PI_Q64: i128 = sincos::PI_Q64;

/// `floor(2π * 2^64)`.
pub const TWO_PI_Q64: i128 = 115904311329233965478;

/// `floor(π/2 * 2^64)`.
pub const PI_HALF_Q64: i128 = sincos::PI_HALF_Q64;

/// `floor(π * 2^32)`, the Q32.32 angle format of `FixedPoint::sincos_wide`.
pub const PI_Q32: i64 = 13493037704;

/// `floor(2π * 2^32)`.
pub const TWO_PI_Q32: i64 = 26986075409;

/// `e^x` for `x` in Q64.64.
///
/// Returns `i128::MAX` when `x >= 41` (the true value is representable up to
/// `x ≈ 43.67` but is not computed) and `0` when `x < -40` (the true value is
/// below 78.4 units there).
#[inline]
pub fn exp_q64(x: i128) -> i128 {
    exp_q64_64_native(x)
}

/// `ln(x)` for `x` in Q64.64; `None` when `x <= 0`.
#[inline]
pub fn ln_q64(x: i128) -> Option<i128> {
    if x <= 0 { None } else { Some(ln_q64_64_native(x)) }
}

/// `sin(x)` for an angle `x` in radians, Q64.64.
#[inline]
pub fn sin_q64(x: i128) -> i128 {
    sincos_q64_64(x).0
}

/// `cos(x)` for an angle `x` in radians, Q64.64.
#[inline]
pub fn cos_q64(x: i128) -> i128 {
    sincos_q64_64(x).1
}

/// `(sin(x), cos(x))` with one shared range reduction.
///
/// Equal to `(sin_q64(x), cos_q64(x))`.
#[inline]
pub fn sincos_q64(x: i128) -> (i128, i128) {
    sincos_q64_64(x)
}

/// Parse a decimal literal to a raw `i128` with `frac_bits` fractional bits.
///
/// For example `try_from_str("1e-5", 64)` for a Q64.64 epsilon.
///
/// Grammar: `[+-]? digits [. digits] [e [+-]? digits]`, also `.5` and `5.`,
/// surrounding whitespace ignored; no expressions or constants. Converted
/// exactly with integer arithmetic and rounded once to nearest, ties toward
/// +infinity, the same rule as `FixedPoint::try_from_str`. Any number of
/// digits is accepted.
///
/// `Err(ParseError)` for a string outside the grammar, `Err(TierOverflow)`
/// when the value does not fit `i128`, `Err(InvalidInput)` when
/// `frac_bits > 127`.
pub fn try_from_str(s: &str, frac_bits: u32) -> Result<i128, OverflowDetected> {
    if frac_bits > 127 {
        return Err(OverflowDetected::InvalidInput);
    }
    match decimal_literal::parse(s, frac_bits, 128) {
        Some(converted) => converted.map(|c| decimal_literal::to_i128(&c)),
        None => Err(OverflowDetected::ParseError),
    }
}
