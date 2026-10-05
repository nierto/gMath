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
//! | [`sigmoid_q64`] | any `i128` | in `[0, 1]` | 3 |
//! | [`softplus_q64`] | any `i128` | saturates to `i128::MAX` | 9 |
//! | [`silu_q64`] | any `i128` | `|result| <= |x|` | 79 at `|x|` near 40 (the sigmoid error times `|x|`) |
//! | [`sqrt_q64`], [`sqrt_q64_to`] | `x >= 0` | `None` for `x < 0` | 0: correctly rounded, proven |
//!
//! The gate functions (0.6.7) were measured on 400 inputs each with
//! `|x| <= 40`, given at 40 fractional bits. Narrowed once with
//! [`narrow_q64`] to 40 or to 20 fractional bits, sigmoid, softplus and silu
//! were the nearest value on every input: the error above is far below half
//! a unit at those precisions, so the narrowed result can differ from the
//! nearest value only when the exact value lies within that error of a
//! rounding boundary. That is rare, not impossible. The square root has no
//! such case: [`sqrt_q64_to`] rounds once from the exact radicand.
//!
//! [`exp_q64`] narrowed the same way was the nearest value on 3,675 of 3,675
//! inputs with `x <= 1`. For larger `x` its error grows with the result
//! (it is relative), and the narrowed value is often not the nearest one
//! (68 of 325 inputs at 40 bits): do not narrow it there and expect a
//! correctly rounded exponential.
//!
//! Integer helpers, exact: [`mul_div_floor`], [`mul_div_nearest`],
//! [`mul_div_floor_u128`] (the product formed at full width) and
//! [`try_ratio_from_str`] (a decimal literal as a fraction in lowest terms).
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
use crate::fixed_point::i256::mul_i128_to_i256;
use crate::fixed_point::{I256, I512};

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

// ============================================================================
// Gate functions at Q64.64
// ============================================================================

/// `round(a * b / 2^64)` for Q64.64 operands, nearest with ties toward
/// +infinity. The caller guarantees the product fits.
#[inline]
fn mul_q64(a: i128, b: i128) -> i128 {
    let p = mul_i128_to_i256(a, b);
    let mut q = p >> 64u32;
    if p.words[0] >= 1u64 << 63 {
        q = q + I256::from_i128(1);
    }
    q.as_i128()
}

/// `round(num * 2^64 / den)` for `0 <= num <= 2^64` and `2^64 <= den <= 2^65`,
/// nearest with ties upward: the quotient of the sigmoid, whose numerator is
/// at most one and whose denominator lies in `[1, 2]`.
#[inline]
fn div_unit_q64(num: u128, den: u128) -> i128 {
    let (q, r) = if num == 1u128 << 64 {
        // 2^128 / den through (2^128 - 1) / den
        let (q, r) = (u128::MAX / den, u128::MAX % den + 1);
        if r == den { (q + 1, 0) } else { (q, r) }
    } else {
        let n = num << 64;
        (n / den, n % den)
    };
    (if 2 * r >= den { q + 1 } else { q }) as i128
}

/// `1 / (1 + e^-x)` for `x` in Q64.64, in `[0, 2^64]`.
///
/// Sign-split so the exponential argument is never positive and the
/// denominator stays in `[1, 2]`: nothing overflows for any `x`. Exactly
/// `2^63` at `x = 0`; `0` for `x < -40` and `2^64` for `x > 40` (the true
/// value is within 78 units of those there).
pub fn sigmoid_q64(x: i128) -> i128 {
    let one = ONE_Q64 as u128;
    if x < 0 {
        let e = exp_q64_64_native(x) as u128; // in [0, 1]
        div_unit_q64(e, one + e)
    } else {
        // -x cannot overflow: x >= 0
        let e = exp_q64_64_native(-x) as u128;
        div_unit_q64(one, one + e)
    }
}

/// `ln(1 + e^x)` for `x` in Q64.64.
///
/// Computed as `max(x, 0) + ln(1 + e^-|x|)`, so the exponential argument is
/// never positive. Saturates to `i128::MAX` when the result does not fit
/// (only for `x` within one unit of the top of the range).
pub fn softplus_q64(x: i128) -> i128 {
    let neg_abs = if x < 0 { x } else { -x };
    let e = exp_q64_64_native(neg_abs); // in [0, 1]
    let corr = ln_q64_64_native(ONE_Q64 + e); // argument in [1, 2]
    if x < 0 { corr } else { x.checked_add(corr).unwrap_or(i128::MAX) }
}

/// `x * sigmoid(x)` for `x` in Q64.64: [`sigmoid_q64`] times `x`, the
/// product rounded once to nearest. `|silu(x)| <= |x|`, so it always fits.
pub fn silu_q64(x: i128) -> i128 {
    mul_q64(x, sigmoid_q64(x))
}

/// `sqrt(x)` for `x` in Q64.64, correctly rounded: the nearest Q64.64 value
/// to the exact root (an exact root is returned as is). `None` for `x < 0`.
///
/// Integer arithmetic on the exact radicand `x * 2^64`; no approximation is
/// involved, so this is a proven bound, not a measured one.
pub fn sqrt_q64(x: i128) -> Option<i128> {
    sqrt_q64_to(x, 64)
}

/// `sqrt(x)` for `x` in Q64.64, correctly rounded to `frac_bits` fractional
/// bits (`frac_bits <= 64`): the nearest value at that precision to the
/// exact root, in one rounding. `None` for `x < 0`.
///
/// Use this instead of narrowing [`sqrt_q64`]: narrowing an already rounded
/// root rounds twice, and the two differ when the exact root lies just
/// below a midpoint (for example `sqrt(1 + 2^-40)` at 40 bits).
///
/// # Panics
/// Panics if `frac_bits > 64`.
pub fn sqrt_q64_to(x: i128, frac_bits: u32) -> Option<i128> {
    assert!(frac_bits <= 64, "sqrt_q64_to: frac_bits above 64");
    if x < 0 {
        return None;
    }
    if x == 0 {
        return Some(0);
    }
    // sqrt(x / 2^64) * 2^f = sqrt(x * 2^(2f - 64)). With t = 2f - 64 (even):
    // t >= 0: n = x * 2^t exactly. t < 0: the root of x / 2^-t, whose floor
    // is the floor root of floor(x / 2^-t).
    let f2 = 2 * frac_bits;
    let (n, down) = if f2 >= 64 {
        (I256::from_u128(x as u128) << (f2 - 64) as usize, 0)
    } else {
        (I256::from_u128((x as u128) >> (64 - f2)), 64 - f2)
    };
    if n.is_zero() {
        // 0 < x / 2^down < 1: the root is below one unit; it is at least one
        // half exactly when x * 4 >= 2^down
        return Some(if (x as u128) >> (down - 2) != 0 { 1 } else { 0 });
    }
    // r = floor(sqrt(n)) by Newton from above; n < 2^191
    let bits = 256 - n.words.iter().rev().enumerate().find(|(_, &w)| w != 0).map_or(256, |(i, w)| i as u32 * 64 + w.leading_zeros());
    let mut r = I256::from_i128(1) << ((bits + 1) / 2) as usize; // >= sqrt(n)
    loop {
        let next = (r + n / r) >> 1u32;
        if next >= r {
            break;
        }
        r = next;
    }
    // nearest: (r + 1/2)^2 = r^2 + r + 1/4, so round up exactly when the
    // radicand exceeds r^2 + r. For t < 0 the radicand is x / 2^down and the
    // test is x > (r^2 + r) * 2^down, on integers.
    let mid = r * r + r;
    let up = if down == 0 { n > mid } else { I256::from_u128(x as u128) > (mid << down as usize) };
    Some((if up { r + I256::from_i128(1) } else { r }).as_i128())
}

/// A Q64.64 value rounded to `frac_bits` fractional bits (`frac_bits <= 64`),
/// nearest with ties toward +infinity: the one narrowing after a wide-tier
/// computation. `narrow_q64(v, 64) == v`.
///
/// # Panics
/// Panics if `frac_bits > 64`.
#[inline]
pub fn narrow_q64(v: i128, frac_bits: u32) -> i128 {
    assert!(frac_bits <= 64, "narrow_q64: frac_bits above 64");
    let shift = 64 - frac_bits;
    if shift == 0 {
        v
    } else {
        (v >> shift) + ((v >> (shift - 1)) & 1)
    }
}

// ============================================================================
// Full-width integer mul_div and exact decimal ratios
// ============================================================================

/// The exact product `a * b` divided by `d`: `(floor quotient, remainder)`
/// with the remainder taking the sign of `d` (or zero).
fn mul_div_parts(a: i128, b: i128, d: i128) -> Result<(I256, I256, I256), OverflowDetected> {
    if d == 0 {
        return Err(OverflowDetected::DivisionByZero);
    }
    let p = mul_i128_to_i256(a, b);
    let dw = I256::from_i128(d);
    let mut q = p / dw; // truncated toward zero
    let mut r = p - q * dw;
    if !r.is_zero() && (r.is_negative() != dw.is_negative()) {
        q = q - I256::from_i128(1);
        r = r + dw;
    }
    Ok((q, r, dw))
}

/// `floor(a * b / d)` with the product formed exactly in 256 bits.
///
/// `Err(DivisionByZero)` for `d == 0`, `Err(TierOverflow)` when the quotient
/// does not fit `i128`. The product itself cannot overflow.
pub fn mul_div_floor(a: i128, b: i128, d: i128) -> Result<i128, OverflowDetected> {
    let (q, _, _) = mul_div_parts(a, b, d)?;
    if q.fits_in_i128() { Ok(q.as_i128()) } else { Err(OverflowDetected::TierOverflow) }
}

/// `a * b / d` rounded to the nearest integer, ties toward +infinity, with
/// the product formed exactly in 256 bits. Errors as [`mul_div_floor`].
pub fn mul_div_nearest(a: i128, b: i128, d: i128) -> Result<i128, OverflowDetected> {
    let (q, r, dw) = mul_div_parts(a, b, d)?;
    // the fractional part is r / d in [0, 1): round up from one half
    let (r_abs, d_abs) = if dw.is_negative() { (-r, -dw) } else { (r, dw) };
    let q = if r_abs + r_abs >= d_abs { q + I256::from_i128(1) } else { q };
    if q.fits_in_i128() { Ok(q.as_i128()) } else { Err(OverflowDetected::TierOverflow) }
}

/// `floor(a * b / d)` for unsigned operands, the product formed exactly.
///
/// `Err(DivisionByZero)` for `d == 0`, `Err(TierOverflow)` when the quotient
/// does not fit `u128`.
pub fn mul_div_floor_u128(a: u128, b: u128, d: u128) -> Result<u128, OverflowDetected> {
    if d == 0 {
        return Err(OverflowDetected::DivisionByZero);
    }
    let q = (I512::from_u128(a) * I512::from_u128(b)) / I512::from_u128(d);
    if q.words[2..].iter().any(|&w| w != 0) {
        return Err(OverflowDetected::TierOverflow);
    }
    Ok(q.words[0] as u128 | (q.words[1] as u128) << 64)
}

/// Parse a decimal literal to the exact fraction it denotes, in lowest terms:
/// `(numerator, denominator)` with `denominator > 0`.
///
/// `"0.25"` gives `(1, 4)`, `"10000000.0"` gives `(10000000, 1)`, `"-1.5e1"`
/// gives `(-15, 1)`. A literal denotes an integer exactly when the
/// denominator is 1. Nothing is rounded.
///
/// Grammar as [`try_from_str`]: `[+-]? digits [. digits] [e [+-]? digits]`,
/// also `.5` and `5.`, surrounding whitespace ignored.
///
/// `Err(ParseError)` for a string outside the grammar, `Err(TierOverflow)`
/// when the reduced numerator or denominator does not fit `i128`.
pub fn try_ratio_from_str(s: &str) -> Result<(i128, i128), OverflowDetected> {
    const PARSE: OverflowDetected = OverflowDetected::ParseError;
    const RANGE: OverflowDetected = OverflowDetected::TierOverflow;
    let s = s.trim().as_bytes();
    let (negative, s) = match s.first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let (mantissa, exponent) = match s.iter().position(|&c| c == b'e' || c == b'E') {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };
    let (int_digits, frac_digits) = match mantissa.iter().position(|&c| c == b'.') {
        Some(i) => (&mantissa[..i], &mantissa[i + 1..]),
        None => (mantissa, &mantissa[..0]),
    };
    let all_digits = |d: &[u8]| d.iter().all(u8::is_ascii_digit);
    if int_digits.len() + frac_digits.len() == 0 || !all_digits(int_digits) || !all_digits(frac_digits) {
        return Err(PARSE);
    }
    let mut exp: i64 = 0;
    if let Some(e) = exponent {
        let (e_neg, e) = match e.first() {
            Some(b'-') => (true, &e[1..]),
            Some(b'+') => (false, &e[1..]),
            _ => (false, e),
        };
        if e.is_empty() || !all_digits(e) {
            return Err(PARSE);
        }
        for &c in e {
            // an exponent this large cannot give a representable value
            exp = (exp * 10 + (c - b'0') as i64).min(1_000_000);
        }
        if e_neg {
            exp = -exp;
        }
    }
    // trailing fraction zeros carry no value and would inflate the denominator
    let frac_digits = &frac_digits[..frac_digits.iter().rposition(|&c| c != b'0').map_or(0, |i| i + 1)];
    let mut num: i128 = 0;
    for &c in int_digits.iter().chain(frac_digits) {
        num = num.checked_mul(10).and_then(|n| n.checked_add((c - b'0') as i128)).ok_or(RANGE)?;
    }
    if num == 0 {
        return Ok((0, 1));
    }
    // value = num * 10^(exp - frac_digits.len())
    let mut scale = exp - frac_digits.len() as i64;
    let mut den: i128 = 1;
    while scale > 0 {
        num = num.checked_mul(10).ok_or(RANGE)?;
        scale -= 1;
    }
    while scale < 0 {
        // divide out what is common before growing the denominator
        if num % 10 == 0 {
            num /= 10;
        } else {
            den = den.checked_mul(10).ok_or(RANGE)?;
        }
        scale += 1;
    }
    let (mut a, mut b) = (num, den);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    Ok((if negative { -(num / a) } else { num / a }, den / a))
}
