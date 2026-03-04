//! Tiered Rational Arithmetic Operations
//!
//! Pure mathematical functions implementing tiered rational arithmetic with overflow detection.
//! Enables precise tier-specific calculations for the UniversalNumber coordination system.
//!
//! ARCHITECTURE: Pure functions with zero side effects and thread-safe by design
//! PRECISION: Mathematically exact results with automatic tier promotion signals
//! INTEGRATION: Compatible with UGOD unified overflow detection

use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;
use crate::fixed_point::i256::I256;

// ================================================================================================
// GCD UTILITY
// ================================================================================================

/// Unsigned GCD via Euclidean algorithm (used by all tiers for reduction)
pub fn gcd_unsigned(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    if a == 0 { 1 } else { a }
}

/// GCD for I256 values (Euclidean algorithm on absolute values)
fn gcd_i256(mut a: I256, mut b: I256) -> I256 {
    if a.is_negative() { a = -a; }
    if b.is_negative() { b = -b; }
    while !b.is_zero() {
        let temp = b;
        b = a % b;
        a = temp;
    }
    if a.is_zero() { I256::from_i128(1) } else { a }
}

/// GCD-reduce an I256 rational and try to fit into i128/u128.
/// Must reduce BEFORE conversion since intermediates (e.g. 2^64 * 2^64 = 2^128)
/// may exceed i128 range even when the reduced result fits.
fn i256_reduce_to_i128(num: I256, den: I256) -> Result<(i128, u128), OverflowDetected> {
    if num.is_zero() { return Ok((0, 1)); }

    let gcd = gcd_i256(num, den);
    let reduced_num = num / gcd;
    let reduced_den = den / gcd;

    // Ensure denominator is positive
    let (final_num, final_den) = if reduced_den.is_negative() {
        (-reduced_num, -reduced_den)
    } else {
        (reduced_num, reduced_den)
    };

    if final_num.fits_in_i128() && final_den.fits_in_i128() {
        Ok((final_num.as_i128(), final_den.as_i128() as u128))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

// ================================================================================================
// TIER 1: TINY (i8/u8) - 99.9% of simple fractions
// ================================================================================================

/// Addition for Tiny tier: (a/b) + (c/d) = (a×d + c×b)/(b×d)
pub fn add_i8_rational(a_num: i8, a_den: u8, b_num: i8, b_den: u8) -> Result<(i8, u8), OverflowDetected> {
    // Handle zero cases for performance
    if a_num == 0 { return Ok((b_num, b_den)); }
    if b_num == 0 { return Ok((a_num, a_den)); }

    // Calculate intermediate values with overflow detection
    let a_expanded = (a_num as i16).checked_mul(b_den as i16)
        .ok_or(OverflowDetected::TierOverflow)?;
    let b_expanded = (b_num as i16).checked_mul(a_den as i16)
        .ok_or(OverflowDetected::TierOverflow)?;
    let common_denominator = (a_den as u16).checked_mul(b_den as u16)
        .ok_or(OverflowDetected::TierOverflow)?;

    let result_numerator = a_expanded.checked_add(b_expanded)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs() as u128, common_denominator as u128);
    let reduced_num = (result_numerator as i128) / (g as i128);
    let reduced_den = (common_denominator as u128) / g;

    if reduced_num >= i8::MIN as i128 && reduced_num <= i8::MAX as i128 &&
       reduced_den <= u8::MAX as u128 {
        Ok((reduced_num as i8, reduced_den as u8))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Subtraction for Tiny tier: (a/b) - (c/d) = (a×d - c×b)/(b×d)
pub fn sub_i8_rational(a_num: i8, a_den: u8, b_num: i8, b_den: u8) -> Result<(i8, u8), OverflowDetected> {
    // Handle zero cases
    if b_num == 0 { return Ok((a_num, a_den)); }

    let a_expanded = (a_num as i16).checked_mul(b_den as i16)
        .ok_or(OverflowDetected::TierOverflow)?;
    let b_expanded = (b_num as i16).checked_mul(a_den as i16)
        .ok_or(OverflowDetected::TierOverflow)?;
    let common_denominator = (a_den as u16).checked_mul(b_den as u16)
        .ok_or(OverflowDetected::TierOverflow)?;

    let result_numerator = a_expanded.checked_sub(b_expanded)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs() as u128, common_denominator as u128);
    let reduced_num = (result_numerator as i128) / (g as i128);
    let reduced_den = (common_denominator as u128) / g;

    if reduced_num >= i8::MIN as i128 && reduced_num <= i8::MAX as i128 &&
       reduced_den <= u8::MAX as u128 {
        Ok((reduced_num as i8, reduced_den as u8))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Multiplication for Tiny tier: (a/b) × (c/d) = (a×c)/(b×d)
pub fn mul_i8_rational(a_num: i8, a_den: u8, b_num: i8, b_den: u8) -> Result<(i8, u8), OverflowDetected> {
    // Handle special cases
    if a_num == 0 || b_num == 0 { return Ok((0, 1)); }
    if a_num == 1 && a_den == 1 { return Ok((b_num, b_den)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    let result_numerator = (a_num as i16).checked_mul(b_num as i16)
        .ok_or(OverflowDetected::TierOverflow)?;
    let result_denominator = (a_den as u16).checked_mul(b_den as u16)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs() as u128, result_denominator as u128);
    let reduced_num = (result_numerator as i128) / (g as i128);
    let reduced_den = (result_denominator as u128) / g;

    if reduced_num >= i8::MIN as i128 && reduced_num <= i8::MAX as i128 &&
       reduced_den <= u8::MAX as u128 {
        Ok((reduced_num as i8, reduced_den as u8))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Division for Tiny tier: (a/b) ÷ (c/d) = (a×d)/(b×c)
pub fn div_i8_rational(a_num: i8, a_den: u8, b_num: i8, b_den: u8) -> Result<(i8, u8), OverflowDetected> {
    // Check for division by zero
    if b_num == 0 { return Err(OverflowDetected::PrecisionLoss); }

    // Handle special cases
    if a_num == 0 { return Ok((0, 1)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    // Division becomes multiplication: (a/b) ÷ (c/d) = (a×d)/(b×c)
    let result_numerator = (a_num as i16).checked_mul(b_den as i16)
        .ok_or(OverflowDetected::TierOverflow)?;
    let result_denominator = (a_den as u16).checked_mul(b_num.abs() as u16)
        .ok_or(OverflowDetected::TierOverflow)?;

    // Handle sign: if b_num was negative, negate result_numerator
    let final_numerator = if b_num < 0 { -result_numerator } else { result_numerator };

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(final_numerator.unsigned_abs() as u128, result_denominator as u128);
    let reduced_num = (final_numerator as i128) / (g as i128);
    let reduced_den = (result_denominator as u128) / g;

    if reduced_num >= i8::MIN as i128 && reduced_num <= i8::MAX as i128 &&
       reduced_den <= u8::MAX as u128 {
        Ok((reduced_num as i8, reduced_den as u8))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

// ================================================================================================
// TIER 2: SMALL (i16/u16) - Mathematical constants
// ================================================================================================

/// Addition for Small tier: (a/b) + (c/d) = (a×d + c×b)/(b×d)
pub fn add_i16_rational(a_num: i16, a_den: u16, b_num: i16, b_den: u16) -> Result<(i16, u16), OverflowDetected> {
    if a_num == 0 { return Ok((b_num, b_den)); }
    if b_num == 0 { return Ok((a_num, a_den)); }

    let a_expanded = (a_num as i32).checked_mul(b_den as i32)
        .ok_or(OverflowDetected::TierOverflow)?;
    let b_expanded = (b_num as i32).checked_mul(a_den as i32)
        .ok_or(OverflowDetected::TierOverflow)?;
    let common_denominator = (a_den as u32).checked_mul(b_den as u32)
        .ok_or(OverflowDetected::TierOverflow)?;

    let result_numerator = a_expanded.checked_add(b_expanded)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs() as u128, common_denominator as u128);
    let reduced_num = (result_numerator as i128) / (g as i128);
    let reduced_den = (common_denominator as u128) / g;

    if reduced_num >= i16::MIN as i128 && reduced_num <= i16::MAX as i128 &&
       reduced_den <= u16::MAX as u128 {
        Ok((reduced_num as i16, reduced_den as u16))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Subtraction for Small tier
pub fn sub_i16_rational(a_num: i16, a_den: u16, b_num: i16, b_den: u16) -> Result<(i16, u16), OverflowDetected> {
    if b_num == 0 { return Ok((a_num, a_den)); }

    let a_expanded = (a_num as i32).checked_mul(b_den as i32)
        .ok_or(OverflowDetected::TierOverflow)?;
    let b_expanded = (b_num as i32).checked_mul(a_den as i32)
        .ok_or(OverflowDetected::TierOverflow)?;
    let common_denominator = (a_den as u32).checked_mul(b_den as u32)
        .ok_or(OverflowDetected::TierOverflow)?;

    let result_numerator = a_expanded.checked_sub(b_expanded)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs() as u128, common_denominator as u128);
    let reduced_num = (result_numerator as i128) / (g as i128);
    let reduced_den = (common_denominator as u128) / g;

    if reduced_num >= i16::MIN as i128 && reduced_num <= i16::MAX as i128 &&
       reduced_den <= u16::MAX as u128 {
        Ok((reduced_num as i16, reduced_den as u16))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Multiplication for Small tier
pub fn mul_i16_rational(a_num: i16, a_den: u16, b_num: i16, b_den: u16) -> Result<(i16, u16), OverflowDetected> {
    if a_num == 0 || b_num == 0 { return Ok((0, 1)); }
    if a_num == 1 && a_den == 1 { return Ok((b_num, b_den)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    let result_numerator = (a_num as i32).checked_mul(b_num as i32)
        .ok_or(OverflowDetected::TierOverflow)?;
    let result_denominator = (a_den as u32).checked_mul(b_den as u32)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs() as u128, result_denominator as u128);
    let reduced_num = (result_numerator as i128) / (g as i128);
    let reduced_den = (result_denominator as u128) / g;

    if reduced_num >= i16::MIN as i128 && reduced_num <= i16::MAX as i128 &&
       reduced_den <= u16::MAX as u128 {
        Ok((reduced_num as i16, reduced_den as u16))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Division for Small tier
pub fn div_i16_rational(a_num: i16, a_den: u16, b_num: i16, b_den: u16) -> Result<(i16, u16), OverflowDetected> {
    if b_num == 0 { return Err(OverflowDetected::PrecisionLoss); }
    if a_num == 0 { return Ok((0, 1)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    let result_numerator = (a_num as i32).checked_mul(b_den as i32)
        .ok_or(OverflowDetected::TierOverflow)?;
    let result_denominator = (a_den as u32).checked_mul(b_num.abs() as u32)
        .ok_or(OverflowDetected::TierOverflow)?;

    let final_numerator = if b_num < 0 { -result_numerator } else { result_numerator };

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(final_numerator.unsigned_abs() as u128, result_denominator as u128);
    let reduced_num = (final_numerator as i128) / (g as i128);
    let reduced_den = (result_denominator as u128) / g;

    if reduced_num >= i16::MIN as i128 && reduced_num <= i16::MAX as i128 &&
       reduced_den <= u16::MAX as u128 {
        Ok((reduced_num as i16, reduced_den as u16))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

// ================================================================================================
// TIER 3: MEDIUM (i32/u32) - Common calculations
// ================================================================================================

/// Addition for Medium tier
pub fn add_i32_rational(a_num: i32, a_den: u32, b_num: i32, b_den: u32) -> Result<(i32, u32), OverflowDetected> {
    if a_num == 0 { return Ok((b_num, b_den)); }
    if b_num == 0 { return Ok((a_num, a_den)); }

    let a_expanded = (a_num as i64).checked_mul(b_den as i64)
        .ok_or(OverflowDetected::TierOverflow)?;
    let b_expanded = (b_num as i64).checked_mul(a_den as i64)
        .ok_or(OverflowDetected::TierOverflow)?;
    let common_denominator = (a_den as u64).checked_mul(b_den as u64)
        .ok_or(OverflowDetected::TierOverflow)?;

    let result_numerator = a_expanded.checked_add(b_expanded)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs() as u128, common_denominator as u128);
    let reduced_num = (result_numerator as i128) / (g as i128);
    let reduced_den = (common_denominator as u128) / g;

    if reduced_num >= i32::MIN as i128 && reduced_num <= i32::MAX as i128 &&
       reduced_den <= u32::MAX as u128 {
        Ok((reduced_num as i32, reduced_den as u32))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Subtraction for Medium tier
pub fn sub_i32_rational(a_num: i32, a_den: u32, b_num: i32, b_den: u32) -> Result<(i32, u32), OverflowDetected> {
    if b_num == 0 { return Ok((a_num, a_den)); }

    let a_expanded = (a_num as i64).checked_mul(b_den as i64)
        .ok_or(OverflowDetected::TierOverflow)?;
    let b_expanded = (b_num as i64).checked_mul(a_den as i64)
        .ok_or(OverflowDetected::TierOverflow)?;
    let common_denominator = (a_den as u64).checked_mul(b_den as u64)
        .ok_or(OverflowDetected::TierOverflow)?;

    let result_numerator = a_expanded.checked_sub(b_expanded)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs() as u128, common_denominator as u128);
    let reduced_num = (result_numerator as i128) / (g as i128);
    let reduced_den = (common_denominator as u128) / g;

    if reduced_num >= i32::MIN as i128 && reduced_num <= i32::MAX as i128 &&
       reduced_den <= u32::MAX as u128 {
        Ok((reduced_num as i32, reduced_den as u32))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Multiplication for Medium tier
pub fn mul_i32_rational(a_num: i32, a_den: u32, b_num: i32, b_den: u32) -> Result<(i32, u32), OverflowDetected> {
    if a_num == 0 || b_num == 0 { return Ok((0, 1)); }
    if a_num == 1 && a_den == 1 { return Ok((b_num, b_den)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    let result_numerator = (a_num as i64).checked_mul(b_num as i64)
        .ok_or(OverflowDetected::TierOverflow)?;
    let result_denominator = (a_den as u64).checked_mul(b_den as u64)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs() as u128, result_denominator as u128);
    let reduced_num = (result_numerator as i128) / (g as i128);
    let reduced_den = (result_denominator as u128) / g;

    if reduced_num >= i32::MIN as i128 && reduced_num <= i32::MAX as i128 &&
       reduced_den <= u32::MAX as u128 {
        Ok((reduced_num as i32, reduced_den as u32))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Division for Medium tier
pub fn div_i32_rational(a_num: i32, a_den: u32, b_num: i32, b_den: u32) -> Result<(i32, u32), OverflowDetected> {
    if b_num == 0 { return Err(OverflowDetected::PrecisionLoss); }
    if a_num == 0 { return Ok((0, 1)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    let result_numerator = (a_num as i64).checked_mul(b_den as i64)
        .ok_or(OverflowDetected::TierOverflow)?;
    let result_denominator = (a_den as u64).checked_mul(b_num.abs() as u64)
        .ok_or(OverflowDetected::TierOverflow)?;

    let final_numerator = if b_num < 0 { -result_numerator } else { result_numerator };

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(final_numerator.unsigned_abs() as u128, result_denominator as u128);
    let reduced_num = (final_numerator as i128) / (g as i128);
    let reduced_den = (result_denominator as u128) / g;

    if reduced_num >= i32::MIN as i128 && reduced_num <= i32::MAX as i128 &&
       reduced_den <= u32::MAX as u128 {
        Ok((reduced_num as i32, reduced_den as u32))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

// ================================================================================================
// TIER 4: LARGE (i64/u64) - Extended precision
// ================================================================================================

/// Addition for Large tier
pub fn add_i64_rational(a_num: i64, a_den: u64, b_num: i64, b_den: u64) -> Result<(i64, u64), OverflowDetected> {
    if a_num == 0 { return Ok((b_num, b_den)); }
    if b_num == 0 { return Ok((a_num, a_den)); }

    let a_expanded = (a_num as i128).checked_mul(b_den as i128)
        .ok_or(OverflowDetected::TierOverflow)?;
    let b_expanded = (b_num as i128).checked_mul(a_den as i128)
        .ok_or(OverflowDetected::TierOverflow)?;
    let common_denominator = (a_den as u128).checked_mul(b_den as u128)
        .ok_or(OverflowDetected::TierOverflow)?;

    let result_numerator = a_expanded.checked_add(b_expanded)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs(), common_denominator);
    let reduced_num = result_numerator / (g as i128);
    let reduced_den = common_denominator / g;

    if reduced_num >= i64::MIN as i128 && reduced_num <= i64::MAX as i128 &&
       reduced_den <= u64::MAX as u128 {
        Ok((reduced_num as i64, reduced_den as u64))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Subtraction for Large tier
pub fn sub_i64_rational(a_num: i64, a_den: u64, b_num: i64, b_den: u64) -> Result<(i64, u64), OverflowDetected> {
    if b_num == 0 { return Ok((a_num, a_den)); }

    let a_expanded = (a_num as i128).checked_mul(b_den as i128)
        .ok_or(OverflowDetected::TierOverflow)?;
    let b_expanded = (b_num as i128).checked_mul(a_den as i128)
        .ok_or(OverflowDetected::TierOverflow)?;
    let common_denominator = (a_den as u128).checked_mul(b_den as u128)
        .ok_or(OverflowDetected::TierOverflow)?;

    let result_numerator = a_expanded.checked_sub(b_expanded)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs(), common_denominator);
    let reduced_num = result_numerator / (g as i128);
    let reduced_den = common_denominator / g;

    if reduced_num >= i64::MIN as i128 && reduced_num <= i64::MAX as i128 &&
       reduced_den <= u64::MAX as u128 {
        Ok((reduced_num as i64, reduced_den as u64))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Multiplication for Large tier
pub fn mul_i64_rational(a_num: i64, a_den: u64, b_num: i64, b_den: u64) -> Result<(i64, u64), OverflowDetected> {
    if a_num == 0 || b_num == 0 { return Ok((0, 1)); }
    if a_num == 1 && a_den == 1 { return Ok((b_num, b_den)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    let result_numerator = (a_num as i128).checked_mul(b_num as i128)
        .ok_or(OverflowDetected::TierOverflow)?;
    let result_denominator = (a_den as u128).checked_mul(b_den as u128)
        .ok_or(OverflowDetected::TierOverflow)?;

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(result_numerator.unsigned_abs(), result_denominator);
    let reduced_num = result_numerator / (g as i128);
    let reduced_den = result_denominator / g;

    if reduced_num >= i64::MIN as i128 && reduced_num <= i64::MAX as i128 &&
       reduced_den <= u64::MAX as u128 {
        Ok((reduced_num as i64, reduced_den as u64))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

/// Division for Large tier
pub fn div_i64_rational(a_num: i64, a_den: u64, b_num: i64, b_den: u64) -> Result<(i64, u64), OverflowDetected> {
    if b_num == 0 { return Err(OverflowDetected::PrecisionLoss); }
    if a_num == 0 { return Ok((0, 1)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    let result_numerator = (a_num as i128).checked_mul(b_den as i128)
        .ok_or(OverflowDetected::TierOverflow)?;
    let result_denominator = (a_den as u128).checked_mul(b_num.abs() as u128)
        .ok_or(OverflowDetected::TierOverflow)?;

    let final_numerator = if b_num < 0 { -result_numerator } else { result_numerator };

    // GCD-reduce at wider type before fit check to avoid false TierOverflow
    let g = gcd_unsigned(final_numerator.unsigned_abs(), result_denominator);
    let reduced_num = final_numerator / (g as i128);
    let reduced_den = result_denominator / g;

    if reduced_num >= i64::MIN as i128 && reduced_num <= i64::MAX as i128 &&
       reduced_den <= u64::MAX as u128 {
        Ok((reduced_num as i64, reduced_den as u64))
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

// ================================================================================================
// TIER 5: HUGE (i128/u128) - Maximum standard precision
// ================================================================================================

/// Addition for Huge tier — uses I256 for intermediate calculations
pub fn add_i128_rational(a_num: i128, a_den: u128, b_num: i128, b_den: u128) -> Result<(i128, u128), OverflowDetected> {
    if a_num == 0 { return Ok((b_num, b_den)); }
    if b_num == 0 { return Ok((a_num, a_den)); }

    let a = I256::from_i128(a_num);
    let b = I256::from_i128(b_num);
    let ad = I256::from_u128(a_den);
    let bd = I256::from_u128(b_den);

    let result_num = a * bd + b * ad;
    let result_den = ad * bd;

    i256_reduce_to_i128(result_num, result_den)
}

/// Subtraction for Huge tier
pub fn sub_i128_rational(a_num: i128, a_den: u128, b_num: i128, b_den: u128) -> Result<(i128, u128), OverflowDetected> {
    if b_num == 0 { return Ok((a_num, a_den)); }

    let a = I256::from_i128(a_num);
    let b = I256::from_i128(b_num);
    let ad = I256::from_u128(a_den);
    let bd = I256::from_u128(b_den);

    let result_num = a * bd - b * ad;
    let result_den = ad * bd;

    i256_reduce_to_i128(result_num, result_den)
}

/// Multiplication for Huge tier
pub fn mul_i128_rational(a_num: i128, a_den: u128, b_num: i128, b_den: u128) -> Result<(i128, u128), OverflowDetected> {
    if a_num == 0 || b_num == 0 { return Ok((0, 1)); }
    if a_num == 1 && a_den == 1 { return Ok((b_num, b_den)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    let a = I256::from_i128(a_num);
    let b = I256::from_i128(b_num);
    let ad = I256::from_u128(a_den);
    let bd = I256::from_u128(b_den);

    let result_num = a * b;
    let result_den = ad * bd;

    i256_reduce_to_i128(result_num, result_den)
}

/// Division for Huge tier
pub fn div_i128_rational(a_num: i128, a_den: u128, b_num: i128, b_den: u128) -> Result<(i128, u128), OverflowDetected> {
    if b_num == 0 { return Err(OverflowDetected::PrecisionLoss); }
    if a_num == 0 { return Ok((0, 1)); }
    if b_num == 1 && b_den == 1 { return Ok((a_num, a_den)); }

    let a = I256::from_i128(a_num);
    let bd = I256::from_u128(b_den);
    let ad = I256::from_u128(a_den);
    let b_abs = I256::from_u128(b_num.unsigned_abs());

    // (a/b) / (c/d) = (a*d) / (b*|c|), with sign adjustment
    let result_num = a * bd;
    let result_den = ad * b_abs;
    let final_num = if b_num < 0 { -result_num } else { result_num };

    i256_reduce_to_i128(final_num, result_den)
}

// ================================================================================================
// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i8_rational_addition() {
        // Test 1/3 + 1/6 = 1/2
        let result = add_i8_rational(1, 3, 1, 6).unwrap();
        assert_eq!(result, (1, 2));

        // Test overflow detection
        let overflow = add_i8_rational(127, 1, 1, 1);
        assert!(overflow.is_err());
    }

    #[test]
    fn test_i8_rational_multiplication() {
        // Test 1/2 × 2/3 = 1/3
        let result = mul_i8_rational(1, 2, 2, 3).unwrap();
        assert_eq!(result, (1, 3));

        // Test zero multiplication
        let zero_result = mul_i8_rational(0, 1, 5, 7).unwrap();
        assert_eq!(zero_result, (0, 1));
    }

    #[test]
    fn test_i8_rational_division() {
        // Test 1/2 ÷ 1/4 = 2
        let result = div_i8_rational(1, 2, 1, 4).unwrap();
        assert_eq!(result, (2, 1));

        // Test division by zero
        let div_zero = div_i8_rational(1, 2, 0, 1);
        assert!(div_zero.is_err());
    }

    #[test]
    fn test_mathematical_identities() {
        // Test additive identity: a + 0 = a
        let result = add_i8_rational(3, 4, 0, 1).unwrap();
        assert_eq!(result, (3, 4));

        // Test multiplicative identity: a × 1 = a
        let result = mul_i8_rational(3, 4, 1, 1).unwrap();
        assert_eq!(result, (3, 4));

        // Test multiplicative zero: a × 0 = 0
        let result = mul_i8_rational(3, 4, 0, 1).unwrap();
        assert_eq!(result, (0, 1));
    }

    #[test]
    fn test_cross_tier_consistency_addition() {
        // Same rational operation across different tiers should give same reduced result

        // 1/3 + 1/6 = 1/2 across all tiers
        let i8_result = add_i8_rational(1, 3, 1, 6).unwrap();
        let i16_result = add_i16_rational(1, 3, 1, 6).unwrap();
        let i32_result = add_i32_rational(1, 3, 1, 6).unwrap();
        let i64_result = add_i64_rational(1, 3, 1, 6).unwrap();

        assert_eq!(i8_result, (1, 2));
        assert_eq!(i16_result, (1, 2));
        assert_eq!(i32_result, (1, 2));
        assert_eq!(i64_result, (1, 2));

        // 2/3 × 3/4 = 1/2 across all tiers
        let i8_mul = mul_i8_rational(2, 3, 3, 4).unwrap();
        let i16_mul = mul_i16_rational(2, 3, 3, 4).unwrap();
        let i32_mul = mul_i32_rational(2, 3, 3, 4).unwrap();
        let i64_mul = mul_i64_rational(2, 3, 3, 4).unwrap();

        assert_eq!(i8_mul, (1, 2));
        assert_eq!(i16_mul, (1, 2));
        assert_eq!(i32_mul, (1, 2));
        assert_eq!(i64_mul, (1, 2));
    }

    #[test]
    fn test_i128_rational_operations() {
        // Test i128 addition via I256 intermediate path
        let result = add_i128_rational(1, 3, 1, 6).unwrap();
        assert_eq!(result, (1, 2));

        // Test i128 multiplication
        let result = mul_i128_rational(2, 3, 3, 4).unwrap();
        assert_eq!(result, (1, 2));

        // Test i128 division
        let result = div_i128_rational(1, 2, 1, 4).unwrap();
        assert_eq!(result, (2, 1));

        // Test i128 subtraction
        let result = sub_i128_rational(1, 2, 1, 3).unwrap();
        assert_eq!(result, (1, 6));
    }

    #[test]
    fn test_gcd_unsigned() {
        assert_eq!(gcd_unsigned(12, 8), 4);
        assert_eq!(gcd_unsigned(17, 13), 1);
        assert_eq!(gcd_unsigned(100, 75), 25);
        assert_eq!(gcd_unsigned(0, 5), 5);
        assert_eq!(gcd_unsigned(0, 0), 1); // convention: gcd(0,0) = 1 to avoid div-by-zero
    }
}
