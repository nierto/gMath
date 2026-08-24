//! Balanced Ternary Division Operations
//!
//! **MISSION**: Tier-specific division with overflow detection and UGOD promotion
//! **ARCHITECTURE**: Extracted from ternary_operations.rs for modular decomposition
//! **PRECISION**: Pure ternary arithmetic preserving exact division by 3
//! **INTEGRATION**: Used by UniversalTernaryFixed::divide() for UGOD dispatch

use super::ternary_types::{
    TernaryTier, TernaryTier1, TernaryTier2, TernaryTier3, TernaryTier4, TernaryTier5, TernaryTier6,
    TernaryValue, UniversalTernaryFixed,
    SCALE_TQ10_10, SCALE_TQ20_20, SCALE_TQ40_40, scale_tq160_160_i1024, scale_tq320_320,
};
use crate::fixed_point::{I256, I512, I1024, I2048};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// TIER 1: TQ10.10 DIVISION (i32 storage)
// ============================================================================

/// TQ10.10 division with overflow detection
#[inline]
pub fn divide_ternary_tq10_10(a: i32, b: i32) -> Result<i32, OverflowDetected> {
    if b == 0 {
        return Err(OverflowDetected::DivisionByZero);
    }

    // Scale numerator to preserve precision
    match (a as i64).checked_mul(SCALE_TQ10_10 as i64) {
        Some(scaled_a) => {
            // 0.5.0: nearest, ties toward +∞ (was truncation). Divisor is
            // arbitrary, so unlike multiply (odd scale) ties CAN occur here.
            let den = b as i64;
            let mut result = scaled_a / den;
            let rem2 = (scaled_a - result * den).unsigned_abs() << 1;
            let dabs = den.unsigned_abs();
            let positive = (scaled_a < 0) == (den < 0);
            if if positive { rem2 >= dabs } else { rem2 > dabs } {
                result += if positive { 1 } else { -1 };
            }
            if result >= i32::MIN as i64 && result <= i32::MAX as i64 {
                Ok(result as i32)
            } else {
                Err(OverflowDetected::TierOverflow)
            }
        }
        None => Err(OverflowDetected::TierOverflow),
    }
}

// ============================================================================
// TIER 2: TQ20.20 DIVISION (i64 storage)
// ============================================================================

/// TQ20.20 division with overflow detection
#[inline]
pub fn divide_ternary_tq20_20(a: i64, b: i64) -> Result<i64, OverflowDetected> {
    if b == 0 {
        return Err(OverflowDetected::DivisionByZero);
    }

    // Scale numerator to preserve precision
    match (a as i128).checked_mul(SCALE_TQ20_20 as i128) {
        Some(scaled_a) => {
            // Nearest, ties toward +∞ (see Tier 1 note).
            let den = b as i128;
            let mut result = scaled_a / den;
            let rem2 = (scaled_a - result * den).unsigned_abs() << 1;
            let dabs = den.unsigned_abs();
            let positive = (scaled_a < 0) == (den < 0);
            if if positive { rem2 >= dabs } else { rem2 > dabs } {
                result += if positive { 1 } else { -1 };
            }
            if result >= i64::MIN as i128 && result <= i64::MAX as i128 {
                Ok(result as i64)
            } else {
                Err(OverflowDetected::TierOverflow)
            }
        }
        None => Err(OverflowDetected::TierOverflow),
    }
}

// ============================================================================
// TIER 3: TQ40.40 DIVISION (i128 storage)
// ============================================================================

/// TQ40.40 division with overflow detection
#[inline]
pub fn divide_ternary_tq40_40(a: i128, b: i128) -> Result<i128, OverflowDetected> {
    if b == 0 {
        return Err(OverflowDetected::DivisionByZero);
    }

    // Use I256 for precision preservation
    let a_extended = I256::from_i128(a);
    let b_extended = I256::from_i128(b);
    let scale = I256::from_i128(SCALE_TQ40_40);

    // Scale numerator and divide — nearest, ties toward +∞ (0.5.0)
    let scaled_a = a_extended * scale;
    let (mut result, rem) =
        crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(scaled_a, b_extended);
    let rem_abs = if rem.is_negative() { -rem } else { rem };
    let den_abs = if b_extended.is_negative() { -b_extended } else { b_extended };
    let positive = scaled_a.is_negative() == b_extended.is_negative();
    let rem2 = rem_abs + rem_abs;
    if if positive { rem2 >= den_abs } else { rem2 > den_abs } {
        result = if positive { result + I256::from_i128(1) } else { result - I256::from_i128(1) };
    }

    if result.fits_in_i128() {
        Ok(result.as_i128())
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

// ============================================================================
// TIER 4: TQ80.80 DIVISION (I256 storage, NEVER FAILS)
// ============================================================================

/// TQ80.80 division (maximum precision, never fails)
#[inline]
pub fn divide_ternary_tq80_80(a: I256, b: I256) -> I256 {
    if b.is_zero() {
        // Return signed infinity representation for division by zero
        return I256::signed_infinity(a.is_negative());
    }

    // Use I512 for precision preservation
    let a_extended = I512::from_i256(a);
    let b_extended = I512::from_i256(b);
    let scale = compute_3_pow_80_i512();

    // Scale numerator and divide — nearest, ties toward +∞ (0.5.0)
    let scaled_a = a_extended * scale;
    let (mut result, rem) =
        crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(scaled_a, b_extended);
    let rem_abs = if rem.is_negative() { -rem } else { rem };
    let den_abs = if b_extended.is_negative() { -b_extended } else { b_extended };
    let positive = scaled_a.is_negative() == b_extended.is_negative();
    let rem2 = rem_abs + rem_abs;
    if if positive { rem2 >= den_abs } else { rem2 > den_abs } {
        result = if positive { result + I512::from_i128(1) } else { result - I512::from_i128(1) };
    }

    // Saturate back to I256
    result.as_i256_saturating()
}

// ============================================================================
// TIER 4 CHECKED VARIANT (for promotion to Tier 5)
// ============================================================================

/// TQ80.80 division with overflow detection (checked variant)
#[inline]
pub fn divide_ternary_tq80_80_checked(a: I256, b: I256) -> Result<I256, OverflowDetected> {
    if b.is_zero() {
        return Err(OverflowDetected::DivisionByZero);
    }

    let a_extended = I512::from_i256(a);
    let b_extended = I512::from_i256(b);
    let scale = compute_3_pow_80_i512();
    let scaled_a = a_extended * scale;
    // Nearest, ties toward +∞ — matches the unchecked variant.
    let (mut result, rem) =
        crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(scaled_a, b_extended);
    let rem_abs = if rem.is_negative() { -rem } else { rem };
    let den_abs = if b_extended.is_negative() { -b_extended } else { b_extended };
    let positive = scaled_a.is_negative() == b_extended.is_negative();
    let rem2 = rem_abs + rem_abs;
    if if positive { rem2 >= den_abs } else { rem2 > den_abs } {
        result = if positive { result + I512::from_i128(1) } else { result - I512::from_i128(1) };
    }

    if result.fits_in_i256() {
        Ok(result.as_i256())
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

// ============================================================================
// TIER 5: TQ160.160 DIVISION (I512 storage, checked)
// ============================================================================

/// TQ160.160 division with overflow detection
#[inline]
pub fn divide_ternary_tq160_160(a: I512, b: I512) -> Result<I512, OverflowDetected> {
    if b.is_zero() {
        return Err(OverflowDetected::DivisionByZero);
    }

    let a_extended = I1024::from_i512(a);
    let b_extended = I1024::from_i512(b);
    let scale = scale_tq160_160_i1024();
    let scaled_a = a_extended * scale;
    // Nearest, ties toward +∞ (0.5.0).
    let mut result = scaled_a / b_extended;
    let rem = scaled_a % b_extended;
    let rem_neg = (rem.words[15] as i64) < 0;
    let den_neg = (b_extended.words[15] as i64) < 0;
    let sa_neg = (scaled_a.words[15] as i64) < 0;
    let rem_abs = if rem_neg { -rem } else { rem };
    let den_abs = if den_neg { -b_extended } else { b_extended };
    let positive = sa_neg == den_neg;
    let rem2 = rem_abs + rem_abs;
    if if positive { rem2 >= den_abs } else { rem2 > den_abs } {
        result = if positive { result + I1024::from_i128(1) } else { result - I1024::from_i128(1) };
    }

    if result.fits_in_i512() {
        Ok(result.as_i512())
    } else {
        Err(OverflowDetected::TierOverflow)
    }
}

// ============================================================================
// TIER 6: TQ320.320 DIVISION (I1024 storage, saturating — NEVER FAILS)
// ============================================================================

/// TQ320.320 division (maximum precision, never fails)
///
/// a / b in TQ320.320: (a * 3^256) / b
/// Since both a and 3^256 are I1024, a * 3^256 needs I2048.
/// Then I2048 / I1024 -> I1024.
#[inline]
pub fn divide_ternary_tq320_320(a: I1024, b: I1024) -> I1024 {
    // Check for division by zero -- return max/min as infinity representation
    let b_i512 = b.as_i512();
    if b_i512.is_zero() && b.fits_in_i512() {
        return I1024::max_value();
    }

    // a * 3^256 using I2048
    let a_ext = I2048::from_i1024(a);
    let scale = I2048::from_i1024(scale_tq320_320());
    let scaled_a = a_ext * scale;

    // Divide I2048 by I1024 — nearest, ties toward +∞ (0.5.0; was
    // truncation). Work on magnitudes: i2048_div_by_i1024 already
    // sign-handles the quotient, so recover the remainder magnitude via
    // |scaled_a| − |q|·|b| and bump per the exact result's sign.
    let q = i2048_div_by_i1024(scaled_a, b);
    let sa_neg = (scaled_a.words[31] as i64) < 0;
    let b_neg = (b.words[15] as i64) < 0;
    let positive = sa_neg == b_neg;
    let abs_sa = if sa_neg { -scaled_a } else { scaled_a };
    let q_neg = (q.words[31] as i64) < 0;
    let abs_q = if q_neg { -q } else { q };
    let abs_b = if b_neg { -b } else { b };
    let qb = abs_q.as_i1024().mul_to_i2048(abs_b); // non-negative magnitudes — safe
    let rem = abs_sa - qb;
    let rem2 = rem + rem;
    let abs_b_wide = I2048::from_i1024(abs_b);
    let one = I2048::from_i1024(I1024::from_i128(1));
    let bumped = if if positive { rem2 >= abs_b_wide } else { rem2 > abs_b_wide } {
        if positive { q + one } else { q - one }
    } else {
        q
    };
    bumped.as_i1024()
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Compute 3^80 as I512 for scaling (TQ80.80 tier-4 scale)
///
/// 3^80 = 147_808_829_414_345_923_316_083_210_206_383_297_601
/// Fits in i128 (max ~1.7x10^38), so construct via from_i128.
pub(super) fn compute_3_pow_80_i512() -> I512 {
    // Precomputed: 3^80 = (3^40)^2 = 12_157_665_459_056_928_801^2
    I512::from_i128(147_808_829_414_345_923_316_083_210_206_383_297_601_i128)
}

/// Divide an I2048 value by an I1024 divisor.
///
/// **ALGORITHM**: Schoolbook long division. The dividend is 2048 bits and the
/// divisor is 1024 bits, so the quotient is at most 1024 bits (fits in I2048
/// for the intermediate result).
///
/// Since I2048 lacks a Div trait, we implement this using shift-and-subtract.
fn i2048_div_by_i1024(dividend: I2048, divisor: I1024) -> I2048 {
    // Handle signs manually
    let dividend_neg = {
        // Check sign: MSB of highest word
        let words = dividend_words(&dividend);
        (words[31] & 0x8000_0000_0000_0000) != 0
    };
    let divisor_neg = {
        let ds = divisor.as_i512();
        // If it fits in I512, check I512 sign. Otherwise check I1024 MSB.
        if divisor.fits_in_i512() {
            ds.is_negative()
        } else {
            // Check word[15] MSB for I1024
            let dw = i1024_words(&divisor);
            (dw[15] & 0x8000_0000_0000_0000) != 0
        }
    };

    let abs_dividend = if dividend_neg { -dividend } else { dividend };
    let abs_divisor = if divisor_neg { -divisor } else { divisor };

    // Convert divisor to I2048 for comparison
    let div_ext = I2048::from_i1024(abs_divisor);

    // Binary long division
    let mut quotient = I2048::zero();
    let mut remainder = I2048::zero();

    // Process 2048 bits from MSB to LSB
    for i in (0..2048).rev() {
        remainder = remainder << 1;
        // Set bit 0 of remainder from dividend bit i
        let word_idx = i / 64;
        let bit_idx = i % 64;
        let dw = dividend_words(&abs_dividend);
        if (dw[word_idx] >> bit_idx) & 1 == 1 {
            remainder = remainder + I2048::one();
        }

        // If remainder >= divisor, subtract and set quotient bit
        if remainder >= div_ext {
            remainder = remainder - div_ext;
            // Set bit i of quotient
            let qw = dividend_words_mut_via_rebuild(&quotient, word_idx, bit_idx);
            quotient = qw;
        }
    }

    // Apply sign
    let result_neg = dividend_neg != divisor_neg;
    if result_neg { -quotient } else { quotient }
}

/// Extract the raw words from an I2048 for bit-level access
fn dividend_words(val: &I2048) -> [u64; 32] {
    val.words
}

/// Extract the raw words from an I1024
fn i1024_words(val: &I1024) -> [u64; 16] {
    val.words
}

/// Set a single bit in an I2048 quotient (rebuild from words)
fn dividend_words_mut_via_rebuild(val: &I2048, word_idx: usize, bit_idx: usize) -> I2048 {
    let mut words = val.words;
    words[word_idx] |= 1u64 << bit_idx;
    I2048::from_words(words)
}

// ============================================================================
// UGOD DIVISION METHOD
// ============================================================================

impl UniversalTernaryFixed {
    /// Division with automatic tier alignment and UGOD overflow promotion
    pub fn divide(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let (aligned_self, aligned_other) = self.align_to_common_tier(other);

        match (&aligned_self.value, &aligned_other.value) {
            (TernaryValue::Tier1(a), TernaryValue::Tier1(b)) => {
                match divide_ternary_tq10_10(a.raw(), b.raw()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier1(TernaryTier1::from_raw(result)), current_tier: TernaryTier::Tier1 }),
                    Err(OverflowDetected::DivisionByZero) => Err(OverflowDetected::DivisionByZero),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier2()?;
                        let p_other = aligned_other.promote_to_tier2()?;
                        p_self.divide(&p_other)
                    }
                }
            }
            (TernaryValue::Tier2(a), TernaryValue::Tier2(b)) => {
                match divide_ternary_tq20_20(a.raw(), b.raw()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier2(TernaryTier2::from_raw(result)), current_tier: TernaryTier::Tier2 }),
                    Err(OverflowDetected::DivisionByZero) => Err(OverflowDetected::DivisionByZero),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier3()?;
                        let p_other = aligned_other.promote_to_tier3()?;
                        p_self.divide(&p_other)
                    }
                }
            }
            (TernaryValue::Tier3(a), TernaryValue::Tier3(b)) => {
                match divide_ternary_tq40_40(a.raw(), b.raw()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier3(TernaryTier3::from_raw(result)), current_tier: TernaryTier::Tier3 }),
                    Err(OverflowDetected::DivisionByZero) => Err(OverflowDetected::DivisionByZero),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier4();
                        let p_other = aligned_other.promote_to_tier4();
                        p_self.divide(&p_other)
                    }
                }
            }
            (TernaryValue::Tier4(a), TernaryValue::Tier4(b)) => {
                match divide_ternary_tq80_80_checked(a.raw().clone(), b.raw().clone()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier4(TernaryTier4::from_raw(result)), current_tier: TernaryTier::Tier4 }),
                    Err(OverflowDetected::DivisionByZero) => Err(OverflowDetected::DivisionByZero),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier5();
                        let p_other = aligned_other.promote_to_tier5();
                        p_self.divide(&p_other)
                    }
                }
            }
            (TernaryValue::Tier5(a), TernaryValue::Tier5(b)) => {
                match divide_ternary_tq160_160(a.raw().clone(), b.raw().clone()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier5(TernaryTier5::from_raw(result)), current_tier: TernaryTier::Tier5 }),
                    Err(OverflowDetected::DivisionByZero) => Err(OverflowDetected::DivisionByZero),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier6();
                        let p_other = aligned_other.promote_to_tier6();
                        p_self.divide(&p_other)
                    }
                }
            }
            (TernaryValue::Tier6(a), TernaryValue::Tier6(b)) => {
                // Check for zero before calling (tq256_256 doesn't return Result)
                if b.raw().clone() == I1024::zero() {
                    return Err(OverflowDetected::DivisionByZero);
                }
                let result = divide_ternary_tq320_320(a.raw().clone(), b.raw().clone());
                Ok(Self { value: TernaryValue::Tier6(TernaryTier6::from_raw(result)), current_tier: TernaryTier::Tier6 })
            }
            _ => unreachable!("align_to_common_tier should ensure matching tiers")
        }
    }
}
