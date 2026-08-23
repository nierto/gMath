//! Binary Fixed-Point Division -- Tier Primitives + UGOD Wrapper
//!
//! **OPERATIONS**: checked_div (Tiers 1-3), div (Tiers 4-6)
//! **UGOD**: UniversalBinaryFixed::divide() with overflow promotion

use super::binary_types::*;
use crate::fixed_point::{I256, I512, I1024};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// TIER 1: Q16.16
// ============================================================================

impl BinaryTier1 {
    pub fn checked_div(&self, other: &Self) -> Option<Self> {
        let fb: u32 = {
            #[cfg(table_format = "q16_16")]
            { crate::fixed_point::frac_config::FRAC_BITS as u32 }
            #[cfg(not(table_format = "q16_16"))]
            { 16 }
        };
        if other.value == 0 { return None; }
        let a_wide = (self.value as i64) << fb;
        let b_wide = other.value as i64;
        let quotient = a_wide / b_wide;
        let remainder = a_wide % b_wide;
        let abs_2rem = (remainder.wrapping_abs() as u64) << 1;
        let abs_div = b_wide.unsigned_abs();
        // 0.5.0 rounding unification: nearest, ties toward +∞ (was
        // half-away, which also mis-signed the bump for exact quotients
        // in (-1, 0): it branched on quotient<0, but q==0 there).
        let positive = (a_wide < 0) == (b_wide < 0);
        let bump = if positive { abs_2rem >= abs_div } else { abs_2rem > abs_div };
        let result = if bump {
            if positive { quotient + 1 } else { quotient - 1 }
        } else {
            quotient
        };
        let result_i32 = result as i32;
        if result_i32 as i64 == result { Some(Self { value: result_i32 }) } else { None }
    }
}

// ============================================================================
// TIER 2: Q32.32
// ============================================================================

impl BinaryTier2 {
    pub fn checked_div(&self, other: &Self) -> Option<Self> {
        if other.value == 0 { return None; }
        let a_wide = (self.value as i128) << 32;
        let b_wide = other.value as i128;
        let quotient = a_wide / b_wide;
        let remainder = a_wide % b_wide;
        let abs_2rem = (remainder.unsigned_abs()) << 1;
        let abs_div = b_wide.unsigned_abs();
        // Nearest, ties toward +∞ (see Tier 1 note).
        let positive = (a_wide < 0) == (b_wide < 0);
        let bump = if positive { abs_2rem >= abs_div } else { abs_2rem > abs_div };
        let result = if bump {
            if positive { quotient + 1 } else { quotient - 1 }
        } else {
            quotient
        };
        let result_i64 = result as i64;
        if result_i64 as i128 == result { Some(Self { value: result_i64 }) } else { None }
    }
}

// ============================================================================
// TIER 3: Q64.64
// ============================================================================

impl BinaryTier3 {
    pub fn checked_div(&self, other: &Self) -> Option<Self> {
        if other.value == 0 { return None; }
        let a_wide = I256::from_i128(self.value) << 64;
        let b_wide = I256::from_i128(other.value);
        let (quotient, remainder) = crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(a_wide, b_wide);
        let abs_2rem = if remainder.is_negative() { (-remainder) << 1 } else { remainder << 1 };
        let abs_div = if b_wide.is_negative() { -b_wide } else { b_wide };
        // Nearest, ties toward +∞ (see Tier 1 note).
        let positive = a_wide.is_negative() == b_wide.is_negative();
        let bump = if positive { abs_2rem >= abs_div } else { abs_2rem > abs_div };
        let result = if bump {
            if positive { quotient + I256::from_i128(1) } else { quotient - I256::from_i128(1) }
        } else {
            quotient
        };
        if result.fits_in_i128() { Some(Self { value: result.as_i128() }) } else { None }
    }
}

// ============================================================================
// TIER 4: Q128.128
// ============================================================================

impl BinaryTier4 {
    pub fn div(&self, other: &Self) -> Option<Self> {
        if other.value.is_zero() { return None; }
        let a_wide = I512::from_i256(self.value) << 128;
        let b_wide = I512::from_i256(other.value);
        let (quotient, remainder) = crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(a_wide, b_wide);
        let abs_2rem = if remainder.is_negative() { (-remainder) << 1 } else { remainder << 1 };
        let abs_div = if b_wide.is_negative() { -b_wide } else { b_wide };
        // Nearest, ties toward +∞ (see Tier 1 note).
        let positive = a_wide.is_negative() == b_wide.is_negative();
        let bump = if positive { abs_2rem >= abs_div } else { abs_2rem > abs_div };
        let result = if bump {
            if positive { quotient + I512::from_i128(1) } else { quotient - I512::from_i128(1) }
        } else {
            quotient
        };
        if result.fits_in_i256() { Some(Self { value: result.as_i256() }) } else { None }
    }
}

// ============================================================================
// TIER 5: Q256.256
// ============================================================================

impl BinaryTier5 {
    pub fn div(&self, other: &Self) -> Option<Self> {
        if other.value.is_zero() { return None; }
        let a_wide = I1024::from_i512(self.value) << 256;
        let b_wide = I1024::from_i512(other.value);
        let quotient = a_wide / b_wide;
        let remainder = a_wide % b_wide;
        let rem_neg = (remainder.words[15] as i64) < 0;
        let div_neg = (b_wide.words[15] as i64) < 0;
        let quot_neg = (quotient.words[15] as i64) < 0;
        let abs_2rem = if rem_neg { (-remainder) << 1 } else { remainder << 1 };
        let abs_div = if div_neg { -b_wide } else { b_wide };
        // Nearest, ties toward +∞ (see Tier 1 note). Exact-result sign
        // from the operand signs, NOT from quotient (wrong at q == 0).
        let _ = quot_neg;
        let a_neg = (a_wide.words[15] as i64) < 0;
        let positive = a_neg == div_neg;
        let bump = if positive { abs_2rem >= abs_div } else { abs_2rem > abs_div };
        let result = if bump {
            if positive { quotient + I1024::from_i128(1) } else { quotient - I1024::from_i128(1) }
        } else {
            quotient
        };
        if result.fits_in_i512() { Some(Self { value: result.as_i512() }) } else { None }
    }
}

// ============================================================================
// TIER 6: Q512.512
// ============================================================================

impl BinaryTier6 {
    pub fn div(&self, other: &Self) -> Option<Self> {
        if other.value == I1024::zero() { return None; }
        use crate::fixed_point::domains::binary_fixed::i2048::{I2048, i2048_div};
        let a_wide = I2048::from_i1024(self.value) << 512;
        let b_wide = I2048::from_i1024(other.value);
        let quotient = i2048_div(a_wide, b_wide);
        // Nearest, ties toward +∞ (was bare truncation — the only tier
        // without an adjust; unified 0.5.0).
        let rem = a_wide - quotient * b_wide;
        let rem_neg = (rem.words[31] as i64) < 0;
        let a_neg = (a_wide.words[31] as i64) < 0;
        let b_neg = (b_wide.words[31] as i64) < 0;
        let abs_2rem = (if rem_neg { -rem } else { rem }) << 1;
        let abs_div = if b_neg { -b_wide } else { b_wide };
        let positive = a_neg == b_neg;
        let bump = if positive { abs_2rem >= abs_div } else { abs_2rem > abs_div };
        let one = I2048::from_i1024(I1024::from_i128(1));
        let quotient = if bump {
            if positive { quotient + one } else { quotient - one }
        } else {
            quotient
        };
        {
            // Checked narrowing (0.5.0 item 1): a quotient beyond I1024
            // (tiny divisor) must be None — ladder top, never a wrap.
            let sign_ext = if (quotient.words[15] as i64) < 0 { u64::MAX } else { 0 };
            if quotient.words[16..32].iter().any(|&w| w != sign_ext) {
                return None;
            }
            Some(Self { value: quotient.as_i1024() })
        }
    }
}

// ============================================================================
// UGOD: UniversalBinaryFixed::divide
// ============================================================================

impl UniversalBinaryFixed {
    pub fn divide(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let (a, b) = self.align_to_common_tier(other);
        // 0.5.0 item 1: decide DivisionByZero ONCE here, so every tier
        // function's None below unambiguously means quotient overflow —
        // previously an overflowing Tier-4 quotient was mislabeled
        // DivisionByZero and the ladder stopped early.
        let divisor_is_zero = match &b.value {
            BinaryValue::Tier1(v) => v.raw() == 0,
            BinaryValue::Tier2(v) => v.raw() == 0,
            BinaryValue::Tier3(v) => v.raw() == 0,
            BinaryValue::Tier4(v) => v.raw().is_zero(),
            BinaryValue::Tier5(v) => v.raw().is_zero(),
            BinaryValue::Tier6(v) => *v.raw() == crate::fixed_point::I1024::zero(),
        };
        if divisor_is_zero {
            return Err(OverflowDetected::DivisionByZero);
        }
        match (&a.value, &b.value) {
            (BinaryValue::Tier1(x), BinaryValue::Tier1(y)) => {
                match x.checked_div(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier1(r), current_tier: 1 }),
                    None => {
                        let x2 = x.to_tier2();
                        let y2 = y.to_tier2();
                        match x2.checked_div(&y2) {
                            Some(r) => Ok(Self { value: BinaryValue::Tier2(r), current_tier: 2 }),
                            None => {
                                let x3 = x.to_tier2().to_tier3();
                                let y3 = y.to_tier2().to_tier3();
                                match x3.checked_div(&y3) {
                                    Some(r) => Ok(Self { value: BinaryValue::Tier3(r), current_tier: 3 }),
                                    None => {
                                        match Self::div_tier4_and_up(&x3.to_tier4(), &y3.to_tier4()) {
                                            Some(r) => Ok(r),
                                            None => Err(OverflowDetected::TierOverflow),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (BinaryValue::Tier2(x), BinaryValue::Tier2(y)) => {
                match x.checked_div(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier2(r), current_tier: 2 }),
                    None => {
                        let x3 = x.to_tier3();
                        let y3 = y.to_tier3();
                        match x3.checked_div(&y3) {
                            Some(r) => Ok(Self { value: BinaryValue::Tier3(r), current_tier: 3 }),
                            None => {
                                match Self::div_tier4_and_up(&x3.to_tier4(), &y3.to_tier4()) {
                                    Some(r) => Ok(r),
                                    None => Err(OverflowDetected::TierOverflow),
                                }
                            }
                        }
                    }
                }
            }
            (BinaryValue::Tier3(x), BinaryValue::Tier3(y)) => {
                match x.checked_div(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier3(r), current_tier: 3 }),
                    None => {
                        match Self::div_tier4_and_up(&x.to_tier4(), &y.to_tier4()) {
                            Some(r) => Ok(r),
                            None => Err(OverflowDetected::TierOverflow),
                        }
                    }
                }
            }
            (BinaryValue::Tier4(x), BinaryValue::Tier4(y)) => {
                match Self::div_tier4_and_up(x, y) {
                    Some(r) => Ok(r),
                    None => Err(OverflowDetected::TierOverflow),
                }
            }
            (BinaryValue::Tier5(x), BinaryValue::Tier5(y)) => {
                match x.div(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier5(r), current_tier: 5 }),
                    None => match x.to_tier6().div(&y.to_tier6()) {
                        Some(r) => Ok(Self { value: BinaryValue::Tier6(r), current_tier: 6 }),
                        None => Err(OverflowDetected::TierOverflow),
                    },
                }
            }
            (BinaryValue::Tier6(x), BinaryValue::Tier6(y)) => {
                match x.div(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier6(r), current_tier: 6 }),
                    None => Err(OverflowDetected::TierOverflow),
                }
            }
            _ => Err(OverflowDetected::InvalidInput),
        }
    }

    /// Divide starting at Tier 4, promoting 5 → 6; None = quotient does
    /// not fit any tier (caller reports TierOverflow; FASC falls back to
    /// the exact rational path). Divisor-zero is decided by the caller.
    fn div_tier4_and_up(x: &BinaryTier4, y: &BinaryTier4) -> Option<Self> {
        if let Some(r) = x.div(y) {
            return Some(Self { value: BinaryValue::Tier4(r), current_tier: 4 });
        }
        if let Some(r) = x.to_tier5().div(&y.to_tier5()) {
            return Some(Self { value: BinaryValue::Tier5(r), current_tier: 5 });
        }
        x.to_tier5().to_tier6().div(&y.to_tier5().to_tier6())
            .map(|r| Self { value: BinaryValue::Tier6(r), current_tier: 6 })
    }
}
