//! Decimal Fixed-Point Tier Types, Constructors, and Promotion
//!
//! **ARCHITECTURE**: Universal 6-tier decimal fixed-point system
//! **SCALING**: Powers-of-10 (value * 10^decimal_places), NOT binary Q-format
//! **TIERS**: D8.8(i16) -> D16.16(i32) -> D32.32(i64) -> D64.64(i128) -> D128.128(D256) -> D256.256(D512)

use crate::fixed_point::domains::binary_fixed::{I256, I512, I1024};
use crate::fixed_point::domains::decimal_fixed::{D256, D512};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// DECIMAL RAW ENUM -- ZASC BOUNDARY TYPE
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub enum DecimalRaw {
    Small(i128),
    Medium(I256),
    Large(I512),
    XLarge(I1024),
}

impl DecimalRaw {
    pub fn natural_tier(&self) -> u8 {
        match self {
            DecimalRaw::Small(_) => 4,
            DecimalRaw::Medium(_) => 5,
            DecimalRaw::Large(_) => 6,
            DecimalRaw::XLarge(_) => 6,
        }
    }
}

// ============================================================================
// DECIMAL TIER TYPES
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalTier1 {
    pub value: i16,
    pub decimal_places: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalTier2 {
    pub value: i32,
    pub decimal_places: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalTier3 {
    pub value: i64,
    pub decimal_places: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalTier4 {
    pub value: i128,
    pub decimal_places: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalTier5 {
    pub value: D256,
    pub decimal_places: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimalTier6 {
    pub value: D512,
    pub decimal_places: u8,
}

// ============================================================================
// TIER CONSTRUCTORS AND PROMOTION
// ============================================================================

impl DecimalTier1 {
    pub fn zero(decimal_places: u8) -> Self {
        Self { value: 0, decimal_places }
    }

    pub fn to_tier2(self) -> DecimalTier2 {
        DecimalTier2 { value: self.value as i32, decimal_places: self.decimal_places }
    }
}

impl DecimalTier2 {
    pub fn zero(decimal_places: u8) -> Self {
        Self { value: 0, decimal_places }
    }

    pub fn to_tier3(self) -> DecimalTier3 {
        DecimalTier3 { value: self.value as i64, decimal_places: self.decimal_places }
    }
}

impl DecimalTier3 {
    pub fn zero(decimal_places: u8) -> Self {
        Self { value: 0, decimal_places }
    }

    pub fn to_tier4(self) -> DecimalTier4 {
        DecimalTier4 { value: self.value as i128, decimal_places: self.decimal_places }
    }
}

impl DecimalTier4 {
    pub fn zero(decimal_places: u8) -> Self {
        Self { value: 0, decimal_places }
    }

    pub fn to_tier5(self) -> DecimalTier5 {
        DecimalTier5 { value: D256::from_i128(self.value), decimal_places: self.decimal_places }
    }
}

impl DecimalTier5 {
    pub fn zero(decimal_places: u8) -> Self {
        Self { value: D256::zero(), decimal_places }
    }

    pub fn to_tier6(self) -> DecimalTier6 {
        DecimalTier6 { value: D512::from_d256(self.value), decimal_places: self.decimal_places }
    }
}

impl DecimalTier6 {
    pub fn zero(decimal_places: u8) -> Self {
        Self { value: D512::zero(), decimal_places }
    }
}

// ============================================================================
// DECIMAL VALUE ENUM
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub enum DecimalValueTiered {
    Tier1(DecimalTier1),
    Tier2(DecimalTier2),
    Tier3(DecimalTier3),
    Tier4(DecimalTier4),
    Tier5(DecimalTier5),
    Tier6(DecimalTier6),
}

// ============================================================================
// UNIVERSAL DECIMAL TIERED -- ZASC + UGOD COMPATIBLE TYPE
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct UniversalDecimalTiered {
    pub(crate) value: DecimalValueTiered,
}

impl UniversalDecimalTiered {
    pub fn from_tier_value(tier: u8, decimal_places: u8, val: i128) -> Result<Self, OverflowDetected> {
        let value = match tier {
            1 => {
                if val < i16::MIN as i128 || val > i16::MAX as i128 {
                    return Err(OverflowDetected::TierOverflow);
                }
                DecimalValueTiered::Tier1(DecimalTier1 { value: val as i16, decimal_places })
            }
            2 => {
                if val < i32::MIN as i128 || val > i32::MAX as i128 {
                    return Err(OverflowDetected::TierOverflow);
                }
                DecimalValueTiered::Tier2(DecimalTier2 { value: val as i32, decimal_places })
            }
            3 => {
                if val < i64::MIN as i128 || val > i64::MAX as i128 {
                    return Err(OverflowDetected::TierOverflow);
                }
                DecimalValueTiered::Tier3(DecimalTier3 { value: val as i64, decimal_places })
            }
            4 => {
                DecimalValueTiered::Tier4(DecimalTier4 { value: val, decimal_places })
            }
            _ => return Err(OverflowDetected::InvalidInput),
        };
        Ok(Self { value })
    }

    pub fn from_tier_raw(tier: u8, decimal_places: u8, raw: DecimalRaw) -> Result<Self, OverflowDetected> {
        let value = match (tier, raw) {
            (1, DecimalRaw::Small(v)) => {
                if v < i16::MIN as i128 || v > i16::MAX as i128 {
                    return Err(OverflowDetected::TierOverflow);
                }
                DecimalValueTiered::Tier1(DecimalTier1 { value: v as i16, decimal_places })
            }
            (2, DecimalRaw::Small(v)) => {
                if v < i32::MIN as i128 || v > i32::MAX as i128 {
                    return Err(OverflowDetected::TierOverflow);
                }
                DecimalValueTiered::Tier2(DecimalTier2 { value: v as i32, decimal_places })
            }
            (3, DecimalRaw::Small(v)) => {
                if v < i64::MIN as i128 || v > i64::MAX as i128 {
                    return Err(OverflowDetected::TierOverflow);
                }
                DecimalValueTiered::Tier3(DecimalTier3 { value: v as i64, decimal_places })
            }
            (4, DecimalRaw::Small(v)) => {
                DecimalValueTiered::Tier4(DecimalTier4 { value: v, decimal_places })
            }
            (5, DecimalRaw::Medium(v)) => {
                DecimalValueTiered::Tier5(DecimalTier5 { value: i256_to_d256(v), decimal_places })
            }
            (6, DecimalRaw::Large(v)) => {
                DecimalValueTiered::Tier6(DecimalTier6 { value: i512_to_d512(v), decimal_places })
            }
            (5, DecimalRaw::Small(v)) => {
                DecimalValueTiered::Tier5(DecimalTier5 { value: D256::from_i128(v), decimal_places })
            }
            (6, DecimalRaw::Small(v)) => {
                DecimalValueTiered::Tier6(DecimalTier6 { value: D512::from_i128(v), decimal_places })
            }
            (6, DecimalRaw::Medium(v)) => {
                DecimalValueTiered::Tier6(DecimalTier6 { value: D512::from_d256(i256_to_d256(v)), decimal_places })
            }
            _ => return Err(OverflowDetected::InvalidInput),
        };
        Ok(Self { value })
    }

    pub fn to_tier_raw(&self) -> (u8, DecimalRaw) {
        match &self.value {
            DecimalValueTiered::Tier1(t) => (1, DecimalRaw::Small(t.value as i128)),
            DecimalValueTiered::Tier2(t) => (2, DecimalRaw::Small(t.value as i128)),
            DecimalValueTiered::Tier3(t) => (3, DecimalRaw::Small(t.value as i128)),
            DecimalValueTiered::Tier4(t) => (4, DecimalRaw::Small(t.value)),
            DecimalValueTiered::Tier5(t) => (5, DecimalRaw::Medium(d256_to_i256(t.value))),
            DecimalValueTiered::Tier6(t) => (6, DecimalRaw::Large(d512_to_i512(t.value))),
        }
    }

    pub fn current_tier(&self) -> u8 {
        match &self.value {
            DecimalValueTiered::Tier1(_) => 1,
            DecimalValueTiered::Tier2(_) => 2,
            DecimalValueTiered::Tier3(_) => 3,
            DecimalValueTiered::Tier4(_) => 4,
            DecimalValueTiered::Tier5(_) => 5,
            DecimalValueTiered::Tier6(_) => 6,
        }
    }

    pub fn decimal_places(&self) -> u8 {
        match &self.value {
            DecimalValueTiered::Tier1(t) => t.decimal_places,
            DecimalValueTiered::Tier2(t) => t.decimal_places,
            DecimalValueTiered::Tier3(t) => t.decimal_places,
            DecimalValueTiered::Tier4(t) => t.decimal_places,
            DecimalValueTiered::Tier5(t) => t.decimal_places,
            DecimalValueTiered::Tier6(t) => t.decimal_places,
        }
    }

    pub fn promote_to_tier(&self, target_tier: u8) -> Option<Self> {
        if target_tier <= self.current_tier() {
            return Some(*self);
        }
        let dp = self.decimal_places();
        match (&self.value, target_tier) {
            (DecimalValueTiered::Tier1(t), 2) => Some(Self { value: DecimalValueTiered::Tier2(t.to_tier2()) }),
            (DecimalValueTiered::Tier1(t), 3) => Some(Self { value: DecimalValueTiered::Tier3(t.to_tier2().to_tier3()) }),
            (DecimalValueTiered::Tier1(t), 4) => Some(Self { value: DecimalValueTiered::Tier4(t.to_tier2().to_tier3().to_tier4()) }),
            (DecimalValueTiered::Tier1(t), 5) => Some(Self { value: DecimalValueTiered::Tier5(t.to_tier2().to_tier3().to_tier4().to_tier5()) }),
            (DecimalValueTiered::Tier1(t), 6) => Some(Self { value: DecimalValueTiered::Tier6(DecimalTier6 { value: D512::from_i128(t.value as i128), decimal_places: dp }) }),
            (DecimalValueTiered::Tier2(t), 3) => Some(Self { value: DecimalValueTiered::Tier3(t.to_tier3()) }),
            (DecimalValueTiered::Tier2(t), 4) => Some(Self { value: DecimalValueTiered::Tier4(t.to_tier3().to_tier4()) }),
            (DecimalValueTiered::Tier2(t), 5) => Some(Self { value: DecimalValueTiered::Tier5(t.to_tier3().to_tier4().to_tier5()) }),
            (DecimalValueTiered::Tier2(t), 6) => Some(Self { value: DecimalValueTiered::Tier6(DecimalTier6 { value: D512::from_i128(t.value as i128), decimal_places: dp }) }),
            (DecimalValueTiered::Tier3(t), 4) => Some(Self { value: DecimalValueTiered::Tier4(t.to_tier4()) }),
            (DecimalValueTiered::Tier3(t), 5) => Some(Self { value: DecimalValueTiered::Tier5(t.to_tier4().to_tier5()) }),
            (DecimalValueTiered::Tier3(t), 6) => Some(Self { value: DecimalValueTiered::Tier6(DecimalTier6 { value: D512::from_i128(t.value as i128), decimal_places: dp }) }),
            (DecimalValueTiered::Tier4(t), 5) => Some(Self { value: DecimalValueTiered::Tier5(t.to_tier5()) }),
            (DecimalValueTiered::Tier4(t), 6) => Some(Self { value: DecimalValueTiered::Tier6(DecimalTier6 { value: D512::from_i128(t.value), decimal_places: dp }) }),
            (DecimalValueTiered::Tier5(t), 6) => Some(Self { value: DecimalValueTiered::Tier6(t.to_tier6()) }),
            _ => None,
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

pub fn max_decimal_places_for_tier(tier: u8) -> u8 {
    match tier {
        1 => 4,
        2 => 9,
        3 => 18,
        4 => 38,
        5 => 76,
        6 => 154,
        _ => 38,
    }
}

pub fn tier_for_decimal_places(decimal_places: u8) -> u8 {
    if decimal_places <= 4 { 1 }
    else if decimal_places <= 9 { 2 }
    else if decimal_places <= 18 { 3 }
    else if decimal_places <= 38 { 4 }
    else if decimal_places <= 76 { 5 }
    else { 6 }
}

pub(crate) fn align_to_common_tier(a: &UniversalDecimalTiered, b: &UniversalDecimalTiered) -> Result<(UniversalDecimalTiered, UniversalDecimalTiered), OverflowDetected> {
    let max_tier = a.current_tier().max(b.current_tier());
    let a_aligned = a.promote_to_tier(max_tier).ok_or(OverflowDetected::TierOverflow)?;
    let b_aligned = b.promote_to_tier(max_tier).ok_or(OverflowDetected::TierOverflow)?;
    Ok((a_aligned, b_aligned))
}

/// Align two decimal values to the same decimal places AND tier.
///
/// Scales up the value with fewer decimal places by multiplying by 10^(dp_max - dp).
/// Then aligns storage tiers. Used by division (and subtraction/addition with mismatched dp).
pub(crate) fn align_decimal_places_and_tier(a: &UniversalDecimalTiered, b: &UniversalDecimalTiered) -> Result<(UniversalDecimalTiered, UniversalDecimalTiered), OverflowDetected> {
    let dp_a = a.decimal_places();
    let dp_b = b.decimal_places();

    let (a_dp_aligned, b_dp_aligned) = if dp_a == dp_b {
        (*a, *b)
    } else if dp_a < dp_b {
        (scale_up_dp(a, dp_b)?, *b)
    } else {
        (*a, scale_up_dp(b, dp_a)?)
    };

    // Now align storage tiers
    align_to_common_tier(&a_dp_aligned, &b_dp_aligned)
}

/// Scale a decimal value up to a higher number of decimal places.
///
/// Multiplies the stored value by 10^(target_dp - current_dp) and updates dp.
fn scale_up_dp(val: &UniversalDecimalTiered, target_dp: u8) -> Result<UniversalDecimalTiered, OverflowDetected> {
    let current_dp = val.decimal_places();
    if target_dp <= current_dp {
        return Ok(*val);
    }
    let dp_diff = target_dp - current_dp;

    // Extract value as i128 (tiers 1-4) or handle wider types
    match &val.value {
        DecimalValueTiered::Tier1(t) => {
            let mut v = t.value as i128;
            for _ in 0..dp_diff {
                v = v.checked_mul(10).ok_or(OverflowDetected::TierOverflow)?;
            }
            let target_tier = tier_for_decimal_places(target_dp);
            store_i128_at_tier(v, target_dp, target_tier)
        }
        DecimalValueTiered::Tier2(t) => {
            let mut v = t.value as i128;
            for _ in 0..dp_diff {
                v = v.checked_mul(10).ok_or(OverflowDetected::TierOverflow)?;
            }
            let target_tier = tier_for_decimal_places(target_dp);
            store_i128_at_tier(v, target_dp, target_tier)
        }
        DecimalValueTiered::Tier3(t) => {
            let mut v = t.value as i128;
            for _ in 0..dp_diff {
                v = v.checked_mul(10).ok_or(OverflowDetected::TierOverflow)?;
            }
            let target_tier = tier_for_decimal_places(target_dp);
            store_i128_at_tier(v, target_dp, target_tier)
        }
        DecimalValueTiered::Tier4(t) => {
            let mut v = t.value;
            for _ in 0..dp_diff {
                v = v.checked_mul(10).ok_or(OverflowDetected::TierOverflow)?;
            }
            let target_tier = tier_for_decimal_places(target_dp);
            store_i128_at_tier(v, target_dp, target_tier)
        }
        DecimalValueTiered::Tier5(t) => {
            let mut v = d256_to_i256(t.value);
            let ten = I256::from_i128(10);
            for _ in 0..dp_diff {
                v = v * ten;
            }
            Ok(UniversalDecimalTiered {
                value: DecimalValueTiered::Tier5(DecimalTier5 { value: i256_to_d256(v), decimal_places: target_dp })
            })
        }
        DecimalValueTiered::Tier6(t) => {
            let mut v = d512_to_i512(t.value);
            let ten = I512::from_i128(10);
            for _ in 0..dp_diff {
                v = v * ten;
            }
            Ok(UniversalDecimalTiered {
                value: DecimalValueTiered::Tier6(DecimalTier6 { value: i512_to_d512(v), decimal_places: target_dp })
            })
        }
    }
}

/// Store an i128 value at the appropriate tier for the given dp.
/// Tries the target tier first, then promotes upward if the value doesn't fit.
fn store_i128_at_tier(val: i128, dp: u8, min_tier: u8) -> Result<UniversalDecimalTiered, OverflowDetected> {
    for tier in min_tier..=4 {
        match UniversalDecimalTiered::from_tier_value(tier, dp, val) {
            Ok(v) => return Ok(v),
            Err(OverflowDetected::TierOverflow) => continue,
            Err(e) => return Err(e),
        }
    }
    // If doesn't fit in i128 tiers, promote to Tier5 (I256)
    Ok(UniversalDecimalTiered {
        value: DecimalValueTiered::Tier5(DecimalTier5 { value: D256::from_i128(val), decimal_places: dp })
    })
}

// ============================================================================
// D256 <-> I256 / D512 <-> I512 CONVERSION HELPERS
// ============================================================================

#[inline(always)]
pub fn i256_to_d256(v: I256) -> D256 {
    D256::from_words(v.words)
}

#[inline(always)]
pub fn d256_to_i256(v: D256) -> I256 {
    I256::from_words(v.words)
}

#[inline(always)]
pub fn i512_to_d512(v: I512) -> D512 {
    D512::from_words(v.words)
}

#[inline(always)]
pub fn d512_to_i512(v: D512) -> I512 {
    I512::from_words(v.words)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_for_decimal_places() {
        assert_eq!(tier_for_decimal_places(2), 1);
        assert_eq!(tier_for_decimal_places(4), 1);
        assert_eq!(tier_for_decimal_places(5), 2);
        assert_eq!(tier_for_decimal_places(9), 2);
        assert_eq!(tier_for_decimal_places(18), 3);
        assert_eq!(tier_for_decimal_places(38), 4);
        assert_eq!(tier_for_decimal_places(76), 5);
        assert_eq!(tier_for_decimal_places(77), 6);
    }

    #[test]
    fn test_decimal_promote_to_tier() {
        let val = UniversalDecimalTiered::from_tier_value(1, 2, 100).unwrap();
        let promoted = val.promote_to_tier(4).unwrap();
        assert_eq!(promoted.current_tier(), 4);
        assert_eq!(promoted.decimal_places(), 2);
        if let DecimalValueTiered::Tier4(t) = promoted.value {
            assert_eq!(t.value, 100);
        } else {
            panic!("Expected tier 4");
        }
    }

    #[test]
    fn test_decimal_tier_raw_roundtrip() {
        let val = UniversalDecimalTiered::from_tier_value(4, 9, 123_456_789).unwrap();
        let (tier, raw) = val.to_tier_raw();
        assert_eq!(tier, 4);
        if let DecimalRaw::Small(v) = raw {
            assert_eq!(v, 123_456_789);
        } else {
            panic!("Expected Small");
        }
        let restored = UniversalDecimalTiered::from_tier_raw(tier, 9, raw).unwrap();
        assert_eq!(restored.current_tier(), 4);
        assert_eq!(restored.decimal_places(), 9);
    }

    #[test]
    fn test_d256_i256_conversion() {
        let d = D256::from_i128(42);
        let i = d256_to_i256(d);
        let d2 = i256_to_d256(i);
        assert_eq!(d, d2);
    }
}
