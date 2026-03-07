//! Binary Fixed-Point Tier Types, Constructors, and Promotion
//!
//! **MISSION**: Type definitions and tier management for UGOD binary fixed-point
//! **ARCHITECTURE**: Q16.16(i32) -> Q32.32(i64) -> Q64.64(i128) -> Q128.128(I256) -> Q256.256(I512) -> Q512.512(I1024)

use crate::fixed_point::{I256, I512, I1024};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// TIER-SPECIFIC STORAGE TYPES
// ============================================================================

/// Tier 1: Q16.16 -- Compact Binary Precision (16 integer + 16 fractional bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryTier1 {
    pub(crate) value: i32,
}

/// Tier 2: Q32.32 -- Standard Binary Precision (32 integer + 32 fractional bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryTier2 {
    pub(crate) value: i64,
}

/// Tier 3: Q64.64 -- Extended Binary Precision (64 integer + 64 fractional bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryTier3 {
    pub(crate) value: i128,
}

/// Tier 4: Q128.128 -- Balanced Precision (128 integer + 128 fractional bits)
#[derive(Debug, Clone)]
pub struct BinaryTier4 {
    pub(crate) value: I256,
}

/// Tier 5: Q256.256 -- Scientific Precision (256 integer + 256 fractional bits)
#[derive(Debug, Clone)]
pub struct BinaryTier5 {
    pub(crate) value: I512,
}

/// Tier 6: Q512.512 -- Extended Precision (512 integer + 512 fractional bits)
#[derive(Debug, Clone)]
pub struct BinaryTier6 {
    pub(crate) value: I1024,
}

// ============================================================================
// RAW BINARY VALUE -- FASC BOUNDARY CROSSING
// ============================================================================

#[derive(Debug, Clone)]
pub enum BinaryRaw {
    Small(i128),
    Medium(I256),
    Large(I512),
    XLarge(I1024),
}

// ============================================================================
// SCALE FACTOR CONSTANTS
// ============================================================================

pub const SCALE_Q16_16: i32 = 1 << 16;
pub const SCALE_Q32_32: i64 = 1 << 32;

// ============================================================================
// TIER 1: Q16.16 -- CONSTRUCTORS AND PROMOTION
// ============================================================================

impl BinaryTier1 {
    pub const fn from_raw(value: i32) -> Self { Self { value } }
    pub const fn raw(&self) -> i32 { self.value }
    pub const fn zero() -> Self { Self { value: 0 } }
    pub const fn one() -> Self { Self { value: SCALE_Q16_16 } }

    pub fn from_integer(n: i16) -> Result<Self, ()> {
        match (n as i32).checked_mul(SCALE_Q16_16) {
            Some(scaled) => Ok(Self { value: scaled }),
            None => Err(()),
        }
    }

    pub fn to_tier2(&self) -> BinaryTier2 {
        BinaryTier2 { value: (self.value as i64) << 16 }
    }
}

// ============================================================================
// TIER 2: Q32.32 -- CONSTRUCTORS AND PROMOTION
// ============================================================================

impl BinaryTier2 {
    pub const fn from_raw(value: i64) -> Self { Self { value } }
    pub const fn raw(&self) -> i64 { self.value }
    pub const fn zero() -> Self { Self { value: 0 } }
    pub const fn one() -> Self { Self { value: SCALE_Q32_32 } }

    pub fn from_integer(n: i32) -> Result<Self, ()> {
        match (n as i64).checked_mul(SCALE_Q32_32) {
            Some(scaled) => Ok(Self { value: scaled }),
            None => Err(()),
        }
    }

    pub fn to_tier3(&self) -> BinaryTier3 {
        BinaryTier3 { value: (self.value as i128) << 32 }
    }

    pub fn to_tier1(&self) -> Result<BinaryTier1, ()> {
        let shifted = self.value >> 16;
        if shifted >= i32::MIN as i64 && shifted <= i32::MAX as i64 {
            Ok(BinaryTier1 { value: shifted as i32 })
        } else {
            Err(())
        }
    }
}

// ============================================================================
// TIER 3: Q64.64 -- CONSTRUCTORS AND PROMOTION
// ============================================================================

impl BinaryTier3 {
    pub const fn from_raw(value: i128) -> Self { Self { value } }
    pub const fn raw(&self) -> i128 { self.value }
    pub const fn zero() -> Self { Self { value: 0 } }

    pub fn one() -> Self { Self { value: 1i128 << 64 } }

    pub fn from_integer(n: i64) -> Result<Self, ()> {
        match (n as i128).checked_mul(1i128 << 64) {
            Some(scaled) => Ok(Self { value: scaled }),
            None => Err(()),
        }
    }

    pub fn to_tier4(&self) -> BinaryTier4 {
        BinaryTier4 { value: I256::from_i128(self.value) << 64 }
    }

    pub fn to_tier2(&self) -> Result<BinaryTier2, ()> {
        let shifted = self.value >> 32;
        if shifted >= i64::MIN as i128 && shifted <= i64::MAX as i128 {
            Ok(BinaryTier2 { value: shifted as i64 })
        } else {
            Err(())
        }
    }
}

// ============================================================================
// TIER 4: Q128.128 -- CONSTRUCTORS AND PROMOTION
// ============================================================================

impl BinaryTier4 {
    pub fn from_raw(value: I256) -> Self { Self { value } }
    pub fn raw(&self) -> &I256 { &self.value }
    pub fn zero() -> Self { Self { value: I256::zero() } }
    pub fn one() -> Self { Self { value: I256::from_i128(1) << 128 } }

    pub fn from_integer(n: i128) -> Self {
        Self { value: I256::from_i128(n) << 128 }
    }

    pub fn to_tier5(&self) -> BinaryTier5 {
        BinaryTier5 { value: I512::from_i256(self.value) << 128 }
    }

    pub fn to_tier3(&self) -> Result<BinaryTier3, ()> {
        let shifted = self.value >> 64;
        if shifted.fits_in_i128() {
            Ok(BinaryTier3 { value: shifted.as_i128() })
        } else {
            Err(())
        }
    }
}

// ============================================================================
// TIER 5: Q256.256 -- CONSTRUCTORS AND PROMOTION
// ============================================================================

impl BinaryTier5 {
    pub fn from_raw(value: I512) -> Self { Self { value } }
    pub fn raw(&self) -> &I512 { &self.value }
    pub fn zero() -> Self { Self { value: I512::zero() } }
    pub fn one() -> Self { Self { value: I512::from_i128(1) << 256 } }

    pub fn from_integer(n: i128) -> Self {
        Self { value: I512::from_i128(n) << 256 }
    }

    pub fn to_tier6(&self) -> BinaryTier6 {
        BinaryTier6 { value: I1024::from_i512(self.value) << 256 }
    }

    pub fn to_tier4(&self) -> Result<BinaryTier4, ()> {
        let shifted = self.value >> 128;
        if shifted.fits_in_i256() {
            Ok(BinaryTier4 { value: shifted.as_i256() })
        } else {
            Err(())
        }
    }
}

// ============================================================================
// TIER 6: Q512.512 -- CONSTRUCTORS AND PROMOTION
// ============================================================================

impl BinaryTier6 {
    pub fn from_raw(value: I1024) -> Self { Self { value } }
    pub fn raw(&self) -> &I1024 { &self.value }
    pub fn zero() -> Self { Self { value: I1024::zero() } }
    pub fn one() -> Self { Self { value: I1024::from_i128(1) << 512 } }

    pub fn from_integer(n: i128) -> Self {
        Self { value: I1024::from_i128(n) << 512 }
    }

    pub fn to_tier5(&self) -> Result<BinaryTier5, ()> {
        let shifted = self.value >> 256;
        if shifted.fits_in_i512() {
            Ok(BinaryTier5 { value: shifted.as_i512() })
        } else {
            Err(())
        }
    }
}

// ============================================================================
// UNIFIED BINARY VALUE ENUM
// ============================================================================

#[derive(Debug, Clone)]
pub enum BinaryValue {
    Tier1(BinaryTier1),
    Tier2(BinaryTier2),
    Tier3(BinaryTier3),
    Tier4(BinaryTier4),
    Tier5(BinaryTier5),
    Tier6(BinaryTier6),
}

// ============================================================================
// UNIVERSAL BINARY FIXED -- RUNTIME UGOD TYPE
// ============================================================================

#[derive(Debug, Clone)]
pub struct UniversalBinaryFixed {
    pub(crate) value: BinaryValue,
    pub(crate) current_tier: u8,
}

impl UniversalBinaryFixed {
    pub fn from_tier_raw(tier: u8, raw: BinaryRaw) -> Result<Self, OverflowDetected> {
        match (tier, raw) {
            (1, BinaryRaw::Small(v)) => Ok(Self {
                value: BinaryValue::Tier1(BinaryTier1::from_raw(v as i32)),
                current_tier: 1,
            }),
            (2, BinaryRaw::Small(v)) => Ok(Self {
                value: BinaryValue::Tier2(BinaryTier2::from_raw(v as i64)),
                current_tier: 2,
            }),
            (3, BinaryRaw::Small(v)) => Ok(Self {
                value: BinaryValue::Tier3(BinaryTier3::from_raw(v)),
                current_tier: 3,
            }),
            (4, BinaryRaw::Medium(v)) => Ok(Self {
                value: BinaryValue::Tier4(BinaryTier4::from_raw(v)),
                current_tier: 4,
            }),
            (5, BinaryRaw::Large(v)) => Ok(Self {
                value: BinaryValue::Tier5(BinaryTier5::from_raw(v)),
                current_tier: 5,
            }),
            (6, BinaryRaw::XLarge(v)) => Ok(Self {
                value: BinaryValue::Tier6(BinaryTier6::from_raw(v)),
                current_tier: 6,
            }),
            _ => Err(OverflowDetected::InvalidInput),
        }
    }

    pub fn to_tier_raw(&self) -> (u8, BinaryRaw) {
        match &self.value {
            BinaryValue::Tier1(t) => (1, BinaryRaw::Small(t.raw() as i128)),
            BinaryValue::Tier2(t) => (2, BinaryRaw::Small(t.raw() as i128)),
            BinaryValue::Tier3(t) => (3, BinaryRaw::Small(t.raw())),
            BinaryValue::Tier4(t) => (4, BinaryRaw::Medium(*t.raw())),
            BinaryValue::Tier5(t) => (5, BinaryRaw::Large(*t.raw())),
            BinaryValue::Tier6(t) => (6, BinaryRaw::XLarge(*t.raw())),
        }
    }

    pub fn from_tier_value(tier: u8, value: i128) -> Result<Self, OverflowDetected> {
        Self::from_tier_raw(tier, BinaryRaw::Small(value))
    }

    pub fn current_tier(&self) -> u8 {
        self.current_tier
    }

    pub fn zero() -> Self {
        Self {
            value: BinaryValue::Tier1(BinaryTier1::zero()),
            current_tier: 1,
        }
    }

    pub fn one() -> Self {
        Self {
            value: BinaryValue::Tier1(BinaryTier1::one()),
            current_tier: 1,
        }
    }

    pub fn from_integer(n: i64) -> Self {
        if n >= i16::MIN as i64 && n <= i16::MAX as i64 {
            if let Ok(t) = BinaryTier1::from_integer(n as i16) {
                return Self { value: BinaryValue::Tier1(t), current_tier: 1 };
            }
        }
        if n >= i32::MIN as i64 && n <= i32::MAX as i64 {
            if let Ok(t) = BinaryTier2::from_integer(n as i32) {
                return Self { value: BinaryValue::Tier2(t), current_tier: 2 };
            }
        }
        if let Ok(t) = BinaryTier3::from_integer(n) {
            return Self { value: BinaryValue::Tier3(t), current_tier: 3 };
        }
        Self {
            value: BinaryValue::Tier4(BinaryTier4::from_integer(n as i128)),
            current_tier: 4,
        }
    }

    pub fn promote_to_tier(&self, target: u8) -> Result<Self, OverflowDetected> {
        if target <= self.current_tier {
            return Ok(self.clone());
        }
        if target > 6 {
            return Err(OverflowDetected::TierOverflow);
        }
        let mut current = self.clone();
        while current.current_tier < target {
            current = current.promote_one_tier()?;
        }
        Ok(current)
    }

    fn promote_one_tier(&self) -> Result<Self, OverflowDetected> {
        match &self.value {
            BinaryValue::Tier1(t) => Ok(Self {
                value: BinaryValue::Tier2(t.to_tier2()),
                current_tier: 2,
            }),
            BinaryValue::Tier2(t) => Ok(Self {
                value: BinaryValue::Tier3(t.to_tier3()),
                current_tier: 3,
            }),
            BinaryValue::Tier3(t) => Ok(Self {
                value: BinaryValue::Tier4(t.to_tier4()),
                current_tier: 4,
            }),
            BinaryValue::Tier4(t) => Ok(Self {
                value: BinaryValue::Tier5(t.to_tier5()),
                current_tier: 5,
            }),
            BinaryValue::Tier5(t) => Ok(Self {
                value: BinaryValue::Tier6(t.to_tier6()),
                current_tier: 6,
            }),
            BinaryValue::Tier6(_) => Err(OverflowDetected::TierOverflow),
        }
    }

    pub(crate) fn align_to_common_tier(&self, other: &Self) -> (Self, Self) {
        let target = self.current_tier.max(other.current_tier);
        let a = self.promote_to_tier(target).unwrap_or_else(|_| self.clone());
        let b = other.promote_to_tier(target).unwrap_or_else(|_| other.clone());
        (a, b)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier1_basics() {
        let zero = BinaryTier1::zero();
        let one = BinaryTier1::one();
        assert_eq!(zero.raw(), 0);
        assert_eq!(one.raw(), SCALE_Q16_16);
    }

    #[test]
    fn test_tier_promotion_lossless() {
        let t1 = BinaryTier1::one();
        let t2 = t1.to_tier2();
        assert_eq!(t2.raw(), SCALE_Q32_32);
    }

    #[test]
    fn test_binary_raw_roundtrip() {
        let ubf = UniversalBinaryFixed::from_integer(42);
        let (tier, raw) = ubf.to_tier_raw();
        let restored = UniversalBinaryFixed::from_tier_raw(tier, raw).unwrap();
        let (tier2, _raw2) = restored.to_tier_raw();
        assert_eq!(tier, tier2);
    }

    #[test]
    fn test_storage_efficiency() {
        use core::mem::size_of;
        assert_eq!(size_of::<BinaryTier1>(), 4);
        assert_eq!(size_of::<BinaryTier2>(), 8);
        assert_eq!(size_of::<BinaryTier3>(), 16);
    }
}
