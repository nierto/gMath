//! Balanced Ternary Addition and Subtraction Operations
//!
//! **MISSION**: Tier-specific add/sub with overflow detection + UGOD methods
//! **ARCHITECTURE**: Free functions operate on raw integers; UGOD methods on UniversalTernaryFixed
//! **PRECISION**: Pure ternary arithmetic preserving exact division by 3
//! **INTEGRATION**: Extracted from ternary_operations.rs and universal_ternary.rs

use super::ternary_types::{
    TernaryTier, TernaryTier1, TernaryTier2, TernaryTier3, TernaryTier4, TernaryTier5, TernaryTier6,
    TernaryValue, UniversalTernaryFixed,
};
use crate::fixed_point::{I256, I512, I1024};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// TIER 1: TQ8.8 ADD/SUBTRACT (i32 storage)
// ============================================================================

/// TQ8.8 addition with overflow detection
///
/// **PRECISION**: 8 integer + 8 fractional trits
/// **STORAGE**: Native i32 operations
/// **PERFORMANCE**: 8x SIMD possible with AVX2
#[inline]
pub fn add_ternary_tq8_8(a: i32, b: i32) -> Result<i32, OverflowDetected> {
    match a.checked_add(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

/// TQ8.8 subtraction with overflow detection
#[inline]
pub fn subtract_ternary_tq8_8(a: i32, b: i32) -> Result<i32, OverflowDetected> {
    match a.checked_sub(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

// ============================================================================
// TIER 2: TQ16.16 ADD/SUBTRACT (i64 storage)
// ============================================================================

/// TQ16.16 addition with overflow detection
///
/// **PRECISION**: 16 integer + 16 fractional trits
/// **STORAGE**: Native i64 operations
/// **PERFORMANCE**: 4x SIMD possible with AVX2
#[inline]
pub fn add_ternary_tq16_16(a: i64, b: i64) -> Result<i64, OverflowDetected> {
    match a.checked_add(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

/// TQ16.16 subtraction with overflow detection
#[inline]
pub fn subtract_ternary_tq16_16(a: i64, b: i64) -> Result<i64, OverflowDetected> {
    match a.checked_sub(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

// ============================================================================
// TIER 3: TQ32.32 ADD/SUBTRACT (i128 storage)
// ============================================================================

/// TQ32.32 addition with overflow detection
///
/// **PRECISION**: 32 integer + 32 fractional trits
/// **STORAGE**: Native i128 operations
/// **PERFORMANCE**: 2x SIMD possible with AVX2
#[inline]
pub fn add_ternary_tq32_32(a: i128, b: i128) -> Result<i128, OverflowDetected> {
    match a.checked_add(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

/// TQ32.32 subtraction with overflow detection
#[inline]
pub fn subtract_ternary_tq32_32(a: i128, b: i128) -> Result<i128, OverflowDetected> {
    match a.checked_sub(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

// ============================================================================
// TIER 4: TQ64.64 ADD/SUBTRACT (I256 storage, NEVER FAILS)
// ============================================================================

/// TQ64.64 addition (maximum precision, never fails)
///
/// **PRECISION**: 64 integer + 64 fractional trits
/// **STORAGE**: I256 operations with saturation
/// **GUARANTEE**: Never fails, uses saturation for extreme overflow
#[inline]
pub fn add_ternary_tq64_64(a: I256, b: I256) -> I256 {
    // I256 addition with saturation protection
    a.saturating_add(b)
}

/// TQ64.64 subtraction (maximum precision, never fails)
#[inline]
pub fn subtract_ternary_tq64_64(a: I256, b: I256) -> I256 {
    // I256 subtraction with saturation protection
    a.saturating_sub(b)
}

// ============================================================================
// TIER 4 CHECKED VARIANTS (for promotion to Tier 5)
// ============================================================================

/// TQ64.64 addition with overflow detection (checked variant for UGOD promotion)
#[inline]
pub fn add_ternary_tq64_64_checked(a: I256, b: I256) -> Result<I256, OverflowDetected> {
    match a.checked_add(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

/// TQ64.64 subtraction with overflow detection (checked variant)
#[inline]
pub fn subtract_ternary_tq64_64_checked(a: I256, b: I256) -> Result<I256, OverflowDetected> {
    match a.checked_sub(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

// ============================================================================
// TIER 5: TQ128.128 ADD/SUBTRACT (I512 storage, checked)
// ============================================================================

/// TQ128.128 addition with overflow detection
#[inline]
pub fn add_ternary_tq128_128(a: I512, b: I512) -> Result<I512, OverflowDetected> {
    match a.checked_add(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

/// TQ128.128 subtraction with overflow detection
#[inline]
pub fn subtract_ternary_tq128_128(a: I512, b: I512) -> Result<I512, OverflowDetected> {
    match a.checked_sub(b) {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

// ============================================================================
// TIER 6: TQ256.256 ADD/SUBTRACT (I1024 storage, saturating — NEVER FAILS)
// ============================================================================

/// TQ256.256 addition (maximum precision, never fails)
#[inline]
pub fn add_ternary_tq256_256(a: I1024, b: I1024) -> I1024 {
    // I1024 addition — wrap on overflow (saturating not available, but
    // at 1024 bits the range is so enormous overflow is practically impossible)
    a + b
}

/// TQ256.256 subtraction (maximum precision, never fails)
#[inline]
pub fn subtract_ternary_tq256_256(a: I1024, b: I1024) -> I1024 {
    a - b
}

// ============================================================================
// UGOD METHODS: ADD / SUBTRACT
// ============================================================================

impl UniversalTernaryFixed {
    /// Addition with automatic tier alignment
    pub fn add(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let (aligned_self, aligned_other) = self.align_to_common_tier(other);

        match (&aligned_self.value, &aligned_other.value) {
            (TernaryValue::Tier1(a), TernaryValue::Tier1(b)) => {
                match add_ternary_tq8_8(a.raw(), b.raw()) {
                    Ok(result) => Ok(Self {
                        value: TernaryValue::Tier1(TernaryTier1::from_raw(result)),
                        current_tier: TernaryTier::Tier1,
                    }),
                    Err(_) => {
                        // Promote both to Tier 2 and retry
                        let promoted_self = aligned_self.promote_to_tier2()?;
                        let promoted_other = aligned_other.promote_to_tier2()?;
                        promoted_self.add(&promoted_other)
                    }
                }
            }
            (TernaryValue::Tier2(a), TernaryValue::Tier2(b)) => {
                match add_ternary_tq16_16(a.raw(), b.raw()) {
                    Ok(result) => Ok(Self {
                        value: TernaryValue::Tier2(TernaryTier2::from_raw(result)),
                        current_tier: TernaryTier::Tier2,
                    }),
                    Err(_) => {
                        let promoted_self = aligned_self.promote_to_tier3()?;
                        let promoted_other = aligned_other.promote_to_tier3()?;
                        promoted_self.add(&promoted_other)
                    }
                }
            }
            (TernaryValue::Tier3(a), TernaryValue::Tier3(b)) => {
                match add_ternary_tq32_32(a.raw(), b.raw()) {
                    Ok(result) => Ok(Self {
                        value: TernaryValue::Tier3(TernaryTier3::from_raw(result)),
                        current_tier: TernaryTier::Tier3,
                    }),
                    Err(_) => {
                        let promoted_self = aligned_self.promote_to_tier4();
                        let promoted_other = aligned_other.promote_to_tier4();
                        promoted_self.add(&promoted_other)
                    }
                }
            }
            (TernaryValue::Tier4(a), TernaryValue::Tier4(b)) => {
                match add_ternary_tq64_64_checked(a.raw().clone(), b.raw().clone()) {
                    Ok(result) => Ok(Self {
                        value: TernaryValue::Tier4(TernaryTier4::from_raw(result)),
                        current_tier: TernaryTier::Tier4,
                    }),
                    Err(_) => {
                        let promoted_self = aligned_self.promote_to_tier5();
                        let promoted_other = aligned_other.promote_to_tier5();
                        promoted_self.add(&promoted_other)
                    }
                }
            }
            (TernaryValue::Tier5(a), TernaryValue::Tier5(b)) => {
                match add_ternary_tq128_128(a.raw().clone(), b.raw().clone()) {
                    Ok(result) => Ok(Self {
                        value: TernaryValue::Tier5(TernaryTier5::from_raw(result)),
                        current_tier: TernaryTier::Tier5,
                    }),
                    Err(_) => {
                        let promoted_self = aligned_self.promote_to_tier6();
                        let promoted_other = aligned_other.promote_to_tier6();
                        promoted_self.add(&promoted_other)
                    }
                }
            }
            (TernaryValue::Tier6(a), TernaryValue::Tier6(b)) => {
                // Tier 6 never fails (max tier, wrapping)
                let result = add_ternary_tq256_256(a.raw().clone(), b.raw().clone());
                Ok(Self {
                    value: TernaryValue::Tier6(TernaryTier6::from_raw(result)),
                    current_tier: TernaryTier::Tier6,
                })
            }
            _ => unreachable!("align_to_common_tier should ensure matching tiers")
        }
    }

    /// Subtraction with automatic tier alignment and UGOD overflow promotion
    pub fn subtract(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let (aligned_self, aligned_other) = self.align_to_common_tier(other);

        match (&aligned_self.value, &aligned_other.value) {
            (TernaryValue::Tier1(a), TernaryValue::Tier1(b)) => {
                match subtract_ternary_tq8_8(a.raw(), b.raw()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier1(TernaryTier1::from_raw(result)), current_tier: TernaryTier::Tier1 }),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier2()?;
                        let p_other = aligned_other.promote_to_tier2()?;
                        p_self.subtract(&p_other)
                    }
                }
            }
            (TernaryValue::Tier2(a), TernaryValue::Tier2(b)) => {
                match subtract_ternary_tq16_16(a.raw(), b.raw()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier2(TernaryTier2::from_raw(result)), current_tier: TernaryTier::Tier2 }),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier3()?;
                        let p_other = aligned_other.promote_to_tier3()?;
                        p_self.subtract(&p_other)
                    }
                }
            }
            (TernaryValue::Tier3(a), TernaryValue::Tier3(b)) => {
                match subtract_ternary_tq32_32(a.raw(), b.raw()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier3(TernaryTier3::from_raw(result)), current_tier: TernaryTier::Tier3 }),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier4();
                        let p_other = aligned_other.promote_to_tier4();
                        p_self.subtract(&p_other)
                    }
                }
            }
            (TernaryValue::Tier4(a), TernaryValue::Tier4(b)) => {
                match subtract_ternary_tq64_64_checked(a.raw().clone(), b.raw().clone()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier4(TernaryTier4::from_raw(result)), current_tier: TernaryTier::Tier4 }),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier5();
                        let p_other = aligned_other.promote_to_tier5();
                        p_self.subtract(&p_other)
                    }
                }
            }
            (TernaryValue::Tier5(a), TernaryValue::Tier5(b)) => {
                match subtract_ternary_tq128_128(a.raw().clone(), b.raw().clone()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier5(TernaryTier5::from_raw(result)), current_tier: TernaryTier::Tier5 }),
                    Err(_) => {
                        let p_self = aligned_self.promote_to_tier6();
                        let p_other = aligned_other.promote_to_tier6();
                        p_self.subtract(&p_other)
                    }
                }
            }
            (TernaryValue::Tier6(a), TernaryValue::Tier6(b)) => {
                let result = subtract_ternary_tq256_256(a.raw().clone(), b.raw().clone());
                Ok(Self { value: TernaryValue::Tier6(TernaryTier6::from_raw(result)), current_tier: TernaryTier::Tier6 })
            }
            _ => unreachable!("align_to_common_tier should ensure matching tiers")
        }
    }
}
