// ============================================================================
// TERNARY NEGATION — Free functions + UGOD method
// ============================================================================
//
// Extracted from ternary_operations.rs (free functions) and
// universal_ternary.rs (UGOD negate method).

use super::ternary_types::{
    TernaryTier, TernaryTier1, TernaryTier2, TernaryTier3, TernaryTier4, TernaryTier5, TernaryTier6,
    TernaryValue, UniversalTernaryFixed,
};
use crate::fixed_point::{I256, I512, I1024};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// FREE FUNCTIONS — Tier-level negation
// ============================================================================

/// TQ10.10 negation
#[inline]
pub fn negate_ternary_tq10_10(a: i32) -> Result<i32, OverflowDetected> {
    match a.checked_neg() {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow), // i32::MIN negation overflows
    }
}

/// TQ20.20 negation
#[inline]
pub fn negate_ternary_tq20_20(a: i64) -> Result<i64, OverflowDetected> {
    match a.checked_neg() {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

/// TQ40.40 negation
#[inline]
pub fn negate_ternary_tq40_40(a: i128) -> Result<i128, OverflowDetected> {
    match a.checked_neg() {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

/// TQ80.80 negation.
///
/// 0.5.0 consistency fix: was `saturating_neg` — the one tier that silently
/// absorbed the binary-MIN edge instead of failing loud like every other
/// tier's `checked_neg`. I256::MIN is unreachable from any valid ternary
/// value (the ternary window is symmetric), so panicking is the correct
/// fail-loud behavior for the impossible case.
#[inline]
pub fn negate_ternary_tq80_80(a: I256) -> I256 {
    assert!(
        a != I256::min_value(),
        "ternary Tier4 negate: I256::MIN is outside the balanced-ternary window"
    );
    -a
}

/// TQ160.160 negation
#[inline]
pub fn negate_ternary_tq160_160(a: I512) -> Result<I512, OverflowDetected> {
    match a.checked_neg() {
        Some(result) => Ok(result),
        None => Err(OverflowDetected::TierOverflow),
    }
}

/// TQ320.320 negation (maximum precision, never fails)
#[inline]
pub fn negate_ternary_tq320_320(a: I1024) -> I1024 {
    -a
}

// ============================================================================
// UGOD METHOD — Negation with overflow promotion
// ============================================================================

impl UniversalTernaryFixed {
    /// Negation with UGOD overflow promotion
    pub fn negate(&self) -> Result<Self, OverflowDetected> {
        match &self.value {
            TernaryValue::Tier1(t) => {
                match negate_ternary_tq10_10(t.raw()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier1(TernaryTier1::from_raw(result)), current_tier: TernaryTier::Tier1 }),
                    Err(_) => {
                        let promoted = self.promote_to_tier2()?;
                        promoted.negate()
                    }
                }
            }
            TernaryValue::Tier2(t) => {
                match negate_ternary_tq20_20(t.raw()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier2(TernaryTier2::from_raw(result)), current_tier: TernaryTier::Tier2 }),
                    Err(_) => {
                        let promoted = self.promote_to_tier3()?;
                        promoted.negate()
                    }
                }
            }
            TernaryValue::Tier3(t) => {
                match negate_ternary_tq40_40(t.raw()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier3(TernaryTier3::from_raw(result)), current_tier: TernaryTier::Tier3 }),
                    Err(_) => {
                        let promoted = self.promote_to_tier4();
                        promoted.negate()
                    }
                }
            }
            TernaryValue::Tier4(t) => {
                let result = negate_ternary_tq80_80(t.raw().clone());
                Ok(Self { value: TernaryValue::Tier4(TernaryTier4::from_raw(result)), current_tier: TernaryTier::Tier4 })
            }
            TernaryValue::Tier5(t) => {
                match negate_ternary_tq160_160(t.raw().clone()) {
                    Ok(result) => Ok(Self { value: TernaryValue::Tier5(TernaryTier5::from_raw(result)), current_tier: TernaryTier::Tier5 }),
                    Err(_) => {
                        let promoted = self.promote_to_tier6();
                        promoted.negate()
                    }
                }
            }
            TernaryValue::Tier6(t) => {
                let result = negate_ternary_tq320_320(t.raw().clone());
                Ok(Self { value: TernaryValue::Tier6(TernaryTier6::from_raw(result)), current_tier: TernaryTier::Tier6 })
            }
        }
    }
}
