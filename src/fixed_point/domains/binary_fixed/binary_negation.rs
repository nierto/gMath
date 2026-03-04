//! Binary Fixed-Point Negation -- Tier Primitives + UGOD Wrapper
//!
//! **OPERATIONS**: checked_neg (Tiers 1-3), neg (Tiers 4-6)
//! **UGOD**: UniversalBinaryFixed::negate() with overflow promotion

use super::binary_types::*;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// TIER 1: Q16.16
// ============================================================================

impl BinaryTier1 {
    pub fn checked_neg(&self) -> Option<Self> {
        self.value.checked_neg().map(|v| Self { value: v })
    }
}

// ============================================================================
// TIER 2: Q32.32
// ============================================================================

impl BinaryTier2 {
    pub fn checked_neg(&self) -> Option<Self> {
        self.value.checked_neg().map(|v| Self { value: v })
    }
}

// ============================================================================
// TIER 3: Q64.64
// ============================================================================

impl BinaryTier3 {
    pub fn checked_neg(&self) -> Option<Self> {
        self.value.checked_neg().map(|v| Self { value: v })
    }
}

// ============================================================================
// TIER 4: Q128.128
// ============================================================================

impl BinaryTier4 {
    pub fn neg(&self) -> Self {
        Self { value: -self.value }
    }
}

// ============================================================================
// TIER 5: Q256.256
// ============================================================================

impl BinaryTier5 {
    pub fn neg(&self) -> Self {
        Self { value: -self.value }
    }
}

// ============================================================================
// TIER 6: Q512.512
// ============================================================================

impl BinaryTier6 {
    pub fn neg(&self) -> Self {
        Self { value: -self.value }
    }
}

// ============================================================================
// UGOD: UniversalBinaryFixed::negate
// ============================================================================

impl UniversalBinaryFixed {
    pub fn negate(&self) -> Result<Self, OverflowDetected> {
        match &self.value {
            BinaryValue::Tier1(t) => {
                match t.checked_neg() {
                    Some(r) => Ok(Self { value: BinaryValue::Tier1(r), current_tier: 1 }),
                    None => {
                        let promoted = t.to_tier2();
                        Ok(Self { value: BinaryValue::Tier2(BinaryTier2::from_raw(-promoted.raw())), current_tier: 2 })
                    }
                }
            }
            BinaryValue::Tier2(t) => {
                match t.checked_neg() {
                    Some(r) => Ok(Self { value: BinaryValue::Tier2(r), current_tier: 2 }),
                    None => {
                        let promoted = t.to_tier3();
                        Ok(Self { value: BinaryValue::Tier3(BinaryTier3::from_raw(-promoted.raw())), current_tier: 3 })
                    }
                }
            }
            BinaryValue::Tier3(t) => {
                match t.checked_neg() {
                    Some(r) => Ok(Self { value: BinaryValue::Tier3(r), current_tier: 3 }),
                    None => {
                        let promoted = t.to_tier4();
                        Ok(Self { value: BinaryValue::Tier4(promoted.neg()), current_tier: 4 })
                    }
                }
            }
            BinaryValue::Tier4(t) => Ok(Self { value: BinaryValue::Tier4(t.neg()), current_tier: 4 }),
            BinaryValue::Tier5(t) => Ok(Self { value: BinaryValue::Tier5(t.neg()), current_tier: 5 }),
            BinaryValue::Tier6(t) => Ok(Self { value: BinaryValue::Tier6(t.neg()), current_tier: 6 }),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negate() {
        let one = UniversalBinaryFixed::from_integer(1);
        let neg = one.negate().unwrap();
        let (_, raw) = neg.to_tier_raw();
        match raw {
            BinaryRaw::Small(v) => assert!(v < 0),
            _ => panic!("Expected Small variant"),
        }
    }
}
