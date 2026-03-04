//! Decimal Fixed-Point Negation -- Tier Primitives + UGOD Wrapper
//!
//! **OPERATIONS**: checked_neg (Tiers 1-4), neg (Tiers 5-6)
//! **UGOD**: UniversalDecimalTiered::negate() with overflow promotion

use super::decimal_types::*;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// TIER 1-4: CHECKED NEG
// ============================================================================

impl DecimalTier1 {
    pub fn checked_neg(self) -> Option<Self> {
        self.value.checked_neg().map(|v| Self { value: v, decimal_places: self.decimal_places })
    }
}

impl DecimalTier2 {
    pub fn checked_neg(self) -> Option<Self> {
        self.value.checked_neg().map(|v| Self { value: v, decimal_places: self.decimal_places })
    }
}

impl DecimalTier3 {
    pub fn checked_neg(self) -> Option<Self> {
        self.value.checked_neg().map(|v| Self { value: v, decimal_places: self.decimal_places })
    }
}

impl DecimalTier4 {
    pub fn checked_neg(self) -> Option<Self> {
        self.value.checked_neg().map(|v| Self { value: v, decimal_places: self.decimal_places })
    }
}

// ============================================================================
// TIER 5-6: NON-FAILING NEG
// ============================================================================

impl DecimalTier5 {
    pub fn neg(self) -> Self {
        Self { value: super::negate_d256(self.value), decimal_places: self.decimal_places }
    }
}

impl DecimalTier6 {
    pub fn neg(self) -> Self {
        Self { value: super::negate_d512(self.value), decimal_places: self.decimal_places }
    }
}

// ============================================================================
// UGOD: UniversalDecimalTiered::negate
// ============================================================================

impl UniversalDecimalTiered {
    pub fn negate(&self) -> Result<Self, OverflowDetected> {
        match &self.value {
            DecimalValueTiered::Tier1(t) => {
                match t.checked_neg() {
                    Some(r) => Ok(Self { value: DecimalValueTiered::Tier1(r) }),
                    None => {
                        let promoted = t.to_tier2();
                        Ok(Self { value: DecimalValueTiered::Tier2(DecimalTier2 { value: -(promoted.value), decimal_places: promoted.decimal_places }) })
                    }
                }
            }
            DecimalValueTiered::Tier2(t) => {
                match t.checked_neg() {
                    Some(r) => Ok(Self { value: DecimalValueTiered::Tier2(r) }),
                    None => {
                        let promoted = t.to_tier3();
                        Ok(Self { value: DecimalValueTiered::Tier3(DecimalTier3 { value: -(promoted.value), decimal_places: promoted.decimal_places }) })
                    }
                }
            }
            DecimalValueTiered::Tier3(t) => {
                match t.checked_neg() {
                    Some(r) => Ok(Self { value: DecimalValueTiered::Tier3(r) }),
                    None => {
                        let promoted = t.to_tier4();
                        Ok(Self { value: DecimalValueTiered::Tier4(DecimalTier4 { value: -(promoted.value), decimal_places: promoted.decimal_places }) })
                    }
                }
            }
            DecimalValueTiered::Tier4(t) => {
                match t.checked_neg() {
                    Some(r) => Ok(Self { value: DecimalValueTiered::Tier4(r) }),
                    None => {
                        let promoted = t.to_tier5();
                        Ok(Self { value: DecimalValueTiered::Tier5(promoted.neg()) })
                    }
                }
            }
            DecimalValueTiered::Tier5(t) => {
                Ok(Self { value: DecimalValueTiered::Tier5(t.neg()) })
            }
            DecimalValueTiered::Tier6(t) => {
                Ok(Self { value: DecimalValueTiered::Tier6(t.neg()) })
            }
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
    fn test_decimal_negate() {
        let val = UniversalDecimalTiered::from_tier_value(4, 2, 1999).unwrap();
        let neg = val.negate().unwrap();
        if let DecimalValueTiered::Tier4(t) = neg.value {
            assert_eq!(t.value, -1999);
        } else {
            panic!("Expected tier 4");
        }
    }
}
