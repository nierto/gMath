//! Binary Fixed-Point Addition and Subtraction -- Tier Primitives + UGOD Wrappers
//!
//! **OPERATIONS**: checked_add/checked_sub (Tiers 1-3), add/sub (Tiers 4-6)
//! **UGOD**: UniversalBinaryFixed::add() and subtract() with overflow promotion

use super::binary_types::*;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// TIER 1: Q16.16
// ============================================================================

impl BinaryTier1 {
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        self.value.checked_add(other.value).map(|v| Self { value: v })
    }

    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        self.value.checked_sub(other.value).map(|v| Self { value: v })
    }
}

// ============================================================================
// TIER 2: Q32.32
// ============================================================================

impl BinaryTier2 {
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        self.value.checked_add(other.value).map(|v| Self { value: v })
    }

    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        self.value.checked_sub(other.value).map(|v| Self { value: v })
    }
}

// ============================================================================
// TIER 3: Q64.64
// ============================================================================

impl BinaryTier3 {
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        self.value.checked_add(other.value).map(|v| Self { value: v })
    }

    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        self.value.checked_sub(other.value).map(|v| Self { value: v })
    }
}

// ============================================================================
// TIER 4: Q128.128
// ============================================================================

impl BinaryTier4 {
    pub fn add(&self, other: &Self) -> Self {
        Self { value: self.value + other.value }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { value: self.value - other.value }
    }
}

// ============================================================================
// TIER 5: Q256.256
// ============================================================================

impl BinaryTier5 {
    pub fn add(&self, other: &Self) -> Self {
        Self { value: self.value + other.value }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { value: self.value - other.value }
    }
}

// ============================================================================
// TIER 6: Q512.512
// ============================================================================

impl BinaryTier6 {
    pub fn add(&self, other: &Self) -> Self {
        Self { value: self.value + other.value }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { value: self.value - other.value }
    }
}

// ============================================================================
// UGOD: UniversalBinaryFixed::add / subtract
// ============================================================================

impl UniversalBinaryFixed {
    pub fn add(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let (a, b) = self.align_to_common_tier(other);
        match (&a.value, &b.value) {
            (BinaryValue::Tier1(x), BinaryValue::Tier1(y)) => {
                match x.checked_add(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier1(r), current_tier: 1 }),
                    None => {
                        let x2 = x.to_tier2();
                        let y2 = y.to_tier2();
                        Ok(Self { value: BinaryValue::Tier2(BinaryTier2::from_raw(x2.raw() + y2.raw())), current_tier: 2 })
                    }
                }
            }
            (BinaryValue::Tier2(x), BinaryValue::Tier2(y)) => {
                match x.checked_add(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier2(r), current_tier: 2 }),
                    None => {
                        let x3 = x.to_tier3();
                        let y3 = y.to_tier3();
                        Ok(Self { value: BinaryValue::Tier3(BinaryTier3::from_raw(x3.raw() + y3.raw())), current_tier: 3 })
                    }
                }
            }
            (BinaryValue::Tier3(x), BinaryValue::Tier3(y)) => {
                match x.checked_add(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier3(r), current_tier: 3 }),
                    None => {
                        let x4 = x.to_tier4();
                        let y4 = y.to_tier4();
                        Ok(Self { value: BinaryValue::Tier4(x4.add(&y4)), current_tier: 4 })
                    }
                }
            }
            (BinaryValue::Tier4(x), BinaryValue::Tier4(y)) => {
                Ok(Self { value: BinaryValue::Tier4(x.add(y)), current_tier: 4 })
            }
            (BinaryValue::Tier5(x), BinaryValue::Tier5(y)) => {
                Ok(Self { value: BinaryValue::Tier5(x.add(y)), current_tier: 5 })
            }
            (BinaryValue::Tier6(x), BinaryValue::Tier6(y)) => {
                Ok(Self { value: BinaryValue::Tier6(x.add(y)), current_tier: 6 })
            }
            _ => Err(OverflowDetected::InvalidInput),
        }
    }

    pub fn subtract(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let (a, b) = self.align_to_common_tier(other);
        match (&a.value, &b.value) {
            (BinaryValue::Tier1(x), BinaryValue::Tier1(y)) => {
                match x.checked_sub(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier1(r), current_tier: 1 }),
                    None => {
                        let x2 = x.to_tier2();
                        let y2 = y.to_tier2();
                        Ok(Self { value: BinaryValue::Tier2(BinaryTier2::from_raw(x2.raw() - y2.raw())), current_tier: 2 })
                    }
                }
            }
            (BinaryValue::Tier2(x), BinaryValue::Tier2(y)) => {
                match x.checked_sub(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier2(r), current_tier: 2 }),
                    None => {
                        let x3 = x.to_tier3();
                        let y3 = y.to_tier3();
                        Ok(Self { value: BinaryValue::Tier3(BinaryTier3::from_raw(x3.raw() - y3.raw())), current_tier: 3 })
                    }
                }
            }
            (BinaryValue::Tier3(x), BinaryValue::Tier3(y)) => {
                match x.checked_sub(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier3(r), current_tier: 3 }),
                    None => {
                        let x4 = x.to_tier4();
                        let y4 = y.to_tier4();
                        Ok(Self { value: BinaryValue::Tier4(x4.sub(&y4)), current_tier: 4 })
                    }
                }
            }
            (BinaryValue::Tier4(x), BinaryValue::Tier4(y)) => {
                Ok(Self { value: BinaryValue::Tier4(x.sub(y)), current_tier: 4 })
            }
            (BinaryValue::Tier5(x), BinaryValue::Tier5(y)) => {
                Ok(Self { value: BinaryValue::Tier5(x.sub(y)), current_tier: 5 })
            }
            (BinaryValue::Tier6(x), BinaryValue::Tier6(y)) => {
                Ok(Self { value: BinaryValue::Tier6(x.sub(y)), current_tier: 6 })
            }
            _ => Err(OverflowDetected::InvalidInput),
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
    fn test_ugod_overflow_promotion() {
        let big = UniversalBinaryFixed {
            value: BinaryValue::Tier1(BinaryTier1::from_raw(i32::MAX - 1)),
            current_tier: 1,
        };
        let one = UniversalBinaryFixed {
            value: BinaryValue::Tier1(BinaryTier1::from_raw(SCALE_Q16_16)),
            current_tier: 1,
        };
        let result = big.add(&one).unwrap();
        assert_eq!(result.current_tier(), 2);
    }

    #[test]
    fn test_cross_tier_alignment() {
        let t1 = UniversalBinaryFixed {
            value: BinaryValue::Tier1(BinaryTier1::one()),
            current_tier: 1,
        };
        let t3 = UniversalBinaryFixed {
            value: BinaryValue::Tier3(BinaryTier3::one()),
            current_tier: 3,
        };
        let result = t1.add(&t3).unwrap();
        assert_eq!(result.current_tier(), 3);
    }
}
