//! Decimal Fixed-Point Addition and Subtraction -- Tier Primitives + UGOD Wrappers
//!
//! **OPERATIONS**: checked_add/checked_sub (Tiers 1-4), add/sub (Tiers 5-6)
//! **UGOD**: UniversalDecimalTiered::add() and subtract() with overflow promotion

use super::decimal_types::*;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// TIER 1-4: CHECKED ADD/SUB
// ============================================================================

impl DecimalTier1 {
    pub fn checked_add(self, other: Self) -> Option<Self> {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        self.value.checked_add(other.value).map(|v| Self { value: v, decimal_places: self.decimal_places })
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        self.value.checked_sub(other.value).map(|v| Self { value: v, decimal_places: self.decimal_places })
    }
}

impl DecimalTier2 {
    pub fn checked_add(self, other: Self) -> Option<Self> {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        self.value.checked_add(other.value).map(|v| Self { value: v, decimal_places: self.decimal_places })
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        self.value.checked_sub(other.value).map(|v| Self { value: v, decimal_places: self.decimal_places })
    }
}

impl DecimalTier3 {
    pub fn checked_add(self, other: Self) -> Option<Self> {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        self.value.checked_add(other.value).map(|v| Self { value: v, decimal_places: self.decimal_places })
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        self.value.checked_sub(other.value).map(|v| Self { value: v, decimal_places: self.decimal_places })
    }
}

impl DecimalTier4 {
    pub fn checked_add(self, other: Self) -> Option<Self> {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        self.value.checked_add(other.value).map(|v| Self { value: v, decimal_places: self.decimal_places })
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        self.value.checked_sub(other.value).map(|v| Self { value: v, decimal_places: self.decimal_places })
    }
}

// ============================================================================
// TIER 5-6: NON-FAILING ADD/SUB
// ============================================================================

impl DecimalTier5 {
    pub fn add(self, other: Self) -> Self {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        Self { value: self.value + other.value, decimal_places: self.decimal_places }
    }

    pub fn sub(self, other: Self) -> Self {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        Self { value: self.value - other.value, decimal_places: self.decimal_places }
    }
}

impl DecimalTier6 {
    pub fn add(self, other: Self) -> Self {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        Self { value: self.value + other.value, decimal_places: self.decimal_places }
    }

    pub fn sub(self, other: Self) -> Self {
        debug_assert_eq!(self.decimal_places, other.decimal_places);
        Self { value: self.value - other.value, decimal_places: self.decimal_places }
    }
}

// ============================================================================
// UGOD: UniversalDecimalTiered::add / subtract
// ============================================================================

impl UniversalDecimalTiered {
    pub fn add(&self, other: &Self) -> Result<Self, OverflowDetected> {
        // Align decimal places AND storage tiers before adding.
        // This ensures both operands have the same dp for exact addition.
        let (a, b) = align_decimal_places_and_tier(self, other)?;
        try_add_same_tier(&a, &b)
    }

    pub fn subtract(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let (a, b) = align_decimal_places_and_tier(self, other)?;
        try_sub_same_tier(&a, &b)
    }
}

fn try_add_same_tier(a: &UniversalDecimalTiered, b: &UniversalDecimalTiered) -> Result<UniversalDecimalTiered, OverflowDetected> {
    match (&a.value, &b.value) {
        (DecimalValueTiered::Tier1(va), DecimalValueTiered::Tier1(vb)) => {
            match va.checked_add(*vb) {
                Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier1(r) }),
                None => {
                    let pa = va.to_tier2();
                    let pb = vb.to_tier2();
                    match pa.checked_add(pb) {
                        Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier2(r) }),
                        None => Err(OverflowDetected::TierOverflow),
                    }
                }
            }
        }
        (DecimalValueTiered::Tier2(va), DecimalValueTiered::Tier2(vb)) => {
            match va.checked_add(*vb) {
                Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier2(r) }),
                None => {
                    let pa = va.to_tier3();
                    let pb = vb.to_tier3();
                    match pa.checked_add(pb) {
                        Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier3(r) }),
                        None => Err(OverflowDetected::TierOverflow),
                    }
                }
            }
        }
        (DecimalValueTiered::Tier3(va), DecimalValueTiered::Tier3(vb)) => {
            match va.checked_add(*vb) {
                Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier3(r) }),
                None => {
                    let pa = va.to_tier4();
                    let pb = vb.to_tier4();
                    match pa.checked_add(pb) {
                        Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier4(r) }),
                        None => Err(OverflowDetected::TierOverflow),
                    }
                }
            }
        }
        (DecimalValueTiered::Tier4(va), DecimalValueTiered::Tier4(vb)) => {
            match va.checked_add(*vb) {
                Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier4(r) }),
                None => {
                    let pa = va.to_tier5();
                    let pb = vb.to_tier5();
                    Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier5(pa.add(pb)) })
                }
            }
        }
        (DecimalValueTiered::Tier5(va), DecimalValueTiered::Tier5(vb)) => {
            Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier5(va.add(*vb)) })
        }
        (DecimalValueTiered::Tier6(va), DecimalValueTiered::Tier6(vb)) => {
            Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier6(va.add(*vb)) })
        }
        _ => Err(OverflowDetected::InvalidInput),
    }
}

fn try_sub_same_tier(a: &UniversalDecimalTiered, b: &UniversalDecimalTiered) -> Result<UniversalDecimalTiered, OverflowDetected> {
    match (&a.value, &b.value) {
        (DecimalValueTiered::Tier1(va), DecimalValueTiered::Tier1(vb)) => {
            match va.checked_sub(*vb) {
                Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier1(r) }),
                None => {
                    let pa = va.to_tier2();
                    let pb = vb.to_tier2();
                    match pa.checked_sub(pb) {
                        Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier2(r) }),
                        None => Err(OverflowDetected::TierOverflow),
                    }
                }
            }
        }
        (DecimalValueTiered::Tier2(va), DecimalValueTiered::Tier2(vb)) => {
            match va.checked_sub(*vb) {
                Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier2(r) }),
                None => {
                    let pa = va.to_tier3();
                    let pb = vb.to_tier3();
                    match pa.checked_sub(pb) {
                        Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier3(r) }),
                        None => Err(OverflowDetected::TierOverflow),
                    }
                }
            }
        }
        (DecimalValueTiered::Tier3(va), DecimalValueTiered::Tier3(vb)) => {
            match va.checked_sub(*vb) {
                Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier3(r) }),
                None => {
                    let pa = va.to_tier4();
                    let pb = vb.to_tier4();
                    match pa.checked_sub(pb) {
                        Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier4(r) }),
                        None => Err(OverflowDetected::TierOverflow),
                    }
                }
            }
        }
        (DecimalValueTiered::Tier4(va), DecimalValueTiered::Tier4(vb)) => {
            match va.checked_sub(*vb) {
                Some(r) => Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier4(r) }),
                None => {
                    let pa = va.to_tier5();
                    let pb = vb.to_tier5();
                    Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier5(pa.sub(pb)) })
                }
            }
        }
        (DecimalValueTiered::Tier5(va), DecimalValueTiered::Tier5(vb)) => {
            Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier5(va.sub(*vb)) })
        }
        (DecimalValueTiered::Tier6(va), DecimalValueTiered::Tier6(vb)) => {
            Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier6(va.sub(*vb)) })
        }
        _ => Err(OverflowDetected::InvalidInput),
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal_tier1_add() {
        let a = DecimalTier1 { value: 100, decimal_places: 2 };
        let b = DecimalTier1 { value: 250, decimal_places: 2 };
        let result = a.checked_add(b).unwrap();
        assert_eq!(result.value, 350);
        assert_eq!(result.decimal_places, 2);
    }

    #[test]
    fn test_decimal_tier1_overflow_promotes() {
        let a = UniversalDecimalTiered::from_tier_value(1, 2, 30000).unwrap();
        let b = UniversalDecimalTiered::from_tier_value(1, 2, 30000).unwrap();
        let result = a.add(&b).unwrap();
        assert_eq!(result.current_tier(), 2);
    }

    #[test]
    fn test_decimal_tier4_add() {
        let a = UniversalDecimalTiered::from_tier_value(4, 2, 1999).unwrap();
        let b = UniversalDecimalTiered::from_tier_value(4, 2, 500).unwrap();
        let result = a.add(&b).unwrap();
        assert_eq!(result.current_tier(), 4);
        if let DecimalValueTiered::Tier4(t) = result.value {
            assert_eq!(t.value, 2499);
        } else {
            panic!("Expected tier 4");
        }
    }

    #[test]
    fn test_cross_tier_add() {
        let a = UniversalDecimalTiered::from_tier_value(1, 2, 100).unwrap();
        let b = UniversalDecimalTiered::from_tier_value(3, 2, 200).unwrap();
        let result = a.add(&b).unwrap();
        assert_eq!(result.current_tier(), 3);
        if let DecimalValueTiered::Tier3(t) = result.value {
            assert_eq!(t.value, 300);
        } else {
            panic!("Expected tier 3");
        }
    }
}
