//! Decimal Fixed-Point Multiplication -- Tier Primitives + UGOD Wrapper
//!
//! **OPERATIONS**: Decimal-scaled multiplication at each tier
//! **UGOD**: UniversalDecimalTiered::multiply() with overflow promotion
//!
//! **KEY INVARIANT**: For exact decimal multiplication, dp_result = dp_a + dp_b.
//! The raw product a_val * b_val (without division) is correctly scaled at dp_result.
//! This preserves exactness: 1.5 * 2.5 → 15*25=375 at dp=2 → 3.75 exactly.

use super::decimal_types::*;
use crate::fixed_point::domains::binary_fixed::{I256, I512, I1024};
use crate::fixed_point::domains::decimal_fixed::D256;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// UGOD: UniversalDecimalTiered::multiply
// ============================================================================

impl UniversalDecimalTiered {
    pub fn multiply(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let dp_a = self.decimal_places();
        let dp_b = other.decimal_places();
        let dp_result_wide = dp_a as u16 + dp_b as u16;
        if dp_result_wide > 154 {
            return Err(OverflowDetected::TierOverflow);
        }
        let dp_result = dp_result_wide as u8;

        // Align storage tiers (but NOT decimal places — we want dp_a + dp_b)
        let (a, b) = align_to_common_tier(self, other)?;
        try_mul_exact(&a, &b, dp_result)
    }
}

/// Exact decimal multiplication: product = a_val * b_val, dp_result = dp_a + dp_b.
/// No division by scale — the raw product at doubled dp is the exact result.
fn try_mul_exact(a: &UniversalDecimalTiered, b: &UniversalDecimalTiered, dp_result: u8) -> Result<UniversalDecimalTiered, OverflowDetected> {
    let min_tier = tier_for_decimal_places(dp_result);

    match (&a.value, &b.value) {
        (DecimalValueTiered::Tier1(va), DecimalValueTiered::Tier1(vb)) => {
            let product = (va.value as i32) * (vb.value as i32);
            store_product_i128(product as i128, dp_result, min_tier)
        }
        (DecimalValueTiered::Tier2(va), DecimalValueTiered::Tier2(vb)) => {
            let product = (va.value as i64) * (vb.value as i64);
            store_product_i128(product as i128, dp_result, min_tier)
        }
        (DecimalValueTiered::Tier3(va), DecimalValueTiered::Tier3(vb)) => {
            let product = (va.value as i128) * (vb.value as i128);
            store_product_i128(product, dp_result, min_tier)
        }
        (DecimalValueTiered::Tier4(va), DecimalValueTiered::Tier4(vb)) => {
            // Product may exceed i128 — compute in I256
            let a_wide = I256::from_i128(va.value);
            let b_wide = I256::from_i128(vb.value);
            let product = a_wide * b_wide;
            if product.fits_in_i128() {
                store_product_i128(product.as_i128(), dp_result, min_tier)
            } else {
                // Store in Tier5 (D256)
                Ok(UniversalDecimalTiered {
                    value: DecimalValueTiered::Tier5(DecimalTier5 {
                        value: i256_to_d256(product),
                        decimal_places: dp_result,
                    })
                })
            }
        }
        (DecimalValueTiered::Tier5(va), DecimalValueTiered::Tier5(vb)) => {
            let a256 = d256_to_i256(va.value);
            let b256 = d256_to_i256(vb.value);
            // Product in I512, downscale to I256 would lose precision
            // Store as Tier5 if fits, otherwise Tier6
            let product_512 = I512::from_i256(a256) * I512::from_i256(b256);
            if product_512.fits_in_i256() {
                Ok(UniversalDecimalTiered {
                    value: DecimalValueTiered::Tier5(DecimalTier5 {
                        value: i256_to_d256(product_512.as_i256()),
                        decimal_places: dp_result,
                    })
                })
            } else {
                Ok(UniversalDecimalTiered {
                    value: DecimalValueTiered::Tier6(DecimalTier6 {
                        value: i512_to_d512(product_512.as_i512()),
                        decimal_places: dp_result,
                    })
                })
            }
        }
        (DecimalValueTiered::Tier6(va), DecimalValueTiered::Tier6(vb)) => {
            let a512 = d512_to_i512(va.value);
            let b512 = d512_to_i512(vb.value);
            let product = I1024::from_i512(a512) * I1024::from_i512(b512);
            // Truncate to I512 — this is the highest tier
            Ok(UniversalDecimalTiered {
                value: DecimalValueTiered::Tier6(DecimalTier6 {
                    value: i512_to_d512(product.as_i512()),
                    decimal_places: dp_result,
                })
            })
        }
        _ => Err(OverflowDetected::InvalidInput),
    }
}

/// Store an i128 product at the best tier for the given dp_result.
/// Tries tiers from min_tier upward until the value fits.
fn store_product_i128(product: i128, dp_result: u8, min_tier: u8) -> Result<UniversalDecimalTiered, OverflowDetected> {
    for tier in min_tier..=4 {
        match UniversalDecimalTiered::from_tier_value(tier, dp_result, product) {
            Ok(v) => return Ok(v),
            Err(OverflowDetected::TierOverflow) => continue,
            Err(e) => return Err(e),
        }
    }
    // Doesn't fit in i128 range — promote to Tier5
    Ok(UniversalDecimalTiered {
        value: DecimalValueTiered::Tier5(DecimalTier5 {
            value: D256::from_i128(product),
            decimal_places: dp_result,
        })
    })
}
