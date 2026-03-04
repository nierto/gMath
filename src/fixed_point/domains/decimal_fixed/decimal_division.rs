//! Decimal Fixed-Point Division -- Tier Primitives + UGOD Wrapper
//!
//! **OPERATIONS**: Decimal-scaled division at each tier
//! **UGOD**: UniversalDecimalTiered::divide() with overflow promotion

use super::decimal_types::*;
use crate::fixed_point::domains::binary_fixed::{I256, I512, I1024};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// HELPER: pow10
// ============================================================================

fn pow10_i128(dp: u8) -> Option<i128> {
    if dp > 38 { return None; }
    let mut result: i128 = 1;
    for _ in 0..dp {
        result = result.checked_mul(10)?;
    }
    Some(result)
}

// ============================================================================
// UGOD: UniversalDecimalTiered::divide
// ============================================================================

impl UniversalDecimalTiered {
    pub fn divide(&self, other: &Self) -> Result<Self, OverflowDetected> {
        // Align decimal places AND storage tiers before dividing.
        // This ensures both operands have the same dp, so the division
        // formula (a_val * 10^dp) / b_val correctly produces a result at dp.
        let (a, b) = align_decimal_places_and_tier(self, other)?;
        try_div_same_tier(&a, &b)
    }
}

fn try_div_same_tier(a: &UniversalDecimalTiered, b: &UniversalDecimalTiered) -> Result<UniversalDecimalTiered, OverflowDetected> {
    match (&a.value, &b.value) {
        (DecimalValueTiered::Tier1(va), DecimalValueTiered::Tier1(vb)) => {
            if vb.value == 0 { return Err(OverflowDetected::DivisionByZero); }
            let dp = va.decimal_places;
            let scale = pow10_i128(dp).ok_or(OverflowDetected::TierOverflow)? as i32;
            let scaled_a = (va.value as i32).checked_mul(scale).ok_or(OverflowDetected::TierOverflow)?;
            let remainder = scaled_a % (vb.value as i32);
            // Inexact division: fall through to rational for exact result
            if remainder != 0 { return Err(OverflowDetected::PrecisionLoss); }
            let quotient = scaled_a / (vb.value as i32);
            if quotient >= i16::MIN as i32 && quotient <= i16::MAX as i32 {
                Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier1(DecimalTier1 { value: quotient as i16, decimal_places: dp }) })
            } else {
                let pa = va.to_tier2();
                let pb = vb.to_tier2();
                try_div_same_tier(
                    &UniversalDecimalTiered { value: DecimalValueTiered::Tier2(pa) },
                    &UniversalDecimalTiered { value: DecimalValueTiered::Tier2(pb) },
                )
            }
        }
        (DecimalValueTiered::Tier2(va), DecimalValueTiered::Tier2(vb)) => {
            if vb.value == 0 { return Err(OverflowDetected::DivisionByZero); }
            let dp = va.decimal_places;
            let scale = pow10_i128(dp).ok_or(OverflowDetected::TierOverflow)? as i64;
            let scaled_a = (va.value as i64).checked_mul(scale).ok_or(OverflowDetected::TierOverflow)?;
            let remainder = scaled_a % (vb.value as i64);
            if remainder != 0 { return Err(OverflowDetected::PrecisionLoss); }
            let quotient = scaled_a / (vb.value as i64);
            if quotient >= i32::MIN as i64 && quotient <= i32::MAX as i64 {
                Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier2(DecimalTier2 { value: quotient as i32, decimal_places: dp }) })
            } else {
                let pa = va.to_tier3();
                let pb = vb.to_tier3();
                try_div_same_tier(
                    &UniversalDecimalTiered { value: DecimalValueTiered::Tier3(pa) },
                    &UniversalDecimalTiered { value: DecimalValueTiered::Tier3(pb) },
                )
            }
        }
        (DecimalValueTiered::Tier3(va), DecimalValueTiered::Tier3(vb)) => {
            if vb.value == 0 { return Err(OverflowDetected::DivisionByZero); }
            let dp = va.decimal_places;
            let scale = pow10_i128(dp).ok_or(OverflowDetected::TierOverflow)?;
            let scaled_a = (va.value as i128).checked_mul(scale).ok_or(OverflowDetected::TierOverflow)?;
            let remainder = scaled_a % (vb.value as i128);
            if remainder != 0 { return Err(OverflowDetected::PrecisionLoss); }
            let quotient = scaled_a / (vb.value as i128);
            if quotient >= i64::MIN as i128 && quotient <= i64::MAX as i128 {
                Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier3(DecimalTier3 { value: quotient as i64, decimal_places: dp }) })
            } else {
                let pa = va.to_tier4();
                let pb = vb.to_tier4();
                try_div_same_tier(
                    &UniversalDecimalTiered { value: DecimalValueTiered::Tier4(pa) },
                    &UniversalDecimalTiered { value: DecimalValueTiered::Tier4(pb) },
                )
            }
        }
        (DecimalValueTiered::Tier4(va), DecimalValueTiered::Tier4(vb)) => {
            if vb.value == 0 { return Err(OverflowDetected::DivisionByZero); }
            let dp = va.decimal_places;
            let scale = pow10_i128(dp).ok_or(OverflowDetected::TierOverflow)?;
            let a_wide = I256::from_i128(va.value) * I256::from_i128(scale);
            let b_wide = I256::from_i128(vb.value);
            let (quotient, remainder) = crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(a_wide, b_wide);
            if !remainder.is_zero() { return Err(OverflowDetected::PrecisionLoss); }
            if quotient.fits_in_i128() {
                Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier4(DecimalTier4 { value: quotient.as_i128(), decimal_places: dp }) })
            } else {
                let pa = va.to_tier5();
                let pb = vb.to_tier5();
                let a256 = d256_to_i256(pa.value);
                let b256 = d256_to_i256(pb.value);
                let s256 = I256::from_i128(scale);
                let scaled = I512::from_i256(a256) * I512::from_i256(s256);
                let b512 = I512::from_i256(b256);
                let (q512, r512) = crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(scaled, b512);
                if !r512.is_zero() { return Err(OverflowDetected::PrecisionLoss); }
                Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier5(DecimalTier5 { value: i256_to_d256(q512.as_i256()), decimal_places: dp }) })
            }
        }
        (DecimalValueTiered::Tier5(va), DecimalValueTiered::Tier5(vb)) => {
            let b256 = d256_to_i256(vb.value);
            if b256.is_zero() { return Err(OverflowDetected::DivisionByZero); }
            let dp = va.decimal_places;
            let scale = pow10_i128(dp).ok_or(OverflowDetected::TierOverflow)?;
            let a256 = d256_to_i256(va.value);
            let s256 = I256::from_i128(scale);
            let scaled = I512::from_i256(a256) * I512::from_i256(s256);
            let b512 = I512::from_i256(b256);
            let (result, rem) = crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(scaled, b512);
            if !rem.is_zero() { return Err(OverflowDetected::PrecisionLoss); }
            Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier5(DecimalTier5 { value: i256_to_d256(result.as_i256()), decimal_places: dp }) })
        }
        (DecimalValueTiered::Tier6(va), DecimalValueTiered::Tier6(vb)) => {
            let b512 = d512_to_i512(vb.value);
            if b512.is_zero() { return Err(OverflowDetected::DivisionByZero); }
            let dp = va.decimal_places;
            let scale = pow10_i128(dp).ok_or(OverflowDetected::TierOverflow)?;
            let a512 = d512_to_i512(va.value);
            let s512 = I512::from_i128(scale);
            let scaled = I1024::from_i512(a512) * I1024::from_i512(s512);
            let b1024 = I1024::from_i512(b512);
            let result = (scaled / b1024).as_i512();
            // Tier 6: best-effort (no wider type available), accept truncated result
            Ok(UniversalDecimalTiered { value: DecimalValueTiered::Tier6(DecimalTier6 { value: i512_to_d512(result), decimal_places: dp }) })
        }
        _ => Err(OverflowDetected::InvalidInput),
    }
}
