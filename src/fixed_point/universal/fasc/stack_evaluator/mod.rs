//! Stack-based evaluator with UGOD integration
//!
//! **MISSION**: Evaluate lazy expressions on thread-local stack without heap allocation
//! **ARCHITECTURE**: Fixed-size workspace with domain routing and overflow delegation
//! **OPTIMIZATION**: Cache-friendly sequential memory access with SIMD potential

pub(crate) mod compute;
pub(crate) mod conversion;
pub(crate) mod domain;
pub(crate) mod formatting;
mod parsing;
mod arithmetic;
mod transcendentals;
#[cfg(test)]
mod tests;

#[allow(unused_imports)]
use super::lazy_expr::{LazyExpr, ConstantId, StackRef};
use crate::fixed_point::universal::ugod::DomainType;
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, OverflowDetected};
use crate::deployment_profiles::DeploymentProfile;
use crate::fixed_point::universal::tier_types::CompactShadow;
#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;

use core::cell::RefCell;
use core::fmt::{self, Display};

// Re-export sub-module functions used by sibling modules and tests
#[allow(unused_imports)]
pub(crate) use conversion::to_binary_storage;
#[allow(unused_imports)]
pub(crate) use compute::{
    downscale_to_storage, upscale_to_compute, sqrt_at_compute_tier, exp_at_compute_tier,
    sinhcosh_at_compute_tier,
    compute_add, compute_subtract, compute_negate, compute_multiply, compute_divide,
    compute_halve, compute_is_zero, compute_is_negative,
};

// ============================================================================
// PROFILE-SPECIFIC STORAGE TYPES (TIER N+1 PRECISION PRESERVATION)
// ============================================================================

/// Profile-specific binary storage type for zero precision loss
///
/// **ARCHITECTURE**: Each profile gets native storage that matches its precision tier
/// **MEMORY COST**:
/// - Embedded: 16 bytes (i128) - Q64.64 storage
/// - Balanced: 32 bytes (I256) - Q128.128 storage
/// - Scientific: 64 bytes (I512) - Q256.256 storage
///
/// **PRECISION GUARANTEE**: Tier N+1 computation results stored without downconversion
#[cfg(table_format = "q256_256")]
pub type BinaryStorage = I512;  // Q256.256: 77 decimals

#[cfg(table_format = "q128_128")]
pub type BinaryStorage = I256;  // Q128.128: 38 decimals

#[cfg(table_format = "q64_64")]
pub type BinaryStorage = i128;  // Q64.64: 19 decimals

#[cfg(table_format = "q32_32")]
pub type BinaryStorage = i64;   // Q32.32: 9 decimals

#[cfg(table_format = "q16_16")]
pub type BinaryStorage = i32;   // Q16.16: 4 decimals

// ============================================================================
// COMPUTE-TIER STORAGE TYPE (TIER N+1 CHAIN PERSISTENCE)
// ============================================================================

/// Compute-tier storage for transcendental chain persistence
///
/// **PURPOSE**: Keep intermediate results at tier N+1 between consecutive
/// transcendentals (e.g., sin(ln(exp(x)))) to avoid precision loss from
/// repeated upscale/downscale cycles.
///
/// **PROFILE MAPPING**:
/// - Embedded: I256 (Q128.128 compute for Q64.64 storage)
/// - Balanced: I512 (Q256.256 compute for Q128.128 storage)
/// - Scientific: I1024 (Q512.512 compute for Q256.256 storage)
#[cfg(table_format = "q256_256")]
pub type ComputeStorage = I1024;  // Q512.512

#[cfg(table_format = "q128_128")]
pub type ComputeStorage = I512;   // Q256.256

#[cfg(table_format = "q64_64")]
pub type ComputeStorage = I256;   // Q128.128

#[cfg(table_format = "q32_32")]
pub type ComputeStorage = i128;   // Q64.64 (tier N+1 for Q32.32)

#[cfg(table_format = "q16_16")]
pub type ComputeStorage = i64;    // Q32.32 (tier N+1 for Q16.16)

/// Maximum decimal places before promoting Decimal to BinaryCompute.
/// Matches each profile's meaningful decimal precision.
/// Beyond this threshold, the scaled integer exceeds BinaryStorage range,
/// so we promote to binary fixed-point (which has FIXED fractional bits, no dp growth).
#[cfg(table_format = "q16_16")]
const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = crate::fixed_point::frac_config::MAX_DECIMAL_DIGITS as u16;
#[cfg(table_format = "q32_32")]
const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 9;
#[cfg(table_format = "q64_64")]
const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 18;
#[cfg(table_format = "q128_128")]
const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 38;
#[cfg(table_format = "q256_256")]
const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 76;

// ============================================================================
// DECIMAL → BINARY STORAGE HELPER (used by as_binary_storage)
// ============================================================================

/// Convert a Decimal (dp, scaled) to the profile's BinaryStorage type via
/// `(scaled << frac_bits) / 10^dp` with round-to-nearest.
///
/// Mirrors `StackEvaluator::to_binary_storage` but is callable without `&mut self`.
pub(crate) fn decimal_to_binary_storage(dp: u8, scaled: BinaryStorage) -> Result<BinaryStorage, OverflowDetected> {
    #[cfg(not(table_format = "q256_256"))]
    use formatting::pow10_i256;
    #[cfg(table_format = "q256_256")]
    use formatting::pow10_i512;

    // 0.5.0 rounding unification: the RESULT domain is binary, so this
    // coercion rounds nearest, ties toward +∞ on every profile (was:
    // truncation on four arms, add-half on q16_16 — per-profile drift).
    #[cfg(table_format = "q256_256")]
    {
        let ten_pow = I1024::from_i512(pow10_i512(dp));
        let num = I1024::from_i512(scaled) << 256;
        let mut q = num / ten_pow;
        let rem = num % ten_pow;
        let rem_neg = (rem.words[15] as i64) < 0;
        let rem_abs = if rem_neg { -rem } else { rem };
        let positive = (num.words[15] as i64) >= 0;
        let rem2 = rem_abs + rem_abs;
        if if positive { rem2 >= ten_pow } else { rem2 > ten_pow } {
            q = if positive { q + I1024::from_i128(1) } else { q - I1024::from_i128(1) };
        }
        Ok(q.as_i512())
    }
    #[cfg(table_format = "q128_128")]
    {
        let ten_pow = pow10_i256(dp);
        let num = I512::from_i256(scaled) << 128;
        let den = I512::from_i256(ten_pow);
        let (mut q, rem) =
            crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(num, den);
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let positive = !num.is_negative();
        let rem2 = rem_abs + rem_abs;
        if if positive { rem2 >= den } else { rem2 > den } {
            q = if positive { q + I512::from_i128(1) } else { q - I512::from_i128(1) };
        }
        Ok(q.as_i256())
    }
    #[cfg(table_format = "q64_64")]
    {
        let ten_pow = pow10_i256(dp);
        let num = I256::from_i128(scaled) << 64;
        let (mut q, rem) =
            crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(num, ten_pow);
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let positive = !num.is_negative();
        let rem2 = rem_abs + rem_abs;
        if if positive { rem2 >= ten_pow } else { rem2 > ten_pow } {
            q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
        }
        Ok(q.as_i128())
    }
    #[cfg(table_format = "q32_32")]
    {
        let ten_pow = pow10_i256(dp);
        let num = I256::from_i128(scaled as i128) << 32;
        let (mut q, rem) =
            crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(num, ten_pow);
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let positive = !num.is_negative();
        let rem2 = rem_abs + rem_abs;
        if if positive { rem2 >= ten_pow } else { rem2 > ten_pow } {
            q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
        }
        Ok(q.as_i128() as i64)
    }
    #[cfg(table_format = "q16_16")]
    {
        use crate::fixed_point::frac_config;
        let ten_pow = pow10_i256(dp);
        let num = I256::from_i128(scaled as i128) << (frac_config::FRAC_BITS as usize);
        let (mut q, rem) =
            crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(num, ten_pow);
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let positive = !num.is_negative();
        let rem2 = rem_abs + rem_abs;
        if if positive { rem2 >= ten_pow } else { rem2 > ten_pow } {
            q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
        }
        Ok(q.as_i128() as i32)
    }
}

/// Public wrapper for use from sibling modules (transcendentals.rs).
pub(crate) fn decimal_compute_to_binary_storage_pub(val: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    decimal_compute_to_binary_storage(val)
}

/// Convert a DecimalCompute value directly to profile BinaryStorage via
/// `val × 2^frac_bits / 10^compute_dp` — no intermediate Decimal materialization.
///
/// This is lossless (within rounding) because we do one big-integer division.
fn decimal_compute_to_binary_storage(val: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    use crate::fixed_point::domains::decimal_fixed::transcendental::DECIMAL_COMPUTE_DP;

    #[cfg(table_format = "q64_64")]
    {
        // val is I256 at decimal compute dp=38. Target: Q64.64 i128.
        // result = val * 2^64 / 10^38 via I512 intermediate
        let num = I512::from_i256(val) << 64usize;
        let mut den = I512::from_i128(1);
        let ten = I512::from_i128(10);
        for _ in 0..DECIMAL_COMPUTE_DP { den = den * ten; }
        let quot = num / den;
        if !quot.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(quot.as_i128())
    }
    #[cfg(table_format = "q128_128")]
    {
        // val is I512 at dp=77. Target: Q128.128 I256.
        // result = val * 2^128 / 10^77 via I1024 intermediate
        let num = I1024::from_i512(val) << 128usize;
        let mut den = I1024::from_i128(1);
        let ten = I1024::from_i128(10);
        for _ in 0..DECIMAL_COMPUTE_DP { den = den * ten; }
        let quot = num / den;
        if !quot.fits_in_i256() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(quot.as_i256())
    }
    #[cfg(table_format = "q256_256")]
    {
        // val is I1024 at dp=154. Target: Q256.256 I512.
        // Use I2048 intermediate.
        use crate::fixed_point::I2048;
        use crate::fixed_point::domains::binary_fixed::i2048::i2048_div;
        let num = I2048::from_i1024(val) << 256usize;
        let mut pow = I1024::from_i128(1);
        let ten = I1024::from_i128(10);
        for _ in 0..DECIMAL_COMPUTE_DP { pow = pow * ten; }
        let den = I2048::from_i1024(pow);
        let quot = i2048_div(num, den);
        // Check fit in I512: upper words (8-31) must be sign extension of word[7]
        let sign = (quot.words[7] as i64) < 0;
        let expected = if sign { u64::MAX } else { 0 };
        for i in 8..32 {
            if quot.words[i] != expected {
                return Err(OverflowDetected::TierOverflow);
            }
        }
        Ok(I512::from_words([
            quot.words[0], quot.words[1], quot.words[2], quot.words[3],
            quot.words[4], quot.words[5], quot.words[6], quot.words[7],
        ]))
    }
    #[cfg(table_format = "q32_32")]
    {
        // val is i128 at dp=19. Target: Q32.32 i64.
        // result = val * 2^32 / 10^19 via I256 intermediate
        let num = I256::from_i128(val) << 32usize;
        let mut den = I256::from_i128(1);
        let ten = I256::from_i128(10);
        for _ in 0..DECIMAL_COMPUTE_DP { den = den * ten; }
        let quot = num / den;
        let q_i128 = quot.as_i128();
        if q_i128 > i64::MAX as i128 || q_i128 < i64::MIN as i128 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(q_i128 as i64)
    }
    #[cfg(table_format = "q16_16")]
    {
        use crate::fixed_point::frac_config;
        // val is i64 at dp=9. Target: Q16.16 i32.
        let num = (val as i128) << (frac_config::FRAC_BITS as usize);
        let mut den: i128 = 1;
        for _ in 0..DECIMAL_COMPUTE_DP { den *= 10; }
        let quot = num / den;
        if quot > i32::MAX as i128 || quot < i32::MIN as i128 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(quot as i32)
    }
}

// ============================================================================
// STACK VALUE — UNIFIED DOMAIN REPRESENTATION
// ============================================================================

/// Stack value - unified representation for all domains
///
/// **ARCHITECTURE**: Tagged union for domain-specific values
/// **MEMORY**: Stack-allocated, no heap pointers
/// **CONVERSION**: Lazy conversion between domains as needed
/// **PRECISION**: Profile-specific storage eliminates downconversion loss
#[derive(Debug, Clone)]
pub enum StackValue {
    /// Binary fixed-point value (tier, mantissa, shadow)
    /// **STORAGE**: Profile-specific (i128 | I256 | I512) - zero precision loss
    /// **SHADOW**: Exact rational representation for precision preservation
    Binary(u8, BinaryStorage, CompactShadow),

    /// Binary compute-tier value for transcendental chain persistence
    /// **STORAGE**: One tier above storage (I256 | I512 | I1024)
    /// **PURPOSE**: Keep intermediate results "hot" at compute precision between
    /// consecutive transcendentals. Materialized to Binary on non-transcendental boundary.
    BinaryCompute(u8, ComputeStorage, CompactShadow),

    /// Decimal fixed-point value (decimals, scaled_value, shadow)
    /// **STORAGE**: Profile-specific (i128 | I256 | I512) - matches binary precision
    /// **SHADOW**: Exact rational representation for precision preservation
    Decimal(u8, BinaryStorage, CompactShadow),

    /// Decimal compute-tier value for transcendental chain persistence.
    ///
    /// **STORAGE**: ComputeStorage at `DECIMAL_COMPUTE_DP` scaling (10^-dp).
    /// **PURPOSE**: Decimal equivalent of BinaryCompute — keeps intermediate
    /// transcendental results at compute-tier precision between chained ops.
    /// Materialized to `Decimal` at non-transcendental boundaries (display, arithmetic
    /// with non-decimal operands, etc.) with dp chosen to guarantee 0 storage ULP.
    ///
    /// **FIELDS**: `(storage_tier, compute_value, shadow)` where `compute_value`
    /// is scaled by `10^DECIMAL_COMPUTE_DP` (per-profile: 9, 19, 38, 77, 154).
    DecimalCompute(u8, ComputeStorage, CompactShadow),

    /// Balanced ternary value (precision_tier, trit_value, shadow)
    /// **STORAGE**: Profile-specific (i128 | I256 | I512) - matches binary precision
    /// **SHADOW**: Exact rational representation for precision preservation
    Ternary(u8, BinaryStorage, CompactShadow),

    /// Symbolic rational value (owned for stack storage)
    /// Symbolic IS exact — no shadow needed
    Symbolic(RationalNumber),

    /// Error state
    Error(OverflowDetected),
}

impl StackValue {
    /// Get domain type for routing decisions
    pub fn domain_type(&self) -> Option<DomainType> {
        match self {
            StackValue::Binary(..) => Some(DomainType::Binary),
            StackValue::BinaryCompute(..) => Some(DomainType::Binary),
            StackValue::Decimal(..) => Some(DomainType::Decimal),
            StackValue::DecimalCompute(..) => Some(DomainType::Decimal),
            StackValue::Ternary(..) => Some(DomainType::Ternary),
            StackValue::Symbolic(_) => Some(DomainType::Symbolic),
            StackValue::Error(_) => None,
        }
    }

    /// Check if value represents an error state
    pub fn is_error(&self) -> bool {
        matches!(self, StackValue::Error(_))
    }

    /// Get the compact shadow for precision preservation
    pub fn shadow(&self) -> CompactShadow {
        match self {
            StackValue::Binary(_, _, s) => s.clone(),
            StackValue::BinaryCompute(_, _, s) => s.clone(),
            StackValue::Decimal(_, _, s) => s.clone(),
            StackValue::DecimalCompute(_, _, s) => s.clone(),
            StackValue::Ternary(_, _, s) => s.clone(),
            StackValue::Symbolic(_) => CompactShadow::None, // Symbolic IS exact
            StackValue::Error(_) => CompactShadow::None,
        }
    }

    /// Convert to rational for cross-domain operations
    ///
    /// **CRITICAL**: Uses full-precision conversion — no i128 truncation.
    /// Binary Q-format values are divided by 2^frac_bits to get the true rational.
    /// Decimal values are divided by 10^decimals.
    /// Ternary values are divided by 3^frac_trits.
    pub fn to_rational(&self) -> Result<RationalNumber, OverflowDetected> {
        match self {
            StackValue::Symbolic(s) => Ok(s.clone()),
            StackValue::BinaryCompute(tier, value, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                // Materialize to storage tier first, then convert
                let storage_val = downscale_to_storage(*value)?;
                let materialized = StackValue::Binary(*tier, storage_val, shadow.clone());
                materialized.to_rational()
            }
            StackValue::DecimalCompute(_tier, value, ref shadow) => {
                // Shadow fast path
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                // DecimalCompute holds raw × 10^-COMPUTE_DP. Materialize at
                // the highest dp that fits (adaptive — 0.5.0 item 0d fix).
                let (target_dp, storage_val) =
                    StackEvaluator::materialize_decimal_compute_adaptive(*value)?;
                let materialized = StackValue::Decimal(target_dp, storage_val, shadow.clone());
                materialized.to_rational()
            }
            StackValue::Binary(tier, value, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                // Binary Q-format: rational = value / 2^frac_bits
                // Tier mapping depends on profile — the storage tier IS the profile's max tier.
                let frac_bits: u32 = match tier {
                    1 => {
                        // Tier 1: Q16.16 on Realtime, raw integer on others
                        #[cfg(table_format = "q16_16")]
                        { 16 }
                        #[cfg(not(table_format = "q16_16"))]
                        { 0 }
                    }
                    2 => 32,   // Q32.32
                    3 => 64,   // Q64.64
                    4 => 128,  // Q128.128
                    5 => 256,  // Q256.256
                    6 => 512,  // Q512.512
                    _ => 64,   // Default
                };

                if frac_bits == 0 {
                    let value_i128 = conversion::binary_storage_to_i128(value);
                    return Ok(RationalNumber::new(value_i128, 1));
                }

                #[cfg(table_format = "q256_256")]
                {
                    if value.is_zero() {
                        return Ok(RationalNumber::new(0, 1));
                    }
                    let tz = conversion::trailing_zeros_i512(value);
                    let tz = if tz > frac_bits { frac_bits } else { tz };
                    let reduced_num = *value >> (tz as usize);
                    let remaining_frac = frac_bits - tz;
                    if remaining_frac == 0 && reduced_num.fits_in_i128() {
                        return Ok(RationalNumber::new(reduced_num.as_i128(), 1));
                    } else if remaining_frac <= 127 && reduced_num.fits_in_i128() {
                        return Ok(RationalNumber::new(reduced_num.as_i128(), 1u128 << remaining_frac));
                    } else {
                        let denom = I512::from_i128(1) << (remaining_frac as usize);
                        return Ok(RationalNumber::from_i512_pair(reduced_num, denom));
                    }
                }

                #[cfg(table_format = "q128_128")]
                {
                    if value.is_zero() {
                        return Ok(RationalNumber::new(0, 1));
                    }
                    let tz = conversion::trailing_zeros_i256(value);
                    let tz = if tz > frac_bits { frac_bits } else { tz };
                    let reduced_num = *value >> tz;
                    let remaining_frac = frac_bits - tz;
                    if remaining_frac == 0 && reduced_num.fits_in_i128() {
                        return Ok(RationalNumber::new(reduced_num.as_i128(), 1));
                    } else if remaining_frac <= 127 && reduced_num.fits_in_i128() {
                        return Ok(RationalNumber::new(reduced_num.as_i128(), 1u128 << remaining_frac));
                    } else {
                        let denom = I256::from_i128(1) << (remaining_frac as usize);
                        return Ok(RationalNumber::from_i256_pair(reduced_num, denom));
                    }
                }

                #[cfg(table_format = "q64_64")]
                {
                    if *value == 0 {
                        return Ok(RationalNumber::new(0, 1));
                    }
                    let trailing = (*value as u128).trailing_zeros().min(frac_bits);
                    let reduced_num = *value >> trailing;
                    let reduced_den = 1u128 << (frac_bits - trailing);
                    return Ok(RationalNumber::new(reduced_num, reduced_den));
                }

                #[cfg(table_format = "q32_32")]
                {
                    if *value == 0 {
                        return Ok(RationalNumber::new(0, 1));
                    }
                    let trailing = (*value as u64).trailing_zeros().min(frac_bits);
                    let reduced_num = (*value >> trailing) as i128;
                    let reduced_den = 1u128 << (frac_bits - trailing);
                    return Ok(RationalNumber::new(reduced_num, reduced_den));
                }

                #[cfg(table_format = "q16_16")]
                {
                    if *value == 0 {
                        return Ok(RationalNumber::new(0, 1));
                    }
                    let trailing = (*value as u32).trailing_zeros().min(frac_bits);
                    let reduced_num = (*value >> trailing) as i128;
                    let reduced_den = 1u128 << (frac_bits - trailing);
                    return Ok(RationalNumber::new(reduced_num, reduced_den));
                }

            }
            StackValue::Decimal(decimals, scaled, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                #[cfg(table_format = "q256_256")]
                {
                    let fits = if scaled.fits_in_i256() { let v = scaled.as_i256(); if v.fits_in_i128() { Some(v.as_i128()) } else { None } } else { None };
                    if let Some(raw) = fits {
                        return conversion::reduce_decimal_to_rational(raw, *decimals);
                    }
                    let mut denom = I512::from_i128(1);
                    for _ in 0..*decimals { denom = denom * I512::from_i128(10); }
                    return Ok(RationalNumber::from_i512_pair(*scaled, denom));
                }
                #[cfg(table_format = "q128_128")]
                {
                    if scaled.fits_in_i128() {
                        return conversion::reduce_decimal_to_rational(scaled.as_i128(), *decimals);
                    }
                    let mut denom = I256::from_i128(1);
                    for _ in 0..*decimals { denom = denom * I256::from_i128(10); }
                    return Ok(RationalNumber::from_i256_pair(*scaled, denom));
                }
                #[cfg(table_format = "q64_64")]
                {
                    return conversion::reduce_decimal_to_rational(*scaled, *decimals);
                }

                #[cfg(table_format = "q32_32")]
                {
                    return conversion::reduce_decimal_to_rational(*scaled as i128, *decimals);
                }

                #[cfg(table_format = "q16_16")]
                {
                    return conversion::reduce_decimal_to_rational(*scaled as i128, *decimals);
                }

            }
            StackValue::Ternary(tier, value, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                domain::ternary_to_rational(*tier, value)
            }
            StackValue::Error(e) => Err(e.clone()),
        }
    }

    /// Extract the raw binary storage value (profile-specific type).
    ///
    /// Returns `None` for non-materializable values and error states.
    /// - `Binary` → returns as-is
    /// - `BinaryCompute` → downscales to storage tier
    /// - `Decimal` → converts to binary Q-format via `(scaled << frac_bits) / 10^dp`
    /// - `DecimalCompute` → downscales to Decimal first, then converts
    pub fn as_binary_storage(&self) -> Option<BinaryStorage> {
        match self {
            StackValue::Binary(_, val, _) => Some(*val),
            StackValue::BinaryCompute(_, val, _) => downscale_to_storage(*val).ok(),
            StackValue::Decimal(dp, scaled, _) => {
                decimal_to_binary_storage(*dp, *scaled).ok()
            }
            StackValue::DecimalCompute(_, val, _) => {
                // Direct conversion: DecimalCompute value × 2^frac_bits / 10^compute_dp
                // (no intermediate Decimal materialization — preserves full precision)
                decimal_compute_to_binary_storage(*val).ok()
            }
            _ => None,
        }
    }

    /// Extract the tier number.
    ///
    /// Returns the precision tier (1-6) for domain values, 0 for errors.
    pub fn tier(&self) -> u8 {
        match self {
            StackValue::Binary(t, _, _) => *t,
            StackValue::BinaryCompute(t, _, _) => *t,
            StackValue::Decimal(t, _, _) => *t,
            StackValue::DecimalCompute(t, _, _) => *t,
            StackValue::Ternary(t, _, _) => *t,
            StackValue::Symbolic(_) => 8, // Symbolic = rational tier
            StackValue::Error(_) => 0,
        }
    }

    /// Convert to decimal string with specified precision.
    ///
    /// Uses integer-only multiply-by-10 extraction (zero floats).
    /// Profile-dispatched: Q64.64 → max 19 digits, Q128.128 → max 38, Q256.256 → max 76.
    ///
    /// For non-binary values: converts to rational first, then extracts digits.
    pub fn to_decimal_string(&self, max_digits: usize) -> String {
        match self {
            StackValue::BinaryCompute(tier, val, shadow) => {
                match downscale_to_storage(*val) {
                    Ok(storage_val) => {
                        let materialized = StackValue::Binary(*tier, storage_val, shadow.clone());
                        materialized.to_decimal_string(max_digits)
                    }
                    Err(_) => "Overflow".to_string(),
                }
            }
            StackValue::DecimalCompute(_tier, val, shadow) => {
                // Materialize at the highest dp that fits (adaptive — 0d fix).
                match StackEvaluator::materialize_decimal_compute_adaptive(*val) {
                    Ok((target_dp, storage_val)) => {
                        let materialized = StackValue::Decimal(target_dp, storage_val, shadow.clone());
                        materialized.to_decimal_string(max_digits)
                    }
                    Err(_) => "Overflow".to_string(),
                }
            }
            StackValue::Binary(_tier, val, _) => {
                formatting::binary_storage_to_decimal_string(*val, max_digits)
            }
            StackValue::Decimal(decimals, val, _) => {
                let full = formatting::decimal_storage_to_string(*decimals, val);
                if max_digits < *decimals as usize {
                    if let Some(dot_pos) = full.find('.') {
                        let end = (dot_pos + 1 + max_digits).min(full.len());
                        full[..end].to_string()
                    } else {
                        full
                    }
                } else {
                    full
                }
            }
            StackValue::Symbolic(r) => {
                formatting::rational_to_decimal_string(r, max_digits)
            }
            StackValue::Ternary(_, _, _) => {
                if let Ok(rational) = self.to_rational() {
                    formatting::rational_to_decimal_string(&rational, max_digits)
                } else {
                    "NaN".to_string()
                }
            }
            StackValue::Error(_) => {
                "NaN".to_string()
            }
        }
    }
}

impl Display for StackValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(table_format = "q16_16")]
        const DEFAULT_DIGITS: usize = 5;
        #[cfg(table_format = "q32_32")]
        const DEFAULT_DIGITS: usize = 9;
        #[cfg(table_format = "q64_64")]
        const DEFAULT_DIGITS: usize = 19;
        #[cfg(table_format = "q128_128")]
        const DEFAULT_DIGITS: usize = 38;
        #[cfg(table_format = "q256_256")]
        const DEFAULT_DIGITS: usize = 77;

        let precision = f.precision().unwrap_or(DEFAULT_DIGITS);

        match self {
            StackValue::BinaryCompute(tier, val, shadow) => {
                match downscale_to_storage(*val) {
                    Ok(storage_val) => {
                        let materialized = StackValue::Binary(*tier, storage_val, shadow.clone());
                        write!(f, "{}", materialized.to_decimal_string(precision))
                    }
                    Err(_) => write!(f, "Overflow"),
                }
            }
            StackValue::DecimalCompute(_tier, val, shadow) => {
                match StackEvaluator::materialize_decimal_compute_adaptive(*val) {
                    Ok((target_dp, storage_val)) => {
                        let materialized = StackValue::Decimal(target_dp, storage_val, shadow.clone());
                        write!(f, "{}", materialized.to_decimal_string(precision))
                    }
                    Err(_) => write!(f, "Overflow"),
                }
            }
            StackValue::Binary(_, val, _) => {
                write!(f, "{}", formatting::binary_storage_to_decimal_string(*val, precision))
            }
            StackValue::Decimal(dec, val, _) => {
                write!(f, "{}", formatting::decimal_storage_to_string(*dec, val))
            }
            StackValue::Ternary(_, _, _) => {
                write!(f, "{}", self.to_decimal_string(precision))
            }
            StackValue::Symbolic(s) => write!(f, "{}", formatting::rational_to_decimal_string(s, precision)),
            StackValue::Error(e) => write!(f, "Error: {:?}", e),
        }
    }
}

// ============================================================================
// STACK EVALUATOR — THREAD-LOCAL COMPUTATION ENGINE
// ============================================================================

/// Universal stack evaluator - thread-local computation engine
///
/// **ARCHITECTURE**: Fixed-size stack with domain-aware evaluation
/// **THREAD-SAFETY**: Each thread gets its own evaluator instance
/// **MEMORY**: Zero heap allocation during computation
pub struct StackEvaluator {
    /// Current workspace offset
    offset: usize,

    /// Stack of computed values (fixed size)
    value_stack: [Option<StackValue>; 256],

    /// Current stack pointer
    stack_ptr: usize,

    /// Deployment profile for tier limits
    deployment_profile: DeploymentProfile,
}

impl StackEvaluator {
    /// Create new evaluator with deployment profile
    pub fn new(deployment_profile: DeploymentProfile) -> Self {
        Self {
            offset: 0,
            value_stack: [const { None }; 256],
            stack_ptr: 0,
            deployment_profile,
        }
    }

    /// Reset evaluator state for new computation
    pub fn reset(&mut self) {
        self.offset = 0;
        self.stack_ptr = 0;
    }

    /// Load value from stack reference
    fn load_ref(&self, stack_ref: StackRef) -> Result<StackValue, OverflowDetected> {
        let idx = stack_ref.0 as usize;
        if idx >= self.stack_ptr {
            return Err(OverflowDetected::InvalidStackReference);
        }

        self.value_stack[idx]
            .clone()
            .ok_or(OverflowDetected::StackCorruption)
    }

    /// Evaluate expression on stack
    pub fn evaluate(&mut self, expr: &LazyExpr) -> Result<StackValue, OverflowDetected> {
        match expr {
            LazyExpr::Literal(s) => self.parse_literal_with_mode(s),
            LazyExpr::Value(v) => Ok((**v).clone()),
            LazyExpr::Constant(c) => self.load_constant(*c),
            LazyExpr::Variable(r) => self.load_ref(*r),
            LazyExpr::Negate(inner) => {
                let val = self.evaluate(inner)?;
                self.negate_value(val)
            }
            LazyExpr::Add(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                self.add_values(l, r)
            }
            LazyExpr::Sub(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                self.subtract_values(l, r)
            }
            LazyExpr::Mul(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                self.multiply_values(l, r)
            }
            LazyExpr::Div(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                self.divide_values(l, r)
            }
            LazyExpr::Exp(inner) => {
                // Identity short-circuit: exp(ln(x)) = x
                if let LazyExpr::Ln(inner_inner) = inner.as_ref() {
                    return self.evaluate(inner_inner);
                }
                let val = self.evaluate(inner)?;
                self.evaluate_exp(val)
            }
            LazyExpr::Ln(inner) => {
                // Identity short-circuit: ln(exp(x)) = x
                if let LazyExpr::Exp(inner_inner) = inner.as_ref() {
                    return self.evaluate(inner_inner);
                }
                let val = self.evaluate(inner)?;
                self.evaluate_ln(val)
            }
            LazyExpr::Sqrt(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_sqrt(val)
            }
            LazyExpr::Pow(base, exponent) => {
                let base_val = self.evaluate(base)?;
                let exp_val = self.evaluate(exponent)?;
                self.evaluate_pow(base_val, exp_val)
            }
            LazyExpr::Sinh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_sinh(val)
            }
            LazyExpr::Cosh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_cosh(val)
            }
            LazyExpr::Tanh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_tanh(val)
            }
            LazyExpr::Asinh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_asinh(val)
            }
            LazyExpr::Acosh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_acosh(val)
            }
            LazyExpr::Atanh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_atanh(val)
            }
            LazyExpr::Sin(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_sin(val)
            }
            LazyExpr::Cos(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_cos(val)
            }
            LazyExpr::Tan(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_tan(val)
            }
            LazyExpr::Asin(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_asin(val)
            }
            LazyExpr::Acos(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_acos(val)
            }
            LazyExpr::Atan(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_atan(val)
            }
            LazyExpr::Atan2(y, x) => {
                let y_val = self.evaluate(y)?;
                let x_val = self.evaluate(x)?;
                self.evaluate_atan2(y_val, x_val)
            }
        }
    }

    /// Materialize BinaryCompute back to Binary (storage tier)
    ///
    /// Called at the top-level evaluate boundary to ensure callers always
    /// receive values at the storage tier, not the internal compute tier.
    ///
    /// Returns `Err(TierOverflow)` if the compute-tier value exceeds the
    /// storage tier's range (UGOD overflow detection).
    /// Materialize a DecimalCompute raw at the highest storage dp that
    /// fits (0.5.0 item 0d fix): try DECIMAL_STORAGE_MAX_DP first and step
    /// down only when the magnitude genuinely needs fewer decimals. The
    /// old fixed `MAX_DP - 2` slack silently cost realtime HALF its digits
    /// (dp 4 → 2: cos(0.1) = 0.9952 displayed as "1.00"), which masqueraded
    /// as a sin/cos kernel plateau for months. Deterministic; the
    /// downscale is checked, so a too-large dp errors and we retry — never
    /// a wrap.
    fn materialize_decimal_compute_adaptive(
        val: ComputeStorage,
    ) -> Result<(u8, BinaryStorage), OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::transcendental::{
            decimal_downscale_to_storage, DECIMAL_STORAGE_MAX_DP,
        };
        let mut dp = DECIMAL_STORAGE_MAX_DP;
        loop {
            match decimal_downscale_to_storage(val, dp) {
                Ok(v) => return Ok((dp, v)),
                Err(e) => {
                    if dp == 0 {
                        return Err(e);
                    }
                    dp -= 1;
                }
            }
        }
    }

    pub(crate) fn materialize_compute(&self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        match value {
            StackValue::BinaryCompute(tier, val, shadow) => {
                let storage = downscale_to_storage(val)?;
                Ok(StackValue::Binary(tier, storage, shadow))
            }
            // DecimalCompute passes through — it's a valid output format.
            // Chain persistence ensures DecimalCompute stays hot between
            // transcendental operations. Display handles formatting.
            other => Ok(other),
        }
    }
}

// ============================================================================
// THREAD-LOCAL EVALUATOR INSTANCE
// ============================================================================

/// Get the compile-time deployment profile based on table format
const fn compile_time_profile() -> DeploymentProfile {
    #[cfg(table_format = "q256_256")]
    { DeploymentProfile::Scientific }

    #[cfg(table_format = "q128_128")]
    { DeploymentProfile::Balanced }

    #[cfg(table_format = "q64_64")]
    { DeploymentProfile::Embedded }

    #[cfg(table_format = "q32_32")]
    { DeploymentProfile::Compact }

    #[cfg(table_format = "q16_16")]
    { DeploymentProfile::Realtime }
}

thread_local! {
    static EVALUATOR: RefCell<StackEvaluator> = RefCell::new(
        StackEvaluator::new(compile_time_profile())
    );
}

/// Evaluate expression using thread-local evaluator
///
/// **MATERIALIZATION**: Any BinaryCompute results are materialized to Binary
/// before returning, ensuring callers always receive storage-tier values.
///
/// **MODE ROUTING**: If a non-Auto output mode is set via `set_mode()`,
/// the result is converted to the requested domain after materialization.
pub fn evaluate(expr: &LazyExpr) -> Result<StackValue, OverflowDetected> {
    EVALUATOR.with(|eval| {
        let mut evaluator = eval.borrow_mut();
        evaluator.reset();
        let result = evaluator.evaluate(expr)?;
        let materialized = evaluator.materialize_compute(result)?;
        evaluator.apply_output_mode(materialized)
    })
}

/// Evaluate sin and cos of the same expression with a single range reduction.
///
/// More efficient than evaluating sin(x) and cos(x) separately — shares the
/// Cody-Waite range reduction and Taylor evaluation at compute tier.
///
/// **USAGE**:
/// ```rust
/// use g_math::canonical::{gmath, evaluate_sincos};
/// let (sin_val, cos_val) = evaluate_sincos(&gmath("0.5")).unwrap();
/// ```
pub fn evaluate_sincos(expr: &LazyExpr) -> Result<(StackValue, StackValue), OverflowDetected> {
    EVALUATOR.with(|eval| {
        let mut evaluator = eval.borrow_mut();
        evaluator.reset();
        let inner_val = evaluator.evaluate(expr)?;
        let (sin_compute, cos_compute) = evaluator.evaluate_sincos(inner_val)?;
        let sin_mat = evaluator.materialize_compute(sin_compute)?;
        let cos_mat = evaluator.materialize_compute(cos_compute)?;
        let sin_out = evaluator.apply_output_mode(sin_mat)?;
        let cos_out = evaluator.apply_output_mode(cos_mat)?;
        Ok((sin_out, cos_out))
    })
}

/// Evaluate sinh and cosh of the same expression with a single shared exp-pair.
///
/// More efficient than evaluating sinh(x) and cosh(x) separately — shares one
/// `exp(x)` + `exp(-x)` evaluation at compute tier. sinh and cosh come from the
/// same `(ep, en)` pair, so their rounding errors are correlated and cancel
/// in downstream expressions like `cosh(θ)·p + (sinh(θ)/θ)·v`.
///
/// Routes to native decimal engine for decimal inputs, binary otherwise.
///
/// **USAGE**:
/// ```rust
/// use g_math::canonical::{gmath, evaluate_sinhcosh};
/// let (sinh_val, cosh_val) = evaluate_sinhcosh(&gmath("0.5")).unwrap();
/// ```
pub fn evaluate_sinhcosh(expr: &LazyExpr) -> Result<(StackValue, StackValue), OverflowDetected> {
    EVALUATOR.with(|eval| {
        let mut evaluator = eval.borrow_mut();
        evaluator.reset();
        let inner_val = evaluator.evaluate(expr)?;
        let (sinh_compute, cosh_compute) = evaluator.evaluate_sinhcosh(inner_val)?;
        let sinh_mat = evaluator.materialize_compute(sinh_compute)?;
        let cosh_mat = evaluator.materialize_compute(cosh_compute)?;
        let sinh_out = evaluator.apply_output_mode(sinh_mat)?;
        let cosh_out = evaluator.apply_output_mode(cosh_mat)?;
        Ok((sinh_out, cosh_out))
    })
}

/// Parse a runtime string into a StackValue
///
/// **PURPOSE**: Bridge for runtime/dynamic string inputs that cannot use
/// `gmath()` (which requires `&'static str`). Eagerly parses through the
/// thread-local evaluator with full mode routing support.
///
/// **USAGE**:
/// ```rust
/// use g_math::canonical::{gmath_parse, evaluate, LazyExpr};
///
/// let user_input = String::from("3.14");
/// let expr = gmath_parse(&user_input).unwrap();
/// let result = evaluate(&(expr + gmath_parse("2.0").unwrap())).unwrap();
/// ```
pub fn gmath_parse(s: &str) -> Result<LazyExpr, OverflowDetected> {
    let value = EVALUATOR.with(|eval| {
        let mut evaluator = eval.borrow_mut();
        evaluator.parse_literal_with_mode(s)
    })?;
    Ok(LazyExpr::Value(Box::new(value)))
}
