//! Decimal Fixed-Point Domain Module
//!
//! DOMAIN-SPECIFIC: Decimal arithmetic components separated from binary domain
//! ARCHITECTURE: D128/D256/D512 integer types optimized for base-10 operations
//! PRECISION: Powers of 10 scaling with 0-ULP decimal arithmetic

// Core DecimalFixed implementation (public API: DecimalFixed, Currency)
pub mod decimal_fixed;

// Domain-specific decimal integer types
pub mod d256;
pub mod d512;

// UGOD tier types, constructors, promotion, helpers
pub mod decimal_types;

// UGOD tier arithmetic operations (called by StackEvaluator)
pub mod decimal_addition;
pub mod decimal_multiplication;
pub mod decimal_division;
pub mod decimal_negation;

// Re-export the main DecimalFixed types
pub use decimal_fixed::{
    DecimalFixed, DecimalFixed2, DecimalFixed3, DecimalFixed6, DecimalFixed9,
    Currency, HighPrecisionCurrency, ParseError, compile_time_power_of_10
};

// Universal 6-tier decimal types for ZASC + UGOD
pub use decimal_types::{
    DecimalRaw, DecimalValueTiered, UniversalDecimalTiered,
    DecimalTier1 as DecimalTierType1, DecimalTier2 as DecimalTierType2,
    DecimalTier3 as DecimalTierType3, DecimalTier4 as DecimalTierType4,
    DecimalTier5 as DecimalTierType5, DecimalTier6 as DecimalTierType6,
    DecimalTier1, DecimalTier2, DecimalTier3, DecimalTier4, DecimalTier5, DecimalTier6,
    tier_for_decimal_places, max_decimal_places_for_tier,
    i256_to_d256, d256_to_i256, i512_to_d512, d512_to_i512,
};

// Domain-separated integer type exports
pub use d256::{D256, DecimalD256, mul_i128_to_d256, mul_d256_to_d512, negate_d256, divmod_d256_by_i128};
pub use d512::{D512, DecimalD512, negate_d512};

// Type aliases for common decimal integer sizes
/// 128-bit decimal integer (standard i128 but domain-specific context)
pub type D128 = i128;

/// Domain-specific banker's rounding for decimal operations
///
/// ALGORITHM: Round half to even (IEEE 754 standard) optimized for decimal domain
/// DETERMINISM: Identical results across all platforms
/// SEPARATION: Isolated from binary domain rounding functions
pub fn banker_round_decimal_i128(quotient: i128, remainder: i128, divisor: i128) -> i128 {
    let half_divisor = divisor / 2;
    let abs_remainder = remainder.abs();

    if abs_remainder < half_divisor {
        // Round down
        quotient
    } else if abs_remainder > half_divisor {
        // Round up
        if remainder >= 0 {
            quotient + 1
        } else {
            quotient - 1
        }
    } else {
        // Exact half - round to even
        if quotient % 2 == 0 {
            quotient
        } else {
            if remainder >= 0 {
                quotient + 1
            } else {
                quotient - 1
            }
        }
    }
}
