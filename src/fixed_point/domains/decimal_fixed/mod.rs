//! Decimal Fixed-Point Domain Module
//!
//! DOMAIN-SPECIFIC: Decimal arithmetic components separated from binary domain
//! ARCHITECTURE: D128/D256/D512 integer types optimized for base-10 operations
//! PRECISION: Powers of 10 scaling for exact decimal arithmetic

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

// Native decimal transcendental functions (tier N+1 with decimal scaling)
pub mod transcendental;

// Re-export the main DecimalFixed types
pub use decimal_fixed::{
    DecimalFixed, DecimalFixed2, DecimalFixed3, DecimalFixed6, DecimalFixed9,
    Currency, HighPrecisionCurrency, ParseError, compile_time_power_of_10
};

// Universal 6-tier decimal types for FASC + UGOD
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
    let abs_divisor = divisor.abs();
    let abs_remainder = remainder.abs();
    // `abs_divisor / 2` truncates; comparing against it directly would (a) treat
    // every exact result as a tie when divisor == 1 (half == 0) and (b) misjudge
    // odd divisors, where the true midpoint is `(D-1)/2 + 0.5`. Compare the
    // floored half and use the divisor parity to resolve the boundary. (We avoid
    // `2 * abs_remainder`, which overflows i128 for large divisors.)
    let half = abs_divisor / 2;
    let bump = if remainder >= 0 { quotient + 1 } else { quotient - 1 };

    if abs_remainder < half {
        quotient // strictly below the midpoint -> round down
    } else if abs_remainder > half {
        bump // strictly above the midpoint -> round up
    } else if abs_divisor % 2 != 0 {
        // Odd divisor: remainder == (D-1)/2 is below the true midpoint D/2.
        // (Covers divisor == 1, where half == 0 and exact results round down.)
        quotient
    } else if quotient % 2 == 0 {
        quotient // exact half, even quotient -> round to even
    } else {
        bump // exact half, odd quotient -> round to even
    }
}
