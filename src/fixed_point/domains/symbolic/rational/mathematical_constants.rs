#![allow(non_snake_case)]
//! Mathematical Constants — Single Source of Truth
//!
//! All constants are generated at build-time with profile-aware precision via pure-Rust
//! BigRational arithmetic. Zero floating-point contamination.
//!
//! ARCHITECTURE: Build-time generation → symbolic rationals → domain conversion
//! PRECISION: Profile-aware (Embedded, Balanced, Scientific)
//! DETERMINISM: Bit-identical results across all platforms and threads

use crate::fixed_point::domains::symbolic::rational::RationalNumber;

// Import extended-precision integer types for tier 6-7 arithmetic
#[allow(unused_imports)]
#[cfg(not(feature = "embedded"))]
use crate::fixed_point::domains::binary_fixed::I256;
#[allow(unused_imports)]
use crate::fixed_point::domains::binary_fixed::I512;

// Include the build-generated constants database
// This provides mathematical constants as rational pairs (numerator, denominator)
#[cfg(feature = "rebuild-tables")]
include!(concat!(env!("OUT_DIR"), "/mathematical_constants.rs"));
#[cfg(not(feature = "rebuild-tables"))]
include!("../../../../generated_tables/mathematical_constants.rs");

/// Symbolic Constants - CANONICAL SOURCE for all mathematical constants across ALL domains
///
/// Single source of truth for precision-preserving constants.
///
/// All mathematical constants (π, e, √2, etc.) are ONLY defined here as exact rational
/// representations. Other domains (Binary, Decimal, Ternary) DO NOT define constants.
/// Instead, constant operations are ROUTED to the Symbolic domain for computation, then
/// the final result is converted back to the target domain.
///
/// **GENERATION**: Build-time via build.rs using algorithmic series:
/// - π: Machin's formula (16*arctan(1/5) - 4*arctan(1/239))
/// - e: Factorial series (Σ(1/n!) for n=0 to profile_terms)
/// - √2: Continued fraction algorithm with profile-aware terms
///
/// **PRECISION**: Profile-aware (Embedded→19, Balanced→38, Scientific→77 decimals)
/// **DETERMINISM**: Bit-identical across all platforms via pure rational arithmetic
/// **ZERO FLOATING POINT**: All constants generated as exact rational pairs
pub struct SymbolicConstants;

impl SymbolicConstants {
    /// High-precision π using build-generated rational approximation
    ///
    /// **PRECISION**: Profile-aware (embedded: 19, balanced: 38, scientific: 77 decimals)
    /// **SOURCE**: Verified 100+ decimal string from OEIS A000796
    /// **ARCHITECTURE**: Scaled-integer rational (numerator / 10^decimals)
    pub fn pi_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::PI();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// High-precision e using build-generated rational approximation
    ///
    /// **PRECISION**: Profile-aware (embedded: 19, balanced: 38, scientific: 77 decimals)
    /// **SOURCE**: Verified 100+ decimal string from OEIS A001113
    pub fn e_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::E();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// High-precision √2 using build-generated rational approximation
    ///
    /// **PRECISION**: Profile-aware (embedded: 19, balanced: 38, scientific: 77 decimals)
    /// **SOURCE**: Verified 100+ decimal string from OEIS A002193
    pub fn sqrt_2_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::SQRT_2();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// High-precision natural logarithm of 2
    ///
    /// **PRECISION**: Profile-aware (embedded: 19, balanced: 38, scientific: 77 decimals)
    /// **SOURCE**: Verified 100+ decimal string from OEIS A002162
    pub fn ln_2_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::LN_2();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// High-precision golden ratio φ = (1 + √5)/2
    ///
    /// **PRECISION**: Profile-aware (embedded: 19, balanced: 38, scientific: 77 decimals)
    /// **SOURCE**: Verified 100+ decimal string from OEIS A001622
    pub fn golden_ratio_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::GOLDEN_RATIO();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// High-precision √3 using build-generated rational approximation
    ///
    /// **PRECISION**: Profile-aware (embedded: 19, balanced: 38, scientific: 77 decimals)
    /// **SOURCE**: Verified 100+ decimal string from OEIS A002194
    pub fn sqrt_3_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::SQRT_3();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// ln(10) - computed from ln(2) relationship
    pub fn ln_10() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::LN_10();
        RationalNumber::from_i512_pair(I512::from_words(num_words), I512::from_words(den_words))
    }
}
