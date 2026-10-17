#![allow(non_snake_case)]
//! Mathematical Constants: Single Source of Truth
//!
//! All constants are generated at build-time at up to 77-decimal precision via pure-Rust
//! BigRational arithmetic. Zero floating-point contamination.
//!
//! ARCHITECTURE: Build-time generation → symbolic rationals → domain conversion
//! PRECISION: per format (Q64.64, Q128.128, Q256.256)
//! DETERMINISM: Bit-identical results across all platforms and threads

use crate::fixed_point::domains::symbolic::rational::RationalNumber;

// Import extended-precision integer types for tier 6-7 arithmetic
#[allow(unused_imports)]
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
/// - e: Factorial series (Σ(1/n!) for n=0 to the number of terms the precision needs)
/// - √2: Continued fraction algorithm with as many terms as the precision needs
///
/// **PRECISION**: per format (Q64.64→19, Q128.128→38, Q256.256→77 decimals)
/// **DETERMINISM**: Bit-identical across all platforms via pure rational arithmetic
/// **ZERO FLOATING POINT**: All constants generated as exact rational pairs
pub struct SymbolicConstants;

impl SymbolicConstants {
    /// High-precision π using build-generated rational approximation
    ///
    /// **PRECISION**: 19 decimals at Q64.64, 38 at Q128.128, 77 at Q256.256
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
    /// **PRECISION**: 19 decimals at Q64.64, 38 at Q128.128, 77 at Q256.256
    /// **SOURCE**: Verified 100+ decimal string from OEIS A001113
    pub fn e_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::E();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// High-precision √2 using build-generated rational approximation
    ///
    /// **PRECISION**: 19 decimals at Q64.64, 38 at Q128.128, 77 at Q256.256
    /// **SOURCE**: Verified 100+ decimal string from OEIS A002193
    pub fn sqrt_2_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::SQRT_2();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// High-precision natural logarithm of 2
    ///
    /// **PRECISION**: 19 decimals at Q64.64, 38 at Q128.128, 77 at Q256.256
    /// **SOURCE**: Verified 100+ decimal string from OEIS A002162
    pub fn ln_2_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::LN_2();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// High-precision golden ratio φ = (1 + √5)/2
    ///
    /// **PRECISION**: 19 decimals at Q64.64, 38 at Q128.128, 77 at Q256.256
    /// **SOURCE**: Verified 100+ decimal string from OEIS A001622
    pub fn golden_ratio_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::GOLDEN_RATIO();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// High-precision √3 using build-generated rational approximation
    ///
    /// **PRECISION**: 19 decimals at Q64.64, 38 at Q128.128, 77 at Q256.256
    /// **SOURCE**: Verified 100+ decimal string from OEIS A002194
    pub fn sqrt_3_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::SQRT_3();
        let num = I512::from_words(num_words);
        let den = I512::from_words(den_words);
        RationalNumber::from_i512_pair(num, den)
    }

    /// Euler-Mascheroni constant gamma: the best rational approximation with
    /// a denominator below 2^192 (error below 2^-382). Before 0.7.0 the
    /// evaluator used 5772/10000 and the `gamma` literal 577215665/10^9.
    pub fn euler_gamma_high_precision() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::EULER_GAMMA();
        RationalNumber::from_i512_pair(I512::from_words(num_words), I512::from_words(den_words))
    }

    /// ln(10) - computed from ln(2) relationship
    pub fn ln_10() -> RationalNumber {
        let (num_words, den_words) = MathematicalConstants::LN_10();
        RationalNumber::from_i512_pair(I512::from_words(num_words), I512::from_words(den_words))
    }
}

#[cfg(test)]
mod precision_tests {
    use super::*;
    use num_bigint::BigInt;

    fn big(words: [u64; 8]) -> BigInt {
        let mut v = BigInt::from(0);
        for w in words.iter().rev() { v = (v << 64usize) + BigInt::from(*w); }
        v
    }

    /// Every constant within 2^-380 of its value (floor(value * 10^120) from
    /// mpmath at 200 digits): |num * 10^120 - D * den| <= den * (10^120 2^-380 + 1).
    #[test]
    fn constants_carry_380_bits() {
        let cases: [(&str, ([u64; 8], [u64; 8]), &str); 10] = [
            ("PI", MathematicalConstants::PI(), "3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647"),
            ("E", MathematicalConstants::E(), "2718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427427466391932003059921"),
            ("SQRT_2", MathematicalConstants::SQRT_2(), "1414213562373095048801688724209698078569671875376948073176679737990732478462107038850387534327641572735013846230912297024"),
            ("LN_2", MathematicalConstants::LN_2(), "693147180559945309417232121458176568075500134360255254120680009493393621969694715605863326996418687542001481020570685733"),
            ("PHI", MathematicalConstants::PHI(), "1618033988749894848204586834365638117720309179805762862135448622705260462818902449707207204189391137484754088075386891752"),
            ("LN_10", MathematicalConstants::LN_10(), "2302585092994045684017991454684364207601101488628772976033327900967572609677352480235997205089598298341967784042286248633"),
            ("SQRT_3", MathematicalConstants::SQRT_3(), "1732050807568877293527446341505872366942805253810380628055806979451933016908800037081146186757248575675626141415406703029"),
            ("SQRT_5", MathematicalConstants::SQRT_5(), "2236067977499789696409173668731276235440618359611525724270897245410520925637804899414414408378782274969508176150773783504"),
            ("EULER_GAMMA", MathematicalConstants::EULER_GAMMA(), "577215664901532860606512090082402431042159335939923598805767234884867726777664670936947063291746749514631447249807082480"),
            ("E_SQUARED", MathematicalConstants::E_SQUARED(), "7389056098930650227230427460575007813180315570551847324087127822522573796079057763384312485079121794773753161265478866123"),
        ];
        let scale = BigInt::from(10).pow(120u32);
        for (name, (n, d), digits) in cases {
            let (n, d) = (big(n), big(d));
            let dd: BigInt = digits.parse().unwrap();
            let diff = &n * &scale - &dd * &d;
            let diff = if diff < BigInt::from(0) { -diff } else { diff };
            let bound = (&d * &scale >> 380usize) + &d;
            assert!(diff <= bound, "{name}: rational farther than 2^-380 from its value");
        }
    }
}
