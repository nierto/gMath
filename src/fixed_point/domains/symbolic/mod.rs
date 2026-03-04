//! Symbolic Mathematics Module — Rational Arithmetic Domain
//!
//! Provides exact rational arithmetic through RationalNumber and tiered operations.
//! All legacy symbolic tracking (SymbolicAdaptiveFixedPoint, Symbol, etc.) has been
//! archived — ZASC uses RationalNumber directly via StackValue::Symbolic.

pub mod rational;

// Tiered symbolic operations (rational arithmetic across all tiers)
mod symbolic_arithmetic;

// Rational arithmetic exports
pub use rational::{
    RationalNumber, RationalStorage, SymbolicConstants,
    symbolic_to_universal_tier, universal_to_symbolic_tier,
};

// Tiered arithmetic exports (tiers 1-5: add/sub/mul/div)
pub use symbolic_arithmetic::{
    add_i8_rational, sub_i8_rational, mul_i8_rational, div_i8_rational,
    add_i16_rational, sub_i16_rational, mul_i16_rational, div_i16_rational,
    add_i32_rational, sub_i32_rational, mul_i32_rational, div_i32_rational,
    add_i64_rational, sub_i64_rational, mul_i64_rational, div_i64_rational,
    add_i128_rational, sub_i128_rational, mul_i128_rational, div_i128_rational,
    gcd_unsigned,
};
