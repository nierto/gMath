//! Binary Fixed-Point Arithmetic Domain
//!
//! ARCHITECTURE: Q64.64 / Q128.128 / Q256.256 binary fixed-point arithmetic
//! DOMAIN: Binary fractions, transcendental functions, wide-integer constants

// UGOD tier types and operations
pub mod binary_types;
pub mod binary_addition;
pub mod binary_multiplication;
pub mod binary_division;
pub mod binary_negation;

// Transcendental functions
pub mod transcendental;

// Integer types
pub mod i256;
pub mod i512;
pub mod i1024;
pub mod i2048;

// Re-export core binary multiplication (AVX2-accelerated with scalar fallback)
pub use binary_multiplication::{
    multiply_binary_i128, multiply_binary_i128_scalar,
};


pub use i256::{I256, mul_i128_to_i256};
pub use i512::I512;
pub use i1024::I1024;
pub use i2048::I2048;

// Binary tier types for runtime UGOD
pub use binary_types::{
    BinaryRaw, BinaryValue, UniversalBinaryFixed,
    BinaryTier1, BinaryTier2, BinaryTier3, BinaryTier4, BinaryTier5, BinaryTier6,
};
