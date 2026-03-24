//! Balanced Ternary Fixed-Point Domain
//!
//! UGOD 6-tier ternary arithmetic with native-sized storage types.
//! Legacy balanced ternary encoding (Trit/TritPair/SIMD) archived.
//!
//! Hotpath: ternary_types → ternary_{addition,multiplication,division,negation}
//! All ternary values flow through StackEvaluator as Ternary(tier, BinaryStorage, CompactShadow).

// UGOD tier types, constructors, promotion, helpers
pub mod ternary_types;

// UGOD tier arithmetic operations
pub mod ternary_addition;
pub mod ternary_multiplication;
pub mod ternary_division;
pub mod ternary_negation;

// TQ1.9 compact ternary format (standalone, for inference weight storage)
pub mod trit_q1_9;

// Trit packing utilities (5 trits/byte base-3 encoding)
pub mod trit_packing;

// UGOD type exports
pub use ternary_types::{
    UniversalTernaryFixed, TernaryTier,
    TernaryTier1, TernaryTier2, TernaryTier3, TernaryTier4, TernaryTier5, TernaryTier6,
    TernaryValue, TernaryRaw,
    SCALE_TQ8_8, SCALE_TQ16_16, SCALE_TQ32_32,
};

// UGOD operation exports
pub use ternary_addition::{
    add_ternary_tq8_8, add_ternary_tq16_16, add_ternary_tq32_32, add_ternary_tq64_64,
    subtract_ternary_tq8_8, subtract_ternary_tq16_16, subtract_ternary_tq32_32, subtract_ternary_tq64_64,
    // Tier 4 checked variants (for UGOD promotion to Tier 5)
    add_ternary_tq64_64_checked, subtract_ternary_tq64_64_checked,
    // Tier 5: TQ128.128
    add_ternary_tq128_128, subtract_ternary_tq128_128,
    // Tier 6: TQ256.256
    add_ternary_tq256_256, subtract_ternary_tq256_256,
};
pub use ternary_multiplication::{
    multiply_ternary_tq8_8, multiply_ternary_tq16_16, multiply_ternary_tq32_32, multiply_ternary_tq64_64,
    multiply_ternary_tq64_64_checked,
    multiply_ternary_tq128_128,
    multiply_ternary_tq256_256,
};
pub use ternary_division::{
    divide_ternary_tq8_8, divide_ternary_tq16_16, divide_ternary_tq32_32, divide_ternary_tq64_64,
    divide_ternary_tq64_64_checked,
    divide_ternary_tq128_128,
    divide_ternary_tq256_256,
};
pub use ternary_negation::{
    negate_ternary_tq8_8, negate_ternary_tq16_16, negate_ternary_tq32_32, negate_ternary_tq64_64,
    negate_ternary_tq128_128,
    negate_ternary_tq256_256,
};
pub use crate::fixed_point::core_types::errors::OverflowDetected;

// TQ1.9 + trit packing exports
pub use trit_q1_9::TritQ1_9;
pub use trit_packing::{Trit, pack_trits, unpack_trits};
