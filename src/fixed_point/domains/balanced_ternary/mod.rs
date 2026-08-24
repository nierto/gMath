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
    SCALE_TQ10_10, SCALE_TQ20_20, SCALE_TQ40_40,
};

// UGOD operation exports
pub use ternary_addition::{
    add_ternary_tq10_10, add_ternary_tq20_20, add_ternary_tq40_40, add_ternary_tq80_80,
    subtract_ternary_tq10_10, subtract_ternary_tq20_20, subtract_ternary_tq40_40, subtract_ternary_tq80_80,
    // Tier 4 checked variants (for UGOD promotion to Tier 5)
    add_ternary_tq80_80_checked, subtract_ternary_tq80_80_checked,
    // Tier 5: TQ160.160
    add_ternary_tq160_160, subtract_ternary_tq160_160,
    // Tier 6: TQ320.320
    add_ternary_tq320_320, subtract_ternary_tq320_320,
};
pub use ternary_multiplication::{
    multiply_ternary_tq10_10, multiply_ternary_tq20_20, multiply_ternary_tq40_40, multiply_ternary_tq80_80,
    multiply_ternary_tq80_80_checked,
    multiply_ternary_tq160_160,
    multiply_ternary_tq320_320,
};
pub use ternary_division::{
    divide_ternary_tq10_10, divide_ternary_tq20_20, divide_ternary_tq40_40, divide_ternary_tq80_80,
    divide_ternary_tq80_80_checked,
    divide_ternary_tq160_160,
    divide_ternary_tq320_320,
};
pub use ternary_negation::{
    negate_ternary_tq10_10, negate_ternary_tq20_20, negate_ternary_tq40_40, negate_ternary_tq80_80,
    negate_ternary_tq160_160,
    negate_ternary_tq320_320,
};
pub use crate::fixed_point::core_types::errors::OverflowDetected;

// TQ1.9 + trit packing exports
pub use trit_q1_9::TritQ1_9;
pub use trit_packing::{Trit, pack_trits, unpack_trits};
