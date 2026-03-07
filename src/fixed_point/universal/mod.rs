//! Universal Fixed-Point Infrastructure
//!
//! Core systems shared across all domains:
//! - `ugod`: Universal Graceful Overflow Delegation (tier management)
//! - `fasc`: Fixed-Allocation Stack Computation (lazy evaluation)
//! - `tier_types`: Universal 6-tier type system shared by all domains

pub mod ugod;
pub mod fasc;
pub mod tier_types;

// UGOD exports
pub use ugod::{
    UniversalTieredArithmetic,
    DomainType
};

// FASC exports
pub use fasc::{
    LazyExpr,
    gmath,
    ConstantId,
    StackEvaluator,
    StackValue
};

// Tier Types exports
pub use tier_types::{
    CompactShadow,
    ShadowConstantId,
};
