//! Universal Fixed-Point Infrastructure
//!
//! Core systems shared across all domains:
//! - `ugod`: Universal Graceful Overflow Delegation (tier management)
//! - `zasc`: Zero-Allocation Stack Computation (lazy evaluation)
//! - `tier_types`: Universal 6-tier type system shared by all domains

pub mod ugod;
pub mod zasc;
pub mod tier_types;

// UGOD exports
pub use ugod::{
    UniversalTieredArithmetic,
    DomainType
};

// ZASC exports
pub use zasc::{
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
