//! # Rational Arithmetic Domain
//!
//! Pure rational arithmetic (a/b) with infinite precision, tiered storage
//! optimization, and exact fraction arithmetic for the symbolic domain.
//!
//! ## Core Components
//!
//! - **rational_number**: Main RationalNumber type with 7-tier storage hierarchy
//! - **rational_conversion**: Cross-domain conversion utilities
//! - **mathematical_constants**: Pre-computed rational constants
//!
//! ## Public API
//!
//! - `RationalNumber`: Arbitrary-precision rational arithmetic type
//! - `RationalStorage`: 7-tier storage optimization (Tiny → Ultra)
//! - `SymbolicConstants`: Pre-computed mathematical constants (π, e, √2, etc.)
//!
//! ## Integration Points
//!
//! - **UGOD**: Automatic tier escalation on overflow
//! - **ZASC**: Zero-allocation stack computation integration

pub mod rational_number;
pub mod rational_conversion;
pub mod mathematical_constants;
#[cfg(test)]
mod tests;

// Re-export primary types for domain-level access
pub use rational_number::{
    RationalNumber, RationalStorage, RationalParts, OverflowDetected,
    symbolic_to_universal_tier, universal_to_symbolic_tier,
};

// Re-export conversion types
pub use rational_conversion::ParseError;

// SymbolicConstants is the canonical source for all mathematical constants across all domains
pub use mathematical_constants::SymbolicConstants;
