//! Fixed-Point Arithmetic Module — Canonical FASC+UGOD Pipeline
//!
//! **Public API**: [`canonical`] module — `use g_math::canonical::{gmath, evaluate};`
//!
//! All other modules are internal implementation details.

/// Canonical FASC entry point — the public API for gMath.
pub mod canonical;

/// Imperative numeric types — `FixedPoint`, `FixedVector`, `FixedMatrix`.
pub mod imperative;
pub use imperative::{FixedPoint, FixedVector, FixedMatrix};

/// TQ1.9 compact ternary operations — standalone, decoupled from FASC/routing.
/// Gated behind the `inference` feature flag.
#[cfg(feature = "inference")]
pub mod tq19;

/// Build-time Q-format configuration (FRAC_BITS and derived constants).
#[doc(hidden)] pub mod frac_config;

// Internal modules — pub for integration test access, hidden from docs
#[doc(hidden)] pub mod domains;
#[doc(hidden)] pub mod core_types;
#[doc(hidden)] pub mod universal;
#[doc(hidden)] pub mod tables;

// ============================================================================
// Internal re-exports — convenience paths for crate-internal use.
// NOT part of the public API. Use `g_math::canonical::*` instead.
// ============================================================================

#[doc(hidden)] pub use domains::binary_fixed;
#[doc(hidden)] pub use domains::symbolic;

// Binary domain types (used across internal codebase)
#[doc(hidden)] pub use domains::binary_fixed::{
    multiply_binary_i128, I256, I512, I1024, I2048,
    transcendental,
};
#[doc(hidden)] pub use domains::binary_fixed::{i256, i512, i1024};

// Decimal domain types (used in rational_conversion)
#[doc(hidden)] pub use domains::decimal_fixed::DecimalFixed;

// Rational arithmetic types (used in stack_evaluator)
#[doc(hidden)] pub use domains::symbolic::rational::{RationalNumber, OverflowDetected};

// Prime table (build-time generated, 1,145 primes up to 9,973)
#[doc(hidden)] pub use tables::prime_table;
