//! The formats and every layer built on them.
//!
//! **Formats**: [`Fixed`] and the aliases [`Q16_16`] .. [`Q256_256`], with the
//! traits [`FixedNum`], [`Storage`], [`WideInt`] and [`Scalar`]. **Layers**:
//! [`canonical`] (routing, `gmath` / `evaluate`), [`imperative`] (vectors,
//! matrices, linear algebra, geometry, fused kernels, intervals),
//! [`compute_tier`] and [`wide`]. Modules marked hidden are internal.

/// Canonical FASC entry point: the public API for gMath.
pub mod canonical;

/// Imperative numeric types: `Fixed`, `FixedVector`, `FixedMatrix`.
pub mod imperative;
pub use imperative::{FixedVector, FixedMatrix, Interval, Scalar};

// Width as a type: a format is a storage word and a number of fractional bits.
pub use fixed_num::{Compute, Fixed, FixedNum, Q16_16, Q32_32, Q64_64, Q128_128, Q256_256};
pub use width::{CircleStorage, Storage, WideInt};
/// The wide signed integers: storage words of `Q128_128` (`I256`) and
/// `Q256_256` (`I512`), compute and accumulator words above them.
pub use domains::binary_fixed::{I256, I512, I1024, I2048};

/// Public compute-tier (tier N+1) transcendentals over a format's compute word.
pub mod compute_tier;

/// Q64.64 transcendentals, π constants and an exact decimal-literal parser
/// for quantities outside the storage and compute tiers. Every format.
pub mod wide;

/// Width traits: the integer words and, per storage word, its compute word
/// and exact accumulators.
pub(crate) mod width;

/// `Fixed<S, F>` and the per-width kernels.
pub(crate) mod fixed_num;
// Correct rounding by two stages: the compute tier with bounds, then Mpf.
pub(crate) mod exact;

// Internal modules — pub for integration test access, hidden from docs
#[doc(hidden)] pub mod domains;
#[doc(hidden)] pub mod core_types;
#[doc(hidden)] pub mod universal;
#[doc(hidden)] pub mod tables;
#[doc(hidden)] pub mod router;

// ============================================================================
// Internal re-exports — convenience paths for crate-internal use.
// NOT part of the public API. Use `g_math::canonical::*` instead.
// ============================================================================

#[doc(hidden)] pub use domains::binary_fixed;
#[doc(hidden)] pub use domains::symbolic;

// Binary domain types (used across internal codebase)
#[doc(hidden)] pub use domains::binary_fixed::{multiply_binary_i128, transcendental};
#[doc(hidden)] pub use domains::binary_fixed::{i256, i512, i1024};

// Decimal domain: `g_math::fixed_point::DecimalFixed` is a supported public type
// (the `domains` module itself stays doc-hidden — this is the documented path).
pub use domains::decimal_fixed::DecimalFixed;
/// `DecimalFixed<2>` and `DecimalFixed<6>`: the money aliases.
pub use domains::decimal_fixed::{Currency, HighPrecisionCurrency};
/// Tie rule for `DecimalFixed` rounding (`try_div_with`, `try_mul_div_with`, ...).
pub use domains::decimal_fixed::DecimalRounding;
/// Certified decimal enclosures: `g_math::fixed_point::DecimalInterval` is a supported public type.
pub use domains::decimal_fixed::DecimalInterval;
/// The compute word `DecimalFixed<D>` runs its transcendentals on, and the
/// marker a generic bound names it through (`where Dp<D>: DecimalTier`).
pub use domains::decimal_fixed::transcendental::{DecimalTier, Dp};

// Rational arithmetic types (used in stack_evaluator)
#[doc(hidden)] pub use domains::symbolic::rational::{RationalNumber, OverflowDetected};

// Prime table (build-time generated, 1,145 primes up to 9,973)
#[doc(hidden)] pub use tables::prime_table;
