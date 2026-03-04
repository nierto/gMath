//! Transcendental Functions — Fixed-Point Implementations
//!
//! 18 transcendental functions across 6 engines, all at 0 ULP via tier N+1 computation.
//!
//! Dedicated engines: `exp`, `ln`, `sqrt`, `sin_cos`, `atan`, `pow`.
//! ZASC-composed: `tan`, `asin`, `acos`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `atan2`.
//!
//! | Profile    | Format    | Decimals | ULP |
//! |------------|-----------|----------|-----|
//! | Embedded   | Q64.64    | 19       | 0   |
//! | Balanced   | Q128.128  | 38       | 0   |
//! | Scientific | Q256.256  | 77       | 0   |
//!
//! Strategy: compute at tier N+1 (double precision), downscale with rounding.
//! Profile-aware compile-time table selection via `GMATH_PROFILE` env var.

pub mod exp_tier_n_plus_1;
pub mod ln_tier_n_plus_1;
pub mod pow_tier_n_plus_1;
pub mod sqrt_tier_n_plus_1;
pub mod sin_cos_tier_n_plus_1;
pub mod atan_tier_n_plus_1;
// Public API - Export profile-aware wrappers (available for all types)
pub use exp_tier_n_plus_1::{exp_binary_i128, exp_binary_i256, exp_binary_i512};
pub use ln_tier_n_plus_1::{ln_binary_i128, ln_binary_i256, ln_binary_i512};
pub use pow_tier_n_plus_1::{pow_binary_i128, pow_binary_i256, pow_binary_i512, pow_integer_i128};
pub use sqrt_tier_n_plus_1::{sqrt_binary_i128, sqrt_binary_i256, sqrt_binary_i512};
pub use sin_cos_tier_n_plus_1::{sin_binary_i128, sin_binary_i256, sin_binary_i512};
pub use sin_cos_tier_n_plus_1::{cos_binary_i128, cos_binary_i256, cos_binary_i512};
#[cfg(table_format = "q64_64")]
pub use sin_cos_tier_n_plus_1::pi_half_i128;
#[cfg(table_format = "q128_128")]
pub use sin_cos_tier_n_plus_1::pi_half_i256;
#[cfg(table_format = "q256_256")]
pub use sin_cos_tier_n_plus_1::pi_half_i512;
pub use atan_tier_n_plus_1::{atan_binary_i128, atan_binary_i256, atan_binary_i512};
pub use atan_tier_n_plus_1::{atan2_binary_i128, atan2_binary_i256, atan2_binary_i512};

// Export I1024 exp/ln/sqrt/sin/cos/atan/atan2 for tier N+1 computation (scientific profile)
#[cfg(table_format = "q256_256")]
pub use exp_tier_n_plus_1::exp_binary_i1024;
#[cfg(table_format = "q256_256")]
pub use ln_tier_n_plus_1::ln_binary_i1024;
#[cfg(table_format = "q256_256")]
pub use sqrt_tier_n_plus_1::sqrt_binary_i1024;
#[cfg(table_format = "q256_256")]
pub use sin_cos_tier_n_plus_1::{sin_binary_i1024, cos_binary_i1024, pi_half_i1024};
#[cfg(table_format = "q256_256")]
pub use atan_tier_n_plus_1::{atan_binary_i1024, atan2_binary_i1024};

// Signed Q512.512 multiply helper - used by all Q512.512 transcendental implementations
#[cfg(table_format = "q256_256")]
pub(crate) use ln_tier_n_plus_1::multiply_i1024_q512_512;

// Compute-tier dispatch functions for StackEvaluator tier N+1 integration
#[cfg(table_format = "q256_256")]
pub use sin_cos_tier_n_plus_1::{sin_compute_tier_i1024, cos_compute_tier_i1024};
#[cfg(table_format = "q128_128")]
pub use sin_cos_tier_n_plus_1::{sin_compute_tier_i512, cos_compute_tier_i512};
#[cfg(table_format = "q64_64")]
pub use sin_cos_tier_n_plus_1::{sin_compute_tier_i256, cos_compute_tier_i256};

#[cfg(table_format = "q256_256")]
pub use atan_tier_n_plus_1::{atan_compute_tier_i1024, atan2_compute_tier_i1024};
#[cfg(table_format = "q128_128")]
pub use atan_tier_n_plus_1::{atan_compute_tier_i512, atan2_compute_tier_i512};
#[cfg(table_format = "q64_64")]
pub use atan_tier_n_plus_1::{atan_compute_tier_i256, atan2_compute_tier_i256};

// ln decomposition functions are internal to ln_tier_n_plus_1.rs (not public API)

// traits.rs and exp.rs archived — ZASC uses exp_tier_n_plus_1 directly

