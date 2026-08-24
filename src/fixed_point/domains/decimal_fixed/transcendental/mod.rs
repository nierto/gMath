//! Decimal domain transcendental functions: native decimal compute-tier implementation.
//!
//! # Architecture
//!
//! Mirrors the binary transcendental architecture but with decimal scaling (value × 10^-dp)
//! throughout. No binary Q-format contamination at any stage.
//!
//! ## Storage vs Compute
//!
//! - **Storage tier**: decimal value at user-specified dp, held in `BinaryStorage` type.
//! - **Compute tier**: decimal value at `DECIMAL_COMPUTE_DP` (≈ 2× storage dp), held in
//!   the profile's `ComputeStorage` type.
//! - Per-profile compute dp: realtime=9, compact=19, embedded=38, balanced=77, scientific=154.
//!
//! ## Iterative Taylor Strategy
//!
//! Unlike binary transcendentals which use large precomputed tables (because binary shifts
//! are "free"), decimal Taylor evaluation computes each term from the previous one:
//!
//! ```text
//! term_k = term_{k-1} × x / k    (exp series)
//! term_k = -term_{k-1} × x² / ((2k)(2k+1))    (sin series)
//! ```
//!
//! This avoids ANY precomputed table. The only "constants" are π (via Machin's formula
//! using our own decimal atan) and ln(10) (via 2·atanh series).
//!
//! ## Validation
//!
//! Every engine is validated against mpmath 50-digit reference values in
//! `tests/decimal_transcendental_validation.rs`.

pub mod decimal_compute;
pub mod exp;
pub mod ln;
pub mod sqrt;
pub mod sin_cos;
pub mod atan;

pub use decimal_compute::{
    DECIMAL_COMPUTE_DP,
    DECIMAL_STORAGE_MAX_DP,
    decimal_compute_scale,
    decimal_compute_zero,
    decimal_compute_one,
    decimal_compute_from_int,
    decimal_compute_add,
    decimal_compute_sub,
    decimal_compute_neg,
    decimal_compute_mul,
    decimal_compute_div,
    decimal_compute_div_int,
    decimal_compute_halve,
    decimal_compute_is_zero,
    decimal_compute_is_negative,
    decimal_upscale_to_compute,
    decimal_downscale_to_storage,
    decimal_compute_abs,
    decimal_compute_cmp,
    i128_upscale_to_compute,
    decimal_compute_to_i128,
};

pub use exp::{decimal_exp, decimal_sinhcosh};
pub use ln::decimal_ln;
pub use sqrt::decimal_sqrt;
pub use sin_cos::{decimal_sin, decimal_cos, decimal_sincos, pi_at_decimal_compute};
pub use atan::{decimal_atan, decimal_atan2};
