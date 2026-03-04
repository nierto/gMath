//! # gMath — Multi-Domain Fixed-Point Arithmetic Library
//!
//! Zero-float, pure-Rust, consensus-safe precision arithmetic with 0 ULP
//! accuracy across 18 transcendental functions on all computation profiles.
//!
//! ## Canonical API (ZASC Pipeline)
//!
//! ```rust,no_run
//! use g_math::canonical::{gmath, evaluate};
//!
//! let result = evaluate(&(gmath("1.5") + gmath("2.5")));
//! let exp_result = evaluate(&gmath("1.0").exp());
//! ```
//!
//! ## Profiles
//!
//! Set via `GMATH_PROFILE` environment variable:
//! - `embedded` — Q64.64, scalar (19 guaranteed decimals)
//! - `performance` — Q64.64, AVX2-optimized (19 guaranteed decimals)
//! - `balanced` — Q128.128 (38 guaranteed decimals)
//! - `scientific` — Q256.256 (77 guaranteed decimals)

// Internal modules
#[doc(hidden)] pub mod deployment_profiles;
pub mod fixed_point;

// Canonical API — the single public entry point
pub use fixed_point::canonical;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
