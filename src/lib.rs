//! # gMath — Multi-Domain Fixed-Point Arithmetic Library
//!
//! Zero-float, pure-Rust, consensus-safe fixed-point arithmetic with
//! 18 transcendental functions computed at tier N+1 for full storage-tier precision.
//!
//! ## Canonical API (FASC Pipeline)
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
//! Default is `embedded`. Set `GMATH_PROFILE` for higher precision:
//! - `embedded` (default) — Q64.64, 19 decimals
//! - `balanced` — Q128.128, 38 decimals
//! - `scientific` — Q256.256, 77 decimals

// Internal modules
#[doc(hidden)] pub mod deployment_profiles;
pub mod fixed_point;

// Canonical API — the single public entry point
pub use fixed_point::canonical;

/// TQ1.9 compact ternary operations — standalone module for inference and signal processing.
pub use fixed_point::tq19;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
