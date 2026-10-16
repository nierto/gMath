//! # gMath: Multi-Domain Fixed-Point Arithmetic Library
//!
//! Zero-float, pure-Rust, consensus-safe fixed-point arithmetic with
//! 18 transcendental functions computed at tier N+1 for full storage-tier precision.
//!
//! Upgrading from 0.4.x: 0.5.0 is more accurate, and as a side effect some
//! results end in a different final digit. This matters only if you store,
//! hash, or compare results across versions. See CHANGELOG.md.
//!
//! ## Canonical API (FASC Pipeline)
//!
//! ```rust,no_run
//! use g_math::canonical::{gmath, evaluate};
//! use g_math::fixed_point::Q64_64;
//!
//! let result = evaluate::<Q64_64>(&(gmath("1.5") + gmath("2.5")));
//! let exp_result = evaluate::<Q64_64>(&gmath("1.0").exp());
//! ```
//!
//! ## Formats
//!
//! The format is a type, `Fixed<S, F>`, chosen per value; any number of
//! formats coexist in one build:
//! - `Fixed<i32, F>` for `2 <= F <= 30` (`Q16_16` is `F = 16`)
//! - `Q32_32`: 9 decimals
//! - `Q64_64`: 19 decimals
//! - `Q128_128`: 38 decimals
//! - `Q256_256`: 77 decimals

// Internal modules
/// Unit-test helpers: a suite written once is instantiated at six formats
/// (`$suite!(module, storage, frac, word)`); `on_words!(w in [w32 w64] { .. })`
/// keeps items for the listed storage words only.
#[cfg(test)]
macro_rules! for_each_format {
    ($suite:ident) => {
        $suite!(q22_10, i32, 10, w32);
        $suite!(q16_16, i32, 16, w32);
        $suite!(q32_32, i64, 32, w64);
        $suite!(q64_64, i128, 64, w128);
        $suite!(q128_128, crate::fixed_point::I256, 128, w256);
        $suite!(q256_256, crate::fixed_point::I512, 256, w512);
        #[cfg(gmath_all_splits)] $suite!(split_q24_8, i32, 8, w32);
        #[cfg(gmath_all_splits)] $suite!(split_q20_12, i32, 12, w32);
        #[cfg(gmath_all_splits)] $suite!(split_q12_20, i32, 20, w32);
        #[cfg(gmath_all_splits)] $suite!(split_q8_24, i32, 24, w32);
    };
}

#[cfg(test)]
macro_rules! on_words {
    ($w:ident in [$($a:ident)*] { $($t:tt)* }) => { on_words!(@scan $w [$($a)*] { $($t)* }); };
    (@scan $w:ident [] { $($t:tt)* }) => {};
    (@scan w32 [w32 $($r:ident)*] { $($t:tt)* }) => { $($t)* };
    (@scan w64 [w64 $($r:ident)*] { $($t:tt)* }) => { $($t)* };
    (@scan w128 [w128 $($r:ident)*] { $($t:tt)* }) => { $($t)* };
    (@scan w256 [w256 $($r:ident)*] { $($t:tt)* }) => { $($t)* };
    (@scan w512 [w512 $($r:ident)*] { $($t:tt)* }) => { $($t)* };
    (@scan $w:ident [$h:ident $($r:ident)*] { $($t:tt)* }) => { on_words!(@scan $w [$($r)*] { $($t)* }); };
}

pub mod fixed_point;

// Canonical API — the single public entry point
pub use fixed_point::canonical;

/// Compute-tier (tier N+1) transcendentals over raw wide integers: exp, ln,
/// sqrt, sinhcosh, sigmoid, softplus, ln1p at 2×FRAC_BITS precision.
pub use fixed_point::compute_tier;

/// Q64.64 exp, ln, sin, cos, π constants and an exact decimal-literal parser.
pub use fixed_point::wide;

// gmath!() macro — available when g_math_macros is published as a separate crate.
// See g_math_macros/ directory for the proc-macro implementation.

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
