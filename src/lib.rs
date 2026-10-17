//! # g_math: multi-domain fixed-point arithmetic
//!
//! Zero-float, pure-Rust, deterministic fixed-point arithmetic. The format is
//! a type, `Fixed<S, F>` (storage word `S`, `F` fraction bits), and any number
//! of formats coexist in one build. The 17 transcendental functions, canonical
//! expression trees and closed-form geometry maps are correctly rounded:
//! computed one tier wider than storage and, where that result lies too close
//! to a rounding boundary, decided at about 1000 bits.
//!
//! Upgrading from 0.6: see `docs/UPGRADING_0.7.md` in the repository (the
//! profiles became types; `FixedPoint` became a format such as `Q64_64` or
//! `Fixed<i32, 10>`). Every change is in CHANGELOG.md.
//!
//! ## Formats
//!
//! - `Fixed<i32, F>` for `2 <= F <= 30` (`Q16_16` is `F = 16`; `Fixed<i32, 10>` is Q22.10)
//! - `Q32_32` = `Fixed<i64, 32>`: about 9 decimals
//! - `Q64_64` = `Fixed<i128, 64>`: about 19 decimals
//! - `Q128_128` = `Fixed<I256, 128>`: about 38 decimals
//! - `Q256_256` = `Fixed<I512, 256>`: about 77 decimals
//!
//! ## Canonical layer (routing, one rounding per tree)
//!
//! ```rust
//! use g_math::canonical::{gmath, evaluate};
//! use g_math::fixed_point::Q64_64;
//!
//! let sum = evaluate::<Q64_64>(&(gmath("1.5") + gmath("2.5"))).unwrap();
//! assert_eq!(sum.to_string(), "4.0");
//! let e = evaluate::<Q64_64>(&gmath("1").exp()).unwrap();
//! assert!(e.to_string().starts_with("2.718281828459045235"));
//! ```
//!
//! ## Imperative layer (direct, `Copy`)
//!
//! ```rust
//! use g_math::fixed_point::{Fixed, FixedVector, Q64_64};
//!
//! type Q = Fixed<i32, 10>; // Q22.10
//! let x = Q::from_str("2").sqrt();
//! assert_eq!(x.raw(), 1448); // 1.4140625, the nearest of sqrt(2) at 10 fraction bits
//!
//! let v: FixedVector<Q64_64> = FixedVector::from_slice(&[Q64_64::from_int(3), Q64_64::from_int(4)]);
//! assert_eq!(v.length(), Q64_64::from_int(5));
//! ```

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

// The examples in the README and the guides are compiled and run as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
#[cfg(doctest)]
#[doc = include_str!("../docs/UPGRADING_0.7.md")]
struct UpgradingExamples;
#[cfg(doctest)]
#[doc = include_str!("../docs/README_IMPERATIVE.md")]
struct ImperativeGuideExamples;
#[cfg(doctest)]
#[doc = include_str!("../docs/README_PRECISION.md")]
struct PrecisionGuideExamples;
#[cfg(doctest)]
#[doc = include_str!("../docs/README_ROUTING.md")]
struct RoutingGuideExamples;
#[cfg(doctest)]
#[doc = include_str!("../docs/README_TRANSCENDENTALS.md")]
struct TranscendentalsGuideExamples;
#[cfg(doctest)]
#[doc = include_str!("../docs/README_FUSED.md")]
struct FusedGuideExamples;
#[cfg(doctest)]
#[doc = include_str!("../docs/README_LINALG.md")]
struct LinalgGuideExamples;
#[cfg(doctest)]
#[doc = include_str!("../docs/README_GEOMETRY.md")]
struct GeometryGuideExamples;
