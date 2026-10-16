//! Shared by the integration tests: one macro that instantiates a test
//! module at every format, so a suite written once runs at six formats in
//! one build (Q22.10, Q16.16, Q32.32, Q64.64, Q128.128, Q256.256); the cfgs
//! `gmath_all_splits` and `gmath_edge_splits` add further 32-bit splits.
//!
//! A suite is a `macro_rules!` taking `($m:ident, $S:ty, $F:literal)` and
//! expanding to `mod $m { ... }` with `type FixedPoint = Fixed<$S, $F>;` at
//! its top; `for_each_format!(suite)` instantiates it at the six formats.
//! A test that only makes sense at one format invokes its suite macro
//! directly with that format.
#![allow(unused_macros, dead_code, unused_imports)]

macro_rules! for_each_format {
    ($suite:ident) => {
        $suite!(q22_10, i32, 10);
        $suite!(q16_16, i32, 16);
        $suite!(q32_32, i64, 32);
        $suite!(q64_64, i128, 64);
        $suite!(q128_128, g_math::fixed_point::I256, 128);
        $suite!(q256_256, g_math::fixed_point::I512, 256);
        // the realtime splits the suite gates (RUSTFLAGS="--cfg gmath_all_splits")
        #[cfg(gmath_all_splits)] $suite!(split_q24_8, i32, 8);
        #[cfg(gmath_all_splits)] $suite!(split_q20_12, i32, 12);
        #[cfg(gmath_all_splits)] $suite!(split_q12_20, i32, 20);
        #[cfg(gmath_all_splits)] $suite!(split_q8_24, i32, 24);
        // the edge splits, for the split-generic correctness gates only
        // (RUSTFLAGS="--cfg gmath_edge_splits")
        #[cfg(gmath_edge_splits)] $suite!(split_q30_2, i32, 2);
        #[cfg(gmath_edge_splits)] $suite!(split_q28_4, i32, 4);
        #[cfg(gmath_edge_splits)] $suite!(split_q26_6, i32, 6);
        #[cfg(gmath_edge_splits)] $suite!(split_q6_26, i32, 26);
        #[cfg(gmath_edge_splits)] $suite!(split_q4_28, i32, 28);
        #[cfg(gmath_edge_splits)] $suite!(split_q2_30, i32, 30);
    };
}

/// The five storage words with their native split (no second 32-bit split).
macro_rules! for_each_word {
    ($suite:ident) => {
        $suite!(q16_16, i32, 16);
        $suite!(q32_32, i64, 32);
        $suite!(q64_64, i128, 64);
        $suite!(q128_128, g_math::fixed_point::I256, 128);
        $suite!(q256_256, g_math::fixed_point::I512, 256);
    };
}
