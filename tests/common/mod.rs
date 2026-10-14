//! Shared by the integration tests: one macro that instantiates a test
//! module at every format, so a suite written once runs at six formats in
//! one build (Q22.10, Q16.16, Q32.32, Q64.64, Q128.128, Q256.256), whatever
//! the build's profile.
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
