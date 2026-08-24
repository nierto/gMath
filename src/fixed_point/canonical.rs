//! Canonical FASC Entry Point
//!
//! **THIS IS THE PRIMARY API FOR gMath.**
//!
//! All arithmetic flows through:
//! `gmath("value") → LazyExpr → StackEvaluator → StackValue → Domain dispatch`
//!
//! ## Quick Start
//!
//! ```rust
//! use g_math::canonical::{gmath, evaluate, LazyExpr};
//!
//! // Build expressions (zero allocation, deferred evaluation)
//! let result = gmath("1.5") + gmath("2.5");
//!
//! // Evaluate via thread-local stack evaluator
//! let value = evaluate(&result).unwrap();
//! println!("{}", value);  // "4.0000000000000000000"
//! ```
//!
//! ## Chaining Results (Compound Interest Example)
//!
//! ```rust
//! use g_math::canonical::{gmath, evaluate, LazyExpr};
//!
//! let rate = gmath("1.05"); // 5% annual rate
//! let mut balance = evaluate(&gmath("1000.00")).unwrap();
//!
//! for year in 1..=5 {
//!     // Feed previous result back in: full precision + shadow preserved
//!     balance = evaluate(&(LazyExpr::from(balance) * gmath("1.05"))).unwrap();
//!     println!("Year {}: {:.2}", year, balance);
//! }
//! ```
//!
//! ## Transcendentals
//!
//! ```rust
//! use g_math::canonical::{gmath, evaluate};
//!
//! let e = evaluate(&gmath("1.0").exp()).unwrap();     // e^1 = 2.718...
//! let ln2 = evaluate(&gmath("2.0").ln()).unwrap();    // ln(2) = 0.693...
//! let root = evaluate(&gmath("2.0").sqrt()).unwrap();  // sqrt(2) = 1.414...
//! println!("{}", e);    // "2.7182818284590452353"
//! ```
//!
//! ## Runtime Strings
//!
//! ```rust
//! use g_math::canonical::{gmath_parse, evaluate, LazyExpr};
//!
//! let user_input = String::from("3.14");
//! let parsed = gmath_parse(&user_input).unwrap();
//! let result = evaluate(&parsed.sin()).unwrap();
//! ```
//!
//! ## Architecture
//!
//! - **LazyExpr**: Expression tree builder with operator overloading
//! - **StackEvaluator**: Thread-local evaluator with fixed-size workspace
//! - **StackValue**: Domain-tagged result (Binary | Decimal | Ternary | Symbolic)
//! - **gmath()**: Entry point for static strings: zero-cost, deferred parsing
//! - **gmath_parse()**: Entry point for runtime strings: eager parsing
//! - **LazyExpr::from(StackValue)**: Chain results into new expressions (zero precision loss)

// Re-export the complete FASC API
pub use super::universal::fasc::{
    LazyExpr,
    LazyMatrixExpr,
    DomainMatrix,
    gmath,
    gmath_parse,
    ConstantId,
    StackEvaluator,
    StackValue,
    evaluate,
    evaluate_matrix,
    evaluate_sincos,
    evaluate_sinhcosh,
    GmathMode,
    ComputeMode,
    OutputMode,
};

// Re-export shadow types for public API
pub use super::universal::tier_types::{CompactShadow, ShadowConstantId};

/// Set compute:output mode. Examples: "binary:ternary", "auto:auto", "decimal:binary"
pub fn set_gmath_mode(mode_str: &str) -> Result<(), &'static str> {
    let mode = GmathMode::from_str(mode_str)?;
    super::universal::fasc::mode::set_mode(mode);
    Ok(())
}

/// Reset to default Auto:Auto mode
pub fn reset_gmath_mode() {
    super::universal::fasc::mode::reset_mode();
}

// ============================================================================
// PRE-PARSED CONSTRUCTORS (called by gmath!() proc-macro)
// ============================================================================

/// Construct a pre-parsed Decimal LazyExpr. Called by `gmath!("0.1")` expansion.
///
/// Skips runtime string parsing: dp, scaled value, and shadow are computed
/// at compile time by the proc-macro. ~60 ns faster than `gmath("0.1")`.
#[doc(hidden)]
pub fn __pre_decimal(dp: u8, scaled: i128, shadow_num: i128, shadow_den: u128) -> LazyExpr {
    use super::universal::fasc::stack_evaluator::conversion::to_binary_storage;
    let storage = to_binary_storage(scaled);
    let shadow = CompactShadow::from_rational(shadow_num, shadow_den);
    LazyExpr::Value(Box::new(StackValue::Decimal(dp, storage, shadow)))
}

/// Construct a pre-parsed Binary integer LazyExpr. Called by `gmath!("255")` expansion.
#[doc(hidden)]
pub fn __pre_integer(value: i128) -> LazyExpr {
    let shadow = CompactShadow::from_rational(value, 1);
    // Binary storage: value << frac_bits
    #[cfg(table_format = "q256_256")]
    let storage = {
        use super::binary_fixed::I512;
        I512::from_i128(value) << 256
    };
    #[cfg(table_format = "q128_128")]
    let storage = {
        use super::binary_fixed::I256;
        I256::from_i128(value) << 128
    };
    #[cfg(table_format = "q64_64")]
    let storage = value << 64;
    #[cfg(table_format = "q32_32")]
    let storage = (value as i64) << 32;
    #[cfg(table_format = "q16_16")]
    let storage = {
        use super::frac_config;
        (value as i32) << frac_config::FRAC_BITS
    };

    // Tier = profile's max binary tier
    let tier = {
        #[cfg(table_format = "q16_16")] { 1u8 }
        #[cfg(table_format = "q32_32")] { 2u8 }
        #[cfg(table_format = "q64_64")] { 3u8 }
        #[cfg(table_format = "q128_128")] { 4u8 }
        #[cfg(table_format = "q256_256")] { 5u8 }
    };

    LazyExpr::Value(Box::new(StackValue::Binary(tier, storage, shadow)))
}
