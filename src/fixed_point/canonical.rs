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
//! use g_math::fixed_point::Q64_64;
//!
//! // Build expressions (zero allocation, deferred evaluation)
//! let result = gmath("1.5") + gmath("2.5");
//!
//! // Evaluate in a format (the thread-local evaluator for that format)
//! let value = evaluate::<Q64_64>(&result).unwrap();
//! println!("{}", value);  // "4.0000000000000000000"
//! ```
//!
//! ## Chaining Results (Compound Interest Example)
//!
//! ```rust
//! use g_math::canonical::{gmath, evaluate, LazyExpr};
//! use g_math::fixed_point::Q64_64;
//!
//! let rate = gmath::<Q64_64>("1.05"); // 5% annual rate
//! let mut balance = evaluate::<Q64_64>(&gmath("1000.00")).unwrap();
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
//! use g_math::fixed_point::Q64_64;
//!
//! let e = evaluate::<Q64_64>(&gmath("1.0").exp()).unwrap();     // e^1 = 2.718...
//! let ln2 = evaluate::<Q64_64>(&gmath("2.0").ln()).unwrap();    // ln(2) = 0.693...
//! let root = evaluate::<Q64_64>(&gmath("2.0").sqrt()).unwrap();  // sqrt(2) = 1.414...
//! println!("{}", e);    // "2.7182818284590452353"
//! ```
//!
//! ## Runtime Strings
//!
//! ```rust
//! use g_math::canonical::{gmath_parse, evaluate, LazyExpr};
//!
//! let user_input = String::from("3.14");
//! let parsed = gmath_parse::<g_math::fixed_point::Q64_64>(&user_input).unwrap();
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
    with_evaluator,
    FascFormat,
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
pub fn __pre_decimal<T: FascFormat>(dp: u8, scaled: i128, shadow_num: i128, shadow_den: u128) -> LazyExpr<T> {
    use super::domains::symbolic::rational::RationalNumber;
    let shadow = CompactShadow::from_rational(shadow_num, shadow_den);
    match T::try_to_binary_storage(scaled) {
        Ok(storage) => LazyExpr::Value(Box::new(StackValue::Decimal(dp, storage, shadow))),
        // beyond the format's decimal storage: the exact rational, as the
        // runtime parser does (this narrowed with a wrapping cast)
        Err(_) => LazyExpr::Value(Box::new(StackValue::Symbolic(
            RationalNumber::new(scaled, 10u128.pow(dp as u32)),
        ))),
    }
}

/// Construct a pre-parsed Binary integer LazyExpr. Called by `gmath!("255")` expansion.
#[doc(hidden)]
pub fn __pre_integer<T: FascFormat>(value: i128) -> LazyExpr<T> {
    use super::domains::symbolic::rational::RationalNumber;
    let shadow = CompactShadow::from_rational(value, 1);
    match T::integer_value(value) {
        Some(storage) => LazyExpr::Value(Box::new(StackValue::Binary(T::MAX_BINARY_TIER, storage, shadow))),
        // beyond the integer range of the storage word: the exact rational,
        // as the runtime parser does (this shifted with a wrapping cast)
        None => LazyExpr::Value(Box::new(StackValue::Symbolic(RationalNumber::from_ratio(value, 1)))),
    }
}
