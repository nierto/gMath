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
//!     // Feed previous result back in — full precision + shadow preserved
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
//! - **gmath()**: Entry point for static strings — zero-cost, deferred parsing
//! - **gmath_parse()**: Entry point for runtime strings — eager parsing
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
