//! Zero-Allocation Stack Computation (ZASC)
//!
//! **MISSION**: Universal lazy evaluation without heap allocation
//! **ARCHITECTURE**: Stack-based computation with UGOD integration
//! **BENEFIT**: All domains share single implementation with zero runtime allocation

pub mod lazy_expr;
pub mod mode;
pub mod stack_evaluator;

// Unified public API
pub use lazy_expr::{LazyExpr, gmath, ConstantId};
pub use mode::{GmathMode, ComputeMode, OutputMode, set_mode, get_mode, reset_mode};
pub use stack_evaluator::{StackEvaluator, StackValue, evaluate, gmath_parse};