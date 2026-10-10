//! Fixed-Allocation Stack Computation (FASC)
//!
//! **MISSION**: Universal lazy evaluation without heap allocation
//! **ARCHITECTURE**: Stack-based computation with UGOD integration
//! **BENEFIT**: All domains share single implementation with zero runtime allocation

pub mod lazy_expr;
pub mod lazy_matrix_expr;
pub mod domain_matrix;
pub mod mode;
pub mod stack_evaluator;

// Unified public API
pub use lazy_expr::{LazyExpr, gmath, gmath_in, ConstantId};
pub use lazy_matrix_expr::{LazyMatrixExpr, evaluate_matrix, evaluate_matrix_in};
pub use domain_matrix::DomainMatrix;
pub use mode::{GmathMode, ComputeMode, OutputMode, set_mode, get_mode, reset_mode};
pub use stack_evaluator::{StackEvaluator, StackValue, FascFormat, with_evaluator, evaluate, evaluate_in, evaluate_sincos, evaluate_sincos_in, evaluate_sinhcosh, evaluate_sinhcosh_in, gmath_parse, gmath_parse_in};