//! Imperative numeric types for direct computation.
//!
//! Provides `FixedPoint`, `FixedVector`, and `FixedMatrix` — Copy-able (FixedPoint)
//! or Clone-able (Vector/Matrix) types with arithmetic operators and transcendentals.

mod fixed_point;
mod fixed_vector;
mod fixed_matrix;

pub use fixed_point::FixedPoint;
pub use fixed_vector::FixedVector;
pub use fixed_matrix::FixedMatrix;

// Re-export the underlying storage type for from_raw()/raw() interop
pub use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
