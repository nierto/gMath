//! Imperative numeric types for direct computation.
//!
//! Provides `FixedPoint`, `FixedVector`, and `FixedMatrix`: Copy-able (FixedPoint)
//! or Clone-able (Vector/Matrix) types with arithmetic operators and transcendentals.

mod fixed_point;
mod fixed_vector;
mod fixed_matrix;
pub(crate) mod linalg;
#[doc(hidden)]
pub mod compute_matrix;
pub(crate) mod wide_matrix;
pub mod fused;
pub mod decompose;
pub mod derived;
pub mod matrix_functions;
pub mod manifold;
pub mod lie_group;
pub mod tensor;
pub mod tensor_decompose;
pub mod ode;
pub mod curvature;
pub mod projective;
pub mod fiber_bundle;
pub mod interval;
pub mod predicates;
pub(crate) mod kernels;
pub(crate) mod decimal_literal;
mod serialization;
mod scalar;

pub use fixed_point::FixedPoint;
pub use fixed_point::OverflowDetected;
pub use fixed_vector::FixedVector;
pub use fixed_matrix::FixedMatrix;
pub use interval::Interval;
pub use scalar::Scalar;

// SCAFFOLDING (width-types phase 7): the build's profile, read by the tests only; removed in 7c.
pub use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;

// Re-export ManifoldPoint for wire transport
pub use serialization::ManifoldPoint;
pub use serialization::{
    MANIFOLD_TAG_EUCLIDEAN, MANIFOLD_TAG_SPHERE, MANIFOLD_TAG_HYPERBOLIC,
    MANIFOLD_TAG_SPD, MANIFOLD_TAG_GRASSMANNIAN,
};
