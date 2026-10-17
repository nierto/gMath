//! Integration tests, geometry: ODE integrators, curvature, manifolds, Lie groups, projective geometry, fiber bundles.
//!
//! Each file in tests/ is one module here (cargo `autotests = false`, one
//! binary per group instead of one per file: the generic library code is
//! compiled once per group). Run one file with
//! `cargo test --release --test geometry <file_name>::`.

#[macro_use]
#[path = "common/mod.rs"]
mod common;

#[path = "curvature_compute_tier_validation.rs"]
mod curvature_compute_tier_validation;
#[path = "geometry_correct_rounding_validation.rs"]
mod geometry_correct_rounding_validation;
#[path = "l1d_l3a_ulp_measurement.rs"]
mod l1d_l3a_ulp_measurement;
#[path = "l2a_l5a_ulp_measurement.rs"]
mod l2a_l5a_ulp_measurement;
#[path = "l2a_ode_validation.rs"]
mod l2a_ode_validation;
#[path = "l3b_curvature_validation.rs"]
mod l3b_curvature_validation;
#[path = "l3c_spd_grassmannian_validation.rs"]
mod l3c_spd_grassmannian_validation;
#[path = "l4a_lie_group_validation.rs"]
mod l4a_lie_group_validation;
#[path = "l4a_mpmath_validation.rs"]
mod l4a_mpmath_validation;
#[path = "l4b_projective_validation.rs"]
mod l4b_projective_validation;
#[path = "l5a_fiber_bundle_validation.rs"]
mod l5a_fiber_bundle_validation;
#[path = "lie_fiber_compute_tier_validation.rs"]
mod lie_fiber_compute_tier_validation;
#[path = "manifold_compute_tier_validation.rs"]
mod manifold_compute_tier_validation;
#[path = "ode_compute_state_validation.rs"]
mod ode_compute_state_validation;
#[path = "projective_tensor_compute_tier_validation.rs"]
mod projective_tensor_compute_tier_validation;
