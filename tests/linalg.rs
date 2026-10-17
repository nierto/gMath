//! Integration tests, linalg: linear algebra and decompositions, fused kernels, tensors, certified intervals and exact predicates.
//!
//! Each file in tests/ is one module here (cargo `autotests = false`, one
//! binary per group instead of one per file: the generic library code is
//! compiled once per group). Run one file with
//! `cargo test --release --test linalg <file_name>::`.

#[macro_use]
#[path = "common/mod.rs"]
mod common;

#[path = "certified_geometry_refs_validation.rs"]
mod certified_geometry_refs_validation;
#[path = "decompose_adversarial.rs"]
mod decompose_adversarial;
#[path = "decompose_mpmath_validation.rs"]
mod decompose_mpmath_validation;
#[path = "decompose_ulp_measurement.rs"]
mod decompose_ulp_measurement;
#[path = "decompose_validation.rs"]
mod decompose_validation;
#[path = "decomposition_convergence_validation.rs"]
mod decomposition_convergence_validation;
#[path = "eigen_svd_schur_validation.rs"]
mod eigen_svd_schur_validation;
#[path = "exact_predicates_validation.rs"]
mod exact_predicates_validation;
#[path = "fused_ops_validation.rs"]
mod fused_ops_validation;
#[path = "interval_enclosure.rs"]
mod interval_enclosure;
#[path = "l1c_l1d_l3a_validation.rs"]
mod l1c_l1d_l3a_validation;
#[path = "layout_and_rotation_validation.rs"]
mod layout_and_rotation_validation;
#[path = "linalg_validation.rs"]
mod linalg_validation;
#[path = "one_rounding_validation.rs"]
mod one_rounding_validation;
#[path = "pd_verdict_validation.rs"]
mod pd_verdict_validation;
#[path = "tensor_validation.rs"]
mod tensor_validation;
