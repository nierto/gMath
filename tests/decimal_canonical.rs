//! Integration tests, decimal_canonical: DecimalFixed, the canonical layer (parsing, routing, modes, UGOD), the ternary domain, overflow and narrowing defects, serialization, the API-gaps gate.
//!
//! Each file in tests/ is one module here (cargo `autotests = false`, one
//! binary per group instead of one per file: the generic library code is
//! compiled once per group). Run one file with
//! `cargo test --release --test decimal_canonical <file_name>::`.

#[macro_use]
#[path = "common/mod.rs"]
mod common;

#[path = "api_gaps_validation.rs"]
mod api_gaps_validation;
#[path = "arithmetic_sweep_validation.rs"]
mod arithmetic_sweep_validation;
#[path = "boundary_stress.rs"]
mod boundary_stress;
#[path = "compound_operations.rs"]
mod compound_operations;
#[path = "decimal_chain_correct_rounding_validation.rs"]
mod decimal_chain_correct_rounding_validation;
#[path = "decimal_contract_validation.rs"]
mod decimal_contract_validation;
#[path = "decimal_exp_range_validation.rs"]
mod decimal_exp_range_validation;
#[path = "decimal_fasc_integration.rs"]
mod decimal_fasc_integration;
#[path = "decimal_interval_enclosure.rs"]
mod decimal_interval_enclosure;
#[path = "decimal_rounding_mode_validation.rs"]
mod decimal_rounding_mode_validation;
#[path = "decimal_tier_by_d_validation.rs"]
mod decimal_tier_by_d_validation;
#[path = "decimal_transcendental_validation.rs"]
mod decimal_transcendental_validation;
#[path = "decimal_try_transcendentals_validation.rs"]
mod decimal_try_transcendentals_validation;
#[path = "decimal_vs_binary_throughput.rs"]
mod decimal_vs_binary_throughput;
#[path = "defects_065_validation.rs"]
mod defects_065_validation;
#[path = "domain_arithmetic_validation.rs"]
mod domain_arithmetic_validation;
#[path = "domain_matrix_validation.rs"]
mod domain_matrix_validation;
#[path = "error_handling.rs"]
mod error_handling;
#[path = "fasc_width_validation.rs"]
mod fasc_width_validation;
#[path = "fractal_router_integration.rs"]
mod fractal_router_integration;
#[path = "lazy_matrix_validation.rs"]
mod lazy_matrix_validation;
#[path = "mode_routing_validation.rs"]
mod mode_routing_validation;
#[path = "narrowing_defects_validation.rs"]
mod narrowing_defects_validation;
#[path = "operator_overflow_validation.rs"]
mod operator_overflow_validation;
#[path = "rounding_unification.rs"]
mod rounding_unification;
#[path = "s1_serialization_extended.rs"]
mod s1_serialization_extended;
#[path = "ternary_domain_validation.rs"]
mod ternary_domain_validation;
#[path = "ternary_path_equivalence.rs"]
mod ternary_path_equivalence;
#[path = "ugod_chain_persistence_test.rs"]
mod ugod_chain_persistence_test;
#[path = "ugod_promotion_validation.rs"]
mod ugod_promotion_validation;
#[path = "wide_integer_sign_semantics.rs"]
mod wide_integer_sign_semantics;
