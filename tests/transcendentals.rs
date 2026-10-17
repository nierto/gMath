//! Integration tests, transcendentals: the transcendental engines, full-range correct rounding, canonical chains, the wide and compute tiers, the float boundary, the width identities.
//!
//! Each file in tests/ is one module here (cargo `autotests = false`, one
//! binary per group instead of one per file: the generic library code is
//! compiled once per group). Run one file with
//! `cargo test --release --test transcendentals <file_name>::`.

#[macro_use]
#[path = "common/mod.rs"]
mod common;

#[path = "chain_correct_rounding_validation.rs"]
mod chain_correct_rounding_validation;
#[path = "compact_profile_validation.rs"]
mod compact_profile_validation;
#[path = "compute_tier_validation.rs"]
mod compute_tier_validation;
#[path = "exp_full_range_validation.rs"]
mod exp_full_range_validation;
#[path = "fasc_ulp_validation.rs"]
mod fasc_ulp_validation;
#[path = "float_boundary_validation.rs"]
mod float_boundary_validation;
#[path = "fused_transcendentals_validation.rs"]
mod fused_transcendentals_validation;
#[path = "incremental_wins_validation.rs"]
mod incremental_wins_validation;
#[path = "negative_operand_battery.rs"]
mod negative_operand_battery;
#[path = "q8_24_validation.rs"]
mod q8_24_validation;
#[path = "realtime_profile_validation.rs"]
mod realtime_profile_validation;
#[path = "sinhcosh_validation.rs"]
mod sinhcosh_validation;
#[path = "sqrt_full_range_validation.rs"]
mod sqrt_full_range_validation;
#[path = "trace_asinh.rs"]
mod trace_asinh;
#[path = "trace_ln.rs"]
mod trace_ln;
#[path = "trace_sqrt.rs"]
mod trace_sqrt;
#[path = "trace_sqrt4.rs"]
mod trace_sqrt4;
#[path = "transcendental_ulp_validation.rs"]
mod transcendental_ulp_validation;
#[path = "try_direct_bypass_validation.rs"]
mod try_direct_bypass_validation;
#[path = "validation_benchmark.rs"]
mod validation_benchmark;
#[path = "wide_gates_validation.rs"]
mod wide_gates_validation;
#[path = "wide_q64_validation.rs"]
mod wide_q64_validation;
#[path = "width_types_validation.rs"]
mod width_types_validation;
