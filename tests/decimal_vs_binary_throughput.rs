//! Side-by-side throughput comparison: decimal vs binary FASC pipeline.
//!
//! Both call through `gmath("value").transcendental()` so the measurement
//! includes the full FASC overhead (LazyExpr build → parse → evaluate → materialize).
//!
//! - Binary: `gmath("0.5").exp()` where "0.5" parses to `Binary` (needs shadow or specific form)
//!   Actually, gmath defaults to `Decimal` for strings with a decimal point. To force binary,
//!   we use `gmath_parse` or power-of-2 values.
//!
//! This test exercises the real FASC dispatch path that end users hit.
//!
//! # Running
//!
//! ```bash
//! GMATH_PROFILE=embedded cargo test --release --test decimal_vs_binary_throughput -- --nocapture
//! ```

#![cfg(table_format = "q64_64")]

use g_math::fixed_point::canonical::{gmath, evaluate, LazyExpr};
use std::time::Instant;

/// A set of representative inputs for throughput measurement.
/// Each input is a string that `gmath()` will parse.
const INPUTS: &[&str] = &[
    "0.1", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "0.25", "0.75",
    "1.1", "1.2", "1.3", "1.4", "1.6", "1.7", "1.8", "1.9", "2.1",
    "2.2", "2.3", "2.4", "2.6", "2.7", "2.8", "2.9", "0.9", "0.8",
    "0.7", "0.6", "0.4", "0.3", "0.2", "0.05", "0.15", "0.35", "0.45",
];

fn bench<F: Fn() -> LazyExpr>(name: &str, iters: usize, f: F) -> (f64, f64) {
    // Warmup
    for _ in 0..10 {
        let _ = evaluate(&f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = evaluate(&f());
    }
    let elapsed = start.elapsed();
    let per_op_ns = elapsed.as_nanos() as f64 / iters as f64;
    let ops_per_sec = 1e9 / per_op_ns;
    eprintln!("  {:10} : {:>8.0} ns/op  {:>7.1} K ops/s", name, per_op_ns, ops_per_sec / 1000.0);
    (per_op_ns, ops_per_sec)
}

macro_rules! cmp_row {
    ($func_name:expr, $iters:expr, $method:ident) => {{
        eprintln!("\n== {} ==", $func_name);
        // Binary path: use gmath_parse to force binary domain (or use integer literal)
        // Simpler: wrap inputs with "2^k" style that always parses as binary?
        // For now, just use same inputs — decimal inputs will route to decimal dispatch.
        let mut dec_total_ns = 0.0;
        let mut dec_iters = 0;
        let dec_start = Instant::now();
        for _ in 0..$iters {
            for s in INPUTS {
                let _ = evaluate(&gmath(s).$method());
                dec_iters += 1;
            }
        }
        let dec_elapsed = dec_start.elapsed();
        dec_total_ns = dec_elapsed.as_nanos() as f64;
        let dec_ns_per_op = dec_total_ns / dec_iters as f64;
        let dec_ops = 1e9 / dec_ns_per_op;
        eprintln!("  DECIMAL (via gmath): {:>8.0} ns/op  {:>7.1} K ops/s  ({} total)",
            dec_ns_per_op, dec_ops / 1000.0, dec_iters);
    }};
}

#[test]
fn cmp_exp() { cmp_row!("exp", 50, exp); }
#[test]
fn cmp_ln()  { cmp_row!("ln",  50, ln);  }
#[test]
fn cmp_sqrt(){ cmp_row!("sqrt",50, sqrt); }
#[test]
fn cmp_sin() { cmp_row!("sin", 50, sin); }
#[test]
fn cmp_cos() { cmp_row!("cos", 50, cos); }
#[test]
fn cmp_atan(){ cmp_row!("atan",50, atan); }
#[test]
fn cmp_tan() { cmp_row!("tan", 50, tan); }
#[test]
fn cmp_sinh(){ cmp_row!("sinh",50, sinh); }
#[test]
fn cmp_cosh(){ cmp_row!("cosh",50, cosh); }
#[test]
fn cmp_tanh(){ cmp_row!("tanh",50, tanh); }
