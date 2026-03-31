//! TQ1.9 throughput measurement.
//!
//! Run: GMATH_PROFILE=compact cargo test --features inference --test tq19_bench --release -- --nocapture

#![cfg(feature = "inference")]

use g_math::fixed_point::tq19::{TQ19Matrix, SCALE, tq19_dot, trit_dot};
use g_math::fixed_point::imperative::{FixedPoint, BinaryStorage};
use std::hint::black_box;
use std::time::Instant;

fn make_activations(n: usize) -> Vec<BinaryStorage> {
    (0..n).map(|i| {
        FixedPoint::from_str(&format!("0.{:04}", (i * 37 + 13) % 9999)).raw()
    }).collect()
}

fn make_weights(rows: usize, cols: usize) -> Vec<i16> {
    (0..rows * cols).map(|i| ((i as i32 * 137 - 5000) % (SCALE / 2)) as i16).collect()
}

fn make_trits(n: usize) -> Vec<i8> {
    (0..n).map(|i| match i % 3 { 0 => 1, 1 => -1, _ => 0 }).collect()
}

#[test]
fn bench_tq19_dot_4096() {
    let n = 4096;
    let weights: Vec<i16> = (0..n).map(|i| ((i * 137 + 42) % (SCALE as usize)) as i16).collect();
    let activations = make_activations(n);

    // Warmup
    for _ in 0..100 {
        let _ = tq19_dot(&weights, &activations);
    }

    let iters = 10_000;
    let start = Instant::now();
    for _ in 0..iters {
        black_box(tq19_dot(black_box(&weights), black_box(&activations)));
    }
    let elapsed = start.elapsed();
    let per_dot_ns = elapsed.as_nanos() as f64 / iters as f64;
    let dots_per_sec = 1e9 / per_dot_ns;
    let elements_per_sec = dots_per_sec * n as f64;

    eprintln!("=== tq19_dot ({n} elements) ===");
    eprintln!("  {per_dot_ns:.0} ns/dot  |  {dots_per_sec:.0} dots/s  |  {:.1}M elem/s", elements_per_sec / 1e6);
}

#[test]
fn bench_trit_dot_4096() {
    let n = 4096;
    let trits = make_trits(n);
    let activations = make_activations(n);

    for _ in 0..100 {
        let _ = trit_dot(&trits, &activations);
    }

    let iters = 10_000;
    let start = Instant::now();
    for _ in 0..iters {
        black_box(trit_dot(black_box(&trits), black_box(&activations)));
    }
    let elapsed = start.elapsed();
    let per_dot_ns = elapsed.as_nanos() as f64 / iters as f64;
    let dots_per_sec = 1e9 / per_dot_ns;

    eprintln!("=== trit_dot ({n} elements, zero-multiply) ===");
    eprintln!("  {per_dot_ns:.0} ns/dot  |  {dots_per_sec:.0} dots/s  |  {:.1}M elem/s", dots_per_sec * n as f64 / 1e6);
}

#[test]
fn bench_matvec_4096x4096() {
    let dim = 4096;
    let m = TQ19Matrix::new(dim, dim, make_weights(dim, dim));
    let activations = make_activations(dim);

    // Warmup
    let _ = m.matvec(&activations);

    let iters = 5;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = m.matvec(&activations);
    }
    let elapsed = start.elapsed();
    let per_matvec_ms = elapsed.as_millis() as f64 / iters as f64;

    // Typical large model: ~196 matvecs per token (7 projections × 28 layers)
    let tokens_per_sec = 1000.0 / (per_matvec_ms * 196.0);

    eprintln!("=== matvec {dim}×{dim} (sequential) ===");
    eprintln!("  {per_matvec_ms:.1} ms/matvec  |  ×196 = {:.1} ms/token  |  {tokens_per_sec:.2} t/s (single-thread)", per_matvec_ms * 196.0);
}

#[test]
fn bench_matvec_batch_4096x4096() {
    let dim = 4096;
    let m = TQ19Matrix::new(dim, dim, make_weights(dim, dim));
    let v1 = make_activations(dim);
    let v2 = make_activations(dim);
    let v3 = make_activations(dim);
    let v4 = make_activations(dim);

    // Sequential: 4 separate matvecs
    let start = Instant::now();
    let _ = m.matvec(&v1);
    let _ = m.matvec(&v2);
    let _ = m.matvec(&v3);
    let _ = m.matvec(&v4);
    let seq_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Batch: weight-centric
    let batch = [v1.as_slice(), v2.as_slice(), v3.as_slice(), v4.as_slice()];
    let start = Instant::now();
    let _ = m.matvec_batch(&batch);
    let batch_ms = start.elapsed().as_secs_f64() * 1000.0;

    let speedup = seq_ms / batch_ms;
    eprintln!("=== batch matvec {dim}×{dim} (batch=4) ===");
    eprintln!("  sequential: {seq_ms:.1} ms  |  batch: {batch_ms:.1} ms  |  speedup: {speedup:.2}×");
}

#[test]
fn bench_matvec_sizes() {
    // Measure across common model dimensions
    for &dim in &[512, 1024, 2048, 4096] {
        let m = TQ19Matrix::new(dim, dim, make_weights(dim, dim));
        let activations = make_activations(dim);

        let iters = if dim <= 1024 { 20 } else { 3 };
        let start = Instant::now();
        for _ in 0..iters {
            black_box(m.matvec(black_box(&activations)));
        }
        let per_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let gops = (dim as f64 * dim as f64 * 2.0) / (per_ms * 1e6); // multiply-accumulate = 2 ops

        eprintln!("  {dim}×{dim}: {per_ms:.1} ms  |  {gops:.2} GOPS");
    }
}
