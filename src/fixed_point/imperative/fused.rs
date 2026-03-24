//! Fused compute-tier operations — entire computation chains at tier N+1.
//!
//! Each function keeps ALL intermediates at compute tier (double width),
//! performing a single downscale at the very end. This eliminates
//! materialization boundaries that cost 1 ULP per boundary.
//!
//! **Typical use cases**:
//! - `sqrt_sum_sq`: Distance/norm computation in high-dimensional spaces
//! - `euclidean_distance`: Metric space nearest-neighbor, manifold geodesics
//! - `softmax`: Attention weight normalization in neural inference
//! - `rms_norm_factor`: Per-layer normalization in transformer architectures
//! - `silu`: Gate activation in SwiGLU MLP layers

use super::FixedPoint;
use super::linalg::{ComputeStorage, upscale_to_compute, round_to_storage};
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_add, compute_subtract, compute_multiply, compute_divide,
    compute_negate, compute_is_zero,
    sqrt_at_compute_tier, exp_at_compute_tier,
};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// Compute-tier helpers
// ============================================================================

#[inline]
fn compute_zero() -> ComputeStorage {
    upscale_to_compute(FixedPoint::ZERO.raw())
}

#[inline]
fn compute_one() -> ComputeStorage {
    upscale_to_compute(FixedPoint::one().raw())
}

// ============================================================================
// FUSED OPERATIONS
// ============================================================================

/// Fused sqrt(Σ x_i²) — norm of a slice, entirely at compute tier.
///
/// Accumulates squares at tier N+1 width, takes sqrt at compute tier,
/// single downscale at the end. Saves 1 materialization vs separate
/// `dot(x,x).sqrt()`.
///
/// **Use case**: Distance/norm in high-dimensional metric spaces.
pub fn sqrt_sum_sq(values: &[FixedPoint]) -> FixedPoint {
    let mut acc = compute_zero();
    for v in values {
        let vc = upscale_to_compute(v.raw());
        acc = compute_add(acc, compute_multiply(vc, vc));
    }
    FixedPoint::from_raw(round_to_storage(sqrt_at_compute_tier(acc)))
}

/// Fused sqrt(Σ (a_i - b_i)²) — Euclidean distance, entirely at compute tier.
///
/// Computes differences, squares, accumulates, and takes sqrt all at tier N+1.
/// Saves 2 materializations vs `(a - b).length()`.
///
/// **Use case**: Nearest-neighbor search, manifold geodesic distance.
pub fn euclidean_distance(a: &[FixedPoint], b: &[FixedPoint]) -> FixedPoint {
    assert_eq!(a.len(), b.len(), "euclidean_distance: dimension mismatch");
    let mut acc = compute_zero();
    for i in 0..a.len() {
        let da = upscale_to_compute(a[i].raw());
        let db = upscale_to_compute(b[i].raw());
        let diff = compute_subtract(da, db);
        acc = compute_add(acc, compute_multiply(diff, diff));
    }
    FixedPoint::from_raw(round_to_storage(sqrt_at_compute_tier(acc)))
}

/// Stable softmax entirely at compute tier.
///
/// Algorithm: find max → subtract max → exp → sum → divide.
/// All exp() results stay at compute tier. Single downscale per output element.
///
/// **Use case**: Attention weight normalization — O(seq_len²) per forward pass.
pub fn softmax(scores: &[FixedPoint]) -> Result<Vec<FixedPoint>, OverflowDetected> {
    if scores.is_empty() {
        return Ok(vec![]);
    }

    // Phase 1: find max at storage tier (no compute needed)
    let mut max_raw = scores[0].raw();
    for s in &scores[1..] {
        if s.raw() > max_raw {
            max_raw = s.raw();
        }
    }
    let max_compute = upscale_to_compute(max_raw);

    // Phase 2: exp(s_i - max) at compute tier, accumulate sum
    let mut exp_values: Vec<ComputeStorage> = Vec::with_capacity(scores.len());
    let mut sum = compute_zero();
    for s in scores {
        let s_compute = upscale_to_compute(s.raw());
        let shifted = compute_subtract(s_compute, max_compute);
        let e = exp_at_compute_tier(shifted);
        sum = compute_add(sum, e);
        exp_values.push(e);
    }

    // Phase 3: divide each exp by sum, single downscale per element
    if compute_is_zero(&sum) {
        return Err(OverflowDetected::DivisionByZero);
    }

    let mut result = Vec::with_capacity(scores.len());
    for e in &exp_values {
        let normalized = compute_divide(*e, sum)?;
        result.push(FixedPoint::from_raw(round_to_storage(normalized)));
    }
    Ok(result)
}

/// Fused 1/sqrt(mean(x²) + eps) — RMSNorm scaling factor at compute tier.
///
/// Computes sum of squares, divides by n, adds epsilon, takes sqrt,
/// then reciprocal — all at tier N+1. Single downscale.
///
/// **Use case**: RMSNorm — called once per layer per token in transformer inference.
pub fn rms_norm_factor(values: &[FixedPoint], eps: FixedPoint) -> Result<FixedPoint, OverflowDetected> {
    if values.is_empty() {
        return Err(OverflowDetected::DivisionByZero);
    }

    // Accumulate x² at compute tier
    let mut sum_sq = compute_zero();
    for v in values {
        let vc = upscale_to_compute(v.raw());
        sum_sq = compute_add(sum_sq, compute_multiply(vc, vc));
    }

    // mean = sum_sq / n
    let n_compute = upscale_to_compute(FixedPoint::from_int(values.len() as i32).raw());
    let mean = compute_divide(sum_sq, n_compute)?;

    // mean + eps
    let eps_compute = upscale_to_compute(eps.raw());
    let mean_eps = compute_add(mean, eps_compute);

    // 1 / sqrt(mean + eps)
    let root = sqrt_at_compute_tier(mean_eps);
    if compute_is_zero(&root) {
        return Err(OverflowDetected::DivisionByZero);
    }
    let inv = compute_divide(compute_one(), root)?;

    Ok(FixedPoint::from_raw(round_to_storage(inv)))
}

/// Fused SiLU activation: x / (1 + exp(-x)) entirely at compute tier.
///
/// SiLU = x * sigmoid(x) = x / (1 + exp(-x)).
/// Keeps exp(-x), addition, and division all at tier N+1.
///
/// **Use case**: SwiGLU gate — called per intermediate activation in MLP layers.
pub fn silu(x: FixedPoint) -> FixedPoint {
    let x_compute = upscale_to_compute(x.raw());
    let neg_x = compute_negate(x_compute);
    let exp_neg = exp_at_compute_tier(neg_x);
    let one_plus_exp = compute_add(compute_one(), exp_neg);

    if compute_is_zero(&one_plus_exp) {
        return FixedPoint::ZERO;
    }

    match compute_divide(x_compute, one_plus_exp) {
        Ok(result) => FixedPoint::from_raw(round_to_storage(result)),
        Err(_) => FixedPoint::ZERO,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(s: &str) -> FixedPoint {
        if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
        else { FixedPoint::from_str(s) }
    }

    #[test]
    fn test_sqrt_sum_sq_basic() {
        // sqrt(3² + 4²) = sqrt(25) = 5
        let vals = [fp("3"), fp("4")];
        let result = sqrt_sum_sq(&vals);
        let diff = (result - fp("5")).abs();
        assert!(diff < fp("0.000000001"), "sqrt(3²+4²) = {}, expected 5", result);
    }

    #[test]
    fn test_sqrt_sum_sq_single() {
        // sqrt(7²) = 7
        let vals = [fp("7")];
        let result = sqrt_sum_sq(&vals);
        let diff = (result - fp("7")).abs();
        assert!(diff < fp("0.000000001"), "sqrt(7²) = {}, expected 7", result);
    }

    #[test]
    fn test_euclidean_distance_basic() {
        // distance([0,0], [3,4]) = 5
        let a = [FixedPoint::ZERO, FixedPoint::ZERO];
        let b = [fp("3"), fp("4")];
        let dist = euclidean_distance(&a, &b);
        let diff = (dist - fp("5")).abs();
        assert!(diff < fp("0.000000001"), "dist([0,0],[3,4]) = {}, expected 5", dist);
    }

    #[test]
    fn test_euclidean_distance_same_point() {
        let a = [fp("1"), fp("2"), fp("3")];
        let dist = euclidean_distance(&a, &a);
        assert!(dist.is_zero() || dist.abs() < fp("0.000000001"),
            "distance to self should be 0, got {}", dist);
    }

    #[test]
    fn test_softmax_uniform() {
        // Softmax of equal values should give uniform distribution
        let scores = vec![fp("1"); 4];
        let result = softmax(&scores).unwrap();
        let expected = fp("0.25");
        for (i, w) in result.iter().enumerate() {
            let diff = (*w - expected).abs();
            assert!(diff < fp("0.001"), "softmax[{}] = {}, expected 0.25", i, w);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let scores = vec![fp("1"), fp("2"), fp("3"), fp("4")];
        let result = softmax(&scores).unwrap();
        let sum: FixedPoint = result.iter().copied().fold(FixedPoint::ZERO, |a, b| a + b);
        let diff = (sum - fp("1")).abs();
        assert!(diff < fp("0.000001"), "softmax sum = {}, expected 1.0", sum);
    }

    #[test]
    fn test_softmax_monotone() {
        // Larger input → larger output
        let scores = vec![fp("1"), fp("2"), fp("3")];
        let result = softmax(&scores).unwrap();
        assert!(result[0] < result[1], "softmax not monotone: {} >= {}", result[0], result[1]);
        assert!(result[1] < result[2], "softmax not monotone: {} >= {}", result[1], result[2]);
    }

    #[test]
    fn test_rms_norm_factor_constant() {
        // RMSNorm of constant vector [c, c, c]: 1/sqrt(c² + eps)
        let c = fp("2");
        let eps = fp("0.000001");
        let vals = vec![c; 4];
        let factor = rms_norm_factor(&vals, eps).unwrap();
        // Expected: 1/sqrt(4 + 0.000001) ≈ 1/2 = 0.5
        let diff = (factor - fp("0.5")).abs();
        assert!(diff < fp("0.001"), "rms_norm_factor = {}, expected ~0.5", factor);
    }

    #[test]
    fn test_silu_zero() {
        // SiLU(0) = 0 / (1 + exp(0)) = 0 / 2 = 0
        let result = silu(FixedPoint::ZERO);
        assert!(result.abs() < fp("0.000000001"), "silu(0) = {}, expected 0", result);
    }

    #[test]
    fn test_silu_positive() {
        // SiLU(x) ≈ x for large positive x (sigmoid ≈ 1)
        let x = fp("10");
        let result = silu(x);
        let diff = (result - x).abs();
        assert!(diff < fp("0.001"), "silu(10) = {}, expected ~10", result);
    }

    #[test]
    fn test_silu_negative() {
        // SiLU(x) ≈ 0 for large negative x (sigmoid ≈ 0)
        let result = silu(fp("-10"));
        assert!(result.abs() < fp("0.001"), "silu(-10) = {}, expected ~0", result);
    }
}
