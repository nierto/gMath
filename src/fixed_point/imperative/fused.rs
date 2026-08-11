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
    compute_add, compute_checked_add, compute_subtract, compute_multiply, compute_divide,
    compute_negate, compute_is_zero, make_compute_int,
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

/// Fused 1/√(Σ vᵢ²) — the reciprocal norm, entirely at compute tier.
///
/// Accumulates squares, takes the square root, and forms the reciprocal all
/// at tier N+1, with one rounding at the final downscale. This is the form
/// normalization actually wants: `inv_sqrt_sum_sq(&v)` then N multiplies
/// replaces a `length()` plus N per-component divisions.
///
/// # Panics
/// Panics if all values are zero (the norm is zero, reciprocal undefined),
/// or if the reciprocal does not fit the storage tier.
pub fn inv_sqrt_sum_sq(values: &[FixedPoint]) -> FixedPoint {
    let mut acc = compute_zero();
    for v in values {
        let vc = upscale_to_compute(v.raw());
        acc = compute_add(acc, compute_multiply(vc, vc));
    }
    let s = sqrt_at_compute_tier(acc);
    let inv = compute_divide(make_compute_int(1), s)
        .expect("inv_sqrt_sum_sq: zero norm has no reciprocal");
    FixedPoint::from_raw(round_to_storage(inv))
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

/// Fused Σ (a_i − b_i)² — squared Euclidean distance at compute tier, no sqrt (U1).
///
/// The no-transcendental half of `euclidean_distance`: metric-tree scoring,
/// squared-space pruning, and Möbius-ratio numerators need the squared
/// value only, and paying a fixed-point sqrt (~15 µs at Q64.64) to
/// immediately re-square it wastes the dominant cost of the kernel.
/// Accumulates at tier N+1, single downscale at the end.
///
/// **Use case**: VP-tree proxy scoring, hyperbolic Möbius-ratio kernels,
/// any comparison that is monotone in the distance.
pub fn euclidean_distance_squared(a: &[FixedPoint], b: &[FixedPoint]) -> FixedPoint {
    assert_eq!(a.len(), b.len(), "euclidean_distance_squared: dimension mismatch");
    let mut acc = compute_zero();
    for i in 0..a.len() {
        let da = upscale_to_compute(a[i].raw());
        let db = upscale_to_compute(b[i].raw());
        let diff = compute_subtract(da, db);
        acc = compute_add(acc, compute_multiply(diff, diff));
    }
    FixedPoint::from_raw(round_to_storage(acc))
}

/// Fused Σ a_i·b_i — dot product entirely at compute tier (U1).
///
/// Accumulates products at tier N+1 width, single downscale at the end —
/// the accumulator cannot wrap the way a storage-tier fold can for large
/// coordinates or many dimensions.
///
/// **Use case**: Möbius denominators, power-diagram distances, cosine
/// numerators.
pub fn dot(a: &[FixedPoint], b: &[FixedPoint]) -> FixedPoint {
    assert_eq!(a.len(), b.len(), "dot: dimension mismatch");
    let mut acc = compute_zero();
    for i in 0..a.len() {
        let da = upscale_to_compute(a[i].raw());
        let db = upscale_to_compute(b[i].raw());
        acc = compute_add(acc, compute_multiply(da, db));
    }
    FixedPoint::from_raw(round_to_storage(acc))
}

/// Fused squared Möbius denominator `|1 − p̄q|² = 1 − 2⟨p,q⟩ + |p|²·|q|²` (U1).
///
/// The denominator of the Poincaré-disk distance ratio
/// `d(p,q) = 2·atanh(|p−q| / |1−p̄q|)`. Computing it fused keeps the
/// dot product, both squared norms, and the combination at tier N+1 with
/// a single downscale — one materialization instead of four, and no
/// intermediate can wrap.
///
/// **Use case**: hyperbolic distance/ratio kernels; combine with
/// [`euclidean_distance_squared`] for a one-sqrt exact kernel:
/// `r = √(dist² / den²)`.
pub fn mobius_denominator_sq(p: &[FixedPoint], q: &[FixedPoint]) -> FixedPoint {
    assert_eq!(p.len(), q.len(), "mobius_denominator_sq: dimension mismatch");
    let mut dot_acc = compute_zero();
    let mut p_sq = compute_zero();
    let mut q_sq = compute_zero();
    for i in 0..p.len() {
        let dp = upscale_to_compute(p[i].raw());
        let dq = upscale_to_compute(q[i].raw());
        dot_acc = compute_add(dot_acc, compute_multiply(dp, dq));
        p_sq = compute_add(p_sq, compute_multiply(dp, dp));
        q_sq = compute_add(q_sq, compute_multiply(dq, dq));
    }
    let one = compute_one();
    let two_dot = compute_add(dot_acc, dot_acc);
    // 1 − 2⟨p,q⟩ + |p|²·|q|², all at tier N+1, one downscale.
    let result = compute_add(
        compute_subtract(one, two_dot),
        compute_multiply(p_sq, q_sq),
    );
    FixedPoint::from_raw(round_to_storage(result))
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
    // If 1 + exp(−x) overflows the compute tier, exp(−x) is at the tier's
    // ceiling (x deeply negative) and silu(x) = x/(1+exp(−x)) rounds to zero
    // at every storage width. Pre-guard, the wrapped sum produced huge
    // garbage for x ≲ −30.
    let one_plus_exp = match compute_checked_add(compute_one(), exp_neg) {
        Ok(v) => v,
        Err(_) => return FixedPoint::ZERO,
    };

    if compute_is_zero(&one_plus_exp) {
        return FixedPoint::ZERO;
    }

    match compute_divide(x_compute, one_plus_exp) {
        Ok(result) => FixedPoint::from_raw(round_to_storage(result)),
        Err(_) => FixedPoint::ZERO,
    }
}

/// Fused softmax + weighted value mix, entirely at compute tier:
///
/// ```text
/// out[d] = Σⱼ softmax(scores)ⱼ · values[j][d]
/// ```
///
/// The softmax weights are **never materialized to storage tier** on the mix
/// path — they stay at compute-tier resolution through the value accumulation,
/// and only the mixed output vector is downscaled (one rounding per output
/// element). This removes the storage-tier resolution floor on attention
/// weights: with FRAC_BITS fractional bits, a materialized weight below
/// 2^-FRAC_BITS truncates to zero and its value vector vanishes from the mix
/// entirely — the cause of long-context attention starvation. Here a weight
/// of any compute-representable magnitude still contributes.
///
/// Returns `(mixed_output[dim], weights[n])`. The returned weights ARE
/// storage-quantized — they are for observers (attention recording,
/// diagnostics), not what the mix used.
///
/// **Use case**: single-query attention `softmax(Q·Kᵀ/√d) · V` — the hot path
/// of autoregressive transformer inference.
pub fn softmax_mix(
    scores: &[FixedPoint],
    values: &[&[FixedPoint]],
) -> Result<(Vec<FixedPoint>, Vec<FixedPoint>), OverflowDetected> {
    assert_eq!(
        scores.len(),
        values.len(),
        "softmax_mix: scores/values length mismatch"
    );
    if scores.is_empty() {
        return Ok((vec![], vec![]));
    }
    let dim = values[0].len();

    // Phase 1: find max at storage tier
    let mut max_raw = scores[0].raw();
    for s in &scores[1..] {
        if s.raw() > max_raw {
            max_raw = s.raw();
        }
    }
    let max_compute = upscale_to_compute(max_raw);

    // Phase 2: exp(s_i - max) at compute tier, accumulate sum. The sum is
    // Σ eⱼ (each eⱼ ≤ 1.0 at compute tier), so it only overflows for
    // astronomically large n — but check it anyway so a wrapped denominator
    // can never masquerade as a valid divisor.
    let mut exp_values: Vec<ComputeStorage> = Vec::with_capacity(scores.len());
    let mut sum = compute_zero();
    for s in scores {
        let shifted = compute_subtract(upscale_to_compute(s.raw()), max_compute);
        let e = exp_at_compute_tier(shifted);
        sum = compute_checked_add(sum, e)?;
        exp_values.push(e);
    }
    if compute_is_zero(&sum) {
        return Err(OverflowDetected::DivisionByZero);
    }

    // Phase 3: accumulate numerators at compute tier, value-row-major for
    // cache locality: num[d] = Σⱼ eⱼ · v[j][d]. This is the module's largest
    // accumulation (scaled by |v|, not bounded by 1.0 like the denominator),
    // so a long context × large activations can exceed the compute envelope —
    // use checked adds and surface TierOverflow rather than wrap silently.
    let mut num: Vec<ComputeStorage> = vec![compute_zero(); dim];
    for (j, v) in values.iter().enumerate() {
        assert_eq!(
            v.len(),
            dim,
            "softmax_mix: value row {j} has length {}, expected {dim}",
            v.len()
        );
        let e = exp_values[j];
        for d in 0..dim {
            num[d] = compute_checked_add(num[d], compute_multiply(e, upscale_to_compute(v[d].raw())))?;
        }
    }

    // Phase 4: single downscale per output element
    let mut out = Vec::with_capacity(dim);
    for n in &num {
        out.push(FixedPoint::from_raw(round_to_storage(compute_divide(*n, sum)?)));
    }

    // Phase 5: observer weights (storage-quantized, NOT used by the mix)
    let mut weights = Vec::with_capacity(scores.len());
    for e in &exp_values {
        weights.push(FixedPoint::from_raw(round_to_storage(compute_divide(*e, sum)?)));
    }

    Ok((out, weights))
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

    /// Profile-appropriate tight tolerance — at least 1 ULP representable.
    fn tight() -> FixedPoint {
        #[cfg(table_format = "q16_16")]
        { fp("0.001") }
        #[cfg(table_format = "q32_32")]
        { fp("0.000000001") }
        #[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256"))]
        { fp("0.000000001") }
    }

    #[test]
    fn test_sqrt_sum_sq_basic() {
        // sqrt(3² + 4²) = sqrt(25) = 5
        let vals = [fp("3"), fp("4")];
        let result = sqrt_sum_sq(&vals);
        let diff = (result - fp("5")).abs();
        assert!(diff < tight(), "sqrt(3²+4²) = {}, expected 5", result);
    }

    #[test]
    fn test_sqrt_sum_sq_single() {
        // sqrt(7²) = 7
        let vals = [fp("7")];
        let result = sqrt_sum_sq(&vals);
        let diff = (result - fp("7")).abs();
        assert!(diff < tight(), "sqrt(7²) = {}, expected 7", result);
    }

    #[test]
    fn test_euclidean_distance_basic() {
        // distance([0,0], [3,4]) = 5
        let a = [FixedPoint::ZERO, FixedPoint::ZERO];
        let b = [fp("3"), fp("4")];
        let dist = euclidean_distance(&a, &b);
        let diff = (dist - fp("5")).abs();
        assert!(diff < tight(), "dist([0,0],[3,4]) = {}, expected 5", dist);
    }

    #[test]
    fn test_euclidean_distance_same_point() {
        let a = [fp("1"), fp("2"), fp("3")];
        let dist = euclidean_distance(&a, &a);
        assert!(dist.is_zero() || dist.abs() < tight(),
            "distance to self should be 0, got {}", dist);
    }

    #[test]
    fn test_euclidean_distance_squared_matches_distance() {
        // dist²([0,0],[3,4]) = 25, and must equal euclidean_distance²
        // within tolerance across varied vectors.
        // Values kept small so dist² fits the narrowest profile (Q8.24
        // integer range is ±127); tolerance scales with d because the
        // d*d side re-quantizes d at storage tier (error ~ d·LSB).
        let cases: [(&[FixedPoint; 2], &[FixedPoint; 2]); 3] = [
            (&[FixedPoint::ZERO, FixedPoint::ZERO], &[fp("3"), fp("4")]),
            (&[fp("0.25"), fp("-0.5")], &[fp("-0.125"), fp("0.75")]),
            (&[fp("3"), fp("-4")], &[fp("-3"), fp("4")]),
        ];
        for (a, b) in cases {
            let sq = euclidean_distance_squared(a, b);
            let d = euclidean_distance(a, b);
            let diff = (sq - d * d).abs();
            let tol = (d.abs() + fp("1")) * tight();
            assert!(diff < tol,
                "dist_sq {} vs dist² {} diverged", sq, d * d);
        }
        let same = [fp("1"), fp("2")];
        assert!(euclidean_distance_squared(&same, &same).abs() < tight());
    }

    /// Oversized fused results must panic loudly, never wrap silently.
    /// Before the round_to_storage fix, this returned a NEGATIVE squared
    /// distance on realtime profiles. from_raw(i32::MAX) keeps the input
    /// constructible and the square out of range at every GMATH_FRAC_BITS.
    #[test]
    #[cfg(table_format = "q16_16")]
    #[should_panic(expected = "exceeds storage tier")]
    fn test_euclidean_distance_squared_overflow_panics() {
        let a = [FixedPoint::from_raw(i32::MAX)];
        let b = [FixedPoint::ZERO];
        let _ = euclidean_distance_squared(&a, &b);
    }

    #[test]
    fn test_dot_basic() {
        // ⟨(1,2,3),(4,5,6)⟩ = 32; orthogonal → 0; sign handling.
        let a = [fp("1"), fp("2"), fp("3")];
        let b = [fp("4"), fp("5"), fp("6")];
        assert!((dot(&a, &b) - fp("32")).abs() < tight());
        let e1 = [fp("1"), FixedPoint::ZERO];
        let e2 = [FixedPoint::ZERO, fp("1")];
        assert!(dot(&e1, &e2).abs() < tight());
        let c = [fp("-0.5"), fp("0.25")];
        let d = [fp("0.5"), fp("0.25")];
        // -0.25 + 0.0625 = -0.1875
        assert!((dot(&c, &d) - fp("-0.1875")).abs() < tight());
    }

    #[test]
    fn test_mobius_denominator_sq() {
        // Against the definition computed at storage tier for small inputs:
        // p=(0.3,0.4), q=(-0.2,0.5): dot=0.14, |p|²=0.25, |q|²=0.29
        // → 1 − 0.28 + 0.0725 = 0.7925
        // 0.3/0.4/0.2 are not exactly representable in binary: the inputs
        // quantize to the storage grid, so the correctly rounded result can
        // legitimately sit 1 LSB away from the rounded decimal literal at
        // coarse FRAC_BITS (seen at Q22.10). Allow 2 LSB.
        let p = [fp("0.3"), fp("0.4")];
        let q = [fp("-0.2"), fp("0.5")];
        let quant_tol = tight() + tight();
        let den = mobius_denominator_sq(&p, &q);
        assert!((den - fp("0.7925")).abs() < quant_tol,
            "mobius_denominator_sq = {}, expected 0.7925", den);
        // p = q = origin → exactly 1.
        let o = [FixedPoint::ZERO, FixedPoint::ZERO];
        assert!((mobius_denominator_sq(&o, &o) - fp("1")).abs() < tight());
        // Identical interior points: 1 − 2|p|² + |p|⁴ = (1 − |p|²)².
        let den_pp = mobius_denominator_sq(&p, &p);
        let w = fp("1") - fp("0.25");
        assert!((den_pp - w * w).abs() < quant_tol);
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
        assert!(diff < tight(), "softmax sum = {}, expected 1.0", sum);
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
    fn test_silu_deep_negative_is_zero() {
        // silu(x) = x/(1+exp(−x)) → 0 super-exponentially as x → −∞: for
        // every x past the exp saturation threshold |silu(x)| is below half
        // an LSB at all storage widths. Pre-fix, the wrapping exp downscale
        // returned huge garbage for x ≲ −30 on realtime profiles
        // (MoE expert gate values can reach ±70).
        for s in ["-30", "-40", "-70", "-100"] {
            let v = silu(fp(s));
            assert!(v.abs() < tight(), "silu({}) = {}, expected ~0", s, v);
        }
    }

    #[test]
    fn test_silu_zero() {
        // SiLU(0) = 0 / (1 + exp(0)) = 0 / 2 = 0
        let result = silu(FixedPoint::ZERO);
        assert!(result.abs() < tight(), "silu(0) = {}, expected 0", result);
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

    #[test]
    fn test_softmax_mix_one_hot() {
        // One dominant score → output ≈ that value row.
        let scores = vec![fp("20"), fp("0"), fp("0")];
        let rows = [
            vec![fp("1"), fp("2")],
            vec![fp("-5"), fp("7")],
            vec![fp("3"), fp("-3")],
        ];
        let refs: Vec<&[FixedPoint]> = rows.iter().map(|r| r.as_slice()).collect();
        let (out, w) = softmax_mix(&scores, &refs).unwrap();
        assert!((out[0] - fp("1")).abs() < fp("0.01"), "out[0] = {}", out[0]);
        assert!((out[1] - fp("2")).abs() < fp("0.01"), "out[1] = {}", out[1]);
        assert!((w[0] - fp("1")).abs() < fp("0.01"), "w[0] = {}", w[0]);
    }

    #[test]
    fn test_softmax_mix_uniform_matches_mean() {
        // Equal scores → output = mean of value rows.
        let scores = vec![FixedPoint::ZERO; 4];
        let rows = [
            vec![fp("4")],
            vec![fp("8")],
            vec![fp("-4")],
            vec![fp("0")],
        ];
        let refs: Vec<&[FixedPoint]> = rows.iter().map(|r| r.as_slice()).collect();
        let (out, _) = softmax_mix(&scores, &refs).unwrap();
        assert!((out[0] - fp("2")).abs() < fp("0.01"), "out[0] = {}", out[0]);
    }

    #[test]
    fn test_softmax_mix_agrees_with_materialized_at_short_length() {
        // At short lengths (weights well above 2^-FRAC_BITS) the fused mix
        // must closely match softmax-then-materialized-mix.
        let scores = vec![fp("1.5"), fp("0.5"), fp("-0.25"), fp("2")];
        let rows = [
            vec![fp("1"), fp("-2")],
            vec![fp("0.5"), fp("3")],
            vec![fp("-1.5"), fp("0.25")],
            vec![fp("2"), fp("1")],
        ];
        let refs: Vec<&[FixedPoint]> = rows.iter().map(|r| r.as_slice()).collect();
        let (fused_out, _) = softmax_mix(&scores, &refs).unwrap();

        let w = softmax(&scores).unwrap();
        for d in 0..2 {
            let mut acc = FixedPoint::ZERO;
            for j in 0..4 {
                acc = acc + w[j] * rows[j][d];
            }
            let diff = (fused_out[d] - acc).abs();
            assert!(
                diff < fp("0.01"),
                "fused vs materialized dim {}: {} vs {}",
                d, fused_out[d], acc
            );
        }
    }

    #[test]
    fn test_softmax_mix_survives_below_storage_floor() {
        // THE regression test for the long-context attention floor.
        // n = 3000 uniform scores → each weight = 0.000333, below HALF the
        // Q22.10 quantum (2^-11 ≈ 0.00049), so round-to-nearest storage
        // materialization sends every weight to zero → mix collapses.
        // Fused path: must recover the true mean of the value rows.
        let n = 3000;
        let scores = vec![FixedPoint::ZERO; n];
        let rows: Vec<Vec<FixedPoint>> = (0..n)
            .map(|j| vec![if j % 2 == 0 { fp("2") } else { fp("4") }])
            .collect();
        let refs: Vec<&[FixedPoint]> = rows.iter().map(|r| r.as_slice()).collect();

        // Fused path recovers the true mean (= 3.0) on EVERY profile — this is
        // the guarantee, and it is asserted unconditionally.
        let (out, _) = softmax_mix(&scores, &refs).unwrap();
        assert!(
            (out[0] - fp("3")).abs() < fp("0.05"),
            "fused mix should recover mean 3.0, got {}",
            out[0]
        );

        // The floor only bites when the storage quantum is coarser than ~1/n
        // (realtime at small FRAC_BITS, e.g. Q22.10): there the *materialized*
        // mix collapses toward zero and the fused path must strictly beat it.
        // On high-precision profiles there is no floor to survive, so this half
        // is conditional on the collapse actually occurring.
        let w = softmax(&scores).unwrap();
        let mut materialized = FixedPoint::ZERO;
        for j in 0..n {
            materialized = materialized + w[j] * rows[j][0];
        }
        if materialized.abs() < fp("0.5") {
            assert!(
                (out[0] - fp("3")).abs() < (out[0] - materialized).abs(),
                "fused mix ({}) should beat the collapsed materialized mix ({})",
                out[0],
                materialized
            );
        }
    }
}
