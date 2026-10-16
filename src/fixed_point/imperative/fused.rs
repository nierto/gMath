//! Fused compute-tier operations: entire computation chains at tier N+1.
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

use super::{FixedMatrix, FixedVector};
use super::kernels;
use crate::fixed_point::fixed_num::{Compute, Fixed, FixedNum};
use crate::fixed_point::width::{Storage, WideInt};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// Compute-tier helpers
// ============================================================================

/// Checked compute-tier sum for the infallible fused operations: a sum
/// beyond the compute tier panics instead of wrapping (plain additions
/// wrapped silently in release builds before 0.6.4).
#[inline]
fn sum_add<S: Storage, const F: u32>(a: Compute<Fixed<S, F>>, b: Compute<Fixed<S, F>>) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::<S, F>::compute_checked_add(a, b).expect("fused: sum exceeds the compute tier")
}

#[inline]
fn compute_zero<S: Storage, const F: u32>() -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::<S, F>::upscale_to_compute(Fixed::ZERO.raw())
}

#[inline]
fn compute_one<S: Storage, const F: u32>() -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::<S, F>::upscale_to_compute(Fixed::one().raw())
}

// ============================================================================
// FUSED OPERATIONS
// ============================================================================

/// Fused sqrt(Σ x_i²): norm of a slice, entirely at compute tier.
///
/// Accumulates squares at tier N+1 width, takes sqrt at compute tier,
/// single downscale at the end. Saves 1 materialization vs separate
/// `dot(x,x).sqrt()`.
///
/// **Use case**: Distance/norm in high-dimensional metric spaces.
pub fn sqrt_sum_sq<S: Storage, const F: u32>(values: &[Fixed<S, F>]) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let mut acc = compute_zero::<S, F>();
    for v in values {
        let vc = Fixed::<S, F>::upscale_to_compute(v.raw());
        acc = sum_add(acc, Fixed::<S, F>::compute_multiply(vc, vc));
    }
    Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(Fixed::<S, F>::sqrt_at_compute_tier(acc)))
}

/// Fused 1/√(Σ vᵢ²): the reciprocal norm, entirely at compute tier.
///
/// Accumulates squares, takes the square root, and forms the reciprocal all
/// at tier N+1, with one rounding at the final downscale. This is the form
/// normalization actually wants: `inv_sqrt_sum_sq(&v)` then N multiplies
/// replaces a `length()` plus N per-component divisions.
///
/// # Panics
/// Panics if all values are zero (the norm is zero, reciprocal undefined),
/// or if the reciprocal does not fit the storage tier.
pub fn inv_sqrt_sum_sq<S: Storage, const F: u32>(values: &[Fixed<S, F>]) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let mut acc = compute_zero::<S, F>();
    for v in values {
        let vc = Fixed::<S, F>::upscale_to_compute(v.raw());
        acc = sum_add(acc, Fixed::<S, F>::compute_multiply(vc, vc));
    }
    let s = Fixed::<S, F>::sqrt_at_compute_tier(acc);
    let inv = Fixed::<S, F>::compute_divide(Fixed::<S, F>::make_compute_int(1), s)
        .expect("inv_sqrt_sum_sq: zero norm has no reciprocal");
    Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(inv))
}

/// Fused sqrt(Σ (a_i - b_i)²): Euclidean distance, entirely at compute tier.
///
/// Computes differences, squares, accumulates, and takes sqrt all at tier N+1.
/// Saves 2 materializations vs `(a - b).length()`.
///
/// **Use case**: Nearest-neighbor search, manifold geodesic distance.
pub fn euclidean_distance<S: Storage, const F: u32>(a: &[Fixed<S, F>], b: &[Fixed<S, F>]) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(a.len(), b.len(), "euclidean_distance: dimension mismatch");
    let mut acc = compute_zero::<S, F>();
    for i in 0..a.len() {
        let da = Fixed::<S, F>::upscale_to_compute(a[i].raw());
        let db = Fixed::<S, F>::upscale_to_compute(b[i].raw());
        let diff = Fixed::<S, F>::compute_subtract(da, db);
        acc = sum_add(acc, Fixed::<S, F>::compute_multiply(diff, diff));
    }
    Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(Fixed::<S, F>::sqrt_at_compute_tier(acc)))
}

/// Fused Σ (a_i − b_i)²: squared Euclidean distance at compute tier, no sqrt (U1).
///
/// The no-transcendental half of `euclidean_distance`: metric-tree scoring,
/// squared-space pruning, and Möbius-ratio numerators need the squared
/// value only, and paying a fixed-point sqrt (~15 µs at Q64.64) to
/// immediately re-square it wastes the dominant cost of the kernel.
/// Accumulates at tier N+1, single downscale at the end.
///
/// **Use case**: VP-tree proxy scoring, hyperbolic Möbius-ratio kernels,
/// any comparison that is monotone in the distance.
pub fn euclidean_distance_squared<S: Storage, const F: u32>(a: &[Fixed<S, F>], b: &[Fixed<S, F>]) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(a.len(), b.len(), "euclidean_distance_squared: dimension mismatch");
    let mut acc = compute_zero::<S, F>();
    for i in 0..a.len() {
        let da = Fixed::<S, F>::upscale_to_compute(a[i].raw());
        let db = Fixed::<S, F>::upscale_to_compute(b[i].raw());
        let diff = Fixed::<S, F>::compute_subtract(da, db);
        acc = sum_add(acc, Fixed::<S, F>::compute_multiply(diff, diff));
    }
    Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(acc))
}

/// Fused Σ a_i·b_i: dot product entirely at compute tier (U1).
///
/// Accumulates products at tier N+1 width, single downscale at the end:
/// the accumulator cannot wrap the way a storage-tier fold can for large
/// coordinates or many dimensions.
///
/// **Use case**: Möbius denominators, power-diagram distances, cosine
/// numerators.
pub fn dot<S: Storage, const F: u32>(a: &[Fixed<S, F>], b: &[Fixed<S, F>]) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(a.len(), b.len(), "dot: dimension mismatch");
    let mut acc = compute_zero::<S, F>();
    for i in 0..a.len() {
        let da = Fixed::<S, F>::upscale_to_compute(a[i].raw());
        let db = Fixed::<S, F>::upscale_to_compute(b[i].raw());
        acc = sum_add(acc, Fixed::<S, F>::compute_multiply(da, db));
    }
    Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(acc))
}

/// Fused quadratic form `v^T M v` with one rounding: every term is an exact triple product and the sum is narrowed once, to nearest.
///
/// The two-stage form (`M v` rounded to storage, then `v . (M v)` rounded
/// again) rounds twice and its error grows with `sum_i |v_i|`. Here every
/// `v_i m_ij v_j` is an exact triple product at `3 * FRAC_BITS` fractional
/// bits on the profile's widest accumulator, the sum is exact and checked,
/// and the single narrowing rounds to nearest with ties toward positive
/// infinity, the binary house rule. The result is the correctly rounded
/// value of `v^T M v` for the stored operands, and it always lies inside
/// `Interval::quadratic_form(v, m)`, which narrows the same exact value
/// outward.
///
/// **Use case**: Mahalanobis distances and metric-tensor scores where the
/// verdict is taken on the scalar and must agree with its certificate.
///
/// Panics if `m` is not square or its size differs from `v`, and on storage
/// overflow; [`try_quadratic_form`] returns `TierOverflow` instead.
pub fn quadratic_form<S: Storage, const F: u32>(v: &FixedVector<Fixed<S, F>>, m: &FixedMatrix<Fixed<S, F>>) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    try_quadratic_form(v, m).expect("quadratic_form: storage overflow")
}

/// Fallible twin of [`quadratic_form`]: `Err(TierOverflow)` where the result leaves the storage tier.
pub fn try_quadratic_form<S: Storage, const F: u32>(v: &FixedVector<Fixed<S, F>>, m: &FixedMatrix<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Ok(Fixed::from_raw(kernels::narrow_triple_nearest::<Fixed<S, F>>(kernels::quadratic_form_exact(Fixed::raw_slice(v.as_slice()), m)?)?))
}

/// Fused squared Möbius denominator `|1 − p̄q|² = 1 − 2⟨p,q⟩ + |p|²·|q|²` (U1).
///
/// The denominator of the Poincaré-disk distance ratio
/// `d(p,q) = 2·atanh(|p−q| / |1−p̄q|)`. Computing it fused keeps the
/// dot product, both squared norms, and the combination at tier N+1 with
/// a single downscale; one materialization instead of four, and no
/// intermediate can wrap.
///
/// **Use case**: hyperbolic distance/ratio kernels; combine with
/// [`euclidean_distance_squared`] for a one-sqrt exact kernel:
/// `r = √(dist² / den²)`.
pub fn mobius_denominator_sq<S: Storage, const F: u32>(p: &[Fixed<S, F>], q: &[Fixed<S, F>]) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(p.len(), q.len(), "mobius_denominator_sq: dimension mismatch");
    let mut dot_acc = compute_zero::<S, F>();
    let mut p_sq = compute_zero::<S, F>();
    let mut q_sq = compute_zero::<S, F>();
    for i in 0..p.len() {
        let dp = Fixed::<S, F>::upscale_to_compute(p[i].raw());
        let dq = Fixed::<S, F>::upscale_to_compute(q[i].raw());
        dot_acc = sum_add(dot_acc, Fixed::<S, F>::compute_multiply(dp, dq));
        p_sq = sum_add(p_sq, Fixed::<S, F>::compute_multiply(dp, dp));
        q_sq = sum_add(q_sq, Fixed::<S, F>::compute_multiply(dq, dq));
    }
    let one = compute_one::<S, F>();
    let two_dot = sum_add(dot_acc, dot_acc);
    // 1 − 2⟨p,q⟩ + |p|²·|q|², all at tier N+1, one downscale.
    let result = sum_add(
        Fixed::<S, F>::compute_subtract(one, two_dot),
        Fixed::<S, F>::compute_multiply(p_sq, q_sq),
    );
    Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(result))
}

/// Stable softmax entirely at compute tier.
///
/// Algorithm: find max → subtract max → exp → sum → divide.
/// All exp() results stay at compute tier. Single downscale per output element.
///
/// **Use case**: Attention weight normalization: O(seq_len²) per forward pass.
///
/// # Errors
/// - `TierOverflow`: the sum of the exponentials leaves the compute tier. Each
///   term is at most 1, so this needs `n >= 2^(63 - 2 * FRAC_BITS)` scores on
///   the realtime profile (about 8.8e12 at Q22.10, 2^31 at Q16.16, 32768 at
///   Q8.24) and cannot happen on wider profiles.
/// - No other error is reachable: the largest term is `exp(0) = 1`, so the
///   sum is never zero and every weight is at most 1.
///
/// An empty input returns an empty vector.
pub fn softmax<S: Storage, const F: u32>(scores: &[Fixed<S, F>]) -> Result<Vec<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
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
    let max_compute = Fixed::<S, F>::upscale_to_compute(max_raw);

    // Phase 2: exp(s_i - max) at compute tier, accumulate sum
    let mut exp_values: Vec<Compute<Fixed<S, F>>> = Vec::with_capacity(scores.len());
    let mut sum = compute_zero::<S, F>();
    for s in scores {
        let s_compute = Fixed::<S, F>::upscale_to_compute(s.raw());
        let shifted = Fixed::<S, F>::compute_subtract(s_compute, max_compute);
        let e = Fixed::<S, F>::exp_at_compute_tier(shifted);
        sum = Fixed::<S, F>::compute_checked_add(sum, e)?;
        exp_values.push(e);
    }

    // Phase 3: divide each exp by sum, single downscale per element
    if Fixed::<S, F>::compute_is_zero(&sum) {
        return Err(OverflowDetected::DivisionByZero);
    }

    let mut result = Vec::with_capacity(scores.len());
    for e in &exp_values {
        let normalized = Fixed::<S, F>::compute_divide(*e, sum)?;
        result.push(Fixed::from_raw(Fixed::<S, F>::downscale_to_storage(normalized)?));
    }
    Ok(result)
}

/// Fused 1/sqrt(mean(x²) + eps): RMSNorm scaling factor at compute tier.
///
/// Computes sum of squares, divides by n, adds epsilon, takes sqrt,
/// then reciprocal: all at tier N+1. Single downscale.
///
/// `eps` is a storage-tier value, so any epsilon below the storage
/// resolution `2^-FRAC_BITS` is zero before it is added (on the realtime
/// profile at Q22.10 both `1e-5` and `1e-6` are zero). Use
/// [`rms_norm_factor_eps_wide`] to apply a small epsilon at the compute tier.
///
/// **Use case**: RMSNorm: called once per layer per token in transformer inference.
///
/// # Errors
/// The same conditions as [`rms_norm_factor_eps_wide`], except that a storage
/// epsilon always fits the compute tier.
pub fn rms_norm_factor<S: Storage, const F: u32>(values: &[Fixed<S, F>], eps: Fixed<S, F>) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    rms_norm_factor_at_compute(values, Fixed::<S, F>::upscale_to_compute(eps.raw()))
}

/// Fused 1/sqrt(mean(x²) + eps) with `eps` given in Q64.64 (`eps * 2^64`).
///
/// Identical to [`rms_norm_factor`] except that epsilon enters at the
/// compute tier (`2 × FRAC_BITS` fractional bits) instead of the storage
/// tier. The Q64.64 value is rounded to the compute tier once, to nearest
/// with ties toward +infinity; it is exact on profiles whose compute tier
/// has at least 64 fractional bits (compact and wider). On the realtime
/// profile the compute tier holds `2 × FRAC_BITS` bits (20 at Q22.10,
/// 32 at Q16.16), so an epsilon below `2^-(2 × FRAC_BITS + 1)` still rounds
/// to zero there, and a small epsilon carries the compute tier's resolution
/// (at Q22.10, `1e-5` becomes `10 / 2^20`).
///
/// # Errors
/// - `DivisionByZero`: `values` is empty, or `mean + eps` is zero at the
///   compute tier (all-zero input with an epsilon that rounds to zero).
/// - `DomainError`: `mean + eps` is negative (only possible with a negative
///   epsilon).
/// - `TierOverflow`: `eps` does not fit the compute tier (realtime only); the
///   sum of squares or `mean + eps` leaves the compute tier (the sum holds
///   `n * x^2` with `2 * FRAC_BITS` fractional bits, so on the realtime
///   profile `n * max|x|^2` must stay below `2^(63 - 2 * FRAC_BITS)`); or the
///   result `1 / sqrt(mean + eps)` exceeds the storage range.
///
/// `g_math::wide::try_from_str("1e-5", 64)` produces the Q64.64 epsilon
/// from a config literal without floats.
pub fn rms_norm_factor_eps_wide<S: Storage, const F: u32>(values: &[Fixed<S, F>], eps_q64: i128) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    rms_norm_factor_at_compute(values, q64_to_compute(eps_q64)?)
}

/// A Q64.64 value at the compute tier: nearest, ties toward +infinity, when
/// the compute tier has fewer than 64 fractional bits; exact otherwise.
fn q64_to_compute<S: Storage, const F: u32>(x: i128) -> Result<Compute<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    // the compute word carries 2F fraction bits: round to nearest when that
    // is below 64 (ties toward +inf), shift up exactly otherwise
    let cf = 2 * F;
    if cf < 64 {
        let shift = 64 - cf;
        let rounded = (x >> shift) + ((x >> (shift - 1)) & 1);
        Compute::<Fixed<S, F>>::from_i128_checked(rounded).ok_or(OverflowDetected::TierOverflow)
    } else {
        Ok(Compute::<Fixed<S, F>>::from_i128(x).shl_bits(cf - 64))
    }
}

fn rms_norm_factor_at_compute<S: Storage, const F: u32>(values: &[Fixed<S, F>], eps_compute: Compute<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if values.is_empty() {
        return Err(OverflowDetected::DivisionByZero);
    }

    // Accumulate x² at compute tier
    let mut sum_sq = compute_zero::<S, F>();
    for v in values {
        let vc = Fixed::<S, F>::upscale_to_compute(v.raw());
        sum_sq = Fixed::<S, F>::compute_checked_add(sum_sq, Fixed::<S, F>::compute_multiply(vc, vc))?;
    }

    // mean = sum_sq / n
    // mean = sum_sq / n, truncated as the compute-tier division truncates
    // (bit-identical to dividing by from_int(n)); the count is never a
    // storage value, so a realtime length past 2^(31 - F) no longer wraps
    let mean = Fixed::<S, F>::compute_div_count(sum_sq, values.len())?;

    // mean + eps
    let mean_eps = Fixed::<S, F>::compute_checked_add(mean, eps_compute)?;

    // 1 / sqrt(mean + eps); a negative radicand (negative eps) is a domain
    // error: the sqrt kernel answers it with a sentinel, not a value
    if Fixed::<S, F>::compute_is_negative(&mean_eps) {
        return Err(OverflowDetected::DomainError);
    }
    let root = Fixed::<S, F>::sqrt_at_compute_tier(mean_eps);
    if Fixed::<S, F>::compute_is_zero(&root) {
        return Err(OverflowDetected::DivisionByZero);
    }
    let inv = Fixed::<S, F>::compute_divide(compute_one::<S, F>(), root)?;

    // Err(TierOverflow), not a panic, when 1/sqrt(mean + eps) exceeds storage
    // (1/sqrt(1e-6) = 1000 is beyond the realtime range past 23 fraction bits)
    Ok(Fixed::from_raw(Fixed::<S, F>::downscale_to_storage(inv)?))
}

/// Fused SiLU activation: x / (1 + exp(-x)) entirely at compute tier.
///
/// SiLU = x * sigmoid(x) = x / (1 + exp(-x)).
/// Keeps exp(-x), addition, and division all at tier N+1.
///
/// **Use case**: SwiGLU gate: called per intermediate activation in MLP layers.
pub fn silu<S: Storage, const F: u32>(x: Fixed<S, F>) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let x_compute = Fixed::<S, F>::upscale_to_compute(x.raw());
    let neg_x = Fixed::<S, F>::compute_negate(x_compute);
    let exp_neg = Fixed::<S, F>::exp_at_compute_tier(neg_x);
    // If 1 + exp(−x) overflows the compute tier, exp(−x) is at the tier's
    // ceiling (x deeply negative) and silu(x) = x/(1+exp(−x)) rounds to zero
    // at every storage width. Pre-guard, the wrapped sum produced huge
    // garbage for x ≲ −30.
    let one_plus_exp = match Fixed::<S, F>::compute_checked_add(compute_one::<S, F>(), exp_neg) {
        Ok(v) => v,
        Err(_) => return Fixed::ZERO,
    };

    if Fixed::<S, F>::compute_is_zero(&one_plus_exp) {
        return Fixed::ZERO;
    }

    match Fixed::<S, F>::compute_divide(x_compute, one_plus_exp) {
        Ok(result) => Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(result)),
        Err(_) => Fixed::ZERO,
    }
}

/// Fused softmax + weighted value mix, entirely at compute tier:
///
/// ```text
/// out[d] = Σⱼ softmax(scores)ⱼ · values[j][d]
/// ```
///
/// The softmax weights are **never materialized to storage tier** on the mix
/// path; they stay at compute-tier resolution through the value accumulation,
/// and only the mixed output vector is downscaled (one rounding per output
/// element). This removes the storage-tier resolution floor on attention
/// weights: with FRAC_BITS fractional bits, a materialized weight below
/// 2^-FRAC_BITS truncates to zero and its value vector vanishes from the mix
/// entirely: the cause of long-context attention starvation. Here a weight
/// of any compute-representable magnitude still contributes.
///
/// Returns `(mixed_output[dim], weights[n])`. The returned weights ARE
/// storage-quantized; they are for observers (attention recording,
/// diagnostics), not what the mix used. [`softmax_mix_values`] skips them
/// (one division per position saved); [`softmax_mix_flat`] and
/// [`softmax_mix_flat_values`] take the value rows as one contiguous buffer.
/// All four return the same mixed output.
///
/// **Use case**: single-query attention `softmax(Q·Kᵀ/√d) · V`: the hot path
/// of autoregressive transformer inference.
///
/// # Errors
/// - `TierOverflow`: a numerator `sum_j e_j * v[j][d]` leaves the compute
///   tier. Each term is at most `max|v|`, so the realtime profile is safe
///   while `n * max|v| < 2^(63 - 2 * FRAC_BITS)` in value terms (in raw
///   terms `n * max|v_raw| < 2^(63 - FRAC_BITS)`; with full-range values that
///   is `n < 2^22` at Q22.10 and `n < 2^16` at Q16.16). Also returned if the
///   sum of exponentials overflows (see [`softmax`]) or a mixed output
///   exceeds storage, which cannot happen for in-range values because the
///   output is a convex combination of them.
///
/// # Panics
/// Panics if `scores` and `values` differ in length or the value rows differ
/// in length. Empty input returns two empty vectors.
pub fn softmax_mix<S: Storage, const F: u32>(
    scores: &[Fixed<S, F>],
    values: &[&[Fixed<S, F>]],
) -> Result<(Vec<Fixed<S, F>>, Vec<Fixed<S, F>>), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(scores.len(), values.len(), "softmax_mix: scores/values length mismatch");
    let dim = values.first().map_or(0, |v| v.len());
    softmax_mix_core(scores, dim, true, |j| {
        let v = values[j];
        assert_eq!(v.len(), dim, "softmax_mix: value row {j} has length {}, expected {dim}", v.len());
        v
    })
}

/// [`softmax_mix`] without the observer weights: only the mixed output.
///
/// Same errors and panics as [`softmax_mix`]; the output is the same value.
pub fn softmax_mix_values<S: Storage, const F: u32>(scores: &[Fixed<S, F>], values: &[&[Fixed<S, F>]]) -> Result<Vec<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(scores.len(), values.len(), "softmax_mix: scores/values length mismatch");
    let dim = values.first().map_or(0, |v| v.len());
    softmax_mix_core(scores, dim, false, |j| {
        let v = values[j];
        assert_eq!(v.len(), dim, "softmax_mix: value row {j} has length {}, expected {dim}", v.len());
        v
    })
    .map(|(out, _)| out)
}

/// [`softmax_mix`] over one contiguous value buffer: row `j` is
/// `values_flat[j * dim..(j + 1) * dim]`.
///
/// # Panics
/// Panics if `values_flat.len() != scores.len() * dim`.
pub fn softmax_mix_flat<S: Storage, const F: u32>(
    scores: &[Fixed<S, F>],
    values_flat: &[Fixed<S, F>],
    dim: usize,
) -> Result<(Vec<Fixed<S, F>>, Vec<Fixed<S, F>>), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(values_flat.len(), scores.len() * dim, "softmax_mix_flat: values_flat length is not scores.len() * dim");
    softmax_mix_core(scores, dim, true, |j| &values_flat[j * dim..(j + 1) * dim])
}

/// [`softmax_mix_flat`] writing into caller-provided slices: the mix into
/// `out` and the observer weights into `weights`. The same values as
/// [`softmax_mix_flat`]; neither result is allocated (the exponentials and
/// the numerators still use scratch memory of `scores.len() + dim` values).
///
/// With empty `scores` nothing is written. On an error the slices may hold
/// partial results.
///
/// # Panics
/// Panics if `values_flat.len() != scores.len() * dim`, `out.len() != dim`
/// or `weights.len() != scores.len()`.
pub fn softmax_mix_flat_into<S: Storage, const F: u32>(
    scores: &[Fixed<S, F>],
    values_flat: &[Fixed<S, F>],
    dim: usize,
    out: &mut [Fixed<S, F>],
    weights: &mut [Fixed<S, F>],
) -> Result<(), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(values_flat.len(), scores.len() * dim, "softmax_mix_flat_into: values_flat length is not scores.len() * dim");
    assert_eq!(out.len(), dim, "softmax_mix_flat_into: out length is not dim");
    assert_eq!(weights.len(), scores.len(), "softmax_mix_flat_into: weights length is not scores.len()");
    if scores.is_empty() {
        return Ok(());
    }
    softmax_mix_core_into(scores, dim, out, Some(weights), |j| &values_flat[j * dim..(j + 1) * dim])
}

/// [`softmax_mix_flat_into`] without the observer weights.
///
/// # Panics
/// Panics if `values_flat.len() != scores.len() * dim` or `out.len() != dim`.
pub fn softmax_mix_flat_values_into<S: Storage, const F: u32>(
    scores: &[Fixed<S, F>],
    values_flat: &[Fixed<S, F>],
    dim: usize,
    out: &mut [Fixed<S, F>],
) -> Result<(), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(values_flat.len(), scores.len() * dim, "softmax_mix_flat_values_into: values_flat length is not scores.len() * dim");
    assert_eq!(out.len(), dim, "softmax_mix_flat_values_into: out length is not dim");
    if scores.is_empty() {
        return Ok(());
    }
    softmax_mix_core_into(scores, dim, out, None, |j| &values_flat[j * dim..(j + 1) * dim])
}

/// [`softmax_mix_flat`] without the observer weights.
pub fn softmax_mix_flat_values<S: Storage, const F: u32>(scores: &[Fixed<S, F>], values_flat: &[Fixed<S, F>], dim: usize) -> Result<Vec<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(values_flat.len(), scores.len() * dim, "softmax_mix_flat: values_flat length is not scores.len() * dim");
    softmax_mix_core(scores, dim, false, |j| &values_flat[j * dim..(j + 1) * dim]).map(|(out, _)| out)
}

fn softmax_mix_core<'a, S: Storage, const F: u32>(
    scores: &[Fixed<S, F>],
    dim: usize,
    want_weights: bool,
    row: impl Fn(usize) -> &'a [Fixed<S, F>],
) -> Result<(Vec<Fixed<S, F>>, Vec<Fixed<S, F>>), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if scores.is_empty() {
        return Ok((vec![], vec![]));
    }
    let mut out = vec![Fixed::ZERO; dim];
    let mut weights = vec![Fixed::ZERO; if want_weights { scores.len() } else { 0 }];
    softmax_mix_core_into(scores, dim, &mut out, want_weights.then_some(&mut weights[..]), row)?;
    Ok((out, weights))
}

/// The mix written into `out` (`dim` elements) and, when asked for, the
/// observer weights into `weights` (`scores.len()` elements). `scores` is
/// not empty. On an error the slices may hold partial results.
fn softmax_mix_core_into<'a, S: Storage, const F: u32>(
    scores: &[Fixed<S, F>],
    dim: usize,
    out: &mut [Fixed<S, F>],
    weights: Option<&mut [Fixed<S, F>]>,
    row: impl Fn(usize) -> &'a [Fixed<S, F>],
) -> Result<(), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{

    // Phase 1: find max at storage tier
    let mut max_raw = scores[0].raw();
    for s in &scores[1..] {
        if s.raw() > max_raw {
            max_raw = s.raw();
        }
    }
    let max_compute = Fixed::<S, F>::upscale_to_compute(max_raw);

    // Phase 2: exp(s_i - max) at compute tier, accumulate sum. The sum is
    // Σ eⱼ (each eⱼ ≤ 1.0 at compute tier), so it only overflows for
    // astronomically large n — but check it anyway so a wrapped denominator
    // can never masquerade as a valid divisor.
    let mut exp_values: Vec<Compute<Fixed<S, F>>> = Vec::with_capacity(scores.len());
    let mut sum = compute_zero::<S, F>();
    for s in scores {
        let shifted = Fixed::<S, F>::compute_subtract(Fixed::<S, F>::upscale_to_compute(s.raw()), max_compute);
        let e = Fixed::<S, F>::exp_at_compute_tier(shifted);
        sum = Fixed::<S, F>::compute_checked_add(sum, e)?;
        exp_values.push(e);
    }
    if Fixed::<S, F>::compute_is_zero(&sum) {
        return Err(OverflowDetected::DivisionByZero);
    }

    // Phase 3: accumulate numerators at compute tier, value-row-major for
    // cache locality: num[d] = Σⱼ eⱼ · v[j][d]. This is the module's largest
    // accumulation (scaled by |v|, not bounded by 1.0 like the denominator),
    // so a long context × large activations can exceed the compute envelope —
    // use checked adds and surface TierOverflow rather than wrap silently.
    let mut num: Vec<Compute<Fixed<S, F>>> = vec![compute_zero::<S, F>(); dim];
    let fast = S::mix_numerators_fast::<F>(&exp_values, &|j| Fixed::raw_slice(row(j)), &mut num);
    if !fast {
        for (j, &e) in exp_values.iter().enumerate() {
            let v = row(j);
            for d in 0..dim {
                num[d] = Fixed::<S, F>::compute_checked_add(num[d], Fixed::<S, F>::compute_multiply(e, Fixed::<S, F>::upscale_to_compute(v[d].raw())))?;
            }
        }
    }

    // Phase 4: single downscale per output element
    for (o, n) in out.iter_mut().zip(&num) {
        *o = Fixed::from_raw(Fixed::<S, F>::downscale_to_storage(Fixed::<S, F>::compute_divide(*n, sum)?)?);
    }

    // Phase 5: observer weights (storage-quantized, NOT used by the mix)
    if let Some(weights) = weights {
        for (w, e) in weights.iter_mut().zip(&exp_values) {
            *w = Fixed::from_raw(Fixed::<S, F>::downscale_to_storage(Fixed::<S, F>::compute_divide(*e, sum)?)?);
        }
    }

    Ok(())
}

/// Realtime numerators in plain i64, where bounds prove that nothing can
/// overflow: with `F = FRAC_BITS <= 15`, `0 <= e <= 2^(2F)` (the exponent is
/// never positive) and `|v| <= 2^31`, each product `e * v` is at most
/// `2^(2F + 31) <= 2^61`, each rounded term at most `2^(F + 31)`, and up to
/// `2^(31 - F)` positions keep every partial sum within `2^62`.
///
/// `(e * v + 2^(F - 1)) >> F` is `Fixed::<S, F>::compute_multiply(e, v << F)` written
/// without the 128-bit product: that function returns
/// `floor(p / 2^(2F)) + bit(2F - 1)` of `p = e * v * 2^F`, which is
/// `floor(e * v / 2^F) + bit(F - 1)` of `e * v`. Same integers as the checked
/// loop, in a form the compiler vectorises. Returns `false`, leaving `num`
/// untouched, when the bounds do not hold.

/// `sigmoid(gate)` at the wide tier for one storage value.
///
/// Realtime and compact: Q64.64 (`wide::sigmoid_q64`), returned with its 64
/// fractional bits. Wider profiles: the compute tier (`2F` fractional bits).
/// sigma(gate) at the compute tier (the sign-split form: the exponential
/// argument is never positive and the denominator stays in [1, 2]).
#[inline]
fn gate_sigmoid<S: Storage, const F: u32>(gate: Fixed<S, F>) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let g = Fixed::<S, F>::upscale_to_compute(gate.raw());
    let one = compute_one::<S, F>();
    let (num, e) = if Fixed::<S, F>::compute_is_negative(&g) {
        let e = Fixed::<S, F>::exp_at_compute_tier(g);
        (e, e)
    } else {
        (one, Fixed::<S, F>::exp_at_compute_tier(Fixed::<S, F>::compute_negate(g)))
    };
    Fixed::<S, F>::compute_divide(num, sum_add(one, e)).expect("sigmoid: denominator in [1, 2]")
}

/// `x * sigmoid(gate)`: the sigmoid at the wide tier, the product exact, one
/// rounding to storage (nearest, ties toward +infinity).
///
/// The sigmoid is evaluated at Q64.64 on the realtime and compact profiles
/// and at the compute tier on the wider ones, so its error reaches the
/// result scaled by `|x|` and stays far below one storage unit for any `x`
/// the profile can hold with a few integer bits to spare. Since
/// `|x * sigmoid(gate)| <= |x|` the result always fits: this cannot fail.
pub fn sigmoid_mul<S: Storage, const F: u32>(x: Fixed<S, F>, gate: Fixed<S, F>) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if let Some(r) = S::sigmoid_mul_fast::<F>(x.raw(), gate.raw()) {
        return Fixed::from_raw(r);
    }
    let p = S::widen_product(S::widen(x.raw()), gate_sigmoid(gate));
    Fixed::from_raw(kernels::narrow_triple_nearest::<Fixed<S, F>>(p).expect("sigmoid_mul: |x * sigmoid| <= |x|"))
}

/// [`sigmoid_mul`] element by element: `out[i] = x[i] * sigmoid(gate[i])`.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn sigmoid_mul_slice<S: Storage, const F: u32>(x: &[Fixed<S, F>], gate: &[Fixed<S, F>]) -> Vec<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(x.len(), gate.len(), "sigmoid_mul_slice: length mismatch");
    x.iter().zip(gate).map(|(&x, &g)| sigmoid_mul(x, g)).collect()
}

/// [`sigmoid_mul`] in place: `x[i] *= sigmoid(gate[i])`.
///
/// # Panics
/// Panics if the slices differ in length.
pub fn sigmoid_mul_in_place<S: Storage, const F: u32>(x: &mut [Fixed<S, F>], gate: &[Fixed<S, F>])
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(x.len(), gate.len(), "sigmoid_mul_in_place: length mismatch");
    for (x, &g) in x.iter_mut().zip(gate) {
        *x = sigmoid_mul(*x, g);
    }
}

/// Shannon entropy `-sum(w * ln(w))` in nats, the terms accumulated at the
/// wide tier and the sum rounded to storage once. A zero weight contributes
/// zero. The weights are used as given: they are not normalised.
///
/// The terms are formed at Q64.64 on the realtime and compact profiles and
/// at the compute tier on the wider ones.
///
/// `Err(DomainError)` if a weight is negative, `Err(TierOverflow)` if the
/// sum or the result leaves its tier.
pub fn entropy<S: Storage, const F: u32>(weights: &[Fixed<S, F>]) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if weights.iter().any(|w| w.is_negative()) {
        return Err(OverflowDetected::DomainError);
    }
    if let Some(r) = S::entropy_fast::<F>(Fixed::raw_slice(weights)) {
        return r.map(Fixed::<S, F>::from_raw);
    }
    let mut sum = compute_zero::<S, F>();
    for w in weights {
        if w.is_zero() {
            continue;
        }
        let wc = Fixed::<S, F>::upscale_to_compute(w.raw());
        sum = Fixed::<S, F>::compute_checked_add(sum, Fixed::<S, F>::compute_checked_multiply(wc, Fixed::<S, F>::ln_at_compute_tier(wc))?)?;
    }
    Ok(Fixed::from_raw(Fixed::<S, F>::downscale_to_storage(Fixed::<S, F>::compute_checked_negate(sum)?)?))
}

/// One query against many keys stored in one contiguous buffer: element `k`
/// of the result is `dot(query, keys_flat[k * dim..(k + 1) * dim])`, each
/// accumulated at the compute tier and rounded to storage once, exactly as
/// [`dot`] and `FixedVector::<Fixed<S, F>>::dot` round.
///
/// On the realtime profile the query is bounded once; when
/// `dim * max|query| * 2^31 < 2^63` no key can overflow the accumulator and
/// every key takes the unchecked, vectorised sum.
///
/// # Panics
/// Panics if `dim != query.len()`, if `keys_flat.len()` is not a multiple of
/// `dim`, or if a result leaves the storage range (as [`dot`] does).
pub fn dot_many<S: Storage, const F: u32>(query: &[Fixed<S, F>], keys_flat: &[Fixed<S, F>], dim: usize) -> Vec<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(query.len(), dim, "dot_many: query length is not dim");
    if dim == 0 {
        assert!(keys_flat.is_empty(), "dot_many: keys_flat must be empty when dim is 0");
        return Vec::new();
    }
    assert_eq!(keys_flat.len() % dim, 0, "dot_many: keys_flat length is not a multiple of dim");
    let mut out = vec![Fixed::ZERO; keys_flat.len() / dim];
    dot_many_into(query, keys_flat, dim, &mut out);
    out
}

/// [`dot_many`] writing into a caller-provided slice: `out[k]` is the dot of
/// the query with key `k`. The same values; nothing is allocated.
///
/// # Panics
/// Panics if `dim != query.len()`, if `keys_flat.len() != out.len() * dim`,
/// or if a result leaves the storage range.
pub fn dot_many_into<S: Storage, const F: u32>(query: &[Fixed<S, F>], keys_flat: &[Fixed<S, F>], dim: usize, out: &mut [Fixed<S, F>])
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(query.len(), dim, "dot_many_into: query length is not dim");
    assert_eq!(keys_flat.len(), out.len() * dim, "dot_many_into: keys_flat length is not out.len() * dim");
    if dim == 0 {
        return;
    }
    if S::dot_many_into_fast::<F>(Fixed::raw_slice(query), Fixed::raw_slice(keys_flat), dim, Fixed::raw_slice_mut(out)) {
        return;
    }
    for (o, key) in out.iter_mut().zip(keys_flat.chunks_exact(dim)) {
        *o = kernels::dot(query, key);
    }
}

/// RMS normalisation with a learned scale:
/// `out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)`, each element rounded
/// once (`eps` in Q64.64).
///
/// The sum of squares is exact. The reciprocal root is taken once per call on
/// a radicand scaled by a power of two into `[1, 4)`, so it keeps its full
/// relative precision whatever the size of the input, and each output is the
/// exact product `x[i] * weight[i] * reciprocal` rounded to storage once, to
/// nearest with ties toward positive infinity. No intermediate is rounded to
/// storage: in particular the factor is not, so this is not
/// `x[i] * rms_norm_factor_eps_wide(x, eps) * weight[i]` (three storage
/// roundings; at 10 fraction bits a factor of 0.05 alone is 0.4% off).
///
/// Accuracy: the reciprocal carries a relative error below `2^-60` on the
/// realtime profile (at every `GMATH_FRAC_BITS`) and below `2^-(2F - 2)` on
/// the wider ones, so an output differs from the correctly rounded value by
/// at most one unit, and only when the exact value lies that close to a
/// rounding boundary. On realtime the epsilon enters exactly as given (it is
/// not first rounded to the compute tier, as in
/// [`rms_norm_factor_eps_wide`]).
///
/// # Errors
/// - `DivisionByZero`: `x` is empty, or `mean + eps` is zero.
/// - `DomainError`: `mean + eps` is negative (a negative epsilon).
/// - `TierOverflow`: an output leaves the storage range, or the sum of
///   squares leaves the working width (realtime: `sum(x^2) + n * eps` at or
///   above `2^62`; wider profiles: the compute tier).
///
/// # Panics
/// Panics if `x` and `weight` differ in length.
pub fn rms_norm<S: Storage, const F: u32>(x: &[Fixed<S, F>], weight: &[Fixed<S, F>], eps_q64: i128) -> Result<Vec<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(x.len(), weight.len(), "rms_norm: x/weight length mismatch");
    let (inv, shift) = rms_reciprocal(x, eps_q64)?;
    let mut out = x.to_vec();
    rms_apply_in_place(&mut out, weight, inv, shift)?;
    Ok(out)
}

/// [`rms_norm`] in place. On `Err` the elements before the failing one have
/// already been replaced.
pub fn rms_norm_in_place<S: Storage, const F: u32>(x: &mut [Fixed<S, F>], weight: &[Fixed<S, F>], eps_q64: i128) -> Result<(), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(x.len(), weight.len(), "rms_norm: x/weight length mismatch");
    let (inv, shift) = rms_reciprocal(x, eps_q64)?;
    rms_apply_in_place(x, weight, inv, shift)
}

/// `x * w * inv / 2^shift`: the exact triple product, rounded once (floor
/// of the doubled value plus one, halved: nearest, ties toward +infinity).
/// `shift >= 1` and `|product| < 2^125`, so nothing here can overflow.
/// `x_i <- x_i * w_i * inv / 2^shift`, each rounded once to storage.
fn rms_apply_in_place<S: Storage, const F: u32>(x: &mut [Fixed<S, F>], w: &[Fixed<S, F>], inv: Compute<Fixed<S, F>>, shift: u32) -> Result<(), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if let Some(r) = S::rms_apply_in_place_fast::<F>(Fixed::raw_slice_mut(x), Fixed::raw_slice(w), inv, shift) {
        return r;
    }
    for (v, &w) in x.iter_mut().zip(w) {
        let product = S::widen_product(S::mul_wide(v.raw(), w.raw()), inv);
        *v = Fixed::from_raw(kernels::narrow_shifted_nearest::<Fixed<S, F>>(product, shift)?);
    }
    Ok(())
}

/// `(inv, shift)` with `x_raw * w_raw * inv / 2^shift` the storage raw of
/// `x * w / sqrt(mean(values^2) + eps)`.
///
/// Realtime: everything in 128-bit integers at Q64.64, independent of the
/// compute tier's `2 * FRAC_BITS`. `T = n * (mean + eps)` is exact; `T / n`
/// and its root are taken on top-aligned words, so `inv` in `(2^61, 2^62]`
/// has a relative error below `2^-60`.
fn rms_reciprocal<S: Storage, const F: u32>(values: &[Fixed<S, F>], eps_q64: i128) -> Result<(Compute<Fixed<S, F>>, u32), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if values.is_empty() {
        return Err(OverflowDetected::DivisionByZero);
    }
    if let Some(r) = S::rms_reciprocal_fast::<F>(Fixed::raw_slice(values), eps_q64) {
        return r;
    }
    let f = F;
    let n = i64::try_from(values.len()).map_err(|_| OverflowDetected::TierOverflow)?;
    let mut total = Fixed::<S, F>::compute_checked_multiply(q64_to_compute(eps_q64)?, Fixed::<S, F>::make_compute_int(n))?;
    for v in values {
        total = Fixed::<S, F>::compute_checked_add(total, S::mul_wide(v.raw(), v.raw()))?;
    }
    if Fixed::<S, F>::compute_is_negative(&total) {
        return Err(OverflowDetected::DomainError);
    }
    if Fixed::<S, F>::compute_is_zero(&total) {
        return Err(OverflowDetected::DivisionByZero);
    }
    // even shifts throughout: the root's power of two stays whole
    let up = (4 * f - 1 - kernels::compute_bit_length::<Fixed<S, F>>(total)) & !1;
    let mean = Fixed::<S, F>::compute_div_count(total.shl_bits(up), values.len())?;
    let excess = kernels::compute_bit_length::<Fixed<S, F>>(mean) as i64 - (2 * f as i64 + 1);
    let (normalised, exponent) = if excess >= 0 {
        let down = (excess & !1) as u32;
        (mean.shr_bits(down), up as i64 - down as i64)
    } else {
        let more = ((1 - excess) & !1) as u32;
        (mean.shl_bits(more), up as i64 + more as i64)
    };
    // normalised = (mean + eps) * 2^exponent in [1, 4)
    let root = Fixed::<S, F>::sqrt_at_compute_tier(normalised);
    let inv = Fixed::<S, F>::compute_divide(compute_one::<S, F>(), root)?; // 2^(-exponent/2) / sqrt(mean + eps)
    let shift = 3 * f as i64 - exponent / 2;
    u32::try_from(shift).map(|shift| (inv, shift)).map_err(|_| OverflowDetected::TierOverflow)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    macro_rules! suite {
        ($m:ident, $S:ty, $F:literal, $W:ident) => {
            mod $m {
                #![allow(dead_code, unused_imports, unreachable_code)]
                use super::super::*;
                type FixedPoint = crate::fixed_point::Fixed<$S, $F>;
                type BinaryStorage = $S;
                type ComputeStorage = <$S as crate::fixed_point::Storage>::Compute;
                const FRAC_BITS: u32 = $F;
                const BITS: u32 = <$S as crate::fixed_point::WideInt>::BITS;
                use super::super::*;

                fn fp(s: &str) -> FixedPoint {
                    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
                    else { FixedPoint::from_str(s) }
                }

                /// `t`, raised to one storage unit where the build's split cannot
                /// represent it (0.001 is 0 raw at GMATH_FRAC_BITS=8, so `diff < t`
                /// could never hold). Unchanged wherever `t` is already >= 1 unit.
                fn at_least_one_unit(t: FixedPoint) -> FixedPoint {
                    let mut u = FixedPoint::one();
                    for _ in 0..FRAC_BITS { u = u / FixedPoint::from_int(2); }
                    if u > t { u } else { t }
                }

                /// The 0.001 tolerance of the approximate checks, at least 1 unit.
                fn loose() -> FixedPoint { at_least_one_unit(fp("0.001")) }

                /// Profile-appropriate tight tolerance: at least 1 ULP representable.
                fn tight() -> FixedPoint {
                    if BITS == 32 { at_least_one_unit(fp("0.001")) } else { fp("0.000000001") }
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

                on_words!($W in [w32] {
                /// Oversized fused results must panic loudly, never wrap silently.
                /// Before the round_to_storage fix, this returned a NEGATIVE squared
                /// distance on realtime profiles. from_raw(i32::MAX) keeps the input
                /// constructible and the square out of range at every GMATH_FRAC_BITS.
                #[test]
                #[should_panic(expected = "exceeds storage tier")]
                fn test_euclidean_distance_squared_overflow_panics() {
                    let a = [FixedPoint::from_raw(i32::MAX)];
                    let b = [FixedPoint::ZERO];
                    let _ = euclidean_distance_squared(&a, &b);
                }
                });

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
                        assert!(diff < loose(), "softmax[{}] = {}, expected 0.25", i, w);
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
                    assert!(diff < loose(), "rms_norm_factor = {}, expected ~0.5", factor);
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
                    assert!(diff < loose(), "silu(10) = {}, expected ~10", result);
                }

                #[test]
                fn test_silu_negative() {
                    // SiLU(x) ≈ 0 for large negative x (sigmoid ≈ 0)
                    let result = silu(fp("-10"));
                    assert!(result.abs() < loose(), "silu(-10) = {}, expected ~0", result);
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

                // ------------------------------------------------------------------
                // 0.6.5: softmax_mix variants, dot_many, rms_norm
                // ------------------------------------------------------------------

                /// The 0.6.4 body of `softmax_mix` (checked compute-tier accumulation),
                /// kept verbatim as the reference for the realtime i64 numerator path.
                fn softmax_mix_reference(
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
                    let max_compute = FixedPoint::upscale_to_compute(max_raw);

                    // Phase 2: exp(s_i - max) at compute tier, accumulate sum. The sum is
                    // Σ eⱼ (each eⱼ ≤ 1.0 at compute tier), so it only overflows for
                    // astronomically large n — but check it anyway so a wrapped denominator
                    // can never masquerade as a valid divisor.
                    let mut exp_values: Vec<ComputeStorage> = Vec::with_capacity(scores.len());
                    let mut sum = compute_zero::<BinaryStorage, FRAC_BITS>();
                    for s in scores {
                        let shifted = FixedPoint::compute_subtract(FixedPoint::upscale_to_compute(s.raw()), max_compute);
                        let e = FixedPoint::exp_at_compute_tier(shifted);
                        sum = FixedPoint::compute_checked_add(sum, e)?;
                        exp_values.push(e);
                    }
                    if FixedPoint::compute_is_zero(&sum) {
                        return Err(OverflowDetected::DivisionByZero);
                    }

                    // Phase 3: accumulate numerators at compute tier, value-row-major for
                    // cache locality: num[d] = Σⱼ eⱼ · v[j][d]. This is the module's largest
                    // accumulation (scaled by |v|, not bounded by 1.0 like the denominator),
                    // so a long context × large activations can exceed the compute envelope —
                    // use checked adds and surface TierOverflow rather than wrap silently.
                    let mut num: Vec<ComputeStorage> = vec![compute_zero::<BinaryStorage, FRAC_BITS>(); dim];
                    for (j, v) in values.iter().enumerate() {
                        assert_eq!(
                            v.len(),
                            dim,
                            "softmax_mix: value row {j} has length {}, expected {dim}",
                            v.len()
                        );
                        let e = exp_values[j];
                        for d in 0..dim {
                            num[d] = FixedPoint::compute_checked_add(num[d], FixedPoint::compute_multiply(e, FixedPoint::upscale_to_compute(v[d].raw())))?;
                        }
                    }

                    // Phase 4: single downscale per output element
                    let mut out = Vec::with_capacity(dim);
                    for n in &num {
                        out.push(FixedPoint::from_raw(FixedPoint::downscale_to_storage(FixedPoint::compute_divide(*n, sum)?)?));
                    }

                    // Phase 5: observer weights (storage-quantized, NOT used by the mix)
                    let mut weights = Vec::with_capacity(scores.len());
                    for e in &exp_values {
                        weights.push(FixedPoint::from_raw(FixedPoint::downscale_to_storage(FixedPoint::compute_divide(*e, sum)?)?));
                    }

                    Ok((out, weights))
                }

                struct Lcg(u64);
                impl Lcg {
                    fn next(&mut self) -> u64 {
                        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        self.0 >> 33
                    }
                    /// k / 64 with |k| <= amp, built from an integer part below 127 so it
                    /// fits every profile and every realtime split the suite gates
                    /// (Q8.24 holds +-128).
                    fn value(&mut self, amp: i32) -> FixedPoint {
                        assert!(amp <= 8000);
                        let k = (self.next() % (2 * amp as u64 + 1)) as i32 - amp;
                        FixedPoint::from_int(k / 64) + FixedPoint::from_int(k % 64) / FixedPoint::from_int(64)
                    }
                }

                fn assert_mix_variants_equal_reference(scores: &[FixedPoint], rows: &[Vec<FixedPoint>], dim: usize) {
                    let refs: Vec<&[FixedPoint]> = rows.iter().map(|r| r.as_slice()).collect();
                    let flat: Vec<FixedPoint> = rows.iter().flat_map(|r| r.iter().copied()).collect();
                    let want = softmax_mix_reference(scores, &refs);
                    assert_eq!(softmax_mix(scores, &refs), want);
                    assert_eq!(softmax_mix_flat(scores, &flat, dim), want);
                    let want_out = want.map(|(out, _)| out);
                    assert_eq!(softmax_mix_values(scores, &refs), want_out);
                    assert_eq!(softmax_mix_flat_values(scores, &flat, dim), want_out);
                }

                #[test]
                fn test_softmax_mix_variants_equal_the_checked_reference() {
                    let mut rng = Lcg(17);
                    for &dim in &[1usize, 3, 8, 64] {
                        for &n in &[1usize, 2, 3, 15, 16, 17, 100, 600] {
                            // score spread: ties, moderate, exps flushed to zero
                            for &amp in &[2i32, 200, 6000] {
                                let scores: Vec<FixedPoint> = (0..n).map(|_| rng.value(amp)).collect();
                                let rows: Vec<Vec<FixedPoint>> = (0..n).map(|_| (0..dim).map(|_| rng.value(6000)).collect()).collect();
                                assert_mix_variants_equal_reference(&scores, &rows, dim);
                            }
                        }
                    }
                    // empty
                    assert_eq!(softmax_mix_values::<BinaryStorage, FRAC_BITS>(&[], &[]), Ok(vec![]));
                    assert_eq!(softmax_mix_flat::<BinaryStorage, FRAC_BITS>(&[], &[], 4), Ok((vec![], vec![])));
                }

                on_words!($W in [w32] {
                /// Realtime: raw-level extremes, including full-range values, lengths
                /// around the fast path's position bound, and inputs where the reference
                /// reports `TierOverflow` (both must agree on the error too).
                #[test]
                fn test_softmax_mix_i64_path_equals_reference_at_extremes() {
                    let mut rng = Lcg(99);
                    let mut raw = |amp: i64| FixedPoint::from_raw(((rng.next() as i64) % (2 * amp + 1) - amp) as i32);
                    for &(n, dim, score_amp, value_amp) in &[
                        (1usize, 1usize, 1i64 << 30, i32::MAX as i64),
                        (2, 1, 3, i32::MAX as i64),
                        (17, 5, 1 << 12, i32::MAX as i64),
                        (2049, 1, 1 << 6, i32::MAX as i64 / 4),
                        (2049, 8, 1 << 16, 1 << 20),
                        (4097, 2, 2, i32::MAX as i64),
                        (70_000, 1, 1, 1 << 30),
                    ] {
                        let scores: Vec<FixedPoint> = (0..n).map(|_| raw(score_amp)).collect();
                        let rows: Vec<Vec<FixedPoint>> = (0..n).map(|_| (0..dim).map(|_| raw(value_amp)).collect()).collect();
                        assert_mix_variants_equal_reference(&scores, &rows, dim);
                    }
                    // every value at the storage minimum and every exp at one
                    let n = 300;
                    let scores = vec![FixedPoint::ZERO; n];
                    let rows = vec![vec![FixedPoint::from_raw(i32::MIN); 3]; n];
                    assert_mix_variants_equal_reference(&scores, &rows, 3);
                }
                });

                #[test]
                fn test_dot_many_equals_dot_per_key() {
                    let mut rng = Lcg(5);
                    for &dim in &[1usize, 3, 15, 16, 17, 64, 128] {
                        for &keys in &[0usize, 1, 2, 9] {
                            // |v| <= 1: a 128-term dot stays inside the narrowest range
                            let query: Vec<FixedPoint> = (0..dim).map(|_| rng.value(64)).collect();
                            let flat: Vec<FixedPoint> = (0..dim * keys).map(|_| rng.value(64)).collect();
                            let got = dot_many(&query, &flat, dim);
                            assert_eq!(got.len(), keys);
                            for k in 0..keys {
                                assert_eq!(got[k], dot(&query, &flat[k * dim..(k + 1) * dim]), "dim {dim} key {k}");
                            }
                        }
                    }
                }

                on_words!($W in [w32] {
                /// Realtime: the bounded sum equals the checked sum on raw extremes
                /// (rounding ties in both signs, large operands), and an overflowing
                /// input panics on both paths.
                #[test]
                fn test_bounded_dot_equals_checked_dot_on_raws() {
                    let mut rng = Lcg(42);
                    let mut raws = |n: usize, amp: i64| -> Vec<FixedPoint> {
                        (0..n).map(|_| FixedPoint::from_raw(((rng.next() as i64) % (2 * amp + 1) - amp) as i32)).collect()
                    };
                    let exact = |a: &[FixedPoint], b: &[FixedPoint]| -> Option<i32> {
                        let acc: i128 = a.iter().zip(b).map(|(x, y)| x.raw() as i128 * y.raw() as i128).sum();
                        let r = (acc >> FRAC_BITS) + ((acc >> (FRAC_BITS - 1)) & 1);
                        i32::try_from(r).ok()
                    };
                    for case in 0..3000usize {
                        let n = [1usize, 7, 16, 17, 64, 128, 256, 1000][case % 8];
                        let (qa, ka) = (if case % 7 == 0 { 1i64 << 25 } else { 1 << 14 }, if case % 11 == 0 { 1i64 << 20 } else { 1 << 12 });
                        let (q, k) = (raws(n, qa), raws(n, ka));
                        let want = exact(&q, &k);
                        let got = std::panic::catch_unwind(|| dot(&q, &k).raw()).ok();
                        assert_eq!(got, want, "case {case} n {n}");
                        let many = std::panic::catch_unwind(|| dot_many(&q, &k, n)[0].raw()).ok();
                        assert_eq!(many, want, "dot_many case {case} n {n}");
                    }
                    // exact ties at the rounding bit, both signs, long enough for the bounded path
                    let half = 1i32 << (FRAC_BITS - 1);
                    for target in [half, -half, 3 * half, -3 * half, half - 1, -half - 1] {
                        let mut q = vec![FixedPoint::ZERO; 32];
                        let mut k = vec![FixedPoint::ZERO; 32];
                        q[5] = FixedPoint::from_raw(target);
                        k[5] = FixedPoint::from_raw(1);
                        assert_eq!(Some(dot(&q, &k).raw()), exact(&q, &k), "tie {target}");
                    }
                    // full-range operands: the bound fails, the checked loop decides
                    let big = vec![FixedPoint::from_raw(i32::MIN); 64];
                    assert!(std::panic::catch_unwind(|| dot(&big, &big)).is_err());
                    assert!(std::panic::catch_unwind(|| dot_many(&big, &big, 64)).is_err());
                }
                });

                #[test]
                fn test_rms_norm_one_rounding() {
                    let one = FixedPoint::one();
                    let half = one / FixedPoint::from_int(2);
                    let two = FixedPoint::from_int(2);
                    // rms 2: the outputs are the weights (exact)
                    let w = [half, -(one + half), two, -one];
                    assert_eq!(rms_norm(&[two, two, -two, two], &w, 0), Ok(vec![half, -(one + half), -two, -one]));
                    // one storage unit among zeros: mean = unit^2 / 4 is a quarter of a
                    // compute unit, yet the output is exactly 2
                    let mut unit = one;
                    for _ in 0..FRAC_BITS { unit = unit / two; }
                    let tiny = [unit, FixedPoint::ZERO, FixedPoint::ZERO, FixedPoint::ZERO];
                    assert_eq!(rms_norm(&tiny, &[one; 4], 0), Ok(vec![two, FixedPoint::ZERO, FixedPoint::ZERO, FixedPoint::ZERO]));
                    // in place gives the same values; an all-zero input is zero when eps > 0
                    let mut rng = Lcg(8);
                    let eps = 1i128 << 44; // about 9.5e-7 in Q64.64
                    for &n in &[1usize, 2, 7, 64, 300] {
                        let x: Vec<FixedPoint> = (0..n).map(|_| rng.value(200)).collect();
                        let w: Vec<FixedPoint> = (0..n).map(|_| rng.value(128)).collect();
                        let want = rms_norm(&x, &w, eps).unwrap();
                        let mut y = x.clone();
                        assert_eq!(rms_norm_in_place(&mut y, &w, eps), Ok(()));
                        assert_eq!(y, want);
                    }
                    assert_eq!(rms_norm(&[FixedPoint::ZERO; 4], &[one; 4], eps), Ok(vec![FixedPoint::ZERO; 4]));
                    // errors
                    assert_eq!(rms_norm(&[FixedPoint::ZERO; 4], &[one; 4], 0), Err(OverflowDetected::DivisionByZero));
                    assert_eq!(rms_norm::<BinaryStorage, FRAC_BITS>(&[], &[], eps), Err(OverflowDetected::DivisionByZero));
                    assert_eq!(rms_norm(&[FixedPoint::ZERO; 4], &[one; 4], -eps), Err(OverflowDetected::DomainError));
                    // an output beyond storage: 2 times the largest power of two
                    let mut big = one;
                    while let Ok(next) = big.try_add(big) { big = next; }
                    assert_eq!(rms_norm(&tiny, &[big, one, one, one], 0), Err(OverflowDetected::TierOverflow));
                }
            }
        };
    }
    for_each_format!(suite);
}
