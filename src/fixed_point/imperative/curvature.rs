//! L3B: Christoffel symbols and curvature tensors with fixed-point arithmetic.
//!
//! Provides numerical differential geometry on Riemannian manifolds:
//! - `numerical_derivative`: central difference with optimal h = 2^(-FRAC_BITS/3)
//! - `christoffel`: Γᵏᵢⱼ = ½gᵏˡ(∂ᵢgⱼˡ+∂ⱼgˡᵢ-∂ˡgᵢⱼ)
//! - `riemann_curvature`: R^l_{ijk} = ∂ⱼΓˡᵢₖ - ∂ₖΓˡᵢⱼ + ΓˡⱼₘΓᵐᵢₖ - ΓˡₖₘΓᵐᵢⱼ
//! - `ricci_tensor`: Rᵢⱼ = R^k_{ikj}
//! - `scalar_curvature`: R = gⁱʲRᵢⱼ
//! - `sectional_curvature`: K(u,v) = R(u,v,v,u)/(|u|²|v|²-<u,v>²)
//!
//! **FASC-UGOD integration:** Numerical differentiation uses h = 2^(-FRAC_BITS/3)
//! (power-of-2, so division by 2h is exact). Metric partials, the inverse
//! metric, Christoffel symbols, their central differences, the Riemann, Ricci,
//! scalar and sectional contractions all stay at the compute tier (2F bits)
//! and each returned value is rounded to storage once: a storage rounding
//! inside a central difference would come out multiplied by 1/(2h). Riemann
//! tensor involves nested finite differences → O(h²) total error; scientific
//! profile recommended for curvature computations.

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::tensor::Tensor;
use super::linalg::{
    compute_product, compute_tier_dot_raw, downscale_to_storage, round_to_storage,
    sincos_at_compute_tier, upscale_to_compute, ComputeStorage,
};
use super::derived::inverse;
use super::compute_matrix::{compute_lu_decompose, ComputeMatrix};
use super::ode::{OdeSystem, rk4_step_compute, state_to_compute};
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_add, compute_checked_add, compute_checked_divide, compute_divide, compute_halve,
    compute_mul_div_int, compute_multiply, compute_negate, compute_subtract,
};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// Numerical differentiation step size
// ============================================================================

/// Optimal step size for central differences: h = 2^(-FRAC_BITS/3).
///
/// This minimizes total error (truncation + rounding) for central differences.
/// Being a power of 2, division by 2h is an exact bit-shift (no rounding).
///
/// Profile values (k = FRAC_BITS/3 rounded to nearest, h = 2^-k):
/// - realtime:  k = (FRAC_BITS + 1) / 3, e.g. 2^(-5) at Q16.16, 2^(-3) at Q22.10
/// - Q32.32:    h = 2^(-11)
/// - Q64.64:    h ≈ 2^(-21) ≈ 4.8e-7
/// - Q128.128:  h ≈ 2^(-43) ≈ 1.1e-13
/// - Q256.256:  h ≈ 2^(-85) ≈ 2.6e-26
pub fn differentiation_step() -> FixedPoint {
    #[cfg(table_format = "q32_32")]
    { FixedPoint::from_raw(1i64 << (32 - 11)) }
    // realtime follows GMATH_FRAC_BITS; a fixed Q16.16 exponent made h = 2.0
    // at Q22.10 (before 0.6.4)
    #[cfg(table_format = "q16_16")]
    { FixedPoint::from_raw(1i32 << (crate::fixed_point::frac_config::FRAC_BITS - step_exponent())) }
    #[cfg(table_format = "q64_64")]
    { FixedPoint::from_raw(1i128 << (64 - 21)) }
    #[cfg(table_format = "q128_128")]
    {
        use crate::fixed_point::I256;
        FixedPoint::from_raw(I256::from_i128(1) << (128usize - 43))
    }
    #[cfg(table_format = "q256_256")]
    {
        use crate::fixed_point::I512;
        FixedPoint::from_raw(I512::from_i128(1) << (256usize - 85))
    }
}

/// k with h = 2^-k on realtime: FRAC_BITS / 3 rounded to nearest (>= 1).
#[cfg(table_format = "q16_16")]
#[inline]
fn step_exponent() -> u32 {
    (crate::fixed_point::frac_config::FRAC_BITS + 1) / 3
}

// ============================================================================
// Metric function type
// ============================================================================

/// A metric function: given a point (as FixedVector of coordinates), returns
/// the metric tensor g_ij as an n×n FixedMatrix.
///
/// This is the fundamental input for Christoffel symbol computation.
/// For known manifolds, this wraps the closed-form metric (e.g., sphere,
/// hyperbolic). For generic manifolds, it can wrap numerical evaluation.
pub trait MetricProvider {
    /// Dimension of the manifold.
    fn dimension(&self) -> usize;
    /// Evaluate the metric tensor g_ij at point p.
    fn metric(&self, p: &FixedVector) -> FixedMatrix;
    /// Evaluate the inverse metric g^ij at point p.
    /// Default: compute via LU inverse of metric().
    fn metric_inverse(&self, p: &FixedVector) -> Result<FixedMatrix, OverflowDetected> {
        inverse(&self.metric(p))
    }
    /// Closed-form Christoffel symbols, if known analytically.
    ///
    /// Override this for known manifolds to avoid numerical differentiation.
    /// Returns None if no closed form is available (falls back to numerical).
    fn christoffel_closed_form(&self, _p: &FixedVector) -> Option<Tensor> {
        None
    }
    /// Closed-form Christoffel symbols at the compute tier, unrounded.
    ///
    /// Implemented by the built-in metrics so that curvature, which
    /// differences the symbols at p +- h and multiplies the difference by
    /// 1 / (2h), does not amplify their storage rounding. The value cannot be
    /// built outside the crate: other providers keep the default `None` and
    /// their [`christoffel_closed_form`](Self::christoffel_closed_form) is used.
    #[doc(hidden)]
    fn christoffel_closed_form_compute(&self, _p: &FixedVector) -> Option<ComputeChristoffel> {
        None
    }
    /// Closed-form scalar curvature, if known analytically.
    ///
    /// Override this for constant-curvature spaces to get exact results.
    fn scalar_curvature_closed_form(&self, _p: &FixedVector) -> Option<FixedPoint> {
        None
    }
}

/// Christoffel symbols Γ^k_{ij} at the compute tier (row-major [k, i, j]).
/// Opaque: only the crate's built-in metrics construct it.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ComputeChristoffel {
    data: Vec<ComputeStorage>,
}

// ============================================================================
// Compute-tier building blocks
// ============================================================================

/// `(a - b) / (2h)` at the compute tier: the exact difference doubled
/// k - 1 times (h = 2^-k), checked. Before 0.6.4 the difference of storage
/// values was shifted in storage, so a rounding in either operand came out
/// multiplied by 2^(k-1) (2^84 on scientific).
fn central_difference(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    let mut d = checked_sub(a, b)?;
    for _ in 1..differentiation_exponent() {
        d = compute_checked_add(d, d)?;
    }
    Ok(d)
}

/// k with h = 2^-k (see [`differentiation_step`]).
fn differentiation_exponent() -> u32 {
    #[cfg(table_format = "q32_32")]
    { 11 }
    #[cfg(table_format = "q16_16")]
    { step_exponent() }
    #[cfg(table_format = "q64_64")]
    { 21 }
    #[cfg(table_format = "q128_128")]
    { 43 }
    #[cfg(table_format = "q256_256")]
    { 85 }
}

fn compute_zero() -> ComputeStorage {
    upscale_to_compute(FixedPoint::ZERO.raw())
}

fn compute_one() -> ComputeStorage {
    upscale_to_compute(FixedPoint::one().raw())
}

/// A compute-tier tensor rounded to storage once per entry.
fn round_tensor(shape: &[usize], data: &[ComputeStorage]) -> Result<Tensor, OverflowDetected> {
    let values = data.iter()
        .map(|&c| downscale_to_storage(c).map(FixedPoint::from_raw))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor::from_data(shape, &values))
}

/// g^{kl} at the compute tier, row-major.
///
/// The provider's `metric_inverse` is authoritative. When it agrees with the
/// compute-tier LU inverse of `metric()` to within one storage unit per
/// entry (the default, and any correctly rounded override), the compute-tier
/// inverse is the same matrix carried to 2F bits and is used unrounded;
/// otherwise the provider's values are used as given.
fn metric_inverse_compute(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<Vec<ComputeStorage>, OverflowDetected> {
    let n = provider.dimension();
    let given = provider.metric_inverse(p)?;
    let given_c: Vec<ComputeStorage> = (0..n * n)
        .map(|e| upscale_to_compute(given.get(e / n, e % n).raw()))
        .collect();
    let wide = compute_lu_decompose(&ComputeMatrix::from_fixed_matrix(&provider.metric(p)))
        .and_then(|lu| lu.inverse());
    let Ok(wide) = wide else { return Ok(given_c) };
    let unit = compute_one_unit();
    let mut out = Vec::with_capacity(n * n);
    for e in 0..n * n {
        let c = wide.get(e / n, e % n);
        let diff = checked_sub(c, given_c[e])?;
        let diff = if diff < compute_zero() { compute_negate(diff) } else { diff };
        if diff > unit { return Ok(given_c); }
        out.push(c);
    }
    Ok(out)
}

/// One storage unit (2^-F) as a compute raw.
fn compute_one_unit() -> ComputeStorage {
    upscale_to_compute(FixedPoint::from_raw(unit_raw()).raw())
}

/// Raw 1 of the storage type.
fn unit_raw() -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    { 1i32 }
    #[cfg(table_format = "q32_32")]
    { 1i64 }
    #[cfg(table_format = "q64_64")]
    { 1i128 }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::I256::from_i128(1) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::I512::from_i128(1) }
}

// ============================================================================
// Partial derivatives of the metric
// ============================================================================

/// Compute ∂_m g_ij at point p via central differences, at the compute tier.
///
/// Returns n×n compute raws, row-major, entry (i,j) = ∂g_ij/∂x^m. The metric
/// values are the provider's storage values; their difference and the
/// division by 2h are exact.
fn metric_partial_compute(
    provider: &dyn MetricProvider,
    p: &FixedVector,
    m: usize,
) -> Result<Vec<ComputeStorage>, OverflowDetected> {
    let h = differentiation_step();
    let n = provider.dimension();

    // p + h*e_m and p - h*e_m
    let mut p_plus = p.clone();
    let mut p_minus = p.clone();
    p_plus[m] = p_plus[m] + h;
    p_minus[m] = p_minus[m] - h;

    let g_plus = provider.metric(&p_plus);
    let g_minus = provider.metric(&p_minus);

    let mut result = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            result.push(central_difference(
                upscale_to_compute(g_plus.get(i, j).raw()),
                upscale_to_compute(g_minus.get(i, j).raw()),
            )?);
        }
    }
    Ok(result)
}

// ============================================================================
// Christoffel symbols of the second kind
// ============================================================================

/// Compute Christoffel symbols Γ^k_{ij} at point p.
///
/// Formula: Γ^k_{ij} = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{li} - ∂_l g_{ij})
///
/// Returns a rank-3 Tensor of shape [n, n, n] where element [k, i, j] = Γ^k_{ij}.
///
/// The metric partials, the inverse metric, the contraction and the factor
/// ½ are all at the compute tier; each symbol is rounded to storage once.
pub fn christoffel(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<Tensor, OverflowDetected> {
    let n = provider.dimension();
    round_tensor(&[n, n, n], &christoffel_compute(provider, p)?)
}

/// Γ^k_{ij} at the compute tier, row-major [k, i, j].
fn christoffel_compute(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<Vec<ComputeStorage>, OverflowDetected> {
    let n = provider.dimension();
    // Prefer closed-form if available (exact, no numerical differentiation)
    if let Some(gamma) = provider.christoffel_closed_form_compute(p) {
        return Ok(gamma.data);
    }
    if let Some(gamma) = provider.christoffel_closed_form(p) {
        let mut out = Vec::with_capacity(n * n * n);
        for k in 0..n { for i in 0..n { for j in 0..n {
            out.push(upscale_to_compute(gamma.get(&[k, i, j]).raw()));
        } } }
        return Ok(out);
    }

    let g_inv = metric_inverse_compute(provider, p)?;

    // All metric partial derivatives ∂_m g_ij for m = 0..n
    let dg: Vec<Vec<ComputeStorage>> = (0..n)
        .map(|m| metric_partial_compute(provider, p, m))
        .collect::<Result<_, _>>()?;

    // Γ^k_{ij} = ½ sum_l g^{kl} (∂_i g_{jl} + ∂_j g_{li} - ∂_l g_{ij})
    let mut gamma = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let mut acc = compute_zero();
                for l in 0..n {
                    let bracket = checked_sub(
                        compute_checked_add(dg[i][j * n + l], dg[j][l * n + i])?,
                        dg[l][i * n + j],
                    )?;
                    acc = compute_checked_add(acc, compute_product(g_inv[k * n + l], bracket)?)?;
                }
                gamma.push(compute_halve(acc));
            }
        }
    }

    Ok(gamma)
}

// ============================================================================
// Riemann curvature tensor
// ============================================================================

/// Compute Riemann curvature tensor R^l_{ijk} at point p.
///
/// Formula: R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij} + Γ^l_{jm} Γ^m_{ik} - Γ^l_{km} Γ^m_{ij}
///
/// Returns a rank-4 Tensor of shape [n, n, n, n] where element [l, i, j, k] = R^l_{ijk}.
///
/// The Christoffel symbols stay at the compute tier through the central
/// difference and the contractions; each component is rounded to storage
/// once. Before 0.6.4 the symbols were rounded to storage and then
/// differenced, which multiplied their rounding by 2^(k-1) (h = 2^-k).
///
/// **Precision warning:** This involves nested finite differences (derivatives of
/// Christoffel symbols). Total error is O(h²) where h = differentiation_step().
/// For best precision, use the scientific profile.
pub fn riemann_curvature(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<Tensor, OverflowDetected> {
    let n = provider.dimension();
    round_tensor(&[n, n, n, n], &riemann_compute(provider, p)?)
}

/// R^l_{ijk} at the compute tier, row-major [l, i, j, k].
fn riemann_compute(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<Vec<ComputeStorage>, OverflowDetected> {
    let n = provider.dimension();
    let h = differentiation_step();
    let at3 = |a: usize, b: usize, c: usize| (a * n + b) * n + c;

    // Compute Christoffel symbols at p and at neighboring points p ± h*e_j
    let gamma = christoffel_compute(provider, p)?;

    // ∂_j Γ^l_{ik} = (Γ^l_{ik}(p + h*e_j) - Γ^l_{ik}(p - h*e_j)) / (2h)
    let mut dgamma: Vec<Vec<ComputeStorage>> = Vec::with_capacity(n);
    for j in 0..n {
        let mut p_plus = p.clone();
        let mut p_minus = p.clone();
        p_plus[j] = p_plus[j] + h;
        p_minus[j] = p_minus[j] - h;

        let gamma_plus = christoffel_compute(provider, &p_plus)?;
        let gamma_minus = christoffel_compute(provider, &p_minus)?;
        dgamma.push(gamma_plus.iter().zip(&gamma_minus)
            .map(|(&a, &b)| central_difference(a, b))
            .collect::<Result<_, _>>()?);
    }

    // Assemble Riemann tensor
    let mut riemann = Vec::with_capacity(n * n * n * n);
    for l in 0..n {
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    // ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij}
                    let mut acc = checked_sub(dgamma[j][at3(l, i, k)], dgamma[k][at3(l, i, j)])?;
                    // + Γ^l_{jm} Γ^m_{ik} - Γ^l_{km} Γ^m_{ij} (sum over m)
                    for m in 0..n {
                        acc = compute_checked_add(acc, compute_product(gamma[at3(l, j, m)], gamma[at3(m, i, k)])?)?;
                        acc = checked_sub(acc, compute_product(gamma[at3(l, k, m)], gamma[at3(m, i, j)])?)?;
                    }
                    riemann.push(acc);
                }
            }
        }
    }

    Ok(riemann)
}

// ============================================================================
// Ricci tensor
// ============================================================================

/// Compute Ricci tensor Rᵢⱼ = R^k_{ikj} at point p.
///
/// This is the trace of the Riemann tensor over the first and third indices.
/// Returns an n×n FixedMatrix. The trace is taken over the compute-tier
/// Riemann tensor and rounded once.
pub fn ricci_tensor(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<FixedMatrix, OverflowDetected> {
    let n = provider.dimension();
    let ricci = ricci_compute(provider, p)?;
    let mut out = FixedMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            out.set(i, j, FixedPoint::from_raw(downscale_to_storage(ricci[i * n + j])?));
        }
    }
    Ok(out)
}

/// R_{ij} = sum_k R^k_{ikj} at the compute tier, row-major.
fn ricci_compute(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<Vec<ComputeStorage>, OverflowDetected> {
    let n = provider.dimension();
    let riemann = riemann_compute(provider, p)?;
    let at4 = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
    let mut ricci = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = compute_zero();
            for k in 0..n {
                acc = compute_checked_add(acc, riemann[at4(k, i, k, j)])?;
            }
            ricci.push(acc);
        }
    }
    Ok(ricci)
}

/// Compute Ricci tensor from a pre-computed Riemann tensor.
pub fn ricci_from_riemann(riemann: &Tensor, n: usize) -> FixedMatrix {
    let mut ricci = FixedMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = FixedPoint::ZERO;
            for k in 0..n {
                sum = sum + riemann.get(&[k, i, k, j]);
            }
            ricci.set(i, j, sum);
        }
    }
    ricci
}

// ============================================================================
// Scalar curvature
// ============================================================================

/// Compute scalar curvature R = g^{ij} R_{ij} at point p.
///
/// The full trace of the Ricci tensor with the inverse metric, both at the
/// compute tier, rounded once. Returns a single FixedPoint.
pub fn scalar_curvature(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<FixedPoint, OverflowDetected> {
    // Prefer closed-form if available (exact for constant-curvature spaces)
    if let Some(r) = provider.scalar_curvature_closed_form(p) {
        return Ok(r);
    }

    let n = provider.dimension();
    let g_inv = metric_inverse_compute(provider, p)?;
    let ricci = ricci_compute(provider, p)?;

    // R = g^{ij} R_{ij} = sum over i,j of g_inv[i,j] * ricci[i,j]
    let mut acc = compute_zero();
    for e in 0..n * n {
        acc = compute_checked_add(acc, compute_product(g_inv[e], ricci[e])?)?;
    }
    Ok(FixedPoint::from_raw(downscale_to_storage(acc)?))
}

/// Compute scalar curvature from pre-computed Ricci tensor and inverse metric.
pub fn scalar_from_ricci(g_inv: &FixedMatrix, ricci: &FixedMatrix) -> FixedPoint {
    let n = g_inv.rows();
    let mut g_flat = Vec::with_capacity(n * n);
    let mut r_flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            g_flat.push(g_inv.get(i, j).raw());
            r_flat.push(ricci.get(i, j).raw());
        }
    }
    FixedPoint::from_raw(compute_tier_dot_raw(&g_flat, &r_flat))
}

// ============================================================================
// Sectional curvature
// ============================================================================

/// Compute sectional curvature K(u, v) at point p.
///
/// K(u,v) = <R(u,v)v, u> / (|u|²|v|² - <u,v>²)
///
/// where <R(u,v)v, u> = g_{ls} R^l_{ijk} v^i u^j v^k u^s (R(∂_j, ∂_k)∂_i =
/// R^l_{ijk} ∂_l). Before 0.6.4 the contraction was R^l_{ijk} u^i v^j v^k,
/// which is zero in exact arithmetic (R^l_{ijk} is antisymmetric in j, k):
/// the result was rounding noise, not the curvature.
///
/// The denominator is the squared area of the parallelogram spanned by u and v.
/// Numerator, the three inner products and the difference in the denominator
/// (which cancels for near-parallel u, v) are at the compute tier; the zero
/// test is on the compute-tier denominator and the quotient is rounded once.
pub fn sectional_curvature(
    provider: &dyn MetricProvider,
    p: &FixedVector,
    u: &FixedVector,
    v: &FixedVector,
) -> Result<FixedPoint, OverflowDetected> {
    let n = provider.dimension();
    let g = provider.metric(p);
    let riemann = riemann_compute(provider, p)?;
    let at4 = |a: usize, b: usize, c: usize, d: usize| ((a * n + b) * n + c) * n + d;
    let uc: Vec<ComputeStorage> = (0..n).map(|i| upscale_to_compute(u[i].raw())).collect();
    let vc: Vec<ComputeStorage> = (0..n).map(|i| upscale_to_compute(v[i].raw())).collect();

    // g u and g v, exact at the compute tier (products of storage values);
    // w_s = g_{ls} u^l lowers the first index (g symmetric)
    let lower = |x: &[ComputeStorage]| -> Result<Vec<ComputeStorage>, OverflowDetected> {
        (0..n).map(|i| {
            let mut acc = compute_zero();
            for j in 0..n {
                acc = compute_checked_add(acc, compute_product(upscale_to_compute(g.get(i, j).raw()), x[j])?)?;
            }
            Ok(acc)
        }).collect()
    };
    let gu = lower(&uc)?;
    let gv = lower(&vc)?;

    // <R(u,v)v, u> = sum_{s,i,j,k} R^s_{ijk} v^i u^j v^k w_s
    let mut numerator = compute_zero();
    for s in 0..n {
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let r = riemann[at4(s, i, j, k)];
                    if r == compute_zero() { continue; }
                    let term = compute_product(compute_product(compute_product(r, vc[i])?, uc[j])?, vc[k])?;
                    numerator = compute_checked_add(numerator, compute_product(term, gu[s])?)?;
                }
            }
        }
    }

    // Denominator: <u,u>*<v,v> - <u,v>², all at the compute tier
    let inner = |x: &[ComputeStorage], gy: &[ComputeStorage]| -> Result<ComputeStorage, OverflowDetected> {
        let mut acc = compute_zero();
        for i in 0..n { acc = compute_checked_add(acc, compute_product(x[i], gy[i])?)?; }
        Ok(acc)
    };
    let uu = inner(&uc, &gu)?;
    let vv = inner(&vc, &gv)?;
    let uv = inner(&uc, &gv)?;
    let denom = checked_sub(compute_product(uu, vv)?, compute_product(uv, uv)?)?;
    if denom == compute_zero() {
        return Err(OverflowDetected::DomainError);
    }

    Ok(FixedPoint::from_raw(downscale_to_storage(compute_checked_divide(numerator, denom)?)?))
}

// ============================================================================
// Built-in metric providers for known manifolds
// ============================================================================

/// Flat Euclidean metric: g_ij = δ_ij.
///
/// All Christoffel symbols and curvature should be exactly zero.
pub struct EuclideanMetric {
    pub dim: usize,
}

impl MetricProvider for EuclideanMetric {
    fn dimension(&self) -> usize { self.dim }

    fn metric(&self, _p: &FixedVector) -> FixedMatrix {
        FixedMatrix::identity(self.dim)
    }

    fn metric_inverse(&self, _p: &FixedVector) -> Result<FixedMatrix, OverflowDetected> {
        Ok(FixedMatrix::identity(self.dim))
    }
}

/// Sphere S^n metric in spherical coordinates.
///
/// For S^2 (standard sphere), coordinates are (θ, φ) with:
///   g = [[1, 0], [0, sin²(θ)]]
///
/// Sectional curvature = 1 everywhere.
pub struct SphereMetric {
    pub radius: FixedPoint,
}

impl SphereMetric {
    /// r² at the compute tier (exact).
    fn radius_sq_compute(&self) -> ComputeStorage {
        let r = upscale_to_compute(self.radius.raw());
        compute_multiply(r, r)
    }
}

impl MetricProvider for SphereMetric {
    fn dimension(&self) -> usize { 2 }

    /// g = r² [[1, 0], [0, sin²θ]], each entry formed at the compute tier and
    /// rounded once (r² sin²θ was three storage roundings before 0.6.4).
    fn metric(&self, p: &FixedVector) -> FixedMatrix {
        // p = [θ, φ]
        let (sin_t, _) = sincos_at_compute_tier(upscale_to_compute(p[0].raw()));
        let r_sq = self.radius_sq_compute();
        let g11 = compute_multiply(r_sq, compute_multiply(sin_t, sin_t));
        let z = FixedPoint::ZERO;
        FixedMatrix::from_slice(2, 2, &[
            FixedPoint::from_raw(round_to_storage(r_sq)), z,
            z, FixedPoint::from_raw(round_to_storage(g11)),
        ])
    }

    /// Exact Christoffel symbols for S² in (θ, φ) coordinates.
    ///
    /// Γ^θ_{φφ} = -sin(θ)cos(θ)
    /// Γ^φ_{θφ} = Γ^φ_{φθ} = cos(θ)/sin(θ) = cot(θ)
    /// All others = 0.
    ///
    /// These are derived analytically from g = r²[[1,0],[0,sin²θ]].
    /// Uses the compute-tier sincos engine, no numerical differentiation
    /// involved; each symbol is rounded to storage once.
    fn christoffel_closed_form(&self, p: &FixedVector) -> Option<Tensor> {
        let gamma = self.christoffel_closed_form_compute(p)?;
        Some(round_tensor(&[2, 2, 2], &gamma.data).expect("sphere: Christoffel symbol exceeds storage"))
    }

    #[doc(hidden)]
    fn christoffel_closed_form_compute(&self, p: &FixedVector) -> Option<ComputeChristoffel> {
        let (sin_t, cos_t) = sincos_at_compute_tier(upscale_to_compute(p[0].raw()));
        let mut data = vec![compute_zero(); 8];
        // Γ^0_{11} = -sin(θ)cos(θ) (radius cancels in Christoffel)
        data[3] = compute_negate(compute_multiply(sin_t, cos_t));
        // Γ^1_{01} = Γ^1_{10} = cos(θ)/sin(θ) = cot(θ)
        if sin_t != compute_zero() {
            let cot_t = compute_divide(cos_t, sin_t).expect("sphere: cot(theta) exceeds the compute tier");
            data[5] = cot_t;
            data[6] = cot_t;
        }
        Some(ComputeChristoffel { data })
    }

    /// Exact scalar curvature for S²: R = 2/r², one rounding.
    fn scalar_curvature_closed_form(&self, _p: &FixedVector) -> Option<FixedPoint> {
        let two = compute_add(compute_one(), compute_one());
        let r = compute_divide(two, self.radius_sq_compute())
            .expect("sphere: 2 / r^2 exceeds the compute tier");
        Some(FixedPoint::from_raw(round_to_storage(r)))
    }
}

/// Hyperbolic space H^2 metric in the upper half-plane model.
///
/// Coordinates (x, y) with y > 0:
///   g = (1/y²) * [[1, 0], [0, 1]]
///
/// Sectional curvature = -1 everywhere.
pub struct HyperbolicMetric;

impl MetricProvider for HyperbolicMetric {
    fn dimension(&self) -> usize { 2 }

    /// g = (1/y²) I, 1/y² formed at the compute tier and rounded once.
    fn metric(&self, p: &FixedVector) -> FixedMatrix {
        // p = [x, y], y > 0
        let y = upscale_to_compute(p[1].raw());
        let scale = compute_divide(compute_one(), compute_multiply(y, y))
            .expect("hyperbolic metric: 1/y^2 exceeds the compute tier");
        let scale = FixedPoint::from_raw(round_to_storage(scale));
        let z = FixedPoint::ZERO;
        FixedMatrix::from_slice(2, 2, &[
            scale, z,
            z, scale,
        ])
    }

    /// Exact Christoffel symbols for H² upper half-plane.
    ///
    /// Γ^x_{xy} = Γ^x_{yx} = -1/y
    /// Γ^y_{xx} = 1/y
    /// Γ^y_{yy} = -1/y
    /// All others = 0.
    ///
    /// Derived analytically from g = (1/y²)·I.
    fn christoffel_closed_form(&self, p: &FixedVector) -> Option<Tensor> {
        let gamma = self.christoffel_closed_form_compute(p)?;
        Some(round_tensor(&[2, 2, 2], &gamma.data).expect("hyperbolic: Christoffel symbol exceeds storage"))
    }

    #[doc(hidden)]
    fn christoffel_closed_form_compute(&self, p: &FixedVector) -> Option<ComputeChristoffel> {
        let y = p[1];
        if y.is_zero() { return None; }
        let inv_y = compute_divide(compute_one(), upscale_to_compute(y.raw()))
            .expect("hyperbolic: 1/y exceeds the compute tier");
        let mut data = vec![compute_zero(); 8];
        // Γ^0_{01} = Γ^0_{10} = -1/y
        data[1] = compute_negate(inv_y);
        data[2] = compute_negate(inv_y);
        // Γ^1_{00} = 1/y
        data[4] = inv_y;
        // Γ^1_{11} = -1/y
        data[7] = compute_negate(inv_y);
        Some(ComputeChristoffel { data })
    }

    /// Exact scalar curvature for H²: R = -2.
    fn scalar_curvature_closed_form(&self, _p: &FixedVector) -> Option<FixedPoint> {
        Some(FixedPoint::from_int(-2))
    }
}

// ============================================================================
// Geodesic ODE and parallel transport ODE
// ============================================================================


/// ODE system for the geodesic equation on a Riemannian manifold.
///
/// State vector: [x^0, ..., x^{n-1}, v^0, ..., v^{n-1}] (position + velocity).
///
/// Equations:
///   dx^k/dt = v^k
///   dv^k/dt = -Γ^k_{ij} v^i v^j
///
/// Christoffel symbols are re-evaluated at each point along the trajectory
/// (either via closed-form or numerical differentiation, depending on the
/// MetricProvider implementation).
///
/// **FASC-UGOD integration:** The Christoffel symbols stay at the compute tier
/// and the Γ·v·v contraction is summed there (exact v^i v^j, one compute-tier
/// product with Γ per term), rounded to storage once per velocity component.
pub struct GeodesicOde<'a> {
    provider: &'a dyn MetricProvider,
}

impl<'a> OdeSystem for GeodesicOde<'a> {
    fn eval(&self, _t: FixedPoint, state: &FixedVector) -> FixedVector {
        let n = self.provider.dimension();
        let mut x = FixedVector::new(n);
        let mut v = FixedVector::new(n);
        for i in 0..n { x[i] = state[i]; v[i] = state[n + i]; }

        // Christoffel symbols at the current position, at the compute tier
        let gamma = match christoffel_compute(self.provider, &x) {
            Ok(g) => g,
            Err(_) => return FixedVector::new(2 * n), // zero on error
        };

        let mut dstate = FixedVector::new(2 * n);
        // dx^k/dt = v^k
        for k in 0..n { dstate[k] = v[k]; }

        // dv^k/dt = -Γ^k_{ij} v^i v^j, summed at the compute tier and rounded
        // once (v^i v^j of two storage values is exact at the compute tier;
        // 0.6.3 rounded it to storage before the dot product)
        let v_c = state_to_compute(&v);
        for k in 0..n {
            let mut acc = upscale_to_compute(FixedPoint::ZERO.raw());
            for i in 0..n {
                for j in 0..n {
                    let vv = compute_multiply(v_c[i], v_c[j]);
                    acc = compute_add(acc, compute_multiply(gamma[(k * n + i) * n + j], vv));
                }
            }
            dstate[n + k] = FixedPoint::from_raw(round_to_storage(compute_negate(acc)));
        }

        dstate
    }
}

/// Integrate the geodesic equation from an initial point and velocity.
///
/// Returns a sequence of points along the geodesic.
///
/// `num_steps` controls the number of RK4 steps. Step size h = total_time / num_steps.
///
/// The step boundaries k T / N, the steps between them and the state
/// [x, v] are carried at the compute tier across all steps; the state is
/// rounded to storage only to evaluate the Christoffel symbols and once per
/// reported point (0.6.3 rounded the state and the step to storage every step:
/// 32 units on a straight line of 96 steps at every split).
pub fn geodesic_integrate(
    provider: &dyn MetricProvider,
    initial_point: &FixedVector,
    initial_velocity: &FixedVector,
    total_time: FixedPoint,
    num_steps: usize,
) -> Result<Vec<FixedVector>, OverflowDetected> {
    let n = provider.dimension();
    // Step k runs from k T / N to (k + 1) T / N, both at the compute tier, so
    // the steps add up to T exactly and each is T / N to 2F bits.
    let time_at = |k: usize| -> ComputeStorage {
        compute_mul_div_int(upscale_to_compute(total_time.raw()), k as i64, num_steps as i64)
            .expect("geodesic: num_steps > 0 and the time fits")
    };

    // Build initial state [x, v] at the compute tier
    let mut state = FixedVector::new(2 * n);
    for i in 0..n { state[i] = initial_point[i]; state[n + i] = initial_velocity[i]; }
    let mut state = state_to_compute(&state);

    let sys = GeodesicOde { provider };
    let mut points = Vec::with_capacity(num_steps + 1);
    let mut t = time_at(0);

    // Position from the compute-tier state, rounded once
    let extract_pos = |s: &[ComputeStorage]| -> Result<FixedVector, OverflowDetected> {
        let mut p = FixedVector::new(n);
        for i in 0..n { p[i] = FixedPoint::from_raw(downscale_to_storage(s[i])?); }
        Ok(p)
    };

    points.push(initial_point.clone());

    for k in 0..num_steps {
        let t_next = time_at(k + 1);
        state = rk4_step_compute(&sys, t, &state, compute_subtract(t_next, t));
        t = t_next;
        points.push(extract_pos(&state)?);
    }

    Ok(points)
}

/// Parallel transport a tangent vector along a discrete curve.
///
/// Solves the parallel transport ODE:
///   dV^k/dt = -Γ^k_{ij} V^i (dx^j/dt)
///
/// where dx/dt is approximated by finite differences along the curve.
///
/// **FASC-UGOD integration:** V is carried at the compute tier along the whole
/// curve and rounded to storage once at the end. At each step:
/// - Christoffel symbols evaluated at current point (compute-tier contractions)
/// - Γ·V·dx contraction and the update of V at the compute tier
/// - Optional re-orthogonalization every `reorthog_interval` steps with the
///   projection coefficient and update at the compute tier
///
/// A compute-tier overflow in the update is `Err(TierOverflow)`, as is a
/// final V beyond the storage range.
///
/// `reorthog_interval`: re-orthogonalize V against the curve tangent every N steps.
/// Set to 0 to disable. Recommended: 10-50 for long curves.
pub fn parallel_transport_ode(
    provider: &dyn MetricProvider,
    curve: &[FixedVector],
    initial_vector: &FixedVector,
    reorthog_interval: usize,
) -> Result<FixedVector, OverflowDetected> {
    if curve.len() < 2 {
        return Ok(initial_vector.clone());
    }

    let n = provider.dimension();
    let mut v = state_to_compute(initial_vector);
    let zero = upscale_to_compute(FixedPoint::ZERO.raw());

    for step in 0..curve.len() - 1 {
        let p = &curve[step];
        let p_next = &curve[step + 1];

        // Curve tangent: dx = p_next - p
        let dx: Vec<FixedPoint> = (0..n).map(|i| p_next[i] - p[i]).collect();

        // Christoffel at current point, at the compute tier
        let gamma = christoffel_compute(provider, p)?;

        // dV^k = -Γ^k_{ij} V^i dx^j (one step of Euler; for RK4 on
        // the transport ODE, we'd need Christoffel at intermediate points),
        // with V, the contraction and the update at the compute tier
        let dx_c: Vec<ComputeStorage> = dx.iter().map(|d| upscale_to_compute(d.raw())).collect();
        let mut v_new = Vec::with_capacity(n);
        for k in 0..n {
            let mut correction = zero;
            for j in 0..n {
                // Sum over i: Γ^k_{ij} V^i
                let mut gamma_v = zero;
                for i in 0..n {
                    gamma_v = compute_checked_add(gamma_v, compute_product(gamma[(k * n + i) * n + j], v[i])?)?;
                }
                correction = compute_checked_add(correction, compute_product(gamma_v, dx_c[j])?)?;
            }
            v_new.push(checked_sub(v[k], correction)?);
        }

        v = v_new;

        // Re-orthogonalization: project V perpendicular to curve tangent
        if reorthog_interval > 0 && (step + 1) % reorthog_interval == 0 {
            let mut dx_norm_sq = zero;
            let mut v_dot_dx = zero;
            for i in 0..n {
                dx_norm_sq = compute_checked_add(dx_norm_sq, compute_product(dx_c[i], dx_c[i])?)?;
                v_dot_dx = compute_checked_add(v_dot_dx, compute_product(v[i], dx_c[i])?)?;
            }
            if dx_norm_sq != zero {
                let coeff = compute_checked_divide(v_dot_dx, dx_norm_sq)?;
                for i in 0..n {
                    v[i] = checked_sub(v[i], compute_product(dx_c[i], coeff)?)?;
                }
            }
        }
    }

    let mut out = FixedVector::new(n);
    for i in 0..n { out[i] = FixedPoint::from_raw(downscale_to_storage(v[i])?); }
    Ok(out)
}

/// `a - b` at the compute tier, `TierOverflow` instead of a wrap.
fn checked_sub(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    #[cfg(not(table_format = "q256_256"))]
    { a.checked_sub(b).ok_or(OverflowDetected::TierOverflow) }
    #[cfg(table_format = "q256_256")]
    {
        // I1024 has no checked_sub; the minimum has no negation
        if b == ComputeStorage::min_value() { return Err(OverflowDetected::TierOverflow); }
        compute_checked_add(a, compute_negate(b))
    }
}
