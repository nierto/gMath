//! L3B: Christoffel symbols and curvature tensors with fixed-point arithmetic.
//!
//! Provides numerical differential geometry on Riemannian manifolds:
//! - `numerical_derivative` — central difference with optimal h = 2^(-FRAC_BITS/3)
//! - `christoffel` — Γᵏᵢⱼ = ½gᵏˡ(∂ᵢgⱼˡ+∂ⱼgˡᵢ-∂ˡgᵢⱼ)
//! - `riemann_curvature` — R^l_{ijk} = ∂ⱼΓˡᵢₖ - ∂ₖΓˡᵢⱼ + ΓˡⱼₘΓᵐᵢₖ - ΓˡₖₘΓᵐᵢⱼ
//! - `ricci_tensor` — Rᵢⱼ = R^k_{ikj}
//! - `scalar_curvature` — R = gⁱʲRᵢⱼ
//! - `sectional_curvature` — K(u,v) = R(u,v,v,u)/(|u|²|v|²-<u,v>²)
//!
//! **FASC-UGOD integration:** Numerical differentiation uses h = 2^(-FRAC_BITS/3)
//! (power-of-2 for exact division by 2h via bit-shift). All contractions (Γ·g⁻¹,
//! Riemann·g) use compute_tier_dot_raw for 1-ULP accumulation. Riemann tensor
//! involves nested finite differences → O(h²) total error; scientific profile
//! recommended for curvature computations.

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::tensor::Tensor;
use super::linalg::compute_tier_dot_raw;
use super::derived::inverse;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// Numerical differentiation step size
// ============================================================================

/// Optimal step size for central differences: h = 2^(-FRAC_BITS/3).
///
/// This minimizes total error (truncation + rounding) for central differences.
/// Being a power of 2, division by 2h is an exact bit-shift (no rounding).
///
/// Profile values:
/// - Q64.64:    h ≈ 2^(-21) ≈ 4.8e-7
/// - Q128.128:  h ≈ 2^(-43) ≈ 1.1e-13
/// - Q256.256:  h ≈ 2^(-85) ≈ 2.6e-26
pub fn differentiation_step() -> FixedPoint {
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

/// Divide by 2h exactly via bit-shift.
/// Since h = 2^(-FRAC_BITS/3), 2h = 2^(-FRAC_BITS/3 + 1), and dividing by 2h
/// is equivalent to shifting left by (FRAC_BITS/3 - 1) then right by FRAC_BITS,
/// or equivalently, just a right-shift of the difference value.
#[inline]
fn divide_by_two_h(val: FixedPoint) -> FixedPoint {
    // val / (2h) where 2h = 2^(-FRAC_BITS/3 + 1)
    // = val * 2^(FRAC_BITS/3 - 1)
    // In Q-format, this means shifting the raw value left by (FRAC_BITS/3 - 1)
    #[cfg(table_format = "q64_64")]
    { FixedPoint::from_raw(val.raw() << 20u32) }  // 21 - 1 = 20
    #[cfg(table_format = "q128_128")]
    { FixedPoint::from_raw(val.raw() << 42usize) }  // 43 - 1 = 42
    #[cfg(table_format = "q256_256")]
    { FixedPoint::from_raw(val.raw() << 84usize) }  // 85 - 1 = 84
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
    /// Closed-form scalar curvature, if known analytically.
    ///
    /// Override this for constant-curvature spaces to get exact results.
    fn scalar_curvature_closed_form(&self, _p: &FixedVector) -> Option<FixedPoint> {
        None
    }
}

// ============================================================================
// Partial derivatives of the metric
// ============================================================================

/// Compute ∂_k g_ij at point p via central differences.
///
/// Returns an n×n matrix where entry (i,j) = ∂g_ij/∂x^k.
fn metric_partial(
    provider: &dyn MetricProvider,
    p: &FixedVector,
    k: usize,
) -> FixedMatrix {
    let h = differentiation_step();
    let n = provider.dimension();

    // p + h*e_k and p - h*e_k
    let mut p_plus = p.clone();
    let mut p_minus = p.clone();
    p_plus[k] = p_plus[k] + h;
    p_minus[k] = p_minus[k] - h;

    let g_plus = provider.metric(&p_plus);
    let g_minus = provider.metric(&p_minus);

    // (g(p+h) - g(p-h)) / (2h), exact bit-shift division
    let mut result = FixedMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let diff = g_plus.get(i, j) - g_minus.get(i, j);
            result.set(i, j, divide_by_two_h(diff));
        }
    }
    result
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
/// All contractions with g^{kl} use compute_tier_dot_raw for 1-ULP accumulation.
pub fn christoffel(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<Tensor, OverflowDetected> {
    // Prefer closed-form if available (exact, no numerical differentiation)
    if let Some(gamma) = provider.christoffel_closed_form(p) {
        return Ok(gamma);
    }

    let n = provider.dimension();
    let g_inv = provider.metric_inverse(p)?;

    // Pre-compute all metric partial derivatives ∂_k g_ij for k = 0..n
    let dg: Vec<FixedMatrix> = (0..n).map(|k| metric_partial(provider, p, k)).collect();

    // Compute Γ^k_{ij} = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{li} - ∂_l g_{ij})
    let mut gamma = Tensor::new(&[n, n, n]);
    let half = FixedPoint::one() / FixedPoint::from_int(2);

    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                // Compute sum over l: g^{kl} * (∂_i g_{jl} + ∂_j g_{li} - ∂_l g_{ij})
                let g_inv_row: Vec<BinaryStorage> = (0..n).map(|l| g_inv.get(k, l).raw()).collect();
                let bracket: Vec<BinaryStorage> = (0..n).map(|l| {
                    // ∂_i g_{jl} + ∂_j g_{li} - ∂_l g_{ij}
                    let term = dg[i].get(j, l) + dg[j].get(l, i) - dg[l].get(i, j);
                    term.raw()
                }).collect();
                let contracted = FixedPoint::from_raw(
                    compute_tier_dot_raw(&g_inv_row, &bracket)
                );
                gamma.set(&[k, i, j], half * contracted);
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
/// **Precision warning:** This involves nested finite differences (derivatives of
/// Christoffel symbols). Total error is O(h²) where h = differentiation_step().
/// For best precision, use the scientific profile.
pub fn riemann_curvature(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<Tensor, OverflowDetected> {
    let n = provider.dimension();
    let h = differentiation_step();

    // Compute Christoffel symbols at p and at neighboring points p ± h*e_k
    let gamma_center = christoffel(provider, p)?;

    // Derivatives of Christoffel: ∂_j Γ^l_{ik} via central difference
    // ∂_j Γ^l_{ik} = (Γ^l_{ik}(p + h*e_j) - Γ^l_{ik}(p - h*e_j)) / (2h)
    let mut dgamma: Vec<Tensor> = Vec::with_capacity(n);
    for j in 0..n {
        let mut p_plus = p.clone();
        let mut p_minus = p.clone();
        p_plus[j] = p_plus[j] + h;
        p_minus[j] = p_minus[j] - h;

        let gamma_plus = christoffel(provider, &p_plus)?;
        let gamma_minus = christoffel(provider, &p_minus)?;

        // (Γ_plus - Γ_minus) / (2h)
        let mut dg_j = Tensor::new(&[n, n, n]);
        for l in 0..n {
            for ii in 0..n {
                for kk in 0..n {
                    let diff = gamma_plus.get(&[l, ii, kk]) - gamma_minus.get(&[l, ii, kk]);
                    dg_j.set(&[l, ii, kk], divide_by_two_h(diff));
                }
            }
        }
        dgamma.push(dg_j);
    }

    // Assemble Riemann tensor
    let mut riemann = Tensor::new(&[n, n, n, n]);
    for l in 0..n {
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    // ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij}
                    let deriv_term = dgamma[j].get(&[l, i, k]) - dgamma[k].get(&[l, i, j]);

                    // Γ^l_{jm} Γ^m_{ik} - Γ^l_{km} Γ^m_{ij} (sum over m)
                    let gamma_jm: Vec<BinaryStorage> = (0..n).map(|m|
                        gamma_center.get(&[l, j, m]).raw()
                    ).collect();
                    let gamma_mik: Vec<BinaryStorage> = (0..n).map(|m|
                        gamma_center.get(&[m, i, k]).raw()
                    ).collect();
                    let gamma_km: Vec<BinaryStorage> = (0..n).map(|m|
                        gamma_center.get(&[l, k, m]).raw()
                    ).collect();
                    let gamma_mij: Vec<BinaryStorage> = (0..n).map(|m|
                        gamma_center.get(&[m, i, j]).raw()
                    ).collect();

                    let contraction_pos = FixedPoint::from_raw(
                        compute_tier_dot_raw(&gamma_jm, &gamma_mik)
                    );
                    let contraction_neg = FixedPoint::from_raw(
                        compute_tier_dot_raw(&gamma_km, &gamma_mij)
                    );

                    riemann.set(&[l, i, j, k], deriv_term + contraction_pos - contraction_neg);
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
/// Returns an n×n FixedMatrix.
pub fn ricci_tensor(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<FixedMatrix, OverflowDetected> {
    let n = provider.dimension();
    let riemann = riemann_curvature(provider, p)?;

    let mut ricci = FixedMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            // R_{ij} = R^k_{ikj} = sum over k of R[k, i, k, j]
            let k_vals: Vec<BinaryStorage> = (0..n).map(|k|
                riemann.get(&[k, i, k, j]).raw()
            ).collect();
            let ones: Vec<BinaryStorage> = (0..n).map(|_|
                FixedPoint::one().raw()
            ).collect();
            let val = FixedPoint::from_raw(compute_tier_dot_raw(&k_vals, &ones));
            ricci.set(i, j, val);
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
/// The full trace of the Ricci tensor with the inverse metric.
/// Returns a single FixedPoint.
pub fn scalar_curvature(
    provider: &dyn MetricProvider,
    p: &FixedVector,
) -> Result<FixedPoint, OverflowDetected> {
    // Prefer closed-form if available (exact for constant-curvature spaces)
    if let Some(r) = provider.scalar_curvature_closed_form(p) {
        return Ok(r);
    }

    let n = provider.dimension();
    let g_inv = provider.metric_inverse(p)?;
    let ricci = ricci_tensor(provider, p)?;

    // R = g^{ij} R_{ij} = sum over i,j of g_inv[i,j] * ricci[i,j]
    let mut g_flat = Vec::with_capacity(n * n);
    let mut r_flat = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            g_flat.push(g_inv.get(i, j).raw());
            r_flat.push(ricci.get(i, j).raw());
        }
    }

    Ok(FixedPoint::from_raw(compute_tier_dot_raw(&g_flat, &r_flat)))
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
/// K(u,v) = R(u,v,v,u) / (|u|²|v|² - <u,v>²)
///
/// where R(u,v,v,u) = R^l_{ijk} u^i v^j v^k g_{ls} u^s (lowered first index).
///
/// The denominator is the squared area of the parallelogram spanned by u and v.
/// Uses compute-tier accumulation throughout.
pub fn sectional_curvature(
    provider: &dyn MetricProvider,
    p: &FixedVector,
    u: &FixedVector,
    v: &FixedVector,
) -> Result<FixedPoint, OverflowDetected> {
    let n = provider.dimension();
    let g = provider.metric(p);
    let riemann = riemann_curvature(provider, p)?;

    // R(u,v,v,u) = R_{lijk} u^i v^j v^k u^l where R_{lijk} = g_{ls} R^s_{ijk}
    // = sum_{i,j,k,l,s} g_{ls} R^s_{ijk} u^i v^j v^k u^l
    // = sum_{i,j,k,s} R^s_{ijk} u^i v^j v^k (sum_l g_{ls} u^l)
    //
    // First lower the first index: w_s = g_{ls} u^l
    let mut w = FixedVector::new(n);
    for s in 0..n {
        let g_row: Vec<BinaryStorage> = (0..n).map(|l| g.get(l, s).raw()).collect();
        let u_raw: Vec<BinaryStorage> = (0..n).map(|l| u[l].raw()).collect();
        w[s] = FixedPoint::from_raw(compute_tier_dot_raw(&g_row, &u_raw));
    }

    // R(u,v,v,u) = sum_{s,i,j,k} R^s_{ijk} * u^i * v^j * v^k * w_s
    let mut numerator = FixedPoint::ZERO;
    for s in 0..n {
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let r_comp = riemann.get(&[s, i, j, k]);
                    if !r_comp.is_zero() {
                        numerator = numerator + r_comp * u[i] * v[j] * v[k] * w[s];
                    }
                }
            }
        }
    }

    // Denominator: <u,u>*<v,v> - <u,v>²
    // where <,> is the metric inner product
    let u_raw: Vec<BinaryStorage> = (0..n).map(|i| u[i].raw()).collect();
    let v_raw: Vec<BinaryStorage> = (0..n).map(|i| v[i].raw()).collect();

    // <u,u> = u^i g_{ij} u^j
    let gu: Vec<BinaryStorage> = (0..n).map(|i| {
        let g_row: Vec<BinaryStorage> = (0..n).map(|j| g.get(i, j).raw()).collect();
        compute_tier_dot_raw(&g_row, &u_raw)
    }).collect();
    let uu = FixedPoint::from_raw(compute_tier_dot_raw(&u_raw, &gu));

    // <v,v>
    let gv: Vec<BinaryStorage> = (0..n).map(|i| {
        let g_row: Vec<BinaryStorage> = (0..n).map(|j| g.get(i, j).raw()).collect();
        compute_tier_dot_raw(&g_row, &v_raw)
    }).collect();
    let vv = FixedPoint::from_raw(compute_tier_dot_raw(&v_raw, &gv));

    // <u,v>
    let uv = FixedPoint::from_raw(compute_tier_dot_raw(&u_raw, &gv));

    let denom = uu * vv - uv * uv;
    if denom.is_zero() {
        return Err(OverflowDetected::DomainError);
    }

    Ok(numerator / denom)
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

impl MetricProvider for SphereMetric {
    fn dimension(&self) -> usize { 2 }

    fn metric(&self, p: &FixedVector) -> FixedMatrix {
        // p = [θ, φ]
        let theta = p[0];
        let r_sq = self.radius * self.radius;
        let sin_theta = theta.sin();
        let z = FixedPoint::ZERO;
        FixedMatrix::from_slice(2, 2, &[
            r_sq, z,
            z, r_sq * sin_theta * sin_theta,
        ])
    }

    /// Exact Christoffel symbols for S² in (θ, φ) coordinates.
    ///
    /// Γ^θ_{φφ} = -sin(θ)cos(θ)
    /// Γ^φ_{θφ} = Γ^φ_{φθ} = cos(θ)/sin(θ) = cot(θ)
    /// All others = 0.
    ///
    /// These are derived analytically from g = r²[[1,0],[0,sin²θ]].
    /// Uses 0-ULP FASC sin/cos — no numerical differentiation involved.
    fn christoffel_closed_form(&self, p: &FixedVector) -> Option<Tensor> {
        let theta = p[0];
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let mut gamma = Tensor::new(&[2, 2, 2]);
        // Γ^0_{11} = -sin(θ)cos(θ) (radius cancels in Christoffel)
        gamma.set(&[0, 1, 1], -sin_t * cos_t);
        // Γ^1_{01} = Γ^1_{10} = cos(θ)/sin(θ) = cot(θ)
        if !sin_t.is_zero() {
            let cot_t = cos_t / sin_t;
            gamma.set(&[1, 0, 1], cot_t);
            gamma.set(&[1, 1, 0], cot_t);
        }
        Some(gamma)
    }

    /// Exact scalar curvature for S²: R = 2/r².
    fn scalar_curvature_closed_form(&self, _p: &FixedVector) -> Option<FixedPoint> {
        let r_sq = self.radius * self.radius;
        Some(FixedPoint::from_int(2) / r_sq)
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

    fn metric(&self, p: &FixedVector) -> FixedMatrix {
        // p = [x, y], y > 0
        let y = p[1];
        let y_sq = y * y;
        let scale = FixedPoint::one() / y_sq;
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
        let y = p[1];
        if y.is_zero() { return None; }
        let inv_y = FixedPoint::one() / y;

        let mut gamma = Tensor::new(&[2, 2, 2]);
        // Γ^0_{01} = Γ^0_{10} = -1/y
        gamma.set(&[0, 0, 1], -inv_y);
        gamma.set(&[0, 1, 0], -inv_y);
        // Γ^1_{00} = 1/y
        gamma.set(&[1, 0, 0], inv_y);
        // Γ^1_{11} = -1/y
        gamma.set(&[1, 1, 1], -inv_y);
        Some(gamma)
    }

    /// Exact scalar curvature for H²: R = -2.
    fn scalar_curvature_closed_form(&self, _p: &FixedVector) -> Option<FixedPoint> {
        Some(FixedPoint::from_int(-2))
    }
}

// ============================================================================
// Geodesic ODE and parallel transport ODE
// ============================================================================

use super::ode::{OdeSystem, rk4_step, OdePoint};

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
/// **FASC-UGOD integration:** The Γ·v·v contraction uses compute_tier_dot_raw
/// for 1-ULP accumulation per velocity component. RK4 weighted sums operate
/// at storage tier (FixedPoint arithmetic with tier N+1 multiplication).
pub struct GeodesicOde<'a> {
    provider: &'a dyn MetricProvider,
}

impl<'a> OdeSystem for GeodesicOde<'a> {
    fn eval(&self, _t: FixedPoint, state: &FixedVector) -> FixedVector {
        let n = self.provider.dimension();
        let mut x = FixedVector::new(n);
        let mut v = FixedVector::new(n);
        for i in 0..n { x[i] = state[i]; v[i] = state[n + i]; }

        // Evaluate Christoffel symbols at current position
        let gamma = match christoffel(self.provider, &x) {
            Ok(g) => g,
            Err(_) => return FixedVector::new(2 * n), // zero on error
        };

        let mut dstate = FixedVector::new(2 * n);
        // dx^k/dt = v^k
        for k in 0..n { dstate[k] = v[k]; }

        // dv^k/dt = -Γ^k_{ij} v^i v^j (compute-tier contraction)
        for k in 0..n {
            let mut gamma_k = Vec::with_capacity(n * n);
            let mut vv = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    gamma_k.push(gamma.get(&[k, i, j]).raw());
                    vv.push((v[i] * v[j]).raw());
                }
            }
            let contraction = FixedPoint::from_raw(
                compute_tier_dot_raw(&gamma_k, &vv)
            );
            dstate[n + k] = -contraction;
        }

        dstate
    }
}

/// Integrate the geodesic equation from an initial point and velocity.
///
/// Returns a sequence of points along the geodesic.
///
/// `num_steps` controls the number of RK4 steps. Step size h = total_time / num_steps.
pub fn geodesic_integrate(
    provider: &dyn MetricProvider,
    initial_point: &FixedVector,
    initial_velocity: &FixedVector,
    total_time: FixedPoint,
    num_steps: usize,
) -> Result<Vec<FixedVector>, OverflowDetected> {
    let n = provider.dimension();
    let h = total_time / FixedPoint::from_int(num_steps as i32);

    // Build initial state [x, v]
    let mut state = FixedVector::new(2 * n);
    for i in 0..n { state[i] = initial_point[i]; state[n + i] = initial_velocity[i]; }

    let sys = GeodesicOde { provider };
    let mut points = Vec::with_capacity(num_steps + 1);
    let mut t = FixedPoint::ZERO;

    // Extract position from state
    let extract_pos = |s: &FixedVector| -> FixedVector {
        let mut p = FixedVector::new(n);
        for i in 0..n { p[i] = s[i]; }
        p
    };

    points.push(extract_pos(&state));

    for _ in 0..num_steps {
        state = rk4_step(&sys, t, &state, h);
        t = t + h;
        points.push(extract_pos(&state));
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
/// **FASC-UGOD integration:** At each step:
/// - Christoffel symbols evaluated at current point (compute-tier contractions)
/// - Γ·V·dx contraction via compute_tier_dot_raw (1 ULP per component)
/// - Optional re-orthogonalization every `reorthog_interval` steps using
///   compute_tier_dot_raw for the projection
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
    let mut v = initial_vector.clone();

    for step in 0..curve.len() - 1 {
        let p = &curve[step];
        let p_next = &curve[step + 1];

        // Curve tangent: dx = p_next - p
        let dx: Vec<FixedPoint> = (0..n).map(|i| p_next[i] - p[i]).collect();

        // Christoffel at current point
        let gamma = christoffel(provider, p)?;

        // dV^k = -Γ^k_{ij} V^i dx^j (one step of Euler — for RK4 on
        // the transport ODE, we'd need Christoffel at intermediate points)
        let mut v_new = FixedVector::new(n);
        for k in 0..n {
            // -Γ^k_{ij} V^i dx^j via compute-tier contraction
            let gamma_k_v: Vec<BinaryStorage> = (0..n).map(|j| {
                // Sum over i: Γ^k_{ij} V^i
                let gamma_ki: Vec<BinaryStorage> = (0..n).map(|i|
                    gamma.get(&[k, i, j]).raw()
                ).collect();
                let v_raw: Vec<BinaryStorage> = (0..n).map(|i| v[i].raw()).collect();
                compute_tier_dot_raw(&gamma_ki, &v_raw)
            }).collect();
            let dx_raw: Vec<BinaryStorage> = dx.iter().map(|d| d.raw()).collect();
            let correction = FixedPoint::from_raw(
                compute_tier_dot_raw(&gamma_k_v, &dx_raw)
            );
            v_new[k] = v[k] - correction;
        }

        v = v_new;

        // Re-orthogonalization: project V perpendicular to curve tangent
        if reorthog_interval > 0 && (step + 1) % reorthog_interval == 0 {
            let dx_vec = FixedVector::from_slice(&dx);
            let dx_norm_sq = dx_vec.dot_precise(&dx_vec);
            if !dx_norm_sq.is_zero() {
                let v_dot_dx = v.dot_precise(&dx_vec);
                let coeff = v_dot_dx / dx_norm_sq;
                v = &v - &(&dx_vec * coeff);
            }
        }
    }

    Ok(v)
}
