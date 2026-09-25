//! Riemannian manifold trait and concrete implementations.
//!
//! L3A: Manifolds with closed-form geodesics (no ODE solver required):
//! - `EuclideanSpace`: R^n, flat
//! - `Sphere`: S^n embedded in R^{n+1}
//! - `HyperbolicSpace`: H^n in the hyperboloid model
//!
//! L3C: Manifolds requiring matrix function infrastructure:
//! - `SPDManifold`: Sym⁺(n), symmetric positive-definite matrices
//! - `Grassmannian`: Gr(k,n), k-dimensional subspaces of R^n

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::compute_matrix::{ComputeMatrix, compute_lu_decompose};
use super::matrix_functions::{matrix_exp_compute, matrix_log_compute, matrix_sqrt_compute};
use super::decompose::{svd_decompose, qr_decompose, cholesky_decompose};
use super::linalg::{ComputeStorage, compute_product, compute_tier_sqrt_dot, downscale_to_storage, exact_dot, sincos_at_compute_tier, upscale_to_compute};
use super::wide_acc::{Wide, divide_to_compute_nearest, exact_dot_compute, narrow_product_to_compute, widen_product};
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_checked_add, compute_halve, compute_is_negative, compute_is_zero, compute_negate, compute_subtract,
    exp_at_compute_tier, exp_sentinel_reached, ln_at_compute_tier, make_compute_int, sqrt_at_compute_tier,
};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// Compute-tier helpers (2 x FRAC_BITS fractional bits)
// ============================================================================
//
// Every multi-step manifold operation below keeps its state at the compute
// tier and narrows each output to storage once. Products of storage values
// are exact at the compute tier; sums of compute-tier products are formed
// exactly on the wide accumulator and rounded once.

type C = ComputeStorage;

#[inline]
fn raws(v: &FixedVector) -> Vec<BinaryStorage> {
    (0..v.len()).map(|i| v[i].raw()).collect()
}

#[inline]
fn up_vec(v: &FixedVector) -> Vec<C> {
    (0..v.len()).map(|i| upscale_to_compute(v[i].raw())).collect()
}

/// One rounding to storage, `Err(TierOverflow)` beyond its range.
#[inline]
fn down(x: C) -> Result<FixedPoint, OverflowDetected> {
    Ok(FixedPoint::from_raw(downscale_to_storage(x)?))
}

fn down_vec(v: &[C]) -> Result<FixedVector, OverflowDetected> {
    let mut out = FixedVector::new(v.len());
    for (i, x) in v.iter().enumerate() { out[i] = down(*x)?; }
    Ok(out)
}

/// A compute matrix narrowed to storage, one rounding per entry, checked.
fn down_matrix(m: &ComputeMatrix) -> Result<FixedMatrix, OverflowDetected> {
    let mut out = FixedMatrix::new(m.rows(), m.cols());
    for r in 0..m.rows() {
        for c in 0..m.cols() { out.set(r, c, down(m.get(r, c))?); }
    }
    Ok(out)
}

#[inline]
fn c_add(a: C, b: C) -> Result<C, OverflowDetected> { compute_checked_add(a, b) }

#[inline]
fn c_sub(a: C, b: C) -> Result<C, OverflowDetected> { compute_checked_add(a, compute_negate(b)) }

/// `sum a_i b_i` of compute values: exact, then one rounding at the compute tier.
fn dot_c(a: &[C], b: &[C]) -> Result<C, OverflowDetected> {
    narrow_product_to_compute(exact_dot_compute(a, b)?)
}

/// Minkowski product `-a_0 b_0 + sum a_i b_i` of compute values, exact then one rounding.
fn mdot_c(a: &[C], b: &[C]) -> Result<C, OverflowDetected> {
    narrow_product_to_compute(exact_dot_compute(&a[1..], &b[1..])?.add_exact(-widen_product(a[0], b[0]))?)
}

/// `x y / z` with the product exact and one rounding (nearest) at the compute tier.
#[inline]
fn mul_div_c(x: C, y: C, z: C) -> Result<C, OverflowDetected> {
    divide_to_compute_nearest(widen_product(x, y), z)
}

/// sqrt of a compute value, a negative (rounding noise) taken as zero.
#[inline]
fn sqrt_c(x: C) -> C {
    if compute_is_negative(&x) { make_compute_int(0) } else { sqrt_at_compute_tier(x) }
}

/// `sqrt(a b - c d)` of compute values, the difference exact before its one rounding.
fn sqrt_det_c(a: C, b: C, c: C, d: C) -> Result<C, OverflowDetected> {
    Ok(sqrt_c(narrow_product_to_compute(widen_product(a, b).add_exact(-widen_product(c, d))?)?))
}

/// atan2(y, x) at the compute tier (the profile's engine).
fn atan2_c(y: C, x: C) -> C {
    use crate::fixed_point::domains::binary_fixed::transcendental::atan_tier_n_plus_1 as engine;
    #[cfg(table_format = "q256_256")]
    { engine::atan2_compute_tier_i1024(y, x) }
    #[cfg(table_format = "q128_128")]
    { engine::atan2_compute_tier_i512(y, x) }
    #[cfg(table_format = "q64_64")]
    { engine::atan2_compute_tier_i256(y, x) }
    #[cfg(table_format = "q32_32")]
    { engine::atan2_binary_i128(y, x) }
    #[cfg(table_format = "q16_16")]
    { engine::atan2_compute_tier_i64(y, x) }
}

/// (sinh x, cosh x) at the compute tier from one exp pair; `Err(TierOverflow)`
/// when an exp reaches its overflow sentinel (the result already exceeds
/// storage, and on some profiles the sentinel would narrow to a plausible
/// maximum instead of failing).
fn sinhcosh_c(x: C) -> Result<(C, C), OverflowDetected> {
    let ep = exp_at_compute_tier(x);
    let en = exp_at_compute_tier(compute_negate(x));
    if exp_sentinel_reached(&ep) || exp_sentinel_reached(&en) {
        return Err(OverflowDetected::TierOverflow);
    }
    Ok((compute_halve(compute_subtract(ep, en)), compute_halve(compute_checked_add(ep, en)?)))
}

/// Squared Frobenius norm of a compute matrix: exact, one rounding.
fn frobenius_sq_c(m: &ComputeMatrix) -> Result<C, OverflowDetected> {
    let mut v = Vec::with_capacity(m.rows() * m.cols());
    for r in 0..m.rows() {
        for c in 0..m.cols() { v.push(m.get(r, c)); }
    }
    dot_c(&v, &v)
}

/// Euclidean norm of a storage vector: exact sum of squares and root at the
/// compute tier, one rounding.
fn euclidean_norm(v: &FixedVector) -> FixedPoint {
    let r = raws(v);
    FixedPoint::from_raw(compute_tier_sqrt_dot(&r, &r).expect("manifold norm exceeds the storage range"))
}

// ============================================================================
// Manifold trait
// ============================================================================

/// A Riemannian manifold with fixed-point arithmetic.
///
/// Points and tangent vectors are represented as `FixedVector`. The embedding
/// dimension may differ from the intrinsic dimension (e.g., S^n uses R^{n+1}).
pub trait Manifold {
    /// Intrinsic dimension of the manifold.
    fn dimension(&self) -> usize;

    /// Riemannian metric: inner product <u, v>_p of tangent vectors at point p.
    fn inner_product(
        &self,
        base: &FixedVector,
        u: &FixedVector,
        v: &FixedVector,
    ) -> FixedPoint;

    /// Norm of a tangent vector: ||v||_p = sqrt(<v, v>_p).
    fn norm(&self, base: &FixedVector, v: &FixedVector) -> FixedPoint {
        self.inner_product(base, v, v).sqrt()
    }

    /// Exponential map: exp_p(v) maps tangent vector v at p to a manifold point.
    fn exp_map(
        &self,
        base: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected>;

    /// Logarithmic map: log_p(q) returns the tangent vector v at p such that exp_p(v) = q.
    fn log_map(
        &self,
        base: &FixedVector,
        target: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected>;

    /// Geodesic distance: d(p, q) = ||log_p(q)||_p.
    fn distance(
        &self,
        p: &FixedVector,
        q: &FixedVector,
    ) -> Result<FixedPoint, OverflowDetected>;

    /// Parallel transport: move tangent vector v from p to q along the geodesic.
    fn parallel_transport(
        &self,
        base: &FixedVector,
        target: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected>;
}

// ============================================================================
// Euclidean space R^n
// ============================================================================

/// Flat Euclidean space R^n.
pub struct EuclideanSpace {
    pub dim: usize,
}

impl Manifold for EuclideanSpace {
    fn dimension(&self) -> usize { self.dim }

    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint {
        u.dot_precise(v)
    }

    /// ||v||: exact sum of squares and root at the compute tier, one rounding.
    fn norm(&self, _base: &FixedVector, v: &FixedVector) -> FixedPoint {
        euclidean_norm(v)
    }

    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        Ok(base + tangent)
    }

    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        Ok(target - base)
    }

    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        Ok(p.metric_distance_safe(q))
    }

    fn parallel_transport(&self, _base: &FixedVector, _target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        Ok(tangent.clone()) // trivial in flat space
    }
}

// ============================================================================
// n-Sphere S^n
// ============================================================================

/// The n-sphere S^n embedded as unit vectors in R^{n+1}.
///
/// Points are (n+1)-dimensional unit vectors.
/// Tangent vectors at p are orthogonal to p in R^{n+1}.
pub struct Sphere {
    pub dim: usize, // intrinsic dimension; ambient = dim + 1
}

impl Sphere {
    /// The geodesic from p to q at the compute tier: the angle
    /// `theta = atan2(|p x q|, p.q)` and the direction `w = q - (p.q / |p|^2) p`
    /// with its length. `p.q`, `|p|^2` and `|q|^2` are exact, `|p x q|^2 =
    /// |p|^2 |q|^2 - (p.q)^2` is exact before its one rounding, and the angle
    /// is scale-invariant, so stored points off the unit sphere by a unit
    /// change it by far less than a unit. acos of p.q rounded to storage (before
    /// 0.6.4) lost half the bits for close points: 1 / theta amplification of
    /// both that rounding and the stored points' distance from the sphere.
    fn geodesic(p: &FixedVector, q: &FixedVector) -> Result<(C, Vec<C>, C), OverflowDetected> {
        assert_eq!(p.len(), q.len(), "Sphere: dimension mismatch");
        let (pr, qr) = (raws(p), raws(q));
        let pp = exact_dot(&pr, &pr)?;
        let qq = exact_dot(&qr, &qr)?;
        if compute_is_zero(&pp) || compute_is_zero(&qq) {
            return Err(OverflowDetected::DomainError);
        }
        let c = exact_dot(&pr, &qr)?;
        let theta = atan2_c(sqrt_det_c(pp, qq, c, c)?, c);
        let mut w = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            let along = mul_div_c(c, upscale_to_compute(pr[i]), pp)?;
            w.push(c_sub(upscale_to_compute(qr[i]), along)?);
        }
        let w_len = sqrt_c(dot_c(&w, &w)?);
        Ok((theta, w, w_len))
    }
}

impl Manifold for Sphere {
    fn dimension(&self) -> usize { self.dim }

    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint {
        u.dot_precise(v)
    }

    /// ||v||: exact sum of squares and root at the compute tier, one rounding.
    fn norm(&self, _base: &FixedVector, v: &FixedVector) -> FixedPoint {
        euclidean_norm(v)
    }

    /// `cos(theta) p + sin(theta) v / theta` with `theta = |v|`: the root,
    /// sin and cos, the quotient and the sum at the compute tier, one rounding
    /// per component. (Before 0.6.4 theta, sin, cos, 1/theta and each product
    /// were rounded to storage, and 1/theta overflowed Q8.24 for |v| < 1/128.)
    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let vr = raws(tangent);
        let theta = sqrt_c(exact_dot(&vr, &vr)?);
        if compute_is_zero(&theta) {
            return Ok(base.clone());
        }
        let (sin_t, cos_t) = sincos_at_compute_tier(theta);
        let mut out = Vec::with_capacity(base.len());
        for i in 0..base.len() {
            let along = compute_product(cos_t, upscale_to_compute(base[i].raw()))?;
            let across = mul_div_c(sin_t, upscale_to_compute(vr[i]), theta)?;
            out.push(c_add(along, across)?);
        }
        down_vec(&out)
    }

    /// `theta w / |w|` from [`Sphere::geodesic`]: angle, direction and
    /// scaling at the compute tier, one rounding per component.
    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let (theta, w, w_len) = Self::geodesic(base, target)?;
        if compute_is_zero(&theta) || compute_is_zero(&w_len) {
            return Ok(FixedVector::new(base.len()));
        }
        let mut out = Vec::with_capacity(w.len());
        for x in &w { out.push(mul_div_c(*x, theta, w_len)?); }
        down_vec(&out)
    }

    /// `atan2(|p x q|, p.q)` at the compute tier, one rounding.
    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        down(Self::geodesic(p, q)?.0)
    }

    /// `v - <v, p+q> / (1 + <p,q>) (p + q)`: the products exact, the
    /// coefficient and each component formed at the compute tier with one
    /// rounding to storage. `Err(DomainError)` for antipodal points, and
    /// `Err(TierOverflow)` when the coefficient near them leaves the range.
    fn parallel_transport(&self, base: &FixedVector, target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let (pr, qr, vr) = (raws(base), raws(target), raws(tangent));
        let denom = c_add(make_compute_int(1), exact_dot(&pr, &qr)?)?;
        if compute_is_zero(&denom) {
            // Antipodal points: transport is ambiguous
            return Err(OverflowDetected::DomainError);
        }
        let num = c_add(exact_dot(&vr, &pr)?, exact_dot(&vr, &qr)?)?;
        let mut out = Vec::with_capacity(vr.len());
        for i in 0..vr.len() {
            let sum_i = c_add(upscale_to_compute(pr[i]), upscale_to_compute(qr[i]))?;
            out.push(c_sub(upscale_to_compute(vr[i]), mul_div_c(num, sum_i, denom)?)?);
        }
        down_vec(&out)
    }
}

// ============================================================================
// Hyperbolic space H^n (hyperboloid model)
// ============================================================================

/// Hyperbolic space H^n in the hyperboloid model.
///
/// Points are (n+1)-dimensional vectors satisfying -x₀² + x₁² + ... + xₙ² = -1, x₀ > 0.
/// Uses the Minkowski inner product: <u,v>_M = -u₀v₀ + u₁v₁ + ... + uₙvₙ.
pub struct HyperbolicSpace {
    pub dim: usize, // intrinsic dimension; ambient = dim + 1
}

impl HyperbolicSpace {
    /// Minkowski inner product: <u,v>_M = -u[0]*v[0] + sum_{i>0} u[i]*v[i].
    ///
    /// Uses compute-tier accumulation for the positive part.
    fn minkowski_dot(u: &FixedVector, v: &FixedVector) -> FixedPoint {
        FixedPoint::from_raw(super::linalg::round_to_storage(Self::minkowski_dot_compute(u, v)))
    }

    /// -u0 v0 + sum_{i>=1} u_i v_i as ONE compute-tier sum. On the hyperboloid
    /// the two parts nearly cancel (-x0^2 + |x|^2 = -1 with x0 large); they
    /// were rounded to storage separately before 0.6.4, which the
    /// cancellation then amplified.
    fn minkowski_dot_compute(u: &FixedVector, v: &FixedVector) -> super::linalg::ComputeStorage {
        use crate::fixed_point::universal::fasc::stack_evaluator::compute::{compute_checked_add, compute_negate};
        assert_eq!(u.len(), v.len());
        let n = u.len();
        if n == 0 { return super::linalg::upscale_to_compute(FixedPoint::ZERO.raw()); }
        let u_raw: Vec<BinaryStorage> = (1..n).map(|i| u[i].raw()).collect();
        let v_raw: Vec<BinaryStorage> = (1..n).map(|i| v[i].raw()).collect();
        let spatial = super::linalg::compute_tier_dot_acc(&u_raw, &v_raw);
        let temporal = super::linalg::compute_tier_dot_acc(&[u[0].raw()], &[v[0].raw()]);
        compute_checked_add(spatial, compute_negate(temporal)).expect("Minkowski product exceeds the compute tier")
    }

    /// Minkowski norm: sqrt(<v,v>_M) for spacelike vectors (tangent vectors).
    /// For tangent vectors on H^n, <v,v>_M >= 0. Product and root at the
    /// compute tier, one rounding.
    fn minkowski_norm(v: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        use crate::fixed_point::universal::fasc::stack_evaluator::compute::{compute_is_negative, compute_negate, sqrt_at_compute_tier};
        let dot = Self::minkowski_dot_compute(v, v);
        // Timelike (negative) shouldn't happen for tangent vectors: handled
        // gracefully as before, with the magnitude's root
        let dot = if compute_is_negative(&dot) { compute_negate(dot) } else { dot };
        Ok(FixedPoint::from_raw(super::linalg::downscale_to_storage(sqrt_at_compute_tier(dot))?))
    }

    /// The geodesic from p to q at the compute tier: the distance
    /// `d = acosh(a / sqrt(PP QQ))` with `a = -<p,q>`, `PP = -<p,p>`,
    /// `QQ = -<q,q>` (all exact), formed as
    /// `ln((a + sqrt(a^2 - PP QQ)) / sqrt(PP QQ))` with `a^2 - PP QQ` exact
    /// before its one rounding, and the direction `w = q - (a / PP) p`
    /// (Minkowski-orthogonal to p) with its length `|w|_L`. The distance is
    /// scale-invariant: stored points off the hyperboloid by a unit change it
    /// by far less than a unit. acosh of `a` rounded to storage (before 0.6.4)
    /// lost half the bits for close points (1 / d amplification).
    /// `Err(DomainError)` unless p and q are timelike on the same sheet.
    fn geodesic(p: &FixedVector, q: &FixedVector) -> Result<(C, Vec<C>, C), OverflowDetected> {
        assert_eq!(p.len(), q.len(), "HyperbolicSpace: dimension mismatch");
        let alpha = compute_negate(Self::minkowski_dot_compute(p, q));
        let pp = compute_negate(Self::minkowski_dot_compute(p, p));
        let qq = compute_negate(Self::minkowski_dot_compute(q, q));
        let zero = make_compute_int(0);
        let positive = |x: &C| !compute_is_negative(x) && !compute_is_zero(x);
        if !(positive(&pp) && positive(&qq) && positive(&alpha)) {
            return Err(OverflowDetected::DomainError);
        }
        let sinh_part = sqrt_det_c(alpha, alpha, pp, qq)?;
        let scale = sqrt_c(narrow_product_to_compute(widen_product(pp, qq))?);
        let ratio = divide_to_compute_nearest(widen_product(c_add(alpha, sinh_part)?, make_compute_int(1)), scale)?;
        // ratio >= 1 exactly; below 1 only by rounding, where d is 0
        let d = ln_at_compute_tier(ratio);
        let d = if compute_is_negative(&d) { zero } else { d };
        let (pc, qc) = (up_vec(p), up_vec(q));
        let mut w = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            w.push(c_sub(qc[i], mul_div_c(alpha, pc[i], pp)?)?);
        }
        let w_len = sqrt_c(mdot_c(&w, &w)?);
        Ok((d, w, w_len))
    }
}

impl Manifold for HyperbolicSpace {
    fn dimension(&self) -> usize { self.dim }

    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint {
        Self::minkowski_dot(u, v)
    }

    fn norm(&self, _base: &FixedVector, v: &FixedVector) -> FixedPoint {
        Self::minkowski_norm(v).unwrap_or(FixedPoint::ZERO)
    }

    /// `cosh(theta) p + sinh(theta) v / theta` with `theta = |v|_L`: the
    /// root, the shared exp pair, the quotient and the sum at the compute
    /// tier, one rounding per component. (Before 0.6.4 theta, sinh, cosh,
    /// 1/theta and each product were rounded to storage.)
    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let tc = up_vec(tangent);
        let theta_sq = Self::minkowski_dot_compute(tangent, tangent);
        // Timelike (negative) shouldn't happen for tangent vectors: handled
        // gracefully as before, with the magnitude's root
        let theta = sqrt_c(if compute_is_negative(&theta_sq) { compute_negate(theta_sq) } else { theta_sq });
        if compute_is_zero(&theta) {
            return Ok(base.clone());
        }
        let (sinh_t, cosh_t) = sinhcosh_c(theta)?;
        let mut out = Vec::with_capacity(base.len());
        for i in 0..base.len() {
            let along = compute_product(cosh_t, upscale_to_compute(base[i].raw()))?;
            out.push(c_add(along, mul_div_c(sinh_t, tc[i], theta)?)?);
        }
        down_vec(&out)
    }

    /// `d w / |w|_L` from [`HyperbolicSpace::geodesic`]: distance, direction
    /// and scaling at the compute tier, one rounding per component.
    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let (d, w, w_len) = Self::geodesic(base, target)?;
        if compute_is_zero(&d) || compute_is_zero(&w_len) {
            return Ok(FixedVector::new(base.len()));
        }
        let mut out = Vec::with_capacity(w.len());
        for x in &w { out.push(mul_div_c(*x, d, w_len)?); }
        down_vec(&out)
    }

    /// `acosh(-<p,q> / sqrt(<p,p> <q,q>))` at the compute tier, one rounding.
    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        down(Self::geodesic(p, q)?.0)
    }

    /// `v + <v,u>_L (sinh(d) p + (cosh(d) - 1) u)` with `u = w / |w|_L` the
    /// unit direction and `d` the distance of [`HyperbolicSpace::geodesic`]:
    /// everything at the compute tier, one rounding per component. (Before
    /// 0.6.4 it went through the rounded log map and its norm, 1/theta, and
    /// each rounded product: about twelve storage roundings.)
    fn parallel_transport(&self, base: &FixedVector, target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let (d, w, w_len) = Self::geodesic(base, target)?;
        if compute_is_zero(&d) || compute_is_zero(&w_len) {
            return Ok(tangent.clone());
        }
        let one = make_compute_int(1);
        let vc = up_vec(tangent);
        // <v, u>_L = <v, w>_L / |w|_L: the product exact, one rounding
        let vw = exact_dot_compute(&vc[1..], &w[1..])?.add_exact(-widen_product(vc[0], w[0]))?;
        let coeff = divide_to_compute_nearest(vw, w_len)?;
        let (sinh_d, cosh_d) = sinhcosh_c(d)?;
        let cosh_m1 = c_sub(cosh_d, one)?;
        let mut out = Vec::with_capacity(vc.len());
        for i in 0..vc.len() {
            let u_i = mul_div_c(w[i], one, w_len)?;
            let dir = c_add(compute_product(sinh_d, upscale_to_compute(base[i].raw()))?, compute_product(cosh_m1, u_i)?)?;
            out.push(c_add(vc[i], compute_product(coeff, dir)?)?);
        }
        down_vec(&out)
    }
}

// ============================================================================
// L3C: SPD manifold Sym⁺(n), symmetric positive-definite matrices
// ============================================================================

/// The manifold of n×n symmetric positive-definite matrices.
///
/// Points are SPD matrices (stored as FixedMatrix).
/// Tangent vectors are symmetric matrices (same storage).
///
/// **Riemannian metric at P:**
///   <U, V>_P = tr(P⁻¹ U P⁻¹ V)
///
/// **Geodesics (closed-form via matrix functions):**
///   exp_P(V) = P^½ expm(P^{-½} V P^{-½}) P^½
///   log_P(Q) = P^½ logm(P^{-½} Q P^{-½}) P^½
///
/// **FASC-UGOD integration:** exp_map, log_map, distance and transport run
/// the compute-tier matrix square root, inverse, products and expm / logm as
/// one ComputeMatrix chain with a single downscale per entry at the end
/// (since 0.6.4; before, P^1/2 and its inverse were rounded to storage).
pub struct SPDManifold {
    pub n: usize,
}

/// Pack a symmetric matrix into a vector (upper triangle, row-major).
/// Dimension: n*(n+1)/2.
fn sym_to_vec(m: &FixedMatrix) -> FixedVector {
    let n = m.rows();
    let dim = n * (n + 1) / 2;
    let mut v = FixedVector::new(dim);
    let mut k = 0;
    for i in 0..n {
        for j in i..n {
            v[k] = m.get(i, j);
            k += 1;
        }
    }
    v
}

/// Unpack a vector into a symmetric matrix.
fn vec_to_sym(v: &FixedVector, n: usize) -> FixedMatrix {
    let mut m = FixedMatrix::new(n, n);
    let mut k = 0;
    for i in 0..n {
        for j in i..n {
            m.set(i, j, v[k]);
            m.set(j, i, v[k]); // symmetric
            k += 1;
        }
    }
    m
}

impl SPDManifold {
    /// P^1/2 and P^-1/2 at the compute tier (Denman-Beavers, then a
    /// compute-tier LU inverse), no storage rounding. `Err(DomainError)` for a
    /// base point that is not positive definite (Cholesky check).
    fn sqrt_and_inv_sqrt(p: &FixedMatrix) -> Result<(ComputeMatrix, ComputeMatrix), OverflowDetected> {
        cholesky_decompose(p)?;
        let sqrt_p = matrix_sqrt_compute(&ComputeMatrix::from_fixed_matrix(p))?;
        let inv_sqrt_p = compute_lu_decompose(&sqrt_p)?.inverse()?;
        Ok((sqrt_p, inv_sqrt_p))
    }

    /// P^-1 at the compute tier.
    fn inverse_compute(p: &FixedMatrix) -> Result<ComputeMatrix, OverflowDetected> {
        compute_lu_decompose(&ComputeMatrix::from_fixed_matrix(p))?.inverse()
    }

    /// tr(A B) of compute matrices: the products exact, one rounding.
    fn trace_of_product(a: &ComputeMatrix, b: &ComputeMatrix) -> Result<C, OverflowDetected> {
        let n = a.rows();
        let mut x = Vec::with_capacity(n * n);
        let mut y = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                x.push(a.get(i, j));
                y.push(b.get(j, i));
            }
        }
        dot_c(&x, &y)
    }
}

impl Manifold for SPDManifold {
    fn dimension(&self) -> usize {
        // Intrinsic dimension of Sym⁺(n) = n*(n+1)/2
        self.n * (self.n + 1) / 2
    }

    /// `tr(P⁻¹ U P⁻¹ V)`: the inverse and both products at the compute tier,
    /// the trace exact, one rounding. Panics if P is singular (it fell back
    /// to the identity metric silently before 0.6.4).
    fn inner_product(
        &self,
        base: &FixedVector,
        u: &FixedVector,
        v: &FixedVector,
    ) -> FixedPoint {
        let p_inv = Self::inverse_compute(&vec_to_sym(base, self.n))
            .expect("SPDManifold::inner_product: base point is singular");
        let a = p_inv.mat_mul(&ComputeMatrix::from_fixed_matrix(&vec_to_sym(u, self.n)));
        let b = p_inv.mat_mul(&ComputeMatrix::from_fixed_matrix(&vec_to_sym(v, self.n)));
        down(Self::trace_of_product(&a, &b).expect("SPDManifold::inner_product exceeds the compute tier"))
            .expect("SPDManifold::inner_product exceeds the storage range")
    }

    /// `sqrt(tr((P⁻¹ V)^2))` at the compute tier, one rounding.
    fn norm(&self, base: &FixedVector, v: &FixedVector) -> FixedPoint {
        let p_inv = Self::inverse_compute(&vec_to_sym(base, self.n))
            .expect("SPDManifold::norm: base point is singular");
        let a = p_inv.mat_mul(&ComputeMatrix::from_fixed_matrix(&vec_to_sym(v, self.n)));
        down(sqrt_c(Self::trace_of_product(&a, &a).expect("SPDManifold::norm exceeds the compute tier")))
            .expect("SPDManifold::norm exceeds the storage range")
    }

    /// `P^1/2 expm(P^-1/2 V P^-1/2) P^1/2`: square root, inverse, products
    /// and exponential all at the compute tier, one rounding per entry.
    /// (Before 0.6.4 P^1/2 and its inverse were rounded to storage, and the
    /// products and expm rounded in between: about six matrix roundings.)
    fn exp_map(
        &self,
        base: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let (sqrt_p, inv_sqrt_p) = Self::sqrt_and_inv_sqrt(&vec_to_sym(base, self.n))?;
        let v = ComputeMatrix::from_fixed_matrix(&vec_to_sym(tangent, self.n));
        let inner = inv_sqrt_p.mat_mul(&v).mat_mul(&inv_sqrt_p);
        let result = sqrt_p.mat_mul(&matrix_exp_compute(&inner)?).mat_mul(&sqrt_p);
        Ok(sym_to_vec(&down_matrix(&result)?))
    }

    /// `P^1/2 logm(P^-1/2 Q P^-1/2) P^1/2`, all at the compute tier, one
    /// rounding per entry.
    fn log_map(
        &self,
        base: &FixedVector,
        target: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let (sqrt_p, inv_sqrt_p) = Self::sqrt_and_inv_sqrt(&vec_to_sym(base, self.n))?;
        let q = ComputeMatrix::from_fixed_matrix(&vec_to_sym(target, self.n));
        let inner = inv_sqrt_p.mat_mul(&q).mat_mul(&inv_sqrt_p);
        let result = sqrt_p.mat_mul(&matrix_log_compute(&inner)?).mat_mul(&sqrt_p);
        Ok(sym_to_vec(&down_matrix(&result)?))
    }

    /// `||logm(P^-1/2 Q P^-1/2)||_F` (= `||log_P(Q)||_P`): the matrix log at
    /// the compute tier, its sum of squares exact, the root at the compute
    /// tier, one rounding. (Before 0.6.4 it rounded log_P(Q), P⁻¹ and the
    /// trace to storage before the root.)
    fn distance(
        &self,
        p: &FixedVector,
        q: &FixedVector,
    ) -> Result<FixedPoint, OverflowDetected> {
        let (_, inv_sqrt_p) = Self::sqrt_and_inv_sqrt(&vec_to_sym(p, self.n))?;
        let q = ComputeMatrix::from_fixed_matrix(&vec_to_sym(q, self.n));
        let log_inner = matrix_log_compute(&inv_sqrt_p.mat_mul(&q).mat_mul(&inv_sqrt_p))?;
        down(sqrt_c(frobenius_sq_c(&log_inner)?))
    }

    /// `E V Eᵀ` with `E = (Q P⁻¹)^1/2`: inverse, product, square root and
    /// transport at the compute tier, one rounding per entry.
    fn parallel_transport(
        &self,
        base: &FixedVector,
        target: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let p = vec_to_sym(base, self.n);
        cholesky_decompose(&p)?;
        let q = ComputeMatrix::from_fixed_matrix(&vec_to_sym(target, self.n));
        let v = ComputeMatrix::from_fixed_matrix(&vec_to_sym(tangent, self.n));
        let e = matrix_sqrt_compute(&q.mat_mul(&Self::inverse_compute(&p)?))?;
        let result = e.mat_mul(&v).mat_mul(&e.transpose());
        Ok(sym_to_vec(&down_matrix(&result)?))
    }
}

// ============================================================================
// L3C: Grassmannian Gr(k, n), k-dimensional subspaces of R^n
// ============================================================================

/// The Grassmann manifold Gr(k, n): k-dimensional subspaces of R^n.
///
/// Points are represented as n×k matrices Q with orthonormal columns (QᵀQ = I_k).
/// Two points Q₁ and Q₂ represent the same subspace if Q₁ = Q₂ R for some
/// orthogonal R ∈ O(k). The manifold operations work modulo this equivalence.
///
/// **Points stored as:** flattened n*k FixedVector (column-major).
///
/// **Geodesics via SVD:**
///   exp_Q(Δ) where Δ is tangent (QᵀΔ = 0):
///     thin SVD: Δ = U Σ Vᵀ, then exp_Q(Δ) = Q V cos(Σ) + U sin(Σ)
///   log_Q(Q') = U Θ Vᵀ where Θ = diag(arctan(σ_i))
///     from thin SVD of (I - QQᵀ)Q' (QᵀQ')⁻¹
///
/// **FASC-UGOD integration:** the SVD from L1B supplies only the right
/// singular vectors (of `Q1ᵀQ2` for log, distance and transport, of the
/// tangent for exp), which a compute-tier Jacobi refinement then corrects;
/// angles, directions and products stay at the compute tier with one
/// rounding per output entry. The SVD's input is still rounded to storage.
pub struct Grassmannian {
    pub k: usize, // subspace dimension
    pub n: usize, // ambient dimension
}

impl Grassmannian {
    /// Pack an n×k matrix into an n*k FixedVector (column-major).
    fn mat_to_vec(m: &FixedMatrix) -> FixedVector {
        let len = m.rows() * m.cols();
        let mut v = FixedVector::new(len);
        let mut idx = 0;
        for c in 0..m.cols() {
            for r in 0..m.rows() {
                v[idx] = m.get(r, c);
                idx += 1;
            }
        }
        v
    }

    /// Unpack an n*k FixedVector into an n×k matrix (column-major).
    fn vec_to_mat(v: &FixedVector, n: usize, k: usize) -> FixedMatrix {
        let mut m = FixedMatrix::new(n, k);
        let mut idx = 0;
        for c in 0..k {
            for r in 0..n {
                m.set(r, c, v[idx]);
                idx += 1;
            }
        }
        m
    }

    /// Project matrix onto the tangent space at Q: Δ - Q(QᵀΔ).
    /// Compute-tier chain: single downscale at end.
    #[allow(dead_code)]
    fn project_tangent(q: &FixedMatrix, delta: &FixedMatrix) -> FixedMatrix {
        let q_c = ComputeMatrix::from_fixed_matrix(q);
        let delta_c = ComputeMatrix::from_fixed_matrix(delta);
        let qt_delta_c = q_c.transpose().mat_mul(&delta_c);
        delta_c.sub(&q_c.mat_mul(&qt_delta_c)).to_fixed_matrix()
    }
}

/// The logarithm of Q2 at Q1 in factored form, at the compute tier:
/// `log = U diag(theta) Aᵀ`.
struct GrassmannLog {
    /// n x k, column i = `P_i / |P_i|` (zero where `P_i` is zero).
    u: ComputeMatrix,
    /// Principal angles `theta_i = atan2(|P_i|, |C_i|)`.
    theta: Vec<C>,
    /// k x k, column i = `C_i / |C_i|`.
    a: ComputeMatrix,
}

impl Grassmannian {
    /// Column norms of a compute matrix: exact sums of squares, root at the
    /// compute tier.
    fn column_norms(m: &ComputeMatrix) -> Result<Vec<C>, OverflowDetected> {
        (0..m.cols()).map(|c| { let col = m.col_vec(c); Ok(sqrt_c(dot_c(&col, &col)?)) }).collect()
    }

    /// `log_Q1(Q2)` at the compute tier from the CS decomposition. With
    /// `Q1ᵀ Q2 = A Σ Bᵀ` (SVD), the columns of `C = Q1ᵀ Q2 B` and of
    /// `P = (I - Q1 Q1ᵀ) Q2 B` have norms `cos theta_i` and `sin theta_i`, so
    /// `theta_i = atan2(|P_i|, |C_i|)` (well conditioned for close and for
    /// orthogonal subspaces) and `log = U diag(theta) Aᵀ` with `U_i = P_i/|P_i|`,
    /// `A_i = C_i/|C_i|`: the standard `U atan(S) Vᵀ` of
    /// `(I - Q1 Q1ᵀ) Q2 (Q1ᵀ Q2)^-1`. Only B comes from the SVD, whose input
    /// `Q1ᵀ Q2` and output are still at storage precision; the angles depend
    /// on B's accuracy to second order. Before 0.6.4 the angles paired the
    /// i-th largest sine with the i-th largest cosine (wrong for k >= 2 with
    /// distinct angles), the log used `B` where `A` belongs (sign wrong for
    /// k = 1 when `Q1ᵀ Q2 < 0`), and the distance took acos of the storage
    /// singular values.
    fn log_parts(&self, q1: &FixedMatrix, q2: &FixedMatrix) -> Result<GrassmannLog, OverflowDetected> {
        let q1_c = ComputeMatrix::from_fixed_matrix(q1);
        let q2_c = ComputeMatrix::from_fixed_matrix(q2);
        let cross = q1_c.transpose().mat_mul(&q2_c); // k×k at compute tier
        let svd = svd_decompose(&down_matrix(&cross)?)?; // storage input to the SVD
        let b = Self::jacobi_refine(&cross.transpose().mat_mul(&cross),
            ComputeMatrix::from_fixed_matrix(&svd.vt.transpose()))?;
        let y = q2_c.mat_mul(&b);
        let c = q1_c.transpose().mat_mul(&y);
        let perp = y.sub(&q1_c.mat_mul(&c));
        let (sin_norms, cos_norms) = (Self::column_norms(&perp)?, Self::column_norms(&c)?);
        let k = c.cols();
        let one = make_compute_int(1);
        let mut u = ComputeMatrix::new(perp.rows(), k);
        let mut a = ComputeMatrix::new(k, k);
        let mut theta = Vec::with_capacity(k);
        for i in 0..k {
            theta.push(atan2_c(sin_norms[i], cos_norms[i]));
            if !compute_is_zero(&sin_norms[i]) {
                for r in 0..perp.rows() { u.set(r, i, mul_div_c(perp.get(r, i), one, sin_norms[i])?); }
            }
            if compute_is_zero(&cos_norms[i]) {
                // orthogonal direction: A_i from the SVD's left vectors
                for r in 0..k { a.set(r, i, upscale_to_compute(svd.u.get(r, i).raw())); }
            } else {
                for r in 0..k { a.set(r, i, mul_div_c(c.get(r, i), one, cos_norms[i])?); }
            }
        }
        Ok(GrassmannLog { u, theta, a })
    }

    /// Rotate the columns of `b` (an orthogonal k×k start, e.g. an SVD's V)
    /// until `bᵀ N b` is diagonal, for a symmetric k×k compute matrix `N`:
    /// cyclic Jacobi at the compute tier, each rotation from `bᵀ N b` formed
    /// afresh. The SVD's vectors come from a storage-rounded input and are
    /// only as accurate as its convergence bound; for nearly equal singular
    /// values (close subspaces) that mixed the principal directions (7 units
    /// in the distance at Q8.24 before this refinement). Rotations are
    /// `t = g / (d + sign(d) sqrt(d^2 + g^2))`, `c = 1 / sqrt(1 + t^2)`,
    /// `s = t c` with `g = G_ij`, `d = (G_jj - G_ii) / 2`: no overflow.
    fn jacobi_refine(nm: &ComputeMatrix, mut b: ComputeMatrix) -> Result<ComputeMatrix, OverflowDetected> {
        let k = b.cols();
        let one = make_compute_int(1);
        for _sweep in 0..6 {
            let mut rotated = false;
            for i in 0..k {
                for j in (i + 1)..k {
                    let bi = b.col_vec(i);
                    let bj = b.col_vec(j);
                    let (ni, nj) = (nm.mul_vector_compute(&bi), nm.mul_vector_compute(&bj));
                    let gij = dot_c(&bi, &nj)?;
                    if compute_is_zero(&gij) { continue; }
                    rotated = true;
                    let d = compute_halve(compute_subtract(dot_c(&bj, &nj)?, dot_c(&bi, &ni)?));
                    let root = sqrt_det_c(d, d, compute_negate(gij), gij)?;
                    let den = if compute_is_negative(&d) { compute_subtract(d, root) } else { c_add(d, root)? };
                    let t = mul_div_c(gij, one, den)?;
                    let c = mul_div_c(one, one, sqrt_c(c_add(one, compute_product(t, t)?)?))?;
                    let s = compute_product(t, c)?;
                    for r in 0..b.rows() {
                        let (x, y) = (bi[r], bj[r]);
                        b.set(r, i, compute_subtract(compute_product(c, x)?, compute_product(s, y)?));
                        b.set(r, j, c_add(compute_product(s, x)?, compute_product(c, y)?)?);
                    }
                }
            }
            if !rotated { break; }
        }
        Ok(b)
    }

    /// A k×k diagonal compute matrix.
    fn diag(d: &[C]) -> ComputeMatrix {
        let zero = make_compute_int(0);
        ComputeMatrix::from_fn(d.len(), d.len(), |r, c| if r == c { d[r] } else { zero })
    }
}

impl Manifold for Grassmannian {
    fn dimension(&self) -> usize {
        // Intrinsic dimension of Gr(k,n) = k*(n-k)
        self.k * (self.n - self.k)
    }

    fn inner_product(
        &self,
        _base: &FixedVector,
        u: &FixedVector,
        v: &FixedVector,
    ) -> FixedPoint {
        // Canonical metric: <U, V> = tr(UᵀV): compute-tier chain, single downscale at trace
        let u_mat = Self::vec_to_mat(u, self.n, self.k);
        let v_mat = Self::vec_to_mat(v, self.n, self.k);
        let u_c = ComputeMatrix::from_fixed_matrix(&u_mat);
        let v_c = ComputeMatrix::from_fixed_matrix(&v_mat);
        u_c.transpose().mat_mul(&v_c).trace_compute()
    }

    /// ||V||_F: exact sum of squares and root at the compute tier, one rounding.
    fn norm(&self, _base: &FixedVector, v: &FixedVector) -> FixedPoint {
        euclidean_norm(v)
    }

    /// `Q V cos(Σ) Vᵀ + Δ V sinc(Σ) Vᵀ` for the thin SVD `Δ = U Σ Vᵀ`, with
    /// `Δ V` (= `U Σ`) and `σ_i = |(Δ V)_i|` formed at the compute tier from
    /// the exact tangent, sin and cos at the compute tier, and one rounding
    /// per entry. V is the SVD's (storage precision); U and Σ are not used.
    fn exp_map(
        &self,
        base: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let q = Self::vec_to_mat(base, self.n, self.k);
        let delta = Self::vec_to_mat(tangent, self.n, self.k);
        let svd = svd_decompose(&delta)?;
        let delta_c = ComputeMatrix::from_fixed_matrix(&delta);
        let v = Self::jacobi_refine(&delta_c.transpose().mat_mul(&delta_c),
            ComputeMatrix::from_fixed_matrix(&svd.vt.transpose()))?;
        let dv = delta_c.mat_mul(&v); // U Σ
        let sigma = Self::column_norms(&dv)?;
        let one = make_compute_int(1);
        let (mut cos_d, mut sinc_d) = (Vec::with_capacity(self.k), Vec::with_capacity(self.k));
        for s in &sigma {
            let (sin_s, cos_s) = sincos_at_compute_tier(*s);
            cos_d.push(cos_s);
            sinc_d.push(if compute_is_zero(s) { one } else { mul_div_c(sin_s, one, *s)? });
        }
        let vt = v.transpose();
        let q_c = ComputeMatrix::from_fixed_matrix(&q);
        let term1 = q_c.mat_mul(&v).mat_mul(&Self::diag(&cos_d)).mat_mul(&vt);
        let term2 = dv.mat_mul(&Self::diag(&sinc_d)).mat_mul(&vt);
        Ok(Self::mat_to_vec(&down_matrix(&term1.add(&term2))?))
    }

    /// `U diag(theta) Aᵀ` from [`Grassmannian::log_parts`], one rounding per entry.
    fn log_map(
        &self,
        base: &FixedVector,
        target: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let q1 = Self::vec_to_mat(base, self.n, self.k);
        let q2 = Self::vec_to_mat(target, self.n, self.k);
        let log = self.log_parts(&q1, &q2)?;
        let result = log.u.mat_mul(&Self::diag(&log.theta)).mat_mul(&log.a.transpose());
        Ok(Self::mat_to_vec(&down_matrix(&result)?))
    }

    /// `sqrt(sum theta_i^2)` of the principal angles of
    /// [`Grassmannian::log_parts`]: sum exact, root at the compute tier, one rounding.
    fn distance(
        &self,
        p: &FixedVector,
        q: &FixedVector,
    ) -> Result<FixedPoint, OverflowDetected> {
        let q1 = Self::vec_to_mat(p, self.n, self.k);
        let q2 = Self::vec_to_mat(q, self.n, self.k);
        let theta = self.log_parts(&q1, &q2)?.theta;
        down(sqrt_c(dot_c(&theta, &theta)?))
    }

    /// Transport along the geodesic with `log_Q1(Q2) = U Θ Aᵀ`:
    /// `PT(Δ) = Δ - Q1 A sin(Θ) Uᵀ Δ + U (cos(Θ) - I) Uᵀ Δ`, the factors from
    /// [`Grassmannian::log_parts`] and the products at the compute tier, one
    /// rounding per entry. (Before 0.6.4 it took a second SVD of the rounded
    /// log map, whose angles were paired wrongly for k >= 2.)
    fn parallel_transport(
        &self,
        base: &FixedVector,
        target: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let q1 = Self::vec_to_mat(base, self.n, self.k);
        let q2 = Self::vec_to_mat(target, self.n, self.k);
        let log = self.log_parts(&q1, &q2)?;
        let one = make_compute_int(1);
        let (mut sin_d, mut cos_m1) = (Vec::with_capacity(self.k), Vec::with_capacity(self.k));
        for t in &log.theta {
            let (s, c) = sincos_at_compute_tier(*t);
            sin_d.push(s);
            cos_m1.push(compute_subtract(c, one));
        }
        let q1_c = ComputeMatrix::from_fixed_matrix(&q1);
        let delta_c = ComputeMatrix::from_fixed_matrix(&Self::vec_to_mat(tangent, self.n, self.k));
        let ut_delta = log.u.transpose().mat_mul(&delta_c);
        let along = q1_c.mat_mul(&log.a).mat_mul(&Self::diag(&sin_d)).mat_mul(&ut_delta);
        let within = log.u.mat_mul(&Self::diag(&cos_m1)).mat_mul(&ut_delta);
        let result = delta_c.sub(&along).add(&within);
        Ok(Self::mat_to_vec(&down_matrix(&result)?))
    }
}

// ============================================================================
// L3C: Stiefel manifold St(k, n), orthonormal k-frames in R^n
// ============================================================================

/// The Stiefel manifold St(k, n): orthonormal k-frames in R^n.
///
/// Points are n×k matrices Q with QᵀQ = I_k (orthonormal columns).
/// Unlike Grassmannian, two points Q₁ ≠ Q₂ even if they span the same subspace.
///
/// **Points stored as:** flattened n*k FixedVector (column-major).
///
/// **Geodesics via QR retraction:**
///   exp_Q(Δ) ≈ qr(Q + Δ).Q: the Q factor of QR decomposition.
///   This is a first-order retraction, not the exact Riemannian exponential,
///   but preserves the orthonormality constraint exactly (QR produces orthonormal Q).
///
/// **FASC-UGOD integration:** QR decomposition uses Householder reflections with
/// compute_tier_dot_raw for all inner products. The retraction preserves
/// orthonormality to machine precision (structural guarantee, not iterative).
pub struct StiefelManifold {
    pub k: usize, // frame dimension (number of columns)
    pub n: usize, // ambient dimension (number of rows)
}

/// Pack an n×k matrix into an n*k FixedVector (column-major).
fn stiefel_mat_to_vec(m: &FixedMatrix) -> FixedVector {
    let len = m.rows() * m.cols();
    let mut v = FixedVector::new(len);
    let mut idx = 0;
    for c in 0..m.cols() {
        for r in 0..m.rows() {
            v[idx] = m.get(r, c);
            idx += 1;
        }
    }
    v
}

/// Unpack an n*k FixedVector into an n×k matrix (column-major).
fn stiefel_vec_to_mat(v: &FixedVector, n: usize, k: usize) -> FixedMatrix {
    let mut m = FixedMatrix::new(n, k);
    let mut idx = 0;
    for c in 0..k {
        for r in 0..n {
            m.set(r, c, v[idx]);
            idx += 1;
        }
    }
    m
}

impl StiefelManifold {
    /// Project a matrix onto the tangent space at Q.
    ///
    /// Tangent vectors Δ satisfy: QᵀΔ + ΔᵀQ = 0 (skew-symmetric QᵀΔ).
    /// Projection: Δ_tangent = Δ - Q · sym(QᵀΔ) where sym(A) = (A+Aᵀ)/2.
    ///
    /// Compute-tier chain: single downscale at end.
    fn project_tangent(q: &FixedMatrix, delta: &FixedMatrix) -> FixedMatrix {
        Self::project_tangent_compute(&ComputeMatrix::from_fixed_matrix(q), &ComputeMatrix::from_fixed_matrix(delta))
            .to_fixed_matrix()
    }

    /// [`StiefelManifold::project_tangent`] on compute matrices, not narrowed.
    fn project_tangent_compute(q_c: &ComputeMatrix, delta_c: &ComputeMatrix) -> ComputeMatrix {
        let qt_delta_c = q_c.transpose().mat_mul(delta_c); // k×k at compute tier
        // sym(QᵀΔ) = (QᵀΔ + ΔᵀQ) / 2 at compute tier
        let sym_c = qt_delta_c.add(&qt_delta_c.transpose()).halve();
        // Δ - Q · sym at compute tier
        delta_c.sub(&q_c.mat_mul(&sym_c))
    }

    /// First-order log `Δ - Q sym(QᵀΔ)`, `Δ = Q' - Q`, at the compute tier.
    fn log_compute(&self, base: &FixedVector, target: &FixedVector) -> ComputeMatrix {
        let q_c = ComputeMatrix::from_fixed_matrix(&stiefel_vec_to_mat(base, self.n, self.k));
        let target_c = ComputeMatrix::from_fixed_matrix(&stiefel_vec_to_mat(target, self.n, self.k));
        Self::project_tangent_compute(&q_c, &target_c.sub(&q_c))
    }
}

impl Manifold for StiefelManifold {
    fn dimension(&self) -> usize {
        // Intrinsic dimension of St(k,n) = nk - k(k+1)/2
        self.n * self.k - self.k * (self.k + 1) / 2
    }

    fn inner_product(
        &self,
        _base: &FixedVector,
        u: &FixedVector,
        v: &FixedVector,
    ) -> FixedPoint {
        // Canonical metric: <U, V> = tr(UᵀV): compute-tier chain, single downscale at trace
        let u_mat = stiefel_vec_to_mat(u, self.n, self.k);
        let v_mat = stiefel_vec_to_mat(v, self.n, self.k);
        let u_c = ComputeMatrix::from_fixed_matrix(&u_mat);
        let v_c = ComputeMatrix::from_fixed_matrix(&v_mat);
        u_c.transpose().mat_mul(&v_c).trace_compute()
    }

    /// ||V||_F: exact sum of squares and root at the compute tier, one rounding.
    fn norm(&self, _base: &FixedVector, v: &FixedVector) -> FixedPoint {
        euclidean_norm(v)
    }

    fn exp_map(
        &self,
        base: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let q = stiefel_vec_to_mat(base, self.n, self.k);
        let delta = stiefel_vec_to_mat(tangent, self.n, self.k);

        // QR retraction: Q_new = qr(Q + Δ).Q
        // This is a first-order retraction that preserves QᵀQ = I exactly.
        let q_plus_delta = &q + &delta;
        let qr = qr_decompose(&q_plus_delta)?;

        // Extract the first k columns of Q from QR
        // qr.q is n×n; we need the first k columns (thin Q)
        let q_new = FixedMatrix::from_fn(self.n, self.k, |r, c| {
            // Ensure positive diagonal in R (sign convention for unique QR)
            let sign = if qr.r.get(c, c).is_negative() {
                FixedPoint::from_int(-1)
            } else {
                FixedPoint::one()
            };
            qr.q.get(r, c) * sign
        });

        Ok(stiefel_mat_to_vec(&q_new))
    }

    /// First-order log `Δ - Q sym(QᵀΔ)` with `Δ = Q' - Q`, at the compute
    /// tier, one rounding per entry.
    fn log_map(
        &self,
        base: &FixedVector,
        target: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        Ok(stiefel_mat_to_vec(&down_matrix(&self.log_compute(base, target))?))
    }

    /// `||log_Q(Q')||_F`: the log at the compute tier, its sum of squares
    /// exact, the root at the compute tier, one rounding. (Before 0.6.4 this
    /// took the square root of the Frobenius norm, returning `||Δ||^(1/2)`.)
    fn distance(
        &self,
        p: &FixedVector,
        q: &FixedVector,
    ) -> Result<FixedPoint, OverflowDetected> {
        down(sqrt_c(frobenius_sq_c(&self.log_compute(p, q))?))
    }

    fn parallel_transport(
        &self,
        _base: &FixedVector,
        target: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        // Transport via projection: project tangent onto tangent space at target
        let q_target = stiefel_vec_to_mat(target, self.n, self.k);
        let delta = stiefel_vec_to_mat(tangent, self.n, self.k);
        let transported = Self::project_tangent(&q_target, &delta);
        Ok(stiefel_mat_to_vec(&transported))
    }
}

// ============================================================================
// Product manifold M₁ × M₂
// ============================================================================

/// Product manifold M₁ × M₂: the Cartesian product of two manifolds.
///
/// Points are concatenated coordinate vectors: `[coords_m1 | coords_m2]`.
/// Metric is block-diagonal: <(u₁,u₂), (v₁,v₂)> = <u₁,v₁>₁ + <u₂,v₂>₂.
/// exp/log/transport operate on each component independently.
///
/// **FASC-UGOD integration:** All operations delegate to the component manifolds.
/// Tier handling is inherited; each component uses its own compute-tier
/// operations internally. The components return storage values, so the
/// combined distance, norm and inner product (combined at the compute tier,
/// one rounding) can add up to one unit to the components' own errors.
pub struct ProductManifold {
    m1: Box<dyn Manifold>,
    m2: Box<dyn Manifold>,
    /// Embedding dimension (FixedVector length) for points on M₁.
    dim1_embed: usize,
    /// Embedding dimension (FixedVector length) for points on M₂.
    dim2_embed: usize,
}

impl ProductManifold {
    /// Create a product manifold M₁ × M₂.
    ///
    /// `dim1_embed` and `dim2_embed` are the FixedVector lengths for points
    /// on each component manifold (may differ from intrinsic dimension).
    pub fn new(
        m1: Box<dyn Manifold>,
        dim1_embed: usize,
        m2: Box<dyn Manifold>,
        dim2_embed: usize,
    ) -> Self {
        Self { m1, m2, dim1_embed, dim2_embed }
    }

    /// Split a concatenated vector into (part1, part2).
    fn split(&self, v: &FixedVector) -> (FixedVector, FixedVector) {
        let mut v1 = FixedVector::new(self.dim1_embed);
        let mut v2 = FixedVector::new(self.dim2_embed);
        for i in 0..self.dim1_embed { v1[i] = v[i]; }
        for i in 0..self.dim2_embed { v2[i] = v[self.dim1_embed + i]; }
        (v1, v2)
    }

    /// Join two vectors into a concatenated vector.
    fn join(v1: &FixedVector, v2: &FixedVector) -> FixedVector {
        let mut v = FixedVector::new(v1.len() + v2.len());
        for i in 0..v1.len() { v[i] = v1[i]; }
        for i in 0..v2.len() { v[v1.len() + i] = v2[i]; }
        v
    }
}

impl Manifold for ProductManifold {
    fn dimension(&self) -> usize {
        self.m1.dimension() + self.m2.dimension()
    }

    fn inner_product(
        &self,
        base: &FixedVector,
        u: &FixedVector,
        v: &FixedVector,
    ) -> FixedPoint {
        let (b1, b2) = self.split(base);
        let (u1, u2) = self.split(u);
        let (v1, v2) = self.split(v);
        // the two parts added at the compute tier (exact), one rounding back;
        // a sum beyond storage panics instead of wrapping
        let first = upscale_to_compute(self.m1.inner_product(&b1, &u1, &v1).raw());
        let second = upscale_to_compute(self.m2.inner_product(&b2, &u2, &v2).raw());
        down(c_add(first, second).expect("ProductManifold::inner_product exceeds the compute tier"))
            .expect("ProductManifold::inner_product exceeds the storage range")
    }

    /// `sqrt(||v1||^2 + ||v2||^2)` of the component norms: squares exact,
    /// sum and root at the compute tier, one rounding.
    fn norm(&self, base: &FixedVector, v: &FixedVector) -> FixedPoint {
        let (b1, b2) = self.split(base);
        let (v1, v2) = self.split(v);
        let parts = [self.m1.norm(&b1, &v1).raw(), self.m2.norm(&b2, &v2).raw()];
        FixedPoint::from_raw(compute_tier_sqrt_dot(&parts, &parts).expect("ProductManifold::norm exceeds the storage range"))
    }

    fn exp_map(
        &self,
        base: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let (b1, b2) = self.split(base);
        let (t1, t2) = self.split(tangent);
        let r1 = self.m1.exp_map(&b1, &t1)?;
        let r2 = self.m2.exp_map(&b2, &t2)?;
        Ok(Self::join(&r1, &r2))
    }

    fn log_map(
        &self,
        base: &FixedVector,
        target: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let (b1, b2) = self.split(base);
        let (t1, t2) = self.split(target);
        let l1 = self.m1.log_map(&b1, &t1)?;
        let l2 = self.m2.log_map(&b2, &t2)?;
        Ok(Self::join(&l1, &l2))
    }

    fn distance(
        &self,
        p: &FixedVector,
        q: &FixedVector,
    ) -> Result<FixedPoint, OverflowDetected> {
        let (p1, p2) = self.split(p);
        let (q1, q2) = self.split(q);
        let d1 = self.m1.distance(&p1, &q1)?;
        let d2 = self.m2.distance(&p2, &q2)?;
        // Product distance sqrt(d₁² + d₂²): squares exact, sum and root at
        // the compute tier, one rounding (was two rounded squares, a rounded
        // sum and a storage root)
        let parts = [d1.raw(), d2.raw()];
        Ok(FixedPoint::from_raw(compute_tier_sqrt_dot(&parts, &parts)?))
    }

    fn parallel_transport(
        &self,
        base: &FixedVector,
        target: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let (b1, b2) = self.split(base);
        let (t1, t2) = self.split(target);
        let (v1, v2) = self.split(tangent);
        let pt1 = self.m1.parallel_transport(&b1, &t1, &v1)?;
        let pt2 = self.m2.parallel_transport(&b2, &t2, &v2)?;
        Ok(Self::join(&pt1, &pt2))
    }
}
