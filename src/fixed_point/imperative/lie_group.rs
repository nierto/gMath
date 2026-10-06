//! L4A: Lie groups and Lie algebras with fixed-point arithmetic.
//!
//! - `SO3`: 3D rotations via closed-form Rodrigues (O(1) trig, not matrix_exp)
//! - `SE3`: 3D rigid motions (rotation + translation) via closed-form V-matrix
//! - `SOn`: General n×n rotations via matrix_exp/matrix_log fallback
//!
//! SO(3) Rodrigues is the foundational primitive for geometric key derivation.
//!
//! Multi-step operations (exp and log, the Manifold chains built on them,
//! adjoints, brackets) carry their state at the compute tier (2 x FRAC_BITS)
//! and round each output to storage once.

use super::FixedVector;
use super::FixedMatrix;
use super::manifold::Manifold;
use super::matrix_functions::{matrix_exp, matrix_log, matrix_exp_compute, matrix_log_compute};
use super::compute_matrix::{ComputeMatrix, compute_lu_decompose};
use super::kernels;
use super::kernels::Orient;
use super::FixedPoint;
use super::Scalar;
use super::matrix_functions::MatrixFunctions;
use crate::fixed_point::fixed_num::{Compute, Fixed, FixedNum};
use crate::fixed_point::width::{Storage, WideInt};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// Compute-tier helpers
// ============================================================================

/// A storage value at the compute tier (exact).
#[inline]
fn up<S: Storage, const F: u32>(x: Fixed<S, F>) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{ Fixed::<S, F>::upscale_to_compute(x.raw()) }

/// One rounding to storage (nearest, ties toward +infinity), checked.
#[inline]
fn down<S: Storage, const F: u32>(c: Compute<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    Ok(Fixed::from_raw(Fixed::<S, F>::downscale_to_storage(c)?))
}

fn down_vector<S: Storage, const F: u32>(v: &[Compute<Fixed<S, F>>]) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let mut out = FixedVector::<Fixed<S, F>>::new(v.len());
    for (i, c) in v.iter().enumerate() { out[i] = down(*c)?; }
    Ok(out)
}

fn down_matrix<S: Storage, const F: u32>(m: &ComputeMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let (rows, cols) = (m.rows(), m.cols());
    let mut out = FixedMatrix::<Fixed<S, F>>::new(rows, cols);
    for r in 0..rows {
        for c in 0..cols { out.set(r, c, down(m.get(r, c))?); }
    }
    Ok(out)
}

/// |v| of a compute-tier vector: squares, sum and root at the compute tier,
/// one rounding.
fn norm_to_storage<S: Storage, const F: u32>(v: &[Compute<Fixed<S, F>>]) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let mut sum = Fixed::<S, F>::make_compute_int(0);
    for c in v { sum = Fixed::<S, F>::compute_add(sum, Fixed::<S, F>::compute_multiply(*c, *c)); }
    down(Fixed::<S, F>::sqrt_at_compute_tier(sum))
}

/// The profile's atan2 at the compute tier (arguments and angle at 2F).
fn atan2_compute<S: Storage, const F: u32>(y: Compute<Fixed<S, F>>, x: Compute<Fixed<S, F>>) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    Fixed::<S, F>::atan2_at_compute_tier(y, x)
}

/// sum_k c_k t^k with c_k = num_k / den_k, at the compute tier.
fn series<S: Storage, const F: u32>(t: Compute<Fixed<S, F>>, coeffs: &[(i64, i64)]) -> Result<Compute<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let mut power = Fixed::<S, F>::make_compute_int(1);
    let mut sum = Fixed::<S, F>::make_compute_int(0);
    for (k, (num, den)) in coeffs.iter().enumerate() {
        if k > 0 { power = Fixed::<S, F>::compute_multiply(power, t); }
        sum = Fixed::<S, F>::compute_add(sum, Fixed::<S, F>::compute_mul_div_int(power, *num, *den)?);
    }
    Ok(sum)
}

/// Exact `sum_k (sum_l a_kl b_kl) * s_k` at 3F (the inner sums exact at 2F,
/// `s_k` storage), rounded to storage once. With `negate` the sum is negated
/// before the rounding (round(-x), not -round(x)).
fn triple_to_storage<S: Storage, const F: u32>(terms: &[(Compute<Fixed<S, F>>, S)], negate: bool) -> Result<Fixed<S, F>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let mut sum = <Orient<Fixed<S, F>> as WideInt>::zero();
    for (c, s) in terms { sum = sum.add_exact(S::widen_product(*c, S::widen(*s)))?; }
    Ok(Fixed::from_raw(kernels::narrow_triple_nearest::<Fixed<S, F>>(if negate { -sum } else { sum })?))
}

/// a x b of storage 3-vectors at the compute tier: each component the
/// difference of two exact products (exact).
fn cross_exact<S: Storage, const F: u32>(a: &FixedVector<Fixed<S, F>>, b: &FixedVector<Fixed<S, F>>) -> [Compute<Fixed<S, F>>; 3]
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let p = |i: usize, j: usize| S::mul_wide(a[i].raw(), b[j].raw());
    [
        Fixed::<S, F>::compute_subtract(p(1, 2), p(2, 1)),
        Fixed::<S, F>::compute_subtract(p(2, 0), p(0, 2)),
        Fixed::<S, F>::compute_subtract(p(0, 1), p(1, 0)),
    ]
}

/// [A, B] = AB - BA of storage matrices, each entry exact at the compute tier.
fn commutator_exact<S: Storage, const F: u32>(a: &FixedMatrix<Fixed<S, F>>, b: &FixedMatrix<Fixed<S, F>>) -> ComputeMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let n = a.rows();
    ComputeMatrix::<Fixed<S, F>>::from_fn(n, n, |i, j| {
        let (mut ab, mut ba) = (Fixed::<S, F>::make_compute_int(0), Fixed::<S, F>::make_compute_int(0));
        for k in 0..n {
            ab = Fixed::<S, F>::compute_add(ab, S::mul_wide(a.get(i, k).raw(), b.get(k, j).raw()));
            ba = Fixed::<S, F>::compute_add(ba, S::mul_wide(b.get(i, k).raw(), a.get(k, j).raw()));
        }
        Fixed::<S, F>::compute_subtract(ab, ba)
    })
}

/// g x h^T of storage matrices with every entry one rounding of its exact
/// value: sum_kl g_ik x_kl h_jl, exact triple products at 3F.
fn conjugate_exact<S: Storage, const F: u32>(g: &FixedMatrix<Fixed<S, F>>, x: &FixedMatrix<Fixed<S, F>>, h: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let n = g.rows();
    let mut out = FixedMatrix::<Fixed<S, F>>::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut terms = Vec::with_capacity(n * n);
            for k in 0..n {
                for l in 0..n {
                    terms.push((S::mul_wide(g.get(i, k).raw(), x.get(k, l).raw()), h.get(j, l).raw()));
                }
            }
            out.set(i, j, triple_to_storage(&terms, false)?);
        }
    }
    Ok(out)
}

/// g^-1 at the compute tier (compute-tier LU, no storage rounding).
fn inverse_compute<S: Storage, const F: u32>(g: &ComputeMatrix<Fixed<S, F>>) -> Result<ComputeMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    compute_lu_decompose(g)?.inverse()
}

/// g xi g^-1 at the compute tier: the conjugation used by the GL(n) and SL(n)
/// adjoints.
fn conjugate_by_inverse<S: Storage, const F: u32>(g: &FixedMatrix<Fixed<S, F>>, xi_hat: &FixedMatrix<Fixed<S, F>>) -> Result<ComputeMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let g_c = ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(g);
    let g_inv = inverse_compute(&g_c)?;
    Ok(g_c.mat_mul(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(xi_hat)).mat_mul(&g_inv))
}

/// The skew part (A - A^T) / 2 at the compute tier.
fn skew_half<S: Storage, const F: u32>(a: &ComputeMatrix<Fixed<S, F>>) -> ComputeMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    a.sub(&a.transpose()).halve()
}

// ============================================================================
// LieGroup trait
// ============================================================================

/// A Lie group with fixed-point arithmetic.
///
/// Group elements are `FixedMatrix<Fixed<S, F>>`. Algebra elements are `FixedVector<Fixed<S, F>>`
/// (coordinate parameterization) with `hat`/`vee` for matrix form.
pub trait LieGroup<T: Scalar = FixedPoint>: Manifold<T> {
    /// Dimension of the Lie algebra.
    fn algebra_dim(&self) -> usize;
    /// Dimension of the matrix representation (n for n×n).
    fn matrix_dim(&self) -> usize;
    /// Group identity element.
    fn identity_element(&self) -> FixedMatrix<T>;
    /// Group composition: g1 * g2.
    fn compose(&self, g1: &FixedMatrix<T>, g2: &FixedMatrix<T>) -> FixedMatrix<T>;
    /// Group inverse: g⁻¹.
    fn group_inverse(&self, g: &FixedMatrix<T>) -> Result<FixedMatrix<T>, OverflowDetected>;
    /// Exponential map: algebra vector → group element.
    fn lie_exp(&self, xi: &FixedVector<T>) -> Result<FixedMatrix<T>, OverflowDetected>;
    /// Logarithmic map: group element → algebra vector.
    fn lie_log(&self, g: &FixedMatrix<T>) -> Result<FixedVector<T>, OverflowDetected>;
    /// Hat map: algebra vector → algebra matrix.
    fn hat(&self, xi: &FixedVector<T>) -> FixedMatrix<T>;
    /// Vee map: algebra matrix → algebra vector.
    fn vee(&self, xi_hat: &FixedMatrix<T>) -> FixedVector<T>;
    /// Adjoint: Ad_g(xi) in algebra coordinates.
    fn adjoint(&self, g: &FixedMatrix<T>, xi: &FixedVector<T>) -> Result<FixedVector<T>, OverflowDetected>;
    /// Lie bracket: [xi, eta].
    fn bracket(&self, xi: &FixedVector<T>, eta: &FixedVector<T>) -> FixedVector<T>;
    /// Group action on a point.
    fn act(&self, g: &FixedMatrix<T>, point: &FixedVector<T>) -> FixedVector<T>;
}

// ============================================================================
// SO(3) — 3D rotations via Rodrigues
// ============================================================================

/// SO(3): Special orthogonal group of 3D rotations.
///
/// Elements: 3×3 orthogonal matrices with det = +1.
/// Algebra so(3): skew-symmetric 3×3 matrices, parameterized by 3-vectors.
/// Uses closed-form Rodrigues formula: O(1) trig, not matrix_exp.
pub struct SO3;

/// The small-angle threshold of the Rodrigues branches: 10^-3 below 128-bit
/// storage, 10^-5 at 128, 10^-10 at 256 and 10^-20 at 512 bits, each the
/// nearest value at the format (what `from_str` makes of the literal), and
/// at least one storage unit: at 8 fraction bits 0.001 rounds to 0, so
/// theta < 0.001 never held and theta = 0 took the general branch and
/// divided by zero (exp(0) erred, log(I) panicked; before 0.6.4).
fn rodrigues_threshold<S: Storage, const F: u32>() -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let k = match S::BITS { 32 | 64 => 3, 128 => 5, 256 => 10, _ => 20 };
    // 10^20 exceeds i64: 10^10 squared on the storage word (512 bits)
    let d = if k == 20 { let e10 = S::from_i64(10_000_000_000); e10.mul_exact(e10) } else { S::from_i64(10i64.pow(k)) };
    // 2^F / 10^k rounded to nearest; never a tie (10^k has the odd factor 5^k)
    let (q, r) = S::one().shl_bits(F).divmod(d);
    let raw = if r.shl_bits(1) >= d { q + S::one() } else { q };
    Fixed::from_raw(if raw == S::ZERO { S::one() } else { raw })
}

/// Below this angle the SE(3) V and V^-1 coefficients come from their Taylor
/// series: 2^(2 - F/2), where the closed forms ((1 - cos)/theta^2,
/// (theta - sin)/theta^3, (1 - alpha)/theta^2) have cancelled to a relative
/// error of about 2^-2F / theta^2 of their result; above it that error stays
/// below 2^-(F + 4) of the output. The series to theta^6 is exact there to
/// far below a unit. (Before 0.6.4 the Taylor branch below the storage
/// threshold kept only the constants 1/2 and 1/6: a theta^3 / 24 |v|
/// truncation, 746 units at theta = 8.8e-6 on Q64.64 and over 10^6 units on
/// Q128.128; mpmath gate lie_fiber_compute_tier_validation.)
fn se3_series_threshold<S: Storage, const F: u32>() -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let mut t = Fixed::<S, F>::make_compute_int(4);
    for _ in 0..F / 2 { t = Fixed::<S, F>::compute_halve(t); }
    t
}

/// theta^2 = |omega|^2 and theta at the compute tier. For storage omega the
/// squares and the sum are exact, so theta^2 is never 0 for a nonzero omega
/// (before 0.6.4 it was rounded to storage: 0 raw for theta below about
/// 2^-(F/2), a DivisionByZero from exp on realtime).
fn angle_compute<S: Storage, const F: u32>(omega: &[Compute<Fixed<S, F>>]) -> (Compute<Fixed<S, F>>, Compute<Fixed<S, F>>)
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let mut theta_sq = Fixed::<S, F>::make_compute_int(0);
    for c in omega { theta_sq = Fixed::<S, F>::compute_add(theta_sq, Fixed::<S, F>::compute_multiply(*c, *c)); }
    (theta_sq, Fixed::<S, F>::sqrt_at_compute_tier(theta_sq))
}

/// [omega]x at the compute tier.
fn hat_compute<S: Storage, const F: u32>(w: &[Compute<Fixed<S, F>>]) -> ComputeMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let z = Fixed::<S, F>::make_compute_int(0);
    let m = [z, Fixed::<S, F>::compute_negate(w[2]), w[1], w[2], z, Fixed::<S, F>::compute_negate(w[0]), Fixed::<S, F>::compute_negate(w[1]), w[0], z];
    ComputeMatrix::<Fixed<S, F>>::from_fn(3, 3, |r, c| m[3 * r + c])
}

/// Rodrigues exp at the compute tier: R = I + sinc [w]x + half_cosc [w]x^2.
fn rodrigues_exp_compute<S: Storage, const F: u32>(omega: &[Compute<Fixed<S, F>>]) -> Result<ComputeMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let (theta_sq, theta) = angle_compute(omega);
    let (sinc, half_cosc) = if theta < Fixed::<S, F>::upscale_to_compute(rodrigues_threshold::<S, F>().raw()) {
        (
            series(theta_sq, &[(1, 1), (-1, 6), (1, 120), (-1, 5040)])?,
            series(theta_sq, &[(1, 2), (-1, 24), (1, 720), (-1, 40320)])?,
        )
    } else {
        let (sin_c, cos_c) = Fixed::<S, F>::sincos_at_compute_tier(theta);
        (
            Fixed::<S, F>::compute_divide(sin_c, theta)?,
            Fixed::<S, F>::compute_divide(Fixed::<S, F>::compute_subtract(Fixed::<S, F>::make_compute_int(1), cos_c), theta_sq)?,
        )
    };
    let k = hat_compute(omega);
    let k2 = k.mat_mul(&k);
    Ok(ComputeMatrix::<Fixed<S, F>>::identity(3).add(&k.scalar_mul(sinc)).add(&k2.scalar_mul(half_cosc)))
}

/// Rodrigues log at the compute tier: (omega, theta) with
/// theta = atan2(|vee(R - R^T)|, tr R - 1). Up to 90 degrees omega =
/// theta vee(R - R^T) / |vee(R - R^T)|, one quotient from the exact product.
/// Beyond 90 degrees the axis comes from the symmetric part,
/// (R + R^T) / 2 - cos(theta) I = (1 - cos(theta)) u u^T, which stays well
/// conditioned up to pi (the skew part vanishes there, and theta / (2 sin)
/// amplified its rounding: up to 70525 units in the mpmath round trip on
/// Q16.16 before 0.6.4),
/// with the sign of u taken from the skew part.
fn rodrigues_log_compute<S: Storage, const F: u32>(r: &ComputeMatrix<Fixed<S, F>>) -> Result<([Compute<Fixed<S, F>>; 3], Compute<Fixed<S, F>>), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let one = Fixed::<S, F>::make_compute_int(1);
    let zero = Fixed::<S, F>::make_compute_int(0);
    let d = [
        Fixed::<S, F>::compute_subtract(r.get(2, 1), r.get(1, 2)),
        Fixed::<S, F>::compute_subtract(r.get(0, 2), r.get(2, 0)),
        Fixed::<S, F>::compute_subtract(r.get(1, 0), r.get(0, 1)),
    ];
    let tr_minus_one = Fixed::<S, F>::compute_subtract(Fixed::<S, F>::compute_add(Fixed::<S, F>::compute_add(r.get(0, 0), r.get(1, 1)), r.get(2, 2)), one);
    let (_, d_norm) = angle_compute(&d);
    let d_zero = Fixed::<S, F>::compute_is_zero(&d_norm);

    if !Fixed::<S, F>::compute_is_negative(&tr_minus_one) {
        if d_zero {
            return Ok(([zero, zero, zero], zero));
        }
        let theta = atan2_compute(d_norm, tr_minus_one);
        let mut omega = [zero; 3];
        for i in 0..3 {
            omega[i] = kernels::divide_to_compute_nearest::<Fixed<S, F>>(S::widen_product(theta, d[i]), d_norm)?;
        }
        return Ok((omega, theta));
    }

    let theta = if d_zero { atan2_compute(zero, tr_minus_one) } else { atan2_compute(d_norm, tr_minus_one) };
    let cos_theta = Fixed::<S, F>::compute_halve(tr_minus_one);
    let one_minus_cos = Fixed::<S, F>::compute_subtract(one, cos_theta);
    // largest diagonal entry: the largest axis component (|u_i| >= 1/sqrt(3))
    let mut i = 0;
    for k in 1..3 {
        if r.get(k, k) > r.get(i, i) { i = k; }
    }
    let ui_sq = Fixed::<S, F>::compute_divide(Fixed::<S, F>::compute_subtract(r.get(i, i), cos_theta), one_minus_cos)?;
    if Fixed::<S, F>::compute_is_negative(&ui_sq) || Fixed::<S, F>::compute_is_zero(&ui_sq) {
        return Err(OverflowDetected::DomainError);
    }
    let ui = Fixed::<S, F>::sqrt_at_compute_tier(ui_sq);
    let den = Fixed::<S, F>::compute_multiply(Fixed::<S, F>::compute_add(one_minus_cos, one_minus_cos), ui);
    let mut u = [zero; 3];
    for k in 0..3 {
        u[k] = if k == i { ui } else { Fixed::<S, F>::compute_divide(Fixed::<S, F>::compute_add(r.get(i, k), r.get(k, i)), den)? };
    }
    // sin(theta) u = vee(R - R^T) / 2: u points along the skew part
    let mut dot = zero;
    for k in 0..3 { dot = Fixed::<S, F>::compute_add(dot, Fixed::<S, F>::compute_multiply(u[k], d[k])); }
    let sign_flip = Fixed::<S, F>::compute_is_negative(&dot);
    let mut omega = [zero; 3];
    for k in 0..3 {
        let w = Fixed::<S, F>::compute_multiply(theta, u[k]);
        omega[k] = if sign_flip { Fixed::<S, F>::compute_negate(w) } else { w };
    }
    Ok((omega, theta))
}

/// SE(3) exp at the compute tier: [[R, V v], [0, 1]].
fn se3_exp_compute<S: Storage, const F: u32>(xi: &[Compute<Fixed<S, F>>]) -> Result<ComputeMatrix<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let omega = &xi[0..3];
    let r = rodrigues_exp_compute(omega)?;
    let (theta_sq, theta) = angle_compute(omega);
    let (c1, c2) = if theta < se3_series_threshold::<S, F>() {
        (
            series(theta_sq, &[(1, 2), (-1, 24), (1, 720), (-1, 40320)])?,
            series(theta_sq, &[(1, 6), (-1, 120), (1, 5040), (-1, 362880)])?,
        )
    } else {
        let (sin_c, cos_c) = Fixed::<S, F>::sincos_at_compute_tier(theta);
        // (theta - sin) / theta^2 / theta: theta^3 itself can round to 0
        // at the compute tier on narrow splits
        (
            Fixed::<S, F>::compute_divide(Fixed::<S, F>::compute_subtract(Fixed::<S, F>::make_compute_int(1), cos_c), theta_sq)?,
            Fixed::<S, F>::compute_divide(Fixed::<S, F>::compute_divide(Fixed::<S, F>::compute_subtract(theta, sin_c), theta_sq)?, theta)?,
        )
    };
    let k = hat_compute(omega);
    let k2 = k.mat_mul(&k);
    let v_mat = ComputeMatrix::<Fixed<S, F>>::identity(3).add(&k.scalar_mul(c1)).add(&k2.scalar_mul(c2));
    let t = v_mat.mul_vector_compute(&xi[3..6]);
    Ok(ComputeMatrix::<Fixed<S, F>>::from_fn(4, 4, |i, j| {
        if i < 3 && j < 3 { r.get(i, j) }
        else if i < 3 { t[i] }
        else if j == 3 { Fixed::<S, F>::make_compute_int(1) }
        else { Fixed::<S, F>::make_compute_int(0) }
    }))
}

/// SE(3) log at the compute tier: [omega, V^-1 t].
fn se3_log_compute<S: Storage, const F: u32>(g: &ComputeMatrix<Fixed<S, F>>) -> Result<[Compute<Fixed<S, F>>; 6], OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let r = ComputeMatrix::<Fixed<S, F>>::from_fn(3, 3, |i, j| g.get(i, j));
    let t = [g.get(0, 3), g.get(1, 3), g.get(2, 3)];
    let (omega, theta) = rodrigues_log_compute(&r)?;
    let theta_sq = Fixed::<S, F>::compute_multiply(theta, theta);
    let c2 = if theta < se3_series_threshold::<S, F>() {
        series(theta_sq, &[(1, 12), (1, 720), (1, 30240), (1, 1209600)])?
    } else {
        let one = Fixed::<S, F>::make_compute_int(1);
        let (sin_c, cos_c) = Fixed::<S, F>::sincos_at_compute_tier(theta);
        // alpha = theta sin(theta) / (2 (1 - cos(theta)))
        let one_minus_cos = Fixed::<S, F>::compute_subtract(one, cos_c);
        let alpha = Fixed::<S, F>::compute_divide(Fixed::<S, F>::compute_multiply(theta, sin_c), Fixed::<S, F>::compute_add(one_minus_cos, one_minus_cos))?;
        Fixed::<S, F>::compute_divide(Fixed::<S, F>::compute_subtract(one, alpha), theta_sq)?
    };
    let k = hat_compute(&omega);
    let k2 = k.mat_mul(&k);
    let v_inv = ComputeMatrix::<Fixed<S, F>>::identity(3).sub(&k.halve()).add(&k2.scalar_mul(c2));
    let v = v_inv.mul_vector_compute(&t);
    Ok([omega[0], omega[1], omega[2], v[0], v[1], v[2]])
}

/// g^-1 = [[R^T, -R^T t], [0, 1]] at the compute tier.
fn se3_inverse_compute<S: Storage, const F: u32>(g: &ComputeMatrix<Fixed<S, F>>) -> ComputeMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let mut out = ComputeMatrix::<Fixed<S, F>>::identity(4);
    for i in 0..3 {
        let mut acc = Fixed::<S, F>::make_compute_int(0);
        for k in 0..3 {
            out.set(i, k, g.get(k, i));
            acc = Fixed::<S, F>::compute_add(acc, Fixed::<S, F>::compute_multiply(g.get(k, i), g.get(k, 3)));
        }
        out.set(i, 3, Fixed::<S, F>::compute_negate(acc));
    }
    out
}

fn up_all<S: Storage, const F: u32>(v: &FixedVector<Fixed<S, F>>) -> Vec<Compute<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{ (0..v.len()).map(|i| up(v[i])).collect() }

/// SO(3) log_map at the compute tier: log(exp(base)^T exp(target)).
fn so3_log_map_compute<S: Storage, const F: u32>(base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<[Compute<Fixed<S, F>>; 3], OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let g_base = rodrigues_exp_compute(&up_all(base))?;
    let g_target = rodrigues_exp_compute(&up_all(target))?;
    Ok(rodrigues_log_compute(&g_base.transpose().mat_mul(&g_target))?.0)
}

/// SE(3) log_map at the compute tier: log(exp(base)^-1 exp(target)).
fn se3_log_map_compute<S: Storage, const F: u32>(base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<[Compute<Fixed<S, F>>; 6], OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let g_base = se3_exp_compute(&up_all(base))?;
    let g_target = se3_exp_compute(&up_all(target))?;
    se3_log_compute(&se3_inverse_compute(&g_base).mat_mul(&g_target))
}

impl SO3 {
    /// hat: ω = [wx, wy, wz] → 3×3 skew-symmetric matrix.
    pub fn hat_so3<S: Storage, const F: u32>(omega: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let (wx, wy, wz) = (omega[0], omega[1], omega[2]);
        let z = Fixed::ZERO;
        FixedMatrix::<Fixed<S, F>>::from_slice(3, 3, &[z, -wz, wy, wz, z, -wx, -wy, wx, z])
    }

    /// vee: extract ω from skew-symmetric matrix.
    pub fn vee_so3<S: Storage, const F: u32>(m: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        FixedVector::<Fixed<S, F>>::from_slice(&[m.get(2, 1), m.get(0, 2), m.get(1, 0)])
    }

    /// Rodrigues exponential: ω (3-vector) → rotation matrix R.
    ///
    /// R = I + sinc(θ)·[ω]× + half_cosc(θ)·[ω]×²
    ///
    /// **Precision:** θ² (exact), θ, the fused sincos, both coefficients (or
    /// their Taylor series below the threshold) and R are all at the compute
    /// tier; each entry is rounded to storage once. Before 0.6.4 θ² was
    /// rounded to storage and used as a divisor, so a θ just above the
    /// threshold with θ² below half a unit returned `Err(DivisionByZero)` on
    /// realtime (omega = [0.002, 0, 0] at Q16.16).
    pub fn rodrigues_exp<S: Storage, const F: u32>(omega: &FixedVector<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        assert_eq!(omega.len(), 3);
        down_matrix(&rodrigues_exp_compute(&up_all(omega))?)
    }

    /// Rodrigues logarithm: rotation matrix R → axis-angle ω.
    ///
    /// **Precision:** the trace, the skew part, θ = atan2(|vee(R - Rᵀ)|, tr R - 1)
    /// and the axis are at the compute tier; each component is rounded once.
    /// Up to 90 degrees ω = θ·vee(R - Rᵀ)/|vee(R - Rᵀ)|; beyond, the axis
    /// comes from the symmetric part of R, which stays well conditioned up
    /// to π (see `rodrigues_log_compute`).
    pub fn rodrigues_log<S: Storage, const F: u32>(r: &FixedMatrix<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        assert!(r.rows() == 3 && r.cols() == 3, "rodrigues_log: R must be 3x3");
        down_vector(&rodrigues_log_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(r))?.0)
    }
}

// SO(3) Manifold implementation
impl<S: Storage, const F: u32> Manifold<Fixed<S, F>> for SO3
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn dimension(&self) -> usize { 3 }

    fn inner_product(&self, _base: &FixedVector<Fixed<S, F>>, u: &FixedVector<Fixed<S, F>>, v: &FixedVector<Fixed<S, F>>) -> Fixed<S, F> {
        u.dot_precise(v)
    }

    /// log(exp(base) exp(tangent)) with both exps, the product and the log
    /// at the compute tier; each component rounded once.
    fn exp_map(&self, base: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        let g_base = rodrigues_exp_compute(&up_all(base))?;
        let g_tangent = rodrigues_exp_compute(&up_all(tangent))?;
        down_vector(&rodrigues_log_compute(&g_base.mat_mul(&g_tangent))?.0)
    }

    /// log(exp(base)ᵀ exp(target)) entirely at the compute tier, rounded once.
    fn log_map(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        down_vector(&so3_log_map_compute(base, target)?)
    }

    /// |log_map| from the compute-tier log, one rounding.
    fn distance(&self, p: &FixedVector<Fixed<S, F>>, q: &FixedVector<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected> {
        norm_to_storage(&so3_log_map_compute(p, q)?)
    }

    /// vee(R_half [v]× R_halfᵀ) = R_half v with R_half = exp(log_map / 2),
    /// the whole chain at the compute tier and each component rounded once.
    fn parallel_transport(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        let diff = so3_log_map_compute(base, target)?;
        let half = [Fixed::<S, F>::compute_halve(diff[0]), Fixed::<S, F>::compute_halve(diff[1]), Fixed::<S, F>::compute_halve(diff[2])];
        let r_half = rodrigues_exp_compute(&half)?;
        down_vector(&r_half.mul_vector_compute(&up_all(tangent)))
    }
}

// SO(3) LieGroup implementation
impl<S: Storage, const F: u32> LieGroup<Fixed<S, F>> for SO3
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn algebra_dim(&self) -> usize { 3 }
    fn matrix_dim(&self) -> usize { 3 }
    fn identity_element(&self) -> FixedMatrix<Fixed<S, F>> { FixedMatrix::<Fixed<S, F>>::identity(3) }
    fn compose(&self, g1: &FixedMatrix<Fixed<S, F>>, g2: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { g1 * g2 }
    fn group_inverse(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> { Ok(g.transpose()) }
    fn lie_exp(&self, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> { SO3::rodrigues_exp(xi) }
    fn lie_log(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> { SO3::rodrigues_log(g) }
    fn hat(&self, xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { SO3::hat_so3(xi) }
    fn vee(&self, xi_hat: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { SO3::vee_so3(xi_hat) }

    fn adjoint(&self, g: &FixedMatrix<Fixed<S, F>>, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        Ok(g.mul_vector(xi)) // Ad_R(ω) = Rω for SO(3)
    }

    /// [ω₁, ω₂] = ω₁ × ω₂, each component one rounding of its exact value.
    fn bracket(&self, xi: &FixedVector<Fixed<S, F>>, eta: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        let c = cross_exact(xi, eta);
        down_vector(&c).expect("SO3::bracket: result exceeds storage")
    }

    fn act(&self, g: &FixedMatrix<Fixed<S, F>>, point: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        g.mul_vector(point)
    }
}

// ============================================================================
// SE(3) — 3D rigid motions
// ============================================================================

/// SE(3): Special Euclidean group of 3D rigid body motions.
///
/// Elements: 4×4 homogeneous [[R, t], [0, 1]] where R ∈ SO(3), t ∈ R³.
/// Algebra se(3): 6-vectors (ω, v) where ω is rotational, v is translational.
pub struct SE3;

impl SE3 {
    /// hat: ξ = [ωx, ωy, ωz, vx, vy, vz] → 4×4 se(3) matrix.
    pub fn hat_se3<S: Storage, const F: u32>(xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        assert_eq!(xi.len(), 6);
        let z = Fixed::ZERO;
        FixedMatrix::<Fixed<S, F>>::from_slice(4, 4, &[
            z,     -xi[2], xi[1], xi[3],
            xi[2],  z,    -xi[0], xi[4],
            -xi[1], xi[0], z,     xi[5],
            z,      z,     z,     z,
        ])
    }

    /// vee: extract 6-vector from 4×4 se(3) matrix.
    pub fn vee_se3<S: Storage, const F: u32>(m: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        FixedVector::<Fixed<S, F>>::from_slice(&[m.get(2, 1), m.get(0, 2), m.get(1, 0), m.get(0, 3), m.get(1, 3), m.get(2, 3)])
    }

    /// Extract R (3×3) from homogeneous matrix.
    pub fn extract_rotation<S: Storage, const F: u32>(g: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        g.submatrix(0, 0, 3, 3)
    }

    /// Extract t (3-vector) from homogeneous matrix.
    pub fn extract_translation<S: Storage, const F: u32>(g: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        FixedVector::<Fixed<S, F>>::from_slice(&[g.get(0, 3), g.get(1, 3), g.get(2, 3)])
    }

    /// Build 4×4 homogeneous from R and t.
    pub fn from_rt<S: Storage, const F: u32>(r: &FixedMatrix<Fixed<S, F>>, t: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let mut m = FixedMatrix::<Fixed<S, F>>::new(4, 4);
        m.set_submatrix(0, 0, r);
        m.set(0, 3, t[0]); m.set(1, 3, t[1]); m.set(2, 3, t[2]);
        m.set(3, 3, Fixed::one());
        m
    }

    /// SE(3) exponential: ξ = [ω, v] → [[R, V·v], [0, 1]].
    ///
    /// **Precision:** θ² (exact), the sincos, R, the V coefficients (their
    /// Taylor series below 2^(2 - F/2)) and V·v are at the compute tier; each
    /// entry is rounded to storage once. Before 0.6.4 θ² was rounded to
    /// storage (a `DivisionByZero` for small θ on realtime) and the small-angle
    /// branch used the constants 1/2 and 1/6 alone (θ³/24·|v| truncation:
    /// 746 units at θ = 8.8e-6 on Q64.64).
    pub fn se3_exp<S: Storage, const F: u32>(xi: &FixedVector<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        assert_eq!(xi.len(), 6);
        down_matrix(&se3_exp_compute(&up_all(xi))?)
    }

    /// SE(3) logarithm: [[R, t], [0, 1]] → [ω, v].
    ///
    /// **Precision:** ω and θ from the compute-tier Rodrigues log (never
    /// rounded to storage before V⁻¹ is formed), the V⁻¹ coefficient (its
    /// Taylor series below 2^(2 - F/2)) and V⁻¹·t at the compute tier; each
    /// component is rounded once.
    pub fn se3_log<S: Storage, const F: u32>(g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        down_vector(&se3_log_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(g))?)
    }
}

// SE(3) Manifold implementation
impl<S: Storage, const F: u32> Manifold<Fixed<S, F>> for SE3
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn dimension(&self) -> usize { 6 }

    fn inner_product(&self, _base: &FixedVector<Fixed<S, F>>, u: &FixedVector<Fixed<S, F>>, v: &FixedVector<Fixed<S, F>>) -> Fixed<S, F> {
        u.dot_precise(v)
    }

    /// log(exp(base) exp(tangent)) entirely at the compute tier, rounded once.
    fn exp_map(&self, base: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        let g_base = se3_exp_compute(&up_all(base))?;
        let g_tangent = se3_exp_compute(&up_all(tangent))?;
        down_vector(&se3_log_compute(&g_base.mat_mul(&g_tangent))?)
    }

    /// log(exp(base)⁻¹ exp(target)) entirely at the compute tier (the inverse
    /// [[Rᵀ, -Rᵀt], [0, 1]] included), rounded once.
    fn log_map(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        down_vector(&se3_log_map_compute(base, target)?)
    }

    /// |log_map| from the compute-tier log, one rounding.
    fn distance(&self, p: &FixedVector<Fixed<S, F>>, q: &FixedVector<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected> {
        norm_to_storage(&se3_log_map_compute(p, q)?)
    }

    fn parallel_transport(&self, _base: &FixedVector<Fixed<S, F>>, _target: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        // Simplified: for bi-invariant-like metric, transport ≈ identity for small motions
        Ok(tangent.clone())
    }
}

// SE(3) LieGroup implementation
impl<S: Storage, const F: u32> LieGroup<Fixed<S, F>> for SE3
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn algebra_dim(&self) -> usize { 6 }
    fn matrix_dim(&self) -> usize { 4 }
    fn identity_element(&self) -> FixedMatrix<Fixed<S, F>> { FixedMatrix::<Fixed<S, F>>::identity(4) }
    fn compose(&self, g1: &FixedMatrix<Fixed<S, F>>, g2: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { g1 * g2 }

    /// [[Rᵀ, -Rᵀt], [0, 1]]: each translation entry one rounding of the exact
    /// negated sum.
    fn group_inverse(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> {
        let r = SE3::extract_rotation(g);
        let t = SE3::extract_translation(g);
        let rt = r.transpose();
        let t_raw: Vec<S> = (0..3).map(|i| t[i].raw()).collect();
        let mut neg_rt_t = FixedVector::<Fixed<S, F>>::new(3);
        for i in 0..3 {
            let col: Vec<S> = (0..3).map(|k| r.get(k, i).raw()).collect();
            neg_rt_t[i] = down(Fixed::<S, F>::compute_negate(kernels::exact_dot::<Fixed<S, F>>(&col, &t_raw)?))?;
        }
        Ok(SE3::from_rt(&rt, &neg_rt_t))
    }

    fn lie_exp(&self, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> { SE3::se3_exp(xi) }
    fn lie_log(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> { SE3::se3_log(g) }
    fn hat(&self, xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { SE3::hat_se3(xi) }
    fn vee(&self, xi_hat: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { SE3::vee_se3(xi_hat) }

    /// Ad_g(ω, v) = (Rω, Rv + t × Rω): Rω and Rv exact at the compute tier,
    /// t × Rω exact at 3F, each component rounded once.
    fn adjoint(&self, g: &FixedMatrix<Fixed<S, F>>, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        let r = SE3::extract_rotation(g);
        let t = SE3::extract_translation(g);
        let omega: Vec<S> = (0..3).map(|i| xi[i].raw()).collect();
        let v: Vec<S> = (0..3).map(|i| xi[3 + i].raw()).collect();
        let mut r_omega = [Fixed::<S, F>::make_compute_int(0); 3];
        let mut r_v = [Fixed::<S, F>::make_compute_int(0); 3];
        for i in 0..3 {
            let row: Vec<S> = (0..3).map(|k| r.get(i, k).raw()).collect();
            r_omega[i] = kernels::exact_dot::<Fixed<S, F>>(&row, &omega)?;
            r_v[i] = kernels::exact_dot::<Fixed<S, F>>(&row, &v)?;
        }
        let one = Fixed::one().raw();
        let mut out = FixedVector::<Fixed<S, F>>::new(6);
        for i in 0..3 {
            out[i] = down(r_omega[i])?;
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            // (t × Rω)_i = t_j (Rω)_k - t_k (Rω)_j
            out[3 + i] = triple_to_storage(&[
                (r_v[i], one),
                (r_omega[k], t[j].raw()),
                (Fixed::<S, F>::compute_negate(r_omega[j]), t[k].raw()),
            ], false)?;
        }
        Ok(out)
    }

    /// [(ω₁, v₁), (ω₂, v₂)] = (ω₁ × ω₂, ω₁ × v₂ - ω₂ × v₁), each component one
    /// rounding of its exact value.
    fn bracket(&self, xi: &FixedVector<Fixed<S, F>>, eta: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        let w1 = FixedVector::<Fixed<S, F>>::from_slice(&[xi[0], xi[1], xi[2]]);
        let v1 = FixedVector::<Fixed<S, F>>::from_slice(&[xi[3], xi[4], xi[5]]);
        let w2 = FixedVector::<Fixed<S, F>>::from_slice(&[eta[0], eta[1], eta[2]]);
        let v2 = FixedVector::<Fixed<S, F>>::from_slice(&[eta[3], eta[4], eta[5]]);
        let w = cross_exact(&w1, &w2);
        let (a, b) = (cross_exact(&w1, &v2), cross_exact(&w2, &v1));
        let out = [w[0], w[1], w[2],
            Fixed::<S, F>::compute_subtract(a[0], b[0]), Fixed::<S, F>::compute_subtract(a[1], b[1]), Fixed::<S, F>::compute_subtract(a[2], b[2])];
        down_vector(&out).expect("SE3::bracket: result exceeds storage")
    }

    fn act(&self, g: &FixedMatrix<Fixed<S, F>>, point: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        let r = SE3::extract_rotation(g);
        let t = SE3::extract_translation(g);
        &r.mul_vector(point) + &t
    }
}

// ============================================================================
// SO(n) — General rotations via matrix_exp/matrix_log
// ============================================================================

/// SO(n): General special orthogonal group.
///
/// For n=3, prefer `SO3` (closed-form Rodrigues, faster + more precise).
/// For n>3, uses `matrix_exp`/`matrix_log` from L1D.
pub struct SOn {
    pub n: usize,
}

impl SOn {
    /// hat: vector → skew-symmetric n×n matrix.
    /// Convention: upper-triangular entries in row-major order.
    pub fn hat_son<S: Storage, const F: u32>(&self, xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let n = self.n;
        let mut m = FixedMatrix::<Fixed<S, F>>::new(n, n);
        let mut k = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                m.set(i, j, xi[k]);
                m.set(j, i, -xi[k]);
                k += 1;
            }
        }
        m
    }

    /// vee: skew-symmetric n×n matrix → vector.
    pub fn vee_son<S: Storage, const F: u32>(&self, m: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let n = self.n;
        let dim = n * (n - 1) / 2;
        let mut v = FixedVector::<Fixed<S, F>>::new(dim);
        let mut k = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                v[k] = m.get(i, j);
                k += 1;
            }
        }
        v
    }

    /// vee of a compute-tier skew matrix, each component rounded once.
    fn vee_son_compute<S: Storage, const F: u32>(&self, m: &ComputeMatrix<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let n = self.n;
        let mut v = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n { v.push(m.get(i, j)); }
        }
        down_vector(&v)
    }

    /// The upper-triangle coordinates of skew(log g) at the compute tier.
    fn log_compute<S: Storage, const F: u32>(&self, g: &ComputeMatrix<Fixed<S, F>>) -> Result<Vec<Compute<Fixed<S, F>>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let skew = skew_half(&matrix_log_compute(g)?);
        let n = self.n;
        let mut v = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n { v.push(skew.get(i, j)); }
        }
        Ok(v)
    }

    fn exp_compute<S: Storage, const F: u32>(&self, xi: &FixedVector<Fixed<S, F>>) -> Result<ComputeMatrix<Fixed<S, F>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        matrix_exp_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(&self.hat_son(xi)))
    }

    fn log_map_compute<S: Storage, const F: u32>(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<Vec<Compute<Fixed<S, F>>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let g = self.exp_compute(base)?.transpose().mat_mul(&self.exp_compute(target)?);
        self.log_compute(&g)
    }
}

impl<S: Storage, const F: u32> Manifold<Fixed<S, F>> for SOn
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn dimension(&self) -> usize { self.n * (self.n - 1) / 2 }
    fn inner_product(&self, _base: &FixedVector<Fixed<S, F>>, u: &FixedVector<Fixed<S, F>>, v: &FixedVector<Fixed<S, F>>) -> Fixed<S, F> { u.dot_precise(v) }

    /// log(exp(base) exp(tangent)): both matrix exps, the product, the matrix
    /// log and its skew part at the compute tier, rounded once.
    fn exp_map(&self, base: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        let g = self.exp_compute(base)?.mat_mul(&self.exp_compute(tangent)?);
        down_vector(&self.log_compute(&g)?)
    }

    /// log(exp(base)ᵀ exp(target)) at the compute tier, rounded once.
    fn log_map(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        down_vector(&self.log_map_compute(base, target)?)
    }

    /// |log_map| from the compute-tier log, one rounding.
    fn distance(&self, p: &FixedVector<Fixed<S, F>>, q: &FixedVector<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected> {
        norm_to_storage(&self.log_map_compute(p, q)?)
    }

    fn parallel_transport(&self, _base: &FixedVector<Fixed<S, F>>, _target: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        Ok(tangent.clone())
    }
}

impl<S: Storage, const F: u32> LieGroup<Fixed<S, F>> for SOn
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn algebra_dim(&self) -> usize { self.n * (self.n - 1) / 2 }
    fn matrix_dim(&self) -> usize { self.n }
    fn identity_element(&self) -> FixedMatrix<Fixed<S, F>> { FixedMatrix::<Fixed<S, F>>::identity(self.n) }
    fn compose(&self, g1: &FixedMatrix<Fixed<S, F>>, g2: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { g1 * g2 }
    fn group_inverse(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> { Ok(g.transpose()) }

    fn lie_exp(&self, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> {
        matrix_exp(&self.hat_son(xi))
    }

    /// vee((log g - (log g)ᵀ) / 2) with the matrix log and the skew part at
    /// the compute tier, each component rounded once.
    fn lie_log(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        down_vector(&self.log_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(g))?)
    }

    fn hat(&self, xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { self.hat_son(xi) }
    fn vee(&self, xi_hat: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { self.vee_son(xi_hat) }

    /// vee(g ξ^ gᵀ), each component one rounding of its exact value.
    fn adjoint(&self, g: &FixedMatrix<Fixed<S, F>>, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        Ok(self.vee_son(&conjugate_exact(g, &self.hat_son(xi), g)?))
    }

    /// vee(AB - BA), each component one rounding of its exact value.
    fn bracket(&self, xi: &FixedVector<Fixed<S, F>>, eta: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        self.vee_son_compute(&commutator_exact(&self.hat_son(xi), &self.hat_son(eta)))
            .expect("SOn::bracket: result exceeds storage")
    }

    fn act(&self, g: &FixedMatrix<Fixed<S, F>>, point: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        g.mul_vector(point)
    }
}

// ============================================================================
// GL(n) — General linear group (invertible n×n matrices)
// ============================================================================

/// GL(n): General linear group of invertible n×n matrices.
///
/// The most general matrix Lie group. Compose = matmul, inverse = LU-based.
/// Algebra gl(n) = all n×n matrices (no constraint), parameterized as n²-vectors.
///
/// **FASC-UGOD integration:** lie_exp/lie_log route through matrix_exp/matrix_log
/// which internally use ComputeMatrix<Fixed<S, F>> at tier N+1. Inverse uses LU with
/// compute_tier_sub_dot_raw. All precision guarantees from L1D flow through.
pub struct GLn {
    pub n: usize,
}

impl GLn {
    /// hat: n²-vector → n×n matrix (row-major).
    pub fn hat_gln<S: Storage, const F: u32>(&self, xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let n = self.n;
        FixedMatrix::<Fixed<S, F>>::from_fn(n, n, |i, j| xi[i * n + j])
    }

    /// vee: n×n matrix → n²-vector (row-major).
    pub fn vee_gln<S: Storage, const F: u32>(&self, m: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let n = self.n;
        let mut v = FixedVector::<Fixed<S, F>>::new(n * n);
        for i in 0..n {
            for j in 0..n {
                v[i * n + j] = m.get(i, j);
            }
        }
        v
    }

    fn vee_gln_compute<S: Storage, const F: u32>(&self, m: &ComputeMatrix<Fixed<S, F>>) -> Vec<Compute<Fixed<S, F>>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let n = self.n;
        (0..n * n).map(|k| m.get(k / n, k % n)).collect()
    }

    fn exp_compute<S: Storage, const F: u32>(&self, xi: &FixedVector<Fixed<S, F>>) -> Result<ComputeMatrix<Fixed<S, F>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        matrix_exp_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(&self.hat_gln(xi)))
    }

    fn log_map_compute<S: Storage, const F: u32>(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<Vec<Compute<Fixed<S, F>>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let g = inverse_compute(&self.exp_compute(base)?)?.mat_mul(&self.exp_compute(target)?);
        Ok(self.vee_gln_compute(&matrix_log_compute(&g)?))
    }
}

impl<S: Storage, const F: u32> Manifold<Fixed<S, F>> for GLn
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn dimension(&self) -> usize { self.n * self.n }

    fn inner_product(&self, _base: &FixedVector<Fixed<S, F>>, u: &FixedVector<Fixed<S, F>>, v: &FixedVector<Fixed<S, F>>) -> Fixed<S, F> {
        u.dot_precise(v)
    }

    /// log(exp(base) exp(tangent)) with the exps, the product and the log at
    /// the compute tier, rounded once.
    fn exp_map(&self, base: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        let g = self.exp_compute(base)?.mat_mul(&self.exp_compute(tangent)?);
        down_vector(&self.vee_gln_compute(&matrix_log_compute(&g)?))
    }

    /// log(exp(base)⁻¹ exp(target)) with a compute-tier LU inverse (the
    /// storage LU inverse carried O(κ) units into the log before 0.6.4),
    /// rounded once.
    fn log_map(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        down_vector(&self.log_map_compute(base, target)?)
    }

    /// |log_map| from the compute-tier log, one rounding.
    fn distance(&self, p: &FixedVector<Fixed<S, F>>, q: &FixedVector<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected> {
        norm_to_storage(&self.log_map_compute(p, q)?)
    }

    fn parallel_transport(&self, _base: &FixedVector<Fixed<S, F>>, _target: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        Ok(tangent.clone())
    }
}

impl<S: Storage, const F: u32> LieGroup<Fixed<S, F>> for GLn
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn algebra_dim(&self) -> usize { self.n * self.n }
    fn matrix_dim(&self) -> usize { self.n }
    fn identity_element(&self) -> FixedMatrix<Fixed<S, F>> { FixedMatrix::<Fixed<S, F>>::identity(self.n) }
    fn compose(&self, g1: &FixedMatrix<Fixed<S, F>>, g2: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { g1 * g2 }
    /// g⁻¹ from a compute-tier LU, each entry rounded once.
    fn group_inverse(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> {
        down_matrix(&inverse_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(g))?)
    }
    fn lie_exp(&self, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> { matrix_exp(&self.hat_gln(xi)) }
    fn lie_log(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> { Ok(self.vee_gln(&matrix_log(g)?)) }
    fn hat(&self, xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { self.hat_gln(xi) }
    fn vee(&self, xi_hat: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { self.vee_gln(xi_hat) }

    /// vee(g ξ^ g⁻¹) with the inverse and both products at the compute tier,
    /// each component rounded once.
    fn adjoint(&self, g: &FixedMatrix<Fixed<S, F>>, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        down_vector(&self.vee_gln_compute(&conjugate_by_inverse(g, &self.hat_gln(xi))?))
    }

    /// vee(AB - BA), each component one rounding of its exact value.
    fn bracket(&self, xi: &FixedVector<Fixed<S, F>>, eta: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        let c = commutator_exact(&self.hat_gln(xi), &self.hat_gln(eta));
        down_vector(&self.vee_gln_compute(&c)).expect("GLn::bracket: result exceeds storage")
    }

    fn act(&self, g: &FixedMatrix<Fixed<S, F>>, point: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { g.mul_vector(point) }
}

// ============================================================================
// O(n) — Orthogonal group (rotations + reflections, det = ±1)
// ============================================================================

/// O(n): Orthogonal group: matrices with QᵀQ = I, det = ±1.
///
/// Same algebra as SO(n) (skew-symmetric), but includes reflections.
/// Inverse = transpose (exact, no LU). Delegates to SOn for exp/log.
///
/// **FASC-UGOD integration:** Identical to SOn: matrix_exp on skew-symmetric
/// input guarantees orthogonal output at tier N+1.
pub struct On {
    pub n: usize,
}

impl<S: Storage, const F: u32> Manifold<Fixed<S, F>> for On
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn dimension(&self) -> usize { self.n * (self.n - 1) / 2 }
    fn inner_product(&self, _base: &FixedVector<Fixed<S, F>>, u: &FixedVector<Fixed<S, F>>, v: &FixedVector<Fixed<S, F>>) -> Fixed<S, F> { u.dot_precise(v) }
    fn exp_map(&self, base: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> { SOn { n: self.n }.exp_map(base, tangent) }
    fn log_map(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> { SOn { n: self.n }.log_map(base, target) }
    fn distance(&self, p: &FixedVector<Fixed<S, F>>, q: &FixedVector<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected> { Ok(self.log_map(p, q)?.length()) }
    fn parallel_transport(&self, _base: &FixedVector<Fixed<S, F>>, _target: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> { Ok(tangent.clone()) }
}

impl<S: Storage, const F: u32> LieGroup<Fixed<S, F>> for On
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn algebra_dim(&self) -> usize { self.n * (self.n - 1) / 2 }
    fn matrix_dim(&self) -> usize { self.n }
    fn identity_element(&self) -> FixedMatrix<Fixed<S, F>> { FixedMatrix::<Fixed<S, F>>::identity(self.n) }
    fn compose(&self, g1: &FixedMatrix<Fixed<S, F>>, g2: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { g1 * g2 }
    fn group_inverse(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> { Ok(g.transpose()) }
    fn lie_exp(&self, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> { SOn { n: self.n }.lie_exp(xi) }
    fn lie_log(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> { SOn { n: self.n }.lie_log(g) }
    fn hat(&self, xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { SOn { n: self.n }.hat_son(xi) }
    fn vee(&self, xi_hat: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { SOn { n: self.n }.vee_son(xi_hat) }
    fn adjoint(&self, g: &FixedMatrix<Fixed<S, F>>, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> { SOn { n: self.n }.adjoint(g, xi) }
    fn bracket(&self, xi: &FixedVector<Fixed<S, F>>, eta: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { SOn { n: self.n }.bracket(xi, eta) }
    fn act(&self, g: &FixedMatrix<Fixed<S, F>>, point: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { g.mul_vector(point) }
}

// ============================================================================
// SL(n) — Special linear group (det = 1, traceless algebra)
// ============================================================================

/// SL(n): Special linear group: n×n matrices with det = 1.
///
/// Algebra sl(n) = traceless n×n matrices (tr(A) = 0), dimension n²-1.
/// det(exp(A)) = exp(tr(A)) = 1 when A is traceless: algebraic guarantee.
///
/// **FASC-UGOD integration:** lie_exp via matrix_exp at tier N+1 (Padé [6/6]).
/// The traceless constraint is preserved exactly by the exponential. lie_log
/// projects back to traceless via `project_traceless` to absorb numerical drift.
/// Inverse uses LU with compute_tier_sub_dot_raw.
pub struct SLn {
    pub n: usize,
}

impl SLn {
    /// hat: (n²-1)-vector → traceless n×n matrix.
    ///
    /// Layout: first (n-1) entries = diagonal d[0]..d[n-2],
    /// remaining n²-n entries = off-diagonal row-major.
    /// d[n-1] = -(d[0]+...+d[n-2]) enforces tr=0.
    pub fn hat_sln<S: Storage, const F: u32>(&self, xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let n = self.n;
        let mut m = FixedMatrix::<Fixed<S, F>>::new(n, n);
        let mut trace_sum = Fixed::ZERO;
        for i in 0..n - 1 {
            m.set(i, i, xi[i]);
            trace_sum = trace_sum + xi[i];
        }
        m.set(n - 1, n - 1, -trace_sum);
        let mut k = n - 1;
        for i in 0..n {
            for j in 0..n {
                if i != j { m.set(i, j, xi[k]); k += 1; }
            }
        }
        m
    }

    /// vee: traceless n×n matrix → (n²-1)-vector.
    pub fn vee_sln<S: Storage, const F: u32>(&self, m: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let n = self.n;
        let mut v = FixedVector::<Fixed<S, F>>::new(n * n - 1);
        for i in 0..n - 1 { v[i] = m.get(i, i); }
        let mut k = n - 1;
        for i in 0..n {
            for j in 0..n {
                if i != j { v[k] = m.get(i, j); k += 1; }
            }
        }
        v
    }

    /// Project matrix onto sl(n) by removing trace: A - (tr(A)/n)·I.
    ///
    /// The trace (exact), tr/n and each diagonal difference are at the compute
    /// tier; each diagonal entry is rounded once (before 0.6.4 tr/n was
    /// rounded to storage and then subtracted, and the trace summed at
    /// storage).
    pub fn project_traceless<S: Storage, const F: u32>(m: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        down_matrix(&project_traceless_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(m)))
            .expect("SLn::project_traceless: result exceeds storage")
    }

    fn vee_sln_compute<S: Storage, const F: u32>(&self, m: &ComputeMatrix<Fixed<S, F>>) -> Vec<Compute<Fixed<S, F>>>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let n = self.n;
        let mut v: Vec<Compute<Fixed<S, F>>> = (0..n - 1).map(|i| m.get(i, i)).collect();
        for i in 0..n {
            for j in 0..n {
                if i != j { v.push(m.get(i, j)); }
            }
        }
        v
    }

    fn exp_compute<S: Storage, const F: u32>(&self, xi: &FixedVector<Fixed<S, F>>) -> Result<ComputeMatrix<Fixed<S, F>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        matrix_exp_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(&self.hat_sln(xi)))
    }

    /// vee(project_traceless(log g)) at the compute tier.
    fn log_compute<S: Storage, const F: u32>(&self, g: &ComputeMatrix<Fixed<S, F>>) -> Result<Vec<Compute<Fixed<S, F>>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        Ok(self.vee_sln_compute(&project_traceless_compute(&matrix_log_compute(g)?)))
    }

    fn log_map_compute<S: Storage, const F: u32>(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<Vec<Compute<Fixed<S, F>>>, OverflowDetected>
    where
        Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
        {
        let g = inverse_compute(&self.exp_compute(base)?)?.mat_mul(&self.exp_compute(target)?);
        self.log_compute(&g)
    }
}

/// A - (tr(A)/n) I at the compute tier.
fn project_traceless_compute<S: Storage, const F: u32>(m: &ComputeMatrix<Fixed<S, F>>) -> ComputeMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    let n = m.rows();
    let mut trace = Fixed::<S, F>::make_compute_int(0);
    for i in 0..n { trace = Fixed::<S, F>::compute_add(trace, m.get(i, i)); }
    let trace_per_n = Fixed::<S, F>::compute_mul_div_int(trace, 1, n as i64).expect("project_traceless: n > 0");
    let mut result = m.copy();
    for i in 0..n { result.set(i, i, Fixed::<S, F>::compute_subtract(m.get(i, i), trace_per_n)); }
    result
}

impl<S: Storage, const F: u32> Manifold<Fixed<S, F>> for SLn
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn dimension(&self) -> usize { self.n * self.n - 1 }
    fn inner_product(&self, _base: &FixedVector<Fixed<S, F>>, u: &FixedVector<Fixed<S, F>>, v: &FixedVector<Fixed<S, F>>) -> Fixed<S, F> { u.dot_precise(v) }

    /// log(exp(base) exp(tangent)), projected traceless, at the compute tier;
    /// rounded once.
    fn exp_map(&self, base: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        let g = self.exp_compute(base)?.mat_mul(&self.exp_compute(tangent)?);
        down_vector(&self.log_compute(&g)?)
    }

    /// log(exp(base)⁻¹ exp(target)) with a compute-tier LU inverse, projected
    /// traceless at the compute tier; rounded once.
    fn log_map(&self, base: &FixedVector<Fixed<S, F>>, target: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        down_vector(&self.log_map_compute(base, target)?)
    }

    /// |log_map| from the compute-tier log, one rounding.
    fn distance(&self, p: &FixedVector<Fixed<S, F>>, q: &FixedVector<Fixed<S, F>>) -> Result<Fixed<S, F>, OverflowDetected> {
        norm_to_storage(&self.log_map_compute(p, q)?)
    }
    fn parallel_transport(&self, _base: &FixedVector<Fixed<S, F>>, _target: &FixedVector<Fixed<S, F>>, tangent: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> { Ok(tangent.clone()) }
}

impl<S: Storage, const F: u32> LieGroup<Fixed<S, F>> for SLn
where
    Fixed<S, F>: FixedNum<Storage = S> + MatrixFunctions,
{
    fn algebra_dim(&self) -> usize { self.n * self.n - 1 }
    fn matrix_dim(&self) -> usize { self.n }
    fn identity_element(&self) -> FixedMatrix<Fixed<S, F>> { FixedMatrix::<Fixed<S, F>>::identity(self.n) }
    fn compose(&self, g1: &FixedMatrix<Fixed<S, F>>, g2: &FixedMatrix<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { g1 * g2 }
    /// g⁻¹ from a compute-tier LU, each entry rounded once.
    fn group_inverse(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> {
        down_matrix(&inverse_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(g))?)
    }
    fn lie_exp(&self, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedMatrix<Fixed<S, F>>, OverflowDetected> { matrix_exp(&self.hat_sln(xi)) }

    /// vee(project_traceless(log g)) with the matrix log and the projection at
    /// the compute tier, each component rounded once.
    fn lie_log(&self, g: &FixedMatrix<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        down_vector(&self.log_compute(&ComputeMatrix::<Fixed<S, F>>::from_fixed_matrix(g))?)
    }

    fn hat(&self, xi: &FixedVector<Fixed<S, F>>) -> FixedMatrix<Fixed<S, F>> { self.hat_sln(xi) }
    fn vee(&self, xi_hat: &FixedMatrix<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { self.vee_sln(xi_hat) }

    /// vee(project_traceless(g ξ^ g⁻¹)) with the inverse, both products and
    /// the projection at the compute tier, each component rounded once.
    fn adjoint(&self, g: &FixedMatrix<Fixed<S, F>>, xi: &FixedVector<Fixed<S, F>>) -> Result<FixedVector<Fixed<S, F>>, OverflowDetected> {
        let conj = conjugate_by_inverse(g, &self.hat_sln(xi))?;
        down_vector(&self.vee_sln_compute(&project_traceless_compute(&conj)))
    }

    /// vee(AB - BA), each component one rounding of its exact value.
    fn bracket(&self, xi: &FixedVector<Fixed<S, F>>, eta: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        let c = commutator_exact(&self.hat_sln(xi), &self.hat_sln(eta));
        down_vector(&self.vee_sln_compute(&c)).expect("SLn::bracket: result exceeds storage")
    }

    fn act(&self, g: &FixedMatrix<Fixed<S, F>>, point: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> { g.mul_vector(point) }
}
