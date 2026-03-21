//! L4A: Lie groups and Lie algebras with fixed-point arithmetic.
//!
//! - `SO3` — 3D rotations via closed-form Rodrigues (O(1) trig, not matrix_exp)
//! - `SE3` — 3D rigid motions (rotation + translation) via closed-form V-matrix
//! - `SOn` — General n×n rotations via matrix_exp/matrix_log fallback
//!
//! SO(3) Rodrigues is the foundational primitive for geometric key derivation.

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::manifold::Manifold;
use super::matrix_functions::{matrix_exp, matrix_log};
use super::compute_matrix::ComputeMatrix;
use super::linalg::{upscale_to_compute, sincos_at_compute_tier, ComputeStorage, round_to_storage};
use super::derived::inverse;
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_subtract, compute_multiply, compute_divide,
};

// ============================================================================
// LieGroup trait
// ============================================================================

/// A Lie group with fixed-point arithmetic.
///
/// Group elements are `FixedMatrix`. Algebra elements are `FixedVector`
/// (coordinate parameterization) with `hat`/`vee` for matrix form.
pub trait LieGroup: Manifold {
    /// Dimension of the Lie algebra.
    fn algebra_dim(&self) -> usize;
    /// Dimension of the matrix representation (n for n×n).
    fn matrix_dim(&self) -> usize;
    /// Group identity element.
    fn identity_element(&self) -> FixedMatrix;
    /// Group composition: g1 * g2.
    fn compose(&self, g1: &FixedMatrix, g2: &FixedMatrix) -> FixedMatrix;
    /// Group inverse: g⁻¹.
    fn group_inverse(&self, g: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected>;
    /// Exponential map: algebra vector → group element.
    fn lie_exp(&self, xi: &FixedVector) -> Result<FixedMatrix, OverflowDetected>;
    /// Logarithmic map: group element → algebra vector.
    fn lie_log(&self, g: &FixedMatrix) -> Result<FixedVector, OverflowDetected>;
    /// Hat map: algebra vector → algebra matrix.
    fn hat(&self, xi: &FixedVector) -> FixedMatrix;
    /// Vee map: algebra matrix → algebra vector.
    fn vee(&self, xi_hat: &FixedMatrix) -> FixedVector;
    /// Adjoint: Ad_g(xi) in algebra coordinates.
    fn adjoint(&self, g: &FixedMatrix, xi: &FixedVector) -> Result<FixedVector, OverflowDetected>;
    /// Lie bracket: [xi, eta].
    fn bracket(&self, xi: &FixedVector, eta: &FixedVector) -> FixedVector;
    /// Group action on a point.
    fn act(&self, g: &FixedMatrix, point: &FixedVector) -> FixedVector;
}

// ============================================================================
// SO(3) — 3D rotations via Rodrigues
// ============================================================================

/// SO(3): Special orthogonal group of 3D rotations.
///
/// Elements: 3×3 orthogonal matrices with det = +1.
/// Algebra so(3): skew-symmetric 3×3 matrices, parameterized by 3-vectors.
/// Uses closed-form Rodrigues formula — O(1) trig, not matrix_exp.
pub struct SO3;

#[cfg(table_format = "q64_64")]
const RODRIGUES_THRESH: &str = "0.00001";
#[cfg(table_format = "q128_128")]
const RODRIGUES_THRESH: &str = "0.0000000001";
#[cfg(table_format = "q256_256")]
const RODRIGUES_THRESH: &str = "0.00000000000000000001";

fn rodrigues_threshold() -> FixedPoint {
    FixedPoint::from_str(RODRIGUES_THRESH)
}

fn clamp_unit(x: FixedPoint) -> FixedPoint {
    let one = FixedPoint::one();
    if x > one { one } else if x < -one { -one } else { x }
}

impl SO3 {
    /// hat: ω = [wx, wy, wz] → 3×3 skew-symmetric matrix.
    pub fn hat_so3(omega: &FixedVector) -> FixedMatrix {
        let (wx, wy, wz) = (omega[0], omega[1], omega[2]);
        let z = FixedPoint::ZERO;
        FixedMatrix::from_slice(3, 3, &[z, -wz, wy, wz, z, -wx, -wy, wx, z])
    }

    /// vee: extract ω from skew-symmetric matrix.
    pub fn vee_so3(m: &FixedMatrix) -> FixedVector {
        FixedVector::from_slice(&[m.get(2, 1), m.get(0, 2), m.get(1, 0)])
    }

    /// Rodrigues exponential: ω (3-vector) → rotation matrix R.
    ///
    /// R = I + sinc(θ)·[ω]× + half_cosc(θ)·[ω]×²
    ///
    /// **Precision:** Entire formula evaluated at compute tier via ComputeMatrix.
    /// Fused sincos computes sin(θ) and cos(θ) from a single range reduction.
    /// Scalar coefficients (sinc, half_cosc) computed at compute tier — zero
    /// mid-chain materializations. Single downscale at the end → 0-1 ULP.
    pub fn rodrigues_exp(omega: &FixedVector) -> Result<FixedMatrix, OverflowDetected> {
        assert_eq!(omega.len(), 3);
        let theta_sq = omega.dot_precise(omega);
        let theta = theta_sq.try_sqrt()?;

        let (sinc_c, hc_c) = if theta < rodrigues_threshold() {
            // Taylor branch: sinc ≈ 1 - θ²/6 + θ⁴/120, half_cosc ≈ 1/2 - θ²/24 + θ⁴/720
            let t2 = theta_sq;
            let t4 = t2 * t2;
            let one = FixedPoint::one();
            let sinc = one - t2 / FixedPoint::from_int(6) + t4 / FixedPoint::from_int(120);
            let half_cosc = one / FixedPoint::from_int(2) - t2 / FixedPoint::from_int(24) + t4 / FixedPoint::from_int(720);
            (upscale_to_compute(sinc.raw()), upscale_to_compute(half_cosc.raw()))
        } else {
            // Fused sincos at compute tier — single range reduction, no materialization
            let theta_c = upscale_to_compute(theta.raw());
            let theta_sq_c = upscale_to_compute(theta_sq.raw());
            let one_c = upscale_to_compute(FixedPoint::one().raw());
            let (sin_c, cos_c) = sincos_at_compute_tier(theta_c);
            let sinc_c = compute_divide(sin_c, theta_c)?;
            let hc_c = compute_divide(compute_subtract(one_c, cos_c), theta_sq_c)?;
            (sinc_c, hc_c)
        };

        // Evaluate R = I + sinc·[ω]× + half_cosc·[ω]×² entirely at compute tier
        let omega_hat = Self::hat_so3(omega);
        let id_c = ComputeMatrix::identity(3);
        let oh_c = ComputeMatrix::from_fixed_matrix(&omega_hat);
        let oh2_c = oh_c.mat_mul(&oh_c); // [ω]×² at compute tier — no storage rounding

        let result = id_c.add(&oh_c.scalar_mul(sinc_c)).add(&oh2_c.scalar_mul(hc_c));
        Ok(result.to_fixed_matrix())
    }

    /// Rodrigues logarithm: rotation matrix R → axis-angle ω.
    ///
    /// Three cases: near-identity (Taylor), near-π (eigenvector), general (arccos).
    pub fn rodrigues_log(r: &FixedMatrix) -> Result<FixedVector, OverflowDetected> {
        let one = FixedPoint::one();
        let two = FixedPoint::from_int(2);
        let trace = r.trace();
        let cos_theta = clamp_unit((trace - one) / two);
        let theta = cos_theta.try_acos()?;

        let thresh = rodrigues_threshold();

        // Near identity: ω ≈ vee((R - Rᵀ) / 2)
        if theta < thresh {
            let skew = &(r - &r.transpose()) * (one / two);
            return Ok(Self::vee_so3(&skew));
        }

        // Near π: extract axis from R + I (rank-1)
        let pi = FixedPoint::from_str("3.14159265358979323846");
        if theta > pi - thresh {
            let r_plus_i = r + &FixedMatrix::identity(3);
            let mut best_col = 0;
            let mut best_norm = FixedPoint::ZERO;
            for j in 0..3 {
                let col = r_plus_i.col(j);
                let n = col.length();
                if n > best_norm {
                    best_norm = n;
                    best_col = j;
                }
            }
            if best_norm.is_zero() {
                return Err(OverflowDetected::DomainError);
            }
            let axis = r_plus_i.col(best_col).normalized();
            return Ok(&axis * theta);
        }

        // General: ω = θ/(2 sin θ) · vee(R - Rᵀ)
        let sin_theta = theta.try_sin()?;
        let coeff = theta / (two * sin_theta);
        let skew = r - &r.transpose();
        Ok(Self::vee_so3(&(&skew * coeff)))
    }
}

// SO(3) Manifold implementation
impl Manifold for SO3 {
    fn dimension(&self) -> usize { 3 }

    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint {
        u.dot_precise(v)
    }

    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = SO3::rodrigues_exp(base)?;
        let g_tangent = SO3::rodrigues_exp(tangent)?;
        SO3::rodrigues_log(&(&g_base * &g_tangent))
    }

    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = SO3::rodrigues_exp(base)?;
        let g_target = SO3::rodrigues_exp(target)?;
        SO3::rodrigues_log(&(&g_base.transpose() * &g_target))
    }

    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        Ok(self.log_map(p, q)?.length())
    }

    fn parallel_transport(&self, base: &FixedVector, target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let diff = self.log_map(base, target)?;
        let half_diff = &diff * (FixedPoint::one() / FixedPoint::from_int(2));
        let r_half = SO3::rodrigues_exp(&half_diff)?;
        let v_hat = SO3::hat_so3(tangent);
        let transported_hat = &(&r_half * &v_hat) * &r_half.transpose();
        Ok(SO3::vee_so3(&transported_hat))
    }
}

// SO(3) LieGroup implementation
impl LieGroup for SO3 {
    fn algebra_dim(&self) -> usize { 3 }
    fn matrix_dim(&self) -> usize { 3 }
    fn identity_element(&self) -> FixedMatrix { FixedMatrix::identity(3) }
    fn compose(&self, g1: &FixedMatrix, g2: &FixedMatrix) -> FixedMatrix { g1 * g2 }
    fn group_inverse(&self, g: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> { Ok(g.transpose()) }
    fn lie_exp(&self, xi: &FixedVector) -> Result<FixedMatrix, OverflowDetected> { SO3::rodrigues_exp(xi) }
    fn lie_log(&self, g: &FixedMatrix) -> Result<FixedVector, OverflowDetected> { SO3::rodrigues_log(g) }
    fn hat(&self, xi: &FixedVector) -> FixedMatrix { SO3::hat_so3(xi) }
    fn vee(&self, xi_hat: &FixedMatrix) -> FixedVector { SO3::vee_so3(xi_hat) }

    fn adjoint(&self, g: &FixedMatrix, xi: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        Ok(g.mul_vector(xi)) // Ad_R(ω) = Rω for SO(3)
    }

    fn bracket(&self, xi: &FixedVector, eta: &FixedVector) -> FixedVector {
        xi.cross(eta) // [ω₁, ω₂] = ω₁ × ω₂
    }

    fn act(&self, g: &FixedMatrix, point: &FixedVector) -> FixedVector {
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
    pub fn hat_se3(xi: &FixedVector) -> FixedMatrix {
        assert_eq!(xi.len(), 6);
        let z = FixedPoint::ZERO;
        FixedMatrix::from_slice(4, 4, &[
            z,     -xi[2], xi[1], xi[3],
            xi[2],  z,    -xi[0], xi[4],
            -xi[1], xi[0], z,     xi[5],
            z,      z,     z,     z,
        ])
    }

    /// vee: extract 6-vector from 4×4 se(3) matrix.
    pub fn vee_se3(m: &FixedMatrix) -> FixedVector {
        FixedVector::from_slice(&[m.get(2, 1), m.get(0, 2), m.get(1, 0), m.get(0, 3), m.get(1, 3), m.get(2, 3)])
    }

    /// Extract R (3×3) from homogeneous matrix.
    pub fn extract_rotation(g: &FixedMatrix) -> FixedMatrix {
        g.submatrix(0, 0, 3, 3)
    }

    /// Extract t (3-vector) from homogeneous matrix.
    pub fn extract_translation(g: &FixedMatrix) -> FixedVector {
        FixedVector::from_slice(&[g.get(0, 3), g.get(1, 3), g.get(2, 3)])
    }

    /// Build 4×4 homogeneous from R and t.
    pub fn from_rt(r: &FixedMatrix, t: &FixedVector) -> FixedMatrix {
        let mut m = FixedMatrix::new(4, 4);
        m.set_submatrix(0, 0, r);
        m.set(0, 3, t[0]); m.set(1, 3, t[1]); m.set(2, 3, t[2]);
        m.set(3, 3, FixedPoint::one());
        m
    }

    /// SE(3) exponential: ξ = [ω, v] → [[R, V·v], [0, 1]].
    ///
    /// **Precision:** Fused sincos at compute tier for coefficients.
    /// V-matrix AND V·v computed entirely at compute tier — single downscale.
    pub fn se3_exp(xi: &FixedVector) -> Result<FixedMatrix, OverflowDetected> {
        assert_eq!(xi.len(), 6);
        let omega = FixedVector::from_slice(&[xi[0], xi[1], xi[2]]);
        let v = FixedVector::from_slice(&[xi[3], xi[4], xi[5]]);

        let theta_sq = omega.dot_precise(&omega);
        let theta = theta_sq.try_sqrt()?;
        let r = SO3::rodrigues_exp(&omega)?;

        let omega_hat = SO3::hat_so3(&omega);

        let (c1_c, c2_c) = if theta < rodrigues_threshold() {
            (
                upscale_to_compute((FixedPoint::one() / FixedPoint::from_int(2)).raw()),
                upscale_to_compute((FixedPoint::one() / FixedPoint::from_int(6)).raw()),
            )
        } else {
            // Fused sincos at compute tier — single range reduction, no materialization
            let theta_c = upscale_to_compute(theta.raw());
            let theta_sq_c = upscale_to_compute(theta_sq.raw());
            let one_c = upscale_to_compute(FixedPoint::one().raw());
            let (sin_c, cos_c) = sincos_at_compute_tier(theta_c);
            // c1 = (1 - cos θ) / θ²
            let c1_c = compute_divide(compute_subtract(one_c, cos_c), theta_sq_c)?;
            // c2 = (θ - sin θ) / θ³
            let theta_cubed_c = compute_multiply(theta_sq_c, theta_c);
            let c2_c = compute_divide(compute_subtract(theta_c, sin_c), theta_cubed_c)?;
            (c1_c, c2_c)
        };

        // V = I + c1·[ω]× + c2·[ω]×² at compute tier
        let id_c = ComputeMatrix::identity(3);
        let oh_c = ComputeMatrix::from_fixed_matrix(&omega_hat);
        let oh2_c = oh_c.mat_mul(&oh_c);
        let v_mat_c = id_c.add(&oh_c.scalar_mul(c1_c)).add(&oh2_c.scalar_mul(c2_c));

        // V·v entirely at compute tier, single downscale
        let v_compute: Vec<ComputeStorage> = (0..3).map(|i| upscale_to_compute(v[i].raw())).collect();
        let t_compute = v_mat_c.mul_vector_compute(&v_compute);
        let t = FixedVector::from_slice(&[
            FixedPoint::from_raw(round_to_storage(t_compute[0])),
            FixedPoint::from_raw(round_to_storage(t_compute[1])),
            FixedPoint::from_raw(round_to_storage(t_compute[2])),
        ]);

        Ok(Self::from_rt(&r, &t))
    }

    /// SE(3) logarithm: [[R, t], [0, 1]] → [ω, v].
    ///
    /// **Precision:** Fused sincos at compute tier for coefficients.
    /// V⁻¹ AND V⁻¹·t computed entirely at compute tier — single downscale.
    pub fn se3_log(g: &FixedMatrix) -> Result<FixedVector, OverflowDetected> {
        let r = Self::extract_rotation(g);
        let t = Self::extract_translation(g);
        let omega = SO3::rodrigues_log(&r)?;
        let theta_sq = omega.dot_precise(&omega);
        let theta = theta_sq.try_sqrt()?;

        let omega_hat = SO3::hat_so3(&omega);

        let (c1_c, c2_c) = if theta < rodrigues_threshold() {
            (
                upscale_to_compute((FixedPoint::one() / FixedPoint::from_int(2)).raw()),
                upscale_to_compute((FixedPoint::one() / FixedPoint::from_int(12)).raw()),
            )
        } else {
            // Fused sincos at compute tier — single range reduction, no materialization
            let theta_c = upscale_to_compute(theta.raw());
            let theta_sq_c = upscale_to_compute(theta_sq.raw());
            let one_c = upscale_to_compute(FixedPoint::one().raw());
            let two_c = upscale_to_compute(FixedPoint::from_int(2).raw());
            let (sin_c, cos_c) = sincos_at_compute_tier(theta_c);
            // alpha = θ·sin(θ) / (2·(1 - cos(θ)))
            let numerator = compute_multiply(theta_c, sin_c);
            let denominator = compute_multiply(two_c, compute_subtract(one_c, cos_c));
            let alpha_c = compute_divide(numerator, denominator)?;
            // c1 = 1/2
            let half_c = upscale_to_compute((FixedPoint::one() / FixedPoint::from_int(2)).raw());
            // c2 = (1 - alpha) / θ²
            let c2_c = compute_divide(compute_subtract(one_c, alpha_c), theta_sq_c)?;
            (half_c, c2_c)
        };

        // V⁻¹ = I - c1·[ω]× + c2·[ω]×² at compute tier
        let id_c = ComputeMatrix::identity(3);
        let oh_c = ComputeMatrix::from_fixed_matrix(&omega_hat);
        let oh2_c = oh_c.mat_mul(&oh_c);
        let v_inv_c = id_c.sub(&oh_c.scalar_mul(c1_c)).add(&oh2_c.scalar_mul(c2_c));

        // V⁻¹·t entirely at compute tier, single downscale
        let t_compute: Vec<ComputeStorage> = (0..3).map(|i| upscale_to_compute(t[i].raw())).collect();
        let v_compute = v_inv_c.mul_vector_compute(&t_compute);
        let v = FixedVector::from_slice(&[
            FixedPoint::from_raw(round_to_storage(v_compute[0])),
            FixedPoint::from_raw(round_to_storage(v_compute[1])),
            FixedPoint::from_raw(round_to_storage(v_compute[2])),
        ]);

        Ok(FixedVector::from_slice(&[omega[0], omega[1], omega[2], v[0], v[1], v[2]]))
    }
}

// SE(3) Manifold implementation
impl Manifold for SE3 {
    fn dimension(&self) -> usize { 6 }

    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint {
        u.dot_precise(v)
    }

    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = SE3::se3_exp(base)?;
        let g_tangent = SE3::se3_exp(tangent)?;
        SE3::se3_log(&(&g_base * &g_tangent))
    }

    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = SE3::se3_exp(base)?;
        let g_target = SE3::se3_exp(target)?;
        let g_base_inv = SE3::from_rt(
            &SE3::extract_rotation(&g_base).transpose(),
            &(-SE3::extract_rotation(&g_base).transpose().mul_vector(&SE3::extract_translation(&g_base))),
        );
        SE3::se3_log(&(&g_base_inv * &g_target))
    }

    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        Ok(self.log_map(p, q)?.length())
    }

    fn parallel_transport(&self, _base: &FixedVector, _target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        // Simplified: for bi-invariant-like metric, transport ≈ identity for small motions
        Ok(tangent.clone())
    }
}

// SE(3) LieGroup implementation
impl LieGroup for SE3 {
    fn algebra_dim(&self) -> usize { 6 }
    fn matrix_dim(&self) -> usize { 4 }
    fn identity_element(&self) -> FixedMatrix { FixedMatrix::identity(4) }
    fn compose(&self, g1: &FixedMatrix, g2: &FixedMatrix) -> FixedMatrix { g1 * g2 }

    fn group_inverse(&self, g: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> {
        let r = SE3::extract_rotation(g);
        let t = SE3::extract_translation(g);
        let rt = r.transpose();
        let neg_rt_t = -rt.mul_vector(&t);
        Ok(SE3::from_rt(&rt, &neg_rt_t))
    }

    fn lie_exp(&self, xi: &FixedVector) -> Result<FixedMatrix, OverflowDetected> { SE3::se3_exp(xi) }
    fn lie_log(&self, g: &FixedMatrix) -> Result<FixedVector, OverflowDetected> { SE3::se3_log(g) }
    fn hat(&self, xi: &FixedVector) -> FixedMatrix { SE3::hat_se3(xi) }
    fn vee(&self, xi_hat: &FixedMatrix) -> FixedVector { SE3::vee_se3(xi_hat) }

    fn adjoint(&self, g: &FixedMatrix, xi: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let r = SE3::extract_rotation(g);
        let t = SE3::extract_translation(g);
        let omega = FixedVector::from_slice(&[xi[0], xi[1], xi[2]]);
        let v = FixedVector::from_slice(&[xi[3], xi[4], xi[5]]);
        let r_omega = r.mul_vector(&omega);
        let t_cross = SO3::hat_so3(&t);
        let r_v = r.mul_vector(&v);
        let t_cross_r_omega = t_cross.mul_vector(&r_omega);
        let adj_v = &r_v + &t_cross_r_omega;
        Ok(FixedVector::from_slice(&[r_omega[0], r_omega[1], r_omega[2], adj_v[0], adj_v[1], adj_v[2]]))
    }

    fn bracket(&self, xi: &FixedVector, eta: &FixedVector) -> FixedVector {
        let w1 = FixedVector::from_slice(&[xi[0], xi[1], xi[2]]);
        let v1 = FixedVector::from_slice(&[xi[3], xi[4], xi[5]]);
        let w2 = FixedVector::from_slice(&[eta[0], eta[1], eta[2]]);
        let v2 = FixedVector::from_slice(&[eta[3], eta[4], eta[5]]);
        let w = w1.cross(&w2);
        let v = &w1.cross(&v2) - &w2.cross(&v1);
        FixedVector::from_slice(&[w[0], w[1], w[2], v[0], v[1], v[2]])
    }

    fn act(&self, g: &FixedMatrix, point: &FixedVector) -> FixedVector {
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
    pub fn hat_son(&self, xi: &FixedVector) -> FixedMatrix {
        let n = self.n;
        let mut m = FixedMatrix::new(n, n);
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
    pub fn vee_son(&self, m: &FixedMatrix) -> FixedVector {
        let n = self.n;
        let dim = n * (n - 1) / 2;
        let mut v = FixedVector::new(dim);
        let mut k = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                v[k] = m.get(i, j);
                k += 1;
            }
        }
        v
    }
}

impl Manifold for SOn {
    fn dimension(&self) -> usize { self.n * (self.n - 1) / 2 }
    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint { u.dot_precise(v) }

    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = self.lie_exp(base)?;
        let g_tangent = self.lie_exp(tangent)?;
        let g_result = &g_base * &g_tangent;
        self.lie_log(&g_result)
    }

    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = self.lie_exp(base)?;
        let g_target = self.lie_exp(target)?;
        self.lie_log(&(&g_base.transpose() * &g_target))
    }

    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        Ok(self.log_map(p, q)?.length())
    }

    fn parallel_transport(&self, _base: &FixedVector, _target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        Ok(tangent.clone())
    }
}

impl LieGroup for SOn {
    fn algebra_dim(&self) -> usize { self.n * (self.n - 1) / 2 }
    fn matrix_dim(&self) -> usize { self.n }
    fn identity_element(&self) -> FixedMatrix { FixedMatrix::identity(self.n) }
    fn compose(&self, g1: &FixedMatrix, g2: &FixedMatrix) -> FixedMatrix { g1 * g2 }
    fn group_inverse(&self, g: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> { Ok(g.transpose()) }

    fn lie_exp(&self, xi: &FixedVector) -> Result<FixedMatrix, OverflowDetected> {
        matrix_exp(&self.hat_son(xi))
    }

    fn lie_log(&self, g: &FixedMatrix) -> Result<FixedVector, OverflowDetected> {
        let log_g = matrix_log(g)?;
        let skew = &(&log_g - &log_g.transpose()) * (FixedPoint::one() / FixedPoint::from_int(2));
        Ok(self.vee_son(&skew))
    }

    fn hat(&self, xi: &FixedVector) -> FixedMatrix { self.hat_son(xi) }
    fn vee(&self, xi_hat: &FixedMatrix) -> FixedVector { self.vee_son(xi_hat) }

    fn adjoint(&self, g: &FixedMatrix, xi: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let xi_hat = self.hat_son(xi);
        let result = &(g * &xi_hat) * &g.transpose();
        Ok(self.vee_son(&result))
    }

    fn bracket(&self, xi: &FixedVector, eta: &FixedVector) -> FixedVector {
        let a = self.hat_son(xi);
        let b = self.hat_son(eta);
        self.vee_son(&(&(&a * &b) - &(&b * &a)))
    }

    fn act(&self, g: &FixedMatrix, point: &FixedVector) -> FixedVector {
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
/// which internally use ComputeMatrix at tier N+1. Inverse uses LU with
/// compute_tier_sub_dot_raw. All precision guarantees from L1D flow through.
pub struct GLn {
    pub n: usize,
}

impl GLn {
    /// hat: n²-vector → n×n matrix (row-major).
    pub fn hat_gln(&self, xi: &FixedVector) -> FixedMatrix {
        let n = self.n;
        FixedMatrix::from_fn(n, n, |i, j| xi[i * n + j])
    }

    /// vee: n×n matrix → n²-vector (row-major).
    pub fn vee_gln(&self, m: &FixedMatrix) -> FixedVector {
        let n = self.n;
        let mut v = FixedVector::new(n * n);
        for i in 0..n {
            for j in 0..n {
                v[i * n + j] = m.get(i, j);
            }
        }
        v
    }
}

impl Manifold for GLn {
    fn dimension(&self) -> usize { self.n * self.n }

    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint {
        u.dot_precise(v)
    }

    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = self.lie_exp(base)?;
        let g_tangent = self.lie_exp(tangent)?;
        self.lie_log(&(&g_base * &g_tangent))
    }

    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = self.lie_exp(base)?;
        let g_target = self.lie_exp(target)?;
        let g_base_inv = inverse(&g_base)?;
        self.lie_log(&(&g_base_inv * &g_target))
    }

    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        Ok(self.log_map(p, q)?.length())
    }

    fn parallel_transport(&self, _base: &FixedVector, _target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        Ok(tangent.clone())
    }
}

impl LieGroup for GLn {
    fn algebra_dim(&self) -> usize { self.n * self.n }
    fn matrix_dim(&self) -> usize { self.n }
    fn identity_element(&self) -> FixedMatrix { FixedMatrix::identity(self.n) }
    fn compose(&self, g1: &FixedMatrix, g2: &FixedMatrix) -> FixedMatrix { g1 * g2 }
    fn group_inverse(&self, g: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> { inverse(g) }
    fn lie_exp(&self, xi: &FixedVector) -> Result<FixedMatrix, OverflowDetected> { matrix_exp(&self.hat_gln(xi)) }
    fn lie_log(&self, g: &FixedMatrix) -> Result<FixedVector, OverflowDetected> { Ok(self.vee_gln(&matrix_log(g)?)) }
    fn hat(&self, xi: &FixedVector) -> FixedMatrix { self.hat_gln(xi) }
    fn vee(&self, xi_hat: &FixedMatrix) -> FixedVector { self.vee_gln(xi_hat) }

    fn adjoint(&self, g: &FixedMatrix, xi: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let xi_hat = self.hat_gln(xi);
        let g_inv = inverse(g)?;
        Ok(self.vee_gln(&(&(g * &xi_hat) * &g_inv)))
    }

    fn bracket(&self, xi: &FixedVector, eta: &FixedVector) -> FixedVector {
        let a = self.hat_gln(xi);
        let b = self.hat_gln(eta);
        self.vee_gln(&(&(&a * &b) - &(&b * &a)))
    }

    fn act(&self, g: &FixedMatrix, point: &FixedVector) -> FixedVector { g.mul_vector(point) }
}

// ============================================================================
// O(n) — Orthogonal group (rotations + reflections, det = ±1)
// ============================================================================

/// O(n): Orthogonal group — matrices with QᵀQ = I, det = ±1.
///
/// Same algebra as SO(n) (skew-symmetric), but includes reflections.
/// Inverse = transpose (exact, no LU). Delegates to SOn for exp/log.
///
/// **FASC-UGOD integration:** Identical to SOn — matrix_exp on skew-symmetric
/// input guarantees orthogonal output at tier N+1.
pub struct On {
    pub n: usize,
}

impl Manifold for On {
    fn dimension(&self) -> usize { self.n * (self.n - 1) / 2 }
    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint { u.dot_precise(v) }
    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> { SOn { n: self.n }.exp_map(base, tangent) }
    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> { SOn { n: self.n }.log_map(base, target) }
    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> { Ok(self.log_map(p, q)?.length()) }
    fn parallel_transport(&self, _base: &FixedVector, _target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> { Ok(tangent.clone()) }
}

impl LieGroup for On {
    fn algebra_dim(&self) -> usize { self.n * (self.n - 1) / 2 }
    fn matrix_dim(&self) -> usize { self.n }
    fn identity_element(&self) -> FixedMatrix { FixedMatrix::identity(self.n) }
    fn compose(&self, g1: &FixedMatrix, g2: &FixedMatrix) -> FixedMatrix { g1 * g2 }
    fn group_inverse(&self, g: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> { Ok(g.transpose()) }
    fn lie_exp(&self, xi: &FixedVector) -> Result<FixedMatrix, OverflowDetected> { SOn { n: self.n }.lie_exp(xi) }
    fn lie_log(&self, g: &FixedMatrix) -> Result<FixedVector, OverflowDetected> { SOn { n: self.n }.lie_log(g) }
    fn hat(&self, xi: &FixedVector) -> FixedMatrix { SOn { n: self.n }.hat_son(xi) }
    fn vee(&self, xi_hat: &FixedMatrix) -> FixedVector { SOn { n: self.n }.vee_son(xi_hat) }
    fn adjoint(&self, g: &FixedMatrix, xi: &FixedVector) -> Result<FixedVector, OverflowDetected> { SOn { n: self.n }.adjoint(g, xi) }
    fn bracket(&self, xi: &FixedVector, eta: &FixedVector) -> FixedVector { SOn { n: self.n }.bracket(xi, eta) }
    fn act(&self, g: &FixedMatrix, point: &FixedVector) -> FixedVector { g.mul_vector(point) }
}

// ============================================================================
// SL(n) — Special linear group (det = 1, traceless algebra)
// ============================================================================

/// SL(n): Special linear group — n×n matrices with det = 1.
///
/// Algebra sl(n) = traceless n×n matrices (tr(A) = 0), dimension n²-1.
/// det(exp(A)) = exp(tr(A)) = 1 when A is traceless — algebraic guarantee.
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
    pub fn hat_sln(&self, xi: &FixedVector) -> FixedMatrix {
        let n = self.n;
        let mut m = FixedMatrix::new(n, n);
        let mut trace_sum = FixedPoint::ZERO;
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
    pub fn vee_sln(&self, m: &FixedMatrix) -> FixedVector {
        let n = self.n;
        let mut v = FixedVector::new(n * n - 1);
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
    pub fn project_traceless(m: &FixedMatrix) -> FixedMatrix {
        let n = m.rows();
        let trace_per_n = m.trace() / FixedPoint::from_int(n as i32);
        let mut result = m.clone();
        for i in 0..n { result.set(i, i, m.get(i, i) - trace_per_n); }
        result
    }
}

impl Manifold for SLn {
    fn dimension(&self) -> usize { self.n * self.n - 1 }
    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint { u.dot_precise(v) }

    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = self.lie_exp(base)?;
        let g_tangent = self.lie_exp(tangent)?;
        self.lie_log(&(&g_base * &g_tangent))
    }

    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let g_base = self.lie_exp(base)?;
        let g_target = self.lie_exp(target)?;
        let g_base_inv = inverse(&g_base)?;
        self.lie_log(&(&g_base_inv * &g_target))
    }

    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> { Ok(self.log_map(p, q)?.length()) }
    fn parallel_transport(&self, _base: &FixedVector, _target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> { Ok(tangent.clone()) }
}

impl LieGroup for SLn {
    fn algebra_dim(&self) -> usize { self.n * self.n - 1 }
    fn matrix_dim(&self) -> usize { self.n }
    fn identity_element(&self) -> FixedMatrix { FixedMatrix::identity(self.n) }
    fn compose(&self, g1: &FixedMatrix, g2: &FixedMatrix) -> FixedMatrix { g1 * g2 }
    fn group_inverse(&self, g: &FixedMatrix) -> Result<FixedMatrix, OverflowDetected> { inverse(g) }
    fn lie_exp(&self, xi: &FixedVector) -> Result<FixedMatrix, OverflowDetected> { matrix_exp(&self.hat_sln(xi)) }

    fn lie_log(&self, g: &FixedMatrix) -> Result<FixedVector, OverflowDetected> {
        Ok(self.vee_sln(&Self::project_traceless(&matrix_log(g)?)))
    }

    fn hat(&self, xi: &FixedVector) -> FixedMatrix { self.hat_sln(xi) }
    fn vee(&self, xi_hat: &FixedMatrix) -> FixedVector { self.vee_sln(xi_hat) }

    fn adjoint(&self, g: &FixedMatrix, xi: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let xi_hat = self.hat_sln(xi);
        let g_inv = inverse(g)?;
        Ok(self.vee_sln(&Self::project_traceless(&(&(g * &xi_hat) * &g_inv))))
    }

    fn bracket(&self, xi: &FixedVector, eta: &FixedVector) -> FixedVector {
        let a = self.hat_sln(xi);
        let b = self.hat_sln(eta);
        self.vee_sln(&(&(&a * &b) - &(&b * &a)))
    }

    fn act(&self, g: &FixedMatrix, point: &FixedVector) -> FixedVector { g.mul_vector(point) }
}
