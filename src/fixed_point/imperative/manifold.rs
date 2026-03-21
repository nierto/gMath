//! Riemannian manifold trait and concrete implementations.
//!
//! L3A — Manifolds with closed-form geodesics (no ODE solver required):
//! - `EuclideanSpace` — R^n, flat
//! - `Sphere` — S^n embedded in R^{n+1}
//! - `HyperbolicSpace` — H^n in the hyperboloid model
//!
//! L3C — Manifolds requiring matrix function infrastructure:
//! - `SPDManifold` — Sym⁺(n), symmetric positive-definite matrices
//! - `Grassmannian` — Gr(k,n), k-dimensional subspaces of R^n

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::compute_matrix::ComputeMatrix;
use super::linalg::compute_tier_dot_raw;
use super::matrix_functions::{matrix_exp, matrix_log, matrix_sqrt};
use super::derived::{inverse_spd, frobenius_norm};
use super::decompose::{svd_decompose, qr_decompose};
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::core_types::errors::OverflowDetected;

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
    /// Clamp a value to [-1, 1] for safe acos input.
    fn clamp_unit(x: FixedPoint) -> FixedPoint {
        let one = FixedPoint::one();
        if x > one { one }
        else if x < -one { -one }
        else { x }
    }
}

impl Manifold for Sphere {
    fn dimension(&self) -> usize { self.dim }

    fn inner_product(&self, _base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint {
        u.dot_precise(v)
    }

    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let theta = tangent.length();
        if theta.is_zero() {
            return Ok(base.clone());
        }
        // exp_p(v) = cos(θ)*p + sin(θ)*(v/θ)
        let cos_t = theta.try_cos()?;
        let sin_t = theta.try_sin()?;
        let direction = tangent * (FixedPoint::one() / theta);
        Ok(base * cos_t + direction * sin_t)
    }

    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let cos_theta = Self::clamp_unit(base.dot_precise(target));
        let theta = cos_theta.try_acos()?;
        if theta.is_zero() {
            return Ok(FixedVector::new(base.len()));
        }
        // direction = target - cos_theta * base (project out the base component)
        let direction = target - &(base * cos_theta);
        let dir_len = direction.length();
        if dir_len.is_zero() {
            return Ok(FixedVector::new(base.len()));
        }
        Ok(&direction * (theta / dir_len))
    }

    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        let cos_theta = Self::clamp_unit(p.dot_precise(q));
        // asin/acos boundary cases (x = ±1) now handled in FASC evaluate_asin
        cos_theta.try_acos()
    }

    fn parallel_transport(&self, base: &FixedVector, target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let cos_theta = Self::clamp_unit(base.dot_precise(target));
        let one = FixedPoint::one();
        let denom = one + cos_theta;
        if denom.is_zero() {
            // Antipodal points — transport is ambiguous
            return Err(OverflowDetected::DomainError);
        }
        // PT(v) = v - <v, p+q>/(1 + <p,q>) * (p + q)
        let p_plus_q = base + target;
        let coeff = tangent.dot_precise(&p_plus_q) / denom;
        Ok(tangent - &(&p_plus_q * coeff))
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
        assert_eq!(u.len(), v.len());
        let n = u.len();
        if n == 0 { return FixedPoint::ZERO; }
        // Compute the spatial part (indices 1..n) at compute tier
        let spatial = if n > 1 {
            let u_raw: Vec<BinaryStorage> = (1..n).map(|i| u[i].raw()).collect();
            let v_raw: Vec<BinaryStorage> = (1..n).map(|i| v[i].raw()).collect();
            FixedPoint::from_raw(compute_tier_dot_raw(&u_raw, &v_raw))
        } else {
            FixedPoint::ZERO
        };
        // Minkowski: -temporal + spatial
        spatial - u[0] * v[0]
    }

    /// Minkowski norm: sqrt(<v,v>_M) for spacelike vectors (tangent vectors).
    /// For tangent vectors on H^n, <v,v>_M >= 0.
    fn minkowski_norm(v: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        let dot = Self::minkowski_dot(v, v);
        if dot.is_negative() {
            // Timelike — shouldn't happen for tangent vectors, but handle gracefully
            return Ok((-dot).sqrt());
        }
        dot.try_sqrt()
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

    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let theta = Self::minkowski_norm(tangent)?;
        if theta.is_zero() {
            return Ok(base.clone());
        }
        // exp_p(v) = cosh(θ)*p + sinh(θ)*(v/θ)
        let cosh_t = theta.try_cosh()?;
        let sinh_t = theta.try_sinh()?;
        let direction = tangent * (FixedPoint::one() / theta);
        Ok(base * cosh_t + direction * sinh_t)
    }

    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        // <p, q>_M for points on H^n: always <= -1
        let minus_alpha = Self::minkowski_dot(base, target); // should be <= -1
        let alpha = -minus_alpha; // >= 1
        // Clamp for safety
        let alpha_clamped = if alpha < FixedPoint::one() { FixedPoint::one() } else { alpha };
        let theta = alpha_clamped.try_acosh()?;
        if theta.is_zero() {
            return Ok(FixedVector::new(base.len()));
        }
        // direction = target - alpha * base (project onto tangent space)
        let direction = target - &(base * (-minus_alpha));
        let dir_norm = Self::minkowski_norm(&direction)?;
        if dir_norm.is_zero() {
            return Ok(FixedVector::new(base.len()));
        }
        Ok(&direction * (theta / dir_norm))
    }

    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, OverflowDetected> {
        let minus_alpha = Self::minkowski_dot(p, q);
        let alpha = -minus_alpha;
        let alpha_clamped = if alpha < FixedPoint::one() { FixedPoint::one() } else { alpha };
        alpha_clamped.try_acosh()
    }

    fn parallel_transport(&self, base: &FixedVector, target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, OverflowDetected> {
        let log_pq = self.log_map(base, target)?;
        let theta = Self::minkowski_norm(&log_pq)?;
        if theta.is_zero() {
            return Ok(tangent.clone());
        }
        // u = log_pq / theta (unit tangent direction)
        let u = &log_pq * (FixedPoint::one() / theta);
        // Coefficient: <v, u>_M
        let vu = Self::minkowski_dot(tangent, &u);
        // PT(v) = v + vu * (sinh(θ)*p + (cosh(θ)-1)*u)
        let sinh_t = theta.try_sinh()?;
        let cosh_t = theta.try_cosh()?;
        let correction = &(base * sinh_t) + &(&u * (cosh_t - FixedPoint::one()));
        Ok(tangent + &(&correction * vu))
    }
}

// ============================================================================
// L3C: SPD manifold Sym⁺(n) — symmetric positive-definite matrices
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
/// **FASC-UGOD integration:** Every operation chains through matrix_sqrt,
/// matrix_exp, matrix_log which internally use ComputeMatrix at tier N+1.
/// The entire exp_map/log_map is a ~7-operation ComputeMatrix chain with
/// a single downscale at the end.
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
    /// Compute P^½ and P^{-½} for an SPD matrix P.
    ///
    /// Returns (sqrt_p, inv_sqrt_p).
    fn sqrt_and_inv_sqrt(p: &FixedMatrix) -> Result<(FixedMatrix, FixedMatrix), OverflowDetected> {
        let sqrt_p = matrix_sqrt(p)?;
        let inv_sqrt_p = inverse_spd(&sqrt_p)?;
        Ok((sqrt_p, inv_sqrt_p))
    }
}

impl Manifold for SPDManifold {
    fn dimension(&self) -> usize {
        // Intrinsic dimension of Sym⁺(n) = n*(n+1)/2
        self.n * (self.n + 1) / 2
    }

    fn inner_product(
        &self,
        base: &FixedVector,
        u: &FixedVector,
        v: &FixedVector,
    ) -> FixedPoint {
        // <U, V>_P = tr(P⁻¹ U P⁻¹ V) — compute-tier chain, single downscale at trace
        let p = vec_to_sym(base, self.n);
        let u_mat = vec_to_sym(u, self.n);
        let v_mat = vec_to_sym(v, self.n);
        let p_inv = inverse_spd(&p).unwrap_or_else(|_| FixedMatrix::identity(self.n));
        let p_inv_c = ComputeMatrix::from_fixed_matrix(&p_inv);
        let u_c = ComputeMatrix::from_fixed_matrix(&u_mat);
        let v_c = ComputeMatrix::from_fixed_matrix(&v_mat);
        // P⁻¹ U P⁻¹ V at compute tier
        let p_inv_u_c = p_inv_c.mat_mul(&u_c);
        let p_inv_v_c = p_inv_c.mat_mul(&v_c);
        let product_c = p_inv_u_c.mat_mul(&p_inv_v_c);
        product_c.trace_compute()
    }

    fn exp_map(
        &self,
        base: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let p = vec_to_sym(base, self.n);
        let v = vec_to_sym(tangent, self.n);

        // exp_P(V) = P^½ expm(P^{-½} V P^{-½}) P^½
        let (sqrt_p, inv_sqrt_p) = Self::sqrt_and_inv_sqrt(&p)?;
        let inner = &(&inv_sqrt_p * &v) * &inv_sqrt_p;
        let exp_inner = matrix_exp(&inner)?;
        let result = &(&sqrt_p * &exp_inner) * &sqrt_p;

        Ok(sym_to_vec(&result))
    }

    fn log_map(
        &self,
        base: &FixedVector,
        target: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let p = vec_to_sym(base, self.n);
        let q = vec_to_sym(target, self.n);

        // log_P(Q) = P^½ logm(P^{-½} Q P^{-½}) P^½
        let (sqrt_p, inv_sqrt_p) = Self::sqrt_and_inv_sqrt(&p)?;
        let inner = &(&inv_sqrt_p * &q) * &inv_sqrt_p;
        let log_inner = matrix_log(&inner)?;
        let result = &(&sqrt_p * &log_inner) * &sqrt_p;

        Ok(sym_to_vec(&result))
    }

    fn distance(
        &self,
        p: &FixedVector,
        q: &FixedVector,
    ) -> Result<FixedPoint, OverflowDetected> {
        let log_v = self.log_map(p, q)?;
        let p_mat = vec_to_sym(p, self.n);
        let v_mat = vec_to_sym(&log_v, self.n);

        // ||V||_P = sqrt(tr(P⁻¹ V P⁻¹ V)) — compute-tier chain, single downscale at trace
        let p_inv = inverse_spd(&p_mat)?;
        let p_inv_c = ComputeMatrix::from_fixed_matrix(&p_inv);
        let v_c = ComputeMatrix::from_fixed_matrix(&v_mat);
        let p_inv_v_c = p_inv_c.mat_mul(&v_c);
        let product_c = p_inv_v_c.mat_mul(&p_inv_v_c);
        product_c.trace_compute().try_sqrt()
    }

    fn parallel_transport(
        &self,
        base: &FixedVector,
        target: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        // Parallel transport on SPD: Γ(P,Q) V = E V Eᵀ
        // where E = (QP⁻¹)^½
        let p = vec_to_sym(base, self.n);
        let q = vec_to_sym(target, self.n);
        let v = vec_to_sym(tangent, self.n);

        let p_inv = inverse_spd(&p)?;
        let qp_inv = &q * &p_inv;
        let e = matrix_sqrt(&qp_inv)?;
        // E V Eᵀ at compute tier — single downscale at end
        let e_c = ComputeMatrix::from_fixed_matrix(&e);
        let v_c = ComputeMatrix::from_fixed_matrix(&v);
        let et_c = e_c.transpose();
        let result = e_c.mat_mul(&v_c).mat_mul(&et_c).to_fixed_matrix();

        Ok(sym_to_vec(&result))
    }
}

// ============================================================================
// L3C: Grassmannian Gr(k, n) — k-dimensional subspaces of R^n
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
/// **FASC-UGOD integration:** Uses SVD from L1B with compute-tier dot products.
/// The thin SVD on n×k matrices (k << n) is O(nk²), much faster than full O(n³).
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
    /// Compute-tier chain — single downscale at end.
    #[allow(dead_code)]
    fn project_tangent(q: &FixedMatrix, delta: &FixedMatrix) -> FixedMatrix {
        let q_c = ComputeMatrix::from_fixed_matrix(q);
        let delta_c = ComputeMatrix::from_fixed_matrix(delta);
        let qt_delta_c = q_c.transpose().mat_mul(&delta_c);
        delta_c.sub(&q_c.mat_mul(&qt_delta_c)).to_fixed_matrix()
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
        // Canonical metric: <U, V> = tr(UᵀV) — compute-tier chain, single downscale at trace
        let u_mat = Self::vec_to_mat(u, self.n, self.k);
        let v_mat = Self::vec_to_mat(v, self.n, self.k);
        let u_c = ComputeMatrix::from_fixed_matrix(&u_mat);
        let v_c = ComputeMatrix::from_fixed_matrix(&v_mat);
        u_c.transpose().mat_mul(&v_c).trace_compute()
    }

    fn exp_map(
        &self,
        base: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let q = Self::vec_to_mat(base, self.n, self.k);
        let delta = Self::vec_to_mat(tangent, self.n, self.k);

        // Thin SVD of tangent: Δ = U Σ Vᵀ (U is n×k, Σ k×k diagonal, Vᵀ k×k)
        let svd = svd_decompose(&delta)?;

        // exp_Q(Δ) = Q V cos(Σ) Vᵀ + U sin(Σ) Vᵀ
        // where cos(Σ) and sin(Σ) are diagonal matrices with cos/sin of singular values
        let kk = svd.sigma.len().min(self.k);

        // Build cos(Σ) and sin(Σ) as k×k diagonal matrices
        let mut cos_sigma = FixedMatrix::new(kk, kk);
        let mut sin_sigma = FixedMatrix::new(kk, kk);
        for i in 0..kk {
            cos_sigma.set(i, i, svd.sigma[i].try_cos()?);
            sin_sigma.set(i, i, svd.sigma[i].try_sin()?);
        }

        // Extract U_thin (n×k from full U) and Vt_thin (k×k from full Vt)
        // Full U is n×n (for m=n) or m×m — we need columns 0..k
        let u_thin = FixedMatrix::from_fn(self.n, kk, |r, c| {
            if r < svd.u.rows() && c < svd.u.cols() { svd.u.get(r, c) } else { FixedPoint::ZERO }
        });
        let vt_thin = FixedMatrix::from_fn(kk, kk, |r, c| {
            if r < svd.vt.rows() && c < svd.vt.cols() { svd.vt.get(r, c) } else { FixedPoint::ZERO }
        });

        // Q_new = Q * V * cos(Σ) * Vᵀ + U * sin(Σ) * Vᵀ — compute-tier chain
        let v_thin = vt_thin.transpose();
        let q_c = ComputeMatrix::from_fixed_matrix(&q);
        let v_thin_c = ComputeMatrix::from_fixed_matrix(&v_thin);
        let cos_sigma_c = ComputeMatrix::from_fixed_matrix(&cos_sigma);
        let sin_sigma_c = ComputeMatrix::from_fixed_matrix(&sin_sigma);
        let vt_thin_c = ComputeMatrix::from_fixed_matrix(&vt_thin);
        let u_thin_c = ComputeMatrix::from_fixed_matrix(&u_thin);
        let term1_c = q_c.mat_mul(&v_thin_c).mat_mul(&cos_sigma_c).mat_mul(&vt_thin_c);
        let term2_c = u_thin_c.mat_mul(&sin_sigma_c).mat_mul(&vt_thin_c);
        let result = term1_c.add(&term2_c).to_fixed_matrix();

        Ok(Self::mat_to_vec(&result))
    }

    fn log_map(
        &self,
        base: &FixedVector,
        target: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let q1 = Self::vec_to_mat(base, self.n, self.k);
        let q2 = Self::vec_to_mat(target, self.n, self.k);

        // Compute (I - Q₁Q₁ᵀ)Q₂ at compute tier: the component of Q₂ orthogonal to Q₁
        let q1_c = ComputeMatrix::from_fixed_matrix(&q1);
        let q2_c = ComputeMatrix::from_fixed_matrix(&q2);
        let qt_q2_c = q1_c.transpose().mat_mul(&q2_c); // k×k at compute tier
        let proj_c = q1_c.mat_mul(&qt_q2_c);            // n×k at compute tier
        let perp_c = q2_c.sub(&proj_c);                 // n×k at compute tier
        let qt_q2 = qt_q2_c.to_fixed_matrix();          // downscale for SVD input
        let perp = perp_c.to_fixed_matrix();             // downscale for SVD input

        // Thin SVD of perp: perp = U Σ Vᵀ
        let svd = svd_decompose(&perp)?;
        let kk = svd.sigma.len().min(self.k);

        // Principal angles: θ_i = atan2(σ_i, (QᵀQ₂ singular values))
        // For well-separated subspaces, θ_i = atan(σ_i / c_i) where c_i are
        // the singular values of QᵀQ₂.
        // Simpler approach: θ_i = atan2(σ_i_perp, σ_i_parallel)
        let svd_parallel = svd_decompose(&qt_q2)?;
        let kk_par = svd_parallel.sigma.len().min(self.k);

        // Build Θ diagonal with atan2(perp_sigma, parallel_sigma)
        let mut theta_diag = FixedMatrix::new(kk, kk);
        for i in 0..kk {
            let s_perp = svd.sigma[i];
            let s_par = if i < kk_par { svd_parallel.sigma[i] } else { FixedPoint::ZERO };
            let theta_i = s_perp.try_atan2(s_par)?;
            theta_diag.set(i, i, theta_i);
        }

        // U_thin from SVD of perp
        let u_thin = FixedMatrix::from_fn(self.n, kk, |r, c| {
            if r < svd.u.rows() && c < svd.u.cols() { svd.u.get(r, c) } else { FixedPoint::ZERO }
        });
        let vt_thin = FixedMatrix::from_fn(kk, kk, |r, c| {
            if r < svd.vt.rows() && c < svd.vt.cols() { svd.vt.get(r, c) } else { FixedPoint::ZERO }
        });

        // Tangent vector: Δ = U Θ Vᵀ — compute-tier chain, single downscale at end
        let u_thin_c = ComputeMatrix::from_fixed_matrix(&u_thin);
        let theta_diag_c = ComputeMatrix::from_fixed_matrix(&theta_diag);
        let vt_thin_c = ComputeMatrix::from_fixed_matrix(&vt_thin);
        let result = u_thin_c.mat_mul(&theta_diag_c).mat_mul(&vt_thin_c).to_fixed_matrix();

        Ok(Self::mat_to_vec(&result))
    }

    fn distance(
        &self,
        p: &FixedVector,
        q: &FixedVector,
    ) -> Result<FixedPoint, OverflowDetected> {
        let q1 = Self::vec_to_mat(p, self.n, self.k);
        let q2 = Self::vec_to_mat(q, self.n, self.k);

        // Distance² = sum of θ_i² (principal angles)
        // Principal angles from SVD of Q₁ᵀQ₂: σ_i = cos(θ_i)
        // Q₁ᵀQ₂ at compute tier, single downscale for SVD input
        let q1_c = ComputeMatrix::from_fixed_matrix(&q1);
        let q2_c = ComputeMatrix::from_fixed_matrix(&q2);
        let qt_q = q1_c.transpose().mat_mul(&q2_c).to_fixed_matrix();
        let svd = svd_decompose(&qt_q)?;
        let kk = svd.sigma.len().min(self.k);

        let one = FixedPoint::one();
        // Collect θ_i values, then use compute_tier_dot_raw for θ·θ accumulation
        let mut thetas: Vec<BinaryStorage> = Vec::with_capacity(kk);
        for i in 0..kk {
            // Clamp to [-1, 1] for acos safety
            let s = if svd.sigma[i] > one { one }
                    else if svd.sigma[i] < -one { -one }
                    else { svd.sigma[i] };
            let theta = s.try_acos()?;
            thetas.push(theta.raw());
        }
        // dist² = θ·θ at compute tier
        let dist_sq = FixedPoint::from_raw(compute_tier_dot_raw(&thetas, &thetas));
        dist_sq.try_sqrt()
    }

    fn parallel_transport(
        &self,
        base: &FixedVector,
        target: &FixedVector,
        tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        // Parallel transport on Grassmannian via the geodesic connection:
        // Move along the geodesic from Q₁ to Q₂, transport Δ.
        //
        // Using the formula:
        //   PT(Δ) = -Q₁ Vᵀ sin(Σ) Uᵀ Δ + U cos(Σ) Uᵀ Δ + (I - UUᵀ) Δ
        // where U Σ Vᵀ = thin SVD of the tangent from Q₁ to Q₂.
        let log_v = self.log_map(base, target)?;
        let q1 = Self::vec_to_mat(base, self.n, self.k);
        let delta_mat = Self::vec_to_mat(tangent, self.n, self.k);
        let tangent_mat = Self::vec_to_mat(&log_v, self.n, self.k);

        let svd = svd_decompose(&tangent_mat)?;
        let kk = svd.sigma.len().min(self.k);

        let u_thin = FixedMatrix::from_fn(self.n, kk, |r, c| {
            if r < svd.u.rows() && c < svd.u.cols() { svd.u.get(r, c) } else { FixedPoint::ZERO }
        });
        let vt_thin = FixedMatrix::from_fn(kk, kk, |r, c| {
            if r < svd.vt.rows() && c < svd.vt.cols() { svd.vt.get(r, c) } else { FixedPoint::ZERO }
        });

        let mut cos_sigma = FixedMatrix::new(kk, kk);
        let mut sin_sigma = FixedMatrix::new(kk, kk);
        for i in 0..kk {
            cos_sigma.set(i, i, svd.sigma[i].try_cos()?);
            sin_sigma.set(i, i, svd.sigma[i].try_sin()?);
        }

        // Entire transport computation at compute tier — single downscale at end
        let q1_c = ComputeMatrix::from_fixed_matrix(&q1);
        let u_thin_c = ComputeMatrix::from_fixed_matrix(&u_thin);
        let vt_thin_c = ComputeMatrix::from_fixed_matrix(&vt_thin);
        let cos_sigma_c = ComputeMatrix::from_fixed_matrix(&cos_sigma);
        let sin_sigma_c = ComputeMatrix::from_fixed_matrix(&sin_sigma);
        let delta_c = ComputeMatrix::from_fixed_matrix(&delta_mat);

        // Uᵀ Δ (k×k) at compute tier
        let ut_delta_c = u_thin_c.transpose().mat_mul(&delta_c);
        // Term 1: -Q₁ V sin(Σ) Uᵀ Δ
        let v_thin_c = vt_thin_c.transpose();
        let term1_c = q1_c.mat_mul(&v_thin_c).mat_mul(&sin_sigma_c).mat_mul(&ut_delta_c).neg();
        // Term 2: U cos(Σ) Uᵀ Δ
        let term2_c = u_thin_c.mat_mul(&cos_sigma_c).mat_mul(&ut_delta_c);
        // Term 3: (I - UUᵀ) Δ = Δ - U(UᵀΔ)
        let term3_c = delta_c.sub(&u_thin_c.mat_mul(&ut_delta_c));

        let result = term1_c.add(&term2_c).add(&term3_c).to_fixed_matrix();
        Ok(Self::mat_to_vec(&result))
    }
}

// ============================================================================
// L3C: Stiefel manifold St(k, n) — orthonormal k-frames in R^n
// ============================================================================

/// The Stiefel manifold St(k, n): orthonormal k-frames in R^n.
///
/// Points are n×k matrices Q with QᵀQ = I_k (orthonormal columns).
/// Unlike Grassmannian, two points Q₁ ≠ Q₂ even if they span the same subspace.
///
/// **Points stored as:** flattened n*k FixedVector (column-major).
///
/// **Geodesics via QR retraction:**
///   exp_Q(Δ) ≈ qr(Q + Δ).Q — the Q factor of QR decomposition.
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
    /// Compute-tier chain — single downscale at end.
    fn project_tangent(q: &FixedMatrix, delta: &FixedMatrix) -> FixedMatrix {
        let q_c = ComputeMatrix::from_fixed_matrix(q);
        let delta_c = ComputeMatrix::from_fixed_matrix(delta);
        let qt_delta_c = q_c.transpose().mat_mul(&delta_c); // k×k at compute tier
        // sym(QᵀΔ) = (QᵀΔ + ΔᵀQ) / 2 at compute tier
        let sym_c = qt_delta_c.add(&qt_delta_c.transpose()).halve();
        // Δ - Q · sym at compute tier, single downscale
        delta_c.sub(&q_c.mat_mul(&sym_c)).to_fixed_matrix()
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
        // Canonical metric: <U, V> = tr(UᵀV) — compute-tier chain, single downscale at trace
        let u_mat = stiefel_vec_to_mat(u, self.n, self.k);
        let v_mat = stiefel_vec_to_mat(v, self.n, self.k);
        let u_c = ComputeMatrix::from_fixed_matrix(&u_mat);
        let v_c = ComputeMatrix::from_fixed_matrix(&v_mat);
        u_c.transpose().mat_mul(&v_c).trace_compute()
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
        // qr.q is n×n — we need the first k columns (thin Q)
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

    fn log_map(
        &self,
        base: &FixedVector,
        target: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let q = stiefel_vec_to_mat(base, self.n, self.k);
        let q_target = stiefel_vec_to_mat(target, self.n, self.k);

        // First-order approximation: Δ = Q' - Q, projected to tangent space
        let diff = &q_target - &q;
        let tangent = Self::project_tangent(&q, &diff);

        Ok(stiefel_mat_to_vec(&tangent))
    }

    fn distance(
        &self,
        p: &FixedVector,
        q: &FixedVector,
    ) -> Result<FixedPoint, OverflowDetected> {
        let log_v = self.log_map(p, q)?;
        let log_mat = stiefel_vec_to_mat(&log_v, self.n, self.k);
        // ||Δ|| = sqrt(tr(ΔᵀΔ)) = Frobenius norm
        frobenius_norm(&log_mat).try_sqrt()
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
/// Tier handling is inherited — each component uses its own compute-tier
/// operations internally. The product structure adds no additional precision loss.
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
        self.m1.inner_product(&b1, &u1, &v1) + self.m2.inner_product(&b2, &u2, &v2)
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
        // Product distance: sqrt(d₁² + d₂²)
        (d1 * d1 + d2 * d2).try_sqrt()
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
