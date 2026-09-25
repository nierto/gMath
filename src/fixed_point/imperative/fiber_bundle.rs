//! L5A: Fiber bundle types and operations with fixed-point arithmetic.
//!
//! Provides:
//! - `FiberBundle` trait: project, fiber_at, lift, local_trivialization
//! - `TrivialBundle`: M × F with flat connection (test baseline)
//! - `VectorBundle`: fiber = R^k, sections are vector fields
//! - `PrincipalBundle`: fiber = Lie group, transition functions, right action
//! - `Connection` trait: horizontal_lift, parallel transport along base curve
//! - `bundle_curvature`: Ω = dω + ω∧ω (structure equation)
//!
//! **FASC-UGOD integration:** All operations delegate to the underlying manifold
//! (L3A) and Lie group (L4A) infrastructure. Parallel transport on bundles reduces
//! to an ODE solved by L2A integrators. Connection curvature uses numerical
//! differentiation from L3B with compute-tier accumulation.

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::linalg::{exact_dot, upscale_to_compute, downscale_to_storage, ComputeStorage};
use super::compute_matrix::{ComputeMatrix, compute_lu_decompose};
use super::wide_acc::{acc, exact_sub_dot_compute, narrow_triple_nearest, widen_product, widen_storage, Wide};
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_checked_add, compute_multiply, compute_subtract, make_compute_int,
};

// ============================================================================
// FiberBundle trait
// ============================================================================

/// A fiber bundle π: E → B with fiber F.
///
/// Total space E, base space B, fiber F. The projection π maps each point
/// in E to a point in B, and each fiber π⁻¹(b) is diffeomorphic to F.
pub trait FiberBundle {
    /// Project from total space to base space: π(e) → b.
    fn project(&self, total_point: &FixedVector) -> FixedVector;

    /// Dimension of the base space.
    fn base_dim(&self) -> usize;

    /// Dimension of the fiber.
    fn fiber_dim(&self) -> usize;

    /// Dimension of the total space (= base_dim + fiber_dim).
    fn total_dim(&self) -> usize {
        self.base_dim() + self.fiber_dim()
    }

    /// Lift: given a base point b and fiber element f, construct a total space point.
    fn lift(&self, base: &FixedVector, fiber: &FixedVector) -> FixedVector;

    /// Local trivialization: decompose a total space point into (base, fiber).
    fn local_trivialization(&self, total_point: &FixedVector) -> (FixedVector, FixedVector);
}

// ============================================================================
// Connection trait
// ============================================================================

/// A connection on a fiber bundle: specifies how fibers relate along the base.
///
/// A connection provides a notion of "horizontal" in the tangent bundle of E,
/// enabling parallel transport along curves in the base.
pub trait BundleConnection: FiberBundle {
    /// Horizontal lift: given a tangent vector v at base point b,
    /// produce the horizontal tangent vector at total point e above b.
    ///
    /// The horizontal lift is the unique vector in T_e(E) that:
    /// 1. Projects down to v under dπ
    /// 2. Lies in the horizontal subspace H_e
    fn horizontal_lift(
        &self,
        total_point: &FixedVector,
        base_tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected>;

    /// Vertical component of a tangent vector at a total space point.
    ///
    /// V_e = ker(dπ): the tangent vectors that project to zero in the base.
    fn vertical_component(
        &self,
        total_point: &FixedVector,
        total_tangent: &FixedVector,
    ) -> FixedVector;

    /// Horizontal component of a tangent vector at a total space point.
    fn horizontal_component(
        &self,
        total_point: &FixedVector,
        total_tangent: &FixedVector,
    ) -> FixedVector {
        let vert = self.vertical_component(total_point, total_tangent);
        total_tangent - &vert
    }

    /// Parallel transport a fiber element along a base curve.
    ///
    /// Given a discrete path in the base space (sequence of points), transport
    /// the fiber element from the first base point to the last.
    ///
    /// This is the discrete approximation to solving the ODE:
    ///   dξ/dt = -ω(γ'(t)) ξ
    /// where ω is the connection form and γ is the base curve.
    fn parallel_transport_along(
        &self,
        base_path: &[FixedVector],
        initial_fiber: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected>;
}

// ============================================================================
// TrivialBundle: M × F with flat connection
// ============================================================================

/// A trivial fiber bundle E = B × F with the flat (product) connection.
///
/// This is the simplest bundle: no twisting, parallel transport is trivial
/// (the fiber element doesn't change). Serves as a test baseline.
pub struct TrivialBundle {
    pub base_dimension: usize,
    pub fiber_dimension: usize,
}

impl FiberBundle for TrivialBundle {
    fn project(&self, total_point: &FixedVector) -> FixedVector {
        let mut base = FixedVector::new(self.base_dimension);
        for i in 0..self.base_dimension {
            base[i] = total_point[i];
        }
        base
    }

    fn base_dim(&self) -> usize { self.base_dimension }
    fn fiber_dim(&self) -> usize { self.fiber_dimension }

    fn lift(&self, base: &FixedVector, fiber: &FixedVector) -> FixedVector {
        let mut total = FixedVector::new(self.base_dimension + self.fiber_dimension);
        for i in 0..self.base_dimension {
            total[i] = base[i];
        }
        for i in 0..self.fiber_dimension {
            total[self.base_dimension + i] = fiber[i];
        }
        total
    }

    fn local_trivialization(&self, total_point: &FixedVector) -> (FixedVector, FixedVector) {
        let mut base = FixedVector::new(self.base_dimension);
        let mut fiber = FixedVector::new(self.fiber_dimension);
        for i in 0..self.base_dimension {
            base[i] = total_point[i];
        }
        for i in 0..self.fiber_dimension {
            fiber[i] = total_point[self.base_dimension + i];
        }
        (base, fiber)
    }
}

impl BundleConnection for TrivialBundle {
    fn horizontal_lift(
        &self,
        _total_point: &FixedVector,
        base_tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        // Flat connection: horizontal lift = (v, 0)
        let mut total_tangent = FixedVector::new(self.total_dim());
        for i in 0..self.base_dimension {
            total_tangent[i] = base_tangent[i];
        }
        Ok(total_tangent)
    }

    fn vertical_component(
        &self,
        _total_point: &FixedVector,
        total_tangent: &FixedVector,
    ) -> FixedVector {
        // Vertical = fiber components (last fiber_dim entries)
        let mut vert = FixedVector::new(self.total_dim());
        for i in 0..self.fiber_dimension {
            vert[self.base_dimension + i] = total_tangent[self.base_dimension + i];
        }
        vert
    }

    fn parallel_transport_along(
        &self,
        _base_path: &[FixedVector],
        initial_fiber: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        // Flat connection: fiber element doesn't change
        Ok(initial_fiber.clone())
    }
}

// ============================================================================
// VectorBundle: fiber = R^k
// ============================================================================

/// A vector bundle with fiber R^k over a base manifold of dimension n.
///
/// The total space has dimension n + k. Sections are vector fields on the base.
/// The connection is specified by connection coefficients (Christoffel-like).
///
/// **Connection coefficients:** A^a_{bi} where a,b are fiber indices (0..k)
/// and i is a base index (0..n). The parallel transport equation is:
///   ∇_i ξ^a = ∂_i ξ^a + A^a_{bi} ξ^b
pub struct VectorBundle {
    pub base_dim_val: usize,
    pub fiber_dim_val: usize,
    /// Connection coefficients A^a_{bi} as a rank-3 tensor [k, k, n].
    /// Entry [a, b, i] = A^a_{bi}.
    /// If None, the connection is flat (A = 0).
    pub connection_coeffs: Option<Vec<FixedPoint>>,
}

impl VectorBundle {
    /// Create a vector bundle with flat connection.
    pub fn flat(base_dim: usize, fiber_dim: usize) -> Self {
        Self {
            base_dim_val: base_dim,
            fiber_dim_val: fiber_dim,
            connection_coeffs: None,
        }
    }

    /// Create a vector bundle with given connection coefficients.
    ///
    /// `coeffs` is a flat array of size k * k * n in row-major order [a, b, i].
    pub fn with_connection(base_dim: usize, fiber_dim: usize, coeffs: Vec<FixedPoint>) -> Self {
        assert_eq!(coeffs.len(), fiber_dim * fiber_dim * base_dim,
            "Connection coefficients must have size fiber_dim² × base_dim");
        Self {
            base_dim_val: base_dim,
            fiber_dim_val: fiber_dim,
            connection_coeffs: Some(coeffs),
        }
    }

    /// Get connection coefficient A^a_{bi}.
    fn get_coeff(&self, a: usize, b: usize, i: usize) -> FixedPoint {
        match &self.connection_coeffs {
            None => FixedPoint::ZERO,
            Some(coeffs) => {
                let k = self.fiber_dim_val;
                let n = self.base_dim_val;
                coeffs[a * k * n + b * n + i]
            }
        }
    }

    /// The raws A^a_{b0..n} of one (a, b) pair.
    fn coeff_row(&self, a: usize, b: usize) -> Vec<BinaryStorage> {
        (0..self.base_dim_val).map(|i| self.get_coeff(a, b, i).raw()).collect()
    }
}

impl FiberBundle for VectorBundle {
    fn project(&self, total_point: &FixedVector) -> FixedVector {
        let mut base = FixedVector::new(self.base_dim_val);
        for i in 0..self.base_dim_val {
            base[i] = total_point[i];
        }
        base
    }

    fn base_dim(&self) -> usize { self.base_dim_val }
    fn fiber_dim(&self) -> usize { self.fiber_dim_val }

    fn lift(&self, base: &FixedVector, fiber: &FixedVector) -> FixedVector {
        let mut total = FixedVector::new(self.base_dim_val + self.fiber_dim_val);
        for i in 0..self.base_dim_val {
            total[i] = base[i];
        }
        for i in 0..self.fiber_dim_val {
            total[self.base_dim_val + i] = fiber[i];
        }
        total
    }

    fn local_trivialization(&self, total_point: &FixedVector) -> (FixedVector, FixedVector) {
        let mut base = FixedVector::new(self.base_dim_val);
        let mut fiber = FixedVector::new(self.fiber_dim_val);
        for i in 0..self.base_dim_val {
            base[i] = total_point[i];
        }
        for i in 0..self.fiber_dim_val {
            fiber[i] = total_point[self.base_dim_val + i];
        }
        (base, fiber)
    }
}

impl BundleConnection for VectorBundle {
    fn horizontal_lift(
        &self,
        total_point: &FixedVector,
        base_tangent: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        let k = self.fiber_dim_val;
        let n = self.base_dim_val;

        // fiber part of total_point
        let fiber: Vec<FixedPoint> = (0..k)
            .map(|a| total_point[n + a])
            .collect();

        let mut total_tangent = FixedVector::new(n + k);
        // Base part: just the base tangent
        for i in 0..n {
            total_tangent[i] = base_tangent[i];
        }
        // Fiber part: -A^a_{bi} ξ^b v^i (negative connection term). The inner
        // sums over i are exact at the compute tier, each times ξ^b exact at
        // 3F, the whole sum negated and rounded once (before 0.6.4 every
        // triple product was rounded to storage twice before the sum).
        let v_raw: Vec<BinaryStorage> = (0..n).map(|i| base_tangent[i].raw()).collect();
        for a in 0..k {
            let mut sum = <acc::Orient as Wide>::zero();
            for b in 0..k {
                let a_dot_v = exact_dot(&self.coeff_row(a, b), &v_raw)?;
                sum = sum.add_exact(widen_product(a_dot_v, widen_storage(fiber[b].raw())))?;
            }
            total_tangent[n + a] = FixedPoint::from_raw(narrow_triple_nearest(-sum)?);
        }
        Ok(total_tangent)
    }

    fn vertical_component(
        &self,
        _total_point: &FixedVector,
        total_tangent: &FixedVector,
    ) -> FixedVector {
        let n = self.base_dim_val;
        let k = self.fiber_dim_val;
        let mut vert = FixedVector::new(n + k);
        for a in 0..k {
            vert[n + a] = total_tangent[n + a];
        }
        vert
    }

    fn parallel_transport_along(
        &self,
        base_path: &[FixedVector],
        initial_fiber: &FixedVector,
    ) -> Result<FixedVector, OverflowDetected> {
        if base_path.len() < 2 {
            return Ok(initial_fiber.clone());
        }

        let k = self.fiber_dim_val;
        let n = self.base_dim_val;
        let up = |x: FixedPoint| upscale_to_compute(x.raw());

        // Discrete parallel transport: at each step
        //   ξ^a_{n+1} = ξ^a_n - A^a_{bi} ξ^b_n Δx^i
        // where Δx = base_path[n+1] - base_path[n]. The fiber state stays at
        // the compute tier across all steps (Δx and sum_i A^a_{bi} Δx^i exact,
        // each update one compute-tier rounding) and is rounded to storage
        // once at the end (before 0.6.4 it was rounded after every step).
        let mut fiber: Vec<ComputeStorage> = (0..k).map(|a| up(initial_fiber[a])).collect();
        for step in 0..base_path.len() - 1 {
            let dx: Vec<ComputeStorage> = (0..n)
                .map(|i| compute_subtract(up(base_path[step + 1][i]), up(base_path[step][i])))
                .collect();
            let mut new_fiber = Vec::with_capacity(k);
            for a in 0..k {
                let mut coeff_dx = Vec::with_capacity(k);
                for b in 0..k {
                    let mut acc = make_compute_int(0);
                    for i in 0..n {
                        // both factors have zero low FRAC_BITS: the product is exact
                        acc = compute_checked_add(acc, compute_multiply(up(self.get_coeff(a, b, i)), dx[i]))?;
                    }
                    coeff_dx.push(acc);
                }
                new_fiber.push(exact_sub_dot_compute(fiber[a], &coeff_dx, &fiber)?);
            }
            fiber = new_fiber;
        }

        let mut out = FixedVector::new(k);
        for a in 0..k { out[a] = FixedPoint::from_raw(downscale_to_storage(fiber[a])?); }
        Ok(out)
    }
}

// ============================================================================
// PrincipalBundle: fiber = Lie group G
// ============================================================================

/// A principal G-bundle where the fiber is a Lie group.
///
/// Transition functions g_{αβ}: U_α ∩ U_β → G satisfy the cocycle condition:
///   g_{αβ} · g_{βγ} = g_{αγ}
///
/// This implementation uses a discrete representation with N open sets (charts)
/// and transition functions stored as matrices (Lie group elements).
pub struct PrincipalBundle {
    pub base_dim_val: usize,
    pub group_dim: usize,   // dimension of the Lie algebra (= fiber_dim)
    pub matrix_dim: usize,  // dimension of the matrix representation
    pub num_charts: usize,
    /// Transition functions g_{αβ} stored as FixedMatrix values.
    /// Index: transitions[α * num_charts + β] = g_{αβ}.
    /// Only defined for α ≠ β. g_{αα} = identity.
    pub transitions: Vec<FixedMatrix>,
}

impl PrincipalBundle {
    /// Create a trivial principal bundle (all transitions are identity).
    pub fn trivial(base_dim: usize, group_dim: usize, matrix_dim: usize, num_charts: usize) -> Self {
        let id = FixedMatrix::identity(matrix_dim);
        let transitions = vec![id; num_charts * num_charts];
        Self {
            base_dim_val: base_dim,
            group_dim,
            matrix_dim,
            num_charts,
            transitions,
        }
    }

    /// Get the transition function g_{αβ}.
    pub fn transition(&self, alpha: usize, beta: usize) -> &FixedMatrix {
        &self.transitions[alpha * self.num_charts + beta]
    }

    /// Set a transition function g_{αβ} and automatically set g_{βα} = g_{αβ}⁻¹.
    ///
    /// The inverse comes from a compute-tier LU, each entry rounded once
    /// (the storage LU inverse carried O(κ) units before 0.6.4).
    pub fn set_transition(
        &mut self,
        alpha: usize,
        beta: usize,
        g: FixedMatrix,
    ) -> Result<(), OverflowDetected> {
        let inv_c = compute_lu_decompose(&ComputeMatrix::from_fixed_matrix(&g))?.inverse()?;
        let mut g_inv = FixedMatrix::new(g.rows(), g.cols());
        for i in 0..g.rows() {
            for j in 0..g.cols() {
                g_inv.set(i, j, FixedPoint::from_raw(downscale_to_storage(inv_c.get(i, j))?));
            }
        }
        self.transitions[alpha * self.num_charts + beta] = g;
        self.transitions[beta * self.num_charts + alpha] = g_inv;
        Ok(())
    }

    /// Verify the cocycle condition: g_{αβ} · g_{βγ} = g_{αγ} for all triples.
    ///
    /// Returns the maximum entry-wise deviation from the cocycle condition.
    pub fn verify_cocycle(&self, tol: FixedPoint) -> (bool, FixedPoint) {
        let mut max_err = FixedPoint::ZERO;
        let mut ok = true;

        for alpha in 0..self.num_charts {
            for beta in 0..self.num_charts {
                for gamma in 0..self.num_charts {
                    let g_ab = self.transition(alpha, beta);
                    let g_bg = self.transition(beta, gamma);
                    let g_ag = self.transition(alpha, gamma);

                    // g_ab * g_bg should equal g_ag
                    let product = g_ab * g_bg;
                    for i in 0..self.matrix_dim {
                        for j in 0..self.matrix_dim {
                            let err = (product.get(i, j) - g_ag.get(i, j)).abs();
                            if err > max_err { max_err = err; }
                            if err > tol { ok = false; }
                        }
                    }
                }
            }
        }

        (ok, max_err)
    }
}

impl FiberBundle for PrincipalBundle {
    fn project(&self, total_point: &FixedVector) -> FixedVector {
        let mut base = FixedVector::new(self.base_dim_val);
        for i in 0..self.base_dim_val {
            base[i] = total_point[i];
        }
        base
    }

    fn base_dim(&self) -> usize { self.base_dim_val }
    fn fiber_dim(&self) -> usize { self.group_dim }

    fn lift(&self, base: &FixedVector, fiber: &FixedVector) -> FixedVector {
        let mut total = FixedVector::new(self.base_dim_val + self.group_dim);
        for i in 0..self.base_dim_val {
            total[i] = base[i];
        }
        for i in 0..self.group_dim {
            total[self.base_dim_val + i] = fiber[i];
        }
        total
    }

    fn local_trivialization(&self, total_point: &FixedVector) -> (FixedVector, FixedVector) {
        let mut base = FixedVector::new(self.base_dim_val);
        let mut fiber = FixedVector::new(self.group_dim);
        for i in 0..self.base_dim_val {
            base[i] = total_point[i];
        }
        for i in 0..self.group_dim {
            fiber[i] = total_point[self.base_dim_val + i];
        }
        (base, fiber)
    }
}

// ============================================================================
// Associated bundle construction
// ============================================================================

/// Apply a transition function (group element) to a fiber element via
/// matrix-vector multiplication (the fundamental representation).
///
/// For a principal bundle P with group G and representation ρ: G → GL(V),
/// the associated bundle P ×_ρ V has fibers transformed by ρ(g).
pub fn apply_representation(
    group_element: &FixedMatrix,
    fiber_element: &FixedVector,
) -> FixedVector {
    group_element.mul_vector(fiber_element)
}

/// Change of chart for a section: ξ_β = g_{αβ} · ξ_α.
///
/// Given a fiber element in chart α, compute the corresponding element in chart β.
pub fn change_chart(
    bundle: &PrincipalBundle,
    alpha: usize,
    beta: usize,
    fiber_alpha: &FixedVector,
) -> FixedVector {
    let g = bundle.transition(alpha, beta);
    g.mul_vector(fiber_alpha)
}

// ============================================================================
// Bundle curvature (connection form Ω = dω + ω∧ω)
// ============================================================================

/// Compute the curvature 2-form of a vector bundle connection at a point.
///
/// For a vector bundle with connection coefficients A^a_{bi}, the curvature is:
///   F^a_{bij} = ∂_i A^a_{bj} - ∂_j A^a_{bi} + A^a_{ci} A^c_{bj} - A^a_{cj} A^c_{bi}
///
/// Returns a rank-4 tensor [k, k, n, n] where entry [a, b, i, j] = F^a_{bij}.
///
/// The ∂A terms vanish (the stored coefficients are constant); each
/// quadratic A·A contraction is one rounding of its exact value.
pub fn vector_bundle_curvature(
    bundle: &VectorBundle,
    _base_point: &FixedVector,
) -> Result<super::tensor::Tensor, OverflowDetected> {
    let k = bundle.fiber_dim_val;
    let n = bundle.base_dim_val;
    let _h = super::curvature::differentiation_step();

    // For flat connections, curvature is zero
    if bundle.connection_coeffs.is_none() {
        return Ok(super::tensor::Tensor::new(&[k, k, n, n]));
    }

    // Compute derivatives of connection coefficients via central differences
    // and the quadratic terms A·A
    let mut curv = super::tensor::Tensor::new(&[k, k, n, n]);

    for a in 0..k {
        for b in 0..k {
            for i in 0..n {
                for j in 0..n {
                    // ∂_i A^a_{bj}: perturb base_point in direction i
                    // We need A^a_{bj} at (point + h*e_i) and (point - h*e_i)
                    // For simplicity, we evaluate the stored coefficients
                    // (which are constant for linear connections)
                    let d_i_a_bj = FixedPoint::ZERO; // constant coefficients
                    let d_j_a_bi = FixedPoint::ZERO;

                    // Quadratic terms: A^a_{ci} A^c_{bj} - A^a_{cj} A^c_{bi}, both
                    // sums exact at the compute tier and their difference
                    // rounded once (before 0.6.4 each sum was rounded, then
                    // subtracted)
                    let a_ci: Vec<BinaryStorage> = (0..k).map(|c| bundle.get_coeff(a, c, i).raw()).collect();
                    let c_bj: Vec<BinaryStorage> = (0..k).map(|c| bundle.get_coeff(c, b, j).raw()).collect();
                    let a_cj: Vec<BinaryStorage> = (0..k).map(|c| bundle.get_coeff(a, c, j).raw()).collect();
                    let c_bi: Vec<BinaryStorage> = (0..k).map(|c| bundle.get_coeff(c, b, i).raw()).collect();
                    let quad = compute_subtract(exact_dot(&a_ci, &c_bj)?, exact_dot(&a_cj, &c_bi)?);
                    let quad = FixedPoint::from_raw(downscale_to_storage(quad)?);

                    curv.set(&[a, b, i, j], d_i_a_bj - d_j_a_bi + quad);
                }
            }
        }
    }

    Ok(curv)
}
