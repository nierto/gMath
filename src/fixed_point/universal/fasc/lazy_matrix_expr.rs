//! Lazy Matrix Expression Tree — Matrix Chain Persistence via FASC
//!
//! **MISSION**: Build matrix expression trees lazily, evaluate entirely at compute tier
//! (tier N+1), single downscale to FixedMatrix at the end — the matrix analog of
//! BinaryCompute chain persistence for scalars.
//!
//! **ARCHITECTURE**: `LazyMatrixExpr` mirrors `LazyExpr` but for matrix operations.
//! All intermediates stay as `ComputeMatrix` during evaluation. This eliminates
//! materialization boundaries that previously caused TierOverflow on intermediate
//! results (e.g., `exp(A) * B * exp(C)` where individual exp results overflow storage).
//!
//! **USAGE**:
//! ```rust,no_run
//! use g_math::canonical::{evaluate_matrix, LazyMatrixExpr};
//! use g_math::fixed_point::FixedMatrix;
//!
//! let a = FixedMatrix::identity(2);
//! let b = FixedMatrix::identity(2);
//! let expr = LazyMatrixExpr::from(a).exp() * LazyMatrixExpr::from(b);
//! let result = evaluate_matrix(&expr).unwrap(); // Single downscale at end
//! ```

use core::ops::{Add, Sub, Mul, Neg};

use crate::fixed_point::imperative::FixedMatrix;
use crate::fixed_point::imperative::FixedPoint;
use crate::fixed_point::imperative::compute_matrix::{ComputeMatrix, compute_lu_decompose};
use crate::fixed_point::imperative::matrix_functions::{
    matrix_exp_compute, matrix_log_compute, matrix_sqrt_compute,
};
use crate::fixed_point::imperative::linalg::upscale_to_compute;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// LazyMatrixExpr — deferred matrix expression tree
// ============================================================================

/// Lazy matrix expression tree.
///
/// All intermediates stay at compute tier (ComputeMatrix) during evaluation.
/// Single downscale to FixedMatrix via `evaluate_matrix()` at the top level.
#[derive(Debug, Clone)]
pub enum LazyMatrixExpr {
    /// A concrete matrix (leaf node).
    Literal(FixedMatrix),

    /// Identity matrix of given dimension.
    Identity(usize),

    /// Matrix addition: A + B
    Add(Box<LazyMatrixExpr>, Box<LazyMatrixExpr>),

    /// Matrix subtraction: A - B
    Sub(Box<LazyMatrixExpr>, Box<LazyMatrixExpr>),

    /// Matrix multiplication (matmul): A * B
    Mul(Box<LazyMatrixExpr>, Box<LazyMatrixExpr>),

    /// Scalar-matrix multiplication: s * A
    ScalarMul(FixedPoint, Box<LazyMatrixExpr>),

    /// Matrix transpose: Aᵀ
    Transpose(Box<LazyMatrixExpr>),

    /// Matrix negation: -A
    Negate(Box<LazyMatrixExpr>),

    /// Matrix inverse: A⁻¹ via compute-tier LU
    Inverse(Box<LazyMatrixExpr>),

    /// Matrix exponential: exp(A) via Padé [6/6] at compute tier
    Exp(Box<LazyMatrixExpr>),

    /// Matrix logarithm: log(A) via inverse scaling-and-squaring at compute tier
    Log(Box<LazyMatrixExpr>),

    /// Matrix square root: A^{1/2} via Denman-Beavers at compute tier
    Sqrt(Box<LazyMatrixExpr>),

    /// Matrix power: A^p = exp(p * log(A)) — entire chain at compute tier
    Pow(Box<LazyMatrixExpr>, FixedPoint),
}

// ============================================================================
// Constructor helpers
// ============================================================================

impl LazyMatrixExpr {
    /// Create from a concrete FixedMatrix.
    pub fn literal(m: FixedMatrix) -> Self {
        LazyMatrixExpr::Literal(m)
    }

    /// Create an identity matrix expression.
    pub fn identity(n: usize) -> Self {
        LazyMatrixExpr::Identity(n)
    }

    /// Matrix exponential: exp(self)
    pub fn exp(self) -> Self {
        LazyMatrixExpr::Exp(Box::new(self))
    }

    /// Matrix logarithm: log(self)
    pub fn log(self) -> Self {
        LazyMatrixExpr::Log(Box::new(self))
    }

    /// Matrix square root: self^{1/2}
    pub fn sqrt(self) -> Self {
        LazyMatrixExpr::Sqrt(Box::new(self))
    }

    /// Matrix power: self^p
    pub fn pow(self, p: FixedPoint) -> Self {
        LazyMatrixExpr::Pow(Box::new(self), p)
    }

    /// Matrix transpose: selfᵀ
    pub fn transpose(self) -> Self {
        LazyMatrixExpr::Transpose(Box::new(self))
    }

    /// Matrix inverse: self⁻¹
    pub fn inverse(self) -> Self {
        LazyMatrixExpr::Inverse(Box::new(self))
    }

    /// Scalar-matrix multiply: s * self
    pub fn scale(self, s: FixedPoint) -> Self {
        LazyMatrixExpr::ScalarMul(s, Box::new(self))
    }

    /// Get expression depth for stack planning.
    pub fn depth(&self) -> usize {
        match self {
            LazyMatrixExpr::Literal(_) | LazyMatrixExpr::Identity(_) => 1,
            LazyMatrixExpr::Transpose(inner) | LazyMatrixExpr::Negate(inner)
            | LazyMatrixExpr::Inverse(inner) | LazyMatrixExpr::Exp(inner)
            | LazyMatrixExpr::Log(inner) | LazyMatrixExpr::Sqrt(inner)
            | LazyMatrixExpr::ScalarMul(_, inner) | LazyMatrixExpr::Pow(inner, _) => {
                1 + inner.depth()
            }
            LazyMatrixExpr::Add(l, r) | LazyMatrixExpr::Sub(l, r)
            | LazyMatrixExpr::Mul(l, r) => {
                1 + l.depth().max(r.depth())
            }
        }
    }

    /// Count operations in expression tree.
    pub fn operation_count(&self) -> usize {
        match self {
            LazyMatrixExpr::Literal(_) | LazyMatrixExpr::Identity(_) => 0,
            LazyMatrixExpr::Transpose(inner) | LazyMatrixExpr::Negate(inner)
            | LazyMatrixExpr::Inverse(inner) | LazyMatrixExpr::Exp(inner)
            | LazyMatrixExpr::Log(inner) | LazyMatrixExpr::Sqrt(inner)
            | LazyMatrixExpr::ScalarMul(_, inner) | LazyMatrixExpr::Pow(inner, _) => {
                1 + inner.operation_count()
            }
            LazyMatrixExpr::Add(l, r) | LazyMatrixExpr::Sub(l, r)
            | LazyMatrixExpr::Mul(l, r) => {
                1 + l.operation_count() + r.operation_count()
            }
        }
    }
}

// ============================================================================
// Operator overloading for natural matrix expression syntax
// ============================================================================

impl From<FixedMatrix> for LazyMatrixExpr {
    fn from(m: FixedMatrix) -> Self {
        LazyMatrixExpr::Literal(m)
    }
}

impl Add for LazyMatrixExpr {
    type Output = LazyMatrixExpr;
    fn add(self, other: Self) -> Self::Output {
        LazyMatrixExpr::Add(Box::new(self), Box::new(other))
    }
}

impl Sub for LazyMatrixExpr {
    type Output = LazyMatrixExpr;
    fn sub(self, other: Self) -> Self::Output {
        LazyMatrixExpr::Sub(Box::new(self), Box::new(other))
    }
}

impl Mul for LazyMatrixExpr {
    type Output = LazyMatrixExpr;
    fn mul(self, other: Self) -> Self::Output {
        LazyMatrixExpr::Mul(Box::new(self), Box::new(other))
    }
}

impl Neg for LazyMatrixExpr {
    type Output = LazyMatrixExpr;
    fn neg(self) -> Self::Output {
        LazyMatrixExpr::Negate(Box::new(self))
    }
}

/// Scalar * LazyMatrixExpr
impl Mul<LazyMatrixExpr> for FixedPoint {
    type Output = LazyMatrixExpr;
    fn mul(self, matrix: LazyMatrixExpr) -> Self::Output {
        LazyMatrixExpr::ScalarMul(self, Box::new(matrix))
    }
}

/// LazyMatrixExpr * Scalar
impl Mul<FixedPoint> for LazyMatrixExpr {
    type Output = LazyMatrixExpr;
    fn mul(self, scalar: FixedPoint) -> Self::Output {
        LazyMatrixExpr::ScalarMul(scalar, Box::new(self))
    }
}

// ============================================================================
// Display
// ============================================================================

impl core::fmt::Display for LazyMatrixExpr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LazyMatrixExpr::Literal(_) => write!(f, "Matrix"),
            LazyMatrixExpr::Identity(n) => write!(f, "I({})", n),
            LazyMatrixExpr::Add(l, r) => write!(f, "({} + {})", l, r),
            LazyMatrixExpr::Sub(l, r) => write!(f, "({} - {})", l, r),
            LazyMatrixExpr::Mul(l, r) => write!(f, "({} * {})", l, r),
            LazyMatrixExpr::ScalarMul(s, inner) => write!(f, "({} * {})", s, inner),
            LazyMatrixExpr::Transpose(inner) => write!(f, "({})ᵀ", inner),
            LazyMatrixExpr::Negate(inner) => write!(f, "-({})", inner),
            LazyMatrixExpr::Inverse(inner) => write!(f, "({})⁻¹", inner),
            LazyMatrixExpr::Exp(inner) => write!(f, "exp({})", inner),
            LazyMatrixExpr::Log(inner) => write!(f, "log({})", inner),
            LazyMatrixExpr::Sqrt(inner) => write!(f, "sqrt({})", inner),
            LazyMatrixExpr::Pow(inner, p) => write!(f, "({})^{}", inner, p),
        }
    }
}

// ============================================================================
// EVALUATOR — recursive tree walk at ComputeMatrix tier
// ============================================================================

/// Evaluate a LazyMatrixExpr entirely at compute tier (tier N+1),
/// returning a ComputeMatrix. No intermediate downscales.
fn eval_compute(expr: &LazyMatrixExpr) -> Result<ComputeMatrix, OverflowDetected> {
    match expr {
        LazyMatrixExpr::Literal(m) => Ok(ComputeMatrix::from_fixed_matrix(m)),

        LazyMatrixExpr::Identity(n) => Ok(ComputeMatrix::identity(*n)),

        LazyMatrixExpr::Add(l, r) => {
            let lc = eval_compute(l)?;
            let rc = eval_compute(r)?;
            Ok(lc.add(&rc))
        }

        LazyMatrixExpr::Sub(l, r) => {
            let lc = eval_compute(l)?;
            let rc = eval_compute(r)?;
            Ok(lc.sub(&rc))
        }

        LazyMatrixExpr::Mul(l, r) => {
            let lc = eval_compute(l)?;
            let rc = eval_compute(r)?;
            Ok(lc.mat_mul(&rc))
        }

        LazyMatrixExpr::ScalarMul(s, inner) => {
            let mc = eval_compute(inner)?;
            let s_compute = upscale_to_compute(s.raw());
            Ok(mc.scalar_mul(s_compute))
        }

        LazyMatrixExpr::Transpose(inner) => {
            let mc = eval_compute(inner)?;
            Ok(mc.transpose())
        }

        LazyMatrixExpr::Negate(inner) => {
            let mc = eval_compute(inner)?;
            Ok(mc.neg())
        }

        LazyMatrixExpr::Inverse(inner) => {
            let mc = eval_compute(inner)?;
            let lu = compute_lu_decompose(&mc)?;
            lu.inverse()
        }

        LazyMatrixExpr::Exp(inner) => {
            let mc = eval_compute(inner)?;
            matrix_exp_compute(&mc)
        }

        LazyMatrixExpr::Log(inner) => {
            let mc = eval_compute(inner)?;
            matrix_log_compute(&mc)
        }

        LazyMatrixExpr::Sqrt(inner) => {
            let mc = eval_compute(inner)?;
            matrix_sqrt_compute(&mc)
        }

        LazyMatrixExpr::Pow(inner, p) => {
            // A^p = exp(p * log(A)) — entire chain at compute tier
            let mc = eval_compute(inner)?;
            let log_a = matrix_log_compute(&mc)?;
            let p_compute = upscale_to_compute(p.raw());
            let p_log_a = log_a.scalar_mul(p_compute);
            matrix_exp_compute(&p_log_a)
        }
    }
}

/// Evaluate a lazy matrix expression with full chain persistence.
///
/// All intermediate operations stay at compute tier (tier N+1).
/// Single downscale to FixedMatrix at the very end → 0-1 ULP.
///
/// **Chain persistence example**:
/// ```rust,no_run
/// use g_math::canonical::{evaluate_matrix, LazyMatrixExpr};
/// use g_math::fixed_point::FixedMatrix;
///
/// let a = LazyMatrixExpr::from(FixedMatrix::identity(2));
/// let b = LazyMatrixExpr::from(FixedMatrix::identity(2));
/// // Entire chain at compute tier — no intermediate materializations
/// let result = evaluate_matrix(&(a.exp() * b.exp())).unwrap();
/// ```
pub fn evaluate_matrix(expr: &LazyMatrixExpr) -> Result<FixedMatrix, OverflowDetected> {
    let compute_result = eval_compute(expr)?;
    Ok(compute_result.to_fixed_matrix())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_matrix_expr_building() {
        let a = LazyMatrixExpr::from(FixedMatrix::identity(2));
        let b = LazyMatrixExpr::from(FixedMatrix::identity(2));
        let expr = a * b;
        assert_eq!(expr.depth(), 2);
        assert_eq!(expr.operation_count(), 1);
    }

    #[test]
    fn test_lazy_matrix_identity_eval() {
        let expr = LazyMatrixExpr::Identity(3);
        let result = evaluate_matrix(&expr).unwrap();
        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(result.get(i, j), FixedPoint::one());
                } else {
                    assert!(result.get(i, j).is_zero());
                }
            }
        }
    }

    #[test]
    fn test_lazy_matrix_add() {
        let a = FixedMatrix::identity(2);
        let expr = LazyMatrixExpr::from(a.clone()) + LazyMatrixExpr::from(a);
        let result = evaluate_matrix(&expr).unwrap();
        // I + I = 2I
        let two = FixedPoint::from_int(2);
        assert_eq!(result.get(0, 0), two);
        assert_eq!(result.get(1, 1), two);
        assert!(result.get(0, 1).is_zero());
    }

    #[test]
    fn test_lazy_matrix_chain_depth() {
        let a = LazyMatrixExpr::from(FixedMatrix::identity(2));
        let expr = a.exp().log(); // exp then log — should roundtrip to ~identity
        assert_eq!(expr.depth(), 3);
        assert_eq!(expr.operation_count(), 2);
    }

    #[test]
    fn test_lazy_matrix_display() {
        let a = LazyMatrixExpr::from(FixedMatrix::identity(2));
        let expr = a.exp();
        let s = format!("{}", expr);
        assert!(s.contains("exp"));
    }
}
