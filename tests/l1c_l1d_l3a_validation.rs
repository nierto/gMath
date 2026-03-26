//! Validation tests for L1C (Derived Ops), L1D (Matrix Functions), L3A (Manifolds).
//! All reference values from mpmath 50-digit precision.

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::derived::*;
use g_math::fixed_point::imperative::matrix_functions::*;
use g_math::fixed_point::imperative::manifold::*;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn tol() -> FixedPoint {
    #[cfg(table_format = "q16_16")]
    { fp("0.1") }  // multi-step algorithms (Padé, Denman-Beavers) accumulate heavily at 16-bit
    #[cfg(not(table_format = "q16_16"))]
    { fp("0.0001") }
}
fn tight() -> FixedPoint {
    #[cfg(table_format = "q16_16")]
    { fp("0.01") }
    #[cfg(table_format = "q32_32")]
    { fp("0.0001") }
    #[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
    { fp("0.000000001") }
}

fn assert_fp(got: FixedPoint, exp: FixedPoint, tol: FixedPoint, name: &str) {
    let d = (got - exp).abs();
    assert!(d < tol, "{}: got {}, expected {}, diff={}", name, got, exp, d);
}

fn mat_near(a: &FixedMatrix, b: &FixedMatrix, tol: FixedPoint, name: &str) {
    assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()), "{}: dim mismatch", name);
    for r in 0..a.rows() {
        for c in 0..a.cols() {
            assert_fp(a.get(r, c), b.get(r, c), tol, &format!("{}[{},{}]", name, r, c));
        }
    }
}

// ============================================================================
// L1C: Norms
// ============================================================================

#[test]
fn test_frobenius_norm() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    // sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
    assert_fp(frobenius_norm(&a), fp("5.47722557505166113"), tight(), "frobenius");
}

#[test]
fn test_norm_1() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    // col 0: |1|+|3|=4, col 1: |2|+|4|=6 → max = 6
    assert_eq!(norm_1(&a), fp("6"));
}

#[test]
fn test_norm_inf() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    // row 0: |1|+|2|=3, row 1: |3|+|4|=7 → max = 7
    assert_eq!(norm_inf(&a), fp("7"));
}

// ============================================================================
// L1C: Solve wrappers
// ============================================================================

#[test]
fn test_solve_wrapper() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("2"), fp("1"), fp("5"), fp("3")]);
    let b = FixedVector::from_slice(&[fp("4"), fp("7")]);
    let x = solve(&a, &b).unwrap();
    assert_fp(x[0], fp("5"), tight(), "solve[0]");
    assert_fp(x[1], fp("-6"), tight(), "solve[1]");
}

#[test]
fn test_determinant_wrapper() {
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("2"), fp("3"), fp("4")]);
    let d = determinant(&a).unwrap();
    assert_fp(d, fp("-2"), tight(), "det");
}

#[test]
fn test_condition_number_1() {
    let id = FixedMatrix::identity(3);
    let cond = condition_number_1(&id).unwrap();
    // cond(I) = 1
    assert_fp(cond, fp("1"), tight(), "cond(I)");
}

// ============================================================================
// L1D: Matrix exponential
// ============================================================================

#[test]
fn test_matrix_exp_zero() {
    // exp(0) = I
    let z = FixedMatrix::new(3, 3);
    let result = matrix_exp(&z).unwrap();
    mat_near(&result, &FixedMatrix::identity(3), tight(), "exp(0)=I");
}

#[test]
fn test_matrix_exp_diagonal() {
    // exp([[1,0],[0,2]]) = [[e, 0],[0, e^2]]
    let a = FixedMatrix::from_slice(2, 2, &[fp("1"), fp("0"), fp("0"), fp("2")]);
    let result = matrix_exp(&a).unwrap();
    assert_fp(result.get(0, 0), fp("2.718281828459045235"), tol(), "exp diag [0,0]");
    assert_fp(result.get(0, 1), fp("0"), tol(), "exp diag [0,1]");
    assert_fp(result.get(1, 0), fp("0"), tol(), "exp diag [1,0]");
    assert_fp(result.get(1, 1), fp("7.389056098930650227"), tol(), "exp diag [1,1]");
}

#[test]
fn test_matrix_exp_rotation() {
    // exp([[0,1],[-1,0]]) = [[cos1, sin1],[-sin1, cos1]]
    let a = FixedMatrix::from_slice(2, 2, &[fp("0"), fp("1"), fp("-1"), fp("0")]);
    let result = matrix_exp(&a).unwrap();
    assert_fp(result.get(0, 0), fp("0.5403023058681397174"), tol(), "exp rot [0,0]");
    assert_fp(result.get(0, 1), fp("0.8414709848078965066"), tol(), "exp rot [0,1]");
    assert_fp(result.get(1, 0), fp("-0.8414709848078965066"), tol(), "exp rot [1,0]");
    assert_fp(result.get(1, 1), fp("0.5403023058681397174"), tol(), "exp rot [1,1]");
}

// ============================================================================
// L1D: Matrix square root
// ============================================================================

#[test]
fn test_matrix_sqrt_diagonal() {
    // sqrt([[4,0],[0,9]]) = [[2,0],[0,3]]
    let a = FixedMatrix::from_slice(2, 2, &[fp("4"), fp("0"), fp("0"), fp("9")]);
    let s = matrix_sqrt(&a).unwrap();
    assert_fp(s.get(0, 0), fp("2"), tol(), "sqrt diag [0,0]");
    assert_fp(s.get(1, 1), fp("3"), tol(), "sqrt diag [1,1]");
    assert_fp(s.get(0, 1), fp("0"), tol(), "sqrt diag [0,1]");
}

#[test]
fn test_matrix_sqrt_reconstruction() {
    // S^2 = A
    let a = FixedMatrix::from_slice(2, 2, &[fp("2"), fp("1"), fp("1"), fp("2")]);
    let s = matrix_sqrt(&a).unwrap();
    let s_sq = &s * &s;
    mat_near(&s_sq, &a, tol(), "sqrt(A)^2 = A");
}

#[test]
fn test_matrix_sqrt_identity() {
    let id = FixedMatrix::identity(3);
    let s = matrix_sqrt(&id).unwrap();
    mat_near(&s, &id, tight(), "sqrt(I) = I");
}

// ============================================================================
// L1D: Matrix logarithm
// ============================================================================

#[test]
fn test_matrix_log_identity() {
    // log(I) = 0
    let id = FixedMatrix::identity(3);
    let result = matrix_log(&id).unwrap();
    mat_near(&result, &FixedMatrix::new(3, 3), tight(), "log(I) = 0");
}

#[test]
fn test_matrix_exp_log_roundtrip() {
    // exp(log(A)) = A for SPD matrix
    let a = FixedMatrix::from_slice(2, 2, &[fp("2"), fp("1"), fp("1"), fp("2")]);
    let log_a = matrix_log(&a).unwrap();
    let exp_log_a = matrix_exp(&log_a).unwrap();
    mat_near(&exp_log_a, &a, tol(), "exp(log(A)) = A");
}

// ============================================================================
// L3A: Euclidean space
// ============================================================================

#[test]
fn test_euclidean_exp_log_roundtrip() {
    let e = EuclideanSpace { dim: 3 };
    let p = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let v = FixedVector::from_slice(&[fp("0.5"), fp("-0.3"), fp("0.1")]);
    let q = e.exp_map(&p, &v).unwrap();
    let v_back = e.log_map(&p, &q).unwrap();
    for i in 0..3 {
        assert_fp(v_back[i], v[i], tight(), &format!("euclidean log(exp(v))[{}]", i));
    }
}

#[test]
fn test_euclidean_distance() {
    let e = EuclideanSpace { dim: 2 };
    let p = FixedVector::from_slice(&[fp("0"), fp("0")]);
    let q = FixedVector::from_slice(&[fp("3"), fp("4")]);
    assert_fp(e.distance(&p, &q).unwrap(), fp("5"), tight(), "euclidean dist");
}

#[test]
fn test_euclidean_parallel_transport() {
    let e = EuclideanSpace { dim: 3 };
    let p = FixedVector::from_slice(&[fp("0"), fp("0"), fp("0")]);
    let q = FixedVector::from_slice(&[fp("1"), fp("1"), fp("1")]);
    let v = FixedVector::from_slice(&[fp("1"), fp("0"), fp("0")]);
    let transported = e.parallel_transport(&p, &q, &v).unwrap();
    // In flat space, parallel transport is identity
    for i in 0..3 {
        assert_fp(transported[i], v[i], tight(), &format!("euclidean PT[{}]", i));
    }
}

// ============================================================================
// L3A: Sphere S^2
// ============================================================================

#[test]
fn test_sphere_distance() {
    let s = Sphere { dim: 2 };
    // Distance between [1,0,0] and [0,1,0] on S^2 = pi/2
    let p = FixedVector::from_slice(&[fp("1"), fp("0"), fp("0")]);
    let q = FixedVector::from_slice(&[fp("0"), fp("1"), fp("0")]);
    let d = s.distance(&p, &q).unwrap();
    assert_fp(d, fp("1.5707963267948966192"), tol(), "sphere dist = pi/2");
}

#[test]
fn test_sphere_exp_log_roundtrip() {
    let s = Sphere { dim: 2 };
    let p = FixedVector::from_slice(&[fp("1"), fp("0"), fp("0")]);
    // Small tangent vector (orthogonal to p)
    let v = FixedVector::from_slice(&[fp("0"), fp("0.3"), fp("0.4")]);
    let q = s.exp_map(&p, &v).unwrap();
    let v_back = s.log_map(&p, &q).unwrap();
    for i in 0..3 {
        assert_fp(v_back[i], v[i], tol(), &format!("sphere log(exp(v))[{}]", i));
    }
}

#[test]
fn test_sphere_distance_same_point() {
    let s = Sphere { dim: 2 };
    let p = FixedVector::from_slice(&[fp("1"), fp("0"), fp("0")]);
    assert_fp(s.distance(&p, &p).unwrap(), fp("0"), tight(), "sphere dist(p,p)=0");
}

// ============================================================================
// L3A: Hyperbolic space H^1
// ============================================================================

#[test]
fn test_hyperbolic_distance() {
    let h = HyperbolicSpace { dim: 1 };
    // Origin of H^1: [1, 0]
    // Point at distance 1: [cosh(1), sinh(1)]
    let p = FixedVector::from_slice(&[fp("1"), fp("0")]);
    let q = FixedVector::from_slice(&[
        fp("1.5430806348152437784"),  // cosh(1)
        fp("1.1752011936438014568"),  // sinh(1)
    ]);
    let d = h.distance(&p, &q).unwrap();
    assert_fp(d, fp("1"), tol(), "hyperbolic dist = 1");
}

#[test]
fn test_hyperbolic_exp_log_roundtrip() {
    let h = HyperbolicSpace { dim: 1 };
    let p = FixedVector::from_slice(&[fp("1"), fp("0")]);
    // Tangent vector at p: must be Minkowski-orthogonal to p, i.e. [0, v1]
    let v = FixedVector::from_slice(&[fp("0"), fp("0.5")]);
    let q = h.exp_map(&p, &v).unwrap();
    let v_back = h.log_map(&p, &q).unwrap();
    for i in 0..2 {
        assert_fp(v_back[i], v[i], tol(), &format!("hyperbolic log(exp(v))[{}]", i));
    }
}

#[test]
fn test_hyperbolic_distance_same_point() {
    let h = HyperbolicSpace { dim: 1 };
    let p = FixedVector::from_slice(&[fp("1"), fp("0")]);
    assert_fp(h.distance(&p, &p).unwrap(), fp("0"), tight(), "hyp dist(p,p)=0");
}

// ============================================================================
// Cross-layer: matrix exp on manifold-relevant matrices
// ============================================================================

#[test]
fn test_exp_preserves_orthogonality() {
    // exp of skew-symmetric matrix should be orthogonal: R^T R = I
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("0"), fp("0.1"), fp("0.2"),
        fp("-0.1"), fp("0"), fp("0.3"),
        fp("-0.2"), fp("-0.3"), fp("0"),
    ]);
    let r = matrix_exp(&a).unwrap();
    let rtr = &r.transpose() * &r;
    mat_near(&rtr, &FixedMatrix::identity(3), tol(), "exp(skew)^T * exp(skew) = I");
}
