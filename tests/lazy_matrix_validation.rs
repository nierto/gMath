//! Validation tests for LazyMatrixExpr — matrix chain persistence.
//!
//! Tests verify that lazy matrix evaluation produces identical results to
//! eager evaluation, and that chains like exp(A)*B stay at compute tier
//! without intermediate materializations.

use g_math::fixed_point::{FixedPoint, FixedMatrix};
use g_math::canonical::{evaluate_matrix, LazyMatrixExpr};
use g_math::fixed_point::imperative::matrix_functions::*;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn tol() -> FixedPoint { fp("0.0001") }
fn tight() -> FixedPoint { fp("0.000000001") }

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
// Basic operations
// ============================================================================

#[test]
fn test_lazy_matrix_identity_roundtrip() {
    let id = FixedMatrix::identity(3);
    let expr = LazyMatrixExpr::from(id.clone());
    let result = evaluate_matrix(&expr).unwrap();
    mat_near(&result, &id, tight(), "identity_roundtrip");
}

#[test]
fn test_lazy_matrix_add() {
    let a = FixedMatrix::identity(2);
    let b = FixedMatrix::identity(2);
    let expr = LazyMatrixExpr::from(a.clone()) + LazyMatrixExpr::from(b.clone());
    let result = evaluate_matrix(&expr).unwrap();
    let expected = &a + &b;
    mat_near(&result, &expected, tight(), "lazy_add");
}

#[test]
fn test_lazy_matrix_sub() {
    let a = FixedMatrix::from_fn(2, 2, |r, c| fp(&format!("{}", (r * 2 + c + 1))));
    let b = FixedMatrix::identity(2);
    let expr = LazyMatrixExpr::from(a.clone()) - LazyMatrixExpr::from(b.clone());
    let result = evaluate_matrix(&expr).unwrap();
    let expected = &a - &b;
    mat_near(&result, &expected, tight(), "lazy_sub");
}

#[test]
fn test_lazy_matrix_mul() {
    let a = FixedMatrix::from_fn(2, 2, |r, c| fp(&format!("{}", (r * 2 + c + 1))));
    let b = FixedMatrix::from_fn(2, 2, |r, c| fp(&format!("{}", (r * 2 + c + 5))));
    let expr = LazyMatrixExpr::from(a.clone()) * LazyMatrixExpr::from(b.clone());
    let result = evaluate_matrix(&expr).unwrap();
    let expected = &a * &b;
    mat_near(&result, &expected, tight(), "lazy_mul");
}

#[test]
fn test_lazy_matrix_neg() {
    let a = FixedMatrix::from_fn(2, 2, |r, c| fp(&format!("{}", (r * 2 + c + 1))));
    let expr = -LazyMatrixExpr::from(a.clone());
    let result = evaluate_matrix(&expr).unwrap();
    let expected = -&a;
    mat_near(&result, &expected, tight(), "lazy_neg");
}

#[test]
fn test_lazy_matrix_transpose() {
    let a = FixedMatrix::from_fn(2, 3, |r, c| fp(&format!("{}", (r * 3 + c + 1))));
    let expr = LazyMatrixExpr::from(a.clone()).transpose();
    let result = evaluate_matrix(&expr).unwrap();
    let expected = a.transpose();
    mat_near(&result, &expected, tight(), "lazy_transpose");
}

#[test]
fn test_lazy_matrix_scalar_mul() {
    let a = FixedMatrix::identity(2);
    let s = fp("3");
    let expr = LazyMatrixExpr::from(a.clone()) * s;
    let result = evaluate_matrix(&expr).unwrap();
    // 3 * I should have 3 on diagonal
    assert_fp(result.get(0, 0), fp("3"), tight(), "scalar_mul[0,0]");
    assert_fp(result.get(1, 1), fp("3"), tight(), "scalar_mul[1,1]");
    assert_fp(result.get(0, 1), FixedPoint::ZERO, tight(), "scalar_mul[0,1]");
}

#[test]
fn test_lazy_matrix_identity_node() {
    let expr = LazyMatrixExpr::identity(4);
    let result = evaluate_matrix(&expr).unwrap();
    assert_eq!(result.rows(), 4);
    assert_eq!(result.cols(), 4);
    assert_fp(result.get(0, 0), fp("1"), tight(), "identity_node[0,0]");
    assert_fp(result.get(2, 3), FixedPoint::ZERO, tight(), "identity_node[2,3]");
}

// ============================================================================
// Matrix transcendentals — lazy vs eager comparison
// ============================================================================

#[test]
fn test_lazy_matrix_exp_vs_eager() {
    // Small matrix so exp converges well
    let a = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("0.1"),
            (0, 1) => fp("0.2"),
            (1, 0) => fp("0.3"),
            (1, 1) => fp("0.4"),
            _ => unreachable!(),
        }
    });
    let lazy_result = evaluate_matrix(&LazyMatrixExpr::from(a.clone()).exp()).unwrap();
    let eager_result = matrix_exp(&a).unwrap();
    mat_near(&lazy_result, &eager_result, tight(), "lazy_exp_vs_eager");
}

#[test]
fn test_lazy_matrix_log_vs_eager() {
    // Matrix close to identity for log convergence
    let a = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("1.1"),
            (0, 1) => fp("0.05"),
            (1, 0) => fp("0.05"),
            (1, 1) => fp("1.2"),
            _ => unreachable!(),
        }
    });
    let lazy_result = evaluate_matrix(&LazyMatrixExpr::from(a.clone()).log()).unwrap();
    let eager_result = matrix_log(&a).unwrap();
    mat_near(&lazy_result, &eager_result, tight(), "lazy_log_vs_eager");
}

#[test]
fn test_lazy_matrix_sqrt_vs_eager() {
    let a = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("4"),
            (0, 1) => fp("0"),
            (1, 0) => fp("0"),
            (1, 1) => fp("9"),
            _ => unreachable!(),
        }
    });
    let lazy_result = evaluate_matrix(&LazyMatrixExpr::from(a.clone()).sqrt()).unwrap();
    let eager_result = matrix_sqrt(&a).unwrap();
    mat_near(&lazy_result, &eager_result, tol(), "lazy_sqrt_vs_eager");
}

// ============================================================================
// Chain persistence — the key feature
// ============================================================================

#[test]
fn test_lazy_matrix_exp_log_roundtrip() {
    // exp(log(A)) should ≈ A for well-conditioned A
    let a = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("1.5"),
            (0, 1) => fp("0.1"),
            (1, 0) => fp("0.1"),
            (1, 1) => fp("2.0"),
            _ => unreachable!(),
        }
    });
    // Lazy chain: exp(log(A)) — all at compute tier, single downscale
    let expr = LazyMatrixExpr::from(a.clone()).log().exp();
    let result = evaluate_matrix(&expr).unwrap();
    mat_near(&result, &a, tol(), "exp_log_chain_roundtrip");
}

#[test]
fn test_lazy_matrix_log_exp_roundtrip() {
    // log(exp(A)) should ≈ A for small A
    let a = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("0.1"),
            (0, 1) => fp("0.05"),
            (1, 0) => fp("0.05"),
            (1, 1) => fp("0.2"),
            _ => unreachable!(),
        }
    });
    let expr = LazyMatrixExpr::from(a.clone()).exp().log();
    let result = evaluate_matrix(&expr).unwrap();
    mat_near(&result, &a, tol(), "log_exp_chain_roundtrip");
}

#[test]
fn test_lazy_matrix_chain_mul_exp() {
    // exp(A) * exp(B) — both exp stay at compute tier, matmul at compute tier
    let a = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("0.1"),
            (0, 1) => fp("0.0"),
            (1, 0) => fp("0.0"),
            (1, 1) => fp("0.2"),
            _ => unreachable!(),
        }
    });
    let b = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("0.3"),
            (0, 1) => fp("0.0"),
            (1, 0) => fp("0.0"),
            (1, 1) => fp("0.1"),
            _ => unreachable!(),
        }
    });
    // Lazy: exp(A) * exp(B) — zero intermediate materializations
    let expr = LazyMatrixExpr::from(a.clone()).exp() * LazyMatrixExpr::from(b.clone()).exp();
    let lazy_result = evaluate_matrix(&expr).unwrap();

    // Eager: two separate materializations
    let exp_a = matrix_exp(&a).unwrap();
    let exp_b = matrix_exp(&b).unwrap();
    let eager_result = &exp_a * &exp_b;

    mat_near(&lazy_result, &eager_result, tol(), "chain_mul_exp");
}

#[test]
fn test_lazy_matrix_pow_chain() {
    // A^p via lazy should match eager
    let a = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("2"),
            (0, 1) => fp("0"),
            (1, 0) => fp("0"),
            (1, 1) => fp("3"),
            _ => unreachable!(),
        }
    });
    let half = fp("0.5");
    let expr = LazyMatrixExpr::from(a.clone()).pow(half);
    let lazy_result = evaluate_matrix(&expr).unwrap();
    let eager_result = matrix_pow(&a, half).unwrap();
    mat_near(&lazy_result, &eager_result, tol(), "pow_chain");
}

#[test]
fn test_lazy_matrix_inverse() {
    let a = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("2"),
            (0, 1) => fp("1"),
            (1, 0) => fp("1"),
            (1, 1) => fp("3"),
            _ => unreachable!(),
        }
    });
    // A * A^{-1} should ≈ I
    let expr = LazyMatrixExpr::from(a.clone()) * LazyMatrixExpr::from(a.clone()).inverse();
    let result = evaluate_matrix(&expr).unwrap();
    let id = FixedMatrix::identity(2);
    mat_near(&result, &id, tol(), "A_times_A_inv");
}

#[test]
fn test_lazy_matrix_complex_chain() {
    // (A + B)^T * exp(A) — 3 operations, all at compute tier
    let a = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("0.1"),
            (0, 1) => fp("0.2"),
            (1, 0) => fp("0.3"),
            (1, 1) => fp("0.1"),
            _ => unreachable!(),
        }
    });
    let b = FixedMatrix::from_fn(2, 2, |r, c| {
        match (r, c) {
            (0, 0) => fp("0.05"),
            (0, 1) => fp("0.1"),
            (1, 0) => fp("0.1"),
            (1, 1) => fp("0.05"),
            _ => unreachable!(),
        }
    });

    let expr = (LazyMatrixExpr::from(a.clone()) + LazyMatrixExpr::from(b.clone())).transpose()
        * LazyMatrixExpr::from(a.clone()).exp();
    let lazy_result = evaluate_matrix(&expr).unwrap();

    // Eager equivalent
    let sum = &a + &b;
    let sum_t = sum.transpose();
    let exp_a = matrix_exp(&a).unwrap();
    let eager_result = &sum_t * &exp_a;

    mat_near(&lazy_result, &eager_result, tol(), "complex_chain");
}

// ============================================================================
// Depth and operation count
// ============================================================================

#[test]
fn test_lazy_matrix_expr_metadata() {
    let a = LazyMatrixExpr::from(FixedMatrix::identity(2));
    let b = LazyMatrixExpr::from(FixedMatrix::identity(2));

    // exp(A) * B + A
    let expr = a.clone().exp() * b + a;
    assert!(expr.depth() >= 3);
    assert!(expr.operation_count() >= 3);
}
