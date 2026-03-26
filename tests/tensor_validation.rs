//! L2B validation tests for Tensor type and operations.
//!
//! Tests verify:
//! 1. Construction, get/set, conversions
//! 2. Contraction correctness (matrix multiply, dot product, rank-3 contraction)
//! 3. Outer product
//! 4. Index operations: transpose, trace, symmetrize, antisymmetrize
//! 5. Metric raise/lower
//! 6. mpmath reference values for specific contractions

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::tensor::{
    Tensor, contract, outer, transpose, trace, raise_index,
    symmetrize, antisymmetrize,
};

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn tol() -> FixedPoint {
    #[cfg(table_format = "q16_16")]
    { fp("0.01") }
    #[cfg(table_format = "q32_32")]
    { fp("0.0001") }
    #[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
    { fp("0.000000001") }
}

// ============================================================================
// Construction and access
// ============================================================================

#[test]
fn test_tensor_new_and_shape() {
    let t = Tensor::new(&[2, 3, 4]);
    assert_eq!(t.rank(), 3);
    assert_eq!(t.shape(), &[2, 3, 4]);
    assert_eq!(t.len(), 24);
}

#[test]
fn test_tensor_get_set() {
    let mut t = Tensor::new(&[2, 3]);
    t.set(&[1, 2], fp("7"));
    assert!((t.get(&[1, 2]) - fp("7")).abs() < tol());
    assert!(t.get(&[0, 0]).is_zero());
}

#[test]
fn test_tensor_from_scalar() {
    let t = Tensor::from(fp("42"));
    assert_eq!(t.rank(), 0);
    assert!((t.to_scalar() - fp("42")).abs() < tol());
}

#[test]
fn test_tensor_from_vector() {
    let v = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let t = Tensor::from(&v);
    assert_eq!(t.rank(), 1);
    assert_eq!(t.shape(), &[3]);
    assert!((t.get(&[1]) - fp("2")).abs() < tol());

    let v2 = t.to_vector();
    assert!((v2[2] - fp("3")).abs() < tol());
}

#[test]
fn test_tensor_from_matrix() {
    let m = FixedMatrix::from_slice(2, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
    ]);
    let t = Tensor::from(&m);
    assert_eq!(t.rank(), 2);
    assert_eq!(t.shape(), &[2, 3]);
    assert!((t.get(&[1, 0]) - fp("4")).abs() < tol());

    let m2 = t.to_matrix();
    assert!((m2.get(0, 2) - fp("3")).abs() < tol());
}

// ============================================================================
// Contraction
// ============================================================================

#[test]
fn test_contraction_dot_product() {
    // [1,2,3] . [4,5,6] = 32
    let a = Tensor::from(&FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]));
    let b = Tensor::from(&FixedVector::from_slice(&[fp("4"), fp("5"), fp("6")]));
    let c = contract(&a, &b, &[(0, 0)]);
    assert_eq!(c.rank(), 0);
    assert!((c.to_scalar() - fp("32")).abs() < tol(),
        "dot product: got {}, expected 32", c.to_scalar());
}

#[test]
fn test_contraction_matrix_multiply() {
    // mpmath: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = Tensor::from(&FixedMatrix::from_slice(2, 2, &[
        fp("1"), fp("2"), fp("3"), fp("4"),
    ]));
    let b = Tensor::from(&FixedMatrix::from_slice(2, 2, &[
        fp("5"), fp("6"), fp("7"), fp("8"),
    ]));
    let c = contract(&a, &b, &[(1, 0)]); // A[i,j]*B[j,k] → C[i,k]
    assert_eq!(c.rank(), 2);
    assert_eq!(c.shape(), &[2, 2]);

    assert!((c.get(&[0, 0]) - fp("19")).abs() < tol(), "C[0,0]={}", c.get(&[0, 0]));
    assert!((c.get(&[0, 1]) - fp("22")).abs() < tol(), "C[0,1]={}", c.get(&[0, 1]));
    assert!((c.get(&[1, 0]) - fp("43")).abs() < tol(), "C[1,0]={}", c.get(&[1, 0]));
    assert!((c.get(&[1, 1]) - fp("50")).abs() < tol(), "C[1,1]={}", c.get(&[1, 1]));
}

#[test]
fn test_contraction_rank3_with_vector() {
    // T[i,j,k] * V[k] = M[i,j]
    // T: 2×2×2, V = [1, -1]
    // M[i,j] = T[i,j,0] - T[i,j,1] = -1 for all
    let mut t = Tensor::new(&[2, 2, 2]);
    // T[0,0,:] = [1,2], T[0,1,:] = [3,4], T[1,0,:] = [5,6], T[1,1,:] = [7,8]
    t.set(&[0, 0, 0], fp("1")); t.set(&[0, 0, 1], fp("2"));
    t.set(&[0, 1, 0], fp("3")); t.set(&[0, 1, 1], fp("4"));
    t.set(&[1, 0, 0], fp("5")); t.set(&[1, 0, 1], fp("6"));
    t.set(&[1, 1, 0], fp("7")); t.set(&[1, 1, 1], fp("8"));

    let v = Tensor::from(&FixedVector::from_slice(&[fp("1"), fp("-1")]));
    let m = contract(&t, &v, &[(2, 0)]); // T[i,j,k]*V[k] → M[i,j]

    assert_eq!(m.rank(), 2);
    assert_eq!(m.shape(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert!((m.get(&[i, j]) - fp("-1")).abs() < tol(),
                "M[{},{}] = {} (expected -1)", i, j, m.get(&[i, j]));
        }
    }
}

// ============================================================================
// Outer product
// ============================================================================

#[test]
fn test_outer_product() {
    let a = Tensor::from(&FixedVector::from_slice(&[fp("1"), fp("2")]));
    let b = Tensor::from(&FixedVector::from_slice(&[fp("3"), fp("4"), fp("5")]));
    let c = outer(&a, &b);
    assert_eq!(c.rank(), 2);
    assert_eq!(c.shape(), &[2, 3]);
    assert!((c.get(&[0, 0]) - fp("3")).abs() < tol());
    assert!((c.get(&[0, 2]) - fp("5")).abs() < tol());
    assert!((c.get(&[1, 0]) - fp("6")).abs() < tol());
    assert!((c.get(&[1, 2]) - fp("10")).abs() < tol());
}

#[test]
fn test_outer_product_rank0() {
    let a = Tensor::from(fp("3"));
    let b = Tensor::from(&FixedVector::from_slice(&[fp("2"), fp("5")]));
    let c = outer(&a, &b);
    assert_eq!(c.rank(), 1);
    assert!((c.get(&[0]) - fp("6")).abs() < tol());
    assert!((c.get(&[1]) - fp("15")).abs() < tol());
}

// ============================================================================
// Transpose
// ============================================================================

#[test]
fn test_transpose_rank2() {
    let m = FixedMatrix::from_slice(2, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
    ]);
    let t = Tensor::from(&m);
    let tt = transpose(&t, &[1, 0]); // swap indices
    assert_eq!(tt.shape(), &[3, 2]);
    assert!((tt.get(&[0, 1]) - fp("4")).abs() < tol());
    assert!((tt.get(&[2, 0]) - fp("3")).abs() < tol());
}

#[test]
fn test_transpose_rank3() {
    let mut t = Tensor::new(&[2, 3, 4]);
    t.set(&[1, 2, 3], fp("42"));
    let tt = transpose(&t, &[2, 0, 1]); // (i,j,k) → (k,i,j)
    assert_eq!(tt.shape(), &[4, 2, 3]);
    assert!((tt.get(&[3, 1, 2]) - fp("42")).abs() < tol());
}

// ============================================================================
// Trace
// ============================================================================

#[test]
fn test_trace_rank2() {
    // tr([[1,2],[3,4]]) = 5
    let m = Tensor::from(&FixedMatrix::from_slice(2, 2, &[
        fp("1"), fp("2"), fp("3"), fp("4"),
    ]));
    let tr = trace(&m, 0, 1);
    assert_eq!(tr.rank(), 0);
    assert!((tr.to_scalar() - fp("5")).abs() < tol(),
        "tr = {} (expected 5)", tr.to_scalar());
}

#[test]
fn test_trace_rank4() {
    // Rank-4 tensor, trace over indices 1 and 3 (both dim 2)
    // T[a,i,b,i] summed over i → result is rank-2 [a,b]
    let mut t = Tensor::new(&[2, 2, 3, 2]);
    t.set(&[0, 0, 1, 0], fp("10")); // contributes to result[0,1]
    t.set(&[0, 1, 1, 1], fp("20")); // contributes to result[0,1]
    let tr = trace(&t, 1, 3);
    assert_eq!(tr.rank(), 2);
    assert_eq!(tr.shape(), &[2, 3]);
    // result[0,1] = T[0,0,1,0] + T[0,1,1,1] = 10 + 20 = 30
    assert!((tr.get(&[0, 1]) - fp("30")).abs() < tol(),
        "trace[0,1] = {} (expected 30)", tr.get(&[0, 1]));
}

// ============================================================================
// Metric raise/lower
// ============================================================================

#[test]
fn test_raise_index_mpmath() {
    // g = [[1, 0.5],[0.5, 2]], v = [3, 1]
    // mpmath: g_inv * v = [3.14285714285714..., -0.285714285714285...]
    let g_inv = Tensor::from(&FixedMatrix::from_slice(2, 2, &[
        fp("1.14285714285714285"), fp("-0.28571428571428571"),
        fp("-0.28571428571428571"), fp("0.57142857142857142"),
    ]));
    let v = Tensor::from(&FixedVector::from_slice(&[fp("3"), fp("1")]));
    let raised = raise_index(&v, 0, &g_inv);
    assert_eq!(raised.rank(), 1);

    let expected_0 = fp("3.14285714285714285");
    let expected_1 = fp("-0.28571428571428571");
    assert!((raised.get(&[0]) - expected_0).abs() < tol(),
        "raised[0] = {} (expected {})", raised.get(&[0]), expected_0);
    assert!((raised.get(&[1]) - expected_1).abs() < tol(),
        "raised[1] = {} (expected {})", raised.get(&[1]), expected_1);
}

// ============================================================================
// Symmetrize / Antisymmetrize
// ============================================================================

#[test]
fn test_symmetrize_rank2() {
    // Symmetrize [[1, 2], [3, 4]] → [[(1+3)/2, (2+2)/2], [(3+1)/2, (4+4)/2]]
    // = [[2, 2], [2, 4]]  wait: sym[i,j] = (T[i,j]+T[j,i])/2
    // sym[0,0] = (1+1)/2 = 1, sym[0,1] = (2+3)/2 = 2.5
    // sym[1,0] = (3+2)/2 = 2.5, sym[1,1] = (4+4)/2 = 4
    let m = Tensor::from(&FixedMatrix::from_slice(2, 2, &[
        fp("1"), fp("2"), fp("3"), fp("4"),
    ]));
    let s = symmetrize(&m, &[0, 1]);
    assert!((s.get(&[0, 1]) - fp("2.5")).abs() < tol(),
        "sym[0,1] = {} (expected 2.5)", s.get(&[0, 1]));
    assert!((s.get(&[1, 0]) - fp("2.5")).abs() < tol());
    // Should be symmetric
    assert!((s.get(&[0, 1]) - s.get(&[1, 0])).abs() < tol());
}

#[test]
fn test_antisymmetrize_rank2() {
    // antisym[i,j] = (T[i,j] - T[j,i]) / 2
    // For [[1,2],[3,4]]: antisym[0,1] = (2-3)/2 = -0.5
    let m = Tensor::from(&FixedMatrix::from_slice(2, 2, &[
        fp("1"), fp("2"), fp("3"), fp("4"),
    ]));
    let a = antisymmetrize(&m, &[0, 1]);
    assert!((a.get(&[0, 1]) - fp("-0.5")).abs() < tol(),
        "antisym[0,1] = {} (expected -0.5)", a.get(&[0, 1]));
    assert!((a.get(&[1, 0]) - fp("0.5")).abs() < tol());
    // Diagonal should be zero
    assert!(a.get(&[0, 0]).abs() < tol());
    assert!(a.get(&[1, 1]).abs() < tol());
    // Should be antisymmetric: a[i,j] = -a[j,i]
    assert!((a.get(&[0, 1]) + a.get(&[1, 0])).abs() < tol());
}

// ============================================================================
// Identity checks
// ============================================================================

#[test]
fn test_contraction_associativity() {
    // (A*B)*C should equal A*(B*C) for matrix multiply
    let a = Tensor::from(&FixedMatrix::from_slice(2, 2, &[fp("1"),fp("2"),fp("3"),fp("4")]));
    let b = Tensor::from(&FixedMatrix::from_slice(2, 2, &[fp("5"),fp("6"),fp("7"),fp("8")]));
    let c = Tensor::from(&FixedMatrix::from_slice(2, 2, &[fp("1"),fp("0"),fp("0"),fp("1")]));

    let ab = contract(&a, &b, &[(1, 0)]);
    let ab_c = contract(&ab, &c, &[(1, 0)]);
    let bc = contract(&b, &c, &[(1, 0)]);
    let a_bc = contract(&a, &bc, &[(1, 0)]);

    for i in 0..2 {
        for j in 0..2 {
            assert!((ab_c.get(&[i, j]) - a_bc.get(&[i, j])).abs() < tol(),
                "(AB)C != A(BC) at [{},{}]", i, j);
        }
    }
}

#[test]
fn test_symmetrize_plus_antisymmetrize_equals_original() {
    // sym(T) + antisym(T) = T
    let m = Tensor::from(&FixedMatrix::from_slice(2, 2, &[
        fp("1"), fp("2"), fp("3"), fp("4"),
    ]));
    let s = symmetrize(&m, &[0, 1]);
    let a = antisymmetrize(&m, &[0, 1]);
    for i in 0..2 {
        for j in 0..2 {
            let sum = s.get(&[i, j]) + a.get(&[i, j]);
            assert!((sum - m.get(&[i, j])).abs() < tol(),
                "sym + antisym != original at [{},{}]", i, j);
        }
    }
}
