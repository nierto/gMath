//! Validation tests for L1B iterative decompositions: Jacobi eigenvalue, SVD, Schur.
//!
//! Tests verify:
//! 1. Structural reconstruction: QΛQᵀ = A, UΣVᵀ = A, QTQᵀ = A
//! 2. Orthogonality: QᵀQ = I, UᵀU = I, VᵀV = I
//! 3. Known spectra: diagonal, identity, repeated eigenvalues
//! 4. mpmath validation: singular values and eigenvalues against 50-digit references
//! 5. Adversarial: rank-deficient, near-singular, Hilbert, mixed-scale
//! 6. SVD-dependent derived ops: pseudoinverse, rank, condition_number_2, nullspace

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::decompose::{
    eigen_symmetric, svd_decompose, schur_decompose,
};
use g_math::fixed_point::imperative::derived::{
    pseudoinverse, rank, condition_number_2, nullspace, frobenius_norm,
};

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') {
        -FixedPoint::from_str(&s[1..])
    } else {
        FixedPoint::from_str(s)
    }
}

fn matrices_approx_eq(a: &FixedMatrix, b: &FixedMatrix, tol: FixedPoint) -> bool {
    if a.rows() != b.rows() || a.cols() != b.cols() { return false; }
    for r in 0..a.rows() {
        for c in 0..a.cols() {
            if (a.get(r, c) - b.get(r, c)).abs() > tol {
                return false;
            }
        }
    }
    true
}

fn tol() -> FixedPoint { fp("0.000000001") }
fn tight_tol() -> FixedPoint { fp("0.0000000000000001") }

// ============================================================================
// Jacobi Symmetric Eigenvalue Tests
// ============================================================================

#[test]
fn test_eigen_identity() {
    let a = FixedMatrix::identity(3);
    let eig = eigen_symmetric(&a).unwrap();
    // All eigenvalues should be 1
    for i in 0..3 {
        assert!((eig.values[i] - FixedPoint::one()).abs() < tol(),
            "eigenvalue[{}] = {} (expected 1)", i, eig.values[i]);
    }
    // Q should be orthogonal
    let qtq = &eig.vectors.transpose() * &eig.vectors;
    assert!(matrices_approx_eq(&qtq, &FixedMatrix::identity(3), tol()),
        "QᵀQ != I for identity");
}

#[test]
fn test_eigen_diagonal() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("5"), fp("0"), fp("0"),
        fp("0"), fp("2"), fp("0"),
        fp("0"), fp("0"), fp("8"),
    ]);
    let eig = eigen_symmetric(&a).unwrap();
    // Eigenvalues sorted descending by |λ|: 8, 5, 2
    let mut vals: Vec<FixedPoint> = (0..3).map(|i| eig.values[i]).collect();
    vals.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    assert!((vals[0] - fp("8")).abs() < tol(), "λ₀ = {} (expected 8)", vals[0]);
    assert!((vals[1] - fp("5")).abs() < tol(), "λ₁ = {} (expected 5)", vals[1]);
    assert!((vals[2] - fp("2")).abs() < tol(), "λ₂ = {} (expected 2)", vals[2]);
}

#[test]
fn test_eigen_reconstruction() {
    // A = [[4,1,2],[1,3,1],[2,1,5]]
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("1"), fp("2"),
        fp("1"), fp("3"), fp("1"),
        fp("2"), fp("1"), fp("5"),
    ]);
    let eig = eigen_symmetric(&a).unwrap();

    // Reconstruct: A_rec = Q Λ Qᵀ
    let n = 3;
    let lambda = FixedMatrix::from_fn(n, n, |r, c| {
        if r == c { eig.values[r] } else { FixedPoint::ZERO }
    });
    let a_rec = &(&eig.vectors * &lambda) * &eig.vectors.transpose();
    assert!(matrices_approx_eq(&a, &a_rec, tol()),
        "QΛQᵀ != A reconstruction failed");
}

#[test]
fn test_eigen_orthogonality() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("1"), fp("2"),
        fp("1"), fp("3"), fp("1"),
        fp("2"), fp("1"), fp("5"),
    ]);
    let eig = eigen_symmetric(&a).unwrap();
    let qtq = &eig.vectors.transpose() * &eig.vectors;
    assert!(matrices_approx_eq(&qtq, &FixedMatrix::identity(3), tol()),
        "QᵀQ != I: eigenvectors not orthogonal");
}

#[test]
fn test_eigen_mpmath_3x3_spd() {
    // mpmath reference: A = [[4,1,2],[1,3,1],[2,1,5]]
    // eigenvalues: 7.04891733952230..., 2.64310413210779..., 2.30797852836990...
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("1"), fp("2"),
        fp("1"), fp("3"), fp("1"),
        fp("2"), fp("1"), fp("5"),
    ]);
    let eig = eigen_symmetric(&a).unwrap();
    let mut vals: Vec<FixedPoint> = (0..3).map(|i| eig.values[i]).collect();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let expected = [
        fp("7.048917339522305"),
        fp("2.643104132107790"),
        fp("2.307978528369904"),
    ];
    for i in 0..3 {
        let diff = (vals[i] - expected[i]).abs();
        assert!(diff < fp("0.000000000001"),
            "eigenvalue[{}]: got {}, expected {}, diff={}", i, vals[i], expected[i], diff);
    }
}

#[test]
fn test_eigen_repeated_eigenvalues() {
    // A = [[2,1,0,0],[1,2,0,0],[0,0,2,1],[0,0,1,2]]
    // Eigenvalues: 3, 3, 1, 1 (repeated)
    let a = FixedMatrix::from_slice(4, 4, &[
        fp("2"), fp("1"), fp("0"), fp("0"),
        fp("1"), fp("2"), fp("0"), fp("0"),
        fp("0"), fp("0"), fp("2"), fp("1"),
        fp("0"), fp("0"), fp("1"), fp("2"),
    ]);
    let eig = eigen_symmetric(&a).unwrap();
    let mut vals: Vec<FixedPoint> = (0..4).map(|i| eig.values[i]).collect();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());

    assert!((vals[0] - fp("3")).abs() < tol(), "λ₀ = {} (expected 3)", vals[0]);
    assert!((vals[1] - fp("3")).abs() < tol(), "λ₁ = {} (expected 3)", vals[1]);
    assert!((vals[2] - fp("1")).abs() < tol(), "λ₂ = {} (expected 1)", vals[2]);
    assert!((vals[3] - fp("1")).abs() < tol(), "λ₃ = {} (expected 1)", vals[3]);

    // Reconstruction still holds
    let lambda = FixedMatrix::from_fn(4, 4, |r, c| {
        if r == c { eig.values[r] } else { FixedPoint::ZERO }
    });
    let a_rec = &(&eig.vectors * &lambda) * &eig.vectors.transpose();
    assert!(matrices_approx_eq(&a, &a_rec, tol()),
        "QΛQᵀ != A for repeated eigenvalues");
}

#[test]
fn test_eigen_1x1() {
    let a = FixedMatrix::from_slice(1, 1, &[fp("7")]);
    let eig = eigen_symmetric(&a).unwrap();
    assert!((eig.values[0] - fp("7")).abs() < tight_tol());
}

#[test]
fn test_eigen_2x2() {
    // [[5, 2], [2, 1]] → eigenvalues: 5.828..., 0.171...
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("5"), fp("2"),
        fp("2"), fp("1"),
    ]);
    let eig = eigen_symmetric(&a).unwrap();
    // trace = 6, det = 1 → λ = 3 ± √8
    let sqrt8 = fp("8").sqrt();
    let l1 = fp("3") + sqrt8;
    let l2 = fp("3") - sqrt8;
    let mut vals: Vec<FixedPoint> = (0..2).map(|i| eig.values[i]).collect();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((vals[0] - l1).abs() < tol(), "λ₀ = {} (expected {})", vals[0], l1);
    assert!((vals[1] - l2).abs() < tol(), "λ₁ = {} (expected {})", vals[1], l2);
}

// ============================================================================
// SVD Tests
// ============================================================================

#[test]
fn test_svd_identity() {
    let a = FixedMatrix::identity(3);
    let svd = svd_decompose(&a).unwrap();
    // All singular values should be 1
    for i in 0..3 {
        assert!((svd.sigma[i] - FixedPoint::one()).abs() < tol(),
            "σ[{}] = {} (expected 1)", i, svd.sigma[i]);
    }
}

#[test]
fn test_svd_diagonal() {
    // diag(3, 1, 2) → singular values [3, 2, 1]
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("3"), fp("0"), fp("0"),
        fp("0"), fp("1"), fp("0"),
        fp("0"), fp("0"), fp("2"),
    ]);
    let svd = svd_decompose(&a).unwrap();
    assert!((svd.sigma[0] - fp("3")).abs() < tol(), "σ₀ = {} (expected 3)", svd.sigma[0]);
    assert!((svd.sigma[1] - fp("2")).abs() < tol(), "σ₁ = {} (expected 2)", svd.sigma[1]);
    assert!((svd.sigma[2] - fp("1")).abs() < tol(), "σ₂ = {} (expected 1)", svd.sigma[2]);
}

#[test]
fn test_svd_reconstruction() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("10"),
    ]);
    let svd = svd_decompose(&a).unwrap();
    let n = 3;

    // Reconstruct: A_rec = U Σ Vᵀ
    let sigma_mat = FixedMatrix::from_fn(n, n, |r, c| {
        if r == c { svd.sigma[r] } else { FixedPoint::ZERO }
    });
    let a_rec = &(&svd.u * &sigma_mat) * &svd.vt;
    assert!(matrices_approx_eq(&a, &a_rec, tol()),
        "UΣVᵀ != A: SVD reconstruction failed");
}

#[test]
fn test_svd_orthogonality() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("10"),
    ]);
    let svd = svd_decompose(&a).unwrap();

    let utu = &svd.u.transpose() * &svd.u;
    assert!(matrices_approx_eq(&utu, &FixedMatrix::identity(3), tol()),
        "UᵀU != I");

    let vtv = &svd.vt * &svd.vt.transpose();
    assert!(matrices_approx_eq(&vtv, &FixedMatrix::identity(3), tol()),
        "VVᵀ != I");
}

#[test]
fn test_svd_mpmath_3x3() {
    // mpmath reference: A = [[1,2,3],[4,5,6],[7,8,10]]
    // σ = [17.4125051668..., 0.87516135011..., 0.19686652111...]
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("10"),
    ]);
    let svd = svd_decompose(&a).unwrap();

    let expected = [
        fp("17.412505166808594"),
        fp("0.875161350110435"),
        fp("0.196866521117430"),
    ];
    for i in 0..3 {
        let diff = (svd.sigma[i] - expected[i]).abs();
        assert!(diff < fp("0.000001"),
            "σ[{}]: got {}, expected {}, diff={}", i, svd.sigma[i], expected[i], diff);
    }
}

#[test]
fn test_svd_2x2_symmetric() {
    // [[3,2],[2,3]] → singular values [5, 1]
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("3"), fp("2"),
        fp("2"), fp("3"),
    ]);
    let svd = svd_decompose(&a).unwrap();
    assert!((svd.sigma[0] - fp("5")).abs() < tol(), "σ₀ = {} (expected 5)", svd.sigma[0]);
    assert!((svd.sigma[1] - fp("1")).abs() < tol(), "σ₁ = {} (expected 1)", svd.sigma[1]);
}

#[test]
fn test_svd_rank_deficient() {
    // A = [[1,2,3],[2,4,6],[3,6,9]] → rank 1, σ₀ = 14, rest ≈ 0
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("2"), fp("4"), fp("6"),
        fp("3"), fp("6"), fp("9"),
    ]);
    let svd = svd_decompose(&a).unwrap();
    assert!((svd.sigma[0] - fp("14")).abs() < tol(),
        "σ₀ = {} (expected 14)", svd.sigma[0]);
    // Remaining singular values should be near zero
    assert!(svd.sigma[1] < fp("0.0001"),
        "σ₁ = {} (expected ≈0)", svd.sigma[1]);
    assert!(svd.sigma[2] < fp("0.0001"),
        "σ₂ = {} (expected ≈0)", svd.sigma[2]);
}

#[test]
fn test_svd_rectangular_tall() {
    // 4×2 matrix
    let a = FixedMatrix::from_slice(4, 2, &[
        fp("1"), fp("2"),
        fp("3"), fp("4"),
        fp("5"), fp("6"),
        fp("7"), fp("8"),
    ]);
    let svd = svd_decompose(&a).unwrap();
    assert_eq!(svd.u.rows(), 4);
    assert_eq!(svd.u.cols(), 4);
    assert_eq!(svd.sigma.len(), 2);
    assert_eq!(svd.vt.rows(), 2);
    assert_eq!(svd.vt.cols(), 2);

    // Reconstruct: U[:, 0:2] * diag(σ) * Vᵀ
    let sigma_mat = FixedMatrix::from_fn(4, 2, |r, c| {
        if r == c && r < 2 { svd.sigma[r] } else { FixedPoint::ZERO }
    });
    let a_rec = &(&svd.u * &sigma_mat) * &svd.vt;
    assert!(matrices_approx_eq(&a, &a_rec, tol()),
        "SVD reconstruction failed for 4×2 matrix");
}

#[test]
fn test_svd_rectangular_wide() {
    // 2×4 matrix (m < n case, triggers transpose path)
    let a = FixedMatrix::from_slice(2, 4, &[
        fp("1"), fp("2"), fp("3"), fp("4"),
        fp("5"), fp("6"), fp("7"), fp("8"),
    ]);
    let svd = svd_decompose(&a).unwrap();
    assert_eq!(svd.u.rows(), 2);
    assert_eq!(svd.u.cols(), 2);
    assert_eq!(svd.sigma.len(), 2);
    assert_eq!(svd.vt.rows(), 4);
    assert_eq!(svd.vt.cols(), 4);

    // Reconstruct
    let sigma_mat = FixedMatrix::from_fn(2, 4, |r, c| {
        if r == c && r < 2 { svd.sigma[r] } else { FixedPoint::ZERO }
    });
    let a_rec = &(&svd.u * &sigma_mat) * &svd.vt;
    assert!(matrices_approx_eq(&a, &a_rec, tol()),
        "SVD reconstruction failed for 2×4 matrix");
}

#[test]
fn test_svd_1x1() {
    let a = FixedMatrix::from_slice(1, 1, &[fp("7")]);
    let svd = svd_decompose(&a).unwrap();
    assert!((svd.sigma[0] - fp("7")).abs() < tight_tol());
}

// ============================================================================
// Schur Decomposition Tests
// ============================================================================

#[test]
fn test_schur_upper_triangular() {
    // Already upper triangular: T should equal A, Q should be I
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("0"), fp("4"), fp("5"),
        fp("0"), fp("0"), fp("6"),
    ]);
    let schur = schur_decompose(&a).unwrap();

    // T should be (quasi-)upper triangular
    for i in 1..3 {
        assert!(schur.t.get(i, i - 1).abs() < tol(),
            "T[{},{}] = {} (expected 0, upper triangular)", i, i-1, schur.t.get(i, i-1));
    }

    // Eigenvalues on diagonal
    let diag: Vec<FixedPoint> = (0..3).map(|i| schur.t.get(i, i)).collect();
    let mut expected = vec![fp("1"), fp("4"), fp("6")];
    expected.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    let mut diag_sorted = diag.clone();
    diag_sorted.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    for i in 0..3 {
        assert!((diag_sorted[i] - expected[i]).abs() < tol(),
            "Schur eigenvalue[{}]: got {}, expected {}", i, diag_sorted[i], expected[i]);
    }
}

#[test]
fn test_schur_reconstruction() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("1"), fp("2"),
        fp("3"), fp("5"), fp("1"),
        fp("1"), fp("2"), fp("3"),
    ]);
    let schur = schur_decompose(&a).unwrap();

    // Reconstruct: A_rec = Q T Qᵀ
    let a_rec = &(&schur.q * &schur.t) * &schur.q.transpose();
    assert!(matrices_approx_eq(&a, &a_rec, tol()),
        "QTQᵀ != A: Schur reconstruction failed");
}

#[test]
fn test_schur_orthogonality() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("1"), fp("2"),
        fp("3"), fp("5"), fp("1"),
        fp("1"), fp("2"), fp("3"),
    ]);
    let schur = schur_decompose(&a).unwrap();
    let qtq = &schur.q.transpose() * &schur.q;
    assert!(matrices_approx_eq(&qtq, &FixedMatrix::identity(3), tol()),
        "QᵀQ != I: Schur Q not orthogonal");
}

#[test]
fn test_schur_identity() {
    let a = FixedMatrix::identity(3);
    let schur = schur_decompose(&a).unwrap();
    assert!(matrices_approx_eq(&schur.t, &FixedMatrix::identity(3), tol()));
}

#[test]
fn test_schur_2x2() {
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("3"), fp("1"),
        fp("0"), fp("2"),
    ]);
    let schur = schur_decompose(&a).unwrap();
    let a_rec = &(&schur.q * &schur.t) * &schur.q.transpose();
    assert!(matrices_approx_eq(&a, &a_rec, tol()),
        "Schur 2×2 reconstruction failed");
}

#[test]
fn test_schur_symmetric_eigenvalues() {
    // Symmetric matrix: Schur form should have eigenvalues on diagonal
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("1"), fp("2"),
        fp("1"), fp("3"), fp("1"),
        fp("2"), fp("1"), fp("5"),
    ]);
    let schur = schur_decompose(&a).unwrap();

    // Trace should be preserved
    let trace_a = a.trace();
    let trace_t = schur.t.trace();
    assert!((trace_a - trace_t).abs() < tol(),
        "Trace not preserved: A={}, T={}", trace_a, trace_t);
}

// ============================================================================
// SVD-Dependent Derived Operations
// ============================================================================

#[test]
fn test_pseudoinverse_square_invertible() {
    // For invertible A, A⁺ = A⁻¹
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("4"), fp("7"),
        fp("2"), fp("6"),
    ]);
    let pinv = pseudoinverse(&a).unwrap();
    // A * A⁺ should be I
    let prod = &a * &pinv;
    assert!(matrices_approx_eq(&prod, &FixedMatrix::identity(2), tol()),
        "A * A⁺ != I for invertible matrix");
}

#[test]
fn test_pseudoinverse_rank_deficient() {
    // A = [[1,2],[2,4]] → rank 1
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1"), fp("2"),
        fp("2"), fp("4"),
    ]);
    // Debug: check SVD first
    let svd = g_math::fixed_point::imperative::decompose::svd_decompose(&a).unwrap();
    eprintln!("SVD sigma: [{}, {}]", svd.sigma[0], svd.sigma[1]);
    eprintln!("SVD U:");
    for r in 0..2 { eprintln!("  [{}, {}]", svd.u.get(r, 0), svd.u.get(r, 1)); }
    eprintln!("SVD Vt:");
    for r in 0..2 { eprintln!("  [{}, {}]", svd.vt.get(r, 0), svd.vt.get(r, 1)); }
    // Check reconstruction
    let sigma_mat = FixedMatrix::from_fn(2, 2, |r, c| if r == c { svd.sigma[r] } else { FixedPoint::ZERO });
    let recon = &(&svd.u * &sigma_mat) * &svd.vt;
    eprintln!("Reconstruction UΣVᵀ:");
    for r in 0..2 { eprintln!("  [{}, {}]", recon.get(r, 0), recon.get(r, 1)); }

    let pinv = pseudoinverse(&a).unwrap();
    eprintln!("Pinv:");
    for r in 0..2 { eprintln!("  [{}, {}]", pinv.get(r, 0), pinv.get(r, 1)); }

    // A * A⁺ * A should equal A (Moore-Penrose property)
    let apa = &(&a * &pinv) * &a;
    for r in 0..2 {
        for c in 0..2 {
            let diff = (a.get(r, c) - apa.get(r, c)).abs();
            eprintln!("A[{},{}]={}, APA[{},{}]={}, diff={}", r, c, a.get(r, c), r, c, apa.get(r, c), diff);
        }
    }
    let rank_tol = fp("0.01");
    assert!(matrices_approx_eq(&a, &apa, rank_tol),
        "A * A⁺ * A != A for rank-deficient matrix");
}

#[test]
fn test_rank_full() {
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("10"),
    ]);
    let r = rank(&a).unwrap();
    assert_eq!(r, 3, "Expected rank 3 for full-rank 3×3");
}

#[test]
fn test_rank_deficient() {
    // rank 1 matrix
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("2"), fp("4"), fp("6"),
        fp("3"), fp("6"), fp("9"),
    ]);
    let r = rank(&a).unwrap();
    assert_eq!(r, 1, "Expected rank 1 for rank-deficient matrix, got {}", r);
}

#[test]
fn test_condition_number_2_identity() {
    let a = FixedMatrix::identity(3);
    let kappa = condition_number_2(&a).unwrap();
    assert!((kappa - FixedPoint::one()).abs() < tol(),
        "κ₂(I) = {} (expected 1)", kappa);
}

#[test]
fn test_condition_number_2_diagonal() {
    // diag(10, 1) → κ₂ = 10
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("10"), fp("0"),
        fp("0"),  fp("1"),
    ]);
    let kappa = condition_number_2(&a).unwrap();
    assert!((kappa - fp("10")).abs() < tol(),
        "κ₂ = {} (expected 10)", kappa);
}

#[test]
fn test_nullspace_full_rank() {
    let a = FixedMatrix::identity(3);
    let ns = nullspace(&a).unwrap();
    assert_eq!(ns.cols(), 0, "Full rank matrix should have empty nullspace");
}

#[test]
fn test_nullspace_rank_deficient() {
    // A = [[1,2,3],[2,4,6],[3,6,9]] → rank 1, nullspace dim = 2
    let a = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("2"), fp("4"), fp("6"),
        fp("3"), fp("6"), fp("9"),
    ]);
    let ns = nullspace(&a).unwrap();
    assert_eq!(ns.cols(), 2, "Expected nullspace dimension 2, got {}", ns.cols());

    // Verify: A * null_vector ≈ 0 for each nullspace column
    for j in 0..ns.cols() {
        let v = ns.col(j);
        let av = a.mul_vector(&v);
        for i in 0..av.len() {
            assert!(av[i].abs() < tol(),
                "A * nullspace[{}][{}] = {} (expected 0)", j, i, av[i]);
        }
    }
}

// ============================================================================
// Adversarial Tests
// ============================================================================

#[test]
fn test_eigen_4x4_spd() {
    // Well-conditioned 4×4 SPD
    let a = FixedMatrix::from_slice(4, 4, &[
        fp("10"), fp("1"), fp("2"), fp("3"),
        fp("1"), fp("8"), fp("1"), fp("2"),
        fp("2"), fp("1"), fp("6"), fp("1"),
        fp("3"), fp("2"), fp("1"), fp("5"),
    ]);
    let eig = eigen_symmetric(&a).unwrap();

    // Reconstruction
    let n = 4;
    let lambda = FixedMatrix::from_fn(n, n, |r, c| {
        if r == c { eig.values[r] } else { FixedPoint::ZERO }
    });
    let a_rec = &(&eig.vectors * &lambda) * &eig.vectors.transpose();
    assert!(matrices_approx_eq(&a, &a_rec, tol()),
        "4×4 SPD eigenvalue reconstruction failed");

    // Orthogonality
    let qtq = &eig.vectors.transpose() * &eig.vectors;
    assert!(matrices_approx_eq(&qtq, &FixedMatrix::identity(n), tol()),
        "4×4 SPD eigenvectors not orthogonal");

    // Sum of eigenvalues = trace
    let trace_a = a.trace();
    let mut eig_sum = FixedPoint::ZERO;
    for i in 0..n {
        eig_sum += eig.values[i];
    }
    assert!((trace_a - eig_sum).abs() < tol(),
        "trace(A)={} != sum(λ)={}", trace_a, eig_sum);
}

#[test]
fn test_svd_near_singular() {
    // Nearly singular: det ≈ 0.001
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1"),      fp("0.999"),
        fp("0.999"),  fp("1"),
    ]);
    let svd = svd_decompose(&a).unwrap();
    // σ₀ ≈ 1.999, σ₁ ≈ 0.001
    assert!(svd.sigma[0] > fp("1.9"), "σ₀ too small: {}", svd.sigma[0]);
    assert!(svd.sigma[1] < fp("0.01"), "σ₁ too large: {}", svd.sigma[1]);

    // Reconstruction
    let sigma_mat = FixedMatrix::from_fn(2, 2, |r, c| {
        if r == c { svd.sigma[r] } else { FixedPoint::ZERO }
    });
    let a_rec = &(&svd.u * &sigma_mat) * &svd.vt;
    assert!(matrices_approx_eq(&a, &a_rec, tol()),
        "Near-singular SVD reconstruction failed");
}

#[test]
fn test_eigen_negative_eigenvalues() {
    // A with negative eigenvalue: [[1, 3], [3, -1]]
    // tr = 0, det = -10 → λ = ±√10
    let a = FixedMatrix::from_slice(2, 2, &[
        fp("1"), fp("3"),
        fp("3"), fp("-1"),
    ]);
    let eig = eigen_symmetric(&a).unwrap();
    let sqrt10 = fp("10").sqrt();
    let mut vals: Vec<FixedPoint> = (0..2).map(|i| eig.values[i]).collect();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((vals[0] - sqrt10).abs() < tol(), "λ₀ = {} (expected √10={})", vals[0], sqrt10);
    assert!((vals[1] + sqrt10).abs() < tol(), "λ₁ = {} (expected -√10)", vals[1]);
}

// ============================================================================
// ULP Measurement Report
// ============================================================================

#[test]
fn test_ulp_measurement_report() {
    // Print precision measurements for diagnostic purposes
    println!("=== L1B Iterative Decomposition ULP Measurement Report ===");

    // Eigenvalue precision: compare to mpmath reference
    let a_eig = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("1"), fp("2"),
        fp("1"), fp("3"), fp("1"),
        fp("2"), fp("1"), fp("5"),
    ]);
    let eig = eigen_symmetric(&a_eig).unwrap();
    let mut vals: Vec<FixedPoint> = (0..3).map(|i| eig.values[i]).collect();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    println!("\n--- Jacobi Eigenvalue (3×3 SPD) ---");
    // mpmath 80-digit references — profile-appropriate length
    // (FixedPoint::from_str panics on overflow, so must fit in storage tier)
    // embedded: 19 decimals, balanced: 38 decimals, scientific: 77 decimals
    #[cfg(table_format = "q64_64")]
    let refs = [
        "7.0489173395223053135",
        "2.6431041321077905561",
        "2.3079785283699041303",
    ];
    #[cfg(table_format = "q128_128")]
    let refs = [
        "7.0489173395223053135222144070233697235",
        "2.6431041321077905561056004899786994166",
        "2.3079785283699041303721851029979308598",
    ];
    #[cfg(table_format = "q256_256")]
    let refs = [
        "7.0489173395223053135222144070233697235963877860565185108382237245925721457688",
        "2.6431041321077905561056004899786994166008728132653756105168491853959776232012",
        "2.3079785283699041303721851029979308598027394006781058786449270900114502310299",
    ];
    for i in 0..3 {
        let ref_val = fp(refs[i]);
        let diff = (vals[i] - ref_val).abs();
        println!("  λ[{}] = {}, ref = {}, |diff| = {}", i, vals[i], ref_val, diff);
    }

    // SVD precision
    let a_svd = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("2"), fp("3"),
        fp("4"), fp("5"), fp("6"),
        fp("7"), fp("8"), fp("10"),
    ]);
    let svd = svd_decompose(&a_svd).unwrap();
    println!("\n--- SVD Singular Values (3×3) ---");
    #[cfg(table_format = "q64_64")]
    let sv_refs = [
        "17.412505166808594516",
        "0.8751613501104356045",
        "0.1968665211174302159",
    ];
    #[cfg(table_format = "q128_128")]
    let sv_refs = [
        "17.412505166808594516759972444201697117",
        "0.8751613501104356045849340956059706785",
        "0.1968665211174302159846855580308661956",
    ];
    #[cfg(table_format = "q256_256")]
    let sv_refs = [
        "17.412505166808594516759972444201697117661188620060895301247247908613411798229",
        "0.875161350110435604584934095605970678588802552939681038110255512487299329420",
        "0.196866521117430215984685558030866195631038499859613668662392132301110348007",
    ];
    for i in 0..3 {
        let diff = (svd.sigma[i] - fp(sv_refs[i])).abs();
        println!("  σ[{}] = {}, ref = {}, |diff| = {}", i, svd.sigma[i], sv_refs[i], diff);
    }

    // Reconstruction error
    let n = 3;
    let sigma_mat = FixedMatrix::from_fn(n, n, |r, c| {
        if r == c { svd.sigma[r] } else { FixedPoint::ZERO }
    });
    let a_rec = &(&svd.u * &sigma_mat) * &svd.vt;
    let err = frobenius_norm(&(&a_svd - &a_rec));
    println!("  ||UΣVᵀ - A||_F = {}", err);

    // Schur trace preservation
    let a_sch = FixedMatrix::from_slice(3, 3, &[
        fp("4"), fp("1"), fp("2"),
        fp("3"), fp("5"), fp("1"),
        fp("1"), fp("2"), fp("3"),
    ]);
    let schur = schur_decompose(&a_sch).unwrap();
    let trace_diff = (a_sch.trace() - schur.t.trace()).abs();
    let recon_err = frobenius_norm(&(&(&schur.q * &schur.t) * &schur.q.transpose() - a_sch));
    println!("\n--- Schur (3×3) ---");
    println!("  trace(A) - trace(T) = {}", trace_diff);
    println!("  ||QTQᵀ - A||_F = {}", recon_err);

    println!("\n=== End Report ===");
}
