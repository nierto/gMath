//! L3B: Christoffel symbols and curvature validation tests.
//!
//! Tests verify:
//! 1. Flat space: all Christoffel symbols and curvature = 0
//! 2. Sphere S²: sectional curvature = 1/r² everywhere
//! 3. Hyperbolic H²: scalar curvature = -2 (upper half-plane model)
//! 4. Christoffel symmetry: Γ^k_{ij} = Γ^k_{ji} (torsion-free)
//! 5. Bianchi identity (first): R^l_{[ijk]} = 0
//! 6. mpmath-validated Christoffel values for sphere
//!
//! All tests run on the active profile (embedded/balanced/scientific).

use g_math::fixed_point::{FixedPoint, FixedVector};
use g_math::fixed_point::imperative::curvature::*;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn tol() -> FixedPoint { fp("0.01") }
fn curv_tol() -> FixedPoint {
    // Curvature involves nested numerical derivatives — tolerance is looser
    #[cfg(table_format = "q16_16")]
    { fp("1") }
    #[cfg(table_format = "q32_32")]
    { fp("0.5") }
    #[cfg(table_format = "q64_64")]
    { fp("0.1") }
    #[cfg(table_format = "q128_128")]
    { fp("0.001") }
    #[cfg(table_format = "q256_256")]
    { fp("0.0000001") }
}

fn assert_fp(got: FixedPoint, exp: FixedPoint, tol: FixedPoint, name: &str) {
    let d = (got - exp).abs();
    assert!(d < tol, "{}: got {}, expected {}, diff={}", name, got, exp, d);
}

// ============================================================================
// Flat Euclidean space: all curvature = 0
// ============================================================================

#[test]
fn test_euclidean_christoffel_zero() {
    let metric = EuclideanMetric { dim: 3 };
    let p = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);

    let gamma = christoffel(&metric, &p).unwrap();

    // All Christoffel symbols should be zero for flat space
    for k in 0..3 {
        for i in 0..3 {
            for j in 0..3 {
                assert_fp(gamma.get(&[k, i, j]), fp("0"), tol(),
                    &format!("Γ^{}_{{{}{}}} should be 0", k, i, j));
            }
        }
    }
}

#[test]
fn test_euclidean_riemann_zero() {
    let metric = EuclideanMetric { dim: 2 };
    let p = FixedVector::from_slice(&[fp("1"), fp("2")]);

    let riemann = riemann_curvature(&metric, &p).unwrap();

    // All components should be zero
    for l in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    assert_fp(riemann.get(&[l, i, j, k]), fp("0"), tol(),
                        &format!("R^{}_{{{}{}{}}} should be 0", l, i, j, k));
                }
            }
        }
    }
}

#[test]
fn test_euclidean_scalar_curvature_zero() {
    let metric = EuclideanMetric { dim: 2 };
    let p = FixedVector::from_slice(&[fp("1"), fp("2")]);

    let r = scalar_curvature(&metric, &p).unwrap();
    assert_fp(r, fp("0"), tol(), "Euclidean scalar curvature");
}

// ============================================================================
// Sphere S² with radius r: sectional curvature = 1/r²
// ============================================================================

#[test]
fn test_sphere_christoffel_symmetry() {
    let metric = SphereMetric { radius: fp("1") };
    // Point: (θ, φ) = (π/4, π/3)
    let p = FixedVector::from_slice(&[fp("0.7853981633974483"), fp("1.0471975511965976")]);

    let gamma = christoffel(&metric, &p).unwrap();

    // Torsion-free: Γ^k_{ij} = Γ^k_{ji}
    for k in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                assert_fp(gamma.get(&[k, i, j]), gamma.get(&[k, j, i]), tol(),
                    &format!("Torsion-free: Γ^{}_{{{}{}}} = Γ^{}_{{{}{}}}",
                        k, i, j, k, j, i));
            }
        }
    }
}

#[test]
fn test_sphere_christoffel_mpmath() {
    // Unit sphere S² in (θ, φ) coords:
    //   g = [[1, 0], [0, sin²θ]]
    //
    // Exact Christoffel symbols:
    //   Γ^θ_{φφ} = -sin(θ)cos(θ)
    //   Γ^φ_{θφ} = Γ^φ_{φθ} = cos(θ)/sin(θ) = cot(θ)
    //   All others = 0
    //
    // At θ = π/4: sin(π/4) = cos(π/4) = √2/2
    //   Γ^0_{11} = -sin(π/4)*cos(π/4) = -1/2
    //   Γ^1_{01} = cot(π/4) = 1
    //
    // mpmath 50 digits at θ=π/4:
    //   -sin(pi/4)*cos(pi/4) = -0.5
    //   cos(pi/4)/sin(pi/4) = 1.0

    let metric = SphereMetric { radius: fp("1") };
    let theta = fp("0.7853981633974483"); // π/4
    let phi = fp("1"); // arbitrary
    let p = FixedVector::from_slice(&[theta, phi]);

    let gamma = christoffel(&metric, &p).unwrap();

    // Γ^0_{11} = -sin(θ)cos(θ) = -0.5 at θ=π/4
    assert_fp(gamma.get(&[0, 1, 1]), fp("-0.5"), curv_tol(),
        "Γ^θ_{φφ} at θ=π/4");

    // Γ^1_{01} = cot(θ) = 1.0 at θ=π/4
    assert_fp(gamma.get(&[1, 0, 1]), fp("1"), curv_tol(),
        "Γ^φ_{θφ} at θ=π/4");

    // Γ^0_{00}, Γ^0_{01}, Γ^1_{00}, Γ^1_{11} should be ≈ 0
    assert_fp(gamma.get(&[0, 0, 0]), fp("0"), curv_tol(), "Γ^0_{00}");
    assert_fp(gamma.get(&[0, 0, 1]), fp("0"), curv_tol(), "Γ^0_{01}");
    assert_fp(gamma.get(&[1, 0, 0]), fp("0"), curv_tol(), "Γ^1_{00}");
    assert_fp(gamma.get(&[1, 1, 1]), fp("0"), curv_tol(), "Γ^1_{11}");
}

#[test]
fn test_sphere_scalar_curvature() {
    // Unit sphere S²: scalar curvature R = 2 (= n(n-1)/r² for S^n, n=2, r=1)
    //
    // mpmath: For S² with radius r, R = 2/r². For r=1, R = 2.

    let metric = SphereMetric { radius: fp("1") };
    // Evaluate at θ = π/3 (away from poles for numerical stability)
    let p = FixedVector::from_slice(&[fp("1.0471975511965976"), fp("1")]);

    let r = scalar_curvature(&metric, &p).unwrap();

    // Tolerance is generous because of nested numerical derivatives
    assert_fp(r, fp("2"), curv_tol(), "S² scalar curvature = 2");
}

// ============================================================================
// Hyperbolic H² (upper half-plane): scalar curvature = -2
// ============================================================================

#[test]
fn test_hyperbolic_scalar_curvature() {
    // Upper half-plane model: g = (1/y²)·I
    // Scalar curvature R = -2 everywhere
    //
    // mpmath: For H² in the Poincaré half-plane, R = -2.

    let metric = HyperbolicMetric;
    let p = FixedVector::from_slice(&[fp("0.5"), fp("1")]); // (x, y) with y=1

    let r = scalar_curvature(&metric, &p).unwrap();
    assert_fp(r, fp("-2"), curv_tol(), "H² scalar curvature = -2");
}

#[test]
fn test_hyperbolic_christoffel_mpmath() {
    // H² upper half-plane: g = (1/y²)·I
    //
    // Exact Christoffel symbols at (x, y):
    //   Γ^0_{01} = Γ^0_{10} = -1/y
    //   Γ^1_{00} = 1/y
    //   Γ^1_{11} = -1/y
    //   Γ^0_{00} = 0
    //   Γ^1_{01} = 0
    //
    // At y = 2: all nonzero = ±0.5

    let metric = HyperbolicMetric;
    let p = FixedVector::from_slice(&[fp("0"), fp("2")]);

    let gamma = christoffel(&metric, &p).unwrap();

    // Γ^0_{01} = -1/y = -0.5
    assert_fp(gamma.get(&[0, 0, 1]), fp("-0.5"), curv_tol(), "Γ^x_{xy} = -1/y");
    // Γ^1_{00} = 1/y = 0.5
    assert_fp(gamma.get(&[1, 0, 0]), fp("0.5"), curv_tol(), "Γ^y_{xx} = 1/y");
    // Γ^1_{11} = -1/y = -0.5
    assert_fp(gamma.get(&[1, 1, 1]), fp("-0.5"), curv_tol(), "Γ^y_{yy} = -1/y");
    // Zero entries
    assert_fp(gamma.get(&[0, 0, 0]), fp("0"), curv_tol(), "Γ^x_{xx} = 0");
    assert_fp(gamma.get(&[0, 1, 1]), fp("0"), curv_tol(), "Γ^x_{yy} = 0");
}

// ============================================================================
// Riemann tensor symmetries
// ============================================================================

#[test]
fn test_riemann_antisymmetry() {
    // R^l_{ijk} = -R^l_{ikj} (antisymmetric in last two indices)
    let metric = SphereMetric { radius: fp("1") };
    let p = FixedVector::from_slice(&[fp("1"), fp("1")]);

    let riemann = riemann_curvature(&metric, &p).unwrap();

    for l in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let val_jk = riemann.get(&[l, i, j, k]);
                    let val_kj = riemann.get(&[l, i, k, j]);
                    assert_fp(val_jk + val_kj, fp("0"), curv_tol(),
                        &format!("R^{}_{{{}{}{}}} + R^{}_{{{}{}{}}} = 0",
                            l, i, j, k, l, i, k, j));
                }
            }
        }
    }
}

#[test]
fn test_first_bianchi_identity() {
    // R^l_{ijk} + R^l_{jki} + R^l_{kij} = 0 (first Bianchi identity)
    let metric = SphereMetric { radius: fp("1") };
    let p = FixedVector::from_slice(&[fp("1"), fp("0.5")]);

    let riemann = riemann_curvature(&metric, &p).unwrap();

    for l in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let cyclic_sum = riemann.get(&[l, i, j, k])
                        + riemann.get(&[l, j, k, i])
                        + riemann.get(&[l, k, i, j]);
                    assert_fp(cyclic_sum, fp("0"), curv_tol(),
                        &format!("Bianchi: R^{}_{{{}{}{}}} + cyc = 0", l, i, j, k));
                }
            }
        }
    }
}

// ============================================================================
// Ricci tensor
// ============================================================================

#[test]
fn test_ricci_from_riemann_consistency() {
    let metric = SphereMetric { radius: fp("1") };
    let p = FixedVector::from_slice(&[fp("1"), fp("1")]);

    let riemann = riemann_curvature(&metric, &p).unwrap();
    let ricci = ricci_tensor(&metric, &p).unwrap();
    let ricci2 = ricci_from_riemann(&riemann, 2);

    for i in 0..2 {
        for j in 0..2 {
            assert_fp(ricci.get(i, j), ricci2.get(i, j), tol(),
                &format!("Ricci consistency [{},{}]", i, j));
        }
    }
}

// ============================================================================
// Differentiation step validation
// ============================================================================

#[test]
fn test_differentiation_step_positive() {
    let h = differentiation_step();
    assert!(!h.is_zero(), "differentiation_step should be nonzero");
    assert!(!h.is_negative(), "differentiation_step should be positive");
}

// ============================================================================
// Scalar curvature from Ricci
// ============================================================================

#[test]
fn test_scalar_from_ricci_consistency() {
    let metric = SphereMetric { radius: fp("1") };
    let p = FixedVector::from_slice(&[fp("1.0471975511965976"), fp("1")]);

    let g_inv = metric.metric_inverse(&p).unwrap();
    let ricci = ricci_tensor(&metric, &p).unwrap();
    let r1 = scalar_curvature(&metric, &p).unwrap();
    let r2 = scalar_from_ricci(&g_inv, &ricci);

    assert_fp(r1, r2, tol(), "scalar curvature consistency");
}
