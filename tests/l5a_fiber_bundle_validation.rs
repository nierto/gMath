//! L5A: Fiber bundle validation tests.
//!
//! Tests verify:
//! 1. Trivial bundle: project ∘ lift = identity
//! 2. Trivial bundle: flat parallel transport preserves fiber element
//! 3. Vector bundle: connection parallel transport modifies fiber correctly
//! 4. Principal bundle: cocycle condition g_{αβ} · g_{βγ} = g_{αγ}
//! 5. Associated bundle: chart change via transition functions
//! 6. Bundle curvature: flat connection → zero curvature
//! 7. Bundle curvature: non-trivial connection → nonzero curvature
//!
//! All tests run on the active profile (embedded/balanced/scientific).

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::fiber_bundle::*;

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

// ============================================================================
// Trivial bundle
// ============================================================================

#[test]
fn test_trivial_project_lift_roundtrip() {
    let bundle = TrivialBundle { base_dimension: 3, fiber_dimension: 2 };
    let base = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let fiber = FixedVector::from_slice(&[fp("4"), fp("5")]);

    let total = bundle.lift(&base, &fiber);
    assert_eq!(total.len(), 5);

    let (base_back, fiber_back) = bundle.local_trivialization(&total);
    for i in 0..3 {
        assert_fp(base_back[i], base[i], tight(), &format!("base[{}]", i));
    }
    for i in 0..2 {
        assert_fp(fiber_back[i], fiber[i], tight(), &format!("fiber[{}]", i));
    }
}

#[test]
fn test_trivial_project_is_base() {
    let bundle = TrivialBundle { base_dimension: 2, fiber_dimension: 3 };
    let total = FixedVector::from_slice(&[fp("1"), fp("2"), fp("10"), fp("20"), fp("30")]);

    let base = bundle.project(&total);
    assert_eq!(base.len(), 2);
    assert_fp(base[0], fp("1"), tight(), "project[0]");
    assert_fp(base[1], fp("2"), tight(), "project[1]");
}

#[test]
fn test_trivial_flat_transport() {
    let bundle = TrivialBundle { base_dimension: 2, fiber_dimension: 2 };
    let fiber = FixedVector::from_slice(&[fp("7"), fp("-3")]);

    // Transport along any path should preserve the fiber element
    let path = vec![
        FixedVector::from_slice(&[fp("0"), fp("0")]),
        FixedVector::from_slice(&[fp("1"), fp("0")]),
        FixedVector::from_slice(&[fp("1"), fp("1")]),
        FixedVector::from_slice(&[fp("2"), fp("1")]),
    ];

    let result = bundle.parallel_transport_along(&path, &fiber).unwrap();
    for i in 0..2 {
        assert_fp(result[i], fiber[i], tight(), &format!("flat_transport[{}]", i));
    }
}

#[test]
fn test_trivial_horizontal_lift() {
    let bundle = TrivialBundle { base_dimension: 2, fiber_dimension: 3 };
    let total_point = FixedVector::from_slice(&[fp("1"), fp("2"), fp("5"), fp("6"), fp("7")]);
    let base_tangent = FixedVector::from_slice(&[fp("0.1"), fp("-0.2")]);

    let h_lift = bundle.horizontal_lift(&total_point, &base_tangent).unwrap();

    // Horizontal lift of (v, 0): base part = v, fiber part = 0
    assert_fp(h_lift[0], fp("0.1"), tight(), "h_lift base[0]");
    assert_fp(h_lift[1], fp("-0.2"), tight(), "h_lift base[1]");
    assert_fp(h_lift[2], fp("0"), tight(), "h_lift fiber[0]");
    assert_fp(h_lift[3], fp("0"), tight(), "h_lift fiber[1]");
    assert_fp(h_lift[4], fp("0"), tight(), "h_lift fiber[2]");
}

#[test]
fn test_trivial_vertical_component() {
    let bundle = TrivialBundle { base_dimension: 2, fiber_dimension: 2 };
    let total_point = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3"), fp("4")]);
    let tangent = FixedVector::from_slice(&[fp("0.5"), fp("-0.3"), fp("0.7"), fp("1.2")]);

    let vert = bundle.vertical_component(&total_point, &tangent);

    // Vertical component: (0, 0, fiber_part)
    assert_fp(vert[0], fp("0"), tight(), "vert base[0]");
    assert_fp(vert[1], fp("0"), tight(), "vert base[1]");
    assert_fp(vert[2], fp("0.7"), tight(), "vert fiber[0]");
    assert_fp(vert[3], fp("1.2"), tight(), "vert fiber[1]");
}

// ============================================================================
// Vector bundle
// ============================================================================

#[test]
fn test_vector_bundle_flat() {
    let bundle = VectorBundle::flat(2, 2);
    let base = FixedVector::from_slice(&[fp("1"), fp("2")]);
    let fiber = FixedVector::from_slice(&[fp("3"), fp("4")]);

    let total = bundle.lift(&base, &fiber);
    let (b, f) = bundle.local_trivialization(&total);

    assert_fp(b[0], fp("1"), tight(), "vb flat base[0]");
    assert_fp(b[1], fp("2"), tight(), "vb flat base[1]");
    assert_fp(f[0], fp("3"), tight(), "vb flat fiber[0]");
    assert_fp(f[1], fp("4"), tight(), "vb flat fiber[1]");
}

#[test]
fn test_vector_bundle_flat_transport() {
    let bundle = VectorBundle::flat(2, 2);
    let fiber = FixedVector::from_slice(&[fp("1"), fp("0")]);

    let path = vec![
        FixedVector::from_slice(&[fp("0"), fp("0")]),
        FixedVector::from_slice(&[fp("1"), fp("1")]),
    ];

    let result = bundle.parallel_transport_along(&path, &fiber).unwrap();
    assert_fp(result[0], fp("1"), tight(), "vb flat transport[0]");
    assert_fp(result[1], fp("0"), tight(), "vb flat transport[1]");
}

#[test]
fn test_vector_bundle_connection_transport() {
    // R² base, R¹ fiber, connection A^0_{0,0}=0.1, A^0_{0,1}=0.2
    // Parallel transport: ξ_{n+1} = ξ_n - A^a_{bi} ξ^b Δx^i
    //
    // Start at fiber=1.0, move by (1, 0):
    //   ξ_new = 1 - 0.1*1*1 = 0.9
    let coeffs = vec![fp("0.1"), fp("0.2")]; // k*k*n = 1*1*2
    let bundle = VectorBundle::with_connection(2, 1, coeffs);

    let path = vec![
        FixedVector::from_slice(&[fp("0"), fp("0")]),
        FixedVector::from_slice(&[fp("1"), fp("0")]),
    ];
    let fiber = FixedVector::from_slice(&[fp("1")]);

    let result = bundle.parallel_transport_along(&path, &fiber).unwrap();
    assert_fp(result[0], fp("0.9"), tol(), "connection transport dx=(1,0)");

    // Move by (0, 1):
    //   ξ_new = 1 - 0.2*1*1 = 0.8
    let path2 = vec![
        FixedVector::from_slice(&[fp("0"), fp("0")]),
        FixedVector::from_slice(&[fp("0"), fp("1")]),
    ];
    let result2 = bundle.parallel_transport_along(&path2, &fiber).unwrap();
    assert_fp(result2[0], fp("0.8"), tol(), "connection transport dx=(0,1)");
}

#[test]
fn test_vector_bundle_horizontal_lift_nonflat() {
    // With connection A^0_{0,0}=0.5, fiber=2:
    // horizontal_lift of base tangent (1,0) at fiber ξ=2:
    //   base part = (1, 0)
    //   fiber part = -A^0_{00}*ξ*v^0 = -0.5*2*1 = -1.0
    let coeffs = vec![fp("0.5"), fp("0")]; // A^0_{0,0}=0.5, A^0_{0,1}=0
    let bundle = VectorBundle::with_connection(2, 1, coeffs);

    let total_point = FixedVector::from_slice(&[fp("0"), fp("0"), fp("2")]); // base=(0,0), fiber=2
    let base_tangent = FixedVector::from_slice(&[fp("1"), fp("0")]);

    let h_lift = bundle.horizontal_lift(&total_point, &base_tangent).unwrap();
    assert_fp(h_lift[0], fp("1"), tol(), "h_lift base[0]");
    assert_fp(h_lift[1], fp("0"), tol(), "h_lift base[1]");
    assert_fp(h_lift[2], fp("-1"), tol(), "h_lift fiber = -A*ξ*v");
}

// ============================================================================
// Principal bundle
// ============================================================================

#[test]
fn test_principal_trivial_cocycle() {
    let bundle = PrincipalBundle::trivial(2, 1, 2, 3); // 3 charts, SO(2) group

    let (ok, max_err) = bundle.verify_cocycle(tol());
    assert!(ok, "Trivial bundle should satisfy cocycle, max_err={}", max_err);
    assert_fp(max_err, fp("0"), tight(), "Trivial cocycle error");
}

#[test]
fn test_principal_set_transition_cocycle() {
    // 3 charts with SO(2) rotations as transition functions
    // g_{01} = R(π/4), g_{12} = R(π/6)
    // Then g_{02} = g_{01} * g_{12} = R(π/4 + π/6) = R(5π/12)
    //
    // For cocycle: g_{01}*g_{12} should equal g_{02}

    let mut bundle = PrincipalBundle::trivial(2, 1, 2, 3);

    // R(π/4)
    let cos_a = fp("0.7071067811865475"); // cos(π/4)
    let sin_a = fp("0.7071067811865475"); // sin(π/4)
    let g01 = FixedMatrix::from_slice(2, 2, &[cos_a, -sin_a, sin_a, cos_a]);

    // R(π/6)
    let cos_b = fp("0.8660254037844386"); // cos(π/6)
    let sin_b = fp("0.5");                 // sin(π/6)
    let g12 = FixedMatrix::from_slice(2, 2, &[cos_b, -sin_b, sin_b, cos_b]);

    // g_{02} = g_{01} * g_{12}
    let g02 = &g01 * &g12;

    bundle.set_transition(0, 1, g01).unwrap();
    bundle.set_transition(1, 2, g12).unwrap();
    bundle.set_transition(0, 2, g02).unwrap();

    let (ok, max_err) = bundle.verify_cocycle(tol());
    assert!(ok, "SO(2) cocycle should hold, max_err={}", max_err);
}

#[test]
fn test_principal_transition_inverse() {
    let mut bundle = PrincipalBundle::trivial(2, 1, 2, 2);

    let cos_a = fp("0.8660254037844386");
    let sin_a = fp("0.5");
    let g = FixedMatrix::from_slice(2, 2, &[cos_a, -sin_a, sin_a, cos_a]);

    bundle.set_transition(0, 1, g).unwrap();

    // g_{10} should be g_{01}⁻¹ = R(-π/6) = R^T
    let g10 = bundle.transition(1, 0);
    assert_fp(g10.get(0, 0), cos_a, tol(), "g_inv[0,0]");
    assert_fp(g10.get(0, 1), sin_a, tol(), "g_inv[0,1]"); // +sin
    assert_fp(g10.get(1, 0), -sin_a, tol(), "g_inv[1,0]"); // -sin
    assert_fp(g10.get(1, 1), cos_a, tol(), "g_inv[1,1]");
}

// ============================================================================
// Associated bundle / change of chart
// ============================================================================

#[test]
fn test_change_chart() {
    let mut bundle = PrincipalBundle::trivial(2, 1, 2, 2);

    // Transition: 90° rotation
    let g = FixedMatrix::from_slice(2, 2, &[
        fp("0"), fp("-1"),
        fp("1"), fp("0"),
    ]);
    bundle.set_transition(0, 1, g).unwrap();

    // Fiber element in chart 0: (1, 0) → in chart 1: (0, 1) (rotated 90°)
    let fiber_0 = FixedVector::from_slice(&[fp("1"), fp("0")]);
    let fiber_1 = change_chart(&bundle, 0, 1, &fiber_0);

    assert_fp(fiber_1[0], fp("0"), tol(), "chart change[0]");
    assert_fp(fiber_1[1], fp("1"), tol(), "chart change[1]");
}

#[test]
fn test_apply_representation() {
    // Rotation matrix acting on a vector
    let r = FixedMatrix::from_slice(2, 2, &[
        fp("0"), fp("-1"),
        fp("1"), fp("0"),
    ]);
    let v = FixedVector::from_slice(&[fp("3"), fp("4")]);
    let result = apply_representation(&r, &v);

    // R(90°) * (3, 4) = (-4, 3)
    assert_fp(result[0], fp("-4"), tol(), "repr[0]");
    assert_fp(result[1], fp("3"), tol(), "repr[1]");
}

// ============================================================================
// Bundle curvature
// ============================================================================

#[test]
fn test_flat_bundle_curvature_zero() {
    let bundle = VectorBundle::flat(2, 2);
    let p = FixedVector::from_slice(&[fp("1"), fp("2")]);

    let curv = vector_bundle_curvature(&bundle, &p).unwrap();

    // Flat connection: all curvature components should be zero
    for a in 0..2 {
        for b in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert_fp(curv.get(&[a, b, i, j]), fp("0"), tol(),
                        &format!("F^{}_{{{}{}{}}} = 0", a, b, i, j));
                }
            }
        }
    }
}

#[test]
fn test_nonflat_bundle_curvature() {
    // Non-commuting connection: A^0_{0,0}=1, A^0_{1,1}=1, A^1_{0,0}=1, A^1_{1,1}=0
    // 2D base, 2D fiber, k*k*n = 2*2*2 = 8 coefficients
    //
    // F^a_{bij} = A^a_{ci} A^c_{bj} - A^a_{cj} A^c_{bi} (for constant A)
    //
    // F^0_{0,01} = A^0_{c0} A^c_{01} - A^0_{c1} A^c_{00}
    // = A^0_{00}*A^0_{01} + A^0_{10}*A^1_{01} - (A^0_{01}*A^0_{00} + A^0_{11}*A^1_{00})
    //
    // Let's use a simpler setup where some curvature is nonzero.
    // A^0_{0,0}=1, A^0_{0,1}=0, A^0_{1,0}=0, A^0_{1,1}=0
    // A^1_{0,0}=0, A^1_{0,1}=1, A^1_{1,0}=0, A^1_{1,1}=0
    //
    // F^0_{0,01} = A^0_{c0}A^c_{01} - A^0_{c1}A^c_{00}
    // = A^0_{00}*A^0_{01} + A^0_{10}*A^1_{01} - (A^0_{01}*A^0_{00} + A^0_{11}*A^1_{00})
    // = 1*0 + 0*1 - (0*1 + 0*0) = 0
    //
    // F^0_{1,01} = A^0_{c0}A^c_{11} - A^0_{c1}A^c_{10}
    // = A^0_{00}*A^0_{11} + A^0_{10}*A^1_{11} - (A^0_{01}*A^0_{10} + A^0_{11}*A^1_{10})
    // = 1*0 + 0*0 - (0*0 + 0*0) = 0
    //
    // Let's try a non-commuting example:
    // k=2, n=2
    // A^0_{00}=0, A^0_{01}=1, A^0_{10}=0, A^0_{11}=0
    // A^1_{00}=0, A^1_{01}=0, A^1_{10}=1, A^1_{11}=0
    //
    // F^0_{0,01} = sum_c A^0_{c0}A^c_{01} - A^0_{c1}A^c_{00}
    //   c=0: A^0_{00}A^0_{01} - A^0_{01}A^0_{00} = 0*1 - 1*0 = 0
    //   c=1: A^0_{10}A^1_{01} - A^0_{11}A^1_{00} = 0*0 - 0*0 = 0
    // → 0
    //
    // F^0_{1,01} = sum_c A^0_{c0}A^c_{11} - A^0_{c1}A^c_{10}
    //   c=0: A^0_{00}A^0_{11} - A^0_{01}A^0_{10} = 0
    //   c=1: A^0_{10}A^1_{11} - A^0_{11}A^1_{10} = 0
    // → 0
    //
    // For genuinely non-trivial curvature we need A_{c,i}A_{c,j} ≠ A_{c,j}A_{c,i}
    // in the fiber indices.
    //
    // A^0_{00}=1, A^0_{01}=0, A^0_{10}=0, A^0_{11}=1
    // A^1_{00}=0, A^1_{01}=1, A^1_{10}=-1, A^1_{11}=0
    // [a*k*n + b*n + i]
    // a=0: [0*4+0*2+0]=1, [0*4+0*2+1]=0, [0*4+1*2+0]=0, [0*4+1*2+1]=1
    // a=1: [1*4+0*2+0]=0, [1*4+0*2+1]=1, [1*4+1*2+0]=-1, [1*4+1*2+1]=0
    let coeffs = vec![
        fp("1"), fp("0"), fp("0"), fp("1"),     // A^0_{bi}
        fp("0"), fp("1"), fp("-1"), fp("0"),     // A^1_{bi}
    ];
    let bundle = VectorBundle::with_connection(2, 2, coeffs);
    let p = FixedVector::from_slice(&[fp("0"), fp("0")]);

    let curv = vector_bundle_curvature(&bundle, &p).unwrap();

    // F^0_{0,01} = A^0_{c0}A^c_{01} - A^0_{c1}A^c_{00}
    //   c=0: A^0_{00}*A^0_{01} - A^0_{01}*A^0_{00} = 1*0 - 0*1 = -0
    //   c=1: A^0_{10}*A^1_{01} - A^0_{11}*A^1_{00} = 0*1 - 1*0 = 0
    // F^0_{0,01} = 0

    // F^0_{1,01} = A^0_{c0}A^c_{11} - A^0_{c1}A^c_{10}
    //   c=0: A^0_{00}*A^0_{11} - A^0_{01}*A^0_{10} = 1*1 - 0*0 = 1
    //   c=1: A^0_{10}*A^1_{11} - A^0_{11}*A^1_{10} = 0*0 - 1*(-1) = 1
    // F^0_{1,01} = 2
    assert_fp(curv.get(&[0, 1, 0, 1]), fp("2"), tol(), "F^0_{1,01}");

    // Antisymmetry in base indices: F^a_{b,ij} = -F^a_{b,ji}
    assert_fp(curv.get(&[0, 1, 1, 0]), fp("-2"), tol(), "F^0_{1,10} = -F^0_{1,01}");
}

// ============================================================================
// Bundle dimensions
// ============================================================================

#[test]
fn test_bundle_dimensions() {
    let trivial = TrivialBundle { base_dimension: 3, fiber_dimension: 2 };
    assert_eq!(trivial.base_dim(), 3);
    assert_eq!(trivial.fiber_dim(), 2);
    assert_eq!(trivial.total_dim(), 5);

    let vb = VectorBundle::flat(4, 3);
    assert_eq!(vb.base_dim(), 4);
    assert_eq!(vb.fiber_dim(), 3);
    assert_eq!(vb.total_dim(), 7);
}
