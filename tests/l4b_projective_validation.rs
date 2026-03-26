//! L4B: Projective and conformal geometry validation tests.
//!
//! Tests verify:
//! 1. Homogeneous coordinate roundtrip
//! 2. Projective transformation preserves collinearity
//! 3. Cross-ratio invariance under projective transforms
//! 4. Stereographic projection roundtrip
//! 5. Möbius composition = matrix multiplication
//! 6. Möbius inverse roundtrip
//! 7. mpmath-validated reference values
//!
//! All tests run on the active profile (embedded/balanced/scientific).

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::projective::*;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn tol() -> FixedPoint {
    #[cfg(table_format = "q16_16")]
    { fp("0.01") }
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

// ============================================================================
// Homogeneous coordinates
// ============================================================================

#[test]
fn test_homogeneous_roundtrip() {
    let v = FixedVector::from_slice(&[fp("3.5"), fp("-2.1"), fp("0.7")]);
    let h = to_homogeneous(&v);

    assert_eq!(h.len(), 4);
    assert_fp(h[3], fp("1"), tight(), "homogeneous w-component");

    let v_back = from_homogeneous(&h).unwrap();
    for i in 0..3 {
        assert_fp(v_back[i], v[i], tight(), &format!("roundtrip[{}]", i));
    }
}

#[test]
fn test_homogeneous_scaled() {
    // [6, -4, 2] in homogeneous should dehomogenize to [3, -2]
    let h = FixedVector::from_slice(&[fp("6"), fp("-4"), fp("2")]);
    let v = from_homogeneous(&h).unwrap();

    assert_eq!(v.len(), 2);
    assert_fp(v[0], fp("3"), tight(), "dehomogenize[0]");
    assert_fp(v[1], fp("-2"), tight(), "dehomogenize[1]");
}

#[test]
fn test_homogeneous_at_infinity() {
    let h = FixedVector::from_slice(&[fp("1"), fp("2"), fp("0")]);
    assert!(from_homogeneous(&h).is_err(), "Point at infinity should error");
    assert!(is_at_infinity(&h, fp("0.001")));
}

// ============================================================================
// Projective transformations
// ============================================================================

#[test]
fn test_projective_identity() {
    let h = FixedMatrix::identity(3); // 2D projective = 3×3 identity
    let p = FixedVector::from_slice(&[fp("1.5"), fp("-2.3")]);

    let result = projective_transform(&h, &p).unwrap();
    assert_fp(result[0], p[0], tight(), "identity[0]");
    assert_fp(result[1], p[1], tight(), "identity[1]");
}

#[test]
fn test_projective_translation() {
    // Translation by (3, -1) in 2D:
    // [[1, 0, 3], [0, 1, -1], [0, 0, 1]]
    let h = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("0"), fp("3"),
        fp("0"), fp("1"), fp("-1"),
        fp("0"), fp("0"), fp("1"),
    ]);
    let p = FixedVector::from_slice(&[fp("2"), fp("5")]);

    let result = projective_transform(&h, &p).unwrap();
    assert_fp(result[0], fp("5"), tol(), "translate[0]");
    assert_fp(result[1], fp("4"), tol(), "translate[1]");
}

#[test]
fn test_projective_composition() {
    // Two translations should compose as sum
    let h1 = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("0"), fp("1"),
        fp("0"), fp("1"), fp("2"),
        fp("0"), fp("0"), fp("1"),
    ]);
    let h2 = FixedMatrix::from_slice(3, 3, &[
        fp("1"), fp("0"), fp("3"),
        fp("0"), fp("1"), fp("-1"),
        fp("0"), fp("0"), fp("1"),
    ]);

    let h12 = compose_projective(&h1, &h2);
    let p = FixedVector::from_slice(&[fp("0"), fp("0")]);

    // h1(h2(0,0)) = h1(3,-1) = (4, 1)
    let result = projective_transform(&h12, &p).unwrap();
    assert_fp(result[0], fp("4"), tol(), "compose[0]");
    assert_fp(result[1], fp("1"), tol(), "compose[1]");
}

// ============================================================================
// Cross-ratio
// ============================================================================

#[test]
fn test_cross_ratio_1d_basic() {
    // CR(0, 1, 2, 3) = (0-2)(1-3)/((0-3)(1-2)) = (-2)(-2)/((-3)(-1)) = 4/3
    // mpmath: cross_ratio(0,1,2,3) = 4/3 = 1.333...
    let cr = cross_ratio_1d(fp("0"), fp("1"), fp("2"), fp("3")).unwrap();
    assert_fp(cr, fp("1.3333333333333333"), tol(), "CR(0,1,2,3)");
}

#[test]
fn test_cross_ratio_projective_invariance() {
    // Cross-ratio should be preserved under projective transformation
    // Test with a general projective map: x ↦ (2x+1)/(x+3)
    let a = fp("0");
    let b = fp("1");
    let c = fp("2");
    let d = fp("3");

    let cr_before = cross_ratio_1d(a, b, c, d).unwrap();

    // Apply Möbius (projective in 1D): T(x) = (2x+1)/(x+3)
    let m = Moebius::new(fp("2"), fp("1"), fp("1"), fp("3"));
    let ta = m.apply(a).unwrap();
    let tb = m.apply(b).unwrap();
    let tc = m.apply(c).unwrap();
    let td = m.apply(d).unwrap();

    let cr_after = cross_ratio_1d(ta, tb, tc, td).unwrap();

    assert_fp(cr_before, cr_after, tol(), "Cross-ratio projective invariance");
}

#[test]
fn test_cross_ratio_nd() {
    // 4 collinear points in R²
    let a = FixedVector::from_slice(&[fp("0"), fp("0")]);
    let b = FixedVector::from_slice(&[fp("1"), fp("1")]);
    let c = FixedVector::from_slice(&[fp("2"), fp("2")]);
    let d = FixedVector::from_slice(&[fp("3"), fp("3")]);

    let cr = cross_ratio(&a, &b, &c, &d).unwrap();
    // Same as 1D case projected onto the line: CR(0, √2, 2√2, 3√2) = 4/3
    assert_fp(cr, fp("1.3333333333333333"), tol(), "CR(R² collinear)");
}

// ============================================================================
// Stereographic projection
// ============================================================================

#[test]
fn test_stereo_roundtrip() {
    // Point on S² (not the north pole)
    let p = FixedVector::from_slice(&[fp("0.5"), fp("0.5"), fp("0.7071067811865475")]);
    // Not exactly on sphere but close enough for roundtrip test

    let x = stereo_project(&p).unwrap();
    let p_back = stereo_unproject(&x);

    // For exact roundtrip, the point must be on the unit sphere
    // stereo_unproject always returns unit sphere points
    // Check that stereo_project(stereo_unproject(x)) = x
    let x_back = stereo_project(&p_back).unwrap();
    for i in 0..x.len() {
        assert_fp(x_back[i], x[i], tol(), &format!("stereo_roundtrip[{}]", i));
    }
}

#[test]
fn test_stereo_unproject_roundtrip() {
    // Start from R^2, go to sphere, come back
    let x = FixedVector::from_slice(&[fp("1"), fp("2")]);

    let p = stereo_unproject(&x);
    assert_eq!(p.len(), 3, "Sphere point should be 3D");

    // Verify it's on the unit sphere: ||p||² = 1
    let norm_sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    assert_fp(norm_sq, fp("1"), tol(), "stereo_unproject on sphere");

    // Now project back
    let x_back = stereo_project(&p).unwrap();
    assert_fp(x_back[0], x[0], tol(), "stereo back [0]");
    assert_fp(x_back[1], x[1], tol(), "stereo back [1]");
}

#[test]
fn test_stereo_origin_to_south_pole() {
    // Origin in R^n maps to the south pole (0, ..., 0, -1)
    // mpmath: stereo_unproject(0, 0) = (0, 0, -1)
    let x = FixedVector::from_slice(&[fp("0"), fp("0")]);
    let p = stereo_unproject(&x);

    assert_fp(p[0], fp("0"), tight(), "south_pole[0]");
    assert_fp(p[1], fp("0"), tight(), "south_pole[1]");
    assert_fp(p[2], fp("-1"), tight(), "south_pole[2]");
}

#[test]
fn test_stereo_mpmath_reference() {
    // mpmath 50 digits: stereo_project((1/√3, 1/√3, 1/√3))
    // where (1/√3, 1/√3, 1/√3) is on S² (norm = 1)
    // p = [a, a, a] with a = 1/√3 ≈ 0.57735026918962576
    //
    // stereo: x_i = p_i / (1 - p_2) = a / (1 - a)
    // 1 - a = 1 - 0.57735... = 0.42264...
    // x = 0.57735.../0.42264... = 1.36602540378443864...
    //
    // mpmath: 1/(sqrt(3) - 1) * sqrt(3)/sqrt(3) = sqrt(3)/(3-sqrt(3))
    //       = 1.36602540378443864676372317...

    let a = fp("0.5773502691896257645");
    let p = FixedVector::from_slice(&[a, a, a]);
    let x = stereo_project(&p).unwrap();

    let expected = fp("1.3660254037844386");
    assert_fp(x[0], expected, tol(), "stereo mpmath x[0]");
    assert_fp(x[1], expected, tol(), "stereo mpmath x[1]");
}

// ============================================================================
// Möbius transformations (real)
// ============================================================================

#[test]
fn test_moebius_identity() {
    let m = Moebius::identity();
    let x = fp("3.7");
    let result = m.apply(x).unwrap();
    assert_fp(result, x, tight(), "Möbius identity");
}

#[test]
fn test_moebius_inverse_roundtrip() {
    let m = Moebius::new(fp("2"), fp("3"), fp("1"), fp("4"));
    let m_inv = m.inverse();

    // m_inv(m(x)) = x
    let x = fp("1.5");
    let mx = m.apply(x).unwrap();
    let result = m_inv.apply(mx).unwrap();
    assert_fp(result, x, tol(), "Möbius inverse roundtrip");
}

#[test]
fn test_moebius_composition() {
    let m1 = Moebius::new(fp("1"), fp("2"), fp("0"), fp("1")); // x ↦ x+2
    let m2 = Moebius::new(fp("3"), fp("0"), fp("0"), fp("1")); // x ↦ 3x

    // m1 ∘ m2: x ↦ 3x + 2
    let composed = m1.compose(&m2);
    let x = fp("1");

    let step_by_step = m1.apply(m2.apply(x).unwrap()).unwrap();
    let one_shot = composed.apply(x).unwrap();
    assert_fp(one_shot, step_by_step, tight(), "Möbius composition");
    assert_fp(one_shot, fp("5"), tight(), "Möbius 3*1+2=5");
}

#[test]
fn test_moebius_determinant() {
    let m = Moebius::new(fp("2"), fp("3"), fp("1"), fp("4"));
    let det = m.determinant();
    // det = 2*4 - 3*1 = 5
    assert_fp(det, fp("5"), tight(), "Möbius det");
}

#[test]
fn test_moebius_fixed_point() {
    // Möbius x ↦ (2x+1)/(x+2) has fixed points where f(x) = x
    // x(x+2) = 2x+1 → x² = 1 → x = ±1
    //
    // mpmath: solve (2x+1)/(x+2) = x → x = 1 or x = -1
    let m = Moebius::new(fp("2"), fp("1"), fp("1"), fp("2"));
    let f1 = m.apply(fp("1")).unwrap();
    assert_fp(f1, fp("1"), tol(), "Möbius fixed point x=1");

    let f_neg1 = m.apply(fp("-1")).unwrap();
    assert_fp(f_neg1, fp("-1"), tol(), "Möbius fixed point x=-1");
}

// ============================================================================
// Complex Möbius transformations
// ============================================================================

#[test]
fn test_complex_moebius_identity() {
    let z = FixedPoint::ZERO;
    let one = FixedPoint::one();
    let m = MoebiusComplex::new((one, z), (z, z), (z, z), (one, z));

    let point = (fp("3"), fp("4")); // 3+4i
    let result = m.apply(point).unwrap();
    assert_fp(result.0, fp("3"), tight(), "complex identity re");
    assert_fp(result.1, fp("4"), tight(), "complex identity im");
}

#[test]
fn test_complex_moebius_rotation() {
    // Rotation by π/2: multiply by i → z ↦ iz / 1
    // a=i, b=0, c=0, d=1
    let z = FixedPoint::ZERO;
    let one = FixedPoint::one();
    let m = MoebiusComplex::new((z, one), (z, z), (z, z), (one, z));

    // i * (3+4i) = 3i + 4i² = -4 + 3i
    let result = m.apply((fp("3"), fp("4"))).unwrap();
    assert_fp(result.0, fp("-4"), tol(), "rotation re");
    assert_fp(result.1, fp("3"), tol(), "rotation im");
}

#[test]
fn test_complex_moebius_inverse_roundtrip() {
    let z = FixedPoint::ZERO;
    let _one = FixedPoint::one();
    let m = MoebiusComplex::new(
        (fp("2"), fp("1")),  // a = 2+i
        (fp("1"), z),         // b = 1
        (z, fp("1")),         // c = i
        (fp("3"), fp("-1")),  // d = 3-i
    );
    let m_inv = m.inverse();
    let point = (fp("1"), fp("2"));

    let transformed = m.apply(point).unwrap();
    let back = m_inv.apply(transformed).unwrap();
    assert_fp(back.0, point.0, tol(), "complex inverse roundtrip re");
    assert_fp(back.1, point.1, tol(), "complex inverse roundtrip im");
}

// ============================================================================
// Projective preserves collinearity
// ============================================================================

#[test]
fn test_projective_collinearity() {
    // Three collinear points: (0,0), (1,1), (2,2)
    // Under a projective transform, they should remain collinear
    let h = FixedMatrix::from_slice(3, 3, &[
        fp("2"), fp("1"), fp("0"),
        fp("0"), fp("3"), fp("1"),
        fp("1"), fp("0"), fp("1"),
    ]);

    let p1 = projective_transform(&h, &FixedVector::from_slice(&[fp("0"), fp("0")])).unwrap();
    let p2 = projective_transform(&h, &FixedVector::from_slice(&[fp("1"), fp("1")])).unwrap();
    let p3 = projective_transform(&h, &FixedVector::from_slice(&[fp("2"), fp("2")])).unwrap();

    // Check collinearity: (p2-p1) × (p3-p1) = 0 (2D cross product)
    let d1x = p2[0] - p1[0];
    let d1y = p2[1] - p1[1];
    let d2x = p3[0] - p1[0];
    let d2y = p3[1] - p1[1];
    let cross = d1x * d2y - d1y * d2x;
    assert_fp(cross, fp("0"), tol(), "Collinearity preservation");
}
