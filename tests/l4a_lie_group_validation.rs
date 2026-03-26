//! L4A: Lie group validation tests — SO(3), SE(3), SO(n).
//! Rodrigues exp/log, orthogonality, determinant, roundtrips, singularities.

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::lie_group::*;
use g_math::fixed_point::imperative::manifold::Manifold;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}
fn tol() -> FixedPoint {
    #[cfg(table_format = "q16_16")]
    { fp("0.01") }  // Rodrigues trig at 16-bit: cos(π/2) ≈ ±0.0001
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
// SO(3) hat/vee
// ============================================================================

#[test]
fn test_so3_hat_vee_roundtrip() {
    let omega = FixedVector::from_slice(&[fp("0.1"), fp("0.2"), fp("0.3")]);
    let hat = SO3::hat_so3(&omega);
    let vee = SO3::vee_so3(&hat);
    for i in 0..3 { assert_fp(vee[i], omega[i], tight(), &format!("hat/vee[{}]", i)); }
}

#[test]
fn test_so3_hat_skew_symmetric() {
    let omega = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
    let hat = SO3::hat_so3(&omega);
    // hat + hat^T should be zero
    let sum = &hat + &hat.transpose();
    for r in 0..3 { for c in 0..3 {
        assert_fp(sum.get(r, c), fp("0"), tight(), &format!("skew[{},{}]", r, c));
    }}
}

// ============================================================================
// SO(3) Rodrigues exp
// ============================================================================

#[test]
fn test_so3_exp_identity() {
    let zero = FixedVector::from_slice(&[fp("0"), fp("0"), fp("0")]);
    let r = SO3::rodrigues_exp(&zero).unwrap();
    let id = FixedMatrix::identity(3);
    for i in 0..3 { for j in 0..3 {
        assert_fp(r.get(i, j), id.get(i, j), tight(), &format!("exp(0)[{},{}]", i, j));
    }}
}

#[test]
fn test_so3_exp_90deg_z() {
    // 90° rotation about z-axis: ω = [0, 0, π/2]
    let pi_half = fp("1.5707963267948966192");
    let omega = FixedVector::from_slice(&[fp("0"), fp("0"), pi_half]);
    let r = SO3::rodrigues_exp(&omega).unwrap();
    // Expected: [[0,-1,0],[1,0,0],[0,0,1]]
    assert_fp(r.get(0, 0), fp("0"), tol(), "Rz90[0,0]");
    assert_fp(r.get(0, 1), fp("-1"), tol(), "Rz90[0,1]");
    assert_fp(r.get(1, 0), fp("1"), tol(), "Rz90[1,0]");
    assert_fp(r.get(1, 1), fp("0"), tol(), "Rz90[1,1]");
    assert_fp(r.get(2, 2), fp("1"), tol(), "Rz90[2,2]");
}

#[test]
fn test_so3_exp_orthogonality() {
    let omega = FixedVector::from_slice(&[fp("0.5"), fp("0.3"), fp("0.7")]);
    let r = SO3::rodrigues_exp(&omega).unwrap();
    let rtr = &r.transpose() * &r;
    let id = FixedMatrix::identity(3);
    for i in 0..3 { for j in 0..3 {
        assert_fp(rtr.get(i, j), id.get(i, j), tol(), &format!("R^TR[{},{}]", i, j));
    }}
}

#[test]
fn test_so3_exp_determinant() {
    let omega = FixedVector::from_slice(&[fp("1.2"), fp("0.3"), fp("0.8")]);
    let r = SO3::rodrigues_exp(&omega).unwrap();
    let det = g_math::fixed_point::imperative::derived::determinant(&r).unwrap();
    assert_fp(det, fp("1"), tol(), "det(R)=1");
}

// ============================================================================
// SO(3) exp/log roundtrip
// ============================================================================

#[test]
fn test_so3_exp_log_roundtrip() {
    let omega = FixedVector::from_slice(&[fp("0.5"), fp("0.3"), fp("0.7")]);
    let r = SO3::rodrigues_exp(&omega).unwrap();
    let omega_back = SO3::rodrigues_log(&r).unwrap();
    for i in 0..3 {
        assert_fp(omega_back[i], omega[i], tol(), &format!("exp/log roundtrip[{}]", i));
    }
}

#[test]
fn test_so3_exp_log_near_identity() {
    // Taylor branch: very small angle
    let omega = FixedVector::from_slice(&[fp("0.000001"), fp("0.000002"), fp("0.000003")]);
    let r = SO3::rodrigues_exp(&omega).unwrap();
    let omega_back = SO3::rodrigues_log(&r).unwrap();
    for i in 0..3 {
        assert_fp(omega_back[i], omega[i], tight(), &format!("near-identity[{}]", i));
    }
}

#[test]
fn test_so3_log_identity() {
    let id = FixedMatrix::identity(3);
    let omega = SO3::rodrigues_log(&id).unwrap();
    for i in 0..3 { assert_fp(omega[i], fp("0"), tight(), &format!("log(I)[{}]", i)); }
}

// ============================================================================
// SO(3) as Manifold
// ============================================================================

#[test]
fn test_so3_manifold_distance_identity() {
    let so3 = SO3;
    let zero = FixedVector::from_slice(&[fp("0"), fp("0"), fp("0")]);
    assert_fp(so3.distance(&zero, &zero).unwrap(), fp("0"), tight(), "d(0,0)");
}

// ============================================================================
// SO(3) LieGroup operations
// ============================================================================

#[test]
fn test_so3_compose_inverse() {
    let so3 = SO3;
    let omega = FixedVector::from_slice(&[fp("0.5"), fp("0.3"), fp("0.7")]);
    let r = so3.lie_exp(&omega).unwrap();
    let r_inv = so3.group_inverse(&r).unwrap();
    let product = so3.compose(&r, &r_inv);
    let id = FixedMatrix::identity(3);
    for i in 0..3 { for j in 0..3 {
        assert_fp(product.get(i, j), id.get(i, j), tol(), &format!("R*R^-1[{},{}]", i, j));
    }}
}

#[test]
fn test_so3_bracket_cross_product() {
    let so3 = SO3;
    let a = FixedVector::from_slice(&[fp("1"), fp("0"), fp("0")]);
    let b = FixedVector::from_slice(&[fp("0"), fp("1"), fp("0")]);
    let bracket = so3.bracket(&a, &b);
    // [e1, e2] = e3
    assert_fp(bracket[0], fp("0"), tight(), "[e1,e2][0]");
    assert_fp(bracket[1], fp("0"), tight(), "[e1,e2][1]");
    assert_fp(bracket[2], fp("1"), tight(), "[e1,e2][2]");
}

#[test]
fn test_so3_act_rotation() {
    let so3 = SO3;
    // 90° about z rotates [1,0,0] → [0,1,0]
    let pi_half = fp("1.5707963267948966192");
    let omega = FixedVector::from_slice(&[fp("0"), fp("0"), pi_half]);
    let r = so3.lie_exp(&omega).unwrap();
    let p = FixedVector::from_slice(&[fp("1"), fp("0"), fp("0")]);
    let rotated = so3.act(&r, &p);
    assert_fp(rotated[0], fp("0"), tol(), "Rz90*ex[0]");
    assert_fp(rotated[1], fp("1"), tol(), "Rz90*ex[1]");
    assert_fp(rotated[2], fp("0"), tol(), "Rz90*ex[2]");
}

// ============================================================================
// SE(3) tests
// ============================================================================

#[test]
fn test_se3_exp_identity() {
    let zero = FixedVector::from_slice(&[fp("0"), fp("0"), fp("0"), fp("0"), fp("0"), fp("0")]);
    let g = SE3::se3_exp(&zero).unwrap();
    let id = FixedMatrix::identity(4);
    for i in 0..4 { for j in 0..4 {
        assert_fp(g.get(i, j), id.get(i, j), tight(), &format!("se3 exp(0)[{},{}]", i, j));
    }}
}

#[test]
fn test_se3_pure_translation() {
    let xi = FixedVector::from_slice(&[fp("0"), fp("0"), fp("0"), fp("1"), fp("2"), fp("3")]);
    let g = SE3::se3_exp(&xi).unwrap();
    // Should be [[I, [1,2,3]], [0,0,0,1]]
    assert_fp(g.get(0, 0), fp("1"), tight(), "pure trans R[0,0]");
    assert_fp(g.get(0, 3), fp("1"), tight(), "pure trans t[0]");
    assert_fp(g.get(1, 3), fp("2"), tight(), "pure trans t[1]");
    assert_fp(g.get(2, 3), fp("3"), tight(), "pure trans t[2]");
}

#[test]
fn test_se3_exp_log_roundtrip() {
    let xi = FixedVector::from_slice(&[fp("0.1"), fp("0.2"), fp("0.3"), fp("1"), fp("2"), fp("3")]);
    let g = SE3::se3_exp(&xi).unwrap();
    let xi_back = SE3::se3_log(&g).unwrap();
    for i in 0..6 {
        assert_fp(xi_back[i], xi[i], tol(), &format!("se3 exp/log[{}]", i));
    }
}

#[test]
fn test_se3_inverse() {
    let se3 = SE3;
    let xi = FixedVector::from_slice(&[fp("0.1"), fp("0.2"), fp("0.3"), fp("1"), fp("2"), fp("3")]);
    let g = se3.lie_exp(&xi).unwrap();
    let g_inv = se3.group_inverse(&g).unwrap();
    let product = se3.compose(&g, &g_inv);
    let id = FixedMatrix::identity(4);
    for i in 0..4 { for j in 0..4 {
        assert_fp(product.get(i, j), id.get(i, j), tol(), &format!("SE3 g*g^-1[{},{}]", i, j));
    }}
}

#[test]
fn test_se3_act_point() {
    let se3 = SE3;
    // Pure translation by [1,2,3]
    let xi = FixedVector::from_slice(&[fp("0"), fp("0"), fp("0"), fp("1"), fp("2"), fp("3")]);
    let g = se3.lie_exp(&xi).unwrap();
    let p = FixedVector::from_slice(&[fp("0"), fp("0"), fp("0")]);
    let transformed = se3.act(&g, &p);
    assert_fp(transformed[0], fp("1"), tight(), "translate[0]");
    assert_fp(transformed[1], fp("2"), tight(), "translate[1]");
    assert_fp(transformed[2], fp("3"), tight(), "translate[2]");
}

// ============================================================================
// SO(n) general tests
// ============================================================================

#[test]
fn test_son_hat_vee_roundtrip() {
    let so4 = SOn { n: 4 };
    // so(4) has dim 6: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    let xi = FixedVector::from_slice(&[fp("0.1"), fp("0.2"), fp("0.3"), fp("0.4"), fp("0.5"), fp("0.6")]);
    let hat = so4.hat_son(&xi);
    let vee = so4.vee_son(&hat);
    for i in 0..6 { assert_fp(vee[i], xi[i], tight(), &format!("SO4 hat/vee[{}]", i)); }
}

#[test]
fn test_son_exp_orthogonality() {
    let so4 = SOn { n: 4 };
    let xi = FixedVector::from_slice(&[fp("0.1"), fp("0.2"), fp("0.3"), fp("0.4"), fp("0.5"), fp("0.6")]);
    let r = so4.lie_exp(&xi).unwrap();
    let rtr = &r.transpose() * &r;
    let id = FixedMatrix::identity(4);
    for i in 0..4 { for j in 0..4 {
        assert_fp(rtr.get(i, j), id.get(i, j), tol(), &format!("SO4 R^TR[{},{}]", i, j));
    }}
}

#[test]
fn test_son3_matches_so3() {
    // SOn(n=3) should give same result as SO3 for same input
    let so3 = SO3;
    let so3n = SOn { n: 3 };
    let omega = FixedVector::from_slice(&[fp("0.5"), fp("0.3"), fp("0.7")]);
    let r_rodrigues = so3.lie_exp(&omega).unwrap();
    // For SOn, the algebra vector ordering is (0,1), (0,2), (1,2) = (ωz, ωy, ωx)
    // Wait — SOn hat convention: entries at (i,j) for i<j in order (0,1),(0,2),(1,2)
    // SO3 hat: omega = [wx, wy, wz], hat[2,1]=wx, hat[0,2]=wy, hat[1,0]=wz
    // SOn hat: xi[0]=hat[0,1], xi[1]=hat[0,2], xi[2]=hat[1,2]
    // So: SOn xi[0] = hat[0,1] = -wz, xi[1] = hat[0,2] = wy, xi[2] = hat[1,2] = -wx
    let xi_son = FixedVector::from_slice(&[-omega[2], omega[1], -omega[0]]);
    let r_matrix_exp = so3n.lie_exp(&xi_son).unwrap();
    for i in 0..3 { for j in 0..3 {
        assert_fp(r_matrix_exp.get(i, j), r_rodrigues.get(i, j), tol(),
            &format!("SOn(3) vs SO3 [{},{}]", i, j));
    }}
}
