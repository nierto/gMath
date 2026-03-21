//! L4A Lie group validation against mpmath 50-digit reference values.

use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};
use g_math::fixed_point::imperative::lie_group::*;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn tol() -> FixedPoint { fp("0.000000001") }

fn assert_fp(got: FixedPoint, exp: FixedPoint, tol: FixedPoint, name: &str) {
    let d = (got - exp).abs();
    assert!(d < tol, "MPMATH FAIL {}: got {}, expected {}, diff={}", name, got, exp, d);
}

fn ulp_diff(a: FixedPoint, b: FixedPoint) -> u64 {
    let raw = (a - b).raw();
    #[cfg(table_format = "q64_64")]
    { raw.unsigned_abs() as u64 }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    { let abs_raw = if raw.is_negative() { -raw } else { raw }; abs_raw.as_i128() as u64 }
}

// ============================================================================
// SO(3) Rodrigues exp vs mpmath
// ============================================================================

#[test]
fn test_mpmath_so3_exp_general() {
    // mpmath: exp([0.5, 0.3, 0.7])
    let omega = FixedVector::from_slice(&[fp("0.5"), fp("0.3"), fp("0.7")]);
    let r = SO3::rodrigues_exp(&omega).unwrap();

    // Full 25-digit mpmath references (avoids truncation error in comparison)
    let refs_25 = [
        (0,0,"0.7295115358427201702495100"), (0,1,"-0.5371528306005544125213872"), (0,2,"0.4234144017982946449592050"),
        (1,0,"0.6770606568888026117498579"), (1,1,"0.6548940284889877914596786"), (1,2,"-0.3357121957015680886469927"),
        (2,0,"-0.0969628071257155220554896"), (2,1,"0.5315831525051155551405404"), (2,2,"0.8414377968733187047796641"),
    ];
    for &(i, j, ref_s) in &refs_25 {
        assert_fp(r.get(i, j), fp(ref_s), tol(), &format!("R[{}][{}]", i, j));
    }

    println!("\n── SO(3) Rodrigues exp ULP (vs mpmath 25-digit) ──");
    let refs = refs_25;
    let mut max_ulp = 0u64;
    for &(i, j, ref_s) in &refs {
        let ulp = ulp_diff(r.get(i, j), fp(ref_s));
        println!("  R[{}][{}]: {} ULP", i, j, ulp);
        max_ulp = max_ulp.max(ulp);
    }
    println!("  MAX ULP: {}", max_ulp);
}

#[test]
fn test_mpmath_so3_exp_log_roundtrip() {
    // mpmath: log(exp([0.5, 0.3, 0.7])) = [0.5, 0.3, 0.7]
    let omega = FixedVector::from_slice(&[fp("0.5"), fp("0.3"), fp("0.7")]);
    let r = SO3::rodrigues_exp(&omega).unwrap();
    let omega_back = SO3::rodrigues_log(&r).unwrap();

    println!("\n── SO(3) exp/log roundtrip ULP ──");
    for i in 0..3 {
        let ulp = ulp_diff(omega_back[i], omega[i]);
        println!("  omega[{}]: {} ULP (got {}, ref {})", i, ulp, omega_back[i], omega[i]);
        assert_fp(omega_back[i], omega[i], tol(), &format!("roundtrip[{}]", i));
    }
}

#[test]
fn test_mpmath_so3_det() {
    // mpmath: det = 1.0 exactly
    let omega = FixedVector::from_slice(&[fp("0.5"), fp("0.3"), fp("0.7")]);
    let r = SO3::rodrigues_exp(&omega).unwrap();
    let det = g_math::fixed_point::imperative::derived::determinant(&r).unwrap();
    let ulp = ulp_diff(det, fp("1"));
    println!("\n── SO(3) det ULP: {} ──", ulp);
    assert_fp(det, fp("1"), tol(), "det(R)");
}

#[test]
fn test_mpmath_so3_orthogonality() {
    // mpmath: R^T R = I (max off-diagonal ≈ 10^-51)
    let omega = FixedVector::from_slice(&[fp("0.5"), fp("0.3"), fp("0.7")]);
    let r = SO3::rodrigues_exp(&omega).unwrap();
    let rtr = &r.transpose() * &r;
    let id = FixedMatrix::identity(3);

    println!("\n── SO(3) orthogonality R^TR ULP ──");
    let mut max_ulp = 0u64;
    for i in 0..3 {
        for j in 0..3 {
            let ulp = ulp_diff(rtr.get(i, j), id.get(i, j));
            if ulp > 0 { println!("  R^TR[{}][{}]: {} ULP", i, j, ulp); }
            max_ulp = max_ulp.max(ulp);
        }
    }
    println!("  MAX orthogonality ULP: {}", max_ulp);
}

// ============================================================================
// SE(3) vs mpmath
// ============================================================================

#[test]
fn test_mpmath_se3_exp_general() {
    // mpmath: exp([0.1, 0.2, 0.3, 1, 2, 3])
    let xi = FixedVector::from_slice(&[fp("0.1"), fp("0.2"), fp("0.3"), fp("1"), fp("2"), fp("3")]);
    let g = SE3::se3_exp(&xi).unwrap();

    // Rotation part
    assert_fp(g.get(0, 0), fp("0.935754803277918907"), tol(), "SE3 R[0][0]");
    assert_fp(g.get(0, 1), fp("-0.283164960565073694"), tol(), "SE3 R[0][1]");
    assert_fp(g.get(1, 0), fp("0.302932713402637110"), tol(), "SE3 R[1][0]");
    assert_fp(g.get(2, 2), fp("0.975290308953045730"), tol(), "SE3 R[2][2]");

    // Translation part — mpmath says t ≈ [1, 2, 3] (very close)
    assert_fp(g.get(0, 3), fp("1"), fp("0.001"), "SE3 t[0]");
    assert_fp(g.get(1, 3), fp("2"), fp("0.001"), "SE3 t[1]");
    assert_fp(g.get(2, 3), fp("3"), fp("0.001"), "SE3 t[2]");

    println!("\n── SE(3) exp ULP ──");
    let r_refs = [
        (0,0,"0.935754803277918907"), (0,1,"-0.283164960565073694"),
        (1,0,"0.302932713402637110"), (2,2,"0.975290308953045730"),
    ];
    for &(i, j, ref_s) in &r_refs {
        let ulp = ulp_diff(g.get(i, j), fp(ref_s));
        println!("  G[{}][{}]: {} ULP", i, j, ulp);
    }
}

#[test]
fn test_mpmath_se3_exp_log_roundtrip() {
    let xi = FixedVector::from_slice(&[fp("0.1"), fp("0.2"), fp("0.3"), fp("1"), fp("2"), fp("3")]);
    let g = SE3::se3_exp(&xi).unwrap();
    let xi_back = SE3::se3_log(&g).unwrap();

    println!("\n── SE(3) exp/log roundtrip ULP ──");
    for i in 0..6 {
        let ulp = ulp_diff(xi_back[i], xi[i]);
        println!("  xi[{}]: {} ULP (got {}, ref {})", i, ulp, xi_back[i], xi[i]);
        assert_fp(xi_back[i], xi[i], fp("0.0001"), &format!("SE3 roundtrip[{}]", i));
    }
}
