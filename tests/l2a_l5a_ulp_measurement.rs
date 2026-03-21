//! ULP measurement for L2A-L5A modules.
//!
//! Reports actual precision in ULP (Units in the Last Place) against
//! mpmath 50+ digit reference values. NOT tolerance-based pass/fail —
//! measures and reports the actual error.
//!
//! mpmath reference computation (Python):
//! ```python
//! from mpmath import mp, exp, sin, cos, sqrt, pi, cot
//! mp.dps = 50
//!
//! # L2A: RK4 single step on dx/dt = -x, x(0)=1, h=0.1
//! # k1=-1, k2=-0.95, k3=-0.9525, k4=-0.90475
//! # x1 = 1 + (0.1/6)*(-1 + 2*(-0.95) + 2*(-0.9525) + (-0.90475))
//! #    = 1 + (0.1/6)*(-5.70975) = 0.90483750000000000...
//! # True: exp(-0.1) = 0.90483741803595957316424905944643...
//!
//! # L3B: Christoffel symbols on S² at θ=π/4
//! # Γ^0_{11} = -sin(π/4)cos(π/4) = -0.5
//! # Γ^1_{01} = cot(π/4) = 1.0
//!
//! # L3B: Scalar curvature S² = 2, H² = -2
//!
//! # L4B: Stereographic from (1/√3, 1/√3, 1/√3):
//! # x = (1/√3) / (1 - 1/√3) = 1.36602540378443864676372317...
//!
//! # L4B: Cross-ratio CR(0,1,2,3) = (0-2)(1-3)/((0-3)(1-2))
//! #    = (-2)(-2)/((-3)(-1)) = 4/3 = 1.33333333333333333...
//! ```

use g_math::fixed_point::{FixedPoint, FixedVector};

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

/// Compute ULP distance between two FixedPoint values.
fn ulp_distance(a: FixedPoint, b: FixedPoint) -> i64 {
    let diff_fp = (a - b).abs();
    let diff = diff_fp.raw();
    #[cfg(table_format = "q64_64")]
    { diff as i64 }
    #[cfg(table_format = "q128_128")]
    { diff.as_i128() as i64 }
    #[cfg(table_format = "q256_256")]
    { diff.as_i128() as i64 }
}

// ============================================================================
// L2A: ODE Solver ULP Measurement
// ============================================================================

#[test]
fn test_l2a_ulp_report() {
    use g_math::fixed_point::imperative::ode::*;

    struct ExpDecay;
    impl OdeSystem for ExpDecay {
        fn eval(&self, _t: FixedPoint, x: &FixedVector) -> FixedVector { -x }
    }

    println!("\n========================================");
    println!("L2A ODE Solver — ULP Measurement Report");
    println!("========================================\n");

    // RK4 single step: x(0.1) from dx/dt = -x, x(0) = 1
    // mpmath: RK4 approximation is exactly 0.9048375 (method-exact, not truncation)
    let sys = ExpDecay;
    let x0 = FixedVector::from_slice(&[fp("1")]);
    let result = rk4_step(&sys, fp("0"), &x0, fp("0.1"));
    let rk4_expected = fp("0.9048375");
    let rk4_ulp = ulp_distance(result[0], rk4_expected);
    println!("RK4 single step (h=0.1):");
    println!("  got:      {}", result[0]);
    println!("  expected: {} (method-exact)", rk4_expected);
    println!("  ULP:      {}", rk4_ulp);

    // RK4 full integration: x(1) = e^(-1)
    // mpmath 50 digits: 0.36787944117144232159647396907024893310018921374266
    let traj = rk4_integrate(&sys, &x0, fp("0"), fp("1"), fp("0.01"));
    let final_val = traj.last().unwrap().x[0];
    #[cfg(table_format = "q64_64")]
    let exp_neg1 = fp("0.3678794411714423215");
    #[cfg(table_format = "q128_128")]
    let exp_neg1 = fp("0.36787944117144232159647396907024893310");
    #[cfg(table_format = "q256_256")]
    let exp_neg1 = fp("0.36787944117144232159647396907024893310018921374266");
    let rk4_100_err = (final_val - exp_neg1).abs();
    println!("\nRK4 100 steps (h=0.01, t=1):");
    println!("  got:       {}", final_val);
    println!("  e^(-1):    {}", exp_neg1);
    println!("  abs error: {}", rk4_100_err);

    // Verlet single step: harmonic oscillator
    struct HarmHam;
    impl HamiltonianSystem for HarmHam {
        fn force(&self, q: &FixedVector, _p: &FixedVector) -> FixedVector { -q }
        fn velocity(&self, _q: &FixedVector, p: &FixedVector) -> FixedVector { p.clone() }
        fn energy(&self, q: &FixedVector, p: &FixedVector) -> FixedPoint {
            (q.dot_precise(q) + p.dot_precise(p)) / FixedPoint::from_int(2)
        }
    }

    let q0 = FixedVector::from_slice(&[fp("1")]);
    let p0 = FixedVector::from_slice(&[fp("0")]);
    let (q_new, p_new) = verlet_step(&HarmHam, &q0, &p0, fp("0.1"));
    // Verlet (2nd order): q ≈ 0.995, p ≈ -0.09975
    // mpmath: cos(0.1) = 0.99500416527802576..., -sin(0.1) = -0.09983341664682815...
    let verlet_q_err = (q_new[0] - fp("0.995")).abs();
    let verlet_p_err = (p_new[0] - fp("-0.09975")).abs();
    println!("\nVerlet single step (h=0.1):");
    println!("  q_new:     {} (method-exact: 0.995)", q_new[0]);
    println!("  p_new:     {} (method-exact: -0.09975)", p_new[0]);
    println!("  q error:   {}", verlet_q_err);
    println!("  p error:   {}", verlet_p_err);

    // Energy conservation over 10000 steps
    let long_traj = verlet_integrate(&HarmHam, &q0, &p0, fp("0"), fp("100"), fp("0.01"));
    let e0 = long_traj[0].energy;
    let mut max_drift = FixedPoint::ZERO;
    for pt in &long_traj {
        let drift = (pt.energy - e0).abs();
        if drift > max_drift { max_drift = drift; }
    }
    println!("\nVerlet energy conservation (10000 steps):");
    println!("  initial E: {}", e0);
    println!("  max drift: {}", max_drift);
}

// ============================================================================
// L3B: Curvature ULP Measurement
// ============================================================================

#[test]
fn test_l3b_ulp_report() {
    use g_math::fixed_point::imperative::curvature::*;

    println!("\n============================================");
    println!("L3B Curvature — ULP Measurement Report");
    println!("============================================\n");

    // Sphere Christoffel: Γ^0_{11} = -0.5 at θ=π/4
    // mpmath: -sin(pi/4)*cos(pi/4) = -0.5 exactly
    let metric = SphereMetric { radius: fp("1") };
    let theta = fp("0.7853981633974483"); // π/4
    let p = FixedVector::from_slice(&[theta, fp("1")]);
    let gamma = christoffel(&metric, &p).unwrap();

    let g011 = gamma.get(&[0, 1, 1]);
    let g011_expected = fp("-0.5");
    let g011_ulp = ulp_distance(g011, g011_expected);
    println!("Sphere Christoffel Γ^0_{{11}} at θ=π/4:");
    println!("  got:      {}", g011);
    println!("  expected: -0.5 (mpmath exact)", );
    println!("  ULP:      {}", g011_ulp);

    let g101 = gamma.get(&[1, 0, 1]);
    let g101_expected = fp("1");
    let g101_ulp = ulp_distance(g101, g101_expected);
    println!("\nSphere Christoffel Γ^1_{{01}} at θ=π/4:");
    println!("  got:      {}", g101);
    println!("  expected: 1.0 (mpmath: cot(π/4) = 1.0)");
    println!("  ULP:      {}", g101_ulp);

    // Sphere scalar curvature = 2
    let p2 = FixedVector::from_slice(&[fp("1.0471975511965976"), fp("1")]);
    let r = scalar_curvature(&metric, &p2).unwrap();
    let r_expected = fp("2");
    let r_ulp = ulp_distance(r, r_expected);
    println!("\nSphere S² scalar curvature at θ=π/3:");
    println!("  got:      {}", r);
    println!("  expected: 2.0 (mpmath exact for S²)");
    println!("  ULP:      {}", r_ulp);
    println!("  abs err:  {}", (r - r_expected).abs());

    // Hyperbolic scalar curvature = -2
    let hmetric = HyperbolicMetric;
    let hp = FixedVector::from_slice(&[fp("0.5"), fp("1")]);
    let rh = scalar_curvature(&hmetric, &hp).unwrap();
    let rh_expected = fp("-2");
    let rh_ulp = ulp_distance(rh, rh_expected);
    println!("\nHyperbolic H² scalar curvature at (0.5, 1):");
    println!("  got:      {}", rh);
    println!("  expected: -2.0 (mpmath exact for H²)");
    println!("  ULP:      {}", rh_ulp);
    println!("  abs err:  {}", (rh - rh_expected).abs());

    // Hyperbolic Christoffel: Γ^0_{01} = -1/y at y=2
    let hp2 = FixedVector::from_slice(&[fp("0"), fp("2")]);
    let hgamma = christoffel(&hmetric, &hp2).unwrap();
    let h001 = hgamma.get(&[0, 0, 1]);
    let h001_expected = fp("-0.5");
    let h001_ulp = ulp_distance(h001, h001_expected);
    println!("\nHyperbolic Christoffel Γ^0_{{01}} at y=2:");
    println!("  got:      {}", h001);
    println!("  expected: -0.5 (mpmath: -1/y = -0.5)");
    println!("  ULP:      {}", h001_ulp);
}

// ============================================================================
// L4B: Projective ULP Measurement
// ============================================================================

#[test]
fn test_l4b_ulp_report() {
    use g_math::fixed_point::imperative::projective::*;

    println!("\n============================================");
    println!("L4B Projective — ULP Measurement Report");
    println!("============================================\n");

    // Cross-ratio CR(0,1,2,3) = 4/3
    // mpmath: 1.33333333333333333333333333333...
    let cr = cross_ratio_1d(fp("0"), fp("1"), fp("2"), fp("3")).unwrap();
    #[cfg(table_format = "q64_64")]
    let cr_expected = fp("1.3333333333333333333");
    #[cfg(table_format = "q128_128")]
    let cr_expected = fp("1.33333333333333333333333333333333333333");
    #[cfg(table_format = "q256_256")]
    let cr_expected = fp("1.33333333333333333333333333333333333333333333333333333333333333333333333333333");
    let cr_ulp = ulp_distance(cr, cr_expected);
    println!("Cross-ratio CR(0,1,2,3) = 4/3:");
    println!("  got:      {}", cr);
    println!("  expected: 4/3 (mpmath)", );
    println!("  ULP:      {}", cr_ulp);

    // Cross-ratio invariance
    let m = Moebius::new(fp("2"), fp("1"), fp("1"), fp("3"));
    let cr_before = cross_ratio_1d(fp("0"), fp("1"), fp("2"), fp("3")).unwrap();
    let ta = m.apply(fp("0")).unwrap();
    let tb = m.apply(fp("1")).unwrap();
    let tc = m.apply(fp("2")).unwrap();
    let td = m.apply(fp("3")).unwrap();
    let cr_after = cross_ratio_1d(ta, tb, tc, td).unwrap();
    let cr_inv_ulp = ulp_distance(cr_before, cr_after);
    println!("\nCross-ratio invariance under Möbius (2x+1)/(x+3):");
    println!("  before:   {}", cr_before);
    println!("  after:    {}", cr_after);
    println!("  ULP diff: {}", cr_inv_ulp);

    // Stereographic from (1/√3, 1/√3, 1/√3)
    // mpmath: 1/(sqrt(3)-1) · √3/√3 = √3/(3-√3) = 1.36602540378443864676372317075294...
    let a = fp("0.5773502691896257645");
    let p = FixedVector::from_slice(&[a, a, a]);
    let x = stereo_project(&p).unwrap();
    let stereo_expected = fp("1.3660254037844386");
    let stereo_ulp = ulp_distance(x[0], stereo_expected);
    println!("\nStereographic from (1/√3, 1/√3, 1/√3):");
    println!("  got:      {}", x[0]);
    println!("  expected: 1.36602540378... (mpmath)");
    println!("  ULP:      {}", stereo_ulp);

    // Stereographic roundtrip
    let x_test = FixedVector::from_slice(&[fp("1"), fp("2")]);
    let p_sphere = stereo_unproject(&x_test);
    let x_back = stereo_project(&p_sphere).unwrap();
    let rt_ulp_0 = ulp_distance(x_back[0], x_test[0]);
    let rt_ulp_1 = ulp_distance(x_back[1], x_test[1]);
    println!("\nStereographic roundtrip (1, 2) → S² → R²:");
    println!("  x[0] ULP: {}", rt_ulp_0);
    println!("  x[1] ULP: {}", rt_ulp_1);

    // Möbius inverse roundtrip
    let m = Moebius::new(fp("2"), fp("3"), fp("1"), fp("4"));
    let x_val = fp("1.5");
    let mx = m.apply(x_val).unwrap();
    let back = m.inverse().apply(mx).unwrap();
    let moeb_rt_ulp = ulp_distance(back, x_val);
    println!("\nMöbius inverse roundtrip (2x+3)/(x+4) at x=1.5:");
    println!("  got:      {}", back);
    println!("  expected: {}", x_val);
    println!("  ULP:      {}", moeb_rt_ulp);
}
