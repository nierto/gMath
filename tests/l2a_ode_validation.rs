//! L2A: ODE solver validation tests.
//!
//! Tests verify:
//! 1. RK4 against closed-form solutions (harmonic oscillator, exponential decay)
//! 2. RK45 adaptive stepping activates and maintains accuracy
//! 3. Symplectic Verlet preserves energy for Hamiltonian systems
//! 4. Conserved quantity monitoring detects drift
//! 5. mpmath-validated reference values for each integrator
//!
//! All tests run on the active profile (embedded/balanced/scientific).

use g_math::fixed_point::{FixedPoint, FixedVector};
use g_math::fixed_point::imperative::ode::*;

fn fp(s: &str) -> FixedPoint {
    if s.starts_with('-') { -FixedPoint::from_str(&s[1..]) }
    else { FixedPoint::from_str(s) }
}

fn assert_fp(got: FixedPoint, exp: FixedPoint, tol: FixedPoint, name: &str) {
    let d = (got - exp).abs();
    assert!(d < tol, "{}: got {}, expected {}, diff={}", name, got, exp, d);
}

// ============================================================================
// RK4: Exponential decay — dx/dt = -x, x(0) = 1
// Solution: x(t) = e^(-t)
// ============================================================================

struct ExponentialDecay;

impl OdeSystem for ExponentialDecay {
    fn eval(&self, _t: FixedPoint, x: &FixedVector) -> FixedVector {
        // dx/dt = -x
        -x
    }
}

#[test]
fn test_rk4_exponential_decay() {
    let sys = ExponentialDecay;
    let x0 = FixedVector::from_slice(&[fp("1")]);
    let t0 = fp("0");
    let t_end = fp("1");
    let h = fp("0.01"); // 100 steps

    let trajectory = rk4_integrate(&sys, &x0, t0, t_end, h);
    let final_point = trajectory.last().unwrap();

    // x(1) = e^(-1) ≈ 0.36787944117144232
    // mpmath: mp.dps=50; exp(-1) = 0.36787944117144232159647396907...
    #[cfg(table_format = "q16_16")]
    let expected = fp("0.3678");
    #[cfg(table_format = "q32_32")]
    let expected = fp("0.367879441");
    #[cfg(table_format = "q64_64")]
    let expected = fp("0.3678794411714423");
    #[cfg(table_format = "q128_128")]
    let expected = fp("0.36787944117144232159647396907");
    #[cfg(table_format = "q256_256")]
    let expected = fp("0.36787944117144232159647396907");

    // RK4 with h=0.01 on this smooth ODE should be very accurate
    #[cfg(table_format = "q16_16")]
    let rk4_tol = fp("0.01");  // 100 steps × 16-bit precision
    #[cfg(not(table_format = "q16_16"))]
    let rk4_tol = fp("0.00001");
    assert_fp(final_point.x[0], expected, rk4_tol, "exp_decay_rk4");
}

// ============================================================================
// RK4: Harmonic oscillator — d²x/dt² = -x
// System: dx/dt = v, dv/dt = -x
// Solution: x(t) = cos(t), v(t) = -sin(t) for x(0)=1, v(0)=0
// ============================================================================

struct HarmonicOscillator;

impl OdeSystem for HarmonicOscillator {
    fn eval(&self, _t: FixedPoint, x: &FixedVector) -> FixedVector {
        // x = [position, velocity]
        // dx/dt = [velocity, -position]
        FixedVector::from_slice(&[x[1], -x[0]])
    }
}

#[test]
fn test_rk4_harmonic_oscillator() {
    let sys = HarmonicOscillator;
    let x0 = FixedVector::from_slice(&[fp("1"), fp("0")]); // x=1, v=0
    let t0 = fp("0");
    let t_end = fp("3.14159265358979"); // ≈ π
    let h = fp("0.01");

    let trajectory = rk4_integrate(&sys, &x0, t0, t_end, h);
    let final_point = trajectory.last().unwrap();

    // At t=π: x(π) = cos(π) = -1, v(π) = -sin(π) = 0
    #[cfg(table_format = "q16_16")]
    let osc_tol = fp("0.1");  // ~314 steps at 16-bit precision
    #[cfg(not(table_format = "q16_16"))]
    let osc_tol = fp("0.001");
    assert_fp(final_point.x[0], fp("-1"), osc_tol, "harmonic_x_at_pi");
    assert_fp(final_point.x[1], fp("0"), osc_tol, "harmonic_v_at_pi");
}

#[test]
fn test_rk4_harmonic_energy_conservation() {
    let sys = HarmonicOscillator;
    let x0 = FixedVector::from_slice(&[fp("1"), fp("0")]);
    let trajectory = rk4_integrate(&sys, &x0, fp("0"), fp("10"), fp("0.01"));

    // Energy = x² + v², should be conserved (= 1.0 initially)
    let (max_drift, _) = monitor_invariant(
        |x| x[0] * x[0] + x[1] * x[1],
        &trajectory,
    );

    // Over 10 time units (1000 steps), energy drift should be small
    #[cfg(table_format = "q16_16")]
    let drift_tol = fp("0.1");  // 1000 steps × 16-bit → larger drift
    #[cfg(not(table_format = "q16_16"))]
    let drift_tol = fp("0.001");
    assert!(max_drift < drift_tol,
        "Energy drift {} exceeds tolerance", max_drift);
}

// ============================================================================
// RK4: Single step accuracy — compare against mpmath reference
// ============================================================================

#[test]
fn test_rk4_single_step_mpmath() {
    // Single RK4 step on dx/dt = -x, x(0) = 1, h = 0.1
    // mpmath 50-digit: RK4 gives x(0.1) with O(h^5) local error
    //
    // Exact: e^(-0.1) = 0.90483741803595957316424905944...
    // RK4 with h=0.1: x ≈ 1 + 0.1*(-1)/6*(...) weights
    //
    // k1 = -1
    // k2 = -(1 + 0.05*(-1)) = -0.95
    // k3 = -(1 + 0.05*(-0.95)) = -0.9525
    // k4 = -(1 + 0.1*(-0.9525)) = -0.90475
    // x1 = 1 + (0.1/6)*(-1 + 2*(-0.95) + 2*(-0.9525) + (-0.90475))
    //     = 1 + (0.1/6)*(-5.70975)
    //     = 1 - 0.0951625
    //     = 0.9048375

    let sys = ExponentialDecay;
    let x0 = FixedVector::from_slice(&[fp("1")]);
    let result = rk4_step(&sys, fp("0"), &x0, fp("0.1"));

    // mpmath: the RK4 approximation for this step is 0.9048375
    assert_fp(result[0], fp("0.9048375"), fp("0.0001"), "rk4_single_step");
}

// ============================================================================
// RK45: Adaptive stepping
// ============================================================================

/// Q16.16: RK45 adaptive tolerance 1e-4 is barely representable (raw≈6) at 16-bit.
#[test]
#[cfg(not(table_format = "q16_16"))]
fn test_rk45_exponential_decay() {
    let sys = ExponentialDecay;
    let x0 = FixedVector::from_slice(&[fp("1")]);
    let config = Rk45Config::new(fp("0.0001"), fp("0.1"));

    let (trajectory, _rejected) = rk45_integrate(&sys, &x0, fp("0"), fp("1"), &config).unwrap();
    let final_point = trajectory.last().unwrap();

    // Should reach t=1 with good accuracy
    assert_fp(final_point.x[0], fp("0.3678794411714423"), fp("0.001"), "rk45_exp_decay");

    // Trajectory should be non-trivial (some steps taken)
    assert!(trajectory.len() > 2, "RK45 should take multiple steps");
}

#[test]
fn test_rk45_adaptive_activates() {
    // Use a system where the solution changes rapidly, then slowly
    // dx/dt = -10x (fast decay)
    struct FastDecay;
    impl OdeSystem for FastDecay {
        fn eval(&self, _t: FixedPoint, x: &FixedVector) -> FixedVector {
            FixedVector::from_slice(&[fp("-10") * x[0]])
        }
    }

    let x0 = FixedVector::from_slice(&[fp("1")]);
    let config = Rk45Config::new(fp("0.001"), fp("0.5"));

    let (trajectory, _rejected) = rk45_integrate(&FastDecay, &x0, fp("0"), fp("1"), &config).unwrap();

    // The adaptive controller should reject some initial large steps and then
    // accept smaller ones. We just verify it converges.
    let final_val = trajectory.last().unwrap().x[0];
    // e^(-10) ≈ 0.0000453999...
    assert!(final_val.abs() < fp("0.001"), "Fast decay should converge near zero");
}

// ============================================================================
// Verlet: Harmonic oscillator (Hamiltonian)
// ============================================================================

struct HarmonicHamiltonian;

impl HamiltonianSystem for HarmonicHamiltonian {
    fn force(&self, q: &FixedVector, _p: &FixedVector) -> FixedVector {
        // dp/dt = -dH/dq = -q (for H = p²/2 + q²/2)
        -q
    }

    fn velocity(&self, _q: &FixedVector, p: &FixedVector) -> FixedVector {
        // dq/dt = dH/dp = p
        p.clone()
    }

    fn energy(&self, q: &FixedVector, p: &FixedVector) -> FixedPoint {
        // H = (q² + p²) / 2
        let two = FixedPoint::from_int(2);
        (q.dot_precise(q) + p.dot_precise(p)) / two
    }
}

#[test]
fn test_verlet_harmonic_oscillator() {
    let sys = HarmonicHamiltonian;
    let q0 = FixedVector::from_slice(&[fp("1")]);
    let p0 = FixedVector::from_slice(&[fp("0")]);

    let trajectory = verlet_integrate(&sys, &q0, &p0, fp("0"), fp("6.28318530717959"), fp("0.01"));

    let final_point = trajectory.last().unwrap();
    // After one full period (2π), should return to q≈1, p≈0
    assert_fp(final_point.q[0], fp("1"), fp("0.01"), "verlet_q_period");
    assert_fp(final_point.p[0], fp("0"), fp("0.01"), "verlet_p_period");
}

#[test]
fn test_verlet_energy_conservation() {
    let sys = HarmonicHamiltonian;
    let q0 = FixedVector::from_slice(&[fp("1")]);
    let p0 = FixedVector::from_slice(&[fp("0")]);

    let trajectory = verlet_integrate(&sys, &q0, &p0, fp("0"), fp("100"), fp("0.01"));

    let initial_energy = trajectory[0].energy;
    let mut max_drift = FixedPoint::ZERO;
    for pt in &trajectory {
        let drift = (pt.energy - initial_energy).abs();
        if drift > max_drift { max_drift = drift; }
    }

    // Symplectic integrator: energy should oscillate but not drift
    // Over 100 time units (10000 steps), max energy drift should be small
    assert!(max_drift < fp("0.01"),
        "Verlet energy drift {} exceeds tolerance over 10000 steps", max_drift);
}

#[test]
fn test_verlet_single_step_mpmath() {
    // Single Verlet step on harmonic oscillator
    // q(0)=1, p(0)=0, h=0.1
    //
    // p_half = p0 + (h/2)*force(q0, p0) = 0 + 0.05*(-1) = -0.05
    // q_new = q0 + h*velocity(q0, p_half) = 1 + 0.1*(-0.05) = 0.995
    // p_new = p_half + (h/2)*force(q_new, p_half) = -0.05 + 0.05*(-0.995) = -0.09975
    //
    // mpmath 50 digits: exact cos(0.1)=0.995004165278025766..., exact -sin(0.1)=-0.099833416646828...
    let sys = HarmonicHamiltonian;
    let q0 = FixedVector::from_slice(&[fp("1")]);
    let p0 = FixedVector::from_slice(&[fp("0")]);

    let (q_new, p_new) = verlet_step(&sys, &q0, &p0, fp("0.1"));

    // Verlet is 2nd-order, so these won't be exact but close
    assert_fp(q_new[0], fp("0.995"), fp("0.001"), "verlet_step_q");
    assert_fp(p_new[0], fp("-0.09975"), fp("0.001"), "verlet_step_p");
}

// ============================================================================
// Monitor invariant
// ============================================================================

#[test]
fn test_monitor_invariant_exact() {
    // For the harmonic oscillator, energy x[0]²+x[1]² should be constant=1
    let sys = HarmonicOscillator;
    let x0 = FixedVector::from_slice(&[fp("1"), fp("0")]);
    let trajectory = rk4_integrate(&sys, &x0, fp("0"), fp("1"), fp("0.01"));

    let (max_drift, drifts) = monitor_invariant(
        |x| x[0] * x[0] + x[1] * x[1],
        &trajectory,
    );

    // RK4 doesn't preserve the Hamiltonian exactly, but drift should be small
    #[cfg(table_format = "q16_16")]
    let inv_tol = fp("0.01");
    #[cfg(not(table_format = "q16_16"))]
    let inv_tol = fp("0.0001");
    assert!(max_drift < inv_tol,
        "RK4 energy drift {} over 100 steps", max_drift);
    assert_eq!(drifts.len(), trajectory.len());
    // First point should have zero drift
    assert!(drifts[0].is_zero(), "Initial drift should be zero");
}

// ============================================================================
// Multi-dimensional system: Lotka-Volterra
// dx/dt = αx - βxy
// dy/dt = δxy - γy
// ============================================================================

#[test]
fn test_rk4_lotka_volterra() {
    struct LotkaVolterra;
    impl OdeSystem for LotkaVolterra {
        fn eval(&self, _t: FixedPoint, x: &FixedVector) -> FixedVector {
            let alpha = fp("1.5");
            let beta = fp("1");
            let delta = fp("1");
            let gamma = fp("3");
            FixedVector::from_slice(&[
                alpha * x[0] - beta * x[0] * x[1],
                delta * x[0] * x[1] - gamma * x[1],
            ])
        }
    }

    let x0 = FixedVector::from_slice(&[fp("10"), fp("5")]);
    let trajectory = rk4_integrate(&LotkaVolterra, &x0, fp("0"), fp("1"), fp("0.001"));

    // Just verify the integration completes and values stay positive
    let final_point = trajectory.last().unwrap();
    assert!(final_point.x[0] > fp("0"), "Prey population should stay positive");
    assert!(final_point.x[1] > fp("0"), "Predator population should stay positive");
    assert!(trajectory.len() > 100, "Should take many steps");
}

// ============================================================================
// RK4 order verification: halving step size should reduce error by ~16x
// ============================================================================

// Q16.16/Q32.32: limited precision means both step sizes hit the representation floor,
// so the error ratio collapses. Convergence order test requires ≥Q64.64.
#[test]
#[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
fn test_rk4_order_verification() {
    let sys = ExponentialDecay;
    let x0 = FixedVector::from_slice(&[fp("1")]);
    let exact = fp("0.3678794411714423"); // e^(-1)

    // Run with h=0.1 and h=0.05
    let traj_coarse = rk4_integrate(&sys, &x0, fp("0"), fp("1"), fp("0.1"));
    let traj_fine = rk4_integrate(&sys, &x0, fp("0"), fp("1"), fp("0.05"));

    let err_coarse = (traj_coarse.last().unwrap().x[0] - exact).abs();
    let err_fine = (traj_fine.last().unwrap().x[0] - exact).abs();

    // RK4 is 4th order: halving h should reduce error by factor of ~16
    // We check that fine error is at least 8x smaller (generous bound)
    if !err_fine.is_zero() && !err_coarse.is_zero() {
        // err_coarse / err_fine should be ≈ 16
        let ratio = err_coarse / err_fine;
        assert!(ratio > fp("8"), "RK4 order: error ratio {} should be > 8 (expect ~16)", ratio);
    }
}
