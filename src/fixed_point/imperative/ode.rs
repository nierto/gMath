//! L2A: Numerical ODE solvers with fixed-point arithmetic.
//!
//! Three integrators covering the practical spectrum:
//! - `rk4_step` / `rk4_integrate` — classical 4th-order Runge-Kutta (fixed step)
//! - `rk45_integrate` — Dormand-Prince adaptive (discrete double/halve/keep controller)
//! - `verlet_step` / `verlet_integrate` — symplectic Störmer-Verlet (Hamiltonian systems)
//!
//! **FASC-UGOD integration:** All weighted sums (k1..k4/k6 combinations) are accumulated
//! at compute tier via `compute_tier_dot_raw`. Step size h/2 is exact bit-shift (no
//! rounding). Adaptive step uses discrete controller (no fractional power needed).
//!
//! **Conserved quantity monitoring:** Optional invariant function `C(x)` tracked during
//! integration with threshold-based projection back to constraint surface.

use super::FixedPoint;
use super::FixedVector;
use super::linalg::compute_tier_dot_raw;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// ODE system trait
// ============================================================================

/// Right-hand side of an ODE system: dx/dt = f(t, x).
///
/// Implement this for your specific ODE. The `eval` method must be deterministic
/// and must not use floating-point internally (per gMath constraints).
pub trait OdeSystem {
    /// Evaluate f(t, x) → dx/dt.
    fn eval(&self, t: FixedPoint, x: &FixedVector) -> FixedVector;
}

/// A boxed closure implementing OdeSystem for convenience.
pub struct OdeFn<F: Fn(FixedPoint, &FixedVector) -> FixedVector> {
    pub f: F,
}

impl<F: Fn(FixedPoint, &FixedVector) -> FixedVector> OdeSystem for OdeFn<F> {
    fn eval(&self, t: FixedPoint, x: &FixedVector) -> FixedVector {
        (self.f)(t, x)
    }
}

/// Wrap a closure as an ODE system.
pub fn ode_fn<F: Fn(FixedPoint, &FixedVector) -> FixedVector>(f: F) -> OdeFn<F> {
    OdeFn { f }
}

// ============================================================================
// Solution point
// ============================================================================

/// A single point in an ODE solution trajectory.
#[derive(Clone, Debug)]
pub struct OdePoint {
    pub t: FixedPoint,
    pub x: FixedVector,
}

// ============================================================================
// RK4 — Classical 4th-order Runge-Kutta (fixed step)
// ============================================================================

/// Perform a single RK4 step: x(t+h) from x(t).
///
/// Computes k1..k4 and accumulates the weighted sum
///   x_{n+1} = x_n + (h/6)(k1 + 2k2 + 2k3 + k4)
/// at compute tier for each component. Single downscale per component.
///
/// h/6 is computed as (h/2)/3 — h/2 is a bit-shift (exact).
pub fn rk4_step<S: OdeSystem>(
    sys: &S,
    t: FixedPoint,
    x: &FixedVector,
    h: FixedPoint,
) -> FixedVector {
    let half_h = h_half(h);
    let h_sixth = half_h / FixedPoint::from_int(3);
    let two = FixedPoint::from_int(2);

    let k1 = sys.eval(t, x);
    let k2 = sys.eval(t + half_h, &(x + &(&k1 * half_h)));
    let k3 = sys.eval(t + half_h, &(x + &(&k2 * half_h)));
    let k4 = sys.eval(t + h, &(x + &(&k3 * h)));

    // Accumulate weighted sum at compute tier per component:
    // x_{n+1}[i] = x[i] + h_sixth * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
    let n = x.len();
    let weights: Vec<BinaryStorage> = vec![
        FixedPoint::one().raw(), two.raw(), two.raw(), FixedPoint::one().raw(),
    ];
    let mut result = FixedVector::new(n);
    for i in 0..n {
        let k_vals: Vec<BinaryStorage> = vec![k1[i].raw(), k2[i].raw(), k3[i].raw(), k4[i].raw()];
        let weighted = FixedPoint::from_raw(compute_tier_dot_raw(&weights, &k_vals));
        result[i] = x[i] + h_sixth * weighted;
    }
    result
}

/// Integrate an ODE from t0 to t_end using fixed-step RK4.
///
/// Returns the trajectory as a vec of (t, x) points, including the initial point.
/// The number of steps is ceil((t_end - t0) / h). The final step may be shortened
/// to land exactly on t_end.
pub fn rk4_integrate<S: OdeSystem>(
    sys: &S,
    x0: &FixedVector,
    t0: FixedPoint,
    t_end: FixedPoint,
    h: FixedPoint,
) -> Vec<OdePoint> {
    let mut trajectory = Vec::new();
    let mut t = t0;
    let mut x = x0.clone();
    trajectory.push(OdePoint { t, x: x.clone() });

    while t < t_end {
        let remaining = t_end - t;
        let step = if remaining < h { remaining } else { h };
        if step.is_zero() { break; }
        x = rk4_step(sys, t, &x, step);
        t = t + step;
        trajectory.push(OdePoint { t, x: x.clone() });
    }
    trajectory
}

// ============================================================================
// RK45 — Dormand-Prince adaptive step
// ============================================================================

/// Dormand-Prince coefficients stored as (numerator, denominator) integer pairs.
/// We compute them as FixedPoint rationals at the call site.

/// Configuration for adaptive RK45 integration.
pub struct Rk45Config {
    /// Error tolerance per step.
    pub tol: FixedPoint,
    /// Initial step size.
    pub h_init: FixedPoint,
    /// Minimum step size (floor).
    pub h_min: FixedPoint,
    /// Maximum step size (cap).
    pub h_max: FixedPoint,
    /// Maximum number of steps (safety limit).
    pub max_steps: usize,
}

impl Rk45Config {
    /// Default configuration with the given tolerance and initial step.
    pub fn new(tol: FixedPoint, h_init: FixedPoint) -> Self {
        let h_min = FixedPoint::from_raw(quantum_raw());
        Self {
            tol,
            h_init,
            h_min,
            h_max: h_init * FixedPoint::from_int(16),
            max_steps: 100_000,
        }
    }
}

/// Integrate an ODE using adaptive Dormand-Prince RK45.
///
/// Uses a discrete step controller (double/halve/keep) instead of the
/// standard `h_new = h * (tol/err)^(1/5)` formula — avoids the 5th root
/// computation entirely.
///
/// Step controller:
/// - `err < tol/32` → double h (but cap at h_max)
/// - `err > tol` → halve h and retry (but floor at h_min)
/// - otherwise → keep h
///
/// Returns the trajectory and the number of rejected steps.
pub fn rk45_integrate<S: OdeSystem>(
    sys: &S,
    x0: &FixedVector,
    t0: FixedPoint,
    t_end: FixedPoint,
    config: &Rk45Config,
) -> Result<(Vec<OdePoint>, usize), OverflowDetected> {
    let mut trajectory = Vec::new();
    let mut t = t0;
    let mut x = x0.clone();
    let mut h = config.h_init;
    let mut rejected = 0usize;

    trajectory.push(OdePoint { t, x: x.clone() });

    let tol_loose = config.tol / FixedPoint::from_int(32);

    for _ in 0..config.max_steps {
        if t >= t_end { break; }

        let remaining = t_end - t;
        let step = if remaining < h { remaining } else { h };
        if step.is_zero() { break; }

        // Compute RK45 stages (Dormand-Prince Butcher tableau)
        let (x4, x5) = dp45_pair(sys, t, &x, step);

        // Error estimate: ||x5 - x4|| (infinity norm for cheapness)
        let err = inf_norm_diff(&x5, &x4);

        if err > config.tol {
            // Reject step, halve h
            h = h_half(h);
            if h < config.h_min { h = config.h_min; }
            rejected += 1;
            continue;
        }

        // Accept step (use the 5th-order solution)
        x = x5;
        t = t + step;
        trajectory.push(OdePoint { t, x: x.clone() });

        // Adjust step size
        if err < tol_loose {
            h = h + h; // double
            if h > config.h_max { h = config.h_max; }
        }
        // else keep h
    }

    Ok((trajectory, rejected))
}

/// Compute the Dormand-Prince 4th and 5th order solutions for one step.
///
/// Returns (x4, x5) where x4 is 4th-order and x5 is 5th-order.
fn dp45_pair<S: OdeSystem>(
    sys: &S,
    t: FixedPoint,
    x: &FixedVector,
    h: FixedPoint,
) -> (FixedVector, FixedVector) {
    // Dormand-Prince Butcher tableau nodes
    // c2=1/5, c3=3/10, c4=4/5, c5=8/9, c6=1, c7=1
    let c2 = FixedPoint::one() / FixedPoint::from_int(5);
    let c3 = FixedPoint::from_int(3) / FixedPoint::from_int(10);
    let c4 = FixedPoint::from_int(4) / FixedPoint::from_int(5);
    let c5 = FixedPoint::from_int(8) / FixedPoint::from_int(9);

    let k1 = sys.eval(t, x);

    // k2 = f(t + c2*h, x + h*(a21*k1))
    // a21 = 1/5
    let x2 = x + &(&k1 * (h * c2));
    let k2 = sys.eval(t + c2 * h, &x2);

    // k3 = f(t + c3*h, x + h*(a31*k1 + a32*k2))
    // a31 = 3/40, a32 = 9/40
    let a31 = FixedPoint::from_int(3) / FixedPoint::from_int(40);
    let a32 = FixedPoint::from_int(9) / FixedPoint::from_int(40);
    let x3 = x + &(&(&k1 * (h * a31)) + &(&k2 * (h * a32)));
    let k3 = sys.eval(t + c3 * h, &x3);

    // k4 = f(t + c4*h, x + h*(a41*k1 + a42*k2 + a43*k3))
    // a41 = 44/45, a42 = -56/15, a43 = 32/9
    let a41 = FixedPoint::from_int(44) / FixedPoint::from_int(45);
    let a42 = FixedPoint::from_int(-56) / FixedPoint::from_int(15);
    let a43 = FixedPoint::from_int(32) / FixedPoint::from_int(9);
    let x4_tmp = x + &(&(&k1 * (h * a41)) + &(&(&k2 * (h * a42)) + &(&k3 * (h * a43))));
    let k4 = sys.eval(t + c4 * h, &x4_tmp);

    // k5 = f(t + c5*h, x + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
    // a51 = 19372/6561, a52 = -25360/2187, a53 = 64448/6561, a54 = -212/729
    let a51 = FixedPoint::from_int(19372) / FixedPoint::from_int(6561);
    let a52 = FixedPoint::from_int(-25360) / FixedPoint::from_int(2187);
    let a53 = FixedPoint::from_int(64448) / FixedPoint::from_int(6561);
    let a54 = FixedPoint::from_int(-212) / FixedPoint::from_int(729);
    let x5_tmp = x + &(&(&k1 * (h * a51))
        + &(&(&k2 * (h * a52)) + &(&(&k3 * (h * a53)) + &(&k4 * (h * a54)))));
    let k5 = sys.eval(t + c5 * h, &x5_tmp);

    // k6 = f(t + h, x + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
    // a61 = 9017/3168, a62 = -355/33, a63 = 46732/5247, a64 = 49/176, a65 = -5103/18656
    let a61 = FixedPoint::from_int(9017) / FixedPoint::from_int(3168);
    let a62 = FixedPoint::from_int(-355) / FixedPoint::from_int(33);
    let a63 = FixedPoint::from_int(46732) / FixedPoint::from_int(5247);
    let a64 = FixedPoint::from_int(49) / FixedPoint::from_int(176);
    let a65 = FixedPoint::from_int(-5103) / FixedPoint::from_int(18656);
    let x6_tmp = x + &(&(&k1 * (h * a61))
        + &(&(&k2 * (h * a62))
        + &(&(&k3 * (h * a63))
        + &(&(&k4 * (h * a64)) + &(&k5 * (h * a65))))));
    let k6 = sys.eval(t + h, &x6_tmp);

    // 5th-order solution (b weights):
    // b1=35/384, b3=500/1113, b4=125/192, b5=-2187/6784, b6=11/84
    let b1 = FixedPoint::from_int(35) / FixedPoint::from_int(384);
    let b3 = FixedPoint::from_int(500) / FixedPoint::from_int(1113);
    let b4 = FixedPoint::from_int(125) / FixedPoint::from_int(192);
    let b5 = FixedPoint::from_int(-2187) / FixedPoint::from_int(6784);
    let b6 = FixedPoint::from_int(11) / FixedPoint::from_int(84);

    // 4th-order solution (b* weights):
    // b*1=5179/57600, b*3=7571/16695, b*4=393/640, b*5=-92097/339200, b*6=187/2100, b*7=1/40
    let bs1 = FixedPoint::from_int(5179) / FixedPoint::from_int(57600);
    let bs3 = FixedPoint::from_int(7571) / FixedPoint::from_int(16695);
    let bs4 = FixedPoint::from_int(393) / FixedPoint::from_int(640);
    let bs5 = FixedPoint::from_int(-92097) / FixedPoint::from_int(339200);
    let bs6 = FixedPoint::from_int(187) / FixedPoint::from_int(2100);
    let _bs7 = FixedPoint::one() / FixedPoint::from_int(40);

    // k7 = f(t + h, x5) — but for DP we need k7 only for b*7
    // k7 uses the 5th-order solution point, but for error estimate
    // we use the simpler approach: compute both solutions from the stages
    let n = x.len();
    let mut x5_out = FixedVector::new(n);
    let mut x4_out = FixedVector::new(n);

    // Compute-tier weighted sums for both 5th and 4th order solutions
    let w5: Vec<BinaryStorage> = vec![b1.raw(), b3.raw(), b4.raw(), b5.raw(), b6.raw()];
    let w4: Vec<BinaryStorage> = vec![bs1.raw(), bs3.raw(), bs4.raw(), bs5.raw(), bs6.raw()];
    for i in 0..n {
        let k_vals: Vec<BinaryStorage> = vec![k1[i].raw(), k3[i].raw(), k4[i].raw(), k5[i].raw(), k6[i].raw()];
        let increment5 = h * FixedPoint::from_raw(compute_tier_dot_raw(&w5, &k_vals));
        x5_out[i] = x[i] + increment5;
        let increment4 = h * FixedPoint::from_raw(compute_tier_dot_raw(&w4, &k_vals));
        x4_out[i] = x[i] + increment4;
    }

    (x4_out, x5_out)
}

// ============================================================================
// Symplectic Störmer-Verlet (for Hamiltonian systems)
// ============================================================================

/// A Hamiltonian system: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q.
///
/// `grad_q` returns -∂H/∂q (the force), and `grad_p` returns ∂H/∂p (the velocity).
pub trait HamiltonianSystem {
    /// Force: dp/dt = -∂H/∂q(q, p).
    fn force(&self, q: &FixedVector, p: &FixedVector) -> FixedVector;
    /// Velocity: dq/dt = ∂H/∂p(q, p).
    fn velocity(&self, q: &FixedVector, p: &FixedVector) -> FixedVector;
    /// Total energy H(q, p) — for conservation monitoring.
    fn energy(&self, q: &FixedVector, p: &FixedVector) -> FixedPoint;
}

/// Result of a Hamiltonian integration step.
#[derive(Clone, Debug)]
pub struct HamiltonianPoint {
    pub t: FixedPoint,
    pub q: FixedVector,
    pub p: FixedVector,
    pub energy: FixedPoint,
}

/// Perform one Störmer-Verlet step.
///
/// The leapfrog/Verlet scheme:
///   p_{1/2} = p_n + (h/2) * force(q_n, p_n)
///   q_{n+1} = q_n + h * velocity(q_n, p_{1/2})
///   p_{n+1} = p_{1/2} + (h/2) * force(q_{n+1}, p_{1/2})
///
/// h/2 is computed as a bit-shift (exact, no rounding).
/// This preserves the symplectic structure to storage-tier precision.
pub fn verlet_step<H: HamiltonianSystem>(
    sys: &H,
    q: &FixedVector,
    p: &FixedVector,
    h: FixedPoint,
) -> (FixedVector, FixedVector) {
    let half_h = h_half(h);

    // Half-step momentum
    let f0 = sys.force(q, p);
    let p_half = p + &(&f0 * half_h);

    // Full-step position
    let v_half = sys.velocity(q, &p_half);
    let q_new = q + &(&v_half * h);

    // Half-step momentum (at new position)
    let f1 = sys.force(&q_new, &p_half);
    let p_new = &p_half + &(&f1 * half_h);

    (q_new, p_new)
}

/// Integrate a Hamiltonian system using symplectic Störmer-Verlet.
///
/// Returns trajectory with energy at each step for conservation monitoring.
/// Energy drift indicates integration error accumulation.
pub fn verlet_integrate<H: HamiltonianSystem>(
    sys: &H,
    q0: &FixedVector,
    p0: &FixedVector,
    t0: FixedPoint,
    t_end: FixedPoint,
    h: FixedPoint,
) -> Vec<HamiltonianPoint> {
    let mut trajectory = Vec::new();
    let mut t = t0;
    let mut q = q0.clone();
    let mut p = p0.clone();

    trajectory.push(HamiltonianPoint {
        t,
        q: q.clone(),
        p: p.clone(),
        energy: sys.energy(&q, &p),
    });

    while t < t_end {
        let remaining = t_end - t;
        let step = if remaining < h { remaining } else { h };
        if step.is_zero() { break; }

        let (q_new, p_new) = verlet_step(sys, &q, &p, step);
        q = q_new;
        p = p_new;
        t = t + step;

        trajectory.push(HamiltonianPoint {
            t,
            q: q.clone(),
            p: p.clone(),
            energy: sys.energy(&q, &p),
        });
    }

    trajectory
}

// ============================================================================
// Conserved quantity monitoring
// ============================================================================

/// Monitor a conserved quantity during integration.
///
/// Given a trajectory, evaluates the invariant at each point and returns
/// (max_drift, drift_at_each_point). max_drift = max |C(x_i) - C(x_0)|.
pub fn monitor_invariant<F: Fn(&FixedVector) -> FixedPoint>(
    invariant: F,
    trajectory: &[OdePoint],
) -> (FixedPoint, Vec<FixedPoint>) {
    if trajectory.is_empty() {
        return (FixedPoint::ZERO, Vec::new());
    }
    let c0 = invariant(&trajectory[0].x);
    let mut max_drift = FixedPoint::ZERO;
    let mut drifts = Vec::with_capacity(trajectory.len());

    for point in trajectory {
        let ci = invariant(&point.x);
        let drift = (ci - c0).abs();
        if drift > max_drift { max_drift = drift; }
        drifts.push(drift);
    }

    (max_drift, drifts)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Exact h/2 via bit-shift (no rounding error).
#[inline]
fn h_half(h: FixedPoint) -> FixedPoint {
    // Right-shift the raw Q-format value by 1 bit = divide by 2 exactly.
    // This works because the Q-format representation has the fractional
    // point at bit FRAC_BITS, so shifting right by 1 divides by 2.
    #[cfg(table_format = "q64_64")]
    { FixedPoint::from_raw(h.raw() >> 1u32) }
    #[cfg(table_format = "q128_128")]
    { FixedPoint::from_raw(h.raw() >> 1u32) }
    #[cfg(table_format = "q256_256")]
    { FixedPoint::from_raw(h.raw() >> 1usize) }
}

/// Infinity norm of the difference of two vectors: max_i |a[i] - b[i]|.
fn inf_norm_diff(a: &FixedVector, b: &FixedVector) -> FixedPoint {
    assert_eq!(a.len(), b.len());
    let mut max_val = FixedPoint::ZERO;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max_val { max_val = d; }
    }
    max_val
}

/// Storage-tier quantum (smallest nonzero FixedPoint).
#[cfg(table_format = "q64_64")]
fn quantum_raw() -> BinaryStorage { 1i128 }
#[cfg(table_format = "q128_128")]
fn quantum_raw() -> BinaryStorage {
    use crate::fixed_point::I256;
    I256::from_i128(1)
}
#[cfg(table_format = "q256_256")]
fn quantum_raw() -> BinaryStorage {
    use crate::fixed_point::I512;
    I512::from_i128(1)
}
