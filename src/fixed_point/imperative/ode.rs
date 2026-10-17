//! L2A: Numerical ODE solvers with fixed-point arithmetic.
//!
//! Three integrators covering the practical spectrum:
//! - `rk4_step` / `rk4_integrate`: classical 4th-order Runge-Kutta (fixed step)
//! - `rk45_integrate`: Dormand-Prince adaptive (discrete double/halve/keep controller)
//! - `verlet_step` / `verlet_integrate`: symplectic Störmer-Verlet (Hamiltonian systems)
//!
//! **FASC-UGOD integration:** Every Runge-Kutta stage and update, `h * sum a_j k_j`, is
//! formed at the compute tier from the exact rational coefficients. The integrators
//! carry their running state at the compute tier (2F bits) across steps: the state is
//! rounded to storage only to call the right-hand side and to report a trajectory
//! point, and the increment is added to the unrounded state, so N steps no longer
//! accumulate N storage roundings. Adaptive step uses discrete controller (no
//! fractional power needed).
//!
//! **Conserved quantity monitoring:** Optional invariant function `C(x)` tracked during
//! integration with threshold-based projection back to constraint surface.

use super::FixedVector;
use super::kernels;
use super::Scalar;
use crate::fixed_point::fixed_num::{Compute, Fixed, FixedNum};
use crate::fixed_point::width::{Storage};
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// ODE system trait
// ============================================================================

/// Right-hand side of an ODE system: dx/dt = f(t, x).
///
/// Implement this for your specific ODE. The `eval` method must be deterministic
/// and must not use floating-point internally (per gMath constraints).
pub trait OdeSystem<T: Scalar> {
    /// Evaluate f(t, x) → dx/dt.
    fn eval(&self, t: T, x: &FixedVector<T>) -> FixedVector<T>;
}

/// A boxed closure implementing OdeSystem for convenience.
pub struct OdeFn<Func> {
    pub f: Func,
}

impl<S: Storage, const F: u32, Func: Fn(Fixed<S, F>, &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>>> OdeSystem<Fixed<S, F>> for OdeFn<Func>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    fn eval(&self, t: Fixed<S, F>, x: &FixedVector<Fixed<S, F>>) -> FixedVector<Fixed<S, F>> {
        (self.f)(t, x)
    }
}

/// Wrap a closure as an ODE system.
pub fn ode_fn<Func>(f: Func) -> OdeFn<Func> {
    OdeFn { f }
}

// ============================================================================
// Solution point
// ============================================================================

/// A single point in an ODE solution trajectory.
#[derive(Clone, Debug)]
pub struct OdePoint<T> {
    pub t: T,
    pub x: FixedVector<T>,
}

// ============================================================================
// RK4: classical 4th-order Runge-Kutta (fixed step)
// ============================================================================

/// Perform a single RK4 step: x(t+h) from x(t).
///
/// Each stage offset and the final update `(h/6)(k1 + 2k2 + 2k3 + k4)` are
/// formed at the compute tier and rounded to storage once. Before 0.6.4, h/6
/// was rounded to storage first: at h = 0.01 at Q22.10 that is 2
/// units instead of 1.67, and every step ran 20% long (38.7 units of error at
/// Q16.16 on the same test).
pub fn rk4_step<S: Storage, const F: u32, Sys: OdeSystem<Fixed<S, F>>>(
    sys: &Sys,
    t: Fixed<S, F>,
    x: &FixedVector<Fixed<S, F>>,
    h: Fixed<S, F>,
) -> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let x_c = state_to_compute(x);
    state_to_storage(&rk4_step_compute(sys, Fixed::<S, F>::upscale_to_compute(t.raw()), &x_c, Fixed::<S, F>::upscale_to_compute(h.raw())))
}

/// One RK4 step on a state held at the compute tier: `t`, `x` and `h` are
/// compute raws, the right-hand side sees them rounded to storage, and the
/// new state `x + (h/6)(k1 + 2k2 + 2k3 + k4)` stays at the compute tier.
pub(crate) fn rk4_step_compute<S: Storage, const F: u32, Sys: OdeSystem<Fixed<S, F>>>(
    sys: &Sys,
    t: Compute<Fixed<S, F>>,
    x: &[Compute<Fixed<S, F>>],
    h: Compute<Fixed<S, F>>,
) -> Vec<Compute<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let t_s = time_to_storage(t);
    let t_half = time_to_storage(Fixed::<S, F>::compute_add(t, scale(h, 1, 2)));
    let t_full = time_to_storage(Fixed::<S, F>::compute_add(t, h));

    let k1 = sys.eval(t_s, &state_to_storage(x));
    let k2 = sys.eval(t_half, &state_to_storage(&stage(x, h, &[(1, 2, &k1)])));
    let k3 = sys.eval(t_half, &state_to_storage(&stage(x, h, &[(1, 2, &k2)])));
    let k4 = sys.eval(t_full, &state_to_storage(&stage(x, h, &[(1, 1, &k3)])));

    stage(x, h, &[(1, 6, &k1), (2, 6, &k2), (2, 6, &k3), (1, 6, &k4)])
}

/// `c * num / den` at the compute tier.
fn scale<S: Storage, const F: u32>(c: Compute<Fixed<S, F>>, num: i64, den: i64) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::<S, F>::compute_mul_div_int(c, num, den).expect("ode: stage term exceeds the compute tier")
}

/// `x + h * sum_j k_j * num_j / den_j` per component, every product, quotient
/// and sum at the compute tier: nothing is rounded to storage, so a stage or
/// a step update is one compute-tier value (Runge-Kutta coefficients, the
/// step and the state are never rounded to storage on their own).
fn stage<S: Storage, const F: u32>(x: &[Compute<Fixed<S, F>>], h: Compute<Fixed<S, F>>, terms: &[(i64, i64, &FixedVector<Fixed<S, F>>)]) -> Vec<Compute<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    (0..x.len()).map(|i| {
        let mut acc = Fixed::<S, F>::upscale_to_compute(Fixed::ZERO.raw());
        for &(num, den, k) in terms {
            acc = Fixed::<S, F>::compute_add(acc, scale(Fixed::<S, F>::upscale_to_compute(k[i].raw()), num, den));
        }
        Fixed::<S, F>::compute_add(x[i], Fixed::<S, F>::compute_multiply(acc, h))
    }).collect()
}

/// A storage vector as compute raws (exact).
pub(crate) fn state_to_compute<S: Storage, const F: u32>(x: &FixedVector<Fixed<S, F>>) -> Vec<Compute<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    (0..x.len()).map(|i| Fixed::<S, F>::upscale_to_compute(x[i].raw())).collect()
}

/// A compute-tier state rounded to storage, nearest (ties toward +infinity).
/// Panics when a component leaves the storage range, like every infallible
/// downscale (never wraps).
pub(crate) fn state_to_storage<S: Storage, const F: u32>(x: &[Compute<Fixed<S, F>>]) -> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    FixedVector::<Fixed<S, F>>::from_slice(&x.iter().map(|&v| Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(v))).collect::<Vec<_>>())
}

/// A compute-tier time rounded to storage for the right-hand side.
fn time_to_storage<S: Storage, const F: u32>(t: Compute<Fixed<S, F>>) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::from_raw(kernels::round_to_storage::<Fixed<S, F>>(t))
}

/// Integrate an ODE from t0 to t_end using fixed-step RK4.
///
/// Returns the trajectory as a vec of (t, x) points, including the initial point.
/// The number of steps is ceil((t_end - t0) / h). The final step may be shortened
/// to land exactly on t_end.
///
/// The state is carried at the compute tier from step to step and rounded to
/// storage only for the right-hand side and for each reported point, so the
/// trajectory is not rounded after every step (0.6.3 rounded the state each
/// step: up to N/2 units after N steps).
pub fn rk4_integrate<S: Storage, const F: u32, Sys: OdeSystem<Fixed<S, F>>>(
    sys: &Sys,
    x0: &FixedVector<Fixed<S, F>>,
    t0: Fixed<S, F>,
    t_end: Fixed<S, F>,
    h: Fixed<S, F>,
) -> Vec<OdePoint<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let mut trajectory = Vec::new();
    let mut t = t0;
    let mut x = state_to_compute(x0);
    trajectory.push(OdePoint::<Fixed<S, F>> { t, x: x0.clone() });

    while t < t_end {
        let remaining = t_end - t;
        let step = if remaining < h { remaining } else { h };
        if step.is_zero() { break; }
        x = rk4_step_compute(sys, Fixed::<S, F>::upscale_to_compute(t.raw()), &x, Fixed::<S, F>::upscale_to_compute(step.raw()));
        t = t + step;
        trajectory.push(OdePoint::<Fixed<S, F>> { t, x: state_to_storage(&x) });
    }
    trajectory
}

// ============================================================================
// RK45: Dormand-Prince adaptive step
// ============================================================================

/// Dormand-Prince coefficients stored as (numerator, denominator) integer pairs.
/// We compute them as Fixed<S, F> rationals at the call site.

/// Configuration for adaptive RK45 integration.
pub struct Rk45Config<T> {
    /// Error tolerance per step.
    pub tol: T,
    /// Initial step size.
    pub h_init: T,
    /// Minimum step size (floor).
    pub h_min: T,
    /// Maximum step size (cap).
    pub h_max: T,
    /// Maximum number of steps (safety limit).
    pub max_steps: usize,
}

impl<S: Storage, const F: u32> Rk45Config<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Default configuration with the given tolerance and initial step.
    pub fn new(tol: Fixed<S, F>, h_init: Fixed<S, F>) -> Self {
        let h_min = Fixed::from_raw(S::one());
        Self {
            tol,
            h_init,
            h_min,
            h_max: h_init.mul_count(16),
            max_steps: 100_000,
        }
    }
}

/// Integrate an ODE using adaptive Dormand-Prince RK45.
///
/// Uses a discrete step controller (double/halve/keep) instead of the
/// standard `h_new = h * (tol/err)^(1/5)` formula: avoids the 5th root
/// computation entirely.
///
/// Step controller:
/// - `err < tol/32` → double h (but cap at h_max)
/// - `err > tol` → halve h and retry (but floor at h_min)
/// - otherwise → keep h
///
/// The state is carried at the compute tier across accepted steps and
/// rounded to storage only for the right-hand side and for each reported
/// point; the error estimate `|x5 - x4|` is the compute-tier difference.
///
/// Returns the trajectory and the number of rejected steps.
pub fn rk45_integrate<S: Storage, const F: u32, Sys: OdeSystem<Fixed<S, F>>>(
    sys: &Sys,
    x0: &FixedVector<Fixed<S, F>>,
    t0: Fixed<S, F>,
    t_end: Fixed<S, F>,
    config: &Rk45Config<Fixed<S, F>>,
) -> Result<(Vec<OdePoint<Fixed<S, F>>>, usize), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let mut trajectory = Vec::new();
    let mut t = t0;
    let mut x = state_to_compute(x0);
    let mut h = config.h_init;
    let mut rejected = 0usize;
    let tol = Fixed::<S, F>::upscale_to_compute(config.tol.raw());

    trajectory.push(OdePoint::<Fixed<S, F>> { t, x: x0.clone() });


    for _ in 0..config.max_steps {
        if t >= t_end { break; }

        let remaining = t_end - t;
        let step = if remaining < h { remaining } else { h };
        if step.is_zero() { break; }

        // Compute RK45 stages (Dormand-Prince Butcher tableau)
        let (x4, x5) = dp45_pair(sys, t, &x, step);

        // Error estimate: ||x5 - x4|| (infinity norm for cheapness), at the
        // compute tier
        let err = inf_norm_diff(&x5, &x4);

        if err > tol {
            // Reject step, halve h
            h = h_half(h);
            if h < config.h_min { h = config.h_min; }
            rejected += 1;
            continue;
        }

        // Accept step (use the 5th-order solution)
        x = x5;
        t = t + step;
        trajectory.push(OdePoint::<Fixed<S, F>> { t, x: state_to_storage(&x) });

        // Adjust step size
        // err < tol / 32, without dividing: tol / 32 rounds to zero for a
        // tolerance of a few units (the step could then never grow)
        let err32 = Fixed::<S, F>::compute_mul_div_int(err, 32, 1);
        if matches!(err32, Ok(e) if e < tol) {
            h = h + h; // double
            if h > config.h_max { h = config.h_max; }
        }
        // else keep h
    }

    Ok((trajectory, rejected))
}

/// Compute the Dormand-Prince 4th and 5th order solutions for one step from
/// a compute-tier state `x`; both stay at the compute tier.
///
/// Returns (x4, x5) where x4 is 4th-order and x5 is 5th-order.
fn dp45_pair<S: Storage, const F: u32, Sys: OdeSystem<Fixed<S, F>>>(
    sys: &Sys,
    t: Fixed<S, F>,
    x: &[Compute<Fixed<S, F>>],
    h: Fixed<S, F>,
) -> (Vec<Compute<Fixed<S, F>>>, Vec<Compute<Fixed<S, F>>>)
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    // Dormand-Prince tableau as exact rationals; every stage is formed at the
    // compute tier and rounded once, to call the right-hand side (the
    // coefficients used to be rounded to storage first, e.g. 212/729 at 10
    // fraction bits).
    let t_c = Fixed::<S, F>::upscale_to_compute(t.raw());
    let h_c = Fixed::<S, F>::upscale_to_compute(h.raw());
    let at = |num: i64, den: i64| time_to_storage(Fixed::<S, F>::compute_add(t_c, scale(h_c, num, den)));
    let st = |terms: &[(i64, i64, &FixedVector<Fixed<S, F>>)]| state_to_storage(&stage(x, h_c, terms));

    let k1 = sys.eval(t, &state_to_storage(x));
    let k2 = sys.eval(at(1, 5), &st(&[(1, 5, &k1)]));
    let k3 = sys.eval(at(3, 10), &st(&[(3, 40, &k1), (9, 40, &k2)]));
    let k4 = sys.eval(at(4, 5), &st(&[(44, 45, &k1), (-56, 15, &k2), (32, 9, &k3)]));
    let k5 = sys.eval(at(8, 9), &st(&[
        (19372, 6561, &k1), (-25360, 2187, &k2), (64448, 6561, &k3), (-212, 729, &k4),
    ]));
    let k6 = sys.eval(t + h, &st(&[
        (9017, 3168, &k1), (-355, 33, &k2), (46732, 5247, &k3), (49, 176, &k4), (-5103, 18656, &k5),
    ]));

    // 5th-order solution (b weights)
    let x5 = stage(x, h_c, &[
        (35, 384, &k1), (500, 1113, &k3), (125, 192, &k4), (-2187, 6784, &k5), (11, 84, &k6),
    ]);
    // 4th-order solution (b* weights) including b*7 k7, k7 = f(t + h, x5)
    // (first-same-as-last). Before 0.6.4 the 1/40 k7 term was left out, so
    // the b* weights summed to 39/40 and |x5 - x4| carried a constant h k / 40
    // instead of the local error the step controller needs.
    let k7 = sys.eval(t + h, &state_to_storage(&x5));
    let x4 = stage(x, h_c, &[
        (5179, 57600, &k1), (7571, 16695, &k3), (393, 640, &k4), (-92097, 339200, &k5),
        (187, 2100, &k6), (1, 40, &k7),
    ]);

    (x4, x5)
}

// ============================================================================
// Symplectic Störmer-Verlet (for Hamiltonian systems)
// ============================================================================

/// A Hamiltonian system: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q.
///
/// `grad_q` returns -∂H/∂q (the force), and `grad_p` returns ∂H/∂p (the velocity).
pub trait HamiltonianSystem<T: Scalar> {
    /// Force: dp/dt = -∂H/∂q(q, p).
    fn force(&self, q: &FixedVector<T>, p: &FixedVector<T>) -> FixedVector<T>;
    /// Velocity: dq/dt = ∂H/∂p(q, p).
    fn velocity(&self, q: &FixedVector<T>, p: &FixedVector<T>) -> FixedVector<T>;
    /// Total energy H(q, p): for conservation monitoring.
    fn energy(&self, q: &FixedVector<T>, p: &FixedVector<T>) -> T;
}

/// Result of a Hamiltonian integration step.
#[derive(Clone, Debug)]
pub struct HamiltonianPoint<T> {
    pub t: T,
    pub q: FixedVector<T>,
    pub p: FixedVector<T>,
    pub energy: T,
}

/// Perform one Störmer-Verlet step.
///
/// The leapfrog/Verlet scheme:
///   p_{1/2} = p_n + (h/2) * force(q_n, p_n)
///   q_{n+1} = q_n + h * velocity(q_n, p_{1/2})
///   p_{n+1} = p_{1/2} + (h/2) * force(q_{n+1}, p_{1/2})
///
/// The half kicks (h/2) f and the drift are formed at the compute tier;
/// p_{1/2} is rounded to storage only to call the system, so p_{n+1} is one
/// rounding of p_n + (h/2)(f_0 + f_1). This preserves the symplectic
/// structure to storage-tier precision.
pub fn verlet_step<S: Storage, const F: u32, H: HamiltonianSystem<Fixed<S, F>>>(
    sys: &H,
    q: &FixedVector<Fixed<S, F>>,
    p: &FixedVector<Fixed<S, F>>,
    h: Fixed<S, F>,
) -> (FixedVector<Fixed<S, F>>, FixedVector<Fixed<S, F>>)
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let (q_new, p_new) = verlet_step_compute(sys, &state_to_compute(q), &state_to_compute(p), Fixed::<S, F>::upscale_to_compute(h.raw()));
    (state_to_storage(&q_new), state_to_storage(&p_new))
}

/// One Störmer-Verlet step on a compute-tier state (q, p); the system sees
/// the state rounded to storage and the new state stays at the compute tier.
fn verlet_step_compute<S: Storage, const F: u32, H: HamiltonianSystem<Fixed<S, F>>>(
    sys: &H,
    q: &[Compute<Fixed<S, F>>],
    p: &[Compute<Fixed<S, F>>],
    h: Compute<Fixed<S, F>>,
) -> (Vec<Compute<Fixed<S, F>>>, Vec<Compute<Fixed<S, F>>>)
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let q_s = state_to_storage(q);

    // Half-step momentum (h/2 at the compute tier: a raw shift floored h/2
    // for an odd raw h)
    let f0 = sys.force(&q_s, &state_to_storage(p));
    let p_half = stage(p, h, &[(1, 2, &f0)]);
    let p_half_s = state_to_storage(&p_half);

    // Full-step position
    let v_half = sys.velocity(&q_s, &p_half_s);
    let q_new = stage(q, h, &[(1, 1, &v_half)]);

    // Half-step momentum (at new position)
    let f1 = sys.force(&state_to_storage(&q_new), &p_half_s);
    let p_new = stage(&p_half, h, &[(1, 2, &f1)]);

    (q_new, p_new)
}

/// Integrate a Hamiltonian system using symplectic Störmer-Verlet.
///
/// Returns trajectory with energy at each step for conservation monitoring.
/// Energy drift indicates integration error accumulation.
///
/// (q, p) are carried at the compute tier from step to step and rounded to
/// storage only for the system calls and for each reported point (0.6.3
/// rounded them after every half kick and drift).
pub fn verlet_integrate<S: Storage, const F: u32, H: HamiltonianSystem<Fixed<S, F>>>(
    sys: &H,
    q0: &FixedVector<Fixed<S, F>>,
    p0: &FixedVector<Fixed<S, F>>,
    t0: Fixed<S, F>,
    t_end: Fixed<S, F>,
    h: Fixed<S, F>,
) -> Vec<HamiltonianPoint<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let mut trajectory = Vec::new();
    let mut t = t0;
    let mut q = state_to_compute(q0);
    let mut p = state_to_compute(p0);

    trajectory.push(HamiltonianPoint {
        t,
        q: q0.clone(),
        p: p0.clone(),
        energy: sys.energy(q0, p0),
    });

    while t < t_end {
        let remaining = t_end - t;
        let step = if remaining < h { remaining } else { h };
        if step.is_zero() { break; }

        let (q_new, p_new) = verlet_step_compute(sys, &q, &p, Fixed::<S, F>::upscale_to_compute(step.raw()));
        q = q_new;
        p = p_new;
        t = t + step;

        let (q_s, p_s) = (state_to_storage(&q), state_to_storage(&p));
        let energy = sys.energy(&q_s, &p_s);
        trajectory.push(HamiltonianPoint::<Fixed<S, F>> { t, q: q_s, p: p_s, energy });
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
pub fn monitor_invariant<S: Storage, const F: u32, Inv: Fn(&FixedVector<Fixed<S, F>>) -> Fixed<S, F>>(
    invariant: Inv,
    trajectory: &[OdePoint<Fixed<S, F>>],
) -> (Fixed<S, F>, Vec<Fixed<S, F>>)
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if trajectory.is_empty() {
        return (Fixed::ZERO, Vec::new());
    }
    let c0 = invariant(&trajectory[0].x);
    let mut max_drift = Fixed::ZERO;
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

/// h/2 via bit-shift: exact for an even raw h, floored for an odd one. Only
/// used to halve a rejected RK45 step, where the exact value does not matter.
#[inline]
fn h_half<S: Storage, const F: u32>(h: Fixed<S, F>) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    // the raw value shifted right by one bit: h / 2 exactly for an even raw h
    Fixed::from_raw(h.raw().shr_bits(1))
}

/// Infinity norm of the difference of two compute-tier vectors:
/// max_i |a[i] - b[i]|, at the compute tier.
fn inf_norm_diff<S: Storage, const F: u32>(a: &[Compute<Fixed<S, F>>], b: &[Compute<Fixed<S, F>>]) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(a.len(), b.len());
    let zero = Fixed::<S, F>::upscale_to_compute(Fixed::ZERO.raw());
    let mut max_val = zero;
    for i in 0..a.len() {
        let d = Fixed::<S, F>::compute_subtract(a[i], b[i]);
        let d = if d < zero { Fixed::<S, F>::compute_negate(d) } else { d };
        if d > max_val { max_val = d; }
    }
    max_val
}

