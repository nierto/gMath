//! ODE and geodesic integrators carry their running state at the compute tier
//! (2F bits) and round it to storage once per reported point, against mpmath.
//!
//! References: `tests/data/ode_compute_state_refs.rs` from
//! `scripts/generate_ode_compute_state_refs.py`: the SAME discrete scheme
//! (RK4, the Dormand-Prince 5th-order solution at a fixed step, Stormer-Verlet,
//! the RK4 geodesic integrator, Euler parallel transport) run in mpmath at 120
//! digits, so every error below is rounding, not discretization. Inputs and
//! steps are dyadic (the geodesic step is T/96), exact on every gated build;
//! each reference is parsed at the build's split and errors are in storage
//! units (one unit = 2^-FRAC_BITS). The callbacks receive storage values, so
//! each stage input is rounded to storage and the right-hand side rounds its
//! own output: those errors are bounded by the integration time, not by the
//! step count, once the state itself no longer rounds after every step.
//!
//! Measured worst error in storage units over realtime at 8, 10, 12, 16, 20
//! and 24 fraction bits, compact, embedded, balanced and scientific,
//! 0.6.3 state rounded every step -> state at the compute tier:
//!   rk4 oscillator (256 steps)      3..17 -> 0..1
//!   rk4 y' = t - y (256 steps)      1..23 -> 0 (1 at 28 fraction bits, outside the gate)
//!   dp45 oscillator (128 steps)      3..7 -> 1
//!   verlet oscillator (256 steps)   6..24 -> 0..1
//!   geodesic flat (96 steps)           32 -> 0 (every build)
//!   geodesic H2 (96 steps)           2..10 -> 0..1
//!   transport H2 (64 segments)  5..1536 -> 0..1 (1536 at 10 fraction bits)

use g_math::fixed_point::imperative::curvature::{
    geodesic_integrate, parallel_transport_ode, EuclideanMetric, HyperbolicMetric,
};
use g_math::fixed_point::imperative::ode::{
    ode_fn, rk45_integrate, rk4_integrate, verlet_integrate, HamiltonianSystem, Rk45Config,
};
use g_math::fixed_point::{FixedPoint, FixedVector};

#[allow(dead_code)]
mod refs {
    include!("data/ode_compute_state_refs.rs");
}

fn fp(s: &str) -> FixedPoint {
    if let Some(rest) = s.strip_prefix('-') { -FixedPoint::from_str(rest) } else { FixedPoint::from_str(s) }
}
fn vecs(v: &[&str]) -> FixedVector { FixedVector::from_slice(&v.iter().map(|s| fp(s)).collect::<Vec<_>>()) }

/// |got - want| in storage units.
fn units(got: FixedPoint, want: FixedPoint) -> i32 {
    let half = fp("0.5");
    let mut unit = FixedPoint::one();
    for _ in 0..g_math::fixed_point::frac_config::FRAC_BITS { unit = unit * half; }
    ((got - want).abs() / unit).to_int()
}

/// Worst error over the reference rows; `state(k)` is the state at step k.
fn worst<const W: usize>(rows: &[(usize, [&str; W])], state: impl Fn(usize) -> FixedVector) -> i32 {
    let mut worst = 0;
    for (k, want) in rows {
        let got = state(*k);
        for i in 0..W { worst = worst.max(units(got[i], fp(want[i]))); }
    }
    worst
}

/// The measured value is printed so the finding can be updated if it moves.
fn check(name: &str, worst: i32, bound: i32) {
    println!("F={} {name}: worst {worst} units (bound {bound})", g_math::fixed_point::frac_config::FRAC_BITS);
    assert!(worst <= bound, "{name}: {worst} units > {bound}");
}

const H64: &str = "0.015625";

#[test]
fn rk4_oscillator_and_forced_decay() {
    let osc = ode_fn(|_t, x: &FixedVector| FixedVector::from_slice(&[x[1], -x[0]]));
    let traj = rk4_integrate(&osc, &vecs(&["1", "0"]), fp("0"), fp("4"), fp(H64));
    assert_eq!(traj.len(), 257);
    check("rk4 oscillator", worst(refs::RK4_OSC, |k| traj[k].x.clone()), BOUND_RK4_OSC);

    let forced = ode_fn(|t: FixedPoint, x: &FixedVector| FixedVector::from_slice(&[t - x[0]]));
    let traj = rk4_integrate(&forced, &vecs(&["1"]), fp("0"), fp("4"), fp(H64));
    assert_eq!(traj.len(), 257);
    check("rk4 y' = t - y", worst(refs::RK4_FORCED, |k| traj[k].x.clone()), BOUND_RK4_FORCED);
}

#[test]
fn dp45_fixed_step_oscillator() {
    // tol far above the local error and h_min = h_max = h_init: 128 accepted
    // steps of 1/32, so the trajectory is the 5th-order solution of the
    // tableau at a fixed step (the adaptive controller is not exercised here)
    let osc = ode_fn(|_t, x: &FixedVector| FixedVector::from_slice(&[x[1], -x[0]]));
    let h = fp("0.03125");
    let config = Rk45Config { tol: fp("1"), h_init: h, h_min: h, h_max: h, max_steps: 1000 };
    let (traj, rejected) = rk45_integrate(&osc, &vecs(&["1", "0"]), fp("0"), fp("4"), &config).unwrap();
    assert_eq!(rejected, 0);
    assert_eq!(traj.len(), 129);
    check("dp45 oscillator", worst(refs::DP45_OSC, |k| traj[k].x.clone()), BOUND_DP45_OSC);
}

struct Oscillator;

impl HamiltonianSystem for Oscillator {
    fn force(&self, q: &FixedVector, _p: &FixedVector) -> FixedVector { -q }
    fn velocity(&self, _q: &FixedVector, p: &FixedVector) -> FixedVector { p.clone() }
    fn energy(&self, q: &FixedVector, p: &FixedVector) -> FixedPoint { (q[0] * q[0] + p[0] * p[0]) * fp("0.5") }
}

#[test]
fn verlet_oscillator() {
    let traj = verlet_integrate(&Oscillator, &vecs(&["1"]), &vecs(&["0"]), fp("0"), fp("4"), fp(H64));
    assert_eq!(traj.len(), 257);
    check("verlet oscillator", worst(refs::VERLET_OSC, |k| FixedVector::from_slice(&[traj[k].q[0], traj[k].p[0]])), BOUND_VERLET_OSC);
}

#[test]
fn geodesics_flat_and_hyperbolic() {
    let flat = geodesic_integrate(&EuclideanMetric { dim: 2 }, &vecs(&["0.5", "-0.25"]), &vecs(&["0.75", "1.25"]), fp("1"), 96).unwrap();
    assert_eq!(flat.len(), 97);
    // a straight line: the exact position x0 + v k/96 at every step
    check("geodesic flat", worst(refs::GEODESIC_FLAT, |k| flat[k].clone()), BOUND_GEODESIC_FLAT);

    let h2 = geodesic_integrate(&HyperbolicMetric, &vecs(&["0", "1"]), &vecs(&["0.75", "0.5"]), fp("1"), 96).unwrap();
    assert_eq!(h2.len(), 97);
    check("geodesic H2", worst(refs::GEODESIC_H2, |k| h2[k].clone()), BOUND_GEODESIC_H2);
}

#[test]
fn parallel_transport_hyperbolic() {
    let dx = fp(H64);
    let dy = fp("0.0078125");
    // x_k = k/64 and y_k = 1 + k/128 by exact repeated addition (from_int(64)
    // is outside the range at 26 or more fraction bits)
    let mut horiz = Vec::new();
    let mut diag = Vec::new();
    let (mut x, mut y) = (FixedPoint::ZERO, FixedPoint::one());
    for _ in 0..65 {
        horiz.push(FixedVector::from_slice(&[x, FixedPoint::one()]));
        diag.push(FixedVector::from_slice(&[x, y]));
        x = x + dx;
        y = y + dy;
    }
    let v0 = vecs(&["0.5", "0.75"]);
    let got = [
        parallel_transport_ode(&HyperbolicMetric, &horiz, &v0, 0).unwrap(),
        parallel_transport_ode(&HyperbolicMetric, &diag, &v0, 0).unwrap(),
        parallel_transport_ode(&HyperbolicMetric, &diag, &v0, 16).unwrap(),
    ];
    check("transport H2", worst(refs::TRANSPORT_H2, |k| got[k].clone()), BOUND_TRANSPORT_H2);
}

// Bounds: the measured worst over realtime at 8, 10, 12, 16, 20 and 24
// fraction bits, compact, embedded, balanced and scientific.
const BOUND_RK4_OSC: i32 = 1;
const BOUND_RK4_FORCED: i32 = 1;
const BOUND_DP45_OSC: i32 = 1;
const BOUND_VERLET_OSC: i32 = 1;
const BOUND_GEODESIC_FLAT: i32 = 0;
const BOUND_GEODESIC_H2: i32 = 1;
const BOUND_TRANSPORT_H2: i32 = 1;
