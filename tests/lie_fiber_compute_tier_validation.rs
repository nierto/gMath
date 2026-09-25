//! Lie groups and fiber bundles at the compute tier, against mpmath.
//!
//! SO(3) / SE(3) exp and log, the Manifold chains built on them (exp_map,
//! log_map, distance, SO(3) parallel transport), SE(3) inverse, adjoint and
//! bracket, the SO(n) / GL(n) / SL(n) chains, adjoints and brackets, SL(n)
//! project_traceless, and the vector bundle's horizontal lift, discrete
//! parallel transport, curvature and a principal bundle transition inverse.
//! Before 0.6.4 each of these rounded its intermediates to storage (theta^2,
//! the Rodrigues coefficients, group products fed to a log, LU inverses,
//! fiber states between transport steps); they now carry them at the compute
//! tier and round each output once.
//!
//! References: `tests/data/lie_fiber_compute_tier_refs.rs` from
//! `scripts/generate_lie_fiber_compute_tier_refs.py` (exact rationals or
//! mpmath at 120 digits). Inputs are dyadic; rows whose inputs need more
//! fraction bits than the build has are skipped. Every reference is parsed at
//! the build's split, so it is the correctly rounded result, and every error
//! below is in storage units (2^-FRAC_BITS).

use g_math::fixed_point::imperative::fiber_bundle::{
    vector_bundle_curvature, BundleConnection, PrincipalBundle, VectorBundle,
};
use g_math::fixed_point::imperative::lie_group::{GLn, LieGroup, SLn, SOn, SE3, SO3};
use g_math::fixed_point::imperative::manifold::Manifold;
use g_math::fixed_point::imperative::tensor::Tensor;
use g_math::fixed_point::{FixedMatrix, FixedPoint, FixedVector};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[allow(dead_code)]
mod refs {
    include!("data/lie_fiber_compute_tier_refs.rs");
}

const FRAC_BITS: u32 = g_math::fixed_point::frac_config::FRAC_BITS as u32;

/// Reported for a call that returned `Err` or panicked.
const FAILED: i64 = 1_000_000_000;
/// Errors are capped here (the tables only need to show "huge").
const CAP: i64 = 1_000_000;

/// Rows of the small / moderate / near-pi groups of the exp and round-trip tables.
const SMALL: usize = 10;
const MODERATE: usize = 8;

fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }
fn vecs(v: &[&str]) -> FixedVector { FixedVector::from_slice(&v.iter().map(|s| fp(s)).collect::<Vec<_>>()) }
fn mat(n: usize, v: &[&str]) -> FixedMatrix { FixedMatrix::from_slice(n, n, &v.iter().map(|s| fp(s)).collect::<Vec<_>>()) }

/// |got - want| in storage units, capped at CAP.
fn units(got: FixedPoint, want: FixedPoint) -> i64 {
    let d = if got > want { got - want } else { want - got };
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    { (d.raw() as i64).min(CAP) }
    #[cfg(table_format = "q64_64")]
    { d.raw().min(CAP as i128) as i64 }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    {
        let half = fp("0.5");
        let mut unit = FixedPoint::one();
        for _ in 0..FRAC_BITS { unit = unit * half; }
        if d >= unit * FixedPoint::from_int(CAP as i32) { CAP } else { (d / unit).to_int() as i64 }
    }
}

/// Worst error of a computed slice against references; a failed call counts
/// as FAILED.
fn worst_of<T>(got: std::thread::Result<Result<T, g_math::fixed_point::OverflowDetected>>, want: &[&str], get: impl Fn(&T, usize) -> FixedPoint) -> i64 {
    match got {
        Ok(Ok(v)) => (0..want.len()).map(|k| units(get(&v, k), fp(want[k]))).max().unwrap_or(0),
        _ => FAILED,
    }
}

fn run<T>(f: impl FnOnce() -> Result<T, g_math::fixed_point::OverflowDetected>) -> std::thread::Result<Result<T, g_math::fixed_point::OverflowDetected>> {
    catch_unwind(AssertUnwindSafe(f))
}

fn vec_entry(v: &FixedVector, k: usize) -> FixedPoint { v[k] }

/// Worst error over a table and the bound it must meet; the measured value
/// is printed so the finding can be updated if it moves.
fn check(name: &str, worst: i64, bound: i64) {
    println!("{name}: worst {worst} units (bound {bound})");
    assert!(worst <= bound, "{name}: {worst} units > {bound}");
}

// ============================================================================
// SO(3) / SE(3) exp, including the small angles where theta^2 was rounded
// ============================================================================

#[test]
fn so3_exp_small_moderate_near_pi() {
    let mut worst = [0i64; 3];
    for (idx, (bits, w, e)) in refs::SO3_EXP.iter().enumerate() {
        if *bits > FRAC_BITS { continue; }
        let g = if idx < SMALL { 0 } else if idx < SMALL + MODERATE { 1 } else { 2 };
        let err = worst_of(run(|| SO3::rodrigues_exp(&vecs(w))), e, |m: &FixedMatrix, k| m.get(k / 3, k % 3));
        worst[g] = worst[g].max(err);
    }
    // all at the compute tier, one rounding per entry. Before 0.6.4 theta^2
    // was rounded to storage: Err(DivisionByZero) for the small rows on
    // realtime (theta^2 below half a unit)
    check("so3 exp small", worst[0], 1);
    check("so3 exp moderate", worst[1], 0);
    check("so3 exp near pi", worst[2], 0);
}

#[test]
fn se3_exp_small_moderate_near_pi() {
    let mut worst = [0i64; 3];
    for (idx, (bits, xi, e)) in refs::SE3_EXP.iter().enumerate() {
        if *bits > FRAC_BITS { continue; }
        let g = if idx < SMALL { 0 } else if idx < SMALL + MODERATE { 1 } else { 2 };
        let err = worst_of(run(|| SE3::se3_exp(&vecs(xi))), e, |m: &FixedMatrix, k| m.get(k / 4, k % 4));
        worst[g] = worst[g].max(err);
    }
    // small rows: before 0.6.4 Err(DivisionByZero) on realtime and the
    // Taylor branch's missing theta^2 terms on the wide profiles (746 units
    // at |omega| = 8.8e-6 on Q64.64, over 10^6 on Q128.128)
    check("se3 exp small", worst[0], 1);
    check("se3 exp moderate", worst[1], 0);
    check("se3 exp near pi", worst[2], 1);
}

// ============================================================================
// log(exp(x)) round trips: the input to log is the STORAGE-rounded group
// element, so these measure the log's sensitivity to that rounding
// ============================================================================

#[test]
fn so3_log_round_trip() {
    let mut worst = [0i64; 3];
    for (idx, (bits, w)) in refs::SO3_ROUND_TRIP.iter().enumerate() {
        if *bits > FRAC_BITS { continue; }
        let g = if idx < SMALL { 0 } else if idx < SMALL + MODERATE { 1 } else { 2 };
        let err = worst_of(run(|| SO3::rodrigues_log(&SO3::rodrigues_exp(&vecs(w))?)), w, vec_entry);
        worst[g] = worst[g].max(err);
    }
    // the log's input is exp(omega) rounded to storage (half a unit per
    // entry), so a unit or two here is that rounding carried through the log,
    // not the log's own error (the chains below, whose group elements stay at
    // the compute tier, meet 1). Near pi the old theta / (2 sin theta) form
    // amplified it to tens of thousands of units
    check("so3 log(exp) small", worst[0], 0);
    check("so3 log(exp) moderate", worst[1], 1);
    check("so3 log(exp) near pi", worst[2], 2);
}

#[test]
fn se3_log_round_trip() {
    let mut worst = [0i64; 3];
    for (idx, (bits, xi)) in refs::SE3_ROUND_TRIP.iter().enumerate() {
        if *bits > FRAC_BITS { continue; }
        let g = if idx < SMALL { 0 } else if idx < SMALL + MODERATE { 1 } else { 2 };
        let err = worst_of(run(|| SE3::se3_log(&SE3::se3_exp(&vecs(xi))?)), xi, vec_entry);
        worst[g] = worst[g].max(err);
    }
    // as above, plus V^-1 (entries up to about 3 near pi) applied to the
    // storage-rounded translation
    check("se3 log(exp) small", worst[0], 1);
    check("se3 log(exp) moderate", worst[1], 2);
    check("se3 log(exp) near pi", worst[2], 3);
}

// ============================================================================
// Manifold chains: the group products stay at the compute tier
// ============================================================================

#[test]
fn so3_manifold_chains() {
    let (mut we, mut wl, mut wd, mut wp) = (0, 0, 0, 0);
    for (a, b, r) in refs::SO3_EXP_MAP {
        we = we.max(worst_of(run(|| SO3.exp_map(&vecs(a), &vecs(b))), r, vec_entry));
    }
    for (a, b, r, d, t, p) in refs::SO3_LOG_MAP {
        wl = wl.max(worst_of(run(|| SO3.log_map(&vecs(a), &vecs(b))), r, vec_entry));
        wd = wd.max(worst_of(run(|| SO3.distance(&vecs(a), &vecs(b))), &[d], |x: &FixedPoint, _| *x));
        wp = wp.max(worst_of(run(|| SO3.parallel_transport(&vecs(a), &vecs(b), &vecs(t))), p, vec_entry));
    }
    check("so3 exp_map", we, 1);
    check("so3 log_map", wl, 1);
    check("so3 distance", wd, 0);
    // 1 at 8 fraction bits: the compute tier's 16 bits leave the exact value
    // close enough to a rounding boundary to land one unit off
    check("so3 parallel transport", wp, 1);
}

#[test]
fn se3_manifold_chains() {
    let (mut we, mut wl, mut wd) = (0, 0, 0);
    for (a, b, em, lm, d) in refs::SE3_MAPS {
        we = we.max(worst_of(run(|| SE3.exp_map(&vecs(a), &vecs(b))), em, vec_entry));
        wl = wl.max(worst_of(run(|| SE3.log_map(&vecs(a), &vecs(b))), lm, vec_entry));
        wd = wd.max(worst_of(run(|| SE3.distance(&vecs(a), &vecs(b))), &[d], |x: &FixedPoint, _| *x));
    }
    check("se3 exp_map", we, 1);
    check("se3 log_map", wl, 1);
    check("se3 distance", wd, 0);
}

#[test]
fn se3_so3_algebra_exact() {
    let (mut wi, mut wa, mut wb, mut ws) = (0, 0, 0, 0);
    for (r, t, xi, inv_t, adj, br, eta, w) in refs::SE3_ALGEBRA {
        let g = SE3::from_rt(&mat(3, r), &vecs(t));
        wi = wi.max(worst_of(run(|| SE3.group_inverse(&g)), inv_t, |m: &FixedMatrix, k| m.get(k, 3)));
        wa = wa.max(worst_of(run(|| SE3.adjoint(&g, &vecs(xi))), adj, vec_entry));
        wb = wb.max(worst_of(run(|| Ok(SE3.bracket(&vecs(xi), &vecs(eta)))), br, vec_entry));
        let (w1, w2) = (vecs(&xi[..3]), vecs(&eta[..3]));
        ws = ws.max(worst_of(run(|| Ok(SO3.bracket(&w1, &w2))), w, vec_entry));
    }
    // exact references: each output is one rounding of an exact sum
    check("se3 inverse translation", wi, 0);
    check("se3 adjoint", wa, 0);
    check("se3 bracket", wb, 0);
    check("so3 bracket", ws, 0);
}

// ============================================================================
// SO(n), GL(n), SL(n)
// ============================================================================

#[test]
fn son_chains_and_algebra() {
    let son = SOn { n: 4 };
    let (mut we, mut wl, mut wd) = (0, 0, 0);
    for (a, b, em, lm, d) in refs::SO4_MAPS {
        we = we.max(worst_of(run(|| son.exp_map(&vecs(a), &vecs(b))), em, vec_entry));
        wl = wl.max(worst_of(run(|| son.log_map(&vecs(a), &vecs(b))), lm, vec_entry));
        wd = wd.max(worst_of(run(|| son.distance(&vecs(a), &vecs(b))), &[d], |x: &FixedPoint, _| *x));
    }
    let (mut wg, mut wa, mut wb) = (0, 0, 0);
    for (g, lg, xi, adj, eta, br) in refs::SO4_ALGEBRA {
        let g = mat(4, g);
        wg = wg.max(worst_of(run(|| son.lie_log(&g)), lg, vec_entry));
        wa = wa.max(worst_of(run(|| son.adjoint(&g, &vecs(xi))), adj, vec_entry));
        wb = wb.max(worst_of(run(|| Ok(son.bracket(&vecs(xi), &vecs(eta)))), br, vec_entry));
    }
    check("so4 exp_map", we, 1);
    check("so4 log_map", wl, 1);
    // 1 at 8 fraction bits: the matrix log at the compute tier's 16 bits
    check("so4 distance", wd, 1);
    check("so4 lie_log", wg, 0);
    check("so4 adjoint", wa, 0);
    check("so4 bracket", wb, 0);
}

fn gl_sl_case<G: LieGroup>(name: &str, group: &G, n: usize,
    maps: &[(&[&str], &[&str], &[&str], &[&str], &str)],
    algebra: &[(&[&str], &[&str], &[&str], &[&str], &[&str], &[&str])],
    bounds: [i64; 6]) {
    let (mut we, mut wl, mut wd, mut wa, mut wb, mut wi) = (0, 0, 0, 0, 0, 0);
    for (a, b, em, lm, d) in maps {
        we = we.max(worst_of(run(|| group.exp_map(&vecs(a), &vecs(b))), em, vec_entry));
        wl = wl.max(worst_of(run(|| group.log_map(&vecs(a), &vecs(b))), lm, vec_entry));
        wd = wd.max(worst_of(run(|| group.distance(&vecs(a), &vecs(b))), &[d], |x: &FixedPoint, _| *x));
    }
    for (g, xi, adj, eta, br, inv) in algebra {
        let g = mat(n, g);
        wa = wa.max(worst_of(run(|| group.adjoint(&g, &vecs(xi))), adj, vec_entry));
        wb = wb.max(worst_of(run(|| Ok(group.bracket(&vecs(xi), &vecs(eta)))), br, vec_entry));
        wi = wi.max(worst_of(run(|| group.group_inverse(&g)), inv, |m: &FixedMatrix, k| m.get(k / n, k % n)));
    }
    check(&format!("{name} exp_map"), we, bounds[0]);
    check(&format!("{name} log_map"), wl, bounds[1]);
    check(&format!("{name} distance"), wd, bounds[2]);
    check(&format!("{name} adjoint"), wa, bounds[3]);
    check(&format!("{name} bracket"), wb, bounds[4]);
    check(&format!("{name} group_inverse"), wi, bounds[5]);
}

#[test]
fn gln_sln_chains_and_algebra() {
    // [exp_map, log_map, distance, adjoint, bracket, group_inverse]: the
    // inverse (compute-tier LU) is not exact, so adjoint and inverse can be a
    // unit off; the bracket is one rounding of an exact value
    gl_sl_case("gl2", &GLn { n: 2 }, 2, refs::GL2_MAPS, refs::GL2_ALGEBRA, [1, 0, 0, 0, 0, 0]);
    gl_sl_case("sl2", &SLn { n: 2 }, 2, refs::SL2_MAPS, refs::SL2_ALGEBRA, [0, 0, 0, 1, 0, 1]);
    // distance and adjoint 1 at 8 and 10 fraction bits: the matrix log and
    // the LU inverse run at the realtime compute tier's 2F bits (16, 20)
    gl_sl_case("sl3", &SLn { n: 3 }, 3, refs::SL3_MAPS, refs::SL3_ALGEBRA, [0, 0, 1, 1, 0, 1]);
}

#[test]
fn sln_project_traceless_is_one_rounding() {
    let mut worst = 0;
    for (n, m, r) in refs::PROJECT_TRACELESS {
        worst = worst.max(worst_of(run(|| Ok(SLn::project_traceless(&mat(*n, m)))), r, |x: &FixedMatrix, k| x.get(k / *n, k % *n)));
    }
    check("sln project_traceless", worst, 0);
}

// ============================================================================
// Fiber bundles
// ============================================================================

fn bundle(n: usize, k: usize, coeffs: &[&str]) -> VectorBundle {
    VectorBundle::with_connection(n, k, coeffs.iter().map(|s| fp(s)).collect())
}

#[test]
fn vector_bundle_lift_transport_curvature() {
    let (mut wh, mut wt, mut wc) = (0, 0, 0);
    for (n, k, c, tp, v, lift) in refs::HORIZONTAL_LIFT {
        let b = bundle(*n, *k, c);
        wh = wh.max(worst_of(run(|| b.horizontal_lift(&vecs(tp), &vecs(v))), lift, vec_entry));
    }
    for (n, k, c, path, xi0, xi) in refs::TRANSPORT_ALONG {
        let b = bundle(*n, *k, c);
        let pts: Vec<FixedVector> = path.chunks(*n).map(|p| vecs(p)).collect();
        wt = wt.max(worst_of(run(|| b.parallel_transport_along(&pts, &vecs(xi0))), xi, vec_entry));
    }
    for (n, k, c, curv) in refs::CURVATURE {
        let b = bundle(*n, *k, c);
        let (n, k) = (*n, *k);
        wc = wc.max(worst_of(run(|| vector_bundle_curvature(&b, &FixedVector::new(n))), curv, |t: &Tensor, idx| {
            let (a, r) = (idx / (k * n * n), idx % (k * n * n));
            let (bb, r) = (r / (n * n), r % (n * n));
            t.get(&[a, bb, r / n, r % n])
        }));
    }
    check("horizontal lift", wh, 0);
    check("parallel transport along", wt, 0);
    check("bundle curvature", wc, 0);
}

#[test]
fn principal_bundle_transition_inverse() {
    let mut worst = 0;
    for (g, inv) in refs::TRANSITION_INVERSE {
        let mut p = PrincipalBundle::trivial(2, 3, 3, 2);
        let r = run(|| { p.set_transition(0, 1, mat(3, g))?; Ok(p.transition(1, 0).clone()) });
        worst = worst.max(worst_of(r, inv, |m: &FixedMatrix, k| m.get(k / 3, k % 3)));
    }
    check("transition inverse", worst, 0);
}
