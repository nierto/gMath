//! Curvature at the compute tier, against exact and mpmath references.
//!
//! References: `tests/data/curvature_compute_tier_refs.rs` from
//! `scripts/generate_curvature_compute_tier_refs.py`: the SAME scheme the
//! library runs (central differences with this build's step h = 2^-k for the
//! metric partials and the Christoffel derivatives, exact inverse metric,
//! exact contractions), so every error below is rounding, not the O(h^2)
//! discretization error. The polynomial metric is exact at every point the
//! scheme evaluates (exact rational references); the sphere and hyperbolic
//! closed forms are mpmath at 300 digits. Each literal is parsed at the
//! build's split and errors are in storage units (one unit = 2^-FRAC_BITS).
//!
//! Measured worst error in storage units over realtime at 8, 10, 12, 16, 20
//! and 24 fraction bits, compact, embedded, balanced and scientific, 0.6.3
//! (storage-rounded symbols, differenced and multiplied by 2^(k-1)) -> 0.6.4
//! (compute tier, one rounding):
//!   poly christoffel                    1 -> 0
//!   poly riemann / ricci   5 (F8) .. 1.0e25 (scientific) -> 0
//!   poly scalar            4 (F10) .. 6.0e24 (scientific) -> 0
//!   sphere riemann / ricci 1 (F10) .. 1.5e25 (scientific) -> 0
//!   sphere / hyperbolic closed forms  1 -> 0
//!   sectional (poly orthogonal, poly near-parallel, sphere): 73 .. 7.5e6 on
//!     realtime, 1.9e9 compact, 8.2e18 embedded, 1.5e38 balanced, Err on
//!     scientific -> 0, except 2 for the near-parallel pair at 8 fraction
//!     bits (its denominator is 1.3 storage units there). Before 0.6.4 the
//!     contraction was R^l_ijk u^i v^j v^k, zero in exact arithmetic: the
//!     results were rounding noise, not the curvature.

use g_math::fixed_point::imperative::curvature::{
    christoffel, differentiation_step, ricci_tensor, riemann_curvature, scalar_curvature,
    sectional_curvature, HyperbolicMetric, MetricProvider, SphereMetric,
};
use g_math::fixed_point::{FixedMatrix, FixedPoint, FixedVector};

#[allow(dead_code)]
mod refs {
    include!("data/curvature_compute_tier_refs.rs");
}

fn fp(s: &str) -> FixedPoint {
    if let Some(rest) = s.strip_prefix('-') { -FixedPoint::from_str(rest) } else { FixedPoint::from_str(s) }
}
fn vecs(v: &[&str]) -> FixedVector { FixedVector::from_slice(&v.iter().map(|s| fp(s)).collect::<Vec<_>>()) }

/// |got - want| in storage units (saturating: an error beyond i128 is
/// reported as i128::MAX).
fn units(got: FixedPoint, want: FixedPoint) -> i128 {
    let d = (got - want).abs().raw();
    #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
    { d as i128 }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    { if d.fits_in_i128() { d.as_i128() } else { i128::MAX } }
}

/// Worst over the listed (got, want) pairs.
fn worst(pairs: impl IntoIterator<Item = (FixedPoint, &'static str)>) -> i128 {
    pairs.into_iter().map(|(g, w)| units(g, fp(w))).max().unwrap_or(0)
}

/// The measured value is printed so the finding can be updated if it moves.
fn check(name: &str, worst: i128, bound: i128) {
    println!("F={} {name}: worst {} units (bound {bound})", g_math::fixed_point::frac_config::FRAC_BITS,
        if worst == i128::MAX { "Err/overflow".to_string() } else { worst.to_string() });
    assert!(worst <= bound, "{name}: {worst} units > {bound}");
}

/// k with differentiation_step() = 2^-k.
fn step_k() -> u32 {
    let h = differentiation_step();
    let mut p = FixedPoint::one();
    for k in 0..200 {
        if p == h { return k; }
        p = p * fp("0.5");
    }
    panic!("differentiation step is not a power of two");
}

/// g = [[1 + x^2, x y / 2], [x y / 2, 1 + y^2]], no closed forms.
struct Poly;

impl MetricProvider for Poly {
    fn dimension(&self) -> usize { 2 }
    fn metric(&self, p: &FixedVector) -> FixedMatrix {
        let (x, y) = (p[0], p[1]);
        let one = FixedPoint::one();
        let off = x * y * fp("0.5");
        FixedMatrix::from_slice(2, 2, &[one + x * x, off, off, one + y * y])
    }
}

fn flat3(t: &g_math::fixed_point::imperative::tensor::Tensor) -> Vec<FixedPoint> {
    let mut v = Vec::new();
    for a in 0..2 { for b in 0..2 { for c in 0..2 { v.push(t.get(&[a, b, c])); } } }
    v
}
fn flat4(t: &g_math::fixed_point::imperative::tensor::Tensor) -> Vec<FixedPoint> {
    let mut v = Vec::new();
    for a in 0..2 { for b in 0..2 { for c in 0..2 { for d in 0..2 { v.push(t.get(&[a, b, c, d])); } } } }
    v
}
fn flat2(m: &FixedMatrix) -> Vec<FixedPoint> { vec![m.get(0, 0), m.get(0, 1), m.get(1, 0), m.get(1, 1)] }

fn pairs<const N: usize>(got: Vec<FixedPoint>, want: &[&'static str; N]) -> Vec<(FixedPoint, &'static str)> {
    got.into_iter().zip(want.iter().copied()).collect()
}

#[test]
fn polynomial_metric_curvature() {
    let k = step_k();
    let rows: Vec<&refs::PolyRefs> = refs::POLY.iter().filter(|r| r.k == k).collect();
    if rows.is_empty() {
        println!("no reference for step 2^-{k} (FRAC_BITS below 8): skipped");
        return;
    }
    let (mut gam, mut rie, mut ric, mut sca, mut sec_orth, mut sec_par) = (0, 0, 0, 0, 0, 0);
    for r in rows {
        let p = vecs(&refs::POINTS[r.point]);
        gam = gam.max(worst(pairs(flat3(&christoffel(&Poly, &p).unwrap()), &r.christoffel)));
        rie = rie.max(worst(pairs(flat4(&riemann_curvature(&Poly, &p).unwrap()), &r.riemann)));
        ric = ric.max(worst(pairs(flat2(&ricci_tensor(&Poly, &p).unwrap()), &r.ricci)));
        sca = sca.max(units(scalar_curvature(&Poly, &p).unwrap(), fp(r.scalar)));
        for (i, uv) in refs::PAIRS.iter().enumerate() {
            let e = match sectional_curvature(&Poly, &p, &vecs(&uv[0]), &vecs(&uv[1])) {
                Ok(s) => units(s, fp(r.sectional[i])),
                Err(_) => i128::MAX,
            };
            if i == 0 { sec_orth = sec_orth.max(e) } else { sec_par = sec_par.max(e) }
        }
    }
    check("poly christoffel", gam, BOUND_POLY_CHRISTOFFEL);
    check("poly riemann", rie, BOUND_POLY_RIEMANN);
    check("poly ricci", ric, BOUND_POLY_RICCI);
    check("poly scalar", sca, BOUND_POLY_SCALAR);
    check("poly sectional orthogonal", sec_orth, BOUND_POLY_SECTIONAL);
    check("poly sectional near-parallel", sec_par, BOUND_POLY_SECTIONAL_NEAR_PARALLEL);
}

#[test]
fn sphere_curvature_from_closed_form_symbols() {
    let k = step_k();
    let Some(r) = refs::SPHERE.iter().find(|r| r.k == k) else {
        println!("no reference for step 2^-{k} (FRAC_BITS below 8): skipped");
        return;
    };
    let s2 = SphereMetric { radius: fp(refs::SPHERE_RADIUS) };
    let p = vecs(&refs::SPHERE_POINT);
    let rie = worst(pairs(flat4(&riemann_curvature(&s2, &p).unwrap()), &r.riemann));
    let ric = worst(pairs(flat2(&ricci_tensor(&s2, &p).unwrap()), &r.ricci));
    let sec = match sectional_curvature(&s2, &p, &vecs(&["1", "0"]), &vecs(&["0", "1"])) {
        Ok(s) => units(s, fp(r.sectional)),
        Err(_) => i128::MAX,
    };
    check("sphere riemann", rie, BOUND_SPHERE_RIEMANN);
    check("sphere ricci", ric, BOUND_SPHERE_RICCI);
    check("sphere sectional", sec, BOUND_SPHERE_SECTIONAL);
}

#[test]
fn closed_forms_round_once() {
    let s2 = SphereMetric { radius: fp(refs::SPHERE_RADIUS) };
    let p = vecs(&refs::SPHERE_POINT);
    let g = s2.metric(&p);
    let gamma = s2.christoffel_closed_form(&p).unwrap();
    let sphere = worst([
        (g.get(1, 1), refs::SPHERE_G11),
        (gamma.get(&[0, 1, 1]), refs::SPHERE_GAMMA_011),
        (gamma.get(&[1, 0, 1]), refs::SPHERE_GAMMA_101),
        (gamma.get(&[1, 1, 0]), refs::SPHERE_GAMMA_101),
        (s2.scalar_curvature_closed_form(&p).unwrap(), refs::SPHERE_SCALAR),
    ]);
    let q = vecs(&refs::HYP_POINT);
    let hyp = worst([
        (HyperbolicMetric.metric(&q).get(0, 0), refs::HYP_G00),
        (HyperbolicMetric.metric(&q).get(1, 1), refs::HYP_G00),
        (HyperbolicMetric.christoffel_closed_form(&q).unwrap().get(&[1, 0, 0]), refs::HYP_GAMMA_100),
    ]);
    check("sphere closed forms", sphere, BOUND_CLOSED_FORMS);
    check("hyperbolic closed forms", hyp, BOUND_CLOSED_FORMS);
}

// Bounds: the measured worst over realtime at 8, 10, 12, 16, 20 and 24
// fraction bits, compact, embedded, balanced and scientific.
const BOUND_POLY_CHRISTOFFEL: i128 = 0;
const BOUND_POLY_RIEMANN: i128 = 0;
const BOUND_POLY_RICCI: i128 = 0;
const BOUND_POLY_SCALAR: i128 = 0;
const BOUND_POLY_SECTIONAL: i128 = 0;
const BOUND_POLY_SECTIONAL_NEAR_PARALLEL: i128 = 2;
const BOUND_SPHERE_RIEMANN: i128 = 0;
const BOUND_SPHERE_RICCI: i128 = 0;
const BOUND_SPHERE_SECTIONAL: i128 = 0;
const BOUND_CLOSED_FORMS: i128 = 0;
