//! Manifold operations against mpmath after the compute-tier rework: every
//! exp map, log map, distance and transport carries its state at the compute
//! tier (2 x FRAC_BITS) and rounds to storage once per output.
//!
//! References: `tests/data/manifold_compute_tier_refs.rs` from
//! `scripts/generate_manifold_compute_tier_refs.py`. SPD, Stiefel, the norms
//! and the product inner product use dyadic inputs (exact on every gated
//! build) with exact or 160-digit references. Points on the sphere, the
//! hyperboloid and the Grassmannian have no dyadic coordinates: their inputs
//! are ideal points written with 100 decimals, rounded at the build's split
//! by the parser, and the generator evaluates the references on the stored
//! values for each split (8, 10, 12, 16, 20, 24 fraction bits and the four
//! wide profiles). Every error below is in storage units against the
//! reference parsed at the build's split. "close" cases are points within
//! 0.001 to 0.03 of each other, where the angle used to come from acos or
//! acosh of a value rounded to storage.

use g_math::fixed_point::imperative::manifold::{
    EuclideanSpace, Grassmannian, HyperbolicSpace, Manifold, ProductManifold, SPDManifold, Sphere, StiefelManifold,
};
use g_math::fixed_point::{FixedPoint, FixedVector};

#[allow(dead_code)]
mod refs {
    include!("data/manifold_compute_tier_refs.rs");
}

const F: u32 = g_math::fixed_point::frac_config::FRAC_BITS;

fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }
fn vecs(v: &[&str]) -> FixedVector { FixedVector::from_slice(&v.iter().map(|s| fp(s)).collect::<Vec<_>>()) }

/// A large sentinel for an error the doubling below cannot resolve (or an Err).
const HUGE: i64 = 1 << 40;

/// |got - want| in storage units (one unit = 2^-FRAC_BITS). Doubles the
/// difference F times (exact) instead of dividing by 2^-F, which overflows
/// the narrow splits for large errors; saturates at HUGE.
fn units(got: FixedPoint, want: FixedPoint) -> i64 {
    let mut d = (got - want).abs();
    let limit = FixedPoint::from_int(16);
    for r in 0..F {
        if d >= limit {
            let left = F - r;
            return if left > 34 { HUGE } else { ((d.to_int() as i64) << left).min(HUGE) };
        }
        d = d + d;
    }
    d.to_int() as i64
}

fn units_res(got: Result<FixedPoint, g_math::fixed_point::imperative::OverflowDetected>, want: &str) -> i64 {
    match got { Ok(g) => units(g, fp(want)), Err(_) => HUGE }
}

fn worst_vec(got: &Result<FixedVector, g_math::fixed_point::imperative::OverflowDetected>, want: &[&str]) -> i64 {
    match got {
        Ok(g) => {
            assert_eq!(g.len(), want.len());
            (0..want.len()).map(|k| units(g[k], fp(want[k]))).max().unwrap_or(0)
        }
        Err(_) => HUGE,
    }
}

/// Measured worst against its bound; the value is printed so the finding can
/// be updated if it moves.
fn check(name: &str, worst: i64, bound: i64) {
    println!("{name}: worst {worst} units (bound {bound})");
    assert!(worst <= bound, "{name}: {worst} units > {bound}");
}

/// Per-split tables exist for the gated splits only.
fn split_has_table() -> bool {
    let found = refs::SPLITS.contains(&F);
    if !found { println!("no per-split references for FRAC_BITS = {F}: skipped"); }
    found
}

/// Worst over (close, far) cases; the per-case list is printed.
fn grouped(name: &str, errs: &[(i64, bool)]) -> (i64, i64) {
    println!("{name} per case: {:?}", errs.iter().map(|e| e.0).collect::<Vec<_>>());
    let close = errs.iter().filter(|e| e.1).map(|e| e.0).max().unwrap_or(0);
    let far = errs.iter().filter(|e| !e.1).map(|e| e.0).max().unwrap_or(0);
    (close, far)
}

#[test]
fn sphere_against_mpmath() {
    if !split_has_table() { return; }
    let s = Sphere { dim: 2 };
    let (mut dist, mut log, mut pt) = (vec![], vec![], vec![]);
    for (f, i, d, l, t) in refs::SPHERE {
        if *f != F { continue; }
        let (p, q, v, close) = refs::SPHERE_PAIRS[*i];
        let (p, q, v) = (vecs(p), vecs(q), vecs(v));
        dist.push((units_res(s.distance(&p, &q), d), close));
        log.push((worst_vec(&s.log_map(&p, &q), l), close));
        pt.push((worst_vec(&s.parallel_transport(&p, &q, &v), t), close));
    }
    let mut exp = vec![];
    for (f, i, e) in refs::SPHERE_EXP {
        if *f != F { continue; }
        let (p, v) = refs::SPHERE_EXPS[*i];
        exp.push((worst_vec(&s.exp_map(&vecs(p), &vecs(v)), e), false));
    }
    let (dc, df) = grouped("sphere distance", &dist);
    let (lc, lf) = grouped("sphere log_map", &log);
    let (pc, pf) = grouped("sphere parallel_transport", &pt);
    let (_, ef) = grouped("sphere exp_map", &exp);
    // atan2(|p x q|, p.q) from exact products, direction and scaling at the
    // compute tier, one rounding: the correctly rounded result on every build.
    // Before 0.6.4 (acos of p.q rounded to storage): close up to 1024 units
    // (Q12.20), log far up to 88, transport near the antipode up to 640, and
    // exp_map / log_map panicked on Q8.24 (1 / theta beyond the range).
    check("sphere distance close", dc, 0);
    check("sphere distance far", df, 0);
    check("sphere log_map close", lc, 0);
    check("sphere log_map far", lf, 0);
    check("sphere parallel_transport close", pc, 0);
    check("sphere parallel_transport far", pf, 0);
    check("sphere exp_map", ef, 0);
}

#[test]
fn hyperbolic_against_mpmath() {
    if !split_has_table() { return; }
    let h = HyperbolicSpace { dim: 2 };
    let (mut dist, mut log, mut pt) = (vec![], vec![], vec![]);
    for (f, i, d, l, t) in refs::HYP {
        if *f != F { continue; }
        let (p, q, v, close) = refs::HYP_PAIRS[*i];
        let (p, q, v) = (vecs(p), vecs(q), vecs(v));
        dist.push((units_res(h.distance(&p, &q), d), close));
        log.push((worst_vec(&h.log_map(&p, &q), l), close));
        pt.push((worst_vec(&h.parallel_transport(&p, &q, &v), t), close));
    }
    let mut exp = vec![];
    for (f, i, e) in refs::HYP_EXP {
        if *f != F { continue; }
        let (p, v) = refs::HYP_EXPS[*i];
        exp.push((worst_vec(&h.exp_map(&vecs(p), &vecs(v)), e), false));
    }
    let (dc, df) = grouped("hyperbolic distance", &dist);
    let (lc, lf) = grouped("hyperbolic log_map", &log);
    let (pc, pf) = grouped("hyperbolic parallel_transport", &pt);
    let (_, ef) = grouped("hyperbolic exp_map", &exp);
    // acosh(a / sqrt(PP QQ)) with a^2 - PP QQ exact, sinh/cosh from one exp
    // pair, everything at the compute tier, one rounding: correctly rounded on
    // every build. Before 0.6.4 (acosh of -<p,q> rounded to storage): close
    // up to 992 units, transport up to 448, exp_map up to 8, and a panic on
    // Q8.24.
    check("hyperbolic distance close", dc, 0);
    check("hyperbolic distance far", df, 0);
    check("hyperbolic log_map close", lc, 0);
    check("hyperbolic log_map far", lf, 0);
    check("hyperbolic parallel_transport close", pc, 0);
    check("hyperbolic parallel_transport far", pf, 0);
    check("hyperbolic exp_map", ef, 0);
}

#[test]
fn grassmannian_against_mpmath() {
    if !split_has_table() { return; }
    let (mut dist, mut log, mut pt) = (vec![], vec![], vec![]);
    for (f, i, d, l, t) in refs::GRASS {
        if *f != F { continue; }
        let (k, n, q1, q2, de, close) = refs::GRASS_CASES[*i];
        let g = Grassmannian { k, n };
        let (q1, q2, de) = (vecs(q1), vecs(q2), vecs(de));
        dist.push((units_res(g.distance(&q1, &q2), d), close));
        log.push((worst_vec(&g.log_map(&q1, &q2), l), close));
        pt.push((worst_vec(&g.parallel_transport(&q1, &q2, &de), t), close));
    }
    let mut exp = vec![];
    for (f, i, e) in refs::GRASS_EXP {
        if *f != F { continue; }
        let (k, n, q, de) = refs::GRASS_EXPS[*i];
        exp.push((worst_vec(&Grassmannian { k, n }.exp_map(&vecs(q), &vecs(de)), e), false));
    }
    let (dc, df) = grouped("grassmannian distance", &dist);
    let (lc, lf) = grouped("grassmannian log_map", &log);
    let (pc, pf) = grouped("grassmannian parallel_transport", &pt);
    let (_, ef) = grouped("grassmannian exp_map", &exp);
    // Principal angles atan2(|P_i|, |C_i|) at the compute tier from the SVD's
    // right vectors refined by compute-tier Jacobi. The log, transport and
    // exp use unit directions of stored frames that are orthonormal only to a
    // unit, and compute-tier products of them: measured 1 unit at most.
    // Before 0.6.4: the log and transport paired the i-th largest sine with
    // the i-th largest cosine and used the wrong right factor (sign flip for
    // k = 1 when Q1^T Q2 < 0): up to 2^40+ units on every build; distance
    // (acos of storage singular values) up to 64 units close.
    check("grassmannian distance close", dc, 0);
    check("grassmannian distance far", df, 0);
    check("grassmannian log_map close", lc, 1);
    check("grassmannian log_map far", lf, 1);
    check("grassmannian parallel_transport close", pc, 1);
    check("grassmannian parallel_transport far", pf, 1);
    check("grassmannian exp_map", ef, 1);
}

#[test]
fn product_distance_against_mpmath() {
    if !split_has_table() { return; }
    let m = ProductManifold::new(Box::new(Sphere { dim: 2 }), 3, Box::new(HyperbolicSpace { dim: 2 }), 3);
    let mut errs = vec![];
    for (f, i, d) in refs::PRODUCT_DISTANCE {
        if *f != F { continue; }
        let (sp, sq, _, close) = refs::SPHERE_PAIRS[*i];
        let (hp, hq, _, _) = refs::HYP_PAIRS[*i];
        let p = vecs(&[sp, hp].concat());
        let q = vecs(&[sq, hq].concat());
        errs.push((units_res(m.distance(&p, &q), d), close));
    }
    let (c, far) = grouped("product distance", &errs);
    // The components' distances arrive rounded (the Manifold trait returns
    // storage values); squares, sum and root at the compute tier then round
    // once: 1 unit at most (before 0.6.4: rounded squares, rounded sum and a
    // storage root, up to 896 units close through the component defects).
    check("product distance close", c, 1);
    check("product distance far", far, 1);
}

#[test]
fn product_inner_product_against_exact() {
    let m = ProductManifold::new(Box::new(Sphere { dim: 2 }), 3, Box::new(HyperbolicSpace { dim: 2 }), 3);
    let base = vecs(&["1", "0", "0", "1", "0", "0"]);
    let worst = refs::PRODUCT_INNER.iter()
        .map(|(u, v, r)| units(m.inner_product(&base, &vecs(u), &vecs(v)), fp(r)))
        .max().unwrap();
    // two correctly rounded parts added exactly: 1 unit at most
    check("product inner_product", worst, 1);
}

#[test]
fn spd_against_mpmath() {
    let (mut inner, mut exp, mut log, mut dist, mut pt, mut norm) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for (n, p, q, u, v, ip, ex, lg, d, t, nr) in refs::SPD {
        let s = SPDManifold { n: *n };
        let (p, q, u, v) = (vecs(p), vecs(q), vecs(u), vecs(v));
        inner.push(units(s.inner_product(&p, &u, &v), fp(ip)));
        exp.push(worst_vec(&s.exp_map(&p, &v), ex));
        log.push(worst_vec(&s.log_map(&p, &q), lg));
        dist.push(units_res(s.distance(&p, &q), d));
        pt.push(worst_vec(&s.parallel_transport(&p, &q, &v), t));
        norm.push(units(s.norm(&p, &v), fp(nr)));
    }
    for (name, e) in [("spd inner_product", &inner), ("spd exp_map", &exp), ("spd log_map", &log),
                      ("spd distance", &dist), ("spd parallel_transport", &pt), ("spd norm", &norm)] {
        println!("{name} per case: {e:?}");
    }
    // P^1/2, its inverse, the products and expm / logm at the compute tier,
    // one rounding per entry. log_map and distance measure 1 unit at 8 to 12
    // fraction bits, where the compute tier's 2F bits leave the matrix log's
    // square roots and its 2^s unscaling about a unit short; 0 elsewhere.
    // Before 0.6.4: exp_map up to 8 units, log_map up to 36, distance up to 9.
    check("spd inner_product", *inner.iter().max().unwrap(), 0);
    check("spd exp_map", *exp.iter().max().unwrap(), 0);
    check("spd log_map", *log.iter().max().unwrap(), 1);
    check("spd distance", *dist.iter().max().unwrap(), 1);
    check("spd parallel_transport", *pt.iter().max().unwrap(), 0);
    check("spd norm", *norm.iter().max().unwrap(), 0);
}

#[test]
fn stiefel_against_exact() {
    let (mut log, mut dist) = (vec![], vec![]);
    for (k, n, q, q2, lg, d) in refs::STIEFEL {
        let s = StiefelManifold { k: *k, n: *n };
        let (q, q2) = (vecs(q), vecs(q2));
        log.push(worst_vec(&s.log_map(&q, &q2), lg));
        dist.push(units_res(s.distance(&q, &q2), d));
    }
    println!("stiefel log_map per case: {log:?}");
    println!("stiefel distance per case: {dist:?}");
    check("stiefel log_map", *log.iter().max().unwrap(), 0);
    // before 0.6.4 distance returned sqrt(||Delta||_F): 64 units at Q24.8,
    // 2^40+ on the wide profiles
    check("stiefel distance", *dist.iter().max().unwrap(), 0);
}

#[test]
fn norms_against_exact() {
    let (mut sph, mut euc, mut gr, mut st) = (0, 0, 0, 0);
    for (v, r) in refs::NORMS {
        let x = vecs(v);
        let n = x.len();
        let want = fp(r);
        sph = sph.max(units(Sphere { dim: n - 1 }.norm(&x, &x), want));
        euc = euc.max(units(EuclideanSpace { dim: n }.norm(&x, &x), want));
        // an n x 1 frame (k = 1) for the frame manifolds
        gr = gr.max(units(Grassmannian { k: 1, n }.norm(&x, &x), want));
        st = st.max(units(StiefelManifold { k: 1, n }.norm(&x, &x), want));
    }
    // exact sum of squares, root at the compute tier (before 0.6.4 the default
    // sqrt of the rounded inner product: up to 15 units at Q22.10)
    check("sphere norm", sph, 0);
    check("euclidean norm", euc, 0);
    check("grassmannian norm", gr, 0);
    check("stiefel norm", st, 0);
}
