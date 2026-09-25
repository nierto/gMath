//! The compute-tier rework of the projective module (cross ratios, projective
//! transforms, inverse stereographic projection, real and complex Moebius
//! maps), `FixedVector::cross` and `normalize`, the tensor decompositions
//! (truncated SVD, Tucker and CP reconstruction, CP-ALS) and the SVD
//! pseudoinverse, against exact rationals and mpmath.
//!
//! References: `tests/data/projective_tensor_compute_tier_refs.rs` from
//! `scripts/generate_projective_tensor_compute_tier_refs.py`. Inputs are
//! dyadic with <= 8 fraction bits, exact on every gated build; each reference
//! is parsed at the build's split, so it is the correctly rounded result, and
//! every error below is in storage units. An `Err` where the exact result
//! exists counts as `FAILED` (a large sentinel).

use g_math::fixed_point::imperative::derived::{condition_number_1, pseudoinverse, pseudoinverse_with_threshold};
use g_math::fixed_point::imperative::projective::{
    cross_ratio, cross_ratio_1d, projective_transform, stereo_unproject, Moebius, MoebiusComplex,
};
use g_math::fixed_point::imperative::tensor::Tensor;
use g_math::fixed_point::imperative::tensor_decompose::{cp_decompose, CPDecomposition, TruncatedSVD, TuckerDecomposition};
use g_math::fixed_point::{FixedMatrix, FixedPoint, FixedVector};

#[allow(dead_code)]
mod refs {
    include!("data/projective_tensor_compute_tier_refs.rs");
}

const FAILED: i64 = 1_000_000;

fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }
fn fps(v: &[&str]) -> Vec<FixedPoint> { v.iter().map(|s| fp(s)).collect() }
fn vecs(v: &[&str]) -> FixedVector { FixedVector::from_slice(&fps(v)) }
fn mat(r: usize, c: usize, v: &[&str]) -> FixedMatrix { FixedMatrix::from_slice(r, c, &fps(v)) }

/// Storage units `unit * 2^k` for k = 0, 1, ... while they fit storage.
fn unit_steps() -> Vec<FixedPoint> {
    let half = fp("0.5");
    let mut unit = FixedPoint::one();
    for _ in 0..g_math::fixed_point::frac_config::FRAC_BITS { unit = unit * half; }
    let mut steps = vec![unit];
    while let Ok(next) = steps[steps.len() - 1].try_add(steps[steps.len() - 1]) { steps.push(next); }
    steps
}

/// |got - want| in storage units (one unit = 2^-FRAC_BITS), rounded down.
/// Counted greedily over unit * 2^k, so a count beyond the storage range
/// (127 units at 24 fraction bits) is still exact; saturates at 2^62.
fn units(got: FixedPoint, want: FixedPoint) -> i64 {
    let mut d = (got - want).abs();
    let mut total: i64 = 0;
    for (k, s) in unit_steps().iter().enumerate().rev() {
        if d >= *s {
            d = d - *s;
            total = total.saturating_add(if k >= 62 { i64::MAX } else { 1i64 << k });
        }
    }
    total
}

/// Bit length of the error in units: 0 for exact, k + 1 for an error in
/// [2^k, 2^(k+1)) units. For errors too large for an `i64` count.
fn unit_bits(got: FixedPoint, want: FixedPoint) -> i64 {
    let d = (got - want).abs();
    unit_steps().iter().rposition(|s| d >= *s).map_or(0, |k| k as i64 + 1)
}

fn units_or_failed<E>(got: Result<FixedPoint, E>, want: &str) -> i64 {
    match got { Ok(g) => units(g, fp(want)), Err(_) => FAILED }
}

/// Worst error over a table and the bound it must meet; the measured value
/// is printed so the finding can be updated if it moves.
fn check(name: &str, worst: i64, bound: i64) {
    println!("{name}: worst {worst} units (bound {bound})");
    assert!(worst <= bound, "{name}: {worst} units > {bound}");
}

fn worst_tensor(got: &Tensor, want: &[&str]) -> i64 {
    got.data().iter().zip(want).map(|(g, w)| units(*g, fp(w))).max().unwrap()
}

#[test]
fn cross_ratio_1d_is_one_rounding_of_the_exact_ratio() {
    let worst = refs::CROSS_RATIO_1D.iter()
        .map(|(a, b, c, d, r)| units_or_failed(cross_ratio_1d(fp(a), fp(b), fp(c), fp(d)), r))
        .max().unwrap();
    // exact products of exact differences, one division rounded once
    check("cross_ratio_1d", worst, 0);
}

#[test]
fn cross_ratio_nd_is_one_rounding_of_the_exact_ratio() {
    let worst = refs::CROSS_RATIO_ND.iter()
        .map(|(a, b, c, d, r)| units_or_failed(cross_ratio(&vecs(a), &vecs(b), &vecs(c), &vecs(d)), r))
        .max().unwrap();
    // exact projections (the |b - a| scale cancels), one division rounded once
    check("cross_ratio", worst, 0);
}

#[test]
fn projective_transform_divides_exact_rows_once() {
    let mut worst = 0;
    for (h, p, x) in refs::PROJECTIVE {
        match projective_transform(&mat(3, 3, h), &vecs(p)) {
            Ok(got) => for i in 0..2 { worst = worst.max(units(got[i], fp(x[i]))); },
            Err(_) => worst = FAILED,
        }
    }
    // exact row sums, each divided by the exact w and rounded once
    check("projective_transform", worst, 0);
}

#[test]
fn stereo_unproject_divides_the_exact_square_once() {
    let mut worst = 0;
    for (x, p) in refs::STEREO_UNPROJECT {
        let got = stereo_unproject(&vecs(x));
        for i in 0..p.len() { worst = worst.max(units(got[i], fp(p[i]))); }
    }
    check("stereo_unproject", worst, 0);
}

#[test]
fn moebius_real_against_exact_rationals() {
    let apply = refs::MOEBIUS_APPLY.iter()
        .map(|(v, r)| {
            let m = Moebius::new(fp(v[0]), fp(v[1]), fp(v[2]), fp(v[3]));
            units_or_failed(m.apply(fp(v[4])), r)
        })
        .max().unwrap();
    let (mut comp, mut det) = (0, 0);
    for (m1, m2, c, d) in refs::MOEBIUS_COMPOSE {
        let a = Moebius::new(fp(m1[0]), fp(m1[1]), fp(m1[2]), fp(m1[3]));
        let b = Moebius::new(fp(m2[0]), fp(m2[1]), fp(m2[2]), fp(m2[3]));
        let got = a.compose(&b);
        for (g, w) in [got.a, got.b, got.c, got.d].iter().zip(c.iter()) { comp = comp.max(units(*g, fp(w))); }
        det = det.max(units(a.determinant(), fp(d)));
    }
    check("moebius apply", apply, 0);
    check("moebius compose", comp, 0);
    check("moebius determinant", det, 0);
}

#[test]
fn moebius_complex_against_exact_rationals() {
    let pair = |v: &[&str], i: usize| (fp(v[2 * i]), fp(v[2 * i + 1]));
    let mut apply = 0;
    for (v, r) in refs::MOEBIUS_COMPLEX_APPLY {
        let m = MoebiusComplex::new(pair(v, 0), pair(v, 1), pair(v, 2), pair(v, 3));
        match m.apply(pair(v, 4)) {
            Ok((re, im)) => apply = apply.max(units(re, fp(r[0]))).max(units(im, fp(r[1]))),
            Err(_) => apply = FAILED,
        }
    }
    let mut comp = 0;
    for (v1, v2, c) in refs::MOEBIUS_COMPLEX_COMPOSE {
        let a = MoebiusComplex::new(pair(v1, 0), pair(v1, 1), pair(v1, 2), pair(v1, 3));
        let b = MoebiusComplex::new(pair(v2, 0), pair(v2, 1), pair(v2, 2), pair(v2, 3));
        let got = a.compose(&b);
        let flat = [got.a.0, got.a.1, got.b.0, got.b.1, got.c.0, got.c.1, got.d.0, got.d.1];
        for (g, w) in flat.iter().zip(c.iter()) { comp = comp.max(units(*g, fp(w))); }
    }
    // numerator and denominator exact, |d|^2 exact, one division each
    check("moebius complex apply", apply, 0);
    // four exact products per entry, one rounding
    check("moebius complex compose", comp, 0);
}

#[test]
fn cross_product_is_one_rounding_of_the_exact_difference() {
    let mut worst = 0;
    for (u, v, w) in refs::CROSS {
        let got = vecs(u).cross(&vecs(v));
        for i in 0..3 { worst = worst.max(units(got[i], fp(w[i]))); }
    }
    check("cross", worst, 0);
}

#[test]
fn normalize_divides_by_the_compute_tier_length() {
    let (mut worst, mut worst_copy) = (0, 0);
    for (v, n) in refs::NORMALIZE {
        let mut got = vecs(v);
        got.normalize();
        for i in 0..n.len() { worst = worst.max(units(got[i], fp(n[i]))); }
        let copy = vecs(v).normalized();
        for i in 0..n.len() { worst_copy = worst_copy.max(units(copy[i], fp(n[i]))); }
    }
    // exact sum of squares, root at the compute tier, one division each
    check("normalize", worst, 0);
    check("normalized", worst_copy, 0);
}

#[test]
fn truncated_svd_reconstruct_is_one_rounding_per_entry() {
    let mut worst = 0;
    for (m, k, n, u, s, vt, r) in refs::TSVD_RECONSTRUCT {
        let t = TruncatedSVD { u: mat(*m, *k, u), sigma: vecs(s), vt: mat(*k, *n, vt) };
        let got = t.reconstruct();
        for i in 0..*m { for j in 0..*n { worst = worst.max(units(got.get(i, j), fp(r[i * n + j]))); } }
    }
    // exact triple products, exact sum, one rounding
    check("truncated svd reconstruct", worst, 0);
}

#[test]
fn tucker_reconstruct_keeps_the_core_at_the_compute_tier() {
    let mut worst = 0;
    for (g, f0, f1, f2, t) in refs::TUCKER_RECONSTRUCT {
        let d = TuckerDecomposition {
            core: Tensor::from_data(&[2, 2, 2], &fps(g)),
            factors: vec![mat(3, 2, f0), mat(3, 2, f1), mat(3, 2, f2)],
        };
        worst = worst.max(worst_tensor(&d.reconstruct(), t));
    }
    // each mode product an exact dot rounded at the compute tier, one
    // rounding to storage after the last mode
    check("tucker reconstruct", worst, 0);
}

#[test]
fn cp_reconstruct_sums_at_the_compute_tier() {
    let mut worst = 0;
    for (w, f0, f1, f2, t) in refs::CP_RECONSTRUCT {
        let d = CPDecomposition { weights: vecs(w), factors: vec![mat(3, 2, f0), mat(3, 2, f1), mat(3, 2, f2)] };
        worst = worst.max(worst_tensor(&d.reconstruct(&[3, 3, 3]), t));
    }
    check("cp reconstruct", worst, 0);
}

/// CP-ALS on exactly rank-1 tensors: in exact arithmetic one sweep from the
/// SVD start returns the weight |w| |a| |b| |c| and the directions a/|a| ...
/// whatever the start (it cancels), so these references are deterministic.
/// The factor signs are not unique; directions are compared in magnitude.
#[test]
fn cp_als_recovers_rank_one_tensors() {
    let (mut weight, mut dirs, mut recon) = (0, 0, 0);
    for (t, w, a, b, c) in refs::CP_RANK1 {
        let tensor = Tensor::from_data(&[3, 3, 3], &fps(t));
        match cp_decompose(&tensor, 1, 3, FixedPoint::ZERO) {
            Ok(cp) => {
                weight = weight.max(units(cp.weights[0], fp(w)));
                for (f, want) in cp.factors.iter().zip([a, b, c]) {
                    for i in 0..3 { dirs = dirs.max(units(f.get(i, 0).abs(), fp(want[i]))); }
                }
                recon = recon.max(worst_tensor(&cp.reconstruct(&[3, 3, 3]), t));
            }
            Err(_) => { weight = FAILED; dirs = FAILED; recon = FAILED; }
        }
    }
    check("cp-als rank-1 weight", weight, 0);
    check("cp-als rank-1 directions", dirs, 0);
    // rebuilt from the storage outputs: the weight and the three unit
    // factors are each rounded once, and their product (entries up to 2)
    // carries those four roundings. Measured 1 or 2 on every build.
    check("cp-als rank-1 reconstruct", recon, 2);
}

/// A = U diag(s) V^T with signed Hadamard / 2 factors (dyadic, orthogonal);
/// A+ = V diag(1/s) U^T. The pseudoinverse's own arithmetic is one exact sum
/// of (v / sigma) u per entry, rounded once; what remains on these cases is
/// the SVD's error in U, sigma, V (its precision contract is relative
/// 2^-(2F/3), 2^-10 on realtime), so the bound is that contract in units.
/// The last four cases are signed permutations of a diagonal, whose SVD
/// factors are exact: there the pseudoinverse alone is measured.
#[test]
fn pseudoinverse_against_exact_rationals() {
    let f = g_math::fixed_point::frac_config::FRAC_BITS as i64;
    let exact_cases = 4;
    let (mut worst, mut worst_t, mut worst_exact) = (0, 0, 0);
    for (case, (m, n, a, p)) in refs::PINV.iter().enumerate() {
        let am = mat(*m, *n, a);
        let got = pseudoinverse(&am).expect("pseudoinverse");
        let got_t = pseudoinverse_with_threshold(&am, fp("0.25")).expect("pseudoinverse_with_threshold");
        for i in 0..*n {
            for j in 0..*m {
                let (g, gt, want) = (got.get(i, j), got_t.get(i, j), fp(p[i * m + j]));
                if case >= refs::PINV.len() - exact_cases {
                    worst_exact = worst_exact.max(units(g, want)).max(units(gt, want));
                } else {
                    worst = worst.max(unit_bits(g, want));
                    worst_t = worst_t.max(unit_bits(gt, want));
                }
            }
        }
    }
    // the SVD contract as an error bit length in units: realtime (F <= 30)
    // 2^(F - 10) units, at least 8; wider profiles 2^(F/3 + 1)
    let svd_bits = if f <= 30 { (f - 10).max(3) + 1 } else { f - 2 * f / 3 + 2 };
    check("pseudoinverse (svd-limited, error bits)", worst, svd_bits);
    check("pseudoinverse_with_threshold (svd-limited, error bits)", worst_t, svd_bits);
    // exact factors: one rounding per entry
    check("pseudoinverse (exact svd factors)", worst_exact, 0);
}

#[test]
fn condition_number_1_against_exact_rationals() {
    let worst = refs::COND1.iter()
        .map(|(n, a, c)| units_or_failed(condition_number_1(&mat(*n, *n, a)), c))
        .max().unwrap();
    check("condition_number_1", worst, 0);
}
