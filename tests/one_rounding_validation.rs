//! The 0.6.4 one-rounding rework against mpmath: operations whose
//! intermediates used to be rounded to storage (sums of squares before a
//! root, the Minkowski product's two halves, QR's ||x||^2 and v^T v, tensor
//! means, Pade coefficients) now form them at the compute tier and round once.
//!
//! References: `tests/data/one_rounding_refs.rs` from
//! `scripts/generate_one_rounding_refs.py` (exact dyadic sums or mpmath at 120
//! digits). Inputs are dyadic with <= 8 fraction bits and |x| <= 8, exact on
//! every gated build; each reference is parsed at the build's split, so it is
//! the correctly rounded result, and every error below is in storage units.

use g_math::fixed_point::imperative::decompose::{cholesky_decompose, eigen_symmetric, lu_decompose, qr_decompose, schur_decompose, svd_decompose};
use g_math::fixed_point::imperative::derived::frobenius_norm;
use g_math::fixed_point::imperative::fused::{rms_norm, rms_norm_in_place};
use g_math::fixed_point::imperative::lie_group::{LieGroup, SO3};
use g_math::fixed_point::imperative::manifold::{HyperbolicSpace, Manifold};
use g_math::fixed_point::imperative::matrix_functions::{matrix_exp, matrix_log, matrix_sqrt};
use g_math::fixed_point::imperative::tensor::{symmetrize, Tensor};
use g_math::fixed_point::{FixedMatrix, FixedPoint, FixedVector};

#[allow(dead_code)]
mod refs {
    include!("data/one_rounding_refs.rs");
}

fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }
fn vecs(v: &[&str]) -> FixedVector { FixedVector::from_slice(&v.iter().map(|s| fp(s)).collect::<Vec<_>>()) }
fn mat(n: usize, v: &[&str]) -> FixedMatrix { FixedMatrix::from_slice(n, n, &v.iter().map(|s| fp(s)).collect::<Vec<_>>()) }

/// |got - want| in storage units (one unit = 2^-FRAC_BITS), saturating at
/// 2^30 (a count beyond the storage range, or beyond i32, reads as 2^30).
fn units(got: FixedPoint, want: FixedPoint) -> i32 {
    let half = fp("0.5");
    let mut unit = FixedPoint::one();
    for _ in 0..g_math::fixed_point::frac_config::FRAC_BITS { unit = unit * half; }
    let cap = FixedPoint::try_from_int(1 << 30).ok();
    match got.try_sub(want).and_then(|d| d.abs().try_div(unit)) {
        Ok(q) if cap.map_or(true, |c| q < c) => q.to_int(),
        _ => 1 << 30,
    }
}

/// Worst error over a table and the bound it must meet; the measured value
/// is printed so the finding can be updated if it moves.
fn check(name: &str, worst: i32, bound: i32) {
    println!("{name}: worst {worst} units (bound {bound})");
    assert!(worst <= bound, "{name}: {worst} units > {bound}");
}

#[test]
fn dot_is_the_nearest_of_the_exact_sum() {
    let worst = refs::DOT.iter().map(|(a, b, r)| units(vecs(a).dot(&vecs(b)), fp(r))).max().unwrap();
    // exact products, one rounding: the correctly rounded sum
    check("dot", worst, 0);
}

#[test]
fn norms_take_the_root_at_the_compute_tier() {
    let length = refs::LENGTH.iter().map(|(v, r)| units(vecs(v).length(), fp(r))).max().unwrap();
    let distance = refs::DISTANCE.iter().map(|(a, b, r)| units(vecs(a).metric_distance_safe(&vecs(b)), fp(r))).max().unwrap();
    let frob = refs::FROBENIUS.iter().map(|(a, r)| units(frobenius_norm(&mat(3, a)), fp(r))).max().unwrap();
    let hyp = HyperbolicSpace { dim: 2 };
    let base = FixedVector::from_slice(&[FixedPoint::one(), FixedPoint::ZERO, FixedPoint::ZERO]);
    let mink = refs::MINKOWSKI.iter().map(|(v, r)| units(hyp.norm(&base, &vecs(v)), fp(r))).max().unwrap();
    // exact sum of squares, compute-tier root, one rounding
    check("length", length, 1);
    check("metric distance", distance, 1);
    check("frobenius", frob, 1);
    check("minkowski norm", mink, 1);
}

#[test]
fn qr_factors_against_mpmath() {
    let (mut worst_r, mut worst_q) = (0, 0);
    for (a, r, q) in refs::QR_R {
        let qr = qr_decompose(&mat(3, a)).expect("qr");
        for i in 0..3 {
            for j in 0..3 {
                if j >= i { worst_r = worst_r.max(units(qr.r.get(i, j), fp(r[3 * i + j]))); }
                worst_q = worst_q.max(units(qr.q.get(i, j), fp(q[3 * i + j])));
            }
        }
    }
    // R and Q at the compute tier through every reflection, rounded once
    // (0.6.3: R up to 23 units, rounded after each reflection)
    check("qr R", worst_r, 1);
    check("qr Q", worst_q, 1);
}

#[test]
fn symmetrize_is_one_rounding_of_the_mean() {
    let mut worst = 0;
    for (t, s) in refs::SYMMETRIZE3 {
        let data: Vec<FixedPoint> = t.iter().map(|x| fp(x)).collect();
        let tensor = Tensor::from_data(&[2, 2, 2], &data);
        let sym = symmetrize(&tensor, &[0, 1, 2]);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    worst = worst.max(units(sym.get(&[i, j, k]), fp(s[4 * i + 2 * j + k])));
                }
            }
        }
    }
    // exact sum of six terms, divided once
    check("symmetrize", worst, 1);
}

#[test]
fn matrix_exp_against_mpmath() {
    let mut worst = 0;
    for (a, e) in refs::EXPM {
        let got = matrix_exp(&mat(2, a)).expect("expm");
        for i in 0..2 {
            for j in 0..2 {
                worst = worst.max(units(got.get(i, j), fp(e[2 * i + j])));
            }
        }
    }
    // Pade [6/6] scaled to the precision (2^-(F + 6) truncation), at the
    // compute tier (Q64.64 on realtime): measured 0 on every build
    check("matrix_exp", worst, 0);
}

#[test]
fn matrix_log_against_mpmath() {
    let mut worst = 0;
    for (a, l) in refs::LOGM {
        let got = matrix_log(&mat(2, a)).expect("logm");
        for i in 0..2 {
            for j in 0..2 {
                worst = worst.max(units(got.get(i, j), fp(l[2 * i + j])));
            }
        }
    }
    // square roots to 2^-m, 22 terms: measured 0 on every build
    check("matrix_log", worst, 0);
}

#[test]
fn matrix_sqrt_against_mpmath() {
    let mut worst = 0;
    for (a, r) in refs::SQRTM {
        let got = matrix_sqrt(&mat(2, a)).expect("sqrtm");
        for i in 0..2 {
            for j in 0..2 {
                worst = worst.max(units(got.get(i, j), fp(r[2 * i + j])));
            }
        }
    }
    // Denman-Beavers to 2^-(F + 8) relative, quadratic convergence: measured 0
    check("matrix_sqrt", worst, 0);
}

#[test]
fn so3_exp_against_mpmath() {
    let mut worst = 0;
    for (w, e) in refs::SO3_EXP {
        let got = SO3.lie_exp(&vecs(w)).expect("so3 exp");
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max(units(got.get(i, j), fp(e[3 * i + j])));
            }
        }
    }
    // measured 1 on every build
    check("so3 exp", worst, 1);
}

fn matn(n: usize, v: &[&str]) -> FixedMatrix { FixedMatrix::from_slice(n, n, &v.iter().map(|s| fp(s)).collect::<Vec<_>>()) }

fn worst_matrix(got: &FixedMatrix, want: &[&str]) -> i32 {
    let n = got.cols();
    (0..got.rows() * n).map(|k| units(got.get(k / n, k % n), fp(want[k]))).max().unwrap()
}

fn worst_vector(got: &FixedVector, want: &[&str]) -> i32 {
    (0..got.len()).map(|k| units(got[k], fp(want[k]))).max().unwrap()
}

#[test]
fn lu_against_exact_rationals() {
    let (mut wl, mut wu, mut wx, mut winv, mut wdet) = (0, 0, 0, 0, 0);
    for (n, a, b, l, u, x, inv, det) in refs::LU {
        let lu = lu_decompose(&matn(*n, a)).expect("lu");
        wl = wl.max(worst_matrix(&lu.l, l));
        wu = wu.max(worst_matrix(&lu.u, u));
        wx = wx.max(worst_vector(&lu.solve(&vecs(b)).expect("solve"), x));
        winv = winv.max(worst_matrix(&lu.inverse().expect("inverse"), inv));
        wdet = wdet.max(units(lu.determinant(), fp(det)));
    }
    check("lu L", wl, 1);
    check("lu U", wu, 1);
    check("lu solve", wx, 1);
    check("lu inverse", winv, 1);
    check("lu determinant", wdet, 1);
}

#[test]
fn cholesky_against_mpmath() {
    let (mut wl, mut wx, mut wdet) = (0, 0, 0);
    for (n, a, b, l, x, det) in refs::CHOLESKY {
        let ch = cholesky_decompose(&matn(*n, a)).expect("cholesky");
        wl = wl.max(worst_matrix(&ch.l, l));
        wx = wx.max(worst_vector(&ch.solve(&vecs(b)).expect("solve"), x));
        wdet = wdet.max(units(ch.determinant(), fp(det)));
    }
    check("cholesky L", wl, 1);
    check("cholesky solve", wx, 1);
    check("cholesky determinant", wdet, 1);
}

#[test]
fn qr_solve_against_exact_rationals() {
    let mut worst = 0;
    for (n, a, b, x) in refs::QR_SOLVE {
        let qr = qr_decompose(&matn(*n, a)).expect("qr");
        worst = worst.max(worst_vector(&qr.solve(&vecs(b)).expect("solve"), x));
    }
    check("qr solve", worst, 1);
}

/// Column `k` of `m` with its sign fixed so its largest-magnitude entry is
/// positive (the references' convention; the first such entry on a tie).
fn signed_column(m: &FixedMatrix, k: usize) -> (Vec<FixedPoint>, bool) {
    let col: Vec<FixedPoint> = (0..m.rows()).map(|r| m.get(r, k)).collect();
    let mut at = 0;
    for r in 1..col.len() {
        if col[r].abs() > col[at].abs() { at = r; }
    }
    let flip = col[at] < FixedPoint::ZERO;
    (if flip { col.iter().map(|x| -*x).collect() } else { col }, flip)
}

#[test]
fn eigen_symmetric_against_mpmath() {
    let (mut wv, mut wq) = (0, 0);
    for (n, a, vals, vecs_ref) in refs::EIGEN_SYM {
        let e = eigen_symmetric(&matn(*n, a)).expect("eigen");
        wv = wv.max(worst_vector(&e.values, vals));
        for k in 0..*n {
            let (col, _) = signed_column(&e.vectors, k);
            for r in 0..*n { wq = wq.max(units(col[r], fp(vecs_ref[r * n + k]))); }
        }
    }
    check("eigen values", wv, 1);
    check("eigen vectors", wq, 1);
}

#[test]
fn svd_against_mpmath() {
    let (mut ws, mut wu, mut wv) = (0, 0, 0);
    for (m, n, a, sig, u_ref, v_ref) in refs::SVD {
        let a = FixedMatrix::from_slice(*m, *n, &a.iter().map(|s| fp(s)).collect::<Vec<_>>());
        let d = svd_decompose(&a).expect("svd");
        ws = ws.max(worst_vector(&d.sigma, sig));
        let v = d.vt.transpose();
        for k in 0..*n {
            let (vcol, flip) = signed_column(&v, k);
            for r in 0..*n { wv = wv.max(units(vcol[r], fp(v_ref[r * n + k]))); }
            for r in 0..*m {
                let x = d.u.get(r, k);
                wu = wu.max(units(if flip { -x } else { x }, fp(u_ref[r * n + k])));
            }
        }
    }
    check("svd sigma", ws, 1);
    check("svd U", wu, 1);
    check("svd V", wv, 1);
}

#[test]
fn schur_eigenvalues_against_exact() {
    let mut worst = 0;
    for (n, a, eig) in refs::SCHUR {
        let d = schur_decompose(&matn(*n, a)).expect("schur");
        let mut diag: Vec<FixedPoint> = (0..*n).map(|i| d.t.get(i, i)).collect();
        diag.sort();
        for i in 0..*n { worst = worst.max(units(diag[i], fp(eig[i]))); }
    }
    check("schur eigenvalues", worst, 1);
}

/// A matrix-function case at this build: its input and reference parsed at
/// the build's split, or `None` when either leaves the storage range.
fn fitting_case(n: usize, a: &[&str], r: &[&str]) -> Option<(FixedMatrix, Vec<FixedPoint>)> {
    let a: Option<Vec<FixedPoint>> = a.iter().map(|s| FixedPoint::try_from_str(s).ok()).collect();
    let r: Option<Vec<FixedPoint>> = r.iter().map(|s| FixedPoint::try_from_str(s).ok()).collect();
    Some((FixedMatrix::from_slice(n, n, &a?), r?))
}

fn wide_table(name: &str, table: &[(usize, &[&str], &[&str])], f: fn(&FixedMatrix) -> Result<FixedMatrix, g_math::fixed_point::OverflowDetected>) -> (i32, usize) {
    let (mut worst, mut ran) = (0, 0);
    for (n, a, r) in table {
        let Some((a, r)) = fitting_case(*n, a, r) else { continue };
        let got = f(&a).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for k in 0..n * n {
            worst = worst.max(units(got.get(k / n, k % n), r[k]));
        }
        ran += 1;
    }
    println!("{name}: {ran} of {} cases fit this build", table.len());
    (worst, ran)
}

#[test]
fn matrix_functions_beyond_small_norms() {
    let (e, _) = wide_table("expm wide", refs::EXPM_WIDE, matrix_exp);
    let (l, _) = wide_table("logm wide", refs::LOGM_WIDE, matrix_log);
    let (s, _) = wide_table("sqrtm wide", refs::SQRTM_WIDE, matrix_sqrt);
    // 0.6.3 on realtime ran these at the compute tier's 2F bits: expm 8 units
    // and logm 6 at 8 fraction bits (1 at 10 to 16). Now Q64.64 there, and
    // Denman-Beavers stops at 2^-(F + 8) relative: measured 0 on every build.
    check("matrix_exp wide", e, 0);
    check("matrix_log wide", l, 0);
    check("matrix_sqrt wide", s, 0);
}

/// Worst error of `rms_norm` over a table, in units, and the cases that fit
/// this build (inputs and references inside the storage range). The in-place
/// form must give the same values.
fn rms_table(name: &str, table: &[(&[&str], &[&str], i128, &[&str])]) -> (i32, usize) {
    let parse = |v: &[&str]| -> Option<Vec<FixedPoint>> { v.iter().map(|s| FixedPoint::try_from_str(s).ok()).collect() };
    let (mut worst, mut ran) = (0, 0);
    for (x, w, eps, r) in table {
        let (Some(x), Some(w), Some(r)) = (parse(x), parse(w), parse(r)) else { continue };
        let got = rms_norm(&x, &w, *eps).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let mut in_place = x.clone();
        rms_norm_in_place(&mut in_place, &w, *eps).unwrap();
        assert_eq!(in_place, got, "{name}: in-place differs");
        for (g, want) in got.iter().zip(&r) {
            worst = worst.max(units(*g, *want));
        }
        ran += 1;
    }
    println!("{name}: {ran} of {} cases fit this build", table.len());
    (worst, ran)
}

#[test]
fn rms_norm_rounds_each_output_once() {
    let (small, ran) = rms_table("rms_norm", refs::RMS_NORM);
    assert_eq!(ran, refs::RMS_NORM.len(), "every small case fits every gated build");
    let (wide, _) = rms_table("rms_norm wide", refs::RMS_NORM_WIDE);
    // exact sum of squares, reciprocal root at full relative precision, one
    // rounding of the exact triple product: the correctly rounded output
    check("rms_norm", small, 0);
    check("rms_norm wide", wide, 0);
}

#[test]
fn rotate_pairs_rounds_each_output_once() {
    let mut worst = 0;
    for (x0, x1, sin, cos, lo, hi) in refs::ROTATE_PAIRS {
        let mut v = FixedVector::from_slice(&[fp(x0), fp(x1)]);
        v.rotate_pairs(&[fp(sin)], &[fp(cos)], 2);
        worst = worst.max(units(v[0], fp(lo))).max(units(v[1], fp(hi)));
    }
    // exact products, exact sum, one rounding
    check("rotate_pairs", worst, 0);
}
