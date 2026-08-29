//! Certified positive-definiteness verdicts: the permanent gate.
//!
//! `pd_verdict` runs Cholesky in certified interval arithmetic. The verdicts
//! it may return are proofs about the stored matrix, so the gate checks:
//! - matrices that are positive definite with exactly representable factors
//!   are PROVEN so (identity, diagonals, dyadic A^T A + I);
//! - matrices with an exactly zero pivot (a zero column, a duplicated column
//!   in a dyadic matrix, whose factor entries are exact) are PROVEN not
//!   positive definite, at the right pivot;
//! - a negative-definite matrix fails at pivot 0;
//! - a duplicated column in a matrix with inexact factor entries is either
//!   proven not positive definite or Inconclusive, never proven positive;
//! - the scalar `cholesky_decompose` never succeeds where the verdict proves
//!   the opposite is impossible: when the verdict is PositiveDefinite the
//!   scalar factorisation succeeds; and the width of the last pivot on
//!   23- and 50-dimensional dyadic SPD matrices stays within a bound set from
//!   measurement (printed under --nocapture).
//!
//! All profiles.

use g_math::fixed_point::imperative::decompose::cholesky_decompose;
use g_math::fixed_point::imperative::predicates::{pd_verdict, PdVerdict};
use g_math::fixed_point::{FixedMatrix, FixedPoint};

fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }

/// Deterministic generator, integer only.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = self.0;
        (x >> 33) ^ x
    }
    /// A dyadic value k / 16 with k in [-8, 8]: exact on every profile.
    fn dyadic(&mut self) -> FixedPoint {
        let k = (self.next() % 17) as i32 - 8;
        FixedPoint::from_int(k) / FixedPoint::from_int(16)
    }
}

fn matrix(rows: &[&[&str]]) -> FixedMatrix {
    let n = rows.len();
    let mut m = FixedMatrix::new(n, n);
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r.len(), n);
        for (j, s) in r.iter().enumerate() {
            m.set(i, j, fp(s));
        }
    }
    m
}

/// A^T A + I from a random dyadic n x n A: SPD, and every entry exact.
fn dyadic_spd(rng: &mut Rng, n: usize) -> FixedMatrix {
    let mut a = FixedMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            a.set(i, j, rng.dyadic());
        }
    }
    let mut m = FixedMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut s = FixedPoint::ZERO;
            for k in 0..n {
                s = s + a.get(k, i) * a.get(k, j); // products of sixteenths: exact
            }
            if i == j { s = s + FixedPoint::one(); }
            m.set(i, j, s);
        }
    }
    m
}

#[test]
fn proven_positive_definite() {
    assert_eq!(pd_verdict(&FixedMatrix::identity(1)).unwrap(), PdVerdict::PositiveDefinite);
    assert_eq!(pd_verdict(&FixedMatrix::identity(7)).unwrap(), PdVerdict::PositiveDefinite);
    let diag = matrix(&[&["2", "0", "0"], &["0", "0.5", "0"], &["0", "0", "0.001"]]);
    assert_eq!(pd_verdict(&diag).unwrap(), PdVerdict::PositiveDefinite);
    // the classic 3x3 with an exact factor: L = [[2,0,0],[1,1,0],[2,0,1]]
    let spd = matrix(&[&["4", "2", "4"], &["2", "2", "2"], &["4", "2", "5"]]);
    assert_eq!(pd_verdict(&spd).unwrap(), PdVerdict::PositiveDefinite);
    let mut rng = Rng(0x5D5D);
    for n in [2usize, 5, 11] {
        for _ in 0..20 {
            let m = dyadic_spd(&mut rng, n);
            assert_eq!(pd_verdict(&m).unwrap(), PdVerdict::PositiveDefinite, "dyadic SPD n = {n}");
            assert!(cholesky_decompose(&m).is_ok());
        }
    }
}

#[test]
fn proven_not_positive_definite_with_exact_zero_pivot() {
    // zero column: pivot 1 is exactly zero
    let z = matrix(&[&["1", "0", "0"], &["0", "0", "0"], &["0", "0", "1"]]);
    assert_eq!(pd_verdict(&z).unwrap(), PdVerdict::NotPositiveDefinite { pivot: 1 });
    // duplicated column in a dyadic matrix: L = [[2,0,0],[1,1,0],[2,0,_]], pivot 2 exactly 0
    let dup = matrix(&[&["4", "2", "4"], &["2", "2", "2"], &["4", "2", "4"]]);
    assert_eq!(pd_verdict(&dup).unwrap(), PdVerdict::NotPositiveDefinite { pivot: 2 });
    // negative definite: fails at the first pivot
    let neg = matrix(&[&["-1", "0"], &["0", "-1"]]);
    assert_eq!(pd_verdict(&neg).unwrap(), PdVerdict::NotPositiveDefinite { pivot: 0 });
    // indefinite: [[1, 2], [2, 1]] has pivot 1 = 1 - 4 = -3
    let indef = matrix(&[&["1", "2"], &["2", "1"]]);
    assert_eq!(pd_verdict(&indef).unwrap(), PdVerdict::NotPositiveDefinite { pivot: 1 });
    // the scalar path agrees on these: it refuses them
    for m in [&z, &dup, &neg, &indef] {
        assert!(cholesky_decompose(m).is_err());
    }
}

#[test]
fn inexact_duplicate_column_is_never_proven_positive() {
    // columns 0 and 2 equal, entries with no exact binary factor
    let dup = matrix(&[&["0.3", "0.1", "0.3"], &["0.1", "0.7", "0.1"], &["0.3", "0.1", "0.3"]]);
    match pd_verdict(&dup).unwrap() {
        PdVerdict::PositiveDefinite => panic!("a singular matrix was proven positive definite"),
        PdVerdict::NotPositiveDefinite { pivot } => assert_eq!(pivot, 2),
        PdVerdict::Inconclusive { pivot, straddle } => {
            assert_eq!(pivot, 2);
            assert!(straddle.contains_zero());
        }
    }
}

#[test]
fn verdict_and_scalar_factorisation_are_consistent() {
    let mut rng = Rng(0xC0C0);
    let mut proven = 0usize;
    for n in [3usize, 6, 9] {
        for _ in 0..30 {
            let m = dyadic_spd(&mut rng, n);
            let v = pd_verdict(&m).unwrap();
            match v {
                PdVerdict::PositiveDefinite => {
                    proven += 1;
                    assert!(cholesky_decompose(&m).is_ok(), "proven PD but scalar Cholesky failed");
                }
                PdVerdict::NotPositiveDefinite { .. } => panic!("A^T A + I proven not PD"),
                PdVerdict::Inconclusive { .. } => {}
            }
        }
    }
    assert!(proven > 80, "expected the dyadic SPD family to be proven PD, got {proven}");
}

/// Width of the last pivot on realistic dimensions. The interval Cholesky is
/// the first multi-step chain in the library; widths compound through the
/// factor. The bound below is set from measurement with a wide margin: the
/// measured values are printed so the finding can be updated if they move.
#[test]
fn last_pivot_width_stays_bounded() {
    use g_math::fixed_point::Interval;
    let mut rng = Rng(0x1D7);
    for n in [23usize, 50] {
        let m = dyadic_spd(&mut rng, n);
        assert_eq!(pd_verdict(&m).unwrap(), PdVerdict::PositiveDefinite, "n = {n}");
        // recover the last pivot's enclosure by re-running the factorisation
        // through the public interval operations, so the width is observable
        let zero = Interval::point(FixedPoint::ZERO);
        let mut l = vec![vec![zero; n]; n];
        let mut last = zero;
        for i in 0..n {
            let d = Interval::point(m.get(i, i)) - Interval::dot_intervals(&l[i][..i], &l[i][..i]);
            let lii = d.sqrt();
            l[i][i] = lii;
            for j in (i + 1)..n {
                let num = Interval::point(m.get(j, i)) - Interval::dot_intervals(&l[j][..i], &l[i][..i]);
                l[j][i] = num / lii;
            }
            last = d;
        }
        let width = last.width();
        println!("pd_verdict last pivot width, n = {n}: {} (value {})", width, last.lo());
        // bound: the pivot stays positive and the width is far below the value
        assert!(last.is_certainly_positive());
        assert!(width < last.lo(), "last pivot width exceeds its own value at n = {n}");
    }
}
