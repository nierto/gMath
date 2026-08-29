//! Exact geometric predicates: the permanent gate.
//!
//! `orient2d`, `orient3d`, `incircle`, `insphere` return a `Sign` decided by
//! exact integer evaluation of a determinant. The gate checks, on every
//! profile:
//! - hand-checkable configurations with all three signs, including the exact
//!   degenerate cases (collinear, coplanar, cocircular, cospherical), which
//!   are the reason the predicates exist;
//! - sign symmetry under permutation parity (swapping two points flips the
//!   sign, cyclic shifts of an odd count preserve it);
//! - agreement with an exact `i128` evaluation of the same determinant on
//!   random raw coordinates small enough that the i128 oracle cannot
//!   overflow (the oracle is exact integer arithmetic, never a float).
//!
//! The circle predicates are not compiled on the scientific profile.

use g_math::fixed_point::imperative::predicates::{orient2d, orient3d, Sign};
#[cfg(not(table_format = "q256_256"))]
use g_math::fixed_point::imperative::predicates::{incircle, insphere};
use g_math::fixed_point::FixedPoint;

fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }

/// A FixedPoint from an i128 raw value, on every profile's storage type.
#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn raw(v: i128) -> FixedPoint { FixedPoint::from_raw(v as _) }
#[cfg(table_format = "q128_128")]
fn raw(v: i128) -> FixedPoint { FixedPoint::from_raw(g_math::fixed_point::I256::from_i128(v)) }
#[cfg(table_format = "q256_256")]
fn raw(v: i128) -> FixedPoint { FixedPoint::from_raw(g_math::fixed_point::I512::from_i128(v)) }
fn p2(x: &str, y: &str) -> [FixedPoint; 2] { [fp(x), fp(y)] }
fn p3(x: &str, y: &str, z: &str) -> [FixedPoint; 3] { [fp(x), fp(y), fp(z)] }

/// Deterministic generator, integer only.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = self.0;
        (x >> 33) ^ x
    }
    /// A raw coordinate with |raw| < 2^bits, as a FixedPoint and as i128.
    fn coord(&mut self, bits: u32) -> (FixedPoint, i128) {
        let v = ((self.next() as u128) & ((1u128 << bits) - 1)) as i128;
        let v = if self.next() & 1 == 1 { -v } else { v };
        (raw(v), v)
    }
}

fn sign_of(x: i128) -> Sign {
    if x < 0 { Sign::Negative } else if x == 0 { Sign::Zero } else { Sign::Positive }
}

// ----------------------------------------------------------------------------
// orient2d
// ----------------------------------------------------------------------------

#[test]
fn orient2d_signs_and_degeneracy() {
    let a = p2("0", "0");
    let b = p2("1", "0");
    let c = p2("0", "1");
    assert_eq!(orient2d(a, b, c), Sign::Positive, "counterclockwise");
    assert_eq!(orient2d(a, c, b), Sign::Negative, "clockwise");
    // exactly collinear, including non-dyadic coordinates: the stored raws of
    // (t, t) are collinear whatever rounding did to t
    assert_eq!(orient2d(p2("0", "0"), p2("1", "1"), p2("2", "2")), Sign::Zero);
    assert_eq!(orient2d(p2("0.1", "0.1"), p2("0.2", "0.2"), p2("0.3", "0.3")), Sign::Zero);
    assert_eq!(orient2d(p2("-3", "7"), p2("-3", "7"), p2("5", "5")), Sign::Zero, "two coincident points");
    // one ulp off the line is decided
    let ulp = raw(1);
    assert_eq!(orient2d(p2("0", "0"), p2("1", "1"), [fp("2"), fp("2") + ulp]), Sign::Positive);
    assert_eq!(orient2d(p2("0", "0"), p2("1", "1"), [fp("2"), fp("2") - ulp]), Sign::Negative);
    // parity
    assert_eq!(orient2d(b, c, a), Sign::Positive, "cyclic shift preserves");
    assert_eq!(orient2d(c, a, b), Sign::Positive);
    assert_eq!(orient2d(b, a, c), Sign::Negative, "swap flips");
}

#[test]
fn orient2d_matches_exact_i128_oracle() {
    let mut rng = Rng(0x0121);
    let mut seen = [0usize; 3];
    for _ in 0..5_000 {
        let (ax, axi) = rng.coord(24);
        let (ay, ayi) = rng.coord(24);
        let (bx, bxi) = rng.coord(24);
        let (by, byi) = rng.coord(24);
        // c on the line ab one time in eight, to hit the zero case
        let (cx, cy, cxi, cyi) = if rng.next() % 8 == 0 {
            (raw(2 * bxi - axi), raw(2 * byi - ayi), 2 * bxi - axi, 2 * byi - ayi)
        } else {
            let (cx, cxi) = rng.coord(24);
            let (cy, cyi) = rng.coord(24);
            (cx, cy, cxi, cyi)
        };
        let exact = (axi - cxi) * (byi - cyi) - (ayi - cyi) * (bxi - cxi);
        let got = orient2d([ax, ay], [bx, by], [cx, cy]);
        assert_eq!(got, sign_of(exact));
        seen[match got { Sign::Negative => 0, Sign::Zero => 1, Sign::Positive => 2 }] += 1;
    }
    assert!(seen.iter().all(|&n| n > 100), "all three signs must occur: {seen:?}");
}

// ----------------------------------------------------------------------------
// orient3d
// ----------------------------------------------------------------------------

#[test]
fn orient3d_signs_and_degeneracy() {
    let a = p3("0", "0", "0");
    let b = p3("1", "0", "0");
    let c = p3("0", "1", "0");
    let d = p3("0", "0", "1");
    // d above the plane of a b c seen counterclockwise from above: Negative
    assert_eq!(orient3d(a, b, c, d), Sign::Negative);
    assert_eq!(orient3d(a, c, b, d), Sign::Positive);
    // coplanar
    assert_eq!(orient3d(a, b, c, p3("2", "3", "0")), Sign::Zero);
    // all on one line, with dyadic coordinates so the multiples are exact in
    // storage (0.1, 0.2, 0.3 would NOT be: each rounds independently, and the
    // predicate correctly reports the stored points as not collinear)
    assert_eq!(orient3d(p3("0.25", "0.5", "0.75"), p3("0.5", "1", "1.5"), p3("0.75", "1.5", "2.25"), p3("1", "2", "3")), Sign::Zero, "all on one line");
    // parity: swapping two points flips, a 3-cycle of a b c preserves
    assert_eq!(orient3d(b, a, c, d), Sign::Positive);
    assert_eq!(orient3d(b, c, a, d), Sign::Negative);
    assert_eq!(orient3d(c, a, b, d), Sign::Negative);
}

#[test]
fn orient3d_matches_exact_i128_oracle() {
    let mut rng = Rng(0x0131);
    let mut seen = [0usize; 3];
    for _ in 0..3_000 {
        let mut p = [[0i128; 3]; 4];
        let mut f = [[FixedPoint::ZERO; 3]; 4];
        for k in 0..4 {
            for i in 0..3 {
                let (v, vi) = rng.coord(20);
                f[k][i] = v;
                p[k][i] = vi;
            }
        }
        if rng.next() % 8 == 0 {
            // make d a point of the plane: d = a + (b - a) + (c - a)
            for i in 0..3 {
                p[3][i] = p[1][i] + p[2][i] - p[0][i];
                f[3][i] = raw(p[3][i]);
            }
        }
        let r = |k: usize| [p[k][0] - p[3][0], p[k][1] - p[3][1], p[k][2] - p[3][2]];
        let (r0, r1, r2) = (r(0), r(1), r(2));
        let exact = r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0]) + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]);
        let got = orient3d(f[0], f[1], f[2], f[3]);
        assert_eq!(got, sign_of(exact));
        seen[match got { Sign::Negative => 0, Sign::Zero => 1, Sign::Positive => 2 }] += 1;
    }
    assert!(seen.iter().all(|&n| n > 50), "all three signs must occur: {seen:?}");
}

// ----------------------------------------------------------------------------
// incircle / insphere
// ----------------------------------------------------------------------------

#[cfg(not(table_format = "q256_256"))]
#[test]
fn incircle_signs_and_degeneracy() {
    // circumcircle of (0,0),(1,0),(1,1): centre (0.5,0.5), passes through (0,1)
    let a = p2("0", "0");
    let b = p2("1", "0");
    let c = p2("1", "1");
    assert_eq!(orient2d(a, b, c), Sign::Positive);
    assert_eq!(incircle(a, b, c, p2("0.5", "0.5")), Sign::Positive, "inside");
    assert_eq!(incircle(a, b, c, p2("2", "2")), Sign::Negative, "outside");
    assert_eq!(incircle(a, b, c, p2("0", "1")), Sign::Zero, "exactly on the circle");
    assert_eq!(incircle(a, b, c, p2("0.5", "-0.5")), Sign::Negative, "outside, below");
    // orientation flips the sign
    assert_eq!(incircle(a, c, b, p2("0.5", "0.5")), Sign::Negative);
    // one ulp inside / outside the circle at (0, 1) is decided
    let ulp = raw(1);
    assert_eq!(incircle(a, b, c, [fp("0") + ulp, fp("1")]), Sign::Positive);
    assert_eq!(incircle(a, b, c, [fp("0") - ulp, fp("1")]), Sign::Negative);
}

#[cfg(not(table_format = "q256_256"))]
#[test]
fn incircle_matches_exact_i128_oracle() {
    let mut rng = Rng(0x01C1);
    let mut seen = [0usize; 3];
    for _ in 0..3_000 {
        let mut p = [[0i128; 2]; 4];
        let mut f = [[FixedPoint::ZERO; 2]; 4];
        for k in 0..4 {
            for i in 0..2 {
                let (v, vi) = rng.coord(20);
                f[k][i] = v;
                p[k][i] = vi;
            }
        }
        let d = p[3];
        let r = |k: usize| [p[k][0] - d[0], p[k][1] - d[1]];
        let (ad, bd, cd) = (r(0), r(1), r(2));
        let lift = |v: [i128; 2]| v[0] * v[0] + v[1] * v[1];
        let det2 = |u: [i128; 2], v: [i128; 2]| u[0] * v[1] - u[1] * v[0];
        let exact = lift(ad) * det2(bd, cd) - lift(bd) * det2(ad, cd) + lift(cd) * det2(ad, bd);
        let got = incircle(f[0], f[1], f[2], f[3]);
        assert_eq!(got, sign_of(exact));
        seen[match got { Sign::Negative => 0, Sign::Zero => 1, Sign::Positive => 2 }] += 1;
    }
    assert!(seen[0] > 100 && seen[2] > 100, "both strict signs must occur: {seen:?}");
}

#[cfg(not(table_format = "q256_256"))]
#[test]
fn insphere_signs_and_degeneracy() {
    // circumsphere of (0,0,0),(1,0,0),(0,1,0),(0,0,1): centre (0.5,0.5,0.5), r^2 = 0.75
    let a = p3("0", "0", "0");
    let b = p3("0", "1", "0"); // ordered so that orient3d(a, b, c, d) is Positive
    let c = p3("1", "0", "0");
    let d = p3("0", "0", "1");
    assert_eq!(orient3d(a, b, c, d), Sign::Positive);
    assert_eq!(insphere(a, b, c, d, p3("0.25", "0.25", "0.25")), Sign::Positive, "inside");
    assert_eq!(insphere(a, b, c, d, p3("2", "2", "2")), Sign::Negative, "outside");
    assert_eq!(insphere(a, b, c, d, p3("1", "1", "0")), Sign::Zero, "on the sphere");
    assert_eq!(insphere(a, b, c, d, p3("1", "1", "1")), Sign::Zero, "on the sphere");
    // orientation flips the sign
    assert_eq!(insphere(a, c, b, d, p3("0.25", "0.25", "0.25")), Sign::Negative);
    // one ulp inside / outside at (1, 1, 0) is decided
    let ulp = raw(1);
    assert_eq!(insphere(a, b, c, d, [fp("1") - ulp, fp("1"), fp("0")]), Sign::Positive);
    assert_eq!(insphere(a, b, c, d, [fp("1") + ulp, fp("1"), fp("0")]), Sign::Negative);
}

#[cfg(not(table_format = "q256_256"))]
#[test]
fn insphere_matches_exact_i128_oracle() {
    let mut rng = Rng(0x0151);
    let mut seen = [0usize; 3];
    for _ in 0..2_000 {
        let mut p = [[0i128; 3]; 5];
        let mut f = [[FixedPoint::ZERO; 3]; 5];
        for k in 0..5 {
            for i in 0..3 {
                let (v, vi) = rng.coord(20); // degree 5 at 2^20: 2^100, fits i128
                f[k][i] = v;
                p[k][i] = vi;
            }
        }
        let e = p[4];
        let r = |k: usize| [p[k][0] - e[0], p[k][1] - e[1], p[k][2] - e[2]];
        let (ae, be, ce, de) = (r(0), r(1), r(2), r(3));
        let lift = |v: [i128; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let det3 = |r0: [i128; 3], r1: [i128; 3], r2: [i128; 3]| {
            r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0]) + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0])
        };
        let exact = -lift(ae) * det3(be, ce, de) + lift(be) * det3(ae, ce, de) - lift(ce) * det3(ae, be, de) + lift(de) * det3(ae, be, ce);
        let got = insphere(f[0], f[1], f[2], f[3], f[4]);
        assert_eq!(got, sign_of(exact));
        seen[match got { Sign::Negative => 0, Sign::Zero => 1, Sign::Positive => 2 }] += 1;
    }
    assert!(seen[0] > 100 && seen[2] > 100, "both strict signs must occur: {seen:?}");
}
