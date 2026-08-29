//! Certified and exact predicates: verdicts that are proven, not estimated.
//!
//! A predicate here returns a trichotomy or a verdict, never a bare `bool`
//! with a tolerance behind it. Two families:
//!
//! - **Exact** ([`orient2d`], [`orient3d`], [`incircle`], [`insphere`]):
//!   each is the sign of a determinant of fixed degree in the input
//!   coordinates, evaluated in exact integer arithmetic on an accumulator
//!   wide enough for the worst case. The accumulator is chosen per profile
//!   from the storage width (orient2d `2W+2` bits, orient3d `3W+3`, incircle
//!   `4W+5`, insphere `5W+6`), never hand-tuned per predicate, and every
//!   multiply asserts its bit-length budget before it runs, so a violation is
//!   loud rather than wrong. The circle predicates are not compiled on the
//!   scientific profile: at `W = 512` they need 2053 and 2566 bits against the
//!   2048 available. If a consumer appears, the route is the arbitrary-precision
//!   type behind the `infinite-precision` gate, not a wider hand-rolled type.
//!   Returns are a trichotomy, [`Sign`], never a `bool`: the exact-zero case
//!   (collinear, coplanar, cocircular, cospherical) is the whole reason exact
//!   predicates exist.
//!
//! - **Certified** ([`pd_verdict`]): decides positive
//!   definiteness by running the Cholesky factorisation in certified interval
//!   arithmetic. If every pivot's interval is strictly positive, the true
//!   pivots of the stored matrix are positive, and the matrix is PROVEN
//!   positive definite; no arbitrary precision is involved. If a pivot's
//!   interval lies at or below zero, the matrix is proven not positive
//!   definite. If a pivot's interval straddles zero the verdict is
//!   [`PdVerdict::Inconclusive`] and the caller decides, with the straddling
//!   interval in hand; regularisation then becomes a documented decision
//!   taken after a diagnosis rather than a blind nudge.
//!
//! The certificate covers the STORED matrix: the verdict is about the
//! `FixedMatrix` passed in, not about whatever statistical quantity it
//! approximates. Only the lower triangle is read, as with
//! `cholesky_decompose`; symmetry is the caller's contract.
//!
//! Design and measurements: `docs/design/CERTIFIED_INTERVALS.md`.

use super::{FixedMatrix, FixedPoint, Interval};
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;

#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
use crate::fixed_point::I256;
#[cfg(any(table_format = "q32_32", table_format = "q64_64"))]
use crate::fixed_point::I512;
#[cfg(any(table_format = "q64_64", table_format = "q128_128"))]
use crate::fixed_point::I1024;
#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
use crate::fixed_point::I2048;

// ============================================================================
// Sign
// ============================================================================

/// The sign of an exactly evaluated determinant.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Sign {
    /// Strictly negative.
    Negative,
    /// Exactly zero: the degenerate configuration, decided exactly.
    Zero,
    /// Strictly positive.
    Positive,
}

impl Sign {
    /// The sign of the negated quantity.
    #[inline]
    pub fn flip(self) -> Sign {
        match self {
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
            Sign::Positive => Sign::Negative,
        }
    }
}

// ============================================================================
// Exact accumulator arithmetic
// ============================================================================

/// Exact signed integer arithmetic on a fixed-width accumulator.
///
/// Products are formed on magnitudes and the sign reapplied, so the wide
/// types' `Mul` implementations are only ever entered with non-negative
/// operands (a schoolbook or a truncated unsigned product, both exact there).
/// Each product first asserts that the operands' bit lengths fit the type,
/// which is the width budget made loud.
trait Wide:
    Copy + PartialEq + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self> + std::ops::Neg<Output = Self>
{
    const BITS: u32;
    fn zero() -> Self;
    fn is_negative(self) -> bool;
    /// Significant bits of a NON-NEGATIVE value.
    fn bit_length(self) -> u32;

    #[inline]
    fn sign(self) -> Sign {
        if self.is_negative() {
            Sign::Negative
        } else if self == Self::zero() {
            Sign::Zero
        } else {
            Sign::Positive
        }
    }

    #[inline]
    fn mul_exact(self, rhs: Self) -> Self {
        let neg = self.is_negative() != rhs.is_negative();
        let a = if self.is_negative() { -self } else { self };
        let b = if rhs.is_negative() { -rhs } else { rhs };
        assert!(
            a.bit_length() + b.bit_length() <= Self::BITS - 1,
            "exact predicate: accumulator width budget violated"
        );
        let p = a * b;
        if neg { -p } else { p }
    }
}

#[inline]
fn words_bit_length(words: &[u64]) -> u32 {
    for i in (0..words.len()).rev() {
        if words[i] != 0 {
            return i as u32 * 64 + (64 - words[i].leading_zeros());
        }
    }
    0
}

impl Wide for i128 {
    const BITS: u32 = 128;
    #[inline] fn zero() -> Self { 0 }
    #[inline] fn is_negative(self) -> bool { self < 0 }
    #[inline] fn bit_length(self) -> u32 { 128 - self.leading_zeros() }
}

#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
impl Wide for I256 {
    const BITS: u32 = 256;
    #[inline] fn zero() -> Self { I256::zero() }
    #[inline] fn is_negative(self) -> bool { I256::is_negative(self) }
    #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
}

#[cfg(any(table_format = "q32_32", table_format = "q64_64"))]
impl Wide for I512 {
    const BITS: u32 = 512;
    #[inline] fn zero() -> Self { I512::zero() }
    #[inline] fn is_negative(self) -> bool { I512::is_negative(self) }
    #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
}

#[cfg(any(table_format = "q64_64", table_format = "q128_128"))]
impl Wide for I1024 {
    const BITS: u32 = 1024;
    #[inline] fn zero() -> Self { I1024::zero() }
    #[inline] fn is_negative(self) -> bool { (self.words[15] as i64) < 0 }
    #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
}

#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
impl Wide for I2048 {
    const BITS: u32 = 2048;
    #[inline] fn zero() -> Self { I2048::zero() }
    #[inline] fn is_negative(self) -> bool { (self.words[31] as i64) < 0 }
    #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
}

// Per-profile accumulator selection, from the storage width W:
//   orient2d 2W+2, orient3d 3W+3 -> Orient;  incircle 4W+5, insphere 5W+6 -> Circle.
#[cfg(table_format = "q16_16")]
mod acc {
    pub type Orient = i128; // W = 32: 66 / 99 bits
    pub type Circle = super::I256; // 133 / 166 bits
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { v as i128 }
    #[inline] pub fn circle(v: super::BinaryStorage) -> Circle { super::I256::from_i128(v as i128) }
}
#[cfg(table_format = "q32_32")]
mod acc {
    pub type Orient = super::I256; // W = 64: 130 / 195 bits
    pub type Circle = super::I512; // 261 / 326 bits
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { super::I256::from_i128(v as i128) }
    #[inline] pub fn circle(v: super::BinaryStorage) -> Circle { super::I512::from_i128(v as i128) }
}
#[cfg(table_format = "q64_64")]
mod acc {
    pub type Orient = super::I512; // W = 128: 258 / 387 bits
    pub type Circle = super::I1024; // 517 / 646 bits
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { super::I512::from_i128(v) }
    #[inline] pub fn circle(v: super::BinaryStorage) -> Circle { super::I1024::from_i128(v) }
}
#[cfg(table_format = "q128_128")]
mod acc {
    pub type Orient = super::I1024; // W = 256: 514 / 771 bits
    pub type Circle = super::I2048; // 1029 / 1286 bits
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { super::I1024::from_i256(v) }
    #[inline] pub fn circle(v: super::BinaryStorage) -> Circle { super::I2048::from_i256(v) }
}
#[cfg(table_format = "q256_256")]
mod acc {
    pub type Orient = super::I2048; // W = 512: 1026 / 1539 bits; circle predicates exceed 2048
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { super::I2048::from_i512(v) }
}

#[inline]
fn p2<W: Wide>(p: [FixedPoint; 2], widen: fn(BinaryStorage) -> W) -> [W; 2] {
    [widen(p[0].raw()), widen(p[1].raw())]
}

#[inline]
fn p3<W: Wide>(p: [FixedPoint; 3], widen: fn(BinaryStorage) -> W) -> [W; 3] {
    [widen(p[0].raw()), widen(p[1].raw()), widen(p[2].raw())]
}

#[inline]
fn sub2<W: Wide>(a: [W; 2], b: [W; 2]) -> [W; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
fn sub3<W: Wide>(a: [W; 3], b: [W; 3]) -> [W; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `| r0 ; r1 ; r2 |`, degree 3 in the entries.
#[inline]
fn det3<W: Wide>(r0: [W; 3], r1: [W; 3], r2: [W; 3]) -> W {
    r0[0].mul_exact(r1[1].mul_exact(r2[2]) - r1[2].mul_exact(r2[1]))
        - r0[1].mul_exact(r1[0].mul_exact(r2[2]) - r1[2].mul_exact(r2[0]))
        + r0[2].mul_exact(r1[0].mul_exact(r2[1]) - r1[1].mul_exact(r2[0]))
}

/// `| r0 ; r1 |`, degree 2.
#[inline]
fn det2<W: Wide>(r0: [W; 2], r1: [W; 2]) -> W {
    r0[0].mul_exact(r1[1]) - r0[1].mul_exact(r1[0])
}

// ============================================================================
// Exact predicates
// ============================================================================

/// Orientation of the triangle `a b c`: `Positive` if counterclockwise,
/// `Negative` if clockwise, `Zero` if the three points are exactly collinear.
///
/// The sign of `| a-c ; b-c |`, evaluated exactly.
pub fn orient2d(a: [FixedPoint; 2], b: [FixedPoint; 2], c: [FixedPoint; 2]) -> Sign {
    let (a, b, c) = (p2(a, acc::orient), p2(b, acc::orient), p2(c, acc::orient));
    det2(sub2(a, c), sub2(b, c)).sign()
}

/// Orientation of the tetrahedron `a b c d`: `Positive` if `d` lies below the
/// plane of `a b c` (the triangle seen counterclockwise from above), `Negative`
/// if above, `Zero` if the four points are exactly coplanar.
///
/// The sign of `| a-d ; b-d ; c-d |`, evaluated exactly.
pub fn orient3d(a: [FixedPoint; 3], b: [FixedPoint; 3], c: [FixedPoint; 3], d: [FixedPoint; 3]) -> Sign {
    let (a, b, c, d) = (p3(a, acc::orient), p3(b, acc::orient), p3(c, acc::orient), p3(d, acc::orient));
    det3(sub3(a, d), sub3(b, d), sub3(c, d)).sign()
}

/// Whether `d` lies inside the circle through `a b c`: `Positive` if inside
/// when `a b c` are counterclockwise (the sign flips with their orientation),
/// `Negative` if outside, `Zero` if exactly on the circle.
///
/// The sign of the lifted determinant `| a-d, |a-d|^2 ; b-d, |b-d|^2 ; c-d,
/// |c-d|^2 |`, expanded along the lift column so that every product pairs a
/// degree-2 lift with a degree-2 minor. Not available on the scientific
/// profile (see the module documentation).
#[cfg(not(table_format = "q256_256"))]
pub fn incircle(a: [FixedPoint; 2], b: [FixedPoint; 2], c: [FixedPoint; 2], d: [FixedPoint; 2]) -> Sign {
    let (a, b, c, d) = (p2(a, acc::circle), p2(b, acc::circle), p2(c, acc::circle), p2(d, acc::circle));
    let (ad, bd, cd) = (sub2(a, d), sub2(b, d), sub2(c, d));
    let lift = |v: [acc::Circle; 2]| v[0].mul_exact(v[0]) + v[1].mul_exact(v[1]);
    (lift(ad).mul_exact(det2(bd, cd)) - lift(bd).mul_exact(det2(ad, cd)) + lift(cd).mul_exact(det2(ad, bd))).sign()
}

/// Whether `e` lies inside the sphere through `a b c d`: `Positive` if inside
/// when `orient3d(a, b, c, d)` is `Positive` (the sign flips with their
/// orientation), `Negative` if outside, `Zero` if exactly on the sphere.
///
/// The sign of the lifted 4 x 4 determinant, expanded along the lift column so
/// that every product pairs a degree-2 lift with a degree-3 minor. Not
/// available on the scientific profile (see the module documentation).
#[cfg(not(table_format = "q256_256"))]
pub fn insphere(
    a: [FixedPoint; 3],
    b: [FixedPoint; 3],
    c: [FixedPoint; 3],
    d: [FixedPoint; 3],
    e: [FixedPoint; 3],
) -> Sign {
    let (a, b, c, d, e) = (
        p3(a, acc::circle),
        p3(b, acc::circle),
        p3(c, acc::circle),
        p3(d, acc::circle),
        p3(e, acc::circle),
    );
    let (ae, be, ce, de) = (sub3(a, e), sub3(b, e), sub3(c, e), sub3(d, e));
    let lift = |v: [acc::Circle; 3]| v[0].mul_exact(v[0]) + v[1].mul_exact(v[1]) + v[2].mul_exact(v[2]);
    (det3(be, ce, de).mul_exact(-lift(ae)) + det3(ae, ce, de).mul_exact(lift(be))
        - det3(ae, be, de).mul_exact(lift(ce))
        + det3(ae, be, ce).mul_exact(lift(de)))
    .sign()
}

/// The outcome of a certified positive-definiteness test.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PdVerdict {
    /// Every Cholesky pivot is certainly positive: proven positive definite.
    PositiveDefinite,
    /// Pivot `pivot` is certainly at or below zero, all earlier pivots being
    /// certainly positive: proven not positive definite (rank deficient or
    /// indefinite).
    NotPositiveDefinite { pivot: usize },
    /// Pivot `pivot` encloses zero with a positive upper endpoint: the stored
    /// matrix is too close to singular for the arithmetic to decide. The
    /// enclosure is returned so the caller can see how close.
    Inconclusive { pivot: usize, straddle: Interval },
}

impl PdVerdict {
    /// `true` only for [`PdVerdict::PositiveDefinite`].
    #[inline]
    pub fn is_proven_positive_definite(self) -> bool {
        matches!(self, PdVerdict::PositiveDefinite)
    }
}

/// Certified positive-definiteness verdict via interval Cholesky.
///
/// Mirrors `cholesky_decompose` step for step with `Interval` entries: pivot
/// `d_i = a_ii - sum_k L_ik^2` and column `L_ji = (a_ji - sum_k L_jk L_ik) /
/// L_ii`, each an exact compute-tier accumulation narrowed once. By induction
/// the exact pivots of the stored matrix lie inside the pivot intervals, so a
/// certainly-positive pivot interval proves a positive pivot (Sylvester's
/// criterion in factored form), and a pivot interval at or below zero proves
/// the opposite.
///
/// Returns `Err(TierOverflow)` if an entry of the factor leaves the storage
/// tier. Panics if `a` is not square.
pub fn pd_verdict(a: &FixedMatrix) -> Result<PdVerdict, OverflowDetected> {
    assert!(a.is_square(), "pd_verdict: matrix must be square");
    let n = a.rows();
    let zero = Interval::point(FixedPoint::ZERO);
    // Lower-triangular factor, row-major, entries are enclosures.
    let mut l: Vec<Vec<Interval>> = vec![vec![zero; n]; n];

    for i in 0..n {
        let pivot = Interval::point(a.get(i, i)).try_sub(Interval::try_dot_intervals(&l[i][..i], &l[i][..i])?)?;
        if pivot.is_certainly_positive() {
            let l_ii = pivot.try_sqrt()?;
            l[i][i] = l_ii;
            for j in (i + 1)..n {
                let numerator = Interval::point(a.get(j, i)).try_sub(Interval::try_dot_intervals(&l[j][..i], &l[i][..i])?)?;
                l[j][i] = numerator.try_div(l_ii)?;
            }
        } else if pivot.hi() <= FixedPoint::ZERO {
            return Ok(PdVerdict::NotPositiveDefinite { pivot: i });
        } else {
            return Ok(PdVerdict::Inconclusive { pivot: i, straddle: pivot });
        }
    }
    Ok(PdVerdict::PositiveDefinite)
}
