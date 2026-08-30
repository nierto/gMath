//! Exact wide accumulators shared by the exact predicates, the certified
//! intervals and the fused quadratic form.
//!
//! One trait, [`Wide`], over the profile's wide integer types: signed
//! multiplication on magnitudes with a loud bit-length budget, checked
//! addition, sign test. One alias module, [`acc`], selecting the accumulator
//! per profile from the storage width `W`: `Orient` (orient2d `2W+2` bits,
//! orient3d `3W+3`, the fused quadratic form `3W-2+2 log2 n`) and `Circle`
//! (incircle `4W+5`, insphere `5W+6`). One fused kernel,
//! [`quadratic_form_exact`], and the three narrowings from `3F` fractional
//! bits down to `F`: floor and ceil for an enclosure, nearest with ties
//! toward positive infinity for a scalar (the binary house rule).
//!
//! Nothing here is public. The predicates, `Interval` and `fused` are the
//! surfaces; this module is the arithmetic they share so that the width
//! budget lives in one place.

use super::interval::exact_product;
use super::{FixedMatrix, FixedVector};
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::universal::fasc::stack_evaluator::{BinaryStorage, ComputeStorage};

#[cfg(table_format = "q16_16")]
use crate::fixed_point::frac_config;
#[cfg(table_format = "q32_32")]
use crate::fixed_point::i256::mul_i128_to_i256;
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
use crate::fixed_point::I256;
#[cfg(any(table_format = "q32_32", table_format = "q64_64"))]
use crate::fixed_point::I512;
#[cfg(any(table_format = "q64_64", table_format = "q128_128"))]
use crate::fixed_point::I1024;
#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
use crate::fixed_point::I2048;

// ============================================================================
// Exact accumulator arithmetic
// ============================================================================

/// Exact signed integer arithmetic on a fixed-width accumulator.
///
/// Products are formed on magnitudes and the sign reapplied, so the wide
/// types' `Mul` implementations are only ever entered with non-negative
/// operands (a schoolbook or a truncated unsigned product, both exact there).
/// Each product first asserts that the operands' bit lengths fit the type,
/// which is the width budget made loud. Sums are checked: a running sum that
/// leaves the accumulator is a typed `TierOverflow`, never a wrap.
pub(crate) trait Wide:
    Copy + PartialEq + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self> + std::ops::Neg<Output = Self>
{
    const BITS: u32;
    fn zero() -> Self;
    fn is_negative(self) -> bool;
    /// Significant bits of a NON-NEGATIVE value.
    fn bit_length(self) -> u32;
    /// Checked signed addition on the accumulator (`None` on overflow).
    fn checked_add_wide(self, rhs: Self) -> Option<Self>;

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

    #[inline]
    fn add_exact(self, rhs: Self) -> Result<Self, OverflowDetected> {
        self.checked_add_wide(rhs).ok_or(OverflowDetected::TierOverflow)
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
    #[inline] fn checked_add_wide(self, rhs: Self) -> Option<Self> { i128::checked_add(self, rhs) }
}

#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
impl Wide for I256 {
    const BITS: u32 = 256;
    #[inline] fn zero() -> Self { I256::zero() }
    #[inline] fn is_negative(self) -> bool { I256::is_negative(self) }
    #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
    #[inline] fn checked_add_wide(self, rhs: Self) -> Option<Self> { I256::checked_add(self, rhs) }
}

#[cfg(any(table_format = "q32_32", table_format = "q64_64"))]
impl Wide for I512 {
    const BITS: u32 = 512;
    #[inline] fn zero() -> Self { I512::zero() }
    #[inline] fn is_negative(self) -> bool { I512::is_negative(self) }
    #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
    #[inline] fn checked_add_wide(self, rhs: Self) -> Option<Self> { I512::checked_add(self, rhs) }
}

#[cfg(any(table_format = "q64_64", table_format = "q128_128"))]
impl Wide for I1024 {
    const BITS: u32 = 1024;
    #[inline] fn zero() -> Self { I1024::zero() }
    #[inline] fn is_negative(self) -> bool { (self.words[15] as i64) < 0 }
    #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
    #[inline] fn checked_add_wide(self, rhs: Self) -> Option<Self> { I1024::checked_add(self, rhs) }
}

#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
impl Wide for I2048 {
    const BITS: u32 = 2048;
    #[inline] fn zero() -> Self { I2048::zero() }
    #[inline] fn is_negative(self) -> bool { (self.words[31] as i64) < 0 }
    #[inline] fn bit_length(self) -> u32 { words_bit_length(&self.words) }
    #[inline] fn checked_add_wide(self, rhs: Self) -> Option<Self> { I2048::checked_add(self, rhs) }
}

// Per-profile accumulator selection, from the storage width W:
//   orient2d 2W+2, orient3d 3W+3, quadratic form 3W-2+2log2(n) -> Orient;
//   incircle 4W+5, insphere 5W+6 -> Circle.
// `orient` widens a storage value exactly.
#[cfg(table_format = "q16_16")]
pub(crate) mod acc {
    pub type Orient = i128; // W = 32: 66 / 99 bits
    pub type Circle = super::I256; // 133 / 166 bits
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { v as i128 }
    #[inline] pub fn circle(v: super::BinaryStorage) -> Circle { super::I256::from_i128(v as i128) }
}
#[cfg(table_format = "q32_32")]
pub(crate) mod acc {
    pub type Orient = super::I256; // W = 64: 130 / 195 bits
    pub type Circle = super::I512; // 261 / 326 bits
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { super::I256::from_i128(v as i128) }
    #[inline] pub fn circle(v: super::BinaryStorage) -> Circle { super::I512::from_i128(v as i128) }
}
#[cfg(table_format = "q64_64")]
pub(crate) mod acc {
    pub type Orient = super::I512; // W = 128: 258 / 387 bits
    pub type Circle = super::I1024; // 517 / 646 bits
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { super::I512::from_i128(v) }
    #[inline] pub fn circle(v: super::BinaryStorage) -> Circle { super::I1024::from_i128(v) }
}
#[cfg(table_format = "q128_128")]
pub(crate) mod acc {
    pub type Orient = super::I1024; // W = 256: 514 / 771 bits
    pub type Circle = super::I2048; // 1029 / 1286 bits
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { super::I1024::from_i256(v) }
    #[inline] pub fn circle(v: super::BinaryStorage) -> Circle { super::I2048::from_i256(v) }
}
#[cfg(table_format = "q256_256")]
pub(crate) mod acc {
    pub type Orient = super::I2048; // W = 512: 1026 / 1539 bits; circle predicates exceed 2048
    #[inline] pub fn orient(v: super::BinaryStorage) -> Orient { super::I2048::from_i512(v) }
}

// ============================================================================
// The fused quadratic form: one exact value at 3F fractional bits
// ============================================================================

/// A storage value sign-extended to the compute width (no shift; exact).
#[inline]
fn widen_storage(v: BinaryStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    {
        v as i64
    }
    #[cfg(table_format = "q32_32")]
    {
        v as i128
    }
    #[cfg(table_format = "q64_64")]
    {
        crate::fixed_point::I256::from_i128(v)
    }
    #[cfg(table_format = "q128_128")]
    {
        crate::fixed_point::I512::from_i256(v)
    }
    #[cfg(table_format = "q256_256")]
    {
        crate::fixed_point::I1024::from_i512(v)
    }
}

/// Sign of a compute-width value (the compute types do not all expose
/// `is_negative`; the top word's sign bit is the definition).
#[inline]
#[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256"))]
fn compute_is_negative(v: ComputeStorage) -> bool {
    (v.words[v.words.len() - 1] as i64) < 0
}

/// Exact product of two compute-width values on the accumulator (compute
/// width doubled). The wide `mul_to_*` family is UNSIGNED by convention, so
/// it is applied to magnitudes and the sign reapplied; the magnitudes'
/// bit lengths are asserted against the accumulator first (the width budget
/// made loud, as in [`Wide::mul_exact`]).
#[inline]
fn widen_product(a: ComputeStorage, b: ComputeStorage) -> acc::Orient {
    #[cfg(table_format = "q16_16")]
    {
        (a as i128) * (b as i128)
    }
    #[cfg(table_format = "q32_32")]
    {
        // sign-correct by itself
        mul_i128_to_i256(a, b)
    }
    #[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256"))]
    {
        let (a_neg, b_neg) = (compute_is_negative(a), compute_is_negative(b));
        let neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        assert!(
            words_bit_length(&abs_a.words) + words_bit_length(&abs_b.words) <= acc::Orient::BITS - 1,
            "exact predicate: accumulator width budget violated"
        );
        #[cfg(table_format = "q64_64")]
        let p = abs_a.mul_to_i512(abs_b);
        #[cfg(table_format = "q128_128")]
        let p = abs_a.mul_to_i1024(abs_b);
        #[cfg(table_format = "q256_256")]
        let p = abs_a.mul_to_i2048(abs_b);
        if neg { -p } else { p }
    }
}

/// Exact `v^T M v` at `3F` fractional bits on the orientation accumulator.
///
/// Every term is an exact triple product: `v_i v_j` is formed exactly at the
/// compute tier (`2F`) and multiplied on the accumulator by the exact entry
/// `m_ii` (or, off the diagonal, by `m_ij + m_ji` at the compute width, which
/// pairs the two symmetric terms into one wide multiply; no symmetry of `M`
/// is assumed). The sum is checked. The value returned has not been rounded
/// at all; the caller narrows it once with [`narrow_triple_floor`],
/// [`narrow_triple_ceil`] or [`narrow_triple_nearest`].
///
/// Budget: `|v_i v_j| < 2^(2W-2)`, `|m_ij + m_ji| < 2^W`, so each product is
/// below `3W-2` bits and the sum of `n(n+1)/2` of them below
/// `3W-2+2 log2 n`; the accumulator has `4W` (or more) bits and every
/// multiply asserts its own budget.
///
/// Panics if `m` is not square or its size differs from `v`.
pub(crate) fn quadratic_form_exact(v: &FixedVector, m: &FixedMatrix) -> Result<acc::Orient, OverflowDetected> {
    let n = v.len();
    assert!(m.is_square() && m.rows() == n, "quadratic_form: dimension mismatch");
    let mut sum = acc::Orient::zero();
    for i in 0..n {
        let vi = v[i].raw();
        let vv = exact_product(vi, vi);
        sum = sum.add_exact(widen_product(widen_storage(m.get(i, i).raw()), vv))?;
        for j in (i + 1)..n {
            // W+1 bits at most: exact at the compute width (2W).
            let pair = widen_storage(m.get(i, j).raw()) + widen_storage(m.get(j, i).raw());
            let vv = exact_product(vi, v[j].raw());
            sum = sum.add_exact(widen_product(pair, vv))?;
        }
    }
    Ok(sum)
}

/// Split a `3F`-scaled accumulator into its floor at the storage scale, the
/// "any discarded bit set" flag and the round bit (bit `2F-1`).
///
/// The arithmetic right shift is floor in two's complement. The floor is
/// fits-checked against the storage tier; a value beyond it is a
/// `TierOverflow`.
#[inline]
fn split_triple(a: acc::Orient) -> Result<(BinaryStorage, bool, bool), OverflowDetected> {
    #[cfg(table_format = "q16_16")]
    {
        let shift = 2 * frac_config::FRAC_BITS;
        let mask = (1i128 << shift) - 1;
        let inexact = (a & mask) != 0;
        let round_bit = ((a >> (shift - 1)) & 1) == 1;
        let shifted = a >> shift;
        if shifted > i32::MAX as i128 || shifted < i32::MIN as i128 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok((shifted as i32, inexact, round_bit))
    }
    #[cfg(table_format = "q32_32")]
    {
        // I256 >> 64: the discarded bits are exactly words[0]
        let inexact = a.words[0] != 0;
        let round_bit = (a.words[0] >> 63) == 1;
        let shifted = a >> 64u32;
        if !shifted.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        let s = shifted.as_i128();
        if s > i64::MAX as i128 || s < i64::MIN as i128 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok((s as i64, inexact, round_bit))
    }
    #[cfg(table_format = "q64_64")]
    {
        // I512 >> 128: the discarded bits are words[0..2]
        let inexact = (a.words[0] | a.words[1]) != 0;
        let round_bit = (a.words[1] >> 63) == 1;
        let shifted = a >> 128usize;
        if !shifted.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok((shifted.as_i128(), inexact, round_bit))
    }
    #[cfg(table_format = "q128_128")]
    {
        // I1024 >> 256: the discarded bits are words[0..4]
        let inexact = (a.words[0] | a.words[1] | a.words[2] | a.words[3]) != 0;
        let round_bit = (a.words[3] >> 63) == 1;
        let shifted = a >> 256usize;
        if !shifted.fits_in_i256() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok((shifted.as_i256(), inexact, round_bit))
    }
    #[cfg(table_format = "q256_256")]
    {
        // I2048 >> 512: the discarded bits are words[0..8]
        let mut low = 0u64;
        for w in &a.words[0..8] {
            low |= *w;
        }
        let inexact = low != 0;
        let round_bit = (a.words[7] >> 63) == 1;
        let shifted = a >> 512usize;
        if !shifted.fits_in_i512() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok((shifted.as_i512(), inexact, round_bit))
    }
}

#[inline]
fn storage_one() -> BinaryStorage {
    #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
    {
        1
    }
    #[cfg(table_format = "q128_128")]
    {
        crate::fixed_point::I256::from_i128(1)
    }
    #[cfg(table_format = "q256_256")]
    {
        crate::fixed_point::I512::from_i128(1)
    }
}

/// Narrow a `3F`-scaled exact value to the storage scale, rounding toward
/// negative infinity. Fits-checked; never wraps.
#[inline]
pub(crate) fn narrow_triple_floor(a: acc::Orient) -> Result<BinaryStorage, OverflowDetected> {
    let (floor, _, _) = split_triple(a)?;
    Ok(floor)
}

/// Narrow a `3F`-scaled exact value to the storage scale, rounding toward
/// positive infinity. The bump above the floor is checked: a floor at the
/// storage maximum has no ceiling in storage, and that is a `TierOverflow`.
#[inline]
pub(crate) fn narrow_triple_ceil(a: acc::Orient) -> Result<BinaryStorage, OverflowDetected> {
    let (floor, inexact, _) = split_triple(a)?;
    if inexact {
        floor.checked_add(storage_one()).ok_or(OverflowDetected::TierOverflow)
    } else {
        Ok(floor)
    }
}

/// Narrow a `3F`-scaled exact value to the storage scale, rounding to nearest
/// with ties toward positive infinity: floor plus the round bit, the binary
/// house rule, on every profile. The bump is checked.
#[inline]
pub(crate) fn narrow_triple_nearest(a: acc::Orient) -> Result<BinaryStorage, OverflowDetected> {
    let (floor, _, round_bit) = split_triple(a)?;
    if round_bit {
        floor.checked_add(storage_one()).ok_or(OverflowDetected::TierOverflow)
    } else {
        Ok(floor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::FixedPoint;

    fn one_third_raw() -> BinaryStorage {
        (FixedPoint::one() / FixedPoint::from_int(3)).raw()
    }

    /// A storage value `x` widened to the accumulator, times `2^(2F)`, narrows
    /// back to `x` under all three rules with nothing discarded.
    #[test]
    fn exact_multiples_narrow_identically_under_every_rule() {
        let x = one_third_raw();
        let v = FixedVector::from_slice(&[FixedPoint::one()]);
        let mut m = FixedMatrix::new(1, 1);
        m.set(0, 0, FixedPoint::from_raw(x));
        let a = quadratic_form_exact(&v, &m).unwrap();
        assert_eq!(narrow_triple_floor(a).unwrap(), x);
        assert_eq!(narrow_triple_ceil(a).unwrap(), x);
        assert_eq!(narrow_triple_nearest(a).unwrap(), x);
    }

    /// `v = [1/2]`, `m = [[2 ulp]]`: the exact value is exactly half an ulp,
    /// a tie. Floor 0, ceil 1, nearest rounds the tie toward +infinity: 1.
    /// The negative twin: floor -1, ceil 0, nearest 0.
    #[test]
    fn constructed_ties_round_toward_positive_infinity() {
        let half = FixedPoint::one() / FixedPoint::from_int(2);
        let two_ulp = FixedPoint::from_raw(storage_one() + storage_one());
        let v = FixedVector::from_slice(&[half]);
        let mut m = FixedMatrix::new(1, 1);
        m.set(0, 0, two_ulp);
        let a = quadratic_form_exact(&v, &m).unwrap();
        let zero = FixedPoint::ZERO.raw();
        assert_eq!(narrow_triple_floor(a).unwrap(), zero);
        assert_eq!(narrow_triple_ceil(a).unwrap(), storage_one());
        assert_eq!(narrow_triple_nearest(a).unwrap(), storage_one());
        m.set(0, 0, -two_ulp);
        let a = quadratic_form_exact(&v, &m).unwrap();
        assert_eq!(narrow_triple_floor(a).unwrap(), -storage_one());
        assert_eq!(narrow_triple_ceil(a).unwrap(), zero);
        assert_eq!(narrow_triple_nearest(a).unwrap(), zero);
    }

    /// The pairing `(m_ij + m_ji) v_i v_j` must not assume symmetry: an
    /// antisymmetric off-diagonal contributes nothing, a one-sided one
    /// contributes its full product.
    #[test]
    fn off_diagonal_terms_are_paired_without_assuming_symmetry() {
        let v = FixedVector::from_slice(&[FixedPoint::from_int(3), FixedPoint::from_int(5)]);
        let mut m = FixedMatrix::new(2, 2);
        m.set(0, 1, FixedPoint::from_int(7));
        m.set(1, 0, FixedPoint::from_int(-7));
        let a = quadratic_form_exact(&v, &m).unwrap();
        assert_eq!(narrow_triple_nearest(a).unwrap(), FixedPoint::ZERO.raw());
        m.set(1, 0, FixedPoint::ZERO);
        let a = quadratic_form_exact(&v, &m).unwrap();
        assert_eq!(narrow_triple_nearest(a).unwrap(), FixedPoint::from_int(105).raw());
    }

    /// A result beyond the storage tier is a typed overflow, never a wrap.
    #[test]
    fn a_result_beyond_storage_is_a_typed_overflow() {
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        let big = FixedPoint::from_raw(BinaryStorage::MAX);
        #[cfg(table_format = "q128_128")]
        let big = FixedPoint::from_raw(crate::fixed_point::I256::max_value());
        #[cfg(table_format = "q256_256")]
        let big = FixedPoint::from_raw(crate::fixed_point::I512::max_value());
        let v = FixedVector::from_slice(&[big]);
        let mut m = FixedMatrix::new(1, 1);
        m.set(0, 0, big);
        let a = quadratic_form_exact(&v, &m).unwrap();
        assert_eq!(narrow_triple_floor(a), Err(OverflowDetected::TierOverflow));
        assert_eq!(narrow_triple_ceil(a), Err(OverflowDetected::TierOverflow));
        assert_eq!(narrow_triple_nearest(a), Err(OverflowDetected::TierOverflow));
    }
}
