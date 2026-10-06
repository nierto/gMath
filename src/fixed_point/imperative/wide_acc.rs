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


// ============================================================================
// Exact accumulator arithmetic
// ============================================================================

// The arithmetic lives in `width`: one integer surface over every word, and
// per storage word the compute word and the accumulators.
pub(crate) use crate::fixed_point::width::WideInt as Wide;
use crate::fixed_point::width::Storage;

// Per-profile accumulator selection, from the storage width W:
//   orient2d 2W+2, orient3d 3W+3, quadratic form 3W-2+2log2(n) -> Orient;
//   incircle 4W+5, insphere 5W+6 -> Circle (no such word at W = 512).
// `orient` and `circle` widen a storage value exactly.
pub(crate) mod acc {
    use super::BinaryStorage;
    use crate::fixed_point::width::Storage;
    #[cfg(not(table_format = "q256_256"))]
    use crate::fixed_point::width::CircleStorage;

    pub type Orient = <BinaryStorage as Storage>::Orient;
    #[cfg(not(table_format = "q256_256"))]
    pub type Circle = <BinaryStorage as CircleStorage>::Circle;
    #[inline] pub fn orient(v: BinaryStorage) -> Orient { Storage::orient(v) }
    #[cfg(not(table_format = "q256_256"))]
    #[inline] pub fn circle(v: BinaryStorage) -> Circle { CircleStorage::circle(v) }
}

// ============================================================================
// The fused quadratic form: one exact value at 3F fractional bits
// ============================================================================

/// A storage value sign-extended to the compute width (no shift; exact).
#[inline]
pub(crate) fn widen_storage(v: BinaryStorage) -> ComputeStorage {
    Storage::widen(v)
}

/// Exact product of two compute-width values on the accumulator (compute
/// width doubled). The magnitudes' bit lengths are asserted against the
/// accumulator first (the width budget made loud, as in [`Wide::mul_exact`]).
#[inline]
pub(crate) fn widen_product(a: ComputeStorage, b: ComputeStorage) -> acc::Orient {
    <BinaryStorage as Storage>::widen_product(a, b)
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

#[cfg(test)]
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
    super::kernels::narrow_triple_floor::<crate::fixed_point::fixed_num::ProfileFormat>(a)
}

/// Narrow a `3F`-scaled exact value to the storage scale, rounding toward
/// positive infinity. The bump above the floor is checked: a floor at the
/// storage maximum has no ceiling in storage, and that is a `TierOverflow`.
#[inline]
pub(crate) fn narrow_triple_ceil(a: acc::Orient) -> Result<BinaryStorage, OverflowDetected> {
    super::kernels::narrow_triple_ceil::<crate::fixed_point::fixed_num::ProfileFormat>(a)
}

/// Narrow a `3F`-scaled exact value to the storage scale, rounding to nearest
/// with ties toward positive infinity: floor plus the round bit, the binary
/// house rule, on every profile. The bump is checked.
#[inline]
pub(crate) fn narrow_triple_nearest(a: acc::Orient) -> Result<BinaryStorage, OverflowDetected> {
    super::kernels::narrow_triple_nearest::<crate::fixed_point::fixed_num::ProfileFormat>(a)
}

/// `a / 2^shift` rounded to nearest with ties toward positive infinity (floor
/// plus the round bit), narrowed to storage (checked). For an exact product
/// whose scale carries a power-of-two normalisation, so the shift is not one
/// of the fixed `2F` or `3F`. (Realtime does the same on a plain `i128` in
/// `fused`.)
#[cfg(not(table_format = "q16_16"))]
pub(crate) fn narrow_shifted_nearest(a: acc::Orient, shift: u32) -> Result<BinaryStorage, OverflowDetected> {
    super::kernels::narrow_shifted_nearest::<crate::fixed_point::fixed_num::ProfileFormat>(a, shift)
}

/// `num / den` rounded to nearest, ties toward positive infinity, narrowed
/// to storage (checked): `num` an exact wide value, `den` a nonzero compute
/// raw. With `num` at `3F` fractional bits and `den` at `2F` the quotient is
/// at the storage scale. Lets a caller divide an exact product by a compute
/// value without forming a quotient that may exceed the compute tier.
pub(crate) fn divide_to_storage_nearest(num: acc::Orient, den: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    super::kernels::divide_to_storage_nearest::<crate::fixed_point::fixed_num::ProfileFormat>(num, den)
}

/// `num / den` rounded to nearest, ties toward positive infinity, narrowed to
/// the compute tier (checked): with `num` at `4F` fractional bits and `den`
/// at `2F` the quotient is a compute raw. The compute-tier twin of
/// [`divide_to_storage_nearest`], for algorithms that keep their state at the
/// compute tier between steps.
pub(crate) fn divide_to_compute_nearest(num: acc::Orient, den: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    super::kernels::divide_to_compute_nearest::<crate::fixed_point::fixed_num::ProfileFormat>(num, den)
}

/// Exact `sum a_i b_i` of compute raws at `4F` fractional bits (checked).
pub(crate) fn exact_dot_compute(a: &[ComputeStorage], b: &[ComputeStorage]) -> Result<acc::Orient, OverflowDetected> {
    super::kernels::exact_dot_compute::<crate::fixed_point::fixed_num::ProfileFormat>(a, b)
}

/// `init - sum a_i b_i` of compute raws: the sum exact at `4F`, `init`
/// widened exactly, one rounding to the compute tier (nearest, ties toward
/// +infinity, checked). The substitution and elimination kernel of the
/// compute-tier factorizations.
pub(crate) fn exact_sub_dot_compute(
    init: ComputeStorage, a: &[ComputeStorage], b: &[ComputeStorage],
) -> Result<ComputeStorage, OverflowDetected> {
    super::kernels::exact_sub_dot_compute::<crate::fixed_point::fixed_num::ProfileFormat>(init, a, b)
}

/// Narrow an exact product of two compute raws (`4F` fractional bits, from
/// [`widen_product`]) to the compute scale (`2F`), rounding to nearest with
/// ties toward positive infinity. The result is fits-checked against the
/// compute width; a value beyond it is a `TierOverflow`, never a wrap.
#[inline]
pub(crate) fn narrow_product_to_compute(a: acc::Orient) -> Result<ComputeStorage, OverflowDetected> {
    super::kernels::narrow_product_to_compute::<crate::fixed_point::fixed_num::ProfileFormat>(a)
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

    /// Dividing by the compute-tier one (2^(2F)) is narrowing by 2^(2F): the
    /// quotient rounding must equal narrow_triple_nearest, ties included
    /// (both signs).
    #[test]
    fn divide_by_compute_one_matches_the_nearest_narrowing() {
        use crate::fixed_point::universal::fasc::stack_evaluator::compute::make_compute_int;
        let one_c = make_compute_int(1);
        let half = FixedPoint::one() / FixedPoint::from_int(2);
        let mut state: u64 = 0x5DEE_CE66_D1CE_4E5B;
        let mut cases = 0;
        // multipliers below 128: storage integers up to 24 fraction bits
        for k in -30i32..=30 {
            for tweak in [0i32, 1, 2, 3] {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let ulps = (state >> 40) as i32 % 16 + k * 3 + tweak;
                let v = FixedVector::from_slice(&[half]);
                let mut m = FixedMatrix::new(1, 1);
                m.set(0, 0, FixedPoint::from_raw(storage_one()) * FixedPoint::from_int(ulps));
                let a = quadratic_form_exact(&v, &m).unwrap();
                assert_eq!(divide_to_storage_nearest(a, one_c), narrow_triple_nearest(a), "{ulps}");
                cases += 1;
            }
        }
        assert!(cases > 200);
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
