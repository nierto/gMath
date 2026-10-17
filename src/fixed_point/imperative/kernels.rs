//! Width-generic kernels shared by the vectors, matrices, decompositions,
//! predicates, intervals and fused operations: dot products and exact
//! accumulations, the narrowings from the accumulator and the compute tier,
//! nearest division, rotations and Householder reflections, and the
//! convergence thresholds of the iterative decompositions.
//!
//! Every function here is generic over a format `T: FixedNum` and works on
//! raw words: `T::Storage`, the compute word `Compute<T>` (twice as wide, `2F`
//! fractional bits) and the accumulator `Orient<T>` (twice as wide again).
//! Each body is the code the 0.6.x profile-bound functions in `linalg` and
//! `wide_acc` carried per profile, written once; those functions now
//! delegate here with the caller's format, so each width returns the
//! integers it always did.

use super::FixedMatrix;
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::fixed_num::{Compute, Fixed, FixedNum};
use crate::fixed_point::width::{Storage, WideInt};

/// The exact accumulator of a format: the orientation accumulator of its
/// storage word.
pub(crate) type Orient<T> = <<T as FixedNum>::Storage as Storage>::Orient;

// ============================================================================
// Storage-level dot products at the compute tier
// ============================================================================

/// `sum a_i b_i` at the compute tier, unrounded. Checked: a sum beyond the
/// compute tier panics instead of wrapping. Takes the width's fast path when
/// it has one and its bound holds.
#[inline]
pub(crate) fn dot_acc<T: FixedNum>(a: &[T::Storage], b: &[T::Storage]) -> Compute<T> {
    assert_eq!(a.len(), b.len(), "compute_tier_dot_raw: length mismatch");
    if let Some(acc) = T::Storage::dot_acc_fast(a, b) {
        return acc;
    }
    dot_acc_checked::<T>(a.iter().copied().zip(b.iter().copied()))
}

/// The checked compute-tier accumulator over `(a_i, b_i)` pairs.
#[inline]
pub(crate) fn dot_acc_checked<T: FixedNum>(pairs: impl Iterator<Item = (T::Storage, T::Storage)>) -> Compute<T> {
    const OVERFLOW: &str = "compute_tier_dot: sum exceeds the compute tier";
    let mut acc = Compute::<T>::zero();
    for (x, y) in pairs {
        acc = acc.checked_add_wide(x.mul_wide(y)).expect(OVERFLOW);
    }
    acc
}

/// `init - sum a_i b_i` at the compute tier, unrounded (the products are
/// exact there). Checked like [`dot_acc`].
pub(crate) fn sub_dot_compute<T: FixedNum>(init: T::Storage, a: &[T::Storage], b: &[T::Storage]) -> Compute<T> {
    const OVERFLOW: &str = "compute_tier_sub_dot: sum exceeds the compute tier";
    assert_eq!(a.len(), b.len(), "compute_tier_sub_dot_compute: length mismatch");
    let mut acc = T::upscale_to_compute(init);
    for (x, y) in a.iter().zip(b) {
        acc = acc.checked_sub_wide(x.mul_wide(*y)).expect(OVERFLOW);
    }
    acc
}

/// `sum a_i b_i` at the compute tier; `Err(TierOverflow)` instead of a panic.
pub(crate) fn exact_dot<T: FixedNum>(a: &[T::Storage], b: &[T::Storage]) -> Result<Compute<T>, OverflowDetected> {
    assert_eq!(a.len(), b.len(), "exact_dot: length mismatch");
    let mut acc = Compute::<T>::zero();
    for (x, y) in a.iter().zip(b) {
        acc = acc.checked_add_wide(x.mul_wide(*y)).ok_or(OverflowDetected::TierOverflow)?;
    }
    Ok(acc)
}

// ============================================================================
// Narrowing from the accumulator (3F or 4F fractional bits)
// ============================================================================

/// Split a `3F`-scaled accumulator into its floor at the storage scale, the
/// "any discarded bit set" flag and the round bit (bit `2F-1`). The floor is
/// fits-checked against the storage tier.
#[inline]
fn split_triple<T: FixedNum>(a: Orient<T>) -> Result<(T::Storage, bool, bool), OverflowDetected> {
    let shift = 2 * T::FRAC_BITS;
    let inexact = a.low_bits_any(shift);
    let round_bit = a.bit(shift - 1);
    let floor = T::Storage::narrow_orient(a.shr_bits(shift)).ok_or(OverflowDetected::TierOverflow)?;
    Ok((floor, inexact, round_bit))
}

/// Floor of a `3F`-scaled accumulator at the storage scale.
#[inline]
pub(crate) fn narrow_triple_floor<T: FixedNum>(a: Orient<T>) -> Result<T::Storage, OverflowDetected> {
    Ok(split_triple::<T>(a)?.0)
}

/// Ceiling of a `3F`-scaled accumulator at the storage scale.
#[inline]
pub(crate) fn narrow_triple_ceil<T: FixedNum>(a: Orient<T>) -> Result<T::Storage, OverflowDetected> {
    let (floor, inexact, _) = split_triple::<T>(a)?;
    if inexact { floor.checked_add_wide(T::Storage::one()).ok_or(OverflowDetected::TierOverflow) } else { Ok(floor) }
}

/// Nearest storage value to a `3F`-scaled accumulator, ties toward +infinity.
#[inline]
pub(crate) fn narrow_triple_nearest<T: FixedNum>(a: Orient<T>) -> Result<T::Storage, OverflowDetected> {
    let (floor, _, round_bit) = split_triple::<T>(a)?;
    if round_bit { floor.checked_add_wide(T::Storage::one()).ok_or(OverflowDetected::TierOverflow) } else { Ok(floor) }
}

/// Nearest storage value to an accumulator shifted right by `shift` bits,
/// ties toward +infinity; `shift == 0` is the checked narrowing itself.
#[allow(dead_code)] // its profile-bound caller is compiled on the wider profiles only
pub(crate) fn narrow_shifted_nearest<T: FixedNum>(a: Orient<T>, shift: u32) -> Result<T::Storage, OverflowDetected> {
    if shift == 0 {
        return T::Storage::narrow_orient(a).ok_or(OverflowDetected::TierOverflow);
    }
    let round_bit = a.bit(shift - 1);
    let floor = T::Storage::narrow_orient(a.shr_bits(shift)).ok_or(OverflowDetected::TierOverflow)?;
    if round_bit { floor.checked_add_wide(T::Storage::one()).ok_or(OverflowDetected::TierOverflow) } else { Ok(floor) }
}

/// An exact product of two compute values (`4F` fractional bits) rounded
/// to the compute tier, nearest with ties toward +infinity; checked.
#[inline]
pub(crate) fn narrow_product_to_compute<T: FixedNum>(a: Orient<T>) -> Result<Compute<T>, OverflowDetected> {
    let shift = 2 * T::FRAC_BITS;
    let round_bit = a.bit(shift - 1);
    let floor = T::Storage::narrow_orient_to_compute(a.shr_bits(shift)).ok_or(OverflowDetected::TierOverflow)?;
    if round_bit { floor.checked_add_wide(Compute::<T>::one()).ok_or(OverflowDetected::TierOverflow) } else { Ok(floor) }
}

// ============================================================================
// Division of an accumulator by a compute value, one rounding
// ============================================================================

/// `num / den` rounded to nearest, ties toward +infinity, on the accumulator.
fn divide_nearest<T: FixedNum>(num: Orient<T>, den: Compute<T>) -> Result<Orient<T>, OverflowDetected> {
    let d = T::Storage::widen_compute(den);
    if d == Orient::<T>::zero() {
        return Err(OverflowDetected::DivisionByZero);
    }
    let (q, r) = num.divmod(d);
    let positive = num.is_negative() == d.is_negative();
    let r_abs = if r.is_negative() { -r } else { r };
    let d_abs = if d.is_negative() { -d } else { d };
    let twice = r_abs + r_abs;
    let bump = if positive { twice >= d_abs } else { twice > d_abs };
    let one = Orient::<T>::one();
    Ok(if bump { if positive { q + one } else { q - one } } else { q })
}

/// `num / den` rounded once to the nearest storage value.
pub(crate) fn divide_to_storage_nearest<T: FixedNum>(num: Orient<T>, den: Compute<T>) -> Result<T::Storage, OverflowDetected> {
    T::Storage::narrow_orient(divide_nearest::<T>(num, den)?).ok_or(OverflowDetected::TierOverflow)
}

/// `num / den` rounded once to the nearest compute value.
pub(crate) fn divide_to_compute_nearest<T: FixedNum>(num: Orient<T>, den: Compute<T>) -> Result<Compute<T>, OverflowDetected> {
    T::Storage::narrow_orient_to_compute(divide_nearest::<T>(num, den)?).ok_or(OverflowDetected::TierOverflow)
}

// ============================================================================
// Compute-level dot products on the accumulator
// ============================================================================

/// The exact `sum a_i b_i` of compute values on the accumulator.
pub(crate) fn exact_dot_compute<T: FixedNum>(a: &[Compute<T>], b: &[Compute<T>]) -> Result<Orient<T>, OverflowDetected> {
    assert_eq!(a.len(), b.len(), "exact_dot_compute: length mismatch");
    let mut sum = Orient::<T>::zero();
    for (x, y) in a.iter().zip(b) {
        sum = sum.add_exact(T::Storage::widen_product(*x, *y))?;
    }
    Ok(sum)
}

/// `init - sum a_i b_i` of compute values, the difference formed exactly on
/// the accumulator and rounded once to the compute tier.
pub(crate) fn exact_sub_dot_compute<T: FixedNum>(init: Compute<T>, a: &[Compute<T>], b: &[Compute<T>]) -> Result<Compute<T>, OverflowDetected> {
    let one = T::make_compute_int(1);
    let dot = exact_dot_compute::<T>(a, b)?;
    narrow_product_to_compute::<T>(T::Storage::widen_product(init, one).add_exact(-dot)?)
}

/// `a * b` of compute values rounded once to the compute tier.
#[inline]
pub(crate) fn compute_product<T: FixedNum>(a: Compute<T>, b: Compute<T>) -> Result<Compute<T>, OverflowDetected> {
    narrow_product_to_compute::<T>(T::Storage::widen_product(a, b))
}

// ============================================================================
// Compute-tier helpers of the decompositions
// ============================================================================

#[inline]
pub(crate) fn compute_abs<T: FixedNum>(v: Compute<T>) -> Compute<T> {
    if v.is_negative() { T::compute_negate(v) } else { v }
}

#[inline]
pub(crate) fn compute_abs_shr<T: FixedNum>(v: Compute<T>, shift: u32) -> Compute<T> {
    compute_abs::<T>(v).shr_bits(shift)
}

/// `2^shift` as a compute raw.
#[inline]
pub(crate) fn compute_quanta<T: FixedNum>(shift: u32) -> Compute<T> {
    Compute::<T>::one().shl_bits(shift)
}

/// Deflation threshold of the iterative decompositions: `2^-(3F/2)` of the
/// magnitude, at least `2^(F/2)` compute quanta.
#[inline]
pub(crate) fn compute_deflation_threshold<T: FixedNum>(magnitude: Compute<T>) -> Compute<T> {
    let f = T::FRAC_BITS;
    compute_abs_shr::<T>(magnitude, f * 3 / 2).max(compute_quanta::<T>(f / 2))
}

#[inline]
pub(crate) fn compute_noise_floor<T: FixedNum>() -> Compute<T> {
    compute_quanta::<T>(T::FRAC_BITS / 2)
}

#[inline]
pub(crate) fn compute_stagnation_threshold<T: FixedNum>(magnitude: Compute<T>) -> Compute<T> {
    let f = T::FRAC_BITS;
    compute_abs_shr::<T>(magnitude, f).max(compute_quanta::<T>(f / 2))
}

/// Significant bits of a compute value's magnitude.
#[inline]
pub(crate) fn compute_bit_length<T: FixedNum>(v: Compute<T>) -> u32 {
    v.magnitude_bit_length()
}

/// The shift that brings the largest of `values` up to the compute unit's
/// bit length (0 if any already reaches it).
pub(crate) fn scale_up_exponent<T: FixedNum>(values: &[Compute<T>]) -> u32 {
    let bits = values.iter().map(|&v| compute_bit_length::<T>(v)).max().unwrap_or(0);
    let unit_bits = 2 * T::FRAC_BITS;
    if bits == 0 || bits >= unit_bits { 0 } else { unit_bits - bits }
}

/// A compute value scaled by `2^-shift`, rounded once to the nearest storage
/// value (ties toward +infinity).
pub(crate) fn downscale_shifted_to_storage<T: FixedNum>(v: Compute<T>, shift: u32) -> Result<T::Storage, OverflowDetected> {
    if shift == 0 {
        return T::downscale_to_storage(v);
    }
    let f = T::FRAC_BITS;
    let half = compute_quanta::<T>(f + shift - 1);
    let q = T::compute_checked_add(v, half)?.shr_bits(f + shift);
    T::downscale_to_storage(q.shl_bits(f))
}

/// The convergence threshold of the iterative decompositions for a value of
/// this magnitude: `|magnitude| >> CONVERGENCE_SHIFT`, at least one quantum.
pub(crate) fn convergence_threshold<S: Storage, const F: u32>(magnitude: Fixed<S, F>) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let shifted = magnitude.abs().raw().shr_bits(<Fixed<S, F> as FixedNum>::CONVERGENCE_SHIFT);
    if shifted == S::zero() { Fixed::from_raw(S::one()) } else { Fixed::from_raw(shifted) }
}

// ============================================================================
// Rotations and reflections on compute-tier state
// ============================================================================

/// A Givens rotation held at the compute tier.
#[derive(Clone, Copy)]
pub(crate) struct Rotation<T: FixedNum> {
    cs: Compute<T>,
    sn: Compute<T>,
}

impl<T: FixedNum> Rotation<T> {
    #[inline]
    pub(crate) fn from_parts(cs: Compute<T>, sn: Compute<T>) -> Self {
        Rotation { cs, sn }
    }

    /// The rotation that zeroes `b` against `a`, from compute-tier values.
    pub(crate) fn zeroing_compute(a: Compute<T>, b: Compute<T>) -> Result<Self, OverflowDetected> {
        let one = T::make_compute_int(1);
        let zero = T::make_compute_int(0);
        if T::compute_is_zero(&b) {
            return Ok(Rotation { cs: one, sn: zero });
        }
        if T::compute_is_zero(&a) {
            let sn = if b.is_negative() { T::compute_negate(one) } else { one };
            return Ok(Rotation { cs: zero, sn });
        }
        if compute_abs::<T>(b) > compute_abs::<T>(a) {
            let tau = T::compute_divide(a, b)?;
            let inv = T::compute_divide(one, T::sqrt_at_compute_tier(T::compute_add(one, T::compute_multiply(tau, tau))))?;
            let sn = if b.is_negative() { T::compute_negate(inv) } else { inv };
            Ok(Rotation { cs: T::compute_multiply(sn, tau), sn })
        } else {
            let tau = T::compute_divide(b, a)?;
            let inv = T::compute_divide(one, T::sqrt_at_compute_tier(T::compute_add(one, T::compute_multiply(tau, tau))))?;
            let cs = if a.is_negative() { T::compute_negate(inv) } else { inv };
            Ok(Rotation { cs, sn: T::compute_multiply(cs, tau) })
        }
    }

    /// `(cs x + sn y, -sn x + cs y)`, each formed exactly and rounded once.
    pub(crate) fn apply_compute(&self, x: Compute<T>, y: Compute<T>) -> Result<(Compute<T>, Compute<T>), OverflowDetected> {
        let first = T::Storage::widen_product(self.cs, x).add_exact(T::Storage::widen_product(self.sn, y))?;
        let second = T::Storage::widen_product(T::compute_negate(self.sn), x).add_exact(T::Storage::widen_product(self.cs, y))?;
        Ok((narrow_product_to_compute::<T>(first)?, narrow_product_to_compute::<T>(second)?))
    }

    /// `cs x + sn y`, formed exactly and rounded once.
    pub(crate) fn combine_compute(&self, x: Compute<T>, y: Compute<T>) -> Result<Compute<T>, OverflowDetected> {
        narrow_product_to_compute::<T>(T::Storage::widen_product(self.cs, x).add_exact(T::Storage::widen_product(self.sn, y))?)
    }

    #[inline]
    pub(crate) fn cos_times_compute(&self, x: Compute<T>) -> Result<Compute<T>, OverflowDetected> {
        compute_product::<T>(self.cs, x)
    }

    #[inline]
    pub(crate) fn sin_times_compute(&self, x: Compute<T>) -> Result<Compute<T>, OverflowDetected> {
        compute_product::<T>(self.sn, x)
    }

    #[inline]
    pub(crate) fn neg_sin_times_compute(&self, x: Compute<T>) -> Result<Compute<T>, OverflowDetected> {
        compute_product::<T>(T::compute_negate(self.sn), x)
    }
}

/// The Householder vector of `x` at the compute tier, with `v . v`, after a
/// power-of-two normalisation of `x` to the compute unit's bit length (so
/// the squared norm neither overflows nor underflows). `None` for a zero
/// vector or when the reflection would be the identity.
pub(crate) fn householder_vector_compute<T: FixedNum>(
    x: &[Compute<T>],
) -> Result<Option<(Vec<Compute<T>>, Compute<T>)>, OverflowDetected> {
    let bits = x.iter().map(|&xi| compute_bit_length::<T>(xi)).max().unwrap_or(0);
    if bits == 0 {
        return Ok(None);
    }
    let unit_bits = 2 * T::FRAC_BITS;
    let scaled: Vec<Compute<T>>;
    let x = if bits < unit_bits {
        scaled = x.iter().map(|&xi| xi.shl_bits(unit_bits - bits)).collect();
        &scaled[..]
    } else if bits > unit_bits {
        scaled = x.iter().map(|&xi| xi.shr_bits(bits - unit_bits)).collect();
        &scaled[..]
    } else {
        x
    };
    let xx = narrow_product_to_compute::<T>(exact_dot_compute::<T>(x, x)?)?;
    if T::compute_is_zero(&xx) {
        return Ok(None);
    }
    let norm = T::sqrt_at_compute_tier(xx);
    let alpha = if x[0].is_negative() { norm } else { T::compute_negate(norm) };
    let mut v = x.to_vec();
    v[0] = T::compute_checked_add(x[0], T::compute_negate(alpha))?;
    let vv = narrow_product_to_compute::<T>(exact_dot_compute::<T>(&v, &v)?)?;
    if T::compute_is_zero(&vv) {
        return Ok(None);
    }
    Ok(Some((v, vv)))
}

/// `w -= 2 (v . w) / (v . v) v` at the compute tier: the dot exact, each
/// update one exact quotient rounded once.
pub(crate) fn reflect_compute<T: FixedNum>(
    w: &mut [Compute<T>], v: &[Compute<T>], v_dot_v: Compute<T>,
) -> Result<(), OverflowDetected> {
    let vw = narrow_product_to_compute::<T>(exact_dot_compute::<T>(v, w)?)?;
    if T::compute_is_zero(&vw) {
        return Ok(());
    }
    let two_vw = T::compute_checked_add(vw, vw)?;
    for (wk, vk) in w.iter_mut().zip(v) {
        let update = divide_to_compute_nearest::<T>(T::Storage::widen_product(two_vw, *vk), v_dot_v)?;
        *wk = T::compute_checked_add(*wk, T::compute_negate(update))?;
    }
    Ok(())
}

// ============================================================================
// Value-level forms used by the containers
// ============================================================================

/// A compute-tier accumulator rounded to the nearest storage value. Panics
/// if the result exceeds the storage tier's range, matching the infallible
/// imperative transcendentals.
#[inline]
pub(crate) fn round_to_storage<T: FixedNum>(acc: Compute<T>) -> T::Storage {
    T::downscale_to_storage(acc).expect("round_to_storage: result exceeds storage tier range")
}

/// The dot product of two slices of values: accumulated at the compute tier
/// and rounded to storage once (nearest, ties toward +infinity).
///
/// Panics if the slices have different lengths or the result leaves storage.
#[inline]
pub(crate) fn dot<S: Storage, const F: u32>(a: &[Fixed<S, F>], b: &[Fixed<S, F>]) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert_eq!(a.len(), b.len(), "compute_tier_dot: length mismatch");
    let acc = dot_acc::<Fixed<S, F>>(Fixed::raw_slice(a), Fixed::raw_slice(b));
    Fixed::from_raw(round_to_storage::<Fixed<S, F>>(acc))
}

/// `sqrt(sum a_i b_i)` with the sum and the root at the compute tier and one
/// rounding to storage. `Err(DomainError)` for a negative sum.
pub(crate) fn sqrt_dot<T: FixedNum>(a: &[T::Storage], b: &[T::Storage]) -> Result<T::Storage, OverflowDetected> {
    let acc = dot_acc::<T>(a, b);
    if acc.is_negative() {
        return Err(OverflowDetected::DomainError);
    }
    T::downscale_to_storage(T::sqrt_at_compute_tier(acc))
}

/// [`dot`] on raw slices.
#[inline]
pub(crate) fn dot_raw<T: FixedNum>(a: &[T::Storage], b: &[T::Storage]) -> T::Storage {
    round_to_storage::<T>(dot_acc::<T>(a, b))
}

/// A compute value scaled by `2^shift` (the caller has bounded the bit length).
#[inline]
pub(crate) fn compute_scale_up<T: FixedNum>(v: Compute<T>, shift: u32) -> Compute<T> {
    v.shl_bits(shift)
}

/// v^T M v exactly at 3F (the orient accumulator): each product of three
/// storage values is exact, and the sum is checked.
pub(crate) fn quadratic_form_exact<S: Storage, const F: u32>(v: &[S], m: &FixedMatrix<Fixed<S, F>>) -> Result<Orient<Fixed<S, F>>, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = v.len();
    assert!(m.is_square() && m.rows() == n, "quadratic_form: dimension mismatch");
    let mut sum = Orient::<Fixed<S, F>>::zero();
    for i in 0..n {
        let vi = v[i];
        let vv = vi.mul_wide(vi);
        sum = sum.add_exact(S::widen_product(m.get(i, i).raw().widen(), vv))?;
        for j in (i + 1)..n {
            // W+1 bits at most: exact at the compute width (2W).
            let pair = m.get(i, j).raw().widen() + m.get(j, i).raw().widen();
            let vv = vi.mul_wide(v[j]);
            sum = sum.add_exact(S::widen_product(pair, vv))?;
        }
    }
    Ok(sum)
}

#[cfg(test)]
mod narrowing_tests {
    //! The three narrowings and the storage-nearest division on the exact
    //! quadratic form, on every format in one build (these were the tests of
    //! the former `wide_acc` module, which was bound to the 0.6.x build's profile).
    use super::*;
    use crate::fixed_point::{I256, I512};

    fn qf<S: Storage, const F: u32>(v: &[Fixed<S, F>], m: &FixedMatrix<Fixed<S, F>>) -> Orient<Fixed<S, F>>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        quadratic_form_exact(Fixed::raw_slice(v), m).unwrap()
    }

    fn check<S: Storage, const F: u32>()
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        let one = Fixed::<S, F>::one();
        let s_one = S::one();
        let zero = S::ZERO;
        // a storage value widened to the accumulator, times 2^(2F), narrows
        // back under all three rules with nothing discarded
        let third = (one / Fixed::<S, F>::from_int(3)).raw();
        let mut m = FixedMatrix::<Fixed<S, F>>::new(1, 1);
        m.set(0, 0, Fixed::from_raw(third));
        let a = qf(&[one], &m);
        assert_eq!(narrow_triple_floor::<Fixed<S, F>>(a).unwrap(), third);
        assert_eq!(narrow_triple_ceil::<Fixed<S, F>>(a).unwrap(), third);
        assert_eq!(narrow_triple_nearest::<Fixed<S, F>>(a).unwrap(), third);
        // v = [1/2], m = [[2 ulp]]: exactly half an ulp, a tie; floor 0, ceil
        // 1, nearest toward +infinity 1; the negative twin -1, 0, 0
        let half = one / Fixed::<S, F>::from_int(2);
        let two_ulp = Fixed::<S, F>::from_raw(s_one + s_one);
        m.set(0, 0, two_ulp);
        let a = qf(&[half], &m);
        assert_eq!(narrow_triple_floor::<Fixed<S, F>>(a).unwrap(), zero);
        assert_eq!(narrow_triple_ceil::<Fixed<S, F>>(a).unwrap(), s_one);
        assert_eq!(narrow_triple_nearest::<Fixed<S, F>>(a).unwrap(), s_one);
        m.set(0, 0, -two_ulp);
        let a = qf(&[half], &m);
        assert_eq!(narrow_triple_floor::<Fixed<S, F>>(a).unwrap(), -s_one);
        assert_eq!(narrow_triple_ceil::<Fixed<S, F>>(a).unwrap(), zero);
        assert_eq!(narrow_triple_nearest::<Fixed<S, F>>(a).unwrap(), zero);
        // dividing by the compute-tier one is narrowing by 2^(2F): the
        // quotient rounding equals narrow_triple_nearest, ties included
        let one_c = Fixed::<S, F>::make_compute_int(1);
        let mut state: u64 = 0x5DEE_CE66_D1CE_4E5B;
        let mut cases = 0;
        for k in -30i32..=30 {
            for tweak in [0i32, 1, 2, 3] {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let ulps = (state >> 40) as i32 % 16 + k * 3 + tweak;
                m.set(0, 0, Fixed::<S, F>::from_raw(s_one) * Fixed::<S, F>::from_int(ulps));
                let a = qf(&[half], &m);
                assert_eq!(
                    divide_to_storage_nearest::<Fixed<S, F>>(a, one_c),
                    narrow_triple_nearest::<Fixed<S, F>>(a),
                    "{ulps}"
                );
                cases += 1;
            }
        }
        assert!(cases > 200);
        // the pairing (m_ij + m_ji) v_i v_j assumes no symmetry: an
        // antisymmetric off-diagonal contributes nothing, a one-sided one its
        // full product
        let v = [Fixed::<S, F>::from_int(3), Fixed::<S, F>::from_int(5)];
        let mut m = FixedMatrix::<Fixed<S, F>>::new(2, 2);
        m.set(0, 1, Fixed::from_int(7));
        m.set(1, 0, Fixed::from_int(-7));
        assert_eq!(narrow_triple_nearest::<Fixed<S, F>>(qf(&v, &m)).unwrap(), zero);
        m.set(1, 0, Fixed::from_raw(zero));
        assert_eq!(narrow_triple_nearest::<Fixed<S, F>>(qf(&v, &m)).unwrap(), Fixed::<S, F>::from_int(105).raw());
        // a result beyond the storage tier is a typed overflow, never a wrap
        let big = Fixed::<S, F>::from_raw(s_one.shl_bits(S::BITS - 2));
        let mut m = FixedMatrix::<Fixed<S, F>>::new(1, 1);
        m.set(0, 0, big);
        let a = qf(&[big], &m);
        assert_eq!(narrow_triple_floor::<Fixed<S, F>>(a), Err(OverflowDetected::TierOverflow));
        assert_eq!(narrow_triple_ceil::<Fixed<S, F>>(a), Err(OverflowDetected::TierOverflow));
        assert_eq!(narrow_triple_nearest::<Fixed<S, F>>(a), Err(OverflowDetected::TierOverflow));
    }

    #[test]
    fn narrowings_on_every_format() {
        check::<i32, 10>();
        check::<i32, 16>();
        check::<i64, 32>();
        check::<i128, 64>();
        check::<I256, 128>();
        check::<I512, 256>();
    }
}
