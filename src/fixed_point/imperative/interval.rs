//! Certified interval arithmetic: enclosures that are sound by construction.
//!
//! An [`Interval<Fixed<S, F>>`] is a pair of storage-tier endpoints `[lo, hi]` carrying the
//! guarantee that the exact mathematical result of the operation that produced
//! it lies within them. Soundness rests on two mechanisms and nothing else:
//!
//! 1. Every product of two storage values is computed EXACTLY at the compute
//!    tier (storage width doubled), and sums of such products accumulate
//!    exactly with checked addition.
//! 2. Every narrowing back to storage rounds the lower endpoint toward
//!    negative infinity and the upper endpoint toward positive infinity.
//!
//! Compound operations ([`Interval::<Fixed<S, F>>::dot`], [`Interval::<Fixed<S, F>>::quadratic_form`])
//! accumulate exactly and narrow once. That is what keeps the enclosure
//! tight: `dot` and `quadratic_form` are `[floor, ceil]` of one exact value,
//! at most 1 ulp wide. The quadratic form's earlier two-narrowing form
//! measured 5 to 64 ulp for 23 dimensions at Q64.64, itself 12 to 24 times
//! tighter than narrowing after every elementary operation. Width is set by
//! the number of narrowings, not by cancellation, because cancellation inside
//! an exact accumulator costs nothing.
//!
//! **What may carry the word certified.** `+`, `-`, `*`, `/` are sound by the
//! standard endpoint argument. [`Interval::<Fixed<S, F>>::sqrt`] is certified a posteriori:
//! the returned floor `k` satisfies `k^2 <= n < (k+1)^2` in exact integers at
//! the compute tier. The candidate comes from an integer Newton iteration at
//! the compute tier, not from the transcendental engine, and the check is
//! bounded: it can fail loudly, never loop. No transcendental is
//! provided. Their accuracy is validated against reference values at chosen
//! points, which is evidence rather than a proof of a bound over the domain,
//! and an interval widened by a measured error would not be an enclosure. Each
//! transcendental joins this type only once its bound is proven.
//!
//! Endpoint arithmetic never wraps. Overflow of the storage tier is a typed
//! `TierOverflow`, because an enclosure that wraps is worse than no enclosure.
//! Infallible operators panic on the same condition; the `try_*` twins return
//! the error.
//!
//! **Where this sits.** An imperative-tier type beside `Fixed<S, F>`,
//! `FixedVector<Fixed<S, F>>`, `FixedMatrix<Fixed<S, F>>` and `fused`: binary Q-format, fixed at the
//! profile's storage tier, intermediates at the compute tier. It is not a
//! FASC domain (no `StackValue` variant, no routing) and not UGOD-tiered
//! (the imperative layer has no wider storage tier to promote into); on
//! storage overflow it fails typed where the scalar operators wrap. See
//! `docs/design/CERTIFIED_INTERVALS.md` for the reasoning.
//!
//! Non-goals: IEEE 1788 decorations, reverse operations, midpoint-radius and
//! affine forms. The measured widths did not call for any of them.

use std::ops::{Add, Div, Mul, Neg, Sub};

use super::{FixedMatrix, FixedVector};
use super::kernels;
use crate::fixed_point::fixed_num::{Compute, Fixed, FixedNum};
use crate::fixed_point::width::{Storage, WideInt};
use crate::fixed_point::core_types::errors::OverflowDetected;


// ============================================================================
// Storage-tier checked helpers (never wrap)
// ============================================================================

#[inline]
fn st_add<S: Storage, const F: u32>(a: S, b: S) -> Result<S, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    a.checked_add_wide(b).ok_or(OverflowDetected::TierOverflow)
}

#[inline]
fn st_sub<S: Storage, const F: u32>(a: S, b: S) -> Result<S, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    a.checked_sub_wide(b).ok_or(OverflowDetected::TierOverflow)
}

#[inline]
fn st_neg<S: Storage, const F: u32>(a: S) -> Result<S, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    a.checked_neg_wide().ok_or(OverflowDetected::TierOverflow)
}

#[inline]
fn compute_zero<S: Storage, const F: u32>() -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::<S, F>::upscale_to_compute(Fixed::ZERO.raw())
}

/// Directed quotient of two storage values: `(floor(a/b), ceil(a/b))` at the
/// storage scale, computed from the truncating integer quotient and its
/// remainder. Mirrors `fixed_divide`'s widening per profile.
#[inline]
fn directed_divide<S: Storage, const F: u32>(a: S, b: S) -> Result<(S, S), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if b == S::ZERO { return Err(OverflowDetected::DivisionByZero); }
    let num = a.widen().shl_bits(F);
    let den = b.widen();
    // truncating quotient and remainder; the directed pair from the remainder's sign
    let (q, rem) = num.divmod(den);
    let one = <Compute<Fixed<S, F>> as WideInt>::one();
    let (f, c) = if rem == <Compute<Fixed<S, F>> as WideInt>::ZERO {
        (q, q)
    } else if num.is_negative() == den.is_negative() {
        (q, q + one)
    } else {
        (q - one, q)
    };
    match (S::narrow(f), S::narrow(c)) {
        (Some(f), Some(c)) => Ok((f, c)),
        _ => Err(OverflowDetected::TierOverflow),
    }
}

/// Certified square root of a storage value: `(floor, ceil)` of `sqrt(x)` at
/// the storage scale, verified by exact integer comparison.
///
/// In raw units `sqrt(x)_raw = isqrt(x_raw << FRAC_BITS)`, and `n = x_raw <<
/// FRAC_BITS` is exactly the compute-tier representation of `x`. The candidate
/// is an integer Newton iteration on `n` at the compute tier, seeded from
/// above by a power of two, which converges to `floor(sqrt(n))` in O(log bits)
/// steps independent of any transcendental engine. The candidate is then
/// verified: `k^2 <= n < (k+1)^2` in exact compute-tier integers, with at most
/// two corrective steps each way; if the certificate still does not hold the
/// function panics rather than loop or return an unverified value.
#[inline]
fn certified_sqrt<S: Storage, const F: u32>(x: S) -> Result<(S, S), OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let n = Fixed::<S, F>::upscale_to_compute(x);
    let mut k = compute_narrow_integer(isqrt_compute(n))?;
    let unit = S::one();
    let mut steps = 0u32;
    while S::mul_wide(k, k) > n {
        k = st_sub(k, unit)?;
        steps += 1;
        assert!(steps <= 2, "certified_sqrt: candidate failed the certificate downward");
    }
    loop {
        let next = st_add(k, unit)?;
        if S::mul_wide(next, next) <= n {
            k = next;
            steps += 1;
            assert!(steps <= 2, "certified_sqrt: candidate failed the certificate upward");
        } else {
            break;
        }
    }
    let floor = k;
    let ceil = if S::mul_wide(k, k) == n { k } else { st_add(k, unit)? };
    Ok((floor, ceil))
}

/// `floor(sqrt(n))` for a non-negative compute-tier integer, by Newton from
/// above: seed `2^ceil(bits/2) >= sqrt(n)`, iterate `k = (k + n/k) / 2` while
/// it decreases. Plain integer arithmetic on the compute type; no Q-format
/// scaling is involved.
#[inline]
fn isqrt_compute<S: Storage, const F: u32>(n: Compute<Fixed<S, F>>) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    debug_assert!(!n.is_negative());
    n.isqrt()
}

/// Significant bits of a non-negative wide compute-tier value.

/// A compute-tier integer that is known to fit the storage type, as storage.
/// This is a plain integer narrowing, not a Q-format downscale.
#[inline]
fn compute_narrow_integer<S: Storage, const F: u32>(v: Compute<Fixed<S, F>>) -> Result<S, OverflowDetected>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    S::narrow(v).ok_or(OverflowDetected::TierOverflow)
}

/// The raw integer 1 (one ulp) in the storage type.

// ============================================================================
// Interval<Fixed<S, F>>
// ============================================================================

/// A certified enclosure `[lo, hi]` of a real value, `lo <= hi`.
///
/// Endpoints are storage-tier `Fixed<S, F>` values. Every operation on this
/// type returns an interval that contains the exact mathematical result for
/// every choice of operands within the input intervals. See the module
/// documentation for what that guarantee rests on and what it excludes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Interval<T> {
    lo: T,
    hi: T,
}

impl<S: Storage, const F: u32> Interval<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// The degenerate interval `[x, x]`.
    #[inline]
    pub fn point(x: Fixed<S, F>) -> Self {
        Interval::<Fixed<S, F>> { lo: x, hi: x }
    }

    /// `[lo, hi]`. Panics if `lo > hi`.
    #[inline]
    pub fn new(lo: Fixed<S, F>, hi: Fixed<S, F>) -> Self {
        Self::try_new(lo, hi).expect("Interval::<Fixed<S, F>>::new: lo > hi")
    }

    /// `[lo, hi]`, or `Err(InvalidInput)` if `lo > hi`.
    #[inline]
    pub fn try_new(lo: Fixed<S, F>, hi: Fixed<S, F>) -> Result<Self, OverflowDetected> {
        if lo > hi {
            return Err(OverflowDetected::InvalidInput);
        }
        Ok(Interval::<Fixed<S, F>> { lo, hi })
    }

    #[inline]
    fn from_raw(lo: S, hi: S) -> Self {
        let iv = Interval::<Fixed<S, F>> { lo: Fixed::from_raw(lo), hi: Fixed::from_raw(hi) };
        debug_assert!(iv.lo <= iv.hi, "Interval<Fixed<S, F>> invariant violated: lo > hi");
        iv
    }

    /// Lower endpoint.
    #[inline]
    pub fn lo(self) -> Fixed<S, F> { self.lo }

    /// Upper endpoint.
    #[inline]
    pub fn hi(self) -> Fixed<S, F> { self.hi }

    /// `hi - lo`. Panics if the width itself does not fit the storage tier.
    #[inline]
    pub fn width(self) -> Fixed<S, F> {
        Fixed::from_raw(st_sub(self.hi.raw(), self.lo.raw()).expect("Interval::<Fixed<S, F>>::width: overflow"))
    }

    /// `lo == hi`.
    #[inline]
    pub fn is_point(self) -> bool { self.lo == self.hi }

    /// `lo <= x <= hi`.
    #[inline]
    pub fn contains(self, x: Fixed<S, F>) -> bool { self.lo <= x && x <= self.hi }

    /// `lo <= 0 <= hi`.
    #[inline]
    pub fn contains_zero(self) -> bool { self.contains(Fixed::ZERO) }

    /// `lo > 0`: every value in the interval is positive.
    #[inline]
    pub fn is_certainly_positive(self) -> bool { self.lo > Fixed::ZERO }

    /// `hi < 0`: every value in the interval is negative.
    #[inline]
    pub fn is_certainly_negative(self) -> bool { self.hi < Fixed::ZERO }

    // ------------------------------------------------------------------
    // Arithmetic, fallible
    // ------------------------------------------------------------------

    /// `[a.lo + b.lo, a.hi + b.hi]`. Storage addition is exact when it fits.
    pub fn try_add(self, rhs: Self) -> Result<Self, OverflowDetected> {
        Ok(Self::from_raw(st_add(self.lo.raw(), rhs.lo.raw())?, st_add(self.hi.raw(), rhs.hi.raw())?))
    }

    /// `[a.lo - b.hi, a.hi - b.lo]`.
    pub fn try_sub(self, rhs: Self) -> Result<Self, OverflowDetected> {
        Ok(Self::from_raw(st_sub(self.lo.raw(), rhs.hi.raw())?, st_sub(self.hi.raw(), rhs.lo.raw())?))
    }

    /// `[-hi, -lo]`.
    pub fn try_neg(self) -> Result<Self, OverflowDetected> {
        Ok(Self::from_raw(st_neg(self.hi.raw())?, st_neg(self.lo.raw())?))
    }

    /// Product: exact corner products at the compute tier, narrowed once.
    ///
    /// The four corner products are formed exactly, the extremes are chosen
    /// exactly, and each is narrowed once: floor for the minimum, ceil for
    /// the maximum.
    pub fn try_mul(self, rhs: Self) -> Result<Self, OverflowDetected> {
        let p = [
            S::mul_wide(self.lo.raw(), rhs.lo.raw()),
            S::mul_wide(self.lo.raw(), rhs.hi.raw()),
            S::mul_wide(self.hi.raw(), rhs.lo.raw()),
            S::mul_wide(self.hi.raw(), rhs.hi.raw()),
        ];
        let mut mn = p[0];
        let mut mx = p[0];
        for q in &p[1..] {
            if *q < mn { mn = *q; }
            if *q > mx { mx = *q; }
        }
        Ok(Self::from_raw(Fixed::<S, F>::downscale_to_storage_floor(mn)?, Fixed::<S, F>::downscale_to_storage_ceil(mx)?))
    }

    /// Quotient; `Err(DivisionByZero)` if the divisor interval contains zero.
    ///
    /// No enclosure exists for a quotient whose divisor may vanish. Otherwise
    /// the quotient is monotone in each operand on the divisor's side of
    /// zero, so the extremes are among the four corner quotients, each taken
    /// with directed rounding.
    pub fn try_div(self, rhs: Self) -> Result<Self, OverflowDetected> {
        if rhs.contains_zero() {
            return Err(OverflowDetected::DivisionByZero);
        }
        let corners = [
            directed_divide(self.lo.raw(), rhs.lo.raw())?,
            directed_divide(self.lo.raw(), rhs.hi.raw())?,
            directed_divide(self.hi.raw(), rhs.lo.raw())?,
            directed_divide(self.hi.raw(), rhs.hi.raw())?,
        ];
        let mut lo = corners[0].0;
        let mut hi = corners[0].1;
        for (f, c) in &corners[1..] {
            if *f < lo { lo = *f; }
            if *c > hi { hi = *c; }
        }
        Ok(Self::from_raw(lo, hi))
    }

    /// Certified square root. `Err(DomainError)` if `lo < 0`.
    ///
    /// `sqrt` is monotone on `[0, inf)`, so the result is
    /// `[floor(sqrt(lo)), ceil(sqrt(hi))]`, each endpoint verified by the
    /// exact integer certificate `k^2 <= n < (k+1)^2` at the compute tier.
    pub fn try_sqrt(self) -> Result<Self, OverflowDetected> {
        if self.lo < Fixed::ZERO {
            return Err(OverflowDetected::DomainError);
        }
        let (lo, _) = certified_sqrt(self.lo.raw())?;
        let (_, hi) = certified_sqrt(self.hi.raw())?;
        Ok(Self::from_raw(lo, hi))
    }

    /// Certified dot product of two point vectors, with one narrowing.
    ///
    /// Every product is exact at the compute tier and the sum is exact with
    /// checked addition; the single narrowing happens at the end.
    ///
    /// Panics if the slices differ in length.
    pub fn try_dot(a: &[Fixed<S, F>], b: &[Fixed<S, F>]) -> Result<Self, OverflowDetected> {
        assert_eq!(a.len(), b.len(), "Interval::<Fixed<S, F>>::dot: length mismatch");
        let mut acc = compute_zero::<S, F>();
        for i in 0..a.len() {
            acc = Fixed::<S, F>::compute_checked_add(acc, S::mul_wide(a[i].raw(), b[i].raw()))?;
        }
        Ok(Self::from_raw(Fixed::<S, F>::downscale_to_storage_floor(acc)?, Fixed::<S, F>::downscale_to_storage_ceil(acc)?))
    }

    /// Certified dot product of two interval vectors, with one narrowing.
    ///
    /// For each term the four exact corner products are formed at the
    /// compute tier and the smallest is added to the lower accumulator, the
    /// largest to the upper; both sums are exact with checked addition and
    /// each is narrowed once. This is what a factorisation needs once its
    /// entries are themselves intervals.
    ///
    /// Panics if the slices differ in length.
    pub fn try_dot_intervals(a: &[Interval<Fixed<S, F>>], b: &[Interval<Fixed<S, F>>]) -> Result<Self, OverflowDetected> {
        assert_eq!(a.len(), b.len(), "Interval::<Fixed<S, F>>::dot_intervals: length mismatch");
        let mut lo_acc = compute_zero::<S, F>();
        let mut hi_acc = compute_zero::<S, F>();
        for i in 0..a.len() {
            let p = [
                S::mul_wide(a[i].lo.raw(), b[i].lo.raw()),
                S::mul_wide(a[i].lo.raw(), b[i].hi.raw()),
                S::mul_wide(a[i].hi.raw(), b[i].lo.raw()),
                S::mul_wide(a[i].hi.raw(), b[i].hi.raw()),
            ];
            let mut mn = p[0];
            let mut mx = p[0];
            for q in &p[1..] {
                if *q < mn { mn = *q; }
                if *q > mx { mx = *q; }
            }
            lo_acc = Fixed::<S, F>::compute_checked_add(lo_acc, mn)?;
            hi_acc = Fixed::<S, F>::compute_checked_add(hi_acc, mx)?;
        }
        Ok(Self::from_raw(Fixed::<S, F>::downscale_to_storage_floor(lo_acc)?, Fixed::<S, F>::downscale_to_storage_ceil(hi_acc)?))
    }

    /// Certified quadratic form `v^T M v` for point inputs, with one narrowing.
    ///
    /// Every term `v_i m_ij v_j` is an exact triple product on the profile's
    /// widest accumulator (the orient3d accumulator: `3W+3` bits and up) and
    /// the sum is exact, so the result is `[floor, ceil]` of ONE exact value:
    /// the width is at most 1 ulp, and exactly 0 when the value is
    /// representable. `fused::quadratic_form` rounds the same exact value to
    /// nearest, so the scalar always lies inside this enclosure. The earlier
    /// two-narrowing form (each `(M v)_i` narrowed, then the outer sum) measured
    /// a mean of 3.2 ulp at 7 dimensions and 5 to 64 ulp at 23.
    ///
    /// Panics if `m` is not square or its size differs from `v`.
    pub fn try_quadratic_form(v: &FixedVector<Fixed<S, F>>, m: &FixedMatrix<Fixed<S, F>>) -> Result<Self, OverflowDetected> {
        let acc = kernels::quadratic_form_exact(Fixed::raw_slice(v.as_slice()), m)?;
        Ok(Self::from_raw(kernels::narrow_triple_floor::<Fixed<S, F>>(acc)?, kernels::narrow_triple_ceil::<Fixed<S, F>>(acc)?))
    }

    // ------------------------------------------------------------------
    // Arithmetic, infallible (panics where the try_ twin errs)
    // ------------------------------------------------------------------

    /// Certified square root; panics on a negative lower endpoint or overflow.
    pub fn sqrt(self) -> Self {
        self.try_sqrt().expect("Interval::<Fixed<S, F>>::sqrt: domain error or overflow")
    }

    /// Certified dot product; panics on overflow.
    pub fn dot(a: &[Fixed<S, F>], b: &[Fixed<S, F>]) -> Self {
        Self::try_dot(a, b).expect("Interval::<Fixed<S, F>>::dot: overflow")
    }

    /// Certified quadratic form; panics on overflow.
    pub fn quadratic_form(v: &FixedVector<Fixed<S, F>>, m: &FixedMatrix<Fixed<S, F>>) -> Self {
        Self::try_quadratic_form(v, m).expect("Interval::<Fixed<S, F>>::quadratic_form: overflow")
    }

    /// Certified dot product of interval vectors; panics on overflow.
    pub fn dot_intervals(a: &[Interval<Fixed<S, F>>], b: &[Interval<Fixed<S, F>>]) -> Self {
        Self::try_dot_intervals(a, b).expect("Interval::<Fixed<S, F>>::dot_intervals: overflow")
    }
}

impl<S: Storage, const F: u32> Add for Interval<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self { self.try_add(rhs).expect("Interval<Fixed<S, F>>: addition overflow") }
}

impl<S: Storage, const F: u32> Sub for Interval<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self { self.try_sub(rhs).expect("Interval<Fixed<S, F>>: subtraction overflow") }
}

impl<S: Storage, const F: u32> Mul for Interval<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self { self.try_mul(rhs).expect("Interval<Fixed<S, F>>: multiplication overflow") }
}

impl<S: Storage, const F: u32> Div for Interval<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self { self.try_div(rhs).expect("Interval<Fixed<S, F>>: division by an interval containing zero, or overflow") }
}

impl<S: Storage, const F: u32> Neg for Interval<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { self.try_neg().expect("Interval<Fixed<S, F>>: negation overflow") }
}

impl<S: Storage, const F: u32> From<Fixed<S, F>> for Interval<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    #[inline]
    fn from(x: Fixed<S, F>) -> Self { Interval::<Fixed<S, F>>::point(x) }
}
