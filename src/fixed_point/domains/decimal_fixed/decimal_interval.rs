//! Certified interval arithmetic over the decimal domain.
//!
//! [`DecimalInterval`] is the decimal counterpart of the binary
//! `fixed_point::Interval`. Endpoints are `DecimalFixed<DECIMALS>` values;
//! products are formed exactly at twice the decimal places in the
//! decimal-domain `D256`; and the single narrowing back to `DECIMALS` places
//! rounds the lower endpoint toward negative infinity and the upper toward
//! positive infinity. Where the binary type narrows by a power of two with a
//! shift, this one narrows by `10^DECIMALS` from the truncating quotient and
//! its remainder. Nothing else about the soundness argument differs; see
//! `docs/design/CERTIFIED_INTERVALS.md`.
//!
//! Profile-independent: `DecimalFixed` is an `i128` scaled by `10^DECIMALS`
//! on every profile, so there is one code path here rather than five.
//!
//! **What may carry the word certified.** `+`, `-`, `*`, `/` by the standard
//! endpoint argument. [`DecimalInterval::sqrt`] a posteriori: the returned
//! floor `k` satisfies `k^2 <= x * 10^DECIMALS < (k+1)^2` in exact `D256`
//! integers. The candidate comes from an integer Newton iteration; the
//! verification loop is the certificate. No transcendental is provided, for
//! the reason given in the binary module: measured accuracy at test points is
//! not a proven bound.
//!
//! Endpoint arithmetic never wraps and never saturates. The scalar
//! `DecimalFixed` operators saturate on overflow and on division by zero;
//! this type returns a typed `TierOverflow` or `DivisionByZero` instead,
//! because an enclosure that saturates is not an enclosure.
//!
//! No `quadratic_form` here: there is no decimal matrix type to take it over.
//! `dot` covers the sums of products that decimal consumers compute.

use std::ops::{Add, Div, Mul, Neg, Sub};

use super::d256::{divmod_d256_by_d256, mul_i128_to_d256, D256};
use super::decimal_fixed::DecimalFixed;
use crate::fixed_point::core_types::errors::OverflowDetected;

// ============================================================================
// i128 storage helpers (never wrap, never saturate)
// ============================================================================

#[inline]
fn st_add(a: i128, b: i128) -> Result<i128, OverflowDetected> {
    a.checked_add(b).ok_or(OverflowDetected::TierOverflow)
}

#[inline]
fn st_sub(a: i128, b: i128) -> Result<i128, OverflowDetected> {
    a.checked_sub(b).ok_or(OverflowDetected::TierOverflow)
}

#[inline]
fn st_neg(a: i128) -> Result<i128, OverflowDetected> {
    a.checked_neg().ok_or(OverflowDetected::TierOverflow)
}

// ============================================================================
// D256 helpers
// ============================================================================

/// `D256` addition with overflow detection: operands of one sign whose sum
/// has the other sign have left 256 bits. `D256::Add` itself carries silently.
#[inline]
fn d256_checked_add(a: D256, b: D256) -> Result<D256, OverflowDetected> {
    let sum = a + b;
    if a.is_negative() == b.is_negative() && sum.is_negative() != a.is_negative() {
        Err(OverflowDetected::TierOverflow)
    } else {
        Ok(sum)
    }
}

#[inline]
fn d256_to_i128(v: D256) -> Result<i128, OverflowDetected> {
    if v.fits_in_i128() { Ok(v.as_i128()) } else { Err(OverflowDetected::TierOverflow) }
}

/// Number of significant bits of a non-negative `D256`.
#[inline]
fn bit_length(v: D256) -> u32 {
    for i in (0..4).rev() {
        if v.words[i] != 0 {
            return i as u32 * 64 + (64 - v.words[i].leading_zeros());
        }
    }
    0
}

/// `2^e` as a `D256`, for `e <= 127`.
#[inline]
fn pow2(e: u32) -> D256 {
    debug_assert!(e <= 127);
    let v = 1u128 << e;
    D256::from_words([v as u64, (v >> 64) as u64, 0, 0])
}

/// `(floor, ceil)` of `p / 10^DECIMALS` for an exact product `p` at
/// `2 * DECIMALS` places, as i128 storage. The scale is positive, so the sign
/// of the exact quotient is the sign of `p`; the remainder's sign convention
/// is never consulted, only whether it is zero.
#[inline]
fn narrow<const DECIMALS: u8>(p: D256) -> Result<(i128, i128), OverflowDetected> {
    let scale = D256::from_i128(DecimalFixed::<DECIMALS>::SCALE);
    let (q, r) = divmod_d256_by_d256(p, scale);
    let one = D256::from_i128(1);
    let (f, c) = if r.is_zero() {
        (q, q)
    } else if p.is_negative() {
        (q - one, q)
    } else {
        (q, q + one)
    };
    Ok((d256_to_i128(f)?, d256_to_i128(c)?))
}

/// `(floor, ceil)` of `a / b` at `DECIMALS` places: `a * 10^DECIMALS` is
/// formed exactly in `D256`, divided with remainder, and the direction is
/// decided by the sign of the exact quotient and whether the remainder is
/// zero. `divmod_d256_by_i128` saturates when the quotient leaves i128, so
/// the 256-bit divide is used and the fit is checked here.
#[inline]
fn directed_divide<const DECIMALS: u8>(a: i128, b: i128) -> Result<(i128, i128), OverflowDetected> {
    if b == 0 {
        return Err(OverflowDetected::DivisionByZero);
    }
    let num = mul_i128_to_d256(a, DecimalFixed::<DECIMALS>::SCALE);
    let (q, r) = divmod_d256_by_d256(num, D256::from_i128(b));
    let one = D256::from_i128(1);
    let (f, c) = if r.is_zero() {
        (q, q)
    } else if (a < 0) == (b < 0) {
        (q, q + one)
    } else {
        (q - one, q)
    };
    Ok((d256_to_i128(f)?, d256_to_i128(c)?))
}

/// Certified square root of a non-negative storage value: `(floor, ceil)` of
/// `sqrt(x)` at `DECIMALS` places, verified by exact integer comparison.
///
/// In raw units `sqrt(x)_raw = isqrt(x_raw * 10^DECIMALS)`, and that product
/// `n` is exact in `D256`. An integer Newton iteration from above (seed
/// `2^ceil(bits/2) >= sqrt(n)`, non-increasing until it stops) produces the
/// candidate; the candidate is then moved until `k^2 <= n < (k+1)^2` holds.
/// The loop is what certifies the result. `k < 2^127` because
/// `n < 2^127 * 10^38 < 2^254`, so `k` and `k + 1` fit i128 and `k^2` is an
/// exact `mul_i128_to_d256`.
fn certified_sqrt<const DECIMALS: u8>(x: i128) -> Result<(i128, i128), OverflowDetected> {
    debug_assert!(x >= 0);
    let n = mul_i128_to_d256(x, DecimalFixed::<DECIMALS>::SCALE);
    if n.is_zero() {
        return Ok((0, 0));
    }
    let two = D256::from_i128(2);
    let mut k = pow2((bit_length(n) + 1) / 2);
    loop {
        let (q, _) = divmod_d256_by_d256(n, k);
        let next = (k + q) / two;
        if next >= k {
            break;
        }
        k = next;
    }
    let mut k = d256_to_i128(k)?;
    let mut steps = 0u32;
    while mul_i128_to_d256(k, k) > n {
        k -= 1;
        steps += 1;
    }
    loop {
        let next = st_add(k, 1)?;
        if mul_i128_to_d256(next, next) <= n {
            k = next;
            steps += 1;
        } else {
            break;
        }
    }
    debug_assert!(steps <= 2, "certified_sqrt: Newton candidate was {steps} units off");
    let ceil = if mul_i128_to_d256(k, k) == n { k } else { st_add(k, 1)? };
    Ok((k, ceil))
}

// ============================================================================
// DecimalInterval
// ============================================================================

/// A certified enclosure `[lo, hi]` of a real value in the decimal domain,
/// `lo <= hi`.
///
/// Endpoints are `DecimalFixed<DECIMALS>` values. Every operation returns an
/// interval containing the exact mathematical result for every choice of
/// operands within the input intervals. See the module documentation for the
/// basis of that guarantee and its boundary.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DecimalInterval<const DECIMALS: u8> {
    lo: DecimalFixed<DECIMALS>,
    hi: DecimalFixed<DECIMALS>,
}

impl<const DECIMALS: u8> DecimalInterval<DECIMALS> {
    /// The degenerate interval `[x, x]`.
    #[inline]
    pub fn point(x: DecimalFixed<DECIMALS>) -> Self {
        DecimalInterval { lo: x, hi: x }
    }

    /// `[lo, hi]`. Panics if `lo > hi`.
    #[inline]
    pub fn new(lo: DecimalFixed<DECIMALS>, hi: DecimalFixed<DECIMALS>) -> Self {
        Self::try_new(lo, hi).expect("DecimalInterval::new: lo > hi")
    }

    /// `[lo, hi]`, or `Err(InvalidInput)` if `lo > hi`.
    #[inline]
    pub fn try_new(lo: DecimalFixed<DECIMALS>, hi: DecimalFixed<DECIMALS>) -> Result<Self, OverflowDetected> {
        if lo > hi {
            return Err(OverflowDetected::InvalidInput);
        }
        Ok(DecimalInterval { lo, hi })
    }

    #[inline]
    fn from_raw(lo: i128, hi: i128) -> Self {
        debug_assert!(lo <= hi, "DecimalInterval invariant violated: lo > hi");
        DecimalInterval { lo: DecimalFixed::from_raw(lo), hi: DecimalFixed::from_raw(hi) }
    }

    /// Lower endpoint.
    #[inline]
    pub fn lo(self) -> DecimalFixed<DECIMALS> { self.lo }

    /// Upper endpoint.
    #[inline]
    pub fn hi(self) -> DecimalFixed<DECIMALS> { self.hi }

    /// `hi - lo`. Panics if the width itself does not fit i128.
    #[inline]
    pub fn width(self) -> DecimalFixed<DECIMALS> {
        DecimalFixed::from_raw(st_sub(self.hi.raw_value(), self.lo.raw_value()).expect("DecimalInterval::width: overflow"))
    }

    /// `lo == hi`.
    #[inline]
    pub fn is_point(self) -> bool { self.lo == self.hi }

    /// `lo <= x <= hi`.
    #[inline]
    pub fn contains(self, x: DecimalFixed<DECIMALS>) -> bool { self.lo <= x && x <= self.hi }

    /// `lo <= 0 <= hi`.
    #[inline]
    pub fn contains_zero(self) -> bool { self.contains(DecimalFixed::ZERO) }

    /// `lo > 0`: every value in the interval is positive.
    #[inline]
    pub fn is_certainly_positive(self) -> bool { self.lo > DecimalFixed::ZERO }

    /// `hi < 0`: every value in the interval is negative.
    #[inline]
    pub fn is_certainly_negative(self) -> bool { self.hi < DecimalFixed::ZERO }

    // ------------------------------------------------------------------
    // Arithmetic, fallible
    // ------------------------------------------------------------------

    /// `[a.lo + b.lo, a.hi + b.hi]`. Storage addition is exact when it fits.
    pub fn try_add(self, rhs: Self) -> Result<Self, OverflowDetected> {
        Ok(Self::from_raw(
            st_add(self.lo.raw_value(), rhs.lo.raw_value())?,
            st_add(self.hi.raw_value(), rhs.hi.raw_value())?,
        ))
    }

    /// `[a.lo - b.hi, a.hi - b.lo]`.
    pub fn try_sub(self, rhs: Self) -> Result<Self, OverflowDetected> {
        Ok(Self::from_raw(
            st_sub(self.lo.raw_value(), rhs.hi.raw_value())?,
            st_sub(self.hi.raw_value(), rhs.lo.raw_value())?,
        ))
    }

    /// `[-hi, -lo]`.
    pub fn try_neg(self) -> Result<Self, OverflowDetected> {
        Ok(Self::from_raw(st_neg(self.hi.raw_value())?, st_neg(self.lo.raw_value())?))
    }

    /// Product: exact corner products at `2 * DECIMALS` places, narrowed once.
    ///
    /// The four corner products are formed exactly in `D256`, the extremes are
    /// chosen exactly, and each is narrowed once: floor for the minimum, ceil
    /// for the maximum.
    pub fn try_mul(self, rhs: Self) -> Result<Self, OverflowDetected> {
        let p = [
            mul_i128_to_d256(self.lo.raw_value(), rhs.lo.raw_value()),
            mul_i128_to_d256(self.lo.raw_value(), rhs.hi.raw_value()),
            mul_i128_to_d256(self.hi.raw_value(), rhs.lo.raw_value()),
            mul_i128_to_d256(self.hi.raw_value(), rhs.hi.raw_value()),
        ];
        let mut mn = p[0];
        let mut mx = p[0];
        for q in &p[1..] {
            if *q < mn { mn = *q; }
            if *q > mx { mx = *q; }
        }
        Ok(Self::from_raw(narrow::<DECIMALS>(mn)?.0, narrow::<DECIMALS>(mx)?.1))
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
            directed_divide::<DECIMALS>(self.lo.raw_value(), rhs.lo.raw_value())?,
            directed_divide::<DECIMALS>(self.lo.raw_value(), rhs.hi.raw_value())?,
            directed_divide::<DECIMALS>(self.hi.raw_value(), rhs.lo.raw_value())?,
            directed_divide::<DECIMALS>(self.hi.raw_value(), rhs.hi.raw_value())?,
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
    /// exact integer certificate `k^2 <= x * 10^DECIMALS < (k+1)^2`.
    pub fn try_sqrt(self) -> Result<Self, OverflowDetected> {
        if self.lo < DecimalFixed::ZERO {
            return Err(OverflowDetected::DomainError);
        }
        let (lo, _) = certified_sqrt::<DECIMALS>(self.lo.raw_value())?;
        let (_, hi) = certified_sqrt::<DECIMALS>(self.hi.raw_value())?;
        Ok(Self::from_raw(lo, hi))
    }

    /// Certified dot product of two point vectors, with one narrowing.
    ///
    /// Every product is exact at `2 * DECIMALS` places and the sum is exact
    /// with overflow-checked `D256` addition; the single narrowing happens at
    /// the end.
    ///
    /// Panics if the slices differ in length.
    pub fn try_dot(a: &[DecimalFixed<DECIMALS>], b: &[DecimalFixed<DECIMALS>]) -> Result<Self, OverflowDetected> {
        assert_eq!(a.len(), b.len(), "DecimalInterval::dot: length mismatch");
        let mut acc = D256::zero();
        for i in 0..a.len() {
            acc = d256_checked_add(acc, mul_i128_to_d256(a[i].raw_value(), b[i].raw_value()))?;
        }
        let (lo, hi) = narrow::<DECIMALS>(acc)?;
        Ok(Self::from_raw(lo, hi))
    }

    // ------------------------------------------------------------------
    // Arithmetic, infallible (panics where the try_ twin errs)
    // ------------------------------------------------------------------

    /// Certified square root; panics on a negative lower endpoint or overflow.
    pub fn sqrt(self) -> Self {
        self.try_sqrt().expect("DecimalInterval::sqrt: domain error or overflow")
    }

    /// Certified dot product; panics on overflow.
    pub fn dot(a: &[DecimalFixed<DECIMALS>], b: &[DecimalFixed<DECIMALS>]) -> Self {
        Self::try_dot(a, b).expect("DecimalInterval::dot: overflow")
    }
}

impl<const DECIMALS: u8> Add for DecimalInterval<DECIMALS> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self { self.try_add(rhs).expect("DecimalInterval: addition overflow") }
}

impl<const DECIMALS: u8> Sub for DecimalInterval<DECIMALS> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self { self.try_sub(rhs).expect("DecimalInterval: subtraction overflow") }
}

impl<const DECIMALS: u8> Mul for DecimalInterval<DECIMALS> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self { self.try_mul(rhs).expect("DecimalInterval: multiplication overflow") }
}

impl<const DECIMALS: u8> Div for DecimalInterval<DECIMALS> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self { self.try_div(rhs).expect("DecimalInterval: division by an interval containing zero, or overflow") }
}

impl<const DECIMALS: u8> Neg for DecimalInterval<DECIMALS> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { self.try_neg().expect("DecimalInterval: negation overflow") }
}

impl<const DECIMALS: u8> From<DecimalFixed<DECIMALS>> for DecimalInterval<DECIMALS> {
    #[inline]
    fn from(x: DecimalFixed<DECIMALS>) -> Self { DecimalInterval::point(x) }
}
