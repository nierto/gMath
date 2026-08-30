//! Certified interval arithmetic: enclosures that are sound by construction.
//!
//! An [`Interval`] is a pair of storage-tier endpoints `[lo, hi]` carrying the
//! guarantee that the exact mathematical result of the operation that produced
//! it lies within them. Soundness rests on two mechanisms and nothing else:
//!
//! 1. Every product of two storage values is computed EXACTLY at the compute
//!    tier (storage width doubled), and sums of such products accumulate
//!    exactly with checked addition.
//! 2. Every narrowing back to storage rounds the lower endpoint toward
//!    negative infinity and the upper endpoint toward positive infinity.
//!
//! Compound operations ([`Interval::dot`], [`Interval::quadratic_form`])
//! accumulate exactly and narrow once. That is what keeps the enclosure
//! tight: `dot` and `quadratic_form` are `[floor, ceil]` of one exact value,
//! at most 1 ulp wide. The quadratic form's earlier two-narrowing form
//! measured 5 to 64 ulp for 23 dimensions at Q64.64, itself 12 to 24 times
//! tighter than narrowing after every elementary operation. Width is set by
//! the number of narrowings, not by cancellation, because cancellation inside
//! an exact accumulator costs nothing.
//!
//! **What may carry the word certified.** `+`, `-`, `*`, `/` are sound by the
//! standard endpoint argument. [`Interval::sqrt`] is certified a posteriori:
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
//! **Where this sits.** An imperative-tier type beside `FixedPoint`,
//! `FixedVector`, `FixedMatrix` and `fused`: binary Q-format, fixed at the
//! profile's storage tier, intermediates at the compute tier. It is not a
//! FASC domain (no `StackValue` variant, no routing) and not UGOD-tiered
//! (the imperative layer has no wider storage tier to promote into); on
//! storage overflow it fails typed where the scalar operators wrap. See
//! `docs/design/CERTIFIED_INTERVALS.md` for the reasoning.
//!
//! Non-goals: IEEE 1788 decorations, reverse operations, midpoint-radius and
//! affine forms. The measured widths did not call for any of them.

use std::ops::{Add, Div, Mul, Neg, Sub};

use super::wide_acc::{narrow_triple_ceil, narrow_triple_floor, quadratic_form_exact};
use super::{FixedMatrix, FixedPoint, FixedVector};
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_checked_add, downscale_to_storage_ceil, downscale_to_storage_floor,
};
use crate::fixed_point::universal::fasc::stack_evaluator::{
    upscale_to_compute, BinaryStorage, ComputeStorage,
};

#[cfg(table_format = "q16_16")]
use crate::fixed_point::frac_config;
#[cfg(table_format = "q64_64")]
use crate::fixed_point::i256::mul_i128_to_i256;
#[cfg(table_format = "q64_64")]
use crate::fixed_point::I256;
#[cfg(table_format = "q128_128")]
use crate::fixed_point::I512;
#[cfg(table_format = "q256_256")]
use crate::fixed_point::{I1024, I512};

// ============================================================================
// Storage-tier checked helpers (never wrap)
// ============================================================================

#[inline]
fn st_add(a: BinaryStorage, b: BinaryStorage) -> Result<BinaryStorage, OverflowDetected> {
    a.checked_add(b).ok_or(OverflowDetected::TierOverflow)
}

#[inline]
fn st_sub(a: BinaryStorage, b: BinaryStorage) -> Result<BinaryStorage, OverflowDetected> {
    a.checked_sub(b).ok_or(OverflowDetected::TierOverflow)
}

#[inline]
fn st_neg(a: BinaryStorage) -> Result<BinaryStorage, OverflowDetected> {
    a.checked_neg().ok_or(OverflowDetected::TierOverflow)
}

#[inline]
fn compute_zero() -> ComputeStorage {
    upscale_to_compute(FixedPoint::ZERO.raw())
}

/// Exact product of two storage values at the compute tier.
///
/// The raw product `a_raw * b_raw` is the value `a * b` at exactly the
/// compute-tier scale (2 * FRAC_BITS), so no rounding occurs here. On the wide
/// profiles the unsigned wide multiply is applied to magnitudes and the sign
/// reapplied, the same pattern as the scalar `fixed_multiply`; on Q64.64
/// `mul_i128_to_i256` is sign-correct by itself.
#[inline]
pub(crate) fn exact_product(a: BinaryStorage, b: BinaryStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    {
        (a as i64) * (b as i64)
    }
    #[cfg(table_format = "q32_32")]
    {
        (a as i128) * (b as i128)
    }
    #[cfg(table_format = "q64_64")]
    {
        mul_i128_to_i256(a, b)
    }
    #[cfg(table_format = "q128_128")]
    {
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let product = abs_a.mul_to_i512(abs_b);
        if a_neg != b_neg { -product } else { product }
    }
    #[cfg(table_format = "q256_256")]
    {
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let product = abs_a.mul_to_i1024(abs_b);
        if a_neg != b_neg { -product } else { product }
    }
}

/// Directed quotient of two storage values: `(floor(a/b), ceil(a/b))` at the
/// storage scale, computed from the truncating integer quotient and its
/// remainder. Mirrors `fixed_divide`'s widening per profile.
#[inline]
fn directed_divide(a: BinaryStorage, b: BinaryStorage) -> Result<(BinaryStorage, BinaryStorage), OverflowDetected> {
    #[cfg(table_format = "q16_16")]
    {
        if b == 0 { return Err(OverflowDetected::DivisionByZero); }
        let num = (a as i64) << frac_config::FRAC_BITS;
        let den = b as i64;
        let q = num / den;
        let rem = num - q * den;
        let (f, c) = if rem == 0 {
            (q, q)
        } else if (num < 0) == (den < 0) {
            (q, q + 1)
        } else {
            (q - 1, q)
        };
        let fits = |v: i64| v >= i32::MIN as i64 && v <= i32::MAX as i64;
        if !fits(f) || !fits(c) { return Err(OverflowDetected::TierOverflow); }
        Ok((f as i32, c as i32))
    }
    #[cfg(table_format = "q32_32")]
    {
        if b == 0 { return Err(OverflowDetected::DivisionByZero); }
        let num = (a as i128) << 32;
        let den = b as i128;
        let q = num / den;
        let rem = num - q * den;
        let (f, c) = if rem == 0 {
            (q, q)
        } else if (num < 0) == (den < 0) {
            (q, q + 1)
        } else {
            (q - 1, q)
        };
        let fits = |v: i128| v >= i64::MIN as i128 && v <= i64::MAX as i128;
        if !fits(f) || !fits(c) { return Err(OverflowDetected::TierOverflow); }
        Ok((f as i64, c as i64))
    }
    #[cfg(table_format = "q64_64")]
    {
        if b == 0 { return Err(OverflowDetected::DivisionByZero); }
        let num = I256::from_i128(a) << 64usize;
        let den = I256::from_i128(b);
        let q = num / den;
        let rem = num - q * den;
        let one = I256::from_i128(1);
        let (f, c) = if rem.is_zero() {
            (q, q)
        } else if num.is_negative() == den.is_negative() {
            (q, q + one)
        } else {
            (q - one, q)
        };
        if !f.fits_in_i128() || !c.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
        Ok((f.as_i128(), c.as_i128()))
    }
    #[cfg(table_format = "q128_128")]
    {
        if b.is_zero() { return Err(OverflowDetected::DivisionByZero); }
        let num = I512::from_i256(a) << 128usize;
        let den = I512::from_i256(b);
        let q = num / den;
        let rem = num - q * den;
        let one = I512::from_i128(1);
        let (f, c) = if rem == I512::zero() {
            (q, q)
        } else if num.is_negative() == den.is_negative() {
            (q, q + one)
        } else {
            (q - one, q)
        };
        if !f.fits_in_i256() || !c.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
        Ok((f.as_i256(), c.as_i256()))
    }
    #[cfg(table_format = "q256_256")]
    {
        if b == I512::zero() { return Err(OverflowDetected::DivisionByZero); }
        let num = I1024::from_i512(a) << 256usize;
        let den = I1024::from_i512(b);
        let q = num / den;
        let rem = num - q * den;
        let one = I1024::from_i128(1);
        let zero = I1024::zero();
        let (f, c) = if rem == zero {
            (q, q)
        } else if (num < zero) == (den < zero) {
            (q, q + one)
        } else {
            (q - one, q)
        };
        if !f.fits_in_i512() || !c.fits_in_i512() { return Err(OverflowDetected::TierOverflow); }
        Ok((f.as_i512(), c.as_i512()))
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
fn certified_sqrt(x: BinaryStorage) -> Result<(BinaryStorage, BinaryStorage), OverflowDetected> {
    let n = upscale_to_compute(x);
    let mut k = compute_narrow_integer(isqrt_compute(n))?;
    let unit = unit_raw();
    let mut steps = 0u32;
    while exact_product(k, k) > n {
        k = st_sub(k, unit)?;
        steps += 1;
        assert!(steps <= 2, "certified_sqrt: candidate failed the certificate downward");
    }
    loop {
        let next = st_add(k, unit)?;
        if exact_product(next, next) <= n {
            k = next;
            steps += 1;
            assert!(steps <= 2, "certified_sqrt: candidate failed the certificate upward");
        } else {
            break;
        }
    }
    let floor = k;
    let ceil = if exact_product(k, k) == n { k } else { st_add(k, unit)? };
    Ok((floor, ceil))
}

/// `floor(sqrt(n))` for a non-negative compute-tier integer, by Newton from
/// above: seed `2^ceil(bits/2) >= sqrt(n)`, iterate `k = (k + n/k) / 2` while
/// it decreases. Plain integer arithmetic on the compute type; no Q-format
/// scaling is involved.
#[inline]
fn isqrt_compute(n: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    {
        debug_assert!(n >= 0);
        (n as u64).isqrt() as i64
    }
    #[cfg(table_format = "q32_32")]
    {
        debug_assert!(n >= 0);
        (n as u128).isqrt() as i128
    }
    #[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256"))]
    {
        if n == ComputeStorage::zero() {
            return n;
        }
        let bits = compute_bit_length(n);
        let mut k = ComputeStorage::from_i128(1) << (((bits + 1) / 2) as usize);
        loop {
            let next = (k + n / k) >> 1;
            if next >= k {
                return k;
            }
            k = next;
        }
    }
}

/// Significant bits of a non-negative wide compute-tier value.
#[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256"))]
#[inline]
fn compute_bit_length(n: ComputeStorage) -> u32 {
    let words = &n.words;
    for i in (0..words.len()).rev() {
        if words[i] != 0 {
            return i as u32 * 64 + (64 - words[i].leading_zeros());
        }
    }
    0
}

/// A compute-tier integer that is known to fit the storage type, as storage.
/// This is a plain integer narrowing, not a Q-format downscale.
#[inline]
fn compute_narrow_integer(v: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    #[cfg(table_format = "q16_16")]
    {
        i32::try_from(v).map_err(|_| OverflowDetected::TierOverflow)
    }
    #[cfg(table_format = "q32_32")]
    {
        i64::try_from(v).map_err(|_| OverflowDetected::TierOverflow)
    }
    #[cfg(table_format = "q64_64")]
    {
        if v.fits_in_i128() { Ok(v.as_i128()) } else { Err(OverflowDetected::TierOverflow) }
    }
    #[cfg(table_format = "q128_128")]
    {
        if v.fits_in_i256() { Ok(v.as_i256()) } else { Err(OverflowDetected::TierOverflow) }
    }
    #[cfg(table_format = "q256_256")]
    {
        if v.fits_in_i512() { Ok(v.as_i512()) } else { Err(OverflowDetected::TierOverflow) }
    }
}

/// The raw integer 1 (one ulp) in the storage type.
#[inline]
fn unit_raw() -> BinaryStorage {
    #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
    { 1 }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::I256::from_i128(1) }
    #[cfg(table_format = "q256_256")]
    { I512::from_i128(1) }
}

// ============================================================================
// Interval
// ============================================================================

/// A certified enclosure `[lo, hi]` of a real value, `lo <= hi`.
///
/// Endpoints are storage-tier `FixedPoint` values. Every operation on this
/// type returns an interval that contains the exact mathematical result for
/// every choice of operands within the input intervals. See the module
/// documentation for what that guarantee rests on and what it excludes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Interval {
    lo: FixedPoint,
    hi: FixedPoint,
}

impl Interval {
    /// The degenerate interval `[x, x]`.
    #[inline]
    pub fn point(x: FixedPoint) -> Self {
        Interval { lo: x, hi: x }
    }

    /// `[lo, hi]`. Panics if `lo > hi`.
    #[inline]
    pub fn new(lo: FixedPoint, hi: FixedPoint) -> Self {
        Self::try_new(lo, hi).expect("Interval::new: lo > hi")
    }

    /// `[lo, hi]`, or `Err(InvalidInput)` if `lo > hi`.
    #[inline]
    pub fn try_new(lo: FixedPoint, hi: FixedPoint) -> Result<Self, OverflowDetected> {
        if lo > hi {
            return Err(OverflowDetected::InvalidInput);
        }
        Ok(Interval { lo, hi })
    }

    #[inline]
    fn from_raw(lo: BinaryStorage, hi: BinaryStorage) -> Self {
        let iv = Interval { lo: FixedPoint::from_raw(lo), hi: FixedPoint::from_raw(hi) };
        debug_assert!(iv.lo <= iv.hi, "Interval invariant violated: lo > hi");
        iv
    }

    /// Lower endpoint.
    #[inline]
    pub fn lo(self) -> FixedPoint { self.lo }

    /// Upper endpoint.
    #[inline]
    pub fn hi(self) -> FixedPoint { self.hi }

    /// `hi - lo`. Panics if the width itself does not fit the storage tier.
    #[inline]
    pub fn width(self) -> FixedPoint {
        FixedPoint::from_raw(st_sub(self.hi.raw(), self.lo.raw()).expect("Interval::width: overflow"))
    }

    /// `lo == hi`.
    #[inline]
    pub fn is_point(self) -> bool { self.lo == self.hi }

    /// `lo <= x <= hi`.
    #[inline]
    pub fn contains(self, x: FixedPoint) -> bool { self.lo <= x && x <= self.hi }

    /// `lo <= 0 <= hi`.
    #[inline]
    pub fn contains_zero(self) -> bool { self.contains(FixedPoint::ZERO) }

    /// `lo > 0`: every value in the interval is positive.
    #[inline]
    pub fn is_certainly_positive(self) -> bool { self.lo > FixedPoint::ZERO }

    /// `hi < 0`: every value in the interval is negative.
    #[inline]
    pub fn is_certainly_negative(self) -> bool { self.hi < FixedPoint::ZERO }

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
            exact_product(self.lo.raw(), rhs.lo.raw()),
            exact_product(self.lo.raw(), rhs.hi.raw()),
            exact_product(self.hi.raw(), rhs.lo.raw()),
            exact_product(self.hi.raw(), rhs.hi.raw()),
        ];
        let mut mn = p[0];
        let mut mx = p[0];
        for q in &p[1..] {
            if *q < mn { mn = *q; }
            if *q > mx { mx = *q; }
        }
        Ok(Self::from_raw(downscale_to_storage_floor(mn)?, downscale_to_storage_ceil(mx)?))
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
        if self.lo < FixedPoint::ZERO {
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
    pub fn try_dot(a: &[FixedPoint], b: &[FixedPoint]) -> Result<Self, OverflowDetected> {
        assert_eq!(a.len(), b.len(), "Interval::dot: length mismatch");
        let mut acc = compute_zero();
        for i in 0..a.len() {
            acc = compute_checked_add(acc, exact_product(a[i].raw(), b[i].raw()))?;
        }
        Ok(Self::from_raw(downscale_to_storage_floor(acc)?, downscale_to_storage_ceil(acc)?))
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
    pub fn try_dot_intervals(a: &[Interval], b: &[Interval]) -> Result<Self, OverflowDetected> {
        assert_eq!(a.len(), b.len(), "Interval::dot_intervals: length mismatch");
        let mut lo_acc = compute_zero();
        let mut hi_acc = compute_zero();
        for i in 0..a.len() {
            let p = [
                exact_product(a[i].lo.raw(), b[i].lo.raw()),
                exact_product(a[i].lo.raw(), b[i].hi.raw()),
                exact_product(a[i].hi.raw(), b[i].lo.raw()),
                exact_product(a[i].hi.raw(), b[i].hi.raw()),
            ];
            let mut mn = p[0];
            let mut mx = p[0];
            for q in &p[1..] {
                if *q < mn { mn = *q; }
                if *q > mx { mx = *q; }
            }
            lo_acc = compute_checked_add(lo_acc, mn)?;
            hi_acc = compute_checked_add(hi_acc, mx)?;
        }
        Ok(Self::from_raw(downscale_to_storage_floor(lo_acc)?, downscale_to_storage_ceil(hi_acc)?))
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
    pub fn try_quadratic_form(v: &FixedVector, m: &FixedMatrix) -> Result<Self, OverflowDetected> {
        let acc = quadratic_form_exact(v, m)?;
        Ok(Self::from_raw(narrow_triple_floor(acc)?, narrow_triple_ceil(acc)?))
    }

    // ------------------------------------------------------------------
    // Arithmetic, infallible (panics where the try_ twin errs)
    // ------------------------------------------------------------------

    /// Certified square root; panics on a negative lower endpoint or overflow.
    pub fn sqrt(self) -> Self {
        self.try_sqrt().expect("Interval::sqrt: domain error or overflow")
    }

    /// Certified dot product; panics on overflow.
    pub fn dot(a: &[FixedPoint], b: &[FixedPoint]) -> Self {
        Self::try_dot(a, b).expect("Interval::dot: overflow")
    }

    /// Certified quadratic form; panics on overflow.
    pub fn quadratic_form(v: &FixedVector, m: &FixedMatrix) -> Self {
        Self::try_quadratic_form(v, m).expect("Interval::quadratic_form: overflow")
    }

    /// Certified dot product of interval vectors; panics on overflow.
    pub fn dot_intervals(a: &[Interval], b: &[Interval]) -> Self {
        Self::try_dot_intervals(a, b).expect("Interval::dot_intervals: overflow")
    }
}

impl Add for Interval {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self { self.try_add(rhs).expect("Interval: addition overflow") }
}

impl Sub for Interval {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self { self.try_sub(rhs).expect("Interval: subtraction overflow") }
}

impl Mul for Interval {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self { self.try_mul(rhs).expect("Interval: multiplication overflow") }
}

impl Div for Interval {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self { self.try_div(rhs).expect("Interval: division by an interval containing zero, or overflow") }
}

impl Neg for Interval {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { self.try_neg().expect("Interval: negation overflow") }
}

impl From<FixedPoint> for Interval {
    #[inline]
    fn from(x: FixedPoint) -> Self { Interval::point(x) }
}
