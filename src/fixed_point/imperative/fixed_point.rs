//! The scalar fixed-point type: [`Fixed<S, F>`](Fixed) and its methods.
//!
//! Arithmetic operators and transcendentals work directly on the raw
//! Q-format integer; the transcendentals compute one tier above storage and
//! round once. `FixedPoint` is the format the build profile selects.

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

use crate::fixed_point::canonical::{
    LazyExpr, StackValue, evaluate_in, gmath_parse_in, CompactShadow, FascFormat,
};
// SCAFFOLDING (width-types phase 7): the build's profile, read by the tests only; removed in 7c.
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::fixed_num::{Compute, FixedNum};
pub use crate::fixed_point::fixed_num::Fixed;
use crate::fixed_point::width::{Storage, WideInt};
use crate::fixed_point::domains::binary_fixed::transcendental::sure;
use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::sincos_q64_64_native;
pub use crate::fixed_point::core_types::errors::OverflowDetected;

/// The engines reduce trigonometric arguments with a pi of fixed precision,
/// so their error grows with |x| faster than the margin model assumes: above
/// 2^24 the sure path (pi at 1500 bits) evaluates sin, cos and tan.
const TRIG_ENGINE_INT_BITS: u32 = 25;

/// The fixed-point format this build's profile selects (`GMATH_PROFILE`,
/// and `GMATH_FRAC_BITS` on the realtime profile).
///
/// Profile-dependent size:
/// - `embedded` (Q64.64): 16 bytes (i128)
/// - `balanced` (Q128.128): 32 bytes (I256)
/// - `scientific` (Q256.256): 64 bytes (I512)
///
/// **Layout guarantee (0.6.5):** `FixedPoint` is `#[repr(transparent)]` over
/// its raw storage integer, so a `FixedPoint` and its raw value have the same
/// size, alignment and bit pattern, and every bit pattern of the raw type is a
/// valid `FixedPoint`. [`FixedPoint::raw_slice`] and
/// [`FixedPoint::from_raw_slice`] reinterpret slices without copying.
// SCAFFOLDING (width-types phase 7): the build's profile, read by the tests only; removed in 7c.
pub type FixedPoint = Fixed<BinaryStorage, { crate::fixed_point::frac_config::FRAC_BITS }>;

impl<S: PartialEq, const F: u32> PartialEq for Fixed<S, F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<S: Eq, const F: u32> Eq for Fixed<S, F> {}

impl<S: Ord, const F: u32> PartialOrd for Fixed<S, F> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: Ord, const F: u32> Ord for Fixed<S, F> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.raw.cmp(&other.raw)
    }
}

// ============================================================================
// Core constructors and accessors
// ============================================================================

impl<S: Copy, const F: u32> Fixed<S, F> {
    /// Create from raw Q-format storage.
    #[inline]
    pub fn from_raw(raw: S) -> Self {
        Self { raw }
    }
    /// Access the raw Q-format storage.
    #[inline]
    pub fn raw(self) -> S {
        self.raw
    }
    /// View a slice of values as their raw storage integers, without copying.
    #[inline]
    pub fn raw_slice(values: &[Self]) -> &[S] {
        // SAFETY: `FixedPoint` is `repr(transparent)` over `S`:
        // same size, alignment and validity, so the slice layouts are equal.
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const S, values.len()) }
    }
    /// View a slice of raw storage integers as values, without copying.
    #[inline]
    pub fn from_raw_slice(raws: &[S]) -> &[Self] {
        // SAFETY: as in `raw_slice`; every raw bit pattern is a valid value.
        unsafe { std::slice::from_raw_parts(raws.as_ptr() as *const Self, raws.len()) }
    }
    /// Mutable form of [`raw_slice`](Self::raw_slice).
    #[inline]
    pub fn raw_slice_mut(values: &mut [Self]) -> &mut [S] {
        // SAFETY: as in `raw_slice`; the borrow is exclusive.
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut S, values.len()) }
    }
    /// Mutable form of [`from_raw_slice`](Self::from_raw_slice).
    #[inline]
    pub fn from_raw_slice_mut(raws: &mut [S]) -> &mut [Self] {
        // SAFETY: as in `raw_slice`; the borrow is exclusive.
        unsafe { std::slice::from_raw_parts_mut(raws.as_mut_ptr() as *mut Self, raws.len()) }
    }
}

impl<S: Storage, const F: u32> Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    /// Zero constant.
    pub const ZERO: Self = Self { raw: S::ZERO };

    /// One (1.0) in Q-format.
    #[inline]
    pub fn one() -> Self {
        Self { raw: Self::one_storage() }
    }







    /// Create from an integer value.
    ///
    /// Panics when `v` is outside the profile's integer range, which only the
    /// realtime profile can reach (`|v| >= 2^(31 - GMATH_FRAC_BITS)`: 32768 at
    /// Q16.16, 128 at Q8.24). Before 0.6.4 the shift wrapped silently.
    /// [`try_from_int`](Self::try_from_int) returns `Err(TierOverflow)` instead.
    #[inline]
    pub fn from_int(v: i32) -> Self {
        Self::try_from_int(v).expect("FixedPoint::from_int: value outside the profile's range")
    }

    /// Create from an integer value, `Err(TierOverflow)` outside the range.
    ///
    /// Only the realtime profile can refuse: every i32 fits the wider ones.
    #[inline]
    pub fn try_from_int(v: i32) -> Result<Self, OverflowDetected> {
        <Self as FixedNum>::try_from_int(v).map(|raw| Self { raw })
    }

    /// `self * n` for a count or dimension `n`: exact at the compute tier (the
    /// count need not fit the storage range), panics if the product does not.
    pub(crate) fn mul_count(self, n: usize) -> Self {
        let c = Self::compute_mul_div_int(Self::upscale_to_compute(self.raw), n as i64, 1)
            .expect("FixedPoint::mul_count: product exceeds the compute tier");
        Self { raw: Self::downscale_to_storage(c).expect("FixedPoint::mul_count: product exceeds storage") }
    }

    /// `self / n` for a count or dimension `n > 0`: the storage division when
    /// `n` fits the storage range (bit-identical to `self / from_int(n)`),
    /// the compute tier when it does not (a realtime count past 2^(31 - F)).
    pub(crate) fn div_count(self, n: usize) -> Self {
        match i32::try_from(n).ok().and_then(|n| Self::try_from_int(n).ok()) {
            Some(nf) => self / nf,
            None => {
                let c = Self::compute_mul_div_int(Self::upscale_to_compute(self.raw), 1, n as i64)
                    .expect("FixedPoint::div_count: n > 0");
                Self { raw: Self::downscale_to_storage(c).expect("FixedPoint::div_count: quotient fits storage") }
            }
        }
    }

    /// Extract the integer part (floor toward negative infinity).
    ///
    /// Panics when the floor is outside the i32 range (`|x| >= 2^31`, reachable
    /// on every profile but realtime); the cast wrapped silently before 0.6.4.
    /// [`try_to_int`](Self::try_to_int) returns `Err(TierOverflow)` instead.
    #[inline]
    pub fn to_int(self) -> i32 {
        self.try_to_int().expect("FixedPoint::to_int: integer part outside the i32 range")
    }

    /// The integer part (floor toward negative infinity), `Err(TierOverflow)`
    /// when it is outside the i32 range.
    #[inline]
    pub fn try_to_int(self) -> Result<i32, OverflowDetected> {
        <Self as FixedNum>::try_to_int(self.raw)
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        if self.is_negative() { -self } else { self }
    }

    /// Check if negative.
    #[inline]
    pub fn is_negative(self) -> bool {
        self.raw.is_negative()
    }

    /// Check if zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.raw == S::ZERO
    }

    // ========================================================================
    // f32/f64 conversions (user-convenience boundary only)
    // ========================================================================

    /// Create from an f32 value, truncated toward zero to the profile's raw step.
    ///
    /// Reads the IEEE 754 bits; no float arithmetic is performed. Panics on NaN,
    /// infinity, or a value outside the profile's range; see
    /// [`try_from_f32`](Self::try_from_f32) for the fallible form.
    pub fn from_f32(v: f32) -> Self {
        Self::try_from_f32(v).unwrap_or_else(|e| panic!("FixedPoint::from_f32: {}", Self::float_input_error(e)))
    }

    /// Create from an f32 value like `from_f32`, returning an error instead of panicking.
    ///
    /// Truncates toward zero to the profile's raw step. `Err(InvalidInput)` for
    /// NaN, `Err(TierOverflow)` for infinity or a value outside the profile's
    /// range.
    pub fn try_from_f32(v: f32) -> Result<Self, OverflowDetected> {
        let bits = v.to_bits();
        let negative = (bits >> 31) != 0;
        let raw_exp = ((bits >> 23) & 0xFF) as i32;
        let fraction = (bits & 0x7F_FFFF) as u64;
        if raw_exp == 0xFF {
            return Err(if fraction == 0 { OverflowDetected::TierOverflow } else { OverflowDetected::InvalidInput });
        }
        let (mantissa, exp_offset) = if raw_exp == 0 {
            // Subnormal (or zero): no implicit 1, exponent -126
            (fraction, -126 - 23)
        } else {
            (fraction | 0x80_0000, raw_exp - 127 - 23)
        };
        Ok(Self { raw: <Self as FixedNum>::truncated_raw(mantissa, exp_offset + F as i32, negative)? })
    }

    /// Create from an f64 value, truncated toward zero to the profile's raw step.
    ///
    /// Reads the IEEE 754 bits; no float arithmetic is performed. Panics on NaN,
    /// infinity, or a value outside the profile's range; see
    /// [`try_from_f64`](Self::try_from_f64) for the fallible form.
    pub fn from_f64(v: f64) -> Self {
        Self::try_from_f64(v).unwrap_or_else(|e| panic!("FixedPoint::from_f64: {}", Self::float_input_error(e)))
    }

    /// Create from an f64 value like `from_f64`, returning an error instead of panicking.
    ///
    /// Truncates toward zero to the profile's raw step. `Err(InvalidInput)` for
    /// NaN, `Err(TierOverflow)` for infinity or a value outside the profile's
    /// range.
    pub fn try_from_f64(v: f64) -> Result<Self, OverflowDetected> {
        let bits = v.to_bits();
        let negative = (bits >> 63) != 0;
        let raw_exp = ((bits >> 52) & 0x7FF) as i32;
        let fraction = bits & 0x000F_FFFF_FFFF_FFFF;
        if raw_exp == 0x7FF {
            return Err(if fraction == 0 { OverflowDetected::TierOverflow } else { OverflowDetected::InvalidInput });
        }
        let (mantissa, exp_offset) = if raw_exp == 0 {
            // Subnormal (or zero): no implicit 1, exponent -1022
            (fraction, -1022 - 52)
        } else {
            (fraction | 0x0010_0000_0000_0000, raw_exp - 1023 - 52)
        };
        Ok(Self { raw: <Self as FixedNum>::truncated_raw(mantissa, exp_offset + F as i32, negative)? })
    }

    fn float_input_error(e: OverflowDetected) -> &'static str {
        match e {
            OverflowDetected::InvalidInput => "NaN",
            _ => "infinity or value outside the profile's range",
        }
    }

    /// Convert to f32: exact when the raw value fits 24 bits, else nearest-even.
    ///
    /// Below f32's normal range the result is subnormal or zero, and beyond its
    /// range (scientific profile only) it is infinite. The f32 is assembled from
    /// the raw integer's bits; no float arithmetic is performed.
    ///
    /// For display and interop only: never decode a value for further
    /// computation through a float.
    pub fn to_f32(self) -> f32 {
        f32::from_bits(<Self as FixedNum>::float_bits(self.raw, 23, 8) as u32)
    }

    /// Convert to f64: exact when the raw value fits 53 bits, else nearest-even.
    ///
    /// Exact for every realtime value and for compact values below 2^21 in
    /// magnitude, and exact conversions round trip:
    /// `FixedPoint::from_f64(x.to_f64()) == x`. The f64 is assembled from the
    /// raw integer's bits; no float arithmetic is performed.
    ///
    /// Before 0.6.3 this printed a truncated decimal string and parsed it;
    /// `x.to_string().parse::<f64>()` reproduces that result.
    ///
    /// For display and interop only: never decode a value for further
    /// computation through a float.
    pub fn to_f64(self) -> f64 {
        f64::from_bits(<Self as FixedNum>::float_bits(self.raw, 52, 11))
    }





    // ========================================================================
    // Transcendentals — direct binary engine calls (bypass FASC)
    //
    // Pattern: upscale → binary engine at compute tier → downscale.
    // Saves ~65 ns per call vs the FASC pipeline (no LazyExpr tree, no TLS,
    // no StackValue boxing, no domain routing). Proven by sincos_wide().
    // ========================================================================

    /// 1.0 at compute tier
    #[inline]
    fn compute_one() -> Compute<Self> {
        Self::upscale_to_compute(Self::one_storage())
    }

    #[inline]
    fn compute_divide_direct(a: Compute<Self>, b: Compute<Self>) -> Compute<Self> {
        Self::compute_divide(a, b).expect("division by zero in transcendental composition")
    }

    /// Upscale self.raw to compute tier, call engine, downscale result.
    #[inline]
    fn direct_unary<E: FnOnce(Compute<Self>) -> Compute<Self>>(self, f: E) -> Self {
        let compute = Self::upscale_to_compute(self.raw);
        let result = f(compute);
        Self { raw: Self::downscale_to_storage(result).expect("transcendental overflow") }
    }

    /// Round a compute-tier result to storage when the engine's error cannot
    /// move it across a rounding boundary; otherwise (or when the engine
    /// saturated) let `sure` decide it at about 1000 bits.
    ///
    /// The engines' error is at most a few units of the compute tier's last
    /// place times the result's magnitude (they compute at `2F` fraction bits
    /// with a relative error). The margin is `2^ABS + (|value| + 1) * 2^REL`
    /// compute units, times `2^extra_bits` for compositions whose error grows
    /// further (pow: the exponent's size); a result whose discarded `F` bits
    /// lie farther than that from one half is decided by the engine alone.
    #[inline]
    fn settle<G: FnOnce() -> Option<S>>(c: Compute<Self>, extra_bits: u32, sure: G) -> Result<S, OverflowDetected> {
        const ABS: u32 = 2;
        let rel = Self::engine_rel_bits();
        type C<S> = <S as Storage>::Compute;
        let one = <C<S> as WideInt>::one();
        let floor = c.shr_bits(F).shl_bits(F);
        let below = c - floor;
        let half = one.shl_bits(F - 1);
        let dist = if below < half { half - below } else { below - half };
        let decided = match c.checked_neg_wide() {
            Some(neg) => {
                let mag = if c.is_negative() { neg } else { c };
                let int_part = mag.shr_bits(2 * F) + one;
                let need = int_part.bit_length() + rel + extra_bits;
                need + 2 < <C<S> as WideInt>::BITS && {
                    let margin = int_part.shl_bits(rel + extra_bits) + one.shl_bits(ABS);
                    dist > margin
                }
            }
            None => false,
        };
        if decided {
            if let Ok(r) = Self::downscale_to_storage(c) { return Ok(r); }
        }
        sure().ok_or(OverflowDetected::TierOverflow)
    }

    /// The engines' relative error bound in compute units, log2: the realtime
    /// engines run the Q64.64 engines and round once to 2F bits (measured
    /// within half a unit at F = 6..28, sin and cos up to 2^20 included); the
    /// wider ones carry a few units relative (measured up to 2^4 for exp).
    /// The same margins serve the canonical evaluator's compute-tier stage.
    #[inline]
    pub(crate) fn engine_rel_bits() -> u32 { if S::BITS == 32 { 1 } else { 6 } }

    /// The extra margin of sin and cos: their error grows with the argument
    /// on the wide engines, not on the realtime ones.
    #[inline]
    fn trig_extra(self) -> u32 { if S::BITS == 32 { 0 } else { self.int_bits() + 4 } }

    /// Bits of the integer part of |x| plus one (the argument's size, for
    /// errors that grow with it).
    #[inline]
    fn int_bits(self) -> u32 {
        let m = match self.raw.checked_neg_wide() { Some(n) => if self.raw.is_negative() { n } else { self.raw }, None => return S::BITS };
        (m.shr_bits(F) + <S as WideInt>::one()).bit_length()
    }

    /// Bits of the integer part of a compute value's magnitude plus one.
    #[inline]
    fn compute_int_bits(c: Compute<Self>) -> u32 {
        let m = match c.checked_neg_wide() { Some(n) => if c.is_negative() { n } else { c }, None => return <Compute<Self> as WideInt>::BITS };
        (m.shr_bits(2 * F) + <Compute<Self> as WideInt>::one()).bit_length()
    }

    /// Fallible version of direct_unary.
    #[inline]
    fn try_direct_unary<E: FnOnce(Compute<Self>) -> Compute<Self>>(self, f: E) -> Result<Self, OverflowDetected> {
        let compute = Self::upscale_to_compute(self.raw);
        let result = f(compute);
        Ok(Self { raw: Self::downscale_to_storage(result)? })
    }

    /// e^x
    pub fn exp(self) -> Self { self.try_exp().expect("transcendental overflow") }
    /// ln(x), x > 0
    pub fn ln(self) -> Self { self.try_ln().expect("transcendental overflow") }
    /// sqrt(x), x >= 0
    pub fn sqrt(self) -> Self { self.try_sqrt().expect("transcendental overflow") }
    /// sin(x)
    pub fn sin(self) -> Self { self.try_sin().expect("transcendental overflow") }
    /// cos(x)
    pub fn cos(self) -> Self { self.try_cos().expect("transcendental overflow") }
    /// Fused (sin(x), cos(x)): single range reduction, ~2× faster than separate calls.
    pub fn sincos(self) -> (Self, Self) {
        self.try_sincos().expect("sincos: overflow or domain error")
    }
    /// tan(x) = sin(x) / cos(x): direct composition, no FASC
    pub fn tan(self) -> Self { self.try_tan().expect("tan overflow") }
    /// atan(x)
    pub fn atan(self) -> Self { self.try_atan().expect("transcendental overflow") }
    /// asin(x) = atan(x / sqrt(1 - x^2)), |x| <= 1: direct composition
    pub fn asin(self) -> Self { self.try_asin().expect("asin overflow") }
    /// acos(x) = pi/2 - asin(x), |x| <= 1: direct composition
    pub fn acos(self) -> Self { self.try_acos().expect("acos overflow") }
    /// sinh(x) = (exp(x) - exp(-x)) / 2, correctly rounded (panics where the
    /// result leaves storage; `try_sinh` returns the error)
    pub fn sinh(self) -> Self { self.try_sinh().expect("sinh overflow") }
    /// cosh(x) = (exp(x) + exp(-x)) / 2, correctly rounded (panics where the
    /// result leaves storage; `try_cosh` returns the error)
    pub fn cosh(self) -> Self { self.try_cosh().expect("cosh overflow") }
    /// Fused (sinh(x), cosh(x)): single shared exp-pair evaluation at compute tier.
    ///
    /// ~2× faster than separate `sinh` + `cosh` (2 exp calls instead of 4).
    /// More importantly, sinh and cosh share the same `(exp(x), exp(-x))` pair,
    /// so their rounding bias is **correlated**: downstream expressions like
    /// `cosh(θ)·p + (sinh(θ)/θ)·v` see errors that cancel rather than accumulate.
    pub fn sinhcosh(self) -> (Self, Self) {
        self.try_sinhcosh().expect("sinhcosh: overflow or domain error")
    }
    /// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1): direct composition
    pub fn tanh(self) -> Self { self.try_tanh().expect("tanh overflow") }
    /// asinh(x) = ln(x + sqrt(x^2 + 1)): direct composition
    pub fn asinh(self) -> Self { self.try_asinh().expect("asinh overflow") }
    /// acosh(x) = ln(x + sqrt(x^2 - 1)), x >= 1: direct composition
    pub fn acosh(self) -> Self { self.try_acosh().expect("acosh overflow") }
    /// atanh(x) = ln((1+x)/(1-x)) / 2, |x| < 1: direct composition
    pub fn atanh(self) -> Self { self.try_atanh().expect("atanh overflow") }

    /// `x^y` for a positive base: the engine composition `exp(y ln x)`,
    /// decided by the sure path when it lands near a rounding boundary (the
    /// exponent's error grows with |y|, hence the extra margin bits). `None`
    /// for a base that is not positive.
    fn pow_positive_base(self, exponent: Self) -> Option<Result<Self, OverflowDetected>> {
        if !(self.raw > S::ZERO) { return None; }
        let xc = Self::upscale_to_compute(self.raw);
        let yc = Self::upscale_to_compute(exponent.raw);
        let sure = || sure::pow_to_raw::<S>(self.raw, exponent.raw, F);
        let y_ln_x = match Self::compute_checked_multiply(yc, Self::direct_ln(xc)) {
            Ok(v) => v,
            Err(_) => return Some(sure().map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow)),
        };
        let e = Self::direct_exp(y_ln_x);
        if Self::exp_sentinel_reached(&e) {
            return Some(sure().map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow));
        }
        let y_mag = if exponent.raw.is_negative() { exponent.raw.checked_neg_wide().unwrap_or(exponent.raw) } else { exponent.raw };
        let extra = (y_mag.shr_bits(F) + <S as WideInt>::one()).bit_length() + 1;
        Some(Self::settle(e, extra, sure).map(|raw| Self { raw }))
    }

    /// x^y = exp(y * ln(x)), correctly rounded for a positive base
    pub fn pow(self, exponent: Self) -> Self {
        if let Some(r) = self.pow_positive_base(exponent) { return r.expect("pow overflow"); }
        let xc = Self::upscale_to_compute(self.raw);
        let yc = Self::upscale_to_compute(exponent.raw);
        let ln_x = Self::direct_ln(xc);
        let y_ln_x = Self::compute_multiply(yc, ln_x);
        Self { raw: Self::downscale_to_storage(Self::direct_exp(y_ln_x)).expect("pow overflow") }
    }

    /// atan2(self=y, x): direct binary engine
    pub fn atan2(self, x: Self) -> Self {
        self.atan2_settled(x).expect("atan2 overflow")
    }

    /// atan2 by the engine, decided by the sure path near a rounding
    /// boundary. (0, 0) is a `DomainError`.
    fn atan2_settled(self, x: Self) -> Result<Self, OverflowDetected> {
        if self.raw == S::ZERO && x.raw == S::ZERO { return Err(OverflowDetected::DomainError); }
        let v = Self::direct_atan2(Self::upscale_to_compute(self.raw), Self::upscale_to_compute(x.raw));
        Ok(Self { raw: Self::settle(v, 4, || sure::atan2_to_raw::<S>(self.raw, x.raw, F))? })
    }

    // ========================================================================
    // UGOD-aware try_* transcendentals — return Result instead of panicking
    // ========================================================================

    /// Fallible e^x: returns `Err(TierOverflow)` if result exceeds storage tier.
    pub fn try_exp(self) -> Result<Self, OverflowDetected> {
        let c = Self::direct_exp(Self::upscale_to_compute(self.raw));
        Ok(Self { raw: Self::settle(c, 0, || sure::exp_raw_to_raw::<S>(self.raw, F))? })
    }
    /// Fallible ln(x): returns `Err(DomainError)` if x <= 0.
    pub fn try_ln(self) -> Result<Self, OverflowDetected> {
        // Domain check before the engine: the raw engine is infallible and
        // signals out-of-domain input with a MIN sentinel that downscale
        // would misreport as TierOverflow.
        if self <= Self::ZERO { return Err(OverflowDetected::DomainError); }
        let v = Self::direct_ln(Self::upscale_to_compute(self.raw));
        Ok(Self { raw: Self::settle(v, 2, || sure::ln_to_raw::<S>(self.raw, F))? })
    }
    /// 1/√x computed at compute tier without materializing √x at storage.
    ///
    /// Reciprocal norms (`1/‖v‖`) are the dominant consumer: one `inv_sqrt`
    /// plus N multiplies replaces N per-component divisions in normalization.
    /// Both the square root and the reciprocal stay at tier N+1; the single
    /// rounding happens at the final downscale.
    ///
    /// # Panics
    /// Panics if `x <= 0`, or if `1/√x` does not fit the storage tier.
    pub fn inv_sqrt(self) -> Self {
        self.try_inv_sqrt().expect("inv_sqrt: domain error or overflow")
    }

    /// Fallible 1/√x: `Err(DomainError)` if x <= 0, `Err(TierOverflow)` if
    /// the result does not fit the storage tier.
    pub fn try_inv_sqrt(self) -> Result<Self, OverflowDetected> {
        if self <= Self::ZERO { return Err(OverflowDetected::DomainError); }
        let s = Self::sqrt_at_compute_tier(Self::upscale_to_compute(self.raw));
        let inv = Self::compute_divide(Self::make_compute_int(1), s)?;
        Ok(Self { raw: Self::downscale_to_storage(inv)? })
    }

    /// Fallible sqrt(x): returns `Err(DomainError)` if x < 0.
    pub fn try_sqrt(self) -> Result<Self, OverflowDetected> {
        if self < Self::ZERO { return Err(OverflowDetected::DomainError); }
        let v = Self::direct_sqrt(Self::upscale_to_compute(self.raw));
        Ok(Self { raw: Self::settle(v, 2, || sure::sqrt_to_raw::<S>(self.raw, F))? })
    }
    /// Fallible sin(x).
    pub fn try_sin(self) -> Result<Self, OverflowDetected> {
        if self.int_bits() > TRIG_ENGINE_INT_BITS { return sure::sin_cos_to_raw::<S>(self.raw, F, false).map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow); }
        let v = Self::direct_sin(Self::upscale_to_compute(self.raw));
        Ok(Self { raw: Self::settle(v, self.trig_extra(), || sure::sin_cos_to_raw::<S>(self.raw, F, false))? })
    }
    /// Fallible cos(x).
    pub fn try_cos(self) -> Result<Self, OverflowDetected> {
        if self.int_bits() > TRIG_ENGINE_INT_BITS { return sure::sin_cos_to_raw::<S>(self.raw, F, true).map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow); }
        let v = Self::direct_cos(Self::upscale_to_compute(self.raw));
        Ok(Self { raw: Self::settle(v, self.trig_extra(), || sure::sin_cos_to_raw::<S>(self.raw, F, true))? })
    }
    /// Fused sin+cos: single shared range reduction at compute tier.
    /// Returns (sin(x), cos(x)). More efficient than separate try_sin + try_cos.
    pub fn try_sincos(self) -> Result<(Self, Self), OverflowDetected> {
        if self.int_bits() > TRIG_ENGINE_INT_BITS { return Ok((self.try_sin()?, self.try_cos()?)); }
        let compute_val = Self::upscale_to_compute(self.raw());
        let (sin_c, cos_c) = Self::sincos_at_compute_tier(compute_val);
        let extra = self.trig_extra();
        Ok((
            Self { raw: Self::settle(sin_c, extra, || sure::sin_cos_to_raw::<S>(self.raw, F, false))? },
            Self { raw: Self::settle(cos_c, extra, || sure::sin_cos_to_raw::<S>(self.raw, F, true))? },
        ))
    }

    /// Fallible fused sinh+cosh: single shared exp-pair at compute tier.
    ///
    /// Returns `Err(TierOverflow)` if either sinh(x) or cosh(x) exceeds the
    /// storage tier (cosh grows fastest: overflows first for large |x|).
    pub fn try_sinhcosh(self) -> Result<(Self, Self), OverflowDetected> {
        let compute_val = Self::upscale_to_compute(self.raw);
        let (sinh_c, cosh_c) = Self::sinhcosh_at_compute_tier(compute_val);
        Ok((
            Self { raw: Self::settle(sinh_c, 0, || sure::sinh_cosh_to_raw::<S>(self.raw, F, false))? },
            Self { raw: Self::settle(cosh_c, 0, || sure::sinh_cosh_to_raw::<S>(self.raw, F, true))? },
        ))
    }


    // Fallible composed transcendentals — direct compute-tier compositions
    // mirroring the infallible methods above (0.5.0 item 2; previously these
    // routed through the FASC pipeline via try_apply_unary(LazyExpr)).
    // Error contract unchanged: DomainError on domain violations (0.4.27),
    // TierOverflow when the result exceeds storage.

    /// Fallible tan(x) = sin(x)/cos(x): `Err(DomainError)` if cos(x) is zero
    /// at the compute tier.
    pub fn try_tan(self) -> Result<Self, OverflowDetected> {
        if self.int_bits() > TRIG_ENGINE_INT_BITS { return sure::tan_to_raw::<S>(self.raw, F).map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow); }
        let c = Self::upscale_to_compute(self.raw);
        let s = Self::direct_sin(c);
        let c_val = Self::direct_cos(c);
        if Self::compute_is_zero(&c_val) { return Err(OverflowDetected::DomainError); }
        let sure = || sure::tan_to_raw::<S>(self.raw, F);
        let t = match Self::compute_checked_divide(s, c_val) {
            Ok(t) => t,
            Err(_) => return Ok(Self { raw: sure().ok_or(OverflowDetected::TierOverflow)? }),
        };
        // sin / cos near a pole multiplies the engines' error by about 1 + tan^2
        let t_int = Self::compute_int_bits(t);
        Ok(Self { raw: Self::settle(t, self.trig_extra() + 2 * t_int + 2, sure)? })
    }
    /// Fallible atan(x).
    pub fn try_atan(self) -> Result<Self, OverflowDetected> {
        let v = Self::direct_atan(Self::upscale_to_compute(self.raw));
        Ok(Self { raw: Self::settle(v, 4, || sure::atan_to_raw::<S>(self.raw, F))? })
    }
    /// Fallible asin(x) = atan(x / sqrt(1 - x²)): `Err(DomainError)` if |x| > 1.
    pub fn try_asin(self) -> Result<Self, OverflowDetected> { self.asin_acos_settled(false) }
    /// Fallible acos(x) = π/2 - asin(x): `Err(DomainError)` if |x| > 1.
    pub fn try_acos(self) -> Result<Self, OverflowDetected> { self.asin_acos_settled(true) }

    /// asin / acos by the engines (atan of x / sqrt(1 - x^2)), decided by
    /// the sure path near a rounding boundary; the error of sqrt(1 - x^2)
    /// grows as 1 / (1 - |x|), hence the extra margin there.
    fn asin_acos_settled(self, acos: bool) -> Result<Self, OverflowDetected> {
        let one = Self::one();
        let neg_one = Self::ZERO - one;
        if self > one || self < neg_one { return Err(OverflowDetected::DomainError); }
        let sure = || sure::asin_acos_to_raw::<S>(self.raw, F, acos);
        if self == one || self == neg_one {
            return Ok(Self { raw: sure().ok_or(OverflowDetected::TierOverflow)? });
        }
        let c = Self::upscale_to_compute(self.raw);
        let x2 = Self::compute_multiply(c, c);
        let denom = Self::direct_sqrt(Self::compute_subtract(Self::compute_one(), x2));
        let asin_c = Self::direct_atan(Self::compute_divide_direct(c, denom));
        let v = if acos { Self::compute_subtract(Self::compute_pi_half(), asin_c) } else { asin_c };
        let mag = if self.raw.is_negative() { -self.raw } else { self.raw };
        let gap = Self::one_storage() - mag;
        let extra = (F + 2).saturating_sub(gap.bit_length()) + 4;
        Ok(Self { raw: Self::settle(v, extra, sure)? })
    }
    /// Fallible sinh(x) = (exp(x) - exp(-x)) / 2: `Err(TierOverflow)` when
    /// the result exceeds the storage tier (a ceiling exp means it already
    /// has, and the ceiling value would downscale to a plausible-wrong max).
    pub fn try_sinh(self) -> Result<Self, OverflowDetected> {
        let c = Self::upscale_to_compute(self.raw);
        let ep = Self::direct_exp(c);
        let en = Self::direct_exp(Self::compute_negate(c));
        let sure = || sure::sinh_cosh_to_raw::<S>(self.raw, F, false);
        if Self::exp_sentinel_reached(&ep) || Self::exp_sentinel_reached(&en) {
            return Ok(Self { raw: sure().ok_or(OverflowDetected::TierOverflow)? });
        }
        Ok(Self { raw: Self::settle(Self::compute_halve(Self::compute_subtract(ep, en)), 0, sure)? })
    }
    /// Fallible cosh(x) = (exp(x) + exp(-x)) / 2: `Err(TierOverflow)` when
    /// the result exceeds the storage tier (a ceiling exp means cosh already
    /// has; the checked add alone cannot see it on wide profiles).
    pub fn try_cosh(self) -> Result<Self, OverflowDetected> {
        let c = Self::upscale_to_compute(self.raw);
        let ep = Self::direct_exp(c);
        let en = Self::direct_exp(Self::compute_negate(c));
        let sure = || sure::sinh_cosh_to_raw::<S>(self.raw, F, true);
        if Self::exp_sentinel_reached(&ep) || Self::exp_sentinel_reached(&en) {
            return Ok(Self { raw: sure().ok_or(OverflowDetected::TierOverflow)? });
        }
        match Self::compute_checked_add(ep, en) {
            Ok(sum) => Ok(Self { raw: Self::settle(Self::compute_halve(sum), 0, sure)? }),
            Err(_) => Ok(Self { raw: sure().ok_or(OverflowDetected::TierOverflow)? }),
        }
    }
    /// Fallible tanh(x) = (exp(2x) - 1) / (exp(2x) + 1). Saturates to exactly
    /// 1 when exp(2x) reaches the compute-tier ceiling (matches `tanh`).
    pub fn try_tanh(self) -> Result<Self, OverflowDetected> {
        let c = Self::upscale_to_compute(self.raw);
        let sure = || sure::tanh_to_raw::<S>(self.raw, F);
        let e2x = Self::direct_exp(Self::compute_add(c, c));
        let one = Self::compute_one();
        let den = match Self::compute_checked_add(e2x, one) {
            Ok(v) => v,
            Err(_) => return Ok(Self { raw: sure().ok_or(OverflowDetected::TierOverflow)? }),
        };
        if Self::exp_sentinel_reached(&e2x) { return Ok(Self { raw: sure().ok_or(OverflowDetected::TierOverflow)? }); }
        let num = Self::compute_subtract(e2x, one);
        Ok(Self { raw: Self::settle(Self::compute_divide_direct(num, den), 4, sure)? })
    }
    /// Fallible asinh(x) = ln(x + sqrt(x² + 1)).
    pub fn try_asinh(self) -> Result<Self, OverflowDetected> {
        // odd: evaluate on |x| (for x < 0, x + sqrt(x^2 + 1) cancels), then
        // negate; the word's minimum has no magnitude, the sure path takes it
        let sure_all = || sure::asinh_to_raw::<S>(self.raw, F);
        let neg = self.raw.is_negative();
        let mag = if neg {
            match self.raw.checked_neg_wide() { Some(m) => m, None => return Ok(Self { raw: sure_all().ok_or(OverflowDetected::TierOverflow)? }) }
        } else { self.raw };
        let c = Self::upscale_to_compute(mag);
        let x2 = match Self::compute_checked_multiply(c, c) {
            Ok(v) => v,
            Err(_) => return Ok(Self { raw: sure_all().ok_or(OverflowDetected::TierOverflow)? }),
        };
        let inner = match Self::compute_checked_add(x2, Self::compute_one()) {
            Ok(v) => Self::direct_sqrt(v),
            Err(_) => return Ok(Self { raw: sure_all().ok_or(OverflowDetected::TierOverflow)? }),
        };
        let v = Self::direct_ln(Self::compute_add(c, inner));
        let r = Self::settle(v, 4, || sure::asinh_to_raw::<S>(mag, F))?;
        Ok(Self { raw: if neg { -r } else { r } })
    }
    /// Fallible acosh(x) = ln(x + sqrt(x² - 1)): `Err(DomainError)` if x < 1.
    pub fn try_acosh(self) -> Result<Self, OverflowDetected> {
        if self < Self::one() { return Err(OverflowDetected::DomainError); }
        let sure = || sure::acosh_to_raw::<S>(self.raw, F);
        let c = Self::upscale_to_compute(self.raw);
        let x2 = match Self::compute_checked_multiply(c, c) {
            Ok(v) => v,
            Err(_) => return Ok(Self { raw: sure().ok_or(OverflowDetected::TierOverflow)? }),
        };
        let inner = Self::direct_sqrt(Self::compute_subtract(x2, Self::compute_one()));
        let v = Self::direct_ln(Self::compute_add(c, inner));
        // sqrt(x^2 - 1) near x = 1 amplifies the error as 1 / sqrt(x - 1)
        let gap = self.raw - Self::one_storage();
        let extra = (F + 2).saturating_sub(gap.bit_length()) + 4;
        Ok(Self { raw: Self::settle(v, extra, sure)? })
    }
    /// Fallible atanh(x) = ln((1+x)/(1-x)) / 2: `Err(DomainError)` if |x| >= 1.
    pub fn try_atanh(self) -> Result<Self, OverflowDetected> {
        let one = Self::one();
        if self >= one || self <= Self::ZERO - one { return Err(OverflowDetected::DomainError); }
        let c = Self::upscale_to_compute(self.raw);
        let one_c = Self::compute_one();
        let sure = || sure::atanh_to_raw::<S>(self.raw, F);
        let ratio = match Self::compute_checked_divide(Self::compute_add(one_c, c), Self::compute_subtract(one_c, c)) {
            Ok(r) => r,
            Err(_) => return Ok(Self { raw: sure().ok_or(OverflowDetected::TierOverflow)? }),
        };
        // the ratio's relative error grows as 1 / (1 - |x|)
        let mag = if self.raw.is_negative() { -self.raw } else { self.raw };
        let gap = Self::one_storage() - mag;
        let extra = (F + 2).saturating_sub(gap.bit_length()) + 4;
        Ok(Self { raw: Self::settle(Self::compute_halve(Self::direct_ln(ratio)), extra, sure)? })
    }



    // ========================================================================
    // Internal helpers
    // ========================================================================





}

// ============================================================================
// Display
// ============================================================================

impl<S: Storage, const F: u32> Fixed<S, F>
where
    Self: FascFormat + FixedNum<Storage = S>,
{
    /// Parse from a decimal string (e.g., "3.14159", "1e-06").
    ///
    /// Panics where [`try_from_str`](Self::try_from_str) returns an error.
    pub fn from_str(s: &str) -> Self {
        Self::try_from_str(s).expect("FixedPoint::from_str: parse failed")
    }
    /// Parse from a string without panicking.
    ///
    /// A decimal literal, optionally in exponent notation
    /// (`[+-]? digits [. digits] [e [+-]? digits]`, also `.5` and `5.`,
    /// surrounding whitespace ignored), is converted exactly with integer
    /// arithmetic and rounded once to the nearest storage value, ties toward
    /// +infinity. Any number of digits is accepted. Any other string (hex
    /// `0x..`, binary `0b..`, ternary `0t..`, fractions `1/3`, repeating
    /// decimals `0.3...`, named constants `pi`) is parsed by the canonical
    /// evaluator in binary mode, as before.
    ///
    /// `Err(ParseError)` for a malformed string, `Err(TierOverflow)` when the
    /// value does not fit storage.
    pub fn try_from_str(s: &str) -> Result<Self, OverflowDetected> {
        match <Self as FascFormat>::literal_to_storage(s) {
            Some(converted) => converted.map(|raw| Self { raw }),
            None => Self::try_from_str_canonical(s),
        }
    }
    /// The literal through the canonical evaluator in binary:binary mode.
    fn try_from_str_canonical(s: &str) -> Result<Self, OverflowDetected> {
        use crate::fixed_point::universal::fasc::mode;

        let old_mode = mode::get_mode();
        mode::set_mode(mode::GmathMode {
            compute: mode::ComputeMode::Binary,
            output: mode::OutputMode::Binary,
        });
        let result = gmath_parse_in::<Self>(s).and_then(|expr| evaluate_in(&expr));
        mode::set_mode(old_mode);

        Self::try_from_stack_value(result?)
    }
    /// Fallible x^y = exp(y * ln(x)).
    pub fn try_pow(self, exponent: Self) -> Result<Self, OverflowDetected> {
        if let Some(r) = self.pow_positive_base(exponent) { return r; }
        let sv1 = self.to_stack_value();
        let sv2 = exponent.to_stack_value();
        let expr = LazyExpr::from(sv1).pow(LazyExpr::from(sv2));
        let result = evaluate_in(&expr)?;
        Self::try_from_stack_value(result)
    }
    /// Fallible atan2(self=y, x).
    pub fn try_atan2(self, x: Self) -> Result<Self, OverflowDetected> {
        if !(self.raw == S::ZERO && x.raw == S::ZERO) { return self.atan2_settled(x); }
        let sv_y = self.to_stack_value();
        let sv_x = x.to_stack_value();
        let expr = LazyExpr::from(sv_y).atan2(LazyExpr::from(sv_x));
        let result = evaluate_in(&expr)?;
        Self::try_from_stack_value(result)
    }
    #[inline]
    pub(crate) fn to_stack_value(self) -> StackValue<Self> {
        StackValue::<Self>::Binary(<Self as FixedNum>::STORAGE_TIER, self.raw, CompactShadow::None)
    }
    pub(crate) fn try_from_stack_value(sv: StackValue<Self>) -> Result<Self, OverflowDetected> {
        match sv.as_binary_storage() {
            Some(raw) => Ok(Self { raw }),
            None => {
                // Non-binary domain — force conversion by adding binary zero
                let zero_sv = StackValue::<Self>::Binary(<Self as FixedNum>::STORAGE_TIER, Self::ZERO.raw, CompactShadow::None);
                let expr = LazyExpr::from(sv) + LazyExpr::from(zero_sv);
                let result = evaluate_in(&expr)?;
                result.as_binary_storage()
                    .map(|raw| Self { raw })
                    .ok_or(OverflowDetected::InvalidInput)
            }
        }
    }
    #[allow(dead_code)]
    fn apply_unary(self, f: fn(LazyExpr<Self>) -> LazyExpr<Self>) -> Self {
        self.try_apply_unary(f).expect("transcendental: overflow or domain error")
    }
    fn try_apply_unary(self, f: fn(LazyExpr<Self>) -> LazyExpr<Self>) -> Result<Self, OverflowDetected> {
        let sv = self.to_stack_value();
        let expr = f(LazyExpr::from(sv));
        let result = evaluate_in(&expr)?;
        Self::try_from_stack_value(result)
    }
}

impl<const F: u32> Fixed<i32, F> {
    /// Fused sin+cos for wide-range angles that exceed storage-tier integer range.
    ///
    /// The angle is a raw i64 in **Q32.32 fixed-point format** (32 integer bits,
    /// 32 fractional bits). This gives ±2.1 billion integer range regardless of
    /// the profile's FRAC_BITS, covering all practical RoPE frequencies.
    ///
    /// Identical to [`sincos_wide_q64`](Self::sincos_wide_q64) of the angle
    /// shifted to Q64.64. Output sin/cos always fits in [-1, 1].
    pub fn sincos_wide(angle_q32_32: i64) -> (Self, Self) {
        // Fixed Q32.32→Q64.64 upscale (always 32, independent of profile FRAC_BITS)
        Self::sincos_wide_q64((angle_q32_32 as i128) << 32)
    }

    /// Fused sin+cos of a Q64.64 angle (`radians * 2^64`), rounded to storage.
    ///
    /// Computes [`wide::sincos_q64`](crate::fixed_point::wide::sincos_q64) and
    /// rounds each result once to storage. The Q64.64 angle needs no caller-side
    /// truncation to Q32.32.
    pub fn sincos_wide_q64(angle_q64: i128) -> (Self, Self) {
        let (sin_q64, cos_q64) = sincos_q64_64_native(angle_q64);
        // Downscale Q64.64 -> storage tier: shift = 64 - F
        let shift = 64 - F;
        let sin_round = (sin_q64 >> (shift - 1)) & 1;
        let cos_round = (cos_q64 >> (shift - 1)) & 1;
        let sin_raw = ((sin_q64 >> shift) + sin_round) as i32;
        let cos_raw = ((cos_q64 >> shift) + cos_round) as i32;
        (Self::from_raw(sin_raw), Self::from_raw(cos_raw))
    }
}

impl Fixed<i64, 32> {
    /// Fused sin+cos for wide-range angles that exceed storage-tier integer range.
    ///
    /// The angle is a raw i64 in **Q32.32 fixed-point format** (32 integer bits,
    /// 32 fractional bits). This gives ±2.1 billion integer range regardless of
    /// the profile's FRAC_BITS, covering all practical RoPE frequencies.
    ///
    /// Identical to [`sincos_wide_q64`](Self::sincos_wide_q64) of the angle
    /// shifted to Q64.64. Output sin/cos always fits in [-1, 1].
    pub fn sincos_wide(angle_q32_32: i64) -> (Self, Self) {
        // Fixed Q32.32→Q64.64 upscale (always 32, independent of profile FRAC_BITS)
        Self::sincos_wide_q64((angle_q32_32 as i128) << 32)
    }

    /// Fused sin+cos of a Q64.64 angle (`radians * 2^64`), rounded to storage.
    ///
    /// Computes [`wide::sincos_q64`](crate::fixed_point::wide::sincos_q64) and
    /// rounds each result once to storage.
    pub fn sincos_wide_q64(angle_q64: i128) -> (Self, Self) {
        let (sin_q64, cos_q64) = sincos_q64_64_native(angle_q64);
        // Q64.64 -> Q32.32: shift right 32 with rounding
        let sin_round = (sin_q64 >> 31) & 1;
        let cos_round = (cos_q64 >> 31) & 1;
        let sin_raw = ((sin_q64 >> 32) + sin_round) as i64;
        let cos_raw = ((cos_q64 >> 32) + cos_round) as i64;
        (Self::from_raw(sin_raw), Self::from_raw(cos_raw))
    }
}

impl<S: Storage, const F: u32> fmt::Display for Fixed<S, F>
where
    Self: FascFormat + FixedNum<Storage = S>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sv = self.to_stack_value();
        fmt::Display::fmt(&sv, f)
    }
}

impl<S: Storage, const F: u32> Default for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

// ============================================================================
// Arithmetic operators — direct Q-format integer ops (no FASC overhead)
// ============================================================================

// The operators never wrap: a result outside the storage range panics, like
// every other infallible path in the crate. Before 0.6.4 `+ - neg` used
// wrapping arithmetic and `* /` narrowed their exact wide results with
// truncating casts, so an overflow returned a plausible wrong value. The
// `try_` methods return `Err(TierOverflow)` / `Err(DivisionByZero)` instead;
// for automatic promotion to a wider tier or an exact rational, evaluate the
// expression through the canonical API (`gmath` / `evaluate`), where UGOD
// applies.

impl<S: Storage, const F: u32> Add for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        // Branch on the checked word directly: building the `Result` here
        // cost a materialised enum per element in vector loops.
        match self.raw.checked_add_wide(rhs.raw) {
            Some(raw) => Self { raw },
            None => panic!("FixedPoint: addition overflow"),
        }
    }
}

impl<S: Storage, const F: u32> Sub for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        // Branch on the checked word directly: building the `Result` here
        // cost a materialised enum per element in vector loops.
        match self.raw.checked_sub_wide(rhs.raw) {
            Some(raw) => Self { raw },
            None => panic!("FixedPoint: subtraction overflow"),
        }
    }
}

impl<S: Storage, const F: u32> Mul for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.try_mul(rhs).expect("FixedPoint: multiplication overflow")
    }
}

impl<S: Storage, const F: u32> Div for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        match self.try_div(rhs) {
            Ok(v) => v,
            Err(OverflowDetected::DivisionByZero) => panic!("FixedPoint: division by zero"),
            Err(_) => panic!("FixedPoint: division overflow"),
        }
    }
}

impl<S: Storage, const F: u32> Neg for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.try_neg().expect("FixedPoint: negation overflow")
    }
}

impl<S: Storage, const F: u32> Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    /// `self + rhs`, `Err(TierOverflow)` when the sum leaves storage.
    #[inline]
    pub fn try_add(self, rhs: Self) -> Result<Self, OverflowDetected> {
        self.raw.checked_add_wide(rhs.raw).map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow)
    }

    /// `self - rhs`, `Err(TierOverflow)` when the difference leaves storage.
    #[inline]
    pub fn try_sub(self, rhs: Self) -> Result<Self, OverflowDetected> {
        self.raw.checked_sub_wide(rhs.raw).map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow)
    }

    /// `-self`, `Err(TierOverflow)` for the storage minimum (no positive twin).
    #[inline]
    pub fn try_neg(self) -> Result<Self, OverflowDetected> {
        self.raw.checked_neg_wide().map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow)
    }

    /// `self * rhs`, rounded to nearest (ties toward +infinity) from the exact
    /// product; `Err(TierOverflow)` when it leaves storage.
    #[inline]
    pub fn try_mul(self, rhs: Self) -> Result<Self, OverflowDetected> {
        <Self as FixedNum>::try_fixed_multiply(self.raw, rhs.raw).map(|raw| Self { raw })
    }

    /// `self / rhs`, rounded to nearest (ties toward +infinity) from the exact
    /// quotient; `Err(DivisionByZero)` or `Err(TierOverflow)`.
    #[inline]
    pub fn try_div(self, rhs: Self) -> Result<Self, OverflowDetected> {
        <Self as FixedNum>::try_fixed_divide(self.raw, rhs.raw).map(|raw| Self { raw })
    }
}

impl<S: Storage, const F: u32> AddAssign for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl<S: Storage, const F: u32> SubAssign for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl<S: Storage, const F: u32> MulAssign for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl<S: Storage, const F: u32> DivAssign for Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

#[cfg(test)]
mod literal_parser_tests {
    use super::FixedPoint;

    /// The exact parser agrees bit for bit with the canonical evaluator on
    /// every plain decimal and integer literal, in range or not: any digit
    /// count (before 0.6.4 the canonical parser truncated past 38 digits and
    /// wrapped `value * 10^dp` into i32 decimal storage on realtime).
    #[test]
    fn exact_parser_matches_the_canonical_path() {
        let int_digits = crate::fixed_point::frac_config::INTEGER_BITS as u64 * 30103 / 100000 + 2;
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |bound: u64| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) % bound
        };
        let mut checked = 0;
        for _ in 0..20_000 {
            let mut s = String::new();
            if next(2) == 0 { s.push('-'); }
            let int_len = 1 + next(int_digits);
            for _ in 0..int_len { s.push((b'0' + next(10) as u8) as char); }
            let frac_len = next(46);
            if frac_len > 0 || next(2) == 0 {
                s.push('.');
                for _ in 0..frac_len.max(1) { s.push((b'0' + next(10) as u8) as char); }
            }
            let exact = FixedPoint::try_from_str(&s);
            let canonical = FixedPoint::try_from_str_canonical(&s);
            match (&exact, &canonical) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "{s}"),
                (Err(_), Err(_)) => {}
                _ => panic!("{s}: exact {exact:?}, canonical {canonical:?}"),
            }
            checked += exact.is_ok() as u32;
        }
        // most literals fit (at 30 fraction bits only |v| < 2 does)
        assert!(checked > 2_000, "{checked}");
    }
}
