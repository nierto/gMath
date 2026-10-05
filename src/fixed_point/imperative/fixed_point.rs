//! FixedPoint: Copy-able binary fixed-point numeric type for imperative computation.
//!
//! Wraps the raw `BinaryStorage` Q-format integer, providing direct arithmetic
//! operators and transcendental methods (routed through FASC).

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

use crate::fixed_point::canonical::{
    LazyExpr, StackValue, evaluate, gmath_parse, CompactShadow,
};
use crate::fixed_point::universal::fasc::stack_evaluator::{
    BinaryStorage, ComputeStorage, upscale_to_compute, downscale_to_storage,
};
pub use crate::fixed_point::core_types::errors::OverflowDetected;

#[cfg(table_format = "q64_64")]
use crate::fixed_point::I256;

#[cfg(table_format = "q128_128")]
use crate::fixed_point::{I256, I512};

#[cfg(table_format = "q256_256")]
use crate::fixed_point::{I512, I1024};

// No extra wide-int imports needed for q32_32 (i64 storage, i128 intermediate)
// No extra wide-int imports needed for q16_16 (i32 storage, i64 intermediate)

// ============================================================================
// Profile-dependent constants
// ============================================================================

#[cfg(table_format = "q16_16")]
const STORAGE_TIER: u8 = 1;
#[cfg(table_format = "q32_32")]
const STORAGE_TIER: u8 = 2;
#[cfg(table_format = "q64_64")]
const STORAGE_TIER: u8 = 3;
#[cfg(table_format = "q128_128")]
const STORAGE_TIER: u8 = 4;
#[cfg(table_format = "q256_256")]
const STORAGE_TIER: u8 = 5;

#[cfg(table_format = "q16_16")]
const FRAC_BITS: i32 = crate::fixed_point::frac_config::FRAC_BITS as i32;
#[cfg(table_format = "q32_32")]
const FRAC_BITS: i32 = 32;
#[cfg(table_format = "q64_64")]
const FRAC_BITS: i32 = 64;
#[cfg(table_format = "q128_128")]
const FRAC_BITS: i32 = 128;
#[cfg(table_format = "q256_256")]
const FRAC_BITS: i32 = 256;

// ============================================================================
// Direct binary engine wrappers (bypass FASC pipeline)
// Each function: ComputeStorage → ComputeStorage via the profile's engine.
// ============================================================================

fn direct_exp(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i64(x) }
    #[cfg(table_format = "q32_32")]
    { crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i128(x) }
    #[cfg(table_format = "q64_64")]
    { crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i256(x) }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i512(x) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i1024(x) }
}

fn direct_ln(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { crate::fixed_point::domains::binary_fixed::transcendental::ln_binary_i64(x) }
    #[cfg(table_format = "q32_32")]
    { crate::fixed_point::domains::binary_fixed::transcendental::ln_binary_i128(x) }
    #[cfg(table_format = "q64_64")]
    { crate::fixed_point::domains::binary_fixed::transcendental::ln_binary_i256(x) }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::domains::binary_fixed::transcendental::ln_binary_i512(x) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::transcendental::ln_binary_i1024(x) }
}

fn direct_sqrt(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sqrt_binary_i64(x) }
    #[cfg(table_format = "q32_32")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sqrt_binary_i128(x) }
    #[cfg(table_format = "q64_64")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sqrt_binary_i256(x) }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sqrt_binary_i512(x) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sqrt_binary_i1024(x) }
}

fn direct_sin(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sin_binary_i64(x) }
    #[cfg(table_format = "q32_32")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sin_binary_i128(x) }
    #[cfg(table_format = "q64_64")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sin_binary_i256(x) }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sin_binary_i512(x) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::transcendental::sin_binary_i1024(x) }
}

fn direct_cos(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { crate::fixed_point::domains::binary_fixed::transcendental::cos_binary_i64(x) }
    #[cfg(table_format = "q32_32")]
    { crate::fixed_point::domains::binary_fixed::transcendental::cos_binary_i128(x) }
    #[cfg(table_format = "q64_64")]
    { crate::fixed_point::domains::binary_fixed::transcendental::cos_binary_i256(x) }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::domains::binary_fixed::transcendental::cos_binary_i512(x) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::transcendental::cos_binary_i1024(x) }
}

fn direct_atan2(y: ComputeStorage, x: ComputeStorage) -> ComputeStorage {
    // realtime: the Q(2F) compute raws go through the Q64.64 engine and back
    // (0.6.3 passed them to the Q64.64 engine unscaled and truncated its
    // Q64.64 angle into the Q(2F) raw: atan2(1, 1) panicked on every split)
    #[cfg(table_format = "q16_16")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan2_compute_tier_i64(y, x) }
    #[cfg(table_format = "q32_32")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan2_binary_i128(y, x) }
    #[cfg(table_format = "q64_64")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan2_binary_i256(y, x) }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan2_binary_i512(y, x) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan2_binary_i1024(y, x) }
}

// Compute-tier arithmetic helpers for direct transcendental composition.
// These operate on ComputeStorage without FASC overhead.
use crate::fixed_point::universal::fasc::stack_evaluator::{
    compute_add, compute_subtract, compute_negate, compute_multiply, compute_divide, compute_halve,
};
use crate::fixed_point::universal::fasc::stack_evaluator::compute::compute_checked_add;

#[inline] fn compute_add_direct(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage { compute_add(a, b) }
#[inline] fn compute_sub_direct(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage { compute_subtract(a, b) }
#[inline] fn compute_neg_direct(a: ComputeStorage) -> ComputeStorage { compute_negate(a) }
#[inline] fn compute_mul_direct(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage { compute_multiply(a, b) }
#[inline] fn compute_divide_direct(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage {
    compute_divide(a, b).expect("division by zero in transcendental composition")
}
#[inline] fn compute_halve_direct(a: ComputeStorage) -> ComputeStorage { compute_halve(a) }

/// True when an exp engine result sits at its overflow sentinel. sinh/cosh
/// built on such a value would be silently wrong: on q128_128 the sentinel
/// equals the storage maximum, so the final downscale does NOT catch it
/// (0.5.0 item 2 find). Shares the per-profile predicate with the FASC
/// pipeline's `exp_at_compute_ceiling`.
#[inline]
fn exp_ceilinged(v: &ComputeStorage) -> bool {
    use crate::fixed_point::universal::fasc::stack_evaluator::compute::exp_sentinel_reached;
    exp_sentinel_reached(v)
}

/// 1.0 at compute tier
fn compute_one() -> ComputeStorage { upscale_to_compute(one_storage()) }

/// pi/2 at compute tier for acos.
/// Upscales from the available pi_half constant to the profile's compute tier.
fn compute_pi_half() -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    {
        use crate::fixed_point::frac_config;
        // pi_half_i128() is Q64.64; the realtime compute tier holds
        // 2·FRAC_BITS fractional bits. The old bare `as i64` cast WRAPPED
        // (π/2·2^64 > i64::MAX), silently corrupting every imperative
        // acos on this profile (0.5.0 item 2 find). Shift down with
        // nearest rounding instead.
        let shift = 64 - (frac_config::COMPUTE_FRAC_BITS as u32);
        let q64 = crate::fixed_point::domains::binary_fixed::transcendental::pi_half_i128();
        let round = (q64 >> (shift - 1)) & 1;
        ((q64 >> shift) + round) as i64
    }
    #[cfg(table_format = "q32_32")]
    { crate::fixed_point::domains::binary_fixed::transcendental::pi_half_i128() }
    #[cfg(table_format = "q64_64")]
    { upscale_to_compute(crate::fixed_point::domains::binary_fixed::transcendental::pi_half_i128()) }
    #[cfg(table_format = "q128_128")]
    { upscale_to_compute(crate::fixed_point::domains::binary_fixed::transcendental::pi_half_i256()) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::transcendental::pi_half_i1024() }
}

/// 1.0 at storage tier
fn one_storage() -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    { 1i32 << crate::fixed_point::frac_config::FRAC_BITS }
    #[cfg(table_format = "q32_32")]
    { 1i64 << 32 }
    #[cfg(table_format = "q64_64")]
    { 1i128 << 64 }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::i256::I256::from_i128(1) << 128 }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::i512::I512::from_i128(1) << 256 }
}

fn direct_atan(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan_binary_i64(x) }
    #[cfg(table_format = "q32_32")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan_binary_i128(x) }
    #[cfg(table_format = "q64_64")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan_binary_i256(x) }
    #[cfg(table_format = "q128_128")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan_binary_i512(x) }
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan_binary_i1024(x) }
}

// ============================================================================
// FixedPoint struct
// ============================================================================

/// A fixed-point number stored as a raw Q-format integer.
///
/// Profile-dependent size:
/// - `embedded` (Q64.64): 16 bytes (i128)
/// - `balanced` (Q128.128): 32 bytes (I256)
/// - `scientific` (Q256.256): 64 bytes (I512)
///
/// Arithmetic is performed directly on the raw Q-format values.
/// Transcendentals route through FASC at tier N+1.
///
/// **Layout guarantee (0.6.5):** `FixedPoint` is `#[repr(transparent)]` over
/// its raw storage integer, so a `FixedPoint` and its raw value have the same
/// size, alignment and bit pattern, and every bit pattern of the raw type is a
/// valid `FixedPoint`. [`FixedPoint::raw_slice`] and
/// [`FixedPoint::from_raw_slice`] reinterpret slices without copying.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct FixedPoint {
    raw: BinaryStorage,
}

// Manual trait impls — delegate to BinaryStorage (which implements all of these)
impl PartialEq for FixedPoint {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for FixedPoint {}

impl PartialOrd for FixedPoint {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FixedPoint {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.raw.cmp(&other.raw)
    }
}

// ============================================================================
// Core constructors and accessors
// ============================================================================

impl FixedPoint {
    /// Zero constant.
    #[cfg(table_format = "q16_16")]
    pub const ZERO: Self = Self { raw: 0i32 };
    #[cfg(table_format = "q32_32")]
    pub const ZERO: Self = Self { raw: 0i64 };
    #[cfg(table_format = "q64_64")]
    pub const ZERO: Self = Self { raw: 0i128 };
    #[cfg(table_format = "q128_128")]
    pub const ZERO: Self = Self { raw: I256::zero() };
    #[cfg(table_format = "q256_256")]
    pub const ZERO: Self = Self { raw: I512::zero() };

    /// One (1.0) in Q-format.
    #[inline]
    pub fn one() -> Self {
        #[cfg(table_format = "q16_16")]
        { Self { raw: 1i32 << FRAC_BITS } }
        #[cfg(table_format = "q32_32")]
        { Self { raw: 1i64 << 32 } }
        #[cfg(table_format = "q64_64")]
        { Self { raw: 1i128 << 64 } }
        #[cfg(table_format = "q128_128")]
        { Self { raw: I256::from_i128(1) << 128usize } }
        #[cfg(table_format = "q256_256")]
        { Self { raw: I512::from_i128(1) << 256usize } }
    }

    /// Create from raw Q-format storage.
    #[inline]
    pub fn from_raw(raw: BinaryStorage) -> Self {
        Self { raw }
    }

    /// Access the raw Q-format storage.
    #[inline]
    pub fn raw(self) -> BinaryStorage {
        self.raw
    }

    /// View a slice of values as their raw storage integers, without copying.
    #[inline]
    pub fn raw_slice(values: &[Self]) -> &[BinaryStorage] {
        // SAFETY: `FixedPoint` is `repr(transparent)` over `BinaryStorage`:
        // same size, alignment and validity, so the slice layouts are equal.
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const BinaryStorage, values.len()) }
    }

    /// View a slice of raw storage integers as values, without copying.
    #[inline]
    pub fn from_raw_slice(raws: &[BinaryStorage]) -> &[Self] {
        // SAFETY: as in `raw_slice`; every raw bit pattern is a valid value.
        unsafe { std::slice::from_raw_parts(raws.as_ptr() as *const Self, raws.len()) }
    }

    /// Mutable form of [`raw_slice`](Self::raw_slice).
    #[inline]
    pub fn raw_slice_mut(values: &mut [Self]) -> &mut [BinaryStorage] {
        // SAFETY: as in `raw_slice`; the borrow is exclusive.
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut BinaryStorage, values.len()) }
    }

    /// Mutable form of [`from_raw_slice`](Self::from_raw_slice).
    #[inline]
    pub fn from_raw_slice_mut(raws: &mut [BinaryStorage]) -> &mut [Self] {
        // SAFETY: as in `raw_slice`; the borrow is exclusive.
        unsafe { std::slice::from_raw_parts_mut(raws.as_mut_ptr() as *mut Self, raws.len()) }
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
        #[cfg(table_format = "q16_16")]
        {
            let raw = (v as i64) << FRAC_BITS;
            i32::try_from(raw).map(|raw| Self { raw }).map_err(|_| OverflowDetected::TierOverflow)
        }
        #[cfg(table_format = "q32_32")]
        { Ok(Self { raw: (v as i64) << 32 }) }
        #[cfg(table_format = "q64_64")]
        { Ok(Self { raw: (v as i128) << 64 }) }
        #[cfg(table_format = "q128_128")]
        { Ok(Self { raw: I256::from_i128(v as i128) << 128usize }) }
        #[cfg(table_format = "q256_256")]
        { Ok(Self { raw: I512::from_i128(v as i128) << 256usize }) }
    }

    /// `self * n` for a count or dimension `n`: exact at the compute tier (the
    /// count need not fit the storage range), panics if the product does not.
    pub(crate) fn mul_count(self, n: usize) -> Self {
        use crate::fixed_point::universal::fasc::stack_evaluator::compute::compute_mul_div_int;
        let c = compute_mul_div_int(upscale_to_compute(self.raw), n as i64, 1)
            .expect("FixedPoint::mul_count: product exceeds the compute tier");
        Self { raw: downscale_to_storage(c).expect("FixedPoint::mul_count: product exceeds storage") }
    }

    /// `self / n` for a count or dimension `n > 0`: the storage division when
    /// `n` fits the storage range (bit-identical to `self / from_int(n)`),
    /// the compute tier when it does not (a realtime count past 2^(31 - F)).
    pub(crate) fn div_count(self, n: usize) -> Self {
        match i32::try_from(n).ok().and_then(|n| Self::try_from_int(n).ok()) {
            Some(nf) => self / nf,
            None => {
                use crate::fixed_point::universal::fasc::stack_evaluator::compute::compute_mul_div_int;
                let c = compute_mul_div_int(upscale_to_compute(self.raw), 1, n as i64)
                    .expect("FixedPoint::div_count: n > 0");
                Self { raw: downscale_to_storage(c).expect("FixedPoint::div_count: quotient fits storage") }
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
        #[cfg(table_format = "q16_16")]
        { Ok(self.raw >> FRAC_BITS) }
        #[cfg(table_format = "q32_32")]
        { i32::try_from(self.raw >> 32).map_err(|_| OverflowDetected::TierOverflow) }
        #[cfg(table_format = "q64_64")]
        { i32::try_from(self.raw >> 64).map_err(|_| OverflowDetected::TierOverflow) }
        #[cfg(table_format = "q128_128")]
        { i32::try_from((self.raw >> 128u32).as_i128()).map_err(|_| OverflowDetected::TierOverflow) }
        #[cfg(table_format = "q256_256")]
        {
            let floor = self.raw >> 256usize;
            if !floor.fits_in_i128() {
                return Err(OverflowDetected::TierOverflow);
            }
            i32::try_from(floor.as_i128()).map_err(|_| OverflowDetected::TierOverflow)
        }
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        if self.is_negative() { -self } else { self }
    }

    /// Check if negative.
    #[inline]
    pub fn is_negative(self) -> bool {
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        { self.raw < 0 }
        #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
        { self.raw.is_negative() }
    }

    /// Check if zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        { self.raw == 0 }
        #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
        { self.raw.is_zero() }
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
        Ok(Self { raw: Self::truncated_raw(mantissa, exp_offset + FRAC_BITS, negative)? })
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
        Ok(Self { raw: Self::truncated_raw(mantissa, exp_offset + FRAC_BITS, negative)? })
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
        f32::from_bits(self.float_bits(23, 8) as u32)
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
        f64::from_bits(self.float_bits(52, 11))
    }

    /// IEEE 754 bits of this value in a format with `fraction_bits` stored
    /// fraction bits and `exponent_bits` exponent bits.
    fn float_bits(self, fraction_bits: u32, exponent_bits: u32) -> u64 {
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        {
            let magnitude = (self.raw as i128).unsigned_abs();
            let words = [magnitude as u64, (magnitude >> 64) as u64];
            ieee_bits(self.raw < 0, &words, FRAC_BITS, fraction_bits, exponent_bits)
        }
        #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
        {
            let negative = self.raw.is_negative();
            // two's complement negation wraps at the minimum, whose words then
            // read as the unsigned magnitude 2^(W-1)
            let magnitude = if negative { -self.raw } else { self.raw };
            ieee_bits(negative, &magnitude.words, FRAC_BITS, fraction_bits, exponent_bits)
        }
    }

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
        match super::decimal_literal::parse(s, FRAC_BITS as u32, crate::fixed_point::frac_config::STORAGE_BITS) {
            Some(converted) => converted.map(|c| Self { raw: super::decimal_literal::to_storage(&c) }),
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
        let result = gmath_parse(s).and_then(|expr| evaluate(&expr));
        mode::set_mode(old_mode);

        Self::try_from_stack_value(result?)
    }

    // ========================================================================
    // Transcendentals — direct binary engine calls (bypass FASC)
    //
    // Pattern: upscale → binary engine at compute tier → downscale.
    // Saves ~65 ns per call vs the FASC pipeline (no LazyExpr tree, no TLS,
    // no StackValue boxing, no domain routing). Proven by sincos_wide().
    // ========================================================================

    /// Upscale self.raw to compute tier, call engine, downscale result.
    #[inline]
    fn direct_unary<F: FnOnce(ComputeStorage) -> ComputeStorage>(self, f: F) -> Self {
        let compute = upscale_to_compute(self.raw);
        let result = f(compute);
        Self { raw: downscale_to_storage(result).expect("transcendental overflow") }
    }

    /// Fallible version of direct_unary.
    #[inline]
    fn try_direct_unary<F: FnOnce(ComputeStorage) -> ComputeStorage>(self, f: F) -> Result<Self, OverflowDetected> {
        let compute = upscale_to_compute(self.raw);
        let result = f(compute);
        Ok(Self { raw: downscale_to_storage(result)? })
    }

    /// e^x
    pub fn exp(self) -> Self { self.direct_unary(direct_exp) }
    /// ln(x), x > 0
    pub fn ln(self) -> Self { self.direct_unary(direct_ln) }
    /// sqrt(x), x >= 0
    pub fn sqrt(self) -> Self { self.direct_unary(direct_sqrt) }
    /// sin(x)
    pub fn sin(self) -> Self { self.direct_unary(direct_sin) }
    /// cos(x)
    pub fn cos(self) -> Self { self.direct_unary(direct_cos) }
    /// Fused (sin(x), cos(x)): single range reduction, ~2× faster than separate calls.
    pub fn sincos(self) -> (Self, Self) {
        self.try_sincos().expect("sincos: overflow or domain error")
    }
    /// tan(x) = sin(x) / cos(x): direct composition, no FASC
    pub fn tan(self) -> Self {
        let c = upscale_to_compute(self.raw);
        let s = direct_sin(c);
        let c_val = direct_cos(c);
        let result = compute_divide_direct(s, c_val);
        Self { raw: downscale_to_storage(result).expect("tan overflow") }
    }
    /// atan(x)
    pub fn atan(self) -> Self { self.direct_unary(direct_atan) }
    /// asin(x) = atan(x / sqrt(1 - x^2)), |x| <= 1: direct composition
    pub fn asin(self) -> Self {
        let c = upscale_to_compute(self.raw);
        let one = compute_one();
        let x2 = compute_mul_direct(c, c);
        let denom = direct_sqrt(compute_sub_direct(one, x2));
        let ratio = compute_divide_direct(c, denom);
        Self { raw: downscale_to_storage(direct_atan(ratio)).expect("asin overflow") }
    }
    /// acos(x) = pi/2 - asin(x), |x| <= 1: direct composition
    pub fn acos(self) -> Self {
        let c = upscale_to_compute(self.raw);
        let one = compute_one();
        let x2 = compute_mul_direct(c, c);
        let denom = direct_sqrt(compute_sub_direct(one, x2));
        let ratio = compute_divide_direct(c, denom);
        let asin_val = direct_atan(ratio);
        let pi_half = compute_pi_half();
        Self { raw: downscale_to_storage(compute_sub_direct(pi_half, asin_val)).expect("acos overflow") }
    }
    /// sinh(x) = (exp(x) - exp(-x)) / 2: direct composition
    pub fn sinh(self) -> Self {
        let c = upscale_to_compute(self.raw);
        let ep = direct_exp(c);
        let en = direct_exp(compute_neg_direct(c));
        // A ceiling exp means |sinh| already exceeds the storage tier, and
        // the ceiling VALUE would downscale cleanly to a plausible-wrong
        // maximum — fail loud instead.
        assert!(
            !exp_ceilinged(&ep) && !exp_ceilinged(&en),
            "sinh overflow: exp at compute ceiling"
        );
        let result = compute_halve_direct(compute_sub_direct(ep, en));
        Self { raw: downscale_to_storage(result).expect("sinh overflow") }
    }
    /// cosh(x) = (exp(x) + exp(-x)) / 2: direct composition
    pub fn cosh(self) -> Self {
        let c = upscale_to_compute(self.raw);
        let ep = direct_exp(c);
        let en = direct_exp(compute_neg_direct(c));
        // A ceiling exp means cosh already exceeds the storage tier. The
        // checked add alone is NOT a sufficient guard: on wide profiles the
        // ceiling fits the compute type and downscales to a plausible-wrong
        // maximum — fail loud instead.
        assert!(
            !exp_ceilinged(&ep) && !exp_ceilinged(&en),
            "cosh overflow: exp at compute ceiling"
        );
        let sum = compute_checked_add(ep, en).expect("cosh overflow");
        let result = compute_halve_direct(sum);
        Self { raw: downscale_to_storage(result).expect("cosh overflow") }
    }
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
    pub fn tanh(self) -> Self {
        let c = upscale_to_compute(self.raw);
        let two_x = compute_add_direct(c, c);
        let e2x = direct_exp(two_x);
        let one = compute_one();
        // exp(2x) at the compute-tier ceiling (saturated downscale or overflow
        // sentinel): tanh = 1 − 2/(exp(2x)+1) rounds to exactly 1 at every
        // storage width — and the denominator add below would wrap.
        let den = match compute_checked_add(e2x, one) {
            Ok(v) => v,
            Err(_) => return Self::one(),
        };
        let num = compute_sub_direct(e2x, one);
        Self { raw: downscale_to_storage(compute_divide_direct(num, den)).expect("tanh overflow") }
    }
    /// asinh(x) = ln(x + sqrt(x^2 + 1)): direct composition
    pub fn asinh(self) -> Self {
        let c = upscale_to_compute(self.raw);
        let one = compute_one();
        let x2 = compute_mul_direct(c, c);
        let inner = direct_sqrt(compute_add_direct(x2, one));
        Self { raw: downscale_to_storage(direct_ln(compute_add_direct(c, inner))).expect("asinh overflow") }
    }
    /// acosh(x) = ln(x + sqrt(x^2 - 1)), x >= 1: direct composition
    pub fn acosh(self) -> Self {
        let c = upscale_to_compute(self.raw);
        let one = compute_one();
        let x2 = compute_mul_direct(c, c);
        let inner = direct_sqrt(compute_sub_direct(x2, one));
        Self { raw: downscale_to_storage(direct_ln(compute_add_direct(c, inner))).expect("acosh overflow") }
    }
    /// atanh(x) = ln((1+x)/(1-x)) / 2, |x| < 1: direct composition
    pub fn atanh(self) -> Self {
        let c = upscale_to_compute(self.raw);
        let one = compute_one();
        let num = compute_add_direct(one, c);
        let den = compute_sub_direct(one, c);
        let ratio = compute_divide_direct(num, den);
        Self { raw: downscale_to_storage(compute_halve_direct(direct_ln(ratio))).expect("atanh overflow") }
    }

    /// x^y = exp(y * ln(x)): direct composition
    pub fn pow(self, exponent: Self) -> Self {
        let xc = upscale_to_compute(self.raw);
        let yc = upscale_to_compute(exponent.raw);
        let ln_x = direct_ln(xc);
        let y_ln_x = compute_mul_direct(yc, ln_x);
        Self { raw: downscale_to_storage(direct_exp(y_ln_x)).expect("pow overflow") }
    }

    /// atan2(self=y, x): direct binary engine
    pub fn atan2(self, x: Self) -> Self {
        let yc = upscale_to_compute(self.raw);
        let xc = upscale_to_compute(x.raw);
        let result = direct_atan2(yc, xc);
        Self { raw: downscale_to_storage(result).expect("atan2 overflow") }
    }

    // ========================================================================
    // UGOD-aware try_* transcendentals — return Result instead of panicking
    // ========================================================================

    /// Fallible e^x: returns `Err(TierOverflow)` if result exceeds storage tier.
    pub fn try_exp(self) -> Result<Self, OverflowDetected> { self.try_direct_unary(direct_exp) }
    /// Fallible ln(x): returns `Err(DomainError)` if x <= 0.
    pub fn try_ln(self) -> Result<Self, OverflowDetected> {
        // Domain check before the engine: the raw engine is infallible and
        // signals out-of-domain input with a MIN sentinel that downscale
        // would misreport as TierOverflow.
        if self <= Self::ZERO { return Err(OverflowDetected::DomainError); }
        self.try_direct_unary(direct_ln)
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
        use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
            compute_divide, make_compute_int, sqrt_at_compute_tier,
        };
        if self <= Self::ZERO { return Err(OverflowDetected::DomainError); }
        let s = sqrt_at_compute_tier(upscale_to_compute(self.raw));
        let inv = compute_divide(make_compute_int(1), s)?;
        Ok(Self { raw: downscale_to_storage(inv)? })
    }

    /// Fallible sqrt(x): returns `Err(DomainError)` if x < 0.
    pub fn try_sqrt(self) -> Result<Self, OverflowDetected> {
        if self < Self::ZERO { return Err(OverflowDetected::DomainError); }
        self.try_direct_unary(direct_sqrt)
    }
    /// Fallible sin(x).
    pub fn try_sin(self) -> Result<Self, OverflowDetected> { self.try_direct_unary(direct_sin) }
    /// Fallible cos(x).
    pub fn try_cos(self) -> Result<Self, OverflowDetected> { self.try_direct_unary(direct_cos) }
    /// Fused sin+cos: single shared range reduction at compute tier.
    /// Returns (sin(x), cos(x)). More efficient than separate try_sin + try_cos.
    pub fn try_sincos(self) -> Result<(Self, Self), OverflowDetected> {
        use super::linalg::{upscale_to_compute, round_to_storage, sincos_at_compute_tier};
        let compute_val = upscale_to_compute(self.raw());
        let (sin_c, cos_c) = sincos_at_compute_tier(compute_val);
        Ok((Self::from_raw(round_to_storage(sin_c)), Self::from_raw(round_to_storage(cos_c))))
    }

    /// Fallible fused sinh+cosh: single shared exp-pair at compute tier.
    ///
    /// Returns `Err(TierOverflow)` if either sinh(x) or cosh(x) exceeds the
    /// storage tier (cosh grows fastest: overflows first for large |x|).
    pub fn try_sinhcosh(self) -> Result<(Self, Self), OverflowDetected> {
        use crate::fixed_point::universal::fasc::stack_evaluator::sinhcosh_at_compute_tier;
        let compute_val = upscale_to_compute(self.raw);
        let (sinh_c, cosh_c) = sinhcosh_at_compute_tier(compute_val);
        Ok((
            Self { raw: downscale_to_storage(sinh_c)? },
            Self { raw: downscale_to_storage(cosh_c)? },
        ))
    }

    /// Fused sin+cos for wide-range angles that exceed storage-tier integer range.
    ///
    /// The angle is a raw i64 in **Q32.32 fixed-point format** (32 integer bits,
    /// 32 fractional bits). This gives ±2.1 billion integer range regardless of
    /// the profile's FRAC_BITS, covering all practical RoPE frequencies.
    ///
    /// Identical to [`sincos_wide_q64`](Self::sincos_wide_q64) of the angle
    /// shifted to Q64.64. Output sin/cos always fits in [-1, 1].
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    pub fn sincos_wide(angle_q32_32: i64) -> (Self, Self) {
        // Fixed Q32.32→Q64.64 upscale (always 32, independent of profile FRAC_BITS)
        Self::sincos_wide_q64((angle_q32_32 as i128) << 32)
    }

    /// Fused sin+cos of a Q64.64 angle (`radians * 2^64`), rounded to storage.
    ///
    /// Computes [`wide::sincos_q64`](crate::fixed_point::wide::sincos_q64) and
    /// rounds each result once to storage. The Q64.64 angle needs no caller-side
    /// truncation to Q32.32.
    ///
    /// # Use case
    /// RoPE position encoding, where `theta^(-2i/d) * position` exceeds the
    /// storage range:
    /// ```ignore
    /// use g_math::fixed_point::{wide, FixedPoint};
    /// let (theta, i, d, position) = (10_000i128, 3i128, 64i128, 4_095i128);
    /// let ln_theta = wide::ln_q64(theta << 64).unwrap();
    /// let inv_freq = wide::exp_q64(-(2 * i * ln_theta) / d);   // Q64.64
    /// let (sin, cos) = FixedPoint::sincos_wide_q64(inv_freq * position);
    /// ```
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    pub fn sincos_wide_q64(angle_q64: i128) -> (Self, Self) {
        use crate::fixed_point::domains::binary_fixed::transcendental::{
            sin_binary_i128, cos_binary_i128,
        };
        let sin_q64 = sin_binary_i128(angle_q64);
        let cos_q64 = cos_binary_i128(angle_q64);
        // Downscale Q64.64 → storage tier: shift = 64 - FRAC_BITS
        #[cfg(table_format = "q16_16")]
        {
            use crate::fixed_point::frac_config;
            let shift = 64 - frac_config::FRAC_BITS;
            let sin_round = (sin_q64 >> (shift - 1)) & 1;
            let cos_round = (cos_q64 >> (shift - 1)) & 1;
            let sin_raw = ((sin_q64 >> shift) + sin_round) as i32;
            let cos_raw = ((cos_q64 >> shift) + cos_round) as i32;
            (Self::from_raw(sin_raw), Self::from_raw(cos_raw))
        }
        #[cfg(table_format = "q32_32")]
        {
            // Q64.64 → Q32.32: shift right 32 with rounding
            let sin_round = (sin_q64 >> 31) & 1;
            let cos_round = (cos_q64 >> 31) & 1;
            let sin_raw = ((sin_q64 >> 32) + sin_round) as i64;
            let cos_raw = ((cos_q64 >> 32) + cos_round) as i64;
            (Self::from_raw(sin_raw), Self::from_raw(cos_raw))
        }
    }
    // Fallible composed transcendentals — direct compute-tier compositions
    // mirroring the infallible methods above (0.5.0 item 2; previously these
    // routed through the FASC pipeline via try_apply_unary(LazyExpr)).
    // Error contract unchanged: DomainError on domain violations (0.4.27),
    // TierOverflow when the result exceeds storage.

    /// Fallible tan(x) = sin(x)/cos(x): `Err(DomainError)` if cos(x) is zero
    /// at the compute tier.
    pub fn try_tan(self) -> Result<Self, OverflowDetected> {
        use crate::fixed_point::universal::fasc::stack_evaluator::compute::compute_is_zero;
        let c = upscale_to_compute(self.raw);
        let s = direct_sin(c);
        let c_val = direct_cos(c);
        if compute_is_zero(&c_val) { return Err(OverflowDetected::DomainError); }
        Ok(Self { raw: downscale_to_storage(compute_divide_direct(s, c_val))? })
    }
    /// Fallible atan(x).
    pub fn try_atan(self) -> Result<Self, OverflowDetected> { self.try_direct_unary(direct_atan) }
    /// Fallible asin(x) = atan(x / sqrt(1 - x²)): `Err(DomainError)` if |x| > 1.
    pub fn try_asin(self) -> Result<Self, OverflowDetected> {
        let one = Self::one();
        let neg_one = Self::ZERO - one;
        if self > one || self < neg_one { return Err(OverflowDetected::DomainError); }
        // Boundary: asin(±1) = ±π/2 exactly (avoids division by sqrt(0))
        if self == one {
            return Ok(Self { raw: downscale_to_storage(compute_pi_half())? });
        }
        if self == neg_one {
            return Ok(Self { raw: downscale_to_storage(compute_neg_direct(compute_pi_half()))? });
        }
        let c = upscale_to_compute(self.raw);
        let x2 = compute_mul_direct(c, c);
        let denom = direct_sqrt(compute_sub_direct(compute_one(), x2));
        let ratio = compute_divide_direct(c, denom);
        Ok(Self { raw: downscale_to_storage(direct_atan(ratio))? })
    }
    /// Fallible acos(x) = π/2 - asin(x): `Err(DomainError)` if |x| > 1.
    pub fn try_acos(self) -> Result<Self, OverflowDetected> {
        let one = Self::one();
        let neg_one = Self::ZERO - one;
        if self > one || self < neg_one { return Err(OverflowDetected::DomainError); }
        let pi_half = compute_pi_half();
        // Boundaries: acos(1) = 0, acos(-1) = π
        let asin_c = if self == one {
            pi_half
        } else if self == neg_one {
            compute_neg_direct(pi_half)
        } else {
            let c = upscale_to_compute(self.raw);
            let x2 = compute_mul_direct(c, c);
            let denom = direct_sqrt(compute_sub_direct(compute_one(), x2));
            direct_atan(compute_divide_direct(c, denom))
        };
        Ok(Self { raw: downscale_to_storage(compute_sub_direct(pi_half, asin_c))? })
    }
    /// Fallible sinh(x) = (exp(x) - exp(-x)) / 2: `Err(TierOverflow)` when
    /// the result exceeds the storage tier (a ceiling exp means it already
    /// has, and the ceiling value would downscale to a plausible-wrong max).
    pub fn try_sinh(self) -> Result<Self, OverflowDetected> {
        let c = upscale_to_compute(self.raw);
        let ep = direct_exp(c);
        let en = direct_exp(compute_neg_direct(c));
        if exp_ceilinged(&ep) || exp_ceilinged(&en) {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(Self { raw: downscale_to_storage(compute_halve_direct(compute_sub_direct(ep, en)))? })
    }
    /// Fallible cosh(x) = (exp(x) + exp(-x)) / 2: `Err(TierOverflow)` when
    /// the result exceeds the storage tier (a ceiling exp means cosh already
    /// has; the checked add alone cannot see it on wide profiles).
    pub fn try_cosh(self) -> Result<Self, OverflowDetected> {
        let c = upscale_to_compute(self.raw);
        let ep = direct_exp(c);
        let en = direct_exp(compute_neg_direct(c));
        if exp_ceilinged(&ep) || exp_ceilinged(&en) {
            return Err(OverflowDetected::TierOverflow);
        }
        let sum = compute_checked_add(ep, en)?;
        Ok(Self { raw: downscale_to_storage(compute_halve_direct(sum))? })
    }
    /// Fallible tanh(x) = (exp(2x) - 1) / (exp(2x) + 1). Saturates to exactly
    /// 1 when exp(2x) reaches the compute-tier ceiling (matches `tanh`).
    pub fn try_tanh(self) -> Result<Self, OverflowDetected> {
        let c = upscale_to_compute(self.raw);
        let e2x = direct_exp(compute_add_direct(c, c));
        let one = compute_one();
        let den = match compute_checked_add(e2x, one) {
            Ok(v) => v,
            Err(_) => return Ok(Self::one()),
        };
        let num = compute_sub_direct(e2x, one);
        Ok(Self { raw: downscale_to_storage(compute_divide_direct(num, den))? })
    }
    /// Fallible asinh(x) = ln(x + sqrt(x² + 1)).
    pub fn try_asinh(self) -> Result<Self, OverflowDetected> {
        // ln argument is provably positive at compute tier: for x >= 0 it is
        // >= 1, and for x < 0 it is ~1/(2|x|) >= 2^-(FRAC_BITS+1), far above
        // one compute-tier ulp for any storable x.
        let c = upscale_to_compute(self.raw);
        let x2 = compute_mul_direct(c, c);
        let inner = direct_sqrt(compute_add_direct(x2, compute_one()));
        Ok(Self { raw: downscale_to_storage(direct_ln(compute_add_direct(c, inner)))? })
    }
    /// Fallible acosh(x) = ln(x + sqrt(x² - 1)): `Err(DomainError)` if x < 1.
    pub fn try_acosh(self) -> Result<Self, OverflowDetected> {
        if self < Self::one() { return Err(OverflowDetected::DomainError); }
        let c = upscale_to_compute(self.raw);
        let x2 = compute_mul_direct(c, c);
        let inner = direct_sqrt(compute_sub_direct(x2, compute_one()));
        Ok(Self { raw: downscale_to_storage(direct_ln(compute_add_direct(c, inner)))? })
    }
    /// Fallible atanh(x) = ln((1+x)/(1-x)) / 2: `Err(DomainError)` if |x| >= 1.
    pub fn try_atanh(self) -> Result<Self, OverflowDetected> {
        let one = Self::one();
        if self >= one || self <= Self::ZERO - one { return Err(OverflowDetected::DomainError); }
        let c = upscale_to_compute(self.raw);
        let one_c = compute_one();
        let ratio = compute_divide_direct(compute_add_direct(one_c, c), compute_sub_direct(one_c, c));
        Ok(Self { raw: downscale_to_storage(compute_halve_direct(direct_ln(ratio)))? })
    }

    /// Fallible x^y = exp(y * ln(x)).
    pub fn try_pow(self, exponent: Self) -> Result<Self, OverflowDetected> {
        let sv1 = self.to_stack_value();
        let sv2 = exponent.to_stack_value();
        let expr = LazyExpr::from(sv1).pow(LazyExpr::from(sv2));
        let result = evaluate(&expr)?;
        Self::try_from_stack_value(result)
    }

    /// Fallible atan2(self=y, x).
    pub fn try_atan2(self, x: Self) -> Result<Self, OverflowDetected> {
        let sv_y = self.to_stack_value();
        let sv_x = x.to_stack_value();
        let expr = LazyExpr::from(sv_y).atan2(LazyExpr::from(sv_x));
        let result = evaluate(&expr)?;
        Self::try_from_stack_value(result)
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    #[inline]
    pub(crate) fn to_stack_value(self) -> StackValue {
        StackValue::Binary(STORAGE_TIER, self.raw, CompactShadow::None)
    }

    pub(crate) fn try_from_stack_value(sv: StackValue) -> Result<Self, OverflowDetected> {
        match sv.as_binary_storage() {
            Some(raw) => Ok(Self { raw }),
            None => {
                // Non-binary domain — force conversion by adding binary zero
                let zero_sv = StackValue::Binary(STORAGE_TIER, Self::ZERO.raw, CompactShadow::None);
                let expr = LazyExpr::from(sv) + LazyExpr::from(zero_sv);
                let result = evaluate(&expr)?;
                result.as_binary_storage()
                    .map(|raw| Self { raw })
                    .ok_or(OverflowDetected::InvalidInput)
            }
        }
    }

    #[allow(dead_code)]
    fn apply_unary(self, f: fn(LazyExpr) -> LazyExpr) -> Self {
        self.try_apply_unary(f).expect("transcendental: overflow or domain error")
    }

    fn try_apply_unary(self, f: fn(LazyExpr) -> LazyExpr) -> Result<Self, OverflowDetected> {
        let sv = self.to_stack_value();
        let expr = f(LazyExpr::from(sv));
        let result = evaluate(&expr)?;
        Self::try_from_stack_value(result)
    }

    /// `mantissa x 2^shift` truncated toward zero, with the sign applied, as a
    /// storage raw. `mantissa` has at most 53 significant bits. A magnitude
    /// outside the storage range (above `2^(W-1) - 1`, or `2^(W-1)` for a
    /// negative value) is a `TierOverflow`, never a wrap.
    fn truncated_raw(mantissa: u64, shift: i32, negative: bool) -> Result<BinaryStorage, OverflowDetected> {
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        {
            #[cfg(table_format = "q16_16")]
            const WIDTH: u32 = 32;
            #[cfg(table_format = "q32_32")]
            const WIDTH: u32 = 64;
            #[cfg(table_format = "q64_64")]
            const WIDTH: u32 = 128;
            let magnitude: u128 = if shift >= 0 {
                // a magnitude of more than 128 bits overflows every width here
                if 64 - mantissa.leading_zeros() as i32 + shift > 128 {
                    return Err(OverflowDetected::TierOverflow);
                }
                (mantissa as u128) << shift
            } else if shift > -128 {
                (mantissa as u128) >> (-shift)
            } else {
                0
            };
            let limit = 1u128 << (WIDTH - 1);
            if magnitude > limit || (magnitude == limit && !negative) {
                return Err(OverflowDetected::TierOverflow);
            }
            let value = if negative { (magnitude as i128).wrapping_neg() } else { magnitude as i128 };
            Ok(value as BinaryStorage)
        }
        #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
        {
            #[cfg(table_format = "q128_128")]
            let (width, one, zero) = (256i32, I256::from_i128(1), I256::zero());
            #[cfg(table_format = "q256_256")]
            let (width, one, zero) = (512i32, I512::from_i128(1), I512::zero());
            let magnitude = if shift < 0 {
                if shift <= -64 { zero } else { from_u64(mantissa >> (-shift)) }
            } else {
                let length = 64 - mantissa.leading_zeros() as i32 + shift;
                if length < width {
                    from_u64(mantissa) << (shift as usize)
                } else if length == width && negative && mantissa.is_power_of_two() {
                    // exactly 2^(W-1): the storage minimum
                    return Ok(one << ((width - 1) as usize));
                } else {
                    return Err(OverflowDetected::TierOverflow);
                }
            };
            Ok(if negative { -magnitude } else { magnitude })
        }
    }
}

/// A `u64` widened to the storage type (wide profiles).
#[cfg(table_format = "q128_128")]
#[inline]
fn from_u64(v: u64) -> I256 {
    I256::from_i128(v as i128)
}
#[cfg(table_format = "q256_256")]
#[inline]
fn from_u64(v: u64) -> I512 {
    I512::from_i128(v as i128)
}

/// IEEE 754 bits of `magnitude x 2^-frac_bits`, `magnitude` given as
/// little-endian 64-bit words, in a binary format with `fraction_bits` stored
/// fraction bits and `exponent_bits` exponent bits: rounded to
/// `fraction_bits + 1` significant bits, nearest with ties to even; subnormal
/// or zero below the normal range, infinite above it. Integer operations only.
fn ieee_bits(negative: bool, magnitude: &[u64], frac_bits: i32, fraction_bits: u32, exponent_bits: u32) -> u64 {
    let length = word_bit_length(magnitude);
    if length == 0 {
        return 0;
    }
    let sign = (negative as u64) << (fraction_bits + exponent_bits);
    let bias = (1i32 << (exponent_bits - 1)) - 1;
    let top = length as i32 - 1 - frac_bits; // exponent of the leading bit
    // weight of the lowest bit kept: fraction_bits below the leading bit, but
    // never below the subnormal quantum
    let lowest = (top - fraction_bits as i32).max(1 - bias - fraction_bits as i32);
    let shift = lowest + frac_bits; // bits of the magnitude below the kept ones
    let mut significand = if shift <= 0 {
        // everything is kept: the magnitude has at most fraction_bits + 1 bits
        magnitude[0] << (-shift) as u32
    } else {
        let shift = shift as u32;
        let kept = word_bits_from(magnitude, shift);
        let round = word_bit(magnitude, shift - 1);
        let sticky = word_any_below(magnitude, shift - 1);
        kept + (round && (sticky || kept & 1 == 1)) as u64
    };
    let mut lowest = lowest;
    if significand >> (fraction_bits + 1) != 0 {
        // rounding carried into a new leading bit
        significand >>= 1;
        lowest += 1;
    }
    if significand == 0 {
        return sign;
    }
    if significand >> fraction_bits == 0 {
        return sign | significand; // subnormal: exponent field 0
    }
    let biased = lowest + fraction_bits as i32 + bias;
    let infinite = (1i32 << exponent_bits) - 1;
    if biased >= infinite {
        return sign | ((infinite as u64) << fraction_bits);
    }
    sign | ((biased as u64) << fraction_bits) | (significand & ((1u64 << fraction_bits) - 1))
}

fn word_bit_length(words: &[u64]) -> u32 {
    for i in (0..words.len()).rev() {
        if words[i] != 0 {
            return i as u32 * 64 + 64 - words[i].leading_zeros();
        }
    }
    0
}

fn word_bit(words: &[u64], i: u32) -> bool {
    words.get((i / 64) as usize).map_or(false, |w| (w >> (i % 64)) & 1 == 1)
}

/// Any bit below position `i` set.
fn word_any_below(words: &[u64], i: u32) -> bool {
    let (w, b) = ((i / 64) as usize, i % 64);
    words.iter().take(w.min(words.len())).any(|&x| x != 0)
        || (b > 0 && words.get(w).map_or(false, |&x| x & ((1u64 << b) - 1) != 0))
}

/// The magnitude shifted right by `i`, low 64 bits (callers keep at most 54).
fn word_bits_from(words: &[u64], i: u32) -> u64 {
    let (w, b) = ((i / 64) as usize, i % 64);
    let low = words.get(w).map_or(0, |&x| x >> b);
    let high = if b == 0 { 0 } else { words.get(w + 1).map_or(0, |&x| x << (64 - b)) };
    low | high
}

// ============================================================================
// Display
// ============================================================================

impl fmt::Display for FixedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sv = self.to_stack_value();
        fmt::Display::fmt(&sv, f)
    }
}

impl Default for FixedPoint {
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

impl Add for FixedPoint {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.try_add(rhs).expect("FixedPoint: addition overflow")
    }
}

impl Sub for FixedPoint {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.try_sub(rhs).expect("FixedPoint: subtraction overflow")
    }
}

impl Mul for FixedPoint {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.try_mul(rhs).expect("FixedPoint: multiplication overflow")
    }
}

impl Div for FixedPoint {
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

impl Neg for FixedPoint {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.try_neg().expect("FixedPoint: negation overflow")
    }
}

impl FixedPoint {
    /// `self + rhs`, `Err(TierOverflow)` when the sum leaves storage.
    #[inline]
    pub fn try_add(self, rhs: Self) -> Result<Self, OverflowDetected> {
        self.raw.checked_add(rhs.raw).map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow)
    }

    /// `self - rhs`, `Err(TierOverflow)` when the difference leaves storage.
    #[inline]
    pub fn try_sub(self, rhs: Self) -> Result<Self, OverflowDetected> {
        self.raw.checked_sub(rhs.raw).map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow)
    }

    /// `-self`, `Err(TierOverflow)` for the storage minimum (no positive twin).
    #[inline]
    pub fn try_neg(self) -> Result<Self, OverflowDetected> {
        self.raw.checked_neg().map(|raw| Self { raw }).ok_or(OverflowDetected::TierOverflow)
    }

    /// `self * rhs`, rounded to nearest (ties toward +infinity) from the exact
    /// product; `Err(TierOverflow)` when it leaves storage.
    #[inline]
    pub fn try_mul(self, rhs: Self) -> Result<Self, OverflowDetected> {
        try_fixed_multiply(self.raw, rhs.raw).map(|raw| Self { raw })
    }

    /// `self / rhs`, rounded to nearest (ties toward +infinity) from the exact
    /// quotient; `Err(DivisionByZero)` or `Err(TierOverflow)`.
    #[inline]
    pub fn try_div(self, rhs: Self) -> Result<Self, OverflowDetected> {
        try_fixed_divide(self.raw, rhs.raw).map(|raw| Self { raw })
    }
}

impl AddAssign for FixedPoint {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl SubAssign for FixedPoint {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl MulAssign for FixedPoint {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl DivAssign for FixedPoint {
    #[inline]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

// ============================================================================
// Q-format fixed-point multiply
// ============================================================================

/// Multiply two Q-format fixed-point values: the exact wide product rounded
/// to nearest (ties toward +infinity) and range-checked.
#[inline]
fn try_fixed_multiply(a: BinaryStorage, b: BinaryStorage) -> Result<BinaryStorage, OverflowDetected> {
    let overflow = OverflowDetected::TierOverflow;
    #[cfg(table_format = "q16_16")]
    {
        // i32*i32->i64, >>FRAC_BITS with round-bit: nearest, ties toward +inf
        let wide = (a as i64) * (b as i64);
        let round_bit = ((wide >> (FRAC_BITS - 1)) & 1) as i32;
        let q = wide >> FRAC_BITS;
        if q as i32 as i64 != q { return Err(overflow); }
        (q as i32).checked_add(round_bit).ok_or(overflow)
    }
    #[cfg(table_format = "q32_32")]
    {
        // i64*i64->i128, >>32 with round-bit: nearest, ties toward +inf
        let wide = (a as i128) * (b as i128);
        let round_bit = ((wide >> 31) & 1) as i64;
        let q = wide >> 32;
        if q as i64 as i128 != q { return Err(overflow); }
        (q as i64).checked_add(round_bit).ok_or(overflow)
    }
    #[cfg(table_format = "q64_64")]
    {
        // i128*i128->I256, >>64 plus the round bit (bit 63 of the product):
        // the rounding of multiply_binary_i128, range-checked
        let product = crate::fixed_point::i256::mul_i128_to_i256(a, b);
        let mut scaled = product >> 64u32;
        if product.words[0] >= 1u64 << 63 { scaled = scaled + I256::from_i128(1); }
        if !scaled.fits_in_i128() { return Err(overflow); }
        Ok(scaled.as_i128())
    }
    #[cfg(table_format = "q128_128")]
    {
        // I256*I256->I512 on magnitudes (mul_to_i512 is unsigned), read by
        // words: the magnitude before rounding is words 2..6, the remainder
        // words 0..2. A positive result rounds the magnitude up on
        // remainder >= half (tie up), a negative one only on remainder > half
        // (tie toward +inf).
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let w = abs_a.mul_to_i512(abs_b).words;
        let top = 1u64 << 63;
        let bump = if result_neg { w[1] > top || (w[1] == top && w[0] != 0) } else { w[1] >= top };
        if w[6] != 0 || w[7] != 0 || w[5] > top {
            return Err(overflow);
        }
        // round: add the bump through the carry chain (the top word is at
        // most 2^63, so it cannot carry out)
        let mut m = [w[2], w[3], w[4], w[5]];
        let mut carry = bump as u64;
        for x in m.iter_mut() {
            let (v, c) = x.overflowing_add(carry);
            *x = v;
            carry = c as u64;
        }
        if m[3] >= top {
            // 2^255 or more: only the negative minimum, magnitude exactly 2^255
            return if result_neg && m == [0, 0, 0, top] { Ok(I256::min_value()) } else { Err(overflow) };
        }
        let mag = I256 { words: m };
        Ok(if result_neg { -mag } else { mag })
    }
    #[cfg(table_format = "q256_256")]
    {
        // I512*I512->I1024 on magnitudes, read by words (see q128_128): the
        // magnitude before rounding is words 4..12, the remainder words 0..4
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let w = abs_a.mul_to_i1024(abs_b).words;
        let top = 1u64 << 63;
        let bump = if result_neg {
            w[3] > top || (w[3] == top && (w[0] | w[1] | w[2]) != 0)
        } else {
            w[3] >= top
        };
        if w[12..].iter().any(|&x| x != 0) || w[11] > top {
            return Err(overflow);
        }
        let mut m = [0u64; 8];
        m.copy_from_slice(&w[4..12]);
        let mut carry = bump as u64;
        for x in m.iter_mut() {
            let (v, c) = x.overflowing_add(carry);
            *x = v;
            carry = c as u64;
        }
        if m[7] >= top {
            // 2^511 or more: only the negative minimum, magnitude exactly 2^511
            let exact_min = m[7] == top && m[..7].iter().all(|&x| x == 0);
            return if result_neg && exact_min { Ok(I512::min_value()) } else { Err(overflow) };
        }
        let mag = I512 { words: m };
        Ok(if result_neg { -mag } else { mag })
    }
}

// ============================================================================
// Q-format fixed-point divide
// ============================================================================

/// Divide two Q-format fixed-point values.
///
/// Uses tier N+1 widening: (a << FRAC_BITS) / b, rounded to nearest with
/// ties toward +∞ (0.5.0 rounding unification: was truncation toward
/// zero). The exact quotient's sign decides the tie direction: positive
/// results bump on 2|rem| >= |den| (tie goes up), negative results bump
/// only on 2|rem| > |den| (tie stays, which is toward +∞).
/// `Err(DivisionByZero)` / `Err(TierOverflow)`; the quotient was narrowed
/// with a truncating cast before 0.6.4.
#[inline]
fn try_fixed_divide(a: BinaryStorage, b: BinaryStorage) -> Result<BinaryStorage, OverflowDetected> {
    let overflow = OverflowDetected::TierOverflow;
    #[cfg(table_format = "q16_16")]
    {
        let num = (a as i64) << FRAC_BITS;
        let den = b as i64;
        if den == 0 { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem2 = (num - q * den).unsigned_abs() << 1;
        let dabs = den.unsigned_abs();
        let positive = (num < 0) == (den < 0);
        let bump = if positive { rem2 >= dabs } else { rem2 > dabs };
        i32::try_from(if bump { q + if positive { 1 } else { -1 } } else { q }).map_err(|_| overflow)
    }
    #[cfg(table_format = "q32_32")]
    {
        let num = (a as i128) << 32;
        let den = b as i128;
        if den == 0 { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem2 = (num - q * den).unsigned_abs() << 1;
        let dabs = den.unsigned_abs();
        let positive = (num < 0) == (den < 0);
        let bump = if positive { rem2 >= dabs } else { rem2 > dabs };
        i64::try_from(if bump { q + if positive { 1 } else { -1 } } else { q }).map_err(|_| overflow)
    }
    #[cfg(table_format = "q64_64")]
    {
        let num = I256::from_i128(a) << 64usize;
        let den = I256::from_i128(b);
        if den.is_zero() { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem = num - q * den;
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let den_abs = if den.is_negative() { -den } else { den };
        let positive = num.is_negative() == den.is_negative();
        let rem2 = rem_abs + rem_abs;
        let bump = if positive { rem2 >= den_abs } else { rem2 > den_abs };
        let one = I256::from_i128(1);
        let q = if bump { if positive { q + one } else { q - one } } else { q };
        if !q.fits_in_i128() { return Err(overflow); }
        Ok(q.as_i128())
    }
    #[cfg(table_format = "q128_128")]
    {
        let num = I512::from_i256(a) << 128usize;
        let den = I512::from_i256(b);
        if den.is_zero() { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem = num - q * den;
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let den_abs = if den.is_negative() { -den } else { den };
        let positive = num.is_negative() == den.is_negative();
        let rem2 = rem_abs + rem_abs;
        let bump = if positive { rem2 >= den_abs } else { rem2 > den_abs };
        let one = I512::from_i128(1);
        let q = if bump { if positive { q + one } else { q - one } } else { q };
        if !q.fits_in_i256() { return Err(overflow); }
        Ok(q.as_i256())
    }
    #[cfg(table_format = "q256_256")]
    {
        let num = I1024::from_i512(a) << 256usize;
        let den = I1024::from_i512(b);
        if den.is_zero() { return Err(OverflowDetected::DivisionByZero); }
        let q = num / den;
        let rem = num - q * den;
        let rem_abs = if rem < I1024::zero() { -rem } else { rem };
        let den_abs = if den < I1024::zero() { -den } else { den };
        let positive = (num < I1024::zero()) == (den < I1024::zero());
        let rem2 = rem_abs + rem_abs;
        let bump = if positive { rem2 >= den_abs } else { rem2 > den_abs };
        let one = I1024::from_i128(1);
        let q = if bump { if positive { q + one } else { q - one } } else { q };
        if !q.fits_in_i512() { return Err(overflow); }
        Ok(q.as_i512())
    }
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
