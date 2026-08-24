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
use crate::fixed_point::multiply_binary_i128;

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
    #[cfg(table_format = "q16_16")]
    { crate::fixed_point::domains::binary_fixed::transcendental::atan2_binary_i128(y as i128, x as i128) as i64 }
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

#[cfg(table_format = "q16_16")]
const MAX_DECIMAL_DIGITS: usize = crate::fixed_point::frac_config::MAX_DECIMAL_DIGITS;
#[cfg(table_format = "q32_32")]
const MAX_DECIMAL_DIGITS: usize = 9;
#[cfg(table_format = "q64_64")]
const MAX_DECIMAL_DIGITS: usize = 19;
#[cfg(table_format = "q128_128")]
const MAX_DECIMAL_DIGITS: usize = 38;
#[cfg(table_format = "q256_256")]
const MAX_DECIMAL_DIGITS: usize = 77;

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
#[derive(Clone, Copy, Debug)]
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

    /// Create from an integer value.
    #[inline]
    pub fn from_int(v: i32) -> Self {
        #[cfg(table_format = "q16_16")]
        { Self { raw: (v as i32) << FRAC_BITS } }
        #[cfg(table_format = "q32_32")]
        { Self { raw: (v as i64) << 32 } }
        #[cfg(table_format = "q64_64")]
        { Self { raw: (v as i128) << 64 } }
        #[cfg(table_format = "q128_128")]
        { Self { raw: I256::from_i128(v as i128) << 128usize } }
        #[cfg(table_format = "q256_256")]
        { Self { raw: I512::from_i128(v as i128) << 256usize } }
    }

    /// Extract the integer part (floor toward negative infinity).
    #[inline]
    pub fn to_int(self) -> i32 {
        #[cfg(table_format = "q16_16")]
        { (self.raw >> FRAC_BITS) as i32 }
        #[cfg(table_format = "q32_32")]
        { (self.raw >> 32) as i32 }
        #[cfg(table_format = "q64_64")]
        { (self.raw >> 64) as i32 }
        #[cfg(table_format = "q128_128")]
        { (self.raw >> 128u32).as_i128() as i32 }
        #[cfg(table_format = "q256_256")]
        { (self.raw >> 256usize).as_i128() as i32 }
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

    /// Create from an f32 value.
    ///
    /// Uses IEEE 754 bit extraction for exact conversion, no float arithmetic
    /// is performed internally. Panics on NaN or infinity.
    pub fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        // Handle +0.0 and -0.0
        if bits & 0x7FFF_FFFF == 0 {
            return Self::ZERO;
        }
        let sign = (bits >> 31) != 0;
        let raw_exp = ((bits >> 23) & 0xFF) as i32;
        let raw_mantissa = bits & 0x7F_FFFF;

        if raw_exp == 0xFF {
            panic!("FixedPoint::from_f32: infinity or NaN");
        }

        let (mantissa, exp_offset) = if raw_exp == 0 {
            // Subnormal: no implicit 1, exponent = -126
            (raw_mantissa as i128, -126 - 23)
        } else {
            // Normal: implicit 1 bit
            ((raw_mantissa | 0x80_0000) as i128, raw_exp - 127 - 23)
        };

        let shift = exp_offset + FRAC_BITS;
        let raw = Self::shift_mantissa_to_raw(mantissa, shift);
        if sign { -Self { raw } } else { Self { raw } }
    }

    /// Create from an f64 value.
    ///
    /// Uses IEEE 754 bit extraction for exact conversion, no float arithmetic
    /// is performed internally. Panics on NaN or infinity.
    pub fn from_f64(v: f64) -> Self {
        let bits = v.to_bits();
        // Handle +0.0 and -0.0
        if bits & 0x7FFF_FFFF_FFFF_FFFF == 0 {
            return Self::ZERO;
        }
        let sign = (bits >> 63) != 0;
        let raw_exp = ((bits >> 52) & 0x7FF) as i32;
        let raw_mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

        if raw_exp == 0x7FF {
            panic!("FixedPoint::from_f64: infinity or NaN");
        }

        let (mantissa, exp_offset) = if raw_exp == 0 {
            (raw_mantissa as i128, -1022 - 52)
        } else {
            ((raw_mantissa | 0x0010_0000_0000_0000) as i128, raw_exp - 1023 - 52)
        };

        let shift = exp_offset + FRAC_BITS;
        let raw = Self::shift_mantissa_to_raw(mantissa, shift);
        if sign { -Self { raw } } else { Self { raw } }
    }

    /// Convert to f32 (lossy: for display/interop only).
    pub fn to_f32(self) -> f32 {
        let sv = self.to_stack_value();
        let s = sv.to_decimal_string(10);
        s.parse::<f32>().unwrap_or(0.0)
    }

    /// Convert to f64 (lossy: for display/interop only).
    pub fn to_f64(self) -> f64 {
        let sv = self.to_stack_value();
        let s = sv.to_decimal_string(MAX_DECIMAL_DIGITS);
        s.parse::<f64>().unwrap_or(0.0)
    }

    /// Parse from a decimal string (e.g., "3.14159").
    ///
    /// Routes through FASC with forced binary mode for correct conversion.
    pub fn from_str(s: &str) -> Self {
        use crate::fixed_point::universal::fasc::mode;

        // Temporarily set binary:binary mode to force binary domain parsing
        let old_mode = mode::get_mode();
        mode::set_mode(mode::GmathMode {
            compute: mode::ComputeMode::Binary,
            output: mode::OutputMode::Binary,
        });

        let expr = gmath_parse(s).expect("FixedPoint::from_str: parse failed");
        let result = evaluate(&expr).expect("FixedPoint::from_str: eval failed");

        // Restore previous mode
        mode::set_mode(old_mode);

        Self::from_stack_value(result)
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
    /// Internally computes at Q64.64 (i128) via the native sincos path,
    /// then narrows to storage tier. Output sin/cos always fits in [-1, 1].
    ///
    /// # Use case
    /// RoPE position encoding where `theta^(2i/d) × position` exceeds storage range.
    /// ```ignore
    /// // Precompute frequency at i64 precision:
    /// let angle_q32: i64 = compute_rope_angle_i64(freq, position);
    /// let (sin_val, cos_val) = FixedPoint::sincos_wide(angle_q32);
    /// ```
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    pub fn sincos_wide(angle_q32_32: i64) -> (Self, Self) {
        use crate::fixed_point::domains::binary_fixed::transcendental::{
            sin_binary_i128, cos_binary_i128,
        };
        // Fixed Q32.32→Q64.64 upscale (always 32, independent of profile FRAC_BITS)
        let angle_q64 = (angle_q32_32 as i128) << 32;
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

    pub(crate) fn from_stack_value(sv: StackValue) -> Self {
        Self::try_from_stack_value(sv).expect("FixedPoint: domain conversion failed")
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

    /// Shift a non-negative mantissa into Q-format raw storage.
    fn shift_mantissa_to_raw(mantissa: i128, shift: i32) -> BinaryStorage {
        #[cfg(table_format = "q16_16")]
        {
            if shift >= 32 {
                panic!("FixedPoint: value too large for Q16.16");
            } else if shift >= 0 {
                // Shift wide, then narrow — mantissa has 53 significant bits
                let wide = mantissa.checked_shl(shift as u32)
                    .expect("FixedPoint: value too large for Q16.16");
                wide as i32
            } else if shift > -128 {
                // Right-shift on full i128 first to preserve precision, then narrow
                (mantissa >> ((-shift) as u32)) as i32
            } else {
                0i32
            }
        }
        #[cfg(table_format = "q32_32")]
        {
            if shift >= 64 {
                panic!("FixedPoint: value too large for Q32.32");
            } else if shift >= 0 {
                // Shift wide, then narrow — mantissa has 53 significant bits
                let wide = mantissa.checked_shl(shift as u32)
                    .expect("FixedPoint: value too large for Q32.32");
                wide as i64
            } else if shift > -128 {
                // Right-shift on full i128 first to preserve precision, then narrow
                (mantissa >> ((-shift) as u32)) as i64
            } else {
                0i64
            }
        }
        #[cfg(table_format = "q64_64")]
        {
            if shift >= 128 {
                panic!("FixedPoint: value too large for Q64.64");
            } else if shift >= 0 {
                mantissa.checked_shl(shift as u32)
                    .expect("FixedPoint: value too large for Q64.64")
            } else if shift > -128 {
                mantissa >> ((-shift) as u32)
            } else {
                0i128
            }
        }
        #[cfg(table_format = "q128_128")]
        {
            let m = I256::from_i128(mantissa);
            if shift >= 256 {
                panic!("FixedPoint: value too large for Q128.128");
            } else if shift >= 0 {
                m << (shift as usize)
            } else if shift > -256 {
                m >> ((-shift) as u32)
            } else {
                I256::zero()
            }
        }
        #[cfg(table_format = "q256_256")]
        {
            let m = I512::from_i128(mantissa);
            if shift >= 512 {
                panic!("FixedPoint: value too large for Q256.256");
            } else if shift >= 0 {
                m << (shift as usize)
            } else if shift > -512 {
                m >> ((-shift) as usize)
            } else {
                I512::zero()
            }
        }
    }
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

impl Add for FixedPoint {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        { Self { raw: self.raw.wrapping_add(rhs.raw) } }
        #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
        { Self { raw: self.raw + rhs.raw } }
    }
}

impl Sub for FixedPoint {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        { Self { raw: self.raw.wrapping_sub(rhs.raw) } }
        #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
        { Self { raw: self.raw - rhs.raw } }
    }
}

impl Mul for FixedPoint {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self { raw: fixed_multiply(self.raw, rhs.raw) }
    }
}

impl Div for FixedPoint {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self { raw: fixed_divide(self.raw, rhs.raw) }
    }
}

impl Neg for FixedPoint {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        { Self { raw: self.raw.wrapping_neg() } }
        #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
        { Self { raw: -self.raw } }
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

/// Multiply two Q-format fixed-point values.
///
/// Uses tier N+1 widening multiplication with right-shift by FRAC_BITS.
#[inline]
fn fixed_multiply(a: BinaryStorage, b: BinaryStorage) -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    {
        // i32*i32→i64, >>FRAC_BITS with round-bit: nearest, ties toward +∞
        // (0.5.0 rounding unification — was floor via bare shift)
        let wide = (a as i64) * (b as i64);
        let round_bit = (wide >> (FRAC_BITS - 1)) & 1;
        ((wide >> FRAC_BITS) + round_bit) as i32
    }
    #[cfg(table_format = "q32_32")]
    {
        // i64*i64→i128, >>32 with round-bit: nearest, ties toward +∞
        // (0.5.0 rounding unification — was floor via bare shift)
        let wide = (a as i128) * (b as i128);
        let round_bit = (wide >> 31) & 1;
        ((wide >> 32) + round_bit) as i64
    }
    #[cfg(table_format = "q64_64")]
    {
        // multiply_binary_i128: i128*i128→I256, >>64, nearest ties toward +∞ (0.5.0)
        multiply_binary_i128(a, b)
    }
    #[cfg(table_format = "q128_128")]
    {
        // I256*I256→I512, >>128: nearest, ties toward +∞ (0.5.0 rounding
        // unification — was truncate toward zero). Computed on magnitudes
        // (mul_to_i512 is unsigned): for a positive result round the
        // magnitude up on remainder >= half (tie goes up = toward +∞);
        // for a negative result round the magnitude up only on
        // remainder > half (tie stays = toward +∞ after negation).
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let product = abs_a.mul_to_i512(abs_b);
        let half = I512::from_i128(1) << 127usize;
        let rem = product & ((I512::from_i128(1) << 128usize) - I512::from_i128(1));
        let mut mag = (product >> 128usize).as_i256();
        let bump = if result_neg { rem > half } else { rem >= half };
        if bump { mag = mag + I256::from_i128(1); }
        if result_neg { -mag } else { mag }
    }
    #[cfg(table_format = "q256_256")]
    {
        // I512*I512→I1024, >>256: nearest, ties toward +∞ (0.5.0 rounding
        // unification — was truncate toward zero; see q128_128 arm).
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let product = abs_a.mul_to_i1024(abs_b);
        let half = I1024::from_i128(1) << 255usize;
        let rem = product & ((I1024::from_i128(1) << 256usize) - I1024::from_i128(1));
        let bump = if result_neg { rem > half } else { rem >= half };
        let mut shifted = (product >> 256usize).as_i512();
        if bump { shifted = shifted + I512::from_i128(1); }
        if result_neg { -shifted } else { shifted }
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
/// Panics on division by zero.
#[inline]
fn fixed_divide(a: BinaryStorage, b: BinaryStorage) -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    {
        let num = (a as i64) << FRAC_BITS;
        let den = b as i64;
        assert!(den != 0, "FixedPoint: division by zero");
        let q = num / den;
        let rem2 = (num - q * den).unsigned_abs() << 1;
        let dabs = den.unsigned_abs();
        let positive = (num < 0) == (den < 0);
        let bump = if positive { rem2 >= dabs } else { rem2 > dabs };
        (if bump { q + if positive { 1 } else { -1 } } else { q }) as i32
    }
    #[cfg(table_format = "q32_32")]
    {
        let num = (a as i128) << 32;
        let den = b as i128;
        assert!(den != 0, "FixedPoint: division by zero");
        let q = num / den;
        let rem2 = (num - q * den).unsigned_abs() << 1;
        let dabs = den.unsigned_abs();
        let positive = (num < 0) == (den < 0);
        let bump = if positive { rem2 >= dabs } else { rem2 > dabs };
        (if bump { q + if positive { 1 } else { -1 } } else { q }) as i64
    }
    #[cfg(table_format = "q64_64")]
    {
        let num = I256::from_i128(a) << 64usize;
        let den = I256::from_i128(b);
        assert!(!den.is_zero(), "FixedPoint: division by zero");
        let q = num / den;
        let rem = num - q * den;
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let den_abs = if den.is_negative() { -den } else { den };
        let positive = num.is_negative() == den.is_negative();
        let rem2 = rem_abs + rem_abs;
        let bump = if positive { rem2 >= den_abs } else { rem2 > den_abs };
        let one = I256::from_i128(1);
        (if bump { if positive { q + one } else { q - one } } else { q }).as_i128()
    }
    #[cfg(table_format = "q128_128")]
    {
        let num = I512::from_i256(a) << 128usize;
        let den = I512::from_i256(b);
        assert!(!den.is_zero(), "FixedPoint: division by zero");
        let q = num / den;
        let rem = num - q * den;
        let rem_abs = if rem.is_negative() { -rem } else { rem };
        let den_abs = if den.is_negative() { -den } else { den };
        let positive = num.is_negative() == den.is_negative();
        let rem2 = rem_abs + rem_abs;
        let bump = if positive { rem2 >= den_abs } else { rem2 > den_abs };
        let one = I512::from_i128(1);
        (if bump { if positive { q + one } else { q - one } } else { q }).as_i256()
    }
    #[cfg(table_format = "q256_256")]
    {
        let num = I1024::from_i512(a) << 256usize;
        let den = I1024::from_i512(b);
        assert!(!den.is_zero(), "FixedPoint: division by zero");
        let q = num / den;
        let rem = num - q * den;
        let rem_abs = if rem < I1024::zero() { -rem } else { rem };
        let den_abs = if den < I1024::zero() { -den } else { den };
        let positive = (num < I1024::zero()) == (den < I1024::zero());
        let rem2 = rem_abs + rem_abs;
        let bump = if positive { rem2 >= den_abs } else { rem2 > den_abs };
        let one = I1024::from_i128(1);
        (if bump { if positive { q + one } else { q - one } } else { q }).as_i512()
    }
}
