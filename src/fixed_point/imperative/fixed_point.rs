//! FixedPoint — Copy-able binary fixed-point numeric type for imperative computation.
//!
//! Wraps the raw `BinaryStorage` Q-format integer, providing direct arithmetic
//! operators and transcendental methods (routed through FASC).

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

use crate::fixed_point::canonical::{
    LazyExpr, StackValue, evaluate, gmath_parse, CompactShadow,
};
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
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
const FRAC_BITS: i32 = 16;
#[cfg(table_format = "q32_32")]
const FRAC_BITS: i32 = 32;
#[cfg(table_format = "q64_64")]
const FRAC_BITS: i32 = 64;
#[cfg(table_format = "q128_128")]
const FRAC_BITS: i32 = 128;
#[cfg(table_format = "q256_256")]
const FRAC_BITS: i32 = 256;

#[cfg(table_format = "q16_16")]
const MAX_DECIMAL_DIGITS: usize = 4;
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
        { Self { raw: 1i32 << 16 } }
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
        { Self { raw: (v as i32) << 16 } }
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
        { (self.raw >> 16) as i32 }
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
    /// Uses IEEE 754 bit extraction for exact conversion — no float arithmetic
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
    /// Uses IEEE 754 bit extraction for exact conversion — no float arithmetic
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

    /// Convert to f32 (lossy — for display/interop only).
    pub fn to_f32(self) -> f32 {
        let sv = self.to_stack_value();
        let s = sv.to_decimal_string(10);
        s.parse::<f32>().unwrap_or(0.0)
    }

    /// Convert to f64 (lossy — for display/interop only).
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
    // Transcendentals — route through FASC at tier N+1
    // ========================================================================

    /// e^x
    pub fn exp(self) -> Self { self.apply_unary(LazyExpr::exp) }
    /// ln(x), x > 0
    pub fn ln(self) -> Self { self.apply_unary(LazyExpr::ln) }
    /// sqrt(x), x >= 0
    pub fn sqrt(self) -> Self { self.apply_unary(LazyExpr::sqrt) }
    /// sin(x)
    pub fn sin(self) -> Self { self.apply_unary(LazyExpr::sin) }
    /// cos(x)
    pub fn cos(self) -> Self { self.apply_unary(LazyExpr::cos) }
    /// tan(x)
    pub fn tan(self) -> Self { self.apply_unary(LazyExpr::tan) }
    /// atan(x)
    pub fn atan(self) -> Self { self.apply_unary(LazyExpr::atan) }
    /// asin(x), |x| <= 1
    pub fn asin(self) -> Self { self.apply_unary(LazyExpr::asin) }
    /// acos(x), |x| <= 1
    pub fn acos(self) -> Self { self.apply_unary(LazyExpr::acos) }
    /// sinh(x)
    pub fn sinh(self) -> Self { self.apply_unary(LazyExpr::sinh) }
    /// cosh(x)
    pub fn cosh(self) -> Self { self.apply_unary(LazyExpr::cosh) }
    /// tanh(x)
    pub fn tanh(self) -> Self { self.apply_unary(LazyExpr::tanh) }
    /// asinh(x)
    pub fn asinh(self) -> Self { self.apply_unary(LazyExpr::asinh) }
    /// acosh(x), x >= 1
    pub fn acosh(self) -> Self { self.apply_unary(LazyExpr::acosh) }
    /// atanh(x), |x| < 1
    pub fn atanh(self) -> Self { self.apply_unary(LazyExpr::atanh) }

    /// x^y = exp(y * ln(x))
    pub fn pow(self, exponent: Self) -> Self {
        self.try_pow(exponent).expect("pow: overflow or domain error")
    }

    /// atan2(self=y, x) — angle of point (x, y)
    pub fn atan2(self, x: Self) -> Self {
        self.try_atan2(x).expect("atan2 failed")
    }

    // ========================================================================
    // UGOD-aware try_* transcendentals — return Result instead of panicking
    // ========================================================================

    /// Fallible e^x — returns `Err(TierOverflow)` if result exceeds storage tier.
    pub fn try_exp(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::exp) }
    /// Fallible ln(x) — returns `Err(DomainError)` if x <= 0.
    pub fn try_ln(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::ln) }
    /// Fallible sqrt(x) — returns `Err(DomainError)` if x < 0.
    pub fn try_sqrt(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::sqrt) }
    /// Fallible sin(x).
    pub fn try_sin(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::sin) }
    /// Fallible cos(x).
    pub fn try_cos(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::cos) }
    /// Fused sin+cos — single shared range reduction at compute tier.
    /// Returns (sin(x), cos(x)). More efficient than separate try_sin + try_cos.
    pub fn try_sincos(self) -> Result<(Self, Self), OverflowDetected> {
        use super::linalg::{upscale_to_compute, round_to_storage, sincos_at_compute_tier};
        let compute_val = upscale_to_compute(self.raw());
        let (sin_c, cos_c) = sincos_at_compute_tier(compute_val);
        Ok((Self::from_raw(round_to_storage(sin_c)), Self::from_raw(round_to_storage(cos_c))))
    }
    /// Fallible tan(x).
    pub fn try_tan(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::tan) }
    /// Fallible atan(x).
    pub fn try_atan(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::atan) }
    /// Fallible asin(x) — returns `Err(DomainError)` if |x| > 1.
    pub fn try_asin(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::asin) }
    /// Fallible acos(x) — returns `Err(DomainError)` if |x| > 1.
    pub fn try_acos(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::acos) }
    /// Fallible sinh(x).
    pub fn try_sinh(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::sinh) }
    /// Fallible cosh(x).
    pub fn try_cosh(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::cosh) }
    /// Fallible tanh(x).
    pub fn try_tanh(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::tanh) }
    /// Fallible asinh(x).
    pub fn try_asinh(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::asinh) }
    /// Fallible acosh(x) — returns `Err(DomainError)` if x < 1.
    pub fn try_acosh(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::acosh) }
    /// Fallible atanh(x) — returns `Err(DomainError)` if |x| >= 1.
    pub fn try_atanh(self) -> Result<Self, OverflowDetected> { self.try_apply_unary(LazyExpr::atanh) }

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
                (mantissa as i32).checked_shl(shift as u32)
                    .expect("FixedPoint: value too large for Q16.16")
            } else if shift > -32 {
                (mantissa as i32) >> ((-shift) as u32)
            } else {
                0i32
            }
        }
        #[cfg(table_format = "q32_32")]
        {
            if shift >= 64 {
                panic!("FixedPoint: value too large for Q32.32");
            } else if shift >= 0 {
                (mantissa as i64).checked_shl(shift as u32)
                    .expect("FixedPoint: value too large for Q32.32")
            } else if shift > -64 {
                (mantissa as i64) >> ((-shift) as u32)
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
        // i32*i32→i64, >>16, truncate to i32
        let wide = (a as i64) * (b as i64);
        (wide >> 16) as i32
    }
    #[cfg(table_format = "q32_32")]
    {
        // i64*i64→i128, >>32, truncate to i64
        let wide = (a as i128) * (b as i128);
        (wide >> 32) as i64
    }
    #[cfg(table_format = "q64_64")]
    {
        // multiply_binary_i128 does: i128*i128→I256, >>64, banker's rounding
        multiply_binary_i128(a, b)
    }
    #[cfg(table_format = "q128_128")]
    {
        // I256*I256→I512, >>128, truncate to I256
        // Sign-correct widening multiply
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let product = abs_a.mul_to_i512(abs_b);
        let shifted = (product >> 128usize).as_i256();
        if result_neg { -shifted } else { shifted }
    }
    #[cfg(table_format = "q256_256")]
    {
        // I512*I512→I1024, >>256, truncate to I512
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let product = abs_a.mul_to_i1024(abs_b);
        let shifted = (product >> 256usize).as_i512();
        if result_neg { -shifted } else { shifted }
    }
}

// ============================================================================
// Q-format fixed-point divide
// ============================================================================

/// Divide two Q-format fixed-point values.
///
/// Uses tier N+1 widening: (a << FRAC_BITS) / b.
/// Panics on division by zero.
#[inline]
fn fixed_divide(a: BinaryStorage, b: BinaryStorage) -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    {
        let num = (a as i64) << 16;
        let den = b as i64;
        assert!(den != 0, "FixedPoint: division by zero");
        (num / den) as i32
    }
    #[cfg(table_format = "q32_32")]
    {
        let num = (a as i128) << 32;
        let den = b as i128;
        assert!(den != 0, "FixedPoint: division by zero");
        (num / den) as i64
    }
    #[cfg(table_format = "q64_64")]
    {
        let num = I256::from_i128(a) << 64usize;
        let den = I256::from_i128(b);
        assert!(!den.is_zero(), "FixedPoint: division by zero");
        (num / den).as_i128()
    }
    #[cfg(table_format = "q128_128")]
    {
        let num = I512::from_i256(a) << 128usize;
        let den = I512::from_i256(b);
        assert!(!den.is_zero(), "FixedPoint: division by zero");
        (num / den).as_i256()
    }
    #[cfg(table_format = "q256_256")]
    {
        let num = I1024::from_i512(a) << 256usize;
        let den = I1024::from_i512(b);
        assert!(!den.is_zero(), "FixedPoint: division by zero");
        (num / den).as_i512()
    }
}
