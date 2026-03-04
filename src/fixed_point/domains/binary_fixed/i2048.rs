//! 2048-bit Integer Arithmetic for Ultra-Precision Fixed-Point
//!
//! PRODUCTION-READY: Maximum precision arithmetic for Q512.512 multiplication
//! ARCHITECTURE: Based on proven I128/I256/I512/I1024 pattern
//! PRECISION: Foundation for scientific profile exp() with 70+ decimal accuracy
//!
//! **CRITICAL**: This type is REQUIRED for native I1024 × I1024 multiplication
//! Without I2048, multiply_i1024_q512_512() loses ~26.5 ULP per exp() call

use std::ops::{Add, Sub, Mul, Shl, Shr, BitOr, BitAnd, Neg};
use std::cmp::{Ord, PartialOrd, Ordering};
use crate::fixed_point::domains::binary_fixed::i1024::I1024;
use crate::fixed_point::domains::binary_fixed::i512::I512;
use crate::fixed_point::domains::binary_fixed::i256::I256;

/// 2048-bit integer type for ultra-precision intermediate calculations
///
/// ARCHITECTURE: Array of 32 × 64-bit words (little-endian)
/// USAGE: Enables I1024 × I1024 → I2048 native multiplication
/// PRIMARY USE: Scientific profile Q512.512 transcendental functions
#[derive(Clone, Copy, Debug)]
pub struct I2048 {
    /// Stored as thirty-two 64-bit words (little-endian)
    pub words: [u64; 32],
}

impl I2048 {
    #[inline(always)]
    pub const fn zero() -> Self {
        I2048 { words: [0; 32] }
    }

    #[inline(always)]
    pub const fn one() -> Self {
        let mut words = [0u64; 32];
        words[0] = 1;
        I2048 { words }
    }

    /// Maximum value for signed 2048-bit integer (2^2047 - 1)
    #[inline(always)]
    pub const fn max_value() -> Self {
        let mut words = [0xFFFF_FFFF_FFFF_FFFF; 32];
        words[31] = 0x7FFF_FFFF_FFFF_FFFF; // sign bit = 0
        I2048 { words }
    }

    /// Minimum value for signed 2048-bit integer (-2^2047)
    #[inline(always)]
    pub const fn min_value() -> Self {
        let mut words = [0u64; 32];
        words[31] = 0x8000_0000_0000_0000; // sign bit = 1
        I2048 { words }
    }

    #[inline(always)]
    pub const fn from_words(words: [u64; 32]) -> Self {
        I2048 { words }
    }

    /// Create I2048 from I1024 with sign extension
    #[inline(always)]
    pub const fn from_i1024(value: I1024) -> Self {
        let is_negative = (value.words[15] as i64) < 0;
        let sign_extend = if is_negative { u64::MAX } else { 0 };

        I2048 {
            words: [
                value.words[0], value.words[1], value.words[2], value.words[3],
                value.words[4], value.words[5], value.words[6], value.words[7],
                value.words[8], value.words[9], value.words[10], value.words[11],
                value.words[12], value.words[13], value.words[14], value.words[15],
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
            ]
        }
    }

    /// Create I2048 from I512 with sign extension
    #[inline(always)]
    pub const fn from_i512(value: I512) -> Self {
        let is_negative = (value.words[7] as i64) < 0;
        let sign_extend = if is_negative { u64::MAX } else { 0 };

        I2048 {
            words: [
                value.words[0], value.words[1], value.words[2], value.words[3],
                value.words[4], value.words[5], value.words[6], value.words[7],
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
            ]
        }
    }

    /// Create I2048 from I256 with sign extension
    #[inline(always)]
    pub const fn from_i256(value: I256) -> Self {
        let is_negative = (value.words[3] as i64) < 0;
        let sign_extend = if is_negative { u64::MAX } else { 0 };

        I2048 {
            words: [
                value.words[0], value.words[1], value.words[2], value.words[3],
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
            ]
        }
    }

    /// Create I2048 from i128 with sign extension
    #[inline(always)]
    pub const fn from_i128(value: i128) -> Self {
        let is_negative = value < 0;
        let sign_extend = if is_negative { u64::MAX } else { 0 };

        I2048 {
            words: [
                value as u64, (value >> 64) as u64,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend,
            ]
        }
    }

    /// Extract lower 1024 bits as I1024
    #[inline(always)]
    pub fn as_i1024(self) -> I1024 {
        let mut words = [0u64; 16];
        words.copy_from_slice(&self.words[0..16]);
        I1024::from_words(words)
    }

    /// Extract lower 512 bits as I512
    #[inline(always)]
    pub fn as_i512(self) -> I512 {
        let mut words = [0u64; 8];
        words.copy_from_slice(&self.words[0..8]);
        I512::from_words(words)
    }

    /// Extract lower 256 bits as I256
    #[inline(always)]
    pub fn as_i256(self) -> I256 {
        I256::from_words([
            self.words[0], self.words[1], self.words[2], self.words[3],
        ])
    }

    /// Extract lower 128 bits as i128
    #[inline(always)]
    pub fn as_i128(self) -> i128 {
        ((self.words[1] as i128) << 64) | (self.words[0] as i128)
    }

    /// Check if value fits in I1024
    #[inline(always)]
    pub fn fits_in_i1024(self) -> bool {
        let is_negative = (self.words[31] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        (16..32).all(|i| self.words[i] == expected_high)
    }

    /// Check if value fits in I512
    #[inline(always)]
    pub fn fits_in_i512(self) -> bool {
        let is_negative = (self.words[7] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        (8..32).all(|i| self.words[i] == expected_high)
    }

    /// Check if value fits in I256
    #[inline(always)]
    pub fn fits_in_i256(self) -> bool {
        let is_negative = (self.words[3] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        (4..32).all(|i| self.words[i] == expected_high)
    }

    /// Check if value fits in i128
    #[inline(always)]
    pub fn fits_in_i128(self) -> bool {
        let is_negative = (self.words[1] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        (2..32).all(|i| self.words[i] == expected_high)
    }
}

// Comparison operations
impl PartialEq for I2048 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words
    }
}

impl Eq for I2048 {}

impl PartialOrd for I2048 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for I2048 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare as signed 2048-bit integers
        let self_negative = (self.words[31] as i64) < 0;
        let other_negative = (other.words[31] as i64) < 0;

        match (self_negative, other_negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => {
                // Same sign, compare magnitude
                for i in (0..32).rev() {
                    match self.words[i].cmp(&other.words[i]) {
                        Ordering::Equal => continue,
                        other => return if self_negative { other.reverse() } else { other },
                    }
                }
                Ordering::Equal
            }
        }
    }
}

// Arithmetic operations
impl Add for I2048 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut result = [0u64; 32];
        let mut carry = 0u64;

        for i in 0..32 {
            let sum = (self.words[i] as u128) + (rhs.words[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }

        I2048 { words: result }
    }
}

impl Sub for I2048 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut result = [0u64; 32];
        let mut borrow = 0u64;

        for i in 0..32 {
            // Subtract with borrow using overflow detection
            let (diff1, b1) = self.words[i].overflowing_sub(rhs.words[i]);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            result[i] = diff2;
            borrow = (b1 || b2) as u64;
        }

        I2048 { words: result }
    }
}

impl Neg for I2048 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        // Two's complement negation
        let mut result = [0u64; 32];
        let mut carry = 1u64;

        for i in 0..32 {
            let sum = (!self.words[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }

        I2048 { words: result }
    }
}

impl Shl<usize> for I2048 {
    type Output = Self;

    #[inline(always)]
    fn shl(self, shift: usize) -> Self {
        if shift == 0 { return self; }
        if shift >= 2048 { return I2048::zero(); }

        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut result = [0u64; 32];

        if bit_shift == 0 {
            // Simple word shift
            for i in word_shift..32 {
                result[i] = self.words[i - word_shift];
            }
        } else {
            // Bit shift within words
            for i in word_shift..32 {
                let low = if i > word_shift { self.words[i - word_shift - 1] >> (64 - bit_shift) } else { 0 };
                let high = self.words[i - word_shift] << bit_shift;
                result[i] = high | low;
            }
        }

        I2048 { words: result }
    }
}

impl Shr<usize> for I2048 {
    type Output = Self;

    #[inline(always)]
    fn shr(self, shift: usize) -> Self {
        if shift == 0 { return self; }

        if shift >= 2048 {
            // Sign extend based on sign bit
            let sign = (self.words[31] as i64) < 0;
            return I2048 { words: if sign { [u64::MAX; 32] } else { [0; 32] } };
        }

        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut result = [0u64; 32];

        if bit_shift == 0 {
            // Simple word shift
            for i in 0..(32 - word_shift) {
                result[i] = self.words[i + word_shift];
            }
            // Sign extend
            let sign = (self.words[31] as i64) < 0;
            for i in (32 - word_shift)..32 {
                result[i] = if sign { u64::MAX } else { 0 };
            }
        } else {
            // Bit shift within words
            for i in 0..(32 - word_shift) {
                let low = self.words[i + word_shift] >> bit_shift;
                let high = if i + word_shift + 1 < 32 {
                    self.words[i + word_shift + 1] << (64 - bit_shift)
                } else {
                    // Sign extend
                    if (self.words[31] as i64) < 0 {
                        u64::MAX << (64 - bit_shift)
                    } else {
                        0
                    }
                };
                result[i] = low | high;
            }
            // Sign extend remaining words
            let sign = (self.words[31] as i64) < 0;
            for i in (32 - word_shift)..32 {
                result[i] = if sign { u64::MAX } else { 0 };
            }
        }

        I2048 { words: result }
    }
}

impl BitOr for I2048 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        let mut result = [0u64; 32];
        for i in 0..32 {
            result[i] = self.words[i] | rhs.words[i];
        }
        I2048 { words: result }
    }
}

impl BitAnd for I2048 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        let mut result = [0u64; 32];
        for i in 0..32 {
            result[i] = self.words[i] & rhs.words[i];
        }
        I2048 { words: result }
    }
}

// Basic multiplication (optimized for smaller operands)
impl Mul for I2048 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Optimize for common cases
        if self.fits_in_i128() && rhs.fits_in_i128() {
            // Use efficient i128 multiplication
            let a = self.as_i128();
            let b = rhs.as_i128();
            let result = crate::fixed_point::i256::mul_i128_to_i256(a, b);
            I2048::from_i256(result)
        } else if self.fits_in_i512() && rhs.fits_in_i512() {
            // Use I512 multiplication
            let a = self.as_i512();
            let b = rhs.as_i512();
            let result = a.mul_to_i1024(b);
            I2048::from_i1024(result)
        } else {
            // Fallback to simple multiplication for very large values
            self.mul_simple(rhs)
        }
    }
}

impl I2048 {
    /// Simple multiplication implementation (not optimized for very large numbers)
    #[inline(always)]
    fn mul_simple(self, rhs: Self) -> Self {
        let mut result = I2048::zero();

        // Handle signs
        let self_neg = (self.words[31] as i64) < 0;
        let rhs_neg = (rhs.words[31] as i64) < 0;
        let result_neg = self_neg != rhs_neg;

        let self_abs = if self_neg { -self } else { self };
        let rhs_abs = if rhs_neg { -rhs } else { rhs };

        // School multiplication algorithm (limited to avoid overflow)
        for i in 0..32 {
            if rhs_abs.words[i] == 0 { continue; }

            let mut carry = 0u64;
            for j in 0..32 {
                if i + j >= 32 { break; }

                let prod = (self_abs.words[j] as u128) * (rhs_abs.words[i] as u128) + (carry as u128) + (result.words[i + j] as u128);
                result.words[i + j] = prod as u64;
                carry = (prod >> 64) as u64;
            }
        }

        if result_neg {
            -result
        } else {
            result
        }
    }
}

// Conversion traits
impl From<I1024> for I2048 {
    #[inline(always)]
    fn from(value: I1024) -> Self {
        I2048::from_i1024(value)
    }
}

impl From<I512> for I2048 {
    #[inline(always)]
    fn from(value: I512) -> Self {
        I2048::from_i512(value)
    }
}

impl From<I256> for I2048 {
    #[inline(always)]
    fn from(value: I256) -> Self {
        I2048::from_i256(value)
    }
}

impl From<i128> for I2048 {
    #[inline(always)]
    fn from(value: i128) -> Self {
        I2048::from_i128(value)
    }
}

impl From<i64> for I2048 {
    #[inline(always)]
    fn from(value: i64) -> Self {
        I2048::from_i128(value as i128)
    }
}

/// Schoolbook long division: I2048 / I2048 → I2048
///
/// **ALGORITHM**: Bit-by-bit long division, processing 2048 bits from MSB to LSB.
/// This exists because I2048 does not implement the Div trait.
/// Used by compute-tier division in the scientific profile (Q512.512 format).
pub fn i2048_div(dividend: I2048, divisor: I2048) -> I2048 {
    if divisor == I2048::zero() {
        // Division by zero — return zero (caller should check)
        return I2048::zero();
    }

    // Handle signs manually
    let dividend_neg = (dividend.words[31] & 0x8000_0000_0000_0000) != 0;
    let divisor_neg = (divisor.words[31] & 0x8000_0000_0000_0000) != 0;

    let abs_dividend = if dividend_neg { -dividend } else { dividend };
    let abs_divisor = if divisor_neg { -divisor } else { divisor };

    let mut quotient = I2048::zero();
    let mut remainder = I2048::zero();

    // Process 2048 bits from MSB to LSB
    for i in (0..2048).rev() {
        remainder = remainder << 1;
        let word_idx = i / 64;
        let bit_idx = i % 64;
        if (abs_dividend.words[word_idx] >> bit_idx) & 1 == 1 {
            remainder = remainder + I2048::one();
        }
        if remainder >= abs_divisor {
            remainder = remainder - abs_divisor;
            let mut q_words = quotient.words;
            q_words[word_idx] |= 1u64 << bit_idx;
            quotient = I2048::from_words(q_words);
        }
    }

    let result_neg = dividend_neg != divisor_neg;
    if result_neg { -quotient } else { quotient }
}

/// Schoolbook long division: I2048 / I1024 → I2048
///
/// Specialized version for dividing an I2048 by an I1024 divisor.
/// More commonly needed than I2048/I2048 (e.g., ternary tier 6 multiply).
pub fn i2048_div_by_i1024(dividend: I2048, divisor: I1024) -> I2048 {
    let divisor_wide = I2048::from_i1024(divisor);
    i2048_div(dividend, divisor_wide)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i2048_basic_operations() {
        let a = I2048::from_i128(42);
        let b = I2048::from_i128(17);

        assert_eq!((a + b).as_i128(), 59);
        assert_eq!((a - b).as_i128(), 25);
        assert_eq!((a * b).as_i128(), 714);
    }

    #[test]
    fn test_i2048_from_i1024() {
        let i1024_val = I1024::from_i128(0x123456789ABCDEF0_i128);
        let i2048_val = I2048::from_i1024(i1024_val);

        assert_eq!(i2048_val.as_i1024(), i1024_val);
        assert_eq!(i2048_val.as_i128(), 0x123456789ABCDEF0_i128);
    }

    #[test]
    fn test_i2048_shift_operations() {
        let value = I2048::from_i128(0xFFFFFFFF00000000_i128);
        let shifted = value >> 32;
        assert_eq!(shifted.as_i128(), 0xFFFFFFFF_i128);

        let left_shifted = I2048::from_i128(1) << 1200;
        assert_ne!(left_shifted.as_i128(), 1); // Should overflow i128 range
    }

    #[test]
    fn test_fit_checks() {
        let small = I2048::from_i128(42);
        assert!(small.fits_in_i128());
        assert!(small.fits_in_i256());
        assert!(small.fits_in_i512());
        assert!(small.fits_in_i1024());

        let large = I2048::from_i1024(I1024::one_q512_512());
        assert!(!large.fits_in_i128());
        assert!(!large.fits_in_i256());
        assert!(!large.fits_in_i512());
    }

    #[test]
    fn test_i2048_negation() {
        let value = I2048::from_i128(42);
        let negated = -value;
        assert_eq!(negated.as_i128(), -42);

        let double_neg = -negated;
        assert_eq!(double_neg, value);
    }

    #[test]
    fn test_i2048_comparison() {
        let a = I2048::from_i128(100);
        let b = I2048::from_i128(200);
        let c = I2048::from_i128(-50);

        assert!(a < b);
        assert!(b > a);
        assert!(c < a);
        assert_eq!(a, I2048::from_i128(100));
    }
}
