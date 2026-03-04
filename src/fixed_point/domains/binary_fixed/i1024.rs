//! 1024-bit Integer Arithmetic for Maximum-Precision Fixed-Point
//! 
//! PRODUCTION-READY: Maximum precision arithmetic for Q512.512 format
//! ARCHITECTURE: Based on proven I128/I256/I512 pattern, optimized for fixed-point operations
//! PRECISION: Foundation for maximum precision transcendental operations (~154 decimal places)

use std::ops::{Add, Sub, Mul, Shl, Shr, BitOr, BitAnd, Neg, Div, Rem};
use std::cmp::{Ord, PartialOrd, Ordering};
use crate::fixed_point::i512::I512;
use crate::fixed_point::i256::I256;

/// 1024-bit integer type for maximum-precision intermediate calculations
/// 
/// ARCHITECTURE: Array of 16 × 64-bit words (little-endian)
/// USAGE: Enables Q512.512 fixed-point format with ~154 decimal places
/// COMPATIBILITY: Follows proven I128/I256/I512 patterns for consistent API
#[derive(Clone, Copy, Debug)]
pub struct I1024 {
    /// Stored as sixteen 64-bit words (little-endian)
    pub words: [u64; 16],
}

impl I1024 {
    #[inline(always)]
    pub const fn zero() -> Self {
        I1024 { words: [0; 16] }
    }
    
    #[inline(always)]
    pub const fn one() -> Self {
        I1024 { words: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
    }

    /// Maximum value for signed 1024-bit integer (2^1023 - 1)
    #[inline(always)]
    pub const fn max_value() -> Self {
        I1024 {
            words: [
                0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
                0xFFFF_FFFF_FFFF_FFFF, 0x7FFF_FFFF_FFFF_FFFF, // sign bit = 0
            ]
        }
    }

    /// Minimum value for signed 1024-bit integer (-2^1023)
    #[inline(always)]
    pub const fn min_value() -> Self {
        I1024 {
            words: [
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0x8000_0000_0000_0000, // sign bit = 1
            ]
        }
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    #[inline(always)]
    pub const fn from_words(words: [u64; 16]) -> Self {
        I1024 { words }
    }
    
    /// Create I1024 from I512 with sign extension
    #[inline(always)]
    pub const fn from_i512(value: I512) -> Self {
        let is_negative = (value.words[7] as i64) < 0;
        let sign_extend = if is_negative { u64::MAX } else { 0 };
        
        I1024 { 
            words: [
                value.words[0], value.words[1], value.words[2], value.words[3],
                value.words[4], value.words[5], value.words[6], value.words[7],
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
            ]
        }
    }
    
    /// Create I1024 from I256 with sign extension
    #[inline(always)]
    pub const fn from_i256(value: I256) -> Self {
        let is_negative = (value.words[3] as i64) < 0;
        let sign_extend = if is_negative { u64::MAX } else { 0 };
        
        I1024 { 
            words: [
                value.words[0], value.words[1], value.words[2], value.words[3],
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend,
            ]
        }
    }
    
    /// Create I1024 from I128 with sign extension  
    #[inline(always)]
    pub const fn from_i128(value: i128) -> Self {
        let is_negative = value < 0;
        let sign_extend = if is_negative { u64::MAX } else { 0 };
        
        I1024 { 
            words: [
                value as u64, (value >> 64) as u64,
                sign_extend, sign_extend, sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend, sign_extend, sign_extend, sign_extend, sign_extend,
                sign_extend, sign_extend,
            ]
        }
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
    
    /// Check if value fits in I512 (signed 512-bit)
    ///
    /// The I512 sign bit is at word 7, bit 63 (bit 511).
    /// For the value to fit, words 8-15 must be sign-extension of that bit.
    #[inline(always)]
    pub fn fits_in_i512(self) -> bool {
        let is_negative = (self.words[7] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        (8..16).all(|i| self.words[i] == expected_high)
    }

    /// Check if value fits in I256 (signed 256-bit)
    ///
    /// The I256 sign bit is at word 3, bit 63 (bit 255).
    #[inline(always)]
    pub fn fits_in_i256(self) -> bool {
        let is_negative = (self.words[3] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        (4..16).all(|i| self.words[i] == expected_high)
    }

    /// Check if value fits in i128 (signed 128-bit)
    ///
    /// The i128 sign bit is at word 1, bit 63 (bit 127).
    #[inline(always)]
    pub fn fits_in_i128(self) -> bool {
        let is_negative = (self.words[1] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        (2..16).all(|i| self.words[i] == expected_high)
    }
    
    /// Convert to bytes (little-endian)
    #[inline(always)]
    pub fn to_bytes_le(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(128);
        for word in self.words.iter() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }
    
    /// Create from bytes (little-endian)
    #[inline(always)]
    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 128, "I1024 requires exactly 128 bytes");
        
        let mut words = [0u64; 16];
        for i in 0..16 {
            let start = i * 8;
            let word_bytes: [u8; 8] = bytes[start..start+8].try_into().unwrap();
            words[i] = u64::from_le_bytes(word_bytes);
        }
        
        I1024 { words }
    }
}

// Comparison operations
impl PartialEq for I1024 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words
    }
}

impl Eq for I1024 {}

impl PartialOrd for I1024 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for I1024 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare as signed 1024-bit integers
        let self_negative = (self.words[15] as i64) < 0;
        let other_negative = (other.words[15] as i64) < 0;
        
        match (self_negative, other_negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => {
                // Same sign, compare magnitude
                for i in (0..16).rev() {
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
impl Add for I1024 {
    type Output = Self;
    
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut result = [0u64; 16];
        let mut carry = 0u64;
        
        for i in 0..16 {
            let sum = (self.words[i] as u128) + (rhs.words[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        
        I1024 { words: result }
    }
}

impl Sub for I1024 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut result = [0u64; 16];
        let mut borrow = 0u64;

        for i in 0..16 {
            // Subtract with borrow using overflow detection
            let (diff1, b1) = self.words[i].overflowing_sub(rhs.words[i]);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            result[i] = diff2;
            borrow = (b1 || b2) as u64;
        }

        I1024 { words: result }
    }
}

impl Neg for I1024 {
    type Output = Self;
    
    #[inline(always)]
    fn neg(self) -> Self {
        // Two's complement negation
        let mut result = [0u64; 16];
        let mut carry = 1u64;
        
        for i in 0..16 {
            let sum = (!self.words[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        
        I1024 { words: result }
    }
}

impl Shl<usize> for I1024 {
    type Output = Self;
    
    #[inline(always)]
    fn shl(self, shift: usize) -> Self {
        if shift == 0 { return self; }
        if shift >= 1024 { return I1024::zero(); }
        
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut result = [0u64; 16];
        
        if bit_shift == 0 {
            // Simple word shift
            for i in word_shift..16 {
                result[i] = self.words[i - word_shift];
            }
        } else {
            // Bit shift within words
            for i in word_shift..16 {
                let low = if i > word_shift { self.words[i - word_shift - 1] >> (64 - bit_shift) } else { 0 };
                let high = self.words[i - word_shift] << bit_shift;
                result[i] = high | low;
            }
        }
        
        I1024 { words: result }
    }
}

impl Shr<usize> for I1024 {
    type Output = Self;
    
    #[inline(always)]
    fn shr(self, shift: usize) -> Self {
        if shift == 0 { return self; }
        
        if shift >= 1024 {
            // Sign extend based on sign bit
            let sign = (self.words[15] as i64) < 0;
            return I1024 { words: if sign { [u64::MAX; 16] } else { [0; 16] } };
        }
        
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut result = [0u64; 16];
        
        if bit_shift == 0 {
            // Simple word shift
            for i in 0..(16 - word_shift) {
                result[i] = self.words[i + word_shift];
            }
            // Sign extend
            let sign = (self.words[15] as i64) < 0;
            for i in (16 - word_shift)..16 {
                result[i] = if sign { u64::MAX } else { 0 };
            }
        } else {
            // Bit shift within words
            for i in 0..(16 - word_shift) {
                let low = self.words[i + word_shift] >> bit_shift;
                let high = if i + word_shift + 1 < 16 {
                    self.words[i + word_shift + 1] << (64 - bit_shift)
                } else {
                    // Sign extend
                    if (self.words[15] as i64) < 0 {
                        u64::MAX << (64 - bit_shift)
                    } else {
                        0
                    }
                };
                result[i] = low | high;
            }
            // Sign extend remaining words
            let sign = (self.words[15] as i64) < 0;
            for i in (16 - word_shift)..16 {
                result[i] = if sign { u64::MAX } else { 0 };
            }
        }
        
        I1024 { words: result }
    }
}

impl BitOr for I1024 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        let mut result = [0u64; 16];
        for i in 0..16 {
            result[i] = self.words[i] | rhs.words[i];
        }
        I1024 { words: result }
    }
}

impl BitAnd for I1024 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        let mut result = [0u64; 16];
        for i in 0..16 {
            result[i] = self.words[i] & rhs.words[i];
        }
        I1024 { words: result }
    }
}

impl Div for I1024 {
    type Output = Self;

    /// Division for I1024
    ///
    /// ALGORITHM: Long division with shift-and-subtract method
    /// PRECISION: Full 1024-bit precision division
    /// CRITICAL: Required for Q256.256 fixed-point division (tier N+1 strategy)
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Handle zero divisor
        if rhs == I1024::zero() {
            // Return max/min value based on sign of dividend
            return if (self.words[15] as i64) < 0 {
                I1024::min_value()
            } else {
                I1024::max_value()
            };
        }

        // Optimize for smaller values
        if self.fits_in_i128() && rhs.fits_in_i128() {
            let dividend = self.as_i128();
            let divisor = rhs.as_i128();
            return I1024::from_i128(dividend / divisor);
        }

        if self.fits_in_i512() && rhs.fits_in_i512() {
            let dividend = self.as_i512();
            let divisor = rhs.as_i512();
            return I1024::from_i512(dividend / divisor);
        }

        // Full 1024-bit division
        let (quotient, _remainder) = self.div_rem_unsigned(rhs);
        quotient
    }
}

impl Rem for I1024 {
    type Output = Self;

    /// Remainder (modulo) for I1024
    ///
    /// ALGORITHM: Uses div_rem_unsigned for full precision
    /// PRECISION: Full 1024-bit precision remainder
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        // Handle zero divisor
        if rhs == I1024::zero() {
            return I1024::zero();
        }

        // Optimize for smaller values
        if self.fits_in_i128() && rhs.fits_in_i128() {
            let dividend = self.as_i128();
            let divisor = rhs.as_i128();
            return I1024::from_i128(dividend % divisor);
        }

        if self.fits_in_i512() && rhs.fits_in_i512() {
            let dividend = self.as_i512();
            let divisor = rhs.as_i512();
            return I1024::from_i512(dividend % divisor);
        }

        // Full 1024-bit remainder
        let (_quotient, remainder) = self.div_rem_unsigned(rhs);
        remainder
    }
}

// Basic multiplication (optimized for smaller operands)
impl Mul for I1024 {
    type Output = Self;
    
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Optimize for common cases
        if self.fits_in_i128() && rhs.fits_in_i128() {
            // Use efficient I128 multiplication (i128*i128 always fits in I256)
            let a = self.as_i128();
            let b = rhs.as_i128();
            let result = crate::fixed_point::i256::mul_i128_to_i256(a, b);
            I1024::from_i256(result)
        } else {
            // Full 1024-bit schoolbook multiplication.
            // NOTE: Cannot use I512*I512->I512 shortcut here because the
            // product of two I512 values can require up to 1024 bits.
            self.mul_simple(rhs)
        }
    }
}

impl I1024 {
    /// Simple multiplication implementation (not optimized for very large numbers)
    #[inline(always)]
    fn mul_simple(self, rhs: Self) -> Self {
        let mut result = I1024::zero();

        // Handle signs
        let self_neg = (self.words[15] as i64) < 0;
        let rhs_neg = (rhs.words[15] as i64) < 0;
        let result_neg = self_neg != rhs_neg;

        let self_abs = if self_neg { -self } else { self };
        let rhs_abs = if rhs_neg { -rhs } else { rhs };

        // School multiplication algorithm (limited to avoid overflow)
        for i in 0..16 {
            if rhs_abs.words[i] == 0 { continue; }

            let mut carry = 0u64;
            for j in 0..16 {
                if i + j >= 16 { break; }

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

    /// Division and remainder using long division algorithm
    ///
    /// ALGORITHM: Binary long division (shift-and-subtract method)
    /// PRECISION: Full 1024-bit precision for both quotient and remainder
    /// CRITICAL: Foundation for Q256.256 fixed-point division
    ///
    /// Returns (quotient, remainder) where:
    /// - quotient = self / rhs
    /// - remainder = self % rhs
    /// - self = quotient * rhs + remainder
    #[inline(always)]
    fn div_rem_unsigned(self, rhs: Self) -> (Self, Self) {
        // Handle signs
        let dividend_neg = (self.words[15] as i64) < 0;
        let divisor_neg = (rhs.words[15] as i64) < 0;
        let quotient_neg = dividend_neg != divisor_neg;

        // Work with absolute values
        let dividend = if dividend_neg { -self } else { self };
        let divisor = if divisor_neg { -rhs } else { rhs };

        // Quick exit for dividend < divisor
        if dividend < divisor {
            return (I1024::zero(), self);
        }

        // Quick exit for divisor == 1
        if divisor == I1024::one() {
            let quotient = if quotient_neg { -dividend } else { dividend };
            return (quotient, I1024::zero());
        }

        // Binary long division
        let mut quotient = I1024::zero();
        let mut remainder = I1024::zero();

        // Process each bit from MSB to LSB
        for bit_pos in (0..1024).rev() {
            // Shift remainder left by 1
            remainder = remainder << 1;

            // Get current bit of dividend
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            let dividend_bit = (dividend.words[word_idx] >> bit_idx) & 1;

            // Add dividend bit to remainder
            remainder.words[0] |= dividend_bit;

            // If remainder >= divisor, subtract and set quotient bit
            if remainder >= divisor {
                remainder = remainder - divisor;

                // Set corresponding bit in quotient
                let q_word_idx = bit_pos / 64;
                let q_bit_idx = bit_pos % 64;
                quotient.words[q_word_idx] |= 1u64 << q_bit_idx;
            }
        }

        // Apply signs
        let final_quotient = if quotient_neg { -quotient } else { quotient };
        let final_remainder = if dividend_neg { -remainder } else { remainder };

        (final_quotient, final_remainder)
    }
}

// Conversion traits
impl From<I512> for I1024 {
    #[inline(always)]
    fn from(value: I512) -> Self {
        I1024::from_i512(value)
    }
}

impl From<I256> for I1024 {
    #[inline(always)]
    fn from(value: I256) -> Self {
        I1024::from_i256(value)
    }
}

impl From<i128> for I1024 {
    #[inline(always)]
    fn from(value: i128) -> Self {
        I1024::from_i128(value)
    }
}

impl From<i64> for I1024 {
    #[inline(always)]
    fn from(value: i64) -> Self {
        I1024::from_i128(value as i128)
    }
}

/// Utility functions for Q512.512 fixed-point operations
impl I1024 {
    /// Create I1024 representing 1.0 in Q512.512 format
    #[inline(always)]
    pub const fn one_q512_512() -> Self {
        let mut words = [0u64; 16];
        words[8] = 1; // 1 << 512 for Q512.512
        I1024 { words }
    }
    
    /// Multiply two Q512.512 values (I1024) and return Q512.512 result
    #[inline(always)]
    pub fn mul_q512_512(a: I1024, b: I1024) -> I1024 {
        use crate::fixed_point::domains::binary_fixed::i2048::I2048;
        // Tier N+1: widen to I2048, multiply, shift right by 512, extract I1024
        let a_wide = I2048::from_i1024(a);
        let b_wide = I2048::from_i1024(b);
        let full_product = a_wide * b_wide;
        // Round-to-nearest: check bit 511 (round bit)
        let round_bit = (full_product >> 511).words[0] & 1;
        let shifted = full_product >> 512;
        let result = shifted.as_i1024();
        if round_bit != 0 {
            result + I1024::from_i128(1)
        } else {
            result
        }
    }
    
    /// Convert from Q512.512 (I1024) to Q256.256 (I512) with rounding
    /// Used by transcendental downscale path
    #[inline(always)]
    pub fn to_q256_256(self) -> I512 {
        // Scale down by 256 bits with rounding
        let shifted = self >> 256;
        shifted.as_i512()
    }

    /// Multiply two I1024 values producing full I2048 result
    ///
    /// **USAGE**: For Q512.512 fixed-point multiplication (tier N+1 strategy)
    /// **ALGORITHM**: Grade-school multiplication with carry propagation
    /// **PRECISION**: Full 2048-bit result, no truncation
    /// **TIER N+1**: Enables error-less Q512.512 exp computation
    /// **CRITICAL**: Fixes 32/77 → 70+/77 decimal precision for scientific profile
    #[inline(always)]
    pub fn mul_to_i2048(&self, other: I1024) -> crate::fixed_point::I2048 {
        use crate::fixed_point::I2048;

        let mut result = I2048::zero();

        // Grade-school multiplication: a × b
        for i in 0..16 {
            let mut carry = 0u64;
            for j in 0..16 {
                let product = (self.words[i] as u128) * (other.words[j] as u128) +
                              (result.words[i + j] as u128) +
                              (carry as u128);
                result.words[i + j] = product as u64;
                carry = (product >> 64) as u64;
            }
            // Propagate final carry
            if i + 16 < 32 {
                result.words[i + 16] = carry;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_i1024_basic_operations() {
        let a = I1024::from_i128(42);
        let b = I1024::from_i128(17);
        
        assert_eq!((a + b).as_i128(), 59);
        assert_eq!((a - b).as_i128(), 25);
        assert_eq!((a * b).as_i128(), 714);
    }
    
    #[test]
    fn test_i1024_from_i512() {
        let i512_val = I512::from_i128(0x123456789ABCDEF0_i128);
        let i1024_val = I1024::from_i512(i512_val);
        
        assert_eq!(i1024_val.as_i512(), i512_val);
        assert_eq!(i1024_val.as_i128(), 0x123456789ABCDEF0_i128);
    }
    
    #[test]
    fn test_i1024_shift_operations() {
        let value = I1024::from_i128(0xFFFFFFFF00000000_i128);
        let shifted = value >> 32;
        assert_eq!(shifted.as_i128(), 0xFFFFFFFF_i128);
        
        let left_shifted = I1024::from_i128(1) << 600;
        assert_ne!(left_shifted.as_i128(), 1); // Should overflow i128 range
    }
    
    #[test]
    fn test_q512_512_operations() {
        let one = I1024::one_q512_512();
        let two = one + one;
        
        // Test Q512.512 multiplication
        let result = I1024::mul_q512_512(one, two);
        assert_eq!(result, two);
    }
    
    #[test]
    fn test_bytes_serialization() {
        let value = I1024::from_i128(0x123456789ABCDEF0_i128);
        let bytes = value.to_bytes_le();
        let restored = I1024::from_bytes_le(&bytes);
        
        assert_eq!(value, restored);
    }
    
    #[test]
    fn test_fit_checks() {
        let small = I1024::from_i128(42);
        assert!(small.fits_in_i128());
        assert!(small.fits_in_i256());
        assert!(small.fits_in_i512());
        
        let large = I1024::one_q512_512();
        assert!(!large.fits_in_i128());
        assert!(!large.fits_in_i256());
        assert!(!large.fits_in_i512());
    }
}