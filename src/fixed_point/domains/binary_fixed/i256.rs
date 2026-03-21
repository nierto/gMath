//! 256-bit Integer Arithmetic Foundation
//! 
//! PRODUCTION-READY: Core 256-bit integer type supporting the entire transcendental suite
//! PRECISION: I256 intermediate used by Q64.64 tier N+1 computations

use std::ops::{Mul, Shr, Neg};
use std::convert::From;

/// Custom 256-bit integer type for intermediate calculations
/// 
/// 256-bit signed integer for Q64.64 compute-tier arithmetic
/// Used by multiplication, division, exponential, and future transcendental functions
#[derive(Clone, Copy, Debug)]
pub struct I256 {
    /// Stored as four 64-bit words (little-endian)
    pub words: [u64; 4],
}

impl I256 {
    #[inline(always)]
    pub const fn zero() -> Self {
        I256 { words: [0, 0, 0, 0] }
    }

    /// Maximum value for signed 256-bit integer (2^255 - 1)
    #[inline(always)]
    pub const fn max_value() -> Self {
        I256 {
            words: [
                0xFFFF_FFFF_FFFF_FFFF,  // word 0
                0xFFFF_FFFF_FFFF_FFFF,  // word 1
                0xFFFF_FFFF_FFFF_FFFF,  // word 2
                0x7FFF_FFFF_FFFF_FFFF,  // word 3 (sign bit = 0)
            ]
        }
    }

    /// Minimum value for signed 256-bit integer (-2^255)
    #[inline(always)]
    pub const fn min_value() -> Self {
        I256 {
            words: [
                0,  // word 0
                0,  // word 1
                0,  // word 2
                0x8000_0000_0000_0000,  // word 3 (sign bit = 1)
            ]
        }
    }

    /// Check if the I256 value is zero
    /// 
    /// ALGORITHM: Check if all words are zero
    /// ARCHITECTURAL_PURPOSE: Supports extended precision arithmetic operations
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.words[0] == 0 && self.words[1] == 0 && self.words[2] == 0 && self.words[3] == 0
    }
    
    #[inline(always)]
    pub const fn from_words(words: [u64; 4]) -> Self {
        I256 { words }
    }
    
    #[inline(always)]
    pub const fn from_i128(value: i128) -> Self {
        let words = if value < 0 {
            // Sign extend for negative values
            [
                value as u64,
                (value >> 64) as u64,
                u64::MAX,
                u64::MAX,
            ]
        } else {
            [
                value as u64,
                (value >> 64) as u64,
                0,
                0,
            ]
        };
        I256 { words }
    }
    
    #[inline(always)]
    pub fn as_i128(self) -> i128 {
        // Take lower 128 bits
        ((self.words[1] as i128) << 64) | (self.words[0] as i128)
    }
    
    #[inline(always)]
    pub fn as_u128(self) -> u128 {
        // Take lower 128 bits
        ((self.words[1] as u128) << 64) | (self.words[0] as u128)
    }
    
    /// Check if the I256 value is negative
    /// 
    /// ALGORITHM: Check the sign bit (most significant bit of highest word)
    /// ARCHITECTURAL_PURPOSE: Supports extended decimal precision overflow detection
    #[inline(always)]
    pub fn is_negative(self) -> bool {
        // Check the sign bit (MSB of the highest word)
        (self.words[3] & 0x8000_0000_0000_0000) != 0
    }
    
    /// Convert to bytes (little-endian, 32 bytes).
    pub fn to_bytes_le(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        for word in self.words.iter() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// Create from bytes (little-endian, must be exactly 32 bytes).
    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 32, "I256 requires exactly 32 bytes");
        let mut words = [0u64; 4];
        for i in 0..4 {
            let start = i * 8;
            let word_bytes: [u8; 8] = bytes[start..start + 8].try_into().unwrap();
            words[i] = u64::from_le_bytes(word_bytes);
        }
        I256 { words }
    }

    #[inline(always)]
    pub const fn from_u128(value: u128) -> Self {
        I256 { 
            words: [
                value as u64,
                (value >> 64) as u64,
                0,
                0,
            ]
        }
    }
    
    
    /// Check if value fits in i128
    #[inline(always)]
    pub fn fits_in_i128(self) -> bool {
        let is_negative = (self.words[1] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        self.words[2] == expected_high && self.words[3] == expected_high
    }

    /// Convenience constructor from u8
    #[inline(always)]
    pub const fn from_u8(value: u8) -> Self {
        I256::from_i128(value as i128)
    }

    /// Saturating addition (returns self on overflow instead of wrapping)
    #[inline(always)]
    pub fn saturating_add(self, rhs: Self) -> Self {
        let result = self + rhs;

        // Check for overflow
        let self_negative = self.is_negative();
        let rhs_negative = rhs.is_negative();
        let result_negative = result.is_negative();

        // Overflow occurs when adding two numbers of the same sign produces opposite sign
        if self_negative == rhs_negative && result_negative != self_negative {
            // Return max or min based on sign
            if self_negative {
                // Negative overflow -> return min (most negative)
                I256::from_words([0, 0, 0, 0x8000_0000_0000_0000])
            } else {
                // Positive overflow -> return max (most positive)
                I256::from_words([u64::MAX, u64::MAX, u64::MAX, 0x7FFF_FFFF_FFFF_FFFF])
            }
        } else {
            result
        }
    }

    /// Saturating subtraction (returns self on overflow instead of wrapping)
    #[inline(always)]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        let result = self - rhs;

        // Check for overflow
        let self_negative = self.is_negative();
        let rhs_negative = rhs.is_negative();
        let result_negative = result.is_negative();

        // Overflow occurs when subtracting numbers of opposite signs produces wrong sign
        if self_negative != rhs_negative && result_negative != self_negative {
            // Return max or min based on sign
            if self_negative {
                // Negative - Positive overflowed -> return min (most negative)
                I256::from_words([0, 0, 0, 0x8000_0000_0000_0000])
            } else {
                // Positive - Negative overflowed -> return max (most positive)
                I256::from_words([u64::MAX, u64::MAX, u64::MAX, 0x7FFF_FFFF_FFFF_FFFF])
            }
        } else {
            result
        }
    }

    /// Saturating negation - returns negated value or saturates at bounds on overflow
    pub fn saturating_neg(self) -> Self {
        // Only I256::MIN would overflow when negated
        // I256::MIN = 0x8000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000
        // -I256::MIN would be I256::MAX + 1, which doesn't exist
        if self == I256::from_words([0, 0, 0, 0x8000_0000_0000_0000]) {
            // Return I256::MAX instead
            I256::from_words([u64::MAX, u64::MAX, u64::MAX, 0x7FFF_FFFF_FFFF_FFFF])
        } else {
            -self
        }
    }

    /// Return signed infinity representation (used for overflow/error cases)
    /// Positive infinity if positive is true, negative infinity otherwise
    #[inline(always)]
    pub fn signed_infinity(positive: bool) -> Self {
        if positive {
            // Positive infinity: maximum representable value
            I256::from_words([u64::MAX, u64::MAX, u64::MAX, 0x7FFF_FFFF_FFFF_FFFF])
        } else {
            // Negative infinity: minimum representable value
            I256::from_words([0, 0, 0, 0x8000_0000_0000_0000])
        }
    }

    /// Multiply two I256 values (for extended precision arithmetic)
    #[inline(always)]
    pub fn multiply_i256(&self, other: &I256) -> crate::fixed_point::I512 {
        // Convert to I512 and multiply
        let a_i512 = crate::fixed_point::I512::from_i256(*self);
        let b_i512 = crate::fixed_point::I512::from_i256(*other);
        a_i512 * b_i512
    }
    
    /// Multiply two I256 values returning I512 result (full precision)
    #[inline(always)]
    pub fn mul_to_i512(self, rhs: I256) -> crate::fixed_point::I512 {
        mul_i256_to_i512(self, rhs)
    }

    /// Checked addition - returns None on overflow
    #[inline(always)]
    pub fn checked_add(self, rhs: I256) -> Option<I256> {
        let result = self + rhs;

        // Check for overflow: result sign different from both operands means overflow
        let self_negative = (self.words[3] as i64) < 0;
        let rhs_negative = (rhs.words[3] as i64) < 0;
        let result_negative = (result.words[3] as i64) < 0;

        // Overflow if both positive and result negative, or both negative and result positive
        if (self_negative == rhs_negative) && (self_negative != result_negative) {
            None
        } else {
            Some(result)
        }
    }

    /// Checked subtraction - returns None on overflow
    #[inline(always)]
    pub fn checked_sub(self, rhs: I256) -> Option<I256> {
        let result = self - rhs;

        // Check for overflow: subtracting opposite signs can overflow
        let self_negative = (self.words[3] as i64) < 0;
        let rhs_negative = (rhs.words[3] as i64) < 0;
        let result_negative = (result.words[3] as i64) < 0;

        // Overflow if signs differ and result sign wrong
        if (self_negative != rhs_negative) && (result_negative != self_negative) {
            None
        } else {
            Some(result)
        }
    }

    /// Checked negation - returns None on overflow (only for MIN value)
    #[inline(always)]
    pub fn checked_neg(self) -> Option<I256> {
        // Only I256::MIN would overflow when negated
        if self == I256::from_words([0, 0, 0, 0x8000_0000_0000_0000]) {
            None
        } else {
            Some(-self)
        }
    }

    /// Checked multiplication - returns None on overflow
    #[inline(always)]
    pub fn checked_mul(self, rhs: I256) -> Option<I256> {
        // Multiply to I512 and check if result fits in I256
        let result_i512 = self.mul_to_i512(rhs);

        // Check if result fits in I256
        if result_i512.fits_in_i256() {
            Some(result_i512.as_i256())
        } else {
            None
        }
    }

    /// Method to extract self as I256 (identity function for compatibility)
    #[inline(always)]
    pub fn as_i256(self) -> I256 {
        self
    }
}

// Comparison operations for I256
impl PartialEq for I256 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words
    }
}

impl Eq for I256 {}

impl PartialOrd for I256 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for I256 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare as signed 256-bit integers
        let self_negative = (self.words[3] as i64) < 0;
        let other_negative = (other.words[3] as i64) < 0;
        
        match (self_negative, other_negative) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                // Same sign: two's complement word comparison is correct as-is.
                // For negatives, smaller raw unsigned value = more negative = Less.
                // No reversal needed.
                for i in (0..4).rev() {
                    match self.words[i].cmp(&other.words[i]) {
                        std::cmp::Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                std::cmp::Ordering::Equal
            }
        }
    }
}

// Arithmetic operations for I256
impl std::ops::Add for I256 {
    type Output = Self;
    
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        
        for i in 0..4 {
            let sum = (self.words[i] as u128) + (rhs.words[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        
        I256 { words: result }
    }
}

impl std::ops::Sub for I256 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut result = [0u64; 4];
        let mut borrow = 0u64;

        for i in 0..4 {
            let a = self.words[i] as u128;
            let b = rhs.words[i] as u128 + borrow as u128;
            if a >= b {
                result[i] = (a - b) as u64;
                borrow = 0;
            } else {
                // Need to borrow from next word: add 2^64 to a
                result[i] = ((1u128 << 64) + a - b) as u64;
                borrow = 1;
            }
        }

        I256 { words: result }
    }
}

impl std::ops::Sub<&I256> for I256 {
    type Output = Self;
    
    #[inline(always)]
    fn sub(self, rhs: &Self) -> Self {
        self - *rhs
    }
}

impl std::ops::Shl<usize> for I256 {
    type Output = Self;
    
    #[inline(always)]
    fn shl(self, shift: usize) -> Self {
        if shift == 0 { return self; }
        if shift >= 256 { return I256::zero(); }
        
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut result = [0u64; 4];
        
        if bit_shift == 0 {
            // Simple word shift
            for i in word_shift..4 {
                result[i] = self.words[i - word_shift];
            }
        } else {
            // Bit shift within words
            for i in word_shift..4 {
                let low = if i > word_shift { self.words[i - word_shift - 1] >> (64 - bit_shift) } else { 0 };
                let high = self.words[i - word_shift] << bit_shift;
                result[i] = high | low;
            }
        }
        
        I256 { words: result }
    }
}

impl std::ops::BitOr for I256 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        I256 {
            words: [
                self.words[0] | rhs.words[0],
                self.words[1] | rhs.words[1],
                self.words[2] | rhs.words[2],
                self.words[3] | rhs.words[3],
            ]
        }
    }
}

impl std::ops::BitAnd for I256 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        I256 {
            words: [
                self.words[0] & rhs.words[0],
                self.words[1] & rhs.words[1],
                self.words[2] & rhs.words[2],
                self.words[3] & rhs.words[3],
            ]
        }
    }
}

impl Shr<u32> for I256 {
    type Output = Self;
    
    #[inline(always)]
    fn shr(self, shift: u32) -> Self {
        if shift == 0 {
            return self;
        }
        
        if shift >= 256 {
            // Sign extend based on sign bit
            let sign = (self.words[3] as i64) < 0;
            return I256 {
                words: if sign { [u64::MAX; 4] } else { [0; 4] }
            };
        }
        
        let word_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;
        
        let mut result = [0u64; 4];
        
        if bit_shift == 0 {
            // Simple word shift
            for i in 0..(4 - word_shift) {
                result[i] = self.words[i + word_shift];
            }
            // Sign extend
            let sign = (self.words[3] as i64) < 0;
            for i in (4 - word_shift)..4 {
                result[i] = if sign { u64::MAX } else { 0 };
            }
        } else {
            // Bit shift within words
            for i in 0..(4 - word_shift) {
                let low = self.words[i + word_shift] >> bit_shift;
                let high = if i + word_shift + 1 < 4 {
                    self.words[i + word_shift + 1] << (64 - bit_shift)
                } else {
                    // Sign extend
                    if (self.words[3] as i64) < 0 {
                        u64::MAX << (64 - bit_shift)
                    } else {
                        0
                    }
                };
                result[i] = low | high;
            }
            // Sign extend remaining words
            let sign = (self.words[3] as i64) < 0;
            for i in (4 - word_shift)..4 {
                result[i] = if sign { u64::MAX } else { 0 };
            }
        }
        
        I256 { words: result }
    }
}

impl Mul for I256 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Proper I256×I256→I512 multiplication, then truncate to I256
        let result_i512 = self.mul_to_i512(rhs);
        result_i512.as_i256()
    }
}

impl std::ops::Div for I256 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        divmod_i256_by_i256(self, rhs).0
    }
}

impl std::ops::Rem for I256 {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        divmod_i256_by_i256(self, rhs).1
    }
}

// Reference implementations for binary domain operations
impl std::ops::Div<&I256> for &I256 {
    type Output = I256;

    #[inline(always)]
    fn div(self, rhs: &I256) -> I256 {
        divmod_i256_by_i256(*self, *rhs).0
    }
}

impl std::ops::Mul<&I256> for &I256 {
    type Output = I256;

    #[inline(always)]
    fn mul(self, rhs: &I256) -> I256 {
        *self * *rhs
    }
}

impl From<u128> for I256 {
    #[inline(always)]
    fn from(value: u128) -> Self {
        I256::from_u128(value)
    }
}

impl From<i128> for I256 {
    #[inline(always)]
    fn from(value: i128) -> Self {
        I256::from_i128(value)
    }
}

impl From<i64> for I256 {
    #[inline(always)]
    fn from(value: i64) -> Self {
        I256::from_i128(value as i128)
    }
}

/// Full 128×128→256 bit multiplication
/// 
/// CRITICAL: This is the core multiplication algorithm used throughout the library
/// Produces exact 256-bit results from 128-bit inputs for maximum precision
#[inline(always)]
pub fn mul_i128_to_i256(a: i128, b: i128) -> I256 {
    // Handle signs
    let a_neg = a < 0;
    let b_neg = b < 0;
    let result_neg = a_neg != b_neg;
    
    let a_abs = a.unsigned_abs();
    let b_abs = b.unsigned_abs();
    
    // Decompose into 64-bit parts
    let a_lo = a_abs as u64;
    let a_hi = (a_abs >> 64) as u64;
    let b_lo = b_abs as u64;
    let b_hi = (b_abs >> 64) as u64;
    
    // 4 partial products (each u128)
    let p00 = (a_lo as u128) * (b_lo as u128);
    let p01 = (a_lo as u128) * (b_hi as u128);
    let p10 = (a_hi as u128) * (b_lo as u128);
    let p11 = (a_hi as u128) * (b_hi as u128);

    // Accumulate with u128 accumulators — carry flows naturally via >> 64
    let w0 = p00 as u64;

    let acc1 = (p00 >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
    let w1 = acc1 as u64;

    let acc2 = (acc1 >> 64) + (p01 >> 64) + (p10 >> 64) + (p11 as u64 as u128);
    let w2 = acc2 as u64;

    let w3 = ((acc2 >> 64) + (p11 >> 64)) as u64;

    let mut result = I256 { words: [w0, w1, w2, w3] };
    
    // Handle sign
    if result_neg {
        // Two's complement negation
        let mut borrow = 1u64;
        for i in 0..4 {
            let (val, b) = (!result.words[i]).overflowing_add(borrow);
            result.words[i] = val;
            borrow = b as u64;
        }
    }
    
    result
}

/// Platform-specific optimizations for multiplication
/// 
/// BMI2 ENHANCED: Uses hardware-accelerated 64×64→128 multiplication when available
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
pub fn mul_i128_to_i256_bmi2(a: i128, b: i128) -> I256 {
    use core::arch::x86_64::*;
    
    unsafe {
        let a_neg = a < 0;
        let b_neg = b < 0;
        let result_neg = a_neg != b_neg;
        
        let a_abs = a.unsigned_abs();
        let b_abs = b.unsigned_abs();
        
        let a_lo = a_abs as u64;
        let a_hi = (a_abs >> 64) as u64;
        let b_lo = b_abs as u64;
        let b_hi = (b_abs >> 64) as u64;
        
        // Use BMI2 instructions for faster multiplication
        let mut lo_lo_hi = 0u64;
        let lo_lo_lo = _mulx_u64(a_lo, b_lo, &mut lo_lo_hi);
        
        let mut lo_hi_hi = 0u64;
        let lo_hi_lo = _mulx_u64(a_lo, b_hi, &mut lo_hi_hi);
        
        let mut hi_lo_hi = 0u64;
        let hi_lo_lo = _mulx_u64(a_hi, b_lo, &mut hi_lo_hi);
        
        let mut hi_hi_hi = 0u64;
        let hi_hi_lo = _mulx_u64(a_hi, b_hi, &mut hi_hi_hi);
        
        // Accumulate with carry chain
        let mut result = I256::zero();
        result.words[0] = lo_lo_lo;
        
        let (sum1, carry1) = lo_lo_hi.overflowing_add(lo_hi_lo);
        let (sum2, carry2) = sum1.overflowing_add(hi_lo_lo);
        result.words[1] = sum2;
        
        let (sum3, carry3) = lo_hi_hi.overflowing_add(hi_lo_hi);
        let (sum4, carry4) = sum3.overflowing_add(hi_hi_lo);
        let (sum5, carry5) = sum4.overflowing_add(carry1 as u64);
        let (sum6, carry6) = sum5.overflowing_add(carry2 as u64);
        result.words[2] = sum6;
        
        result.words[3] = hi_hi_hi + (carry3 as u64) + (carry4 as u64) + (carry5 as u64) + (carry6 as u64);
        
        // Handle sign
        if result_neg {
            let mut borrow = 1u64;
            for i in 0..4 {
                let (val, b) = (!result.words[i]).overflowing_add(borrow);
                result.words[i] = val;
                borrow = b as u64;
            }
        }
        
        result
    }
}

/// Full I256×I256→I512 multiplication for maximum precision
#[inline(always)]
pub fn mul_i256_to_i512(a: I256, b: I256) -> crate::fixed_point::I512 {
    // Implement proper 256×256→512 bit multiplication
    // Algorithm: (a_hi*2^128 + a_lo) * (b_hi*2^128 + b_lo)
    //          = a_hi*b_hi*2^256 + (a_hi*b_lo + a_lo*b_hi)*2^128 + a_lo*b_lo
    use crate::fixed_point::I512;

    // Decompose each I256 into two UNSIGNED 128-bit parts to avoid sign extension bugs
    let a_lo = ((a.words[1] as u128) << 64) | (a.words[0] as u128);
    let a_hi = ((a.words[3] as u128) << 64) | (a.words[2] as u128);
    let b_lo = ((b.words[1] as u128) << 64) | (b.words[0] as u128);
    let b_hi = ((b.words[3] as u128) << 64) | (b.words[2] as u128);

    // Compute the four partial products using UNSIGNED multiplication
    let lo_lo = mul_u128_to_u256(a_lo, b_lo);     // a_lo * b_lo
    let lo_hi = mul_u128_to_u256(a_lo, b_hi);     // a_lo * b_hi
    let hi_lo = mul_u128_to_u256(a_hi, b_lo);     // a_hi * b_lo
    let hi_hi = mul_u128_to_u256(a_hi, b_hi);     // a_hi * b_hi

    // Accumulate into 512-bit result with proper carry propagation
    let mut result = I512::zero();

    // Step 1: Add lo_lo at position 0 (bits 0-255)
    result.words[0] = lo_lo[0];
    result.words[1] = lo_lo[1];
    result.words[2] = lo_lo[2];
    result.words[3] = lo_lo[3];

    // Step 2: Add lo_hi at position 2 (bits 128-383)
    let mut carry = 0u64;
    for i in 0..4 {
        let word_idx = i + 2;
        if word_idx < 8 {
            let sum = (result.words[word_idx] as u128) + (lo_hi[i] as u128) + (carry as u128);
            result.words[word_idx] = sum as u64;
            carry = (sum >> 64) as u64;
        }
    }
    // Propagate final carry from lo_hi addition
    if carry > 0 {
        let word_idx = 6;
        if word_idx < 8 {
            let sum = (result.words[word_idx] as u128) + (carry as u128);
            result.words[word_idx] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        if carry > 0 && word_idx + 1 < 8 {
            result.words[word_idx + 1] += carry;
        }
    }

    // Step 3: Add hi_lo at position 2 (bits 128-383)
    carry = 0;
    for i in 0..4 {
        let word_idx = i + 2;
        if word_idx < 8 {
            let sum = (result.words[word_idx] as u128) + (hi_lo[i] as u128) + (carry as u128);
            result.words[word_idx] = sum as u64;
            carry = (sum >> 64) as u64;
        }
    }
    // Propagate final carry from hi_lo addition
    if carry > 0 {
        let word_idx = 6;
        if word_idx < 8 {
            let sum = (result.words[word_idx] as u128) + (carry as u128);
            result.words[word_idx] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        if carry > 0 && word_idx + 1 < 8 {
            result.words[word_idx + 1] += carry;
        }
    }

    // Step 4: Add hi_hi at position 4 (bits 256-511)
    carry = 0;
    for i in 0..4 {
        let word_idx = i + 4;
        if word_idx < 8 {
            let sum = (result.words[word_idx] as u128) + (hi_hi[i] as u128) + (carry as u128);
            result.words[word_idx] = sum as u64;
            carry = (sum >> 64) as u64;
        }
    }

    result
}

/// Multiply two u128 values to get u256 result (as 4×u64 array)
#[inline]
fn mul_u128_to_u256(a: u128, b: u128) -> [u64; 4] {
    // Decompose into 64-bit parts
    let a_lo = a as u64;
    let a_hi = (a >> 64) as u64;
    let b_lo = b as u64;
    let b_hi = (b >> 64) as u64;

    // Compute partial products (each up to 128 bits)
    let p00 = (a_lo as u128) * (b_lo as u128);
    let p01 = (a_lo as u128) * (b_hi as u128);
    let p10 = (a_hi as u128) * (b_lo as u128);
    let p11 = (a_hi as u128) * (b_hi as u128);

    // Accumulate using u128 accumulator to avoid carry overflow
    let mut result = [0u64; 4];

    // Position 0: p00 lower 64 bits
    result[0] = p00 as u64;
    let mut acc = (p00 >> 64) as u128;

    // Position 1: p01_lo + p10_lo + carry from position 0
    acc += (p01 as u64) as u128 + (p10 as u64) as u128;
    result[1] = acc as u64;
    acc >>= 64;

    // Position 2: p01_hi + p10_hi + p11_lo + carry from position 1
    acc += ((p01 >> 64) as u64) as u128 + ((p10 >> 64) as u64) as u128 + (p11 as u64) as u128;
    result[2] = acc as u64;
    acc >>= 64;

    // Position 3: p11_hi + carry from position 2
    acc += ((p11 >> 64) as u64) as u128;
    result[3] = acc as u64;

    result
}

/// Division with remainder for I256 by I256 (binary domain)
///
/// ALGORITHM: Full 256-bit by 256-bit long division with exact remainder calculation
/// PRECISION: Binary domain - optimized for binary fractions and transcendental functions
/// DOMAIN: Pure binary domain - never mixed with decimal
pub fn divmod_i256_by_i256(dividend: I256, divisor: I256) -> (I256, I256) {
    // Handle division by zero with saturation
    if divisor.is_zero() {
        let saturated_quotient = if dividend.is_negative() {
            I256 { words: [0, 0, 0, 0x8000_0000_0000_0000] } // i256::MIN
        } else {
            I256 { words: [u64::MAX, u64::MAX, u64::MAX, 0x7FFF_FFFF_FFFF_FFFF] } // i256::MAX
        };
        return (saturated_quotient, I256::zero());
    }

    // Optimize for cases where both fit in i128
    if dividend.fits_in_i128() && divisor.fits_in_i128() {
        let dividend_i128 = dividend.as_i128();
        let divisor_i128 = divisor.as_i128();
        let quotient = dividend_i128 / divisor_i128;
        let remainder = dividend_i128 % divisor_i128;
        return (I256::from_i128(quotient), I256::from_i128(remainder));
    }

    // Determine signs for proper signed division
    let dividend_negative = dividend.is_negative();
    let divisor_negative = divisor.is_negative();
    let quotient_negative = dividend_negative != divisor_negative;

    // Work with absolute values
    let abs_dividend = if dividend_negative {
        negate_i256(dividend)
    } else {
        dividend
    };

    let abs_divisor = if divisor_negative {
        negate_i256(divisor)
    } else {
        divisor
    };

    // PRODUCTION: 256-bit by 256-bit long division algorithm (binary domain)
    let mut quotient_words = [0u64; 4];
    let mut remainder = I256::zero();

    // Process each bit from most significant to least significant
    for word_idx in (0..4).rev() {
        for bit_idx in (0..64).rev() {
            // Shift remainder left by 1
            remainder = shift_left_i256_by_1(remainder);

            // Set the least significant bit to the current dividend bit
            let dividend_bit = (abs_dividend.words[word_idx] >> bit_idx) & 1;
            remainder.words[0] |= dividend_bit;

            // Try to subtract divisor from remainder
            if compare_i256_unsigned(remainder, abs_divisor) >= 0 {
                remainder = subtract_i256_unsigned(remainder, abs_divisor);
                // Set the corresponding quotient bit
                quotient_words[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    let mut quotient = I256 { words: quotient_words };

    // Apply signs
    if quotient_negative && !quotient.is_zero() {
        quotient = negate_i256(quotient);
    }

    if dividend_negative && !remainder.is_zero() {
        remainder = negate_i256(remainder);
    }

    (quotient, remainder)
}

/// Helper: Negate I256 value (two's complement) - binary domain
#[inline(always)]
fn negate_i256(value: I256) -> I256 {
    let mut result = [0u64; 4];
    let mut carry = 1u64;

    for i in 0..4 {
        let inverted = !value.words[i];
        let sum = (inverted as u128) + (carry as u128);
        result[i] = sum as u64;
        carry = (sum >> 64) as u64;
    }

    I256 { words: result }
}

/// Helper: Shift I256 left by 1 bit (for long division algorithm)
#[inline(always)]
fn shift_left_i256_by_1(value: I256) -> I256 {
    let mut result = [0u64; 4];
    let mut carry = 0u64;

    for i in 0..4 {
        let word = value.words[i];
        result[i] = (word << 1) | carry;
        carry = word >> 63;
    }

    I256 { words: result }
}

/// Helper: Compare two I256 values as unsigned (for long division)
#[inline(always)]
fn compare_i256_unsigned(a: I256, b: I256) -> i8 {
    for i in (0..4).rev() {
        if a.words[i] > b.words[i] {
            return 1;
        } else if a.words[i] < b.words[i] {
            return -1;
        }
    }
    0 // Equal
}

/// Helper: Subtract b from a (unsigned, for long division)
#[inline(always)]
fn subtract_i256_unsigned(a: I256, b: I256) -> I256 {
    let mut result = [0u64; 4];
    let mut borrow = 0i128;

    for i in 0..4 {
        let diff = (a.words[i] as i128) - (b.words[i] as i128) - borrow;
        if diff < 0 {
            result[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            result[i] = diff as u64;
            borrow = 0;
        }
    }

    I256 { words: result }
}

// Type alias for convenience
#[allow(non_camel_case_types)]
pub type i256 = I256;

/// Display trait implementation for I256
impl std::fmt::Display for I256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Decimal representation via i128 truncation
        // Values exceeding i128 range are displayed truncated
        let as_i128 = self.as_i128();
        write!(f, "{}", as_i128)
    }
}

/// Negation trait implementation for I256
impl Neg for I256 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        // Two's complement negation: invert all bits and add 1
        let mut result = [0u64; 4];
        let mut carry = 1u64;

        for i in 0..4 {
            let sum = (!self.words[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }

        I256 { words: result }
    }
}

/// Tests for I256 arithmetic
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_i256_multiplication() {
        let a = 0x123456789ABCDEF0123456789ABCDEF_i128;
        let b = 0xFEDCBA9876543210FEDCBA9876543210_u128 as i128;
        
        let result = mul_i128_to_i256(a, b);
        
        // Verify against known result
        assert_eq!(result.words[0], 0x2236d88fe5618cf0);
        assert_eq!(result.words[1], 0x458fab20783af122);
        assert_eq!(result.words[2], 0x23578729b6a56d85);
        assert_eq!(result.words[3], 0xfffeb49923cc0953);
    }
    
    #[test]
    fn test_i256_from_conversions() {
        let value = 0x123456789ABCDEF0_i128;
        let i256_val = I256::from_i128(value);
        assert_eq!(i256_val.as_i128(), value);
    }
    
    #[test]
    fn test_i256_shift_operations() {
        let value = I256::from_i128(0xFFFFFFFF00000000_i128);
        let shifted = value >> 32;
        assert_eq!(shifted.as_i128(), 0xFFFFFFFF_i128);
    }
}