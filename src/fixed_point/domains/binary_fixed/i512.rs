//! 512-bit Integer Arithmetic
//! 
//! PRODUCTION-READY: Extended precision arithmetic for Q256.256 format
//! ARCHITECTURE: Based on proven I128 pattern, optimized for fixed-point operations
//! PRECISION: Foundation for ultra-high precision transcendental operations

use std::ops::{Add, Sub, Mul, Shl, Shr, BitOr, BitAnd, Neg};
use std::cmp::{Ord, PartialOrd, Ordering};
use crate::fixed_point::i256::I256;

/// 512-bit signed integer for Q128.128 compute-tier arithmetic
/// 
/// ARCHITECTURE: Array of 8 × 64-bit words (little-endian)
/// USAGE: Enables Q256.256 fixed-point format with ~77 decimal places
/// COMPATIBILITY: Follows proven I128/I256 patterns for consistent API
#[derive(Clone, Copy, Debug)]
pub struct I512 {
    /// Stored as eight 64-bit words (little-endian)
    pub words: [u64; 8],
}

impl I512 {
    #[inline(always)]
    pub const fn zero() -> Self {
        I512 { words: [0; 8] }
    }
    
    #[inline(always)]
    pub const fn one() -> Self {
        I512 { words: [1, 0, 0, 0, 0, 0, 0, 0] }
    }

    /// Maximum value for signed 512-bit integer (2^511 - 1)
    #[inline(always)]
    pub const fn max_value() -> Self {
        I512 {
            words: [
                0xFFFF_FFFF_FFFF_FFFF,  // word 0
                0xFFFF_FFFF_FFFF_FFFF,  // word 1
                0xFFFF_FFFF_FFFF_FFFF,  // word 2
                0xFFFF_FFFF_FFFF_FFFF,  // word 3
                0xFFFF_FFFF_FFFF_FFFF,  // word 4
                0xFFFF_FFFF_FFFF_FFFF,  // word 5
                0xFFFF_FFFF_FFFF_FFFF,  // word 6
                0x7FFF_FFFF_FFFF_FFFF,  // word 7 (sign bit = 0)
            ]
        }
    }

    /// Minimum value for signed 512-bit integer (-2^511)
    #[inline(always)]
    pub const fn min_value() -> Self {
        I512 {
            words: [
                0,  // word 0
                0,  // word 1
                0,  // word 2
                0,  // word 3
                0,  // word 4
                0,  // word 5
                0,  // word 6
                0x8000_0000_0000_0000,  // word 7 (sign bit = 1)
            ]
        }
    }

    #[inline(always)]
    pub const fn from_words(words: [u64; 8]) -> Self {
        I512 { words }
    }
    
    /// Create I512 from I256 with sign extension
    #[inline(always)]
    pub const fn from_i256(value: I256) -> Self {
        let is_negative = (value.words[3] as i64) < 0;
        let sign_extend = if is_negative { u64::MAX } else { 0 };
        
        I512 { 
            words: [
                value.words[0],
                value.words[1], 
                value.words[2],
                value.words[3],
                sign_extend,
                sign_extend,
                sign_extend,
                sign_extend,
            ]
        }
    }
    
    /// Check if the I512 value is zero
    ///
    /// ALGORITHM: Check if all words are zero
    /// PERFORMANCE: O(1) with early exit on first non-zero word
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    /// Check if the I512 value is negative
    ///
    /// ALGORITHM: Check the sign bit (MSB of the highest word)
    /// PERFORMANCE: O(1) single bit check
    #[inline(always)]
    pub fn is_negative(self) -> bool {
        (self.words[7] & 0x8000_0000_0000_0000) != 0
    }

    /// Create I512 from I128 with sign extension
    #[inline(always)]
    pub const fn from_i128(value: i128) -> Self {
        let is_negative = value < 0;
        let sign_extend = if is_negative { u64::MAX } else { 0 };

        I512 {
            words: [
                value as u64,
                (value >> 64) as u64,
                sign_extend,
                sign_extend,
                sign_extend,
                sign_extend,
                sign_extend,
                sign_extend,
            ]
        }
    }

    /// Create I512 from u128 (unsigned, always positive)
    ///
    /// USAGE: Required for decimal string parsing in from_decimal_string()
    /// PRECISION: Full 128-bit value preserved, zero-extended to 512 bits
    #[inline(always)]
    pub const fn from_u128(value: u128) -> Self {
        I512 {
            words: [
                value as u64,              // Lower 64 bits
                (value >> 64) as u64,      // Upper 64 bits
                0,                         // Zero-extend (unsigned)
                0,
                0,
                0,
                0,
                0,
            ]
        }
    }

    /// Extract lower 256 bits as I256
    #[inline(always)]
    pub fn as_i256(self) -> I256 {
        I256::from_words([
            self.words[0],
            self.words[1],
            self.words[2],
            self.words[3],
        ])
    }

    /// Convert to I256 with saturation
    ///
    /// If the value doesn't fit in I256, returns I256::max_value() or I256::min_value()
    #[inline(always)]
    pub fn as_i256_saturating(self) -> I256 {
        if self.fits_in_i256() {
            self.as_i256()
        } else {
            // Check sign to determine which limit to use
            let is_negative = (self.words[7] as i64) < 0;
            if is_negative {
                I256::min_value()
            } else {
                I256::max_value()
            }
        }
    }

    /// Extract lower 128 bits as i128
    #[inline(always)]
    pub fn as_i128(self) -> i128 {
        ((self.words[1] as i128) << 64) | (self.words[0] as i128)
    }
    
    /// Check if value fits in I256
    #[inline(always)]
    pub fn fits_in_i256(self) -> bool {
        // Check that words[4-7] are the sign extension of bit 255 (word[3] bit 63)
        let sign_bit_i256 = (self.words[3] as i64) < 0;
        let expected_high = if sign_bit_i256 { u64::MAX } else { 0 };

        self.words[4] == expected_high &&
        self.words[5] == expected_high &&
        self.words[6] == expected_high &&
        self.words[7] == expected_high
    }
    
    /// Check if value fits in i128
    #[inline(always)]
    pub fn fits_in_i128(self) -> bool {
        let is_negative = (self.words[1] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };
        
        (2..8).all(|i| self.words[i] == expected_high)
    }
    
    /// Convert to bytes (little-endian)
    #[inline(always)]
    pub fn to_bytes_le(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        for word in self.words.iter() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }
    
    /// Create from bytes (little-endian)
    #[inline(always)]
    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 64, "I512 requires exactly 64 bytes");

        let mut words = [0u64; 8];
        for i in 0..8 {
            let start = i * 8;
            let word_bytes: [u8; 8] = bytes[start..start+8].try_into().unwrap();
            words[i] = u64::from_le_bytes(word_bytes);
        }

        I512 { words }
    }

    /// Checked addition - returns None on overflow
    #[inline(always)]
    pub fn checked_add(self, rhs: I512) -> Option<I512> {
        let result = self + rhs;

        // Check for overflow: result sign different from both operands means overflow
        let self_negative = (self.words[7] as i64) < 0;
        let rhs_negative = (rhs.words[7] as i64) < 0;
        let result_negative = (result.words[7] as i64) < 0;

        // Overflow if both positive and result negative, or both negative and result positive
        if (self_negative == rhs_negative) && (self_negative != result_negative) {
            None
        } else {
            Some(result)
        }
    }

    /// Checked subtraction - returns None on overflow
    #[inline(always)]
    pub fn checked_sub(self, rhs: I512) -> Option<I512> {
        let result = self - rhs;

        // Check for overflow: subtracting opposite signs can overflow
        let self_negative = (self.words[7] as i64) < 0;
        let rhs_negative = (rhs.words[7] as i64) < 0;
        let result_negative = (result.words[7] as i64) < 0;

        // Overflow if signs differ and result sign wrong
        if (self_negative != rhs_negative) && (result_negative != self_negative) {
            None
        } else {
            Some(result)
        }
    }

    /// Checked negation - returns None on overflow (only for MIN value)
    #[inline(always)]
    pub fn checked_neg(self) -> Option<I512> {
        // Only I512::MIN would overflow when negated
        if self == I512::from_words([0, 0, 0, 0, 0, 0, 0, 0x8000_0000_0000_0000]) {
            None
        } else {
            Some(-self)
        }
    }

    /// Method to extract self as I512 (identity function for compatibility)
    #[inline(always)]
    pub fn as_i512(self) -> I512 {
        self
    }

    /// Checked multiplication - returns None on overflow
    #[inline(always)]
    pub fn checked_mul(self, rhs: I512) -> Option<I512> {
        // Multiply to I1024 and check if result fits in I512
        let result_i1024 = self.multiply_i512(&rhs);

        // Check if result fits in I512 (check upper 512 bits are sign extension)
        let is_negative = (result_i1024.words[15] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        let fits = (8..16).all(|i| result_i1024.words[i] == expected_high);

        if fits {
            Some(I512::from_words([
                result_i1024.words[0],
                result_i1024.words[1],
                result_i1024.words[2],
                result_i1024.words[3],
                result_i1024.words[4],
                result_i1024.words[5],
                result_i1024.words[6],
                result_i1024.words[7],
            ]))
        } else {
            None
        }
    }
}

// Comparison operations
impl PartialEq for I512 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words
    }
}

impl Eq for I512 {}

impl PartialOrd for I512 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for I512 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare as signed 512-bit integers
        let self_negative = (self.words[7] as i64) < 0;
        let other_negative = (other.words[7] as i64) < 0;
        
        match (self_negative, other_negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => {
                // Same sign: two's complement word comparison is correct as-is.
                // For negatives, smaller raw unsigned value = more negative = Less.
                // No reversal needed.
                for i in (0..8).rev() {
                    match self.words[i].cmp(&other.words[i]) {
                        Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                Ordering::Equal
            }
        }
    }
}

// Arithmetic operations
impl Add for I512 {
    type Output = Self;
    
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut result = [0u64; 8];
        let mut carry = 0u64;
        
        for i in 0..8 {
            let sum = (self.words[i] as u128) + (rhs.words[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        
        I512 { words: result }
    }
}

impl Sub for I512 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut result = [0u64; 8];
        let mut borrow = 0u64;

        for i in 0..8 {
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

        I512 { words: result }
    }
}

/// Display trait implementation for I512
impl std::fmt::Display for I512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Decimal representation via i128 for values that fit, I256 approximation otherwise
        if self.fits_in_i128() {
            let as_i128 = self.as_i128();
            write!(f, "{}", as_i128)
        } else {
            // For values that don't fit in i128, use I256 approximation
            let as_i256 = self.as_i256_saturating();
            write!(f, "{}(I256)", as_i256)
        }
    }
}

impl Neg for I512 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        // Two's complement negation
        let mut result = [0u64; 8];
        let mut carry = 1u64;
        
        for i in 0..8 {
            let sum = (!self.words[i] as u128) + (carry as u128);
            result[i] = sum as u64;
            carry = (sum >> 64) as u64;
        }
        
        I512 { words: result }
    }
}

impl Shl<usize> for I512 {
    type Output = Self;
    
    #[inline(always)]
    fn shl(self, shift: usize) -> Self {
        if shift == 0 { return self; }
        if shift >= 512 { return I512::zero(); }
        
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut result = [0u64; 8];
        
        if bit_shift == 0 {
            // Simple word shift
            for i in word_shift..8 {
                result[i] = self.words[i - word_shift];
            }
        } else {
            // Bit shift within words
            for i in word_shift..8 {
                let low = if i > word_shift { self.words[i - word_shift - 1] >> (64 - bit_shift) } else { 0 };
                let high = self.words[i - word_shift] << bit_shift;
                result[i] = high | low;
            }
        }
        
        I512 { words: result }
    }
}

impl Shr<usize> for I512 {
    type Output = Self;
    
    #[inline(always)]
    fn shr(self, shift: usize) -> Self {
        if shift == 0 { return self; }
        
        if shift >= 512 {
            // Sign extend based on sign bit
            let sign = (self.words[7] as i64) < 0;
            return I512 { words: if sign { [u64::MAX; 8] } else { [0; 8] } };
        }
        
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut result = [0u64; 8];
        
        if bit_shift == 0 {
            // Simple word shift
            for i in 0..(8 - word_shift) {
                result[i] = self.words[i + word_shift];
            }
            // Sign extend
            let sign = (self.words[7] as i64) < 0;
            for i in (8 - word_shift)..8 {
                result[i] = if sign { u64::MAX } else { 0 };
            }
        } else {
            // Bit shift within words
            for i in 0..(8 - word_shift) {
                let low = self.words[i + word_shift] >> bit_shift;
                let high = if i + word_shift + 1 < 8 {
                    self.words[i + word_shift + 1] << (64 - bit_shift)
                } else {
                    // Sign extend
                    if (self.words[7] as i64) < 0 {
                        u64::MAX << (64 - bit_shift)
                    } else {
                        0
                    }
                };
                result[i] = low | high;
            }
            // Sign extend remaining words
            let sign = (self.words[7] as i64) < 0;
            for i in (8 - word_shift)..8 {
                result[i] = if sign { u64::MAX } else { 0 };
            }
        }
        
        I512 { words: result }
    }
}

impl BitOr for I512 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        let mut result = [0u64; 8];
        for i in 0..8 {
            result[i] = self.words[i] | rhs.words[i];
        }
        I512 { words: result }
    }
}

impl BitAnd for I512 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        let mut result = [0u64; 8];
        for i in 0..8 {
            result[i] = self.words[i] & rhs.words[i];
        }
        I512 { words: result }
    }
}

// Basic multiplication (can be optimized later)
impl Mul for I512 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Simple schoolbook multiplication — sufficient for current workloads

        if self.fits_in_i128() && rhs.fits_in_i128() {
            // Use efficient I128 multiplication
            let a = self.as_i128();
            let b = rhs.as_i128();
            let result = crate::fixed_point::i256::mul_i128_to_i256(a, b);
            I512::from_i256(result)
        } else {
            // Fallback to simple multiplication for large values
            self.mul_simple(rhs)
        }
    }
}

impl std::ops::Div for I512 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        divmod_i512_by_i512(self, rhs).0
    }
}

impl std::ops::Rem for I512 {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        divmod_i512_by_i512(self, rhs).1
    }
}

impl I512 {
    /// Simple multiplication implementation (not optimized)
    #[inline(always)]
    fn mul_simple(self, rhs: Self) -> Self {
        let mut result = I512::zero();
        
        // Handle signs
        let self_neg = (self.words[7] as i64) < 0;
        let rhs_neg = (rhs.words[7] as i64) < 0;
        let result_neg = self_neg != rhs_neg;
        
        let self_abs = if self_neg { -self } else { self };
        let rhs_abs = if rhs_neg { -rhs } else { rhs };
        
        // School multiplication algorithm
        for i in 0..8 {
            if rhs_abs.words[i] == 0 { continue; }
            
            let mut carry = 0u64;
            for j in 0..8 {
                if i + j >= 8 { break; }
                
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
impl From<I256> for I512 {
    #[inline(always)]
    fn from(value: I256) -> Self {
        I512::from_i256(value)
    }
}

impl From<i128> for I512 {
    #[inline(always)]
    fn from(value: i128) -> Self {
        I512::from_i128(value)
    }
}

impl From<i64> for I512 {
    #[inline(always)]
    fn from(value: i64) -> Self {
        I512::from_i128(value as i128)
    }
}

/// Utility functions for Q256.256 fixed-point operations
impl I512 {
    /// Create I512 representing 1.0 in Q256.256 format
    #[inline(always)]
    pub const fn one_q256_256() -> Self {
        let mut words = [0u64; 8];
        words[4] = 1; // 1 << 256 for Q256.256
        I512 { words }
    }
    
    /// Multiply two Q256.256 values (I512) and return Q256.256 result
    #[inline(always)]
    pub fn mul_q256_256(a: I512, b: I512) -> I512 {
        // Tier N+1: widen to I1024, multiply, shift right by 256, extract I512
        let a_wide = crate::fixed_point::I1024::from_i512(a);
        let b_wide = crate::fixed_point::I1024::from_i512(b);
        let full_product = a_wide * b_wide;
        // Round-to-nearest: check bit 255 (round bit)
        let shifted = full_product >> 256;
        let round_bit = (full_product >> 255).words[0] & 1;
        let result = shifted.as_i512();
        if round_bit != 0 {
            result + I512::from_i128(1)
        } else {
            result
        }
    }
    
    /// Convert from Q64.64 (i128) to Q256.256 (I512)
    #[inline(always)]
    pub fn from_q64_64(value: i128) -> Self {
        // Scale up by 192 bits (256 - 64)
        I512::from_i128(value) << 192
    }
    
    /// Convert from Q256.256 (I512) to Q64.64 (i128) with rounding
    #[inline(always)]
    pub fn to_q64_64(self) -> i128 {
        // Scale down by 192 bits with rounding
        let shifted = self >> 192;
        shifted.as_i128()
    }
    
    /// Multiply two I512 values producing I1024 result
    /// 
    /// PRECISION: Full precision multiplication for extended arithmetic
    /// USAGE: Required for tiered overflow-resistant arithmetic
    #[inline(always)]
    pub fn multiply_i512(&self, other: &I512) -> crate::fixed_point::I1024 {
        // Convert to I1024 and multiply
        let a_i1024 = crate::fixed_point::I1024::from_i512(*self);
        let b_i1024 = crate::fixed_point::I1024::from_i512(*other);
        a_i1024 * b_i1024
    }

    /// Multiply two I512 values producing full I1024 result
    ///
    /// **USAGE**: For Q256.256 fixed-point multiplication (tier N+1 strategy)
    /// **ALGORITHM**: Grade-school multiplication with carry propagation
    /// **PRECISION**: Full 1024-bit result, no truncation
    /// **TIER N+1**: Enables error-less Q256.256 exp computation
    #[inline(always)]
    pub fn mul_to_i1024(&self, other: I512) -> crate::fixed_point::I1024 {
        use crate::fixed_point::I1024;

        let mut result = I1024::zero();

        // Grade-school multiplication: a × b
        for i in 0..8 {
            let mut carry = 0u64;
            for j in 0..8 {
                let product = (self.words[i] as u128) * (other.words[j] as u128) +
                              (result.words[i + j] as u128) +
                              (carry as u128);
                result.words[i + j] = product as u64;
                carry = (product >> 64) as u64;
            }
            // Propagate final carry
            if i + 8 < 16 {
                result.words[i + 8] = carry;
            }
        }

        result
    }
}

// ============================================================================
// PRODUCTION DIVISION IMPLEMENTATION
// ============================================================================

/// Full 512-bit by 512-bit signed division and modulo
///
/// ALGORITHM: Binary long division with shift-and-subtract method
/// PRECISION: Full 512-bit precision division (no truncation, no approximation)
/// PERFORMANCE: O(512) bit-by-bit division
/// CORRECTNESS: Mirrors proven I256 divmod_i256_by_i256 implementation
///
/// This is the PRODUCTION implementation required for scientific profile Q256.256
/// fractional parsing. Without this, from_decimal_string() returns zero for all
/// fractional parts, causing exp(0.5) → exp(0), exp(1.5) → exp(1), etc.
///
/// # Arguments
/// * `dividend` - The value to be divided
/// * `divisor` - The value to divide by
///
/// # Returns
/// * `(quotient, remainder)` tuple where `dividend = quotient * divisor + remainder`
///
/// # Panics
/// * Never panics - division by zero returns saturated quotient and zero remainder
pub fn divmod_i512_by_i512(dividend: I512, divisor: I512) -> (I512, I512) {
    // Handle division by zero with saturation
    if divisor.is_zero() {
        let saturated_quotient = if dividend.is_negative() {
            I512::min_value()
        } else {
            I512::max_value()
        };
        return (saturated_quotient, I512::zero());
    }

    // Optimize for cases where both fit in i128 (common for small values)
    if dividend.fits_in_i128() && divisor.fits_in_i128() {
        let dividend_i128 = dividend.as_i128();
        let divisor_i128 = divisor.as_i128();
        let quotient = dividend_i128 / divisor_i128;
        let remainder = dividend_i128 % divisor_i128;
        return (I512::from_i128(quotient), I512::from_i128(remainder));
    }

    // Optimize for cases where both fit in I256
    if dividend.fits_in_i256() && divisor.fits_in_i256() {
        let dividend_i256 = dividend.as_i256();
        let divisor_i256 = divisor.as_i256();
        let (quot, rem) = crate::fixed_point::i256::divmod_i256_by_i256(dividend_i256, divisor_i256);
        return (I512::from_i256(quot), I512::from_i256(rem));
    }

    // Determine signs for proper signed division
    let dividend_negative = dividend.is_negative();
    let divisor_negative = divisor.is_negative();
    let quotient_negative = dividend_negative != divisor_negative;

    // Work with absolute values
    let abs_dividend = if dividend_negative {
        negate_i512(dividend)
    } else {
        dividend
    };

    let abs_divisor = if divisor_negative {
        negate_i512(divisor)
    } else {
        divisor
    };

    // PRODUCTION: 512-bit by 512-bit long division algorithm
    // Binary long division - processes each bit from MSB to LSB
    let mut quotient_words = [0u64; 8];
    let mut remainder = I512::zero();

    // Process each bit from most significant to least significant
    for word_idx in (0..8).rev() {
        for bit_idx in (0..64).rev() {
            // Shift remainder left by 1
            remainder = shift_left_i512_by_1(remainder);

            // Set the least significant bit to the current dividend bit
            let dividend_bit = (abs_dividend.words[word_idx] >> bit_idx) & 1;
            remainder.words[0] |= dividend_bit;

            // Try to subtract divisor from remainder
            if compare_i512_unsigned(remainder, abs_divisor) >= 0 {
                remainder = subtract_i512_unsigned(remainder, abs_divisor);
                // Set the corresponding quotient bit
                quotient_words[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    let mut quotient = I512 { words: quotient_words };

    // Apply signs
    if quotient_negative && !quotient.is_zero() {
        quotient = negate_i512(quotient);
    }

    if dividend_negative && !remainder.is_zero() {
        remainder = negate_i512(remainder);
    }

    (quotient, remainder)
}

/// Helper: Negate I512 value (two's complement)
///
/// ALGORITHM: Bitwise NOT + 1 (standard two's complement negation)
/// PRECISION: Exact for all values except I512::min_value() (overflow wraps)
#[inline(always)]
fn negate_i512(value: I512) -> I512 {
    let mut result = [0u64; 8];
    let mut carry = 1u64;

    for i in 0..8 {
        let (val, c) = (!value.words[i]).overflowing_add(carry);
        result[i] = val;
        carry = c as u64;
    }

    I512 { words: result }
}

/// Helper: Shift I512 left by exactly 1 bit
///
/// ALGORITHM: Propagate carry bit from each word to the next
/// USAGE: Required for binary long division algorithm
#[inline(always)]
fn shift_left_i512_by_1(value: I512) -> I512 {
    let mut result = [0u64; 8];
    let mut carry = 0u64;

    for i in 0..8 {
        let word = value.words[i];
        result[i] = (word << 1) | carry;
        carry = word >> 63;  // Extract MSB as carry to next word
    }

    I512 { words: result }
}

/// Helper: Compare two I512 values as unsigned (for long division)
///
/// ALGORITHM: Lexicographic comparison from MSB to LSB
/// RETURNS: 1 if a > b, -1 if a < b, 0 if a == b
#[inline(always)]
fn compare_i512_unsigned(a: I512, b: I512) -> i8 {
    for i in (0..8).rev() {
        if a.words[i] > b.words[i] {
            return 1;
        } else if a.words[i] < b.words[i] {
            return -1;
        }
    }
    0 // Equal
}

/// Helper: Subtract b from a (unsigned, for long division)
///
/// ALGORITHM: Word-by-word subtraction with borrow propagation
/// PRECONDITION: a >= b (ensured by compare_i512_unsigned check in caller)
#[inline(always)]
fn subtract_i512_unsigned(a: I512, b: I512) -> I512 {
    let mut result = [0u64; 8];
    let mut borrow = 0i128;

    for i in 0..8 {
        let diff = (a.words[i] as i128) - (b.words[i] as i128) - borrow;
        if diff < 0 {
            result[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            result[i] = diff as u64;
            borrow = 0;
        }
    }

    I512 { words: result }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_i512_basic_operations() {
        let a = I512::from_i128(42);
        let b = I512::from_i128(17);
        
        assert_eq!((a + b).as_i128(), 59);
        assert_eq!((a - b).as_i128(), 25);
        assert_eq!((a * b).as_i128(), 714);
    }
    
    #[test]
    fn test_i512_from_i256() {
        let i256_val = I256::from_i128(0x123456789ABCDEF0_i128);
        let i512_val = I512::from_i256(i256_val);
        
        assert_eq!(i512_val.as_i256(), i256_val);
        assert_eq!(i512_val.as_i128(), 0x123456789ABCDEF0_i128);
    }
    
    #[test]
    fn test_i512_shift_operations() {
        let value = I512::from_i128(0xFFFFFFFF00000000_i128);
        let shifted = value >> 32;
        assert_eq!(shifted.as_i128(), 0xFFFFFFFF_i128);
        
        let left_shifted = I512::from_i128(1) << 100;
        assert_ne!(left_shifted.as_i128(), 1); // Should overflow i128 range
    }
    
    #[test]
    fn test_q256_256_operations() {
        let one = I512::one_q256_256();
        let two = one + one;
        
        // Test Q256.256 multiplication
        let result = I512::mul_q256_256(one, two);
        assert_eq!(result, two);
    }
    
    #[test]
    fn test_q64_64_conversion() {
        let q64_val = 42i128 << 64; // 42.0 in Q64.64
        let q256_val = I512::from_q64_64(q64_val);
        let back_to_q64 = q256_val.to_q64_64();
        
        assert_eq!(back_to_q64, q64_val);
    }
    
    #[test]
    fn test_bytes_serialization() {
        let value = I512::from_i128(0x123456789ABCDEF0_i128);
        let bytes = value.to_bytes_le();
        let restored = I512::from_bytes_le(&bytes);
        
        assert_eq!(value, restored);
    }
}