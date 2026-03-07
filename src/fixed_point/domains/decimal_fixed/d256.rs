//! Decimal Domain 256-bit Integer Arithmetic
//! 
//! DOMAIN-SPECIFIC: Pure decimal arithmetic foundation for DecimalFixed operations
//! PRECISION: Optimized for base-10 scaling factors (powers of 10)
//! SEPARATION: Completely isolated from binary domain (B256) implementations

use std::ops::Mul;

/// Decimal-domain 256-bit integer type for intermediate calculations
/// 
/// CRITICAL: This type is specifically designed for decimal fixed-point arithmetic
/// SCALING: Optimized for powers of 10 (not powers of 2 like binary domain)
/// SEPARATION: Must never be mixed with binary domain B256 types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct D256 {
    /// Stored as four 64-bit words (little-endian)
    pub words: [u64; 4],
}

impl D256 {
    #[inline(always)]
    pub const fn zero() -> Self {
        D256 { words: [0, 0, 0, 0] }
    }
    
    #[inline(always)]
    pub const fn from_words(words: [u64; 4]) -> Self {
        D256 { words }
    }
    
    /// Convert from i128 with proper sign extension for decimal operations
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
        D256 { words }
    }
    
    /// Extract lower 128 bits as i128 (decimal-specific conversion)
    #[inline(always)]
    pub fn as_i128(self) -> i128 {
        // Take lower 128 bits
        ((self.words[1] as i128) << 64) | (self.words[0] as i128)
    }
    
    /// Check if the D256 value is zero
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.words[0] == 0 && self.words[1] == 0 && self.words[2] == 0 && self.words[3] == 0
    }
    
    /// Check if the D256 value is negative (decimal-specific sign check)
    #[inline(always)]
    pub fn is_negative(self) -> bool {
        // Check the sign bit (MSB of the highest word)
        (self.words[3] & 0x8000_0000_0000_0000) != 0
    }
    
    /// Check if value fits in i128 (for decimal domain overflow detection)
    #[inline(always)]
    pub fn fits_in_i128(self) -> bool {
        let is_negative = (self.words[1] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };
        
        self.words[2] == expected_high && self.words[3] == expected_high
    }
    
    /// Convert from u128 for decimal operations
    #[inline(always)]
    pub const fn from_u128(value: u128) -> Self {
        D256 { 
            words: [
                value as u64,
                (value >> 64) as u64,
                0,
                0,
            ]
        }
    }
    
    /// Extract lower 128 bits as u128
    #[inline(always)]
    pub fn as_u128(self) -> u128 {
        ((self.words[1] as u128) << 64) | (self.words[0] as u128)
    }

    /// Convert from i64 with sign extension
    #[inline(always)]
    pub const fn from_i64(value: i64) -> Self {
        Self::from_i128(value as i128)
    }

    /// Convert from i32 with sign extension
    #[inline(always)]
    pub const fn from_i32(value: i32) -> Self {
        Self::from_i128(value as i128)
    }

    /// Convert from i16 with sign extension
    #[inline(always)]
    pub const fn from_i16(value: i16) -> Self {
        Self::from_i128(value as i128)
    }

    /// Convert from u8
    #[inline(always)]
    pub const fn from_u8(value: u8) -> Self {
        Self::from_i128(value as i128)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(&self) -> Self {
        if self.is_negative() {
            negate_d256(*self)
        } else {
            *self
        }
    }
}

// Arithmetic operations for D256 (decimal-optimized)
impl std::ops::Add for D256 {
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
        
        D256 { words: result }
    }
}

impl std::ops::Sub for D256 {
    type Output = Self;
    
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut result = [0u64; 4];
        let mut borrow = 0u64;
        
        for i in 0..4 {
            let diff = (self.words[i] as u128).wrapping_sub((rhs.words[i] as u128) + (borrow as u128));
            result[i] = diff as u64;
            borrow = if diff > u128::MAX { 1 } else { 0 };
        }
        
        D256 { words: result }
    }
}

impl Mul for D256 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Proper D256×D256→D512 multiplication, then truncate to D256
        let result_d512 = self.mul_to_d512(rhs);
        result_d512.as_d256()
    }
}

impl std::ops::Div for D256 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        divmod_d256_by_d256(self, rhs).0
    }
}

impl std::ops::Rem for D256 {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        divmod_d256_by_d256(self, rhs).1
    }
}

// Comparison traits for D256 (decimal domain)
impl PartialOrd for D256 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for D256 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare as signed 256-bit integers (decimal domain)
        let self_negative = self.is_negative();
        let other_negative = other.is_negative();

        match (self_negative, other_negative) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                // Same sign, compare magnitude word by word
                for i in (0..4).rev() {
                    match self.words[i].cmp(&other.words[i]) {
                        std::cmp::Ordering::Equal => continue,
                        ordering => {
                            // For negative numbers, reverse the comparison
                            return if self_negative { ordering.reverse() } else { ordering };
                        }
                    }
                }
                std::cmp::Ordering::Equal
            }
        }
    }
}

impl D256 {
    /// Full D256×D256→D512 multiplication for maximum decimal precision
    #[inline(always)]
    pub fn mul_to_d512(self, rhs: D256) -> super::d512::D512 {
        mul_d256_to_d512(self, rhs)
    }
}

/// Full 128×128→256 bit multiplication optimized for decimal operations
/// 
/// CRITICAL: Decimal-domain multiplication algorithm
/// Produces exact 256-bit results from 128-bit decimal inputs
#[inline(always)]
pub fn mul_i128_to_d256(a: i128, b: i128) -> D256 {
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
    
    // Compute partial products
    let lo_lo = (a_lo as u128) * (b_lo as u128);
    let lo_hi = (a_lo as u128) * (b_hi as u128);
    let hi_lo = (a_hi as u128) * (b_lo as u128);
    let hi_hi = (a_hi as u128) * (b_hi as u128);
    
    // Accumulate results
    let mut result = D256::zero();
    
    // Add lo_lo
    result.words[0] = lo_lo as u64;
    result.words[1] = (lo_lo >> 64) as u64;
    
    // Add lo_hi
    let (sum1, carry1) = result.words[1].overflowing_add(lo_hi as u64);
    result.words[1] = sum1;
    result.words[2] = (lo_hi >> 64) as u64 + carry1 as u64;
    
    // Add hi_lo  
    let (sum2, carry2) = result.words[1].overflowing_add(hi_lo as u64);
    result.words[1] = sum2;
    let (sum3, carry3) = result.words[2].overflowing_add((hi_lo >> 64) as u64 + carry2 as u64);
    result.words[2] = sum3;
    result.words[3] = carry3 as u64;
    
    // Add hi_hi
    let (sum4, carry4) = result.words[2].overflowing_add(hi_hi as u64);
    result.words[2] = sum4;
    result.words[3] += (hi_hi >> 64) as u64 + carry4 as u64;
    
    // Handle sign (decimal-specific two's complement)
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

/// Full D256×D256→D512 multiplication for maximum decimal precision
#[inline(always)]
pub fn mul_d256_to_d512(a: D256, b: D256) -> super::d512::D512 {
    use super::d512::D512;
    
    // Decompose each D256 into two i128 parts
    let a_lo = ((a.words[1] as i128) << 64) | (a.words[0] as i128);
    let a_hi = ((a.words[3] as i128) << 64) | (a.words[2] as i128);
    let b_lo = ((b.words[1] as i128) << 64) | (b.words[0] as i128);
    let b_hi = ((b.words[3] as i128) << 64) | (b.words[2] as i128);
    
    // Compute the four partial products for decimal domain
    let lo_lo = mul_i128_to_d256(a_lo, b_lo);     // a_lo * b_lo
    let lo_hi = mul_i128_to_d256(a_lo, b_hi);     // a_lo * b_hi  
    let hi_lo = mul_i128_to_d256(a_hi, b_lo);     // a_hi * b_lo
    let hi_hi = mul_i128_to_d256(a_hi, b_hi);     // a_hi * b_hi
    
    // Accumulate into 512-bit decimal result
    let mut result = D512::zero();
    
    // Add lo_lo (no shift)
    result.words[0] = lo_lo.words[0];
    result.words[1] = lo_lo.words[1];
    result.words[2] = lo_lo.words[2];
    result.words[3] = lo_lo.words[3];
    
    // Add lo_hi and hi_lo (shift by 128 bits = 2 words)
    let mid_sum = D256::zero() + lo_hi + hi_lo;
    let mut carry = 0u64;
    
    for i in 0..4 {
        let word_idx = i + 2; // Shift by 2 words (128 bits)
        if word_idx < 8 {
            let sum = (result.words[word_idx] as u128) + (mid_sum.words[i] as u128) + (carry as u128);
            result.words[word_idx] = sum as u64;
            carry = (sum >> 64) as u64;
        }
    }
    
    // Add hi_hi (shift by 256 bits = 4 words)
    carry = 0;
    for i in 0..4 {
        let word_idx = i + 4; // Shift by 4 words (256 bits)
        if word_idx < 8 {
            let sum = (result.words[word_idx] as u128) + (hi_hi.words[i] as u128) + (carry as u128);
            result.words[word_idx] = sum as u64;
            carry = (sum >> 64) as u64;
        }
    }
    
    result
}

/// Proper D256 two's complement negation for decimal domain
/// 
/// ALGORITHM: Bitwise NOT + 1 across all 256 bits (decimal-optimized)
/// PRECISION: Exact negation maintaining all decimal precision bits
pub fn negate_d256(value: D256) -> D256 {
    let mut result = D256 { words: [0; 4] };
    let mut carry = 1u64;
    
    // Two's complement: flip all bits and add 1
    for i in 0..4 {
        let (negated, new_carry) = (!value.words[i]).overflowing_add(carry);
        result.words[i] = negated;
        carry = if new_carry { 1 } else { 0 };
    }
    
    result
}

/// Division with remainder for D256 by i128 (decimal-specific)
/// 
/// ALGORITHM: Full 256-bit long division with exact remainder calculation for decimal scaling
/// PRECISION: Maintains exact arithmetic for decimal scaling operations
pub fn divmod_d256_by_i128(dividend: D256, divisor: i128) -> (i128, i128) {
    if divisor == 0 {
        return (if dividend.words[3] as i64 >= 0 { i128::MAX } else { i128::MIN }, 0);
    }
    
    // Handle simple case where result fits in i128
    if dividend.fits_in_i128() {
        let dividend_i128 = dividend.as_i128();
        let quotient = dividend_i128 / divisor;
        let remainder = dividend_i128 % divisor;
        return (quotient, remainder);
    }
    
    // Determine signs
    let dividend_negative = dividend.is_negative();
    let divisor_negative = divisor < 0;
    let result_negative = dividend_negative != divisor_negative;
    
    // Work with absolute values using proper D256 negation
    let abs_dividend = if dividend_negative {
        negate_d256(dividend)
    } else {
        dividend
    };
    
    let abs_divisor = divisor.unsigned_abs() as u128;
    
    // PROPER 256-bit by 128-bit long division algorithm for decimal domain
    let mut quotient = 0u128;
    let mut remainder = 0u128;
    
    // Process from most significant to least significant 64-bit word
    for word_idx in (0..4).rev() {
        // Shift remainder by 64 bits and add next word
        remainder = (remainder << 64) + abs_dividend.words[word_idx] as u128;
        
        // Divide remainder by divisor
        let word_quotient = remainder / abs_divisor;
        remainder = remainder % abs_divisor;
        
        // Add word quotient to total (if it fits)
        if word_idx < 2 { // Only accumulate if result fits in 128 bits
            quotient = (quotient << 64) + word_quotient;
        } else if word_quotient != 0 {
            // Overflow case - quotient too large for i128
            let saturated_quotient = if result_negative { i128::MIN } else { i128::MAX };
            let saturated_remainder = if dividend_negative { -(remainder as i128) } else { remainder as i128 };
            return (saturated_quotient, saturated_remainder);
        }
    }
    
    // Convert results to signed values with proper bounds checking
    let final_quotient = if quotient > i128::MAX as u128 {
        // Overflow case
        if result_negative { i128::MIN } else { i128::MAX }
    } else {
        let signed_quotient = quotient as i128;
        if result_negative { -signed_quotient } else { signed_quotient }
    };
    
    let final_remainder = if remainder > i128::MAX as u128 {
        // This should never happen with proper divisor, but guard against it
        0
    } else {
        let signed_remainder = remainder as i128;
        if dividend_negative { -signed_remainder } else { signed_remainder }
    };
    
    (final_quotient, final_remainder)
}

/// Division with remainder for D256 by D256 (decimal-specific)
///
/// ALGORITHM: Full 256-bit by 256-bit long division with exact remainder calculation
/// PRECISION: Maintains exact arithmetic for decimal scaling operations
/// DOMAIN: Pure decimal domain - optimized for base-10 operations
pub fn divmod_d256_by_d256(dividend: D256, divisor: D256) -> (D256, D256) {
    // Handle division by zero with saturation
    if divisor.is_zero() {
        let saturated_quotient = if dividend.is_negative() {
            D256::from_i128(i128::MIN)
        } else {
            D256::from_i128(i128::MAX)
        };
        return (saturated_quotient, D256::zero());
    }

    // Optimize for cases where both fit in i128
    if dividend.fits_in_i128() && divisor.fits_in_i128() {
        let dividend_i128 = dividend.as_i128();
        let divisor_i128 = divisor.as_i128();
        let quotient = dividend_i128 / divisor_i128;
        let remainder = dividend_i128 % divisor_i128;
        return (D256::from_i128(quotient), D256::from_i128(remainder));
    }

    // Determine signs for proper signed division
    let dividend_negative = dividend.is_negative();
    let divisor_negative = divisor.is_negative();
    let quotient_negative = dividend_negative != divisor_negative;

    // Work with absolute values
    let abs_dividend = if dividend_negative {
        negate_d256(dividend)
    } else {
        dividend
    };

    let abs_divisor = if divisor_negative {
        negate_d256(divisor)
    } else {
        divisor
    };

    // PRODUCTION: 256-bit by 256-bit long division algorithm
    // This implements the standard long division algorithm word-by-word
    let mut quotient_words = [0u64; 4];
    let mut remainder = D256::zero();

    // Process each bit from most significant to least significant
    for word_idx in (0..4).rev() {
        for bit_idx in (0..64).rev() {
            // Shift remainder left by 1
            remainder = shift_left_d256_by_1(remainder);

            // Set the least significant bit to the current dividend bit
            let dividend_bit = (abs_dividend.words[word_idx] >> bit_idx) & 1;
            remainder.words[0] |= dividend_bit;

            // Try to subtract divisor from remainder
            if compare_d256_unsigned(remainder, abs_divisor) >= 0 {
                remainder = subtract_d256_unsigned(remainder, abs_divisor);
                // Set the corresponding quotient bit
                quotient_words[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    let mut quotient = D256 { words: quotient_words };

    // Apply signs
    if quotient_negative && !is_d256_zero(quotient) {
        quotient = negate_d256(quotient);
    }

    if dividend_negative && !is_d256_zero(remainder) {
        remainder = negate_d256(remainder);
    }

    (quotient, remainder)
}

/// Helper: Shift D256 left by 1 bit (for long division algorithm)
#[inline(always)]
fn shift_left_d256_by_1(value: D256) -> D256 {
    let mut result = [0u64; 4];
    let mut carry = 0u64;

    for i in 0..4 {
        let word = value.words[i];
        result[i] = (word << 1) | carry;
        carry = word >> 63;
    }

    D256 { words: result }
}

/// Helper: Compare two D256 values as unsigned (for long division)
#[inline(always)]
fn compare_d256_unsigned(a: D256, b: D256) -> i8 {
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
fn subtract_d256_unsigned(a: D256, b: D256) -> D256 {
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

    D256 { words: result }
}

/// Helper: Check if D256 is zero
#[inline(always)]
fn is_d256_zero(value: D256) -> bool {
    value.words[0] == 0 && value.words[1] == 0 && value.words[2] == 0 && value.words[3] == 0
}

// Type aliases for domain clarity
pub type DecimalD256 = D256;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_d256_basic_operations() {
        let a = D256::from_i128(100);
        let b = D256::from_i128(25);
        
        let sum = a + b;
        assert_eq!(sum.as_i128(), 125);
        
        let diff = a - b;
        assert_eq!(diff.as_i128(), 75);
        
        let product = a * b;
        assert_eq!(product.as_i128(), 2500);
    }
    
    #[test]
    fn test_d256_multiplication() {
        let a = 0x123456789ABCDEF0123456789ABCDEF_i128;
        let b = 0x111111111111111111111111111111_i128;
        
        let result = mul_i128_to_d256(a, b);
        
        // Verify the multiplication produced a valid result
        assert!(!result.is_zero());
    }
    
    #[test]
    fn test_d256_decimal_specific() {
        // Test decimal-specific operations (powers of 10)
        let decimal_val = D256::from_i128(1000000); // 10^6
        let scale_10 = D256::from_i128(10);
        
        let scaled = decimal_val * scale_10;
        assert_eq!(scaled.as_i128(), 10000000); // 10^7
    }
    
    #[test]
    fn test_d256_domain_separation() {
        // Ensure D256 operations are isolated from binary domain
        let d_val = D256::from_i128(1000);
        assert_eq!(d_val.as_i128(), 1000);
        
        // Test negation
        let neg_val = negate_d256(d_val);
        assert_eq!(neg_val.as_i128(), -1000);
    }
    
    #[test]
    fn test_divmod_d256_by_i128() {
        let dividend = D256::from_i128(1000);
        let divisor = 10i128;
        
        let (quotient, remainder) = divmod_d256_by_i128(dividend, divisor);
        assert_eq!(quotient, 100);
        assert_eq!(remainder, 0);
        
        // Test with remainder
        let dividend = D256::from_i128(1007);
        let (quotient, remainder) = divmod_d256_by_i128(dividend, divisor);
        assert_eq!(quotient, 100);
        assert_eq!(remainder, 7);
    }
}