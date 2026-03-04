//! Decimal Domain 512-bit Integer Arithmetic
//! 
//! DOMAIN-SPECIFIC: Pure decimal arithmetic foundation for extended DecimalFixed operations
//! PRECISION: Optimized for base-10 scaling factors with maximum intermediate precision
//! SEPARATION: Completely isolated from binary domain (B512) implementations

/// Decimal-domain 512-bit integer type for extended intermediate calculations
/// 
/// CRITICAL: This type is specifically designed for decimal fixed-point arithmetic
/// SCALING: Optimized for powers of 10 with extended precision
/// SEPARATION: Must never be mixed with binary domain B512 types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct D512 {
    /// Stored as eight 64-bit words (little-endian)
    pub words: [u64; 8],
}

impl D512 {
    #[inline(always)]
    pub const fn zero() -> Self {
        D512 { words: [0; 8] }
    }
    
    #[inline(always)]
    pub const fn from_words(words: [u64; 8]) -> Self {
        D512 { words }
    }
    
    /// Convert from D256 with proper extension for decimal operations
    #[inline(always)]
    pub const fn from_d256(value: super::d256::D256) -> Self {
        let is_negative = (value.words[3] as i64) < 0;
        let extend_value = if is_negative { u64::MAX } else { 0 };
        
        D512 {
            words: [
                value.words[0],
                value.words[1],
                value.words[2],
                value.words[3],
                extend_value,
                extend_value,
                extend_value,
                extend_value,
            ]
        }
    }
    
    /// Extract lower 256 bits as D256 (decimal-specific conversion)
    #[inline(always)]
    pub fn as_d256(self) -> super::d256::D256 {
        super::d256::D256 {
            words: [
                self.words[0],
                self.words[1],
                self.words[2],
                self.words[3],
            ]
        }
    }
    
    /// Check if the D512 value is zero
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Check if the D512 value is negative (decimal-specific sign check)
    #[inline(always)]
    pub fn is_negative(&self) -> bool {
        // Check the sign bit (MSB of the highest word)
        (self.words[7] & 0x8000_0000_0000_0000) != 0
    }

    /// Check if value fits in D256 (for overflow detection)
    #[inline(always)]
    pub fn fits_in_d256(&self) -> bool {
        let is_negative = (self.words[3] as i64) < 0;
        let expected_high = if is_negative { u64::MAX } else { 0 };

        self.words[4] == expected_high
            && self.words[5] == expected_high
            && self.words[6] == expected_high
            && self.words[7] == expected_high
    }

    /// Minimum value for saturation (most negative D512)
    #[inline(always)]
    pub const fn min_value() -> Self {
        D512 {
            words: [0, 0, 0, 0, 0, 0, 0, 0x8000_0000_0000_0000]
        }
    }

    /// Maximum value for saturation (most positive D512)
    #[inline(always)]
    pub const fn max_value() -> Self {
        D512 {
            words: [u64::MAX, u64::MAX, u64::MAX, u64::MAX,
                    u64::MAX, u64::MAX, u64::MAX, 0x7FFF_FFFF_FFFF_FFFF]
        }
    }

    /// Convert from i16 with sign extension
    #[inline(always)]
    pub const fn from_i16(value: i16) -> Self {
        let extend_value = if value < 0 { u64::MAX } else { 0 };
        let low_word = value as i64 as u64; // Sign-extend to u64

        D512 {
            words: [low_word, extend_value, extend_value, extend_value,
                    extend_value, extend_value, extend_value, extend_value]
        }
    }

    /// Convert from i32 with sign extension
    #[inline(always)]
    pub const fn from_i32(value: i32) -> Self {
        let extend_value = if value < 0 { u64::MAX } else { 0 };
        let low_word = value as i64 as u64; // Sign-extend to u64

        D512 {
            words: [low_word, extend_value, extend_value, extend_value,
                    extend_value, extend_value, extend_value, extend_value]
        }
    }

    /// Convert from i64 with sign extension
    #[inline(always)]
    pub const fn from_i64(value: i64) -> Self {
        let extend_value = if value < 0 { u64::MAX } else { 0 };
        let low_word = value as u64;

        D512 {
            words: [low_word, extend_value, extend_value, extend_value,
                    extend_value, extend_value, extend_value, extend_value]
        }
    }

    /// Convert from i128 with sign extension
    #[inline(always)]
    pub const fn from_i128(value: i128) -> Self {
        let extend_value = if value < 0 { u64::MAX } else { 0 };

        D512 {
            words: [
                value as u64,
                (value >> 64) as u64,
                extend_value,
                extend_value,
                extend_value,
                extend_value,
                extend_value,
                extend_value,
            ]
        }
    }

    /// Convert from u8
    #[inline(always)]
    pub const fn from_u8(value: u8) -> Self {
        Self::from_i128(value as i128)
    }

    /// Check if value is even
    #[inline(always)]
    pub fn is_even(&self) -> bool {
        (self.words[0] & 1) == 0
    }

    /// Get sign as -1, 0, or 1
    #[inline(always)]
    pub fn signum(&self) -> Self {
        if self.is_zero() {
            D512::zero()
        } else if self.is_negative() {
            D512::from_i128(-1)
        } else {
            D512::from_i128(1)
        }
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(&self) -> Self {
        if self.is_negative() {
            // Two's complement negation
            let mut result = D512 { words: [0; 8] };
            let mut carry = 1u64;
            for i in 0..8 {
                let (negated, new_carry) = (!self.words[i]).overflowing_add(carry);
                result.words[i] = negated;
                carry = if new_carry { 1 } else { 0 };
            }
            result
        } else {
            *self
        }
    }

}

// Arithmetic operations for D512 (decimal-optimized)
impl std::ops::Add for D512 {
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
        
        D512 { words: result }
    }
}

impl std::ops::Sub for D512 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut result = [0u64; 8];
        let mut borrow = 0u64;

        for i in 0..8 {
            let diff = (self.words[i] as u128).wrapping_sub((rhs.words[i] as u128) + (borrow as u128));
            result[i] = diff as u64;
            borrow = if diff > u128::MAX { 1 } else { 0 };
        }

        D512 { words: result }
    }
}

impl std::ops::Mul for D512 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // D512×D512 multiplication (truncates to 512 bits)

        let mut result = [0u64; 8];

        // Perform multi-precision multiplication
        for i in 0..8 {
            let mut carry = 0u128;
            for j in 0..8 {
                if i + j < 8 {
                    let product = (self.words[i] as u128) * (rhs.words[j] as u128);
                    let sum = (result[i + j] as u128) + product + carry;
                    result[i + j] = sum as u64;
                    carry = sum >> 64;
                }
            }
        }

        D512 { words: result }
    }
}

impl std::ops::Div for D512 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        divmod_d512_by_d512(self, rhs).0
    }
}

impl std::ops::Rem for D512 {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        divmod_d512_by_d512(self, rhs).1
    }
}

// Comparison traits for D512 (decimal domain)
impl PartialOrd for D512 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for D512 {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare as signed 512-bit integers (decimal domain)
        let self_negative = self.is_negative();
        let other_negative = other.is_negative();

        match (self_negative, other_negative) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                // Same sign, compare magnitude word by word
                for i in (0..8).rev() {
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

/// Division with remainder for D512 by D512 (decimal-specific)
///
/// ALGORITHM: Full 512-bit by 512-bit long division with exact remainder calculation
/// PRECISION: Maintains exact arithmetic for decimal scaling operations - CRITICAL FOR 0-ULP
/// DOMAIN: Pure decimal domain - optimized for base-10 operations
pub fn divmod_d512_by_d512(dividend: D512, divisor: D512) -> (D512, D512) {
    // Handle division by zero with saturation
    if divisor.is_zero() {
        let saturated_quotient = if dividend.is_negative() {
            D512::min_value()
        } else {
            D512::max_value()
        };
        return (saturated_quotient, D512::zero());
    }

    // Optimize for cases where both fit in D256
    if dividend.fits_in_d256() && divisor.fits_in_d256() {
        let dividend_d256 = dividend.as_d256();
        let divisor_d256 = divisor.as_d256();
        let (quotient, remainder) = super::d256::divmod_d256_by_d256(dividend_d256, divisor_d256);
        return (D512::from_d256(quotient), D512::from_d256(remainder));
    }

    // Determine signs for proper signed division
    let dividend_negative = dividend.is_negative();
    let divisor_negative = divisor.is_negative();
    let quotient_negative = dividend_negative != divisor_negative;

    // Work with absolute values
    let abs_dividend = if dividend_negative {
        negate_d512(dividend)
    } else {
        dividend
    };

    let abs_divisor = if divisor_negative {
        negate_d512(divisor)
    } else {
        divisor
    };

    // PRODUCTION: 512-bit by 512-bit long division algorithm
    let mut quotient_words = [0u64; 8];
    let mut remainder = D512::zero();

    // Process each bit from most significant to least significant
    for word_idx in (0..8).rev() {
        for bit_idx in (0..64).rev() {
            // Shift remainder left by 1
            remainder = shift_left_d512_by_1(remainder);

            // Set the least significant bit to the current dividend bit
            let dividend_bit = (abs_dividend.words[word_idx] >> bit_idx) & 1;
            remainder.words[0] |= dividend_bit;

            // Try to subtract divisor from remainder
            if compare_d512_unsigned(remainder, abs_divisor) >= 0 {
                remainder = subtract_d512_unsigned(remainder, abs_divisor);
                // Set the corresponding quotient bit
                quotient_words[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    let mut quotient = D512 { words: quotient_words };

    // Apply signs
    if quotient_negative && !is_d512_zero(quotient) {
        quotient = negate_d512(quotient);
    }

    if dividend_negative && !is_d512_zero(remainder) {
        remainder = negate_d512(remainder);
    }

    (quotient, remainder)
}

/// Helper: Negate D512 value (two's complement)
#[inline(always)]
pub fn negate_d512(value: D512) -> D512 {
    let mut result = [0u64; 8];
    let mut carry = 1u64;

    for i in 0..8 {
        let inverted = !value.words[i];
        let sum = (inverted as u128) + (carry as u128);
        result[i] = sum as u64;
        carry = (sum >> 64) as u64;
    }

    D512 { words: result }
}

/// Helper: Shift D512 left by 1 bit (for long division algorithm)
#[inline(always)]
fn shift_left_d512_by_1(value: D512) -> D512 {
    let mut result = [0u64; 8];
    let mut carry = 0u64;

    for i in 0..8 {
        let word = value.words[i];
        result[i] = (word << 1) | carry;
        carry = word >> 63;
    }

    D512 { words: result }
}

/// Helper: Compare two D512 values as unsigned (for long division)
#[inline(always)]
fn compare_d512_unsigned(a: D512, b: D512) -> i8 {
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
#[inline(always)]
fn subtract_d512_unsigned(a: D512, b: D512) -> D512 {
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

    D512 { words: result }
}

/// Helper: Check if D512 is zero
#[inline(always)]
fn is_d512_zero(value: D512) -> bool {
    value.words.iter().all(|&w| w == 0)
}

// Type aliases for domain clarity
pub type DecimalD512 = D512;

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::d256::D256;
    
    #[test]
    fn test_d512_basic_operations() {
        let a = D512::from_d256(D256::from_i128(1000));
        let b = D512::from_d256(D256::from_i128(500));
        
        let sum = a + b;
        assert_eq!(sum.as_d256().as_i128(), 1500);
        
        let diff = a - b;
        assert_eq!(diff.as_d256().as_i128(), 500);
    }
    
    #[test]
    fn test_d512_from_d256_conversion() {
        let d256_val = D256::from_i128(-1000);
        let d512_val = D512::from_d256(d256_val);
        
        // Should preserve the negative value
        assert_eq!(d512_val.as_d256().as_i128(), -1000);
    }
}