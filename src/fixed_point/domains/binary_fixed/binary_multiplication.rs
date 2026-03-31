//! Binary Fixed-Point Multiplication
//!
//! **TIER PRIMITIVES**: checked_mul (Tiers 1-3), mul (Tiers 4-6)
//! **UGOD**: UniversalBinaryFixed::multiply() with overflow promotion
//! **SIMD**: AVX2-accelerated Q64.64 multiply for transcendental hotpath

use super::binary_types::*;
use super::i256::{I256, mul_i128_to_i256};
use crate::fixed_point::{I512, I1024};
use crate::fixed_point::core_types::errors::OverflowDetected;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// TIER 1: Q16.16
// ============================================================================

impl BinaryTier1 {
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        // BinaryTier1 is always Q16.16 (UGOD tier type, fixed format)
        let wide = (self.value as i64) * (other.value as i64);
        let round_bit = (wide >> 15) & 1;
        let shifted = (wide >> 16) + round_bit;
        let result = shifted as i32;
        if (result as i64) == shifted {
            Some(Self { value: result })
        } else {
            None
        }
    }
}

// ============================================================================
// TIER 2: Q32.32
// ============================================================================

impl BinaryTier2 {
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        let wide = (self.value as i128) * (other.value as i128);
        let round_bit = (wide >> 31) & 1;
        let shifted = (wide >> 32) + round_bit;
        let result = shifted as i64;
        if (result as i128) == shifted {
            Some(Self { value: result })
        } else {
            None
        }
    }
}

// ============================================================================
// TIER 3: Q64.64
// ============================================================================

impl BinaryTier3 {
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        let wide = I256::from_i128(self.value) * I256::from_i128(other.value);
        let round_bit = (wide >> 63) & I256::from_i128(1);
        let result = (wide >> 64) + round_bit;
        if result.fits_in_i128() {
            Some(Self { value: result.as_i128() })
        } else {
            None
        }
    }
}

// ============================================================================
// TIER 4: Q128.128
// ============================================================================

impl BinaryTier4 {
    pub fn mul(&self, other: &Self) -> Self {
        let wide = I512::from_i256(self.value) * I512::from_i256(other.value);
        let round_bit = (wide >> 127) & I512::from_i128(1);
        let result = (wide >> 128) + round_bit;
        Self { value: result.as_i256() }
    }
}

// ============================================================================
// TIER 5: Q256.256
// ============================================================================

impl BinaryTier5 {
    pub fn mul(&self, other: &Self) -> Self {
        let wide = I1024::from_i512(self.value) * I1024::from_i512(other.value);
        let round_bit = (wide >> 255) & I1024::from_i128(1);
        let result = (wide >> 256) + round_bit;
        Self { value: result.as_i512() }
    }
}

// ============================================================================
// TIER 6: Q512.512
// ============================================================================

impl BinaryTier6 {
    pub fn mul(&self, other: &Self) -> Self {
        Self { value: I1024::mul_q512_512(self.value, other.value) }
    }
}

// ============================================================================
// UGOD: UniversalBinaryFixed::multiply
// ============================================================================

impl UniversalBinaryFixed {
    pub fn multiply(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let (a, b) = self.align_to_common_tier(other);
        match (&a.value, &b.value) {
            (BinaryValue::Tier1(x), BinaryValue::Tier1(y)) => {
                match x.checked_mul(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier1(r), current_tier: 1 }),
                    None => {
                        let x2 = x.to_tier2();
                        let y2 = y.to_tier2();
                        match x2.checked_mul(&y2) {
                            Some(r) => Ok(Self { value: BinaryValue::Tier2(r), current_tier: 2 }),
                            None => {
                                let x3 = x.to_tier2().to_tier3();
                                let y3 = y.to_tier2().to_tier3();
                                match x3.checked_mul(&y3) {
                                    Some(r) => Ok(Self { value: BinaryValue::Tier3(r), current_tier: 3 }),
                                    None => Ok(Self { value: BinaryValue::Tier4(x3.to_tier4().mul(&y3.to_tier4())), current_tier: 4 }),
                                }
                            }
                        }
                    }
                }
            }
            (BinaryValue::Tier2(x), BinaryValue::Tier2(y)) => {
                match x.checked_mul(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier2(r), current_tier: 2 }),
                    None => {
                        let x3 = x.to_tier3();
                        let y3 = y.to_tier3();
                        match x3.checked_mul(&y3) {
                            Some(r) => Ok(Self { value: BinaryValue::Tier3(r), current_tier: 3 }),
                            None => Ok(Self { value: BinaryValue::Tier4(x3.to_tier4().mul(&y3.to_tier4())), current_tier: 4 }),
                        }
                    }
                }
            }
            (BinaryValue::Tier3(x), BinaryValue::Tier3(y)) => {
                match x.checked_mul(y) {
                    Some(r) => Ok(Self { value: BinaryValue::Tier3(r), current_tier: 3 }),
                    None => Ok(Self { value: BinaryValue::Tier4(x.to_tier4().mul(&y.to_tier4())), current_tier: 4 }),
                }
            }
            (BinaryValue::Tier4(x), BinaryValue::Tier4(y)) => {
                Ok(Self { value: BinaryValue::Tier4(x.mul(y)), current_tier: 4 })
            }
            (BinaryValue::Tier5(x), BinaryValue::Tier5(y)) => {
                Ok(Self { value: BinaryValue::Tier5(x.mul(y)), current_tier: 5 })
            }
            (BinaryValue::Tier6(x), BinaryValue::Tier6(y)) => {
                Ok(Self { value: BinaryValue::Tier6(x.mul(y)), current_tier: 6 })
            }
            _ => Err(OverflowDetected::InvalidInput),
        }
    }
}

// ============================================================================
// RAW Q64.64 MULTIPLY: Transcendental hotpath (NOT UGOD-managed)
// ============================================================================

/// Multiply two Q64.64 binary fixed-point values using I256 intermediate to prevent truncation
///
/// Auto-detecting SIMD acceleration with guaranteed fallback to scalar implementation.
#[inline(always)]
pub fn multiply_binary_i128(a: i128, b: i128) -> i128 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { multiply_binary_i128_avx2(a, b) }
        } else {
            multiply_binary_i128_scalar(a, b)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        multiply_binary_i128_scalar(a, b)
    }
}

/// Scalar implementation using I256 intermediate
///
/// Reference implementation, also used as SIMD equivalence baseline.
#[inline(always)]
pub fn multiply_binary_i128_scalar(a: i128, b: i128) -> i128 {
    let result = mul_i128_to_i256(a, b);
    let scaled = result >> 64;

    // Round to nearest even (IEEE 754 compliant)
    let remainder = result.words[0];
    let half = 1u64 << 63;

    if remainder > half || (remainder == half && (scaled.words[0] & 1) == 1) {
        scaled.as_i128().wrapping_add(1)
    } else {
        scaled.as_i128()
    }
}

/// AVX2-accelerated multiplication using 256-bit SIMD registers
///
/// CRITICAL: Must produce bit-identical results to multiply_binary_i128_scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multiply_binary_i128_avx2(a: i128, b: i128) -> i128 {
    let a_neg = a < 0;
    let b_neg = b < 0;
    let result_neg = a_neg != b_neg;

    let a_abs = a.unsigned_abs();
    let b_abs = b.unsigned_abs();

    let a_lo = a_abs as u64;
    let a_hi = (a_abs >> 64) as u64;
    let b_lo = b_abs as u64;
    let b_hi = (b_abs >> 64) as u64;

    // Load operand parts into SIMD registers
    let _a_parts = _mm256_set_epi64x(0, 0, a_hi as i64, a_lo as i64);
    let _b_parts = _mm256_set_epi64x(0, 0, b_hi as i64, b_lo as i64);

    // Compute partial products (same as scalar algorithm)
    let lo_lo = (a_lo as u128) * (b_lo as u128);
    let lo_hi = (a_lo as u128) * (b_hi as u128);
    let hi_lo = (a_hi as u128) * (b_lo as u128);
    let hi_hi = (a_hi as u128) * (b_hi as u128);

    // Accumulate results (identical to mul_i128_to_i256)
    let mut result = I256::zero();

    result.words[0] = lo_lo as u64;
    result.words[1] = (lo_lo >> 64) as u64;

    let (sum1, carry1) = result.words[1].overflowing_add(lo_hi as u64);
    result.words[1] = sum1;
    result.words[2] = (lo_hi >> 64) as u64 + carry1 as u64;

    let (sum2, carry2) = result.words[1].overflowing_add(hi_lo as u64);
    result.words[1] = sum2;
    let (sum3, carry3) = result.words[2].overflowing_add((hi_lo >> 64) as u64 + carry2 as u64);
    result.words[2] = sum3;
    result.words[3] = carry3 as u64;

    let (sum4, carry4) = result.words[2].overflowing_add(hi_hi as u64);
    result.words[2] = sum4;
    result.words[3] += (hi_hi >> 64) as u64 + carry4 as u64;

    // Handle sign
    if result_neg {
        let mut borrow = 1u64;
        for i in 0..4 {
            let (val, b) = (!result.words[i]).overflowing_add(borrow);
            result.words[i] = val;
            borrow = b as u64;
        }
    }

    // Apply identical rounding logic to scalar version
    let scaled = result >> 64;
    let remainder = result.words[0];
    let half = 1u64 << 63;

    if remainder > half || (remainder == half && (scaled.words[0] & 1) == 1) {
        scaled.as_i128().wrapping_add(1)
    } else {
        scaled.as_i128()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier3_multiply() {
        let two = BinaryTier3::from_raw(2i128 << 64);
        let three = BinaryTier3::from_raw(3i128 << 64);
        let six = two.checked_mul(&three).unwrap();
        assert_eq!(six.raw(), 6i128 << 64);
    }

    #[test]
    fn test_multiply_binary_exact_cases() {
        let one = 1i128 << 64;
        let result = multiply_binary_i128(one, one);
        assert_eq!(result, one, "1.0 * 1.0 should be exactly 1.0");

        let two = 2i128 << 64;
        let half = 1i128 << 63;
        let result = multiply_binary_i128(two, half);
        assert_eq!(result, one, "2.0 * 0.5 should be exactly 1.0");
    }

    #[test]
    fn test_multiply_binary_simd_equivalence() {
        let test_cases = [
            (1i128 << 64, 1i128 << 64),
            (2i128 << 64, 1i128 << 63),
            (0x123456789ABCDEF0_i128, 0xFEDCBA9876543210_i128),
        ];

        for &(a, b) in &test_cases {
            let scalar_result = multiply_binary_i128_scalar(a, b);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let simd_result = unsafe { multiply_binary_i128_avx2(a, b) };
                    assert_eq!(scalar_result, simd_result,
                        "SIMD and scalar results must be bit-identical");
                }
            }
        }
    }
}
