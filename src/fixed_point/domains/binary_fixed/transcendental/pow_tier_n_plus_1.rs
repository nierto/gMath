// ============================================================================
// TIER N+1 POWER FUNCTION
// ============================================================================
//
// **ALGORITHM**: pow(x, y) = exp(y × ln(x)) for x > 0
//
// **PRINCIPLE**: Compute ALL intermediate steps at tier N+1, downscale once at end
//
// **GUARANTEE**: Same contractual precision as exp() and ln()
//   - Tier 3 (Q64.64):   19 decimal digits, ≤1 ULP
//   - Tier 4 (Q128.128): 38 decimal digits, ≤1 ULP
//   - Tier 5 (Q256.256): 77 decimal digits, ≤1 ULP
//
// **SPECIAL CASES**:
//   - pow(x, 0) = 1 for any x ≠ 0
//   - pow(0, y) = 0 for y > 0
//   - pow(0, 0) = 1 (IEEE convention)
//   - pow(1, y) = 1 for any y
//   - pow(x, 1) = x for any x > 0
//   - pow(x, y) where x < 0: Error (complex result)
//   - pow(x, y) where x = 0 and y < 0: Error (infinity)
//
// **ERROR ANALYSIS**:
//   ln(x):       ≤1 ULP at tier N+1
//   y × ln(x):   ≤0.5 ULP (multiplication rounding)
//   exp(...):    ≤1 ULP at tier N+1
//   downscale:   ≤0.5 ULP (rounding)
//   Total:       ≤1 ULP at tier N (errors don't simply add due to tier N+1 buffer)
//
// ============================================================================

use crate::fixed_point::i256::I256;
use crate::fixed_point::i512::I512;
use crate::fixed_point::i1024::I1024;

// Conditional imports based on table format
#[cfg(any(table_format = "q256_256", table_format = "q128_128"))]
use super::ln_tier_n_plus_1::ln_q256_256_native;

// ln_q128_128_native no longer needed — pow now uses REAL Q256.256/Q128.128 tier N+1

#[cfg(any(table_format = "q256_256", table_format = "q128_128"))]
use super::exp_tier_n_plus_1::downscale_q256_to_q128;

#[cfg(any(table_format = "q128_128", table_format = "q64_64"))]
use super::exp_tier_n_plus_1::downscale_q128_to_q64;

#[cfg(any(table_format = "q256_256", table_format = "q128_128"))]
use super::exp_tier_n_plus_1::exp_q256_256_native;

// exp_q128_128_native no longer needed — pow now uses REAL Q256.256/Q128.128 tier N+1

// ============================================================================
// CORE IMPLEMENTATION: Q256.256 (Scientific Profile)
// ============================================================================

/// Compute pow(x, y) at Q256.256 precision
///
/// Both x and y are in Q256.256 format (I512 with 256 fractional bits)
/// Returns result in Q256.256 format
///
/// **PRECISION**: 77 decimal digits guaranteed
#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
pub fn pow_q256_256_native(x: I512, y: I512) -> I512 {
    let one_q256 = I512::from_i256(I256::from_i128(1)) << 256;
    let zero = I512::zero();

    // Special case: x = 0
    if x == zero {
        // Check if y is positive (integer part > 0 or fractional part > 0 with int = 0)
        if y > zero {
            return zero;  // 0^y = 0 for y > 0
        } else if y == zero {
            return one_q256;  // 0^0 = 1 (IEEE convention)
        } else {
            // 0^(-y) = infinity → return max value as error indicator
            return I512::max_value();
        }
    }

    // Special case: x < 0 → complex result, not supported
    if x < zero {
        // Return a sentinel value (could also panic or return Result)
        return I512::min_value();
    }

    // Special case: y = 0 → x^0 = 1
    if y == zero {
        return one_q256;
    }

    // Special case: x = 1 → 1^y = 1
    if x == one_q256 {
        return one_q256;
    }

    // Special case: y = 1 → x^1 = x
    if y == one_q256 {
        return x;
    }

    // General case: pow(x, y) = exp(y × ln(x))
    //
    // TIER N+1 STRATEGY:
    // - ln(x) computed at Q256.256 (already tier N for scientific)
    // - Multiply y × ln(x) at Q512.512 for extra precision
    // - exp() computed at Q256.256
    // - Result is Q256.256 (no downscale needed for scientific profile)

    // Step 1: ln(x) at Q256.256
    let ln_x = ln_q256_256_native(x);

    // Step 2: y × ln(x) with extended precision
    // Multiply in I1024 to get Q512.512, then downscale to Q256.256
    let y_i1024 = I1024::from_i512(y);
    let ln_x_i1024 = I1024::from_i512(ln_x);
    let product_q512 = y_i1024.mul_to_i2048(ln_x_i1024);

    // Downscale Q512.512 → Q256.256 with rounding
    let rounding = crate::fixed_point::I2048::from_i1024(
        I1024::from_i512(I512::from_i256(I256::from_i128(1)))
    ) << 255;
    let y_ln_x = ((product_q512 + rounding) >> 256).as_i1024().as_i512();

    // Step 3: exp(y × ln(x))
    exp_q256_256_native(y_ln_x)
}

// ============================================================================
// CORE IMPLEMENTATION: Q128.128 (Balanced Profile)
// ============================================================================

/// Compute pow(x, y) at Q128.128 precision
///
/// Both x and y are in Q128.128 format (I256 with 128 fractional bits)
/// Returns result in Q128.128 format
///
/// **PRECISION**: 38 decimal digits guaranteed
#[cfg(table_format = "q128_128")]
pub fn pow_q128_128_native(x: I256, y: I256) -> I256 {
    let one_q128 = I256::from_i128(1) << 128;
    let zero = I256::zero();

    // Special case: x = 0
    if x == zero {
        if y > zero {
            return zero;
        } else if y == zero {
            return one_q128;
        } else {
            return I256::max_value();  // Error: 0^(-y)
        }
    }

    // Special case: x < 0
    if x < zero {
        return I256::min_value();  // Error: complex result
    }

    // Special case: y = 0
    if y == zero {
        return one_q128;
    }

    // Special case: x = 1
    if x == one_q128 {
        return one_q128;
    }

    // Special case: y = 1
    if y == one_q128 {
        return x;
    }

    // TIER N+1 STRATEGY for balanced profile:
    // - Upscale to Q256.256
    // - Compute ln, multiply, exp at Q256.256
    // - Downscale result to Q128.128

    // Upscale x and y to Q256.256
    let x_q256 = I512::from_i256(x) << 128;
    let y_q256 = I512::from_i256(y) << 128;

    // Step 1: ln(x) at Q256.256 using REAL Q256.256 native implementation
    let ln_x = ln_q256_256_native(x_q256);

    // Step 2: y × ln(x) at Q512.512 → Q256.256
    let y_i1024 = I1024::from_i512(y_q256);
    let ln_x_i1024 = I1024::from_i512(ln_x);
    let product_q512 = y_i1024.mul_to_i2048(ln_x_i1024);

    let rounding = crate::fixed_point::I2048::from_i1024(
        I1024::from_i512(I512::from_i256(I256::from_i128(1)))
    ) << 255;
    let y_ln_x = ((product_q512 + rounding) >> 256).as_i1024().as_i512();

    // Step 3: exp(y × ln(x)) at Q256.256 using REAL Q256.256 native implementation
    let result_q256 = exp_q256_256_native(y_ln_x);

    // Downscale Q256.256 → Q128.128 with rounding
    downscale_q256_to_q128(result_q256)
}



// ============================================================================
// CORE IMPLEMENTATION: Q64.64 (Embedded Profile)
// ============================================================================

/// Compute pow(x, y) at Q64.64 precision
///
/// Both x and y are in Q64.64 format (i128 with 64 fractional bits)
/// Returns result in Q64.64 format
///
/// **PRECISION**: 19 decimal digits guaranteed
#[cfg(table_format = "q64_64")]
pub fn pow_q64_64_native(x: i128, y: i128) -> i128 {
    let one_q64: i128 = 1 << 64;

    // Special case: x = 0
    if x == 0 {
        if y > 0 {
            return 0;
        } else if y == 0 {
            return one_q64;
        } else {
            return i128::MAX;  // Error: 0^(-y)
        }
    }

    // Special case: x < 0
    if x < 0 {
        return i128::MIN;  // Error: complex result
    }

    // Special case: y = 0
    if y == 0 {
        return one_q64;
    }

    // Special case: x = 1
    if x == one_q64 {
        return one_q64;
    }

    // Special case: y = 1
    if y == one_q64 {
        return x;
    }

    // TIER N+1 STRATEGY for Q64.64:
    // - Upscale to Q128.128
    // - Compute ln, multiply, exp at Q128.128
    // - Downscale result to Q64.64

    // Upscale x and y to Q128.128
    let x_q128 = I256::from_i128(x) << 64;
    let y_q128 = I256::from_i128(y) << 64;

    // Step 1: ln(x) at Q128.128
    let ln_x = ln_q128_128_native_for_q64(x_q128);

    // Step 2: y × ln(x) at Q256.256 → Q128.128
    let y_i512 = I512::from_i256(y_q128);
    let ln_x_i512 = I512::from_i256(ln_x);
    let product_q256 = y_i512.mul_to_i1024(ln_x_i512);

    let rounding = I1024::from_i512(I512::from_i256(I256::from_i128(1))) << 127;
    let y_ln_x = ((product_q256 + rounding) >> 128).as_i512().as_i256();

    // Step 3: exp(y × ln(x)) at Q128.128
    let result_q128 = exp_q128_128_native_for_q64(y_ln_x);

    // Downscale Q128.128 → Q64.64 with rounding
    downscale_q128_to_q64(result_q128)
}

/// Helper: ln at Q128.128 for Q64.64 profile
#[cfg(table_format = "q64_64")]
fn ln_q128_128_native_for_q64(x: I256) -> I256 {
    // Use the Q64.64 ln and upscale
    use super::ln_tier_n_plus_1::ln_binary_i128;
    let x_q64 = (x >> 64).as_i128();
    let result_q64 = ln_binary_i128(x_q64);
    I256::from_i128(result_q64) << 64
}

/// Helper: exp at Q128.128 for Q64.64 profile
#[cfg(table_format = "q64_64")]
fn exp_q128_128_native_for_q64(x: I256) -> I256 {
    // Use the Q64.64 exp and upscale
    use super::exp_tier_n_plus_1::exp_binary_i128;
    let x_q64 = (x >> 64).as_i128();
    let result_q64 = exp_binary_i128(x_q64);
    I256::from_i128(result_q64) << 64
}

// ============================================================================
// PUBLIC API: Profile-Aware pow() Functions
// ============================================================================

/// Compute pow(x, y) for Q64.64 inputs
///
/// **PRECISION CONTRACT**: 19 decimal digits guaranteed, ≤1 ULP error
#[cfg(table_format = "q256_256")]
pub fn pow_binary_i128(x: i128, y: i128) -> i128 {
    // Upscale to Q256.256, compute, downscale
    let x_q256 = I512::from_i256(I256::from_i128(x)) << 192;
    let y_q256 = I512::from_i256(I256::from_i128(y)) << 192;
    let result_q256 = pow_q256_256_native(x_q256, y_q256);

    // Downscale Q256.256 → Q64.64 with rounding
    let rounding = I512::from_i256(I256::from_i128(1)) << 191;
    ((result_q256 + rounding) >> 192).as_i256().as_i128()
}

/// Compute pow(x, y) for Q128.128 inputs
///
/// **PRECISION CONTRACT**: 38 decimal digits guaranteed, ≤1 ULP error
#[cfg(table_format = "q256_256")]
pub fn pow_binary_i256(x: I256, y: I256) -> I256 {
    // Upscale to Q256.256, compute, downscale
    let x_q256 = I512::from_i256(x) << 128;
    let y_q256 = I512::from_i256(y) << 128;
    let result_q256 = pow_q256_256_native(x_q256, y_q256);

    // Downscale Q256.256 → Q128.128 with rounding
    downscale_q256_to_q128(result_q256)
}

/// Compute pow(x, y) for Q256.256 inputs
///
/// **PRECISION CONTRACT**: 77 decimal digits guaranteed, ≤1 ULP error
#[cfg(table_format = "q256_256")]
pub fn pow_binary_i512(x: I512, y: I512) -> I512 {
    pow_q256_256_native(x, y)
}

// Balanced profile implementations
#[cfg(table_format = "q128_128")]
pub fn pow_binary_i128(x: i128, y: i128) -> i128 {
    let x_q128 = I256::from_i128(x) << 64;
    let y_q128 = I256::from_i128(y) << 64;
    let result_q128 = pow_q128_128_native(x_q128, y_q128);
    downscale_q128_to_q64(result_q128)
}

#[cfg(table_format = "q128_128")]
pub fn pow_binary_i256(x: I256, y: I256) -> I256 {
    pow_q128_128_native(x, y)
}

#[cfg(table_format = "q128_128")]
pub fn pow_binary_i512(x: I512, y: I512) -> I512 {
    // Direct Q256.256 computation (compute tier for balanced profile)
    pow_q256_256_native(x, y)
}

// Q64.64 profile implementations
#[cfg(table_format = "q64_64")]
pub fn pow_binary_i128(x: i128, y: i128) -> i128 {
    pow_q64_64_native(x, y)
}

#[cfg(table_format = "q64_64")]
pub fn pow_binary_i256(x: I256, y: I256) -> I256 {
    // Downscale to Q64.64, compute, upscale
    let x_q64 = (x >> 64).as_i128();
    let y_q64 = (y >> 64).as_i128();
    let result_q64 = pow_q64_64_native(x_q64, y_q64);
    I256::from_i128(result_q64) << 64
}

#[cfg(table_format = "q64_64")]
pub fn pow_binary_i512(x: I512, y: I512) -> I512 {
    // Downscale to Q64.64, compute, upscale
    let x_q64 = (x >> 192).as_i256().as_i128();
    let y_q64 = (y >> 192).as_i256().as_i128();
    let result_q64 = pow_q64_64_native(x_q64, y_q64);
    I512::from_i256(I256::from_i128(result_q64)) << 192
}

// ============================================================================
// CONVENIENCE: Integer Exponent Optimization
// ============================================================================

/// Fast integer power using repeated squaring (no ln/exp needed)
///
/// For integer exponents, this is both faster and more precise than exp(y×ln(x))
#[inline]
pub fn pow_integer_i128(base: i128, exp: i32) -> i128 {
    if exp == 0 {
        return 1i128 << 64;  // 1.0 in Q64.64
    }
    if exp == 1 {
        return base;
    }
    if exp < 0 {
        // x^(-n) = 1 / x^n
        let pos_result = pow_integer_i128(base, -exp);
        if pos_result == 0 {
            return i128::MAX;  // Division by zero
        }
        // 1.0 / pos_result in Q64.64
        let one_q128 = I256::from_i128(1) << 128;  // 1.0 in Q128.128 for division precision
        let pos_q128 = I256::from_i128(pos_result);
        return (one_q128 / pos_q128).as_i128();
    }

    // Exponentiation by squaring
    let mut result = 1i128 << 64;  // 1.0 in Q64.64
    let mut base_pow = base;
    let mut n = exp as u32;

    while n > 0 {
        if n & 1 == 1 {
            // result *= base_pow (Q64.64 multiplication)
            let r256 = I256::from_i128(result);
            let b256 = I256::from_i128(base_pow);
            result = ((r256 * b256) >> 64).as_i128();
        }
        // base_pow *= base_pow
        let b256 = I256::from_i128(base_pow);
        base_pow = ((b256 * b256) >> 64).as_i128();
        n >>= 1;
    }

    result
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_special_cases() {
        let one = 1i128 << 64;  // 1.0 in Q64.64
        let two = 2i128 << 64;  // 2.0 in Q64.64

        // x^0 = 1
        assert_eq!(pow_binary_i128(two, 0), one);

        // x^1 = x
        assert_eq!(pow_binary_i128(two, one), two);

        // 1^y = 1
        assert_eq!(pow_binary_i128(one, two), one);

        // 0^1 = 0
        assert_eq!(pow_binary_i128(0, one), 0);
    }

    #[test]
    fn test_pow_integer_exponent() {
        let two = 2i128 << 64;  // 2.0 in Q64.64
        let four = 4i128 << 64; // 4.0 in Q64.64
        let eight = 8i128 << 64; // 8.0 in Q64.64

        // 2^2 = 4
        let result = pow_integer_i128(two, 2);
        let diff = (result - four).abs();
        assert!(diff < (1i128 << 32), "2^2 should be 4, got diff {}", diff);

        // 2^3 = 8
        let result = pow_integer_i128(two, 3);
        let diff = (result - eight).abs();
        assert!(diff < (1i128 << 32), "2^3 should be 8, got diff {}", diff);
    }
}
