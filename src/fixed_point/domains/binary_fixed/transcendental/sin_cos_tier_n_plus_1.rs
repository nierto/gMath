// ============================================================================
// TIER N+1 SINE AND COSINE
// ============================================================================
//
// **PRINCIPLE**: Compute at native tier using build.rs-generated constants
//
// **ALGORITHM**: Cody-Waite range reduction + Taylor series
//   1. Reduce x into [-π/4, π/4] via k = round(x * 2/π)
//   2. Compute sin(r) and cos(r) using Taylor/Horner polynomials
//   3. Apply quadrant correction based on k mod 4
//
// **PRECISION**: Uses build.rs-generated constants (77-digit π) and Taylor coefficients
//
// **PROFILES**:
//   - Q64.64 (i128): 11 Taylor terms → 19 correct digits
//   - Q128.128 (I256): 21 Taylor terms → 38 correct digits
//   - Q256.256 (I512): 41 Taylor terms → 77 correct digits
//
// ============================================================================

#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::i1024::I1024;
#[allow(unused_imports)]
use crate::fixed_point::I2048;

// Include trig constants (pre-built or regenerated via --features rebuild-tables).
// Not all constants are used on every profile — #[allow(dead_code)] prevents warnings.
#[allow(dead_code)]
mod sincos_trig_consts {
    #[cfg(feature = "rebuild-tables")]
    include!(concat!(env!("OUT_DIR"), "/trig_constants.rs"));
    #[cfg(not(feature = "rebuild-tables"))]
    include!("../../../../generated_tables/trig_constants.rs");
}
#[allow(unused_imports)]
pub use sincos_trig_consts::*;

// Reuse upscale/downscale helpers from exp_tier_n_plus_1
#[allow(unused_imports)]
use super::exp_tier_n_plus_1::{
    upscale_q64_to_q128, upscale_q128_to_q256, upscale_q64_to_q256,
    downscale_q128_to_q64, downscale_q256_to_q128, downscale_q256_to_q64,
};

// ============================================================================
// Q64.64 IMPLEMENTATION (i128)
// ============================================================================

/// Sine function in Q64.64 format
///
/// **INPUT**: i128 in Q64.64 (angle in radians)
/// **OUTPUT**: i128 in Q64.64 (sin value in [-1, 1])
#[cfg(table_format = "q64_64")]
pub fn sin_binary_i128(x: i128) -> i128 {
    sin_q64_64(x)
}

/// Cosine function in Q64.64 format
#[cfg(table_format = "q64_64")]
pub fn cos_binary_i128(x: i128) -> i128 {
    cos_q64_64(x)
}

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
pub fn sin_binary_i256(x: I256) -> I256 {
    // Tier N+1: compute sin at Q128.128 natively
    sin_q128_128_for_embedded(x)
}

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
pub fn cos_binary_i256(x: I256) -> I256 {
    // Tier N+1: compute cos at Q128.128 natively
    cos_q128_128_for_embedded(x)
}

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
pub fn sin_binary_i512(x: I512) -> I512 {
    // Downscale to Q128.128, compute, upscale back to Q256.256
    let x_q128 = (x >> 128).as_i256();
    let result_q128 = sin_q128_128_for_embedded(x_q128);
    I512::from_i256(result_q128) << 128
}

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
pub fn cos_binary_i512(x: I512) -> I512 {
    // Downscale to Q128.128, compute, upscale back to Q256.256
    let x_q128 = (x >> 128).as_i256();
    let result_q128 = cos_q128_128_for_embedded(x_q128);
    I512::from_i256(result_q128) << 128
}

/// Core Q64.64 sin/cos implementation
#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn sin_q64_64(x: i128) -> i128 {
    let (sin_val, _cos_val) = sincos_q64_64(x);
    sin_val
}

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn cos_q64_64(x: i128) -> i128 {
    let (_sin_val, cos_val) = sincos_q64_64(x);
    cos_val
}

/// Compute both sin and cos simultaneously (shared range reduction)
#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn sincos_q64_64(x: i128) -> (i128, i128) {
    let one_q64: i128 = 1i128 << 64;

    // Handle x = 0
    if x == 0 {
        return (0, one_q64);
    }

    // Step 1: Range reduction — k = round(x * 2/π)
    // Multiply x by 2/π in Q64.64, then round to nearest integer
    let two_over_pi = TWO_OVER_PI_Q64;
    // x * (2/π) in Q64.64: (x * two_over_pi) >> 64
    let product = {
        let x_wide = I256::from_i128(x);
        let twopi_wide = I256::from_i128(two_over_pi);
        let prod = x_wide * twopi_wide;
        // Result is in Q128.128 format (64+64 frac bits), shift right 64 to get Q64.64
        (prod >> 64).as_i128()
    };

    // Round to nearest integer: k = floor(product + 0.5) = floor((product + half_q64) >> 64)
    let half_q64 = 1i128 << 63; // 0.5 in Q64.64
    let k = (product + half_q64) >> 64;

    // Step 2: Compute remainder r = x - k * (π/2) using extended precision
    // Use two-part π/2 for Cody-Waite: C1 + C2 = π/2
    // For Q64.64: single step is sufficient given 64-bit precision
    let k_times_pi_half = {
        let k_wide = I256::from_i128(k);
        let pi_half_wide = I256::from_i128(PI_HALF_Q64);
        (k_wide * pi_half_wide).as_i128()
    };
    let r = x - k_times_pi_half;

    // Step 3: Compute sin(r) and cos(r) via Taylor series
    // sin(r) = r - r³/3! + r⁵/5! - r⁷/7! + ...
    // cos(r) = 1 - r²/2! + r⁴/4! - r⁶/6! + ...
    let sin_r = taylor_sin_q64_64(r);
    let cos_r = taylor_cos_q64_64(r);

    // Step 4: Quadrant correction based on k mod 4
    let quadrant = ((k % 4) + 4) % 4; // Handle negative k
    match quadrant {
        0 => (sin_r, cos_r),          // sin(x), cos(x)
        1 => (cos_r, -sin_r),         // cos(r), -sin(r)
        2 => (-sin_r, -cos_r),        // -sin(r), -cos(r)
        3 => (-cos_r, sin_r),         // -cos(r), sin(r)
        _ => unreachable!(),
    }
}

/// Taylor series for sin(r) in Q64.64 format
/// sin(r) = Σ (-1)^k * r^(2k+1) / (2k+1)!
#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn taylor_sin_q64_64(r: i128) -> i128 {
    // Horner form: sin(r) = r * (1 - r²/6 * (1 - r²/20 * (1 - r²/42 * ...)))
    let r_sq = {
        let r_wide = I256::from_i128(r);
        ((r_wide * r_wide) >> 64).as_i128()
    };

    // Use coefficients from build.rs: SIN_COEFFS_Q64[k] = 1/(2k+1)!
    // Compute: result = r * (c[0] - r² * (c[1] - r² * (c[2] - ...)))
    // Start from innermost term
    let num_terms = 11;
    let mut result: i128 = 0;

    // Horner's method from inside out
    for k in (1..num_terms).rev() {
        // result = SIN_COEFFS_Q64[k] - r² * result / one
        let coeff = SIN_COEFFS_Q64[k];
        let term = {
            let r_sq_wide = I256::from_i128(r_sq);
            let result_wide = I256::from_i128(result);
            ((r_sq_wide * result_wide) >> 64).as_i128()
        };
        // Horner: all subtractions — the alternating series signs are
        // inherent in the Horner nesting: c0 - s*(c1 - s*(c2 - ...))
        result = coeff - term;
    }

    // Final: multiply by r and subtract from leading term
    // sin(r) = r * (SIN_COEFFS_Q64[0] - r² * result)
    let inner = {
        let r_sq_wide = I256::from_i128(r_sq);
        let result_wide = I256::from_i128(result);
        ((r_sq_wide * result_wide) >> 64).as_i128()
    };
    let poly = SIN_COEFFS_Q64[0] - inner;

    // Multiply by r
    let sin_val = {
        let r_wide = I256::from_i128(r);
        let poly_wide = I256::from_i128(poly);
        ((r_wide * poly_wide) >> 64).as_i128()
    };

    sin_val
}

/// Taylor series for cos(r) in Q64.64 format
/// cos(r) = Σ (-1)^k * r^(2k) / (2k)!
#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn taylor_cos_q64_64(r: i128) -> i128 {
    let r_sq = {
        let r_wide = I256::from_i128(r);
        ((r_wide * r_wide) >> 64).as_i128()
    };

    // Horner's method: cos(r) = c[0] - r²*(c[1] - r²*(c[2] - ...))
    let num_terms = 11;
    let mut result: i128 = 0;

    for k in (1..num_terms).rev() {
        let coeff = COS_COEFFS_Q64[k];
        let term = {
            let r_sq_wide = I256::from_i128(r_sq);
            let result_wide = I256::from_i128(result);
            ((r_sq_wide * result_wide) >> 64).as_i128()
        };
        result = coeff - term;
    }

    // Final: cos(r) = COS_COEFFS_Q64[0] - r² * result
    let inner = {
        let r_sq_wide = I256::from_i128(r_sq);
        let result_wide = I256::from_i128(result);
        ((r_sq_wide * result_wide) >> 64).as_i128()
    };
    COS_COEFFS_Q64[0] - inner
}

// ============================================================================
// Q128.128 IMPLEMENTATION (I256)
// ============================================================================

#[cfg(table_format = "q128_128")]
pub fn sin_binary_i128(x: i128) -> i128 {
    let x_q128 = upscale_q64_to_q128(x);
    downscale_q128_to_q64(sin_q128_128(x_q128))
}

#[cfg(table_format = "q128_128")]
pub fn cos_binary_i128(x: i128) -> i128 {
    let x_q128 = upscale_q64_to_q128(x);
    downscale_q128_to_q64(cos_q128_128(x_q128))
}

#[cfg(table_format = "q128_128")]
pub fn sin_binary_i256(x: I256) -> I256 {
    // UGOD Tier N+1: Q128.128 storage → compute at Q256.256 → downscale
    let x_q256 = upscale_q128_to_q256(x);
    downscale_q256_to_q128(sin_q256_256_for_balanced(x_q256))
}

#[cfg(table_format = "q128_128")]
pub fn cos_binary_i256(x: I256) -> I256 {
    // UGOD Tier N+1: Q128.128 storage → compute at Q256.256 → downscale
    let x_q256 = upscale_q128_to_q256(x);
    downscale_q256_to_q128(cos_q256_256_for_balanced(x_q256))
}

#[cfg(table_format = "q128_128")]
pub fn sin_binary_i512(x: I512) -> I512 {
    // Direct Q256.256 computation (compute tier for balanced profile)
    sin_q256_256_for_balanced(x)
}

#[cfg(table_format = "q128_128")]
pub fn cos_binary_i512(x: I512) -> I512 {
    // Direct Q256.256 computation (compute tier for balanced profile)
    cos_q256_256_for_balanced(x)
}

#[cfg(table_format = "q128_128")]
fn sin_q128_128(x: I256) -> I256 {
    let (sin_val, _) = sincos_q128_128(x);
    sin_val
}

#[cfg(table_format = "q128_128")]
fn cos_q128_128(x: I256) -> I256 {
    let (_, cos_val) = sincos_q128_128(x);
    cos_val
}

#[cfg(table_format = "q128_128")]
fn sincos_q128_128(x: I256) -> (I256, I256) {
    let one_q128 = I256::from_i128(1) << 128;

    if x == I256::zero() {
        return (I256::zero(), one_q128);
    }

    // Range reduction: k = round(x * 2/π)
    let two_over_pi = I256::from_words(TWO_OVER_PI_Q128);
    let product = {
        let x_wide = I512::from_i256(x);
        let twopi_wide = I512::from_i256(two_over_pi);
        let prod = x_wide * twopi_wide;
        (prod >> 128).as_i256()
    };

    let half_q128 = I256::from_i128(1) << 127;
    let k_i256 = (product + half_q128) >> 128;
    let k = k_i256.as_i128();

    // Remainder: r = x - k * π/2
    let pi_half = I256::from_words(PI_HALF_Q128);
    let k_times_pi_half = {
        let k_wide = I512::from_i256(I256::from_i128(k));
        let pi_half_wide = I512::from_i256(pi_half);
        (k_wide * pi_half_wide).as_i256()
    };
    let r = x - k_times_pi_half;

    // Taylor series
    let sin_r = taylor_sin_q128_128(r);
    let cos_r = taylor_cos_q128_128(r);

    // Quadrant correction
    let quadrant = ((k % 4) + 4) % 4;
    match quadrant {
        0 => (sin_r, cos_r),
        1 => (cos_r, I256::zero() - sin_r),
        2 => (I256::zero() - sin_r, I256::zero() - cos_r),
        3 => (I256::zero() - cos_r, sin_r),
        _ => unreachable!(),
    }
}

#[cfg(table_format = "q128_128")]
fn taylor_sin_q128_128(r: I256) -> I256 {
    let r_sq = {
        let r_wide = I512::from_i256(r);
        ((r_wide * r_wide) >> 128).as_i256()
    };

    let num_terms = 21;
    let mut result = I256::zero();

    for k in (1..num_terms).rev() {
        let coeff = I256::from_words(SIN_COEFFS_Q128[k]);
        let term = {
            let r_sq_wide = I512::from_i256(r_sq);
            let result_wide = I512::from_i256(result);
            ((r_sq_wide * result_wide) >> 128).as_i256()
        };
        result = coeff - term;
    }

    let inner = {
        let r_sq_wide = I512::from_i256(r_sq);
        let result_wide = I512::from_i256(result);
        ((r_sq_wide * result_wide) >> 128).as_i256()
    };
    let poly = I256::from_words(SIN_COEFFS_Q128[0]) - inner;

    let r_wide = I512::from_i256(r);
    let poly_wide = I512::from_i256(poly);
    ((r_wide * poly_wide) >> 128).as_i256()
}

#[cfg(table_format = "q128_128")]
fn taylor_cos_q128_128(r: I256) -> I256 {
    let r_sq = {
        let r_wide = I512::from_i256(r);
        ((r_wide * r_wide) >> 128).as_i256()
    };

    let num_terms = 21;
    let mut result = I256::zero();

    for k in (1..num_terms).rev() {
        let coeff = I256::from_words(COS_COEFFS_Q128[k]);
        let term = {
            let r_sq_wide = I512::from_i256(r_sq);
            let result_wide = I512::from_i256(result);
            ((r_sq_wide * result_wide) >> 128).as_i256()
        };
        result = coeff - term;
    }

    let inner = {
        let r_sq_wide = I512::from_i256(r_sq);
        let result_wide = I512::from_i256(result);
        ((r_sq_wide * result_wide) >> 128).as_i256()
    };
    I256::from_words(COS_COEFFS_Q128[0]) - inner
}

// ============================================================================
// Q256.256 IMPLEMENTATION (I512)
// ============================================================================

#[cfg(table_format = "q256_256")]
pub fn sin_binary_i128(x: i128) -> i128 {
    let x_q256 = upscale_q64_to_q256(x);
    downscale_q256_to_q64(sin_q256_256(x_q256))
}

#[cfg(table_format = "q256_256")]
pub fn cos_binary_i128(x: i128) -> i128 {
    let x_q256 = upscale_q64_to_q256(x);
    downscale_q256_to_q64(cos_q256_256(x_q256))
}

#[cfg(table_format = "q256_256")]
pub fn sin_binary_i256(x: I256) -> I256 {
    let x_q256 = upscale_q128_to_q256(x);
    downscale_q256_to_q128(sin_q256_256(x_q256))
}

#[cfg(table_format = "q256_256")]
pub fn cos_binary_i256(x: I256) -> I256 {
    let x_q256 = upscale_q128_to_q256(x);
    downscale_q256_to_q128(cos_q256_256(x_q256))
}

#[cfg(table_format = "q256_256")]
pub fn sin_binary_i512(x: I512) -> I512 {
    sin_q256_256(x)
}

#[cfg(table_format = "q256_256")]
pub fn cos_binary_i512(x: I512) -> I512 {
    cos_q256_256(x)
}

#[cfg(table_format = "q256_256")]
fn sin_q256_256(x: I512) -> I512 {
    let (sin_val, _) = sincos_q256_256(x);
    sin_val
}

#[cfg(table_format = "q256_256")]
fn cos_q256_256(x: I512) -> I512 {
    let (_, cos_val) = sincos_q256_256(x);
    cos_val
}

#[cfg(table_format = "q256_256")]
fn sincos_q256_256(x: I512) -> (I512, I512) {
    let one_q256 = I512::from_i128(1) << 256;

    if x == I512::zero() {
        return (I512::zero(), one_q256);
    }

    // Range reduction: k = round(x * 2/π)
    let two_over_pi = I512::from_words(TWO_OVER_PI_Q256);
    let product = {
        let x_wide = I1024::from_i512(x);
        let twopi_wide = I1024::from_i512(two_over_pi);
        let prod = x_wide * twopi_wide;
        (prod >> 256).as_i512()
    };

    let half_q256 = I512::from_i128(1) << 255;
    let k_i512 = (product + half_q256) >> 256;
    let k = k_i512.as_i256().as_i128();

    // Remainder: r = x - k * π/2
    let pi_half = I512::from_words(PI_HALF_Q256);
    let k_times_pi_half = {
        let k_wide = I1024::from_i512(I512::from_i128(k));
        let pi_half_wide = I1024::from_i512(pi_half);
        (k_wide * pi_half_wide).as_i512()
    };
    let r = x - k_times_pi_half;

    // Taylor series
    let sin_r = taylor_sin_q256_256(r);
    let cos_r = taylor_cos_q256_256(r);

    // Quadrant correction
    let quadrant = ((k % 4) + 4) % 4;
    match quadrant {
        0 => (sin_r, cos_r),
        1 => (cos_r, I512::zero() - sin_r),
        2 => (I512::zero() - sin_r, I512::zero() - cos_r),
        3 => (I512::zero() - cos_r, sin_r),
        _ => unreachable!(),
    }
}

#[cfg(table_format = "q256_256")]
fn taylor_sin_q256_256(r: I512) -> I512 {
    let r_sq = {
        let r_wide = I1024::from_i512(r);
        ((r_wide * r_wide) >> 256).as_i512()
    };

    let num_terms = 41;
    let mut result = I512::zero();

    for k in (1..num_terms).rev() {
        let coeff = I512::from_words(SIN_COEFFS_Q256[k]);
        let term = {
            let r_sq_wide = I1024::from_i512(r_sq);
            let result_wide = I1024::from_i512(result);
            ((r_sq_wide * result_wide) >> 256).as_i512()
        };
        result = coeff - term;
    }

    let inner = {
        let r_sq_wide = I1024::from_i512(r_sq);
        let result_wide = I1024::from_i512(result);
        ((r_sq_wide * result_wide) >> 256).as_i512()
    };
    let poly = I512::from_words(SIN_COEFFS_Q256[0]) - inner;

    let r_wide = I1024::from_i512(r);
    let poly_wide = I1024::from_i512(poly);
    ((r_wide * poly_wide) >> 256).as_i512()
}

#[cfg(table_format = "q256_256")]
fn taylor_cos_q256_256(r: I512) -> I512 {
    let r_sq = {
        let r_wide = I1024::from_i512(r);
        ((r_wide * r_wide) >> 256).as_i512()
    };

    let num_terms = 41;
    let mut result = I512::zero();

    for k in (1..num_terms).rev() {
        let coeff = I512::from_words(COS_COEFFS_Q256[k]);
        let term = {
            let r_sq_wide = I1024::from_i512(r_sq);
            let result_wide = I1024::from_i512(result);
            ((r_sq_wide * result_wide) >> 256).as_i512()
        };
        result = coeff - term;
    }

    let inner = {
        let r_sq_wide = I1024::from_i512(r_sq);
        let result_wide = I1024::from_i512(result);
        ((r_sq_wide * result_wide) >> 256).as_i512()
    };
    I512::from_words(COS_COEFFS_Q256[0]) - inner
}

// ============================================================================
// PUBLIC CONSTANT ACCESSORS (for use by StackEvaluator)
// ============================================================================

/// Returns π/2 in Q64.64 format
#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
pub fn pi_half_i128() -> i128 {
    PI_HALF_Q64
}

/// Returns π/2 in Q128.128 format
#[cfg(table_format = "q128_128")]
pub fn pi_half_i256() -> I256 {
    I256::from_words(PI_HALF_Q128)
}

/// Returns π/2 in Q256.256 format
#[cfg(table_format = "q256_256")]
pub fn pi_half_i512() -> I512 {
    I512::from_words(PI_HALF_Q256)
}

/// Returns π/2 in Q512.512 format (compute tier for scientific profile)
#[cfg(table_format = "q256_256")]
pub fn pi_half_i1024() -> I1024 {
    I1024::from_words(PI_HALF_Q512)
}

// ============================================================================
// Q512.512 IMPLEMENTATION (I1024) — COMPUTE TIER FOR SCIENTIFIC PROFILE
// ============================================================================
//
// This implementation is ONLY compiled under q256_256 (scientific profile).
// It serves as the tier N+1 compute tier for Q256.256 storage.

#[cfg(table_format = "q256_256")]
pub fn sin_binary_i1024(x: I1024) -> I1024 {
    sincos_q512_512(x).0
}

#[cfg(table_format = "q256_256")]
pub fn cos_binary_i1024(x: I1024) -> I1024 {
    sincos_q512_512(x).1
}

#[cfg(table_format = "q256_256")]
fn sincos_q512_512(x: I1024) -> (I1024, I1024) {
    use super::ln_tier_n_plus_1::multiply_i1024_q512_512;

    let one_q512 = I1024::from_i128(1) << 512;

    if x == I1024::zero() {
        return (I1024::zero(), one_q512);
    }

    // Range reduction: k = round(x * 2/π)
    let two_over_pi = I1024::from_words(TWO_OVER_PI_Q512);
    // SIGNED Q512.512 multiply (mul_to_i2048 is unsigned — x can be negative)
    let product = multiply_i1024_q512_512(x, two_over_pi);

    let half_q512 = I1024::from_i128(1) << 511;
    let k_i1024 = (product + half_q512) >> 512;
    let k = k_i1024.as_i512().as_i256().as_i128();

    // Remainder: r = x - k * π/2
    let pi_half = I1024::from_words(PI_HALF_Q512);
    // k is a small integer — use signed I1024 * operator (safe: |k*pi_half| << 2^1023)
    let k_times_pi_half = I1024::from_i128(k) * pi_half;
    let r = x - k_times_pi_half;

    // Taylor series
    let sin_r = taylor_sin_q512_512(r);
    let cos_r = taylor_cos_q512_512(r);

    // Quadrant correction
    let quadrant = ((k % 4) + 4) % 4;
    match quadrant {
        0 => (sin_r, cos_r),
        1 => (cos_r, I1024::zero() - sin_r),
        2 => (I1024::zero() - sin_r, I1024::zero() - cos_r),
        3 => (I1024::zero() - cos_r, sin_r),
        _ => unreachable!(),
    }
}

#[cfg(table_format = "q256_256")]
fn taylor_sin_q512_512(r: I1024) -> I1024 {
    use super::ln_tier_n_plus_1::multiply_i1024_q512_512;

    // SIGNED multiply: r can be negative, r_sq must be positive
    let r_sq = multiply_i1024_q512_512(r, r);

    let num_terms = 65;
    let mut result = I1024::zero();

    for k in (1..num_terms).rev() {
        let coeff = I1024::from_words(SIN_COEFFS_Q512[k]);
        // SIGNED multiply: result can be negative in Horner scheme
        let term = multiply_i1024_q512_512(r_sq, result);
        result = coeff - term;
    }

    let inner = multiply_i1024_q512_512(r_sq, result);
    let poly = I1024::from_words(SIN_COEFFS_Q512[0]) - inner;

    // SIGNED multiply: r can be negative
    multiply_i1024_q512_512(r, poly)
}

#[cfg(table_format = "q256_256")]
fn taylor_cos_q512_512(r: I1024) -> I1024 {
    use super::ln_tier_n_plus_1::multiply_i1024_q512_512;

    // SIGNED multiply: r can be negative, r_sq must be positive
    let r_sq = multiply_i1024_q512_512(r, r);

    let num_terms = 65;
    let mut result = I1024::zero();

    for k in (1..num_terms).rev() {
        let coeff = I1024::from_words(COS_COEFFS_Q512[k]);
        // SIGNED multiply: result can be negative in Horner scheme
        let term = multiply_i1024_q512_512(r_sq, result);
        result = coeff - term;
    }

    let inner = multiply_i1024_q512_512(r_sq, result);
    I1024::from_words(COS_COEFFS_Q512[0]) - inner
}

// ============================================================================
// COMPUTE-TIER DISPATCH FUNCTIONS
// ============================================================================
//
// These dispatch to the appropriate tier based on the compile-time profile.
// Used by StackEvaluator to call sin/cos at the compute tier (tier N+1).

/// Compute sin at compute tier (tier N+1)
#[cfg(table_format = "q256_256")]
pub fn sin_compute_tier_i1024(x: I1024) -> I1024 {
    sin_binary_i1024(x)
}

/// Compute cos at compute tier (tier N+1)
#[cfg(table_format = "q256_256")]
pub fn cos_compute_tier_i1024(x: I1024) -> I1024 {
    cos_binary_i1024(x)
}

/// Compute sin at compute tier for balanced profile (Q256.256 on I512)
#[cfg(table_format = "q128_128")]
pub fn sin_compute_tier_i512(x: I512) -> I512 {
    sin_q256_256_for_balanced(x)
}

/// Compute cos at compute tier for balanced profile (Q256.256 on I512)
#[cfg(table_format = "q128_128")]
pub fn cos_compute_tier_i512(x: I512) -> I512 {
    cos_q256_256_for_balanced(x)
}

/// Compute sin at compute tier for embedded/performance profile (Q128.128 on I256)
#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
pub fn sin_compute_tier_i256(x: I256) -> I256 {
    sin_q128_128_for_embedded(x)
}

/// Compute cos at compute tier for embedded/performance profile (Q128.128 on I256)
#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
pub fn cos_compute_tier_i256(x: I256) -> I256 {
    cos_q128_128_for_embedded(x)
}

/// Fused sin+cos at compute tier (single shared range reduction).
/// Returns (sin, cos) — saves one range reduction vs calling sin + cos separately.
#[cfg(table_format = "q256_256")]
pub fn sincos_compute_tier_i1024(x: I1024) -> (I1024, I1024) {
    sincos_q512_512(x)
}

/// Fused sin+cos at compute tier for balanced profile (Q256.256 on I512).
#[cfg(table_format = "q128_128")]
pub fn sincos_compute_tier_i512(x: I512) -> (I512, I512) {
    sincos_q256_256_impl(x)
}

/// Fused sin+cos at compute tier for embedded profile (Q128.128 on I256).
#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
pub fn sincos_compute_tier_i256(x: I256) -> (I256, I256) {
    sincos_q128_128_impl(x)
}

// ============================================================================
// CROSS-PROFILE IMPLEMENTATIONS (tier N+1 compute functions)
// ============================================================================
//
// Q128.128 sin/cos available under q64_64 for tier N+1 computation

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn sin_q128_128_for_embedded(x: I256) -> I256 {
    sincos_q128_128_impl(x).0
}

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn cos_q128_128_for_embedded(x: I256) -> I256 {
    sincos_q128_128_impl(x).1
}

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn sincos_q128_128_impl(x: I256) -> (I256, I256) {
    let one_q128 = I256::from_i128(1) << 128;

    if x == I256::zero() {
        return (I256::zero(), one_q128);
    }

    // Range reduction: k = round(x * 2/π)
    let two_over_pi = I256::from_words(TWO_OVER_PI_Q128);
    let product = {
        let x_wide = I512::from_i256(x);
        let twopi_wide = I512::from_i256(two_over_pi);
        let prod = x_wide * twopi_wide;
        (prod >> 128).as_i256()
    };

    let half_q128 = I256::from_i128(1) << 127;
    let k_i256 = (product + half_q128) >> 128;
    let k = k_i256.as_i128();

    // Remainder: r = x - k * π/2
    let pi_half = I256::from_words(PI_HALF_Q128);
    let k_times_pi_half = {
        let k_wide = I512::from_i256(I256::from_i128(k));
        let pi_half_wide = I512::from_i256(pi_half);
        (k_wide * pi_half_wide).as_i256()
    };
    let r = x - k_times_pi_half;

    let sin_r = taylor_sin_q128_128_impl(r);
    let cos_r = taylor_cos_q128_128_impl(r);

    let quadrant = ((k % 4) + 4) % 4;
    match quadrant {
        0 => (sin_r, cos_r),
        1 => (cos_r, I256::zero() - sin_r),
        2 => (I256::zero() - sin_r, I256::zero() - cos_r),
        3 => (I256::zero() - cos_r, sin_r),
        _ => unreachable!(),
    }
}

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn taylor_sin_q128_128_impl(r: I256) -> I256 {
    let r_sq = {
        let r_wide = I512::from_i256(r);
        ((r_wide * r_wide) >> 128).as_i256()
    };

    let num_terms = 21;
    let mut result = I256::zero();

    for k in (1..num_terms).rev() {
        let coeff = I256::from_words(SIN_COEFFS_Q128[k]);
        let term = {
            let r_sq_wide = I512::from_i256(r_sq);
            let result_wide = I512::from_i256(result);
            ((r_sq_wide * result_wide) >> 128).as_i256()
        };
        result = coeff - term;
    }

    let inner = {
        let r_sq_wide = I512::from_i256(r_sq);
        let result_wide = I512::from_i256(result);
        ((r_sq_wide * result_wide) >> 128).as_i256()
    };
    let poly = I256::from_words(SIN_COEFFS_Q128[0]) - inner;

    let r_wide = I512::from_i256(r);
    let poly_wide = I512::from_i256(poly);
    ((r_wide * poly_wide) >> 128).as_i256()
}

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn taylor_cos_q128_128_impl(r: I256) -> I256 {
    let r_sq = {
        let r_wide = I512::from_i256(r);
        ((r_wide * r_wide) >> 128).as_i256()
    };

    let num_terms = 21;
    let mut result = I256::zero();

    for k in (1..num_terms).rev() {
        let coeff = I256::from_words(COS_COEFFS_Q128[k]);
        let term = {
            let r_sq_wide = I512::from_i256(r_sq);
            let result_wide = I512::from_i256(result);
            ((r_sq_wide * result_wide) >> 128).as_i256()
        };
        result = coeff - term;
    }

    let inner = {
        let r_sq_wide = I512::from_i256(r_sq);
        let result_wide = I512::from_i256(result);
        ((r_sq_wide * result_wide) >> 128).as_i256()
    };
    I256::from_words(COS_COEFFS_Q128[0]) - inner
}

// Q256.256 sin/cos available under q128_128 for tier N+1 computation

#[cfg(table_format = "q128_128")]
fn sin_q256_256_for_balanced(x: I512) -> I512 {
    sincos_q256_256_impl(x).0
}

#[cfg(table_format = "q128_128")]
fn cos_q256_256_for_balanced(x: I512) -> I512 {
    sincos_q256_256_impl(x).1
}

#[cfg(table_format = "q128_128")]
fn sincos_q256_256_impl(x: I512) -> (I512, I512) {
    let one_q256 = I512::from_i128(1) << 256;

    if x == I512::zero() {
        return (I512::zero(), one_q256);
    }

    // Range reduction: k = round(x * 2/π)
    let two_over_pi = I512::from_words(TWO_OVER_PI_Q256);
    let product = {
        let x_wide = I1024::from_i512(x);
        let twopi_wide = I1024::from_i512(two_over_pi);
        let prod = x_wide * twopi_wide;
        (prod >> 256).as_i512()
    };

    let half_q256 = I512::from_i128(1) << 255;
    let k_i512 = (product + half_q256) >> 256;
    let k = k_i512.as_i256().as_i128();

    // Remainder: r = x - k * π/2
    let pi_half = I512::from_words(PI_HALF_Q256);
    let k_times_pi_half = {
        let k_wide = I1024::from_i512(I512::from_i128(k));
        let pi_half_wide = I1024::from_i512(pi_half);
        (k_wide * pi_half_wide).as_i512()
    };
    let r = x - k_times_pi_half;

    let sin_r = taylor_sin_q256_256_impl(r);
    let cos_r = taylor_cos_q256_256_impl(r);

    let quadrant = ((k % 4) + 4) % 4;
    match quadrant {
        0 => (sin_r, cos_r),
        1 => (cos_r, I512::zero() - sin_r),
        2 => (I512::zero() - sin_r, I512::zero() - cos_r),
        3 => (I512::zero() - cos_r, sin_r),
        _ => unreachable!(),
    }
}

#[cfg(table_format = "q128_128")]
fn taylor_sin_q256_256_impl(r: I512) -> I512 {
    let r_sq = {
        let r_wide = I1024::from_i512(r);
        ((r_wide * r_wide) >> 256).as_i512()
    };

    let num_terms = 41;
    let mut result = I512::zero();

    for k in (1..num_terms).rev() {
        let coeff = I512::from_words(SIN_COEFFS_Q256[k]);
        let term = {
            let r_sq_wide = I1024::from_i512(r_sq);
            let result_wide = I1024::from_i512(result);
            ((r_sq_wide * result_wide) >> 256).as_i512()
        };
        result = coeff - term;
    }

    let inner = {
        let r_sq_wide = I1024::from_i512(r_sq);
        let result_wide = I1024::from_i512(result);
        ((r_sq_wide * result_wide) >> 256).as_i512()
    };
    let poly = I512::from_words(SIN_COEFFS_Q256[0]) - inner;

    let r_wide = I1024::from_i512(r);
    let poly_wide = I1024::from_i512(poly);
    ((r_wide * poly_wide) >> 256).as_i512()
}

#[cfg(table_format = "q128_128")]
fn taylor_cos_q256_256_impl(r: I512) -> I512 {
    let r_sq = {
        let r_wide = I1024::from_i512(r);
        ((r_wide * r_wide) >> 256).as_i512()
    };

    let num_terms = 41;
    let mut result = I512::zero();

    for k in (1..num_terms).rev() {
        let coeff = I512::from_words(COS_COEFFS_Q256[k]);
        let term = {
            let r_sq_wide = I1024::from_i512(r_sq);
            let result_wide = I1024::from_i512(result);
            ((r_sq_wide * result_wide) >> 256).as_i512()
        };
        result = coeff - term;
    }

    let inner = {
        let r_sq_wide = I1024::from_i512(r_sq);
        let result_wide = I1024::from_i512(result);
        ((r_sq_wide * result_wide) >> 256).as_i512()
    };
    I512::from_words(COS_COEFFS_Q256[0]) - inner
}

// ============================================================================
// Q32.32 / Q16.16 PROFILE WRAPPERS (i64 storage)
// ============================================================================

/// sin() for Q32.32 storage (i64) — tier N+1 via Q64.64
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn sin_binary_i64(x: i64) -> i64 {
    use super::exp_tier_n_plus_1::{upscale_q32_to_q64, downscale_q64_to_q32};
    let x_q64 = upscale_q32_to_q64(x);
    let result_q64 = sin_q64_64(x_q64);
    downscale_q64_to_q32(result_q64)
}

/// cos() for Q32.32 storage (i64) — tier N+1 via Q64.64
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn cos_binary_i64(x: i64) -> i64 {
    use super::exp_tier_n_plus_1::{upscale_q32_to_q64, downscale_q64_to_q32};
    let x_q64 = upscale_q32_to_q64(x);
    let result_q64 = cos_q64_64(x_q64);
    downscale_q64_to_q32(result_q64)
}

/// sin() at compute tier for Q16.16 profile (Q32.32 on i64)
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn sin_compute_tier_i64(x: i64) -> i64 {
    use super::exp_tier_n_plus_1::{upscale_q32_to_q64, downscale_q64_to_q32};
    let x_q64 = upscale_q32_to_q64(x);
    let result_q64 = sin_q64_64(x_q64);
    downscale_q64_to_q32(result_q64)
}

/// cos() at compute tier for Q16.16 profile (Q32.32 on i64)
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn cos_compute_tier_i64(x: i64) -> i64 {
    use super::exp_tier_n_plus_1::{upscale_q32_to_q64, downscale_q64_to_q32};
    let x_q64 = upscale_q32_to_q64(x);
    let result_q64 = cos_q64_64(x_q64);
    downscale_q64_to_q32(result_q64)
}

/// Fused sin+cos at compute tier for Q16.16 profile (Q32.32 on i64)
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn sincos_compute_tier_i64(x: i64) -> (i64, i64) {
    use super::exp_tier_n_plus_1::{upscale_q32_to_q64, downscale_q64_to_q32};
    let x_q64 = upscale_q32_to_q64(x);
    let (sin_val, cos_val) = sincos_q64_64(x_q64);
    (downscale_q64_to_q32(sin_val), downscale_q64_to_q32(cos_val))
}

/// sin() for Q32.32 profile — i128 is the compute tier (Q64.64)
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn sin_binary_i128(x: i128) -> i128 {
    sin_q64_64(x)
}

/// cos() for Q32.32 profile — i128 is the compute tier (Q64.64)
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn cos_binary_i128(x: i128) -> i128 {
    cos_q64_64(x)
}
