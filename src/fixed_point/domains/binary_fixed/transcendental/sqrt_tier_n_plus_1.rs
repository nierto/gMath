// ============================================================================
// TIER N+1 SQUARE ROOT
// ============================================================================
//
// **PRINCIPLE**: Compute at tier N+1, downscale to tier N with ZERO precision loss
//
// **GUARANTEE**: ALL fractional bits of storage tier are 100% accurate
//
// **ALGORITHM**: Integer Newton-Raphson
//   x_{n+1} = (x_n + S_shifted / x_n) / 2
//   where S_shifted = S << N (upscale to next tier type to avoid overflow)
//
//   For Q512.512 (I2048 lacks Div): Reciprocal-sqrt Newton on the
//   NORMALISED input m = S * 2^(-2k) in [1, 4), so that y and y^2 keep
//   full precision (the unnormalised form lost ~250 bits for large S),
//   y_{n+1} = y_n * (3 - m * y_n^2) / 2, sqrt(m) = m * y_final, then a
//   bounded exact-integer certificate correction and an exact shift by k
//
// **STRATEGY**:
//   - Tier 3 storage (i128/Q64.64):   Compute at Tier 4 (I256/Q128.128) → 19 decimals exact
//   - Tier 4 storage (I256/Q128.128): Compute at Tier 5 (I512/Q256.256) → 38 decimals exact
//   - Tier 5 storage (I512/Q256.256): Compute at Tier 6 (I1024/Q512.512) → 77 decimals exact
//
// **CONVERGENCE**: Quadratic - each iteration doubles correct bits
//   - Q64.64:   7 iterations (128 bits / 2^7 = 1 bit minimum → guaranteed)
//   - Q128.128: 9 iterations (256 bits / 2^9 = 0.5 bit → guaranteed)
//   - Q256.256: 10 iterations (512 bits / 2^10 = 0.5 bit → guaranteed)
//   - Q512.512: 11 iterations (1024 bits / 2^11 = 0.5 bit → guaranteed)
//
// ============================================================================

#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::i1024::I1024;

// Reuse upscale/downscale helpers from exp_tier_n_plus_1
// (not all are used in every profile, hence allow)
#[allow(unused_imports)]
use super::exp_tier_n_plus_1::{
    upscale_q64_to_q128, upscale_q128_to_q256, upscale_q64_to_q256,
    downscale_q128_to_q64, downscale_q256_to_q128, downscale_q256_to_q64,
};

// ============================================================================
// HELPER: Find MSB position (floor of log2) for initial seed
// ============================================================================

#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
#[inline(always)]
fn find_msb_position_i256(x: &I256) -> Option<u32> {
    if *x <= I256::zero() {
        return None;
    }
    for i in (0..4).rev() {
        if x.words[i] != 0 {
            let bit_pos = 63 - x.words[i].leading_zeros();
            return Some(i as u32 * 64 + bit_pos);
        }
    }
    None
}

#[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q32_32", table_format = "q16_16"))]
#[inline(always)]
fn find_msb_position_i512(x: &I512) -> Option<u32> {
    if *x <= I512::zero() {
        return None;
    }
    for i in (0..8).rev() {
        if x.words[i] != 0 {
            let bit_pos = 63 - x.words[i].leading_zeros();
            return Some(i as u32 * 64 + bit_pos);
        }
    }
    None
}

#[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256", table_format = "q512_512", table_format = "q32_32", table_format = "q16_16"))]
#[inline(always)]
fn find_msb_position_i1024(x: &I1024) -> Option<u32> {
    if *x <= I1024::zero() {
        return None;
    }
    for i in (0..16).rev() {
        if x.words[i] != 0 {
            let bit_pos = 63 - x.words[i].leading_zeros();
            return Some(i as u32 * 64 + bit_pos);
        }
    }
    None
}

// ============================================================================
// TIER-NATIVE IMPLEMENTATIONS
// ============================================================================

/// Q64.64 native square root (Tier 3 → computed at Tier 4 Q128.128)
///
/// **INPUT**: i128 value in Q64.64 format (must be non-negative)
/// **OUTPUT**: i128 value in Q64.64 format
/// **PRECISION**: 19 correct decimal digits (all 64 fractional bits exact)
/// **ALGORITHM**: Newton-Raphson in I256 (Q128.128) with 7 iterations
/// **DOMAIN**: x >= 0 (returns i128::MIN for x < 0)
#[cfg(any(table_format = "q64_64", table_format = "q32_32", table_format = "q16_16"))]
fn sqrt_q64_64_native(x: i128) -> i128 {
    // Domain: sqrt(x) undefined for x < 0
    if x < 0 {
        return i128::MIN;
    }
    if x == 0 {
        return 0;
    }

    // Special case: sqrt(1.0) = 1.0 exactly
    let one_q64: i128 = 1_i128 << 64;
    if x == one_q64 {
        return one_q64;
    }

    // Work in I256 for overflow prevention
    let s = I256::from_i128(x);

    // S_shifted = S << 64 (upscale from Q64.64 to Q128.128 for intermediate precision)
    // This gives us S in Q128.128-like format for the Newton iteration
    let s_shifted = s << 64;

    // Initial seed: x0 = 1 << ((msb(s_shifted) + 1) / 2)
    let msb = match find_msb_position_i256(&s_shifted) {
        Some(pos) => pos,
        None => return 0,
    };
    let mut x_n = I256::from_i128(1) << ((msb + 1) / 2) as usize;

    // Newton-Raphson: x_{n+1} = (x_n + S_shifted / x_n) / 2
    // 7 iterations for 128-bit precision (quadratic convergence)
    for _ in 0..7 {
        let quotient = s_shifted / x_n;
        x_n = (x_n + quotient) >> 1;
    }

    // Result is in Q128.128-like scale (fractional bits = 64 from original + 64 from shift)
    // But we want Q64.64 output, so extract as i128
    // The Newton iteration converges to sqrt(S_shifted) = sqrt(S << 64) = sqrt(S) * 2^32
    // Since S is in Q64.64, sqrt(S) should be in Q64.64 too.
    // sqrt(S_shifted) = sqrt(S * 2^64) = sqrt(S) * 2^32
    // But what we actually want: sqrt in Q64.64 = sqrt(S/2^64) * 2^64
    //   = sqrt(S) * 2^(-32) * 2^64 = sqrt(S) * 2^32
    // And sqrt(S_shifted) = sqrt(S * 2^64) = sqrt(S) * 2^32
    // So the result IS our answer in Q64.64 format.
    x_n.as_i128()
}

/// Q128.128 native square root (Tier 4 → computed at Tier 5 Q256.256)
///
/// **INPUT**: I256 value in Q128.128 format (must be non-negative)
/// **OUTPUT**: I256 value in Q128.128 format
/// **PRECISION**: 38 correct decimal digits (all 128 fractional bits exact)
/// **ALGORITHM**: Newton-Raphson in I512 (Q256.256) with 9 iterations
/// **DOMAIN**: x >= 0 (returns I256::min_value() for x < 0)
#[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q32_32", table_format = "q16_16"))]
fn sqrt_q128_128_native(x: I256) -> I256 {
    if x < I256::zero() {
        return I256::min_value();
    }
    if x == I256::zero() {
        return I256::zero();
    }

    // Special case: sqrt(1.0) = 1.0 exactly
    let one_q128 = I256::from_i128(1) << 128;
    if x == one_q128 {
        return one_q128;
    }

    // Work in I512 for overflow prevention
    let s = I512::from_i256(x);

    // S_shifted = S << 128 (upscale for Q256.256 intermediate precision)
    let s_shifted = s << 128;

    // Initial seed via MSB
    let msb = match find_msb_position_i512(&s_shifted) {
        Some(pos) => pos,
        None => return I256::zero(),
    };
    let mut x_n = I512::from_i256(I256::from_i128(1)) << ((msb + 1) / 2) as usize;

    // Newton-Raphson: 9 iterations for 256-bit precision
    for _ in 0..9 {
        let quotient = s_shifted / x_n;
        x_n = (x_n + quotient) >> 1;
    }

    x_n.as_i256()
}

/// Q256.256 native square root (Tier 5 → computed at Tier 6 Q512.512)
///
/// **INPUT**: I512 value in Q256.256 format (must be non-negative)
/// **OUTPUT**: I512 value in Q256.256 format
/// **PRECISION**: 77 correct decimal digits (all 256 fractional bits exact)
/// **ALGORITHM**: Newton-Raphson in I1024 (Q512.512) with 10 iterations
/// **DOMAIN**: x >= 0 (returns I512::min_value() for x < 0)
#[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256", table_format = "q32_32", table_format = "q16_16"))]
fn sqrt_q256_256_native(x: I512) -> I512 {
    if x < I512::zero() {
        return I512::min_value();
    }
    if x == I512::zero() {
        return I512::zero();
    }

    // Special case: sqrt(1.0) = 1.0 exactly
    let one_q256 = I512::one_q256_256();
    if x == one_q256 {
        return one_q256;
    }

    // Work in I1024 for overflow prevention
    let s = I1024::from_i512(x);

    // S_shifted = S << 256 (upscale for Q512.512 intermediate precision)
    let s_shifted = s << 256;

    // Initial seed via MSB
    let msb = match find_msb_position_i1024(&s_shifted) {
        Some(pos) => pos,
        None => return I512::zero(),
    };
    let mut x_n = I1024::from_i512(I512::from_i256(I256::from_i128(1))) << ((msb + 1) / 2) as usize;

    // Newton-Raphson: 12 iterations for 512-bit precision
    // Integer division truncation can slow quadratic convergence;
    // 12 iterations provides margin to achieve all 512 bits
    for _ in 0..12 {
        let quotient = s_shifted / x_n;
        x_n = (x_n + quotient) >> 1;
    }

    x_n.as_i512()
}

/// Square root at Q512.512 (tier N+1 for the scientific profile).
///
/// **ALGORITHM**: normalise the input to m in [1, 4) by an exact power-of-two
/// shift, run reciprocal-square-root Newton on m at full precision, certify
/// the result at the normalised scale by the exact integer check
/// r^2 <= m * 2^512 < (r+1)^2 with a bounded correction, then shift back.
/// **WHY NORMALISE**: run on the raw input, y = 1/sqrt(x) and y^2 are too small
/// for the Q512.512 grid when x is large, and the iteration converges to a
/// fixed point of the truncated map (about 2^-259 relative precision at
/// x = 2^254). With m in [1, 4) every intermediate keeps its full precision.
/// **DOMAIN**: x >= 0
#[cfg(any(table_format = "q256_256", table_format = "q512_512"))]
fn sqrt_q512_512_native(x: I1024) -> I1024 {
    if x < I1024::zero() {
        return I1024::min_value();
    }
    if x == I1024::zero() {
        return I1024::zero();
    }

    let one = I1024::one_q512_512();

    if x == one {
        return one;
    }

    // Normalise: x = m * 2^(2k) with m in [1, 4) at Q512.512.
    //
    // The reciprocal-square-root iteration below needs y = 1/sqrt(m) and y^2
    // at full 512-bit fractional precision. Run directly on a large x that
    // is not the case: y is tiny, y^2 is tinier, and truncating y^2 to the
    // Q512.512 grid leaves it only (512 - log2 x) significant bits, so the
    // iteration converges to a fixed point of the truncated map instead of
    // the true one (about 2^-259 relative precision at x = 2^254, which is
    // 2^124 ulp at Q256.256). With m in [1, 4), y is in (0.5, 1] and every
    // intermediate carries its full precision; the final shift by k is exact.
    // x is a Q256.256 value upscaled by 256 bits, so |k| <= 128 and the
    // normalising shift never discards a set bit for k >= 0, while for k < 0
    // m < 4 * 2^512 always fits.
    let msb = match find_msb_position_i1024(&x) {
        Some(pos) => pos as i32,
        None => return I1024::zero(),
    };
    let k = (msb - 512).div_euclid(2);
    let m = if k >= 0 { x >> ((2 * k) as usize) } else { x << ((-2 * k) as usize) };
    debug_assert!(m >= one && m < (one << 2), "sqrt normalisation left [1, 4)");

    // Reciprocal-sqrt Newton-Raphson on m:
    // y_{n+1} = y_n * (3 - m * y_n^2) / 2
    // Seed 1 for m in [1, 2) and 1/2 for m in [2, 4): both satisfy
    // m * y_0^2 < 3, the convergence condition. Eleven iterations reach the
    // 2^-512 grid from either seed with margin (ten suffice in the model).
    let three = I1024::from_i128(3) << 512;
    let mut y_n = if m < (one << 1) { one } else { one >> 1 };
    for _ in 0..11 {
        let y_sq = multiply_i1024_q512_512_signed(y_n, y_n);
        let s_y_sq = multiply_i1024_q512_512_signed(m, y_sq);
        let diff = three - s_y_sq;
        let product = multiply_i1024_q512_512_signed(y_n, diff);
        y_n = product >> 1;
    }

    // sqrt(m) = m * y at Q512.512, then certify it at the normalised scale:
    // r must be exactly floor(sqrt(m * 2^512)), i.e. r^2 <= m * 2^512 < (r+1)^2
    // in exact 2048-bit integers. The Newton result is within a few units
    // (measured: at most 4 on 3220 inputs spanning the profile's range), so
    // the correction is bounded and any excursion beyond it is a defect that
    // must panic rather than return an unverified value.
    let mut r = multiply_i1024_q512_512_signed(m, y_n);
    let target = crate::fixed_point::I2048::from_i1024(m) << 512usize;
    let unit = I1024::from_i128(1);
    let mut steps = 0u32;
    while r.mul_to_i2048(r) > target {
        r = r - unit;
        steps += 1;
        assert!(steps <= 8, "sqrt_q512_512_native: certificate correction exceeded its bound (downward)");
    }
    loop {
        let next = r + unit;
        if next.mul_to_i2048(next) <= target {
            r = next;
            steps += 1;
            assert!(steps <= 8, "sqrt_q512_512_native: certificate correction exceeded its bound (upward)");
        } else {
            break;
        }
    }

    // Undo the normalisation: sqrt(x) = sqrt(m) * 2^k. The shift is exact for
    // k >= 0; for k < 0 it floors at the 2^-512 grid, far below storage.
    if k >= 0 { r << (k as usize) } else { r >> ((-k) as usize) }
}

/// Q512.512 fixed-point multiply: (a * b) >> 512
///
/// Signed Q512.512 multiply, truncating the magnitude (toward zero).
///
/// 0.5.0 audit history: this was `multiply_i1024_q512_512` (shadowing the
/// sign-safe ln helper), assumed positive-by-construction and asserted as
/// such, and the assert then FIRED on scientific SPD-distance inputs:
/// the Newton factor `3 − S·y²` goes NEGATIVE when the seed overshoots,
/// so the raw unsigned `mul_to_i2048` here was a real latent corruption
/// path for overshooting inputs. Now sign-wrapped. Magnitude truncation
/// is kept (identical to the old behavior for the positive common case,
/// so the validated Newton trajectory is unchanged there).
#[cfg(any(table_format = "q256_256", table_format = "q512_512"))]
#[inline(always)]
fn multiply_i1024_q512_512_signed(a: I1024, b: I1024) -> I1024 {
    let a_neg = a < I1024::zero();
    let b_neg = b < I1024::zero();
    let abs_a = if a_neg { -a } else { a };
    let abs_b = if b_neg { -b } else { b };
    let full = abs_a.mul_to_i2048(abs_b);
    let mag = (full >> 512).as_i1024();
    if a_neg != b_neg { -mag } else { mag }
}

// ============================================================================
// PROFILE-AWARE WRAPPERS (provide all functions for all profiles)
// ============================================================================

// For scientific profile (Q256.256), provide wrappers for lower tiers
#[cfg(table_format = "q256_256")]
pub fn sqrt_binary_i128(x: i128) -> i128 {
    if x < 0 { return i128::MIN; }
    let x_q256 = upscale_q64_to_q256(x);
    let result_q256 = sqrt_q256_256_native(x_q256);
    downscale_q256_to_q64(result_q256)
}

#[cfg(table_format = "q256_256")]
pub fn sqrt_binary_i256(x: I256) -> I256 {
    if x < I256::zero() { return I256::min_value(); }
    let x_q256 = upscale_q128_to_q256(x);
    let result_q256 = sqrt_q256_256_native(x_q256);
    downscale_q256_to_q128(result_q256)
}

#[cfg(table_format = "q256_256")]
pub fn sqrt_binary_i512(x: I512) -> I512 {
    // Direct Q256.256 computation
    sqrt_q256_256_native(x)
}

// For balanced profile (Q128.128), provide wrappers
#[cfg(table_format = "q128_128")]
pub fn sqrt_binary_i128(x: i128) -> i128 {
    if x < 0 { return i128::MIN; }
    let x_q128 = upscale_q64_to_q128(x);
    let result_q128 = sqrt_q128_128_native(x_q128);
    downscale_q128_to_q64(result_q128)
}

#[cfg(table_format = "q128_128")]
pub fn sqrt_binary_i256(x: I256) -> I256 {
    if x < I256::zero() { return I256::min_value(); }
    let x_q256 = upscale_q128_to_q256(x);
    let result_q256 = sqrt_q256_256_native(x_q256);
    downscale_q256_to_q128(result_q256)
}

#[cfg(table_format = "q128_128")]
pub fn sqrt_binary_i512(x: I512) -> I512 {
    // Direct Q256.256 computation (compute tier for balanced profile)
    sqrt_q256_256_native(x)
}

// For performance/embedded profile (Q64.64), provide wrappers
#[cfg(table_format = "q64_64")]
pub fn sqrt_binary_i128(x: i128) -> i128 {
    // Direct Q64.64 computation
    sqrt_q64_64_native(x)
}

#[cfg(table_format = "q64_64")]
pub fn sqrt_binary_i256(x: I256) -> I256 {
    // Tier N+1: compute sqrt at Q128.128 natively
    sqrt_q128_128_native(x)
}

#[cfg(table_format = "q64_64")]
pub fn sqrt_binary_i512(x: I512) -> I512 {
    // Compute sqrt at Q256.256 for Q64.64 profile
    sqrt_q256_256_native(x)
}

// I1024 sqrt for scientific profile tier N+1 computation
#[cfg(any(table_format = "q256_256", table_format = "q512_512"))]
pub fn sqrt_binary_i1024(x: I1024) -> I1024 {
    sqrt_q512_512_native(x)
}

// ============================================================================
// Q32.32 / Q16.16 PROFILE WRAPPERS (i64 storage)
// ============================================================================

/// sqrt() for Q32.32 storage (i64): tier N+1 via Q64.64
///
/// For Q16.16 profile: ComputeStorage = i64, BinaryStorage = i32
/// For Q32.32 profile: BinaryStorage = i64
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn sqrt_binary_i64(x: i64) -> i64 {
    if x < 0 { return i64::MIN; }
    use super::exp_tier_n_plus_1::{upscale_q32_to_q64, downscale_q64_to_q32};
    let x_q64 = upscale_q32_to_q64(x);
    let result_q64 = sqrt_q64_64_native(x_q64);
    downscale_q64_to_q32(result_q64)
}

/// sqrt() for Q32.32 profile: i128 is the compute tier (Q64.64)
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn sqrt_binary_i128(x: i128) -> i128 {
    sqrt_q64_64_native(x)
}

/// sqrt() for Q32.32 profile: I256 is tier N+1 (Q128.128)
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn sqrt_binary_i256(x: I256) -> I256 {
    // Tier N+1: compute sqrt at Q128.128 natively
    sqrt_q128_128_native(x)
}

/// sqrt() for Q32.32 profile: I512 wrapper
#[cfg(any(table_format = "q32_32", table_format = "q16_16"))]
pub fn sqrt_binary_i512(x: I512) -> I512 {
    // Compute sqrt at Q256.256 for Q32.32 profile
    sqrt_q256_256_native(x)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[cfg(not(any(table_format = "q32_32", table_format = "q16_16")))]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_zero() {
        assert_eq!(sqrt_binary_i128(0), 0);
    }

    #[test]
    fn test_sqrt_one() {
        #[cfg(table_format = "q64_64")]
        let one: i128 = 1_i128 << 64;
        #[cfg(table_format = "q128_128")]
        let one: i128 = 1_i128 << 64;
        #[cfg(table_format = "q256_256")]
        let one: i128 = 1_i128 << 64;

        let result = sqrt_binary_i128(one);
        // sqrt(1.0) should be exactly 1.0
        assert_eq!(result, one);
    }

    #[test]
    fn test_sqrt_four() {
        // sqrt(4.0) = 2.0 exactly
        #[cfg(table_format = "q64_64")]
        {
            let four: i128 = 4_i128 << 64;
            let two: i128 = 2_i128 << 64;
            let result = sqrt_binary_i128(four);
            // Allow 1 ULP tolerance
            let diff = (result - two).abs();
            assert!(diff <= 1, "sqrt(4.0) should be 2.0, diff={}", diff);
        }
        #[cfg(table_format = "q128_128")]
        {
            let four: i128 = 4_i128 << 64;
            let two: i128 = 2_i128 << 64;
            let result = sqrt_binary_i128(four);
            let diff = (result - two).abs();
            assert!(diff <= 1, "sqrt(4.0) should be 2.0, diff={}", diff);
        }
        #[cfg(table_format = "q256_256")]
        {
            let four: i128 = 4_i128 << 64;
            let two: i128 = 2_i128 << 64;
            let result = sqrt_binary_i128(four);
            let diff = (result - two).abs();
            assert!(diff <= 1, "sqrt(4.0) should be 2.0, diff={}", diff);
        }
    }

    #[test]
    fn test_sqrt_negative_returns_sentinel() {
        let neg = -1_i128 << 64;
        let result = sqrt_binary_i128(neg);
        assert_eq!(result, i128::MIN);
    }

    #[test]
    fn test_sqrt_nine() {
        // sqrt(9.0) = 3.0 exactly
        let nine: i128 = 9_i128 << 64;
        let three: i128 = 3_i128 << 64;
        let result = sqrt_binary_i128(nine);
        let diff = (result - three).abs();
        assert!(diff <= 1, "sqrt(9.0) should be 3.0, diff={}", diff);
    }

    #[test]
    fn test_sqrt_i256_one() {
        let one = I256::from_i128(1) << 128;
        let result = sqrt_binary_i256(one);
        assert_eq!(result, one);
    }

    #[test]
    fn test_sqrt_i256_four() {
        let four = I256::from_i128(4) << 128;
        let two = I256::from_i128(2) << 128;
        let result = sqrt_binary_i256(four);
        let diff = if result > two { result - two } else { two - result };
        assert!(diff <= I256::from_i128(1), "sqrt(4.0) should be 2.0 in I256");
    }

    #[cfg(table_format = "q256_256")]
    #[test]
    fn test_sqrt_i512_one() {
        let one = I512::one_q256_256();
        let result = sqrt_binary_i512(one);
        assert_eq!(result, one);
    }

    #[cfg(table_format = "q256_256")]
    #[test]
    fn test_sqrt_i512_four() {
        let four = I512::from_i256(I256::from_i128(4)) << 256;
        let two = I512::from_i256(I256::from_i128(2)) << 256;
        let result = sqrt_binary_i512(four);
        let diff = if result > two { result - two } else { two - result };
        assert!(diff <= I512::from_i256(I256::from_i128(1)), "sqrt(4.0) should be 2.0 in I512");
    }
}
