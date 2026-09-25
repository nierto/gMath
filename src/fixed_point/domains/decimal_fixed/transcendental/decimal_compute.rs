//! Decimal compute-tier primitives: arithmetic on values scaled by 10^DECIMAL_COMPUTE_DP.
//!
//! All values in this module are interpreted as `raw_int × 10^(-DECIMAL_COMPUTE_DP)`.
//! Mirrors binary Q-format primitives but with decimal (base-10) scaling.
//!
//! ## Widening Multiply
//!
//! `a × b` on two compute-tier values overflows the compute type. We use the next-wider
//! integer type for the product, then divide by 10^compute_dp to return to compute tier.
//!
//! | Profile    | Compute    | Wide       | Widening op                |
//! |------------|------------|------------|----------------------------|
//! | realtime   | i64        | i128       | (val as i128)              |
//! | compact    | i128       | I256       | I256::from_i128            |
//! | embedded   | I256       | I512       | I256::mul_to_i512          |
//! | balanced   | I512       | I1024      | I512::mul_to_i1024         |
//! | scientific | I1024      | I2048      | I1024::mul_to_i2048        |

#[allow(unused_imports)]
pub use crate::fixed_point::universal::fasc::stack_evaluator::{BinaryStorage, ComputeStorage};
#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;
#[cfg(table_format = "q256_256")]
use crate::fixed_point::I2048;
#[cfg(table_format = "q256_256")]
use crate::fixed_point::domains::binary_fixed::i2048::i2048_div;

use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

// ============================================================================
// PROFILE-SPECIFIC CONSTANTS
// ============================================================================

/// Decimal compute-tier precision (decimal places): matches binary compute frac bits.
///
/// Chosen so `compute_dp ≈ 2 × storage_dp`, giving the decimal equivalent of binary
/// tier N+1 precision.
#[cfg(table_format = "q16_16")]
pub const DECIMAL_COMPUTE_DP: u8 = 9;
#[cfg(table_format = "q32_32")]
pub const DECIMAL_COMPUTE_DP: u8 = 19;
#[cfg(table_format = "q64_64")]
pub const DECIMAL_COMPUTE_DP: u8 = 38;
#[cfg(table_format = "q128_128")]
pub const DECIMAL_COMPUTE_DP: u8 = 77;
#[cfg(table_format = "q256_256")]
pub const DECIMAL_COMPUTE_DP: u8 = 154;

/// Maximum storage dp supported for decimal transcendentals (conservative bound).
#[cfg(table_format = "q16_16")]
pub const DECIMAL_STORAGE_MAX_DP: u8 = 4;
#[cfg(table_format = "q32_32")]
pub const DECIMAL_STORAGE_MAX_DP: u8 = 9;
#[cfg(table_format = "q64_64")]
pub const DECIMAL_STORAGE_MAX_DP: u8 = 19;
#[cfg(table_format = "q128_128")]
pub const DECIMAL_STORAGE_MAX_DP: u8 = 38;
#[cfg(table_format = "q256_256")]
pub const DECIMAL_STORAGE_MAX_DP: u8 = 77;

// ============================================================================
// WIDE COMPUTE TYPE — one tier wider than ComputeStorage for widening multiply
// ============================================================================

#[cfg(table_format = "q16_16")]
pub(crate) type WideCompute = i128;
#[cfg(table_format = "q32_32")]
pub(crate) type WideCompute = I256;
#[cfg(table_format = "q64_64")]
pub(crate) type WideCompute = I512;
#[cfg(table_format = "q128_128")]
pub(crate) type WideCompute = I1024;
#[cfg(table_format = "q256_256")]
pub(crate) type WideCompute = I2048;

// ============================================================================
// CONSTANTS — zero, one, scale factor (10^DECIMAL_COMPUTE_DP)
// ============================================================================

/// Compute-tier `0`.
#[inline]
pub fn decimal_compute_zero() -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { 0i64 }
    #[cfg(table_format = "q32_32")]
    { 0i128 }
    #[cfg(table_format = "q64_64")]
    { I256::zero() }
    #[cfg(table_format = "q128_128")]
    { I512::zero() }
    #[cfg(table_format = "q256_256")]
    { I1024::zero() }
}

/// Compute-tier `1.0` = `10^DECIMAL_COMPUTE_DP`.
#[inline]
pub fn decimal_compute_scale() -> ComputeStorage {
    pow10_compute_ct(DECIMAL_COMPUTE_DP)
}

/// Alias for `decimal_compute_scale`: represents the value 1.0 at compute dp.
#[inline]
pub fn decimal_compute_one() -> ComputeStorage {
    decimal_compute_scale()
}

/// Convert a small integer (abs value ≤ 2^62) to decimal compute tier.
#[inline]
pub fn decimal_compute_from_int(n: i64) -> ComputeStorage {
    let one = decimal_compute_one();
    #[cfg(table_format = "q16_16")]
    { one.checked_mul(n).expect("decimal_compute_from_int: integer outside the compute range") }
    #[cfg(table_format = "q32_32")]
    { one * (n as i128) }
    #[cfg(table_format = "q64_64")]
    { one * I256::from_i128(n as i128) }
    #[cfg(table_format = "q128_128")]
    { one * I512::from_i128(n as i128) }
    #[cfg(table_format = "q256_256")]
    { one * I1024::from_i128(n as i128) }
}

/// Wide-tier scale factor `10^DECIMAL_COMPUTE_DP` in the widening type.
#[inline]
fn wide_scale() -> WideCompute {
    #[cfg(table_format = "q16_16")]
    { pow10_i128(DECIMAL_COMPUTE_DP as u32) }
    #[cfg(table_format = "q32_32")]
    { I256::from_i128(pow10_i128(DECIMAL_COMPUTE_DP as u32)) }
    #[cfg(table_format = "q64_64")]
    { I512::from_i256(pow10_compute_ct(DECIMAL_COMPUTE_DP)) }
    #[cfg(table_format = "q128_128")]
    { I1024::from_i512(pow10_compute_ct(DECIMAL_COMPUTE_DP)) }
    #[cfg(table_format = "q256_256")]
    { I2048::from_i1024(pow10_compute_ct(DECIMAL_COMPUTE_DP)) }
}

// ============================================================================
// POW10 HELPERS — compute 10^n at various tiers
// ============================================================================

/// 10^exp as i128. Valid for exp ≤ 38.
#[inline]
#[allow(dead_code)]
pub(crate) fn pow10_i128(exp: u32) -> i128 {
    let mut result: i128 = 1;
    let mut i = 0;
    while i < exp {
        result *= 10;
        i += 1;
    }
    result
}

/// 10^exp in ComputeStorage type. Valid for exp ≤ DECIMAL_COMPUTE_DP.
#[inline]
pub(crate) fn pow10_compute_ct(exp: u8) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    {
        let mut result: i64 = 1;
        for _ in 0..exp { result = result.saturating_mul(10); }
        result
    }
    #[cfg(table_format = "q32_32")]
    {
        let mut result: i128 = 1;
        for _ in 0..exp { result *= 10; }
        result
    }
    #[cfg(table_format = "q64_64")]
    {
        let mut result = I256::from_i128(1);
        let ten = I256::from_i128(10);
        for _ in 0..exp { result = result * ten; }
        result
    }
    #[cfg(table_format = "q128_128")]
    {
        let mut result = I512::from_i128(1);
        let ten = I512::from_i128(10);
        for _ in 0..exp { result = result * ten; }
        result
    }
    #[cfg(table_format = "q256_256")]
    {
        let mut result = I1024::from_i128(1);
        let ten = I1024::from_i128(10);
        for _ in 0..exp { result = result * ten; }
        result
    }
}

// ============================================================================
// PROFILE-SPECIFIC WIDENING / NARROWING
// ============================================================================

#[inline]
fn widen(v: ComputeStorage) -> WideCompute {
    #[cfg(table_format = "q16_16")]
    { v as i128 }
    #[cfg(table_format = "q32_32")]
    { I256::from_i128(v) }
    #[cfg(table_format = "q64_64")]
    { I512::from_i256(v) }
    #[cfg(table_format = "q128_128")]
    { I1024::from_i512(v) }
    #[cfg(table_format = "q256_256")]
    { I2048::from_i1024(v) }
}

/// Narrow a wide value to the compute tier, `Err(TierOverflow)` when it does
/// not fit (an unchecked cast before 0.6.4, which wrapped silently).
#[inline]
fn narrow(w: WideCompute) -> Result<ComputeStorage, OverflowDetected> {
    #[cfg(table_format = "q16_16")]
    { i64::try_from(w).map_err(|_| OverflowDetected::TierOverflow) }
    #[cfg(table_format = "q32_32")]
    { if w.fits_in_i128() { Ok(w.as_i128()) } else { Err(OverflowDetected::TierOverflow) } }
    #[cfg(table_format = "q64_64")]
    { if w.fits_in_i256() { Ok(w.as_i256()) } else { Err(OverflowDetected::TierOverflow) } }
    #[cfg(table_format = "q128_128")]
    { if w.fits_in_i512() { Ok(w.as_i512()) } else { Err(OverflowDetected::TierOverflow) } }
    #[cfg(table_format = "q256_256")]
    { if w.fits_in_i1024() { Ok(w.as_i1024()) } else { Err(OverflowDetected::TierOverflow) } }
}

/// Widening multiply: ComputeStorage × ComputeStorage → WideCompute (no overflow).
///
/// **CRITICAL**: The underlying `mul_to_*` functions use UNSIGNED word arithmetic,
/// which gives incorrect results for negative I256/I512/I1024 inputs (because the
/// sign-extended high words are treated as huge positive magnitudes). We must
/// extract signs ourselves, multiply the absolute values, then re-apply the sign.
#[inline]
fn wide_mul(a: ComputeStorage, b: ComputeStorage) -> WideCompute {
    #[cfg(table_format = "q16_16")]
    {
        // i64 × i64 → i128: Rust signed mul handles sign correctly
        (a as i128) * (b as i128)
    }
    #[cfg(table_format = "q32_32")]
    {
        // i128 × i128: explicit sign handling
        let a_neg = a < 0;
        let b_neg = b < 0;
        let result_neg = a_neg ^ b_neg;
        // negate after widening: i128::MIN has no i128 negation (it panicked)
        let abs_a = if a_neg { I256::zero() - I256::from_i128(a) } else { I256::from_i128(a) };
        let abs_b = if b_neg { I256::zero() - I256::from_i128(b) } else { I256::from_i128(b) };
        // Use I256 unsigned-word multiply: both operands non-negative now
        let abs_product = abs_a * abs_b;
        if result_neg { I256::zero() - abs_product } else { abs_product }
    }
    #[cfg(table_format = "q64_64")]
    {
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg ^ b_neg;
        let abs_a = if a_neg { I256::zero() - a } else { a };
        let abs_b = if b_neg { I256::zero() - b } else { b };
        let abs_product = abs_a.mul_to_i512(abs_b);
        if result_neg { I512::zero() - abs_product } else { abs_product }
    }
    #[cfg(table_format = "q128_128")]
    {
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg ^ b_neg;
        let abs_a = if a_neg { I512::zero() - a } else { a };
        let abs_b = if b_neg { I512::zero() - b } else { b };
        let abs_product = abs_a.mul_to_i1024(abs_b);
        if result_neg { I1024::zero() - abs_product } else { abs_product }
    }
    #[cfg(table_format = "q256_256")]
    {
        let a_neg = (a.words[15] as i64) < 0;
        let b_neg = (b.words[15] as i64) < 0;
        let result_neg = a_neg ^ b_neg;
        let abs_a = if a_neg { I1024::zero() - a } else { a };
        let abs_b = if b_neg { I1024::zero() - b } else { b };
        let abs_product = abs_a.mul_to_i2048(abs_b);
        if result_neg { I2048::zero() - abs_product } else { abs_product }
    }
}

/// Divide at wide tier (handles I2048 via free function).
#[inline]
fn wide_div(a: WideCompute, b: WideCompute) -> WideCompute {
    #[cfg(table_format = "q256_256")]
    { i2048_div(a, b) }
    #[cfg(not(table_format = "q256_256"))]
    { a / b }
}

/// `w / 10^DECIMAL_COMPUTE_DP` truncated toward zero: bit-identical to
/// `wide_div(w, wide_scale())`. On balanced and scientific the wide integers
/// divide bit by bit (2048 steps for I2048, about 69 us a call in release),
/// once per decimal multiply, which made FASC decimal transcendentals about
/// 570x slower than the binary engine there. Dividing the magnitude by 10^19
/// limb by limb, ceil(DP / 19) times, is O(words) per pass:
/// floor(floor(x / a) / b) = floor(x / (a b)) for x >= 0.
#[inline]
fn wide_div_scale(w: WideCompute) -> WideCompute {
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    {
        let negative = wide_is_negative(&w);
        let mut m = if negative { -w } else { w };
        let mut left = DECIMAL_COMPUTE_DP as u32;
        while left > 0 {
            let k = left.min(19);
            let d = 10u128.pow(k);
            let mut rem: u128 = 0;
            for word in m.words.iter_mut().rev() {
                let cur = (rem << 64) | *word as u128;
                *word = (cur / d) as u64;
                rem = cur % d;
            }
            left -= k;
        }
        if negative { -m } else { m }
    }
    #[cfg(not(any(table_format = "q128_128", table_format = "q256_256")))]
    { wide_div(w, wide_scale()) }
}

/// `w >> 1` at wide tier (halve). Used for rounding.
#[inline]
fn wide_halve(w: WideCompute) -> WideCompute {
    #[cfg(table_format = "q16_16")]
    { w >> 1 }
    #[cfg(table_format = "q32_32")]
    { w >> 1 }
    #[cfg(table_format = "q64_64")]
    { w >> 1 }
    #[cfg(table_format = "q128_128")]
    { w >> 1 }
    #[cfg(table_format = "q256_256")]
    { w >> 1 }
}

#[inline]
fn wide_is_negative(w: &WideCompute) -> bool {
    #[cfg(table_format = "q16_16")]
    { *w < 0 }
    #[cfg(table_format = "q32_32")]
    { *w < I256::zero() }
    #[cfg(table_format = "q64_64")]
    { *w < I512::zero() }
    #[cfg(table_format = "q128_128")]
    { *w < I1024::zero() }
    #[cfg(table_format = "q256_256")]
    {
        // I2048 sign bit is the MSB of word 31
        (w.words[31] & 0x8000_0000_0000_0000) != 0
    }
}

#[inline]
#[allow(dead_code)]
fn wide_zero() -> WideCompute {
    #[cfg(table_format = "q16_16")]
    { 0i128 }
    #[cfg(table_format = "q32_32")]
    { I256::zero() }
    #[cfg(table_format = "q64_64")]
    { I512::zero() }
    #[cfg(table_format = "q128_128")]
    { I1024::zero() }
    #[cfg(table_format = "q256_256")]
    { I2048::zero() }
}

#[inline]
fn wide_neg(w: WideCompute) -> WideCompute {
    #[cfg(table_format = "q16_16")]
    { -w }
    #[cfg(table_format = "q32_32")]
    { I256::zero() - w }
    #[cfg(table_format = "q64_64")]
    { I512::zero() - w }
    #[cfg(table_format = "q128_128")]
    { I1024::zero() - w }
    #[cfg(table_format = "q256_256")]
    { I2048::zero() - w }
}

// ============================================================================
// ARITHMETIC PRIMITIVES — add, sub, neg, halve
// ============================================================================

/// Add two compute-tier decimal values.
#[inline]
pub fn decimal_compute_add(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage {
    try_decimal_compute_add(a, b).expect("decimal compute tier: addition overflow")
}

/// `a + b` of compute-tier decimal values, `Err(TierOverflow)` when the sum
/// leaves the compute tier (0.6.3 wrapped).
#[inline]
pub fn try_decimal_compute_add(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    a.checked_add(b).ok_or(OverflowDetected::TierOverflow)
}

/// Subtract two compute-tier decimal values.
#[inline]
pub fn decimal_compute_sub(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage {
    try_decimal_compute_sub(a, b).expect("decimal compute tier: subtraction overflow")
}

/// `a - b` of compute-tier decimal values, `Err(TierOverflow)` when the
/// difference leaves the compute tier (0.6.3 wrapped).
#[inline]
pub fn try_decimal_compute_sub(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    a.checked_add(try_decimal_compute_neg(b)?).ok_or(OverflowDetected::TierOverflow)
}

/// `-a` of a compute-tier decimal value, `Err(TierOverflow)` for the tier's
/// minimum.
#[inline]
pub fn try_decimal_compute_neg(a: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    #[cfg(not(table_format = "q256_256"))]
    { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    #[cfg(table_format = "q256_256")]
    { if a == I1024::min_value() { Err(OverflowDetected::TierOverflow) } else { Ok(I1024::zero() - a) } }
}

/// Negate a compute-tier decimal value.
#[inline]
pub fn decimal_compute_neg(a: ComputeStorage) -> ComputeStorage {
    try_decimal_compute_neg(a).expect("decimal compute tier: negation overflow")
}

/// Divide compute-tier value by 2 (exact for even values, rounds toward -∞ for odd).
#[inline]
pub fn decimal_compute_halve(a: ComputeStorage) -> ComputeStorage {
    a >> 1
}

/// Absolute value at compute tier.
#[inline]
pub fn decimal_compute_abs(a: ComputeStorage) -> ComputeStorage {
    if decimal_compute_is_negative(&a) {
        decimal_compute_neg(a)
    } else {
        a
    }
}

#[inline]
pub fn decimal_compute_is_zero(a: &ComputeStorage) -> bool {
    #[cfg(table_format = "q16_16")]
    { *a == 0 }
    #[cfg(table_format = "q32_32")]
    { *a == 0 }
    #[cfg(table_format = "q64_64")]
    { a.is_zero() }
    #[cfg(table_format = "q128_128")]
    { a.is_zero() }
    #[cfg(table_format = "q256_256")]
    { a.is_zero() }
}

#[inline]
pub fn decimal_compute_is_negative(a: &ComputeStorage) -> bool {
    #[cfg(table_format = "q16_16")]
    { *a < 0 }
    #[cfg(table_format = "q32_32")]
    { *a < 0 }
    #[cfg(table_format = "q64_64")]
    { a.is_negative() }
    #[cfg(table_format = "q128_128")]
    { a.is_negative() }
    #[cfg(table_format = "q256_256")]
    { (a.words[15] as i64) < 0 }
}

#[inline]
pub fn decimal_compute_cmp(a: &ComputeStorage, b: &ComputeStorage) -> std::cmp::Ordering {
    a.cmp(b)
}

// ============================================================================
// MULTIPLY — (a × b) / 10^compute_dp with round-half-away-from-zero
// ============================================================================

/// Multiply two compute-tier decimal values, rescaling back to compute dp.
///
/// Panics when the product leaves the compute tier (it wrapped silently
/// before 0.6.4); see [`try_decimal_compute_mul`].
#[inline]
pub fn decimal_compute_mul(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage {
    try_decimal_compute_mul(a, b).expect("decimal_compute_mul: product outside the decimal compute tier")
}

/// Multiply two compute-tier decimal values, rescaling back to compute dp,
/// `Err(TierOverflow)` when the product does not fit the compute tier.
///
/// Algorithm:
/// 1. Widening multiply: `product = a × b` in the wide type (no overflow).
/// 2. Divide by `10^DECIMAL_COMPUTE_DP` with round-half-away-from-zero.
/// 3. Narrow back to ComputeStorage (checked).
#[inline]
pub fn try_decimal_compute_mul(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    let product = wide_mul(a, b);
    let scale = wide_scale();
    let half_scale = wide_halve(scale);

    // Round-half-away-from-zero: for positive products add half, for negative subtract half.
    let rounded = if wide_is_negative(&product) {
        product - half_scale
    } else {
        product + half_scale
    };

    narrow(wide_div_scale(rounded))
}

// ============================================================================
// DIVIDE — (a × 10^compute_dp) / b with rounding
// ============================================================================

/// Divide two compute-tier decimal values: `a / b`, result at compute dp.
///
/// Formula: `result = round((a × 10^dp) / b)`. Uses wide type for the numerator.
/// `Err(TierOverflow)` when the quotient does not fit the compute tier.
#[inline]
pub fn decimal_compute_div(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    if decimal_compute_is_zero(&b) {
        return Err(OverflowDetected::DomainError);
    }

    // numerator = a × 10^dp (in wide type)
    let numerator = wide_mul(a, decimal_compute_scale());
    let divisor = widen(b);

    // Round-half-away-from-zero
    let half_divisor = wide_halve(if wide_is_negative(&divisor) { wide_neg(divisor) } else { divisor });
    let result_neg = wide_is_negative(&numerator) ^ wide_is_negative(&divisor);
    let num_abs = if wide_is_negative(&numerator) { wide_neg(numerator) } else { numerator };
    let div_abs = if wide_is_negative(&divisor) { wide_neg(divisor) } else { divisor };

    let rounded_num = num_abs + half_divisor;
    let quotient = wide_div(rounded_num, div_abs);

    let final_quot = if result_neg { wide_neg(quotient) } else { quotient };

    narrow(final_quot)
}

/// Divide compute-tier value by a small positive integer, with round-half-away-from-zero.
///
/// Fast path for Taylor series where each term is divided by `n` (term index).
#[inline]
pub fn decimal_compute_div_int(a: ComputeStorage, n: u64) -> ComputeStorage {
    if n == 0 {
        return decimal_compute_zero();
    }
    if n == 1 {
        return a;
    }

    // Round-half-away-from-zero at compute tier (no widening needed — n is small).
    let n_compute: ComputeStorage = {
        #[cfg(table_format = "q16_16")]
        { n as i64 }
        #[cfg(table_format = "q32_32")]
        { n as i128 }
        #[cfg(table_format = "q64_64")]
        { I256::from_i128(n as i128) }
        #[cfg(table_format = "q128_128")]
        { I512::from_i128(n as i128) }
        #[cfg(table_format = "q256_256")]
        { I1024::from_i128(n as i128) }
    };

    let half_n = n_compute >> 1;
    let rounded = if decimal_compute_is_negative(&a) {
        a - half_n
    } else {
        a + half_n
    };

    rounded / n_compute
}

// ============================================================================
// UPSCALE / DOWNSCALE — bridge between storage dp and compute dp
// ============================================================================

/// Upscale a Decimal StackValue component (dp, scaled) to compute-tier decimal format.
///
/// - If `storage_dp < DECIMAL_COMPUTE_DP`: multiply by `10^(compute_dp - storage_dp)`.
/// - If `storage_dp == DECIMAL_COMPUTE_DP`: widen only.
/// - If `storage_dp > DECIMAL_COMPUTE_DP`: divide (precision loss possible: rare case).
#[inline]
pub fn decimal_upscale_to_compute(scaled: BinaryStorage, storage_dp: u8) -> Result<ComputeStorage, OverflowDetected> {
    if storage_dp == DECIMAL_COMPUTE_DP {
        return Ok(binary_storage_to_compute_widen(scaled));
    }

    if storage_dp < DECIMAL_COMPUTE_DP {
        let diff = DECIMAL_COMPUTE_DP - storage_dp;
        if diff > DECIMAL_COMPUTE_DP {
            return Err(OverflowDetected::Overflow);
        }
        let factor = pow10_compute_ct(diff);
        let widened = binary_storage_to_compute_widen(scaled);
        // Check that widened × factor fits in ComputeStorage.
        // For our profiles this is always true because storage_dp ≤ DECIMAL_STORAGE_MAX_DP
        // and compute_dp ≤ 2 × DECIMAL_STORAGE_MAX_DP, so the product is bounded.
        return Ok(widened * factor);
    }

    // storage_dp > compute_dp: divide, rounding half to even (the decimal rule).
    let diff = storage_dp - DECIMAL_COMPUTE_DP;
    let divisor = pow10_compute_ct(diff);
    let widened = binary_storage_to_compute_widen(scaled);
    Ok(div_round_half_even(widened, divisor))
}

/// `value / divisor` rounded to nearest, ties to even (banker's, the decimal
/// domain's rule wherever rounding occurs), for an even `divisor > 0` such
/// as a power of ten. One division; the remainder comes from `q * divisor`.
#[inline]
fn div_round_half_even(value: ComputeStorage, divisor: ComputeStorage) -> ComputeStorage {
    let q = value / divisor;
    let r = value - q * divisor; // sign of value, |r| < divisor
    let r_abs = decimal_compute_abs(r);
    // compare |r| with divisor - |r| (no doubling, so nothing can overflow)
    let bump = match r_abs.cmp(&(divisor - r_abs)) {
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => decimal_compute_halve(q) + decimal_compute_halve(q) != q,
    };
    if !bump {
        q
    } else if decimal_compute_is_negative(&value) {
        q - decimal_compute_unit()
    } else {
        q + decimal_compute_unit()
    }
}

/// The raw integer 1 (one unit at the compute dp, not the value 1.0).
#[inline]
fn decimal_compute_unit() -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { 1i64 }
    #[cfg(table_format = "q32_32")]
    { 1i128 }
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(1) }
    #[cfg(table_format = "q128_128")]
    { I512::from_i128(1) }
    #[cfg(table_format = "q256_256")]
    { I1024::from_i128(1) }
}

/// Widen a BinaryStorage-typed scaled decimal value to ComputeStorage (no rescale).
#[inline]
fn binary_storage_to_compute_widen(scaled: BinaryStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { scaled as i64 }
    #[cfg(table_format = "q32_32")]
    { scaled as i128 }
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(scaled) }
    #[cfg(table_format = "q128_128")]
    { I512::from_i256(scaled) }
    #[cfg(table_format = "q256_256")]
    { I1024::from_i512(scaled) }
}

/// Downscale compute-tier decimal value to storage-tier (BinaryStorage, dp).
///
/// The caller provides `target_dp`: the number of decimal places the result should have.
/// Rounding is half to even (banker's); it was half away from zero before 0.6.4.
///
/// Returns `Err(TierOverflow)` if the result doesn't fit in BinaryStorage.
#[inline]
pub fn decimal_downscale_to_storage(compute_val: ComputeStorage, target_dp: u8) -> Result<BinaryStorage, OverflowDetected> {
    if target_dp > DECIMAL_COMPUTE_DP {
        // User wants more precision than we have — upscale the value.
        // This only makes sense if target_dp ≤ DECIMAL_STORAGE_MAX_DP.
        if target_dp > DECIMAL_STORAGE_MAX_DP {
            return Err(OverflowDetected::Overflow);
        }
        let diff = target_dp - DECIMAL_COMPUTE_DP;
        let factor = pow10_compute_ct(diff);
        let scaled = compute_val * factor;
        return narrow_compute_to_storage(scaled);
    }

    if target_dp == DECIMAL_COMPUTE_DP {
        return narrow_compute_to_storage(compute_val);
    }

    // target_dp < compute_dp: divide, rounding half to even
    let diff = DECIMAL_COMPUTE_DP - target_dp;
    let divisor = pow10_compute_ct(diff);
    narrow_compute_to_storage(div_round_half_even(compute_val, divisor))
}

/// Narrow ComputeStorage to BinaryStorage with overflow check.
#[inline]
fn narrow_compute_to_storage(v: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    #[cfg(table_format = "q16_16")]
    {
        if v > i32::MAX as i64 || v < i32::MIN as i64 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(v as i32)
    }
    #[cfg(table_format = "q32_32")]
    {
        if v > i64::MAX as i128 || v < i64::MIN as i128 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(v as i64)
    }
    #[cfg(table_format = "q64_64")]
    {
        if !v.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(v.as_i128())
    }
    #[cfg(table_format = "q128_128")]
    {
        if !v.fits_in_i256() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(v.as_i256())
    }
    #[cfg(table_format = "q256_256")]
    {
        if !v.fits_in_i512() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(v.as_i512())
    }
}

// ============================================================================
// i128 ↔ ComputeStorage CONVERSIONS (for DecimalFixed imperative path)
// ============================================================================

/// Widen an i128 value to ComputeStorage (no rescale, just type widening).
///
/// Panics on realtime when `value` does not fit the i64 compute tier (the
/// cast truncated it silently before 0.6.4).
#[inline]
pub fn i128_to_compute(value: i128) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { i64::try_from(value).expect("i128_to_compute: value outside the realtime compute tier") }
    #[cfg(table_format = "q32_32")]
    { value }
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(value) }
    #[cfg(table_format = "q128_128")]
    { I512::from_i128(value) }
    #[cfg(table_format = "q256_256")]
    { I1024::from_i128(value) }
}

/// Upscale an i128 decimal value at `dp` decimal places to ComputeStorage at DECIMAL_COMPUTE_DP.
///
/// For DecimalFixed<DECIMALS>: value is `real × 10^dp`. Result is `real × 10^DECIMAL_COMPUTE_DP`.
/// Panics when the value does not fit the decimal compute tier; see
/// [`try_i128_upscale_to_compute`].
pub fn i128_upscale_to_compute(value: i128, dp: u8) -> ComputeStorage {
    try_i128_upscale_to_compute(value, dp)
        .expect("DecimalFixed: value outside the decimal compute range")
}

/// Upscale an i128 decimal value at `dp` decimal places to the compute dp,
/// `Err(TierOverflow)` when it does not fit the compute tier (realtime and
/// compact only: every i128 fits the wider tiers at every dp).
///
/// A `dp` beyond the compute dp rounds half to even (the decimal rule), in
/// i128 before the value is narrowed. Before 0.6.4 realtime cast the i128 to
/// i64 first (silently dropping the high bits), the upscale multiply was
/// unchecked, and the division truncated.
pub fn try_i128_upscale_to_compute(value: i128, dp: u8) -> Result<ComputeStorage, OverflowDetected> {
    if dp > DECIMAL_COMPUTE_DP {
        let diff = (dp - DECIMAL_COMPUTE_DP) as u32;
        // |value| < 2^127 < 10^39 / 2, so any diff >= 39 rounds to zero
        let q = if diff > 38 {
            0
        } else {
            let d = pow10_i128(diff);
            super::super::banker_round_decimal_i128(value / d, value % d, d)
        };
        return try_i128_to_compute(q);
    }
    let widened = try_i128_to_compute(value)?;
    if dp == DECIMAL_COMPUTE_DP {
        return Ok(widened);
    }
    let factor = pow10_compute_ct(DECIMAL_COMPUTE_DP - dp);
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    { widened.checked_mul(factor).ok_or(OverflowDetected::TierOverflow) }
    // |value| * 10^DECIMAL_COMPUTE_DP < 2^127 * 10^38 < 2^254 (embedded),
    // < 2^383 (balanced), < 2^639 (scientific): always inside the tier
    #[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
    { Ok(widened * factor) }
}

#[inline]
fn try_i128_to_compute(value: i128) -> Result<ComputeStorage, OverflowDetected> {
    #[cfg(table_format = "q16_16")]
    { i64::try_from(value).map_err(|_| OverflowDetected::TierOverflow) }
    #[cfg(not(table_format = "q16_16"))]
    { Ok(i128_to_compute(value)) }
}

/// Downscale ComputeStorage at DECIMAL_COMPUTE_DP to i128 at target `dp` decimal places.
///
/// Rounds half to even (banker's, the `DecimalFixed` rule; half away from
/// zero before 0.6.4). Panics when the result does not fit i128; see
/// [`try_decimal_compute_to_i128`].
pub fn decimal_compute_to_i128(value: ComputeStorage, dp: u8) -> i128 {
    try_decimal_compute_to_i128(value, dp)
        .expect("DecimalFixed: result outside the i128 storage range")
}

/// Downscale ComputeStorage at DECIMAL_COMPUTE_DP to i128 at `dp` decimal
/// places, half to even, `Err(TierOverflow)` when the result does not fit
/// i128. Before 0.6.4 the wide profiles narrowed with an unchecked cast, so
/// e.g. `DecimalFixed::<19>::exp(50)` on embedded returned a wrapped value.
pub fn try_decimal_compute_to_i128(value: ComputeStorage, dp: u8) -> Result<i128, OverflowDetected> {
    if dp > DECIMAL_COMPUTE_DP {
        // realtime and compact only (the wider compute dps exceed every i128 scale)
        let factor = pow10_i128((dp - DECIMAL_COMPUTE_DP) as u32);
        return compute_to_i128(value)?.checked_mul(factor).ok_or(OverflowDetected::TierOverflow);
    }
    let result = if dp < DECIMAL_COMPUTE_DP {
        div_round_half_even(value, pow10_compute_ct(DECIMAL_COMPUTE_DP - dp))
    } else {
        value
    };
    compute_to_i128(result)
}

/// A compute-tier value as i128, `Err(TierOverflow)` if it does not fit.
#[inline]
fn compute_to_i128(v: ComputeStorage) -> Result<i128, OverflowDetected> {
    #[cfg(table_format = "q16_16")]
    { Ok(v as i128) }
    #[cfg(table_format = "q32_32")]
    { Ok(v) }
    #[cfg(not(any(table_format = "q16_16", table_format = "q32_32")))]
    {
        if !v.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(v.as_i128())
    }
}

#[cfg(all(test, table_format = "q64_64"))]
mod tests {
    use super::*;

    #[test]
    fn scale_is_10_pow_38() {
        // 10^38 should equal decimal_compute_scale()
        let expected = pow10_compute_ct(38);
        assert_eq!(decimal_compute_scale(), expected);
    }

    #[test]
    fn one_times_one_is_one() {
        let one = decimal_compute_one();
        let result = decimal_compute_mul(one, one);
        assert_eq!(result, one);
    }

    #[test]
    fn two_times_three_is_six() {
        let two = decimal_compute_from_int(2);
        let three = decimal_compute_from_int(3);
        let six = decimal_compute_from_int(6);
        assert_eq!(decimal_compute_mul(two, three), six);
    }

    #[test]
    fn six_div_three_is_two() {
        let six = decimal_compute_from_int(6);
        let three = decimal_compute_from_int(3);
        let two = decimal_compute_from_int(2);
        assert_eq!(decimal_compute_div(six, three).unwrap(), two);
    }

    #[test]
    fn one_div_two_is_half() {
        let one = decimal_compute_one();
        let two = decimal_compute_from_int(2);
        let half = decimal_compute_div(one, two).unwrap();
        // half = 5 × 10^37
        let expected = pow10_compute_ct(37) * I256::from_i128(5);
        assert_eq!(half, expected);
    }

    #[test]
    fn neg_and_abs() {
        let two = decimal_compute_from_int(2);
        let neg_two = decimal_compute_neg(two);
        assert!(decimal_compute_is_negative(&neg_two));
        assert_eq!(decimal_compute_abs(neg_two), two);
    }

    #[test]
    fn upscale_dp_0_to_compute() {
        // 42 at dp=0 → 42 × 10^38 at compute tier
        let result = decimal_upscale_to_compute(42i128, 0).unwrap();
        let expected = decimal_compute_from_int(42);
        assert_eq!(result, expected);
    }

    #[test]
    fn upscale_dp_1_to_compute() {
        // 0.1 is stored as (scaled=1, dp=1). At compute tier it becomes 10^37.
        let result = decimal_upscale_to_compute(1i128, 1).unwrap();
        let expected = pow10_compute_ct(37);
        assert_eq!(result, expected);
    }

    #[test]
    fn downscale_to_storage_dp_2() {
        // 1.00 at compute tier (10^38) → dp=2 should give 100
        let one = decimal_compute_one();
        let result = decimal_downscale_to_storage(one, 2).unwrap();
        assert_eq!(result, 100i128);
    }

    #[test]
    fn downscale_rounding() {
        // 0.125 at compute tier → dp=2 gives 12 (half to even: 12.5 → 12),
        // 0.135 gives 14, -0.125 gives -12 (13 before 0.6.4, half away)
        // 0.125 × 10^38 = 125 × 10^35
        let val = I256::from_i128(125) * pow10_compute_ct(35);
        assert_eq!(decimal_downscale_to_storage(val, 2).unwrap(), 12i128);
        let val = I256::from_i128(135) * pow10_compute_ct(35);
        assert_eq!(decimal_downscale_to_storage(val, 2).unwrap(), 14i128);
        let val = I256::from_i128(-125) * pow10_compute_ct(35);
        assert_eq!(decimal_downscale_to_storage(val, 2).unwrap(), -12i128);
    }
}

#[cfg(all(test, any(table_format = "q128_128", table_format = "q256_256")))]
mod wide_div_scale_tests {
    use super::*;

    /// The limb-wise division by 10^DP is bit-identical to the general one.
    #[test]
    fn matches_the_general_division() {
        let mut state: u64 = 0x1234_5678_9ABC_DEF1;
        let scale = wide_scale();
        for case in 0..400 {
            let mut w = wide_scale();
            // random magnitudes from below 10^DP up to the product range
            let mut words = w.words;
            let used = case % words.len() + 1;
            for (i, word) in words.iter_mut().enumerate() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *word = if i < used { state } else { 0 };
            }
            let last = words.len() - 1;
            words[last] &= 0x3FFF_FFFF_FFFF_FFFF;
            w.words = words;
            for v in [w, -w, w >> 1, -(w >> 3)] {
                assert_eq!(wide_div_scale(v), wide_div(v, scale), "case {case}");
            }
        }
    }
}
