//! Decimal compute-tier primitives: arithmetic on values scaled by 10^C::DP.
//!
//! All values in this module are interpreted as `raw_int × 10^(-C::DP)`.
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
#[allow(unused_imports)]
use super::word::DecimalWord;
#[allow(unused_imports)]
use crate::fixed_point::width::WideInt;
#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;
#[cfg(table_format = "q256_256")]
use crate::fixed_point::I2048;

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
// WIDE COMPUTE TYPE — one tier wider than C for widening multiply
// ============================================================================

#[cfg(table_format = "q16_16")]
#[allow(dead_code)]
pub(crate) type WideCompute = i128;
#[cfg(table_format = "q32_32")]
#[allow(dead_code)]
pub(crate) type WideCompute = I256;
#[cfg(table_format = "q64_64")]
#[allow(dead_code)]
pub(crate) type WideCompute = I512;
#[cfg(table_format = "q128_128")]
#[allow(dead_code)]
pub(crate) type WideCompute = I1024;
#[cfg(table_format = "q256_256")]
#[allow(dead_code)]
pub(crate) type WideCompute = I2048;

// ============================================================================
// CONSTANTS — zero, one, scale factor (10^C::DP)
// ============================================================================

/// Compute-tier `0`.
#[inline]
pub fn decimal_compute_zero<C: DecimalWord>() -> C {
    C::decimal_compute_zero()
}

/// Compute-tier `1.0` = `10^C::DP`.
#[inline]
pub fn decimal_compute_scale<C: DecimalWord>() -> C {
    C::pow10_compute_ct(C::DP)
}

/// Alias for `decimal_compute_scale`: represents the value 1.0 at compute dp.
#[inline]
pub fn decimal_compute_one<C: DecimalWord>() -> C {
    decimal_compute_scale::<C>()
}

/// Convert a small integer (abs value ≤ 2^62) to decimal compute tier.
#[inline]
pub fn decimal_compute_from_int<C: DecimalWord>(n: i64) -> C {
    C::decimal_compute_from_int(n)
}

#[cfg(test)]
#[cfg(test)]
#[allow(dead_code)]
/// Wide-tier scale factor `10^C::DP` in the widening type.
#[inline]
fn wide_scale<C: DecimalWord>() -> C::Wide {
    C::wide_scale()
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

/// 10^exp in C type. Valid for exp ≤ C::DP.
#[inline]
#[allow(dead_code)]
pub(crate) fn pow10_compute_ct<C: DecimalWord>(exp: u8) -> C {
    C::pow10_compute_ct(exp)
}

// ============================================================================
// PROFILE-SPECIFIC WIDENING / NARROWING
// ============================================================================

#[cfg(test)]
#[cfg(test)]
#[allow(dead_code)]
/// Divide at wide tier (handles I2048 via free function).
#[inline]
fn wide_div<C: DecimalWord>(a: C::Wide, b: C::Wide) -> C::Wide {
    C::wide_div(a, b)
}

#[cfg(test)]
#[cfg(test)]
#[allow(dead_code)]
/// `w / 10^C::DP` truncated toward zero: bit-identical to
/// `C::wide_div(w, C::wide_scale())`. On balanced and scientific the wide integers
/// divide bit by bit (2048 steps for I2048, about 69 us a call in release),
/// once per decimal multiply, which made FASC decimal transcendentals about
/// 570x slower than the binary engine there. Dividing the magnitude by 10^19
/// limb by limb, ceil(DP / 19) times, is O(words) per pass:
/// floor(floor(x / a) / b) = floor(x / (a b)) for x >= 0.
#[inline]
fn wide_div_scale<C: DecimalWord>(w: C::Wide) -> C::Wide {
    C::wide_div_scale(w)
}

// ============================================================================
// ARITHMETIC PRIMITIVES — add, sub, neg, halve
// ============================================================================

/// Add two compute-tier decimal values.
#[inline]
pub fn decimal_compute_add<C: DecimalWord>(a: C, b: C) -> C {
    try_decimal_compute_add(a, b).expect("decimal compute tier: addition overflow")
}

/// `a + b` of compute-tier decimal values, `Err(TierOverflow)` when the sum
/// leaves the compute tier (0.6.3 wrapped).
#[inline]
pub fn try_decimal_compute_add<C: DecimalWord>(a: C, b: C) -> Result<C, OverflowDetected> {
    a.checked_add_wide(b).ok_or(OverflowDetected::TierOverflow)
}

/// Subtract two compute-tier decimal values.
#[inline]
pub fn decimal_compute_sub<C: DecimalWord>(a: C, b: C) -> C {
    try_decimal_compute_sub(a, b).expect("decimal compute tier: subtraction overflow")
}

/// `a - b` of compute-tier decimal values, `Err(TierOverflow)` when the
/// difference leaves the compute tier (0.6.3 wrapped).
#[inline]
pub fn try_decimal_compute_sub<C: DecimalWord>(a: C, b: C) -> Result<C, OverflowDetected> {
    a.checked_add_wide(C::try_decimal_compute_neg(b)?).ok_or(OverflowDetected::TierOverflow)
}

/// `-a` of a compute-tier decimal value, `Err(TierOverflow)` for the tier's
/// minimum.
#[inline]
pub fn try_decimal_compute_neg<C: DecimalWord>(a: C) -> Result<C, OverflowDetected> {
    C::try_decimal_compute_neg(a)
}

/// Negate a compute-tier decimal value.
#[inline]
pub fn decimal_compute_neg<C: DecimalWord>(a: C) -> C {
    C::try_decimal_compute_neg(a).expect("decimal compute tier: negation overflow")
}

/// Divide compute-tier value by 2 (exact for even values, rounds toward -∞ for odd).
#[inline]
pub fn decimal_compute_halve<C: DecimalWord>(a: C) -> C {
    a.shr_bits(1)
}

/// Absolute value at compute tier.
#[inline]
pub fn decimal_compute_abs<C: DecimalWord>(a: C) -> C {
    if C::decimal_compute_is_negative(&a) {
        decimal_compute_neg(a)
    } else {
        a
    }
}

#[inline]
pub fn decimal_compute_is_zero<C: DecimalWord>(a: &C) -> bool {
    C::decimal_compute_is_zero(a)
}

#[inline]
pub fn decimal_compute_is_negative<C: DecimalWord>(a: &C) -> bool {
    C::decimal_compute_is_negative(a)
}

#[inline]
pub fn decimal_compute_cmp<C: DecimalWord>(a: &C, b: &C) -> std::cmp::Ordering {
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
pub fn decimal_compute_mul<C: DecimalWord>(a: C, b: C) -> C {
    try_decimal_compute_mul(a, b).expect("decimal_compute_mul: product outside the decimal compute tier")
}

/// Multiply two compute-tier decimal values, rescaling back to compute dp,
/// `Err(TierOverflow)` when the product does not fit the compute tier.
///
/// Algorithm:
/// 1. Widening multiply: `product = a × b` in the wide type (no overflow).
/// 2. Divide by `10^C::DP` with round-half-away-from-zero.
/// 3. Narrow back to C (checked).
#[inline]
pub fn try_decimal_compute_mul<C: DecimalWord>(a: C, b: C) -> Result<C, OverflowDetected> {
    let product = C::wide_mul(a, b);
    let scale = C::wide_scale();
    let half_scale = C::wide_halve(scale);

    // Round-half-away-from-zero: for positive products add half, for negative subtract half.
    let rounded = if C::wide_is_negative(&product) {
        product - half_scale
    } else {
        product + half_scale
    };

    C::narrow_from_wide(C::wide_div_scale(rounded))
}

// ============================================================================
// DIVIDE — (a × 10^compute_dp) / b with rounding
// ============================================================================

/// Divide two compute-tier decimal values: `a / b`, result at compute dp.
///
/// Formula: `result = round((a × 10^dp) / b)`. Uses wide type for the numerator.
/// `Err(TierOverflow)` when the quotient does not fit the compute tier.
#[inline]
pub fn decimal_compute_div<C: DecimalWord>(a: C, b: C) -> Result<C, OverflowDetected> {
    if C::decimal_compute_is_zero(&b) {
        return Err(OverflowDetected::DomainError);
    }

    // numerator = a × 10^dp (in wide type)
    let numerator = C::wide_mul(a, decimal_compute_scale::<C>());
    let divisor = C::widen_to_wide(b);

    // Round-half-away-from-zero
    let half_divisor = C::wide_halve(if C::wide_is_negative(&divisor) { C::wide_neg(divisor) } else { divisor });
    let result_neg = C::wide_is_negative(&numerator) ^ C::wide_is_negative(&divisor);
    let num_abs = if C::wide_is_negative(&numerator) { C::wide_neg(numerator) } else { numerator };
    let div_abs = if C::wide_is_negative(&divisor) { C::wide_neg(divisor) } else { divisor };

    let rounded_num = num_abs + half_divisor;
    let quotient = C::wide_div(rounded_num, div_abs);

    let final_quot = if result_neg { C::wide_neg(quotient) } else { quotient };

    C::narrow_from_wide(final_quot)
}

/// Divide compute-tier value by a small positive integer, with round-half-away-from-zero.
///
/// Fast path for Taylor series where each term is divided by `n` (term index).
#[inline]
pub fn decimal_compute_div_int<C: DecimalWord>(a: C, n: u64) -> C {
    C::decimal_compute_div_int(a, n)
}

// ============================================================================
// UPSCALE / DOWNSCALE — bridge between storage dp and compute dp
// ============================================================================

/// Upscale a Decimal StackValue component (dp, scaled) to compute-tier decimal format.
///
/// - If `storage_dp < C::DP`: multiply by `10^(compute_dp - storage_dp)`.
/// - If `storage_dp == C::DP`: widen only.
/// - If `storage_dp > C::DP`: divide (precision loss possible: rare case).
#[inline]
pub fn decimal_upscale_to_compute<C: DecimalWord>(scaled: C::Narrow, storage_dp: u8) -> Result<C, OverflowDetected> {
    if storage_dp == C::DP {
        return Ok(C::binary_storage_to_compute_widen(scaled));
    }

    if storage_dp < C::DP {
        let diff = C::DP - storage_dp;
        if diff > C::DP {
            return Err(OverflowDetected::Overflow);
        }
        let factor = C::pow10_compute_ct(diff);
        let widened = C::binary_storage_to_compute_widen(scaled);
        // Check that widened × factor fits in C.
        // For our profiles this is always true because storage_dp ≤ C::STORAGE_MAX_DP
        // and compute_dp ≤ 2 × C::STORAGE_MAX_DP, so the product is bounded.
        return Ok(widened * factor);
    }

    // storage_dp > compute_dp: divide, rounding half to even (the decimal rule).
    let diff = storage_dp - C::DP;
    let divisor = C::pow10_compute_ct(diff);
    let widened = C::binary_storage_to_compute_widen(scaled);
    Ok(div_round_half_even(widened, divisor))
}

/// `value / divisor` rounded to nearest, ties to even (banker's, the decimal
/// domain's rule wherever rounding occurs), for an even `divisor > 0` such
/// as a power of ten. One division; the remainder comes from `q * divisor`.
#[inline]
fn div_round_half_even<C: DecimalWord>(value: C, divisor: C) -> C {
    let q = value.divmod(divisor).0;
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
    } else if C::decimal_compute_is_negative(&value) {
        q - C::decimal_compute_unit()
    } else {
        q + C::decimal_compute_unit()
    }
}

/// Downscale compute-tier decimal value to storage-tier (C::Narrow, dp).
///
/// The caller provides `target_dp`: the number of decimal places the result should have.
/// Rounding is half to even (banker's); it was half away from zero before 0.6.4.
///
/// Returns `Err(TierOverflow)` if the result doesn't fit in C::Narrow.
#[inline]
pub fn decimal_downscale_to_storage<C: DecimalWord>(compute_val: C, target_dp: u8) -> Result<C::Narrow, OverflowDetected> {
    if target_dp > C::DP {
        // User wants more precision than we have — upscale the value.
        // This only makes sense if target_dp ≤ C::STORAGE_MAX_DP.
        if target_dp > C::STORAGE_MAX_DP {
            return Err(OverflowDetected::Overflow);
        }
        let diff = target_dp - C::DP;
        let factor = C::pow10_compute_ct(diff);
        let scaled = compute_val * factor;
        return C::narrow_compute_to_storage(scaled);
    }

    if target_dp == C::DP {
        return C::narrow_compute_to_storage(compute_val);
    }

    // target_dp < compute_dp: divide, rounding half to even
    let diff = C::DP - target_dp;
    let divisor = C::pow10_compute_ct(diff);
    C::narrow_compute_to_storage(div_round_half_even(compute_val, divisor))
}

// ============================================================================
// i128 ↔ C CONVERSIONS (for DecimalFixed imperative path)
// ============================================================================

/// Widen an i128 value to C (no rescale, just type widening).
///
/// Panics on realtime when `value` does not fit the i64 compute tier (the
/// cast truncated it silently before 0.6.4).
#[inline]
pub fn i128_to_compute<C: DecimalWord>(value: i128) -> C {
    C::i128_to_compute(value)
}

/// Upscale an i128 decimal value at `dp` decimal places to C at C::DP.
///
/// For DecimalFixed<DECIMALS>: value is `real × 10^dp`. Result is `real × 10^C::DP`.
/// Panics when the value does not fit the decimal compute tier; see
/// [`try_i128_upscale_to_compute`].
pub fn i128_upscale_to_compute<C: DecimalWord>(value: i128, dp: u8) -> C {
    C::try_i128_upscale_to_compute(value, dp)
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
pub fn try_i128_upscale_to_compute<C: DecimalWord>(value: i128, dp: u8) -> Result<C, OverflowDetected> {
    C::try_i128_upscale_to_compute(value, dp)
}

/// Downscale C at C::DP to i128 at target `dp` decimal places.
///
/// Rounds half to even (banker's, the `DecimalFixed` rule; half away from
/// zero before 0.6.4). Panics when the result does not fit i128; see
/// [`try_decimal_compute_to_i128`].
pub fn decimal_compute_to_i128<C: DecimalWord>(value: C, dp: u8) -> i128 {
    try_decimal_compute_to_i128::<C>(value, dp)
        .expect("DecimalFixed: result outside the i128 storage range")
}

/// Downscale C at C::DP to i128 at `dp` decimal
/// places, half to even, `Err(TierOverflow)` when the result does not fit
/// i128. Before 0.6.4 the wide profiles narrowed with an unchecked cast, so
/// e.g. `DecimalFixed::<19>::exp(50)` on embedded returned a wrapped value.
pub fn try_decimal_compute_to_i128<C: DecimalWord>(value: C, dp: u8) -> Result<i128, OverflowDetected> {
    if dp > C::DP {
        // realtime and compact only (the wider compute dps exceed every i128 scale)
        let factor = pow10_i128((dp - C::DP) as u32);
        return C::compute_to_i128(value)?.checked_mul(factor).ok_or(OverflowDetected::TierOverflow);
    }
    let result = if dp < C::DP {
        div_round_half_even(value, C::pow10_compute_ct(C::DP - dp))
    } else {
        value
    };
    C::compute_to_i128(result)
}

/// [`try_decimal_compute_to_i128`] that refuses to round when the digits
/// it would discard lie within a margin of a rounding boundary (zero, a
/// half, or a whole unit at `dp`): `None` means the caller must recompute
/// on a wider word. The margin is 10^-4 of the guard range, which is about
/// a thousand times the engines' measured error at their last digit; with
/// fewer than five guard digits every rounding counts as ambiguous.
pub fn try_decimal_compute_to_i128_unless_ambiguous<C: DecimalWord>(value: C, dp: u8) -> Option<Result<i128, OverflowDetected>> {
    if dp >= C::DP {
        return Some(try_decimal_compute_to_i128::<C>(value, dp));
    }
    let guard = C::DP - dp;
    if guard < 5 {
        return None;
    }
    // one power of ten (a loop of wide multiplications), the divisor from it
    let margin = C::pow10_compute_ct(guard - 4);
    let divisor = margin * C::from_i64(10_000);
    let q = value.divmod(divisor).0;
    let r = decimal_compute_abs(value - q * divisor); // |r| < divisor
    let d_minus_r = divisor - r;
    // |2r - divisor| without doubling
    let half_gap = if r < d_minus_r { d_minus_r - r } else { r - d_minus_r };
    if r < margin || d_minus_r < margin || half_gap < margin + margin {
        return None;
    }
    // round half to even from the quotient and remainder in hand (one wide
    // division, not a second one in `div_round_half_even`); the tie itself
    // is inside the margin, so only the strict comparison decides
    let rounded = if r > d_minus_r {
        if C::decimal_compute_is_negative(&value) { q - C::decimal_compute_unit() } else { q + C::decimal_compute_unit() }
    } else {
        q
    };
    Some(C::compute_to_i128(rounded))
}

#[cfg(all(test, table_format = "q64_64"))]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use super::super::decimal_compute::*;

    #[test]
    fn scale_is_10_pow_38() {
        // 10^38 should equal decimal_compute_scale::<ComputeStorage>()
        let expected = pow10_compute_ct::<ComputeStorage>(38);
        assert_eq!(decimal_compute_scale::<ComputeStorage>(), expected);
    }

    #[test]
    fn one_times_one_is_one() {
        let one = decimal_compute_one::<ComputeStorage>();
        let result = decimal_compute_mul(one, one);
        assert_eq!(result, one);
    }

    #[test]
    fn two_times_three_is_six() {
        let two = decimal_compute_from_int::<ComputeStorage>(2);
        let three = decimal_compute_from_int::<ComputeStorage>(3);
        let six = decimal_compute_from_int::<ComputeStorage>(6);
        assert_eq!(decimal_compute_mul(two, three), six);
    }

    #[test]
    fn six_div_three_is_two() {
        let six = decimal_compute_from_int::<ComputeStorage>(6);
        let three = decimal_compute_from_int::<ComputeStorage>(3);
        let two = decimal_compute_from_int::<ComputeStorage>(2);
        assert_eq!(decimal_compute_div(six, three).unwrap(), two);
    }

    #[test]
    fn one_div_two_is_half() {
        let one = decimal_compute_one::<ComputeStorage>();
        let two = decimal_compute_from_int::<ComputeStorage>(2);
        let half = decimal_compute_div(one, two).unwrap();
        // half = 5 × 10^37
        let expected = pow10_compute_ct::<ComputeStorage>(37) * I256::from_i128(5);
        assert_eq!(half, expected);
    }

    #[test]
    fn neg_and_abs() {
        let two = decimal_compute_from_int::<ComputeStorage>(2);
        let neg_two = decimal_compute_neg(two);
        assert!(decimal_compute_is_negative(&neg_two));
        assert_eq!(decimal_compute_abs(neg_two), two);
    }

    #[test]
    fn upscale_dp_0_to_compute() {
        // 42 at dp=0 → 42 × 10^38 at compute tier
        let result = decimal_upscale_to_compute::<ComputeStorage>(42i128, 0).unwrap();
        let expected = decimal_compute_from_int::<ComputeStorage>(42);
        assert_eq!(result, expected);
    }

    #[test]
    fn upscale_dp_1_to_compute() {
        // 0.1 is stored as (scaled=1, dp=1). At compute tier it becomes 10^37.
        let result = decimal_upscale_to_compute::<ComputeStorage>(1i128, 1).unwrap();
        let expected = pow10_compute_ct::<ComputeStorage>(37);
        assert_eq!(result, expected);
    }

    #[test]
    fn downscale_to_storage_dp_2() {
        // 1.00 at compute tier (10^38) → dp=2 should give 100
        let one = decimal_compute_one::<ComputeStorage>();
        let result = decimal_downscale_to_storage::<ComputeStorage>(one, 2).unwrap();
        assert_eq!(result, 100i128);
    }

    #[test]
    fn downscale_rounding() {
        // 0.125 at compute tier → dp=2 gives 12 (half to even: 12.5 → 12),
        // 0.135 gives 14, -0.125 gives -12 (13 before 0.6.4, half away)
        // 0.125 × 10^38 = 125 × 10^35
        let val = I256::from_i128(125) * pow10_compute_ct::<ComputeStorage>(35);
        assert_eq!(decimal_downscale_to_storage::<ComputeStorage>(val, 2).unwrap(), 12i128);
        let val = I256::from_i128(135) * pow10_compute_ct::<ComputeStorage>(35);
        assert_eq!(decimal_downscale_to_storage::<ComputeStorage>(val, 2).unwrap(), 14i128);
        let val = I256::from_i128(-125) * pow10_compute_ct::<ComputeStorage>(35);
        assert_eq!(decimal_downscale_to_storage::<ComputeStorage>(val, 2).unwrap(), -12i128);
    }
}

#[cfg(all(test, any(table_format = "q128_128", table_format = "q256_256")))]
mod wide_div_scale_tests {
    use super::*;
    #[allow(unused_imports)]
    use super::super::decimal_compute::*;

    /// The limb-wise division by 10^DP is bit-identical to the general one.
    #[test]
    fn matches_the_general_division() {
        let mut state: u64 = 0x1234_5678_9ABC_DEF1;
        let scale = wide_scale::<ComputeStorage>();
        for case in 0..400 {
            let mut w = wide_scale::<ComputeStorage>();
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
                assert_eq!(wide_div_scale::<ComputeStorage>(v), wide_div::<ComputeStorage>(v, scale), "case {case}");
            }
        }
    }
}
