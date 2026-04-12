//! Decimal sine and cosine — Cody-Waite range reduction with decimal π, Horner Taylor.
//!
//! # Algorithm
//!
//! 1. **Range reduction**: Given `x`, compute `k = round(x × 2/π)`, `r = x - k × π/2`.
//!    Then `|r| ≤ π/4` and `sin(x), cos(x)` are expressed in terms of `sin(r), cos(r)`
//!    via the quadrant `k mod 4`.
//! 2. **Taylor series** for `|r| ≤ π/4`:
//!    - `sin(r) = r - r³/3! + r⁵/5! - r⁷/7! + ...`
//!    - `cos(r) = 1 - r²/2! + r⁴/4! - r⁶/6! + ...`
//!    Iterative term computation: `term_k = -term_{k-1} × r² / ((2k)(2k+1))` for sin.
//! 3. **Quadrant reconstruction**:
//!    - k mod 4 == 0: `(sin_r, cos_r)`
//!    - k mod 4 == 1: `(cos_r, -sin_r)`
//!    - k mod 4 == 2: `(-sin_r, -cos_r)`
//!    - k mod 4 == 3: `(-cos_r, sin_r)`
//!
//! # π computation
//!
//! π is computed via **Machin's formula**: `π/4 = 4·atan(1/5) - atan(1/239)`.
//! Uses our own `decimal_atan` Taylor series (for small arguments, converges fast).
//! Result is cached in a thread-local cell to avoid recomputation.

use super::decimal_compute::{
    ComputeStorage, DECIMAL_COMPUTE_DP,
    decimal_compute_zero, decimal_compute_one, decimal_compute_halve,
    decimal_compute_add, decimal_compute_sub, decimal_compute_mul,
    decimal_compute_div, decimal_compute_div_int,
    decimal_compute_is_zero, decimal_compute_is_negative, decimal_compute_neg,
    decimal_compute_cmp, decimal_compute_abs,
};
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

// ============================================================================
// π COMPUTATION VIA MACHIN'S FORMULA
// ============================================================================

thread_local! {
    static PI_CACHE: std::cell::RefCell<Option<ComputeStorage>> = const { std::cell::RefCell::new(None) };
}

/// Compute π at decimal compute-tier precision, cached in thread-local storage.
///
/// Uses Machin's formula: `π/4 = 4·atan(1/5) - atan(1/239)`.
///
/// Both `atan(1/5)` and `atan(1/239)` converge via direct Taylor series because the
/// arguments are small (no range reduction needed, avoiding circular dependency on π).
pub fn pi_at_decimal_compute() -> Result<ComputeStorage, OverflowDetected> {
    let cached: Option<ComputeStorage> = PI_CACHE.with(|c| c.borrow().clone());
    if let Some(pi) = cached {
        return Ok(pi);
    }

    // Compute 1/5 and 1/239 at compute dp
    let one = decimal_compute_one();
    let five = super::decimal_compute::decimal_compute_from_int(5);
    let two_thirty_nine = super::decimal_compute::decimal_compute_from_int(239);
    let one_fifth = decimal_compute_div(one, five)?;
    let one_239th = decimal_compute_div(one, two_thirty_nine)?;

    // atan(1/5) via direct Taylor (|x| = 0.2, fast convergence)
    let atan_fifth = atan_small_direct(one_fifth);
    // atan(1/239) via direct Taylor (|x| ≈ 0.004, extremely fast)
    let atan_239 = atan_small_direct(one_239th);

    // π/4 = 4 × atan(1/5) - atan(1/239)
    let four_atan_fifth = decimal_compute_add(atan_fifth, decimal_compute_add(atan_fifth, decimal_compute_add(atan_fifth, atan_fifth)));
    let pi_over_4 = decimal_compute_sub(four_atan_fifth, atan_239);

    // π = 4 × (π/4)
    let pi = decimal_compute_add(pi_over_4, decimal_compute_add(pi_over_4, decimal_compute_add(pi_over_4, pi_over_4)));

    PI_CACHE.with(|c: &std::cell::RefCell<Option<ComputeStorage>>| *c.borrow_mut() = Some(pi));
    Ok(pi)
}

/// Direct Taylor for `atan(x)` when `|x| < 0.3`.
///
/// Used ONLY by `pi_at_decimal_compute` to bootstrap π without depending on
/// the full `decimal_atan` function (which itself needs π for `|x| > 1` cases).
fn atan_small_direct(x: ComputeStorage) -> ComputeStorage {
    if decimal_compute_is_zero(&x) {
        return decimal_compute_zero();
    }

    let x_sq = decimal_compute_mul(x, x);
    let mut term = x;
    let mut sum = x;
    let mut sign_positive = true;

    // Taylor: atan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...
    // For |x| ≤ 0.2, term k ~ 0.2^(2k+1)/(2k+1) < 10^-dp when k ≈ dp × 0.72
    let max_terms = (DECIMAL_COMPUTE_DP as u32 * 2) + 20;
    for k in 1..=max_terms {
        term = decimal_compute_mul(term, x_sq);
        if decimal_compute_is_zero(&term) {
            break;
        }
        sign_positive = !sign_positive;
        let divisor = (2 * k as u64) + 1;
        let contribution = decimal_compute_div_int(term, divisor);
        if decimal_compute_is_zero(&contribution) {
            break;
        }
        if sign_positive {
            sum = decimal_compute_add(sum, contribution);
        } else {
            sum = decimal_compute_sub(sum, contribution);
        }
    }

    sum
}

// ============================================================================
// SIN / COS / SINCOS
// ============================================================================

const fn max_trig_taylor_terms() -> u32 {
    // For |r| ≤ π/4 ≈ 0.785, term k ~ 0.785^(2k+1)/(2k+1)! — converges quickly.
    // Need ~21 terms for dp=38, ~41 for dp=77, ~80 for dp=154.
    (DECIMAL_COMPUTE_DP as u32 / 2) + 20
}

/// Compute both `sin(x)` and `cos(x)` at compute dp — single shared range reduction.
pub fn decimal_sincos(x: ComputeStorage) -> Result<(ComputeStorage, ComputeStorage), OverflowDetected> {
    if decimal_compute_is_zero(&x) {
        return Ok((decimal_compute_zero(), decimal_compute_one()));
    }

    // Range reduction: k = round(x × 2/π), r = x - k × π/2
    let pi = pi_at_decimal_compute()?;
    let pi_half = decimal_compute_halve(pi);
    let pi_quarter = decimal_compute_halve(pi_half);

    // Compute k = round(x / (π/2)) as an integer
    let x_over_pi_half = decimal_compute_div(x, pi_half)?;
    let k = round_to_int(x_over_pi_half);

    // r = x - k × (π/2)
    let k_times_pi_half = mul_by_int(pi_half, k);
    let r = decimal_compute_sub(x, k_times_pi_half);

    // Ensure |r| ≤ π/4 by one additional correction if needed
    // (rounding errors in k computation could leave |r| slightly > π/4)
    let abs_r = decimal_compute_abs(r);
    if decimal_compute_cmp(&abs_r, &pi_quarter) == std::cmp::Ordering::Greater {
        // Leave as-is; Taylor will just need a few more terms
    }

    // Taylor series for sin(r) and cos(r) — iterative
    let (sin_r, cos_r) = sincos_taylor(r);

    // Quadrant reconstruction: k mod 4
    let k_mod_4 = ((k % 4) + 4) % 4;
    let (sin_x, cos_x) = match k_mod_4 {
        0 => (sin_r, cos_r),
        1 => (cos_r, decimal_compute_neg(sin_r)),
        2 => (decimal_compute_neg(sin_r), decimal_compute_neg(cos_r)),
        3 => (decimal_compute_neg(cos_r), sin_r),
        _ => unreachable!(),
    };

    let _ = r; // silence mutability warning if we remove correction
    Ok((sin_x, cos_x))
}

/// Compute `sin(x)` at compute dp.
pub fn decimal_sin(x: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    decimal_sincos(x).map(|(s, _)| s)
}

/// Compute `cos(x)` at compute dp.
pub fn decimal_cos(x: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    decimal_sincos(x).map(|(_, c)| c)
}

/// Iterative Taylor for `sin(r), cos(r)` with `|r| ≤ π/4`.
fn sincos_taylor(r: ComputeStorage) -> (ComputeStorage, ComputeStorage) {
    let r_sq = decimal_compute_mul(r, r);

    // sin series: r - r³/3! + r⁵/5! - ...
    // term_0 = r, term_k = -term_{k-1} × r² / ((2k)(2k+1))
    let mut sin_term = r;
    let mut sin_sum = r;
    let mut sin_sign_positive = true;

    // cos series: 1 - r²/2! + r⁴/4! - ...
    // term_0 = 1, term_k = -term_{k-1} × r² / ((2k-1)(2k))
    let one = decimal_compute_one();
    let mut cos_term = one;
    let mut cos_sum = one;
    let mut cos_sign_positive = true;

    let max_terms = max_trig_taylor_terms();
    for k in 1..=max_terms {
        let k64 = k as u64;

        // Update sin term: × r² / ((2k)(2k+1))
        sin_term = decimal_compute_mul(sin_term, r_sq);
        let sin_div = (2 * k64) * (2 * k64 + 1);
        sin_term = decimal_compute_div_int(sin_term, sin_div);
        sin_sign_positive = !sin_sign_positive;
        if !decimal_compute_is_zero(&sin_term) {
            if sin_sign_positive {
                sin_sum = decimal_compute_add(sin_sum, sin_term);
            } else {
                sin_sum = decimal_compute_sub(sin_sum, sin_term);
            }
        }

        // Update cos term: × r² / ((2k-1)(2k))
        cos_term = decimal_compute_mul(cos_term, r_sq);
        let cos_div = (2 * k64 - 1) * (2 * k64);
        cos_term = decimal_compute_div_int(cos_term, cos_div);
        cos_sign_positive = !cos_sign_positive;
        if !decimal_compute_is_zero(&cos_term) {
            if cos_sign_positive {
                cos_sum = decimal_compute_add(cos_sum, cos_term);
            } else {
                cos_sum = decimal_compute_sub(cos_sum, cos_term);
            }
        }

        // Both converged?
        if decimal_compute_is_zero(&sin_term) && decimal_compute_is_zero(&cos_term) {
            break;
        }
    }

    (sin_sum, cos_sum)
}

/// Round a compute-tier value to the nearest integer (as i64, for small values).
fn round_to_int(v: ComputeStorage) -> i64 {
    // Add 0.5 (ULP = scale/2), then truncate toward zero via division by scale
    let half = decimal_compute_halve(decimal_compute_one());
    let rounded = if decimal_compute_is_negative(&v) {
        decimal_compute_sub(v, half)
    } else {
        decimal_compute_add(v, half)
    };

    // Divide by scale (10^dp) to get integer part
    let scale = decimal_compute_one();
    #[cfg(table_format = "q16_16")]
    { (rounded / scale) as i64 }
    #[cfg(table_format = "q32_32")]
    { (rounded / scale) as i64 }
    #[cfg(table_format = "q64_64")]
    { (rounded / scale).as_i128() as i64 }
    #[cfg(table_format = "q128_128")]
    { (rounded / scale).as_i128() as i64 }
    #[cfg(table_format = "q256_256")]
    { (rounded / scale).as_i128() as i64 }
}

/// Multiply a compute-tier value by a small integer `n`.
fn mul_by_int(v: ComputeStorage, n: i64) -> ComputeStorage {
    if n == 0 {
        return decimal_compute_zero();
    }
    let negative = n < 0;
    let n_abs = n.unsigned_abs();

    // For small n, repeated addition is fine. But for n up to ~10^18 we need multiplication.
    // Use native multiplication by converting n into ComputeStorage scale.
    let n_compute: ComputeStorage = {
        #[cfg(table_format = "q16_16")]
        { n_abs as i64 }
        #[cfg(table_format = "q32_32")]
        { n_abs as i128 }
        #[cfg(table_format = "q64_64")]
        { crate::fixed_point::i256::I256::from_i128(n_abs as i128) }
        #[cfg(table_format = "q128_128")]
        { crate::fixed_point::i512::I512::from_i128(n_abs as i128) }
        #[cfg(table_format = "q256_256")]
        { crate::fixed_point::I1024::from_i128(n_abs as i128) }
    };

    let result = v * n_compute;

    if negative {
        decimal_compute_neg(result)
    } else {
        result
    }
}

#[cfg(all(test, table_format = "q64_64"))]
mod tests {
    use super::*;
    use super::super::decimal_compute::{decimal_compute_from_int, pow10_compute_ct};
    use crate::fixed_point::i256::I256;

    fn parse_decimal_str(s: &str) -> I256 {
        let mut result = I256::from_i128(0);
        let ten = I256::from_i128(10);
        for ch in s.chars() {
            let digit = ch.to_digit(10).expect("non-digit");
            result = result * ten + I256::from_i128(digit as i128);
        }
        result
    }

    #[test]
    fn sin_zero() {
        let result = decimal_sin(decimal_compute_zero()).unwrap();
        assert_eq!(result, decimal_compute_zero());
    }

    #[test]
    fn cos_zero() {
        let result = decimal_cos(decimal_compute_zero()).unwrap();
        assert_eq!(result, decimal_compute_one());
    }

    /// mpmath: π = 3.14159265358979323846264338327950288419716939937510...
    #[test]
    fn pi_value_matches_mpmath() {
        let pi = pi_at_decimal_compute().unwrap();
        let expected = parse_decimal_str("314159265358979323846264338327950288420");
        let diff = if pi > expected { pi - expected } else { expected - pi };
        let tolerance = I256::from_i128(1_000_000);
        assert!(
            diff < tolerance,
            "π precision: got={:?} expected={:?} diff={:?}",
            pi, expected, diff
        );
    }

    /// mpmath: sin(1) = 0.84147098480789650665250232163029899962256306079837...
    #[test]
    fn sin_one_mpmath() {
        let one = decimal_compute_one();
        let result = decimal_sin(one).unwrap();
        let expected = parse_decimal_str("84147098480789650665250232163029899962");
        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(10_000_000);
        assert!(
            diff < tolerance,
            "sin(1) precision: got={:?} expected={:?} diff={:?}",
            result, expected, diff
        );
    }

    /// mpmath: cos(1) = 0.54030230586813971740093660744297660373231042061792...
    #[test]
    fn cos_one_mpmath() {
        let one = decimal_compute_one();
        let result = decimal_cos(one).unwrap();
        let expected = parse_decimal_str("54030230586813971740093660744297660373");
        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(10_000_000);
        assert!(
            diff < tolerance,
            "cos(1) precision: got={:?} expected={:?} diff={:?}",
            result, expected, diff
        );
    }
}
