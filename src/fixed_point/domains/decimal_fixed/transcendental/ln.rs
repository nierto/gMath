//! Decimal natural logarithm: `ln(x)` at compute dp.
//!
//! # Algorithm
//!
//! 1. **Domain check**: `x > 0` or `DomainError`.
//! 2. **Range reduction** (halving/doubling): reduce `x` to `m ∈ [1, 2)` by extracting
//!    `k` such that `x = 2^k × m`. Then `ln(x) = k × ln(2) + ln(m)`.
//! 3. **Core series** for `m ∈ [1, 2)`: use `atanh` series with `s = (m-1)/(m+1) ∈ [0, 1/3)`:
//!    `ln(m) = 2 × (s + s³/3 + s⁵/5 + s⁷/7 + ...)`
//!    For `|s| ≤ 1/3`, each term shrinks by factor `1/9`, giving very fast convergence.
//! 4. **`ln(2)` constant**: computed once via the same `atanh` series applied to `1/3`:
//!    `ln(2) = 2 × atanh(1/3)`. Cached thread-local.

use super::decimal_compute::{
    ComputeStorage, DECIMAL_COMPUTE_DP,
    decimal_compute_zero, decimal_compute_one, decimal_compute_halve,
    decimal_compute_add, decimal_compute_sub, decimal_compute_mul,
    decimal_compute_div, decimal_compute_div_int,
    decimal_compute_is_zero, decimal_compute_is_negative,
    decimal_compute_from_int, decimal_compute_cmp, decimal_compute_neg,
};
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

thread_local! {
    static LN2_CACHE: std::cell::RefCell<Option<ComputeStorage>> = const { std::cell::RefCell::new(None) };
}

/// Compute `ln(2)` at decimal compute dp, cached.
///
/// Formula: `ln(2) = 2 × atanh(1/3) = 2 × (1/3 + (1/3)³/3 + (1/3)⁵/5 + ...)`
fn ln2_at_compute() -> Result<ComputeStorage, OverflowDetected> {
    let cached: Option<ComputeStorage> = LN2_CACHE.with(|c| c.borrow().clone());
    if let Some(v) = cached {
        return Ok(v);
    }

    let one = decimal_compute_one();
    let three = decimal_compute_from_int(3);
    let one_third = decimal_compute_div(one, three)?;

    // atanh(1/3) via direct series (|s| = 1/3, fast convergence)
    let s = one_third;
    let s_sq = decimal_compute_mul(s, s);
    let mut term = s;
    let mut sum = s;

    // atanh(s) = s + s³/3 + s⁵/5 + ...
    // All positive terms.
    let max_terms = (DECIMAL_COMPUTE_DP as u32) + 20;
    for k in 1..=max_terms {
        term = decimal_compute_mul(term, s_sq);
        if decimal_compute_is_zero(&term) {
            break;
        }
        let divisor = (2 * k as u64) + 1;
        let contribution = decimal_compute_div_int(term, divisor);
        if decimal_compute_is_zero(&contribution) {
            break;
        }
        sum = decimal_compute_add(sum, contribution);
    }

    // ln(2) = 2 × atanh(1/3)
    let ln2 = decimal_compute_add(sum, sum);

    LN2_CACHE.with(|c: &std::cell::RefCell<Option<ComputeStorage>>| *c.borrow_mut() = Some(ln2));
    Ok(ln2)
}

/// Compute `ln(x)` for `x > 0` at compute dp.
pub fn decimal_ln(x: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    if decimal_compute_is_negative(&x) || decimal_compute_is_zero(&x) {
        return Err(OverflowDetected::DomainError);
    }

    let one = decimal_compute_one();
    let two = decimal_compute_from_int(2);

    // Range reduction: find k such that x / 2^k ∈ [1, 2)
    // Halve while ≥ 2, double while < 1
    let mut m = x;
    let mut k: i64 = 0;

    // First: if x is much larger than 2, halve repeatedly
    while decimal_compute_cmp(&m, &two) != std::cmp::Ordering::Less {
        m = decimal_compute_halve(m);
        k += 1;
    }

    // If x < 1, double repeatedly
    while decimal_compute_cmp(&m, &one) == std::cmp::Ordering::Less {
        // Double: m = m + m
        m = decimal_compute_add(m, m);
        k -= 1;
    }

    // Now m ∈ [1, 2). Compute ln(m) via atanh series.
    // s = (m - 1) / (m + 1), |s| ∈ [0, 1/3)
    let m_minus_one = decimal_compute_sub(m, one);
    let m_plus_one = decimal_compute_add(m, one);
    let s = decimal_compute_div(m_minus_one, m_plus_one)?;

    // ln(m) = 2 × atanh(s) = 2 × (s + s³/3 + s⁵/5 + ...)
    let s_sq = decimal_compute_mul(s, s);
    let mut term = s;
    let mut atanh_sum = s;

    let max_terms = (DECIMAL_COMPUTE_DP as u32) + 20;
    for j in 1..=max_terms {
        term = decimal_compute_mul(term, s_sq);
        if decimal_compute_is_zero(&term) {
            break;
        }
        let divisor = (2 * j as u64) + 1;
        let contribution = decimal_compute_div_int(term, divisor);
        if decimal_compute_is_zero(&contribution) {
            break;
        }
        atanh_sum = decimal_compute_add(atanh_sum, contribution);
    }

    let ln_m = decimal_compute_add(atanh_sum, atanh_sum); // 2 × atanh(s)

    // Final: ln(x) = k × ln(2) + ln(m)
    if k == 0 {
        return Ok(ln_m);
    }

    let ln2 = ln2_at_compute()?;
    let k_ln2 = if k > 0 {
        mul_compute_by_int(ln2, k as u64)
    } else {
        decimal_compute_neg(mul_compute_by_int(ln2, (-k) as u64))
    };

    Ok(decimal_compute_add(k_ln2, ln_m))
}

/// Multiply compute-tier value by a small positive integer.
fn mul_compute_by_int(v: ComputeStorage, n: u64) -> ComputeStorage {
    if n == 0 {
        return decimal_compute_zero();
    }
    if n == 1 {
        return v;
    }

    let n_compute: ComputeStorage = {
        #[cfg(table_format = "q16_16")]
        { n as i64 }
        #[cfg(table_format = "q32_32")]
        { n as i128 }
        #[cfg(table_format = "q64_64")]
        { crate::fixed_point::i256::I256::from_i128(n as i128) }
        #[cfg(table_format = "q128_128")]
        { crate::fixed_point::i512::I512::from_i128(n as i128) }
        #[cfg(table_format = "q256_256")]
        { crate::fixed_point::I1024::from_i128(n as i128) }
    };

    v * n_compute
}

#[cfg(all(test, table_format = "q64_64"))]
mod tests {
    use super::*;
    use super::super::decimal_compute::pow10_compute_ct;
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
    fn ln_one_is_zero() {
        let result = decimal_ln(decimal_compute_one()).unwrap();
        assert_eq!(result, decimal_compute_zero());
    }

    #[test]
    fn ln_negative_errors() {
        let neg_one = decimal_compute_neg(decimal_compute_one());
        assert!(decimal_ln(neg_one).is_err());
    }

    #[test]
    fn ln_zero_errors() {
        assert!(decimal_ln(decimal_compute_zero()).is_err());
    }

    /// mpmath: ln(2) = 0.69314718055994530941723212145817656807550013436025...
    #[test]
    fn ln_two_mpmath() {
        let two = decimal_compute_from_int(2);
        let result = decimal_ln(two).unwrap();
        let expected = parse_decimal_str("69314718055994530941723212145817656808");
        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(1_000_000);
        assert!(
            diff < tolerance,
            "ln(2) precision: got={:?} expected={:?} diff={:?}",
            result, expected, diff
        );
    }

    /// mpmath: ln(10) = 2.30258509299404568401799145468436420760110148862877...
    #[test]
    fn ln_ten_mpmath() {
        let ten = decimal_compute_from_int(10);
        let result = decimal_ln(ten).unwrap();
        let expected = parse_decimal_str("230258509299404568401799145468436420760");
        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(10_000_000);
        assert!(
            diff < tolerance,
            "ln(10) precision: got={:?} expected={:?} diff={:?}",
            result, expected, diff
        );
    }

    /// mpmath: ln(0.5) = -0.69314718055994530941723212145817656807550013436026...
    #[test]
    fn ln_half_mpmath() {
        let half = pow10_compute_ct(37) * I256::from_i128(5);
        let result = decimal_ln(half).unwrap();
        let expected_abs = parse_decimal_str("69314718055994530941723212145817656808");
        // result should be -expected_abs
        let result_abs = if decimal_compute_is_negative(&result) {
            decimal_compute_neg(result)
        } else {
            result
        };
        let diff = if result_abs > expected_abs { result_abs - expected_abs } else { expected_abs - result_abs };
        let tolerance = I256::from_i128(1_000_000);
        assert!(
            diff < tolerance,
            "ln(0.5) precision: got={:?} expected_abs={:?}",
            result, expected_abs
        );
        assert!(decimal_compute_is_negative(&result), "ln(0.5) should be negative");
    }
}
