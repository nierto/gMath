//! Decimal square root: `sqrt(x)` via Newton-Raphson at compute dp.
//!
//! # Algorithm
//!
//! Newton-Raphson iteration: `y_{n+1} = (y_n + x/y_n) / 2` converges quadratically
//! (digits double each iteration). For a good initial guess, ~8 iterations suffice
//! at dp=38, ~10 at dp=77, ~12 at dp=154.
//!
//! # Initial Guess
//!
//! Extract the "integer sqrt" of the raw compute-tier integer, then adjust for the
//! 10^dp scaling. For value `v` at compute dp, `v = raw × 10^-dp`, so
//! `sqrt(v) = sqrt(raw) × 10^(-dp/2)`. For odd dp we shift `raw` by 10.

#[allow(unused_imports)]
use super::word::DecimalWord;
#[allow(unused_imports)]
use crate::fixed_point::width::WideInt;
#[allow(unused_imports)]
use super::decimal_compute::{decimal_compute_one, decimal_compute_add, decimal_compute_div, decimal_compute_halve, decimal_compute_cmp};
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

/// Compute `sqrt(x)` for x ≥ 0 at compute dp.
///
/// # Algorithm
///
/// Newton-Raphson `y_{n+1} = (y_n + x/y_n) / 2` with magnitude-aware initial guess.
///
/// **Initial guess**: Use bit length of `x` to estimate the order of magnitude.
/// If x has `b` significant bits, then `sqrt(x)` has approximately `b/2` significant
/// bits, so `y_0 = 1 << (b/2)` is a good starting point.
pub fn decimal_sqrt<C: DecimalWord>(x: C) -> Result<C, OverflowDetected> {
    if C::decimal_compute_is_negative(&x) {
        return Err(OverflowDetected::DomainError);
    }
    if C::decimal_compute_is_zero(&x) {
        return Ok(C::decimal_compute_zero());
    }

    // Magnitude-aware initial guess: use the bit length of x.
    //
    // x at compute dp represents `x_actual = x_raw / 10^DP`.
    // sqrt(x_actual) = sqrt(x_raw) / sqrt(10^DP) = sqrt(x_raw) × 10^(-DP/2)
    //
    // At compute dp, sqrt(x_actual) at compute dp = sqrt(x_raw) × 10^(DP/2).
    //
    // For initial guess: estimate sqrt(x_raw) by halving the bit length, then scale.
    let bits = C::sqrt_bit_length(&x);
    // y_0 = 2^half_bits — order of magnitude estimate of sqrt(x_raw)
    // Then we want y_0 × 10^(DP/2) for compute storage. But that's complex to construct.
    //
    // Simpler: y_0 = x >> (bits - half_bits) — this gives roughly sqrt(x_raw) magnitude
    // when interpreted as compute storage at compute dp.
    //
    // Actually the simplest correct approach: integer sqrt of x_raw, then ×10^(DP/2).
    // But that's also complex. Use bit-based estimate that converges in ~6 iterations:
    let one = decimal_compute_one::<C>();
    let y_init = if bits == 0 {
        decimal_compute_halve(one)
    } else {
        // We want y ≈ sqrt(x). Bit estimate: y has bits/2 + extra bits.
        // For x at compute dp, x_raw bit length includes 10^DP scaling (~127 bits for dp=38).
        // sqrt(x_raw) has half those bits → sqrt(x_actual) at compute dp has bits/2 + DP×log2(10)/2 bits.
        // Simpler: y_0 = 1 << ((bits + dp_bits) / 2)
        let dp_bits = (C::DP as u32) * 33219 / 10000; // ≈ DP × log2(10)
        let target_bits = (bits + dp_bits) / 2;
        C::shift_one_left(target_bits)
    };

    let mut y = if is_compute_zero_or_neg(&y_init) {
        decimal_compute_halve(one)
    } else {
        y_init
    };

    // Newton-Raphson: y_{n+1} = (y_n + x/y_n) / 2
    const MAX_ITERATIONS: u32 = 200;
    let mut prev = C::decimal_compute_zero();

    for _ in 0..MAX_ITERATIONS {
        let x_div_y = decimal_compute_div(x, y)?;
        let sum = decimal_compute_add(y, x_div_y);
        let new_y = decimal_compute_halve(sum);

        if decimal_compute_cmp(&new_y, &y) == std::cmp::Ordering::Equal {
            return Ok(new_y);
        }
        if decimal_compute_cmp(&new_y, &prev) == std::cmp::Ordering::Equal {
            return Ok(new_y);
        }
        prev = y;
        y = new_y;
    }

    Ok(y)
}

fn is_compute_zero_or_neg<C: DecimalWord>(v: &C) -> bool {
    C::decimal_compute_is_zero(v) || C::decimal_compute_is_negative(v)
}

#[cfg(test)]
mod tests {
    //! On the 38-digit compute word (I256), in every build.
    use super::*;
    type ComputeStorage = crate::fixed_point::I256;
    #[allow(unused_imports)]
    use super::super::decimal_compute::*;
    use super::super::decimal_compute::decimal_compute_from_int;
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
    fn sqrt_zero() {
        let result = decimal_sqrt(decimal_compute_zero::<ComputeStorage>()).unwrap();
        assert_eq!(result, decimal_compute_zero::<ComputeStorage>());
    }

    #[test]
    fn sqrt_one() {
        let result = decimal_sqrt(decimal_compute_one::<ComputeStorage>()).unwrap();
        assert_eq!(result, decimal_compute_one::<ComputeStorage>());
    }

    #[test]
    fn sqrt_four_is_two() {
        let four = decimal_compute_from_int::<ComputeStorage>(4);
        let two = decimal_compute_from_int::<ComputeStorage>(2);
        let result = decimal_sqrt(four).unwrap();
        assert_eq!(result, two);
    }

    #[test]
    fn sqrt_nine_is_three() {
        let nine = decimal_compute_from_int::<ComputeStorage>(9);
        let three = decimal_compute_from_int::<ComputeStorage>(3);
        let result = decimal_sqrt(nine).unwrap();
        assert_eq!(result, three);
    }

    /// mpmath: sqrt(2) = 1.41421356237309504880168872420969807856967187537694...
    #[test]
    fn sqrt_two_mpmath() {
        let two = decimal_compute_from_int::<ComputeStorage>(2);
        let result = decimal_sqrt(two).unwrap();
        // 1.41421356237309504880168872420969807857 × 10^38
        // = 141421356237309504880168872420969807857
        let expected = parse_decimal_str("141421356237309504880168872420969807857");
        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(1000);
        assert!(
            diff < tolerance,
            "sqrt(2) precision: got={:?} expected={:?} diff={:?}",
            result, expected, diff
        );
    }

    /// mpmath: sqrt(3) = 1.73205080756887729352744634150587236694280525381038...
    #[test]
    fn sqrt_three_mpmath() {
        let three = decimal_compute_from_int::<ComputeStorage>(3);
        let result = decimal_sqrt(three).unwrap();
        let expected = parse_decimal_str("173205080756887729352744634150587236694");
        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(1000);
        assert!(
            diff < tolerance,
            "sqrt(3) precision: got={:?} expected={:?} diff={:?}",
            result, expected, diff
        );
    }
}
