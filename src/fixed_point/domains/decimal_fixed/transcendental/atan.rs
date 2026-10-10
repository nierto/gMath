//! Decimal arctangent: `atan(x)` and `atan2(y, x)` at compute dp.
//!
//! # Algorithm
//!
//! For `|x| < 1`: direct Taylor series `atan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...`
//!
//! For `|x| ≥ 1`: use the identity `atan(x) = π/2 - atan(1/x)` to reduce.
//!
//! For faster convergence when `|x|` is close to 1, we apply the half-angle reduction:
//! `atan(x) = 2 × atan(x / (1 + sqrt(1 + x²)))`. But for simplicity in the first
//! implementation, we accept slower convergence near 1 and let the Taylor loop
//! iterate up to `max_terms()` times.
//!
//! # π dependency
//!
//! `atan` itself does NOT need π (for `|x| < 1`). The reduction `atan(x) = π/2 - atan(1/x)`
//! needs π/2, but we can avoid calling `atan` on values ≥ 1 for most use cases by
//! accepting a direct input in `[-1, 1]`. For the `|x| > 1` case, the caller must
//! supply `pi_half` (which `sin_cos` computes via Machin's formula using this same
//! `decimal_atan` function; one-time bootstrap).

#[allow(unused_imports)]
use super::word::DecimalWord;
#[allow(unused_imports)]
use crate::fixed_point::width::WideInt;
#[allow(unused_imports)]
use super::decimal_compute::ComputeStorage;
use super::decimal_compute::{decimal_compute_one, decimal_compute_halve, decimal_compute_add, decimal_compute_sub, decimal_compute_mul, decimal_compute_div, decimal_compute_abs, decimal_compute_cmp, decimal_compute_neg};
use super::sqrt::decimal_sqrt;
use super::sin_cos::pi_at_decimal_compute;
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

fn max_taylor_terms<C: DecimalWord>() -> u32 {
    // For |x| ≤ tan(π/8) ≈ 0.414, term k ~ 0.414^(2k+1)/(2k+1) < 10^-dp when k ≈ dp * 1.3
    (C::DP as u32 * 2) + 20
}

/// Core Taylor series `atan(x) = x - x³/3 + x⁵/5 - x⁷/7 + ...`
///
/// Only converges reasonably fast for `|x| < 0.5` or so. The caller must range-reduce.
fn atan_taylor<C: DecimalWord>(x: C) -> C {
    if C::decimal_compute_is_zero(&x) {
        return C::decimal_compute_zero();
    }

    let x_sq = decimal_compute_mul(x, x);
    let mut term = x; // term_k corresponds to x^(2k+1)/(2k+1), starting at k=0
    let mut sum = x;
    let mut sign_positive = true;

    let max_terms = max_taylor_terms::<C>();
    for k in 1..=max_terms {
        // Next odd power: term = term × x²
        term = decimal_compute_mul(term, x_sq);
        if C::decimal_compute_is_zero(&term) {
            break;
        }
        sign_positive = !sign_positive;
        // Divide by (2k+1)
        let divisor = (2 * k as u64) + 1;
        let contribution = super::decimal_compute::decimal_compute_div_int(term, divisor);
        if C::decimal_compute_is_zero(&contribution) {
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

/// Compute `atan(x)` for x at compute dp.
///
/// Handles the full range `x ∈ ℝ`:
/// - `|x| ≤ tan(π/12) ≈ 0.268`: direct Taylor (fast convergence)
/// - `0.268 < |x| ≤ 1`: half-angle reduction `atan(x) = 2·atan(x/(1+sqrt(1+x²)))`
/// - `|x| > 1`: reduce via `atan(x) = π/2 - atan(1/x)`
pub fn decimal_atan<C: DecimalWord>(x: C) -> Result<C, OverflowDetected> {
    if C::decimal_compute_is_zero(&x) {
        return Ok(C::decimal_compute_zero());
    }

    let negative = C::decimal_compute_is_negative(&x);

    // Compare |x| against 1 without forming |x|: the compute tier's minimum
    // has no negation (decimal_compute_abs panicked there)
    let one = decimal_compute_one::<C>();
    let beyond_one = decimal_compute_cmp(&x, &one) == std::cmp::Ordering::Greater
        || decimal_compute_cmp(&x, &decimal_compute_neg(one)) == std::cmp::Ordering::Less;

    let result = if beyond_one {
        // |x| > 1: atan(|x|) = π/2 - atan(1/|x|); the quotient rounds half
        // away from zero, so |1/x| equals 1/|x| bit for bit
        let reciprocal = decimal_compute_abs(decimal_compute_div(one, x)?);
        let atan_recip = atan_reduced(reciprocal)?;
        let pi_half = decimal_compute_halve(pi_at_decimal_compute::<C>()?);
        decimal_compute_sub(pi_half, atan_recip)
    } else {
        // |x| ≤ 1
        atan_reduced(decimal_compute_abs(x))?
    };

    if negative {
        Ok(decimal_compute_neg(result))
    } else {
        Ok(result)
    }
}

/// `atan(x)` for `0 ≤ x ≤ 1`.
///
/// Applies half-angle reduction `atan(x) = 2·atan(x/(1+sqrt(1+x²)))` to bring `x`
/// below `tan(π/12) ≈ 0.268` for fast Taylor convergence.
fn atan_reduced<C: DecimalWord>(x: C) -> Result<C, OverflowDetected> {
    // Apply half-angle reduction twice: |x| ≤ 1 → ≤ tan(π/8) ≈ 0.414 → ≤ tan(π/16) ≈ 0.199
    let threshold = C::tan_pi_over_16_approx();

    let mut y = x;
    let mut halvings: u32 = 0;
    while decimal_compute_cmp(&y, &threshold) == std::cmp::Ordering::Greater && halvings < 8 {
        // y_new = y / (1 + sqrt(1 + y²))
        let y_sq = decimal_compute_mul(y, y);
        let one_plus_y_sq = decimal_compute_add(decimal_compute_one::<C>(), y_sq);
        let sqrt_val = decimal_sqrt(one_plus_y_sq)?;
        let denom = decimal_compute_add(decimal_compute_one::<C>(), sqrt_val);
        y = decimal_compute_div(y, denom)?;
        halvings += 1;
    }

    // Direct Taylor for small y
    let mut result = atan_taylor(y);

    // Multiply result by 2^halvings
    for _ in 0..halvings {
        result = decimal_compute_add(result, result);
    }

    Ok(result)
}

/// Compute `atan2(y, x)` for y, x at compute dp.
///
/// Standard 4-quadrant atan2 using the decimal `atan` function.
pub fn decimal_atan2<C: DecimalWord>(y: C, x: C) -> Result<C, OverflowDetected> {
    let y_zero = C::decimal_compute_is_zero(&y);
    let x_zero = C::decimal_compute_is_zero(&x);

    if x_zero && y_zero {
        return Err(OverflowDetected::DomainError);
    }

    let pi = pi_at_decimal_compute::<C>()?;
    let pi_half = decimal_compute_halve(pi);

    if x_zero {
        // On the y-axis
        return if C::decimal_compute_is_negative(&y) {
            Ok(decimal_compute_neg(pi_half))
        } else {
            Ok(pi_half)
        };
    }

    let ratio = match decimal_compute_div(y, x) {
        Ok(r) => r,
        // |y/x| beyond the compute tier: atan2(y, x) = sign(y) π/2 - atan(x/y)
        // (y != 0 here), where x/y is tiny. Before 0.6.4 this was an error.
        Err(OverflowDetected::TierOverflow) => {
            let signed_half = if C::decimal_compute_is_negative(&y) { decimal_compute_neg(pi_half) } else { pi_half };
            return Ok(decimal_compute_sub(signed_half, decimal_atan(decimal_compute_div(x, y)?)?));
        }
        Err(e) => return Err(e),
    };
    let atan_val = decimal_atan(ratio)?;

    if !C::decimal_compute_is_negative(&x) {
        // First or fourth quadrant
        Ok(atan_val)
    } else if !C::decimal_compute_is_negative(&y) {
        // Second quadrant: atan2 = atan(y/x) + π
        Ok(decimal_compute_add(atan_val, pi))
    } else {
        // Third quadrant: atan2 = atan(y/x) - π
        Ok(decimal_compute_sub(atan_val, pi))
    }
}


#[cfg(all(test, table_format = "q64_64"))]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use super::super::decimal_compute::*;
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
    fn atan_zero() {
        let result = decimal_atan(decimal_compute_zero::<ComputeStorage>()).unwrap();
        assert_eq!(result, decimal_compute_zero::<ComputeStorage>());
    }

    /// mpmath: atan(1) = π/4 = 0.78539816339744830961566084581987572104929234984...
    #[test]
    fn atan_one_is_pi_over_4() {
        let one = decimal_compute_one::<ComputeStorage>();
        let result = decimal_atan(one).unwrap();
        // π/4 at dp=38
        let expected = parse_decimal_str("78539816339744830961566084581987572105");
        let diff = if result > expected { result - expected } else { expected - result };
        // atan(1) bootstraps π, so tolerance is a bit wider for this first release
        let tolerance = I256::from_i128(1_000_000);
        assert!(
            diff < tolerance,
            "atan(1) precision: got={:?} expected={:?} diff={:?}",
            result, expected, diff
        );
    }

    /// mpmath: atan(0.5) = 0.46364760900080611621425623146121440202853705428612...
    #[test]
    fn atan_half() {
        let half = pow10_compute_ct::<ComputeStorage>(37) * I256::from_i128(5);
        let result = decimal_atan(half).unwrap();
        let expected = parse_decimal_str("46364760900080611621425623146121440203");
        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(1_000_000);
        assert!(
            diff < tolerance,
            "atan(0.5) precision: got={:?} expected={:?} diff={:?}",
            result, expected, diff
        );
    }
}
