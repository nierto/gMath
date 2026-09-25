//! Decimal sine and cosine: exact range reduction at a wide precision, Taylor
//! at the compute dp.
//!
//! # Algorithm
//!
//! 1. **Range reduction** at `HP_DP` (see `hp.rs`): `k = round(|x| 2/pi)`,
//!    `r = |x| - k pi/2`, with pi/2 held to `2 HP_DP` digits as a split
//!    constant, so `r` is exact to half a unit at `HP_DP` for every `k` the
//!    compute tier can produce. `r` is then rounded once to the compute dp.
//! 2. **Taylor series** for `|r| <= pi/4`:
//!    - `sin(r) = r - r^3/3! + r^5/5! - r^7/7! + ...`
//!    - `cos(r) = 1 - r^2/2! + r^4/4! - r^6/6! + ...`
//!    Iterative term computation: `term_k = -term_{k-1} * r^2 / ((2k)(2k+1))` for sin.
//! 3. **Quadrant reconstruction** from `k mod 4`, then `sin(-x) = -sin(x)`.
//!
//! Before 0.6.4 the reduction used pi at the compute dp and an i64 quadrant
//! count: the error of pi grew with `k` (realtime `sin(10^5)` at 4 decimals
//! was 4 units off, `sin(10^9)` had the wrong sign; embedded `sin(10^17)` at
//! 19 decimals 1 unit), and a count beyond i64 wrapped.
//!
//! # pi
//!
//! Machin's formula `pi/4 = 4 atan(1/5) - atan(1/239)`, each arctangent
//! summed at `2 HP_DP` digits with divisions by small integers only. The
//! compute-dp pi (`pi_at_decimal_compute`) is that value rounded half to even.

use super::decimal_compute::{
    ComputeStorage, DECIMAL_COMPUTE_DP,
    decimal_compute_zero, decimal_compute_one,
    decimal_compute_add, decimal_compute_sub, decimal_compute_mul,
    decimal_compute_div_int,
    decimal_compute_is_zero, decimal_compute_is_negative, decimal_compute_neg,
};
use super::hp::*;
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;
use std::cell::RefCell;
use std::rc::Rc;

// ============================================================================
// pi AT 2 HP_DP DIGITS (built once per thread)
// ============================================================================

struct PiConsts {
    /// pi/2 to `2 HP_DP` digits, split: `pi/2 = (hi + lo / 10^HP_DP) / 10^HP_DP`.
    half_hi: Hp,
    half_lo: Hp,
    /// Half a unit at `HP_DP` (`10^HP_DP / 2`).
    half_unit: Hp,
    /// pi/2 and pi/4 at `HP_DP` (rounded), 2/pi at `HP_DP` (truncated).
    half_pi: Hp,
    quarter_pi: Hp,
    two_over_pi: Hp,
    /// pi at the compute dp, rounded half to even.
    pi_compute: ComputeStorage,
}

thread_local! {
    static PI_CONSTS: RefCell<Option<Rc<PiConsts>>> = const { RefCell::new(None) };
}

/// `atan(1/m)` at `2 HP_DP` digits: `sum (-1)^k / ((2k+1) m^(2k+1))`, each
/// term truncated (the sum is off by at most a few hundred units there).
fn atan_inverse_wide(m: u64) -> Hp {
    let mut p = hp_divmod_small(hp_pow10(2 * HP_DP), m).0;
    let (mut plus, mut minus) = (p, hp_small(0));
    let mut k: u64 = 1;
    loop {
        p = hp_divmod_small(p, m * m).0;
        if hp_is_zero(&p) {
            break;
        }
        let term = hp_divmod_small(p, 2 * k + 1).0;
        if k % 2 == 1 { minus = minus + term } else { plus = plus + term }
        k += 1;
    }
    plus - minus
}

fn build_pi_consts() -> Result<PiConsts, OverflowDetected> {
    // pi/2 = 8 atan(1/5) - 2 atan(1/239); 1.6 * 10^(2 HP_DP) fits Hp
    let half_wide = hp_mul_small(atan_inverse_wide(5), 8) - hp_mul_small(atan_inverse_wide(239), 2);
    let half_hi = hp_div_pow10(half_wide, HP_DP);
    let half_lo = half_wide - hp_mul_pow10(half_hi, HP_DP);
    let half_unit = hp_divmod_small(hp_pow10(HP_DP), 2).0;
    let half_pi = half_hi + hp_div_pow10(half_lo + half_unit, HP_DP);
    let quarter_pi = hp_div_pow10(hp_divmod_small(half_wide, 2).0 + half_unit, HP_DP);
    let two_over_pi = hp_div(hp_pow10(2 * HP_DP), half_pi);
    let pi_wide = hp_mul_small(half_wide, 2);
    let pi_compute = hp_to_compute(&hp_div_round_half_even(pi_wide, 2 * HP_DP - DECIMAL_COMPUTE_DP as u32, 0))?;
    Ok(PiConsts { half_hi, half_lo, half_unit, half_pi, quarter_pi, two_over_pi, pi_compute })
}

fn pi_consts() -> Result<Rc<PiConsts>, OverflowDetected> {
    if let Some(c) = PI_CONSTS.with(|c| c.borrow().clone()) {
        return Ok(c);
    }
    let c = Rc::new(build_pi_consts()?);
    PI_CONSTS.with(|slot| *slot.borrow_mut() = Some(c.clone()));
    Ok(c)
}

/// pi at decimal compute-tier precision, rounded half to even, cached per thread.
pub fn pi_at_decimal_compute() -> Result<ComputeStorage, OverflowDetected> {
    Ok(pi_consts()?.pi_compute)
}

// ============================================================================
// SIN / COS / SINCOS
// ============================================================================

const fn max_trig_taylor_terms() -> u32 {
    // For |r| ≤ π/4 ≈ 0.785, term k ~ 0.785^(2k+1)/(2k+1)! — converges quickly.
    // Need ~21 terms for dp=38, ~41 for dp=77, ~80 for dp=154.
    (DECIMAL_COMPUTE_DP as u32 / 2) + 20
}

/// `a = k pi/2 + r` for `a >= 0` at the compute dp (held in `Hp`, so the
/// magnitude of the compute tier's minimum fits): the quadrant count `k`
/// (exact, in `Hp`) and `r` at `HP_DP` (signed, `|r| <= pi/4` up to a unit).
fn reduce(c: &PiConsts, a_c: Hp) -> (Hp, Hp) {
    // k within one of floor(a 2/pi): 2/pi has HP_DP digits, k fewer
    let mut k = hp_div_pow10(a_c * c.two_over_pi, DECIMAL_COMPUTE_DP as u32 + HP_DP);
    let a_wide = hp_mul_pow10(a_c, HP_DP - DECIMAL_COMPUTE_DP as u32);
    let k_half_pi = k * c.half_hi + hp_div_pow10(k * c.half_lo + c.half_unit, HP_DP);
    let mut r = a_wide - k_half_pi;
    let minus_quarter = -c.quarter_pi;
    while r > c.quarter_pi {
        k = k + hp_small(1);
        r = r - c.half_pi;
    }
    while !hp_is_zero(&k) && r < minus_quarter {
        k = k - hp_small(1);
        r = r + c.half_pi;
    }
    (k, r)
}

/// Compute both `sin(x)` and `cos(x)` at compute dp: single shared range reduction.
pub fn decimal_sincos(x: ComputeStorage) -> Result<(ComputeStorage, ComputeStorage), OverflowDetected> {
    if decimal_compute_is_zero(&x) {
        return Ok((decimal_compute_zero(), decimal_compute_one()));
    }
    let negative = decimal_compute_is_negative(&x);
    // |x| in Hp: negating at the compute tier panicked at its minimum
    let xh = compute_to_hp(x);
    let a = if negative { -xh } else { xh };
    let c = pi_consts()?;
    let (k, r) = reduce(&c, a);
    let (sin_r, cos_r) = sincos_taylor(hp_signed_to_compute(r)?);

    // Quadrant reconstruction: k mod 4
    let (sin_a, cos_a) = match k.words[0] & 3 {
        0 => (sin_r, cos_r),
        1 => (cos_r, decimal_compute_neg(sin_r)),
        2 => (decimal_compute_neg(sin_r), decimal_compute_neg(cos_r)),
        _ => (decimal_compute_neg(cos_r), sin_r),
    };
    Ok((if negative { decimal_compute_neg(sin_a) } else { sin_a }, cos_a))
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

#[cfg(all(test, table_format = "q64_64"))]
mod tests {
    use super::*;
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
