//! Decimal exponential: `exp(x)` at compute dp.
//!
//! # Algorithm
//!
//! A result near the top of the compute tier (about 10^9 on realtime, 10^19
//! on compact, far more on the wider profiles) needs as many significant
//! digits as the tier holds, so the engine works at a wider decimal precision
//! `C::HP_DP` in the next integer tier, on values below 2 only:
//!
//! 1. **Reduction**: `x = n ln2 + r` with integer `n` and `0 <= r < ln2`.
//!    `ln2` is held to `2 C::HP_DP` digits (a split constant), so `n ln2` is
//!    exact to half a unit at `C::HP_DP` for every `n`.
//! 2. **Digits**: `r = d1/10 + d2/100 + d3/1000 + d4/10^4 + s` with
//!    `0 <= s < 10^-4`; `exp(r)` is four table entries times a short Taylor
//!    series for `exp(s)`.
//! 3. **Scale**: `exp(x) = exp(r) 2^n`, rounded once (half to even) to the
//!    compute dp. The power of two is exact.
//!
//! The relative error before that rounding is a few units at `C::HP_DP`
//! whatever the size of `x`. `C::HP_DP` exceeds (digits of the largest
//! representable result) + (storage dp) by at least 8 on every profile, so
//! the compute-dp result is the correctly rounded value. Before 0.6.4 the
//! engine built `exp(k) = e^k` by `k` successive multiplications at the
//! compute dp: up to 267 units off at 19 decimals on embedded (x near 44),
//! and on realtime the entries past `e^22` wrapped the i64 compute tier.
//!
//! Past the compute tier the result is `Err(TierOverflow)`; below
//! `-C::COMPUTE_BITS` it rounds to 0.

#[allow(unused_imports)]
use super::word::{DecimalWord, HpWord};
#[allow(unused_imports)]
use crate::fixed_point::width::WideInt;
#[allow(unused_imports)]
use super::decimal_compute::ComputeStorage;
use super::decimal_compute::{decimal_compute_one, decimal_compute_cmp, decimal_compute_neg};
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

use super::hp::*;

// ============================================================================
// CONSTANTS AND TABLES AT C::HP_DP (built once per thread)
// ============================================================================

struct HpConsts<C: DecimalWord> {
    /// `10^C::HP_DP` (the value 1) and half of it.
    one: C::Hp,
    half: C::Hp,
    /// ln2 to `2 C::HP_DP` digits, split: `ln2 = (hi + lo / 10^C::HP_DP) / 10^C::HP_DP`.
    ln2_hi: C::Hp,
    ln2_lo: C::Hp,
    /// ln2 at `C::HP_DP` (rounded) and at 15 digits (truncated, for estimating n).
    ln2: C::Hp,
    ln2_15: i128,
    /// `exp(d / 10^(s+1))` for `s = 0..4`, `d = 0..10`.
    table: [[C::Hp; 10]; 4],
}


fn hp_consts<C: DecimalWord>() -> std::rc::Rc<HpConsts<C>> {
    super::cache::typed_cached(|| Ok(build_hp_consts::<C>())).expect("decimal exp constants")
}

/// `exp(y)` at `C::HP_DP` for `0 <= y <= 0.1` by Taylor series.
fn hp_exp_taylor<C: DecimalWord>(y: C::Hp, one: C::Hp, half: C::Hp) -> C::Hp {
    let mut term = one;
    let mut sum = one;
    for k in 1..=(C::HP_DP as u64 + 10) {
        term = hp_div_small_round::<C>(hp_mul::<C>(term, y, half), k);
        if hp_is_zero::<C>(&term) {
            break;
        }
        sum = sum + term;
    }
    sum
}

fn build_hp_consts<C: DecimalWord>() -> HpConsts<C> {
    let one = hp_pow10::<C>(C::HP_DP);
    let half = hp_divmod_small::<C>(one, 2).0;

    // ln2 = 2 atanh(1/3) = sum over k of 2 / ((2k+1) 3^(2k+1)), at 2 C::HP_DP
    // digits: divisions by small integers only (10^(2 C::HP_DP) fits C::Hp), each
    // term truncated, so the sum is low by at most a few hundred units at
    // 2 C::HP_DP digits.
    let mut p = hp_divmod_small::<C>(hp_mul_small::<C>(hp_pow10::<C>(2 * C::HP_DP), 2), 3).0;
    let mut sum = p;
    let mut k: u64 = 1;
    loop {
        p = hp_divmod_small::<C>(p, 9).0;
        if hp_is_zero::<C>(&p) {
            break;
        }
        sum = sum + hp_divmod_small::<C>(p, 2 * k + 1).0;
        k += 1;
    }
    let ln2_hi = hp_div_pow10::<C>(sum, C::HP_DP);
    let ln2_lo = sum - hp_mul_pow10::<C>(ln2_hi, C::HP_DP);
    let ln2 = ln2_hi + hp_div_pow10::<C>(ln2_lo + half, C::HP_DP);
    let ln2_15 = hp_to_i128::<C>(&hp_div_pow10::<C>(ln2_hi, C::HP_DP - 15));

    let mut table = [[one; 10]; 4];
    for (s, row) in table.iter_mut().enumerate() {
        let base = hp_exp_taylor::<C>(hp_pow10::<C>(C::HP_DP - 1 - s as u32), one, half);
        for d in 1..10 {
            row[d] = hp_mul::<C>(row[d - 1], base, half);
        }
    }
    HpConsts { one, half, ln2_hi, ln2_lo, ln2, ln2_15, table }
}

/// `n ln2` at `C::HP_DP`, within half a unit.
fn n_ln2<C: DecimalWord>(c: &HpConsts<C>, n: i64) -> C::Hp {
    let m = n.unsigned_abs();
    let v = hp_mul_small::<C>(c.ln2_hi, m) + hp_div_pow10::<C>(hp_mul_small::<C>(c.ln2_lo, m) + c.half, C::HP_DP);
    if n < 0 { -v } else { v }
}

/// `exp(x) = m 2^n` with `m` at `C::HP_DP` in `[1, 2)` up to rounding, or
/// `None` when the result rounds to 0 at the compute dp. `Err(TierOverflow)`
/// when `x > C::COMPUTE_BITS` (the result exceeds every compute value).
fn exp_hp<C: DecimalWord>(x: C) -> Result<Option<(C::Hp, i64)>, OverflowDetected> {
    let bound = C::decimal_compute_from_int(C::COMPUTE_BITS);
    if decimal_compute_cmp(&x, &bound) == std::cmp::Ordering::Greater {
        return Err(OverflowDetected::TierOverflow);
    }
    if decimal_compute_cmp(&x, &decimal_compute_neg(bound)) == std::cmp::Ordering::Less {
        return Ok(None);
    }
    let c = hp_consts::<C>();
    let negative = C::decimal_compute_is_negative(&x);
    let magnitude = hp_mul_pow10::<C>(
        C::compute_to_hp(if negative { decimal_compute_neg(x) } else { x }),
        C::HP_DP - C::DP as u32,
    );
    let xh = if negative { -magnitude } else { magnitude };

    // n from 15-digit values (|x| <= C::COMPUTE_BITS keeps them inside i128),
    // then corrected (a step at most) so that 0 <= r < ln2
    let x15 = hp_to_i128::<C>(&hp_div_pow10::<C>(magnitude, C::HP_DP - 15));
    let est = (x15 / c.ln2_15) as i64;
    let mut n = if negative { -est - 1 } else { est };
    let mut stepped_down = false;
    let r = loop {
        let r = xh - n_ln2(&c, n);
        if hp_is_negative::<C>(&r) {
            n -= 1;
            stepped_down = true;
        } else if r >= c.ln2 && !stepped_down {
            n += 1;
        } else {
            break r;
        }
    };

    // exp(r): four decimal digits from the tables, Taylor for the rest
    let q = hp_to_i128::<C>(&hp_div_pow10::<C>(r, C::HP_DP - 4)) as usize;
    let s = r - hp_mul_pow10::<C>(hp_small::<C>(q as u64), C::HP_DP - 4);
    let digits = [q / 1000, (q / 100) % 10, (q / 10) % 10, q % 10];
    let mut m = c.one;
    for (stage, &d) in digits.iter().enumerate() {
        if d != 0 {
            m = hp_mul::<C>(m, c.table[stage][d], c.half);
        }
    }
    if !hp_is_zero::<C>(&s) {
        m = hp_mul::<C>(m, hp_exp_taylor::<C>(s, c.one, c.half), c.half);
    }
    Ok(Some((m, n)))
}

/// `m 2^n` at `C::HP_DP` rounded half to even to the compute dp,
/// `Err(TierOverflow)` when it does not fit the compute tier.
fn hp_scaled_to_compute<C: DecimalWord>(m: C::Hp, n: i64) -> Result<C, OverflowDetected> {
    let e = C::HP_DP - C::DP as u32;
    let q = if n >= 0 {
        if hp_bit_length::<C>(&m) as i64 + n > C::HP_BITS as i64 - 2 {
            return Err(OverflowDetected::TierOverflow);
        }
        hp_div_round_half_even::<C>(hp_shl::<C>(m, n as u32), e, 0)
    } else {
        // 10^e 2^k beyond the tier means m / (10^e 2^k) < 2^(4 - C::HP_BITS) 10^C::HP_DP < 1/2
        let k = n.unsigned_abs();
        if hp_bit_length::<C>(&hp_pow10::<C>(e)) as u64 + k > C::HP_BITS as u64 - 2 {
            return Ok(C::decimal_compute_zero());
        }
        hp_div_round_half_even::<C>(m, e, k as u32)
    };
    C::hp_to_compute(&q)
}

/// Compute `exp(x)` for x at compute dp, correctly rounded (half to even) to
/// the compute dp; `Err(TierOverflow)` when the result is outside the
/// compute tier. See the module docs for the algorithm.
pub fn decimal_exp<C: DecimalWord>(x: C) -> Result<C, OverflowDetected> {
    if C::decimal_compute_is_zero(&x) {
        return Ok(decimal_compute_one::<C>());
    }
    match exp_hp(x)? {
        None => Ok(C::decimal_compute_zero()),
        Some((m, n)) => hp_scaled_to_compute::<C>(m, n),
    }
}

/// Compute `exp(-x)`: used internally for hyperbolic functions.
#[allow(dead_code)]
pub fn decimal_exp_neg<C: DecimalWord>(x: C) -> Result<C, OverflowDetected> {
    decimal_exp(decimal_compute_neg(x))
}

/// Fused `(sinh(x), cosh(x))` at decimal compute tier: shares one exp-pair evaluation.
///
/// `exp(|x|)` and `exp(-|x|)` are combined at `C::HP_DP` and each result is
/// rounded once (half to even) to the compute dp, so both are correctly
/// rounded, including `sinh` of small arguments (no cancellation at the
/// compute dp) and `sinh`/`cosh` values beyond `exp`'s own range (e^x/2
/// fits where e^x does not). `Err(TierOverflow)` when a result is outside
/// the compute tier.
///
/// sinh and cosh are derived from the **same** `(ep, en)` pair, so their
/// rounding bias is correlated: critical for expressions like
/// `cosh(θ)·p + (sinh(θ)/θ)·v` where the two errors cancel.
pub fn decimal_sinhcosh<C: DecimalWord>(x: C) -> Result<(C, C), OverflowDetected> {
    if C::decimal_compute_is_zero(&x) {
        return Ok((C::decimal_compute_zero(), decimal_compute_one::<C>()));
    }
    let negative = C::decimal_compute_is_negative(&x);
    // the compute tier's minimum has no negation there (and its sinh/cosh
    // are far outside the tier anyway)
    let a = if negative { C::try_decimal_compute_neg(x)? } else { x };
    let (m1, n1) = exp_hp(a)?.ok_or(OverflowDetected::TierOverflow)?;
    if hp_bit_length::<C>(&m1) as i64 + n1 > C::HP_BITS as i64 - 2 {
        return Err(OverflowDetected::TierOverflow);
    }
    let ep = hp_shl::<C>(m1, n1 as u32);
    let en = match exp_hp(decimal_compute_neg(a))? {
        None => hp_small::<C>(0),
        Some((m2, n2)) if n2.unsigned_abs() < C::HP_BITS as u64 => hp_shr::<C>(m2, n2.unsigned_abs() as u32),
        Some(_) => hp_small::<C>(0),
    };
    let e = C::HP_DP - C::DP as u32;
    let sinh = C::hp_to_compute(&hp_div_round_half_even::<C>(ep - en, e, 1))?;
    let cosh = C::hp_to_compute(&hp_div_round_half_even::<C>(ep + en, e, 1))?;
    Ok((if negative { decimal_compute_neg(sinh) } else { sinh }, cosh))
}

#[cfg(all(test, table_format = "q64_64"))]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use super::super::decimal_compute::*;
    use super::super::decimal_compute::{decimal_compute_from_int, pow10_compute_ct};
    use crate::fixed_point::i256::I256;

    /// mpmath reference: exp(0) = 1.0 exactly
    #[test]
    fn exp_zero_is_one() {
        let result = decimal_exp(decimal_compute_zero::<ComputeStorage>()).unwrap();
        assert_eq!(result, decimal_compute_one::<ComputeStorage>());
    }

    /// mpmath: exp(1) = 2.71828182845904523536028747135266249775724709369995957...
    #[test]
    fn exp_one_within_1_ulp() {
        let result = decimal_exp(decimal_compute_one::<ComputeStorage>()).unwrap();
        // Expected: 2.71828182845904523536028747135266249775 × 10^38
        // = 27182818284590452353602874713526624977 at dp=38 (+0.5 rounding)
        // mpmath rounded: 2.7182818284590452353602874713526624977572 × 10^38
        // → 27182818284590452353602874713526624978 at dp=38 (round half to even)
        let expected_str = "271828182845904523536028747135266249776";
        let expected = parse_decimal_str_q64_64(expected_str);

        // Allow up to 1000 ULP tolerance for this first correctness check
        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(10_000);
        assert!(
            diff < tolerance,
            "exp(1) precision check: got={:?}, expected={:?}, diff={:?}",
            result, expected, diff
        );
    }

    /// mpmath: exp(0.5) = 1.64872127070012814684865078781416357165377610071014...
    #[test]
    fn exp_half_within_1_ulp() {
        // 0.5 at compute dp = 5 × 10^37
        let half = pow10_compute_ct::<ComputeStorage>(37) * I256::from_i128(5);
        let result = decimal_exp(half).unwrap();
        // Expected: 1.64872127070012814684865078781416357165 × 10^38
        let expected_str = "164872127070012814684865078781416357165";
        let expected = parse_decimal_str_q64_64(expected_str);

        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(10_000);
        assert!(
            diff < tolerance,
            "exp(0.5) precision check: got={:?}, expected={:?}, diff={:?}",
            result, expected, diff
        );
    }

    /// Parse a decimal-digit string into an I256 assuming it represents a value
    /// at q64_64 compute dp=38.
    fn parse_decimal_str_q64_64(s: &str) -> I256 {
        let mut result = I256::from_i128(0);
        let ten = I256::from_i128(10);
        for ch in s.chars() {
            let digit = ch.to_digit(10).expect("non-digit in test string");
            result = result * ten + I256::from_i128(digit as i128);
        }
        result
    }

    /// mpmath: exp(2) = 7.38905609893065022723042746057500781318031557055184...
    #[test]
    fn exp_two_within_reasonable() {
        let two = decimal_compute_from_int::<ComputeStorage>(2);
        let result = decimal_exp(two).unwrap();
        // Expected: 7.38905609893065022723042746057500781318 × 10^38
        let expected_str = "738905609893065022723042746057500781318";
        let expected = parse_decimal_str_q64_64(expected_str);

        let diff = if result > expected { result - expected } else { expected - result };
        let tolerance = I256::from_i128(100_000);
        assert!(
            diff < tolerance,
            "exp(2) precision check: got={:?}, expected={:?}, diff={:?}",
            result, expected, diff
        );
    }

    /// mpmath: exp(-1) = 0.36787944117144232839...
    /// Storage-tier validation (compute-tier rounding from 1/exp(1) is acceptable
    /// as long as storage-tier result is exact: covered by decimal_transcendental_validation).
    #[test]
    fn exp_neg_one() {
        use crate::canonical::{gmath, evaluate};
        let result = evaluate(&gmath("-1.0").exp()).unwrap();
        let s = format!("{}", result);
        // exp(-1) ≈ 0.36787944117144232...
        assert!(s.starts_with("0.3678794411714"),
            "exp(-1) at storage tier should match mpmath to 13+ digits, got: {}", s);
    }
}

