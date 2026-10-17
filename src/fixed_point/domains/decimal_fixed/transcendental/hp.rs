//! Wide decimal working precision shared by the decimal exp and sin/cos
//! reductions: non-negative values at `C::HP_DP` decimal places in `C::Hp`, the
//! integer tier above the compute tier (I256 for the i64 and i128 compute words).
//!
//! Word-level helpers only (scalar multiply and divide by u64, shifts,
//! products of two values below 2 at `C::HP_DP`); every narrowing is checked.

#[allow(unused_imports)]
use super::word::{DecimalWord, HpWord};
#[allow(unused_imports)]
use crate::fixed_point::width::WideInt;
use super::decimal_compute::decimal_compute_neg;
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;


/// Working decimal precision of the engine. Products of two values below 2
/// at this dp fit `C::Hp`; the storage result needs (result digits + storage
/// dp), per 0.6.x profile: realtime 10 + 4, compact 20 + 9, embedded 20 + 19, balanced 39 + 38,
/// scientific 77 + 77.

/// Bits of `C::Hp`.

/// Bits of the compute tier. `e^C::COMPUTE_BITS` exceeds every compute value,
/// and `e^-C::COMPUTE_BITS` is below half a unit at the compute dp.

// ============================================================================
// WORD ARITHMETIC ON NON-NEGATIVE C::Hp VALUES
// ============================================================================

#[inline]
pub(super) fn hp_small<C: DecimalWord>(v: u64) -> C::Hp {
    <C::Hp>::from_i128(v as i128)
}

#[inline]
pub(super) fn hp_is_negative<C: DecimalWord>(v: &C::Hp) -> bool {
    v.words()[v.words().len() - 1] >> 63 == 1
}

#[inline]
pub(super) fn hp_is_zero<C: DecimalWord>(v: &C::Hp) -> bool {
    v.words().iter().all(|&w| w == 0)
}

/// Number of significant bits of a non-negative value.
pub(super) fn hp_bit_length<C: DecimalWord>(v: &C::Hp) -> u32 {
    for i in (0..v.words().len()).rev() {
        if v.words()[i] != 0 {
            return i as u32 * 64 + (64 - v.words()[i].leading_zeros());
        }
    }
    0
}

/// `v * m` for `v >= 0`; panics if the product leaves `C::Hp` (the callers'
/// bounds make that unreachable).
pub(super) fn hp_mul_small<C: DecimalWord>(mut v: C::Hp, m: u64) -> C::Hp {
    let mut carry: u128 = 0;
    for w in v.words_mut().iter_mut() {
        let cur = (*w as u128) * (m as u128) + carry;
        *w = cur as u64;
        carry = cur >> 64;
    }
    assert!(carry == 0 && !hp_is_negative::<C>(&v), "decimal exp: working value outside its tier");
    v
}

/// `(v / d, v % d)` for `v >= 0`, `d > 0`.
pub(super) fn hp_divmod_small<C: DecimalWord>(mut v: C::Hp, d: u64) -> (C::Hp, u64) {
    let mut rem: u128 = 0;
    for w in v.words_mut().iter_mut().rev() {
        let cur = (rem << 64) | *w as u128;
        *w = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
    (v, rem as u64)
}

/// `v * 10^e` for `v >= 0`.
pub(super) fn hp_mul_pow10<C: DecimalWord>(mut v: C::Hp, mut e: u32) -> C::Hp {
    while e > 0 {
        let k = e.min(19);
        v = hp_mul_small::<C>(v, 10u64.pow(k));
        e -= k;
    }
    v
}

/// `floor(v / 10^e)` for `v >= 0`.
pub(super) fn hp_div_pow10<C: DecimalWord>(mut v: C::Hp, mut e: u32) -> C::Hp {
    while e > 0 {
        let k = e.min(19);
        v = hp_divmod_small::<C>(v, 10u64.pow(k)).0;
        e -= k;
    }
    v
}

pub(super) fn hp_pow10<C: DecimalWord>(e: u32) -> C::Hp {
    hp_mul_pow10::<C>(hp_small::<C>(1), e)
}

/// `v * 2^k` for `v >= 0`; the caller has checked the bit length.
pub(super) fn hp_shl<C: DecimalWord>(v: C::Hp, k: u32) -> C::Hp {
    let mut out = hp_small::<C>(0);
    let (words, bits) = ((k / 64) as usize, k % 64);
    let n = v.words().len();
    for i in (words..n).rev() {
        let lo = v.words()[i - words];
        let carry = if bits > 0 && i > words { v.words()[i - words - 1] >> (64 - bits) } else { 0 };
        out.words_mut()[i] = if bits > 0 { (lo << bits) | carry } else { lo };
    }
    out
}

/// `floor(v / 2^k)` for `v >= 0`.
pub(super) fn hp_shr<C: DecimalWord>(v: C::Hp, k: u32) -> C::Hp {
    let mut out = hp_small::<C>(0);
    let (words, bits) = ((k / 64) as usize, k % 64);
    let n = v.words().len();
    for i in 0..n.saturating_sub(words) {
        let hi = if bits > 0 && i + words + 1 < n { v.words()[i + words + 1] << (64 - bits) } else { 0 };
        out.words_mut()[i] = if bits > 0 { (v.words()[i + words] >> bits) | hi } else { v.words()[i + words] };
    }
    out
}

/// A small non-negative value as i128.
pub(super) fn hp_to_i128<C: DecimalWord>(v: &C::Hp) -> i128 {
    assert!(hp_bit_length::<C>(v) <= 126, "decimal exp: small value out of range");
    (v.words()[0] as i128) | ((v.words()[1] as i128) << 64)
}

/// Round `a * b / 10^C::HP_DP` (half up) for `a, b >= 0` below 2 at `C::HP_DP`.
pub(super) fn hp_mul<C: DecimalWord>(a: C::Hp, b: C::Hp, half: C::Hp) -> C::Hp {
    hp_div_pow10::<C>(a * b + half, C::HP_DP)
}

/// Round `v / n` (half up) for `v >= 0`.
pub(super) fn hp_div_small_round<C: DecimalWord>(v: C::Hp, n: u64) -> C::Hp {
    hp_divmod_small::<C>(v + hp_small::<C>(n / 2), n).0
}

/// `round(t / (10^e 2^k))`, half to even, for `t >= 0`; the caller has
/// checked that `10^e 2^k` fits.
pub(super) fn hp_div_round_half_even<C: DecimalWord>(t: C::Hp, e: u32, k: u32) -> C::Hp {
    let d = hp_shl::<C>(hp_pow10::<C>(e), k);
    let q = hp_shr::<C>(hp_div_pow10::<C>(t, e), k);
    let rem = t - hp_mul_pow10::<C>(hp_shl::<C>(q, k), e);
    match rem.cmp(&(d - rem)) {
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Greater => q + hp_small::<C>(1),
        std::cmp::Ordering::Equal => if q.words()[0] & 1 == 1 { q + hp_small::<C>(1) } else { q },
    }
}

/// A signed value at `C::HP_DP` rounded half to even to the compute dp,
/// `Err(TierOverflow)` when it does not fit the compute tier.
pub(super) fn hp_signed_to_compute<C: DecimalWord>(v: C::Hp) -> Result<C, OverflowDetected> {
    let negative = hp_is_negative::<C>(&v);
    let magnitude = if negative { -v } else { v };
    let q = C::hp_to_compute(&hp_div_round_half_even::<C>(magnitude, C::HP_DP - C::DP as u32, 0))?;
    Ok(if negative { decimal_compute_neg(q) } else { q })
}

