//! Wide decimal working precision shared by the decimal exp and sin/cos
//! reductions: non-negative values at `HP_DP` decimal places in `Hp`, the
//! integer tier above the compute tier (I256 on realtime and compact).
//!
//! Word-level helpers only (scalar multiply and divide by u64, shifts,
//! products of two values below 2 at `HP_DP`); every narrowing is checked.

use super::decimal_compute::{ComputeStorage, DECIMAL_COMPUTE_DP, decimal_compute_neg};
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
pub(super) use crate::fixed_point::i256::I256 as Hp;
#[cfg(table_format = "q64_64")]
pub(super) use crate::fixed_point::i512::I512 as Hp;
#[cfg(table_format = "q128_128")]
pub(super) use crate::fixed_point::I1024 as Hp;
#[cfg(table_format = "q256_256")]
pub(super) use crate::fixed_point::I2048 as Hp;

/// Working decimal precision of the engine. Products of two values below 2
/// at this dp fit `Hp`; the storage result needs (result digits + storage
/// dp): realtime 10 + 4, compact 20 + 9, embedded 20 + 19, balanced 39 + 38,
/// scientific 77 + 77.
#[cfg(table_format = "q16_16")]
pub(super) const HP_DP: u32 = 28;
#[cfg(table_format = "q32_32")]
pub(super) const HP_DP: u32 = 38;
#[cfg(table_format = "q64_64")]
pub(super) const HP_DP: u32 = 60;
#[cfg(table_format = "q128_128")]
pub(super) const HP_DP: u32 = 90;
#[cfg(table_format = "q256_256")]
pub(super) const HP_DP: u32 = 170;

/// Bits of `Hp`.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
pub(super) const HP_BITS: u32 = 256;
#[cfg(table_format = "q64_64")]
pub(super) const HP_BITS: u32 = 512;
#[cfg(table_format = "q128_128")]
pub(super) const HP_BITS: u32 = 1024;
#[cfg(table_format = "q256_256")]
pub(super) const HP_BITS: u32 = 2048;

/// Bits of the compute tier. `e^COMPUTE_BITS` exceeds every compute value,
/// and `e^-COMPUTE_BITS` is below half a unit at the compute dp.
#[cfg(table_format = "q16_16")]
pub(super) const COMPUTE_BITS: i64 = 64;
#[cfg(table_format = "q32_32")]
pub(super) const COMPUTE_BITS: i64 = 128;
#[cfg(table_format = "q64_64")]
pub(super) const COMPUTE_BITS: i64 = 256;
#[cfg(table_format = "q128_128")]
pub(super) const COMPUTE_BITS: i64 = 512;
#[cfg(table_format = "q256_256")]
pub(super) const COMPUTE_BITS: i64 = 1024;

// ============================================================================
// WORD ARITHMETIC ON NON-NEGATIVE Hp VALUES
// ============================================================================

#[inline]
pub(super) fn hp_small(v: u64) -> Hp {
    Hp::from_i128(v as i128)
}

#[inline]
pub(super) fn hp_is_negative(v: &Hp) -> bool {
    v.words[v.words.len() - 1] >> 63 == 1
}

#[inline]
pub(super) fn hp_is_zero(v: &Hp) -> bool {
    v.words.iter().all(|&w| w == 0)
}

/// Number of significant bits of a non-negative value.
pub(super) fn hp_bit_length(v: &Hp) -> u32 {
    for i in (0..v.words.len()).rev() {
        if v.words[i] != 0 {
            return i as u32 * 64 + (64 - v.words[i].leading_zeros());
        }
    }
    0
}

/// `v * m` for `v >= 0`; panics if the product leaves `Hp` (the callers'
/// bounds make that unreachable).
pub(super) fn hp_mul_small(mut v: Hp, m: u64) -> Hp {
    let mut carry: u128 = 0;
    for w in v.words.iter_mut() {
        let cur = (*w as u128) * (m as u128) + carry;
        *w = cur as u64;
        carry = cur >> 64;
    }
    assert!(carry == 0 && !hp_is_negative(&v), "decimal exp: working value outside its tier");
    v
}

/// `(v / d, v % d)` for `v >= 0`, `d > 0`.
pub(super) fn hp_divmod_small(mut v: Hp, d: u64) -> (Hp, u64) {
    let mut rem: u128 = 0;
    for w in v.words.iter_mut().rev() {
        let cur = (rem << 64) | *w as u128;
        *w = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
    (v, rem as u64)
}

/// `v * 10^e` for `v >= 0`.
pub(super) fn hp_mul_pow10(mut v: Hp, mut e: u32) -> Hp {
    while e > 0 {
        let k = e.min(19);
        v = hp_mul_small(v, 10u64.pow(k));
        e -= k;
    }
    v
}

/// `floor(v / 10^e)` for `v >= 0`.
pub(super) fn hp_div_pow10(mut v: Hp, mut e: u32) -> Hp {
    while e > 0 {
        let k = e.min(19);
        v = hp_divmod_small(v, 10u64.pow(k)).0;
        e -= k;
    }
    v
}

pub(super) fn hp_pow10(e: u32) -> Hp {
    hp_mul_pow10(hp_small(1), e)
}

/// `v * 2^k` for `v >= 0`; the caller has checked the bit length.
pub(super) fn hp_shl(v: Hp, k: u32) -> Hp {
    let mut out = hp_small(0);
    let (words, bits) = ((k / 64) as usize, k % 64);
    let n = v.words.len();
    for i in (words..n).rev() {
        let lo = v.words[i - words];
        let carry = if bits > 0 && i > words { v.words[i - words - 1] >> (64 - bits) } else { 0 };
        out.words[i] = if bits > 0 { (lo << bits) | carry } else { lo };
    }
    out
}

/// `floor(v / 2^k)` for `v >= 0`.
pub(super) fn hp_shr(v: Hp, k: u32) -> Hp {
    let mut out = hp_small(0);
    let (words, bits) = ((k / 64) as usize, k % 64);
    let n = v.words.len();
    for i in 0..n.saturating_sub(words) {
        let hi = if bits > 0 && i + words + 1 < n { v.words[i + words + 1] << (64 - bits) } else { 0 };
        out.words[i] = if bits > 0 { (v.words[i + words] >> bits) | hi } else { v.words[i + words] };
    }
    out
}

/// A small non-negative value as i128.
pub(super) fn hp_to_i128(v: &Hp) -> i128 {
    assert!(hp_bit_length(v) <= 126, "decimal exp: small value out of range");
    (v.words[0] as i128) | ((v.words[1] as i128) << 64)
}

/// Round `a * b / 10^HP_DP` (half up) for `a, b >= 0` below 2 at `HP_DP`.
pub(super) fn hp_mul(a: Hp, b: Hp, half: Hp) -> Hp {
    hp_div_pow10(a * b + half, HP_DP)
}

/// Round `v / n` (half up) for `v >= 0`.
pub(super) fn hp_div_small_round(v: Hp, n: u64) -> Hp {
    hp_divmod_small(v + hp_small(n / 2), n).0
}

/// `round(t / (10^e 2^k))`, half to even, for `t >= 0`; the caller has
/// checked that `10^e 2^k` fits.
pub(super) fn hp_div_round_half_even(t: Hp, e: u32, k: u32) -> Hp {
    let d = hp_shl(hp_pow10(e), k);
    let q = hp_shr(hp_div_pow10(t, e), k);
    let rem = t - hp_mul_pow10(hp_shl(q, k), e);
    match rem.cmp(&(d - rem)) {
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Greater => q + hp_small(1),
        std::cmp::Ordering::Equal => if q.words[0] & 1 == 1 { q + hp_small(1) } else { q },
    }
}

pub(super) fn compute_to_hp(x: ComputeStorage) -> Hp {
    #[cfg(table_format = "q16_16")]
    { Hp::from_i128(x as i128) }
    #[cfg(table_format = "q32_32")]
    { Hp::from_i128(x) }
    #[cfg(table_format = "q64_64")]
    { Hp::from_i256(x) }
    #[cfg(table_format = "q128_128")]
    { Hp::from_i512(x) }
    #[cfg(table_format = "q256_256")]
    { Hp::from_i1024(x) }
}

/// A non-negative value as a compute value, `Err(TierOverflow)` if it does
/// not fit.
pub(super) fn hp_to_compute(v: &Hp) -> Result<ComputeStorage, OverflowDetected> {
    if hp_bit_length(v) > COMPUTE_BITS as u32 - 1 {
        return Err(OverflowDetected::TierOverflow);
    }
    let w = &v.words;
    #[cfg(table_format = "q16_16")]
    { Ok(w[0] as i64) }
    #[cfg(table_format = "q32_32")]
    { Ok((w[0] as i128) | ((w[1] as i128) << 64)) }
    #[cfg(table_format = "q64_64")]
    { Ok(crate::fixed_point::i256::I256::from_words([w[0], w[1], w[2], w[3]])) }
    #[cfg(table_format = "q128_128")]
    {
        let mut words = [0u64; 8];
        words.copy_from_slice(&w[..8]);
        Ok(crate::fixed_point::i512::I512::from_words(words))
    }
    #[cfg(table_format = "q256_256")]
    {
        let mut words = [0u64; 16];
        words.copy_from_slice(&w[..16]);
        Ok(crate::fixed_point::I1024::from_words(words))
    }
}

/// A signed value at `HP_DP` rounded half to even to the compute dp,
/// `Err(TierOverflow)` when it does not fit the compute tier.
pub(super) fn hp_signed_to_compute(v: Hp) -> Result<ComputeStorage, OverflowDetected> {
    let negative = hp_is_negative(&v);
    let magnitude = if negative { -v } else { v };
    let q = hp_to_compute(&hp_div_round_half_even(magnitude, HP_DP - DECIMAL_COMPUTE_DP as u32, 0))?;
    Ok(if negative { decimal_compute_neg(q) } else { q })
}

/// `floor(a / b)` for `a >= 0`, `b > 0` (the type's long division; used
/// once per thread for constants).
pub(super) fn hp_div(a: Hp, b: Hp) -> Hp {
    #[cfg(table_format = "q256_256")]
    { crate::fixed_point::domains::binary_fixed::i2048::i2048_div(a, b) }
    #[cfg(not(table_format = "q256_256"))]
    { a / b }
}
