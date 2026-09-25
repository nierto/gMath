//! Exact decimal-literal conversion to binary fixed point, integers only.
//!
//! Grammar: `[+-]? (digits ('.' digits?)? | '.' digits) ([eE] [+-]? digits)?`,
//! surrounding ASCII whitespace ignored. The literal is converted exactly and
//! rounded once to the nearest multiple of `2^-frac_bits`, ties toward
//! +infinity (the crate's binary rounding rule). Any number of digits is
//! accepted; digits that cannot affect the rounding only set a sticky flag.

use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
#[cfg(table_format = "q128_128")]
use crate::fixed_point::I256;
#[cfg(table_format = "q256_256")]
use crate::fixed_point::I512;

/// Magnitude words: 512 bits of storage plus room for the rounding carry.
pub(crate) const WORDS: usize = 9;
/// Fraction digits that can matter: every rounding boundary of a Q.F grid,
/// `(2k + 1) / 2^(F + 1)`, has exactly F + 1 decimal fraction digits, so
/// digits past position F + 1 only decide ties. F <= 256.
const MAX_FRAC_DIGITS: usize = 257;
/// Base-10^9 limbs holding MAX_FRAC_DIGITS digits.
const FRAC_LIMBS: usize = (MAX_FRAC_DIGITS + 8) / 9;
/// 10^156 > 2^512: a longer integer part cannot fit any storage width.
const MAX_INT_DIGITS: i64 = 156;
const BILLION: u64 = 1_000_000_000;

/// Signed magnitude of a converted literal.
pub(crate) struct Converted {
    pub negative: bool,
    pub words: [u64; WORDS],
}

/// `None` if `s` does not match the grammar; otherwise the literal rounded
/// to `frac_bits` fractional bits as a two's complement value of `width`
/// bits (`Err(TierOverflow)` when it does not fit; the minimum
/// `-2^(width - 1)` fits).
pub(crate) fn parse(s: &str, frac_bits: u32, width: u32) -> Option<Result<Converted, OverflowDetected>> {
    debug_assert!(frac_bits as usize + 1 <= MAX_FRAC_DIGITS && width as usize <= 64 * (WORDS - 1));
    let b = s.trim().as_bytes();
    let len = b.len();
    let mut i = 0;
    let negative = match b.first() {
        Some(b'-') => { i = 1; true }
        Some(b'+') => { i = 1; false }
        _ => false,
    };
    let int_start = i;
    while i < len && b[i].is_ascii_digit() { i += 1; }
    let int_digits = &b[int_start..i];
    let mut frac_digits: &[u8] = &[];
    if i < len && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < len && b[i].is_ascii_digit() { i += 1; }
        frac_digits = &b[frac_start..i];
    }
    if int_digits.is_empty() && frac_digits.is_empty() {
        return None;
    }
    let mut exp10: i64 = 0;
    if i < len && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        let exp_negative = match b.get(i) {
            Some(b'-') => { i += 1; true }
            Some(b'+') => { i += 1; false }
            _ => false,
        };
        let exp_start = i;
        while i < len && b[i].is_ascii_digit() {
            // saturate far beyond any representable magnitude
            exp10 = (exp10 * 10 + (b[i] - b'0') as i64).min(1_000_000_000);
            i += 1;
        }
        if i == exp_start {
            return None;
        }
        if exp_negative { exp10 = -exp10; }
    }
    if i != len {
        return None;
    }
    Some(convert(negative, int_digits, frac_digits, exp10, frac_bits, width))
}

fn convert(
    negative: bool,
    int_digits: &[u8],
    frac_digits: &[u8],
    exp10: i64,
    frac_bits: u32,
    width: u32,
) -> Result<Converted, OverflowDetected> {
    let total = int_digits.len() + frac_digits.len();
    let digit = |k: usize| -> u8 {
        if k < int_digits.len() { int_digits[k] - b'0' } else { frac_digits[k - int_digits.len()] - b'0' }
    };
    // Significant digits: digit(lead..end), no leading or trailing zeros.
    let mut lead = 0;
    while lead < total && digit(lead) == 0 { lead += 1; }
    if lead == total {
        return Ok(Converted { negative: false, words: [0; WORDS] });
    }
    let mut end = total;
    while digit(end - 1) == 0 { end -= 1; }
    let n = (end - lead) as i64;
    let sig = |k: i64| -> u8 { if k >= 0 && k < n { digit(lead + k as usize) } else { 0 } };
    // The point sits after `point` significant digits (negative: leading zeros).
    let point = int_digits.len() as i64 + exp10 - lead as i64;
    if point > MAX_INT_DIGITS {
        return Err(OverflowDetected::TierOverflow);
    }

    // Integer part: sig(0..point), zero-padded.
    let mut int_words = [0u64; WORDS];
    for k in 0..point.max(0) {
        let mut carry = sig(k) as u128;
        for w in int_words.iter_mut() {
            let v = (*w as u128) * 10 + carry;
            *w = v as u64;
            carry = v >> 64;
        }
    }
    let int_bits = bit_length(&int_words);
    if int_bits > 0 && int_bits + frac_bits > width {
        return Err(OverflowDetected::TierOverflow);
    }
    let mut words = shl(&int_words, frac_bits);

    // Fraction part: the first frac_bits + 1 digits after the point, in
    // base-10^9 limbs (limb 0 most significant); later digits are sticky.
    let frac_len = (n - point).max(0);
    let kept = frac_len.min(frac_bits as i64 + 1);
    let mut limbs = [0u32; FRAC_LIMBS];
    for j in 0..kept {
        let limb = (j / 9) as usize;
        limbs[limb] = limbs[limb] * 10 + sig(point + j) as u32;
    }
    // Right-pad the last partial limb with zeros.
    if kept % 9 != 0 {
        let last = (kept / 9) as usize;
        for _ in 0..(9 - kept % 9) { limbs[last] *= 10; }
    }
    let used = ((kept + 8) / 9) as usize;
    let mut sticky = frac_len > kept;

    // Doubling the fraction k times shifts its next k bits into the carry.
    let mut round_bit = false;
    let mut produced = 0u32;
    while produced < frac_bits + 1 {
        let k = (frac_bits + 1 - produced).min(32);
        let mut carry = 0u64;
        for limb in limbs[..used].iter_mut().rev() {
            let v = ((*limb as u64) << k) + carry;
            *limb = (v % BILLION) as u32;
            carry = v / BILLION;
        }
        for u in 0..k {
            let bit = (carry >> (k - 1 - u)) & 1 == 1;
            let t = produced + u; // weight 2^-(t + 1)
            if t < frac_bits {
                if bit {
                    let pos = (frac_bits - 1 - t) as usize;
                    words[pos / 64] |= 1u64 << (pos % 64);
                }
            } else {
                round_bit = bit;
            }
        }
        produced += k;
    }
    sticky |= limbs[..used].iter().any(|&l| l != 0);

    // Nearest, ties toward +infinity: a tie rounds a positive magnitude up
    // and a negative magnitude down.
    if round_bit && (!negative || sticky) {
        for w in words.iter_mut() {
            let (v, overflow) = w.overflowing_add(1);
            *w = v;
            if !overflow { break; }
        }
    }

    let bits = bit_length(&words);
    let is_zero = bits == 0;
    let fits = bits < width
        || (negative && bits == width && trailing_zeros_below(&words, width - 1));
    if !fits {
        return Err(OverflowDetected::TierOverflow);
    }
    Ok(Converted { negative: negative && !is_zero, words })
}

fn bit_length(words: &[u64; WORDS]) -> u32 {
    for (i, &w) in words.iter().enumerate().rev() {
        if w != 0 {
            return i as u32 * 64 + (64 - w.leading_zeros());
        }
    }
    0
}

/// True when every bit below `pos` is zero.
fn trailing_zeros_below(words: &[u64; WORDS], pos: u32) -> bool {
    let full = (pos / 64) as usize;
    words[..full].iter().all(|&w| w == 0) && (words[full] & ((1u64 << (pos % 64)) - 1)) == 0
}

fn shl(words: &[u64; WORDS], shift: u32) -> [u64; WORDS] {
    let mut out = [0u64; WORDS];
    let (word_shift, bit_shift) = ((shift / 64) as usize, shift % 64);
    for i in (word_shift..WORDS).rev() {
        let src = i - word_shift;
        out[i] = words[src] << bit_shift;
        if bit_shift != 0 && src > 0 {
            out[i] |= words[src - 1] >> (64 - bit_shift);
        }
    }
    out
}

/// A converted literal of at most 128 bits as `i128`.
pub(crate) fn to_i128(c: &Converted) -> i128 {
    let magnitude = ((c.words[1] as u128) << 64 | c.words[0] as u128) as i128;
    // the minimum's magnitude 2^127 reads as i128::MIN, its own negation
    if c.negative { magnitude.wrapping_neg() } else { magnitude }
}

/// A converted literal (range-checked for the storage width) as a storage raw.
pub(crate) fn to_storage(c: &Converted) -> BinaryStorage {
    let w = &c.words;
    // the minimum's magnitude 2^(W-1) reads as the minimum, its own negation
    #[cfg(table_format = "q16_16")]
    let magnitude = w[0] as u32 as i32;
    #[cfg(table_format = "q32_32")]
    let magnitude = w[0] as i64;
    #[cfg(table_format = "q64_64")]
    let magnitude = ((w[1] as u128) << 64 | w[0] as u128) as i128;
    #[cfg(table_format = "q128_128")]
    let magnitude = I256::from_words([w[0], w[1], w[2], w[3]]);
    #[cfg(table_format = "q256_256")]
    let magnitude = I512::from_words([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
    #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
    { if c.negative { magnitude.wrapping_neg() } else { magnitude } }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    { if c.negative { -magnitude } else { magnitude } }
}
