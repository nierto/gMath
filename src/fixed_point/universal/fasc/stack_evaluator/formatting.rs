//! Output formatting: value to decimal string conversion
//!
//! All conversions are float-free, using integer long division.

use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
use crate::fixed_point::I1024;
use crate::fixed_point::domains::symbolic::rational::rational_number::RationalNumber;

/// Convert a RationalNumber (num/den) to a human-readable decimal string
/// via I1024 long division. Always exact for the digits produced.
///
/// For integer rationals (den=1), returns just the integer.
/// For non-integer rationals, produces up to `max_digits` fractional digits.
/// Trailing zeros are stripped.
///
/// Uses I1024 arithmetic (always available, no feature gates) to avoid overflow
/// during `remainder * 10`. Covers tiers 1-7 via `try_as_i512_pair()`.
pub(super) fn rational_to_decimal_string(r: &RationalNumber, max_digits: usize) -> String {

    let parts = r.extract_native();
    let (num_512, den_512) = match parts.try_as_i512_pair() {
        Some(pair) => pair,
        None => {
            // Tier 8 (BigInt) — fall back to RationalNumber Display (num/den format)
            return format!("{}", r);
        }
    };

    let num = I1024::from_i512(num_512);
    let den = I1024::from_i512(den_512);
    let zero = I1024::zero();
    let ten = I1024::from_i128(10);

    let is_negative = (num.words[15] & 0x8000_0000_0000_0000) != 0;
    let abs_num = if is_negative { -num } else { num };

    let int_part = abs_num / den;
    let mut remainder = abs_num % den;

    // Convert integer part to decimal string (digit-at-a-time extraction)
    let int_str = if int_part == zero {
        "0".to_string()
    } else {
        let mut digits = Vec::new();
        let mut n = int_part;
        while n != zero {
            let digit = (n % ten).as_i128() as u8;
            digits.push((b'0' + digit) as char);
            n = n / ten;
        }
        digits.reverse();
        digits.into_iter().collect()
    };

    if remainder == zero {
        // Exact integer
        if is_negative {
            return format!("-{}", int_str);
        } else {
            return int_str;
        }
    }

    // Fractional digits via long division
    let mut frac_digits = String::with_capacity(max_digits);
    for _ in 0..max_digits {
        remainder = remainder * ten;
        let digit = (remainder / den).as_i128() as u8;
        remainder = remainder % den;
        frac_digits.push((b'0' + digit) as char);
        if remainder == zero {
            break; // Exact terminating decimal
        }
    }

    // Strip trailing zeros
    let trimmed = frac_digits.trim_end_matches('0');

    if is_negative {
        format!("-{}.{}", int_str, trimmed)
    } else {
        format!("{}.{}", int_str, trimmed)
    }
}

/// Compute 10^exp as I256: O(1) via const lookup table (covers 10^0..10^76)
#[allow(dead_code)]
pub(super) fn pow10_i256(exp: u8) -> I256 {
    use super::compute::pow10_table_i256;
    const TABLE: [I256; 39] = pow10_table_i256();
    if (exp as usize) < TABLE.len() {
        TABLE[exp as usize]
    } else {
        // Fallback for exp >= 39 (I256 can hold ~10^76)
        let mut result = TABLE[38];
        let ten = I256::from_i128(10);
        for _ in 38..exp {
            result = result * ten;
        }
        result
    }
}

/// Convert I256 to decimal string (unsigned, no leading zeros)
#[allow(dead_code)]
pub(super) fn i256_to_decimal_string(mut val: I256) -> String {
    if val == I256::zero() {
        return "0".to_string();
    }
    let ten = I256::from_i128(10);
    let mut digits = Vec::new();
    while val > I256::zero() {
        let digit = (val % ten).as_i128() as u8;
        digits.push(b'0' + digit);
        val = val / ten;
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}

/// Convert I256 to zero-padded decimal string of exactly `width` digits
#[allow(dead_code)]
pub(super) fn i256_to_padded_decimal_string(mut val: I256, width: usize) -> String {
    let ten = I256::from_i128(10);
    let mut digits = Vec::with_capacity(width);
    for _ in 0..width {
        let digit = (val % ten).as_i128().unsigned_abs() as u8;
        digits.push(b'0' + digit);
        val = val / ten;
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}

/// Compute 10^exp as i128: covers 10^0..10^38 (i128 max ~1.7e38)
#[allow(dead_code)]
pub(super) fn pow10_i128(exp: u8) -> i128 {
    const TABLE: [i128; 39] = {
        let mut t = [0i128; 39];
        t[0] = 1;
        let mut i = 1;
        while i < 39 {
            t[i] = t[i - 1] * 10;
            i += 1;
        }
        t
    };
    if (exp as usize) < TABLE.len() {
        TABLE[exp as usize]
    } else {
        // Fallback for exp >= 39 — unlikely for Q32.32 but safe
        let mut result = TABLE[38];
        let mut i = 38u8;
        while i < exp {
            result = result.saturating_mul(10);
            i += 1;
        }
        result
    }
}

/// Compute 10^exp as i64: covers 10^0..10^18 (i64 max ~9.2e18)
#[allow(dead_code)]
pub(super) fn pow10_i64(exp: u8) -> i64 {
    const TABLE: [i64; 19] = {
        let mut t = [0i64; 19];
        t[0] = 1;
        let mut i = 1;
        while i < 19 {
            t[i] = t[i - 1] * 10;
            i += 1;
        }
        t
    };
    if (exp as usize) < TABLE.len() {
        TABLE[exp as usize]
    } else {
        // Fallback for exp >= 19 — unlikely for Q16.16 but safe
        let mut result = TABLE[18];
        let mut i = 18u8;
        while i < exp {
            result = result.saturating_mul(10);
            i += 1;
        }
        result
    }
}

pub(super) fn pow10_i512(exp: u8) -> I512 {
    use super::compute::pow10_table_i512;
    const TABLE: [I512; 77] = pow10_table_i512();
    if (exp as usize) < TABLE.len() {
        TABLE[exp as usize]
    } else {
        // Fallback for exp >= 77 (I512 can hold ~10^153)
        let mut result = TABLE[76];
        let ten = I512::from_i128(10);
        for _ in 76..exp {
            result = result * ten;
        }
        result
    }
}

pub(super) fn i512_to_decimal_string(mut val: I512) -> String {
    if val == I512::zero() {
        return "0".to_string();
    }
    let ten = I512::from_i128(10);
    let mut digits = Vec::new();
    while val > I512::zero() {
        let digit = (val % ten).as_i128() as u8;
        digits.push(b'0' + digit);
        val = val / ten;
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}

pub(super) fn i512_to_padded_decimal_string(mut val: I512, width: usize) -> String {
    let ten = I512::from_i128(10);
    let mut digits = Vec::with_capacity(width);
    for _ in 0..width {
        let digit = (val % ten).as_i128().unsigned_abs() as u8;
        digits.push(b'0' + digit);
        val = val / ten;
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}

pub(super) fn pow10_i1024(exp: u8) -> I1024 {
    let mut result = I1024::from_i128(1);
    let ten = I1024::from_i128(10);
    for _ in 0..exp {
        result = result * ten;
    }
    result
}

pub(super) fn i1024_to_decimal_string(mut val: I1024) -> String {
    if val == I1024::zero() {
        return "0".to_string();
    }
    let ten = I1024::from_i128(10);
    let mut digits = Vec::new();
    while val > I1024::zero() {
        let digit = (val % ten).as_i128() as u8;
        digits.push(b'0' + digit);
        val = val / ten;
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}

pub(super) fn i1024_to_padded_decimal_string(mut val: I1024, width: usize) -> String {
    let ten = I1024::from_i128(10);
    let mut digits = Vec::with_capacity(width);
    for _ in 0..width {
        let digit = (val % ten).as_i128().unsigned_abs() as u8;
        digits.push(b'0' + digit);
        val = val / ten;
    }
    digits.reverse();
    String::from_utf8(digits).unwrap()
}

