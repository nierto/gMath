//! Output formatting — value to decimal string conversion
//!
//! All conversions are float-free, using integer long division.

use super::BinaryStorage;
#[cfg(table_format = "q16_16")]
use crate::fixed_point::frac_config;
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

/// Convert BinaryStorage to decimal string using integer-only arithmetic.
///
/// Profile-dispatched: uses the correct Q-format for each profile.
pub(super) fn binary_storage_to_decimal_string(val: BinaryStorage, max_digits: usize) -> String {
    #[cfg(table_format = "q64_64")]
    {
        // Q64.64: i128
        let is_negative = val < 0;
        let abs_val = if is_negative { -val } else { val };
        let int_part = (abs_val >> 64) as u64;
        let frac_part = abs_val as u64; // Lower 64 bits

        let digits = max_digits.min(19);
        let mut frac_str = String::with_capacity(digits);
        let mut remainder = frac_part as u128;
        for _ in 0..digits {
            remainder *= 10;
            let digit = (remainder >> 64) as u8;
            frac_str.push((b'0' + digit) as char);
            remainder &= (1u128 << 64) - 1;
        }

        let sign = if is_negative { "-" } else { "" };
        format!("{}{}.{}", sign, int_part, frac_str)
    }

    #[cfg(table_format = "q32_32")]
    {
        // Q32.32: i64
        let is_negative = val < 0;
        let abs_val = if is_negative { -val } else { val };
        let int_part = (abs_val >> 32) as u32;
        let frac_part = abs_val as u32; // Lower 32 bits

        let digits = max_digits.min(9);
        let mut frac_str = String::with_capacity(digits);
        let mut remainder = frac_part as u64;
        for _ in 0..digits {
            remainder *= 10;
            let digit = (remainder >> 32) as u8;
            frac_str.push((b'0' + digit) as char);
            remainder &= (1u64 << 32) - 1;
        }

        let sign = if is_negative { "-" } else { "" };
        format!("{}{}.{}", sign, int_part, frac_str)
    }

    #[cfg(table_format = "q16_16")]
    {
        // i32 Q-format with configurable FRAC_BITS
        let fb = frac_config::FRAC_BITS;
        let is_negative = val < 0;
        let abs_val = if is_negative { -val } else { val };
        let int_part = (abs_val >> fb) as u32;
        let frac_part = abs_val as u32 & ((1u32 << fb) - 1);

        let digits = max_digits.min(frac_config::MAX_DECIMAL_DIGITS);
        let mut frac_str = String::with_capacity(digits);
        let mut remainder = frac_part;
        for _ in 0..digits {
            remainder *= 10;
            let digit = (remainder >> fb) as u8;
            frac_str.push((b'0' + digit) as char);
            remainder &= (1u32 << fb) - 1;
        }

        let sign = if is_negative { "-" } else { "" };
        format!("{}{}.{}", sign, int_part, frac_str)
    }

    #[cfg(table_format = "q128_128")]
    {
        // Q128.128: I256
        let is_negative = val < I256::zero();
        let abs_val = if is_negative { -val } else { val };
        let int_part = (abs_val >> 128).as_i128() as u128;
        let mut frac = abs_val.as_i128() as u128;

        let digits = max_digits.min(38);
        let mut frac_str = String::with_capacity(digits);
        for _ in 0..digits {
            let frac_i256 = I256::from_u128(frac);
            let multiplied = frac_i256 * I256::from_i128(10);
            let digit = (multiplied >> 128).as_i128() as u8;
            frac_str.push((b'0' + digit) as char);
            frac = multiplied.as_i128() as u128;
        }

        let sign = if is_negative { "-" } else { "" };
        format!("{}{}.{}", sign, int_part, frac_str)
    }

    #[cfg(table_format = "q256_256")]
    {
        // Q256.256: I512
        let is_negative = val < I512::zero();
        let abs_val = if is_negative { -val } else { val };
        let int_part = (abs_val >> 256).as_i256().as_i128() as u128;

        let mask_lower_256 = I512::from_words([
            0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
            0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF,
            0, 0, 0, 0
        ]);
        let mut frac = abs_val & mask_lower_256;

        let digits = max_digits.min(76);
        let mut frac_str = String::with_capacity(digits);
        for _ in 0..digits {
            let frac_i1024 = I1024::from_i512(frac);
            let multiplied = frac_i1024 * I1024::from_i128(10);
            let digit = ((multiplied >> 256).as_i512().as_i256().as_i128() & 0xFF) as u8;
            frac_str.push((b'0' + digit) as char);
            frac = (multiplied & I1024::from_i512(mask_lower_256)).as_i512();
        }

        let sign = if is_negative { "-" } else { "" };
        format!("{}{}.{}", sign, int_part, frac_str)
    }

}

/// Format a Decimal StackValue as "integer.fraction" using native-width arithmetic.
///
/// Uses BinaryStorage (i128/I256/I512) for the division so that large decimal-place
/// counts (e.g., dec=62 from chained multiplications) don't overflow.
pub(super) fn decimal_storage_to_string(dec: u8, val: &BinaryStorage) -> String {
    let dp = dec as usize;

    #[cfg(table_format = "q64_64")]
    {
        // Q64.64: BinaryStorage = i128. Promote to I256 for division so
        // we never overflow on 10^dec (which exceeds i128 when dec > 38).
        let v = *val;
        let is_negative = v < 0;
        let abs_val = if is_negative { I256::from_i128(v.wrapping_neg()) } else { I256::from_i128(v) };
        let divisor = pow10_i256(dec);
        let integer_part = abs_val / divisor;
        let fractional_part = abs_val % divisor;
        let sign = if is_negative { "-" } else { "" };
        if dp == 0 {
            format!("{}{}", sign, i256_to_decimal_string(integer_part))
        } else {
            format!("{}{}.{}", sign, i256_to_decimal_string(integer_part), i256_to_padded_decimal_string(fractional_part, dp))
        }
    }

    #[cfg(table_format = "q32_32")]
    {
        // Q32.32: BinaryStorage = i64. Promote to i128 for division so
        // we never overflow on 10^dec.
        let v = *val;
        let is_negative = v < 0;
        let abs_val = if is_negative { (v as i128).wrapping_neg() } else { v as i128 };
        let divisor = pow10_i128(dec);
        let integer_part = abs_val / divisor;
        let fractional_part = abs_val % divisor;
        let sign = if is_negative { "-" } else { "" };
        if dp == 0 {
            format!("{}{}", sign, integer_part)
        } else {
            format!("{}{}.{:0>width$}", sign, integer_part, fractional_part, width = dp)
        }
    }

    #[cfg(table_format = "q16_16")]
    {
        // Q16.16: BinaryStorage = i32. Promote to i64 for division so
        // we never overflow on 10^dec.
        let v = *val;
        let is_negative = v < 0;
        let abs_val = if is_negative { (v as i64).wrapping_neg() } else { v as i64 };
        let divisor = pow10_i64(dec);
        let integer_part = abs_val / divisor;
        let fractional_part = abs_val % divisor;
        let sign = if is_negative { "-" } else { "" };
        if dp == 0 {
            format!("{}{}", sign, integer_part)
        } else {
            format!("{}{}.{:0>width$}", sign, integer_part, fractional_part, width = dp)
        }
    }

    #[cfg(table_format = "q128_128")]
    {
        // Q128.128: BinaryStorage = I256. Division done natively in I256.
        let is_negative = *val < I256::zero();
        let abs_val = if is_negative { -*val } else { *val };
        let divisor = pow10_i256(dec);
        let integer_part = abs_val / divisor;
        let fractional_part = abs_val % divisor;
        let sign = if is_negative { "-" } else { "" };
        if dp == 0 {
            format!("{}{}", sign, i256_to_decimal_string(integer_part))
        } else {
            format!("{}{}.{}", sign, i256_to_decimal_string(integer_part), i256_to_padded_decimal_string(fractional_part, dp))
        }
    }

    #[cfg(table_format = "q256_256")]
    {
        // Q256.256: BinaryStorage = I512. For dec > 76, promote to I1024.
        let is_negative = *val < I512::zero();
        let abs_val = if is_negative { -*val } else { *val };

        if dp <= 154 {
            // I512 can hold 10^154. Use native I512 division.
            let divisor = pow10_i512(dec);
            let integer_part = abs_val / divisor;
            let fractional_part = abs_val % divisor;
            let sign = if is_negative { "-" } else { "" };
            if dp == 0 {
                format!("{}{}", sign, i512_to_decimal_string(integer_part))
            } else {
                format!("{}{}.{}", sign, i512_to_decimal_string(integer_part), i512_to_padded_decimal_string(fractional_part, dp))
            }
        } else {
            // dp > 154: promote to I1024
            let v1024 = I1024::from_i512(abs_val);
            let divisor = pow10_i1024(dec);
            let integer_part = v1024 / divisor;
            let fractional_part = v1024 % divisor;
            let sign = if is_negative { "-" } else { "" };
            if dp == 0 {
                format!("{}{}", sign, i1024_to_decimal_string(integer_part))
            } else {
                format!("{}{}.{}", sign, i1024_to_decimal_string(integer_part), i1024_to_padded_decimal_string(fractional_part, dp))
            }
        }
    }
}

/// Compute 10^exp as I256 — O(1) via const lookup table (covers 10^0..10^76)
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

/// Compute 10^exp as i128 — covers 10^0..10^38 (i128 max ~1.7e38)
#[cfg(any(table_format = "q32_32"))]
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

/// Compute 10^exp as i64 — covers 10^0..10^18 (i64 max ~9.2e18)
#[cfg(any(table_format = "q16_16"))]
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

#[cfg(any(table_format = "q256_256"))]
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

#[cfg(any(table_format = "q256_256"))]
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

#[cfg(any(table_format = "q256_256"))]
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

#[cfg(any(table_format = "q256_256"))]
pub(super) fn pow10_i1024(exp: u8) -> I1024 {
    let mut result = I1024::from_i128(1);
    let ten = I1024::from_i128(10);
    for _ in 0..exp {
        result = result * ten;
    }
    result
}

#[cfg(any(table_format = "q256_256"))]
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

#[cfg(any(table_format = "q256_256"))]
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

