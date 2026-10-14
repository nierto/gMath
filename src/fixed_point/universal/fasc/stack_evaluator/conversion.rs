//! Storage type conversion helpers
//!
//! Decimal-to-rational reduction and trailing zero detection.

#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, OverflowDetected};

/// Reduce a decimal rational (scaled_value / 10^dp) by common factors of 2 and 5.
pub(super) fn reduce_decimal_to_rational(raw_num: i128, decimals: u8) -> Result<RationalNumber, OverflowDetected> {
    if raw_num == 0 {
        return Ok(RationalNumber::new(0, 1));
    }
    let mut num = raw_num;
    let mut den = 10_u128.pow(decimals as u32);
    while num % 2 == 0 && den % 2 == 0 {
        num /= 2;
        den /= 2;
    }
    while num % 5 == 0 && den % 5 == 0 {
        num /= 5;
        den /= 5;
    }
    Ok(RationalNumber::new(num, den))
}

/// Count trailing zero bits in I256 (little-endian words)
#[allow(dead_code)]
pub(super) fn trailing_zeros_i256(val: &I256) -> u32 {
    let w = val.words;
    if w[0] != 0 { return w[0].trailing_zeros(); }
    if w[1] != 0 { return 64 + w[1].trailing_zeros(); }
    if w[2] != 0 { return 128 + w[2].trailing_zeros(); }
    if w[3] != 0 { return 192 + w[3].trailing_zeros(); }
    256 // value is zero
}

/// Count trailing zero bits in I512 (little-endian words)
pub(super) fn trailing_zeros_i512(val: &I512) -> u32 {
    let w = val.words;
    for i in 0..8 {
        if w[i] != 0 {
            return (i as u32) * 64 + w[i].trailing_zeros();
        }
    }
    512 // value is zero
}
