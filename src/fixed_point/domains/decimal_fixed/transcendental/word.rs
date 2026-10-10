//! The decimal compute tier's word. Every decimal engine is generic over
//! the compute word `C`; the tier below it (`Narrow`), the word above it
//! (`Wide`), the high-precision working word (`Hp`) and the digit budgets
//! are the word's associated items. The bodies are the 0.6.7 profile arms,
//! transplanted; the profile-bound names in `decimal_compute` delegate here.

#![allow(clippy::all, unused_braces)]

use crate::fixed_point::domains::binary_fixed::i2048::i2048_div;
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;
use crate::fixed_point::width::{Storage, WideInt};
use crate::fixed_point::{I1024, I2048, I256, I512};
#[allow(unused_imports)]
use super::decimal_compute::*;
#[allow(unused_imports)]
use super::hp::*;

/// A multi-word integer whose limbs the high-precision helpers walk.
pub trait HpWord: WideInt {
    #[doc(hidden)]
    fn words(&self) -> &[u64];
    #[doc(hidden)]
    fn words_mut(&mut self) -> &mut [u64];
}
macro_rules! hp_word {
    ($($t:ty),*) => { $(impl HpWord for $t {
        #[inline] fn words(&self) -> &[u64] { &self.words }
        #[inline] fn words_mut(&mut self) -> &mut [u64] { &mut self.words }
    })* };
}
hp_word!(I256, I512, I1024, I2048);

/// The compute word of a decimal tier.
pub trait DecimalWord: WideInt {
    /// The storage word the tier narrows to.
    type Narrow: Storage<Compute = Self>;
    /// The word a product of two compute values needs.
    type Wide: WideInt;
    /// The working word of the exp, sin and cos engines.
    type Hp: HpWord;
    /// Decimal places of the compute tier.
    const DP: u8;
    /// Decimal places the storage word holds.
    const STORAGE_MAX_DP: u8;
    /// Decimal places of the high-precision working value.
    const HP_DP: u32;
    /// Bits of the high-precision word.
    const HP_BITS: u32;
    /// Bits of the compute word.
    const COMPUTE_BITS: i64;

    #[doc(hidden)]
    fn decimal_compute_zero() -> Self;
    #[doc(hidden)]
    fn decimal_compute_from_int(n: i64) -> Self;
    #[doc(hidden)]
    fn wide_scale() -> Self::Wide;
    #[doc(hidden)]
    fn pow10_compute_ct(exp: u8) -> Self;
    #[doc(hidden)]
    fn widen_to_wide(v: Self) -> Self::Wide;
    #[doc(hidden)]
    fn narrow_from_wide(w: Self::Wide) -> Result<Self, OverflowDetected>;
    #[doc(hidden)]
    fn wide_mul(a: Self, b: Self) -> Self::Wide;
    #[doc(hidden)]
    fn wide_div(a: Self::Wide, b: Self::Wide) -> Self::Wide;
    #[doc(hidden)]
    fn wide_div_scale(w: Self::Wide) -> Self::Wide;
    #[doc(hidden)]
    fn wide_halve(w: Self::Wide) -> Self::Wide;
    #[doc(hidden)]
    fn wide_is_negative(w: &Self::Wide) -> bool;
    #[doc(hidden)]
    fn wide_zero() -> Self::Wide;
    #[doc(hidden)]
    fn wide_neg(w: Self::Wide) -> Self::Wide;
    #[doc(hidden)]
    fn try_decimal_compute_neg(a: Self) -> Result<Self, OverflowDetected>;
    #[doc(hidden)]
    fn decimal_compute_is_zero(a: &Self) -> bool;
    #[doc(hidden)]
    fn decimal_compute_is_negative(a: &Self) -> bool;
    #[doc(hidden)]
    fn decimal_compute_div_int(a: Self, n: u64) -> Self;
    #[doc(hidden)]
    fn decimal_compute_unit() -> Self;
    #[doc(hidden)]
    fn binary_storage_to_compute_widen(scaled: Self::Narrow) -> Self;
    #[doc(hidden)]
    fn narrow_compute_to_storage(v: Self) -> Result<Self::Narrow, OverflowDetected>;
    #[doc(hidden)]
    fn i128_to_compute(value: i128) -> Self;
    #[doc(hidden)]
    fn try_i128_upscale_to_compute(value: i128, dp: u8) -> Result<Self, OverflowDetected>;
    #[doc(hidden)]
    fn try_i128_to_compute(value: i128) -> Result<Self, OverflowDetected>;
    #[doc(hidden)]
    fn compute_to_i128(v: Self) -> Result<i128, OverflowDetected>;
    #[doc(hidden)]
    fn compute_to_hp(x: Self) -> Self::Hp;
    #[doc(hidden)]
    fn hp_to_compute(v: &Self::Hp) -> Result<Self, OverflowDetected>;
    #[doc(hidden)]
    fn hp_div(a: Self::Hp, b: Self::Hp) -> Self::Hp;
    #[doc(hidden)]
    fn sqrt_bit_length(v: &Self) -> u32;
    #[doc(hidden)]
    fn shift_one_left(n: u32) -> Self;
    #[doc(hidden)]
    fn mul_compute_by_int(v: Self, n: u64) -> Self;
    #[doc(hidden)]
    fn tan_pi_over_16_approx() -> Self;
}

impl DecimalWord for i64 {
    type Narrow = i32;
    type Wide = i128;
    type Hp = I256;
    const DP: u8 = 9;
    const STORAGE_MAX_DP: u8 = 4;
    const HP_DP: u32 = 28;
    const HP_BITS: u32 = 256;
    const COMPUTE_BITS: i64 = 64;
    #[inline]
    fn decimal_compute_zero() -> i64 {
        { 0i64 }
    }
    #[inline]
    fn decimal_compute_from_int(n: i64) -> i64 {
        let one = decimal_compute_one::<Self>();
        { one.checked_mul(n).expect("decimal_compute_from_int: integer outside the compute range") }
    }
    #[inline]
    fn wide_scale() -> i128 {
        { pow10_i128(Self::DP as u32) }
    }
    #[inline]
    fn pow10_compute_ct(exp: u8) -> i64 {
        {
            let mut result: i64 = 1;
            for _ in 0..exp { result = result.saturating_mul(10); }
            result
        }
    }
    #[inline]
    fn widen_to_wide(v: i64) -> i128 {
        { v as i128 }
    }
    #[inline]
    fn narrow_from_wide(w: i128) -> Result<i64, OverflowDetected> {
        { i64::try_from(w).map_err(|_| OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn wide_mul(a: i64, b: i64) -> i128 {
        {
            // i64 × i64 → i128: Rust signed mul handles sign correctly
            (a as i128) * (b as i128)
        }
    }
    #[inline]
    fn wide_div(a: i128, b: i128) -> i128 {
        { a / b }
    }
    #[inline]
    fn wide_div_scale(w: i128) -> i128 {
        { Self::wide_div(w, Self::wide_scale()) }
    }
    #[inline]
    fn wide_halve(w: i128) -> i128 {
        { w >> 1 }
    }
    #[inline]
    fn wide_is_negative(w: &i128) -> bool {
        { *w < 0 }
    }
    #[inline]
    fn wide_zero() -> i128 {
        { 0i128 }
    }
    #[inline]
    fn wide_neg(w: i128) -> i128 {
        { -w }
    }
    #[inline]
    fn try_decimal_compute_neg(a: i64) -> Result<i64, OverflowDetected> {
        { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn decimal_compute_is_zero(a: &i64) -> bool {
        { *a == 0 }
    }
    #[inline]
    fn decimal_compute_is_negative(a: &i64) -> bool {
        { *a < 0 }
    }
    #[inline]
    fn decimal_compute_div_int(a: i64, n: u64) -> i64 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return a;
        }

        // Round-half-away-from-zero at compute tier (no widening needed — n is small).
        let n_compute: i64 = {
            { n as i64 }
        };

        let half_n = n_compute >> 1;
        let rounded = if Self::decimal_compute_is_negative(&a) {
            a - half_n
        } else {
            a + half_n
        };

        rounded / n_compute
    }
    #[inline]
    fn decimal_compute_unit() -> i64 {
        { 1i64 }
    }
    #[inline]
    fn binary_storage_to_compute_widen(scaled: i32) -> i64 {
        { scaled as i64 }
    }
    #[inline]
    fn narrow_compute_to_storage(v: i64) -> Result<i32, OverflowDetected> {
        {
            if v > i32::MAX as i64 || v < i32::MIN as i64 {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(v as i32)
        }
    }
    #[inline]
    fn i128_to_compute(value: i128) -> i64 {
        { i64::try_from(value).expect("i128_to_compute: value outside the realtime compute tier") }
    }
    fn try_i128_upscale_to_compute(value: i128, dp: u8) -> Result<i64, OverflowDetected> {
        if dp > Self::DP {
            let diff = (dp - Self::DP) as u32;
            // |value| < 2^127 < 10^39 / 2, so any diff >= 39 rounds to zero
            let q = if diff > 38 {
                0
            } else {
                let d = pow10_i128(diff);
                super::super::banker_round_decimal_i128(value / d, value % d, d)
            };
            return Self::try_i128_to_compute(q);
        }
        let widened = Self::try_i128_to_compute(value)?;
        if dp == Self::DP {
            return Ok(widened);
        }
        let factor = Self::pow10_compute_ct(Self::DP - dp);
        { widened.checked_mul(factor).ok_or(OverflowDetected::TierOverflow) }
        // |value| * 10^Self::DP < 2^127 * 10^38 < 2^254 (embedded),
        // < 2^383 (balanced), < 2^639 (scientific): always inside the tier
    }
    #[inline]
    fn try_i128_to_compute(value: i128) -> Result<i64, OverflowDetected> {
        { i64::try_from(value).map_err(|_| OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn compute_to_i128(v: i64) -> Result<i128, OverflowDetected> {
        { Ok(v as i128) }
    }
    fn compute_to_hp(x: i64) -> I256 {
        { I256::from_i128(x as i128) }
    }
    fn hp_to_compute(v: &I256) -> Result<i64, OverflowDetected> {
        if hp_bit_length::<Self>(v) > Self::COMPUTE_BITS as u32 - 1 {
            return Err(OverflowDetected::TierOverflow);
        }
        let w = &v.words();
        { Ok(w[0] as i64) }
    }
    fn hp_div(a: I256, b: I256) -> I256 {
        { a / b }
    }
    fn sqrt_bit_length(v: &i64) -> u32 {
        {
            if *v <= 0 { 0 } else { 64 - v.leading_zeros() }
        }
    }
    fn shift_one_left(n: u32) -> i64 {
        {
            if n >= 63 { i64::MAX } else { 1i64 << n }
        }
    }
    fn mul_compute_by_int(v: i64, n: u64) -> i64 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return v;
        }

        let n_compute: i64 = {
            { n as i64 }
        };

        v * n_compute
    }
    fn tan_pi_over_16_approx() -> i64 {
        // 0.2 at compute dp = 2 × 10^(dp-1)
        if Self::DP >= 1 {
            let two = {
                { 2i64 }
            };
            Self::pow10_compute_ct(Self::DP - 1) * two
        } else {
            decimal_compute_one::<Self>()
        }
    }
}

impl DecimalWord for i128 {
    type Narrow = i64;
    type Wide = I256;
    type Hp = I256;
    const DP: u8 = 19;
    const STORAGE_MAX_DP: u8 = 9;
    const HP_DP: u32 = 38;
    const HP_BITS: u32 = 256;
    const COMPUTE_BITS: i64 = 128;
    #[inline]
    fn decimal_compute_zero() -> i128 {
        { 0i128 }
    }
    #[inline]
    fn decimal_compute_from_int(n: i64) -> i128 {
        let one = decimal_compute_one::<Self>();
        { one * (n as i128) }
    }
    #[inline]
    fn wide_scale() -> I256 {
        { I256::from_i128(pow10_i128(Self::DP as u32)) }
    }
    #[inline]
    fn pow10_compute_ct(exp: u8) -> i128 {
        {
            let mut result: i128 = 1;
            for _ in 0..exp { result *= 10; }
            result
        }
    }
    #[inline]
    fn widen_to_wide(v: i128) -> I256 {
        { I256::from_i128(v) }
    }
    #[inline]
    fn narrow_from_wide(w: I256) -> Result<i128, OverflowDetected> {
        { if w.fits_in_i128() { Ok(w.as_i128()) } else { Err(OverflowDetected::TierOverflow) } }
    }
    #[inline]
    fn wide_mul(a: i128, b: i128) -> I256 {
        {
            // i128 × i128: explicit sign handling
            let a_neg = a < 0;
            let b_neg = b < 0;
            let result_neg = a_neg ^ b_neg;
            // negate after widening: i128::MIN has no i128 negation (it panicked)
            let abs_a = if a_neg { I256::zero() - I256::from_i128(a) } else { I256::from_i128(a) };
            let abs_b = if b_neg { I256::zero() - I256::from_i128(b) } else { I256::from_i128(b) };
            // Use I256 unsigned-word multiply: both operands non-negative now
            let abs_product = abs_a * abs_b;
            if result_neg { I256::zero() - abs_product } else { abs_product }
        }
    }
    #[inline]
    fn wide_div(a: I256, b: I256) -> I256 {
        { a / b }
    }
    #[inline]
    fn wide_div_scale(w: I256) -> I256 {
        { Self::wide_div(w, Self::wide_scale()) }
    }
    #[inline]
    fn wide_halve(w: I256) -> I256 {
        { w >> 1 }
    }
    #[inline]
    fn wide_is_negative(w: &I256) -> bool {
        { *w < I256::zero() }
    }
    #[inline]
    fn wide_zero() -> I256 {
        { I256::zero() }
    }
    #[inline]
    fn wide_neg(w: I256) -> I256 {
        { I256::zero() - w }
    }
    #[inline]
    fn try_decimal_compute_neg(a: i128) -> Result<i128, OverflowDetected> {
        { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn decimal_compute_is_zero(a: &i128) -> bool {
        { *a == 0 }
    }
    #[inline]
    fn decimal_compute_is_negative(a: &i128) -> bool {
        { *a < 0 }
    }
    #[inline]
    fn decimal_compute_div_int(a: i128, n: u64) -> i128 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return a;
        }

        // Round-half-away-from-zero at compute tier (no widening needed — n is small).
        let n_compute: i128 = {
            { n as i128 }
        };

        let half_n = n_compute >> 1;
        let rounded = if Self::decimal_compute_is_negative(&a) {
            a - half_n
        } else {
            a + half_n
        };

        rounded / n_compute
    }
    #[inline]
    fn decimal_compute_unit() -> i128 {
        { 1i128 }
    }
    #[inline]
    fn binary_storage_to_compute_widen(scaled: i64) -> i128 {
        { scaled as i128 }
    }
    #[inline]
    fn narrow_compute_to_storage(v: i128) -> Result<i64, OverflowDetected> {
        {
            if v > i64::MAX as i128 || v < i64::MIN as i128 {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(v as i64)
        }
    }
    #[inline]
    fn i128_to_compute(value: i128) -> i128 {
        { value }
    }
    fn try_i128_upscale_to_compute(value: i128, dp: u8) -> Result<i128, OverflowDetected> {
        if dp > Self::DP {
            let diff = (dp - Self::DP) as u32;
            // |value| < 2^127 < 10^39 / 2, so any diff >= 39 rounds to zero
            let q = if diff > 38 {
                0
            } else {
                let d = pow10_i128(diff);
                super::super::banker_round_decimal_i128(value / d, value % d, d)
            };
            return Self::try_i128_to_compute(q);
        }
        let widened = Self::try_i128_to_compute(value)?;
        if dp == Self::DP {
            return Ok(widened);
        }
        let factor = Self::pow10_compute_ct(Self::DP - dp);
        { widened.checked_mul(factor).ok_or(OverflowDetected::TierOverflow) }
        // |value| * 10^Self::DP < 2^127 * 10^38 < 2^254 (embedded),
        // < 2^383 (balanced), < 2^639 (scientific): always inside the tier
    }
    #[inline]
    fn try_i128_to_compute(value: i128) -> Result<i128, OverflowDetected> {
        { Ok(Self::i128_to_compute(value)) }
    }
    #[inline]
    fn compute_to_i128(v: i128) -> Result<i128, OverflowDetected> {
        { Ok(v) }
    }
    fn compute_to_hp(x: i128) -> I256 {
        { I256::from_i128(x) }
    }
    fn hp_to_compute(v: &I256) -> Result<i128, OverflowDetected> {
        if hp_bit_length::<Self>(v) > Self::COMPUTE_BITS as u32 - 1 {
            return Err(OverflowDetected::TierOverflow);
        }
        let w = &v.words();
        { Ok((w[0] as i128) | ((w[1] as i128) << 64)) }
    }
    fn hp_div(a: I256, b: I256) -> I256 {
        { a / b }
    }
    fn sqrt_bit_length(v: &i128) -> u32 {
        {
            if *v <= 0 { 0 } else { 128 - v.leading_zeros() }
        }
    }
    fn shift_one_left(n: u32) -> i128 {
        {
            if n >= 127 { i128::MAX } else { 1i128 << n }
        }
    }
    fn mul_compute_by_int(v: i128, n: u64) -> i128 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return v;
        }

        let n_compute: i128 = {
            { n as i128 }
        };

        v * n_compute
    }
    fn tan_pi_over_16_approx() -> i128 {
        // 0.2 at compute dp = 2 × 10^(dp-1)
        if Self::DP >= 1 {
            let two = {
                { 2i128 }
            };
            Self::pow10_compute_ct(Self::DP - 1) * two
        } else {
            decimal_compute_one::<Self>()
        }
    }
}

impl DecimalWord for I256 {
    type Narrow = i128;
    type Wide = I512;
    type Hp = I512;
    const DP: u8 = 38;
    const STORAGE_MAX_DP: u8 = 19;
    const HP_DP: u32 = 60;
    const HP_BITS: u32 = 512;
    const COMPUTE_BITS: i64 = 256;
    #[inline]
    fn decimal_compute_zero() -> I256 {
        { I256::zero() }
    }
    #[inline]
    fn decimal_compute_from_int(n: i64) -> I256 {
        let one = decimal_compute_one::<Self>();
        { one * I256::from_i128(n as i128) }
    }
    #[inline]
    fn wide_scale() -> I512 {
        { I512::from_i256(Self::pow10_compute_ct(Self::DP)) }
    }
    #[inline]
    fn pow10_compute_ct(exp: u8) -> I256 {
        {
            let mut result = I256::from_i128(1);
            let ten = I256::from_i128(10);
            for _ in 0..exp { result = result * ten; }
            result
        }
    }
    #[inline]
    fn widen_to_wide(v: I256) -> I512 {
        { I512::from_i256(v) }
    }
    #[inline]
    fn narrow_from_wide(w: I512) -> Result<I256, OverflowDetected> {
        { if w.fits_in_i256() { Ok(w.as_i256()) } else { Err(OverflowDetected::TierOverflow) } }
    }
    #[inline]
    fn wide_mul(a: I256, b: I256) -> I512 {
        {
            let a_neg = a.is_negative();
            let b_neg = b.is_negative();
            let result_neg = a_neg ^ b_neg;
            let abs_a = if a_neg { I256::zero() - a } else { a };
            let abs_b = if b_neg { I256::zero() - b } else { b };
            let abs_product = abs_a.mul_to_i512(abs_b);
            if result_neg { I512::zero() - abs_product } else { abs_product }
        }
    }
    #[inline]
    fn wide_div(a: I512, b: I512) -> I512 {
        { a / b }
    }
    #[inline]
    fn wide_div_scale(w: I512) -> I512 {
        { Self::wide_div(w, Self::wide_scale()) }
    }
    #[inline]
    fn wide_halve(w: I512) -> I512 {
        { w >> 1 }
    }
    #[inline]
    fn wide_is_negative(w: &I512) -> bool {
        { *w < I512::zero() }
    }
    #[inline]
    fn wide_zero() -> I512 {
        { I512::zero() }
    }
    #[inline]
    fn wide_neg(w: I512) -> I512 {
        { I512::zero() - w }
    }
    #[inline]
    fn try_decimal_compute_neg(a: I256) -> Result<I256, OverflowDetected> {
        { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn decimal_compute_is_zero(a: &I256) -> bool {
        { a.is_zero() }
    }
    #[inline]
    fn decimal_compute_is_negative(a: &I256) -> bool {
        { a.is_negative() }
    }
    #[inline]
    fn decimal_compute_div_int(a: I256, n: u64) -> I256 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return a;
        }

        // Round-half-away-from-zero at compute tier (no widening needed — n is small).
        let n_compute: I256 = {
            { I256::from_i128(n as i128) }
        };

        let half_n = n_compute >> 1;
        let rounded = if Self::decimal_compute_is_negative(&a) {
            a - half_n
        } else {
            a + half_n
        };

        rounded / n_compute
    }
    #[inline]
    fn decimal_compute_unit() -> I256 {
        { I256::from_i128(1) }
    }
    #[inline]
    fn binary_storage_to_compute_widen(scaled: i128) -> I256 {
        { I256::from_i128(scaled) }
    }
    #[inline]
    fn narrow_compute_to_storage(v: I256) -> Result<i128, OverflowDetected> {
        {
            if !v.fits_in_i128() {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(v.as_i128())
        }
    }
    #[inline]
    fn i128_to_compute(value: i128) -> I256 {
        { I256::from_i128(value) }
    }
    fn try_i128_upscale_to_compute(value: i128, dp: u8) -> Result<I256, OverflowDetected> {
        if dp > Self::DP {
            let diff = (dp - Self::DP) as u32;
            // |value| < 2^127 < 10^39 / 2, so any diff >= 39 rounds to zero
            let q = if diff > 38 {
                0
            } else {
                let d = pow10_i128(diff);
                super::super::banker_round_decimal_i128(value / d, value % d, d)
            };
            return Self::try_i128_to_compute(q);
        }
        let widened = Self::try_i128_to_compute(value)?;
        if dp == Self::DP {
            return Ok(widened);
        }
        let factor = Self::pow10_compute_ct(Self::DP - dp);
        // |value| * 10^Self::DP < 2^127 * 10^38 < 2^254 (embedded),
        // < 2^383 (balanced), < 2^639 (scientific): always inside the tier
        { Ok(widened * factor) }
    }
    #[inline]
    fn try_i128_to_compute(value: i128) -> Result<I256, OverflowDetected> {
        { Ok(Self::i128_to_compute(value)) }
    }
    #[inline]
    fn compute_to_i128(v: I256) -> Result<i128, OverflowDetected> {
        {
            if !v.fits_in_i128() {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(v.as_i128())
        }
    }
    fn compute_to_hp(x: I256) -> I512 {
        { I512::from_i256(x) }
    }
    fn hp_to_compute(v: &I512) -> Result<I256, OverflowDetected> {
        if hp_bit_length::<Self>(v) > Self::COMPUTE_BITS as u32 - 1 {
            return Err(OverflowDetected::TierOverflow);
        }
        let w = &v.words();
        { Ok(crate::fixed_point::i256::I256::from_words([w[0], w[1], w[2], w[3]])) }
    }
    fn hp_div(a: I512, b: I512) -> I512 {
        { a / b }
    }
    fn sqrt_bit_length(v: &I256) -> u32 {
        {
            // I256 — count bits via word inspection (highest non-zero word)
            for i in (0..4).rev() {
                if v.words[i] != 0 {
                    return (i as u32) * 64 + (64 - v.words[i].leading_zeros());
                }
            }
            0
        }
    }
    fn shift_one_left(n: u32) -> I256 {
        {
            use crate::fixed_point::i256::I256;
            if n >= 254 { I256::from_i128(1) << 254usize } else { I256::from_i128(1) << (n as usize) }
        }
    }
    fn mul_compute_by_int(v: I256, n: u64) -> I256 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return v;
        }

        let n_compute: I256 = {
            { crate::fixed_point::i256::I256::from_i128(n as i128) }
        };

        v * n_compute
    }
    fn tan_pi_over_16_approx() -> I256 {
        // 0.2 at compute dp = 2 × 10^(dp-1)
        if Self::DP >= 1 {
            let two = {
                { crate::fixed_point::i256::I256::from_i128(2) }
            };
            Self::pow10_compute_ct(Self::DP - 1) * two
        } else {
            decimal_compute_one::<Self>()
        }
    }
}

impl DecimalWord for I512 {
    type Narrow = I256;
    type Wide = I1024;
    type Hp = I1024;
    const DP: u8 = 77;
    const STORAGE_MAX_DP: u8 = 38;
    const HP_DP: u32 = 90;
    const HP_BITS: u32 = 1024;
    const COMPUTE_BITS: i64 = 512;
    #[inline]
    fn decimal_compute_zero() -> I512 {
        { I512::zero() }
    }
    #[inline]
    fn decimal_compute_from_int(n: i64) -> I512 {
        let one = decimal_compute_one::<Self>();
        { one * I512::from_i128(n as i128) }
    }
    #[inline]
    fn wide_scale() -> I1024 {
        { I1024::from_i512(Self::pow10_compute_ct(Self::DP)) }
    }
    #[inline]
    fn pow10_compute_ct(exp: u8) -> I512 {
        {
            let mut result = I512::from_i128(1);
            let ten = I512::from_i128(10);
            for _ in 0..exp { result = result * ten; }
            result
        }
    }
    #[inline]
    fn widen_to_wide(v: I512) -> I1024 {
        { I1024::from_i512(v) }
    }
    #[inline]
    fn narrow_from_wide(w: I1024) -> Result<I512, OverflowDetected> {
        { if w.fits_in_i512() { Ok(w.as_i512()) } else { Err(OverflowDetected::TierOverflow) } }
    }
    #[inline]
    fn wide_mul(a: I512, b: I512) -> I1024 {
        {
            let a_neg = a.is_negative();
            let b_neg = b.is_negative();
            let result_neg = a_neg ^ b_neg;
            let abs_a = if a_neg { I512::zero() - a } else { a };
            let abs_b = if b_neg { I512::zero() - b } else { b };
            let abs_product = abs_a.mul_to_i1024(abs_b);
            if result_neg { I1024::zero() - abs_product } else { abs_product }
        }
    }
    #[inline]
    fn wide_div(a: I1024, b: I1024) -> I1024 {
        { a / b }
    }
    #[inline]
    fn wide_div_scale(w: I1024) -> I1024 {
        {
            let negative = Self::wide_is_negative(&w);
            let mut m = if negative { -w } else { w };
            let mut left = Self::DP as u32;
            while left > 0 {
                let k = left.min(19);
                let d = 10u128.pow(k);
                let mut rem: u128 = 0;
                for word in m.words.iter_mut().rev() {
                    let cur = (rem << 64) | *word as u128;
                    *word = (cur / d) as u64;
                    rem = cur % d;
                }
                left -= k;
            }
            if negative { -m } else { m }
        }
    }
    #[inline]
    fn wide_halve(w: I1024) -> I1024 {
        { w >> 1 }
    }
    #[inline]
    fn wide_is_negative(w: &I1024) -> bool {
        { *w < I1024::zero() }
    }
    #[inline]
    fn wide_zero() -> I1024 {
        { I1024::zero() }
    }
    #[inline]
    fn wide_neg(w: I1024) -> I1024 {
        { I1024::zero() - w }
    }
    #[inline]
    fn try_decimal_compute_neg(a: I512) -> Result<I512, OverflowDetected> {
        { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn decimal_compute_is_zero(a: &I512) -> bool {
        { a.is_zero() }
    }
    #[inline]
    fn decimal_compute_is_negative(a: &I512) -> bool {
        { a.is_negative() }
    }
    #[inline]
    fn decimal_compute_div_int(a: I512, n: u64) -> I512 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return a;
        }

        // Round-half-away-from-zero at compute tier (no widening needed — n is small).
        let n_compute: I512 = {
            { I512::from_i128(n as i128) }
        };

        let half_n = n_compute >> 1;
        let rounded = if Self::decimal_compute_is_negative(&a) {
            a - half_n
        } else {
            a + half_n
        };

        rounded / n_compute
    }
    #[inline]
    fn decimal_compute_unit() -> I512 {
        { I512::from_i128(1) }
    }
    #[inline]
    fn binary_storage_to_compute_widen(scaled: I256) -> I512 {
        { I512::from_i256(scaled) }
    }
    #[inline]
    fn narrow_compute_to_storage(v: I512) -> Result<I256, OverflowDetected> {
        {
            if !v.fits_in_i256() {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(v.as_i256())
        }
    }
    #[inline]
    fn i128_to_compute(value: i128) -> I512 {
        { I512::from_i128(value) }
    }
    fn try_i128_upscale_to_compute(value: i128, dp: u8) -> Result<I512, OverflowDetected> {
        if dp > Self::DP {
            let diff = (dp - Self::DP) as u32;
            // |value| < 2^127 < 10^39 / 2, so any diff >= 39 rounds to zero
            let q = if diff > 38 {
                0
            } else {
                let d = pow10_i128(diff);
                super::super::banker_round_decimal_i128(value / d, value % d, d)
            };
            return Self::try_i128_to_compute(q);
        }
        let widened = Self::try_i128_to_compute(value)?;
        if dp == Self::DP {
            return Ok(widened);
        }
        let factor = Self::pow10_compute_ct(Self::DP - dp);
        // |value| * 10^Self::DP < 2^127 * 10^38 < 2^254 (embedded),
        // < 2^383 (balanced), < 2^639 (scientific): always inside the tier
        { Ok(widened * factor) }
    }
    #[inline]
    fn try_i128_to_compute(value: i128) -> Result<I512, OverflowDetected> {
        { Ok(Self::i128_to_compute(value)) }
    }
    #[inline]
    fn compute_to_i128(v: I512) -> Result<i128, OverflowDetected> {
        {
            if !v.fits_in_i128() {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(v.as_i128())
        }
    }
    fn compute_to_hp(x: I512) -> I1024 {
        { I1024::from_i512(x) }
    }
    fn hp_to_compute(v: &I1024) -> Result<I512, OverflowDetected> {
        if hp_bit_length::<Self>(v) > Self::COMPUTE_BITS as u32 - 1 {
            return Err(OverflowDetected::TierOverflow);
        }
        let w = &v.words();
        {
            let mut words = [0u64; 8];
            words.copy_from_slice(&w[..8]);
            Ok(crate::fixed_point::i512::I512::from_words(words))
        }
    }
    fn hp_div(a: I1024, b: I1024) -> I1024 {
        { a / b }
    }
    fn sqrt_bit_length(v: &I512) -> u32 {
        {
            for i in (0..8).rev() {
                if v.words[i] != 0 {
                    return (i as u32) * 64 + (64 - v.words[i].leading_zeros());
                }
            }
            0
        }
    }
    fn shift_one_left(n: u32) -> I512 {
        {
            use crate::fixed_point::i512::I512;
            if n >= 510 { I512::from_i128(1) << 510usize } else { I512::from_i128(1) << (n as usize) }
        }
    }
    fn mul_compute_by_int(v: I512, n: u64) -> I512 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return v;
        }

        let n_compute: I512 = {
            { crate::fixed_point::i512::I512::from_i128(n as i128) }
        };

        v * n_compute
    }
    fn tan_pi_over_16_approx() -> I512 {
        // 0.2 at compute dp = 2 × 10^(dp-1)
        if Self::DP >= 1 {
            let two = {
                { crate::fixed_point::i512::I512::from_i128(2) }
            };
            Self::pow10_compute_ct(Self::DP - 1) * two
        } else {
            decimal_compute_one::<Self>()
        }
    }
}

impl DecimalWord for I1024 {
    type Narrow = I512;
    type Wide = I2048;
    type Hp = I2048;
    const DP: u8 = 154;
    const STORAGE_MAX_DP: u8 = 77;
    const HP_DP: u32 = 170;
    const HP_BITS: u32 = 2048;
    const COMPUTE_BITS: i64 = 1024;
    #[inline]
    fn decimal_compute_zero() -> I1024 {
        { I1024::zero() }
    }
    #[inline]
    fn decimal_compute_from_int(n: i64) -> I1024 {
        let one = decimal_compute_one::<Self>();
        { one * I1024::from_i128(n as i128) }
    }
    #[inline]
    fn wide_scale() -> I2048 {
        { I2048::from_i1024(Self::pow10_compute_ct(Self::DP)) }
    }
    #[inline]
    fn pow10_compute_ct(exp: u8) -> I1024 {
        {
            let mut result = I1024::from_i128(1);
            let ten = I1024::from_i128(10);
            for _ in 0..exp { result = result * ten; }
            result
        }
    }
    #[inline]
    fn widen_to_wide(v: I1024) -> I2048 {
        { I2048::from_i1024(v) }
    }
    #[inline]
    fn narrow_from_wide(w: I2048) -> Result<I1024, OverflowDetected> {
        { if w.fits_in_i1024() { Ok(w.as_i1024()) } else { Err(OverflowDetected::TierOverflow) } }
    }
    #[inline]
    fn wide_mul(a: I1024, b: I1024) -> I2048 {
        {
            let a_neg = (a.words[15] as i64) < 0;
            let b_neg = (b.words[15] as i64) < 0;
            let result_neg = a_neg ^ b_neg;
            let abs_a = if a_neg { I1024::zero() - a } else { a };
            let abs_b = if b_neg { I1024::zero() - b } else { b };
            let abs_product = abs_a.mul_to_i2048(abs_b);
            if result_neg { I2048::zero() - abs_product } else { abs_product }
        }
    }
    #[inline]
    fn wide_div(a: I2048, b: I2048) -> I2048 {
        { i2048_div(a, b) }
    }
    #[inline]
    fn wide_div_scale(w: I2048) -> I2048 {
        {
            let negative = Self::wide_is_negative(&w);
            let mut m = if negative { -w } else { w };
            let mut left = Self::DP as u32;
            while left > 0 {
                let k = left.min(19);
                let d = 10u128.pow(k);
                let mut rem: u128 = 0;
                for word in m.words.iter_mut().rev() {
                    let cur = (rem << 64) | *word as u128;
                    *word = (cur / d) as u64;
                    rem = cur % d;
                }
                left -= k;
            }
            if negative { -m } else { m }
        }
    }
    #[inline]
    fn wide_halve(w: I2048) -> I2048 {
        { w >> 1 }
    }
    #[inline]
    fn wide_is_negative(w: &I2048) -> bool {
        {
            // I2048 sign bit is the MSB of word 31
            (w.words[31] & 0x8000_0000_0000_0000) != 0
        }
    }
    #[inline]
    fn wide_zero() -> I2048 {
        { I2048::zero() }
    }
    #[inline]
    fn wide_neg(w: I2048) -> I2048 {
        { I2048::zero() - w }
    }
    #[inline]
    fn try_decimal_compute_neg(a: I1024) -> Result<I1024, OverflowDetected> {
        { if a == I1024::min_value() { Err(OverflowDetected::TierOverflow) } else { Ok(I1024::zero() - a) } }
    }
    #[inline]
    fn decimal_compute_is_zero(a: &I1024) -> bool {
        { a.is_zero() }
    }
    #[inline]
    fn decimal_compute_is_negative(a: &I1024) -> bool {
        { (a.words[15] as i64) < 0 }
    }
    #[inline]
    fn decimal_compute_div_int(a: I1024, n: u64) -> I1024 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return a;
        }

        // Round-half-away-from-zero at compute tier (no widening needed — n is small).
        let n_compute: I1024 = {
            { I1024::from_i128(n as i128) }
        };

        let half_n = n_compute >> 1;
        let rounded = if Self::decimal_compute_is_negative(&a) {
            a - half_n
        } else {
            a + half_n
        };

        rounded / n_compute
    }
    #[inline]
    fn decimal_compute_unit() -> I1024 {
        { I1024::from_i128(1) }
    }
    #[inline]
    fn binary_storage_to_compute_widen(scaled: I512) -> I1024 {
        { I1024::from_i512(scaled) }
    }
    #[inline]
    fn narrow_compute_to_storage(v: I1024) -> Result<I512, OverflowDetected> {
        {
            if !v.fits_in_i512() {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(v.as_i512())
        }
    }
    #[inline]
    fn i128_to_compute(value: i128) -> I1024 {
        { I1024::from_i128(value) }
    }
    fn try_i128_upscale_to_compute(value: i128, dp: u8) -> Result<I1024, OverflowDetected> {
        if dp > Self::DP {
            let diff = (dp - Self::DP) as u32;
            // |value| < 2^127 < 10^39 / 2, so any diff >= 39 rounds to zero
            let q = if diff > 38 {
                0
            } else {
                let d = pow10_i128(diff);
                super::super::banker_round_decimal_i128(value / d, value % d, d)
            };
            return Self::try_i128_to_compute(q);
        }
        let widened = Self::try_i128_to_compute(value)?;
        if dp == Self::DP {
            return Ok(widened);
        }
        let factor = Self::pow10_compute_ct(Self::DP - dp);
        // |value| * 10^Self::DP < 2^127 * 10^38 < 2^254 (embedded),
        // < 2^383 (balanced), < 2^639 (scientific): always inside the tier
        { Ok(widened * factor) }
    }
    #[inline]
    fn try_i128_to_compute(value: i128) -> Result<I1024, OverflowDetected> {
        { Ok(Self::i128_to_compute(value)) }
    }
    #[inline]
    fn compute_to_i128(v: I1024) -> Result<i128, OverflowDetected> {
        {
            if !v.fits_in_i128() {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(v.as_i128())
        }
    }
    fn compute_to_hp(x: I1024) -> I2048 {
        { I2048::from_i1024(x) }
    }
    fn hp_to_compute(v: &I2048) -> Result<I1024, OverflowDetected> {
        if hp_bit_length::<Self>(v) > Self::COMPUTE_BITS as u32 - 1 {
            return Err(OverflowDetected::TierOverflow);
        }
        let w = &v.words();
        {
            let mut words = [0u64; 16];
            words.copy_from_slice(&w[..16]);
            Ok(crate::fixed_point::I1024::from_words(words))
        }
    }
    fn hp_div(a: I2048, b: I2048) -> I2048 {
        { crate::fixed_point::domains::binary_fixed::i2048::i2048_div(a, b) }
    }
    fn sqrt_bit_length(v: &I1024) -> u32 {
        {
            for i in (0..16).rev() {
                if v.words[i] != 0 {
                    return (i as u32) * 64 + (64 - v.words[i].leading_zeros());
                }
            }
            0
        }
    }
    fn shift_one_left(n: u32) -> I1024 {
        {
            use crate::fixed_point::I1024;
            if n >= 1020 { I1024::from_i128(1) << 1020usize } else { I1024::from_i128(1) << (n as usize) }
        }
    }
    fn mul_compute_by_int(v: I1024, n: u64) -> I1024 {
        if n == 0 {
            return Self::decimal_compute_zero();
        }
        if n == 1 {
            return v;
        }

        let n_compute: I1024 = {
            { crate::fixed_point::I1024::from_i128(n as i128) }
        };

        v * n_compute
    }
    fn tan_pi_over_16_approx() -> I1024 {
        // 0.2 at compute dp = 2 × 10^(dp-1)
        if Self::DP >= 1 {
            let two = {
                { crate::fixed_point::I1024::from_i128(2) }
            };
            Self::pow10_compute_ct(Self::DP - 1) * two
        } else {
            decimal_compute_one::<Self>()
        }
    }
}
