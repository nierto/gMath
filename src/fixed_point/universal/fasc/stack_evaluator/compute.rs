//! Compute-tier arithmetic and transcendental dispatchers
//!
//! All functions operate at tier N+1 (ComputeStorage) for maximum precision.

#[allow(unused_imports)]
use super::{BinaryStorage, ComputeStorage};

// Build-time Q-format constants (FRAC_BITS, COMPUTE_FRAC_BITS, etc.)
#[cfg(table_format = "q16_16")]
use crate::fixed_point::frac_config;
#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;
#[cfg(table_format = "q256_256")]
use crate::fixed_point::I2048;
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

// Compute-tier sqrt for decimal domain
#[allow(unused_imports)]
#[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256", table_format = "q32_32", table_format = "q16_16"))]
use crate::fixed_point::domains::binary_fixed::transcendental::{
    sqrt_binary_i256, sqrt_binary_i512,
};

#[allow(unused_imports)]
#[cfg(table_format = "q256_256")]
use crate::fixed_point::domains::binary_fixed::transcendental::ln_binary_i1024;

#[cfg(table_format = "q256_256")]
use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::{
    sin_compute_tier_i1024, cos_compute_tier_i1024, sincos_compute_tier_i1024, pi_half_i1024,
};
#[cfg(table_format = "q128_128")]
use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::{
    sin_compute_tier_i512, cos_compute_tier_i512, sincos_compute_tier_i512,
};
// i256 compute-tier transcendentals (used by q64_64 only — q32_32/q16_16 now use native tier)
#[cfg(table_format = "q64_64")]
use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::{
    sin_compute_tier_i256, cos_compute_tier_i256, sincos_compute_tier_i256,
};
// i64 compute-tier transcendentals (used by q16_16 dispatch)
#[cfg(table_format = "q16_16")]
use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::{
    sin_compute_tier_i64, cos_compute_tier_i64, sincos_compute_tier_i64,
};

#[cfg(table_format = "q256_256")]
use crate::fixed_point::domains::binary_fixed::transcendental::atan_tier_n_plus_1::{
    atan_compute_tier_i1024, atan2_compute_tier_i1024,
};
#[cfg(table_format = "q128_128")]
use crate::fixed_point::domains::binary_fixed::transcendental::atan_tier_n_plus_1::{
    atan_compute_tier_i512, atan2_compute_tier_i512,
};
#[cfg(table_format = "q64_64")]
use crate::fixed_point::domains::binary_fixed::transcendental::atan_tier_n_plus_1::{
    atan_compute_tier_i256, atan2_compute_tier_i256,
};
#[cfg(table_format = "q16_16")]
use crate::fixed_point::domains::binary_fixed::transcendental::atan_tier_n_plus_1::{
    atan_compute_tier_i64, atan2_compute_tier_i64,
};

#[cfg(table_format = "q256_256")]
use crate::fixed_point::domains::binary_fixed::transcendental::sqrt_tier_n_plus_1::sqrt_binary_i1024;
#[cfg(table_format = "q16_16")]
use crate::fixed_point::domains::binary_fixed::transcendental::sqrt_tier_n_plus_1::sqrt_binary_i64;

// q32_32: ComputeStorage = i128 (Q64.64) — native Q64.64 transcendentals
#[cfg(table_format = "q32_32")]
use crate::fixed_point::domains::binary_fixed::transcendental::{
    sin_binary_i128, cos_binary_i128, sqrt_binary_i128,
    atan_binary_i128, atan2_binary_i128,
};
#[cfg(table_format = "q32_32")]
use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::pi_half_i128;

pub(crate) fn upscale_to_compute(val: BinaryStorage) -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    {
        // I512 → I1024, shift left 256
        I1024::from_i512(val) << 256
    }
    #[cfg(table_format = "q128_128")]
    {
        // I256 → I512, shift left 128
        I512::from_i256(val) << 128
    }
    #[cfg(table_format = "q64_64")]
    {
        // i128 → I256, shift left 64
        I256::from_i128(val) << 64
    }
    #[cfg(table_format = "q32_32")]
    {
        // i64 → i128, shift left 32 (Q32.32 → Q64.64)
        (val as i128) << 32
    }
    #[cfg(table_format = "q16_16")]
    {
        // i32 → i64, shift left by FRAC_BITS (storage → compute tier)
        (val as i64) << frac_config::FRAC_BITS
    }
}

/// Downscale from compute tier to storage tier with round-to-nearest.
///
/// Shifts right by (COMPUTE_FRAC_BITS - STORAGE_FRAC_BITS) with rounding.
///
/// Returns `Err(TierOverflow)` if the compute-tier value does not fit in
/// the storage tier — this is the UGOD overflow detection point that
/// prevents silent truncation of large results (e.g., exp(44) in Q64.64).
#[inline]
pub(crate) fn downscale_to_storage(val: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    #[cfg(table_format = "q256_256")]
    {
        // I1024 → I512, shift right 256 with rounding
        let round_bit = (val & (I1024::from_i128(1) << 255)) != I1024::zero();
        let shifted = val >> 256;
        if !shifted.fits_in_i512() {
            return Err(OverflowDetected::TierOverflow);
        }
        let mut result = shifted.as_i512();
        if round_bit {
            result = result + I512::from_i128(1);
        }
        Ok(result)
    }
    #[cfg(table_format = "q128_128")]
    {
        // I512 → I256, shift right 128 with rounding
        let round_bit = (val & (I512::from_i128(1) << 127)) != I512::zero();
        let shifted = val >> 128;
        if !shifted.fits_in_i256() {
            return Err(OverflowDetected::TierOverflow);
        }
        let mut result = shifted.as_i256();
        if round_bit {
            result = result + I256::from_i128(1);
        }
        Ok(result)
    }
    #[cfg(table_format = "q64_64")]
    {
        // I256 → i128, shift right 64 with rounding
        let round_bit = (val & (I256::from_i128(1) << 63)) != I256::zero();
        let shifted = val >> 64;
        if !shifted.fits_in_i128() {
            return Err(OverflowDetected::TierOverflow);
        }
        let mut result = shifted.as_i128();
        if round_bit {
            result += 1;
        }
        Ok(result)
    }
    #[cfg(table_format = "q32_32")]
    {
        // i128 → i64, shift right 32 with rounding (Q64.64 → Q32.32)
        let round_bit = (val & (1i128 << 31)) != 0;
        let shifted = val >> 32;
        if shifted > i64::MAX as i128 || shifted < i64::MIN as i128 {
            return Err(OverflowDetected::TierOverflow);
        }
        let mut result = shifted as i64;
        if round_bit {
            result += 1;
        }
        Ok(result)
    }
    #[cfg(table_format = "q16_16")]
    {
        // i64 → i32, shift right by FRAC_BITS with rounding
        let round_bit = (val & (1i64 << frac_config::FRAC_ROUND_BIT)) != 0;
        let shifted = val >> frac_config::FRAC_BITS;
        if shifted > i32::MAX as i64 || shifted < i32::MIN as i64 {
            return Err(OverflowDetected::TierOverflow);
        }
        let mut result = shifted as i32;
        if round_bit {
            result += 1;
        }
        Ok(result)
    }
}

/// Convert a Decimal StackValue directly to ComputeStorage at full compute-tier precision.
///
/// This avoids the precision loss of going through BinaryStorage first.
/// For Q64.64 profile: converts to Q128.128 (I256) instead of Q64.64→upscale
/// For Q128.128 profile: converts to Q256.256 (I512) instead of Q128.128→upscale
/// For Q256.256 profile: converts to Q512.512 (I1024) instead of Q256.256→upscale
#[inline]
pub(super) fn decimal_to_compute_storage(decimals: u8, scaled: BinaryStorage) -> Result<ComputeStorage, OverflowDetected> {
    if decimals == 0 {
        // Integer value — shift directly into compute Q-format
        #[cfg(table_format = "q256_256")]
        { return Ok(I1024::from_i512(scaled) << 512); }
        #[cfg(table_format = "q128_128")]
        { return Ok(I512::from_i256(scaled) << 256); }
        #[cfg(table_format = "q64_64")]
        { return Ok(I256::from_i128(scaled) << 128); }
        #[cfg(table_format = "q32_32")]
        { return Ok((scaled as i128) << 64); }
        #[cfg(table_format = "q16_16")]
        { return Ok((scaled as i64) << frac_config::COMPUTE_FRAC_BITS); }
    }

    let den = pow10_compute(decimals)?;

    #[cfg(table_format = "q256_256")]
    {
        let num = I1024::from_i512(scaled) << 512;
        Ok(num / den)
    }
    #[cfg(table_format = "q128_128")]
    {
        let num = I512::from_i256(scaled) << 256;
        Ok(num / den)
    }
    #[cfg(table_format = "q64_64")]
    {
        let num = I256::from_i128(scaled) << 128;
        Ok(num / den)
    }
    #[cfg(table_format = "q32_32")]
    {
        // BinaryStorage=i64 → promote to i128, shift by 64 (Q64.64 compute format)
        let num = (scaled as i128) << 64;
        Ok(num / den)
    }
    #[cfg(table_format = "q16_16")]
    {
        // BinaryStorage=i32 → promote to i64, shift by COMPUTE_FRAC_BITS (compute format)
        let num = (scaled as i64) << frac_config::COMPUTE_FRAC_BITS;
        Ok(num / den)
    }
}

/// Compute 10^exp at ComputeStorage precision — O(1) via const lookup table.
///
/// Tables are built at compile time via const fn word-level multiplication.
/// Covers all exponents up to the profile's DP promotion threshold.
/// Returns `Err(Overflow)` if exponent exceeds the type's safe representable range.
#[inline]
pub(super) fn pow10_compute(exp: u8) -> Result<ComputeStorage, OverflowDetected> {
    #[cfg(table_format = "q64_64")]
    {
        // ComputeStorage = I256. Table covers 0..=38. I256 safe max ~10^76.
        const TABLE: [I256; 39] = pow10_table_i256();
        if (exp as usize) < TABLE.len() {
            Ok(TABLE[exp as usize])
        } else if exp <= 76 {
            Ok(pow10_compute_fallback_i256(exp))
        } else {
            Err(OverflowDetected::Overflow)
        }
    }
    #[cfg(table_format = "q128_128")]
    {
        // ComputeStorage = I512. Table covers 0..=76. I512 safe max ~10^153.
        const TABLE: [I512; 77] = pow10_table_i512();
        if (exp as usize) < TABLE.len() {
            Ok(TABLE[exp as usize])
        } else if exp <= 153 {
            Ok(pow10_compute_fallback_i512(exp))
        } else {
            Err(OverflowDetected::Overflow)
        }
    }
    #[cfg(table_format = "q256_256")]
    {
        // ComputeStorage = I1024. Table covers 0..=154. I1024 safe max ~10^307.
        // u8 max is 255 < 308, so all fallback values are safe.
        const TABLE: [I1024; 155] = pow10_table_i1024();
        if (exp as usize) < TABLE.len() {
            Ok(TABLE[exp as usize])
        } else {
            Ok(pow10_compute_fallback_i1024(exp))
        }
    }
    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128. i128 safe max ~10^38.
        const TABLE: [i128; 39] = pow10_table_i128();
        if (exp as usize) < TABLE.len() {
            Ok(TABLE[exp as usize])
        } else {
            Err(OverflowDetected::Overflow)
        }
    }
    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64. i64 safe max ~10^18.
        const TABLE: [i64; 19] = pow10_table_i64();
        if (exp as usize) < TABLE.len() {
            Ok(TABLE[exp as usize])
        } else {
            Err(OverflowDetected::Overflow)
        }
    }
}

// ============================================================================
// CONST POW10 TABLE GENERATORS — compile-time word-level multiplication
// ============================================================================

/// Multiply a little-endian u64 word array by 10, returning the new words.
/// Pure const fn — no trait dispatch, no heap.
const fn mul10_words<const N: usize>(words: [u64; N]) -> [u64; N] {
    let mut result = [0u64; N];
    let mut carry: u64 = 0;
    let mut i = 0;
    while i < N {
        let prod = words[i] as u128 * 10 + carry as u128;
        result[i] = prod as u64;
        carry = (prod >> 64) as u64;
        i += 1;
    }
    result
}

/// Build const pow10 table for I256 (4 words, covers 10^0..10^38)
#[allow(dead_code)]
pub(super) const fn pow10_table_i256() -> [I256; 39] {
    let mut table = [I256::from_words([0; 4]); 39];
    let mut words = [0u64; 4];
    words[0] = 1; // 10^0 = 1
    table[0] = I256::from_words(words);
    let mut i = 1;
    while i < 39 {
        words = mul10_words(words);
        table[i] = I256::from_words(words);
        i += 1;
    }
    table
}

/// Build const pow10 table for I512 (8 words, covers 10^0..10^76)
#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
pub(super) const fn pow10_table_i512() -> [I512; 77] {
    let mut table = [I512::from_words([0; 8]); 77];
    let mut words = [0u64; 8];
    words[0] = 1;
    table[0] = I512::from_words(words);
    let mut i = 1;
    while i < 77 {
        words = mul10_words(words);
        table[i] = I512::from_words(words);
        i += 1;
    }
    table
}

/// Build const pow10 table for I1024 (16 words, covers 10^0..10^154)
#[cfg(table_format = "q256_256")]
pub(super) const fn pow10_table_i1024() -> [I1024; 155] {
    let mut table = [I1024::from_words([0; 16]); 155];
    let mut words = [0u64; 16];
    words[0] = 1;
    table[0] = I1024::from_words(words);
    let mut i = 1;
    while i < 155 {
        words = mul10_words(words);
        table[i] = I1024::from_words(words);
        i += 1;
    }
    table
}

/// Build const pow10 table for i128 (covers 10^0..10^38)
#[cfg(table_format = "q32_32")]
pub(super) const fn pow10_table_i128() -> [i128; 39] {
    let mut table = [0i128; 39];
    table[0] = 1;
    let mut i = 1;
    while i < 39 {
        table[i] = table[i - 1] * 10;
        i += 1;
    }
    table
}

/// Build const pow10 table for i64 (covers 10^0..10^18)
#[cfg(table_format = "q16_16")]
pub(super) const fn pow10_table_i64() -> [i64; 19] {
    let mut table = [0i64; 19];
    table[0] = 1;
    let mut i = 1;
    while i < 19 {
        table[i] = table[i - 1] * 10;
        i += 1;
    }
    table
}

// Fallback loops for exponents beyond the precomputed table
#[cold]
#[allow(dead_code)]
fn pow10_compute_fallback_i256(exp: u8) -> I256 {
    let mut result = I256::from_i128(1);
    let ten = I256::from_i128(10);
    for _ in 0..exp { result = result * ten; }
    result
}

#[cold]
#[cfg(table_format = "q128_128")]
fn pow10_compute_fallback_i512(exp: u8) -> I512 {
    let mut result = I512::from_i128(1);
    let ten = I512::from_i128(10);
    for _ in 0..exp { result = result * ten; }
    result
}

#[cold]
#[cfg(table_format = "q256_256")]
fn pow10_compute_fallback_i1024(exp: u8) -> I1024 {
    let mut result = I1024::from_i128(1);
    let ten = I1024::from_i128(10);
    for _ in 0..exp { result = result * ten; }
    result
}

/// Convert a Symbolic rational (num/den as i128) directly to ComputeStorage at full compute-tier precision.
#[inline]
pub(super) fn symbolic_to_compute_storage(num: i128, den: i128) -> Result<ComputeStorage, OverflowDetected> {
    #[cfg(table_format = "q256_256")]
    {
        let n = I1024::from_i512(I512::from_i256(I256::from_i128(num))) << 512;
        let d = I1024::from_i512(I512::from_i256(I256::from_i128(den)));
        Ok(n / d)
    }
    #[cfg(table_format = "q128_128")]
    {
        let n = I512::from_i256(I256::from_i128(num)) << 256;
        let d = I512::from_i256(I256::from_i128(den));
        Ok(n / d)
    }
    #[cfg(table_format = "q64_64")]
    {
        let n = I256::from_i128(num) << 128;
        let d = I256::from_i128(den);
        Ok(n / d)
    }
    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128 (Q64.64). Use I256 intermediate for (num << 64) / den.
        let n = I256::from_i128(num) << 64;
        let d = I256::from_i128(den);
        Ok((n / d).as_i128())
    }
    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64. Use i128 intermediate for (num << COMPUTE_FRAC_BITS) / den.
        let n = (num as i128) << frac_config::COMPUTE_FRAC_BITS;
        let d = den as i128;
        Ok((n / d) as i64)
    }
}

/// Convert Symbolic rational (Massive I256 or Ultra I512) to ComputeStorage.
/// Called when i128 extraction fails due to constants stored at higher tiers.
pub(super) fn symbolic_wide_to_compute_storage(
    rational: &crate::fixed_point::domains::symbolic::rational::RationalNumber,
) -> Result<ComputeStorage, OverflowDetected> {
    let parts = rational.extract_native();

    #[cfg(table_format = "q128_128")]
    {
        // Try I256 extraction (Massive tier)
        if let Some((num_i256, den_i256)) = parts.try_as_i256_pair() {
            let n = I512::from_i256(num_i256) << 256;
            let d = I512::from_i256(den_i256);
            return Ok(n / d);
        }
        // Try I512 extraction (Ultra tier)
        if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
            let n = I1024::from_i512(num_i512) << 256;
            let d = I1024::from_i512(den_i512);
            return Ok((n / d).as_i512());
        }
    }

    #[cfg(table_format = "q256_256")]
    {
        if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
            let n = I1024::from_i512(num_i512) << 512;
            let d = I1024::from_i512(den_i512);
            return Ok(n / d);
        }
    }

    #[cfg(table_format = "q64_64")]
    {
        if let Some((num_i256, den_i256)) = parts.try_as_i256_pair() {
            let n = I256::from_i128(num_i256.as_i128()) << 128;
            let d = I256::from_i128(den_i256.as_i128());
            return Ok(n / d);
        }
        // Ultra tier (I512): mathematical constants (e, π, etc.) are stored at
        // 77-digit precision as I512/I512 rationals. Use I1024 intermediate to
        // convert to Q128.128 (ComputeStorage for q64_64) without overflow.
        if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
            let n = I1024::from_i512(num_i512) << 128;
            let d = I1024::from_i512(den_i512);
            return Ok((n / d).as_i256());
        }
    }

    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128 (Q64.64). I256 intermediate for shift+divide.
        if let Some((num_i256, den_i256)) = parts.try_as_i256_pair() {
            let n = I256::from_i128(num_i256.as_i128()) << 64;
            let d = I256::from_i128(den_i256.as_i128());
            return Ok((n / d).as_i128());
        }
        if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
            let n = I1024::from_i512(num_i512) << 64;
            let d = I1024::from_i512(den_i512);
            return Ok((n / d).as_i512().as_i256().as_i128());
        }
    }

    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64. Use I256 intermediate for shift+divide → i64.
        if let Some((num_i256, den_i256)) = parts.try_as_i256_pair() {
            let n = I256::from_i128(num_i256.as_i128()) << (frac_config::COMPUTE_FRAC_BITS as usize);
            let d = I256::from_i128(den_i256.as_i128());
            return Ok((n / d).as_i128() as i64);
        }
        if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
            let n = I1024::from_i512(num_i512) << (frac_config::COMPUTE_FRAC_BITS as usize);
            let d = I1024::from_i512(den_i512);
            return Ok((n / d).as_i512().as_i256().as_i128() as i64);
        }
    }

    Err(OverflowDetected::TierOverflow)
}

// ============================================================================
// COMPUTE-TIER ARITHMETIC HELPERS
// ============================================================================

/// Add two compute-tier values
#[inline]
pub(crate) fn compute_add(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage {
    a + b
}

/// Subtract two compute-tier values
#[inline]
pub(crate) fn compute_subtract(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage {
    a - b
}

/// Negate a compute-tier value
#[inline]
pub(crate) fn compute_negate(a: ComputeStorage) -> ComputeStorage {
    -a
}

/// Multiply two compute-tier values (needs double-width intermediate)
#[inline]
pub(crate) fn compute_multiply(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    {
        // I1024 × I1024 → I2048 >> 512 → I1024
        // MUST use signed multiply: mul_to_i2048 is unsigned
        use crate::fixed_point::domains::binary_fixed::transcendental::multiply_i1024_q512_512;
        multiply_i1024_q512_512(a, b)
    }
    #[cfg(table_format = "q128_128")]
    {
        // I512 × I512 → I1024 >> 256 → I512 (with rounding)
        let a_wide = I1024::from_i512(a);
        let b_wide = I1024::from_i512(b);
        let product = a_wide * b_wide;
        let round_bit = (product & (I1024::from_i128(1) << 255)) != I1024::zero();
        let mut result = (product >> 256).as_i512();
        if round_bit {
            result = result + I512::from_i128(1);
        }
        result
    }
    #[cfg(table_format = "q64_64")]
    {
        // I256 × I256 → I512 >> 128 → I256 (with rounding)
        let a_wide = I512::from_i256(a);
        let b_wide = I512::from_i256(b);
        let product = a_wide * b_wide;
        let round_bit = (product & (I512::from_i128(1) << 127)) != I512::zero();
        let mut result = (product >> 128).as_i256();
        if round_bit {
            result = result + I256::from_i128(1);
        }
        result
    }
    #[cfg(table_format = "q32_32")]
    {
        // i128 × i128 → I256 >> 64 → i128 (with rounding)
        // ComputeStorage = i128 (Q64.64), COMPUTE_FRAC = 64
        let a_wide = I256::from_i128(a);
        let b_wide = I256::from_i128(b);
        let product = a_wide * b_wide;
        let round_bit = (product & (I256::from_i128(1) << 63)) != I256::zero();
        let mut result = (product >> 64).as_i128();
        if round_bit {
            result += 1;
        }
        result
    }
    #[cfg(table_format = "q16_16")]
    {
        // i64 × i64 → i128 >> COMPUTE_FRAC_BITS → i64 (with rounding)
        let product = (a as i128) * (b as i128);
        let round_bit = (product & (1i128 << frac_config::COMPUTE_ROUND_BIT)) != 0;
        let mut result = (product >> frac_config::COMPUTE_FRAC_BITS) as i64;
        if round_bit {
            result += 1;
        }
        result
    }
}

/// Divide two compute-tier values (needs double-width intermediate for numerator shift)
#[inline]
pub(crate) fn compute_divide(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    #[cfg(table_format = "q256_256")]
    {
        // (I1024 << 512) / I1024 — uses I2048 for numerator
        if b == I1024::zero() { return Err(OverflowDetected::DivisionByZero); }
        let a_wide = I2048::from_i1024(a) << 512;
        let b_wide = I2048::from_i1024(b);
        // I2048 has no Div trait — use the schoolbook division helper
        use crate::fixed_point::domains::binary_fixed::i2048::i2048_div;
        Ok(i2048_div(a_wide, b_wide).as_i1024())
    }
    #[cfg(table_format = "q128_128")]
    {
        // (I512 << 256) / I512 — use I1024 for shifted numerator
        if b == I512::zero() { return Err(OverflowDetected::DivisionByZero); }
        let a_wide = I1024::from_i512(a) << 256;
        let b_wide = I1024::from_i512(b);
        Ok((a_wide / b_wide).as_i512())
    }
    #[cfg(table_format = "q64_64")]
    {
        // (I256 << 128) / I256 — use I512 for shifted numerator
        if b == I256::zero() { return Err(OverflowDetected::DivisionByZero); }
        let a_wide = I512::from_i256(a) << 128;
        let b_wide = I512::from_i256(b);
        Ok((a_wide / b_wide).as_i256())
    }
    #[cfg(table_format = "q32_32")]
    {
        // (i128 << 64) / i128 — use I256 for shifted numerator
        // ComputeStorage = i128 (Q64.64), COMPUTE_FRAC = 64
        if b == 0i128 { return Err(OverflowDetected::DivisionByZero); }
        let a_wide = I256::from_i128(a) << 64;
        let b_wide = I256::from_i128(b);
        Ok((a_wide / b_wide).as_i128())
    }
    #[cfg(table_format = "q16_16")]
    {
        // (i64 << COMPUTE_FRAC_BITS) / i64 — use i128 for shifted numerator
        if b == 0i64 { return Err(OverflowDetected::DivisionByZero); }
        let a_wide = (a as i128) << frac_config::COMPUTE_FRAC_BITS;
        let b_wide = b as i128;
        Ok((a_wide / b_wide) as i64)
    }
}

/// Halve a compute-tier value (right shift by 1)
#[inline]
pub(crate) fn compute_halve(a: ComputeStorage) -> ComputeStorage {
    a >> 1
}

/// Create a compute-tier integer (e.g., 1, 2 in the compute Q-format)
#[allow(dead_code)]
#[inline]
pub(crate) fn make_compute_int(n: i64) -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    { I1024::from_i128(n as i128) << 512 }
    #[cfg(table_format = "q128_128")]
    { I512::from_i128(n as i128) << 256 }
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(n as i128) << 128 }
    #[cfg(table_format = "q32_32")]
    { (n as i128) << 64 }
    #[cfg(table_format = "q16_16")]
    { n << frac_config::COMPUTE_FRAC_BITS }
}

/// Check if a compute-tier value is zero
#[inline]
pub(crate) fn compute_is_zero(a: &ComputeStorage) -> bool {
    #[cfg(table_format = "q256_256")]
    { *a == I1024::zero() }
    #[cfg(table_format = "q128_128")]
    { *a == I512::zero() }
    #[cfg(table_format = "q64_64")]
    { *a == I256::zero() }
    #[cfg(table_format = "q32_32")]
    { *a == 0i128 }
    #[cfg(table_format = "q16_16")]
    { *a == 0i64 }
}

/// Check if a compute-tier value is negative
#[inline]
pub(crate) fn compute_is_negative(a: &ComputeStorage) -> bool {
    #[cfg(table_format = "q256_256")]
    { (a.words[15] & 0x8000_0000_0000_0000) != 0 }
    #[cfg(table_format = "q128_128")]
    { *a < I512::zero() }
    #[cfg(table_format = "q64_64")]
    { *a < I256::zero() }
    #[cfg(table_format = "q32_32")]
    { *a < 0i128 }
    #[cfg(table_format = "q16_16")]
    { *a < 0i64 }
}

// ============================================================================
// COMPUTE-TIER TRANSCENDENTAL DISPATCH FUNCTIONS
// ============================================================================
//
// These free functions dispatch to the profile-appropriate compute-tier
// implementations. They are called by the StackEvaluator evaluate_* methods.

/// Compute sin at compute tier (tier N+1)
#[inline]
pub(super) fn sin_at_compute_tier(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    { sin_compute_tier_i1024(x) }
    #[cfg(table_format = "q128_128")]
    { sin_compute_tier_i512(x) }
    #[cfg(table_format = "q64_64")]
    { sin_compute_tier_i256(x) }
    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128 (Q64.64) — native Q64.64 computation
        sin_binary_i128(x)
    }
    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64 — upscale to Q64.64 (i128), compute, downscale
        sin_compute_tier_i64(x)
    }
}

/// Compute cos at compute tier (tier N+1)
#[inline]
pub(super) fn cos_at_compute_tier(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    { cos_compute_tier_i1024(x) }
    #[cfg(table_format = "q128_128")]
    { cos_compute_tier_i512(x) }
    #[cfg(table_format = "q64_64")]
    { cos_compute_tier_i256(x) }
    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128 (Q64.64) — native Q64.64 computation
        cos_binary_i128(x)
    }
    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64 — upscale to Q64.64 (i128), compute, downscale
        cos_compute_tier_i64(x)
    }
}

/// Fused sin+cos at compute tier (tier N+1) — single shared range reduction.
/// Returns (sin(x), cos(x)) saving one range reduction vs separate calls.
#[inline]
pub(crate) fn sincos_at_compute_tier(x: ComputeStorage) -> (ComputeStorage, ComputeStorage) {
    #[cfg(table_format = "q256_256")]
    { sincos_compute_tier_i1024(x) }
    #[cfg(table_format = "q128_128")]
    { sincos_compute_tier_i512(x) }
    #[cfg(table_format = "q64_64")]
    { sincos_compute_tier_i256(x) }
    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128 (Q64.64) — native Q64.64 sin and cos
        (sin_binary_i128(x), cos_binary_i128(x))
    }
    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64 — fused sincos via Q64.64 (i128)
        sincos_compute_tier_i64(x)
    }
}

/// Compute atan at compute tier (tier N+1)
#[inline]
pub(super) fn atan_at_compute_tier(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    { atan_compute_tier_i1024(x) }
    #[cfg(table_format = "q128_128")]
    { atan_compute_tier_i512(x) }
    #[cfg(table_format = "q64_64")]
    { atan_compute_tier_i256(x) }
    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128 (Q64.64) — native Q64.64 computation
        atan_binary_i128(x)
    }
    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64 — upscale to Q64.64 (i128), compute, downscale
        atan_compute_tier_i64(x)
    }
}

/// Compute atan2 at compute tier (tier N+1)
#[inline]
pub(super) fn atan2_at_compute_tier(y: ComputeStorage, x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    { atan2_compute_tier_i1024(y, x) }
    #[cfg(table_format = "q128_128")]
    { atan2_compute_tier_i512(y, x) }
    #[cfg(table_format = "q64_64")]
    { atan2_compute_tier_i256(y, x) }
    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128 (Q64.64) — native Q64.64 computation
        atan2_binary_i128(y, x)
    }
    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64 — upscale to Q64.64 (i128), compute, downscale
        atan2_compute_tier_i64(y, x)
    }
}

/// Compute exp at compute tier (tier N+1) — free function variant.
///
/// Returns ComputeStorage directly (no StackValue wrapping).
/// Used by fused operations that need exp without evaluator context.
#[inline]
pub(crate) fn exp_at_compute_tier(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    {
        use crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i1024;
        exp_binary_i1024(x)
    }
    #[cfg(table_format = "q128_128")]
    {
        use crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i512;
        exp_binary_i512(x)
    }
    #[cfg(table_format = "q64_64")]
    {
        use crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i256;
        exp_binary_i256(x)
    }
    #[cfg(table_format = "q32_32")]
    {
        use crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i128;
        exp_binary_i128(x)
    }
    #[cfg(table_format = "q16_16")]
    {
        use crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i64;
        exp_binary_i64(x)
    }
}

/// Compute sqrt at compute tier (tier N+1)
#[inline]
pub(crate) fn sqrt_at_compute_tier(x: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    { sqrt_binary_i1024(x) }
    #[cfg(table_format = "q128_128")]
    {
        // ComputeStorage = I512 (Q256.256) — use native sqrt_binary_i512
        sqrt_binary_i512(x)
    }
    #[cfg(table_format = "q64_64")]
    {
        // ComputeStorage = I256 (Q128.128) — use native sqrt_binary_i256
        sqrt_binary_i256(x)
    }
    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128 (Q64.64) — native Q64.64 computation
        sqrt_binary_i128(x)
    }
    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64 — upscale to Q64.64 (i128), compute, downscale
        sqrt_binary_i64(x)
    }
}

/// Return π/2 at compute tier (tier N+1) with full precision
///
/// Uses the build.rs-generated constants at compute-tier resolution.
/// PI_HALF_Q128/Q256/Q512 are all generated unconditionally by build.rs.
#[inline]
pub(super) fn pi_half_at_compute_tier() -> ComputeStorage {
    #[cfg(table_format = "q256_256")]
    {
        // Q512.512 — full precision from PI_HALF_Q512
        pi_half_i1024()
    }
    #[cfg(table_format = "q128_128")]
    {
        // Compute tier is Q256.256 on I512
        // PI_HALF_Q256 constant is at full Q256.256 precision
        use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::PI_HALF_Q256;
        I512::from_words(PI_HALF_Q256)
    }
    #[cfg(table_format = "q64_64")]
    {
        // Compute tier is Q128.128 on I256
        // PI_HALF_Q128 constant is at full Q128.128 precision
        use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::PI_HALF_Q128;
        I256::from_words(PI_HALF_Q128)
    }
    #[cfg(table_format = "q32_32")]
    {
        // Compute tier is Q64.64 on i128
        // PI_HALF_Q64 constant is at full Q64.64 precision
        pi_half_i128()
    }
    #[cfg(table_format = "q16_16")]
    {
        // Derive from PI_HALF_Q64 (always available): shift right by NATIVE_UPSHIFT
        // NATIVE_UPSHIFT = 64 - COMPUTE_FRAC_BITS = 64 - 2*FRAC_BITS
        use crate::fixed_point::domains::binary_fixed::transcendental::sin_cos_tier_n_plus_1::PI_HALF_Q64;
        (PI_HALF_Q64 >> frac_config::NATIVE_UPSHIFT) as i64
    }
}
