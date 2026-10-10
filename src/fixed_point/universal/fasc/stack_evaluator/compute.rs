//! Compute-tier arithmetic and transcendental dispatchers
//!
//! All functions operate at tier N+1 (ComputeStorage) for maximum precision.

use super::{BinaryStorage, ComputeStorage};

// Build-time Q-format constants (FRAC_BITS, COMPUTE_FRAC_BITS, etc.)
#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

/// Downscale from compute tier to storage tier with round-to-nearest.
///
/// Shifts right by (COMPUTE_FRAC_BITS - STORAGE_FRAC_BITS) with rounding.
///
/// Returns `Err(TierOverflow)` if the compute-tier value does not fit in
/// the storage tier; this is the UGOD overflow detection point that
/// prevents silent truncation of large results (e.g., exp(44) in Q64.64).
#[inline]
pub(crate) fn downscale_to_storage(val: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::downscale_to_storage(val)
}
// ============================================================================
// CONST POW10 TABLE GENERATORS — compile-time word-level multiplication
// ============================================================================

/// Multiply a little-endian u64 word array by 10, returning the new words.
/// Pure const fn, no trait dispatch, no heap.
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
pub(super) fn pow10_compute_fallback_i256(exp: u8) -> I256 {
    let mut result = I256::from_i128(1);
    let ten = I256::from_i128(10);
    for _ in 0..exp { result = result * ten; }
    result
}

#[cold]
pub(super) fn pow10_compute_fallback_i512(exp: u8) -> I512 {
    let mut result = I512::from_i128(1);
    let ten = I512::from_i128(10);
    for _ in 0..exp { result = result * ten; }
    result
}

#[cold]
pub(super) fn pow10_compute_fallback_i1024(exp: u8) -> I1024 {
    let mut result = I1024::from_i128(1);
    let ten = I1024::from_i128(10);
    for _ in 0..exp { result = result * ten; }
    result
}

// ============================================================================
// COMPUTE-TIER ARITHMETIC HELPERS
// ============================================================================
/// Create a compute-tier integer (e.g., 1, 2 in the compute Q-format).
///
/// On realtime the tier holds only +-2^(63 - 2F): a larger `n` panics
/// instead of wrapping (use `compute_mul_div_int` to scale by a large integer).
#[allow(dead_code)]
#[inline]
pub(crate) fn make_compute_int(n: i64) -> ComputeStorage {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::make_compute_int(n)
}
// ============================================================================
// COMPUTE-TIER TRANSCENDENTAL DISPATCH FUNCTIONS
// ============================================================================
//
// These free functions dispatch to the profile-appropriate compute-tier
// implementations. They are called by the StackEvaluator evaluate_* methods.

