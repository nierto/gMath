//! Compute-tier arithmetic and transcendental dispatchers
//!
//! All functions operate at tier N+1 (ComputeStorage) for maximum precision.

#[allow(unused_imports)]
use super::format::FascFormat;
use crate::fixed_point::fixed_num::ProfileFormat;
use super::{BinaryStorage, ComputeStorage};

// Build-time Q-format constants (FRAC_BITS, COMPUTE_FRAC_BITS, etc.)
#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;
use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;

pub(crate) fn upscale_to_compute(val: BinaryStorage) -> ComputeStorage {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::upscale_to_compute(val)
}

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
/// Convert a Decimal StackValue directly to ComputeStorage at full compute-tier precision.
///
/// This avoids the precision loss of going through BinaryStorage first.
/// For Q64.64 profile: converts to Q128.128 (I256) instead of Q64.64→upscale
/// For Q128.128 profile: converts to Q256.256 (I512) instead of Q128.128→upscale
/// For Q256.256 profile: converts to Q512.512 (I1024) instead of Q256.256→upscale
#[inline]
pub(super) fn decimal_to_compute_storage(decimals: u8, scaled: BinaryStorage) -> Result<ComputeStorage, OverflowDetected> {
    <ProfileFormat as FascFormat>::decimal_to_compute_storage(decimals, scaled)
}

/// Compute 10^exp at ComputeStorage precision: O(1) via const lookup table.
///
/// Tables are built at compile time via const fn word-level multiplication.
/// Covers all exponents up to the profile's DP promotion threshold.
/// Returns `Err(Overflow)` if exponent exceeds the type's safe representable range.
#[inline]
pub(super) fn pow10_compute(exp: u8) -> Result<ComputeStorage, OverflowDetected> {
    <ProfileFormat as FascFormat>::pow10_compute(exp)
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

/// Convert a Symbolic rational (num/den as i128) directly to ComputeStorage at full compute-tier precision.
#[inline]
pub(super) fn symbolic_to_compute_storage(num: i128, den: i128) -> Result<ComputeStorage, OverflowDetected> {
    <ProfileFormat as FascFormat>::symbolic_to_compute_storage(num, den)
}

/// Convert Symbolic rational (Massive I256 or Ultra I512) to ComputeStorage.
/// Called when i128 extraction fails due to constants stored at higher tiers.
pub(super) fn symbolic_wide_to_compute_storage(
    rational: &crate::fixed_point::domains::symbolic::rational::RationalNumber,
) -> Result<ComputeStorage, OverflowDetected> {
    <ProfileFormat as FascFormat>::symbolic_wide_to_compute_storage(rational)
}

// ============================================================================
// COMPUTE-TIER ARITHMETIC HELPERS
// ============================================================================
/// Add two compute-tier values, detecting signed overflow instead of wrapping.
///
/// `compute_add` wraps silently (native ints in release, schoolbook wrap on the
/// wide types). Long accumulations that can legitimately exceed the compute-tier
/// envelope: e.g. `Σⱼ eⱼ·vⱼ` in `fused::softmax_mix` over a long context: must
/// surface the overflow as an error rather than return wrapped garbage.
#[inline]
pub(crate) fn compute_checked_add(
    a: ComputeStorage,
    b: ComputeStorage,
) -> Result<ComputeStorage, OverflowDetected> {
    a.checked_add(b).ok_or(OverflowDetected::TierOverflow)
}

/// `a - b` at the compute tier, `Err(TierOverflow)` when it leaves the tier.
pub(crate) fn compute_checked_subtract(
    a: ComputeStorage,
    b: ComputeStorage,
) -> Result<ComputeStorage, OverflowDetected> {
    compute_checked_add(a, compute_checked_negate(b)?)
}

/// `-a` at the compute tier, `Err(TierOverflow)` for the tier's minimum.
pub(crate) fn compute_checked_negate(a: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    <ProfileFormat as FascFormat>::compute_checked_negate(a)
}
/// True when an exp engine result sits at its overflow sentinel.
///
/// The sentinel differs by profile: the q16_16/q32_32/q256_256 engines
/// saturate at the compute type's maximum, but the q64_64/q128_128 engines
/// return `i128::MAX` at storage scale, which on q128_128 DOWNSCALES
/// CLEANLY into the storage maximum, so a `== compute_ceiling()` check (the
/// old guard) silently missed it and cosh/sinh built on the sentinel
/// returned plausible-wrong values (0.5.0 item 2 find). `>=` because no
/// legitimate exp result can exceed the sentinel.
#[inline]
pub(crate) fn exp_sentinel_reached(v: &ComputeStorage) -> bool {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::exp_sentinel_reached(v)
}
/// `a * b` at the compute tier, rounded to nearest from the exact product;
/// `Err(TierOverflow)` when it leaves the compute tier.
pub(crate) fn compute_checked_multiply(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::compute_checked_multiply(a, b)
}

/// Divide two compute-tier values (needs double-width intermediate for numerator shift).
///
/// `Err(TierOverflow)` when the quotient leaves the compute tier. Before 0.6.4
/// every arm narrowed the quotient with a truncating cast and returned the
/// wrapped value as `Ok`; it is now [`compute_checked_divide`].
#[inline]
pub(crate) fn compute_divide(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    compute_checked_divide(a, b)
}

/// Divide two compute-tier values, rounded to nearest with ties toward
/// +infinity (the crate's binary rule) from the exact quotient; a quotient
/// beyond the compute tier is `TierOverflow`. Before 0.6.4 every arm
/// truncated toward zero, so each compute-tier quotient in a chain carried
/// up to one unit of one-sided error.
#[inline]
pub(crate) fn compute_checked_divide(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::compute_checked_divide(a, b)
}

/// Halve a compute-tier value (right shift by 1)
#[inline]
pub(crate) fn compute_halve(a: ComputeStorage) -> ComputeStorage {
    a >> 1
}

/// Create a compute-tier integer (e.g., 1, 2 in the compute Q-format).
///
/// On realtime the tier holds only +-2^(63 - 2F): a larger `n` panics
/// instead of wrapping (use `compute_mul_div_int` to scale by a large integer).
#[allow(dead_code)]
#[inline]
pub(crate) fn make_compute_int(n: i64) -> ComputeStorage {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::make_compute_int(n)
}
/// Check if a compute-tier value is zero
#[inline]
pub(crate) fn compute_is_zero(a: &ComputeStorage) -> bool {
    <ProfileFormat as FascFormat>::compute_is_zero(a)
}

/// Check if a compute-tier value is negative
#[inline]
pub(crate) fn compute_is_negative(a: &ComputeStorage) -> bool {
    <ProfileFormat as FascFormat>::compute_is_negative(a)
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
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::sin_at_compute_tier(x)
}

/// Compute cos at compute tier (tier N+1)
#[inline]
pub(super) fn cos_at_compute_tier(x: ComputeStorage) -> ComputeStorage {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::cos_at_compute_tier(x)
}

/// Fused sinh+cosh at compute tier (tier N+1): shares one exp-pair evaluation.
///
/// Evaluates `exp(x)` and `exp(-x)` once at compute tier, then derives
/// `sinh(x) = (exp(x) - exp(-x)) / 2` and `cosh(x) = (exp(x) + exp(-x)) / 2`.
///
/// Benefits over separate calls:
///   • 2 exp evaluations instead of 4 (`sinh` + `cosh` each call exp twice today).
///   • sinh and cosh are derived from the **same** `(ep, en)` pair, so their
///     rounding bias is correlated: critical for expressions like
///     `cosh(θ)·p + (sinh(θ)/θ)·v` where the two errors ought to cancel.
#[inline]
pub(crate) fn sinhcosh_at_compute_tier(x: ComputeStorage) -> (ComputeStorage, ComputeStorage) {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::sinhcosh_at_compute_tier(x)
}

/// Fused sin+cos at compute tier (tier N+1): single shared range reduction.
/// Returns (sin(x), cos(x)) saving one range reduction vs separate calls.
#[inline]
pub(crate) fn sincos_at_compute_tier(x: ComputeStorage) -> (ComputeStorage, ComputeStorage) {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::sincos_at_compute_tier(x)
}

/// Compute atan at compute tier (tier N+1)
#[inline]
pub(super) fn atan_at_compute_tier(x: ComputeStorage) -> ComputeStorage {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::atan_at_compute_tier(x)
}

/// Compute atan2 at compute tier (tier N+1)
#[inline]
pub(super) fn atan2_at_compute_tier(y: ComputeStorage, x: ComputeStorage) -> ComputeStorage {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::atan2_at_compute_tier(y, x)
}
/// Compute sqrt at compute tier (tier N+1)
#[inline]
pub(crate) fn sqrt_at_compute_tier(x: ComputeStorage) -> ComputeStorage {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::sqrt_at_compute_tier(x)
}

/// Compute ln at compute tier (tier N+1): free function variant.
///
/// Returns ComputeStorage directly (no StackValue wrapping).
/// Propagates the kernel's domain sentinel (type MIN) for x <= 0;
/// callers must domain-check before calling or handle the sentinel.
/// Only consumed by the feature-gated `compute_tier` module, hence the
/// dead_code allowance for no-feature builds (same as `make_compute_int`).
#[allow(dead_code)]
#[inline]
pub(crate) fn ln_at_compute_tier(x: ComputeStorage) -> ComputeStorage {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::ln_at_compute_tier(x)
}

/// Return π/2 at compute tier (tier N+1) with full precision
///
/// Uses the build.rs-generated constants at compute-tier resolution.
/// PI_HALF_Q128/Q256/Q512 are all generated unconditionally by build.rs.
#[inline]
pub(super) fn pi_half_at_compute_tier() -> ComputeStorage {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::pi_half_at_compute_tier()
}
