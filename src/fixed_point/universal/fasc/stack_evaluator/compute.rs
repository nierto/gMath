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

/// Downscale from compute tier to storage tier rounding toward negative
/// infinity (floor).
///
/// The arithmetic right shift IS floor in two's complement, so this is
/// `downscale_to_storage` without its round-bit bump. Fits-checked the same
/// way. Used for the lower endpoint of certified intervals; never on a scalar
/// path, which keeps the one-rounding-rule-per-domain contract intact.
#[inline]
pub(crate) fn downscale_to_storage_floor(val: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::downscale_to_storage_floor(val)
}

/// Downscale from compute tier to storage tier rounding toward positive
/// infinity (ceil).
///
/// Floor plus one whenever any discarded bit is set. The bump is checked:
/// a value whose floor is exactly the storage maximum has no ceiling in the
/// storage tier, and that is a `TierOverflow`, not a wrap. Used for the upper
/// endpoint of certified intervals.
#[inline]
pub(crate) fn downscale_to_storage_ceil(val: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::downscale_to_storage_ceil(val)
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
        {
            // checked: at 24 fraction bits the compute tier holds +-2^15
            let v = (scaled as i128) << frac_config::COMPUTE_FRAC_BITS;
            return i64::try_from(v).map_err(|_| OverflowDetected::TierOverflow);
        }
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
        // i128: scaled << 2F can pass i64 at high fraction bits (it wrapped:
        // "100.1234" at 24 fraction bits is 1001234 * 2^48; before 0.6.4)
        let num = (scaled as i128) << frac_config::COMPUTE_FRAC_BITS;
        i64::try_from(num / den as i128).map_err(|_| OverflowDetected::TierOverflow)
    }
}

/// Compute 10^exp at ComputeStorage precision: O(1) via const lookup table.
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
        let q = n / d;
        if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
        Ok(q.as_i128())
    }
    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64. I256 intermediate: num << 2F can pass i128.
        // Range-checked: the bare `as i64` wrapped a symbolic value beyond
        // the realtime compute range (atan(999999999) at 24 fraction bits
        // came out as -pi/2; before 0.6.4).
        let n = I256::from_i128(num) << (frac_config::COMPUTE_FRAC_BITS as usize);
        let q = n / I256::from_i128(den);
        if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
        i64::try_from(q.as_i128()).map_err(|_| OverflowDetected::TierOverflow)
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
            let q = n / d;
            if !q.fits_in_i512() { return Err(OverflowDetected::TierOverflow); }
            return Ok(q.as_i512());
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
            // I512: an I256 numerator shifted by 128 (was truncated to i128)
            let q = (I512::from_i256(num_i256) << 128) / I512::from_i256(den_i256);
            if !q.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
            return Ok(q.as_i256());
        }
        // Ultra tier (I512): mathematical constants (e, π, etc.) are stored at
        // 77-digit precision as I512/I512 rationals. Use I1024 intermediate to
        // convert to Q128.128 (ComputeStorage for q64_64) without overflow.
        if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
            let n = I1024::from_i512(num_i512) << 128;
            let d = I1024::from_i512(den_i512);
            let q = n / d;
            if !q.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
            return Ok(q.as_i256());
        }
    }

    #[cfg(table_format = "q32_32")]
    {
        // ComputeStorage = i128 (Q64.64). I256 intermediate for shift+divide.
        if let Some((num_i256, den_i256)) = parts.try_as_i256_pair() {
            let q = (I512::from_i256(num_i256) << 64) / I512::from_i256(den_i256);
            if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
            return Ok(q.as_i128());
        }
        if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
            let n = I1024::from_i512(num_i512) << 64;
            let d = I1024::from_i512(den_i512);
            let q = n / d;
            if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
            return Ok(q.as_i512().as_i256().as_i128());
        }
    }

    #[cfg(table_format = "q16_16")]
    {
        // ComputeStorage = i64. Use I256 intermediate for shift+divide → i64.
        if let Some((num_i256, den_i256)) = parts.try_as_i256_pair() {
            let q = (I512::from_i256(num_i256) << (frac_config::COMPUTE_FRAC_BITS as usize)) / I512::from_i256(den_i256);
            if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
            return i64::try_from(q.as_i128()).map_err(|_| OverflowDetected::TierOverflow);
        }
        if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
            let n = I1024::from_i512(num_i512) << (frac_config::COMPUTE_FRAC_BITS as usize);
            let d = I1024::from_i512(den_i512);
            let q = n / d;
            if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
            return i64::try_from(q.as_i128()).map_err(|_| OverflowDetected::TierOverflow);
        }
    }

    Err(OverflowDetected::TierOverflow)
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
    #[cfg(not(table_format = "q256_256"))]
    { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    #[cfg(table_format = "q256_256")]
    { if a == I1024::min_value() { Err(OverflowDetected::TierOverflow) } else { Ok(-a) } }
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
