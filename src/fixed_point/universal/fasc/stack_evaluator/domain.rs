//! Domain marshaling and shadow arithmetic
//!
//! Bridges between StackValue representation and domain-specific UGOD types
//! (UniversalTernaryFixed, UniversalDecimalTiered, UniversalBinaryFixed).
//! Also maintains CompactShadow precision during arithmetic operations.

use super::BinaryStorage;
use crate::fixed_point::i256::I256;
use crate::fixed_point::i512::I512;
use crate::fixed_point::I1024;
use crate::fixed_point::domains::balanced_ternary::ternary_types::{UniversalTernaryFixed, TernaryRaw};
use crate::fixed_point::domains::binary_fixed::binary_types::{UniversalBinaryFixed, BinaryRaw};
use crate::fixed_point::domains::decimal_fixed::decimal_types::UniversalDecimalTiered;
use crate::fixed_point::universal::tier_types::CompactShadow;
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, OverflowDetected};
use super::conversion::{to_binary_storage, binary_storage_to_i128};

pub(super) fn ternary_from_storage(tier: u8, storage: &BinaryStorage) -> Result<UniversalTernaryFixed, OverflowDetected> {
    match tier {
        1..=3 => {
            // Tiers 1-3 fit in i128
            let val = binary_storage_to_i128(storage);
            UniversalTernaryFixed::from_tier_value(tier, val)
        }
        4 => {
            // Tier 4 uses I256
            #[cfg(table_format = "q256_256")]
            { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(storage.as_i256())) }

            #[cfg(table_format = "q128_128")]
            { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(*storage)) }

            #[cfg(table_format = "q64_64")]
            { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(I256::from_i128(*storage))) }

            #[cfg(table_format = "q32_32")]
            { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(I256::from_i128(*storage as i128))) }

            #[cfg(table_format = "q16_16")]
            { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(I256::from_i128(*storage as i128))) }

        }
        5 => {
            // Tier 5 uses I512
            #[cfg(table_format = "q256_256")]
            { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(*storage)) }

            #[cfg(table_format = "q128_128")]
            { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(I512::from_i256(*storage))) }

            #[cfg(table_format = "q64_64")]
            { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(I512::from_i128(*storage))) }

            #[cfg(table_format = "q32_32")]
            { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(I512::from_i128(*storage as i128))) }

            #[cfg(table_format = "q16_16")]
            { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(I512::from_i128(*storage as i128))) }

        }
        6 => {
            // Tier 6 uses I1024
            #[cfg(table_format = "q256_256")]
            { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i512(*storage))) }

            #[cfg(table_format = "q128_128")]
            { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i256(*storage))) }

            #[cfg(table_format = "q64_64")]
            { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i128(*storage))) }

            #[cfg(table_format = "q32_32")]
            { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i128(*storage as i128))) }

            #[cfg(table_format = "q16_16")]
            { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i128(*storage as i128))) }

        }
        _ => Err(OverflowDetected::InvalidInput)
    }
}

/// Checked narrowing of an i128 tier raw into BinaryStorage.
///
/// Returns TierOverflow instead of wrapping — the bare `as i32`/`as i64`
/// casts this replaces silently corrupted Tier-2+ ternary values on the
/// realtime/compact profiles (wrap-defect class, fixed 0.4.33).
fn i128_to_binary_storage_checked(val: i128) -> Result<BinaryStorage, OverflowDetected> {
    #[cfg(table_format = "q256_256")]
    { Ok(I512::from_i128(val)) }

    #[cfg(table_format = "q128_128")]
    { Ok(I256::from_i128(val)) }

    #[cfg(table_format = "q64_64")]
    { Ok(val) }

    #[cfg(table_format = "q32_32")]
    { i64::try_from(val).map_err(|_| OverflowDetected::TierOverflow) }

    #[cfg(table_format = "q16_16")]
    { i32::try_from(val).map_err(|_| OverflowDetected::TierOverflow) }
}

/// Convert UniversalTernaryFixed result back to (tier, BinaryStorage) — full precision.
///
/// Fallible: a tier raw that does not fit the profile\'s BinaryStorage is a
/// TierOverflow error, never a silent wrap (e.g. `0t3281` parses to Tier 2
/// with raw 3281·3^16 ≈ 1.4e11, which cannot live in realtime\'s i32).
pub(super) fn ternary_to_storage(ternary: &UniversalTernaryFixed) -> Result<(u8, BinaryStorage), OverflowDetected> {
    let (tier, raw) = ternary.to_tier_raw();
    let storage = match raw {
        TernaryRaw::Small(v) => i128_to_binary_storage_checked(v)?,
        TernaryRaw::Medium(v) => {
            // I256 → BinaryStorage
            #[cfg(table_format = "q256_256")]
            { I512::from_i256(v) }

            #[cfg(table_format = "q128_128")]
            { v }

            #[cfg(not(any(table_format = "q256_256", table_format = "q128_128")))]
            {
                if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                i128_to_binary_storage_checked(v.as_i128())?
            }
        }
        TernaryRaw::Large(v) => {
            // I512 → BinaryStorage
            #[cfg(table_format = "q256_256")]
            { v }

            #[cfg(table_format = "q128_128")]
            {
                if !v.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
                v.as_i256()
            }

            #[cfg(not(any(table_format = "q256_256", table_format = "q128_128")))]
            {
                if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                i128_to_binary_storage_checked(v.as_i128())?
            }
        }
        TernaryRaw::XLarge(v) => {
            // I1024 → BinaryStorage
            #[cfg(table_format = "q256_256")]
            {
                if !v.fits_in_i512() { return Err(OverflowDetected::TierOverflow); }
                v.as_i512()
            }

            #[cfg(table_format = "q128_128")]
            {
                if !v.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
                v.as_i256()
            }

            #[cfg(not(any(table_format = "q256_256", table_format = "q128_128")))]
            {
                if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                i128_to_binary_storage_checked(v.as_i128())?
            }
        }
    };
    Ok((tier, storage))
}

/// Convert ternary StackValue fields to RationalNumber.
///
/// **PURPOSE**: Ternary stores `value * 3^frac_trits`. This converts to `value / 3^frac_trits`
/// as a RationalNumber. Used by to_rational(), to_compute_storage(), to_binary_storage(), Display.
pub(super) fn ternary_to_rational(tier: u8, value: &BinaryStorage) -> Result<RationalNumber, OverflowDetected> {
    let ternary = ternary_from_storage(tier, value)?;
    let (ternary_tier, _raw) = ternary.to_tier_raw();

    let frac_trits: u32 = match ternary_tier {
        1 => 8,    // TQ8.8
        2 => 16,   // TQ16.16
        3 => 32,   // TQ32.32
        4 => 64,   // TQ64.64
        5 => 128,  // TQ128.128
        6 => 256,  // TQ256.256
        _ => 32,
    };

    if frac_trits <= 32 {
        let mut denom = 1i128;
        for _ in 0..frac_trits {
            denom *= 3;
        }
        let val_i128 = binary_storage_to_i128(value);
        return Ok(RationalNumber::new(val_i128, denom as u128));
    }

    #[cfg(table_format = "q256_256")]
    {
        let val = *value;

        // GCD-reduce: ternary denominators are 3^N, divide out common 3s
        // from numerator to minimize the rational tier required.
        // E.g., 10 * 3^128 / 3^128 reduces to 10/1 (Huge tier, not Ultra).
        let three = I512::from_i128(3);
        let zero = I512::zero();
        let mut reduced_num = val;
        let mut remaining_pow = frac_trits;
        while remaining_pow > 0 && (reduced_num % three) == zero {
            reduced_num = reduced_num / three;
            remaining_pow -= 1;
        }

        // Select smallest RationalNumber tier that fits
        if remaining_pow == 0 {
            if reduced_num.fits_in_i128() {
                return Ok(RationalNumber::new(reduced_num.as_i128(), 1));
            }
            if reduced_num.fits_in_i256() {
                return Ok(RationalNumber::from_i256_pair(reduced_num.as_i256(), I256::from_i128(1)));
            }
            return Ok(RationalNumber::from_i512_pair(reduced_num, I512::from_i128(1)));
        }

        // Rebuild reduced denominator (3^remaining_pow)
        let mut reduced_den = I512::from_i128(1);
        for _ in 0..remaining_pow {
            reduced_den = reduced_den * three;
        }
        if reduced_num.fits_in_i128() && reduced_den.fits_in_i128() {
            return Ok(RationalNumber::new(reduced_num.as_i128(), reduced_den.as_i128() as u128));
        }
        if reduced_num.fits_in_i256() && reduced_den.fits_in_i256() {
            return Ok(RationalNumber::from_i256_pair(reduced_num.as_i256(), reduced_den.as_i256()));
        }
        return Ok(RationalNumber::from_i512_pair(reduced_num, reduced_den));
    }

    #[cfg(table_format = "q128_128")]
    {
        let val = *value;

        // GCD-reduce: divide out common powers of 3
        let three = I256::from_i128(3);
        let zero = I256::zero();
        let mut reduced_num = val;
        let mut remaining_pow = frac_trits;
        while remaining_pow > 0 && (reduced_num % three) == zero {
            reduced_num = reduced_num / three;
            remaining_pow -= 1;
        }

        if remaining_pow == 0 {
            if reduced_num.fits_in_i128() {
                return Ok(RationalNumber::new(reduced_num.as_i128(), 1));
            }
            return Ok(RationalNumber::from_i256_pair(reduced_num, I256::from_i128(1)));
        }

        let mut reduced_den = I256::from_i128(1);
        for _ in 0..remaining_pow {
            reduced_den = reduced_den * three;
        }
        if reduced_num.fits_in_i128() && reduced_den.fits_in_i128() {
            return Ok(RationalNumber::new(reduced_num.as_i128(), reduced_den.as_i128() as u128));
        }
        return Ok(RationalNumber::from_i256_pair(reduced_num, reduced_den));
    }

    #[cfg(table_format = "q64_64")]
    {
        let val_i128 = *value;
        let mut denom = 1i128;
        for _ in 0..frac_trits {
            denom *= 3;
        }
        return Ok(RationalNumber::new(val_i128, denom as u128));
    }

    #[cfg(table_format = "q32_32")]
    {
        let val_i128 = *value as i128;
        let mut denom = 1i128;
        for _ in 0..frac_trits {
            denom *= 3;
        }
        return Ok(RationalNumber::new(val_i128, denom as u128));
    }

    #[cfg(table_format = "q16_16")]
    {
        let val_i128 = *value as i128;
        let mut denom = 1i128;
        for _ in 0..frac_trits {
            denom *= 3;
        }
        return Ok(RationalNumber::new(val_i128, denom as u128));
    }
}

/// Create UniversalDecimalTiered from StackValue decimal storage — full precision
///
/// **PURPOSE**: Bridge BinaryStorage → typed decimal representation for UGOD dispatch.
/// Decimal values are stored as (decimal_places, BinaryStorage) in the evaluator.
/// This function determines the appropriate tier and creates a properly typed value.
pub(super) fn decimal_from_storage(decimal_places: u8, storage: &BinaryStorage) -> Result<UniversalDecimalTiered, OverflowDetected> {
    use crate::fixed_point::domains::decimal_fixed::decimal_types::{DecimalRaw, tier_for_decimal_places};
    let tier = tier_for_decimal_places(decimal_places);

    match tier {
        1..=4 => {
            // UGOD: on wider profiles, the BinaryStorage (I256/I512) may exceed i128
            // even when tier_for_decimal_places suggests tiers 1-4 based on dp alone.
            // Check actual value magnitude and promote to tier 5/6 if needed.
            #[cfg(table_format = "q128_128")]
            {
                if !storage.fits_in_i128() {
                    // Value exceeds i128 — promote to tier 5 (I256)
                    return UniversalDecimalTiered::from_tier_raw(
                        5, decimal_places, DecimalRaw::Medium(*storage)
                    );
                }
            }
            #[cfg(table_format = "q256_256")]
            {
                if !storage.fits_in_i128() {
                    if storage.fits_in_i256() {
                        // Value fits in I256 — promote to tier 5
                        return UniversalDecimalTiered::from_tier_raw(
                            5, decimal_places, DecimalRaw::Medium(storage.as_i256())
                        );
                    } else {
                        // Value needs full I512 — promote to tier 6
                        return UniversalDecimalTiered::from_tier_raw(
                            6, decimal_places, DecimalRaw::Large(*storage)
                        );
                    }
                }
            }

            // Value fits in i128 — standard tier 1-4 path
            let val = binary_storage_to_i128(storage);
            for t in tier..=4 {
                match UniversalDecimalTiered::from_tier_raw(t, decimal_places, DecimalRaw::Small(val)) {
                    Ok(v) => return Ok(v),
                    Err(OverflowDetected::TierOverflow) => continue,
                    Err(e) => return Err(e),
                }
            }
            // Doesn't fit in tiers 1-4, promote to tier 5
            UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(I256::from_i128(val)))
                .or_else(|_| UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Small(val)))
        }
        5 => {
            // Tier 5: I256 backing
            #[cfg(table_format = "q256_256")]
            { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(storage.as_i256())) }

            #[cfg(table_format = "q128_128")]
            { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(*storage)) }

            #[cfg(table_format = "q64_64")]
            { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(I256::from_i128(*storage))) }

            #[cfg(table_format = "q32_32")]
            { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(I256::from_i128(*storage as i128))) }

            #[cfg(table_format = "q16_16")]
            { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(I256::from_i128(*storage as i128))) }

        }
        6 => {
            // Tier 6: I512 backing
            #[cfg(table_format = "q256_256")]
            { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(*storage)) }

            #[cfg(table_format = "q128_128")]
            { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(I512::from_i256(*storage))) }

            #[cfg(table_format = "q64_64")]
            { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(I512::from_i128(*storage))) }

            #[cfg(table_format = "q32_32")]
            { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(I512::from_i128(*storage as i128))) }

            #[cfg(table_format = "q16_16")]
            { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(I512::from_i128(*storage as i128))) }

        }
        _ => Err(OverflowDetected::InvalidInput)
    }
}

/// Convert UniversalDecimalTiered result back to (decimal_places, BinaryStorage) — full precision
pub(super) fn decimal_to_storage(decimal: &UniversalDecimalTiered) -> (u8, BinaryStorage) {
    use crate::fixed_point::domains::decimal_fixed::decimal_types::DecimalRaw;
    let (tier, raw) = decimal.to_tier_raw();
    let decimal_places = decimal.decimal_places();
    let storage = match raw {
        DecimalRaw::Small(v) => to_binary_storage(v),
        DecimalRaw::Medium(v) => {
            #[cfg(table_format = "q256_256")]
            { I512::from_i256(v) }

            #[cfg(table_format = "q128_128")]
            { v }

            #[cfg(table_format = "q64_64")]
            { v.as_i128() }

            #[cfg(table_format = "q32_32")]
            { v.as_i128() as i64 }

            #[cfg(table_format = "q16_16")]
            { v.as_i128() as i32 }

        }
        DecimalRaw::Large(v) => {
            #[cfg(table_format = "q256_256")]
            { v }

            #[cfg(table_format = "q128_128")]
            { v.as_i256() }

            #[cfg(table_format = "q64_64")]
            { v.as_i128() }

            #[cfg(table_format = "q32_32")]
            { v.as_i128() as i64 }

            #[cfg(table_format = "q16_16")]
            { v.as_i128() as i32 }

        }
        DecimalRaw::XLarge(v) => {
            #[cfg(table_format = "q256_256")]
            { v.as_i512() }

            #[cfg(table_format = "q128_128")]
            { v.as_i256() }

            #[cfg(table_format = "q64_64")]
            { v.as_i128() }

            #[cfg(table_format = "q32_32")]
            { v.as_i128() as i64 }

            #[cfg(table_format = "q16_16")]
            { v.as_i128() as i32 }

        }
    };
    let _ = tier; // tier stored implicitly via decimal_places
    (decimal_places, storage)
}

/// Create UniversalBinaryFixed from StackValue binary storage — full precision
///
/// **PURPOSE**: Bridge BinaryStorage → typed binary representation for UGOD dispatch.
/// Binary values are stored as (tier, BinaryStorage) in the evaluator.
/// This function converts to the typed tier system for proper UGOD arithmetic.
pub(super) fn binary_from_storage(tier: u8, storage: &BinaryStorage) -> Result<UniversalBinaryFixed, OverflowDetected> {
    match tier {
        1..=3 => {
            // Tiers 1-3 fit in i128
            let val = binary_storage_to_i128(storage);
            UniversalBinaryFixed::from_tier_value(tier, val)
        }
        4 => {
            // Tier 4 uses I256
            #[cfg(table_format = "q256_256")]
            { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(storage.as_i256())) }

            #[cfg(table_format = "q128_128")]
            { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(*storage)) }

            #[cfg(table_format = "q64_64")]
            { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(I256::from_i128(*storage))) }

            #[cfg(table_format = "q32_32")]
            { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(I256::from_i128(*storage as i128))) }

            #[cfg(table_format = "q16_16")]
            { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(I256::from_i128(*storage as i128))) }

        }
        5 => {
            // Tier 5 uses I512
            #[cfg(table_format = "q256_256")]
            { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(*storage)) }

            #[cfg(table_format = "q128_128")]
            { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(I512::from_i256(*storage))) }

            #[cfg(table_format = "q64_64")]
            { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(I512::from_i128(*storage))) }

            #[cfg(table_format = "q32_32")]
            { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(I512::from_i128(*storage as i128))) }

            #[cfg(table_format = "q16_16")]
            { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(I512::from_i128(*storage as i128))) }

        }
        6 => {
            // Tier 6 uses I1024
            #[cfg(table_format = "q256_256")]
            { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i512(*storage))) }

            #[cfg(table_format = "q128_128")]
            { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i256(*storage))) }

            #[cfg(table_format = "q64_64")]
            { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i128(*storage))) }

            #[cfg(table_format = "q32_32")]
            { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i128(*storage as i128))) }

            #[cfg(table_format = "q16_16")]
            { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i128(*storage as i128))) }

        }
        _ => Err(OverflowDetected::InvalidInput)
    }
}

/// Convert UniversalBinaryFixed result back to (tier, BinaryStorage) — full precision
pub(super) fn binary_to_storage(binary: &UniversalBinaryFixed) -> (u8, BinaryStorage) {
    let (tier, raw) = binary.to_tier_raw();
    match raw {
        BinaryRaw::Small(v) => (tier, to_binary_storage(v)),
        BinaryRaw::Medium(v) => {
            // I256 → BinaryStorage
            #[cfg(table_format = "q256_256")]
            { (tier, I512::from_i256(v)) }

            #[cfg(table_format = "q128_128")]
            { (tier, v) }

            #[cfg(table_format = "q64_64")]
            { (tier, v.as_i128()) }

            #[cfg(table_format = "q32_32")]
            { (tier, v.as_i128() as i64) }

            #[cfg(table_format = "q16_16")]
            { (tier, v.as_i128() as i32) }

        }
        BinaryRaw::Large(v) => {
            // I512 → BinaryStorage
            #[cfg(table_format = "q256_256")]
            { (tier, v) }

            #[cfg(table_format = "q128_128")]
            { (tier, v.as_i256()) }

            #[cfg(table_format = "q64_64")]
            { (tier, v.as_i128()) }

            #[cfg(table_format = "q32_32")]
            { (tier, v.as_i128() as i64) }

            #[cfg(table_format = "q16_16")]
            { (tier, v.as_i128() as i32) }

        }
        BinaryRaw::XLarge(v) => {
            // I1024 → BinaryStorage
            #[cfg(table_format = "q256_256")]
            { (tier, v.as_i512()) }

            #[cfg(table_format = "q128_128")]
            { (tier, v.as_i256()) }

            #[cfg(table_format = "q64_64")]
            { (tier, v.as_i128()) }

            #[cfg(table_format = "q32_32")]
            { (tier, v.as_i128() as i64) }

            #[cfg(table_format = "q16_16")]
            { (tier, v.as_i128() as i32) }

        }
    }
}

// ============================================================================
// SHADOW PROPAGATION HELPERS
// ============================================================================

/// GCD for shadow reduction (Euclidean algorithm on u128)
pub(super) fn shadow_gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Negate a shadow: -(a/b) = (-a)/b
pub(super) fn shadow_negate(s: &CompactShadow) -> CompactShadow {
    match s.as_rational() {
        Some((num, den)) => CompactShadow::from_rational(-num, den),
        None => CompactShadow::None,
    }
}

/// Add two shadows: a/b + c/d = (a*d + c*b) / (b*d), with overflow → None
pub(super) fn shadow_add(a: &CompactShadow, b: &CompactShadow) -> CompactShadow {
    let (an, ad) = match a.as_rational() {
        Some(v) => v,
        None => return CompactShadow::None,
    };
    let (bn, bd) = match b.as_rational() {
        Some(v) => v,
        None => return CompactShadow::None,
    };
    // (an * bd + bn * ad) / (ad * bd) — all checked for overflow
    let ad_128 = ad as i128;
    let bd_128 = bd as i128;
    let num = an.checked_mul(bd_128).and_then(|x| bn.checked_mul(ad_128).and_then(|y| x.checked_add(y)));
    let den = (ad as u128).checked_mul(bd as u128);
    match (num, den) {
        (Some(n), Some(d)) if d > 0 => {
            let g = shadow_gcd(n.unsigned_abs(), d);
            let rn = n / g as i128;
            let rd = d / g;
            CompactShadow::from_rational(rn, rd)
        }
        _ => CompactShadow::None,
    }
}

/// Subtract two shadows: a/b - c/d = (a*d - c*b) / (b*d), with overflow → None
pub(super) fn shadow_subtract(a: &CompactShadow, b: &CompactShadow) -> CompactShadow {
    let (an, ad) = match a.as_rational() {
        Some(v) => v,
        None => return CompactShadow::None,
    };
    let (bn, bd) = match b.as_rational() {
        Some(v) => v,
        None => return CompactShadow::None,
    };
    let ad_128 = ad as i128;
    let bd_128 = bd as i128;
    let num = an.checked_mul(bd_128).and_then(|x| bn.checked_mul(ad_128).and_then(|y| x.checked_sub(y)));
    let den = (ad as u128).checked_mul(bd as u128);
    match (num, den) {
        (Some(n), Some(d)) if d > 0 => {
            let g = shadow_gcd(n.unsigned_abs(), d);
            let rn = n / g as i128;
            let rd = d / g;
            CompactShadow::from_rational(rn, rd)
        }
        _ => CompactShadow::None,
    }
}

/// Multiply two shadows: (a/b) * (c/d) = (a*c) / (b*d), with overflow → None
pub(super) fn shadow_multiply(a: &CompactShadow, b: &CompactShadow) -> CompactShadow {
    let (an, ad) = match a.as_rational() {
        Some(v) => v,
        None => return CompactShadow::None,
    };
    let (bn, bd) = match b.as_rational() {
        Some(v) => v,
        None => return CompactShadow::None,
    };
    let num = an.checked_mul(bn);
    let den = (ad as u128).checked_mul(bd as u128);
    match (num, den) {
        (Some(n), Some(d)) if d > 0 => {
            let g = shadow_gcd(n.unsigned_abs(), d);
            let rn = n / g as i128;
            let rd = d / g;
            CompactShadow::from_rational(rn, rd)
        }
        _ => CompactShadow::None,
    }
}

/// Divide two shadows: (a/b) / (c/d) = (a*d) / (b*c), with overflow → None
pub(super) fn shadow_divide(a: &CompactShadow, b: &CompactShadow) -> CompactShadow {
    let (an, ad) = match a.as_rational() {
        Some(v) => v,
        None => return CompactShadow::None,
    };
    let (bn, bd) = match b.as_rational() {
        Some(v) => v,
        None => return CompactShadow::None,
    };
    if bn == 0 { return CompactShadow::None; } // division by zero
    // (a/ad) / (b/bd) = (a * bd) / (ad * |b|), sign from b
    let bd_128 = bd as i128;
    let num = an.checked_mul(bd_128);
    let den = (ad as u128).checked_mul(bn.unsigned_abs());
    let sign = if bn < 0 { -1i128 } else { 1i128 };
    match (num, den) {
        (Some(n), Some(d)) if d > 0 => {
            match n.checked_mul(sign) {
                Some(signed_n) => {
                    let g = shadow_gcd(signed_n.unsigned_abs(), d);
                    let rn = signed_n / g as i128;
                    let rd = d / g;
                    CompactShadow::from_rational(rn, rd)
                }
                None => CompactShadow::None,
            }
        }
        _ => CompactShadow::None,
    }
}

#[cfg(test)]
mod ternary_storage_arm_tests {
    //! The Medium/Large/XLarge arms of `ternary_to_storage` are unreachable
    //! from `0t` literals (from_str caps at Tier 3 — i64 integer parts keep
    //! raws inside i128), so they are pinned here at the unit level: small
    //! raws convert exactly on every profile, raws exceeding the profile's
    //! BinaryStorage are a loud TierOverflow, never a wrap.
    use super::*;

    fn tier4(v: i128) -> UniversalTernaryFixed {
        UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(I256::from_i128(v))).unwrap()
    }

    #[test]
    fn medium_arm_small_raw_converts_everywhere() {
        let (tier, _storage) = ternary_to_storage(&tier4(12_345)).expect("small Medium raw");
        assert_eq!(tier, 4);
        let (tier_n, _s) = ternary_to_storage(&tier4(-12_345)).expect("small negative Medium raw");
        assert_eq!(tier_n, 4);
    }

    #[test]
    fn medium_arm_oversized_raw_fails_loud_on_narrow_storage() {
        // I256 raw beyond i128: fits only I256/I512 storage.
        let big = UniversalTernaryFixed::from_tier_raw(
            4,
            TernaryRaw::Medium(I256::from_i128(i128::MAX) + I256::from_i128(i128::MAX)),
        )
        .unwrap();
        let res = ternary_to_storage(&big);
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        assert!(res.is_err(), "beyond-i128 Medium raw must be TierOverflow on narrow storage");
        #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
        assert!(res.is_ok(), "I256+ storage holds beyond-i128 Medium raws");
    }

    #[test]
    fn large_and_xlarge_arms_small_and_oversized() {
        // Small Tier-5/6 raws (fit i128) convert on every profile.
        let t5 = UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(I512::from_i128(-777)))
            .unwrap();
        assert_eq!(ternary_to_storage(&t5).expect("small Large raw").0, 5);
        let t6 =
            UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i128(777)))
                .unwrap();
        assert_eq!(ternary_to_storage(&t6).expect("small XLarge raw").0, 6);

        // Oversized: I512 raw beyond i128 — Err on <= i128 storage.
        let big5 = UniversalTernaryFixed::from_tier_raw(
            5,
            TernaryRaw::Large(I512::from_i128(1) << 200),
        )
        .unwrap();
        let res5 = ternary_to_storage(&big5);
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        assert!(res5.is_err());
        #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
        assert!(res5.is_ok());

        // I1024 raw beyond I512 — Err on EVERY profile (even scientific).
        let big6 = UniversalTernaryFixed::from_tier_raw(
            6,
            TernaryRaw::XLarge(I1024::from_i128(1) << 600),
        )
        .unwrap();
        assert!(
            ternary_to_storage(&big6).is_err(),
            "beyond-I512 XLarge raw must be TierOverflow everywhere"
        );
    }
}
