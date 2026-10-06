//! Domain marshaling and shadow arithmetic
//!
//! Bridges between StackValue representation and domain-specific UGOD types
//! (UniversalTernaryFixed, UniversalDecimalTiered, UniversalBinaryFixed).
//! Also maintains CompactShadow precision during arithmetic operations.

use super::format::FascFormat;
use crate::fixed_point::fixed_num::ProfileFormat;
use super::BinaryStorage;
#[cfg(test)]
use crate::fixed_point::i256::I256;
#[cfg(test)]
use crate::fixed_point::i512::I512;
#[cfg(test)]
use crate::fixed_point::I1024;
use crate::fixed_point::domains::balanced_ternary::ternary_types::UniversalTernaryFixed;
#[cfg(test)]
use crate::fixed_point::domains::balanced_ternary::ternary_types::TernaryRaw;
use crate::fixed_point::domains::binary_fixed::binary_types::UniversalBinaryFixed;
use crate::fixed_point::domains::decimal_fixed::decimal_types::UniversalDecimalTiered;
use crate::fixed_point::universal::tier_types::CompactShadow;
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, OverflowDetected};

pub(super) fn ternary_from_storage(tier: u8, storage: &BinaryStorage) -> Result<UniversalTernaryFixed, OverflowDetected> {
    <ProfileFormat as FascFormat>::ternary_from_storage(tier, storage)
}

/// Checked narrowing of an i128 tier raw into BinaryStorage.
///
/// Returns TierOverflow instead of wrapping: the bare `as i32`/`as i64`
/// casts this replaces silently corrupted Tier-2+ ternary values on the
/// realtime/compact profiles (wrap-defect class, fixed 0.4.33).
fn i128_to_binary_storage_checked(val: i128) -> Result<BinaryStorage, OverflowDetected> {
    <ProfileFormat as FascFormat>::i128_to_binary_storage_checked(val)
}

/// Convert UniversalTernaryFixed result back to (tier, BinaryStorage): full precision.
///
/// Fallible: a tier raw that does not fit the profile\'s BinaryStorage is a
/// TierOverflow error, never a silent wrap (e.g. `0t3281` parses to Tier 2
/// with raw 3281·3^16 ≈ 1.4e11, which cannot live in realtime\'s i32).
pub(super) fn ternary_to_storage(ternary: &UniversalTernaryFixed) -> Result<(u8, BinaryStorage), OverflowDetected> {
    <ProfileFormat as FascFormat>::ternary_to_storage(ternary)
}

/// Convert ternary StackValue fields to RationalNumber.
///
/// **PURPOSE**: Ternary stores `value * 3^frac_trits`. This converts to `value / 3^frac_trits`
/// as a RationalNumber. Used by to_rational(), to_compute_storage(), to_binary_storage(), Display.
pub(super) fn ternary_to_rational(tier: u8, value: &BinaryStorage) -> Result<RationalNumber, OverflowDetected> {
    <ProfileFormat as FascFormat>::ternary_to_rational(tier, value)
}

/// Create UniversalDecimalTiered from StackValue decimal storage: full precision
///
/// **PURPOSE**: Bridge BinaryStorage → typed decimal representation for UGOD dispatch.
/// Decimal values are stored as (decimal_places, BinaryStorage) in the evaluator.
/// This function determines the appropriate tier and creates a properly typed value.
pub(super) fn decimal_from_storage(decimal_places: u8, storage: &BinaryStorage) -> Result<UniversalDecimalTiered, OverflowDetected> {
    <ProfileFormat as FascFormat>::decimal_from_storage(decimal_places, storage)
}

/// Convert UniversalDecimalTiered result back to (decimal_places, BinaryStorage): full precision
///
/// `Err(TierOverflow)` when the scaled value does not fit storage; callers
/// fall back to the exact rational. Every arm used to narrow with a
/// truncating cast, so a decimal result beyond storage (realtime:
/// `214748.3647 + 1`) wrapped into a wrong value on every profile.
pub(super) fn decimal_to_storage(decimal: &UniversalDecimalTiered) -> Result<(u8, BinaryStorage), OverflowDetected> {
    <ProfileFormat as FascFormat>::decimal_to_storage(decimal)
}

/// Create UniversalBinaryFixed from StackValue binary storage: full precision
///
/// **PURPOSE**: Bridge BinaryStorage → typed binary representation for UGOD dispatch.
/// Binary values are stored as (tier, BinaryStorage) in the evaluator.
/// This function converts to the typed tier system for proper UGOD arithmetic.
pub(super) fn binary_from_storage(tier: u8, storage: &BinaryStorage) -> Result<UniversalBinaryFixed, OverflowDetected> {
    <ProfileFormat as FascFormat>::binary_from_storage(tier, storage)
}

/// Convert UniversalBinaryFixed result back to (tier, BinaryStorage): full precision
/// Checked narrowing of an i128 binary tier raw into BinaryStorage.
/// TierOverflow instead of the silent wrap the bare casts produced
/// (0.5.0 item 1: 9e18 + 9e18 on embedded returned 0.0: the promoted
/// Tier-4 result wrapped through as_i128; binary twin of the 0.4.33
/// ternary_to_storage fix).
fn i128_to_binary_storage_checked_b(val: i128) -> Result<BinaryStorage, OverflowDetected> {
    <ProfileFormat as FascFormat>::i128_to_binary_storage_checked_b(val)
}

/// Convert UniversalBinaryFixed back to (tier, BinaryStorage): checked.
///
/// Fallible (0.5.0 item 1): a promoted tier raw that does not fit the
/// profile's BinaryStorage is a TierOverflow, never a wrap. FASC's
/// arithmetic arms catch that error and fall back to the exact rational
/// path: the true top of the UGOD ladder.
pub(super) fn binary_to_storage(binary: &UniversalBinaryFixed) -> Result<(u8, BinaryStorage), OverflowDetected> {
    <ProfileFormat as FascFormat>::binary_to_storage(binary)
}

/// Euclid's gcd for shadow normalization.
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
    //! from `0t` literals (from_str caps at Tier 3: i64 integer parts keep
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
