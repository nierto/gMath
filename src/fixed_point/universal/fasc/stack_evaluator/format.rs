//! The evaluator's format-dependent pieces as a trait: the decimal digit
//! budgets, the tier-raw widenings and narrowings, the decimal and symbolic
//! conversions to the compute tier, the decimal formatter and the rational
//! roundings, one implementation per format. The bodies are the profile arms
//! of 0.6.7, transplanted; the evaluator reaches them through the profile's
//! format until it is generic itself.

#![allow(clippy::all, unused_braces)]

use crate::deployment_profiles::DeploymentProfile;
use crate::fixed_point::domains::balanced_ternary::ternary_types::{TernaryRaw, UniversalTernaryFixed};
use crate::fixed_point::domains::binary_fixed::binary_types::{BinaryRaw, UniversalBinaryFixed};
use crate::fixed_point::domains::decimal_fixed::decimal_types::UniversalDecimalTiered;
use crate::fixed_point::domains::symbolic::rational::rational_number::{OverflowDetected, RationalNumber};
use crate::fixed_point::fixed_num::{Compute, Fixed, FixedNum};
#[allow(unused_imports)]
use crate::fixed_point::{I1024, I2048, I256, I512};

/// Every format's evaluator-facing constants and raw-word conversions.
pub(crate) trait FascFormat: FixedNum {
    /// Decimal places beyond which a decimal literal promotes to the next tier.
    const DECIMAL_DP_PROMOTION_THRESHOLD: u16;
    /// Decimal places of the decimal compute tier.
    const DECIMAL_COMPUTE_DP: u8;
    /// Decimal places the storage word holds for the decimal domain.
    const DECIMAL_STORAGE_MAX_DP: u8;
    /// Digits `Display` prints without a precision.
    const DEFAULT_DIGITS: usize;
    /// Fractional digits a decimal literal may carry before it leaves the decimal domain.
    const PARSE_MAX_FRAC: usize;
    /// Faithfully representable decimal digits of the binary format.
    const MAX_DECIMAL_DIGITS: usize;
    /// The top of the binary UGOD ladder for this format.
    const MAX_BINARY_TIER: u8;
    /// The top of the ternary UGOD ladder for this format.
    const MAX_TERNARY_TIER: u8;
    /// The deployment profile this format was.
    const DEPLOYMENT_PROFILE: DeploymentProfile;

    fn decimal_to_binary_storage(dp: u8, scaled: Self::Storage) -> Result<Self::Storage, OverflowDetected>;
    fn decimal_compute_to_binary_storage(val: Compute<Self>) -> Result<Self::Storage, OverflowDetected>;
    fn ternary_from_storage(tier: u8, storage: &Self::Storage) -> Result<UniversalTernaryFixed, OverflowDetected>;
    fn i128_to_binary_storage_checked(val: i128) -> Result<Self::Storage, OverflowDetected>;
    fn ternary_to_storage(ternary: &UniversalTernaryFixed) -> Result<(u8, Self::Storage), OverflowDetected>;
    fn ternary_to_rational(tier: u8, value: &Self::Storage) -> Result<RationalNumber, OverflowDetected>;
    fn decimal_from_storage(decimal_places: u8, storage: &Self::Storage) -> Result<UniversalDecimalTiered, OverflowDetected>;
    fn decimal_to_storage(decimal: &UniversalDecimalTiered) -> Result<(u8, Self::Storage), OverflowDetected>;
    fn binary_from_storage(tier: u8, storage: &Self::Storage) -> Result<UniversalBinaryFixed, OverflowDetected>;
    fn i128_to_binary_storage_checked_b(val: i128) -> Result<Self::Storage, OverflowDetected>;
    fn binary_to_storage(binary: &UniversalBinaryFixed) -> Result<(u8, Self::Storage), OverflowDetected>;
    fn try_to_binary_storage(val: i128) -> Result<Self::Storage, OverflowDetected>;
    fn binary_storage_to_i128(val: &Self::Storage) -> i128;
    fn binary_storage_to_decimal_string(val: Self::Storage, max_digits: usize) -> String;
    fn decimal_storage_to_string(dec: u8, val: &Self::Storage) -> String;
    fn decimal_to_compute_storage(decimals: u8, scaled: Self::Storage) -> Result<Compute<Self>, OverflowDetected>;
    fn pow10_compute(exp: u8) -> Result<Compute<Self>, OverflowDetected>;
    fn symbolic_to_compute_storage(num: i128, den: i128) -> Result<Compute<Self>, OverflowDetected>;
    fn symbolic_wide_to_compute_storage(
    rational: &crate::fixed_point::domains::symbolic::rational::RationalNumber,
    ) -> Result<Compute<Self>, OverflowDetected>;
    fn compute_checked_negate(a: Compute<Self>) -> Result<Compute<Self>, OverflowDetected>;
    fn compute_is_zero(a: &Compute<Self>) -> bool;
    fn compute_is_negative(a: &Compute<Self>) -> bool;
    fn rational_i128_to_storage_checked(num: i128, den: i128) -> Result<Self::Storage, OverflowDetected>;
    fn rational_to_decimal_components(rational: &RationalNumber) -> Result<(u8, i128), OverflowDetected>;
}

impl<const F: u32> FascFormat for Fixed<i32, F> {
    const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = Self::MAX_DECIMAL_DIGITS as u16;
    const DECIMAL_COMPUTE_DP: u8 = 9;
    const DECIMAL_STORAGE_MAX_DP: u8 = 4;
    const DEFAULT_DIGITS: usize = 5;
    const PARSE_MAX_FRAC: usize = 4;
    const MAX_DECIMAL_DIGITS: usize = ((F as u64 * 30103) / 100000) as usize;
    const MAX_BINARY_TIER: u8 = 1;
    const MAX_TERNARY_TIER: u8 = 1;
    const DEPLOYMENT_PROFILE: DeploymentProfile = DeploymentProfile::Realtime;
    fn decimal_to_binary_storage(dp: u8, scaled: i32) -> Result<i32, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // 0.5.0 rounding unification: the RESULT domain is binary, so this
        // coercion rounds nearest, ties toward +∞ on every profile (was:
        // truncation on four arms, add-half on q16_16 — per-profile drift).
        {
            let frac_bits = F as usize;
            let ten_pow = super::formatting::pow10_i256(dp);
            let num = I256::from_i128(scaled as i128) << frac_bits;
            let (mut q, rem) =
                crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(num, ten_pow);
            let rem_abs = if rem.is_negative() { -rem } else { rem };
            let positive = !num.is_negative();
            let rem2 = rem_abs + rem_abs;
            if if positive { rem2 >= ten_pow } else { rem2 > ten_pow } {
                q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
            }
            if !q.fits_in_i128() { return Err(overflow); }
            Self::try_to_binary_storage(q.as_i128())
        }
    }
    fn decimal_compute_to_binary_storage(val: i64) -> Result<i32, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // Nearest, ties toward +infinity (the binary rule): before 0.6.4 this
        // divided with truncation toward zero, up to one ulp off.
        // `bump` = whether the truncated quotient moves one unit away from zero.
        #[inline(always)]
        fn bump(positive: bool, at_least_half: bool, above_half: bool) -> bool {
            if positive { at_least_half } else { above_half }
        }

        {
            // val is i64 at dp=9. Target: Q16.16 i32.
            let num = (val as i128) << (F as usize);
            let mut den: i128 = 1;
            for _ in 0..Self::DECIMAL_COMPUTE_DP { den *= 10; }
            let (mut q, rem) = (num / den, num % den);
            if bump(num >= 0, 2 * rem.abs() >= den, 2 * rem.abs() > den) {
                q = if num >= 0 { q + 1 } else { q - 1 };
            }
            i32::try_from(q).map_err(|_| overflow)
        }
    }
    fn ternary_from_storage(tier: u8, storage: &i32) -> Result<UniversalTernaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalTernaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256

                { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(I256::from_i128(*storage as i128))) }

            }
            5 => {
                // Tier 5 uses I512

                { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(I512::from_i128(*storage as i128))) }

            }
            6 => {
                // Tier 6 uses I1024

                { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i128(*storage as i128))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked(val: i128) -> Result<i32, OverflowDetected> {

        { i32::try_from(val).map_err(|_| OverflowDetected::TierOverflow) }
    }
    fn ternary_to_storage(ternary: &UniversalTernaryFixed) -> Result<(u8, i32), OverflowDetected> {
        let (tier, raw) = ternary.to_tier_raw();
        let storage = match raw {
            TernaryRaw::Small(v) => Self::i128_to_binary_storage_checked(v)?,
            TernaryRaw::Medium(v) => {
                // I256 → i32

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked(v.as_i128())?
                }
            }
            TernaryRaw::Large(v) => {
                // I512 → i32

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked(v.as_i128())?
                }
            }
            TernaryRaw::XLarge(v) => {
                // I1024 → i32

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked(v.as_i128())?
                }
            }
        };
        Ok((tier, storage))
    }
    fn ternary_to_rational(tier: u8, value: &i32) -> Result<RationalNumber, OverflowDetected> {
        let ternary = Self::ternary_from_storage(tier, value)?;
        let (ternary_tier, _raw) = ternary.to_tier_raw();

        let frac_trits: u32 = match ternary_tier {
            1 => 10,   // TQ10.10
            2 => 20,   // TQ20.20
            3 => 40,   // TQ40.40
            4 => 80,   // TQ80.80
            5 => 160,  // TQ160.160
            6 => 320,  // TQ320.320
            _ => 40,
        };

        // Tiers 1-3 stay on the i128 fast path: 3^40 = 1.2e19 fits i128.
        if frac_trits <= 40 {
            let mut denom = 1i128;
            for _ in 0..frac_trits {
                denom *= 3;
            }
            let val_i128 = Self::binary_storage_to_i128(value);
            return Ok(RationalNumber::new(val_i128, denom as u128));
        }

        {
            let val_i128 = *value as i128;
            let mut denom = 1i128;
            for _ in 0..frac_trits {
                denom *= 3;
            }
            return Ok(RationalNumber::new(val_i128, denom as u128));
        }
    }
    fn decimal_from_storage(decimal_places: u8, storage: &i32) -> Result<UniversalDecimalTiered, OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::{DecimalRaw, tier_for_decimal_places};
        let tier = tier_for_decimal_places(decimal_places);

        match tier {
            1..=4 => {
                // UGOD: on wider profiles, the i32 (I256/I512) may exceed i128
                // even when tier_for_decimal_places suggests tiers 1-4 based on dp alone.
                // Check actual value magnitude and promote to tier 5/6 if needed.

                // Value fits in i128 — standard tier 1-4 path
                let val = Self::binary_storage_to_i128(storage);
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

                { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(I256::from_i128(*storage as i128))) }

            }
            6 => {
                // Tier 6: I512 backing

                { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(I512::from_i128(*storage as i128))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn decimal_to_storage(decimal: &UniversalDecimalTiered) -> Result<(u8, i32), OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::DecimalRaw;
        let (_tier, raw) = decimal.to_tier_raw();
        let decimal_places = decimal.decimal_places();
        let overflow = OverflowDetected::TierOverflow;
        let storage = match raw {
            DecimalRaw::Small(v) => Self::try_to_binary_storage(v)?,
            DecimalRaw::Medium(v) => {
                { if v.fits_in_i128() { Self::try_to_binary_storage(v.as_i128())? } else { return Err(overflow) } }
            }
            DecimalRaw::Large(v) => {
                { if v.fits_in_i128() { Self::try_to_binary_storage(v.as_i128())? } else { return Err(overflow) } }
            }
            DecimalRaw::XLarge(v) => {
                { if v.fits_in_i128() { Self::try_to_binary_storage(v.as_i128())? } else { return Err(overflow) } }
            }
        };
        Ok((decimal_places, storage))
    }
    fn binary_from_storage(tier: u8, storage: &i32) -> Result<UniversalBinaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalBinaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256

                { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(I256::from_i128(*storage as i128))) }

            }
            5 => {
                // Tier 5 uses I512

                { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(I512::from_i128(*storage as i128))) }

            }
            6 => {
                // Tier 6 uses I1024

                { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i128(*storage as i128))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked_b(val: i128) -> Result<i32, OverflowDetected> {
        { i32::try_from(val).map_err(|_| OverflowDetected::TierOverflow) }
    }
    fn binary_to_storage(binary: &UniversalBinaryFixed) -> Result<(u8, i32), OverflowDetected> {
        let (tier, raw) = binary.to_tier_raw();
        let storage = match raw {
            BinaryRaw::Small(v) => Self::i128_to_binary_storage_checked_b(v)?,
            BinaryRaw::Medium(v) => {
                // I256 → i32

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked_b(v.as_i128())?
                }
            }
            BinaryRaw::Large(v) => {
                // I512 → i32

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked_b(v.as_i128())?
                }
            }
            BinaryRaw::XLarge(v) => {
                // I1024 → i32

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked_b(v.as_i128())?
                }
            }
        };
        Ok((tier, storage))
    }
    #[inline(always)]
    fn try_to_binary_storage(val: i128) -> Result<i32, OverflowDetected> {

        { i32::try_from(val).map_err(|_| OverflowDetected::TierOverflow) }
    }
    #[inline(always)]
    fn binary_storage_to_i128(val: &i32) -> i128 {

        { *val as i128 }

    }
    fn binary_storage_to_decimal_string(val: i32, max_digits: usize) -> String {

        {
            // i32 Q-format with configurable FRAC_BITS
            let fb = F;
            let is_negative = val < 0;
            let abs_val = if is_negative { -val } else { val };
            let int_part = (abs_val >> fb) as u32;
            let frac_part = abs_val as u32 & ((1u32 << fb) - 1);

            let digits = max_digits.min(Self::MAX_DECIMAL_DIGITS);
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

    }
    fn decimal_storage_to_string(dec: u8, val: &i32) -> String {
        let dp = dec as usize;

        {
            // Q16.16: i32 = i32. Promote to i64 for division so
            // we never overflow on 10^dec.
            let v = *val;
            let is_negative = v < 0;
            let abs_val = if is_negative { (v as i64).wrapping_neg() } else { v as i64 };
            let divisor = super::formatting::pow10_i64(dec);
            let integer_part = abs_val / divisor;
            let fractional_part = abs_val % divisor;
            let sign = if is_negative { "-" } else { "" };
            if dp == 0 {
                format!("{}{}", sign, integer_part)
            } else {
                format!("{}{}.{:0>width$}", sign, integer_part, fractional_part, width = dp)
            }
        }

    }
    #[inline]
    fn decimal_to_compute_storage(decimals: u8, scaled: i32) -> Result<i64, OverflowDetected> {
        if decimals == 0 {
            // Integer value — shift directly into compute Q-format
            {
                // checked: at 24 fraction bits the compute tier holds +-2^15
                let v = (scaled as i128) << (2 * F);
                return i64::try_from(v).map_err(|_| OverflowDetected::TierOverflow);
            }
        }

        let den = Self::pow10_compute(decimals)?;

        {
            // i128: scaled << 2F can pass i64 at high fraction bits (it wrapped:
            // "100.1234" at 24 fraction bits is 1001234 * 2^48; before 0.6.4)
            let num = (scaled as i128) << (2 * F);
            i64::try_from(num / den as i128).map_err(|_| OverflowDetected::TierOverflow)
        }
    }
    #[inline]
    fn pow10_compute(exp: u8) -> Result<i64, OverflowDetected> {
        {
            // i64 = i64. i64 safe max ~10^18.
            const TABLE: [i64; 19] = super::compute::pow10_table_i64();
            if (exp as usize) < TABLE.len() {
                Ok(TABLE[exp as usize])
            } else {
                Err(OverflowDetected::Overflow)
            }
        }
    }
    #[inline]
    fn symbolic_to_compute_storage(num: i128, den: i128) -> Result<i64, OverflowDetected> {
        {
            // i64 = i64. I256 intermediate: num << 2F can pass i128.
            // Range-checked: the bare `as i64` wrapped a symbolic value beyond
            // the realtime compute range (atan(999999999) at 24 fraction bits
            // came out as -pi/2; before 0.6.4).
            let n = I256::from_i128(num) << ((2 * F) as usize);
            let q = n / I256::from_i128(den);
            if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
            i64::try_from(q.as_i128()).map_err(|_| OverflowDetected::TierOverflow)
        }
    }
    fn symbolic_wide_to_compute_storage(
        rational: &crate::fixed_point::domains::symbolic::rational::RationalNumber,
    ) -> Result<i64, OverflowDetected> {
        let parts = rational.extract_native();

        {
            // i64 = i64. Use I256 intermediate for shift+divide → i64.
            if let Some((num_i256, den_i256)) = parts.try_as_i256_pair() {
                let q = (I512::from_i256(num_i256) << ((2 * F) as usize)) / I512::from_i256(den_i256);
                if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                return i64::try_from(q.as_i128()).map_err(|_| OverflowDetected::TierOverflow);
            }
            if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
                let n = I1024::from_i512(num_i512) << ((2 * F) as usize);
                let d = I1024::from_i512(den_i512);
                let q = n / d;
                if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                return i64::try_from(q.as_i128()).map_err(|_| OverflowDetected::TierOverflow);
            }
        }

        Err(OverflowDetected::TierOverflow)
    }
    fn compute_checked_negate(a: i64) -> Result<i64, OverflowDetected> {
        { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn compute_is_zero(a: &i64) -> bool {
        { *a == 0i64 }
    }
    #[inline]
    fn compute_is_negative(a: &i64) -> bool {
        { *a < 0i64 }
    }
    fn rational_i128_to_storage_checked(num: i128, den: i128) -> Result<i32, OverflowDetected> {
        if den == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }
        {
            let fb: usize = F as usize;
            let n = I256::from_i128(num) << fb;
            let d = I256::from_i128(den);
            let (mut q, r) = crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(n, d);
            let r_abs = if r.is_negative() { -r } else { r };
            let d_abs = if d.is_negative() { -d } else { d };
            let positive = n.is_negative() == d.is_negative();
            let r2 = r_abs + r_abs;
            if if positive { r2 >= d_abs } else { r2 > d_abs } {
                q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
            }
            if !q.fits_in_i128() {
                return Err(OverflowDetected::TierOverflow);
            }
            let q128 = q.as_i128();
            {
                i32::try_from(q128).map_err(|_| OverflowDetected::TierOverflow)
            }
        }
    }
    fn rational_to_decimal_components(rational: &RationalNumber) -> Result<(u8, i128), OverflowDetected> {
        // Try i128 extraction (works for all profiles ≤ Q128.128 and reduced rationals)
        if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
            return super::transcendentals::rational_to_decimal_components_i128(num, den);
        }

        // Q256.256 fallback: extract as I512 for Ultra-tier rationals

        Err(OverflowDetected::Overflow)
    }
}

impl FascFormat for Fixed<i64, 32> {
    const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 9;
    const DECIMAL_COMPUTE_DP: u8 = 19;
    const DECIMAL_STORAGE_MAX_DP: u8 = 9;
    const DEFAULT_DIGITS: usize = 9;
    const PARSE_MAX_FRAC: usize = 9;
    const MAX_DECIMAL_DIGITS: usize = 9;
    const MAX_BINARY_TIER: u8 = 2;
    const MAX_TERNARY_TIER: u8 = 2;
    const DEPLOYMENT_PROFILE: DeploymentProfile = DeploymentProfile::Compact;
    fn decimal_to_binary_storage(dp: u8, scaled: i64) -> Result<i64, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // 0.5.0 rounding unification: the RESULT domain is binary, so this
        // coercion rounds nearest, ties toward +∞ on every profile (was:
        // truncation on four arms, add-half on q16_16 — per-profile drift).
        {
            let frac_bits = 32usize;
            let ten_pow = super::formatting::pow10_i256(dp);
            let num = I256::from_i128(scaled as i128) << frac_bits;
            let (mut q, rem) =
                crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(num, ten_pow);
            let rem_abs = if rem.is_negative() { -rem } else { rem };
            let positive = !num.is_negative();
            let rem2 = rem_abs + rem_abs;
            if if positive { rem2 >= ten_pow } else { rem2 > ten_pow } {
                q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
            }
            if !q.fits_in_i128() { return Err(overflow); }
            Self::try_to_binary_storage(q.as_i128())
        }
    }
    fn decimal_compute_to_binary_storage(val: i128) -> Result<i64, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // Nearest, ties toward +infinity (the binary rule): before 0.6.4 this
        // divided with truncation toward zero, up to one ulp off.
        // `bump` = whether the truncated quotient moves one unit away from zero.
        #[inline(always)]
        fn bump(positive: bool, at_least_half: bool, above_half: bool) -> bool {
            if positive { at_least_half } else { above_half }
        }

        {
            // val is i128 at dp=19. Target: Q32.32 i64.
            // result = val * 2^32 / 10^19 via I256 intermediate
            let num = I256::from_i128(val) << 32usize;
            let mut den = I256::from_i128(1);
            let ten = I256::from_i128(10);
            for _ in 0..Self::DECIMAL_COMPUTE_DP { den = den * ten; }
            let (mut q, rem) = crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(num, den);
            let rem_abs = if rem.is_negative() { -rem } else { rem };
            let positive = !num.is_negative();
            if bump(positive, rem_abs + rem_abs >= den, rem_abs + rem_abs > den) {
                q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
            }
            if !q.fits_in_i128() { return Err(overflow); }
            i64::try_from(q.as_i128()).map_err(|_| overflow)
        }
    }
    fn ternary_from_storage(tier: u8, storage: &i64) -> Result<UniversalTernaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalTernaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256

                { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(I256::from_i128(*storage as i128))) }

            }
            5 => {
                // Tier 5 uses I512

                { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(I512::from_i128(*storage as i128))) }

            }
            6 => {
                // Tier 6 uses I1024

                { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i128(*storage as i128))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked(val: i128) -> Result<i64, OverflowDetected> {

        { i64::try_from(val).map_err(|_| OverflowDetected::TierOverflow) }

    }
    fn ternary_to_storage(ternary: &UniversalTernaryFixed) -> Result<(u8, i64), OverflowDetected> {
        let (tier, raw) = ternary.to_tier_raw();
        let storage = match raw {
            TernaryRaw::Small(v) => Self::i128_to_binary_storage_checked(v)?,
            TernaryRaw::Medium(v) => {
                // I256 → i64

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked(v.as_i128())?
                }
            }
            TernaryRaw::Large(v) => {
                // I512 → i64

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked(v.as_i128())?
                }
            }
            TernaryRaw::XLarge(v) => {
                // I1024 → i64

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked(v.as_i128())?
                }
            }
        };
        Ok((tier, storage))
    }
    fn ternary_to_rational(tier: u8, value: &i64) -> Result<RationalNumber, OverflowDetected> {
        let ternary = Self::ternary_from_storage(tier, value)?;
        let (ternary_tier, _raw) = ternary.to_tier_raw();

        let frac_trits: u32 = match ternary_tier {
            1 => 10,   // TQ10.10
            2 => 20,   // TQ20.20
            3 => 40,   // TQ40.40
            4 => 80,   // TQ80.80
            5 => 160,  // TQ160.160
            6 => 320,  // TQ320.320
            _ => 40,
        };

        // Tiers 1-3 stay on the i128 fast path: 3^40 = 1.2e19 fits i128.
        if frac_trits <= 40 {
            let mut denom = 1i128;
            for _ in 0..frac_trits {
                denom *= 3;
            }
            let val_i128 = Self::binary_storage_to_i128(value);
            return Ok(RationalNumber::new(val_i128, denom as u128));
        }

        {
            let val_i128 = *value as i128;
            let mut denom = 1i128;
            for _ in 0..frac_trits {
                denom *= 3;
            }
            return Ok(RationalNumber::new(val_i128, denom as u128));
        }

    }
    fn decimal_from_storage(decimal_places: u8, storage: &i64) -> Result<UniversalDecimalTiered, OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::{DecimalRaw, tier_for_decimal_places};
        let tier = tier_for_decimal_places(decimal_places);

        match tier {
            1..=4 => {
                // UGOD: on wider profiles, the i64 (I256/I512) may exceed i128
                // even when tier_for_decimal_places suggests tiers 1-4 based on dp alone.
                // Check actual value magnitude and promote to tier 5/6 if needed.

                // Value fits in i128 — standard tier 1-4 path
                let val = Self::binary_storage_to_i128(storage);
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

                { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(I256::from_i128(*storage as i128))) }

            }
            6 => {
                // Tier 6: I512 backing

                { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(I512::from_i128(*storage as i128))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn decimal_to_storage(decimal: &UniversalDecimalTiered) -> Result<(u8, i64), OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::DecimalRaw;
        let (_tier, raw) = decimal.to_tier_raw();
        let decimal_places = decimal.decimal_places();
        let overflow = OverflowDetected::TierOverflow;
        let storage = match raw {
            DecimalRaw::Small(v) => Self::try_to_binary_storage(v)?,
            DecimalRaw::Medium(v) => {
                { if v.fits_in_i128() { Self::try_to_binary_storage(v.as_i128())? } else { return Err(overflow) } }
            }
            DecimalRaw::Large(v) => {
                { if v.fits_in_i128() { Self::try_to_binary_storage(v.as_i128())? } else { return Err(overflow) } }
            }
            DecimalRaw::XLarge(v) => {
                { if v.fits_in_i128() { Self::try_to_binary_storage(v.as_i128())? } else { return Err(overflow) } }
            }
        };
        Ok((decimal_places, storage))
    }
    fn binary_from_storage(tier: u8, storage: &i64) -> Result<UniversalBinaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalBinaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256

                { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(I256::from_i128(*storage as i128))) }

            }
            5 => {
                // Tier 5 uses I512

                { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(I512::from_i128(*storage as i128))) }

            }
            6 => {
                // Tier 6 uses I1024

                { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i128(*storage as i128))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked_b(val: i128) -> Result<i64, OverflowDetected> {
        { i64::try_from(val).map_err(|_| OverflowDetected::TierOverflow) }
    }
    fn binary_to_storage(binary: &UniversalBinaryFixed) -> Result<(u8, i64), OverflowDetected> {
        let (tier, raw) = binary.to_tier_raw();
        let storage = match raw {
            BinaryRaw::Small(v) => Self::i128_to_binary_storage_checked_b(v)?,
            BinaryRaw::Medium(v) => {
                // I256 → i64

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked_b(v.as_i128())?
                }
            }
            BinaryRaw::Large(v) => {
                // I512 → i64

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked_b(v.as_i128())?
                }
            }
            BinaryRaw::XLarge(v) => {
                // I1024 → i64

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked_b(v.as_i128())?
                }
            }
        };
        Ok((tier, storage))
    }
    #[inline(always)]
    fn try_to_binary_storage(val: i128) -> Result<i64, OverflowDetected> {

        { i64::try_from(val).map_err(|_| OverflowDetected::TierOverflow) }

    }
    #[inline(always)]
    fn binary_storage_to_i128(val: &i64) -> i128 {

        { *val as i128 }

    }
    fn binary_storage_to_decimal_string(val: i64, max_digits: usize) -> String {

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

    }
    fn decimal_storage_to_string(dec: u8, val: &i64) -> String {
        let dp = dec as usize;

        {
            // Q32.32: i64 = i64. Promote to i128 for division so
            // we never overflow on 10^dec.
            let v = *val;
            let is_negative = v < 0;
            let abs_val = if is_negative { (v as i128).wrapping_neg() } else { v as i128 };
            let divisor = super::formatting::pow10_i128(dec);
            let integer_part = abs_val / divisor;
            let fractional_part = abs_val % divisor;
            let sign = if is_negative { "-" } else { "" };
            if dp == 0 {
                format!("{}{}", sign, integer_part)
            } else {
                format!("{}{}.{:0>width$}", sign, integer_part, fractional_part, width = dp)
            }
        }

    }
    #[inline]
    fn decimal_to_compute_storage(decimals: u8, scaled: i64) -> Result<i128, OverflowDetected> {
        if decimals == 0 {
            // Integer value — shift directly into compute Q-format
            { return Ok((scaled as i128) << 64); }
        }

        let den = Self::pow10_compute(decimals)?;

        {
            // i64=i64 → promote to i128, shift by 64 (Q64.64 compute format)
            let num = (scaled as i128) << 64;
            Ok(num / den)
        }
    }
    #[inline]
    fn pow10_compute(exp: u8) -> Result<i128, OverflowDetected> {
        {
            // i128 = i128. i128 safe max ~10^38.
            const TABLE: [i128; 39] = super::compute::pow10_table_i128();
            if (exp as usize) < TABLE.len() {
                Ok(TABLE[exp as usize])
            } else {
                Err(OverflowDetected::Overflow)
            }
        }
    }
    #[inline]
    fn symbolic_to_compute_storage(num: i128, den: i128) -> Result<i128, OverflowDetected> {
        {
            // i128 = i128 (Q64.64). Use I256 intermediate for (num << 64) / den.
            let n = I256::from_i128(num) << 64;
            let d = I256::from_i128(den);
            let q = n / d;
            if !q.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
            Ok(q.as_i128())
        }
    }
    fn symbolic_wide_to_compute_storage(
        rational: &crate::fixed_point::domains::symbolic::rational::RationalNumber,
    ) -> Result<i128, OverflowDetected> {
        let parts = rational.extract_native();

        {
            // i128 = i128 (Q64.64). I256 intermediate for shift+divide.
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

        Err(OverflowDetected::TierOverflow)
    }
    fn compute_checked_negate(a: i128) -> Result<i128, OverflowDetected> {
        { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn compute_is_zero(a: &i128) -> bool {
        { *a == 0i128 }
    }
    #[inline]
    fn compute_is_negative(a: &i128) -> bool {
        { *a < 0i128 }
    }
    fn rational_i128_to_storage_checked(num: i128, den: i128) -> Result<i64, OverflowDetected> {
        if den == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }
        {
            let fb: usize = 32;
            let n = I256::from_i128(num) << fb;
            let d = I256::from_i128(den);
            let (mut q, r) = crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(n, d);
            let r_abs = if r.is_negative() { -r } else { r };
            let d_abs = if d.is_negative() { -d } else { d };
            let positive = n.is_negative() == d.is_negative();
            let r2 = r_abs + r_abs;
            if if positive { r2 >= d_abs } else { r2 > d_abs } {
                q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
            }
            if !q.fits_in_i128() {
                return Err(OverflowDetected::TierOverflow);
            }
            let q128 = q.as_i128();
            {
                i64::try_from(q128).map_err(|_| OverflowDetected::TierOverflow)
            }
        }
    }
    fn rational_to_decimal_components(rational: &RationalNumber) -> Result<(u8, i128), OverflowDetected> {
        // Try i128 extraction (works for all profiles ≤ Q128.128 and reduced rationals)
        if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
            return super::transcendentals::rational_to_decimal_components_i128(num, den);
        }

        // Q256.256 fallback: extract as I512 for Ultra-tier rationals

        Err(OverflowDetected::Overflow)
    }
}

impl FascFormat for Fixed<i128, 64> {
    const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 18;
    const DECIMAL_COMPUTE_DP: u8 = 38;
    const DECIMAL_STORAGE_MAX_DP: u8 = 19;
    const DEFAULT_DIGITS: usize = 19;
    const PARSE_MAX_FRAC: usize = 38;
    const MAX_DECIMAL_DIGITS: usize = 19;
    const MAX_BINARY_TIER: u8 = 3;
    const MAX_TERNARY_TIER: u8 = 3;
    const DEPLOYMENT_PROFILE: DeploymentProfile = DeploymentProfile::Embedded;
    fn decimal_to_binary_storage(dp: u8, scaled: i128) -> Result<i128, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // 0.5.0 rounding unification: the RESULT domain is binary, so this
        // coercion rounds nearest, ties toward +∞ on every profile (was:
        // truncation on four arms, add-half on q16_16 — per-profile drift).
        {
            let frac_bits = 64usize;
            let ten_pow = super::formatting::pow10_i256(dp);
            let num = I256::from_i128(scaled as i128) << frac_bits;
            let (mut q, rem) =
                crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(num, ten_pow);
            let rem_abs = if rem.is_negative() { -rem } else { rem };
            let positive = !num.is_negative();
            let rem2 = rem_abs + rem_abs;
            if if positive { rem2 >= ten_pow } else { rem2 > ten_pow } {
                q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
            }
            if !q.fits_in_i128() { return Err(overflow); }
            Self::try_to_binary_storage(q.as_i128())
        }
    }
    fn decimal_compute_to_binary_storage(val: I256) -> Result<i128, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // Nearest, ties toward +infinity (the binary rule): before 0.6.4 this
        // divided with truncation toward zero, up to one ulp off.
        // `bump` = whether the truncated quotient moves one unit away from zero.
        #[inline(always)]
        fn bump(positive: bool, at_least_half: bool, above_half: bool) -> bool {
            if positive { at_least_half } else { above_half }
        }

        {
            // val is I256 at decimal compute dp=38. Target: Q64.64 i128.
            // result = val * 2^64 / 10^38 via I512 intermediate
            let num = I512::from_i256(val) << 64usize;
            let mut den = I512::from_i128(1);
            let ten = I512::from_i128(10);
            for _ in 0..Self::DECIMAL_COMPUTE_DP { den = den * ten; }
            let (mut q, rem) = crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(num, den);
            let rem_abs = if rem.is_negative() { -rem } else { rem };
            let positive = !num.is_negative();
            if bump(positive, rem_abs + rem_abs >= den, rem_abs + rem_abs > den) {
                q = if positive { q + I512::from_i128(1) } else { q - I512::from_i128(1) };
            }
            if !q.fits_in_i128() { return Err(overflow); }
            Ok(q.as_i128())
        }
    }
    fn ternary_from_storage(tier: u8, storage: &i128) -> Result<UniversalTernaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalTernaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256

                { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(I256::from_i128(*storage))) }

            }
            5 => {
                // Tier 5 uses I512

                { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(I512::from_i128(*storage))) }

            }
            6 => {
                // Tier 6 uses I1024

                { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i128(*storage))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked(val: i128) -> Result<i128, OverflowDetected> {

        { Ok(val) }

    }
    fn ternary_to_storage(ternary: &UniversalTernaryFixed) -> Result<(u8, i128), OverflowDetected> {
        let (tier, raw) = ternary.to_tier_raw();
        let storage = match raw {
            TernaryRaw::Small(v) => Self::i128_to_binary_storage_checked(v)?,
            TernaryRaw::Medium(v) => {
                // I256 → i128

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked(v.as_i128())?
                }
            }
            TernaryRaw::Large(v) => {
                // I512 → i128

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked(v.as_i128())?
                }
            }
            TernaryRaw::XLarge(v) => {
                // I1024 → i128

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked(v.as_i128())?
                }
            }
        };
        Ok((tier, storage))
    }
    fn ternary_to_rational(tier: u8, value: &i128) -> Result<RationalNumber, OverflowDetected> {
        let ternary = Self::ternary_from_storage(tier, value)?;
        let (ternary_tier, _raw) = ternary.to_tier_raw();

        let frac_trits: u32 = match ternary_tier {
            1 => 10,   // TQ10.10
            2 => 20,   // TQ20.20
            3 => 40,   // TQ40.40
            4 => 80,   // TQ80.80
            5 => 160,  // TQ160.160
            6 => 320,  // TQ320.320
            _ => 40,
        };

        // Tiers 1-3 stay on the i128 fast path: 3^40 = 1.2e19 fits i128.
        if frac_trits <= 40 {
            let mut denom = 1i128;
            for _ in 0..frac_trits {
                denom *= 3;
            }
            let val_i128 = Self::binary_storage_to_i128(value);
            return Ok(RationalNumber::new(val_i128, denom as u128));
        }

        {
            let val_i128 = *value;
            let mut denom = 1i128;
            for _ in 0..frac_trits {
                denom *= 3;
            }
            return Ok(RationalNumber::new(val_i128, denom as u128));
        }

    }
    fn decimal_from_storage(decimal_places: u8, storage: &i128) -> Result<UniversalDecimalTiered, OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::{DecimalRaw, tier_for_decimal_places};
        let tier = tier_for_decimal_places(decimal_places);

        match tier {
            1..=4 => {
                // UGOD: on wider profiles, the i128 (I256/I512) may exceed i128
                // even when tier_for_decimal_places suggests tiers 1-4 based on dp alone.
                // Check actual value magnitude and promote to tier 5/6 if needed.

                // Value fits in i128 — standard tier 1-4 path
                let val = Self::binary_storage_to_i128(storage);
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

                { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(I256::from_i128(*storage))) }

            }
            6 => {
                // Tier 6: I512 backing

                { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(I512::from_i128(*storage))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn decimal_to_storage(decimal: &UniversalDecimalTiered) -> Result<(u8, i128), OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::DecimalRaw;
        let (_tier, raw) = decimal.to_tier_raw();
        let decimal_places = decimal.decimal_places();
        let overflow = OverflowDetected::TierOverflow;
        let storage = match raw {
            DecimalRaw::Small(v) => Self::try_to_binary_storage(v)?,
            DecimalRaw::Medium(v) => {
                { if v.fits_in_i128() { Self::try_to_binary_storage(v.as_i128())? } else { return Err(overflow) } }
            }
            DecimalRaw::Large(v) => {
                { if v.fits_in_i128() { Self::try_to_binary_storage(v.as_i128())? } else { return Err(overflow) } }
            }
            DecimalRaw::XLarge(v) => {
                { if v.fits_in_i128() { Self::try_to_binary_storage(v.as_i128())? } else { return Err(overflow) } }
            }
        };
        Ok((decimal_places, storage))
    }
    fn binary_from_storage(tier: u8, storage: &i128) -> Result<UniversalBinaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalBinaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256

                { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(I256::from_i128(*storage))) }

            }
            5 => {
                // Tier 5 uses I512

                { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(I512::from_i128(*storage))) }

            }
            6 => {
                // Tier 6 uses I1024

                { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i128(*storage))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked_b(val: i128) -> Result<i128, OverflowDetected> {
        { Ok(val) }
    }
    fn binary_to_storage(binary: &UniversalBinaryFixed) -> Result<(u8, i128), OverflowDetected> {
        let (tier, raw) = binary.to_tier_raw();
        let storage = match raw {
            BinaryRaw::Small(v) => Self::i128_to_binary_storage_checked_b(v)?,
            BinaryRaw::Medium(v) => {
                // I256 → i128

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked_b(v.as_i128())?
                }
            }
            BinaryRaw::Large(v) => {
                // I512 → i128

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked_b(v.as_i128())?
                }
            }
            BinaryRaw::XLarge(v) => {
                // I1024 → i128

                {
                    if !v.fits_in_i128() { return Err(OverflowDetected::TierOverflow); }
                    Self::i128_to_binary_storage_checked_b(v.as_i128())?
                }
            }
        };
        Ok((tier, storage))
    }
    #[inline(always)]
    fn try_to_binary_storage(val: i128) -> Result<i128, OverflowDetected> {

        { Ok(val) }

    }
    #[inline(always)]
    fn binary_storage_to_i128(val: &i128) -> i128 {

        { *val }

    }
    fn binary_storage_to_decimal_string(val: i128, max_digits: usize) -> String {
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

    }
    fn decimal_storage_to_string(dec: u8, val: &i128) -> String {
        let dp = dec as usize;

        {
            // Q64.64: i128 = i128. Promote to I256 for division so
            // we never overflow on 10^dec (which exceeds i128 when dec > 38).
            let v = *val;
            let is_negative = v < 0;
            let abs_val = if is_negative { I256::from_i128(v.wrapping_neg()) } else { I256::from_i128(v) };
            let divisor = super::formatting::pow10_i256(dec);
            let integer_part = abs_val / divisor;
            let fractional_part = abs_val % divisor;
            let sign = if is_negative { "-" } else { "" };
            if dp == 0 {
                format!("{}{}", sign, super::formatting::i256_to_decimal_string(integer_part))
            } else {
                format!("{}{}.{}", sign, super::formatting::i256_to_decimal_string(integer_part), super::formatting::i256_to_padded_decimal_string(fractional_part, dp))
            }
        }

    }
    #[inline]
    fn decimal_to_compute_storage(decimals: u8, scaled: i128) -> Result<I256, OverflowDetected> {
        if decimals == 0 {
            // Integer value — shift directly into compute Q-format
            { return Ok(I256::from_i128(scaled) << 128); }
        }

        let den = Self::pow10_compute(decimals)?;

        {
            let num = I256::from_i128(scaled) << 128;
            Ok(num / den)
        }
    }
    #[inline]
    fn pow10_compute(exp: u8) -> Result<I256, OverflowDetected> {
        {
            // I256 = I256. Table covers 0..=38. I256 safe max ~10^76.
            const TABLE: [I256; 39] = super::compute::pow10_table_i256();
            if (exp as usize) < TABLE.len() {
                Ok(TABLE[exp as usize])
            } else if exp <= 76 {
                Ok(super::compute::pow10_compute_fallback_i256(exp))
            } else {
                Err(OverflowDetected::Overflow)
            }
        }
    }
    #[inline]
    fn symbolic_to_compute_storage(num: i128, den: i128) -> Result<I256, OverflowDetected> {
        {
            let n = I256::from_i128(num) << 128;
            let d = I256::from_i128(den);
            Ok(n / d)
        }
    }
    fn symbolic_wide_to_compute_storage(
        rational: &crate::fixed_point::domains::symbolic::rational::RationalNumber,
    ) -> Result<I256, OverflowDetected> {
        let parts = rational.extract_native();

        {
            if let Some((num_i256, den_i256)) = parts.try_as_i256_pair() {
                // I512: an I256 numerator shifted by 128 (was truncated to i128)
                let q = (I512::from_i256(num_i256) << 128) / I512::from_i256(den_i256);
                if !q.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
                return Ok(q.as_i256());
            }
            // Ultra tier (I512): mathematical constants (e, π, etc.) are stored at
            // 77-digit precision as I512/I512 rationals. Use I1024 intermediate to
            // convert to Q128.128 (I256 for q64_64) without overflow.
            if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
                let n = I1024::from_i512(num_i512) << 128;
                let d = I1024::from_i512(den_i512);
                let q = n / d;
                if !q.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
                return Ok(q.as_i256());
            }
        }

        Err(OverflowDetected::TierOverflow)
    }
    fn compute_checked_negate(a: I256) -> Result<I256, OverflowDetected> {
        { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn compute_is_zero(a: &I256) -> bool {
        { *a == I256::zero() }
    }
    #[inline]
    fn compute_is_negative(a: &I256) -> bool {
        { *a < I256::zero() }
    }
    fn rational_i128_to_storage_checked(num: i128, den: i128) -> Result<i128, OverflowDetected> {
        if den == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }
        {
            let fb: usize = 64;
            let n = I256::from_i128(num) << fb;
            let d = I256::from_i128(den);
            let (mut q, r) = crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(n, d);
            let r_abs = if r.is_negative() { -r } else { r };
            let d_abs = if d.is_negative() { -d } else { d };
            let positive = n.is_negative() == d.is_negative();
            let r2 = r_abs + r_abs;
            if if positive { r2 >= d_abs } else { r2 > d_abs } {
                q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
            }
            if !q.fits_in_i128() {
                return Err(OverflowDetected::TierOverflow);
            }
            let q128 = q.as_i128();
            { Ok(q128) }
        }
    }
    fn rational_to_decimal_components(rational: &RationalNumber) -> Result<(u8, i128), OverflowDetected> {
        // Try i128 extraction (works for all profiles ≤ Q128.128 and reduced rationals)
        if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
            return super::transcendentals::rational_to_decimal_components_i128(num, den);
        }

        // Q256.256 fallback: extract as I512 for Ultra-tier rationals

        Err(OverflowDetected::Overflow)
    }
}

impl FascFormat for Fixed<I256, 128> {
    const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 38;
    const DECIMAL_COMPUTE_DP: u8 = 77;
    const DECIMAL_STORAGE_MAX_DP: u8 = 38;
    const DEFAULT_DIGITS: usize = 38;
    const PARSE_MAX_FRAC: usize = 38;
    const MAX_DECIMAL_DIGITS: usize = 38;
    const MAX_BINARY_TIER: u8 = 4;
    const MAX_TERNARY_TIER: u8 = 4;
    const DEPLOYMENT_PROFILE: DeploymentProfile = DeploymentProfile::Balanced;
    fn decimal_to_binary_storage(dp: u8, scaled: I256) -> Result<I256, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // 0.5.0 rounding unification: the RESULT domain is binary, so this
        // coercion rounds nearest, ties toward +∞ on every profile (was:
        // truncation on four arms, add-half on q16_16 — per-profile drift).
        {
            let ten_pow = super::formatting::pow10_i256(dp);
            let num = I512::from_i256(scaled) << 128;
            let den = I512::from_i256(ten_pow);
            let (mut q, rem) =
                crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(num, den);
            let rem_abs = if rem.is_negative() { -rem } else { rem };
            let positive = !num.is_negative();
            let rem2 = rem_abs + rem_abs;
            if if positive { rem2 >= den } else { rem2 > den } {
                q = if positive { q + I512::from_i128(1) } else { q - I512::from_i128(1) };
            }
            if !q.fits_in_i256() { return Err(overflow); }
            Ok(q.as_i256())
        }
    }
    fn decimal_compute_to_binary_storage(val: I512) -> Result<I256, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // Nearest, ties toward +infinity (the binary rule): before 0.6.4 this
        // divided with truncation toward zero, up to one ulp off.
        // `bump` = whether the truncated quotient moves one unit away from zero.
        #[inline(always)]
        fn bump(positive: bool, at_least_half: bool, above_half: bool) -> bool {
            if positive { at_least_half } else { above_half }
        }

        {
            // val is I512 at dp=77. Target: Q128.128 I256.
            // result = val * 2^128 / 10^77 via I1024 intermediate
            let num = I1024::from_i512(val) << 128usize;
            let mut den = I1024::from_i128(1);
            let ten = I1024::from_i128(10);
            for _ in 0..Self::DECIMAL_COMPUTE_DP { den = den * ten; }
            let mut q = num / den;
            let rem = num % den;
            let rem_abs = if (rem.words[15] as i64) < 0 { -rem } else { rem };
            let positive = (num.words[15] as i64) >= 0;
            if bump(positive, rem_abs + rem_abs >= den, rem_abs + rem_abs > den) {
                q = if positive { q + I1024::from_i128(1) } else { q - I1024::from_i128(1) };
            }
            if !q.fits_in_i256() { return Err(overflow); }
            Ok(q.as_i256())
        }
    }
    fn ternary_from_storage(tier: u8, storage: &I256) -> Result<UniversalTernaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalTernaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256

                { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(*storage)) }

            }
            5 => {
                // Tier 5 uses I512

                { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(I512::from_i256(*storage))) }

            }
            6 => {
                // Tier 6 uses I1024

                { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i256(*storage))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked(val: i128) -> Result<I256, OverflowDetected> {

        { Ok(I256::from_i128(val)) }

    }
    fn ternary_to_storage(ternary: &UniversalTernaryFixed) -> Result<(u8, I256), OverflowDetected> {
        let (tier, raw) = ternary.to_tier_raw();
        let storage = match raw {
            TernaryRaw::Small(v) => Self::i128_to_binary_storage_checked(v)?,
            TernaryRaw::Medium(v) => {
                // I256 → I256

                { v }

            }
            TernaryRaw::Large(v) => {
                // I512 → I256

                {
                    if !v.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
                    v.as_i256()
                }

            }
            TernaryRaw::XLarge(v) => {
                // I1024 → I256

                {
                    if !v.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
                    v.as_i256()
                }

            }
        };
        Ok((tier, storage))
    }
    fn ternary_to_rational(tier: u8, value: &I256) -> Result<RationalNumber, OverflowDetected> {
        let ternary = Self::ternary_from_storage(tier, value)?;
        let (ternary_tier, _raw) = ternary.to_tier_raw();

        let frac_trits: u32 = match ternary_tier {
            1 => 10,   // TQ10.10
            2 => 20,   // TQ20.20
            3 => 40,   // TQ40.40
            4 => 80,   // TQ80.80
            5 => 160,  // TQ160.160
            6 => 320,  // TQ320.320
            _ => 40,
        };

        // Tiers 1-3 stay on the i128 fast path: 3^40 = 1.2e19 fits i128.
        if frac_trits <= 40 {
            let mut denom = 1i128;
            for _ in 0..frac_trits {
                denom *= 3;
            }
            let val_i128 = Self::binary_storage_to_i128(value);
            return Ok(RationalNumber::new(val_i128, denom as u128));
        }

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

    }
    fn decimal_from_storage(decimal_places: u8, storage: &I256) -> Result<UniversalDecimalTiered, OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::{DecimalRaw, tier_for_decimal_places};
        let tier = tier_for_decimal_places(decimal_places);

        match tier {
            1..=4 => {
                // UGOD: on wider profiles, the I256 (I256/I512) may exceed i128
                // even when tier_for_decimal_places suggests tiers 1-4 based on dp alone.
                // Check actual value magnitude and promote to tier 5/6 if needed.
                {
                    if !storage.fits_in_i128() {
                        // Value exceeds i128 — promote to tier 5 (I256)
                        return UniversalDecimalTiered::from_tier_raw(
                            5, decimal_places, DecimalRaw::Medium(*storage)
                        );
                    }
                }

                // Value fits in i128 — standard tier 1-4 path
                let val = Self::binary_storage_to_i128(storage);
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

                { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(*storage)) }

            }
            6 => {
                // Tier 6: I512 backing

                { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(I512::from_i256(*storage))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn decimal_to_storage(decimal: &UniversalDecimalTiered) -> Result<(u8, I256), OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::DecimalRaw;
        let (_tier, raw) = decimal.to_tier_raw();
        let decimal_places = decimal.decimal_places();
        let overflow = OverflowDetected::TierOverflow;
        let storage = match raw {
            DecimalRaw::Small(v) => Self::try_to_binary_storage(v)?,
            DecimalRaw::Medium(v) => {
                { v }
            }
            DecimalRaw::Large(v) => {
                { if v.fits_in_i256() { v.as_i256() } else { return Err(overflow) } }
            }
            DecimalRaw::XLarge(v) => {
                { if v.fits_in_i256() { v.as_i256() } else { return Err(overflow) } }
            }
        };
        Ok((decimal_places, storage))
    }
    fn binary_from_storage(tier: u8, storage: &I256) -> Result<UniversalBinaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalBinaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256

                { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(*storage)) }

            }
            5 => {
                // Tier 5 uses I512

                { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(I512::from_i256(*storage))) }

            }
            6 => {
                // Tier 6 uses I1024

                { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i256(*storage))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked_b(val: i128) -> Result<I256, OverflowDetected> {
        { Ok(I256::from_i128(val)) }
    }
    fn binary_to_storage(binary: &UniversalBinaryFixed) -> Result<(u8, I256), OverflowDetected> {
        let (tier, raw) = binary.to_tier_raw();
        let storage = match raw {
            BinaryRaw::Small(v) => Self::i128_to_binary_storage_checked_b(v)?,
            BinaryRaw::Medium(v) => {
                // I256 → I256

                { v }

            }
            BinaryRaw::Large(v) => {
                // I512 → I256

                {
                    if !v.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
                    v.as_i256()
                }

            }
            BinaryRaw::XLarge(v) => {
                // I1024 → I256

                {
                    if !v.fits_in_i256() { return Err(OverflowDetected::TierOverflow); }
                    v.as_i256()
                }

            }
        };
        Ok((tier, storage))
    }
    #[inline(always)]
    fn try_to_binary_storage(val: i128) -> Result<I256, OverflowDetected> {

        { Ok(I256::from_i128(val)) }

    }
    #[inline(always)]
    fn binary_storage_to_i128(val: &I256) -> i128 {

        { val.as_i256().as_i128() }

    }
    fn binary_storage_to_decimal_string(val: I256, max_digits: usize) -> String {

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

    }
    fn decimal_storage_to_string(dec: u8, val: &I256) -> String {
        let dp = dec as usize;

        {
            // Q128.128: I256 = I256. Division done natively in I256.
            let is_negative = *val < I256::zero();
            let abs_val = if is_negative { -*val } else { *val };
            let divisor = super::formatting::pow10_i256(dec);
            let integer_part = abs_val / divisor;
            let fractional_part = abs_val % divisor;
            let sign = if is_negative { "-" } else { "" };
            if dp == 0 {
                format!("{}{}", sign, super::formatting::i256_to_decimal_string(integer_part))
            } else {
                format!("{}{}.{}", sign, super::formatting::i256_to_decimal_string(integer_part), super::formatting::i256_to_padded_decimal_string(fractional_part, dp))
            }
        }

    }
    #[inline]
    fn decimal_to_compute_storage(decimals: u8, scaled: I256) -> Result<I512, OverflowDetected> {
        if decimals == 0 {
            // Integer value — shift directly into compute Q-format
            { return Ok(I512::from_i256(scaled) << 256); }
        }

        let den = Self::pow10_compute(decimals)?;

        {
            let num = I512::from_i256(scaled) << 256;
            Ok(num / den)
        }
    }
    #[inline]
    fn pow10_compute(exp: u8) -> Result<I512, OverflowDetected> {
        {
            // I512 = I512. Table covers 0..=76. I512 safe max ~10^153.
            const TABLE: [I512; 77] = super::compute::pow10_table_i512();
            if (exp as usize) < TABLE.len() {
                Ok(TABLE[exp as usize])
            } else if exp <= 153 {
                Ok(super::compute::pow10_compute_fallback_i512(exp))
            } else {
                Err(OverflowDetected::Overflow)
            }
        }
    }
    #[inline]
    fn symbolic_to_compute_storage(num: i128, den: i128) -> Result<I512, OverflowDetected> {
        {
            let n = I512::from_i256(I256::from_i128(num)) << 256;
            let d = I512::from_i256(I256::from_i128(den));
            Ok(n / d)
        }
    }
    fn symbolic_wide_to_compute_storage(
        rational: &crate::fixed_point::domains::symbolic::rational::RationalNumber,
    ) -> Result<I512, OverflowDetected> {
        let parts = rational.extract_native();

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

        Err(OverflowDetected::TierOverflow)
    }
    fn compute_checked_negate(a: I512) -> Result<I512, OverflowDetected> {
        { a.checked_neg().ok_or(OverflowDetected::TierOverflow) }
    }
    #[inline]
    fn compute_is_zero(a: &I512) -> bool {
        { *a == I512::zero() }
    }
    #[inline]
    fn compute_is_negative(a: &I512) -> bool {
        { *a < I512::zero() }
    }
    fn rational_i128_to_storage_checked(num: i128, den: i128) -> Result<I256, OverflowDetected> {
        if den == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }
        {
            let n = I512::from_i256(I256::from_i128(num)) << 128;
            let d = I512::from_i256(I256::from_i128(den));
            let (mut q, r) = crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(n, d);
            let r_abs = if r.is_negative() { -r } else { r };
            let d_abs = if d.is_negative() { -d } else { d };
            let positive = n.is_negative() == d.is_negative();
            let r2 = r_abs + r_abs;
            if if positive { r2 >= d_abs } else { r2 > d_abs } {
                q = if positive { q + I512::from_i128(1) } else { q - I512::from_i128(1) };
            }
            if !q.fits_in_i256() {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(q.as_i256())
        }
    }
    fn rational_to_decimal_components(rational: &RationalNumber) -> Result<(u8, i128), OverflowDetected> {
        // Try i128 extraction (works for all profiles ≤ Q128.128 and reduced rationals)
        if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
            return super::transcendentals::rational_to_decimal_components_i128(num, den);
        }

        // Q256.256 fallback: extract as I512 for Ultra-tier rationals

        Err(OverflowDetected::Overflow)
    }
}

impl FascFormat for Fixed<I512, 256> {
    const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 76;
    const DECIMAL_COMPUTE_DP: u8 = 154;
    const DECIMAL_STORAGE_MAX_DP: u8 = 77;
    const DEFAULT_DIGITS: usize = 77;
    const PARSE_MAX_FRAC: usize = 76;
    const MAX_DECIMAL_DIGITS: usize = 77;
    const MAX_BINARY_TIER: u8 = 5;
    const MAX_TERNARY_TIER: u8 = 5;
    const DEPLOYMENT_PROFILE: DeploymentProfile = DeploymentProfile::Scientific;
    fn decimal_to_binary_storage(dp: u8, scaled: I512) -> Result<I512, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // 0.5.0 rounding unification: the RESULT domain is binary, so this
        // coercion rounds nearest, ties toward +∞ on every profile (was:
        // truncation on four arms, add-half on q16_16 — per-profile drift).
        {
            let ten_pow = I1024::from_i512(super::formatting::pow10_i512(dp));
            let num = I1024::from_i512(scaled) << 256;
            let mut q = num / ten_pow;
            let rem = num % ten_pow;
            let rem_neg = (rem.words[15] as i64) < 0;
            let rem_abs = if rem_neg { -rem } else { rem };
            let positive = (num.words[15] as i64) >= 0;
            let rem2 = rem_abs + rem_abs;
            if if positive { rem2 >= ten_pow } else { rem2 > ten_pow } {
                q = if positive { q + I1024::from_i128(1) } else { q - I1024::from_i128(1) };
            }
            if !q.fits_in_i512() { return Err(overflow); }
            Ok(q.as_i512())
        }
    }
    fn decimal_compute_to_binary_storage(val: I1024) -> Result<I512, OverflowDetected> {
        let overflow = OverflowDetected::TierOverflow;

        // Nearest, ties toward +infinity (the binary rule): before 0.6.4 this
        // divided with truncation toward zero, up to one ulp off.
        // `bump` = whether the truncated quotient moves one unit away from zero.
        #[inline(always)]
        fn bump(positive: bool, at_least_half: bool, above_half: bool) -> bool {
            if positive { at_least_half } else { above_half }
        }

        {
            // val is I1024 at dp=154. Target: Q256.256 I512.
            // Use I2048 intermediate.
            use crate::fixed_point::I2048;
            use crate::fixed_point::domains::binary_fixed::i2048::i2048_divmod;
            let num = I2048::from_i1024(val) << 256usize;
            let mut pow = I1024::from_i128(1);
            let ten = I1024::from_i128(10);
            for _ in 0..Self::DECIMAL_COMPUTE_DP { pow = pow * ten; }
            let den = I2048::from_i1024(pow);
            let (mut q, rem) = i2048_divmod(num, den);
            let positive = (num.words[31] as i64) >= 0;
            let rem_abs = if (rem.words[31] as i64) < 0 { -rem } else { rem };
            if bump(positive, rem_abs + rem_abs >= den, rem_abs + rem_abs > den) {
                q = if positive { q + I2048::one() } else { q - I2048::one() };
            }
            if !q.fits_in_i512() { return Err(overflow); }
            Ok(I512::from_words([
                q.words[0], q.words[1], q.words[2], q.words[3],
                q.words[4], q.words[5], q.words[6], q.words[7],
            ]))
        }
    }
    fn ternary_from_storage(tier: u8, storage: &I512) -> Result<UniversalTernaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalTernaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256
                { UniversalTernaryFixed::from_tier_raw(4, TernaryRaw::Medium(storage.as_i256())) }

            }
            5 => {
                // Tier 5 uses I512
                { UniversalTernaryFixed::from_tier_raw(5, TernaryRaw::Large(*storage)) }

            }
            6 => {
                // Tier 6 uses I1024
                { UniversalTernaryFixed::from_tier_raw(6, TernaryRaw::XLarge(I1024::from_i512(*storage))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked(val: i128) -> Result<I512, OverflowDetected> {
        { Ok(I512::from_i128(val)) }

    }
    fn ternary_to_storage(ternary: &UniversalTernaryFixed) -> Result<(u8, I512), OverflowDetected> {
        let (tier, raw) = ternary.to_tier_raw();
        let storage = match raw {
            TernaryRaw::Small(v) => Self::i128_to_binary_storage_checked(v)?,
            TernaryRaw::Medium(v) => {
                // I256 → I512
                { I512::from_i256(v) }

            }
            TernaryRaw::Large(v) => {
                // I512 → I512
                { v }

            }
            TernaryRaw::XLarge(v) => {
                // I1024 → I512
                {
                    if !v.fits_in_i512() { return Err(OverflowDetected::TierOverflow); }
                    v.as_i512()
                }

            }
        };
        Ok((tier, storage))
    }
    fn ternary_to_rational(tier: u8, value: &I512) -> Result<RationalNumber, OverflowDetected> {
        let ternary = Self::ternary_from_storage(tier, value)?;
        let (ternary_tier, _raw) = ternary.to_tier_raw();

        let frac_trits: u32 = match ternary_tier {
            1 => 10,   // TQ10.10
            2 => 20,   // TQ20.20
            3 => 40,   // TQ40.40
            4 => 80,   // TQ80.80
            5 => 160,  // TQ160.160
            6 => 320,  // TQ320.320
            _ => 40,
        };

        // Tiers 1-3 stay on the i128 fast path: 3^40 = 1.2e19 fits i128.
        if frac_trits <= 40 {
            let mut denom = 1i128;
            for _ in 0..frac_trits {
                denom *= 3;
            }
            let val_i128 = Self::binary_storage_to_i128(value);
            return Ok(RationalNumber::new(val_i128, denom as u128));
        }

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

    }
    fn decimal_from_storage(decimal_places: u8, storage: &I512) -> Result<UniversalDecimalTiered, OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::{DecimalRaw, tier_for_decimal_places};
        let tier = tier_for_decimal_places(decimal_places);

        match tier {
            1..=4 => {
                // UGOD: on wider profiles, the I512 (I256/I512) may exceed i128
                // even when tier_for_decimal_places suggests tiers 1-4 based on dp alone.
                // Check actual value magnitude and promote to tier 5/6 if needed.
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
                let val = Self::binary_storage_to_i128(storage);
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
                { UniversalDecimalTiered::from_tier_raw(5, decimal_places, DecimalRaw::Medium(storage.as_i256())) }

            }
            6 => {
                // Tier 6: I512 backing
                { UniversalDecimalTiered::from_tier_raw(6, decimal_places, DecimalRaw::Large(*storage)) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn decimal_to_storage(decimal: &UniversalDecimalTiered) -> Result<(u8, I512), OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::decimal_types::DecimalRaw;
        let (_tier, raw) = decimal.to_tier_raw();
        let decimal_places = decimal.decimal_places();
        let overflow = OverflowDetected::TierOverflow;
        let storage = match raw {
            DecimalRaw::Small(v) => Self::try_to_binary_storage(v)?,
            DecimalRaw::Medium(v) => {
                { I512::from_i256(v) }
            }
            DecimalRaw::Large(v) => {
                { v }
            }
            DecimalRaw::XLarge(v) => {
                { if v.fits_in_i512() { v.as_i512() } else { return Err(overflow) } }
            }
        };
        Ok((decimal_places, storage))
    }
    fn binary_from_storage(tier: u8, storage: &I512) -> Result<UniversalBinaryFixed, OverflowDetected> {
        match tier {
            1..=3 => {
                // Tiers 1-3 fit in i128
                let val = Self::binary_storage_to_i128(storage);
                UniversalBinaryFixed::from_tier_value(tier, val)
            }
            4 => {
                // Tier 4 uses I256
                { UniversalBinaryFixed::from_tier_raw(4, BinaryRaw::Medium(storage.as_i256())) }

            }
            5 => {
                // Tier 5 uses I512
                { UniversalBinaryFixed::from_tier_raw(5, BinaryRaw::Large(*storage)) }

            }
            6 => {
                // Tier 6 uses I1024
                { UniversalBinaryFixed::from_tier_raw(6, BinaryRaw::XLarge(I1024::from_i512(*storage))) }

            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }
    fn i128_to_binary_storage_checked_b(val: i128) -> Result<I512, OverflowDetected> {
        { Ok(I512::from_i128(val)) }
    }
    fn binary_to_storage(binary: &UniversalBinaryFixed) -> Result<(u8, I512), OverflowDetected> {
        let (tier, raw) = binary.to_tier_raw();
        let storage = match raw {
            BinaryRaw::Small(v) => Self::i128_to_binary_storage_checked_b(v)?,
            BinaryRaw::Medium(v) => {
                // I256 → I512
                { I512::from_i256(v) }

            }
            BinaryRaw::Large(v) => {
                // I512 → I512
                { v }

            }
            BinaryRaw::XLarge(v) => {
                // I1024 → I512
                {
                    if !v.fits_in_i512() { return Err(OverflowDetected::TierOverflow); }
                    v.as_i512()
                }

            }
        };
        Ok((tier, storage))
    }
    #[inline(always)]
    fn try_to_binary_storage(val: i128) -> Result<I512, OverflowDetected> {
        { Ok(I512::from_i128(val)) }

    }
    #[inline(always)]
    fn binary_storage_to_i128(val: &I512) -> i128 {
        { val.as_i512().as_i256().as_i128() }

    }
    fn binary_storage_to_decimal_string(val: I512, max_digits: usize) -> String {

        {
            // Q256.256: I512
            let is_negative = val < I512::zero();
            let abs_val = if is_negative { -val } else { val };
            // Integer part can exceed u128 (up to 2^255-1): extract digits at
            // I256 width. (0.5.0 item 1: the old `as_i128 as u128` squeeze
            // wrapped any integer part >= 2^127 — 1e20*1e20 displayed as
            // 1e40 mod 2^128.)
            let int_i256 = (abs_val >> 256).as_i256();
            let int_part = if int_i256 == I256::zero() {
                "0".to_string()
            } else {
                let ten = I256::from_i128(10);
                let mut digits = Vec::new();
                let mut n = int_i256;
                while n != I256::zero() {
                    let digit = (n % ten).as_i128() as u8;
                    digits.push((b'0' + digit) as char);
                    n = n / ten;
                }
                digits.reverse();
                digits.into_iter().collect::<String>()
            };

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
    fn decimal_storage_to_string(dec: u8, val: &I512) -> String {
        let dp = dec as usize;

        {
            // Q256.256: I512 = I512. For dec > 76, promote to I1024.
            let is_negative = *val < I512::zero();
            let abs_val = if is_negative { -*val } else { *val };

            if dp <= 154 {
                // I512 can hold 10^154. Use native I512 division.
                let divisor = super::formatting::pow10_i512(dec);
                let integer_part = abs_val / divisor;
                let fractional_part = abs_val % divisor;
                let sign = if is_negative { "-" } else { "" };
                if dp == 0 {
                    format!("{}{}", sign, super::formatting::i512_to_decimal_string(integer_part))
                } else {
                    format!("{}{}.{}", sign, super::formatting::i512_to_decimal_string(integer_part), super::formatting::i512_to_padded_decimal_string(fractional_part, dp))
                }
            } else {
                // dp > 154: promote to I1024
                let v1024 = I1024::from_i512(abs_val);
                let divisor = super::formatting::pow10_i1024(dec);
                let integer_part = v1024 / divisor;
                let fractional_part = v1024 % divisor;
                let sign = if is_negative { "-" } else { "" };
                if dp == 0 {
                    format!("{}{}", sign, super::formatting::i1024_to_decimal_string(integer_part))
                } else {
                    format!("{}{}.{}", sign, super::formatting::i1024_to_decimal_string(integer_part), super::formatting::i1024_to_padded_decimal_string(fractional_part, dp))
                }
            }
        }
    }
    #[inline]
    fn decimal_to_compute_storage(decimals: u8, scaled: I512) -> Result<I1024, OverflowDetected> {
        if decimals == 0 {
            // Integer value — shift directly into compute Q-format
            { return Ok(I1024::from_i512(scaled) << 512); }
        }

        let den = Self::pow10_compute(decimals)?;

        {
            let num = I1024::from_i512(scaled) << 512;
            Ok(num / den)
        }
    }
    #[inline]
    fn pow10_compute(exp: u8) -> Result<I1024, OverflowDetected> {
        {
            // I1024 = I1024. Table covers 0..=154. I1024 safe max ~10^307.
            // u8 max is 255 < 308, so all fallback values are safe.
            const TABLE: [I1024; 155] = super::compute::pow10_table_i1024();
            if (exp as usize) < TABLE.len() {
                Ok(TABLE[exp as usize])
            } else {
                Ok(super::compute::pow10_compute_fallback_i1024(exp))
            }
        }
    }
    #[inline]
    fn symbolic_to_compute_storage(num: i128, den: i128) -> Result<I1024, OverflowDetected> {
        {
            let n = I1024::from_i512(I512::from_i256(I256::from_i128(num))) << 512;
            let d = I1024::from_i512(I512::from_i256(I256::from_i128(den)));
            Ok(n / d)
        }
    }
    fn symbolic_wide_to_compute_storage(
        rational: &crate::fixed_point::domains::symbolic::rational::RationalNumber,
    ) -> Result<I1024, OverflowDetected> {
        let parts = rational.extract_native();

        {
            if let Some((num_i512, den_i512)) = parts.try_as_i512_pair() {
                let n = I1024::from_i512(num_i512) << 512;
                let d = I1024::from_i512(den_i512);
                return Ok(n / d);
            }
        }

        Err(OverflowDetected::TierOverflow)
    }
    fn compute_checked_negate(a: I1024) -> Result<I1024, OverflowDetected> {
        { if a == I1024::min_value() { Err(OverflowDetected::TierOverflow) } else { Ok(-a) } }
    }
    #[inline]
    fn compute_is_zero(a: &I1024) -> bool {
        { *a == I1024::zero() }
    }
    #[inline]
    fn compute_is_negative(a: &I1024) -> bool {
        { (a.words[15] & 0x8000_0000_0000_0000) != 0 }
    }
    fn rational_i128_to_storage_checked(num: i128, den: i128) -> Result<I512, OverflowDetected> {
        if den == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }
        {
            use crate::fixed_point::I1024;
            let n = I1024::from_i512(I512::from_i256(I256::from_i128(num))) << 256;
            let d = I1024::from_i512(I512::from_i256(I256::from_i128(den)));
            let mut q = n / d;
            let r = n % d;
            let r_neg = (r.words[15] as i64) < 0;
            let d_neg = (d.words[15] as i64) < 0;
            let n_neg = (n.words[15] as i64) < 0;
            let r_abs = if r_neg { -r } else { r };
            let d_abs = if d_neg { -d } else { d };
            let positive = n_neg == d_neg;
            let r2 = r_abs + r_abs;
            if if positive { r2 >= d_abs } else { r2 > d_abs } {
                q = if positive { q + I1024::from_i128(1) } else { q - I1024::from_i128(1) };
            }
            if !q.fits_in_i512() {
                return Err(OverflowDetected::TierOverflow);
            }
            Ok(q.as_i512())
        }
    }
    fn rational_to_decimal_components(rational: &RationalNumber) -> Result<(u8, i128), OverflowDetected> {
        // Try i128 extraction (works for all profiles ≤ Q128.128 and reduced rationals)
        if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
            return super::transcendentals::rational_to_decimal_components_i128(num, den);
        }

        // Q256.256 fallback: extract as I512 for Ultra-tier rationals
        {
            let parts = rational.extract_native();
            if let Some((num, den)) = parts.try_as_i512_pair() {
                return super::transcendentals::rational_to_decimal_components_i512(num, den);
            }
        }

        Err(OverflowDetected::Overflow)
    }
}
