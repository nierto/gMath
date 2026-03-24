//! Parsing — string literal to StackValue conversion
//!
//! 5-phase parsing: fractions, repeating decimals, hex/binary, decimals, named constants, integers.
//! Mode routing integration for compute_mode override.

use super::{StackValue, StackEvaluator};
use super::conversion::to_binary_storage;
use super::domain::ternary_to_storage;
#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
use crate::fixed_point::universal::tier_types::CompactShadow;
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, OverflowDetected};
use crate::fixed_point::universal::ConstantId;

impl StackEvaluator {
    /// Parse literal string — single-pass byte classifier
    ///
    /// **ARCHITECTURE**: Prefix checks + single byte scan + dispatch. Zero heap allocation.
    /// Replaces former CHD hash (~100ns) + NumberClassifier (~300-500ns) pipeline.
    ///
    /// **ROUTING ORDER** (fastest-exit-first):
    /// 1. Prefix check: 0x/0b/0t → Binary/Ternary (2 byte reads, no scan)
    /// 2. First byte: alphabetic → named constant (1 byte read, no scan)
    /// 3. Single-pass scan: detect '/' and '.' simultaneously
    ///    a. '/' found → Fraction → Symbolic (short-circuits scan)
    ///    b. '.' + "..." suffix → Repeating decimal → Symbolic
    ///    c. '.' found → Decimal (proven exact: 10^N = 2^N × 5^N)
    /// 4. Default → integer → Binary
    pub(crate) fn parse_literal(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let bytes = s.as_bytes();
        let len = bytes.len();
        if len == 0 {
            return Err(OverflowDetected::ParseError);
        }

        // FAST PREFIX CHECK: Hex/binary/ternary prefixes (0x/0b/0t)
        // Checked before byte scan because prefix is 2 bytes at known position.
        if len > 1 && bytes[0] == b'0' {
            match bytes[1] {
                b'x' | b'X' => return self.parse_binary_hex(s),
                b'b' | b'B' => return self.parse_binary_bin(s),
                b't' | b'T' => return self.parse_ternary_value(&s[2..]),
                _ => {}
            }
        }

        // FAST FIRST-BYTE CHECK: Alphabetic → named constant (pi, e, sqrt2, ...)
        // No scan needed — single byte read.
        if bytes[0].is_ascii_alphabetic() || bytes[0] > 127 {
            return self.parse_named_constant(s);
        }

        // SINGLE-PASS BYTE SCAN: Detect '/' and '.' in one traversal.
        // For the dominant case ("1.5", "3.14159"), this replaces two separate
        // full-string scans with one. Short-circuits on '/' (fractions are rare).
        let mut has_slash = false;
        let mut has_dot = false;
        for &b in bytes {
            if b == b'/' { has_slash = true; break; } // slash found — no need to continue
            if b == b'.' { has_dot = true; }           // dot found — keep scanning for '/'
        }

        if has_slash {
            return self.parse_fraction(s);
        }

        // REPEATING DECIMAL CHECK: "0.333..." — must have dot AND end with "..."
        // Only checked when we already know a dot exists (from the scan above).
        if has_dot && len > 3
            && bytes[len - 3] == b'.'
            && bytes[len - 2] == b'.'
            && bytes[len - 1] == b'.'
        {
            return self.parse_repeating_decimal(s);
        }

        // DECIMAL: Contains '.' → Decimal domain (proven exact)
        // MATHEMATICAL PROOF: ALL decimal strings are exact in Decimal domain.
        // Denominator = 10^N = 2^N × 5^N. GCD cannot introduce new primes.
        if has_dot {
            return self.parse_decimal(s);
        }

        // DEFAULT: Integer → Binary
        self.parse_integer(s)
    }

    /// Parse decimal string
    ///
    /// For ≤38 fractional digits: direct i128 path (Decimal domain).
    /// For >38 fractional digits: use two-part parsing (Q256.256 only) or
    /// truncate to 38 digits (i128 scale factor limit).
    ///
    /// All profiles accept up to 38 fractional digits via the i128 path.
    /// The decimal→binary conversion naturally rounds to the closest
    /// representable Q-format value. Do NOT truncate to profile precision
    /// before conversion — truncation introduces different rounding than
    /// the mathematically correct round-to-nearest.
    pub(crate) fn parse_decimal(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let dot_pos = s.find('.').ok_or(OverflowDetected::ParseError)?;
        if s[dot_pos + 1..].contains('.') {
            return Err(OverflowDetected::ParseError);
        }

        let is_negative = s.starts_with('-');
        let integer_str = if is_negative { &s[1..dot_pos] } else { &s[..dot_pos] };
        let fractional_str = &s[dot_pos + 1..];
        let frac_len = fractional_str.len();

        // All profiles use i128 path for ≤38 digits (10^38 fits in i128).
        // Q256.256 extends to 76 digits via two-part parsing.
        #[cfg(table_format = "q16_16")]
        let max_frac: usize = 4;
        #[cfg(table_format = "q32_32")]
        let max_frac: usize = 9;
        #[cfg(table_format = "q64_64")]
        let max_frac: usize = 38;
        #[cfg(table_format = "q128_128")]
        let max_frac: usize = 38;
        #[cfg(table_format = "q256_256")]
        let max_frac: usize = 76; // 77 sig digits minus integer part headroom

        // Truncate to what the profile can represent
        let effective_frac_len = frac_len.min(max_frac);
        let effective_frac_str = &fractional_str[..effective_frac_len];

        // For ≤38 digits: single i128 path
        if effective_frac_len <= 38 {
            let integer_part = integer_str.parse::<i128>()
                .map_err(|_| OverflowDetected::ParseError)?;
            let integer_part = if is_negative { -integer_part } else { integer_part };
            let decimals = effective_frac_len as u8;
            let fractional_part = effective_frac_str.parse::<i128>()
                .map_err(|_| OverflowDetected::ParseError)?;
            let scale = 10_i128.pow(decimals as u32);
            let scaled = integer_part
                .checked_mul(scale)
                .and_then(|v| v.checked_add(if integer_part < 0 { -fractional_part } else { fractional_part }))
                .ok_or(OverflowDetected::Overflow)?;
            let shadow = CompactShadow::from_rational(scaled, scale as u128);
            Ok(StackValue::Decimal(decimals, to_binary_storage(scaled), shadow))
        } else {
            // >38 fractional digits (scientific profile Q256.256).
            // Split into two 38-digit halves, parse each into i128,
            // then combine at I512 width for BinaryStorage.
            //
            // Strategy: parse "7.HHHH...LLLL..." as
            //   value = integer + high/10^38 + low/10^76
            // Each piece fits in i128. Combine via I512 arithmetic,
            // then convert to Q-format via I1024 shift-and-divide.
            let high_len = 38usize.min(effective_frac_len);
            let low_len = effective_frac_len - high_len;
            let high_frac = &effective_frac_str[..high_len];
            let low_frac = &effective_frac_str[high_len..];

            let integer_part = integer_str.parse::<i128>()
                .map_err(|_| OverflowDetected::ParseError)?;
            let high_part = high_frac.parse::<i128>()
                .map_err(|_| OverflowDetected::ParseError)?;
            let low_part = if low_len > 0 {
                low_frac.parse::<i128>().map_err(|_| OverflowDetected::ParseError)?
            } else { 0i128 };

            #[cfg(table_format = "q256_256")]
            {
                use crate::fixed_point::i1024::I1024;

                // Build the numerator at I512 width:
                // numerator = integer * 10^N + high * 10^low_len + low
                // where N = high_len + low_len = effective_frac_len
                let scale_high = {
                    // 10^low_len (≤ 10^38, fits in i128)
                    let mut s = I512::from_i128(1);
                    for _ in 0..low_len { s = s * I512::from_i128(10); }
                    s
                };
                let scale_total = {
                    // 10^N = 10^(high_len + low_len) ≤ 10^76
                    let mut s = I512::from_i128(1);
                    for _ in 0..effective_frac_len { s = s * I512::from_i128(10); }
                    s
                };

                let numerator = I512::from_i128(integer_part) * scale_total
                    + I512::from_i128(high_part) * scale_high
                    + I512::from_i128(low_part);
                let numerator = if is_negative { -numerator } else { numerator };

                // Convert to Q256.256: raw = numerator * 2^256 / 10^N
                let num_wide = I1024::from_i512(numerator) << 256usize;
                let den_wide = I1024::from_i512(scale_total);
                // Add rounding: (num_wide + den_wide/2) / den_wide
                let half_den = den_wide >> 1usize;
                let rounded = if is_negative {
                    num_wide - half_den
                } else {
                    num_wide + half_den
                };
                let q_value = (rounded / den_wide).as_i512();

                let shadow = CompactShadow::None;
                let tier = self.profile_max_binary_tier();
                Ok(StackValue::Binary(tier, q_value, shadow))
            }

            #[cfg(not(table_format = "q256_256"))]
            {
                let _ = (integer_part, high_part, low_part, high_len, low_len);
                Err(OverflowDetected::Overflow)
            }
        }
    }

    /// Parse decimal as rational (for inexact decimals like 0.333)
    /// Parse repeating decimal notation (0.333..., 0.142857...)
    ///
    /// **MATHEMATICAL ALGORITHM**: Convert repeating decimal to exact rational
    /// - Formula: 0.abc... = abc / (10^n - 1) where n = length of repeating part
    /// - For mixed: k.abc... = k + (abc / (10^n - 1))
    ///
    /// **EXAMPLES**:
    /// - "0.333..." → numerator = 333, denominator = 999 → reduces to 1/3
    /// - "0.142857..." → numerator = 142857, denominator = 999999 → reduces to 1/7
    /// - "2.333..." → 2 + 333/999 → 2331/999 → reduces to 7/3
    pub(crate) fn parse_repeating_decimal(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        // Remove ellipsis
        let trimmed = s.trim_end_matches("...");
        let dot_pos = trimmed.find('.').ok_or(OverflowDetected::ParseError)?;
        // Reject multiple dots
        if trimmed[dot_pos + 1..].contains('.') {
            return Err(OverflowDetected::ParseError);
        }

        let integer_part = trimmed[..dot_pos].parse::<i128>()
            .map_err(|_| OverflowDetected::ParseError)?;
        let repeating_str = &trimmed[dot_pos + 1..];
        // Guard: 10^rep_len must fit in i128 (max 38 digits)
        if repeating_str.len() > 38 {
            return Err(OverflowDetected::Overflow);
        }
        let rep_len = repeating_str.len() as u32;
        let rep_value = repeating_str.parse::<i128>()
            .map_err(|_| OverflowDetected::ParseError)?;

        // Safe: rep_len <= 38, so 10^38 fits in i128
        // Denominator is 999...9 (n nines) = 10^n - 1
        let denominator = 10_i128.pow(rep_len)
            .checked_sub(1)
            .ok_or(OverflowDetected::Overflow)?;

        // Numerator = integer_part * denominator + repeating_value
        let numerator = integer_part
            .checked_mul(denominator)
            .and_then(|v| v.checked_add(rep_value))
            .ok_or(OverflowDetected::Overflow)?;

        // RationalNumber will reduce to lowest terms (333/999 → 1/3)
        let rational = RationalNumber::new(numerator, denominator as u128);
        Ok(StackValue::Symbolic(rational))
    }

    /// Parse hex literal to binary domain
    pub(crate) fn parse_binary_hex(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let hex_str = s.trim_start_matches("0x").trim_start_matches("0X");
        let value = i128::from_str_radix(hex_str, 16)
            .map_err(|_| OverflowDetected::ParseError)?;
        let shadow = CompactShadow::from_rational(value, 1);
        Ok(StackValue::Binary(1, to_binary_storage(value), shadow))
    }

    /// Parse binary literal
    pub(crate) fn parse_binary_bin(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let bin_str = s.trim_start_matches("0b").trim_start_matches("0B");
        let value = i128::from_str_radix(bin_str, 2)
            .map_err(|_| OverflowDetected::ParseError)?;
        let shadow = CompactShadow::from_rational(value, 1);
        Ok(StackValue::Binary(1, to_binary_storage(value), shadow))
    }

    /// Parse fraction to symbolic domain
    pub(crate) fn parse_fraction(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let slash_pos = s.find('/').ok_or(OverflowDetected::ParseError)?;
        // Reject multiple slashes (e.g. "1/2/3")
        if s[slash_pos + 1..].contains('/') {
            return Err(OverflowDetected::ParseError);
        }

        let numerator = s[..slash_pos].parse::<i128>()
            .map_err(|_| OverflowDetected::ParseError)?;
        let denominator = s[slash_pos + 1..].parse::<u128>()  // Parse as u128 for denominator
            .map_err(|_| OverflowDetected::ParseError)?;

        // Guard: zero denominator would panic in RationalNumber::new (assert_ne!)
        if denominator == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }

        let rational = RationalNumber::new(numerator, denominator);
        Ok(StackValue::Symbolic(rational))
    }

    /// Parse integer to most appropriate domain
    ///
    /// **CRITICAL**: Must convert integer to Q-format by shifting left
    /// - Q64.64: value << 64
    /// - Q128.128: value << 128
    /// - Q256.256: value << 256
    pub(crate) fn parse_integer(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let value = s.parse::<i128>()
            .map_err(|_| OverflowDetected::ParseError)?;

        // Create shadow: exact rational = value/1
        let shadow = CompactShadow::from_rational(value, 1);

        // Convert to Q-format based on profile (shift left by fractional bits)
        let storage_tier = self.profile_max_binary_tier();

        #[cfg(table_format = "q256_256")]
        {
            let q_value = I512::from_i128(value) << 256;
            Ok(StackValue::Binary(storage_tier, q_value, shadow))
        }

        #[cfg(table_format = "q128_128")]
        {
            let q_value = I256::from_i128(value) << 128;
            Ok(StackValue::Binary(storage_tier, q_value, shadow))
        }

        #[cfg(table_format = "q64_64")]
        {
            // Guard: value << 64 wraps silently for |value| > i64::MAX.
            // Q64.64 has 64 integer bits (signed), so valid range is i64.
            if value > i64::MAX as i128 || value < i64::MIN as i128 {
                return Err(OverflowDetected::Overflow);
            }
            let q_value = value << 64;
            Ok(StackValue::Binary(storage_tier, q_value, shadow))
        }

        #[cfg(table_format = "q32_32")]
        {
            // Q32.32 has 32 integer bits (signed), valid range is i32.
            if value > i32::MAX as i128 || value < i32::MIN as i128 {
                return Err(OverflowDetected::Overflow);
            }
            let q_value = (value as i64) << 32;
            Ok(StackValue::Binary(storage_tier, q_value, shadow))
        }

        #[cfg(table_format = "q16_16")]
        {
            // Q16.16 has 16 integer bits (signed), valid range is i16.
            if value > i16::MAX as i128 || value < i16::MIN as i128 {
                return Err(OverflowDetected::Overflow);
            }
            let q_value = (value as i32) << 16;
            Ok(StackValue::Binary(storage_tier, q_value, shadow))
        }

    }

    /// Parse named mathematical constant via fast hardcoded match
    ///
    /// **PURPOSE**: Replaces NumberClassifier + parse_mathematical_constant() path
    /// for alphabetic inputs. O(1) match on ~10 known constant names.
    ///
    /// **PRECISION**: Uses SymbolicConstants high-precision methods (profile-aware:
    /// 19/38/77 decimals) — NOT crude rational approximations.
    ///
    /// **EXAMPLES**: "pi" → π (full precision), "e" → Euler's number, "sqrt2" → √2
    pub(crate) fn parse_named_constant(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        use crate::fixed_point::domains::symbolic::rational::mathematical_constants::SymbolicConstants;

        // Fast match on ASCII constant names and UTF-8 symbols.
        // Returns full profile-aware precision from build.rs-generated constants.
        let rational = match s {
            "pi" | "PI" | "Pi" | "π" => {
                SymbolicConstants::pi_high_precision()
            }
            "e" | "E" => {
                SymbolicConstants::e_high_precision()
            }
            "sqrt2" | "SQRT2" | "√2" => {
                SymbolicConstants::sqrt_2_high_precision()
            }
            "sqrt3" | "SQRT3" | "√3" => {
                SymbolicConstants::sqrt_3_high_precision()
            }
            "phi" | "PHI" | "φ" => {
                SymbolicConstants::golden_ratio_high_precision()
            }
            "ln2" | "LN2" => {
                SymbolicConstants::ln_2_high_precision()
            }
            "ln10" | "LN10" => {
                SymbolicConstants::ln_10()
            }
            "gamma" | "γ" => {
                // Euler-Mascheroni — no high-precision build.rs constant available
                RationalNumber::new(577215665, 1000000000)
            }
            _ => {
                return Err(OverflowDetected::ParseError);
            }
        };

        Ok(StackValue::Symbolic(rational))
    }

    /// Parse value into ternary domain via `0t` prefix.
    ///
    /// **FLOAT-FREE**: Pure integer parsing — no f64 contamination
    /// **ALGORITHM**: Converts decimal string to base-3 scaled integer via UniversalTernaryFixed
    /// **STORAGE**: Ternary(tier, raw_value) where raw_value = value × 3^frac_trits
    /// **PREFIX**: `0t1.5` → parse "1.5" → base-3 fixed-point representation
    pub(crate) fn parse_ternary_value(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        use crate::fixed_point::domains::balanced_ternary::UniversalTernaryFixed;
        let ternary = UniversalTernaryFixed::from_str(s)?;
        let (tier, storage) = ternary_to_storage(&ternary);

        // Create shadow: raw_value / 3^frac_trits (when both fit in i128/u128)
        let shadow = {
            let frac_trits: u32 = match tier {
                1 => 8, 2 => 16, 3 => 32, 4 => 64, _ => 128,
            };
            // 3^64 fits in u128, 3^128 does not
            if frac_trits <= 64 {
                let den = 3u128.pow(frac_trits);

                #[cfg(table_format = "q16_16")]
                let raw_opt: Option<i128> = Some(storage as i128);

                #[cfg(table_format = "q32_32")]
                let raw_opt: Option<i128> = Some(storage as i128);

                #[cfg(table_format = "q64_64")]
                let raw_opt: Option<i128> = Some(storage);

                #[cfg(table_format = "q128_128")]
                let raw_opt: Option<i128> = if storage.fits_in_i128() { Some(storage.as_i128()) } else { None };

                #[cfg(table_format = "q256_256")]
                let raw_opt: Option<i128> = if storage.fits_in_i128() { Some(storage.as_i128()) } else { None };

                match raw_opt {
                    Some(raw) => CompactShadow::from_rational(raw, den),
                    None => CompactShadow::None,
                }
            } else {
                CompactShadow::None
            }
        };
        Ok(StackValue::Ternary(tier, storage, shadow))
    }

    /// Load mathematical constant with full profile-aware precision
    pub(crate) fn load_constant(&mut self, constant: ConstantId) -> Result<StackValue, OverflowDetected> {
        use crate::fixed_point::domains::symbolic::rational::mathematical_constants::SymbolicConstants;

        match constant {
            ConstantId::Pi => {
                Ok(StackValue::Symbolic(SymbolicConstants::pi_high_precision()))
            }
            ConstantId::E => {
                Ok(StackValue::Symbolic(SymbolicConstants::e_high_precision()))
            }
            ConstantId::Sqrt2 => {
                Ok(StackValue::Symbolic(SymbolicConstants::sqrt_2_high_precision()))
            }
            ConstantId::Phi => {
                Ok(StackValue::Symbolic(SymbolicConstants::golden_ratio_high_precision()))
            }
            ConstantId::EulerGamma => {
                // Euler-Mascheroni — no high-precision build.rs constant available
                let rational = RationalNumber::from_ratio(5772, 10000);
                Ok(StackValue::Symbolic(rational))
            }
        }
    }

}
