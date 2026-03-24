//! TQ1.9 — Compact Ternary Fixed-Point for Neural Network Inference
//!
//! **FORMAT**: 1 integer trit + 9 fractional trits = 10 trits total
//! **STORAGE**: i16 (max raw value 29524 < i16::MAX 32767)
//! **SCALE**: 3^9 = 19683 (fractional scaling factor)
//! **PRECISION**: ~4.3 decimal digits (uniform) — beats fp16's ~3.3 variable digits
//! **RANGE**: ±1.5 (exact: ±29524/19683 ≈ ±1.49987)
//!
//! Standalone type with conversion bridge to FixedPoint. NOT a deployment profile.
//! UGOD-aware: promotes to TernaryTier1 (i32/TQ8.8) on overflow.
//!
//! Designed for ternary neural network weight storage where values
//! cluster in [-1.5, +1.5] and uniform precision matters more than dynamic range.

use crate::fixed_point::core_types::errors::OverflowDetected;

/// Scale factor: 3^9 = 19683
pub const SCALE_TQ1_9: i16 = 19_683;

/// Scale factor as i32 for intermediate computations
const SCALE_TQ1_9_I32: i32 = 19_683;

/// Maximum raw value: (3^10 - 1) / 2 = 29524
const MAX_RAW: i16 = 29_524;

/// Minimum raw value
const MIN_RAW: i16 = -29_524;

/// TQ1.9 — Compact ternary fixed-point (1 integer trit + 9 fractional trits)
///
/// Stores `value * 3^9` as i16. Same byte cost as fp16 but ~30% more precision
/// for values in the [-1.5, +1.5] range typical of neural network weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TritQ1_9 {
    raw: i16,
}

impl TritQ1_9 {
    // ════════════════════════════════════════════════════════════════
    // Construction
    // ════════════════════════════════════════════════════════════════

    /// Create from raw scaled value (value * 3^9)
    #[inline]
    pub const fn from_raw(raw: i16) -> Self {
        Self { raw }
    }

    /// Get raw scaled value
    #[inline]
    pub const fn raw(&self) -> i16 {
        self.raw
    }

    /// Zero value
    #[inline]
    pub const fn zero() -> Self {
        Self { raw: 0 }
    }

    /// One (1.0 in TQ1.9 = 19683)
    #[inline]
    pub const fn one() -> Self {
        Self { raw: SCALE_TQ1_9 }
    }

    /// Negative one (-1.0 in TQ1.9 = -19683)
    #[inline]
    pub const fn neg_one() -> Self {
        Self { raw: -SCALE_TQ1_9 }
    }

    /// Maximum representable value (~1.49987)
    #[inline]
    pub const fn max_value() -> Self {
        Self { raw: MAX_RAW }
    }

    /// Minimum representable value (~-1.49987)
    #[inline]
    pub const fn min_value() -> Self {
        Self { raw: MIN_RAW }
    }

    /// Create from a rational value (numerator / denominator), rounding to nearest
    pub fn from_rational(num: i64, den: u64) -> Result<Self, OverflowDetected> {
        if den == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }
        // (num * SCALE) / den with rounding
        let scaled = num.checked_mul(SCALE_TQ1_9_I32 as i64)
            .ok_or(OverflowDetected::TierOverflow)?;
        let raw = if scaled >= 0 {
            (scaled + den as i64 / 2) / den as i64
        } else {
            (scaled - den as i64 / 2) / den as i64
        };
        if raw < MIN_RAW as i64 || raw > MAX_RAW as i64 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(Self { raw: raw as i16 })
    }

    /// Create from an integer
    pub fn from_integer(n: i32) -> Result<Self, OverflowDetected> {
        let raw = (n as i32).checked_mul(SCALE_TQ1_9_I32)
            .ok_or(OverflowDetected::TierOverflow)?;
        if raw < MIN_RAW as i32 || raw > MAX_RAW as i32 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(Self { raw: raw as i16 })
    }

    // ════════════════════════════════════════════════════════════════
    // Trit access (balanced ternary digits: {-1, 0, +1})
    // ════════════════════════════════════════════════════════════════

    /// Decompose into 10 balanced ternary trits [t0, t1, ..., t8, t9]
    /// where trits[9] is the integer trit and trits[0] is the least significant fractional trit.
    /// Value = sum(trits[i] * 3^i) for i in 0..10, divided by 3^9.
    ///
    /// Standard balanced ternary conversion: at each step, take remainder mod 3,
    /// then adjust to {-1, 0, +1} range by carrying into the next digit.
    pub fn to_trits(&self) -> [i8; 10] {
        let mut trits = [0i8; 10];
        let mut remaining = self.raw as i32;
        for i in 0..10 {
            // Euclidean remainder in [0, 3)
            let r = ((remaining % 3) + 3) % 3;
            remaining = (remaining - r) / 3;
            // Map {0, 1, 2} → {0, 1, -1} (balanced ternary)
            if r == 2 {
                trits[i] = -1;
                remaining += 1; // carry
            } else {
                trits[i] = r as i8;
            }
        }
        trits
    }

    /// Construct from 10 balanced ternary trits [t9, t8, ..., t1, t0]
    /// Each trit must be in {-1, 0, +1}.
    pub fn from_trits(trits: [i8; 10]) -> Result<Self, OverflowDetected> {
        let mut value: i32 = 0;
        let mut power: i32 = 1; // 3^0
        for i in 0..10 {
            let t = trits[i];
            if t < -1 || t > 1 {
                return Err(OverflowDetected::InvalidInput);
            }
            value += t as i32 * power;
            if i < 9 {
                power *= 3;
            }
        }
        if value < MIN_RAW as i32 || value > MAX_RAW as i32 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(Self { raw: value as i16 })
    }

    // ════════════════════════════════════════════════════════════════
    // Checked arithmetic (i32 intermediate for overflow safety)
    // ════════════════════════════════════════════════════════════════

    /// Checked addition. Returns Err on overflow.
    #[inline]
    pub fn checked_add(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let sum = self.raw as i32 + other.raw as i32;
        if sum < MIN_RAW as i32 || sum > MAX_RAW as i32 {
            Err(OverflowDetected::TierOverflow)
        } else {
            Ok(Self { raw: sum as i16 })
        }
    }

    /// Checked subtraction. Returns Err on overflow.
    #[inline]
    pub fn checked_sub(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let diff = self.raw as i32 - other.raw as i32;
        if diff < MIN_RAW as i32 || diff > MAX_RAW as i32 {
            Err(OverflowDetected::TierOverflow)
        } else {
            Ok(Self { raw: diff as i16 })
        }
    }

    /// Checked multiplication. Uses i32 intermediate ÷ 19683.
    /// Rounds to nearest for the ternary fixed-point result.
    #[inline]
    pub fn checked_mul(&self, other: &Self) -> Result<Self, OverflowDetected> {
        let product = self.raw as i32 * other.raw as i32;
        // Round-to-nearest division by scale
        let result = if product >= 0 {
            (product + SCALE_TQ1_9_I32 / 2) / SCALE_TQ1_9_I32
        } else {
            (product - SCALE_TQ1_9_I32 / 2) / SCALE_TQ1_9_I32
        };
        if result < MIN_RAW as i32 || result > MAX_RAW as i32 {
            Err(OverflowDetected::TierOverflow)
        } else {
            Ok(Self { raw: result as i16 })
        }
    }

    /// Checked division. Returns Err on division by zero or overflow.
    #[inline]
    pub fn checked_div(&self, other: &Self) -> Result<Self, OverflowDetected> {
        if other.raw == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }
        // (a * scale) / b with rounding
        let scaled = self.raw as i32 * SCALE_TQ1_9_I32;
        let result = if scaled >= 0 {
            (scaled + other.raw.abs() as i32 / 2) / other.raw as i32
        } else {
            (scaled - other.raw.abs() as i32 / 2) / other.raw as i32
        };
        if result < MIN_RAW as i32 || result > MAX_RAW as i32 {
            Err(OverflowDetected::TierOverflow)
        } else {
            Ok(Self { raw: result as i16 })
        }
    }

    /// Checked negation. Always succeeds for TQ1.9 (no i16::MIN asymmetry in range).
    #[inline]
    pub fn checked_neg(&self) -> Result<Self, OverflowDetected> {
        Ok(Self { raw: -self.raw })
    }

    /// Absolute value
    #[inline]
    pub fn abs(&self) -> Self {
        Self { raw: self.raw.abs() }
    }

    // ════════════════════════════════════════════════════════════════
    // UGOD: Promote to TernaryTier1 (TQ8.8, i32) on overflow
    // ════════════════════════════════════════════════════════════════

    /// Promote to TernaryTier1 (TQ8.8, scale 3^8 = 6561).
    /// TQ1.9 scale is 3^9, TQ8.8 scale is 3^8, so: tier1_raw = tq1_9_raw / 3
    /// (with rounding, since we lose 1 fractional trit precision but gain integer range).
    pub fn to_ternary_tier1(&self) -> super::ternary_types::TernaryTier1 {
        // 3^9 / 3^8 = 3, so divide by 3 with rounding
        let raw32 = self.raw as i32;
        let tier1_val = if raw32 >= 0 {
            (raw32 + 1) / 3 // round to nearest
        } else {
            (raw32 - 1) / 3
        };
        super::ternary_types::TernaryTier1::from_raw(tier1_val)
    }

    // ════════════════════════════════════════════════════════════════
    // Conversion: to/from f64 (convenience only, NOT for internal logic)
    // ════════════════════════════════════════════════════════════════

    /// Convert to f64 (convenience only — NOT for internal computation)
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.raw as f64 / SCALE_TQ1_9 as f64
    }

    /// Convert from f64 (convenience only — NOT for internal computation)
    pub fn from_f64(val: f64) -> Result<Self, OverflowDetected> {
        let scaled = (val * SCALE_TQ1_9 as f64).round() as i32;
        if scaled < MIN_RAW as i32 || scaled > MAX_RAW as i32 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(Self { raw: scaled as i16 })
    }

    // ════════════════════════════════════════════════════════════════
    // Utility
    // ════════════════════════════════════════════════════════════════

    /// Check if this value is zero
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.raw == 0
    }

    /// Check if this value is negative
    #[inline]
    pub const fn is_negative(&self) -> bool {
        self.raw < 0
    }

    /// Check if this value is positive
    #[inline]
    pub const fn is_positive(&self) -> bool {
        self.raw > 0
    }

    /// Number of trits in this format
    #[inline]
    pub const fn total_trits() -> usize {
        10
    }

    /// Number of fractional trits
    #[inline]
    pub const fn frac_trits() -> usize {
        9
    }

    /// Number of integer trits
    #[inline]
    pub const fn int_trits() -> usize {
        1
    }
}

impl core::fmt::Display for TritQ1_9 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Display as decimal fraction: raw / 19683
        let abs_raw = self.raw.unsigned_abs() as u32;
        let integer = abs_raw / SCALE_TQ1_9 as u32;
        let remainder = abs_raw % SCALE_TQ1_9 as u32;

        if self.raw < 0 {
            write!(f, "-")?;
        }

        if remainder == 0 {
            write!(f, "{}", integer)
        } else {
            // Compute 5 decimal digits of fractional part
            let frac = (remainder as u64 * 100_000) / SCALE_TQ1_9 as u64;
            let frac_str = format!("{:05}", frac);
            let trimmed = frac_str.trim_end_matches('0');
            write!(f, "{}.{}", integer, trimmed)
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Standard operator traits
// ════════════════════════════════════════════════════════════════

impl core::ops::Add for TritQ1_9 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.checked_add(&rhs).expect("TQ1.9 addition overflow")
    }
}

impl core::ops::Sub for TritQ1_9 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.checked_sub(&rhs).expect("TQ1.9 subtraction overflow")
    }
}

impl core::ops::Mul for TritQ1_9 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.checked_mul(&rhs).expect("TQ1.9 multiplication overflow")
    }
}

impl core::ops::Neg for TritQ1_9 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { raw: -self.raw }
    }
}

// ════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(TritQ1_9::zero().raw(), 0);
        assert_eq!(TritQ1_9::one().raw(), 19683);
        assert_eq!(TritQ1_9::neg_one().raw(), -19683);
        assert_eq!(TritQ1_9::max_value().raw(), 29524);
        assert_eq!(TritQ1_9::min_value().raw(), -29524);
    }

    #[test]
    fn test_from_integer() {
        let one = TritQ1_9::from_integer(1).unwrap();
        assert_eq!(one.raw(), 19683);
        let neg = TritQ1_9::from_integer(-1).unwrap();
        assert_eq!(neg.raw(), -19683);
        let zero = TritQ1_9::from_integer(0).unwrap();
        assert_eq!(zero.raw(), 0);
        // 2 overflows TQ1.9 range (max ~1.5)
        assert!(TritQ1_9::from_integer(2).is_err());
    }

    #[test]
    fn test_from_rational() {
        // 1/2 = 0.5 → 0.5 * 19683 = 9841.5 → rounds to 9842
        let half = TritQ1_9::from_rational(1, 2).unwrap();
        assert_eq!(half.raw(), 9842);
        // 1/3 = 0.333... → 0.333... * 19683 = 6561.0 exactly (3^9/3 = 3^8)
        let third = TritQ1_9::from_rational(1, 3).unwrap();
        assert_eq!(third.raw(), 6561);
    }

    #[test]
    fn test_checked_add() {
        let a = TritQ1_9::from_rational(1, 2).unwrap(); // 0.5
        let b = TritQ1_9::from_rational(1, 3).unwrap(); // 0.333...
        let sum = a.checked_add(&b).unwrap();
        // 0.5 + 0.333... ≈ 0.833... → raw = 9842 + 6561 = 16403
        assert_eq!(sum.raw(), 16403);
    }

    #[test]
    fn test_checked_mul() {
        let a = TritQ1_9::one();
        let b = TritQ1_9::one();
        let product = a.checked_mul(&b).unwrap();
        assert_eq!(product.raw(), 19683); // 1.0 * 1.0 = 1.0

        let half = TritQ1_9::from_rational(1, 2).unwrap();
        let result = half.checked_mul(&half).unwrap();
        // 0.5 * 0.5 = 0.25 → raw ≈ 0.25 * 19683 = 4920.75 → rounds to 4921
        assert_eq!(result.raw(), 4921);
    }

    #[test]
    fn test_checked_div_overflow() {
        // 1.0 / 0.5 = 2.0 → exceeds TQ1.9 range (±1.5), must overflow
        let one = TritQ1_9::one();
        let half = TritQ1_9::from_rational(1, 2).unwrap();
        assert!(one.checked_div(&half).is_err());
    }

    #[test]
    fn test_checked_div_identity() {
        let one = TritQ1_9::one();
        let result = one.checked_div(&one).unwrap();
        assert_eq!(result.raw(), 19683); // 1.0 / 1.0 = 1.0
    }

    #[test]
    fn test_checked_div_exact() {
        // 1/3 ÷ 1/3 = 1.0
        let third = TritQ1_9::from_rational(1, 3).unwrap(); // raw = 6561
        let result = third.checked_div(&third).unwrap();
        assert_eq!(result.raw(), 19683); // 1.0

        // 1.0 ÷ (-1.0) = -1.0
        let one = TritQ1_9::one();
        let neg = TritQ1_9::neg_one();
        let result = one.checked_div(&neg).unwrap();
        assert_eq!(result.raw(), -19683);
    }

    #[test]
    fn test_div_by_zero() {
        let one = TritQ1_9::one();
        let zero = TritQ1_9::zero();
        assert!(one.checked_div(&zero).is_err());
    }

    #[test]
    fn test_overflow_add() {
        let max = TritQ1_9::max_value();
        let one = TritQ1_9::one();
        assert!(max.checked_add(&one).is_err());
    }

    #[test]
    fn test_negation() {
        let pos = TritQ1_9::one();
        let neg = pos.checked_neg().unwrap();
        assert_eq!(neg.raw(), -19683);
        assert_eq!((-pos).raw(), -19683);
    }

    #[test]
    fn test_from_trits_roundtrip() {
        // Value 1.0 in balanced ternary: 1 * 3^9 = 19683
        // = 1*3^9 + 0*3^8 + ... + 0*3^0 → trits = [0,0,0,0,0,0,0,0,0,1]
        let one = TritQ1_9::one();
        let trits = one.to_trits();
        let reconstructed = TritQ1_9::from_trits(trits).unwrap();
        assert_eq!(one, reconstructed);
    }

    #[test]
    fn test_zero_trits() {
        let zero = TritQ1_9::zero();
        let trits = zero.to_trits();
        assert_eq!(trits, [0i8; 10]);
        let reconstructed = TritQ1_9::from_trits(trits).unwrap();
        assert_eq!(zero, reconstructed);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TritQ1_9::zero()), "0");
        assert_eq!(format!("{}", TritQ1_9::one()), "1");
        assert_eq!(format!("{}", TritQ1_9::neg_one()), "-1");
        let half = TritQ1_9::from_rational(1, 2).unwrap();
        let s = format!("{}", half);
        assert!(s.starts_with("0.5"), "half display: {}", s);
    }

    #[test]
    fn test_precision_vs_fp16() {
        // fp16 at value 1.0 has precision ~0.001 (2^-10)
        // TQ1.9 at value 1.0 has precision 1/19683 ≈ 0.0000508
        // TQ1.9 is ~20x more precise near 1.0
        let increment = 1.0 / SCALE_TQ1_9 as f64;
        assert!(increment < 0.001, "TQ1.9 increment {} should be < fp16 precision 0.001", increment);
    }

    #[test]
    fn test_tq1_9_fits_in_i16() {
        // Critical: max balanced ternary 10-trit value must fit in i16
        assert!(MAX_RAW <= i16::MAX);
        assert!(MIN_RAW >= i16::MIN);
    }

    #[test]
    fn test_ugod_promotion_to_tier1() {
        let val = TritQ1_9::one();
        let tier1 = val.to_ternary_tier1();
        // TQ1.9 one = 19683, TQ8.8 one = 6561
        // 19683 / 3 = 6561 exactly
        assert_eq!(tier1.raw(), 6561);
    }

    #[test]
    fn test_trits_roundtrip_exhaustive_samples() {
        // Verify roundtrip for a representative set of values
        let test_values: [i16; 11] = [
            0, 1, -1, SCALE_TQ1_9, -SCALE_TQ1_9,
            MAX_RAW, MIN_RAW,
            6561, -6561, // 1/3
            100, -100,
        ];
        for &raw in &test_values {
            let original = TritQ1_9::from_raw(raw);
            let trits = original.to_trits();
            // Verify all trits are in {-1, 0, 1}
            for &t in &trits {
                assert!(t >= -1 && t <= 1, "invalid trit {} for raw {}", t, raw);
            }
            let reconstructed = TritQ1_9::from_trits(trits).unwrap();
            assert_eq!(original, reconstructed,
                "roundtrip failed for raw {}: trits={:?}", raw, trits);
        }
    }

    #[test]
    fn test_mul_exact_integer_verification() {
        // Verify: 1/3 * 1/3 = 1/9
        // 1/3 raw = 6561 (= 3^8 = 19683/3)
        // product = 6561 * 6561 = 43046721
        // result = (43046721 + 9841) / 19683 = (43056562) / 19683 = 2187
        // 2187 = 3^7 = 19683/9, so raw 2187 = 1/9 exactly
        let third = TritQ1_9::from_rational(1, 3).unwrap();
        assert_eq!(third.raw(), 6561);
        let ninth = third.checked_mul(&third).unwrap();
        assert_eq!(ninth.raw(), 2187); // 3^7 = 1/9 of 3^9

        // Verify: 2187 / 19683 = 1/9 ≈ 0.11111...
        // Pure integer check: 2187 * 9 = 19683 ✓
        assert_eq!(2187 * 9, 19683);
    }

    #[test]
    fn test_add_sub_inverse() {
        let a = TritQ1_9::from_rational(2, 3).unwrap();
        let b = TritQ1_9::from_rational(1, 4).unwrap();
        let sum = a.checked_add(&b).unwrap();
        let back = sum.checked_sub(&b).unwrap();
        assert_eq!(back.raw(), a.raw());
    }

    #[test]
    fn test_scale_constant_is_power_of_3() {
        // Verify 3^9 = 19683
        let mut p: i32 = 1;
        for _ in 0..9 { p *= 3; }
        assert_eq!(p, SCALE_TQ1_9_I32);
        assert_eq!(SCALE_TQ1_9 as i32, SCALE_TQ1_9_I32);
    }

    #[test]
    fn test_max_raw_is_balanced_ternary_max() {
        // Max 10-trit balanced ternary = (3^10 - 1) / 2
        let three_pow_10: i32 = 59049; // 3^10
        let max = (three_pow_10 - 1) / 2;
        assert_eq!(max, MAX_RAW as i32);
        assert_eq!(max, 29524);
    }
}
