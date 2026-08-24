//! Balanced Ternary Types, Constructors, and Tier Promotion
//!
//! **MISSION**: Unified type definitions + runtime precision management for UGOD ternary
//! **ARCHITECTURE**: TQ10.10(i32) → TQ20.20(i64) → TQ40.40(i128) → TQ80.80(I256)
//!     → TQ160.160(I512) → TQ320.320(I1024)
//! **PRECISION**: Each tier uses minimal storage for its precision level
//! **INTEGRATION**: UGOD-compatible with runtime overflow delegation
//!
//! This file contains:
//! - Tier-specific storage types (TernaryTier1-6)
//! - Scale factor constants (powers of 3)
//! - TernaryTier enum and UniversalTernaryFixed struct
//! - Constructors, promotion, alignment, FASC boundary helpers
//! - Ternary-specific div3/mul3 operations
//!
//! Arithmetic UGOD methods (add, subtract, multiply, divide, negate) live in
//! separate operation files — NOT here.

use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::{I256, I512, I1024};

// ============================================================================
// TIER-SPECIFIC STORAGE TYPES
// ============================================================================

/// Tier 1: TQ10.10 - Compact Geometric Precision (8 integer + 8 fractional trits)
///
/// **STORAGE**: i32 (32 bits for ~25.4 bits of information)
/// **RANGE**: ±(3^10-1)/2 = ±29,524
/// **USE CASE**: Basic geometric coordinates, simple triangular operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TernaryTier1 {
    /// Raw value: actual_value × 3^10
    pub(crate) value: i32,
}

/// Tier 2: TQ20.20 - Standard Geometric Precision (16 integer + 16 fractional trits)
///
/// **STORAGE**: i64 (64 bits for ~50.7 bits of information)
/// **RANGE**: ±(3^20-1)/2 ≈ ±1.74×10^9
/// **USE CASE**: Standard geometric transformations, barycentric coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TernaryTier2 {
    /// Raw value: actual_value × 3^20
    pub(crate) value: i64,
}

/// Tier 3: TQ40.40 - Extended Geometric Precision (32 integer + 32 fractional trits)
///
/// **STORAGE**: i128 (128 bits for ~101.4 bits of information)
/// **RANGE**: ±(3^40-1)/2 ≈ ±6.1×10^18
/// **USE CASE**: Complex geometric calculations, high-precision transformations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TernaryTier3 {
    /// Raw value: actual_value × 3^40
    pub(crate) value: i128,
}

/// Tier 4: TQ80.80 - Ultra Geometric Precision (64 integer + 64 fractional trits)
///
/// **STORAGE**: I256 (256 bits for ~202.9 bits of information)
/// **RANGE**: ±(3^80-1)/2 ≈ ±7.4×10^37
/// **USE CASE**: Research-grade geometric computations, high-precision applications
#[derive(Debug, Clone)]
pub struct TernaryTier4 {
    /// Raw value: actual_value × 3^80
    pub(crate) value: I256,
}

/// Tier 5: TQ160.160 - Balanced Profile Precision (128 integer + 128 fractional trits)
///
/// **STORAGE**: I512 (512 bits for ~405.7 bits of information)
/// **RANGE**: ±(3^160-1)/2 ≈ ±1.1×10^76
/// **USE CASE**: Cross-domain operations matching balanced profile (38 decimals)
#[derive(Debug, Clone)]
pub struct TernaryTier5 {
    /// Raw value: actual_value × 3^160
    pub(crate) value: I512,
}

/// Tier 6: TQ320.320 - Scientific Profile Precision (256 integer + 256 fractional trits)
///
/// **STORAGE**: I1024 (1024 bits for ~811.4 bits of information)
/// **RANGE**: ±(3^320-1)/2 ≈ ±2.4×10^152
/// **USE CASE**: Cross-domain operations matching scientific profile (77 decimals)
#[derive(Debug, Clone)]
pub struct TernaryTier6 {
    /// Raw value: actual_value × 3^320
    pub(crate) value: I1024,
}

/// Raw ternary value -- preserves full precision for all tiers when crossing
/// the FASC boundary (avoids i128 truncation for Tier 4+).
#[derive(Debug, Clone)]
pub enum TernaryRaw {
    /// Tiers 1-3: value fits in i128
    Small(i128),
    /// Tier 4: I256 storage
    Medium(I256),
    /// Tier 5: I512 storage
    Large(I512),
    /// Tier 6: I1024 storage
    XLarge(I1024),
}

/// Unified ternary value enum (similar to DecimalValue)
#[derive(Debug, Clone)]
pub enum TernaryValue {
    Tier1(TernaryTier1),
    Tier2(TernaryTier2),
    Tier3(TernaryTier3),
    Tier4(TernaryTier4),
    Tier5(TernaryTier5),
    Tier6(TernaryTier6),
}

// ============================================================================
// SCALE FACTOR CONSTANTS (Powers of 3)
// ============================================================================

/// Scale factor for TQ10.10: 3^10 = 59,049
pub const SCALE_TQ10_10: i32 = 59_049;

/// Scale factor for TQ20.20: 3^20 = 3,486,784,401
pub const SCALE_TQ20_20: i64 = 3_486_784_401;

/// Scale factor for TQ40.40: 3^40 = 12,157,665,459,056,928,801
pub const SCALE_TQ40_40: i128 = 12_157_665_459_056_928_801;

// Scale factor for TQ80.80 needs I256 computation
fn scale_tq80_80() -> I256 {
    // 3^80 = compute at runtime or use precomputed constant
    compute_power_of_3_i256(80)
}

/// 3^160 as I512 for Tier 5 scaling (~2.2×10^76, needs ~254 bits, fits in I512's 511)
pub(crate) fn scale_tq160_160() -> I512 {
    compute_power_of_3_i512(160)
}

/// 3^160 as I1024 for Tier 5 multiplication intermediate
pub(crate) fn scale_tq160_160_i1024() -> I1024 {
    I1024::from_i512(scale_tq160_160())
}

/// 3^320 as I1024 for Tier 6 scaling (~6.9×10^152, needs ~508 bits, fits in I1024's 1023)
pub(crate) fn scale_tq320_320() -> I1024 {
    compute_power_of_3_i1024(320)
}

// ============================================================================
// HELPER FUNCTIONS (Powers of 3)
// ============================================================================

/// Compute 3^n as I256 for large powers
fn compute_power_of_3_i256(n: u32) -> I256 {
    let mut result = I256::from_u8(1);
    let three = I256::from_u8(3);
    for _ in 0..n {
        result = result * three;
    }
    result
}

/// Compute 3^n as I512 for Tier 5 scale factors
pub(crate) fn compute_power_of_3_i512(n: u32) -> I512 {
    let mut result = I512::from_i128(1);
    let three = I512::from_i128(3);
    for _ in 0..n {
        result = result * three;
    }
    result
}

/// Compute 3^n as I1024 for Tier 6 scale factors
pub(crate) fn compute_power_of_3_i1024(n: u32) -> I1024 {
    let mut result = I1024::from_i128(1);
    let three = I1024::from_i128(3);
    for _ in 0..n {
        result = result * three;
    }
    result
}

// ============================================================================
// TIER 1: TQ10.10 IMPLEMENTATION
// ============================================================================

impl TernaryTier1 {
    /// Create from raw i32 value
    pub const fn from_raw(value: i32) -> Self {
        Self { value }
    }

    /// Get raw value
    pub const fn raw(&self) -> i32 {
        self.value
    }

    /// Create zero
    pub const fn zero() -> Self {
        Self { value: 0 }
    }

    /// Create one (1.0 in TQ10.10 format)
    pub const fn one() -> Self {
        Self { value: SCALE_TQ10_10 }
    }

    /// Convert from integer
    pub fn from_integer(n: i16) -> Result<Self, ()> {
        match (n as i32).checked_mul(SCALE_TQ10_10) {
            Some(scaled) => Ok(Self { value: scaled }),
            None => Err(()), // Overflow
        }
    }

    /// Promote to Tier 2 (lossless)
    pub fn to_tier2(&self) -> TernaryTier2 {
        // Scale up by 3^10 to convert TQ10.10 to TQ20.20
        let scale_factor = SCALE_TQ20_20 / SCALE_TQ10_10 as i64; // 3^10
        TernaryTier2 {
            value: (self.value as i64) * scale_factor,
        }
    }

    /// Ternary right-shift: drop the lowest trit (nearest — tie-free since 3 is odd)
    pub fn div3(&self) -> Self {
        // 0.5.0: nearest (true ternary shift — drop lowest trit). 3 is odd
        // so no tie exists; |rem| >= 2 rounds away, |rem| <= 1 truncates.
        let q = self.value / 3;
        let r = self.value - q * 3;
        Self {
            value: if r >= 2 { q + 1 } else if r <= -2 { q - 1 } else { q },
        }
    }

    /// Exact multiplication by 3 (shift operation in ternary)
    pub fn mul3(&self) -> Result<Self, ()> {
        match self.value.checked_mul(3) {
            Some(result) => Ok(Self { value: result }),
            None => Err(()), // Overflow
        }
    }
}

// ============================================================================
// TIER 2: TQ20.20 IMPLEMENTATION
// ============================================================================

impl TernaryTier2 {
    /// Create from raw i64 value
    pub const fn from_raw(value: i64) -> Self {
        Self { value }
    }

    /// Get raw value
    pub const fn raw(&self) -> i64 {
        self.value
    }

    /// Create zero
    pub const fn zero() -> Self {
        Self { value: 0 }
    }

    /// Create one (1.0 in TQ20.20 format)
    pub const fn one() -> Self {
        Self { value: SCALE_TQ20_20 }
    }

    /// Convert from integer
    pub fn from_integer(n: i32) -> Result<Self, ()> {
        match (n as i64).checked_mul(SCALE_TQ20_20) {
            Some(scaled) => Ok(Self { value: scaled }),
            None => Err(()), // Overflow
        }
    }

    /// Promote to Tier 3 (lossless)
    pub fn to_tier3(&self) -> TernaryTier3 {
        // Scale up by 3^20 to convert TQ20.20 to TQ40.40
        let scale_factor = SCALE_TQ40_40 / SCALE_TQ20_20 as i128; // 3^20
        TernaryTier3 {
            value: (self.value as i128) * scale_factor,
        }
    }

    /// Demote to Tier 1 (may lose precision)
    pub fn to_tier1(&self) -> Result<TernaryTier1, ()> {
        let scale_factor = SCALE_TQ20_20 / SCALE_TQ10_10 as i64; // 3^8
        let scaled = self.value / scale_factor;

        if scaled >= i32::MIN as i64 && scaled <= i32::MAX as i64 {
            Ok(TernaryTier1 {
                value: scaled as i32,
            })
        } else {
            Err(()) // Value too large for Tier 1
        }
    }

    /// Exact division by 3
    pub fn div3(&self) -> Self {
        // 0.5.0: nearest (true ternary shift — drop lowest trit). 3 is odd
        // so no tie exists; |rem| >= 2 rounds away, |rem| <= 1 truncates.
        let q = self.value / 3;
        let r = self.value - q * 3;
        Self {
            value: if r >= 2 { q + 1 } else if r <= -2 { q - 1 } else { q },
        }
    }

    /// Exact multiplication by 3
    pub fn mul3(&self) -> Result<Self, ()> {
        match self.value.checked_mul(3) {
            Some(result) => Ok(Self { value: result }),
            None => Err(()), // Overflow
        }
    }
}

// ============================================================================
// TIER 3: TQ40.40 IMPLEMENTATION
// ============================================================================

impl TernaryTier3 {
    /// Create from raw i128 value
    pub const fn from_raw(value: i128) -> Self {
        Self { value }
    }

    /// Get raw value
    pub const fn raw(&self) -> i128 {
        self.value
    }

    /// Create zero
    pub const fn zero() -> Self {
        Self { value: 0 }
    }

    /// Create one (1.0 in TQ40.40 format)
    pub const fn one() -> Self {
        Self { value: SCALE_TQ40_40 }
    }

    /// Convert from integer
    pub fn from_integer(n: i64) -> Result<Self, ()> {
        match (n as i128).checked_mul(SCALE_TQ40_40) {
            Some(scaled) => Ok(Self { value: scaled }),
            None => Err(()), // Overflow
        }
    }

    /// Promote to Tier 4 (lossless)
    pub fn to_tier4(&self) -> TernaryTier4 {
        let base = I256::from_i128(self.value);
        let scale_factor = compute_power_of_3_i256(40); // 3^40 as I256 (tier 3->4 delta)
        TernaryTier4 {
            value: base * scale_factor,
        }
    }

    /// Demote to Tier 2 (may lose precision)
    pub fn to_tier2(&self) -> Result<TernaryTier2, ()> {
        let scale_factor = SCALE_TQ40_40 / SCALE_TQ20_20 as i128; // 3^16
        let scaled = self.value / scale_factor;

        if scaled >= i64::MIN as i128 && scaled <= i64::MAX as i128 {
            Ok(TernaryTier2 {
                value: scaled as i64,
            })
        } else {
            Err(()) // Value too large for Tier 2
        }
    }

    /// Exact division by 3
    pub fn div3(&self) -> Self {
        // 0.5.0: nearest (true ternary shift — drop lowest trit). 3 is odd
        // so no tie exists; |rem| >= 2 rounds away, |rem| <= 1 truncates.
        let q = self.value / 3;
        let r = self.value - q * 3;
        Self {
            value: if r >= 2 { q + 1 } else if r <= -2 { q - 1 } else { q },
        }
    }

    /// Exact multiplication by 3
    pub fn mul3(&self) -> Result<Self, ()> {
        match self.value.checked_mul(3) {
            Some(result) => Ok(Self { value: result }),
            None => Err(()), // Overflow
        }
    }
}

// ============================================================================
// TIER 4: TQ80.80 IMPLEMENTATION
// ============================================================================

impl TernaryTier4 {
    /// Create from raw I256 value
    pub fn from_raw(value: I256) -> Self {
        Self { value }
    }

    /// Get raw value
    pub fn raw(&self) -> &I256 {
        &self.value
    }

    /// Create zero
    pub fn zero() -> Self {
        Self {
            value: I256::zero(),
        }
    }

    /// Create one (1.0 in TQ80.80 format)
    pub fn one() -> Self {
        Self {
            value: scale_tq80_80(),
        }
    }

    /// Convert from integer
    pub fn from_integer(n: i128) -> Self {
        let base = I256::from_i128(n);
        let scale = scale_tq80_80();
        Self {
            value: base * scale,
        }
    }

    /// Promote to Tier 5 (lossless)
    pub fn to_tier5(&self) -> TernaryTier5 {
        let base = I512::from_i256(self.value);
        let scale_factor = compute_power_of_3_i512(80); // 3^80 as I512 (tier 4->5 delta)
        TernaryTier5 {
            value: base * scale_factor,
        }
    }

    /// Demote to Tier 3 (may lose precision)
    pub fn to_tier3(&self) -> Result<TernaryTier3, ()> {
        let scale_factor = compute_power_of_3_i256(40); // 3^40 (tier 4->3 delta)
        let scaled = &self.value / &scale_factor;

        if scaled.fits_in_i128() {
            Ok(TernaryTier3 {
                value: scaled.as_i128(),
            })
        } else {
            Err(()) // Value too large for Tier 3
        }
    }

    /// Exact division by 3
    pub fn div3(&self) -> Self {
        Self {
            value: {
                // Nearest ternary shift (0.5.0) — see the native-tier div3.
                let three = I256::from_u8(3);
                let q = &self.value / &three;
                let r = self.value - q * three;
                let two = I256::from_u8(2);
                if r >= two { q + I256::from_u8(1) }
                else if r <= -two { q - I256::from_u8(1) }
                else { q }
            },
        }
    }

    /// Exact multiplication by 3
    pub fn mul3(&self) -> Self {
        Self {
            value: &self.value * &I256::from_u8(3),
        }
    }
}

// ============================================================================
// TIER 5: TQ160.160 IMPLEMENTATION
// ============================================================================

impl TernaryTier5 {
    /// Create from raw I512 value
    pub fn from_raw(value: I512) -> Self {
        Self { value }
    }

    /// Get raw value
    pub fn raw(&self) -> &I512 {
        &self.value
    }

    /// Create zero
    pub fn zero() -> Self {
        Self {
            value: I512::zero(),
        }
    }

    /// Create one (1.0 in TQ160.160 format)
    pub fn one() -> Self {
        Self {
            value: scale_tq160_160(),
        }
    }

    /// Convert from integer
    pub fn from_integer(n: i128) -> Self {
        let base = I512::from_i128(n);
        let scale = scale_tq160_160();
        Self {
            value: base * scale,
        }
    }

    /// Promote to Tier 6 (lossless)
    pub fn to_tier6(&self) -> TernaryTier6 {
        let base = I1024::from_i512(self.value);
        let scale_factor = I1024::from_i512(compute_power_of_3_i512(160)); // 3^160 as I1024 (tier 5->6 delta)
        TernaryTier6 {
            value: base * scale_factor,
        }
    }

    /// Demote to Tier 4 (may lose precision)
    pub fn to_tier4(&self) -> Result<TernaryTier4, ()> {
        let scale_factor = compute_power_of_3_i512(80); // 3^80 (tier 5->4 delta)
        let scaled = self.value / scale_factor;

        if scaled.fits_in_i256() {
            Ok(TernaryTier4 {
                value: scaled.as_i256(),
            })
        } else {
            Err(()) // Value too large for Tier 4
        }
    }

    /// Exact division by 3
    pub fn div3(&self) -> Self {
        Self {
            value: {
                // Nearest ternary shift (0.5.0) — see the native-tier div3.
                let three = I512::from_i128(3);
                let q = self.value / three;
                let r = self.value - q * three;
                let two = I512::from_i128(2);
                if r >= two { q + I512::from_i128(1) }
                else if r <= -two { q - I512::from_i128(1) }
                else { q }
            },
        }
    }

    /// Exact multiplication by 3
    pub fn mul3(&self) -> Self {
        Self {
            value: self.value * I512::from_i128(3),
        }
    }
}

// ============================================================================
// TIER 6: TQ320.320 IMPLEMENTATION
// ============================================================================

impl TernaryTier6 {
    /// Create from raw I1024 value
    pub fn from_raw(value: I1024) -> Self {
        Self { value }
    }

    /// Get raw value
    pub fn raw(&self) -> &I1024 {
        &self.value
    }

    /// Create zero
    pub fn zero() -> Self {
        Self {
            value: I1024::zero(),
        }
    }

    /// Create one (1.0 in TQ320.320 format)
    pub fn one() -> Self {
        Self {
            value: scale_tq320_320(),
        }
    }

    /// Convert from integer
    pub fn from_integer(n: i128) -> Self {
        let base = I1024::from_i128(n);
        let scale = scale_tq320_320();
        Self {
            value: base * scale,
        }
    }

    /// Demote to Tier 5 (may lose precision)
    pub fn to_tier5(&self) -> Result<TernaryTier5, ()> {
        let scale_factor = I1024::from_i512(compute_power_of_3_i512(160)); // 3^160 (tier 6->5 delta)
        let scaled = self.value / scale_factor;

        if scaled.fits_in_i512() {
            Ok(TernaryTier5 {
                value: scaled.as_i512(),
            })
        } else {
            Err(()) // Value too large for Tier 5
        }
    }

    /// Exact division by 3
    pub fn div3(&self) -> Self {
        Self {
            value: {
                // Nearest ternary shift (0.5.0) — see the native-tier div3.
                let three = I1024::from_i128(3);
                let q = self.value / three;
                let r = self.value - q * three;
                let two = I1024::from_i128(2);
                if r >= two { q + I1024::from_i128(1) }
                else if r <= -two { q - I1024::from_i128(1) }
                else { q }
            },
        }
    }

    /// Exact multiplication by 3
    pub fn mul3(&self) -> Self {
        Self {
            value: self.value * I1024::from_i128(3),
        }
    }
}

// ============================================================================
// CONVERSION FUNCTIONS BETWEEN TIERS
// ============================================================================

// ============================================================================
// TERNARY TIER ENUM + UNIVERSAL TERNARY FIXED STRUCT
// ============================================================================

/// UGOD-compatible ternary precision tiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryTier {
    Tier1,  // TQ10.10    - 16 trits (i32 storage)
    Tier2,  // TQ20.20  - 32 trits (i64 storage)
    Tier3,  // TQ40.40  - 64 trits (i128 storage)
    Tier4,  // TQ80.80  - 128 trits (I256 storage)
    Tier5,  // TQ160.160 - 256 trits (I512 storage)
    Tier6,  // TQ320.320 - 512 trits (I1024 storage, never fails)
}

/// **UniversalTernaryFixed** - UGOD-compatible ternary type with runtime precision
///
/// **RUNTIME PRECISION**: Automatically promotes between tiers on overflow
/// **TIER PROGRESSION**: TQ10.10 → TQ20.20 → TQ40.40 → TQ80.80 → TQ160.160 → TQ320.320
/// **UGOD COMPATIBLE**: Implements UniversalTieredArithmetic trait
/// **GEOMETRIC READY**: Handles triangular coordinates with adaptive precision
#[derive(Debug, Clone)]
pub struct UniversalTernaryFixed {
    pub(crate) value: TernaryValue,
    pub(crate) current_tier: TernaryTier,
}

// ============================================================================
// CORE IMPLEMENTATION
// ============================================================================

impl UniversalTernaryFixed {
    /// Create from string with automatic precision detection
    pub fn from_str(input: &str) -> Result<Self, OverflowDetected> {
        // Sign is handled here, once: "-0.5" used to parse as +0.5 because
        // "-0".parse::<i64>() loses the sign before the fraction is applied.
        // Parse the magnitude, then negate — identical results for inputs
        // with a nonzero integer part, correct results for "-0.x".
        let trimmed = input.trim();
        if let Some(rest) = trimmed.strip_prefix('-') {
            if rest.starts_with('-') {
                return Err(OverflowDetected::InvalidInput);
            }
            // Parse the magnitude but thread the FINAL sign into the
            // conversion, so the ties-toward-+∞ boundary rule (contract
            // §5, 0.5.0 unification) sees the signed value: a positive
            // tie rounds the magnitude up, a negative tie rounds the
            // magnitude down (toward zero), then the result is negated.
            return Self::from_str_signed(rest, true)?.negate();
        }
        Self::from_str_signed(trimmed, false)
    }

    fn from_str_signed(input: &str, final_neg: bool) -> Result<Self, OverflowDetected> {
        // Parse the decimal string and convert to balanced ternary
        let (integer_part, fractional_part) = Self::parse_decimal_string(input)?;

        // Try to fit in most efficient tier first
        if let Ok(value) = Self::try_create_tier1(integer_part, &fractional_part, final_neg) {
            return Ok(value);
        }
        if let Ok(value) = Self::try_create_tier2(integer_part, &fractional_part, final_neg) {
            return Ok(value);
        }
        if let Ok(value) = Self::try_create_tier3(integer_part, &fractional_part, final_neg) {
            return Ok(value);
        }

        // Fallback to maximum precision tier (never fails)
        Ok(Self::create_tier4(integer_part, &fractional_part, final_neg))
    }

    /// Create zero value efficiently
    pub fn zero() -> Self {
        Self {
            value: TernaryValue::Tier1(TernaryTier1::zero()),
            current_tier: TernaryTier::Tier1,
        }
    }

    /// Create one value efficiently
    pub fn one() -> Self {
        Self {
            value: TernaryValue::Tier1(TernaryTier1::one()),
            current_tier: TernaryTier::Tier1,
        }
    }

    /// Create from integer with automatic tier selection
    pub fn from_integer(value: i64) -> Result<Self, OverflowDetected> {
        // Try Tier 1 first (most efficient)
        if value >= i16::MIN as i64 && value <= i16::MAX as i64 {
            if let Ok(tier1) = TernaryTier1::from_integer(value as i16) {
                return Ok(Self {
                    value: TernaryValue::Tier1(tier1),
                    current_tier: TernaryTier::Tier1,
                });
            }
        }

        // Try Tier 2
        if value >= i32::MIN as i64 && value <= i32::MAX as i64 {
            if let Ok(tier2) = TernaryTier2::from_integer(value as i32) {
                return Ok(Self {
                    value: TernaryValue::Tier2(tier2),
                    current_tier: TernaryTier::Tier2,
                });
            }
        }

        // Try Tier 3
        if let Ok(tier3) = TernaryTier3::from_integer(value) {
            return Ok(Self {
                value: TernaryValue::Tier3(tier3),
                current_tier: TernaryTier::Tier3,
            });
        }

        // Fallback to Tier 4 (never fails)
        Ok(Self {
            value: TernaryValue::Tier4(TernaryTier4::from_integer(value as i128)),
            current_tier: TernaryTier::Tier4,
        })
    }

    /// Get current tier for UGOD coordination
    pub fn current_tier(&self) -> TernaryTier {
        self.current_tier
    }

    /// Promote to higher tier (UGOD requirement)
    pub fn promote_to_tier(&self, target_tier: TernaryTier) -> Result<Self, OverflowDetected> {
        if target_tier as u8 <= self.current_tier as u8 {
            return Ok(self.clone()); // Already at sufficient tier
        }

        match target_tier {
            TernaryTier::Tier2 => self.promote_to_tier2(),
            TernaryTier::Tier3 => self.promote_to_tier3(),
            TernaryTier::Tier4 => Ok(self.promote_to_tier4()),
            TernaryTier::Tier5 => Ok(self.promote_to_tier5()),
            TernaryTier::Tier6 => Ok(self.promote_to_tier6()),
            TernaryTier::Tier1 => Err(OverflowDetected::InvalidInput), // Cannot demote
        }
    }

    /// Check if values can be combined at a common tier
    pub fn align_to_common_tier(&self, other: &Self) -> (Self, Self) {
        let common_tier = std::cmp::max(self.current_tier as u8, other.current_tier as u8);
        let common_tier_enum = match common_tier {
            0 => TernaryTier::Tier1,
            1 => TernaryTier::Tier2,
            2 => TernaryTier::Tier3,
            3 => TernaryTier::Tier4,
            4 => TernaryTier::Tier5,
            _ => TernaryTier::Tier6,
        };

        let aligned_self = self.promote_to_tier(common_tier_enum).unwrap_or_else(|_| self.clone());
        let aligned_other = other.promote_to_tier(common_tier_enum).unwrap_or_else(|_| other.clone());

        (aligned_self, aligned_other)
    }

    /// Create UniversalTernaryFixed from tier and i128 value (for FASC integration)
    ///
    /// **PURPOSE**: Convert StackValue::Ternary(tier, value) back to UniversalTernaryFixed
    /// **WARNING**: Tier 4+ truncates to i128 - use from_tier_raw() for full precision
    pub fn from_tier_value(tier: u8, value: i128) -> Result<Self, OverflowDetected> {
        match tier {
            1 => {
                let truncated = value as i32;
                Ok(Self {
                    value: TernaryValue::Tier1(TernaryTier1::from_raw(truncated)),
                    current_tier: TernaryTier::Tier1,
                })
            }
            2 => {
                let truncated = value as i64;
                Ok(Self {
                    value: TernaryValue::Tier2(TernaryTier2::from_raw(truncated)),
                    current_tier: TernaryTier::Tier2,
                })
            }
            3 => {
                Ok(Self {
                    value: TernaryValue::Tier3(TernaryTier3::from_raw(value)),
                    current_tier: TernaryTier::Tier3,
                })
            }
            4 => {
                let expanded = I256::from_i128(value);
                Ok(Self {
                    value: TernaryValue::Tier4(TernaryTier4::from_raw(expanded)),
                    current_tier: TernaryTier::Tier4,
                })
            }
            5 => {
                let expanded = I512::from_i128(value);
                Ok(Self {
                    value: TernaryValue::Tier5(TernaryTier5::from_raw(expanded)),
                    current_tier: TernaryTier::Tier5,
                })
            }
            6 => {
                let expanded = I1024::from_i128(value);
                Ok(Self {
                    value: TernaryValue::Tier6(TernaryTier6::from_raw(expanded)),
                    current_tier: TernaryTier::Tier6,
                })
            }
            _ => Err(OverflowDetected::InvalidInput)
        }
    }

    /// Extract tier and i128 value for FASC storage (legacy -- truncates Tier 4+)
    ///
    /// **WARNING**: Use to_tier_raw() for full precision.
    pub fn to_tier_value(&self) -> (u8, i128) {
        match &self.value {
            TernaryValue::Tier1(tier1) => (1, tier1.raw() as i128),
            TernaryValue::Tier2(tier2) => (2, tier2.raw() as i128),
            TernaryValue::Tier3(tier3) => (3, tier3.raw()),
            TernaryValue::Tier4(tier4) => (4, tier4.raw().as_i128()),
            TernaryValue::Tier5(tier5) => (5, tier5.raw().as_i128()),
            TernaryValue::Tier6(tier6) => (6, tier6.raw().as_i128()),
        }
    }

    /// Create UniversalTernaryFixed from tier and raw value -- NO truncation
    ///
    /// **PURPOSE**: Full-precision FASC boundary crossing for all tiers
    pub fn from_tier_raw(tier: u8, raw: TernaryRaw) -> Result<Self, OverflowDetected> {
        match (tier, raw) {
            // Checked narrowing (0.5.0): the old bare `as i32`/`as i64`
            // casts silently wrapped oversized raws into garbage values.
            (1, TernaryRaw::Small(v)) => Ok(Self {
                value: TernaryValue::Tier1(TernaryTier1::from_raw(
                    i32::try_from(v).map_err(|_| OverflowDetected::TierOverflow)?,
                )),
                current_tier: TernaryTier::Tier1,
            }),
            (2, TernaryRaw::Small(v)) => Ok(Self {
                value: TernaryValue::Tier2(TernaryTier2::from_raw(
                    i64::try_from(v).map_err(|_| OverflowDetected::TierOverflow)?,
                )),
                current_tier: TernaryTier::Tier2,
            }),
            (3, TernaryRaw::Small(v)) => Ok(Self {
                value: TernaryValue::Tier3(TernaryTier3::from_raw(v)),
                current_tier: TernaryTier::Tier3,
            }),
            (4, TernaryRaw::Medium(v)) => Ok(Self {
                value: TernaryValue::Tier4(TernaryTier4::from_raw(v)),
                current_tier: TernaryTier::Tier4,
            }),
            (5, TernaryRaw::Large(v)) => Ok(Self {
                value: TernaryValue::Tier5(TernaryTier5::from_raw(v)),
                current_tier: TernaryTier::Tier5,
            }),
            (6, TernaryRaw::XLarge(v)) => Ok(Self {
                value: TernaryValue::Tier6(TernaryTier6::from_raw(v)),
                current_tier: TernaryTier::Tier6,
            }),
            _ => Err(OverflowDetected::InvalidInput),
        }
    }

    /// Extract tier and raw value -- NO truncation for any tier
    ///
    /// **PURPOSE**: Full-precision FASC boundary crossing for all tiers
    pub fn to_tier_raw(&self) -> (u8, TernaryRaw) {
        match &self.value {
            TernaryValue::Tier1(tier1) => (1, TernaryRaw::Small(tier1.raw() as i128)),
            TernaryValue::Tier2(tier2) => (2, TernaryRaw::Small(tier2.raw() as i128)),
            TernaryValue::Tier3(tier3) => (3, TernaryRaw::Small(tier3.raw())),
            TernaryValue::Tier4(tier4) => (4, TernaryRaw::Medium(tier4.raw().clone())),
            TernaryValue::Tier5(tier5) => (5, TernaryRaw::Large(tier5.raw().clone())),
            TernaryValue::Tier6(tier6) => (6, TernaryRaw::XLarge(tier6.raw().clone())),
        }
    }
}

// ============================================================================
// TIER CREATION METHODS
// ============================================================================

impl UniversalTernaryFixed {
    /// Try to create in Tier 1 (TQ10.10) - most efficient
    fn try_create_tier1(integer_part: i64, fractional_part: &str, final_neg: bool) -> Result<Self, OverflowDetected> {
        // Check if value fits in Tier 1 range
        if integer_part.abs() > 29_524 || fractional_part.len() > 10 {
            return Err(OverflowDetected::TierOverflow);
        }

        let ternary_value = Self::convert_decimal_to_ternary_tier1(integer_part, fractional_part, final_neg)?;

        Ok(Self {
            value: TernaryValue::Tier1(TernaryTier1::from_raw(ternary_value)),
            current_tier: TernaryTier::Tier1,
        })
    }

    /// Try to create in Tier 2 (TQ20.20)
    fn try_create_tier2(integer_part: i64, fractional_part: &str, final_neg: bool) -> Result<Self, OverflowDetected> {
        // Check if value fits in Tier 2 range
        if integer_part.abs() > 1_743_392_200 || fractional_part.len() > 20 {
            return Err(OverflowDetected::TierOverflow);
        }

        let ternary_value = Self::convert_decimal_to_ternary_tier2(integer_part, fractional_part, final_neg)?;

        Ok(Self {
            value: TernaryValue::Tier2(TernaryTier2::from_raw(ternary_value)),
            current_tier: TernaryTier::Tier2,
        })
    }

    /// Try to create in Tier 3 (TQ40.40)
    fn try_create_tier3(integer_part: i64, fractional_part: &str, final_neg: bool) -> Result<Self, OverflowDetected> {
        // Most values should fit in Tier 3
        let ternary_value = Self::convert_decimal_to_ternary_tier3(integer_part, fractional_part, final_neg)?;

        Ok(Self {
            value: TernaryValue::Tier3(TernaryTier3::from_raw(ternary_value)),
            current_tier: TernaryTier::Tier3,
        })
    }

    /// Create in Tier 4 (TQ80.80) - never fails, maximum precision
    fn create_tier4(integer_part: i64, fractional_part: &str, final_neg: bool) -> Self {
        let ternary_value = Self::convert_decimal_to_ternary_tier4(integer_part, fractional_part, final_neg);

        Self {
            value: TernaryValue::Tier4(TernaryTier4::from_raw(ternary_value)),
            current_tier: TernaryTier::Tier4,
        }
    }
}

// ============================================================================
// TIER PROMOTION METHODS (UGOD CORE FUNCTIONALITY)
// ============================================================================

impl UniversalTernaryFixed {
    /// Promote to Tier 2 (TQ20.20)
    pub(crate) fn promote_to_tier2(&self) -> Result<Self, OverflowDetected> {
        match &self.value {
            TernaryValue::Tier1(tier1) => {
                let promoted = tier1.to_tier2();
                Ok(Self {
                    value: TernaryValue::Tier2(promoted),
                    current_tier: TernaryTier::Tier2,
                })
            }
            _ => Ok(self.clone()) // Already at higher tier
        }
    }

    /// Promote to Tier 3 (TQ40.40)
    pub(crate) fn promote_to_tier3(&self) -> Result<Self, OverflowDetected> {
        let tier2_value = if matches!(&self.value, TernaryValue::Tier1(_)) {
            self.promote_to_tier2()?
        } else {
            self.clone()
        };

        match &tier2_value.value {
            TernaryValue::Tier2(tier2) => {
                let promoted = tier2.to_tier3();
                Ok(Self {
                    value: TernaryValue::Tier3(promoted),
                    current_tier: TernaryTier::Tier3,
                })
            }
            _ => Ok(tier2_value) // Already at higher tier
        }
    }

    /// Promote to Tier 4 (TQ80.80)
    pub(crate) fn promote_to_tier4(&self) -> Self {
        let tier3_value = self.promote_to_tier3().unwrap_or_else(|_| self.clone());

        match &tier3_value.value {
            TernaryValue::Tier3(tier3) => {
                let promoted = tier3.to_tier4();
                Self {
                    value: TernaryValue::Tier4(promoted),
                    current_tier: TernaryTier::Tier4,
                }
            }
            TernaryValue::Tier4(_) | TernaryValue::Tier5(_) | TernaryValue::Tier6(_) => tier3_value,
            _ => unreachable!("promote_to_tier3 should guarantee Tier3 or higher")
        }
    }

    /// Promote to Tier 5 (TQ160.160)
    pub(crate) fn promote_to_tier5(&self) -> Self {
        let tier4_value = self.promote_to_tier4();

        match &tier4_value.value {
            TernaryValue::Tier4(tier4) => {
                let promoted = tier4.to_tier5();
                Self {
                    value: TernaryValue::Tier5(promoted),
                    current_tier: TernaryTier::Tier5,
                }
            }
            TernaryValue::Tier5(_) | TernaryValue::Tier6(_) => tier4_value,
            _ => unreachable!("promote_to_tier4 should guarantee Tier4 or higher")
        }
    }

    /// Promote to Tier 6 (TQ320.320) - never fails, maximum ternary precision
    pub(crate) fn promote_to_tier6(&self) -> Self {
        let tier5_value = self.promote_to_tier5();

        match &tier5_value.value {
            TernaryValue::Tier5(tier5) => {
                let promoted = tier5.to_tier6();
                Self {
                    value: TernaryValue::Tier6(promoted),
                    current_tier: TernaryTier::Tier6,
                }
            }
            TernaryValue::Tier6(_) => tier5_value,
            _ => unreachable!("promote_to_tier5 should guarantee Tier5 or higher")
        }
    }
}

// ============================================================================
// TERNARY-SPECIFIC OPERATIONS: div3 / mul3
// ============================================================================

impl UniversalTernaryFixed {
    /// Division by 3 - exact operation in balanced ternary
    pub fn div3(&self) -> Self {
        match &self.value {
            TernaryValue::Tier1(tier1) => Self {
                value: TernaryValue::Tier1(tier1.div3()),
                current_tier: TernaryTier::Tier1,
            },
            TernaryValue::Tier2(tier2) => Self {
                value: TernaryValue::Tier2(tier2.div3()),
                current_tier: TernaryTier::Tier2,
            },
            TernaryValue::Tier3(tier3) => Self {
                value: TernaryValue::Tier3(tier3.div3()),
                current_tier: TernaryTier::Tier3,
            },
            TernaryValue::Tier4(tier4) => Self {
                value: TernaryValue::Tier4(tier4.div3()),
                current_tier: TernaryTier::Tier4,
            },
            TernaryValue::Tier5(tier5) => Self {
                value: TernaryValue::Tier5(tier5.div3()),
                current_tier: TernaryTier::Tier5,
            },
            TernaryValue::Tier6(tier6) => Self {
                value: TernaryValue::Tier6(tier6.div3()),
                current_tier: TernaryTier::Tier6,
            },
        }
    }

    /// Multiplication by 3 - exact operation in balanced ternary
    pub fn mul3(&self) -> Result<Self, OverflowDetected> {
        match &self.value {
            TernaryValue::Tier1(tier1) => {
                match tier1.mul3() {
                    Ok(result) => Ok(Self {
                        value: TernaryValue::Tier1(result),
                        current_tier: TernaryTier::Tier1,
                    }),
                    Err(_) => {
                        let promoted = self.promote_to_tier2()?;
                        promoted.mul3()
                    }
                }
            }
            TernaryValue::Tier2(tier2) => {
                match tier2.mul3() {
                    Ok(result) => Ok(Self {
                        value: TernaryValue::Tier2(result),
                        current_tier: TernaryTier::Tier2,
                    }),
                    Err(_) => {
                        let promoted = self.promote_to_tier3()?;
                        promoted.mul3()
                    }
                }
            }
            TernaryValue::Tier3(tier3) => {
                match tier3.mul3() {
                    Ok(result) => Ok(Self {
                        value: TernaryValue::Tier3(result),
                        current_tier: TernaryTier::Tier3,
                    }),
                    Err(_) => {
                        let promoted = self.promote_to_tier4();
                        promoted.mul3()
                    }
                }
            }
            TernaryValue::Tier4(tier4) => {
                Ok(Self {
                    value: TernaryValue::Tier4(tier4.mul3()),
                    current_tier: TernaryTier::Tier4,
                })
            }
            TernaryValue::Tier5(tier5) => {
                Ok(Self {
                    value: TernaryValue::Tier5(tier5.mul3()),
                    current_tier: TernaryTier::Tier5,
                })
            }
            TernaryValue::Tier6(tier6) => {
                Ok(Self {
                    value: TernaryValue::Tier6(tier6.mul3()),
                    current_tier: TernaryTier::Tier6,
                })
            }
        }
    }
}

// ============================================================================
// HELPER METHODS: Parsing and Decimal-to-Ternary Conversion
// ============================================================================

impl UniversalTernaryFixed {
    /// Parse decimal string into (integer_part, fractional_part)
    fn parse_decimal_string(input: &str) -> Result<(i64, &str), OverflowDetected> {
        let trimmed = input.trim();

        if let Some(dot_pos) = trimmed.find('.') {
            let integer_str = &trimmed[..dot_pos];
            let fractional_str = &trimmed[dot_pos + 1..];

            let integer_part = integer_str.parse::<i64>()
                .map_err(|_| OverflowDetected::InvalidInput)?;

            Ok((integer_part, fractional_str))
        } else {
            let integer_part = trimmed.parse::<i64>()
                .map_err(|_| OverflowDetected::InvalidInput)?;
            Ok((integer_part, ""))
        }
    }

    /// Convert decimal to balanced ternary for Tier 1 (TQ10.10)
    ///
    /// **FORMAT**: value = integer_part x 3^10 + fractional_encoding
    /// **STORAGE**: i32
    /// **ALGORITHM**: Pure integer arithmetic -- scale factor is 3^10 = 59049
    fn convert_decimal_to_ternary_tier1(integer: i64, fractional: &str, final_neg: bool) -> Result<i32, OverflowDetected> {
        let scale = SCALE_TQ10_10 as i64; // 3^10 = 59_049

        // Integer part: integer x scale
        let integer_scaled = integer.checked_mul(scale)
            .ok_or(OverflowDetected::TierOverflow)?;

        // Fractional part: parse digits, compute frac_digits x scale / 10^len
        let frac_scaled = if fractional.is_empty() {
            0i64
        } else {
            let frac_digits = fractional.parse::<i64>()
                .map_err(|_| OverflowDetected::InvalidInput)?;
            let ten_pow = 10_i64.pow(fractional.len() as u32);
            // Nearest; tie direction from the FINAL sign (ties toward +∞):
            // positive value → magnitude up, negative value → magnitude down.
            let sn = frac_digits * scale;
            let mut q = sn / ten_pow;
            let rem2 = (sn - q * ten_pow) << 1;
            if rem2 > ten_pow || (rem2 == ten_pow && !final_neg) { q += 1; }
            q
        };

        let total = if integer >= 0 {
            integer_scaled.checked_add(frac_scaled)
        } else {
            integer_scaled.checked_sub(frac_scaled)
        }.ok_or(OverflowDetected::TierOverflow)?;

        // Check i32 range
        if total > i32::MAX as i64 || total < i32::MIN as i64 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(total as i32)
    }

    /// Convert decimal to balanced ternary for Tier 2 (TQ20.20)
    ///
    /// **FORMAT**: value = integer_part x 3^20 + fractional_encoding
    /// **STORAGE**: i64
    fn convert_decimal_to_ternary_tier2(integer: i64, fractional: &str, final_neg: bool) -> Result<i64, OverflowDetected> {
        let scale = SCALE_TQ20_20; // 3^20 = 3_486_784_401

        // Integer part: integer x scale (use i128 intermediate to avoid overflow)
        let integer_scaled = (integer as i128).checked_mul(scale as i128)
            .ok_or(OverflowDetected::TierOverflow)?;

        // Fractional part
        let frac_scaled = if fractional.is_empty() {
            0i128
        } else {
            let frac_digits = fractional.parse::<i128>()
                .map_err(|_| OverflowDetected::InvalidInput)?;
            let ten_pow = 10_i128.pow(fractional.len() as u32);
            // Nearest, ties toward +∞ via the final sign (see tier 1).
            let sn = frac_digits * scale as i128;
            let mut q = sn / ten_pow;
            let rem2 = (sn - q * ten_pow) << 1;
            if rem2 > ten_pow || (rem2 == ten_pow && !final_neg) { q += 1; }
            q
        };

        let total = if integer >= 0 {
            integer_scaled.checked_add(frac_scaled)
        } else {
            integer_scaled.checked_sub(frac_scaled)
        }.ok_or(OverflowDetected::TierOverflow)?;

        // Check i64 range
        if total > i64::MAX as i128 || total < i64::MIN as i128 {
            return Err(OverflowDetected::TierOverflow);
        }
        Ok(total as i64)
    }

    /// Convert decimal to balanced ternary for Tier 3 (TQ40.40)
    ///
    /// **FORMAT**: value = integer_part x 3^40 + fractional_encoding
    /// **STORAGE**: i128
    fn convert_decimal_to_ternary_tier3(integer: i64, fractional: &str, final_neg: bool) -> Result<i128, OverflowDetected> {
        let scale = SCALE_TQ40_40; // 3^40 = 12_157_665_459_056_928_801

        // Integer part: integer x scale
        let integer_scaled = (integer as i128).checked_mul(scale)
            .ok_or(OverflowDetected::TierOverflow)?;

        // Fractional part
        let frac_scaled = if fractional.is_empty() {
            0i128
        } else {
            let frac_digits = fractional.parse::<i128>()
                .map_err(|_| OverflowDetected::InvalidInput)?;
            let ten_pow = 10_i128.pow(fractional.len() as u32);
            // Nearest, ties toward +∞ via the final sign (see tier 1).
            // Checked: 10^len * 3^40 exceeds i128 for len >= 20 -- long
            // fractions cascade to Tier 4 instead of overflowing here.
            let sn = frac_digits.checked_mul(scale)
                .ok_or(OverflowDetected::TierOverflow)?;
            let mut q = sn / ten_pow;
            let rem2 = (sn - q * ten_pow) << 1;
            if rem2 > ten_pow || (rem2 == ten_pow && !final_neg) { q += 1; }
            q
        };

        let total = if integer >= 0 {
            integer_scaled.checked_add(frac_scaled)
        } else {
            integer_scaled.checked_sub(frac_scaled)
        }.ok_or(OverflowDetected::TierOverflow)?;

        Ok(total)
    }

    /// Convert decimal to balanced ternary for Tier 4 (TQ80.80)
    ///
    /// **FORMAT**: value = integer_part x 3^64 + fractional_encoding
    /// **STORAGE**: I256 (never overflows at this tier)
    fn convert_decimal_to_ternary_tier4(integer: i64, fractional: &str, final_neg: bool) -> I256 {
        // 3^64 = 3_433_683_820_292_512_484_657_849_089_281 (fits in i128)
        let scale = I256::from_i128(3_433_683_820_292_512_484_657_849_089_281_i128);

        // Integer part: integer x scale
        let integer_i256 = I256::from_i128(integer as i128);
        let integer_scaled = integer_i256 * scale;

        // Fractional part
        let frac_scaled = if fractional.is_empty() {
            I256::zero()
        } else {
            let frac_digits = fractional.parse::<i128>().unwrap_or(0);
            let ten_pow = 10_i128.pow(fractional.len() as u32);
            let frac_i256 = I256::from_i128(frac_digits);
            // Nearest, ties toward +∞ via the final sign (see tier 1).
            let sn = frac_i256 * scale;
            let tp = I256::from_i128(ten_pow);
            let (mut q, rem) =
                crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(sn, tp);
            let rem2 = rem + rem;
            if rem2 > tp || (rem2 == tp && !final_neg) { q = q + I256::from_i128(1); }
            q
        };

        if integer >= 0 {
            integer_scaled + frac_scaled
        } else {
            integer_scaled - frac_scaled
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tests from ternary_tiers ---

    #[test]
    fn test_tier1_basics() {
        let zero = TernaryTier1::zero();
        let one = TernaryTier1::one();

        assert_eq!(zero.raw(), 0);
        assert_eq!(one.raw(), SCALE_TQ10_10);
    }

    #[test]
    fn test_tier_promotion_tiers() {
        let tier1_val = TernaryTier1::from_integer(10).unwrap();
        let tier2_val = tier1_val.to_tier2();

        // Value should be preserved but scaled up
        assert_eq!(tier2_val.raw(), 10 * SCALE_TQ20_20);
    }

    #[test]
    fn test_exact_division_by_3_tier1() {
        let val = TernaryTier1::from_integer(9).unwrap();
        let divided = val.div3();

        // 9 / 3 = 3 in ternary (exact)
        assert_eq!(divided.raw(), 3 * SCALE_TQ10_10);
    }

    #[test]
    fn test_storage_efficiency() {
        use std::mem::size_of;

        assert_eq!(size_of::<TernaryTier1>(), 4);  // 32 bits
        assert_eq!(size_of::<TernaryTier2>(), 8);  // 64 bits
        assert_eq!(size_of::<TernaryTier3>(), 16); // 128 bits
    }

    // --- Tests from universal_ternary ---

    #[test]
    fn test_create_from_integer() {
        // Small integer should use Tier 1
        let small = UniversalTernaryFixed::from_integer(100).unwrap();
        assert!(matches!(small.current_tier(), TernaryTier::Tier1));

        // Large integer should use higher tier
        let large = UniversalTernaryFixed::from_integer(1_000_000).unwrap();
        assert!(matches!(large.current_tier(), TernaryTier::Tier2 | TernaryTier::Tier3));
    }

    #[test]
    fn test_universal_tier_promotion() {
        let tier1_val = UniversalTernaryFixed::from_integer(10).unwrap();
        assert!(matches!(tier1_val.current_tier(), TernaryTier::Tier1));

        // Promote to higher tier
        let promoted = tier1_val.promote_to_tier(TernaryTier::Tier3).unwrap();
        assert!(matches!(promoted.current_tier(), TernaryTier::Tier3));
    }

    #[test]
    fn test_exact_division_by_3_universal() {
        let val = UniversalTernaryFixed::from_integer(9).unwrap();
        let _divided = val.div3();

        // Division by 3 should be exact in balanced ternary
    }

    #[test]
    fn test_common_tier_alignment() {
        let tier1_val = UniversalTernaryFixed::from_integer(10).unwrap();
        let tier2_val = UniversalTernaryFixed::from_integer(100_000).unwrap();

        let (aligned1, aligned2) = tier1_val.align_to_common_tier(&tier2_val);

        // Both should now be at the higher tier
        assert_eq!(aligned1.current_tier() as u8, aligned2.current_tier() as u8);
    }

    #[test]
    fn test_zero_and_one() {
        let zero = UniversalTernaryFixed::zero();
        let one = UniversalTernaryFixed::one();

        assert!(matches!(zero.current_tier(), TernaryTier::Tier1));
        assert!(matches!(one.current_tier(), TernaryTier::Tier1));
    }
}
