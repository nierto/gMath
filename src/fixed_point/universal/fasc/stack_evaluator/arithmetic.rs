//! Arithmetic dispatch — add, subtract, multiply, divide with UGOD fallback
//!
//! Each operation attempts native-domain UGOD arithmetic first,
//! falling back to rational for cross-domain operations.

#[allow(unused_imports)]
use super::{BinaryStorage, ComputeStorage, StackValue, StackEvaluator, DECIMAL_DP_PROMOTION_THRESHOLD};
use super::compute::{upscale_to_compute, compute_add, compute_subtract, compute_multiply, compute_divide, compute_negate};
use super::domain::{
    binary_from_storage, binary_to_storage,
    decimal_from_storage, decimal_to_storage,
    ternary_from_storage, ternary_to_storage,
    shadow_negate, shadow_add, shadow_subtract, shadow_multiply, shadow_divide,
};
use crate::fixed_point::universal::tier_types::CompactShadow;
#[allow(unused_imports)]
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, OverflowDetected};
use crate::deployment_profiles::DeploymentProfile;

impl StackEvaluator {
    /// Negate value with overflow handling
    pub(crate) fn negate_value(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        match value {
            StackValue::BinaryCompute(tier, val, ref shadow) => {
                Ok(StackValue::BinaryCompute(tier, compute_negate(val), shadow_negate(shadow)))
            }
            StackValue::Binary(tier, val, ref shadow) => {
                // Full-precision binary negation with UGOD tier promotion
                let binary = binary_from_storage(tier, &val)?;
                let result = binary.negate()?;
                let (new_tier, storage) = binary_to_storage(&result);
                Ok(StackValue::Binary(new_tier, storage, shadow_negate(shadow)))
            }
            StackValue::Decimal(dec, val, ref shadow) => {
                // Full-precision decimal negation with UGOD tier promotion
                let decimal = decimal_from_storage(dec, &val)?;
                let result = decimal.negate()?;
                let (new_dec, storage) = decimal_to_storage(&result);
                Ok(StackValue::Decimal(new_dec, storage, shadow_negate(shadow)))
            }
            StackValue::Ternary(tier, val, ref shadow) => {
                // Full-precision ternary negation with UGOD tier promotion
                let ternary = ternary_from_storage(tier, &val)?;
                let result = ternary.negate()?;
                let (new_tier, storage) = ternary_to_storage(&result);
                Ok(StackValue::Ternary(new_tier, storage, shadow_negate(shadow)))
            }
            StackValue::Symbolic(s) => {
                Ok(StackValue::Symbolic(s.try_negate()?))
            }
            StackValue::Error(e) => Ok(StackValue::Error(e)),
        }
    }

    /// Add values with UGOD overflow handling
    pub(crate) fn add_values(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        // Handle BinaryCompute: if either operand is BinaryCompute, operate at compute tier
        match (&left, &right) {
            (StackValue::BinaryCompute(t1, v1, s1), StackValue::BinaryCompute(_t2, v2, s2)) => {
                return Ok(StackValue::BinaryCompute(*t1, compute_add(*v1, *v2), shadow_add(s1, s2)));
            }
            (StackValue::BinaryCompute(t1, v1, s1), StackValue::Binary(_, v2, s2)) => {
                let v2_compute = upscale_to_compute(*v2);
                return Ok(StackValue::BinaryCompute(*t1, compute_add(*v1, v2_compute), shadow_add(s1, s2)));
            }
            (StackValue::Binary(_, v1, s1), StackValue::BinaryCompute(t2, v2, s2)) => {
                let v1_compute = upscale_to_compute(*v1);
                return Ok(StackValue::BinaryCompute(*t2, compute_add(v1_compute, *v2), shadow_add(s1, s2)));
            }
            (StackValue::BinaryCompute(t1, v1, s1), other) | (other, StackValue::BinaryCompute(t1, v1, s1)) => {
                // BinaryCompute + non-binary: convert other directly to compute tier (full precision)
                let other_compute = self.to_compute_storage(other)?;
                return Ok(StackValue::BinaryCompute(*t1, compute_add(*v1, other_compute), shadow_add(s1, &other.shadow())));
            }
            _ => {}
        }

        // If domains match, try native addition
        if left.domain_type() == right.domain_type() {
            match (&left, &right) {
                (StackValue::Binary(t1, v1, s1), StackValue::Binary(t2, v2, s2)) => {
                    // Full-precision binary addition with UGOD tier promotion
                    let binary_a = binary_from_storage(*t1, v1)?;
                    let binary_b = binary_from_storage(*t2, v2)?;
                    let result = binary_a.add(&binary_b)?;
                    let (tier, storage) = binary_to_storage(&result);
                    Ok(StackValue::Binary(tier, storage, shadow_add(s1, s2)))
                }
                (StackValue::Decimal(d1, v1, s1), StackValue::Decimal(d2, v2, s2)) => {
                    // Full-precision decimal addition with UGOD tier promotion
                    // dp alignment handled inside decimal add (align_decimal_places_and_tier)
                    // On overflow, fall through to rational for exact result
                    match (decimal_from_storage(*d1, v1), decimal_from_storage(*d2, v2)) {
                        (Ok(decimal_a), Ok(decimal_b)) => {
                            match decimal_a.add(&decimal_b) {
                                Ok(result) => {
                                    let (dec, storage) = decimal_to_storage(&result);
                                    Ok(StackValue::Decimal(dec, storage, shadow_add(s1, s2)))
                                }
                                Err(_) => self.add_via_rational(left, right),
                            }
                        }
                        _ => self.add_via_rational(left, right),
                    }
                }
                (StackValue::Ternary(t1, v1, s1), StackValue::Ternary(t2, v2, s2)) => {
                    // Full-precision ternary addition with UGOD tier promotion
                    let ternary_a = ternary_from_storage(*t1, v1)?;
                    let ternary_b = ternary_from_storage(*t2, v2)?;
                    let result = ternary_a.add(&ternary_b)?;
                    let (tier, storage) = ternary_to_storage(&result);
                    Ok(StackValue::Ternary(tier, storage, shadow_add(s1, s2)))
                }
                _ => {
                    // Cross-domain - convert through rational
                    self.add_via_rational(left, right)
                }
            }
        } else {
            // Cross-domain: exact symbolic arithmetic via rational representation.
            // a/b + c/d = (ad+bc)/bd — always exact, no rounding.
            self.add_via_rational(left, right)
        }
    }

    /// Subtract values with UGOD overflow handling
    pub(crate) fn subtract_values(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        // Handle BinaryCompute: if either operand is BinaryCompute, operate at compute tier
        match (&left, &right) {
            (StackValue::BinaryCompute(t1, v1, s1), StackValue::BinaryCompute(_t2, v2, s2)) => {
                return Ok(StackValue::BinaryCompute(*t1, compute_subtract(*v1, *v2), shadow_subtract(s1, s2)));
            }
            (StackValue::BinaryCompute(t1, v1, s1), StackValue::Binary(_, v2, s2)) => {
                let v2_compute = upscale_to_compute(*v2);
                return Ok(StackValue::BinaryCompute(*t1, compute_subtract(*v1, v2_compute), shadow_subtract(s1, s2)));
            }
            (StackValue::Binary(_, v1, s1), StackValue::BinaryCompute(t2, v2, s2)) => {
                let v1_compute = upscale_to_compute(*v1);
                return Ok(StackValue::BinaryCompute(*t2, compute_subtract(v1_compute, *v2), shadow_subtract(s1, s2)));
            }
            (StackValue::BinaryCompute(t1, v1, s1), other) => {
                let other_compute = self.to_compute_storage(other)?;
                return Ok(StackValue::BinaryCompute(*t1, compute_subtract(*v1, other_compute), shadow_subtract(s1, &other.shadow())));
            }
            (other, StackValue::BinaryCompute(t2, v2, s2)) => {
                let other_compute = self.to_compute_storage(other)?;
                return Ok(StackValue::BinaryCompute(*t2, compute_subtract(other_compute, *v2), shadow_subtract(&other.shadow(), s2)));
            }
            _ => {}
        }

        // Similar to add but with subtraction
        if left.domain_type() == right.domain_type() {
            match (&left, &right) {
                (StackValue::Binary(t1, v1, s1), StackValue::Binary(t2, v2, s2)) => {
                    // Full-precision binary subtraction with UGOD tier promotion
                    let binary_a = binary_from_storage(*t1, v1)?;
                    let binary_b = binary_from_storage(*t2, v2)?;
                    let result = binary_a.subtract(&binary_b)?;
                    let (tier, storage) = binary_to_storage(&result);
                    Ok(StackValue::Binary(tier, storage, shadow_subtract(s1, s2)))
                }
                (StackValue::Decimal(d1, v1, s1), StackValue::Decimal(d2, v2, s2)) => {
                    // Full-precision decimal subtraction with UGOD tier promotion
                    // dp alignment handled inside decimal subtract (align_decimal_places_and_tier)
                    // On overflow, fall through to rational for exact result
                    match (decimal_from_storage(*d1, v1), decimal_from_storage(*d2, v2)) {
                        (Ok(decimal_a), Ok(decimal_b)) => {
                            match decimal_a.subtract(&decimal_b) {
                                Ok(result) => {
                                    let (dec, storage) = decimal_to_storage(&result);
                                    Ok(StackValue::Decimal(dec, storage, shadow_subtract(s1, s2)))
                                }
                                Err(_) => self.subtract_via_rational(left, right),
                            }
                        }
                        _ => self.subtract_via_rational(left, right),
                    }
                }
                (StackValue::Ternary(t1, v1, s1), StackValue::Ternary(t2, v2, s2)) => {
                    // Full-precision ternary subtraction with UGOD tier promotion
                    let ternary_a = ternary_from_storage(*t1, v1)?;
                    let ternary_b = ternary_from_storage(*t2, v2)?;
                    let result = ternary_a.subtract(&ternary_b)?;
                    let (tier, storage) = ternary_to_storage(&result);
                    Ok(StackValue::Ternary(tier, storage, shadow_subtract(s1, s2)))
                }
                _ => {
                    self.subtract_via_rational(left, right)
                }
            }
        } else {
            // Cross-domain: exact symbolic arithmetic via rational representation.
            // a/b - c/d = (ad-bc)/bd — always exact, no rounding.
            self.subtract_via_rational(left, right)
        }
    }

    /// Multiply values with UGOD overflow handling
    pub(crate) fn multiply_values(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        // Handle BinaryCompute: if either operand is BinaryCompute, operate at compute tier
        match (&left, &right) {
            (StackValue::BinaryCompute(t1, v1, s1), StackValue::BinaryCompute(_t2, v2, s2)) => {
                return Ok(StackValue::BinaryCompute(*t1, compute_multiply(*v1, *v2), shadow_multiply(s1, s2)));
            }
            (StackValue::BinaryCompute(t1, v1, s1), StackValue::Binary(_, v2, s2)) => {
                let v2_compute = upscale_to_compute(*v2);
                return Ok(StackValue::BinaryCompute(*t1, compute_multiply(*v1, v2_compute), shadow_multiply(s1, s2)));
            }
            (StackValue::Binary(_, v1, s1), StackValue::BinaryCompute(t2, v2, s2)) => {
                let v1_compute = upscale_to_compute(*v1);
                return Ok(StackValue::BinaryCompute(*t2, compute_multiply(v1_compute, *v2), shadow_multiply(s1, s2)));
            }
            (StackValue::BinaryCompute(t1, v1, s1), other) => {
                let other_compute = self.to_compute_storage(other)?;
                return Ok(StackValue::BinaryCompute(*t1, compute_multiply(*v1, other_compute), shadow_multiply(s1, &other.shadow())));
            }
            (other, StackValue::BinaryCompute(t2, v2, s2)) => {
                let other_compute = self.to_compute_storage(other)?;
                return Ok(StackValue::BinaryCompute(*t2, compute_multiply(other_compute, *v2), shadow_multiply(&other.shadow(), s2)));
            }
            _ => {}
        }

        if left.domain_type() == right.domain_type() {
            match (&left, &right) {
                (StackValue::Binary(t1, v1, s1), StackValue::Binary(t2, v2, s2)) => {
                    // Full-precision binary multiplication with UGOD tier promotion
                    let binary_a = binary_from_storage(*t1, v1)?;
                    let binary_b = binary_from_storage(*t2, v2)?;
                    let result = binary_a.multiply(&binary_b)?;
                    let (tier, storage) = binary_to_storage(&result);
                    Ok(StackValue::Binary(tier, storage, shadow_multiply(s1, s2)))
                }
                (StackValue::Decimal(d1, v1, s1), StackValue::Decimal(d2, v2, s2)) => {
                    let dp_result = *d1 as u16 + *d2 as u16;
                    if dp_result > DECIMAL_DP_PROMOTION_THRESHOLD {
                        // Decimal places would exceed profile precision — promote to Symbolic.
                        // Rational representation is always exact: (a/10^dp_a) × (b/10^dp_b)
                        // = (a×b) / (10^(dp_a+dp_b)). No rounding, no ULP error.
                        return self.multiply_via_rational(left, right);
                    }
                    // dp fits within profile precision — normal decimal multiply
                    // On overflow (storage or arithmetic), fall through to rational
                    match (decimal_from_storage(*d1, v1), decimal_from_storage(*d2, v2)) {
                        (Ok(decimal_a), Ok(decimal_b)) => {
                            match decimal_a.multiply(&decimal_b) {
                                Ok(result) => {
                                    let (dp, storage) = decimal_to_storage(&result);
                                    Ok(StackValue::Decimal(dp, storage, shadow_multiply(s1, s2)))
                                }
                                Err(_) => self.multiply_via_rational(left, right),
                            }
                        }
                        _ => self.multiply_via_rational(left, right),
                    }
                }
                (StackValue::Ternary(t1, v1, s1), StackValue::Ternary(t2, v2, s2)) => {
                    // Full-precision ternary multiplication with UGOD tier promotion
                    let ternary_a = ternary_from_storage(*t1, v1)?;
                    let ternary_b = ternary_from_storage(*t2, v2)?;
                    let result = ternary_a.multiply(&ternary_b)?;
                    let (tier, storage) = ternary_to_storage(&result);
                    Ok(StackValue::Ternary(tier, storage, shadow_multiply(s1, s2)))
                }
                _ => {
                    self.multiply_via_rational(left, right)
                }
            }
        } else {
            // Cross-domain: exact symbolic arithmetic via rational representation.
            // (a/b) × (c/d) = (ac)/(bd) — always exact, no rounding.
            self.multiply_via_rational(left, right)
        }
    }

    /// Divide values with UGOD overflow handling
    pub(crate) fn divide_values(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        // Handle BinaryCompute: if either operand is BinaryCompute, operate at compute tier
        match (&left, &right) {
            (StackValue::BinaryCompute(t1, v1, s1), StackValue::BinaryCompute(_t2, v2, s2)) => {
                return Ok(StackValue::BinaryCompute(*t1, compute_divide(*v1, *v2)?, shadow_divide(s1, s2)));
            }
            (StackValue::BinaryCompute(t1, v1, s1), StackValue::Binary(_, v2, s2)) => {
                let v2_compute = upscale_to_compute(*v2);
                return Ok(StackValue::BinaryCompute(*t1, compute_divide(*v1, v2_compute)?, shadow_divide(s1, s2)));
            }
            (StackValue::Binary(_, v1, s1), StackValue::BinaryCompute(t2, v2, s2)) => {
                let v1_compute = upscale_to_compute(*v1);
                return Ok(StackValue::BinaryCompute(*t2, compute_divide(v1_compute, *v2)?, shadow_divide(s1, s2)));
            }
            (StackValue::BinaryCompute(t1, v1, s1), other) => {
                let other_compute = self.to_compute_storage(other)?;
                return Ok(StackValue::BinaryCompute(*t1, compute_divide(*v1, other_compute)?, shadow_divide(s1, &other.shadow())));
            }
            (other, StackValue::BinaryCompute(t2, v2, s2)) => {
                let other_compute = self.to_compute_storage(other)?;
                return Ok(StackValue::BinaryCompute(*t2, compute_divide(other_compute, *v2)?, shadow_divide(&other.shadow(), s2)));
            }
            _ => {}
        }

        // Division often requires symbolic computation for exactness
        match (&left, &right) {
            (StackValue::Symbolic(l), StackValue::Symbolic(r)) => {
                Ok(StackValue::Symbolic(l.try_divide(r)?))
            }
            (StackValue::Binary(t1, v1, s1), StackValue::Binary(t2, v2, s2)) => {
                // Full-precision binary division with UGOD tier promotion
                let binary_a = binary_from_storage(*t1, v1)?;
                let binary_b = binary_from_storage(*t2, v2)?;
                let result = binary_a.divide(&binary_b)?;
                let (tier, storage) = binary_to_storage(&result);
                Ok(StackValue::Binary(tier, storage, shadow_divide(s1, s2)))
            }
            (StackValue::Decimal(d1, v1, s1), StackValue::Decimal(d2, v2, s2)) => {
                // Full-precision decimal division with UGOD tier promotion
                // On overflow or inexact result, fall through to rational for exact answer
                match (decimal_from_storage(*d1, v1), decimal_from_storage(*d2, v2)) {
                    (Ok(decimal_a), Ok(decimal_b)) => {
                        match decimal_a.divide(&decimal_b) {
                            Ok(result) => {
                                let (dp, storage) = decimal_to_storage(&result);
                                Ok(StackValue::Decimal(dp, storage, shadow_divide(s1, s2)))
                            }
                            Err(_) => self.divide_via_rational(left, right),
                        }
                    }
                    _ => self.divide_via_rational(left, right),
                }
            }
            (StackValue::Ternary(t1, v1, s1), StackValue::Ternary(t2, v2, s2)) => {
                // Full-precision ternary division with UGOD tier promotion
                let ternary_a = ternary_from_storage(*t1, v1)?;
                let ternary_b = ternary_from_storage(*t2, v2)?;
                let result = ternary_a.divide(&ternary_b)?;
                let (tier, storage) = ternary_to_storage(&result);
                Ok(StackValue::Ternary(tier, storage, shadow_divide(s1, s2)))
            }
            _ => {
                // Cross-domain: exact symbolic arithmetic via rational representation.
                // (a/b) ÷ (c/d) = (a/b) × (d/c) — always exact, no rounding.
                self.divide_via_rational(left, right)
            }
        }
    }

    // ============================================================================
    // RATIONAL FALLBACK OPERATIONS FOR CROSS-DOMAIN AND OVERFLOW
    // ============================================================================

    pub(crate) fn add_via_rational(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        let l_rational = left.to_rational()?;
        let r_rational = right.to_rational()?;
        let result = l_rational.try_add(&r_rational)?;
        Ok(StackValue::Symbolic(result))
    }

    pub(crate) fn subtract_via_rational(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        let l_rational = left.to_rational()?;
        let r_rational = right.to_rational()?;
        let result = l_rational.try_subtract(&r_rational)?;
        Ok(StackValue::Symbolic(result))
    }

    pub(crate) fn multiply_via_rational(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        let l_rational = left.to_rational()?;
        let r_rational = right.to_rational()?;
        match l_rational.try_multiply(&r_rational) {
            Ok(result) => Ok(StackValue::Symbolic(result)),
            Err(OverflowDetected::TierOverflow) | Err(OverflowDetected::PrecisionLimit) => {
                // UGOD fallback: rational overflow → compute-tier Binary (≤0.5 ULP)
                let tier = self.profile_max_binary_tier();
                let l_compute = self.to_compute_storage(&left)?;
                let r_compute = self.to_compute_storage(&right)?;
                Ok(StackValue::BinaryCompute(tier, compute_multiply(l_compute, r_compute), CompactShadow::None))
            }
            Err(e) => Err(e),
        }
    }

    pub(crate) fn divide_via_rational(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        let l_rational = left.to_rational()?;
        let r_rational = right.to_rational()?;
        match l_rational.try_divide(&r_rational) {
            Ok(result) => Ok(StackValue::Symbolic(result)),
            Err(OverflowDetected::TierOverflow) | Err(OverflowDetected::PrecisionLimit) => {
                // UGOD fallback: rational overflow → compute-tier Binary (≤0.5 ULP)
                let tier = self.profile_max_binary_tier();
                let l_compute = self.to_compute_storage(&left)?;
                let r_compute = self.to_compute_storage(&right)?;
                Ok(StackValue::BinaryCompute(tier, compute_divide(l_compute, r_compute)?, CompactShadow::None))
            }
            Err(e) => Err(e),
        }
    }

    // ============================================================================
    // TIER PROMOTION HELPERS
    // ============================================================================

    // Tier promotion is now handled by UniversalBinaryFixed/UniversalDecimalTiered UGOD arithmetic

    // ============================================================================
    // TRANSCENDENTAL FUNCTION EVALUATION
    // ============================================================================

    /// Get the maximum binary tier for the current deployment profile
    ///
    /// **ARCHITECTURE**: Transcendentals always use profile-max tier (not overflow-based)
    /// **RATIONALE**: exp/ln/sin/cos benefit from maximum precision upfront
    /// **CONTRAST**: Basic ops (add/mul/sub/div) promote on overflow detection
    pub(crate) fn profile_max_binary_tier(&self) -> u8 {
        match self.deployment_profile {
            DeploymentProfile::Realtime => 1,      // Q16.16 (i32)
            DeploymentProfile::Compact => 2,       // Q32.32 (i64)
            DeploymentProfile::Embedded => 3,      // Q64.64 (i128)
            DeploymentProfile::Balanced => 4,      // Q128.128 (I256)
            DeploymentProfile::Scientific => 5,    // Q256.256 (I512)
            DeploymentProfile::Custom => 3,        // Default to Q64.64 for custom
        }
    }

    /// Get the maximum ternary tier for the current deployment profile
    ///
    /// **ARCHITECTURE**: Mirrors profile_max_binary_tier() for ternary domain.
    /// Maps each profile to the ternary tier whose storage type matches the profile's native type.
    /// **TIER MAPPING**:
    ///   - Embedded (i128) → Tier 3 TQ32.32 (i128, 32 frac trits ≈ 15 decimals)
    ///   - Balanced (I256)             → Tier 4 TQ64.64 (I256, 64 frac trits ≈ 30 decimals)
    ///   - Scientific (I512)           → Tier 5 TQ128.128 (I512, 128 frac trits ≈ 61 decimals)
    pub(crate) fn profile_max_ternary_tier(&self) -> u8 {
        match self.deployment_profile {
            DeploymentProfile::Realtime => 1,      // TQ8.8 (i32, 8 frac trits)
            DeploymentProfile::Compact => 2,       // TQ16.16 (i64, 16 frac trits)
            DeploymentProfile::Embedded => 3,      // TQ32.32 (i128, 32 frac trits)
            DeploymentProfile::Balanced => 4,      // TQ64.64 (I256, 64 frac trits)
            DeploymentProfile::Scientific => 5,    // TQ128.128 (I512, 128 frac trits)
            DeploymentProfile::Custom => 3,        // Default TQ32.32
        }
    }
}
