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
use crate::fixed_point::router::fractal_topology::{
    classify, route_binary_op, coerce_to_decimal, OpId, DomainChoice,
};

impl StackEvaluator {
    /// Negate value with overflow handling
    pub(crate) fn negate_value(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        match value {
            StackValue::BinaryCompute(tier, val, ref shadow) => {
                Ok(StackValue::BinaryCompute(tier, compute_negate(val), shadow_negate(shadow)))
            }
            StackValue::DecimalCompute(tier, val, ref shadow) => {
                use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_compute_neg;
                Ok(StackValue::DecimalCompute(tier, decimal_compute_neg(val), shadow_negate(shadow)))
            }
            StackValue::Binary(tier, val, ref shadow) => {
                // Full-precision binary negation with UGOD tier promotion
                let binary = binary_from_storage(tier, &val)?;
                let result = binary.negate()?;
                let (new_tier, storage) = binary_to_storage(&result)?;
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
                let (new_tier, storage) = ternary_to_storage(&result)?;
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
        // Handle DecimalCompute: if both operands are decimal-domain, stay at decimal compute tier
        use crate::fixed_point::domains::decimal_fixed::transcendental::{
            decimal_compute_add, decimal_upscale_to_compute,
        };
        match (&left, &right) {
            (StackValue::DecimalCompute(t1, v1, s1), StackValue::DecimalCompute(_t2, v2, s2)) => {
                return Ok(StackValue::DecimalCompute(*t1, decimal_compute_add(*v1, *v2), shadow_add(s1, s2)));
            }
            (StackValue::DecimalCompute(t1, v1, s1), StackValue::Decimal(dp, scaled, s2)) => {
                let v2 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t1, decimal_compute_add(*v1, v2), shadow_add(s1, s2)));
            }
            (StackValue::Decimal(dp, scaled, s1), StackValue::DecimalCompute(t2, v2, s2)) => {
                let v1 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t2, decimal_compute_add(v1, *v2), shadow_add(s1, s2)));
            }
            _ => {}
        }
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
                    // Ladder top (0.5.0 item 1): if the promoted result
                    // cannot fit the profile storage — or the UGOD ladder
                    // itself overflows — fall back to the exact rational
                    // path instead of erroring (and never wrapping; the
                    // storage conversion is checked now).
                    match binary_a.add(&binary_b).and_then(|r| binary_to_storage(&r)) {
                        Ok((tier, storage)) => {
                            Ok(StackValue::Binary(tier, storage, shadow_add(s1, s2)))
                        }
                        Err(OverflowDetected::TierOverflow) => {
                            self.add_via_rational(left.clone(), right.clone())
                        }
                        Err(e) => Err(e),
                    }
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
                    let (tier, storage) = ternary_to_storage(&result)?;
                    Ok(StackValue::Ternary(tier, storage, shadow_add(s1, s2)))
                }
                _ => {
                    // Cross-domain - convert through rational
                    self.add_via_rational(left, right)
                }
            }
        } else {
            // Cross-domain: try router-guided coercion before rational fallback.
            // The fractal router classifies both operands by shadow denominator factoring
            // and picks the lowest-rank domain where both are exact. If coercion succeeds,
            // the recursive call hits the same-domain path — no infinite recursion.
            if let Some((cl, cr)) = self.try_route_coerce(OpId::Add, &left, &right) {
                self.add_values(cl, cr)
            } else {
                self.add_via_rational(left, right)
            }
        }
    }

    /// Subtract values with UGOD overflow handling
    pub(crate) fn subtract_values(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        // DecimalCompute propagation
        use crate::fixed_point::domains::decimal_fixed::transcendental::{
            decimal_compute_sub, decimal_upscale_to_compute,
        };
        match (&left, &right) {
            (StackValue::DecimalCompute(t1, v1, s1), StackValue::DecimalCompute(_t2, v2, s2)) => {
                return Ok(StackValue::DecimalCompute(*t1, decimal_compute_sub(*v1, *v2), shadow_subtract(s1, s2)));
            }
            (StackValue::DecimalCompute(t1, v1, s1), StackValue::Decimal(dp, scaled, s2)) => {
                let v2 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t1, decimal_compute_sub(*v1, v2), shadow_subtract(s1, s2)));
            }
            (StackValue::Decimal(dp, scaled, s1), StackValue::DecimalCompute(t2, v2, s2)) => {
                let v1 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t2, decimal_compute_sub(v1, *v2), shadow_subtract(s1, s2)));
            }
            _ => {}
        }
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
                    // Ladder top: rational fallback (see add arm).
                    match binary_a.subtract(&binary_b).and_then(|r| binary_to_storage(&r)) {
                        Ok((tier, storage)) => {
                            Ok(StackValue::Binary(tier, storage, shadow_subtract(s1, s2)))
                        }
                        Err(OverflowDetected::TierOverflow) => {
                            self.subtract_via_rational(left.clone(), right.clone())
                        }
                        Err(e) => Err(e),
                    }
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
                    let (tier, storage) = ternary_to_storage(&result)?;
                    Ok(StackValue::Ternary(tier, storage, shadow_subtract(s1, s2)))
                }
                _ => {
                    self.subtract_via_rational(left, right)
                }
            }
        } else {
            // Cross-domain: try router-guided coercion before rational fallback
            if let Some((cl, cr)) = self.try_route_coerce(OpId::Sub, &left, &right) {
                self.subtract_values(cl, cr)
            } else {
                self.subtract_via_rational(left, right)
            }
        }
    }

    /// Multiply values with UGOD overflow handling
    pub(crate) fn multiply_values(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        // DecimalCompute propagation
        use crate::fixed_point::domains::decimal_fixed::transcendental::{
            decimal_compute_mul, decimal_upscale_to_compute,
        };
        match (&left, &right) {
            (StackValue::DecimalCompute(t1, v1, s1), StackValue::DecimalCompute(_t2, v2, s2)) => {
                return Ok(StackValue::DecimalCompute(*t1, decimal_compute_mul(*v1, *v2), shadow_multiply(s1, s2)));
            }
            (StackValue::DecimalCompute(t1, v1, s1), StackValue::Decimal(dp, scaled, s2)) => {
                let v2 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t1, decimal_compute_mul(*v1, v2), shadow_multiply(s1, s2)));
            }
            (StackValue::Decimal(dp, scaled, s1), StackValue::DecimalCompute(t2, v2, s2)) => {
                let v1 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t2, decimal_compute_mul(v1, *v2), shadow_multiply(s1, s2)));
            }
            _ => {}
        }
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
                    // Ladder top (0.5.0 item 1): if the promoted result
                    // cannot fit the profile storage — or the UGOD ladder
                    // itself overflows — fall back to the exact rational
                    // path instead of erroring (and never wrapping; the
                    // storage conversion is checked now).
                    match binary_a.multiply(&binary_b).and_then(|r| binary_to_storage(&r)) {
                        Ok((tier, storage)) => {
                            Ok(StackValue::Binary(tier, storage, shadow_multiply(s1, s2)))
                        }
                        Err(OverflowDetected::TierOverflow) => {
                            self.multiply_via_rational(left.clone(), right.clone())
                        }
                        Err(e) => Err(e),
                    }
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
                    let (tier, storage) = ternary_to_storage(&result)?;
                    Ok(StackValue::Ternary(tier, storage, shadow_multiply(s1, s2)))
                }
                _ => {
                    self.multiply_via_rational(left, right)
                }
            }
        } else {
            // Cross-domain: try router-guided coercion before rational fallback
            if let Some((cl, cr)) = self.try_route_coerce(OpId::Mul, &left, &right) {
                self.multiply_values(cl, cr)
            } else {
                self.multiply_via_rational(left, right)
            }
        }
    }

    /// Divide values with UGOD overflow handling
    pub(crate) fn divide_values(&mut self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        // DecimalCompute propagation
        use crate::fixed_point::domains::decimal_fixed::transcendental::{
            decimal_compute_div, decimal_upscale_to_compute,
        };
        match (&left, &right) {
            (StackValue::DecimalCompute(t1, v1, s1), StackValue::DecimalCompute(_t2, v2, s2)) => {
                return Ok(StackValue::DecimalCompute(*t1, decimal_compute_div(*v1, *v2)?, shadow_divide(s1, s2)));
            }
            (StackValue::DecimalCompute(t1, v1, s1), StackValue::Decimal(dp, scaled, s2)) => {
                let v2 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t1, decimal_compute_div(*v1, v2)?, shadow_divide(s1, s2)));
            }
            (StackValue::Decimal(dp, scaled, s1), StackValue::DecimalCompute(t2, v2, s2)) => {
                let v1 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t2, decimal_compute_div(v1, *v2)?, shadow_divide(s1, s2)));
            }
            _ => {}
        }
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
                // Ladder top: rational fallback (see add arm).
                match binary_a.divide(&binary_b).and_then(|r| binary_to_storage(&r)) {
                    Ok((tier, storage)) => {
                        Ok(StackValue::Binary(tier, storage, shadow_divide(s1, s2)))
                    }
                    Err(OverflowDetected::TierOverflow) => {
                        self.divide_via_rational(left.clone(), right.clone())
                    }
                    Err(e) => Err(e),
                }
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
                let (tier, storage) = ternary_to_storage(&result)?;
                Ok(StackValue::Ternary(tier, storage, shadow_divide(s1, s2)))
            }
            _ => {
                // Cross-domain: try router-guided coercion before rational fallback
                if let Some((cl, cr)) = self.try_route_coerce(OpId::Div, &left, &right) {
                    self.divide_values(cl, cr)
                } else {
                    self.divide_via_rational(left, right)
                }
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
    // ROUTER-GUIDED CROSS-DOMAIN COERCION
    // ============================================================================

    /// Try router-guided coercion for a cross-domain binary operation.
    ///
    /// Returns `Some((coerced_left, coerced_right))` if both operands can be
    /// coerced to a common domain. Returns `None` if coercion is impossible
    /// (should fall back to rational arithmetic).
    ///
    /// **Performance**: ~35 ns (classify × 2 + table lookup + shadow coercion).
    fn try_route_coerce(
        &self,
        op: OpId,
        left: &StackValue,
        right: &StackValue,
    ) -> Option<(StackValue, StackValue)> {
        let lc = classify(left);
        let rc = classify(right);
        match route_binary_op(op, lc, rc) {
            DomainChoice::Decimal => {
                let dl = coerce_to_decimal(left)?;
                let dr = coerce_to_decimal(right)?;
                Some((dl, dr))
            }
            DomainChoice::Binary => {
                let bl = self.coerce_to_binary_sv(left).ok()?;
                let br = self.coerce_to_binary_sv(right).ok()?;
                Some((bl, br))
            }
            DomainChoice::Ternary => {
                // Add/Sub of 3-adic values (denominator 3^k) — exact in
                // ternary, replaces the rational-pair fallback. Coercion is
                // exact for this class (num·3^F/den divides evenly). On
                // narrow profiles a large value can overflow ternary
                // storage: `.ok()?` makes that a silent fall-through to the
                // rational path, so routing can never introduce a failure
                // (docs/design/TERNARY_ROUTING_COLUMN.md, Decision 2).
                let tl = self.convert_to_ternary(left.clone()).ok()?;
                let tr = self.convert_to_ternary(right.clone()).ok()?;
                Some((tl, tr))
            }
            DomainChoice::Symbolic => None,
        }
    }

    /// Coerce a StackValue to Binary domain.
    ///
    /// - Binary/BinaryCompute: pass through
    /// - Decimal: use decimal_to_binary_storage (Q-format conversion)
    /// - Other: use to_binary_storage (general conversion via shadow/rational)
    fn coerce_to_binary_sv(&self, value: &StackValue) -> Result<StackValue, OverflowDetected> {
        match value {
            StackValue::Binary(..) | StackValue::BinaryCompute(..) => Ok(value.clone()),
            StackValue::Decimal(dp, scaled, shadow) => {
                let binary_storage = super::decimal_to_binary_storage(*dp, *scaled)?;
                let tier = self.profile_max_binary_tier();
                Ok(StackValue::Binary(tier, binary_storage, *shadow))
            }
            _ => {
                let storage = self.to_binary_storage(value)?;
                let tier = self.profile_max_binary_tier();
                Ok(StackValue::Binary(tier, storage, value.shadow()))
            }
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
