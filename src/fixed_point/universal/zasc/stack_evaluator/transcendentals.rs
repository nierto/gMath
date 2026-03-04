//! Transcendental function evaluation — 18 functions with BinaryCompute chain persistence
//!
//! 5 dedicated (exp, ln, sqrt, pow, atan2) + 13 ZASC-composed from core functions.
//! All return BinaryCompute for chain persistence — single downscale at materialization.
//! Also contains mode routing (parse_literal_with_mode, apply_output_mode)
//! and conversion helpers (to_compute_storage, to_binary_storage).

#[allow(unused_imports)]
use super::{BinaryStorage, ComputeStorage, StackValue, StackEvaluator, DECIMAL_DP_PROMOTION_THRESHOLD};
use super::compute::*;
#[allow(unused_imports)]
use super::conversion::{to_binary_storage, reduce_decimal_to_rational};
#[allow(unused_imports)]
use super::domain::{ternary_to_rational, ternary_to_storage, decimal_from_storage, decimal_to_storage, binary_to_storage};
#[allow(unused_imports)]
use super::formatting::pow10_i256;
#[cfg(table_format = "q256_256")]
#[allow(unused_imports)]
use super::formatting::pow10_i512;
use crate::fixed_point::domains::balanced_ternary::ternary_types::{UniversalTernaryFixed, TernaryRaw};
use crate::fixed_point::i256::I256;
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;
use crate::fixed_point::universal::tier_types::CompactShadow;
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, OverflowDetected};

// Transcendental functions called directly (not re-exported through compute::*)
#[cfg(table_format = "q256_256")]
use crate::fixed_point::domains::binary_fixed::transcendental::ln_binary_i1024;
#[cfg(table_format = "q128_128")]
use crate::fixed_point::domains::binary_fixed::transcendental::{exp_binary_i512, ln_binary_i512};
#[cfg(table_format = "q64_64")]
use crate::fixed_point::domains::binary_fixed::transcendental::{exp_binary_i256, ln_binary_i256};

impl StackEvaluator {

    /// Compute exp directly on ComputeStorage (already at compute tier)
    ///
    /// Returns BinaryCompute to keep the result "hot" for transcendental chaining.
    pub(crate) fn exp_at_compute_tier(&self, compute_val: ComputeStorage, storage_tier: u8) -> Result<StackValue, OverflowDetected> {
        #[cfg(table_format = "q256_256")]
        {
            // ComputeStorage = I1024 (Q512.512)
            use crate::fixed_point::domains::binary_fixed::transcendental::exp_binary_i1024;
            let result = exp_binary_i1024(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
        #[cfg(table_format = "q128_128")]
        {
            // ComputeStorage = I512 (Q256.256)
            let result = exp_binary_i512(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
        #[cfg(table_format = "q64_64")]
        {
            // ComputeStorage = I256 (Q128.128)
            let result = exp_binary_i256(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
    }

    /// Evaluate exponential function on stack value with TIER N+1 ULTRA-PRECISION
    ///
    /// All inputs are upscaled to compute tier (tier N+1) before computation.
    /// Returns BinaryCompute for transcendental chain persistence.
    pub(crate) fn evaluate_exp(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        let compute_val = self.to_compute_storage(&value)?;
        self.exp_at_compute_tier(compute_val, storage_tier)
    }

    // ============================================================================
    // NATURAL LOGARITHM FUNCTION EVALUATION
    // ============================================================================

    /// Compute ln directly on ComputeStorage (already at compute tier)
    ///
    /// Returns BinaryCompute to keep the result "hot" for transcendental chaining.
    pub(crate) fn ln_at_compute_tier(&self, compute_val: ComputeStorage, storage_tier: u8) -> Result<StackValue, OverflowDetected> {
        // Domain check: ln(x) requires x > 0
        if compute_is_negative(&compute_val) || compute_is_zero(&compute_val) {
            return Err(OverflowDetected::DomainError);
        }

        #[cfg(table_format = "q256_256")]
        {
            // ComputeStorage = I1024 (Q512.512) — compute ln at full compute tier
            let result = ln_binary_i1024(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
        #[cfg(table_format = "q128_128")]
        {
            // ComputeStorage = I512 (Q256.256)
            let result = ln_binary_i512(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
        #[cfg(table_format = "q64_64")]
        {
            // ComputeStorage = I256 (Q128.128)
            let result = ln_binary_i256(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
    }

    /// Evaluate natural logarithm function on stack value with TIER N+1 ULTRA-PRECISION
    ///
    /// All inputs are upscaled to compute tier (tier N+1) before computation.
    /// Returns BinaryCompute for transcendental chain persistence.
    ///
    /// **DOMAIN ERROR**: Returns DomainError for x <= 0
    pub(crate) fn evaluate_ln(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        let compute_val = self.to_compute_storage(&value)?;

        // Domain check: ln(x) requires x > 0 (checked in ln_at_compute_tier too,
        // but check here to avoid unnecessary conversion overhead for error cases)
        self.ln_at_compute_tier(compute_val, storage_tier)
    }

    // ============================================================================
    // SQUARE ROOT AND POWER FUNCTION EVALUATION
    // ============================================================================

    /// sqrt(x) — tier N+1 computation returning BinaryCompute
    /// **DOMAIN**: x >= 0 (returns error for x < 0)
    pub(crate) fn evaluate_sqrt(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        let compute_val = self.to_compute_storage(&value)?;

        // Domain check: sqrt(x) undefined for x < 0
        if compute_is_negative(&compute_val) {
            return Err(OverflowDetected::DomainError);
        }

        let result = sqrt_at_compute_tier(compute_val);
        Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
    }

    /// pow(x, y) = exp(y × ln(x)) — ZASC-composed at compute tier
    ///
    /// The entire chain (ln → multiply → exp) stays at compute tier via BinaryCompute propagation.
    pub(crate) fn evaluate_pow(&mut self, base: StackValue, exponent: StackValue) -> Result<StackValue, OverflowDetected> {
        // Fast path: integer exponent via shadow → exponentiation-by-squaring (0 ULP)
        if let Some((exp_num, exp_den)) = exponent.shadow().as_rational() {
            if exp_den == 1 && exp_num.unsigned_abs() <= 1000 {
                return self.pow_integer(base, exp_num as i64);
            }
        }

        // Slow path: pow(x, y) = exp(y * ln(x))
        // Chain stays at compute tier: ln returns BinaryCompute, multiply keeps it,
        // exp accepts BinaryCompute directly.
        let ln_base = self.evaluate_ln(base)?;
        let product = self.multiply_values(exponent, ln_base)?;
        self.evaluate_exp(product)
    }

    /// Integer exponentiation-by-squaring: x^n using multiply_values
    ///
    /// Produces 0 ULP for integer exponents. Negative exponents handled
    /// via 1/x^|n| using binary_divide.
    pub(crate) fn pow_integer(&mut self, base: StackValue, exp: i64) -> Result<StackValue, OverflowDetected> {
        let negative = exp < 0;
        let mut n = exp.unsigned_abs();

        if n == 0 {
            return Ok(self.make_binary_int(1));
        }

        // Convert to binary to avoid Decimal*Decimal → rational overflow in squaring
        let mut result = self.make_binary_int(1);
        let mut b = self.to_binary_value(&base)?;

        while n > 0 {
            if n & 1 == 1 {
                result = self.multiply_values(result, b.clone())?;
            }
            n >>= 1;
            if n > 0 {
                b = self.multiply_values(b.clone(), b)?;
            }
        }

        if negative {
            let one = self.make_binary_int(1);
            result = self.binary_divide(one, result)?;
        }

        Ok(result)
    }

    // ============================================================================
    // HYPERBOLIC FUNCTIONS (ZASC-composed from exp/ln/sqrt)
    // ============================================================================

    /// Create a Q-format binary constant from an integer value
    pub(crate) fn make_binary_int(&self, value: i128) -> StackValue {
        let tier = self.profile_max_binary_tier();
        #[cfg(table_format = "q256_256")]
        { StackValue::Binary(tier, I512::from_i128(value) << 256, CompactShadow::from_rational(value, 1)) }
        #[cfg(table_format = "q128_128")]
        { StackValue::Binary(tier, I256::from_i128(value) << 128, CompactShadow::from_rational(value, 1)) }
        #[cfg(table_format = "q64_64")]
        { StackValue::Binary(tier, value << 64, CompactShadow::from_rational(value, 1)) }
    }

    /// Convert any StackValue to BinaryCompute (compute tier).
    /// Used by composed transcendental functions to:
    /// 1. Avoid Decimal*Decimal → rational overflow
    /// 2. Avoid Binary*Binary → UGOD promotion → i128 truncation for large values
    /// Since BinaryCompute uses ComputeStorage (I256 for q64_64), it has enough range
    /// for intermediate squaring and other arithmetic.
    pub(crate) fn to_binary_value(&mut self, val: &StackValue) -> Result<StackValue, OverflowDetected> {
        match val {
            StackValue::BinaryCompute(..) => Ok(val.clone()),
            _ => {
                let compute = self.to_compute_storage(val)?;
                let tier = self.profile_max_binary_tier();
                Ok(StackValue::BinaryCompute(tier, compute, val.shadow()))
            }
        }
    }

    /// Halve a binary StackValue (divide by 2 via right-shift) — avoids rational conversion
    pub(crate) fn halve_binary(&self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        match value {
            StackValue::BinaryCompute(tier, val, _) => {
                Ok(StackValue::BinaryCompute(tier, compute_halve(val), CompactShadow::None))
            }
            StackValue::Binary(tier, val, _) => {
                #[cfg(table_format = "q256_256")]
                { Ok(StackValue::Binary(tier, val >> 1, CompactShadow::None)) }
                #[cfg(table_format = "q128_128")]
                { Ok(StackValue::Binary(tier, val >> 1, CompactShadow::None)) }
                #[cfg(table_format = "q64_64")]
                { Ok(StackValue::Binary(tier, val >> 1, CompactShadow::None)) }
            }
            other => {
                // Fallback to rational division for non-binary
                let _two = self.make_binary_int(2);
                // Can't call divide_values on &self so convert to binary first
                let binary_val = self.to_binary_storage(&other)?;
                let tier = self.profile_max_binary_tier();
                #[cfg(table_format = "q256_256")]
                { Ok(StackValue::Binary(tier, binary_val >> 1, CompactShadow::None)) }
                #[cfg(table_format = "q128_128")]
                { Ok(StackValue::Binary(tier, binary_val >> 1, CompactShadow::None)) }
                #[cfg(table_format = "q64_64")]
                { Ok(StackValue::Binary(tier, binary_val >> 1, CompactShadow::None)) }
            }
        }
    }

    /// Divide a binary value by another in Q-format (native binary division)
    pub(crate) fn binary_divide(&self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        // Handle BinaryCompute: if either operand is BinaryCompute, use compute-tier division
        match (&left, &right) {
            (StackValue::BinaryCompute(t, v1, _), StackValue::BinaryCompute(_, v2, _)) => {
                return Ok(StackValue::BinaryCompute(*t, compute_divide(*v1, *v2)?, CompactShadow::None));
            }
            (StackValue::BinaryCompute(t, v1, _), StackValue::Binary(_, v2, _)) => {
                let v2_compute = upscale_to_compute(*v2);
                return Ok(StackValue::BinaryCompute(*t, compute_divide(*v1, v2_compute)?, CompactShadow::None));
            }
            (StackValue::Binary(_, v1, _), StackValue::BinaryCompute(t, v2, _)) => {
                let v1_compute = upscale_to_compute(*v1);
                return Ok(StackValue::BinaryCompute(*t, compute_divide(v1_compute, *v2)?, CompactShadow::None));
            }
            (StackValue::BinaryCompute(t, v1, _), other) => {
                let other_compute = self.to_compute_storage(other)?;
                return Ok(StackValue::BinaryCompute(*t, compute_divide(*v1, other_compute)?, CompactShadow::None));
            }
            (other, StackValue::BinaryCompute(t, v2, _)) => {
                let other_compute = self.to_compute_storage(other)?;
                return Ok(StackValue::BinaryCompute(*t, compute_divide(other_compute, *v2)?, CompactShadow::None));
            }
            _ => {}
        }

        let num = self.to_binary_storage(&left)?;
        let den = self.to_binary_storage(&right)?;
        let tier = self.profile_max_binary_tier();

        #[cfg(table_format = "q256_256")]
        {
            // Q256.256: (num << 256) / den — use I1024 intermediate
            let num_wide = I1024::from_i512(num) << 256;
            let den_wide = I1024::from_i512(den);
            if den_wide == I1024::zero() { return Err(OverflowDetected::DivisionByZero); }
            let result = (num_wide / den_wide).as_i512();
            Ok(StackValue::Binary(tier, result, CompactShadow::None))
        }
        #[cfg(table_format = "q128_128")]
        {
            let num_wide = I512::from_i256(num) << 128;
            let den_wide = I512::from_i256(den);
            if den_wide == I512::zero() { return Err(OverflowDetected::DivisionByZero); }
            let result = (num_wide / den_wide).as_i256();
            Ok(StackValue::Binary(tier, result, CompactShadow::None))
        }
        #[cfg(table_format = "q64_64")]
        {
            let num_wide = I256::from_i128(num) << 64;
            let den_wide = I256::from_i128(den);
            if den_wide == I256::zero() { return Err(OverflowDetected::DivisionByZero); }
            let result = (num_wide / den_wide).as_i128();
            Ok(StackValue::Binary(tier, result, CompactShadow::None))
        }
    }

    /// sinh(x) = (exp(x) - exp(-x)) / 2
    pub(crate) fn evaluate_sinh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let exp_x = self.evaluate_exp(value.clone())?;
        let neg_x = self.negate_value(value)?;
        let exp_neg_x = self.evaluate_exp(neg_x)?;
        let diff = self.subtract_values(exp_x, exp_neg_x)?;
        self.halve_binary(diff)
    }

    /// cosh(x) = (exp(x) + exp(-x)) / 2
    pub(crate) fn evaluate_cosh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let exp_x = self.evaluate_exp(value.clone())?;
        let neg_x = self.negate_value(value)?;
        let exp_neg_x = self.evaluate_exp(neg_x)?;
        let sum = self.add_values(exp_x, exp_neg_x)?;
        self.halve_binary(sum)
    }

    /// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    /// Optimized: only 1 exp call instead of 3
    pub(crate) fn evaluate_tanh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let value = self.to_binary_value(&value)?;
        let two = self.make_binary_int(2);
        let two_x = self.multiply_values(two, value)?;
        let exp_2x = self.evaluate_exp(two_x)?;
        let one = self.make_binary_int(1);
        let numerator = self.subtract_values(exp_2x.clone(), one.clone())?;
        let denominator = self.add_values(exp_2x, one)?;
        self.binary_divide(numerator, denominator)
    }

    /// asinh(x) = ln(x + sqrt(x² + 1))
    pub(crate) fn evaluate_asinh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        // Convert to binary to avoid Decimal*Decimal → rational overflow
        let value = self.to_binary_value(&value)?;
        let x_sq = self.multiply_values(value.clone(), value.clone())?;
        let one = self.make_binary_int(1);
        let x_sq_plus_1 = self.add_values(x_sq, one)?;
        let sqrt_val = self.evaluate_sqrt(x_sq_plus_1)?;
        let sum = self.add_values(value, sqrt_val)?;
        self.evaluate_ln(sum)
    }

    /// acosh(x) = ln(x + sqrt(x² - 1))
    /// **DOMAIN**: x >= 1 (returns DomainError otherwise)
    pub(crate) fn evaluate_acosh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        // Domain check: x >= 1
        let binary_val = self.to_binary_storage(&value)?;
        let one_binary = self.to_binary_storage(&self.make_binary_int(1))?;
        if binary_val < one_binary { return Err(OverflowDetected::DomainError); }

        // Convert to binary to avoid Decimal*Decimal → rational overflow
        let value = self.to_binary_value(&value)?;
        let x_sq = self.multiply_values(value.clone(), value.clone())?;
        let one = self.make_binary_int(1);
        let x_sq_minus_1 = self.subtract_values(x_sq, one)?;
        let sqrt_val = self.evaluate_sqrt(x_sq_minus_1)?;
        let sum = self.add_values(value, sqrt_val)?;
        self.evaluate_ln(sum)
    }

    /// atanh(x) = ln((1+x)/(1-x)) / 2
    /// **DOMAIN**: |x| < 1 (returns DomainError otherwise)
    pub(crate) fn evaluate_atanh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        // Domain check: |x| < 1
        let binary_val = self.to_binary_storage(&value)?;
        let one_binary = self.to_binary_storage(&self.make_binary_int(1))?;

        // Check |x| >= 1 (profile-generic comparison since BinaryStorage implements Ord)
        #[cfg(table_format = "q256_256")]
        {
            let neg_one = I512::zero() - one_binary;
            if binary_val >= one_binary || binary_val <= neg_one {
                return Err(OverflowDetected::DomainError);
            }
        }
        #[cfg(table_format = "q128_128")]
        {
            let neg_one = I256::zero() - one_binary;
            if binary_val >= one_binary || binary_val <= neg_one {
                return Err(OverflowDetected::DomainError);
            }
        }
        #[cfg(table_format = "q64_64")]
        {
            if binary_val >= one_binary || binary_val <= -one_binary {
                return Err(OverflowDetected::DomainError);
            }
        }

        // Convert to binary to avoid cross-domain rational overflow
        let value = self.to_binary_value(&value)?;
        let one = self.make_binary_int(1);
        let one_plus_x = self.add_values(one.clone(), value.clone())?;
        let one_minus_x = self.subtract_values(one, value)?;
        let ratio = self.binary_divide(one_plus_x, one_minus_x)?;
        let ln_ratio = self.evaluate_ln(ratio)?;
        self.halve_binary(ln_ratio)
    }

    // ============================================================================
    // TRIGONOMETRIC FUNCTIONS
    // ============================================================================

    /// sin(x) — tier N+1 computation returning BinaryCompute
    pub(crate) fn evaluate_sin(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        let compute_val = self.to_compute_storage(&value)?;
        let result = sin_at_compute_tier(compute_val);
        Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
    }

    /// cos(x) — tier N+1 computation returning BinaryCompute
    pub(crate) fn evaluate_cos(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        let compute_val = self.to_compute_storage(&value)?;
        let result = cos_at_compute_tier(compute_val);
        Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
    }

    /// tan(x) = sin(x) / cos(x) — ZASC-composed at compute tier
    pub(crate) fn evaluate_tan(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        // Both sin and cos return BinaryCompute — division stays at compute tier
        let sin_val = self.evaluate_sin(value.clone())?;
        let cos_val = self.evaluate_cos(value)?;
        // Check for cos == 0 at compute tier
        if let StackValue::BinaryCompute(_, c, _) = &cos_val {
            if compute_is_zero(c) {
                return Err(OverflowDetected::DomainError);
            }
        }
        self.binary_divide(sin_val, cos_val)
    }

    /// asin(x) = atan(x / sqrt(1 - x²)) — ZASC-composed at compute tier
    /// Domain: |x| <= 1
    pub(crate) fn evaluate_asin(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        // Domain check: |x| <= 1
        let binary_val = self.to_binary_storage(&value)?;
        let one_bs = self.to_binary_storage(&self.make_binary_int(1))?;

        #[cfg(table_format = "q256_256")]
        {
            let abs_val = if binary_val < I512::zero() { I512::zero() - binary_val } else { binary_val };
            if abs_val > one_bs { return Err(OverflowDetected::DomainError); }
        }
        #[cfg(table_format = "q128_128")]
        {
            let abs_val = if binary_val < I256::zero() { I256::zero() - binary_val } else { binary_val };
            if abs_val > one_bs { return Err(OverflowDetected::DomainError); }
        }
        #[cfg(table_format = "q64_64")]
        {
            let abs_val = if binary_val < 0 { -binary_val } else { binary_val };
            if abs_val > one_bs { return Err(OverflowDetected::DomainError); }
        }

        // asin(x) = atan(x / sqrt(1 - x²))
        // Convert to binary to avoid Decimal*Decimal → rational overflow
        let value = self.to_binary_value(&value)?;
        let one = self.make_binary_int(1);
        let x_sq = self.multiply_values(value.clone(), value.clone())?;
        let one_minus_x_sq = self.subtract_values(one, x_sq)?;
        let sqrt_val = self.evaluate_sqrt(one_minus_x_sq)?;
        let ratio = self.binary_divide(value, sqrt_val)?;
        self.evaluate_atan(ratio)
    }

    /// acos(x) = π/2 - asin(x) — ZASC-composed at compute tier
    /// Domain: |x| <= 1
    pub(crate) fn evaluate_acos(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let asin_val = self.evaluate_asin(value)?;

        // π/2 at compute tier
        let storage_tier = self.profile_max_binary_tier();
        let pi_half_compute = pi_half_at_compute_tier();
        let pi_half_val = StackValue::BinaryCompute(storage_tier, pi_half_compute, CompactShadow::None);
        self.subtract_values(pi_half_val, asin_val)
    }

    /// atan(x) — tier N+1 computation returning BinaryCompute
    pub(crate) fn evaluate_atan(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        let compute_val = self.to_compute_storage(&value)?;
        let result = atan_at_compute_tier(compute_val);
        Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
    }

    /// atan2(y, x) — tier N+1 computation returning BinaryCompute
    pub(crate) fn evaluate_atan2(&mut self, y: StackValue, x: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        let y_compute = self.to_compute_storage(&y)?;
        let x_compute = self.to_compute_storage(&x)?;
        let result = atan2_at_compute_tier(y_compute, x_compute);
        Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
    }

    /// Convert any StackValue to ComputeStorage (tier N+1)
    ///
    /// - BinaryCompute: pass through directly
    /// - Binary: upscale to compute tier
    /// - Decimal/Symbolic/Ternary: convert to BinaryStorage first, then upscale
    pub(crate) fn to_compute_storage(&self, value: &StackValue) -> Result<ComputeStorage, OverflowDetected> {
        match value {
            StackValue::BinaryCompute(_, val, _) => Ok(*val),
            StackValue::Binary(_, val, _) => Ok(upscale_to_compute(*val)),
            // Convert Decimal/Symbolic DIRECTLY to compute tier at full precision.
            // This avoids the precision loss from to_binary_storage() → upscale
            // (which fills lower bits with zeros instead of real precision).
            StackValue::Decimal(decimals, scaled, _) => {
                decimal_to_compute_storage(*decimals, *scaled)
            }
            StackValue::Symbolic(rational) => {
                // Try i128 extraction first (tiers 1-5)
                if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
                    return symbolic_to_compute_storage(num, den);
                }
                // Fall back to wider extraction for Massive/Ultra tier rationals
                return symbolic_wide_to_compute_storage(rational);
            }
            StackValue::Ternary(tier, value, _) => {
                // Ternary: convert through rational (value / 3^frac_trits)
                let rational = ternary_to_rational(*tier, value)?;
                if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
                    symbolic_to_compute_storage(num, den)
                } else {
                    symbolic_wide_to_compute_storage(&rational)
                }
            }
            StackValue::Error(e) => Err(e.clone()),
        }
    }

    /// Convert any StackValue to profile-specific BinaryStorage format
    pub(crate) fn to_binary_storage(&self, value: &StackValue) -> Result<BinaryStorage, OverflowDetected> {
        match value {
            StackValue::Binary(_, val, _) => Ok(*val),
            StackValue::BinaryCompute(_, val, _) => Ok(downscale_to_storage(*val)),
            StackValue::Decimal(decimals, scaled, _) => {
                #[cfg(table_format = "q256_256")]
                {
                    use crate::fixed_point::I1024;
                    let ten_pow = pow10_i512(*decimals);
                    let scaled_i1024 = I1024::from_i512(*scaled) << 256;
                    Ok((scaled_i1024 / I1024::from_i512(ten_pow)).as_i512())
                }

                #[cfg(table_format = "q128_128")]
                {
                    let ten_pow = pow10_i256(*decimals);
                    let num = I512::from_i256(*scaled) << 128;
                    let den = I512::from_i256(ten_pow);
                    Ok((num / den).as_i256())
                }

                #[cfg(table_format = "q64_64")]
                {
                    let ten_pow = pow10_i256(*decimals);
                    let num = I256::from_i128(*scaled) << 64;
                    Ok((num / ten_pow).as_i128())
                }
            }
            StackValue::Symbolic(rational) => {
                // Try i128 extraction first (tiers 1-5)
                if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
                    #[cfg(table_format = "q256_256")]
                    {
                        use crate::fixed_point::I1024;
                        let num_i1024 = I1024::from_i512(I512::from_i256(I256::from_i128(num))) << 256;
                        let den_i512 = I512::from_i256(I256::from_i128(den));
                        return Ok((num_i1024 / I1024::from_i512(den_i512)).as_i512());
                    }

                    #[cfg(table_format = "q128_128")]
                    {
                        return Ok((I256::from_i128(num) << 128) / I256::from_i128(den));
                    }

                    #[cfg(table_format = "q64_64")]
                    {
                        return Ok((num << 64) / den);
                    }
                }
                // Fall back to wider extraction for Massive/Ultra tier rationals
                let cs = symbolic_wide_to_compute_storage(rational)?;
                Ok(downscale_to_storage(cs))
            }
            StackValue::Ternary(tier, value, _) => {
                // Ternary: convert through rational (value / 3^frac_trits), then to Q-format
                let rational = ternary_to_rational(*tier, value)?;
                if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
                    #[cfg(table_format = "q256_256")]
                    {
                        use crate::fixed_point::I1024;
                        let num_i1024 = I1024::from_i512(I512::from_i256(I256::from_i128(num))) << 256;
                        let den_i512 = I512::from_i256(I256::from_i128(den));
                        return Ok((num_i1024 / I1024::from_i512(den_i512)).as_i512());
                    }

                    #[cfg(table_format = "q128_128")]
                    {
                        return Ok((I256::from_i128(num) << 128) / I256::from_i128(den));
                    }

                    #[cfg(table_format = "q64_64")]
                    {
                        return Ok((num << 64) / den);
                    }
                } else {
                    let cs = symbolic_wide_to_compute_storage(&rational)?;
                    Ok(downscale_to_storage(cs))
                }
            }
            StackValue::Error(e) => Err(*e),
        }
    }

    // ========================================================================
    // MODE ROUTING — compute_mode:output_mode
    // ========================================================================

    /// Parse literal with mode override (Auto delegates to parse_literal unchanged)
    pub(crate) fn parse_literal_with_mode(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        use crate::fixed_point::universal::zasc::mode::{ComputeMode, get_mode};
        let mode = get_mode();
        match mode.compute {
            ComputeMode::Auto => self.parse_literal(s),
            ComputeMode::Binary => self.parse_as_binary(s),
            ComputeMode::Decimal => self.parse_as_decimal(s),
            ComputeMode::Symbolic => self.parse_as_symbolic(s),
            ComputeMode::Ternary => self.parse_as_ternary(s),
        }
    }

    /// Force-parse into Binary domain
    pub(crate) fn parse_as_binary(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let value = self.parse_literal(s)?;
        self.convert_to_binary(value)
    }

    /// Force-parse into Decimal domain
    pub(crate) fn parse_as_decimal(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let value = self.parse_literal(s)?;
        self.convert_to_decimal(value)
    }

    /// Force-parse into Symbolic domain
    pub(crate) fn parse_as_symbolic(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let value = self.parse_literal(s)?;
        self.convert_to_symbolic(value)
    }

    /// Force-parse into Ternary domain
    pub(crate) fn parse_as_ternary(&mut self, s: &str) -> Result<StackValue, OverflowDetected> {
        let value = self.parse_literal(s)?;
        self.convert_to_ternary(value)
    }

    /// Convert any StackValue to Binary domain
    pub(crate) fn convert_to_binary(&self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        match &value {
            StackValue::Binary(_, _, _) => Ok(value),
            StackValue::BinaryCompute(_, _, _) => Ok(self.materialize_compute(value)),
            _ => {
                let storage = self.to_binary_storage(&value)?;
                let tier = self.profile_max_binary_tier();
                let shadow = value.shadow();
                Ok(StackValue::Binary(tier, storage, shadow))
            }
        }
    }

    /// Convert any StackValue to Decimal domain
    pub(crate) fn convert_to_decimal(&self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        match &value {
            StackValue::Decimal(_, _, _) => Ok(value),
            _ => {
                let rational = value.to_rational()?;
                let (decimals, scaled) = rational_to_decimal_components(&rational)?;
                let shadow = value.shadow();
                Ok(StackValue::Decimal(decimals, to_binary_storage(scaled), shadow))
            }
        }
    }

    /// Convert any StackValue to Symbolic domain
    pub(crate) fn convert_to_symbolic(&self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        match &value {
            StackValue::Symbolic(_) => Ok(value),
            _ => {
                let rational = value.to_rational()?;
                Ok(StackValue::Symbolic(rational))
            }
        }
    }

    /// Convert any StackValue to Ternary domain using UGOD profile-aware tier selection
    ///
    /// Uses rational-based conversion: extract exact rational (num/den), then compute
    /// `num * 3^frac_trits / den` at the profile-appropriate ternary tier.
    ///
    /// **UGOD STRATEGY**: Profile determines tier (matching convert_to_binary pattern):
    ///   - Embedded → Tier 3 TQ32.32 (i128 arithmetic, 32 frac trits)
    ///   - Balanced → Tier 4 TQ64.64 (I256 arithmetic, 64 frac trits)
    ///   - Scientific → Tier 5 TQ128.128 (I512 arithmetic, 128 frac trits)
    pub(crate) fn convert_to_ternary(&self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        match &value {
            StackValue::Ternary(_, _, _) => Ok(value),
            _ => {
                let shadow = value.shadow();
                let rational = value.to_rational()?;
                let tier = self.profile_max_ternary_tier();

                // Q256.256 tier 5: extract I512 directly to handle Ultra-tier rationals
                // (numerator_i128() returns None for Ultra-tier from binary to_rational)
                #[cfg(table_format = "q256_256")]
                {
                    if tier == 5 {
                        let parts = rational.extract_native();
                        let (num512, den512) = parts.try_as_i512_pair()
                            .ok_or(OverflowDetected::Overflow)?;
                        if den512.is_zero() {
                            return Err(OverflowDetected::DivisionByZero);
                        }
                        let scale = {
                            let mut s = I512::from_i128(1);
                            let three = I512::from_i128(3);
                            for _ in 0..128 { s = s * three; }
                            s
                        };
                        // Split to avoid overflow: int_part * scale + (rem * scale) / den
                        let int_part = num512 / den512;
                        let remainder = num512 - int_part * den512;
                        let stored = int_part * scale + (remainder * scale) / den512;
                        let (t, bs) = ternary_to_storage(
                            &UniversalTernaryFixed::from_tier_raw(tier, TernaryRaw::Large(stored))?
                        );
                        return Ok(StackValue::Ternary(t, bs, shadow));
                    }
                }

                // Other tiers: i128 extraction (sufficient for embedded/balanced)
                let num = rational.numerator_i128().ok_or(OverflowDetected::Overflow)?;
                let den = rational.denominator_i128().ok_or(OverflowDetected::Overflow)?;
                if den == 0 {
                    return Err(OverflowDetected::DivisionByZero);
                }
                match tier {
                    // Tier 3: TQ32.32 — 32 frac trits, i128 arithmetic
                    3 => {
                        // 3^32 = 1,853,020,188,851,841
                        let scale: i128 = 1_853_020_188_851_841;
                        let stored = if let Some(product) = num.checked_mul(scale) {
                            product / den
                        } else {
                            let quotient = num / den;
                            let remainder = num % den;
                            quotient.checked_mul(scale).ok_or(OverflowDetected::Overflow)?
                                + remainder.checked_mul(scale).ok_or(OverflowDetected::Overflow)? / den
                        };
                        Ok(StackValue::Ternary(tier, to_binary_storage(stored), shadow))
                    }
                    // Tier 4: TQ64.64 — 64 frac trits, I256 arithmetic
                    4 => {
                        let scale = {
                            let mut s = I256::from_u8(1);
                            let three = I256::from_u8(3);
                            for _ in 0..64 { s = s * three; }
                            s
                        };
                        let num256 = I256::from_i128(num);
                        let den256 = I256::from_i128(den);
                        let stored = (num256 * scale) / den256;
                        let (t, bs) = ternary_to_storage(
                            &UniversalTernaryFixed::from_tier_raw(tier, TernaryRaw::Medium(stored))?
                        );
                        Ok(StackValue::Ternary(t, bs, shadow))
                    }
                    // Tier 5: TQ128.128 — handled by cfg block above on Q256.256
                    5 => {
                        let scale = {
                            let mut s = I512::from_i128(1);
                            let three = I512::from_i128(3);
                            for _ in 0..128 { s = s * three; }
                            s
                        };
                        let num512 = I512::from_i128(num);
                        let den512 = I512::from_i128(den);
                        let stored = (num512 * scale) / den512;
                        let (t, bs) = ternary_to_storage(
                            &UniversalTernaryFixed::from_tier_raw(tier, TernaryRaw::Large(stored))?
                        );
                        Ok(StackValue::Ternary(t, bs, shadow))
                    }
                    // Fallback: use tier 3 for any unexpected tier value
                    _ => {
                        let scale: i128 = 1_853_020_188_851_841;
                        let stored = if let Some(product) = num.checked_mul(scale) {
                            product / den
                        } else {
                            let quotient = num / den;
                            let remainder = num % den;
                            quotient.checked_mul(scale).ok_or(OverflowDetected::Overflow)?
                                + remainder.checked_mul(scale).ok_or(OverflowDetected::Overflow)? / den
                        };
                        Ok(StackValue::Ternary(3, to_binary_storage(stored), shadow))
                    }
                }
            }
        }
    }

    /// Apply output mode conversion after evaluation
    pub(crate) fn apply_output_mode(&self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        use crate::fixed_point::universal::zasc::mode::{OutputMode, get_mode};
        let mode = get_mode();
        match mode.output {
            OutputMode::Auto => Ok(value),
            OutputMode::Binary => self.convert_to_binary(value),
            OutputMode::Decimal => self.convert_to_decimal(value),
            OutputMode::Symbolic => self.convert_to_symbolic(value),
            OutputMode::Ternary => self.convert_to_ternary(value),
        }
    }
}

/// Convert rational to (decimal_places, scaled_value) for Decimal domain construction.
/// Uses I256 intermediate arithmetic to avoid i128 overflow when num * 10^decimals
/// exceeds i128 range (e.g., large-numerator rationals from multiply_via_rational).
/// On Q256.256, falls back to I512 arithmetic for Ultra-tier rationals.
pub(super) fn rational_to_decimal_components(rational: &RationalNumber) -> Result<(u8, i128), OverflowDetected> {
    // Try i128 extraction (works for all profiles ≤ Q128.128 and reduced rationals)
    if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
        return rational_to_decimal_components_i128(num, den);
    }

    // Q256.256 fallback: extract as I512 for Ultra-tier rationals
    #[cfg(table_format = "q256_256")]
    {
        let parts = rational.extract_native();
        if let Some((num, den)) = parts.try_as_i512_pair() {
            return rational_to_decimal_components_i512(num, den);
        }
    }

    Err(OverflowDetected::Overflow)
}

/// i128 path for rational_to_decimal_components (original algorithm)
fn rational_to_decimal_components_i128(num: i128, den: i128) -> Result<(u8, i128), OverflowDetected> {
    if den == 0 {
        return Err(OverflowDetected::DivisionByZero);
    }
    // Factor out 2s and 5s from denominator to find exact decimal places
    let mut d = den.unsigned_abs() as u128;
    let mut decimals: u32 = 0;
    while d % 10 == 0 { d /= 10; decimals += 1; }
    while d % 5 == 0 { d /= 5; decimals += 1; }
    while d % 2 == 0 { d /= 2; decimals += 1; }
    if d != 1 {
        // Not exactly representable in decimal — use max precision
        decimals = 19;
    }
    if decimals > 19 {
        decimals = 19;
    }
    // I256 intermediate: handles num(≤1.7e38) * 10^19(=1e19) = up to ~1.7e57,
    // well within I256 range (~5.8e76).
    let num_wide = I256::from_i128(num);
    let den_wide = I256::from_i128(den);
    let ten = I256::from_i128(10);
    let mut scale_wide = I256::from_i128(1);
    for _ in 0..decimals { scale_wide = scale_wide * ten; }
    let scaled_wide = num_wide * scale_wide / den_wide;
    if scaled_wide.fits_in_i128() {
        return Ok((decimals as u8, scaled_wide.as_i128()));
    }
    // Result exceeds i128 — reduce decimal places until it fits
    let mut dec = decimals;
    while dec > 0 {
        dec -= 1;
        let mut s = I256::from_i128(1);
        for _ in 0..dec { s = s * ten; }
        let result = num_wide * s / den_wide;
        if result.fits_in_i128() {
            return Ok((dec as u8, result.as_i128()));
        }
    }
    let result = num_wide / den_wide;
    if result.fits_in_i128() {
        Ok((0, result.as_i128()))
    } else {
        Err(OverflowDetected::Overflow)
    }
}

/// I512 path for rational_to_decimal_components (Q256.256 Ultra-tier fallback).
/// Computes num * 10^decimals / den using I512 arithmetic, splitting into
/// integer_part and fractional_part to avoid overflow.
#[cfg(table_format = "q256_256")]
fn rational_to_decimal_components_i512(num: I512, den: I512) -> Result<(u8, i128), OverflowDetected> {
    if den.is_zero() {
        return Err(OverflowDetected::DivisionByZero);
    }
    // For Ultra-tier rationals (denominators like 2^256 or 3^128),
    // exact decimal representation is rarely possible — use max i128 precision.
    let decimals: u32 = 19;
    let ten = I512::from_i128(10);
    let mut scale = I512::from_i128(1);
    for _ in 0..decimals { scale = scale * ten; }

    // Split computation to avoid I512 overflow:
    // result = integer_part * scale + (remainder * scale) / den
    let integer_part = num / den;
    let remainder = num - integer_part * den;
    let frac_scaled = (remainder * scale) / den;
    let result = integer_part * scale + frac_scaled;

    if result.fits_in_i128() {
        return Ok((decimals as u8, result.as_i128()));
    }
    // Reduce decimal places until it fits
    let mut dec = decimals;
    while dec > 0 {
        dec -= 1;
        let mut s = I512::from_i128(1);
        for _ in 0..dec { s = s * ten; }
        let int_part = num / den;
        let rem = num - int_part * den;
        let frac = (rem * s) / den;
        let r = int_part * s + frac;
        if r.fits_in_i128() {
            return Ok((dec as u8, r.as_i128()));
        }
    }
    let r = num / den;
    if r.fits_in_i128() {
        Ok((0, r.as_i128()))
    } else {
        Err(OverflowDetected::Overflow)
    }
}
