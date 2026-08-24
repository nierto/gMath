//! Transcendental function evaluation — 18 functions with BinaryCompute chain persistence
//!
//! 5 dedicated (exp, ln, sqrt, pow, atan2) + 13 FASC-composed from core functions.
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

/// True when a binary exp result sits at its overflow sentinel — building
/// on it (cosh/tanh adds, sinh halving) would produce a plausible-wrong
/// value. The old `== compute_ceiling()` check missed the q64_64/q128_128
/// engines' `i128::MAX`-at-storage-scale sentinel (0.5.0 item 2 find);
/// the shared per-profile predicate covers every engine.
fn exp_at_compute_ceiling(v: &StackValue) -> bool {
    use crate::fixed_point::universal::fasc::stack_evaluator::compute::exp_sentinel_reached;
    matches!(v, StackValue::BinaryCompute(_, val, _) if exp_sentinel_reached(val))
}

// Transcendental functions called directly (not re-exported through compute::*)
#[cfg(table_format = "q256_256")]
use crate::fixed_point::domains::binary_fixed::transcendental::ln_binary_i1024;
#[cfg(table_format = "q128_128")]
use crate::fixed_point::domains::binary_fixed::transcendental::{exp_binary_i512, ln_binary_i512};
// Native tier dispatch: Q64.64 uses I256, Q32.32 uses i128, Q16.16 uses i64
#[cfg(table_format = "q64_64")]
use crate::fixed_point::domains::binary_fixed::transcendental::{exp_binary_i256, ln_binary_i256};
#[cfg(table_format = "q32_32")]
use crate::fixed_point::domains::binary_fixed::transcendental::{exp_binary_i128, ln_binary_i128};
#[cfg(table_format = "q16_16")]
use crate::fixed_point::domains::binary_fixed::transcendental::{exp_binary_i64, ln_binary_i64};

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
        #[cfg(table_format = "q32_32")]
        {
            // ComputeStorage = i128 (Q64.64) — native Q64.64 computation
            let result = exp_binary_i128(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
        #[cfg(table_format = "q16_16")]
        {
            // ComputeStorage = i64 — upscale to Q64.64 (i128), compute, downscale
            let result = exp_binary_i64(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
    }

    /// Extract decimal compute-tier value if input is Decimal or DecimalCompute.
    ///
    /// Returns `Some(compute_val)` for decimal-domain inputs, `None` otherwise.
    /// Used to route transcendental evaluation to decimal engines when the input
    /// came from the decimal domain, preserving decimal exactness end-to-end.
    pub(crate) fn try_decimal_compute(&self, value: &StackValue) -> Option<ComputeStorage> {
        use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_upscale_to_compute;
        match value {
            StackValue::DecimalCompute(_, val, _) => Some(*val),
            StackValue::Decimal(dp, scaled, _) => decimal_upscale_to_compute(*scaled, *dp).ok(),
            _ => None,
        }
    }

    /// Evaluate exponential function on stack value at tier N+1
    ///
    /// Routes to decimal engine for decimal-domain inputs, binary engine otherwise.
    /// Returns BinaryCompute or DecimalCompute for transcendental chain persistence.
    pub(crate) fn evaluate_exp(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        // Decimal fast path: preserve decimal exactness through the chain
        if let Some(dec_compute) = self.try_decimal_compute(&value) {
            use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_exp;
            let result = decimal_exp(dec_compute)?;
            return Ok(StackValue::DecimalCompute(storage_tier, result, CompactShadow::None));
        }
        // Binary path (default)
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
        #[cfg(table_format = "q32_32")]
        {
            // ComputeStorage = i128 (Q64.64) — native Q64.64 computation
            let result = ln_binary_i128(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
        #[cfg(table_format = "q16_16")]
        {
            // ComputeStorage = i64 — upscale to Q64.64 (i128), compute, downscale
            let result = ln_binary_i64(compute_val);
            Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
        }
    }

    /// Evaluate natural logarithm on stack value at tier N+1
    ///
    /// All inputs are upscaled to compute tier (tier N+1) before computation.
    /// Returns BinaryCompute for transcendental chain persistence.
    ///
    /// **DOMAIN ERROR**: Returns DomainError for x <= 0
    pub(crate) fn evaluate_ln(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        // Decimal fast path
        if let Some(dec_compute) = self.try_decimal_compute(&value) {
            use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_ln;
            let result = decimal_ln(dec_compute)?;
            return Ok(StackValue::DecimalCompute(storage_tier, result, CompactShadow::None));
        }
        let compute_val = self.to_compute_storage(&value)?;
        self.ln_at_compute_tier(compute_val, storage_tier)
    }

    // ============================================================================
    // SQUARE ROOT AND POWER FUNCTION EVALUATION
    // ============================================================================

    /// sqrt(x) — tier N+1 computation returning BinaryCompute or DecimalCompute
    /// **DOMAIN**: x >= 0 (returns error for x < 0)
    pub(crate) fn evaluate_sqrt(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        // Decimal fast path
        if let Some(dec_compute) = self.try_decimal_compute(&value) {
            use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_sqrt;
            let result = decimal_sqrt(dec_compute)?;
            return Ok(StackValue::DecimalCompute(storage_tier, result, CompactShadow::None));
        }
        let compute_val = self.to_compute_storage(&value)?;

        // Domain check: sqrt(x) undefined for x < 0
        if compute_is_negative(&compute_val) {
            return Err(OverflowDetected::DomainError);
        }

        let result = sqrt_at_compute_tier(compute_val);
        Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
    }

    /// pow(x, y) = exp(y × ln(x)) — FASC-composed at compute tier
    ///
    /// The entire chain (ln → multiply → exp) stays at compute tier via BinaryCompute propagation.
    pub(crate) fn evaluate_pow(&mut self, base: StackValue, exponent: StackValue) -> Result<StackValue, OverflowDetected> {
        // Fast path: integer exponent via shadow → exponentiation-by-squaring (exact)
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
    /// Exact for integer exponents. Negative exponents handled
    /// via 1/x^|n| using divide_at_compute.
    pub(crate) fn pow_integer(&mut self, base: StackValue, exp: i64) -> Result<StackValue, OverflowDetected> {
        let negative = exp < 0;
        let mut n = exp.unsigned_abs();

        if n == 0 {
            return Ok(self.make_int_like(&base, 1));
        }

        // to_compute_value preserves decimal domain (returns DecimalCompute for decimal input)
        let mut b = self.to_compute_value(&base)?;
        // Start with "1" in the same domain as base — avoids Binary×DecimalCompute mixing
        let mut result = self.make_int_like(&b, 1);

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
            let one = self.make_int_like(&result, 1);
            result = self.divide_at_compute(one, result)?;
        }

        Ok(result)
    }

    // ============================================================================
    // HYPERBOLIC FUNCTIONS (FASC-composed from exp/ln/sqrt)
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
        #[cfg(table_format = "q32_32")]
        { StackValue::Binary(tier, (value as i64) << 32, CompactShadow::from_rational(value, 1)) }
        #[cfg(table_format = "q16_16")]
        {
            use crate::fixed_point::frac_config;
            StackValue::Binary(tier, (value as i32) << frac_config::FRAC_BITS, CompactShadow::from_rational(value, 1))
        }
    }

    /// Create an integer constant in the same domain as `reference`.
    ///
    /// Returns `DecimalCompute` if `reference` is `Decimal`/`DecimalCompute`,
    /// otherwise `Binary`. Used by composed transcendentals (asinh, acosh, atanh,
    /// asin, etc.) to avoid decimal/binary domain mixing during composition.
    pub(crate) fn make_int_like(&self, reference: &StackValue, value: i128) -> StackValue {
        match reference {
            StackValue::DecimalCompute(..) | StackValue::Decimal(..) => {
                use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_compute_from_int;
                let tier = self.profile_max_binary_tier();
                StackValue::DecimalCompute(
                    tier,
                    decimal_compute_from_int(value as i64),
                    CompactShadow::from_rational(value, 1),
                )
            }
            _ => self.make_binary_int(value),
        }
    }

    /// PATH: FASC (canonical) composed-transcendental machinery. Domain-agnostic
    /// despite the legacy `binary` ancestry — the imperative `FixedPoint` /
    /// `DecimalFixed` types do NOT use these helpers (they call the compute-tier
    /// engines directly).
    ///
    /// Convert any StackValue to a compute-tier form suitable for transcendental
    /// composition, preserving domain:
    /// - `BinaryCompute` → unchanged (already at binary compute tier)
    /// - `DecimalCompute` → unchanged (already at decimal compute tier)
    /// - `Decimal` → `DecimalCompute` (upscale, preserve decimal identity)
    /// - Everything else (incl. `Symbolic`) → `BinaryCompute`
    ///
    /// This keeps composed transcendentals (sinh, asinh, tan, ...) in their
    /// native domain through the whole chain. NOTE: the final arm sends a
    /// `Symbolic` operand to binary compute; this is correct-rounded in practice
    /// (the compute tier absorbs representation error — measured 0 ULP), but a
    /// future guard could route `Symbolic` through decimal compute for domain
    /// consistency. See tests/oracle_golden/FINDINGS.md.
    pub(crate) fn to_compute_value(&mut self, val: &StackValue) -> Result<StackValue, OverflowDetected> {
        match val {
            StackValue::BinaryCompute(..) => Ok(val.clone()),
            StackValue::DecimalCompute(..) => Ok(val.clone()),
            StackValue::Decimal(dp, scaled, shadow) => {
                use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_upscale_to_compute;
                let dec_compute = decimal_upscale_to_compute(*scaled, *dp)?;
                let tier = self.profile_max_binary_tier();
                Ok(StackValue::DecimalCompute(tier, dec_compute, shadow.clone()))
            }
            _ => {
                let compute = self.to_compute_storage(val)?;
                let tier = self.profile_max_binary_tier();
                Ok(StackValue::BinaryCompute(tier, compute, val.shadow()))
            }
        }
    }

    /// PATH: FASC composed-transcendental helper. Domain-agnostic divide-by-2:
    /// `DecimalCompute` → `decimal_compute_halve`, `BinaryCompute`/`Binary` →
    /// arithmetic right-shift. Preserves the operand's domain (the `binary` in
    /// the old name was a misnomer).
    pub(crate) fn halve_value(&self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        match value {
            StackValue::BinaryCompute(tier, val, _) => {
                Ok(StackValue::BinaryCompute(tier, compute_halve(val), CompactShadow::None))
            }
            StackValue::DecimalCompute(tier, val, _) => {
                use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_compute_halve;
                Ok(StackValue::DecimalCompute(tier, decimal_compute_halve(val), CompactShadow::None))
            }
            StackValue::Binary(tier, val, _) => {
                #[cfg(table_format = "q256_256")]
                { Ok(StackValue::Binary(tier, val >> 1, CompactShadow::None)) }
                #[cfg(table_format = "q128_128")]
                { Ok(StackValue::Binary(tier, val >> 1, CompactShadow::None)) }
                #[cfg(table_format = "q64_64")]
                { Ok(StackValue::Binary(tier, val >> 1, CompactShadow::None)) }
                #[cfg(table_format = "q32_32")]
                { Ok(StackValue::Binary(tier, val >> 1, CompactShadow::None)) }
                #[cfg(table_format = "q16_16")]
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
                #[cfg(table_format = "q32_32")]
                { Ok(StackValue::Binary(tier, binary_val >> 1, CompactShadow::None)) }
                #[cfg(table_format = "q16_16")]
                { Ok(StackValue::Binary(tier, binary_val >> 1, CompactShadow::None)) }
            }
        }
    }

    /// PATH: FASC composed-transcendental helper — the compute-tier divide used
    /// inside transcendental composition. Distinct from `arithmetic::divide_values`
    /// (the general `&mut self` arithmetic divide with UGOD/rational fallback);
    /// this one is `&self` and stays at the compute tier. Domain-agnostic
    /// `left / right`: `DecimalCompute` operands stay decimal
    /// (`decimal_compute_div`), `BinaryCompute` stay binary; unrecognized domains
    /// fall back to binary Q-format division. (Old name `binary_divide`.)
    pub(crate) fn divide_at_compute(&self, left: StackValue, right: StackValue) -> Result<StackValue, OverflowDetected> {
        // DecimalCompute propagation — mirror the BinaryCompute pattern for decimal
        use crate::fixed_point::domains::decimal_fixed::transcendental::{
            decimal_compute_div, decimal_upscale_to_compute,
        };
        match (&left, &right) {
            (StackValue::DecimalCompute(t, v1, _), StackValue::DecimalCompute(_, v2, _)) => {
                return Ok(StackValue::DecimalCompute(*t, decimal_compute_div(*v1, *v2)?, CompactShadow::None));
            }
            (StackValue::DecimalCompute(t, v1, _), StackValue::Decimal(dp, scaled, _)) => {
                let v2 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t, decimal_compute_div(*v1, v2)?, CompactShadow::None));
            }
            (StackValue::Decimal(dp, scaled, _), StackValue::DecimalCompute(t, v2, _)) => {
                let v1 = decimal_upscale_to_compute(*scaled, *dp)?;
                return Ok(StackValue::DecimalCompute(*t, decimal_compute_div(v1, *v2)?, CompactShadow::None));
            }
            _ => {}
        }
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
        #[cfg(table_format = "q32_32")]
        {
            // Q32.32: (num << 32) / den — use i128 intermediate
            let num_wide = (num as i128) << 32;
            let den_wide = den as i128;
            if den_wide == 0 { return Err(OverflowDetected::DivisionByZero); }
            let result = (num_wide / den_wide) as i64;
            Ok(StackValue::Binary(tier, result, CompactShadow::None))
        }
        #[cfg(table_format = "q16_16")]
        {
            use crate::fixed_point::frac_config;
            let num_wide = (num as i64) << frac_config::FRAC_BITS;
            let den_wide = den as i64;
            if den_wide == 0 { return Err(OverflowDetected::DivisionByZero); }
            let result = (num_wide / den_wide) as i32;
            Ok(StackValue::Binary(tier, result, CompactShadow::None))
        }
    }

    /// sinh(x) = (exp(x) - exp(-x)) / 2
    pub(crate) fn evaluate_sinh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let exp_x = self.evaluate_exp(value.clone())?;
        let neg_x = self.negate_value(value)?;
        let exp_neg_x = self.evaluate_exp(neg_x)?;
        let diff = self.subtract_values(exp_x, exp_neg_x)?;
        self.halve_value(diff)
    }

    /// cosh(x) = (exp(x) + exp(-x)) / 2
    pub(crate) fn evaluate_cosh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let exp_x = self.evaluate_exp(value.clone())?;
        let neg_x = self.negate_value(value)?;
        let exp_neg_x = self.evaluate_exp(neg_x)?;
        // A ceiling exp (saturated downscale / overflow sentinel) means cosh
        // already exceeds the storage tier — and the add below would wrap.
        if exp_at_compute_ceiling(&exp_x) || exp_at_compute_ceiling(&exp_neg_x) {
            return Err(OverflowDetected::TierOverflow);
        }
        let sum = self.add_values(exp_x, exp_neg_x)?;
        self.halve_value(sum)
    }

    /// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    /// Optimized: only 1 exp call instead of 3
    pub(crate) fn evaluate_tanh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let value = self.to_compute_value(&value)?;
        let two = self.make_int_like(&value, 2);
        let two_x = self.multiply_values(two, value.clone())?;
        let exp_2x = self.evaluate_exp(two_x)?;
        // exp(2x) at the compute-tier ceiling: tanh = 1 − 2/(exp(2x)+1) rounds
        // to exactly 1 at every storage width, and the +1 below would wrap.
        if exp_at_compute_ceiling(&exp_2x) {
            return Ok(self.make_int_like(&value, 1));
        }
        let one = self.make_int_like(&value, 1);
        let numerator = self.subtract_values(exp_2x.clone(), one.clone())?;
        let denominator = self.add_values(exp_2x, one)?;
        self.divide_at_compute(numerator, denominator)
    }

    /// asinh(x) = ln(x + sqrt(x² + 1))
    pub(crate) fn evaluate_asinh(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        // to_compute_value preserves decimal domain (returns DecimalCompute for decimal input)
        let value = self.to_compute_value(&value)?;
        let x_sq = self.multiply_values(value.clone(), value.clone())?;
        let one = self.make_int_like(&value, 1);
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

        // to_compute_value preserves decimal domain
        let value = self.to_compute_value(&value)?;
        let x_sq = self.multiply_values(value.clone(), value.clone())?;
        let one = self.make_int_like(&value, 1);
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
        #[cfg(table_format = "q32_32")]
        {
            if binary_val >= one_binary || binary_val <= -one_binary {
                return Err(OverflowDetected::DomainError);
            }
        }
        #[cfg(table_format = "q16_16")]
        {
            if binary_val >= one_binary || binary_val <= -one_binary {
                return Err(OverflowDetected::DomainError);
            }
        }

        // to_compute_value preserves decimal domain
        let value = self.to_compute_value(&value)?;
        let one = self.make_int_like(&value, 1);
        let one_plus_x = self.add_values(one.clone(), value.clone())?;
        let one_minus_x = self.subtract_values(one, value)?;
        let ratio = self.divide_at_compute(one_plus_x, one_minus_x)?;
        let ln_ratio = self.evaluate_ln(ratio)?;
        self.halve_value(ln_ratio)
    }

    // ============================================================================
    // TRIGONOMETRIC FUNCTIONS
    // ============================================================================

    /// sin(x) — tier N+1 computation returning BinaryCompute or DecimalCompute
    pub(crate) fn evaluate_sin(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        if let Some(dec_compute) = self.try_decimal_compute(&value) {
            use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_sin;
            let result = decimal_sin(dec_compute)?;
            return Ok(StackValue::DecimalCompute(storage_tier, result, CompactShadow::None));
        }
        let compute_val = self.to_compute_storage(&value)?;
        let result = sin_at_compute_tier(compute_val);
        Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
    }

    /// cos(x) — tier N+1 computation returning BinaryCompute or DecimalCompute
    pub(crate) fn evaluate_cos(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        if let Some(dec_compute) = self.try_decimal_compute(&value) {
            use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_cos;
            let result = decimal_cos(dec_compute)?;
            return Ok(StackValue::DecimalCompute(storage_tier, result, CompactShadow::None));
        }
        let compute_val = self.to_compute_storage(&value)?;
        let result = cos_at_compute_tier(compute_val);
        Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
    }

    /// Fused sin+cos at compute tier — single shared range reduction.
    /// Returns (sin_val, cos_val) both as BinaryCompute or DecimalCompute.
    pub(crate) fn evaluate_sincos(&mut self, value: StackValue) -> Result<(StackValue, StackValue), OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        if let Some(dec_compute) = self.try_decimal_compute(&value) {
            use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_sincos;
            let (sin_result, cos_result) = decimal_sincos(dec_compute)?;
            return Ok((
                StackValue::DecimalCompute(storage_tier, sin_result, CompactShadow::None),
                StackValue::DecimalCompute(storage_tier, cos_result, CompactShadow::None),
            ));
        }
        let compute_val = self.to_compute_storage(&value)?;
        let (sin_result, cos_result) = sincos_at_compute_tier(compute_val);
        Ok((
            StackValue::BinaryCompute(storage_tier, sin_result, CompactShadow::None),
            StackValue::BinaryCompute(storage_tier, cos_result, CompactShadow::None),
        ))
    }

    /// Fused sinh+cosh at compute tier — single shared exp-pair evaluation.
    /// Returns (sinh_val, cosh_val) both as BinaryCompute or DecimalCompute.
    ///
    /// Routes to `decimal_sinhcosh` for decimal-domain inputs, binary
    /// `sinhcosh_at_compute_tier` otherwise. sinh and cosh come from the
    /// same `(exp(x), exp(-x))` pair, preserving their error correlation.
    pub(crate) fn evaluate_sinhcosh(&mut self, value: StackValue) -> Result<(StackValue, StackValue), OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        if let Some(dec_compute) = self.try_decimal_compute(&value) {
            use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_sinhcosh;
            let (sinh_result, cosh_result) = decimal_sinhcosh(dec_compute)?;
            return Ok((
                StackValue::DecimalCompute(storage_tier, sinh_result, CompactShadow::None),
                StackValue::DecimalCompute(storage_tier, cosh_result, CompactShadow::None),
            ));
        }
        let compute_val = self.to_compute_storage(&value)?;
        let (sinh_result, cosh_result) = sinhcosh_at_compute_tier(compute_val);
        Ok((
            StackValue::BinaryCompute(storage_tier, sinh_result, CompactShadow::None),
            StackValue::BinaryCompute(storage_tier, cosh_result, CompactShadow::None),
        ))
    }

    /// tan(x) = sin(x) / cos(x) — uses fused sincos for single range reduction
    pub(crate) fn evaluate_tan(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let (sin_val, cos_val) = self.evaluate_sincos(value)?;
        // Check for cos == 0 at compute tier
        if let StackValue::BinaryCompute(_, c, _) = &cos_val {
            if compute_is_zero(c) {
                return Err(OverflowDetected::DomainError);
            }
        }
        self.divide_at_compute(sin_val, cos_val)
    }

    /// asin(x) = atan(x / sqrt(1 - x²)) — FASC-composed at compute tier
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
        #[cfg(table_format = "q32_32")]
        {
            let abs_val = if binary_val < 0 { -binary_val } else { binary_val };
            if abs_val > one_bs { return Err(OverflowDetected::DomainError); }
        }
        #[cfg(table_format = "q16_16")]
        {
            let abs_val = if binary_val < 0 { -binary_val } else { binary_val };
            if abs_val > one_bs { return Err(OverflowDetected::DomainError); }
        }

        // Boundary cases: asin(±1) = ±π/2 exactly (avoids division by sqrt(0))
        if binary_val == one_bs {
            let storage_tier = self.profile_max_binary_tier();
            let pi_half = pi_half_at_compute_tier();
            return Ok(StackValue::BinaryCompute(storage_tier, pi_half, CompactShadow::None));
        }
        #[cfg(table_format = "q64_64")]
        let neg_one_bs: i128 = -one_bs;
        #[cfg(table_format = "q128_128")]
        let neg_one_bs = I256::zero() - one_bs;
        #[cfg(table_format = "q256_256")]
        let neg_one_bs = I512::zero() - one_bs;
        #[cfg(table_format = "q32_32")]
        let neg_one_bs: i64 = -one_bs;
        #[cfg(table_format = "q16_16")]
        let neg_one_bs: i32 = -one_bs;
        if binary_val == neg_one_bs {
            let storage_tier = self.profile_max_binary_tier();
            let pi_half = pi_half_at_compute_tier();
            return Ok(StackValue::BinaryCompute(storage_tier, -pi_half, CompactShadow::None));
        }

        // asin(x) = atan(x / sqrt(1 - x²))
        // to_compute_value preserves decimal domain
        let value = self.to_compute_value(&value)?;
        let one = self.make_int_like(&value, 1);
        let x_sq = self.multiply_values(value.clone(), value.clone())?;
        let one_minus_x_sq = self.subtract_values(one, x_sq)?;
        let sqrt_val = self.evaluate_sqrt(one_minus_x_sq)?;
        let ratio = self.divide_at_compute(value, sqrt_val)?;
        self.evaluate_atan(ratio)
    }

    /// acos(x) = π/2 - asin(x) — FASC-composed at compute tier
    /// Domain: |x| <= 1
    pub(crate) fn evaluate_acos(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let asin_val = self.evaluate_asin(value)?;

        // π/2 at compute tier — match asin_val's domain
        let storage_tier = self.profile_max_binary_tier();
        let pi_half_val = match &asin_val {
            StackValue::DecimalCompute(..) => {
                // Use decimal π/2 = pi_at_decimal_compute / 2
                use crate::fixed_point::domains::decimal_fixed::transcendental::{
                    pi_at_decimal_compute, decimal_compute_halve,
                };
                let pi = pi_at_decimal_compute()?;
                StackValue::DecimalCompute(storage_tier, decimal_compute_halve(pi), CompactShadow::None)
            }
            _ => {
                let pi_half_compute = pi_half_at_compute_tier();
                StackValue::BinaryCompute(storage_tier, pi_half_compute, CompactShadow::None)
            }
        };
        self.subtract_values(pi_half_val, asin_val)
    }

    /// atan(x) — tier N+1 computation returning BinaryCompute or DecimalCompute
    pub(crate) fn evaluate_atan(&mut self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        if let Some(dec_compute) = self.try_decimal_compute(&value) {
            use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_atan;
            let result = decimal_atan(dec_compute)?;
            return Ok(StackValue::DecimalCompute(storage_tier, result, CompactShadow::None));
        }
        let compute_val = self.to_compute_storage(&value)?;
        let result = atan_at_compute_tier(compute_val);
        Ok(StackValue::BinaryCompute(storage_tier, result, CompactShadow::None))
    }

    /// atan2(y, x) — tier N+1 computation returning BinaryCompute
    pub(crate) fn evaluate_atan2(&mut self, y: StackValue, x: StackValue) -> Result<StackValue, OverflowDetected> {
        let storage_tier = self.profile_max_binary_tier();
        // Decimal fast path: both operands decimal → decimal engine
        if let (Some(y_dec), Some(x_dec)) = (self.try_decimal_compute(&y), self.try_decimal_compute(&x)) {
            use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_atan2;
            let result = decimal_atan2(y_dec, x_dec)?;
            return Ok(StackValue::DecimalCompute(storage_tier, result, CompactShadow::None));
        }
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
            StackValue::DecimalCompute(_tier, val, _) => {
                // Direct conversion at tier N+1 (no lossy Decimal intermediate):
                // binary_compute = val × 2^COMPUTE_FRAC_BITS / 10^DECIMAL_COMPUTE_DP
                // Done in the next-wider integer type to preserve full precision.
                use crate::fixed_point::domains::decimal_fixed::transcendental::DECIMAL_COMPUTE_DP;
                #[cfg(table_format = "q64_64")]
                {
                    // val: I256 at dp=38. Target: Q128.128 I256.
                    // result = val × 2^128 / 10^38 via I512 intermediate
                    let num = I512::from_i256(*val) << 128usize;
                    let mut den = I512::from_i128(1);
                    let ten = I512::from_i128(10);
                    for _ in 0..DECIMAL_COMPUTE_DP { den = den * ten; }
                    Ok((num / den).as_i256())
                }
                #[cfg(table_format = "q128_128")]
                {
                    // val: I512 at dp=77. Target: Q256.256 I512.
                    let num = I1024::from_i512(*val) << 256usize;
                    let mut den = I1024::from_i128(1);
                    let ten = I1024::from_i128(10);
                    for _ in 0..DECIMAL_COMPUTE_DP { den = den * ten; }
                    Ok((num / den).as_i512())
                }
                #[cfg(table_format = "q256_256")]
                {
                    // val: I1024 at dp=154. Target: Q512.512 I1024.
                    use crate::fixed_point::I2048;
                    use crate::fixed_point::domains::binary_fixed::i2048::i2048_div;
                    let num = I2048::from_i1024(*val) << 512usize;
                    let mut pow = I1024::from_i128(1);
                    let ten = I1024::from_i128(10);
                    for _ in 0..DECIMAL_COMPUTE_DP { pow = pow * ten; }
                    let den = I2048::from_i1024(pow);
                    let quot = i2048_div(num, den);
                    Ok(I1024::from_words([
                        quot.words[0], quot.words[1], quot.words[2], quot.words[3],
                        quot.words[4], quot.words[5], quot.words[6], quot.words[7],
                        quot.words[8], quot.words[9], quot.words[10], quot.words[11],
                        quot.words[12], quot.words[13], quot.words[14], quot.words[15],
                    ]))
                }
                #[cfg(table_format = "q32_32")]
                {
                    // val: i128 at dp=19. Target: Q64.64 i128.
                    let num = I256::from_i128(*val) << 64usize;
                    let mut den = I256::from_i128(1);
                    let ten = I256::from_i128(10);
                    for _ in 0..DECIMAL_COMPUTE_DP { den = den * ten; }
                    Ok((num / den).as_i128())
                }
                #[cfg(table_format = "q16_16")]
                {
                    use crate::fixed_point::frac_config;
                    let num = (*val as i128) << (frac_config::COMPUTE_FRAC_BITS as usize);
                    let mut den: i128 = 1;
                    for _ in 0..DECIMAL_COMPUTE_DP { den *= 10; }
                    Ok((num / den) as i64)
                }
            }
            StackValue::Symbolic(rational) => {
                // Try i128 extraction first (tiers 1-5)
                if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
                    symbolic_to_compute_storage(num, den)
                } else {
                    // Fall back to wider extraction for Massive/Ultra tier rationals
                    symbolic_wide_to_compute_storage(rational)
                }
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

    /// num/den → Q-format BinaryStorage: nearest, ties toward +∞, CHECKED
    /// (0.5.0 item 1): the old per-arm `(num << F) / den` shifted i128
    /// before any range check — a symbolic 1e20 coerced to binary on
    /// embedded wrapped mod 2^64 and produced a plausible WRONG value.
    fn rational_i128_to_storage_checked(num: i128, den: i128) -> Result<BinaryStorage, OverflowDetected> {
        if den == 0 {
            return Err(OverflowDetected::DivisionByZero);
        }
        #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
        {
            #[cfg(table_format = "q16_16")]
            let fb: usize = crate::fixed_point::frac_config::FRAC_BITS as usize;
            #[cfg(table_format = "q32_32")]
            let fb: usize = 32;
            #[cfg(table_format = "q64_64")]
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
            #[cfg(table_format = "q64_64")]
            { Ok(q128) }
            #[cfg(table_format = "q32_32")]
            {
                i64::try_from(q128).map_err(|_| OverflowDetected::TierOverflow)
            }
            #[cfg(table_format = "q16_16")]
            {
                i32::try_from(q128).map_err(|_| OverflowDetected::TierOverflow)
            }
        }
        #[cfg(table_format = "q128_128")]
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
        #[cfg(table_format = "q256_256")]
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

    /// Convert any StackValue to profile-specific BinaryStorage format
    pub(crate) fn to_binary_storage(&self, value: &StackValue) -> Result<BinaryStorage, OverflowDetected> {
        match value {
            StackValue::Binary(_, val, _) => Ok(*val),
            StackValue::BinaryCompute(_, val, _) => downscale_to_storage(*val),
            StackValue::DecimalCompute(_tier, val, _shadow) => {
                // Direct tier N+1 conversion (no lossy intermediate)
                crate::fixed_point::universal::fasc::stack_evaluator::decimal_compute_to_binary_storage_pub(*val)
            }
            StackValue::Decimal(decimals, scaled, _) => {
                // Delegate to the single checked, ties-+∞ implementation
                // (0.5.0 item 1: this arm previously duplicated the
                // conversion with unchecked truncating variants per profile).
                crate::fixed_point::universal::fasc::stack_evaluator::decimal_to_binary_storage(*decimals, *scaled)
            }
            StackValue::Symbolic(rational) => {
                // Try i128 extraction first (tiers 1-5)
                if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
                    return Self::rational_i128_to_storage_checked(num, den);
                }
                // Fall back to wider extraction for Massive/Ultra tier rationals
                let cs = symbolic_wide_to_compute_storage(rational)?;
                downscale_to_storage(cs)
            }
            StackValue::Ternary(tier, value, _) => {
                // Ternary: convert through rational (value / 3^frac_trits), then to Q-format
                let rational = ternary_to_rational(*tier, value)?;
                if let (Some(num), Some(den)) = (rational.numerator_i128(), rational.denominator_i128()) {
                    Self::rational_i128_to_storage_checked(num, den)
                } else {
                    let cs = symbolic_wide_to_compute_storage(&rational)?;
                    downscale_to_storage(cs)
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
        use crate::fixed_point::universal::fasc::mode::{ComputeMode, get_mode};
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
            StackValue::BinaryCompute(_, _, _) => self.materialize_compute(value),
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
    ///   - Embedded → Tier 3 TQ40.40 (i128 arithmetic, 40 frac trits)
    ///   - Balanced → Tier 4 TQ80.80 (I256 arithmetic, 80 frac trits)
    ///   - Scientific → Tier 5 TQ160.160 (I512 arithmetic, 160 frac trits)
    /// (num * scale) / den rounded to nearest, ties toward +INF — the
    /// documented ternary conversion-boundary rule (0.5.0 unification;
    /// contract §5). All arms run wide (I256/I512) since the 0.5.0 tier
    /// resize: num·3^40 can exceed i128 for binary-raw numerators.
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
                            for _ in 0..160 { s = s * three; }
                            s
                        };
                        // Split to avoid overflow: int_part * scale + (rem * scale) / den
                        let int_part = num512 / den512;
                        let remainder = num512 - int_part * den512;
                        // Nearest, ties toward +INF on the fractional part (0.5.0).
                        let sn = remainder * scale;
                        let (mut frac, rem) =
                            crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(sn, den512);
                        {
                            let rem_abs = if rem.is_negative() { -rem } else { rem };
                            let den_abs = if den512.is_negative() { -den512 } else { den512 };
                            let positive = sn.is_negative() == den512.is_negative();
                            let rem2 = rem_abs + rem_abs;
                            if if positive { rem2 >= den_abs } else { rem2 > den_abs } {
                                frac = if positive { frac + I512::from_i128(1) } else { frac - I512::from_i128(1) };
                            }
                        }
                        let stored = int_part * scale + frac;
                        let (t, bs) = ternary_to_storage(
                            &UniversalTernaryFixed::from_tier_raw(tier, TernaryRaw::Large(stored))?
                        )?;
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
                    // Tiers 1-3 — i128-storable raws at the PROFILE's own
                    // scale (3^10 / 3^20 / 3^40). Pre-resize the narrow
                    // profiles borrowed the tier-3 arm, whose raws no longer
                    // fit their storage (value·3^40 exceeds i64 for any
                    // |value| ≥ 1). The num·scale product can exceed i128
                    // (binary raws reach 2^64), so the multiply-divide runs
                    // at I256 width and narrows back with a fits check.
                    t @ (1 | 2 | 3) => {
                        let scale_i128: i128 = match t {
                            1 => 59_049,                     // 3^10
                            2 => 3_486_784_401,              // 3^20
                            _ => 12_157_665_459_056_928_801, // 3^40
                        };
                        let scale = I256::from_i128(scale_i128);
                        let num256 = I256::from_i128(num);
                        let den256 = I256::from_i128(den);
                        // Nearest, ties toward +INF (0.5.0; contract §5).
                        let sn = num256 * scale;
                        let (mut q, rem) =
                            crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(sn, den256);
                        {
                            let rem_abs = if rem.is_negative() { -rem } else { rem };
                            let den_abs = if den256.is_negative() { -den256 } else { den256 };
                            let positive = sn.is_negative() == den256.is_negative();
                            let rem2 = rem_abs + rem_abs;
                            if if positive { rem2 >= den_abs } else { rem2 > den_abs } {
                                q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
                            }
                        }
                        if !q.fits_in_i128() {
                            return Err(OverflowDetected::TierOverflow);
                        }
                        let stored = q.as_i128();
                        // Checked narrowing — the bare to_binary_storage cast
                        // silently wrapped tier-3 raws on narrow profiles
                        // (same class as the 0.4.33 ternary_to_storage fix).
                        let (t, bs) = ternary_to_storage(
                            &UniversalTernaryFixed::from_tier_raw(tier, TernaryRaw::Small(stored))?
                        )?;
                        Ok(StackValue::Ternary(t, bs, shadow))
                    }
                    // Tier 4: TQ80.80 — 80 frac trits, I256 arithmetic
                    4 => {
                        let scale = {
                            let mut s = I256::from_u8(1);
                            let three = I256::from_u8(3);
                            for _ in 0..80 { s = s * three; }
                            s
                        };
                        let num256 = I256::from_i128(num);
                        let den256 = I256::from_i128(den);
                        // Nearest, ties toward +INF (0.5.0; contract §5).
                        let sn = num256 * scale;
                        let (mut stored, rem) =
                            crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(sn, den256);
                        {
                            let rem_abs = if rem.is_negative() { -rem } else { rem };
                            let den_abs = if den256.is_negative() { -den256 } else { den256 };
                            let positive = sn.is_negative() == den256.is_negative();
                            let rem2 = rem_abs + rem_abs;
                            if if positive { rem2 >= den_abs } else { rem2 > den_abs } {
                                stored = if positive { stored + I256::from_i128(1) } else { stored - I256::from_i128(1) };
                            }
                        }
                        let (t, bs) = ternary_to_storage(
                            &UniversalTernaryFixed::from_tier_raw(tier, TernaryRaw::Medium(stored))?
                        )?;
                        Ok(StackValue::Ternary(t, bs, shadow))
                    }
                    // Tier 5: TQ160.160 — handled by cfg block above on Q256.256
                    5 => {
                        let scale = {
                            let mut s = I512::from_i128(1);
                            let three = I512::from_i128(3);
                            for _ in 0..160 { s = s * three; }
                            s
                        };
                        let num512 = I512::from_i128(num);
                        let den512 = I512::from_i128(den);
                        // Nearest, ties toward +INF (0.5.0; contract §5).
                        let sn = num512 * scale;
                        let (mut stored, rem) =
                            crate::fixed_point::domains::binary_fixed::i512::divmod_i512_by_i512(sn, den512);
                        {
                            let rem_abs = if rem.is_negative() { -rem } else { rem };
                            let den_abs = if den512.is_negative() { -den512 } else { den512 };
                            let positive = sn.is_negative() == den512.is_negative();
                            let rem2 = rem_abs + rem_abs;
                            if if positive { rem2 >= den_abs } else { rem2 > den_abs } {
                                stored = if positive { stored + I512::from_i128(1) } else { stored - I512::from_i128(1) };
                            }
                        }
                        let (t, bs) = ternary_to_storage(
                            &UniversalTernaryFixed::from_tier_raw(tier, TernaryRaw::Large(stored))?
                        )?;
                        Ok(StackValue::Ternary(t, bs, shadow))
                    }
                    // Fallback: use tier 3 for any unexpected tier value
                    _ => {
                        // I256-width multiply-divide (see tier-3 arm).
                        let scale = I256::from_i128(12_157_665_459_056_928_801);
                        let num256 = I256::from_i128(num);
                        let den256 = I256::from_i128(den);
                        let sn = num256 * scale;
                        let (mut q, rem) =
                            crate::fixed_point::domains::binary_fixed::i256::divmod_i256_by_i256(sn, den256);
                        {
                            let rem_abs = if rem.is_negative() { -rem } else { rem };
                            let den_abs = if den256.is_negative() { -den256 } else { den256 };
                            let positive = sn.is_negative() == den256.is_negative();
                            let rem2 = rem_abs + rem_abs;
                            if if positive { rem2 >= den_abs } else { rem2 > den_abs } {
                                q = if positive { q + I256::from_i128(1) } else { q - I256::from_i128(1) };
                            }
                        }
                        if !q.fits_in_i128() {
                            return Err(OverflowDetected::TierOverflow);
                        }
                        let stored = q.as_i128();
                        // Checked narrowing (see tier-3 arm above): on
                        // realtime/compact a tier-3 raw beyond i32/i64 is a
                        // loud TierOverflow, never a wrap.
                        let (t, bs) = ternary_to_storage(
                            &UniversalTernaryFixed::from_tier_raw(3, TernaryRaw::Small(stored))?
                        )?;
                        Ok(StackValue::Ternary(t, bs, shadow))
                    }
                }
            }
        }
    }

    /// Apply output mode conversion after evaluation
    pub(crate) fn apply_output_mode(&self, value: StackValue) -> Result<StackValue, OverflowDetected> {
        use crate::fixed_point::universal::fasc::mode::{OutputMode, get_mode};
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
