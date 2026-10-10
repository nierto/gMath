//! Stack-based evaluator with UGOD integration
//!
//! **MISSION**: Evaluate lazy expressions on thread-local stack without heap allocation
//! **ARCHITECTURE**: Fixed-size workspace with domain routing and overflow delegation
//! **OPTIMIZATION**: Cache-friendly sequential memory access with SIMD potential

pub(crate) mod compute;
pub(crate) mod conversion;
pub(crate) mod domain;
pub(crate) mod formatting;
pub(crate) mod format;
mod parsing;
mod arithmetic;
mod transcendentals;
#[cfg(test)]
mod tests;

#[allow(unused_imports)]
use super::lazy_expr::{LazyExpr, ConstantId, StackRef};
use crate::fixed_point::universal::ugod::DomainType;
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, OverflowDetected};
use crate::fixed_point::universal::tier_types::CompactShadow;
#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;

use core::cell::RefCell;
#[cfg(test)]
use crate::fixed_point::fixed_num::ProfileFormat;
use core::marker::PhantomData;
use std::any::{Any, TypeId};
use std::rc::Rc;
use crate::fixed_point::fixed_num::Compute;
use crate::fixed_point::imperative::FixedPoint;
pub use format::FascFormat;
use core::fmt::{self, Display};

// Re-export sub-module functions used by sibling modules and tests
#[allow(unused_imports)]
pub(crate) use conversion::try_to_binary_storage;
#[allow(unused_imports)]
pub(crate) use compute::{downscale_to_storage};

// ============================================================================
// PROFILE-SPECIFIC STORAGE TYPES (TIER N+1 PRECISION PRESERVATION)
// ============================================================================

/// Profile-specific binary storage type for zero precision loss
///
/// **ARCHITECTURE**: Each profile gets native storage that matches its precision tier
/// **MEMORY COST**:
/// - Embedded: 16 bytes (i128) - Q64.64 storage
/// - Balanced: 32 bytes (I256) - Q128.128 storage
/// - Scientific: 64 bytes (I512) - Q256.256 storage
///
/// **PRECISION GUARANTEE**: Tier N+1 computation results stored without downconversion
#[cfg(table_format = "q256_256")]
pub type BinaryStorage = I512;  // Q256.256: 77 decimals

#[cfg(table_format = "q128_128")]
pub type BinaryStorage = I256;  // Q128.128: 38 decimals

#[cfg(table_format = "q64_64")]
pub type BinaryStorage = i128;  // Q64.64: 19 decimals

#[cfg(table_format = "q32_32")]
pub type BinaryStorage = i64;   // Q32.32: 9 decimals

#[cfg(table_format = "q16_16")]
pub type BinaryStorage = i32;   // Q16.16: 4 decimals

// ============================================================================
// COMPUTE-TIER STORAGE TYPE (TIER N+1 CHAIN PERSISTENCE)
// ============================================================================

/// Compute-tier storage for transcendental chain persistence
///
/// **PURPOSE**: Keep intermediate results at tier N+1 between consecutive
/// transcendentals (e.g., sin(ln(exp(x)))) to avoid precision loss from
/// repeated upscale/downscale cycles.
///
/// **PROFILE MAPPING**:
/// - Embedded: I256 (Q128.128 compute for Q64.64 storage)
/// - Balanced: I512 (Q256.256 compute for Q128.128 storage)
/// - Scientific: I1024 (Q512.512 compute for Q256.256 storage)
#[cfg(table_format = "q256_256")]
pub type ComputeStorage = I1024;  // Q512.512

#[cfg(table_format = "q128_128")]
pub type ComputeStorage = I512;   // Q256.256

#[cfg(table_format = "q64_64")]
pub type ComputeStorage = I256;   // Q128.128

#[cfg(table_format = "q32_32")]
pub type ComputeStorage = i128;   // Q64.64 (tier N+1 for Q32.32)

#[cfg(table_format = "q16_16")]
pub type ComputeStorage = i64;    // Q32.32 (tier N+1 for Q16.16)

/// Maximum decimal places before promoting Decimal to BinaryCompute.
/// Matches each profile's meaningful decimal precision.
/// Beyond this threshold, the scaled integer exceeds BinaryStorage range,
/// so we promote to binary fixed-point (which has FIXED fractional bits, no dp growth).

// ============================================================================
// DECIMAL → BINARY STORAGE HELPER (used by as_binary_storage)
// ============================================================================

#[cfg(test)]
#[cfg(test)]
#[allow(dead_code)]
/// Convert a DecimalCompute value directly to profile BinaryStorage via
/// `val × 2^frac_bits / 10^compute_dp`, no intermediate Decimal materialization.
///
/// This is lossless (within rounding) because we do one big-integer division.
fn decimal_compute_to_binary_storage(val: ComputeStorage) -> Result<BinaryStorage, OverflowDetected> {
    <ProfileFormat as FascFormat>::decimal_compute_to_binary_storage(val)
}

// ============================================================================
// STACK VALUE — UNIFIED DOMAIN REPRESENTATION
// ============================================================================

/// Stack value - unified representation for all domains
///
/// **ARCHITECTURE**: Tagged union for domain-specific values
/// **MEMORY**: Stack-allocated, no heap pointers
/// **CONVERSION**: Lazy conversion between domains as needed
/// **PRECISION**: Profile-specific storage eliminates downconversion loss
#[derive(Debug, Clone)]
pub enum StackValue<T: FascFormat = FixedPoint> {
    /// Binary fixed-point value (tier, mantissa, shadow)
    /// **STORAGE**: Profile-specific (i128 | I256 | I512) - zero precision loss
    /// **SHADOW**: Exact rational representation for precision preservation
    Binary(u8, T::Storage, CompactShadow),

    /// Binary compute-tier value for transcendental chain persistence
    /// **STORAGE**: One tier above storage (I256 | I512 | I1024)
    /// **PURPOSE**: Keep intermediate results "hot" at compute precision between
    /// consecutive transcendentals. Materialized to Binary on non-transcendental boundary.
    BinaryCompute(u8, Compute<T>, CompactShadow),

    /// Decimal fixed-point value (decimals, scaled_value, shadow)
    /// **STORAGE**: Profile-specific (i128 | I256 | I512) - matches binary precision
    /// **SHADOW**: Exact rational representation for precision preservation
    Decimal(u8, T::Storage, CompactShadow),

    /// Decimal compute-tier value for transcendental chain persistence.
    ///
    /// **STORAGE**: Compute<T> at `DECIMAL_COMPUTE_DP` scaling (10^-dp).
    /// **PURPOSE**: Decimal equivalent of BinaryCompute: keeps intermediate
    /// transcendental results at compute-tier precision between chained ops.
    /// Materialized to `Decimal` at non-transcendental boundaries (display, arithmetic
    /// with non-decimal operands, etc.) with dp chosen to guarantee 0 storage ULP.
    ///
    /// **FIELDS**: `(storage_tier, compute_value, shadow)` where `compute_value`
    /// is scaled by `10^DECIMAL_COMPUTE_DP` (per-profile: 9, 19, 38, 77, 154).
    DecimalCompute(u8, Compute<T>, CompactShadow),

    /// Balanced ternary value (precision_tier, trit_value, shadow)
    /// **STORAGE**: Profile-specific (i128 | I256 | I512) - matches binary precision
    /// **SHADOW**: Exact rational representation for precision preservation
    Ternary(u8, T::Storage, CompactShadow),

    /// Symbolic rational value (owned for stack storage)
    /// Symbolic IS exact, no shadow needed
    Symbolic(RationalNumber),

    /// Error state
    Error(OverflowDetected),
}

impl<T: FascFormat> StackValue<T> {
    /// Get domain type for routing decisions
    pub fn domain_type(&self) -> Option<DomainType> {
        match self {
            StackValue::<T>::Binary(..) => Some(DomainType::Binary),
            StackValue::<T>::BinaryCompute(..) => Some(DomainType::Binary),
            StackValue::<T>::Decimal(..) => Some(DomainType::Decimal),
            StackValue::<T>::DecimalCompute(..) => Some(DomainType::Decimal),
            StackValue::<T>::Ternary(..) => Some(DomainType::Ternary),
            StackValue::<T>::Symbolic(_) => Some(DomainType::Symbolic),
            StackValue::<T>::Error(_) => None,
        }
    }

    /// Check if value represents an error state
    pub fn is_error(&self) -> bool {
        matches!(self, StackValue::<T>::Error(_))
    }

    /// Get the compact shadow for precision preservation
    pub fn shadow(&self) -> CompactShadow {
        match self {
            StackValue::<T>::Binary(_, _, s) => s.clone(),
            StackValue::<T>::BinaryCompute(_, _, s) => s.clone(),
            StackValue::<T>::Decimal(_, _, s) => s.clone(),
            StackValue::<T>::DecimalCompute(_, _, s) => s.clone(),
            StackValue::<T>::Ternary(_, _, s) => s.clone(),
            StackValue::<T>::Symbolic(_) => CompactShadow::None, // Symbolic IS exact
            StackValue::<T>::Error(_) => CompactShadow::None,
        }
    }

    /// Convert to rational for cross-domain operations
    ///
    /// **CRITICAL**: Uses full-precision conversion, no i128 truncation.
    /// Binary Q-format values are divided by 2^frac_bits to get the true rational.
    /// Decimal values are divided by 10^decimals.
    /// Ternary values are divided by 3^frac_trits.
    pub fn to_rational(&self) -> Result<RationalNumber, OverflowDetected> {
        match self {
            StackValue::<T>::Symbolic(s) => Ok(s.clone()),
            StackValue::<T>::BinaryCompute(tier, value, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                // Materialize to storage tier first, then convert
                let storage_val = T::downscale_to_storage(*value)?;
                let materialized = StackValue::<T>::Binary(*tier, storage_val, shadow.clone());
                materialized.to_rational()
            }
            StackValue::<T>::DecimalCompute(_tier, value, ref shadow) => {
                // Shadow fast path
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                // DecimalCompute holds raw × 10^-COMPUTE_DP. Materialize at
                // the highest dp that fits (adaptive — 0.5.0 item 0d fix).
                let (target_dp, storage_val) =
                    StackEvaluator::<T>::materialize_decimal_compute_adaptive(*value)?;
                let materialized = StackValue::<T>::Decimal(target_dp, storage_val, shadow.clone());
                materialized.to_rational()
            }
            StackValue::<T>::Binary(tier, value, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                // Binary Q-format: rational = value / 2^frac_bits
                // Tier mapping depends on profile — the storage tier IS the profile's max tier.
                let frac_bits: u32 = match tier {
                    1 => {
                        // Tier 1: the realtime storage tier, raw integer on others.
                        // It follows GMATH_FRAC_BITS; a fixed 16 read every
                        // shadowless realtime value 64x too small at Q22.10
                        // (before 0.6.4: -1000 + 1/1 became -999/64).
                        T::TIER1_RATIONAL_FRAC_BITS
                    }
                    2 => 32,   // Q32.32
                    3 => 64,   // Q64.64
                    4 => 128,  // Q128.128
                    5 => 256,  // Q256.256
                    6 => 512,  // Q512.512
                    _ => 64,   // Default
                };

                if frac_bits == 0 {
                    let value_i128 = T::binary_storage_to_i128(value);
                    return Ok(RationalNumber::new(value_i128, 1));
                }

                T::binary_to_rational(frac_bits, value)

            }
            StackValue::<T>::Decimal(decimals, scaled, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                T::decimal_to_rational(*decimals, scaled)

            }
            StackValue::<T>::Ternary(tier, value, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                T::ternary_to_rational(*tier, value)
            }
            StackValue::<T>::Error(e) => Err(e.clone()),
        }
    }

    /// Extract the raw binary storage value (profile-specific type).
    ///
    /// Returns `None` for non-materializable values and error states.
    /// - `Binary` → returns as-is
    /// - `BinaryCompute` → downscales to storage tier
    /// - `Decimal` → converts to binary Q-format via `(scaled << frac_bits) / 10^dp`
    /// - `DecimalCompute` → downscales to Decimal first, then converts
    pub fn as_binary_storage(&self) -> Option<T::Storage> {
        match self {
            StackValue::<T>::Binary(_, val, _) => Some(*val),
            StackValue::<T>::BinaryCompute(_, val, _) => T::downscale_to_storage(*val).ok(),
            StackValue::<T>::Decimal(dp, scaled, _) => {
                T::decimal_to_binary_storage(*dp, *scaled).ok()
            }
            StackValue::<T>::DecimalCompute(_, val, _) => {
                // Direct conversion: DecimalCompute value × 2^frac_bits / 10^compute_dp
                // (no intermediate Decimal materialization — preserves full precision)
                T::decimal_compute_to_binary_storage(*val).ok()
            }
            _ => None,
        }
    }

    /// Extract the tier number.
    ///
    /// Returns the precision tier (1-6) for domain values, 0 for errors.
    pub fn tier(&self) -> u8 {
        match self {
            StackValue::<T>::Binary(t, _, _) => *t,
            StackValue::<T>::BinaryCompute(t, _, _) => *t,
            StackValue::<T>::Decimal(t, _, _) => *t,
            StackValue::<T>::DecimalCompute(t, _, _) => *t,
            StackValue::<T>::Ternary(t, _, _) => *t,
            StackValue::<T>::Symbolic(_) => 8, // Symbolic = rational tier
            StackValue::<T>::Error(_) => 0,
        }
    }

    /// Convert to decimal string with specified precision.
    ///
    /// Uses integer-only multiply-by-10 extraction (zero floats).
    /// Profile-dispatched: Q64.64 → max 19 digits, Q128.128 → max 38, Q256.256 → max 76.
    ///
    /// For non-binary values: converts to rational first, then extracts digits.
    pub fn to_decimal_string(&self, max_digits: usize) -> String {
        match self {
            StackValue::<T>::BinaryCompute(tier, val, shadow) => {
                match T::downscale_to_storage(*val) {
                    Ok(storage_val) => {
                        let materialized = StackValue::<T>::Binary(*tier, storage_val, shadow.clone());
                        materialized.to_decimal_string(max_digits)
                    }
                    Err(_) => "Overflow".to_string(),
                }
            }
            StackValue::<T>::DecimalCompute(_tier, val, shadow) => {
                // Materialize at the highest dp that fits (adaptive — 0d fix).
                match StackEvaluator::<T>::materialize_decimal_compute_adaptive(*val) {
                    Ok((target_dp, storage_val)) => {
                        let materialized = StackValue::<T>::Decimal(target_dp, storage_val, shadow.clone());
                        materialized.to_decimal_string(max_digits)
                    }
                    Err(_) => "Overflow".to_string(),
                }
            }
            StackValue::<T>::Binary(_tier, val, _) => {
                T::binary_storage_to_decimal_string(*val, max_digits)
            }
            StackValue::<T>::Decimal(decimals, val, _) => {
                let full = T::decimal_storage_to_string(*decimals, val);
                if max_digits < *decimals as usize {
                    if let Some(dot_pos) = full.find('.') {
                        let end = (dot_pos + 1 + max_digits).min(full.len());
                        full[..end].to_string()
                    } else {
                        full
                    }
                } else {
                    full
                }
            }
            StackValue::<T>::Symbolic(r) => {
                formatting::rational_to_decimal_string(r, max_digits)
            }
            StackValue::<T>::Ternary(_, _, _) => {
                if let Ok(rational) = self.to_rational() {
                    formatting::rational_to_decimal_string(&rational, max_digits)
                } else {
                    "NaN".to_string()
                }
            }
            StackValue::<T>::Error(_) => {
                "NaN".to_string()
            }
        }
    }
}

impl<T: FascFormat> Display for StackValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        let precision = f.precision().unwrap_or(T::DEFAULT_DIGITS);

        match self {
            StackValue::<T>::BinaryCompute(tier, val, shadow) => {
                match T::downscale_to_storage(*val) {
                    Ok(storage_val) => {
                        let materialized = StackValue::<T>::Binary(*tier, storage_val, shadow.clone());
                        write!(f, "{}", materialized.to_decimal_string(precision))
                    }
                    Err(_) => write!(f, "Overflow"),
                }
            }
            StackValue::<T>::DecimalCompute(_tier, val, shadow) => {
                match StackEvaluator::<T>::materialize_decimal_compute_adaptive(*val) {
                    Ok((target_dp, storage_val)) => {
                        let materialized = StackValue::<T>::Decimal(target_dp, storage_val, shadow.clone());
                        write!(f, "{}", materialized.to_decimal_string(precision))
                    }
                    Err(_) => write!(f, "Overflow"),
                }
            }
            StackValue::<T>::Binary(_, val, _) => {
                write!(f, "{}", T::binary_storage_to_decimal_string(*val, precision))
            }
            StackValue::<T>::Decimal(dec, val, _) => {
                write!(f, "{}", T::decimal_storage_to_string(*dec, val))
            }
            StackValue::<T>::Ternary(_, _, _) => {
                write!(f, "{}", self.to_decimal_string(precision))
            }
            StackValue::<T>::Symbolic(s) => write!(f, "{}", formatting::rational_to_decimal_string(s, precision)),
            StackValue::<T>::Error(e) => write!(f, "Error: {:?}", e),
        }
    }
}

// ============================================================================
// STACK EVALUATOR — THREAD-LOCAL COMPUTATION ENGINE
// ============================================================================

/// Universal stack evaluator - thread-local computation engine
///
/// **ARCHITECTURE**: Fixed-size stack with domain-aware evaluation
/// **THREAD-SAFETY**: Each thread gets its own evaluator instance
/// **MEMORY**: Zero heap allocation during computation
pub struct StackEvaluator<T: FascFormat = FixedPoint> {
    /// Current workspace offset
    offset: usize,

    /// Stack of computed values (fixed size)
    value_stack: [Option<StackValue<T>>; 256],

    /// Current stack pointer
    stack_ptr: usize,

    /// The format evaluated: its tier limits, digit budgets and words.
    _format: PhantomData<T>,
}

impl<T: FascFormat> Default for StackEvaluator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FascFormat> StackEvaluator<T> {
    /// A new evaluator for the format `T`.
    pub fn new() -> Self {
        Self {
            offset: 0,
            value_stack: [const { None }; 256],
            stack_ptr: 0,
            _format: PhantomData,
        }
    }

    /// Reset evaluator state for new computation
    pub fn reset(&mut self) {
        self.offset = 0;
        self.stack_ptr = 0;
    }

    /// Load value from stack reference
    fn load_ref(&self, stack_ref: StackRef) -> Result<StackValue<T>, OverflowDetected> {
        let idx = stack_ref.0 as usize;
        if idx >= self.stack_ptr {
            return Err(OverflowDetected::InvalidStackReference);
        }

        self.value_stack[idx]
            .clone()
            .ok_or(OverflowDetected::StackCorruption)
    }

    /// Evaluate expression on stack
    pub fn evaluate(&mut self, expr: &LazyExpr<T>) -> Result<StackValue<T>, OverflowDetected> {
        match expr {
            LazyExpr::Literal(s) => self.parse_literal_with_mode(s),
            LazyExpr::Value(v) => Ok((**v).clone()),
            LazyExpr::Constant(c) => self.load_constant(*c),
            LazyExpr::Variable(r) => self.load_ref(*r),
            LazyExpr::Negate(inner) => {
                let val = self.evaluate(inner)?;
                self.negate_value(val)
            }
            LazyExpr::Add(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                self.add_values(l, r)
            }
            LazyExpr::Sub(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                self.subtract_values(l, r)
            }
            LazyExpr::Mul(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                self.multiply_values(l, r)
            }
            LazyExpr::Div(left, right) => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                self.divide_values(l, r)
            }
            LazyExpr::Exp(inner) => {
                // Identity short-circuit: exp(ln(x)) = x
                if let LazyExpr::Ln(inner_inner) = inner.as_ref() {
                    return self.evaluate(inner_inner);
                }
                let val = self.evaluate(inner)?;
                self.evaluate_exp(val)
            }
            LazyExpr::Ln(inner) => {
                // Identity short-circuit: ln(exp(x)) = x
                if let LazyExpr::Exp(inner_inner) = inner.as_ref() {
                    return self.evaluate(inner_inner);
                }
                let val = self.evaluate(inner)?;
                self.evaluate_ln(val)
            }
            LazyExpr::Sqrt(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_sqrt(val)
            }
            LazyExpr::Pow(base, exponent) => {
                let base_val = self.evaluate(base)?;
                let exp_val = self.evaluate(exponent)?;
                self.evaluate_pow(base_val, exp_val)
            }
            LazyExpr::Sinh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_sinh(val)
            }
            LazyExpr::Cosh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_cosh(val)
            }
            LazyExpr::Tanh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_tanh(val)
            }
            LazyExpr::Asinh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_asinh(val)
            }
            LazyExpr::Acosh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_acosh(val)
            }
            LazyExpr::Atanh(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_atanh(val)
            }
            LazyExpr::Sin(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_sin(val)
            }
            LazyExpr::Cos(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_cos(val)
            }
            LazyExpr::Tan(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_tan(val)
            }
            LazyExpr::Asin(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_asin(val)
            }
            LazyExpr::Acos(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_acos(val)
            }
            LazyExpr::Atan(inner) => {
                let val = self.evaluate(inner)?;
                self.evaluate_atan(val)
            }
            LazyExpr::Atan2(y, x) => {
                let y_val = self.evaluate(y)?;
                let x_val = self.evaluate(x)?;
                self.evaluate_atan2(y_val, x_val)
            }
        }
    }

    /// Materialize BinaryCompute back to Binary (storage tier)
    ///
    /// Called at the top-level evaluate boundary to ensure callers always
    /// receive values at the storage tier, not the internal compute tier.
    ///
    /// Returns `Err(TierOverflow)` if the compute-tier value exceeds the
    /// storage tier's range (UGOD overflow detection).
    /// Materialize a DecimalCompute raw at the highest storage dp that
    /// fits (0.5.0 item 0d fix): try DECIMAL_STORAGE_MAX_DP first and step
    /// down only when the magnitude genuinely needs fewer decimals. The
    /// old fixed `MAX_DP - 2` slack silently cost realtime HALF its digits
    /// (dp 4 → 2: cos(0.1) = 0.9952 displayed as "1.00"), which masqueraded
    /// as a sin/cos kernel plateau for months. Deterministic; the
    /// downscale is checked, so a too-large dp errors and we retry, never
    /// a wrap.
    fn materialize_decimal_compute_adaptive(
        val: Compute<T>,
    ) -> Result<(u8, T::Storage), OverflowDetected> {
        use crate::fixed_point::domains::decimal_fixed::transcendental::decimal_downscale_to_storage;
        let mut dp = T::DECIMAL_STORAGE_MAX_DP;
        loop {
            match decimal_downscale_to_storage(val, dp) {
                Ok(v) => return Ok((dp, v)),
                Err(e) => {
                    if dp == 0 {
                        return Err(e);
                    }
                    dp -= 1;
                }
            }
        }
    }

    pub(crate) fn materialize_compute(&self, value: StackValue<T>) -> Result<StackValue<T>, OverflowDetected> {
        match value {
            StackValue::BinaryCompute(tier, val, shadow) => {
                let storage = T::downscale_to_storage(val)?;
                Ok(StackValue::Binary(tier, storage, shadow))
            }
            // DecimalCompute passes through — it's a valid output format.
            // Chain persistence ensures DecimalCompute stays hot between
            // transcendental operations. Display handles formatting.
            other => Ok(other),
        }
    }
}

// ============================================================================
// THREAD-LOCAL EVALUATOR INSTANCE
// ============================================================================

thread_local! {
    /// One evaluator per format used on this thread, keyed by the format's type.
    static EVALUATORS: RefCell<Vec<(TypeId, Rc<dyn Any>)>> = const { RefCell::new(Vec::new()) };
    /// The entry used last: one type comparison on the common path.
    static LAST: RefCell<Option<(TypeId, Rc<dyn Any>)>> = const { RefCell::new(None) };
}

/// Run `f` on this thread's evaluator for the format `T` (created on first
/// use). Nested calls for the same format are a borrow error, as before.
pub fn with_evaluator<T: FascFormat, R>(f: impl FnOnce(&mut StackEvaluator<T>) -> R) -> R {
    let id = TypeId::of::<T>();
    let hit = LAST.with(|last| match &*last.borrow() {
        Some((t, rc)) if *t == id => Some(rc.clone()),
        _ => None,
    });
    let cell: Rc<RefCell<StackEvaluator<T>>> = match hit {
        Some(rc) => rc.downcast::<RefCell<StackEvaluator<T>>>().expect("evaluator registry: the key is the type"),
        None => {
            let rc = EVALUATORS.with(|regs| {
                let mut regs = regs.borrow_mut();
                if let Some((_, rc)) = regs.iter().find(|(t, _)| *t == id) {
                    return rc.clone();
                }
                let rc: Rc<dyn Any> = Rc::new(RefCell::new(StackEvaluator::<T>::new()));
                regs.push((id, rc.clone()));
                rc
            });
            LAST.with(|last| *last.borrow_mut() = Some((id, rc.clone())));
            rc.downcast::<RefCell<StackEvaluator<T>>>().expect("evaluator registry: the key is the type")
        }
    };
    let mut evaluator = cell.borrow_mut();
    f(&mut evaluator)
}

/// Evaluate `expr` in the format `T`: the tree is parsed, routed and
/// computed by this thread's evaluator for `T`, the result materialized
/// to its storage tier, then converted per the output mode.
pub fn evaluate_in<T: FascFormat>(expr: &LazyExpr<T>) -> Result<StackValue<T>, OverflowDetected> {
    with_evaluator(|evaluator| {
        evaluator.reset();
        let result = evaluator.evaluate(expr)?;
        let materialized = evaluator.materialize_compute(result)?;
        evaluator.apply_output_mode(materialized)
    })
}

/// Evaluate expression using thread-local evaluator
///
/// **ARCHITECTURE**: Thread-local StackEvaluator instance, one per thread
/// **BENEFIT**: Zero mutex contention, no heap allocation per call
/// **COMPUTE**: BinaryCompute results are materialized to storage tier at the top level
pub fn evaluate(expr: &LazyExpr) -> Result<StackValue, OverflowDetected> {
    evaluate_in(expr)
}

/// `sin` and `cos` of `expr` from one range reduction, in the format `T`.
pub fn evaluate_sincos_in<T: FascFormat>(expr: &LazyExpr<T>) -> Result<(StackValue<T>, StackValue<T>), OverflowDetected> {
    with_evaluator(|evaluator| {
        evaluator.reset();
        let inner_val = evaluator.evaluate(expr)?;
        let (sin_compute, cos_compute) = evaluator.evaluate_sincos(inner_val)?;
        let sin_mat = evaluator.materialize_compute(sin_compute)?;
        let cos_mat = evaluator.materialize_compute(cos_compute)?;
        let sin_out = evaluator.apply_output_mode(sin_mat)?;
        let cos_out = evaluator.apply_output_mode(cos_mat)?;
        Ok((sin_out, cos_out))
    })
}

/// Evaluate sin and cos of an expression simultaneously (shared range reduction).
///
/// Returns `(sin, cos)` as materialized StackValues. The inner expression is
/// evaluated once; both results come from the same compute-tier computation.
pub fn evaluate_sincos(expr: &LazyExpr) -> Result<(StackValue, StackValue), OverflowDetected> {
    evaluate_sincos_in(expr)
}

/// `sinh` and `cosh` of `expr` from one exponential, in the format `T`.
pub fn evaluate_sinhcosh_in<T: FascFormat>(expr: &LazyExpr<T>) -> Result<(StackValue<T>, StackValue<T>), OverflowDetected> {
    with_evaluator(|evaluator| {
        evaluator.reset();
        let inner_val = evaluator.evaluate(expr)?;
        let (sinh_compute, cosh_compute) = evaluator.evaluate_sinhcosh(inner_val)?;
        let sinh_mat = evaluator.materialize_compute(sinh_compute)?;
        let cosh_mat = evaluator.materialize_compute(cosh_compute)?;
        let sinh_out = evaluator.apply_output_mode(sinh_mat)?;
        let cosh_out = evaluator.apply_output_mode(cosh_mat)?;
        Ok((sinh_out, cosh_out))
    })
}

/// Evaluate sinh and cosh of an expression simultaneously (shared exp computation).
///
/// Returns `(sinh, cosh)` as materialized StackValues. The inner expression is
/// evaluated once; both results come from the same compute-tier exp evaluation.
pub fn evaluate_sinhcosh(expr: &LazyExpr) -> Result<(StackValue, StackValue), OverflowDetected> {
    evaluate_sinhcosh_in(expr)
}

/// Parse a runtime string into a pre-evaluated expression in the format `T`.
pub fn gmath_parse_in<T: FascFormat>(s: &str) -> Result<LazyExpr<T>, OverflowDetected> {
    let value = with_evaluator(|evaluator: &mut StackEvaluator<T>| evaluator.parse_literal_with_mode(s))?;
    Ok(LazyExpr::Value(Box::new(value)))
}

/// Parse a runtime string into a LazyExpr (for non-'static strings).
///
/// Unlike `gmath()` which requires `&'static str`, this parses immediately
/// and returns a `LazyExpr::Value` holding the parsed StackValue.
pub fn gmath_parse(s: &str) -> Result<LazyExpr, OverflowDetected> {
    gmath_parse_in(s)
}
