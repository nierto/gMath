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
mod exact;

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
use core::marker::PhantomData;
use std::any::{Any, TypeId};
use std::rc::Rc;
use crate::fixed_point::fixed_num::Compute;
pub use format::FascFormat;
use core::fmt::{self, Display};

// Re-export sub-module functions used by sibling modules and tests
#[allow(unused_imports)]
#[allow(unused_imports)]

/// Maximum decimal places before promoting Decimal to BinaryCompute.
/// Matches each profile's meaningful decimal precision.
/// Beyond this threshold, the scaled integer exceeds BinaryStorage range,
/// so we promote to binary fixed-point (which has FIXED fractional bits, no dp growth).

// ============================================================================
// DECIMAL → BINARY STORAGE HELPER (used by as_binary_storage)
// ============================================================================

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
pub enum StackValue<T: FascFormat> {
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
pub struct StackEvaluator<T: FascFormat> {
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
pub fn evaluate<T: FascFormat>(expr: &LazyExpr<T>) -> Result<StackValue<T>, OverflowDetected> {
    with_evaluator(|evaluator| {
        evaluator.reset();
        let materialized = evaluator.evaluate(expr).and_then(|r| evaluator.materialize_compute(r));
        let materialized = settle_tree(evaluator, expr, materialized)?;
        evaluator.apply_output_mode(materialized)
    })
}

/// A tree with at least one operation, whose result is a binary storage
/// value (or is wanted as one, or failed on the compute tier), is rounded
/// once from its exact value: one function of stored values by the
/// imperative ladder, anything deeper by the exact evaluation on `Big`
/// (`exact.rs`, on `Mpf` from just above the format's width). Where the exact evaluation cannot decide (or does not
/// cover the tree), the compute-tier result stands.
fn settle_tree<T: FascFormat>(
    evaluator: &mut StackEvaluator<T>,
    expr: &LazyExpr<T>,
    materialized: Result<StackValue<T>, OverflowDetected>,
) -> Result<StackValue<T>, OverflowDetected> {
    use crate::fixed_point::domains::binary_fixed::transcendental::mpf::MpfErr;
    use crate::fixed_point::universal::fasc::mode::{get_mode, OutputMode};
    if matches!(expr, LazyExpr::Literal(_) | LazyExpr::Value(_) | LazyExpr::Constant(_) | LazyExpr::Variable(_)) {
        return materialized;
    }
    let want_binary = matches!(get_mode().output, OutputMode::Binary);
    // a decimal result: its reading at the storage places must be the exact
    // value rounded once; the compute value stays when it already reads so
    if let Ok(StackValue::<T>::DecimalCompute(tier, val, _)) = &materialized {
        if !want_binary {
            let (tier, val) = (*tier, *val);
            return match exact::rounded_decimal(evaluator, expr) {
                Ok((dp, scaled)) => {
                    if StackEvaluator::<T>::materialize_decimal_compute_adaptive(val).ok() == Some((dp, scaled)) {
                        materialized
                    } else {
                        match crate::fixed_point::domains::decimal_fixed::transcendental::decimal_upscale_to_compute::<Compute<T>>(scaled, dp) {
                            Ok(v) => Ok(StackValue::<T>::DecimalCompute(tier, v, CompactShadow::None)),
                            Err(_) => materialized,
                        }
                    }
                }
                Err(MpfErr::Domain) => Err(OverflowDetected::DomainError),
                Err(_) => materialized,
            };
        }
    }
    let (tier, old_raw, shadow) = match &materialized {
        Ok(StackValue::<T>::Binary(tier, raw, shadow)) if *tier == T::STORAGE_TIER => (*tier, Some(*raw), shadow.clone()),
        Ok(StackValue::<T>::Binary(..)) => return materialized,
        // an evaluator error can be a compute-tier artifact (tanh rounded to 1
        // before atanh): a value the exact evaluation decides replaces it;
        // otherwise the error stands with its own kind (below)
        Err(_) => (T::STORAGE_TIER, None, CompactShadow::None),
        Ok(_) if want_binary => (T::STORAGE_TIER, None, CompactShadow::None),
        Ok(_) => return materialized,
    };
    if let Some(settled) = settle_single_function(evaluator, expr, &materialized) {
        return settled;
    }
    match exact::rounded(evaluator, expr) {
        Ok(raw) => {
            // a shadow describes the old raw's exact value; keep it only if the raw stands
            let shadow = if old_raw == Some(raw) { shadow } else { CompactShadow::None };
            Ok(StackValue::<T>::Binary(tier, raw, shadow))
        }
        // a value the evaluator returned is replaced by the exact verdict;
        // an error it returned keeps its own kind
        Err(MpfErr::Domain) if materialized.is_ok() => Err(OverflowDetected::DomainError),
        Err(MpfErr::Overflow) if materialized.is_ok() => Err(OverflowDetected::TierOverflow),
        Err(_) => materialized,
    }
}

/// A tree that is one transcendental of stored binary values is rounded
/// like the imperative function: the compute-tier result, decided near a
/// rounding boundary by the sure path. Deeper trees keep the compute-tier
/// chain and its single rounding.
fn settle_single_function<T: FascFormat>(
    evaluator: &mut StackEvaluator<T>,
    expr: &LazyExpr<T>,
    materialized: &Result<StackValue<T>, OverflowDetected>,
) -> Option<Result<StackValue<T>, OverflowDetected>> {
    use format::SettledOp as Op;
    let (op, a, b): (Op, &LazyExpr<T>, Option<&LazyExpr<T>>) = match expr {
        LazyExpr::Exp(a) => (Op::Exp, a, None), LazyExpr::Ln(a) => (Op::Ln, a, None), LazyExpr::Sqrt(a) => (Op::Sqrt, a, None),
        LazyExpr::Sin(a) => (Op::Sin, a, None), LazyExpr::Cos(a) => (Op::Cos, a, None), LazyExpr::Tan(a) => (Op::Tan, a, None),
        LazyExpr::Atan(a) => (Op::Atan, a, None), LazyExpr::Asin(a) => (Op::Asin, a, None), LazyExpr::Acos(a) => (Op::Acos, a, None),
        LazyExpr::Sinh(a) => (Op::Sinh, a, None), LazyExpr::Cosh(a) => (Op::Cosh, a, None), LazyExpr::Tanh(a) => (Op::Tanh, a, None),
        LazyExpr::Asinh(a) => (Op::Asinh, a, None), LazyExpr::Acosh(a) => (Op::Acosh, a, None), LazyExpr::Atanh(a) => (Op::Atanh, a, None),
        LazyExpr::Pow(a, b) => (Op::Pow, a, Some(b)), LazyExpr::Atan2(a, b) => (Op::Atan2, a, Some(b)),
        _ => return None,
    };
    // a binary result is replaced by the settled one; an error of the
    // compute-tier composition (an intermediate leaving the compute word
    // while the result fits) is retried on the imperative path too
    let (tier, shadow) = match &materialized {
        Ok(StackValue::<T>::Binary(tier, _, shadow)) => (*tier, shadow.clone()),
        Err(_) => (T::STORAGE_TIER, CompactShadow::None),
        _ => (T::STORAGE_TIER, CompactShadow::None),
    };
    let stored = |evaluator: &mut StackEvaluator<T>, e: &LazyExpr<T>| -> Option<T::Storage> {
        if !matches!(e, LazyExpr::Value(_) | LazyExpr::Literal(_)) { return None; }
        match evaluator.evaluate(e).ok()? {
            StackValue::<T>::Binary(_, raw, _) => Some(raw),
            _ => None,
        }
    };
    let x = match stored(evaluator, a) { Some(v) => v, None => return None };
    let y = match b {
        Some(b) => match stored(evaluator, b) { Some(v) => v, None => return None },
        None => <T::Storage as crate::fixed_point::width::WideInt>::ZERO,
    };
    match T::settled_function(op, x, y) {
        Some(Ok(raw)) => Some(Ok(StackValue::<T>::Binary(tier, raw, shadow))),
        _ => None,
    }
}


/// `sin` and `cos` of `expr` from one range reduction, in the format `T`.
pub fn evaluate_sincos<T: FascFormat>(expr: &LazyExpr<T>) -> Result<(StackValue<T>, StackValue<T>), OverflowDetected> {
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


/// `sinh` and `cosh` of `expr` from one exponential, in the format `T`.
pub fn evaluate_sinhcosh<T: FascFormat>(expr: &LazyExpr<T>) -> Result<(StackValue<T>, StackValue<T>), OverflowDetected> {
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


/// Parse a runtime string into a pre-evaluated expression in the format `T`.
pub fn gmath_parse<T: FascFormat>(s: &str) -> Result<LazyExpr<T>, OverflowDetected> {
    let value = with_evaluator(|evaluator: &mut StackEvaluator<T>| evaluator.parse_literal_with_mode(s))?;
    Ok(LazyExpr::Value(Box::new(value)))
}

