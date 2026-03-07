//! Stack-based evaluator with UGOD integration
//!
//! **MISSION**: Evaluate lazy expressions on thread-local stack without heap allocation
//! **ARCHITECTURE**: Fixed-size workspace with domain routing and overflow delegation
//! **OPTIMIZATION**: Cache-friendly sequential memory access with SIMD potential

pub(crate) mod compute;
pub(crate) mod conversion;
pub(crate) mod domain;
pub(crate) mod formatting;
mod parsing;
mod arithmetic;
mod transcendentals;
#[cfg(test)]
mod tests;

#[allow(unused_imports)]
use super::lazy_expr::{LazyExpr, ConstantId, StackRef};
use crate::fixed_point::universal::ugod::DomainType;
use crate::fixed_point::domains::symbolic::rational::rational_number::{RationalNumber, OverflowDetected};
use crate::deployment_profiles::DeploymentProfile;
use crate::fixed_point::universal::tier_types::CompactShadow;
#[allow(unused_imports)]
use crate::fixed_point::i256::I256;
#[allow(unused_imports)]
use crate::fixed_point::i512::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;

use core::cell::RefCell;
use core::fmt::{self, Display};

// Re-export sub-module functions used by sibling modules and tests
#[allow(unused_imports)]
pub(crate) use conversion::to_binary_storage;
#[allow(unused_imports)]
pub(crate) use compute::{downscale_to_storage, upscale_to_compute};

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

/// Maximum decimal places before promoting Decimal to BinaryCompute.
/// Matches each profile's meaningful decimal precision.
/// Beyond this threshold, the scaled integer exceeds BinaryStorage range,
/// so we promote to binary fixed-point (which has FIXED fractional bits, no dp growth).
#[cfg(table_format = "q64_64")]
const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 18;
#[cfg(table_format = "q128_128")]
const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 38;
#[cfg(table_format = "q256_256")]
const DECIMAL_DP_PROMOTION_THRESHOLD: u16 = 76;

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
pub enum StackValue {
    /// Binary fixed-point value (tier, mantissa, shadow)
    /// **STORAGE**: Profile-specific (i128 | I256 | I512) - zero precision loss
    /// **SHADOW**: Exact rational representation for precision preservation
    Binary(u8, BinaryStorage, CompactShadow),

    /// Binary compute-tier value for transcendental chain persistence
    /// **STORAGE**: One tier above storage (I256 | I512 | I1024)
    /// **PURPOSE**: Keep intermediate results "hot" at compute precision between
    /// consecutive transcendentals. Materialized to Binary on non-transcendental boundary.
    BinaryCompute(u8, ComputeStorage, CompactShadow),

    /// Decimal fixed-point value (decimals, scaled_value, shadow)
    /// **STORAGE**: Profile-specific (i128 | I256 | I512) - matches binary precision
    /// **SHADOW**: Exact rational representation for precision preservation
    Decimal(u8, BinaryStorage, CompactShadow),

    /// Balanced ternary value (precision_tier, trit_value, shadow)
    /// **STORAGE**: Profile-specific (i128 | I256 | I512) - matches binary precision
    /// **SHADOW**: Exact rational representation for precision preservation
    Ternary(u8, BinaryStorage, CompactShadow),

    /// Symbolic rational value (owned for stack storage)
    /// Symbolic IS exact — no shadow needed
    Symbolic(RationalNumber),

    /// Error state
    Error(OverflowDetected),
}

impl StackValue {
    /// Get domain type for routing decisions
    pub fn domain_type(&self) -> Option<DomainType> {
        match self {
            StackValue::Binary(..) => Some(DomainType::Binary),
            StackValue::BinaryCompute(..) => Some(DomainType::Binary),
            StackValue::Decimal(..) => Some(DomainType::Decimal),
            StackValue::Ternary(..) => Some(DomainType::Ternary),
            StackValue::Symbolic(_) => Some(DomainType::Symbolic),
            StackValue::Error(_) => None,
        }
    }

    /// Check if value represents an error state
    pub fn is_error(&self) -> bool {
        matches!(self, StackValue::Error(_))
    }

    /// Get the compact shadow for precision preservation
    pub fn shadow(&self) -> CompactShadow {
        match self {
            StackValue::Binary(_, _, s) => s.clone(),
            StackValue::BinaryCompute(_, _, s) => s.clone(),
            StackValue::Decimal(_, _, s) => s.clone(),
            StackValue::Ternary(_, _, s) => s.clone(),
            StackValue::Symbolic(_) => CompactShadow::None, // Symbolic IS exact
            StackValue::Error(_) => CompactShadow::None,
        }
    }

    /// Convert to rational for cross-domain operations
    ///
    /// **CRITICAL**: Uses full-precision conversion — no i128 truncation.
    /// Binary Q-format values are divided by 2^frac_bits to get the true rational.
    /// Decimal values are divided by 10^decimals.
    /// Ternary values are divided by 3^frac_trits.
    pub fn to_rational(&self) -> Result<RationalNumber, OverflowDetected> {
        match self {
            StackValue::Symbolic(s) => Ok(s.clone()),
            StackValue::BinaryCompute(tier, value, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                // Materialize to storage tier first, then convert
                let storage_val = downscale_to_storage(*value);
                let materialized = StackValue::Binary(*tier, storage_val, shadow.clone());
                materialized.to_rational()
            }
            StackValue::Binary(tier, value, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                // Binary Q-format: rational = value / 2^frac_bits
                let frac_bits: u32 = match tier {
                    1 => 0,    // Raw integer (hex/bin parse)
                    2 => 32,   // Q32.32
                    3 => 64,   // Q64.64
                    4 => 128,  // Q128.128
                    5 => 256,  // Q256.256
                    6 => 512,  // Q512.512
                    _ => 64,   // Default
                };

                if frac_bits == 0 {
                    let value_i128 = conversion::binary_storage_to_i128(value);
                    return Ok(RationalNumber::new(value_i128, 1));
                }

                #[cfg(table_format = "q256_256")]
                {
                    if value.is_zero() {
                        return Ok(RationalNumber::new(0, 1));
                    }
                    let tz = conversion::trailing_zeros_i512(value);
                    let tz = if tz > frac_bits { frac_bits } else { tz };
                    let reduced_num = *value >> (tz as usize);
                    let remaining_frac = frac_bits - tz;
                    if remaining_frac == 0 && reduced_num.fits_in_i128() {
                        return Ok(RationalNumber::new(reduced_num.as_i128(), 1));
                    } else if remaining_frac <= 127 && reduced_num.fits_in_i128() {
                        return Ok(RationalNumber::new(reduced_num.as_i128(), 1u128 << remaining_frac));
                    } else {
                        let denom = I512::from_i128(1) << (remaining_frac as usize);
                        return Ok(RationalNumber::from_i512_pair(reduced_num, denom));
                    }
                }

                #[cfg(table_format = "q128_128")]
                {
                    if value.is_zero() {
                        return Ok(RationalNumber::new(0, 1));
                    }
                    let tz = conversion::trailing_zeros_i256(value);
                    let tz = if tz > frac_bits { frac_bits } else { tz };
                    let reduced_num = *value >> tz;
                    let remaining_frac = frac_bits - tz;
                    if remaining_frac == 0 && reduced_num.fits_in_i128() {
                        return Ok(RationalNumber::new(reduced_num.as_i128(), 1));
                    } else if remaining_frac <= 127 && reduced_num.fits_in_i128() {
                        return Ok(RationalNumber::new(reduced_num.as_i128(), 1u128 << remaining_frac));
                    } else {
                        let denom = I256::from_i128(1) << (remaining_frac as usize);
                        return Ok(RationalNumber::from_i256_pair(reduced_num, denom));
                    }
                }

                #[cfg(table_format = "q64_64")]
                {
                    if *value == 0 {
                        return Ok(RationalNumber::new(0, 1));
                    }
                    let trailing = (*value as u128).trailing_zeros().min(frac_bits);
                    let reduced_num = *value >> trailing;
                    let reduced_den = 1u128 << (frac_bits - trailing);
                    return Ok(RationalNumber::new(reduced_num, reduced_den));
                }

            }
            StackValue::Decimal(decimals, scaled, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                #[cfg(table_format = "q256_256")]
                {
                    let fits = if scaled.fits_in_i256() { let v = scaled.as_i256(); if v.fits_in_i128() { Some(v.as_i128()) } else { None } } else { None };
                    if let Some(raw) = fits {
                        return conversion::reduce_decimal_to_rational(raw, *decimals);
                    }
                    let mut denom = I512::from_i128(1);
                    for _ in 0..*decimals { denom = denom * I512::from_i128(10); }
                    return Ok(RationalNumber::from_i512_pair(*scaled, denom));
                }
                #[cfg(table_format = "q128_128")]
                {
                    if scaled.fits_in_i128() {
                        return conversion::reduce_decimal_to_rational(scaled.as_i128(), *decimals);
                    }
                    let mut denom = I256::from_i128(1);
                    for _ in 0..*decimals { denom = denom * I256::from_i128(10); }
                    return Ok(RationalNumber::from_i256_pair(*scaled, denom));
                }
                #[cfg(table_format = "q64_64")]
                {
                    return conversion::reduce_decimal_to_rational(*scaled, *decimals);
                }

            }
            StackValue::Ternary(tier, value, ref shadow) => {
                // Shadow fast path: O(1) when shadow exists
                if let Some((num, den)) = shadow.as_rational() {
                    return Ok(RationalNumber::new(num, den));
                }
                domain::ternary_to_rational(*tier, value)
            }
            StackValue::Error(e) => Err(e.clone()),
        }
    }

    /// Extract the raw binary storage value (profile-specific type).
    ///
    /// Returns `None` for non-binary values and error states.
    /// For `BinaryCompute`, materializes (downscales) to storage tier first.
    pub fn as_binary_storage(&self) -> Option<BinaryStorage> {
        match self {
            StackValue::Binary(_, val, _) => Some(*val),
            StackValue::BinaryCompute(_, val, _) => Some(downscale_to_storage(*val)),
            _ => None,
        }
    }

    /// Extract the tier number.
    ///
    /// Returns the precision tier (1-6) for domain values, 0 for errors.
    pub fn tier(&self) -> u8 {
        match self {
            StackValue::Binary(t, _, _) => *t,
            StackValue::BinaryCompute(t, _, _) => *t,
            StackValue::Decimal(t, _, _) => *t,
            StackValue::Ternary(t, _, _) => *t,
            StackValue::Symbolic(_) => 8, // Symbolic = rational tier
            StackValue::Error(_) => 0,
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
            StackValue::BinaryCompute(tier, val, shadow) => {
                let storage_val = downscale_to_storage(*val);
                let materialized = StackValue::Binary(*tier, storage_val, shadow.clone());
                materialized.to_decimal_string(max_digits)
            }
            StackValue::Binary(_tier, val, _) => {
                formatting::binary_storage_to_decimal_string(*val, max_digits)
            }
            StackValue::Decimal(decimals, val, _) => {
                let full = formatting::decimal_storage_to_string(*decimals, val);
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
            StackValue::Symbolic(r) => {
                formatting::rational_to_decimal_string(r, max_digits)
            }
            StackValue::Ternary(_, _, _) => {
                if let Ok(rational) = self.to_rational() {
                    formatting::rational_to_decimal_string(&rational, max_digits)
                } else {
                    "NaN".to_string()
                }
            }
            StackValue::Error(_) => {
                "NaN".to_string()
            }
        }
    }
}

impl Display for StackValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(table_format = "q64_64")]
        const DEFAULT_DIGITS: usize = 19;
        #[cfg(table_format = "q128_128")]
        const DEFAULT_DIGITS: usize = 38;
        #[cfg(table_format = "q256_256")]
        const DEFAULT_DIGITS: usize = 77;

        let precision = f.precision().unwrap_or(DEFAULT_DIGITS);

        match self {
            StackValue::BinaryCompute(tier, val, shadow) => {
                let storage_val = downscale_to_storage(*val);
                let materialized = StackValue::Binary(*tier, storage_val, shadow.clone());
                write!(f, "{}", materialized.to_decimal_string(precision))
            }
            StackValue::Binary(_, val, _) => {
                write!(f, "{}", formatting::binary_storage_to_decimal_string(*val, precision))
            }
            StackValue::Decimal(dec, val, _) => {
                write!(f, "{}", formatting::decimal_storage_to_string(*dec, val))
            }
            StackValue::Ternary(_, _, _) => {
                write!(f, "{}", self.to_decimal_string(precision))
            }
            StackValue::Symbolic(s) => write!(f, "{}", formatting::rational_to_decimal_string(s, precision)),
            StackValue::Error(e) => write!(f, "Error: {:?}", e),
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
pub struct StackEvaluator {
    /// Current workspace offset
    offset: usize,

    /// Stack of computed values (fixed size)
    value_stack: [Option<StackValue>; 256],

    /// Current stack pointer
    stack_ptr: usize,

    /// Deployment profile for tier limits
    deployment_profile: DeploymentProfile,
}

impl StackEvaluator {
    /// Create new evaluator with deployment profile
    pub fn new(deployment_profile: DeploymentProfile) -> Self {
        Self {
            offset: 0,
            value_stack: [const { None }; 256],
            stack_ptr: 0,
            deployment_profile,
        }
    }

    /// Reset evaluator state for new computation
    pub fn reset(&mut self) {
        self.offset = 0;
        self.stack_ptr = 0;
    }

    /// Load value from stack reference
    fn load_ref(&self, stack_ref: StackRef) -> Result<StackValue, OverflowDetected> {
        let idx = stack_ref.0 as usize;
        if idx >= self.stack_ptr {
            return Err(OverflowDetected::InvalidStackReference);
        }

        self.value_stack[idx]
            .clone()
            .ok_or(OverflowDetected::StackCorruption)
    }

    /// Evaluate expression on stack
    pub fn evaluate(&mut self, expr: &LazyExpr) -> Result<StackValue, OverflowDetected> {
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
                let val = self.evaluate(inner)?;
                self.evaluate_exp(val)
            }
            LazyExpr::Ln(inner) => {
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
    fn materialize_compute(&self, value: StackValue) -> StackValue {
        match value {
            StackValue::BinaryCompute(tier, val, shadow) => {
                StackValue::Binary(tier, downscale_to_storage(val), shadow)
            }
            other => other,
        }
    }
}

// ============================================================================
// THREAD-LOCAL EVALUATOR INSTANCE
// ============================================================================

/// Get the compile-time deployment profile based on table format
const fn compile_time_profile() -> DeploymentProfile {
    #[cfg(table_format = "q256_256")]
    { DeploymentProfile::Scientific }

    #[cfg(table_format = "q128_128")]
    { DeploymentProfile::Balanced }

    #[cfg(table_format = "q64_64")]
    { DeploymentProfile::Embedded }
}

thread_local! {
    static EVALUATOR: RefCell<StackEvaluator> = RefCell::new(
        StackEvaluator::new(compile_time_profile())
    );
}

/// Evaluate expression using thread-local evaluator
///
/// **MATERIALIZATION**: Any BinaryCompute results are materialized to Binary
/// before returning, ensuring callers always receive storage-tier values.
///
/// **MODE ROUTING**: If a non-Auto output mode is set via `set_mode()`,
/// the result is converted to the requested domain after materialization.
pub fn evaluate(expr: &LazyExpr) -> Result<StackValue, OverflowDetected> {
    EVALUATOR.with(|eval| {
        let mut evaluator = eval.borrow_mut();
        evaluator.reset();
        let result = evaluator.evaluate(expr)?;
        let materialized = evaluator.materialize_compute(result);
        evaluator.apply_output_mode(materialized)
    })
}

/// Parse a runtime string into a StackValue
///
/// **PURPOSE**: Bridge for runtime/dynamic string inputs that cannot use
/// `gmath()` (which requires `&'static str`). Eagerly parses through the
/// thread-local evaluator with full mode routing support.
///
/// **USAGE**:
/// ```rust
/// use g_math::canonical::{gmath_parse, evaluate, LazyExpr};
///
/// let user_input = String::from("3.14");
/// let expr = gmath_parse(&user_input).unwrap();
/// let result = evaluate(&(expr + gmath_parse("2.0").unwrap())).unwrap();
/// ```
pub fn gmath_parse(s: &str) -> Result<LazyExpr, OverflowDetected> {
    let value = EVALUATOR.with(|eval| {
        let mut evaluator = eval.borrow_mut();
        evaluator.parse_literal_with_mode(s)
    })?;
    Ok(LazyExpr::Value(Box::new(value)))
}
