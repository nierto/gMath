//! Fractal Topology Router — Value-Adaptive Domain Dispatch
//!
//! **PURPOSE**: Eliminate domain crisscross in arithmetic by classifying operands
//! via CompactShadow denominator factoring and dispatching through a compile-time
//! routing table.
//!
//! **ARCHITECTURE**:
//! - Layer 1: Shadow classifier (~15 ns) — factors denominator, sets exact_in bits
//! - Layer 2: Routing table (~3 ns) — static lookup in .rodata, zero sync
//! - Layer 3: Tree walker (~200 ns) — bottom-up LazyExpr annotation
//!
//! **INVARIANTS**:
//! - Zero mutable state; routing table in .rodata
//! - Thread-safe by construction (all inputs → pure functions → output)
//! - Integer-only classification (no f32/f64 anywhere)
//! - O(1) per-node routing decision via table lookup

use crate::fixed_point::universal::tier_types::CompactShadow;
use crate::fixed_point::universal::fasc::stack_evaluator::StackValue;
use crate::fixed_point::universal::fasc::lazy_expr::LazyExpr;

// ============================================================================
// DOMAIN CHOICE
// ============================================================================

/// Target domain for an operation, determined by the routing table.
///
/// Precedence: Binary (fastest for integers) < Decimal (exact base-10) < Symbolic (always exact).
/// Ternary routing is deliberately sequenced, not abandoned: the classifier already
/// computes `TERNARY_BIT`, but no table column consumes it (ternary-exact values fall
/// through to Binary) until the balanced-ternary domain has its dedicated validation
/// suite. Adding `Ternary` here plus a table column then rescues denominator-3^k
/// values (e.g. 1/3 + 1/3) from the symbolic fallback into exact fixed-point.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DomainChoice {
    /// Binary fixed-point (Q-format). Fastest for integers and powers of 2.
    Binary = 0,
    /// Decimal fixed-point (scaled integer). Exact for base-10 values.
    Decimal = 1,
    /// Symbolic rational (a/b). Always exact, always available, slowest.
    Symbolic = 2,
}

// ============================================================================
// OPERAND CLASSIFICATION
// ============================================================================

/// Bitmask constants for the `exact_in` field of `OperandClass`.
pub const TERNARY_BIT: u8 = 1 << 0; // 0b0001
pub const BINARY_BIT: u8 = 1 << 1;  // 0b0010
pub const DECIMAL_BIT: u8 = 1 << 2; // 0b0100
pub const SYMBOLIC_BIT: u8 = 1 << 3; // 0b1000

/// Operand classification for routing table lookup.
///
/// `exact_in` is a 4-bit bitmask indicating which domains can represent
/// this value exactly. Indexes directly into the 16×16 routing table.
///
/// **Classification algorithm** (~15 ns via shadow, ~5 ns variant fallback):
/// 1. Extract (num, den) from CompactShadow
/// 2. Strip factors of 2 from den → if remainder == 1, set binary-exact bit
/// 3. Strip factors of 2 and 5 → if remainder == 1, set decimal-exact bit
/// 4. Strip factors of 3 → if remainder == 1, set ternary-exact bit
/// 5. Symbolic bit always set for finite rationals
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OperandClass {
    pub exact_in: u8,
}

impl OperandClass {
    /// Exact in all four domains (pure integers, e.g. "42", "0xFF").
    pub const ALL: Self = Self {
        exact_in: TERNARY_BIT | BINARY_BIT | DECIMAL_BIT | SYMBOLIC_BIT,
    };
    /// Exact only in binary domain (BinaryCompute intermediates without shadow).
    pub const BINARY_ONLY: Self = Self { exact_in: BINARY_BIT };
    /// Exact in decimal and symbolic (e.g. "0.1", "3.14").
    pub const DECIMAL_SYMBOLIC: Self = Self {
        exact_in: DECIMAL_BIT | SYMBOLIC_BIT,
    };
    /// Exact only in symbolic (e.g. 1/7, π).
    pub const SYMBOLIC_ONLY: Self = Self { exact_in: SYMBOLIC_BIT };

    /// Index into routing table (lower 4 bits, 0..15).
    #[inline(always)]
    pub const fn index(self) -> usize {
        (self.exact_in & 0x0F) as usize
    }
}

// ============================================================================
// OPERATION IDS
// ============================================================================

/// Operation identifier for routing table indexing.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OpId {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Exp = 4,
    Ln = 5,
    Sqrt = 6,
    Sin = 7,
    Cos = 8,
    Tan = 9,
    Atan = 10,
    Atan2 = 11,
    Pow = 12,
    Asin = 13,
    Acos = 14,
    Sinh = 15,
    Cosh = 16,
    Tanh = 17,
    Asinh = 18,
    Acosh = 19,
    Atanh = 20,
}

const NUM_OPS: usize = 21;

// ============================================================================
// SHADOW CLASSIFIER (~15 ns)
// ============================================================================

/// Classify a StackValue by factoring its CompactShadow denominator.
///
/// **Performance**: ~15 ns for shadow path (typical), ~5 ns for variant fallback.
/// **Coverage**: ~95% of parsed values have shadows (from parse_literal).
#[inline]
pub fn classify(value: &StackValue) -> OperandClass {
    let shadow = value.shadow();

    // Fast path: shadow carries rational (num, den) — factor denominator
    if let Some((_num, den)) = shadow.as_rational() {
        return classify_denominator(den);
    }

    // Mathematical constant (π, e, √2) → binary + symbolic
    if shadow.constant_id().is_some() {
        return OperandClass {
            exact_in: BINARY_BIT | SYMBOLIC_BIT,
        };
    }

    // No shadow: fall back to StackValue variant
    classify_by_variant(value)
}

/// Classify by denominator factoring (pure integer arithmetic, no f64).
///
/// A rational num/den is:
/// - **ternary-exact** iff den = 3^k for some k ≥ 0
/// - **binary-exact** iff den = 2^k for some k ≥ 0
/// - **decimal-exact** iff den = 2^a × 5^b for some a,b ≥ 0
/// - **symbolic-exact** always (all finite rationals are symbolic)
#[inline]
pub fn classify_denominator(den: u128) -> OperandClass {
    if den == 0 {
        return OperandClass::SYMBOLIC_ONLY;
    }

    let mut exact_in: u8 = SYMBOLIC_BIT; // all rationals are symbolic-exact

    // Binary-exact: den = 2^k
    let mut d = den;
    while d & 1 == 0 {
        d >>= 1;
    }
    if d == 1 {
        exact_in |= BINARY_BIT;
    }

    // Decimal-exact: den = 2^a × 5^b
    d = den;
    while d % 2 == 0 {
        d /= 2;
    }
    while d % 5 == 0 {
        d /= 5;
    }
    if d == 1 {
        exact_in |= DECIMAL_BIT;
    }

    // Ternary-exact: den = 3^k
    d = den;
    while d % 3 == 0 {
        d /= 3;
    }
    if d == 1 {
        exact_in |= TERNARY_BIT;
    }

    OperandClass { exact_in }
}

/// Fallback classification when no shadow is available.
#[inline]
fn classify_by_variant(value: &StackValue) -> OperandClass {
    match value {
        StackValue::Binary(..) | StackValue::BinaryCompute(..) => OperandClass::BINARY_ONLY,
        StackValue::Decimal(..) | StackValue::DecimalCompute(..) => OperandClass {
            exact_in: DECIMAL_BIT,
        },
        StackValue::Ternary(..) => OperandClass {
            exact_in: TERNARY_BIT,
        },
        StackValue::Symbolic(_) => OperandClass::SYMBOLIC_ONLY,
        StackValue::Error(_) => OperandClass { exact_in: 0 },
    }
}

// ============================================================================
// ROUTING TABLE (~3 ns lookup, 5.25 KB .rodata)
// ============================================================================

/// Static routing table: 21 ops × 16 left classes × 16 right classes.
///
/// Lives in .rodata, shared across all threads, zero sync overhead.
/// Size: 21 × 16 × 16 × 1 byte = 5,376 bytes.
///
/// For arithmetic (Add/Sub/Mul/Div): picks lowest-rank domain where both
/// operands are exact. For transcendentals: routes to the input's
/// preferred engine.
static ROUTING_TABLE: [[[DomainChoice; 16]; 16]; NUM_OPS] = build_routing_table();

const fn build_routing_table() -> [[[DomainChoice; 16]; 16]; NUM_OPS] {
    let mut table = [[[DomainChoice::Symbolic; 16]; 16]; NUM_OPS];
    let mut op = 0usize;
    while op < NUM_OPS {
        let mut left = 0u8;
        while left < 16 {
            let mut right = 0u8;
            while right < 16 {
                table[op][left as usize][right as usize] =
                    compute_route(op, left, right);
                right += 1;
            }
            left += 1;
        }
        op += 1;
    }
    table
}

/// Compute route for a single (op, left_class, right_class) triple.
const fn compute_route(op: usize, left_class: u8, right_class: u8) -> DomainChoice {
    if op <= 3 {
        // Arithmetic (Add, Sub, Mul, Div): intersection → lowest rank
        let intersection = left_class & right_class;
        lowest_rank_domain(intersection)
    } else {
        // Transcendentals: route by input's domain (left operand only)
        transcendental_route(left_class)
    }
}

/// Pick the lowest-rank (most efficient) domain from an exact_in bitmask.
///
/// Precedence: Binary < Decimal < Symbolic.
/// Ternary routing deferred — ternary bit is skipped.
const fn lowest_rank_domain(exact_in: u8) -> DomainChoice {
    if exact_in & BINARY_BIT != 0 {
        DomainChoice::Binary
    } else if exact_in & DECIMAL_BIT != 0 {
        DomainChoice::Decimal
    } else {
        DomainChoice::Symbolic
    }
}

/// Route a transcendental to the input's preferred engine.
///
/// Decimal engine used when input is decimal-exact but NOT binary-exact
/// (honors the "stated domain" — "3.0" → decimal engine, "3" → binary engine).
/// Binary engine used for everything else (always available).
const fn transcendental_route(class: u8) -> DomainChoice {
    if class & DECIMAL_BIT != 0 && class & BINARY_BIT == 0 {
        DomainChoice::Decimal
    } else {
        DomainChoice::Binary
    }
}

// ============================================================================
// ROUTE FUNCTIONS (public API)
// ============================================================================

/// Route a binary operation given classified operands.
///
/// ~3 ns: single array lookup, L1-cache resident (5.25 KB table).
#[inline(always)]
pub fn route_binary_op(op: OpId, left: OperandClass, right: OperandClass) -> DomainChoice {
    ROUTING_TABLE[op as usize][left.index()][right.index()]
}

/// Route a unary operation given classified operand.
///
/// ~3 ns: single array lookup. Right operand index fixed at 0.
#[inline(always)]
pub fn route_unary_op(op: OpId, operand: OperandClass) -> DomainChoice {
    ROUTING_TABLE[op as usize][operand.index()][0]
}

// ============================================================================
// CROSS-DOMAIN COERCION
// ============================================================================

/// Coerce a StackValue to the Decimal domain using its CompactShadow.
///
/// Returns `Some(Decimal(dp, scaled, shadow))` if the shadow's denominator
/// has only factors of 2 and 5 (decimal-representable).
/// Returns `None` if coercion is impossible.
///
/// **Performance**: ~20 ns (shadow extraction + denominator factoring + scaling).
pub fn coerce_to_decimal(value: &StackValue) -> Option<StackValue> {
    match value {
        StackValue::Decimal(..) | StackValue::DecimalCompute(..) => Some(value.clone()),
        _ => {
            let shadow = value.shadow();
            if let Some((num, den)) = shadow.as_rational() {
                shadow_to_decimal(num, den, shadow)
            } else {
                None
            }
        }
    }
}

/// Convert a rational (num/den) to a Decimal StackValue via denominator factoring.
///
/// Given value = num/den where den = 2^a × 5^b:
///   dp = max(a, b)
///   scaled = num × 2^(dp-a) × 5^(dp-b)
///   result = Decimal(dp, to_binary_storage(scaled), shadow)
fn shadow_to_decimal(num: i128, den: u128, shadow: CompactShadow) -> Option<StackValue> {
    if den == 0 {
        return None;
    }

    let mut d = den;
    let mut count2 = 0u32;
    let mut count5 = 0u32;
    while d % 2 == 0 {
        d /= 2;
        count2 += 1;
    }
    while d % 5 == 0 {
        d /= 5;
        count5 += 1;
    }
    if d != 1 {
        return None; // Not decimal-representable
    }

    let dp = count2.max(count5);
    if dp > 38 {
        return None; // Would overflow i128 scaling
    }

    // scaled = num × 2^(dp - count2) × 5^(dp - count5)
    let mut scaled = num;
    for _ in 0..(dp - count2) {
        scaled = scaled.checked_mul(2)?;
    }
    for _ in 0..(dp - count5) {
        scaled = scaled.checked_mul(5)?;
    }

    use crate::fixed_point::universal::fasc::stack_evaluator::conversion::to_binary_storage;
    let storage = to_binary_storage(scaled);
    Some(StackValue::Decimal(dp as u8, storage, shadow))
}

// ============================================================================
// TREE WALKER — O(N) bottom-up expression routing
// ============================================================================

/// Walk a LazyExpr tree bottom-up, returning the root's OperandClass.
///
/// **Complexity**: O(N) in tree size, single recursive pass.
/// **Performance**: ~200-300 ns for typical 10-node expressions.
///
/// The returned OperandClass represents the domains that can produce an
/// exact (or best-available) result for the entire expression.
pub fn route_expression(expr: &LazyExpr) -> OperandClass {
    match expr {
        LazyExpr::Literal(s) => classify_literal_string(s),
        LazyExpr::Value(v) => classify(v),
        LazyExpr::Constant(_) => OperandClass {
            exact_in: BINARY_BIT | SYMBOLIC_BIT,
        },
        LazyExpr::Variable(_) => OperandClass::ALL,

        // Negation preserves exactness class
        LazyExpr::Negate(inner) => route_expression(inner),

        // Binary arithmetic: intersection of children's exactness
        LazyExpr::Add(l, r)
        | LazyExpr::Sub(l, r)
        | LazyExpr::Mul(l, r)
        | LazyExpr::Div(l, r) => {
            let lc = route_expression(l);
            let rc = route_expression(r);
            OperandClass {
                exact_in: lc.exact_in & rc.exact_in,
            }
        }

        // Binary transcendentals (pow, atan2)
        LazyExpr::Pow(base, _exp) | LazyExpr::Atan2(base, _exp) => {
            // Result is in the compute domain of the base
            let child = route_expression(base);
            let domain = transcendental_route(child.exact_in);
            domain_to_class(domain)
        }

        // Unary transcendentals: output is in the engine's compute domain
        LazyExpr::Exp(inner)
        | LazyExpr::Ln(inner)
        | LazyExpr::Sqrt(inner)
        | LazyExpr::Sin(inner)
        | LazyExpr::Cos(inner)
        | LazyExpr::Tan(inner)
        | LazyExpr::Asin(inner)
        | LazyExpr::Acos(inner)
        | LazyExpr::Atan(inner)
        | LazyExpr::Sinh(inner)
        | LazyExpr::Cosh(inner)
        | LazyExpr::Tanh(inner)
        | LazyExpr::Asinh(inner)
        | LazyExpr::Acosh(inner)
        | LazyExpr::Atanh(inner) => {
            let child = route_expression(inner);
            let domain = transcendental_route(child.exact_in);
            domain_to_class(domain)
        }
    }
}

/// Convert a DomainChoice to the OperandClass a compute engine produces.
fn domain_to_class(domain: DomainChoice) -> OperandClass {
    match domain {
        DomainChoice::Binary => OperandClass::BINARY_ONLY,
        DomainChoice::Decimal => OperandClass {
            exact_in: DECIMAL_BIT,
        },
        DomainChoice::Symbolic => OperandClass::SYMBOLIC_ONLY,
    }
}

/// Classify a literal string without full parsing.
///
/// Determines domain exactness from string format. Conservative for decimals
/// (doesn't compute GCD to check binary-exactness of e.g. "0.5" = 1/2).
/// The shadow classifier (post-parse) provides the full picture.
pub fn classify_literal_string(s: &str) -> OperandClass {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return OperandClass::SYMBOLIC_ONLY;
    }

    // Hex/binary/ternary prefix → integer → all domains
    if bytes.len() > 1 && bytes[0] == b'0' {
        match bytes[1] {
            b'x' | b'X' | b'b' | b'B' | b't' | b'T' => return OperandClass::ALL,
            _ => {}
        }
    }

    // Named constant (pi, e, sqrt2) → binary + symbolic
    if bytes[0].is_ascii_alphabetic() || bytes[0] > 127 {
        return OperandClass {
            exact_in: BINARY_BIT | SYMBOLIC_BIT,
        };
    }

    // Fraction: classify denominator
    if let Some(slash_pos) = bytes.iter().position(|&b| b == b'/') {
        let den_str = &s[slash_pos + 1..];
        let den_str = den_str.trim();
        if let Ok(den) = den_str.parse::<u128>() {
            return classify_denominator(den);
        }
        return OperandClass::SYMBOLIC_ONLY;
    }

    // Repeating decimal → symbolic
    if s.len() > 3 && s.ends_with("...") {
        return OperandClass::SYMBOLIC_ONLY;
    }

    // Decimal point → decimal-exact + symbolic (conservative)
    if bytes.contains(&b'.') {
        return OperandClass::DECIMAL_SYMBOLIC;
    }

    // Pure integer → exact in all domains
    OperandClass::ALL
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Denominator classification ---

    #[test]
    fn classify_den_1_is_all_domains() {
        let c = classify_denominator(1);
        assert_ne!(c.exact_in & BINARY_BIT, 0, "den=1 should be binary-exact");
        assert_ne!(c.exact_in & DECIMAL_BIT, 0, "den=1 should be decimal-exact");
        assert_ne!(c.exact_in & TERNARY_BIT, 0, "den=1 should be ternary-exact");
        assert_ne!(c.exact_in & SYMBOLIC_BIT, 0, "den=1 should be symbolic-exact");
    }

    #[test]
    fn classify_den_2_is_binary_decimal() {
        let c = classify_denominator(2);
        assert_ne!(c.exact_in & BINARY_BIT, 0, "1/2 should be binary-exact");
        assert_ne!(c.exact_in & DECIMAL_BIT, 0, "1/2 should be decimal-exact");
        assert_eq!(c.exact_in & TERNARY_BIT, 0, "1/2 should NOT be ternary-exact");
    }

    #[test]
    fn classify_den_10_is_decimal_only() {
        let c = classify_denominator(10);
        assert_eq!(c.exact_in & BINARY_BIT, 0, "1/10 should NOT be binary-exact");
        assert_ne!(c.exact_in & DECIMAL_BIT, 0, "1/10 should be decimal-exact");
        assert_eq!(c.exact_in & TERNARY_BIT, 0, "1/10 should NOT be ternary-exact");
    }

    #[test]
    fn classify_den_100_is_decimal_only() {
        let c = classify_denominator(100);
        assert_eq!(c.exact_in & BINARY_BIT, 0);
        assert_ne!(c.exact_in & DECIMAL_BIT, 0);
    }

    #[test]
    fn classify_den_3_is_ternary_only() {
        let c = classify_denominator(3);
        assert_eq!(c.exact_in & BINARY_BIT, 0);
        assert_eq!(c.exact_in & DECIMAL_BIT, 0);
        assert_ne!(c.exact_in & TERNARY_BIT, 0);
    }

    #[test]
    fn classify_den_27_is_ternary_only() {
        let c = classify_denominator(27); // 3^3
        assert_ne!(c.exact_in & TERNARY_BIT, 0);
        assert_eq!(c.exact_in & BINARY_BIT, 0);
    }

    #[test]
    fn classify_den_7_is_symbolic_only() {
        let c = classify_denominator(7);
        assert_eq!(c.exact_in & BINARY_BIT, 0);
        assert_eq!(c.exact_in & DECIMAL_BIT, 0);
        assert_eq!(c.exact_in & TERNARY_BIT, 0);
        assert_ne!(c.exact_in & SYMBOLIC_BIT, 0);
    }

    #[test]
    fn classify_den_8_is_binary_decimal() {
        let c = classify_denominator(8); // 2^3
        assert_ne!(c.exact_in & BINARY_BIT, 0, "1/8 binary-exact");
        assert_ne!(c.exact_in & DECIMAL_BIT, 0, "1/8 = 0.125, decimal-exact");
    }

    #[test]
    fn classify_den_5_is_decimal_only() {
        let c = classify_denominator(5);
        assert_eq!(c.exact_in & BINARY_BIT, 0, "1/5 not binary-exact");
        assert_ne!(c.exact_in & DECIMAL_BIT, 0, "1/5 = 0.2, decimal-exact");
    }

    #[test]
    fn classify_den_6_is_symbolic() {
        let c = classify_denominator(6); // 2 × 3
        assert_eq!(c.exact_in & BINARY_BIT, 0);
        assert_eq!(c.exact_in & DECIMAL_BIT, 0);
        assert_eq!(c.exact_in & TERNARY_BIT, 0);
        assert_ne!(c.exact_in & SYMBOLIC_BIT, 0);
    }

    // --- Routing table ---

    #[test]
    fn route_decimal_plus_integer_is_decimal() {
        // "0.1" (decimal+symbolic) + "255" (all) → Decimal
        let lc = OperandClass::DECIMAL_SYMBOLIC;
        let rc = OperandClass::ALL;
        assert_eq!(route_binary_op(OpId::Add, lc, rc), DomainChoice::Decimal);
    }

    #[test]
    fn route_integer_plus_integer_is_binary() {
        // "3" (all) + "6" (all) → Binary (lowest rank)
        let lc = OperandClass::ALL;
        let rc = OperandClass::ALL;
        assert_eq!(route_binary_op(OpId::Add, lc, rc), DomainChoice::Binary);
    }

    #[test]
    fn route_decimal_plus_ternary_fraction_is_symbolic() {
        // "0.1" (decimal+symbolic) + "1/3" (ternary+symbolic)
        let lc = OperandClass::DECIMAL_SYMBOLIC;
        let rc = OperandClass {
            exact_in: TERNARY_BIT | SYMBOLIC_BIT,
        };
        assert_eq!(
            route_binary_op(OpId::Add, lc, rc),
            DomainChoice::Symbolic
        );
    }

    #[test]
    fn route_binary_half_plus_decimal_is_decimal() {
        // "0.5" (binary+decimal+symbolic) + "0.1" (decimal+symbolic)
        let lc = OperandClass {
            exact_in: BINARY_BIT | DECIMAL_BIT | SYMBOLIC_BIT,
        };
        let rc = OperandClass::DECIMAL_SYMBOLIC;
        // Intersection: decimal + symbolic → lowest rank = Decimal
        // (Binary bit only in lc, not rc)
        assert_eq!(route_binary_op(OpId::Mul, lc, rc), DomainChoice::Decimal);
    }

    #[test]
    fn route_mul_all_domains() {
        assert_eq!(
            route_binary_op(OpId::Mul, OperandClass::ALL, OperandClass::ALL),
            DomainChoice::Binary
        );
    }

    #[test]
    fn route_div_decimal_by_integer() {
        let lc = OperandClass::DECIMAL_SYMBOLIC;
        let rc = OperandClass::ALL;
        assert_eq!(route_binary_op(OpId::Div, lc, rc), DomainChoice::Decimal);
    }

    // --- Transcendental routing ---

    #[test]
    fn route_exp_decimal_input() {
        // Decimal-only input → decimal engine
        let c = OperandClass::DECIMAL_SYMBOLIC;
        assert_eq!(route_unary_op(OpId::Exp, c), DomainChoice::Decimal);
    }

    #[test]
    fn route_exp_integer_input() {
        // Integer (all domains) → binary engine (binary bit set)
        let c = OperandClass::ALL;
        assert_eq!(route_unary_op(OpId::Exp, c), DomainChoice::Binary);
    }

    #[test]
    fn route_exp_binary_only() {
        let c = OperandClass::BINARY_ONLY;
        assert_eq!(route_unary_op(OpId::Exp, c), DomainChoice::Binary);
    }

    // --- Literal string classification ---

    #[test]
    fn literal_decimal() {
        let c = classify_literal_string("0.1");
        assert_ne!(c.exact_in & DECIMAL_BIT, 0);
        assert_eq!(c.exact_in & BINARY_BIT, 0);
    }

    #[test]
    fn literal_integer() {
        assert_eq!(classify_literal_string("255"), OperandClass::ALL);
    }

    #[test]
    fn literal_negative_integer() {
        // Starts with '-', not alphabetic, not '.', not '/' → integer path
        // Actually '-' is not alphabetic, not a digit... let's check
        // bytes[0] = b'-' — not alpha, not '0', not '.'
        // No slash, no dot → integer path → ALL
        assert_eq!(classify_literal_string("-42"), OperandClass::ALL);
    }

    #[test]
    fn literal_hex() {
        assert_eq!(classify_literal_string("0xFF"), OperandClass::ALL);
    }

    #[test]
    fn literal_fraction_third() {
        let c = classify_literal_string("1/3");
        assert_ne!(c.exact_in & TERNARY_BIT, 0);
        assert_eq!(c.exact_in & BINARY_BIT, 0);
        assert_eq!(c.exact_in & DECIMAL_BIT, 0);
    }

    #[test]
    fn literal_fraction_half() {
        let c = classify_literal_string("1/2");
        assert_ne!(c.exact_in & BINARY_BIT, 0);
        assert_ne!(c.exact_in & DECIMAL_BIT, 0);
    }

    #[test]
    fn literal_named_constant() {
        let c = classify_literal_string("pi");
        assert_ne!(c.exact_in & BINARY_BIT, 0);
        assert_eq!(c.exact_in & DECIMAL_BIT, 0);
    }

    #[test]
    fn literal_repeating() {
        let c = classify_literal_string("0.333...");
        assert_eq!(c, OperandClass::SYMBOLIC_ONLY);
    }

    // --- Tree walker ---

    #[test]
    fn tree_walker_literal_add() {
        // gmath("0.1") + gmath("255")
        let expr = LazyExpr::Add(
            Box::new(LazyExpr::Literal("0.1")),
            Box::new(LazyExpr::Literal("255")),
        );
        let class = route_expression(&expr);
        // "0.1" = decimal+symbolic, "255" = all → intersection = decimal+symbolic
        assert_ne!(class.exact_in & DECIMAL_BIT, 0, "should be decimal-exact");
        assert_eq!(class.exact_in & BINARY_BIT, 0, "should NOT be binary-exact");
    }

    #[test]
    fn tree_walker_integer_add() {
        let expr = LazyExpr::Add(
            Box::new(LazyExpr::Literal("3")),
            Box::new(LazyExpr::Literal("6")),
        );
        let class = route_expression(&expr);
        assert_eq!(class, OperandClass::ALL);
    }

    #[test]
    fn tree_walker_transcendental() {
        // exp("0.1") → decimal engine → decimal-only class
        let expr = LazyExpr::Exp(Box::new(LazyExpr::Literal("0.1")));
        let class = route_expression(&expr);
        assert_ne!(class.exact_in & DECIMAL_BIT, 0);
        assert_eq!(class.exact_in & BINARY_BIT, 0);
    }

    #[test]
    fn tree_walker_mixed_chain() {
        // (0.1 + 255).exp() → decimal add → decimal exp
        let inner = LazyExpr::Add(
            Box::new(LazyExpr::Literal("0.1")),
            Box::new(LazyExpr::Literal("255")),
        );
        let expr = LazyExpr::Exp(Box::new(inner));
        let class = route_expression(&expr);
        assert_ne!(class.exact_in & DECIMAL_BIT, 0);
    }
}
