//! Lazy Expression Tree - Fixed Allocation Design
//!
//! **MISSION**: Build expression trees without heap allocation until evaluation
//! **ARCHITECTURE**: Recursive enum with operator overloading for natural syntax
//! **OPTIMIZATION**: Compile-time constant detection and expression simplification

use core::ops::{Add, Sub, Mul, Div, Neg};
use core::fmt::{self, Display};

/// Stack reference (index into evaluation stack)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct StackRef(pub u16);

/// Mathematical constant identifiers for pattern recognition
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConstantId {
    Pi,
    E,
    Sqrt2,
    Phi,        // Golden ratio
    EulerGamma, // Euler-Mascheroni constant
}

/// Lazy expression tree - no heap allocations during construction
///
/// **ARCHITECTURE**: Expression nodes build tree structure lazily
/// **EVALUATION**: Deferred until explicitly requested or display
/// **OPTIMIZATION**: Compile-time constant folding where possible
#[derive(Debug, Clone)]
pub enum LazyExpr {
    /// String literal (compile-time or static reference)
    Literal(&'static str),

    /// Pre-evaluated value — enables chaining results across evaluate() calls.
    ///
    /// Preserves full precision + shadow from previous computation.
    /// The evaluator passes this through directly — no re-parsing, no precision loss.
    ///
    /// ```rust
    /// use g_math::canonical::{gmath, evaluate, LazyExpr};
    /// let balance = evaluate(&(gmath("1000.00") * gmath("1.05"))).unwrap();
    /// // Feed result back into next expression — shadow + precision preserved
    /// let next = evaluate(&(LazyExpr::from(balance) * gmath("1.05"))).unwrap();
    /// ```
    Value(Box<super::stack_evaluator::StackValue>),

    /// Mathematical constant by ID
    Constant(ConstantId),

    /// Stack variable reference (for intermediate values)
    Variable(StackRef),

    /// Negation (unary operation)
    Negate(Box<LazyExpr>),

    /// Binary operations (Box for recursive structure only)
    Add(Box<LazyExpr>, Box<LazyExpr>),
    Sub(Box<LazyExpr>, Box<LazyExpr>),
    Mul(Box<LazyExpr>, Box<LazyExpr>),
    Div(Box<LazyExpr>, Box<LazyExpr>),

    /// Transcendental operations (unary)
    Exp(Box<LazyExpr>),
    Ln(Box<LazyExpr>),
    Sqrt(Box<LazyExpr>),

    /// Power function (binary): x^y = exp(y × ln(x))
    Pow(Box<LazyExpr>, Box<LazyExpr>),

    // Hyperbolic functions (FASC-composed from exp/ln/sqrt)
    Sinh(Box<LazyExpr>),
    Cosh(Box<LazyExpr>),
    Tanh(Box<LazyExpr>),
    Asinh(Box<LazyExpr>),
    Acosh(Box<LazyExpr>),
    Atanh(Box<LazyExpr>),

    // Trigonometric functions
    Sin(Box<LazyExpr>),
    Cos(Box<LazyExpr>),
    Tan(Box<LazyExpr>),

    // Inverse trigonometric functions
    Asin(Box<LazyExpr>),
    Acos(Box<LazyExpr>),
    Atan(Box<LazyExpr>),
    Atan2(Box<LazyExpr>, Box<LazyExpr>),
}

impl LazyExpr {
    /// Check if expression is a compile-time constant
    pub fn is_constant(&self) -> bool {
        match self {
            LazyExpr::Literal(_) | LazyExpr::Constant(_) | LazyExpr::Value(_) => true,
            LazyExpr::Variable(_) => false,
            LazyExpr::Negate(inner) | LazyExpr::Exp(inner) | LazyExpr::Ln(inner) | LazyExpr::Sqrt(inner)
            | LazyExpr::Sinh(inner) | LazyExpr::Cosh(inner) | LazyExpr::Tanh(inner)
            | LazyExpr::Asinh(inner) | LazyExpr::Acosh(inner) | LazyExpr::Atanh(inner)
            | LazyExpr::Sin(inner) | LazyExpr::Cos(inner) | LazyExpr::Tan(inner)
            | LazyExpr::Asin(inner) | LazyExpr::Acos(inner) | LazyExpr::Atan(inner) => inner.is_constant(),
            LazyExpr::Add(l, r) | LazyExpr::Sub(l, r) |
            LazyExpr::Mul(l, r) | LazyExpr::Div(l, r) |
            LazyExpr::Pow(l, r) | LazyExpr::Atan2(l, r) => {
                l.is_constant() && r.is_constant()
            }
        }
    }

    /// Get expression depth for stack allocation planning
    pub fn depth(&self) -> usize {
        match self {
            LazyExpr::Literal(_) | LazyExpr::Constant(_) | LazyExpr::Variable(_) | LazyExpr::Value(_) => 1,
            LazyExpr::Negate(inner) | LazyExpr::Exp(inner) | LazyExpr::Ln(inner) | LazyExpr::Sqrt(inner)
            | LazyExpr::Sinh(inner) | LazyExpr::Cosh(inner) | LazyExpr::Tanh(inner)
            | LazyExpr::Asinh(inner) | LazyExpr::Acosh(inner) | LazyExpr::Atanh(inner)
            | LazyExpr::Sin(inner) | LazyExpr::Cos(inner) | LazyExpr::Tan(inner)
            | LazyExpr::Asin(inner) | LazyExpr::Acos(inner) | LazyExpr::Atan(inner) => 1 + inner.depth(),
            LazyExpr::Add(l, r) | LazyExpr::Sub(l, r) |
            LazyExpr::Mul(l, r) | LazyExpr::Div(l, r) |
            LazyExpr::Pow(l, r) | LazyExpr::Atan2(l, r) => {
                1 + l.depth().max(r.depth())
            }
        }
    }

    /// Count number of operations in expression tree
    pub fn operation_count(&self) -> usize {
        match self {
            LazyExpr::Literal(_) | LazyExpr::Constant(_) | LazyExpr::Variable(_) | LazyExpr::Value(_) => 0,
            LazyExpr::Negate(inner) | LazyExpr::Exp(inner) | LazyExpr::Ln(inner) | LazyExpr::Sqrt(inner)
            | LazyExpr::Sinh(inner) | LazyExpr::Cosh(inner) | LazyExpr::Tanh(inner)
            | LazyExpr::Asinh(inner) | LazyExpr::Acosh(inner) | LazyExpr::Atanh(inner)
            | LazyExpr::Sin(inner) | LazyExpr::Cos(inner) | LazyExpr::Tan(inner)
            | LazyExpr::Asin(inner) | LazyExpr::Acos(inner) | LazyExpr::Atan(inner) => 1 + inner.operation_count(),
            LazyExpr::Add(l, r) | LazyExpr::Sub(l, r) |
            LazyExpr::Mul(l, r) | LazyExpr::Div(l, r) |
            LazyExpr::Pow(l, r) | LazyExpr::Atan2(l, r) => {
                1 + l.operation_count() + r.operation_count()
            }
        }
    }

    /// Power function: self^exponent
    ///
    /// **ALGORITHM**: pow(x, y) = exp(y × ln(x))
    /// **PRECISION**: Same contractual guarantees as exp() and ln()
    pub fn pow(self, exponent: LazyExpr) -> LazyExpr {
        LazyExpr::Pow(Box::new(self), Box::new(exponent))
    }
}

/// Convert a pre-evaluated StackValue into a LazyExpr for chaining.
///
/// Preserves full precision and shadow — no string round-trip.
impl From<super::stack_evaluator::StackValue> for LazyExpr {
    fn from(val: super::stack_evaluator::StackValue) -> Self {
        LazyExpr::Value(Box::new(val))
    }
}

/// Universal entry point - zero allocation expression builder
///
/// **USAGE**: `let expr = gmath("123.456");`
/// **ARCHITECTURE**: Returns expression node without parsing
/// **EVALUATION**: Deferred until display or explicit evaluation
pub const fn gmath(input: &'static str) -> LazyExpr {
    LazyExpr::Literal(input)
}

// ============================================================================
// OPERATOR OVERLOADING FOR NATURAL EXPRESSION SYNTAX
// ============================================================================

impl Add for LazyExpr {
    type Output = LazyExpr;

    fn add(self, other: Self) -> Self::Output {
        LazyExpr::Add(Box::new(self), Box::new(other))
    }
}

impl Sub for LazyExpr {
    type Output = LazyExpr;

    fn sub(self, other: Self) -> Self::Output {
        LazyExpr::Sub(Box::new(self), Box::new(other))
    }
}

impl Mul for LazyExpr {
    type Output = LazyExpr;

    fn mul(self, other: Self) -> Self::Output {
        LazyExpr::Mul(Box::new(self), Box::new(other))
    }
}

impl Div for LazyExpr {
    type Output = LazyExpr;

    fn div(self, other: Self) -> Self::Output {
        LazyExpr::Div(Box::new(self), Box::new(other))
    }
}

impl Neg for LazyExpr {
    type Output = LazyExpr;

    fn neg(self) -> Self::Output {
        LazyExpr::Negate(Box::new(self))
    }
}

// ============================================================================
// TRANSCENDENTAL OPERATIONS
// ============================================================================

impl LazyExpr {
    /// Exponential function (e^x)
    ///
    /// **ARCHITECTURE**: Lazy evaluation - builds expression tree without computation
    /// **EVALUATION**: Deferred until display or explicit evaluation
    /// **PRECISION**: Profile-aware tier selection (Q64.64, Q128.128, or Q256.256)
    pub fn exp(self) -> Self {
        LazyExpr::Exp(Box::new(self))
    }

    /// Natural logarithm function (ln(x))
    ///
    /// **ARCHITECTURE**: Lazy evaluation - builds expression tree without computation
    /// **EVALUATION**: Deferred until display or explicit evaluation
    /// **PRECISION**: Profile-aware tier selection (Q64.64, Q128.128, or Q256.256)
    /// **DOMAIN**: x > 0 (returns error for x <= 0)
    pub fn ln(self) -> Self {
        LazyExpr::Ln(Box::new(self))
    }

    /// Square root function (sqrt(x))
    ///
    /// **ALGORITHM**: Newton-Raphson at tier N+1 for full precision
    /// **DOMAIN**: x >= 0 (returns error for x < 0)
    pub fn sqrt(self) -> Self {
        LazyExpr::Sqrt(Box::new(self))
    }

    // Hyperbolic functions — FASC-composed from exp/ln/sqrt

    /// Hyperbolic sine: sinh(x) = (exp(x) - exp(-x)) / 2
    pub fn sinh(self) -> Self {
        LazyExpr::Sinh(Box::new(self))
    }

    /// Hyperbolic cosine: cosh(x) = (exp(x) + exp(-x)) / 2
    pub fn cosh(self) -> Self {
        LazyExpr::Cosh(Box::new(self))
    }

    /// Hyperbolic tangent: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    pub fn tanh(self) -> Self {
        LazyExpr::Tanh(Box::new(self))
    }

    /// Inverse hyperbolic sine: asinh(x) = ln(x + sqrt(x² + 1))
    pub fn asinh(self) -> Self {
        LazyExpr::Asinh(Box::new(self))
    }

    /// Inverse hyperbolic cosine: acosh(x) = ln(x + sqrt(x² - 1))
    /// **DOMAIN**: x >= 1
    pub fn acosh(self) -> Self {
        LazyExpr::Acosh(Box::new(self))
    }

    /// Inverse hyperbolic tangent: atanh(x) = ln((1+x)/(1-x)) / 2
    /// **DOMAIN**: |x| < 1
    pub fn atanh(self) -> Self {
        LazyExpr::Atanh(Box::new(self))
    }

    // Trigonometric functions

    /// Sine function (dedicated binary-level implementation)
    pub fn sin(self) -> Self {
        LazyExpr::Sin(Box::new(self))
    }

    /// Cosine function (dedicated binary-level implementation)
    pub fn cos(self) -> Self {
        LazyExpr::Cos(Box::new(self))
    }

    /// Tangent function: tan(x) = sin(x) / cos(x) (FASC-composed)
    pub fn tan(self) -> Self {
        LazyExpr::Tan(Box::new(self))
    }

    // Inverse trigonometric functions

    /// Arcsine: asin(x) = atan(x / sqrt(1 - x²))
    /// **DOMAIN**: |x| <= 1
    pub fn asin(self) -> Self {
        LazyExpr::Asin(Box::new(self))
    }

    /// Arccosine: acos(x) = π/2 - asin(x)
    /// **DOMAIN**: |x| <= 1
    pub fn acos(self) -> Self {
        LazyExpr::Acos(Box::new(self))
    }

    /// Arctangent (dedicated binary-level implementation)
    pub fn atan(self) -> Self {
        LazyExpr::Atan(Box::new(self))
    }

    /// Two-argument arctangent: atan2(y, x)
    pub fn atan2(self, x: LazyExpr) -> LazyExpr {
        LazyExpr::Atan2(Box::new(self), Box::new(x))
    }
}

// ============================================================================
// DISPLAY TRAIT - TRIGGERS LAZY EVALUATION
// ============================================================================

impl Display for LazyExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display shows expression structure; evaluation is handled by StackEvaluator
        match self {
            LazyExpr::Literal(s) => write!(f, "{}", s),
            LazyExpr::Value(v) => write!(f, "{}", v),
            LazyExpr::Constant(c) => write!(f, "{:?}", c),
            LazyExpr::Variable(r) => write!(f, "var[{}]", r.0),
            LazyExpr::Negate(inner) => write!(f, "-({})", inner),
            LazyExpr::Add(l, r) => write!(f, "({} + {})", l, r),
            LazyExpr::Sub(l, r) => write!(f, "({} - {})", l, r),
            LazyExpr::Mul(l, r) => write!(f, "({} * {})", l, r),
            LazyExpr::Div(l, r) => write!(f, "({} / {})", l, r),
            LazyExpr::Exp(inner) => write!(f, "exp({})", inner),
            LazyExpr::Ln(inner) => write!(f, "ln({})", inner),
            LazyExpr::Sqrt(inner) => write!(f, "sqrt({})", inner),
            LazyExpr::Pow(base, exp) => write!(f, "pow({}, {})", base, exp),
            LazyExpr::Sinh(inner) => write!(f, "sinh({})", inner),
            LazyExpr::Cosh(inner) => write!(f, "cosh({})", inner),
            LazyExpr::Tanh(inner) => write!(f, "tanh({})", inner),
            LazyExpr::Asinh(inner) => write!(f, "asinh({})", inner),
            LazyExpr::Acosh(inner) => write!(f, "acosh({})", inner),
            LazyExpr::Atanh(inner) => write!(f, "atanh({})", inner),
            LazyExpr::Sin(inner) => write!(f, "sin({})", inner),
            LazyExpr::Cos(inner) => write!(f, "cos({})", inner),
            LazyExpr::Tan(inner) => write!(f, "tan({})", inner),
            LazyExpr::Asin(inner) => write!(f, "asin({})", inner),
            LazyExpr::Acos(inner) => write!(f, "acos({})", inner),
            LazyExpr::Atan(inner) => write!(f, "atan({})", inner),
            LazyExpr::Atan2(y, x) => write!(f, "atan2({}, {})", y, x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_building() {
        let expr = gmath("10") + gmath("20") * gmath("30");
        assert_eq!(expr.depth(), 3);
        assert_eq!(expr.operation_count(), 2);
    }

    #[test]
    fn test_constant_detection() {
        let const_expr = gmath("10") + gmath("20");
        assert!(const_expr.is_constant());

        let var_expr = LazyExpr::Variable(StackRef(0)) + gmath("10");
        assert!(!var_expr.is_constant());
    }
}