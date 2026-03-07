//! Error Types for Arithmetic Operations
//!
//! Unified error type for all domains (Binary, Decimal, Symbolic, Ternary).
//! Enables graceful overflow handling through UGOD
//! (Universal Graceful Overflow Delegation) architecture.

/// Canonical overflow detection result for all arithmetic domains
///
/// **UNIFIED**: Single error type shared across Binary, Decimal, Symbolic, and Ternary domains
///
/// Indicates why an arithmetic operation cannot proceed in current tier/domain:
/// - **TierOverflow**: Result exceeds current tier capacity (promote to higher tier)
/// - **PrecisionLimit**: Result exceeds deployment profile maximum precision
/// - **PrecisionLoss**: Intermediate calculation would lose unacceptable precision
/// - **DivisionByZero**: Division by zero detected
/// - **InvalidInput**: Invalid input format or value
/// - **ParseError**: Parsing error (string to number conversion failed)
/// - **Overflow**: Generic overflow (fallback for non-tiered contexts)
/// - **StackUnderflow**: Stack underflow (FASC VM - pop from empty stack)
/// - **StackCorruption**: Stack corruption detected (FASC VM - invalid state)
/// - **InvalidStackReference**: Invalid stack reference (FASC VM - out of bounds)
/// - **DomainError**: Domain error (e.g., ln(x) for x <= 0, sqrt(x) for x < 0)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowDetected {
    /// Result would exceed current tier capacity
    TierOverflow,
    /// Result would exceed deployment profile precision limit
    PrecisionLimit,
    /// Intermediate calculation would lose precision
    PrecisionLoss,
    /// Division by zero detected
    DivisionByZero,
    /// Invalid input format or value
    InvalidInput,
    /// Parsing error (string to number conversion failed)
    ParseError,
    /// Generic overflow (fallback for non-tiered contexts)
    Overflow,
    /// Stack underflow (FASC VM - pop from empty stack)
    StackUnderflow,
    /// Stack corruption detected (FASC VM - invalid state)
    StackCorruption,
    /// Invalid stack reference (FASC VM - out of bounds)
    InvalidStackReference,
    /// Domain error (e.g., ln(x) for x <= 0, sqrt(x) for x < 0)
    DomainError,
}
