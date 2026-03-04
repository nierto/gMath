//! Precision Management Types
//!
//! Precision tracking for shadow materialization decisions.
//! PrecisionTier is used by ShadowMetadata to determine shadow strategy.

/// Precision tier classification for shadow materialization
///
/// Tracks precision loss levels to determine appropriate shadow strategy:
/// - Exact: No precision loss (shadow unnecessary)
/// - Minor: <1 ULP loss (compressed shadow sufficient)
/// - Moderate: 1-10 ULP loss (materialized shadow recommended)
/// - Significant: >10 ULP loss (shared shadow cache for efficiency)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PrecisionTier {
    Exact = 0,       // No precision loss (shadow unnecessary)
    Minor = 1,       // <1 ULP loss (compressed shadow)
    Moderate = 2,    // 1-10 ULP loss (materialized shadow)
    Significant = 3, // >10 ULP loss (shared shadow cache)
}

