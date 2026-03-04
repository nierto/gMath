//! UniversalTieredArithmetic: Cross-Domain Tiered Overflow Coordination
//!
//! **MISSION**: Unified trait interface for tiered overflow delegation across all precision domains
//! **ARCHITECTURE**: Universal 6-tier system aligned across all 4 domains
//! **COORDINATION**: Automatic tier promotion with rational infinite precision fallback
//!
//! ## Universal 6-Tier System (all domains aligned)
//!
//! | Tier | Bits | Backing | Binary    | Decimal   | Ternary     | Symbolic    |
//! |------|------|---------|-----------|-----------|-------------|-------------|
//! | 1    | 32   | i32     | Q16.16    | D16.16    | TQ8.8       | i16/u16     |
//! | 2    | 64   | i64     | Q32.32    | D32.32    | TQ16.16     | i32/u32     |
//! | 3    | 128  | i128    | Q64.64    | D64.64    | TQ32.32     | i64/u64     |
//! | 4    | 256  | I256    | Q128.128  | D128.128  | TQ64.64     | i128/u128   |
//! | 5    | 512  | I512    | Q256.256  | D256.256  | TQ128.128   | I256/I256   |
//! | 6    | 1024 | I1024   | Q512.512  | D512.512  | TQ256.256   | I512/I512   |
//!
//! ## Overflow Behavior
//!
//! - Tiers 1-5: Overflow → promote to next tier (UGOD)
//! - Tier 6: Overflow → rational fallback (infinite precision)
//!
//! ## Implementation Status
//!
//! - Binary: `UniversalBinaryFixed` (binary_types.rs) — full 6-tier UGOD
//! - Decimal: `UniversalDecimalTiered` (decimal_types.rs) — full 6-tier UGOD
//! - Ternary: `UniversalTernaryFixed` (ternary_types.rs) — full 6-tier UGOD
//! - Symbolic: `RationalNumber` (rational_number.rs) — 6+1 tiers (BigInt gated)
//! - StackEvaluator: typed dispatch through all domain tier systems
//! - Shadow: `CompactShadow` propagation through all arithmetic ops

use crate::fixed_point::domains::symbolic::rational::rational_number::OverflowDetected;
use std::fmt::Debug;

// DomainType is canonically defined in core_types::domain_metadata (with Reserved4-7 for future domains).
// Re-exported here for convenience with existing `ugod::DomainType` imports.
pub use crate::fixed_point::core_types::domain_metadata::DomainType;

/// Universal trait for tiered arithmetic operations across all precision domains
/// 
/// **ARCHITECTURE**: Each domain implements this trait with its own tier mapping strategy
/// **COORDINATION**: Enables cross-domain operations and universal overflow delegation
/// **FALLBACK**: Rational infinite precision provides guaranteed computation success
pub trait UniversalTieredArithmetic: Clone + Sized + Debug {
    /// Error type for overflow detection (standardized to OverflowDetected)
    type Error: From<OverflowDetected> + Debug;
    
    // ============================================================================
    // FUNDAMENTAL ARITHMETIC OPERATIONS WITH OVERFLOW DETECTION
    // ============================================================================
    
    /// Addition with overflow detection and tier promotion
    fn try_add(&self, other: &Self) -> Result<Self, Self::Error>;
    
    /// Subtraction with overflow detection and tier promotion
    fn try_subtract(&self, other: &Self) -> Result<Self, Self::Error>;
    
    /// Multiplication with overflow detection and tier promotion
    fn try_multiply(&self, other: &Self) -> Result<Self, Self::Error>;
    
    /// Division with overflow detection and tier promotion
    fn try_divide(&self, other: &Self) -> Result<Self, Self::Error>;
    
    /// Negation with overflow detection and tier promotion
    fn try_negate(&self) -> Result<Self, Self::Error>;
    
    // ============================================================================
    // TIER MANAGEMENT AND PROMOTION INFRASTRUCTURE
    // ============================================================================
    
    /// Get current tier level (domain-specific mapping)
    fn current_tier(&self) -> u8;
    
    /// Check if value can be promoted to target tier within deployment profile limits
    fn can_promote_to_tier(&self, tier: u8) -> bool;
    
    /// Promote to target tier (returns None if impossible)
    fn promote_to_tier(&self, tier: u8) -> Option<Self>;
    
    /// Get maximum tier supported by this domain
    fn max_tier() -> u8 where Self: Sized;
    
    /// Get domain type for universal coordination
    fn domain_type() -> DomainType where Self: Sized;
    
    // ============================================================================
    // UNIVERSAL COORDINATION HELPERS
    // ============================================================================
    
    /// Check if this value can accommodate operations with another domain's tier
    fn can_accommodate_symbolic_tier(&self, symbolic_tier: u8) -> bool {
        // Default implementation: check if current tier can handle the symbolic tier
        self.current_tier() >= Self::symbolic_tier_mapping(symbolic_tier)
    }
    
    /// Map symbolic tier to this domain's tier (domain-specific implementation)
    fn symbolic_tier_mapping(symbolic_tier: u8) -> u8 where Self: Sized {
        // Default: 1:1 mapping, domains should override this
        symbolic_tier
    }
    
    /// Get deployment profile maximum tier limit
    fn max_tier_for_profile(profile: DeploymentProfile) -> u8 where Self: Sized {
        match profile {
            DeploymentProfile::Embedded => 4,        // Conservative limits
            DeploymentProfile::Balanced => 5,        // Balanced precision/performance
            DeploymentProfile::Scientific => 6,      // High precision for research
            DeploymentProfile::Custom => 7,          // Full capability for custom profiles
        }
    }
}

/// Import deployment profile
use crate::deployment_profiles::DeploymentProfile;

// ============================================================================
// DOMAIN IMPLEMENTATIONS
// ============================================================================
//
// Each domain has its own typed tier system with UGOD overflow promotion:
//   - Binary:   UniversalBinaryFixed  (binary_types.rs)  — add/sub/mul/neg
//   - Decimal:  UniversalDecimalTiered (decimal_types.rs) — add/sub/neg
//   - Ternary:  UniversalTernaryFixed  (ternary_types.rs) — add/sub/mul/div/neg
//   - Symbolic: RationalNumber         (rational_number.rs) — try_add/sub/mul/div/neg
//
// The StackEvaluator (stack_evaluator.rs) dispatches through these typed systems
// via binary_from_storage()/decimal_from_storage()/ternary_from_storage() bridges.
// Cross-domain operations fall back to rational arithmetic for exactness.