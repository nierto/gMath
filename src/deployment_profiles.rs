//! Deployment profile definitions for the FASC+UGOD multi-domain architecture

/// Deployment profiles with precise precision limits
///
/// Profiles map directly to Q-format tiers and build-time table generation.
/// Selected via `GMATH_PROFILE` env var at build time (processed by build.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentProfile {
    /// Realtime: Q16.16, i32 storage, i64 compute, 4 decimal places
    /// PRECISION: 4 decimal places | TIER: 1 (i16/u16)
    Realtime,

    /// Compact/Fast: Q32.32, i64 storage, i128 compute, 9 decimal places
    /// PRECISION: 9 decimal places | TIER: 2 (i32/u32)
    Compact,

    /// Embedded systems: Q64.64, scalar-only, minimal memory
    /// PRECISION: 19 decimal places | TIER: 4 (i64/u64)
    Embedded,

    /// General purpose: Q128.128, balanced precision/performance
    /// PRECISION: 38 decimal places | TIER: 5 (i128/u128)
    Balanced,

    /// Scientific computing: Q256.256, maximum precision
    /// PRECISION: 77 decimal places | TIER: 6 (I256/I256)
    Scientific,

    /// Research/Custom: ENV-configured precision, automatic profile fallback
    Custom,
}

impl Default for DeploymentProfile {
    fn default() -> Self {
        DeploymentProfile::Balanced
    }
}

#[cfg(test)]
mod tests;
