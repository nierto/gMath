//! Shadow Types — Relocated from router/shadow/ for hotpath access
//!
//! These types are used by RationalNumber and NumberClassifier (both on the
//! canonical FASC hotpath). The rest of the shadow system (decision engine,
//! preservation, materialization) has been archived.

/// Domain classification for precision routing (future-proof for 8 domains)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DomainType {
    /// Binary fixed-point (Q64.64)
    Binary = 0,
    /// Decimal fixed-point (DecimalFixed<N>)
    Decimal = 1,
    /// Balanced ternary (geometric operations)
    Ternary = 2,
    /// Symbolic rational (exact a/b)
    Symbolic = 3,
    /// Future domain 4
    Reserved4 = 4,
    /// Future domain 5
    Reserved5 = 5,
    /// Future domain 6
    Reserved6 = 6,
    /// Future domain 7
    Reserved7 = 7,
}

impl DomainType {
    /// Create from packed bits (for metadata extraction)
    pub fn from_bits(bits: u32) -> Self {
        match bits & 0x7 {
            0 => Self::Binary,
            1 => Self::Decimal,
            2 => Self::Ternary,
            3 => Self::Symbolic,
            4 => Self::Reserved4,
            5 => Self::Reserved5,
            6 => Self::Reserved6,
            _ => Self::Reserved7,
        }
    }
}

/// Type of exactness (implies shadow strategy)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExactnessType {
    /// Exactly representable, no shadow needed
    Exact = 0,
    /// Inexact in domain, permanent shadow needed
    Inexact = 1,
    /// Mathematical constant, permanent shadow with pattern
    MathConstant = 2,
    /// Irrational, permanent shadow required
    Irrational = 3,
    /// Exact but in computation with inexact (temporary shadow)
    ExactWithInexact = 4,
    /// Repeating decimal pattern
    RepeatingDecimal = 5,
    /// Reserved for future
    Reserved6 = 6,
    Reserved7 = 7,
}

/// Ultra-compact shadow metadata (4 bytes)
///
/// BIT ALLOCATION (32 bits total):
/// - [0-2]:   DomainType (3 bits = 8 domains)
/// - [3-5]:   ExactnessType (3 bits = 8 types)
/// - [6-8]:   TierLevel (3 bits = 8 tiers, 0-7)
/// - [9-11]:  DecimalPlaces (3 bits)
/// - [12-19]: ConfidenceScore (8 bits = 0-255)
/// - [20-27]: PatternId (8 bits = 256 patterns)
/// - [28-31]: Reserved (4 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShadowMetadata {
    packed: u32,
}

impl ShadowMetadata {
    /// Create new metadata
    pub fn new(domain: DomainType, exactness: ExactnessType) -> Self {
        let mut packed = 0u32;
        packed |= (domain as u32) & 0x7;
        packed |= ((exactness as u32) & 0x7) << 3;
        packed |= 128u32 << 12; // Default confidence 50%
        Self { packed }
    }

    fn exactness_type(&self) -> ExactnessType {
        match (self.packed >> 3) & 0x7 {
            0 => ExactnessType::Exact,
            1 => ExactnessType::Inexact,
            2 => ExactnessType::MathConstant,
            3 => ExactnessType::Irrational,
            4 => ExactnessType::ExactWithInexact,
            5 => ExactnessType::RepeatingDecimal,
            6 => ExactnessType::Reserved6,
            _ => ExactnessType::Reserved7,
        }
    }

    /// Map ExactnessType to PrecisionTier for compatibility
    pub fn precision_tier(&self) -> crate::fixed_point::core_types::precision::PrecisionTier {
        use crate::fixed_point::core_types::precision::PrecisionTier;
        match self.exactness_type() {
            ExactnessType::Exact => PrecisionTier::Exact,
            ExactnessType::ExactWithInexact | ExactnessType::RepeatingDecimal => PrecisionTier::Minor,
            ExactnessType::Inexact => PrecisionTier::Moderate,
            _ => PrecisionTier::Significant,
        }
    }
}
