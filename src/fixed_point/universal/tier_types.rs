//! Universal Tier Types for Unified Domain Architecture
//!
//! **MISSION**: Single type system shared by all domains (binary, decimal, ternary, symbolic)
//! **ARCHITECTURE**: 6-tier hierarchy aligned across all domains, with optional tier 7 BigInt
//!
//! | Tier | Bits | Backing | Binary    | Decimal   | Ternary   | Symbolic  |
//! |------|------|---------|-----------|-----------|-----------|-----------|
//! | 1    | 32   | i32     | Q16.16    | D16.16    | TQ8.8     | i16/u16   |
//! | 2    | 64   | i64     | Q32.32    | D32.32    | TQ16.16   | i32/u32   |
//! | 3    | 128  | i128    | Q64.64    | D64.64    | TQ32.32   | i64/u64   |
//! | 4    | 256  | I256    | Q128.128  | D128.128  | TQ64.64   | i128/u128 |
//! | 5    | 512  | I512    | Q256.256  | D256.256  | TQ128.128 | I256/I256 |
//! | 6    | 1024 | I1024   | Q512.512  | D512.512  | TQ256.256 | I512/I512 |

// ============================================================================
// COMPACT SHADOW
// ============================================================================

/// Stack-allocated shadow for exactness preservation.
///
/// **PURPOSE**: Preserves the exact rational value alongside a domain-specific
/// approximation. When 0.5 is stored as binary Q64.64, the shadow keeps "1/2"
/// so cross-domain operations remain exact.
///
/// **MEMORY**: 0-32 bytes, entirely stack-allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactShadow {
    /// No shadow needed (value is exact in its domain)
    None,
    /// 2 bytes: covers 1/2, 1/3, 3/4, etc.
    Tiny(i8, u8),
    /// 4 bytes: covers 355/113 (pi approx), etc.
    Small(i16, u16),
    /// 8 bytes: larger fractions
    Medium(i32, u32),
    /// 16 bytes: still stack-allocated
    Large(i64, u64),
    /// 32 bytes: max stack shadow
    Full(i128, u128),
    /// 1 byte: known mathematical constant (pi, e, sqrt2, phi, gamma)
    ConstantRef(ShadowConstantId),
}

/// Identifier for well-known mathematical constants in shadows.
///
/// When a value is a mathematical constant, we store just the ID rather
/// than a rational approximation — the constant tables provide full precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShadowConstantId {
    Pi = 0,
    E = 1,
    Sqrt2 = 2,
    Phi = 3,        // Golden ratio
    Ln2 = 4,
    Ln10 = 5,
    EulerGamma = 6,
}

impl CompactShadow {
    /// Check if this shadow is empty
    #[inline]
    pub fn is_none(&self) -> bool {
        matches!(self, CompactShadow::None)
    }

    /// Check if this shadow carries a value
    #[inline]
    pub fn is_some(&self) -> bool {
        !self.is_none()
    }

    /// Create the smallest shadow that can represent num/den
    pub fn from_rational(num: i128, den: u128) -> Self {
        if den == 0 {
            return CompactShadow::None;
        }
        if den == 1 && num == 0 {
            return CompactShadow::None; // zero needs no shadow
        }

        // Try to fit in the smallest tier
        if let (Ok(n), Ok(d)) = (i8::try_from(num), u8::try_from(den)) {
            return CompactShadow::Tiny(n, d);
        }
        if let (Ok(n), Ok(d)) = (i16::try_from(num), u16::try_from(den)) {
            return CompactShadow::Small(n, d);
        }
        if let (Ok(n), Ok(d)) = (i32::try_from(num), u32::try_from(den)) {
            return CompactShadow::Medium(n, d);
        }
        if let (Ok(n), Ok(d)) = (i64::try_from(num), u64::try_from(den)) {
            return CompactShadow::Large(n, d);
        }
        CompactShadow::Full(num, den)
    }

    /// Extract as (numerator, denominator) if this is a rational shadow
    pub fn as_rational(&self) -> Option<(i128, u128)> {
        match self {
            CompactShadow::None => None,
            CompactShadow::Tiny(n, d) => Some((*n as i128, *d as u128)),
            CompactShadow::Small(n, d) => Some((*n as i128, *d as u128)),
            CompactShadow::Medium(n, d) => Some((*n as i128, *d as u128)),
            CompactShadow::Large(n, d) => Some((*n as i128, *d as u128)),
            CompactShadow::Full(n, d) => Some((*n, *d)),
            CompactShadow::ConstantRef(_) => None,
        }
    }

    /// Memory footprint in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            CompactShadow::None => 0,
            CompactShadow::Tiny(_, _) => 2,
            CompactShadow::Small(_, _) => 4,
            CompactShadow::Medium(_, _) => 8,
            CompactShadow::Large(_, _) => 16,
            CompactShadow::Full(_, _) => 32,
            CompactShadow::ConstantRef(_) => 1,
        }
    }

    /// Get the numerator (None for None/ConstantRef)
    pub fn numerator(&self) -> Option<i128> {
        self.as_rational().map(|(n, _)| n)
    }

    /// Get the denominator (None for None/ConstantRef)
    pub fn denominator(&self) -> Option<u128> {
        self.as_rational().map(|(_, d)| d)
    }

    /// Get the constant ID if this is a ConstantRef
    pub fn constant_id(&self) -> Option<ShadowConstantId> {
        match self {
            CompactShadow::ConstantRef(id) => Some(*id),
            _ => None,
        }
    }
}

impl core::fmt::Display for CompactShadow {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CompactShadow::None => write!(f, "none"),
            CompactShadow::ConstantRef(c) => write!(f, "{}", c),
            other => {
                // All rational variants — as_rational() always succeeds for non-None/non-ConstantRef
                match other.as_rational() {
                    Some((num, den)) if den == 1 => write!(f, "{}", num),
                    Some((num, den)) => write!(f, "{}/{}", num, den),
                    None => write!(f, "none"),
                }
            }
        }
    }
}

impl core::fmt::Display for ShadowConstantId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShadowConstantId::Pi => write!(f, "\u{03C0}"),
            ShadowConstantId::E => write!(f, "e"),
            ShadowConstantId::Sqrt2 => write!(f, "\u{221A}2"),
            ShadowConstantId::Phi => write!(f, "\u{03C6}"),
            ShadowConstantId::Ln2 => write!(f, "ln2"),
            ShadowConstantId::Ln10 => write!(f, "ln10"),
            ShadowConstantId::EulerGamma => write!(f, "\u{03B3}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_shadow_auto_tiering() {
        // 1/3 fits in Tiny
        let s = CompactShadow::from_rational(1, 3);
        assert!(matches!(s, CompactShadow::Tiny(1, 3)));
        assert_eq!(s.size_bytes(), 2);

        // 355/113 fits in Small
        let s = CompactShadow::from_rational(355, 113);
        assert!(matches!(s, CompactShadow::Small(355, 113)));
        assert_eq!(s.size_bytes(), 4);

        // Large values go to appropriate tiers
        let s = CompactShadow::from_rational(100_000, 7);
        assert!(matches!(s, CompactShadow::Medium(100_000, 7)));
    }

    #[test]
    fn compact_shadow_roundtrip() {
        let s = CompactShadow::from_rational(1, 3);
        assert_eq!(s.as_rational(), Some((1, 3)));

        let s = CompactShadow::ConstantRef(ShadowConstantId::Pi);
        assert_eq!(s.as_rational(), None);
    }
}
