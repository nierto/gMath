//! Trit Packing Utilities — Public API for Balanced Ternary Weight Storage
//!
//! **ENCODING**: 5 trits per byte using base-3 packing (3^5 = 243 ≤ 255)
//! **EFFICIENCY**: 1.6 bits per trit (vs theoretical minimum 1.585 bits)
//!
//! Trit values are balanced ternary: {-1, 0, +1}
//! Stored as {0, 1, 2} in the packing (offset by +1 for unsigned encoding)
//!
//! This module provides the canonical g_math trit packing format.

use crate::fixed_point::core_types::errors::OverflowDetected;

/// A balanced ternary digit: -1, 0, or +1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum Trit {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Trit {
    /// Convert from i8. Returns Err if not in {-1, 0, 1}.
    #[inline]
    pub fn from_i8(val: i8) -> Result<Self, OverflowDetected> {
        match val {
            -1 => Ok(Trit::Neg),
            0 => Ok(Trit::Zero),
            1 => Ok(Trit::Pos),
            _ => Err(OverflowDetected::InvalidInput),
        }
    }

    /// Convert to i8
    #[inline]
    pub const fn as_i8(self) -> i8 {
        self as i8
    }

    /// Convert to unsigned encoding for packing: {-1, 0, +1} → {0, 1, 2}
    #[inline]
    const fn to_packed(self) -> u8 {
        (self as i8 + 1) as u8
    }

    /// Convert from unsigned encoding: {0, 1, 2} → {-1, 0, +1}
    #[inline]
    fn from_packed(val: u8) -> Result<Self, OverflowDetected> {
        match val {
            0 => Ok(Trit::Neg),
            1 => Ok(Trit::Zero),
            2 => Ok(Trit::Pos),
            _ => Err(OverflowDetected::InvalidInput),
        }
    }
}

/// Pack trits into bytes, 5 trits per byte using base-3 encoding.
///
/// Each byte stores: `d[0]*81 + d[1]*27 + d[2]*9 + d[3]*3 + d[4]`
/// where `d[i] ∈ {0, 1, 2}` (trit value + 1).
///
/// The last byte may contain fewer than 5 trits; unused positions are filled with 0 (Zero trit).
///
/// # Encoding
/// - 5 trits per byte: 3^5 = 243 ≤ 255 (fits in u8)
/// - Storage: ceil(count / 5) bytes
/// - Efficiency: 1.6 bits per trit
pub fn pack_trits(trits: &[Trit]) -> Vec<u8> {
    let num_bytes = (trits.len() + 4) / 5;
    let mut packed = Vec::with_capacity(num_bytes);

    let mut i = 0;
    while i < trits.len() {
        let mut byte: u8 = 0;
        // Pack up to 5 trits into one byte: d0*81 + d1*27 + d2*9 + d3*3 + d4
        for j in 0..5 {
            byte *= 3;
            if i + j < trits.len() {
                byte += trits[i + j].to_packed();
            } else {
                byte += Trit::Zero.to_packed(); // pad with zero
            }
        }
        packed.push(byte);
        i += 5;
    }

    packed
}

/// Unpack bytes back to trits.
///
/// `count` is the exact number of trits to extract (important for the last byte
/// which may have been zero-padded during packing).
pub fn unpack_trits(data: &[u8], count: usize) -> Result<Vec<Trit>, OverflowDetected> {
    let mut trits = Vec::with_capacity(count);
    let mut extracted = 0;

    for &byte in data {
        if extracted >= count {
            break;
        }

        // Extract 5 trits from byte (most-significant first)
        let mut remaining = byte;
        let mut chunk = [Trit::Zero; 5];
        for j in (0..5).rev() {
            let d = remaining % 3;
            remaining /= 3;
            chunk[j] = Trit::from_packed(d)?;
        }

        for j in 0..5 {
            if extracted >= count {
                break;
            }
            trits.push(chunk[j]);
            extracted += 1;
        }
    }

    Ok(trits)
}

// quantize_to_trits is profile-dependent and will be added
// alongside the profile infrastructure in Phase 1 (Q32.32) and Phase 2 (Q16.16).
// For now, consumers can implement threshold-based quantization using
// pack_trits() + their own threshold logic.

// ════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let trits = vec![Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos, Trit::Neg];
        let packed = pack_trits(&trits);
        assert_eq!(packed.len(), 1); // 5 trits = 1 byte
        let unpacked = unpack_trits(&packed, 5).unwrap();
        assert_eq!(trits, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_partial() {
        // 7 trits → 2 bytes (5 + 2, padded)
        let trits = vec![
            Trit::Pos, Trit::Zero, Trit::Neg, Trit::Pos, Trit::Zero,
            Trit::Neg, Trit::Pos,
        ];
        let packed = pack_trits(&trits);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_trits(&packed, 7).unwrap();
        assert_eq!(trits, unpacked);
    }

    #[test]
    fn test_pack_all_zeros() {
        let trits = vec![Trit::Zero; 10];
        let packed = pack_trits(&trits);
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_trits(&packed, 10).unwrap();
        assert_eq!(trits, unpacked);
    }

    #[test]
    fn test_pack_all_pos() {
        let trits = vec![Trit::Pos; 5];
        let packed = pack_trits(&trits);
        // All +1 → packed as 2: 2*81 + 2*27 + 2*9 + 2*3 + 2 = 242
        assert_eq!(packed[0], 242);
        let unpacked = unpack_trits(&packed, 5).unwrap();
        assert_eq!(trits, unpacked);
    }

    #[test]
    fn test_pack_all_neg() {
        let trits = vec![Trit::Neg; 5];
        let packed = pack_trits(&trits);
        // All -1 → packed as 0: 0*81 + 0*27 + 0*9 + 0*3 + 0 = 0
        assert_eq!(packed[0], 0);
        let unpacked = unpack_trits(&packed, 5).unwrap();
        assert_eq!(trits, unpacked);
    }

    #[test]
    fn test_encoding_fits_in_u8() {
        // Max encoding: all +1 → 242 ≤ 255
        // 3^5 - 1 = 242, which is the maximum packed byte value
        assert!(242u8 <= u8::MAX);
        // Verify 3^5 = 243
        assert_eq!(3u32.pow(5), 243);
    }

    #[test]
    fn test_empty() {
        let trits: Vec<Trit> = vec![];
        let packed = pack_trits(&trits);
        assert!(packed.is_empty());
        let unpacked = unpack_trits(&packed, 0).unwrap();
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_single_trit() {
        for trit in [Trit::Neg, Trit::Zero, Trit::Pos] {
            let packed = pack_trits(&[trit]);
            let unpacked = unpack_trits(&packed, 1).unwrap();
            assert_eq!(unpacked[0], trit);
        }
    }

    #[test]
    fn test_large_roundtrip() {
        // 1000 trits → 200 bytes
        let mut trits = Vec::with_capacity(1000);
        for i in 0..1000 {
            trits.push(match i % 3 {
                0 => Trit::Neg,
                1 => Trit::Zero,
                _ => Trit::Pos,
            });
        }
        let packed = pack_trits(&trits);
        assert_eq!(packed.len(), 200);
        let unpacked = unpack_trits(&packed, 1000).unwrap();
        assert_eq!(trits, unpacked);
    }

    #[test]
    fn test_trit_conversions() {
        assert_eq!(Trit::from_i8(-1).unwrap(), Trit::Neg);
        assert_eq!(Trit::from_i8(0).unwrap(), Trit::Zero);
        assert_eq!(Trit::from_i8(1).unwrap(), Trit::Pos);
        assert!(Trit::from_i8(2).is_err());
        assert!(Trit::from_i8(-2).is_err());
    }
}
