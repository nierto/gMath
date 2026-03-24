//! Binary serialization for FixedPoint, FixedVector, FixedMatrix, Tensor,
//! and ManifoldPoint.
//!
//! Wire format: big-endian byte order with a 1-byte profile tag.
//!
//! ## FixedPoint format
//! `[u8 profile_tag][raw bytes in big-endian]`
//!
//! Profile tags:
//! - 0x01: Q64.64 (16 bytes payload)
//! - 0x02: Q128.128 (32 bytes payload)
//! - 0x03: Q256.256 (64 bytes payload)
//!
//! ## FixedVector format
//! `[u32 len (big-endian)][FixedPoint × len]` (without per-element profile tags)
//!
//! ## FixedMatrix format
//! `[u32 rows (big-endian)][u32 cols (big-endian)][FixedPoint × rows*cols]`
//!
//! ## Tensor format
//! `[u8 rank][u32 × rank (shape dims)][FixedPoint × product(shape)]`
//!
//! ## ManifoldPoint format
//! `[u8 manifold_tag][u32 param...][FixedVector coordinate_data]`
//!
//! Manifold tags:
//! - 0x01: EuclideanSpace(dim)
//! - 0x02: Sphere(dim)
//! - 0x03: HyperbolicSpace(dim)
//! - 0x04: SPDManifold(n)
//! - 0x05: Grassmannian(k, n)
//!
//! ## Compact encoding
//! `[u8 flags][payload]` where flags encode:
//! - bit 0: sign (0=positive, 1=negative)
//! - bit 1-2: size class (0=zero, 1=small, 2=medium, 3=full)
//! Small/medium elide trailing zero bytes in the fractional part.

use super::FixedPoint;
use super::FixedVector;
use super::FixedMatrix;
use super::tensor::Tensor;
use crate::fixed_point::core_types::errors::OverflowDetected;

#[cfg(table_format = "q64_64")]
const PROFILE_TAG: u8 = 0x01;
#[cfg(table_format = "q128_128")]
const PROFILE_TAG: u8 = 0x02;
#[cfg(table_format = "q256_256")]
const PROFILE_TAG: u8 = 0x03;
#[cfg(table_format = "q32_32")]
const PROFILE_TAG: u8 = 0x04;
#[cfg(table_format = "q16_16")]
const PROFILE_TAG: u8 = 0x05;

#[cfg(table_format = "q64_64")]
const RAW_BYTE_LEN: usize = 16;
#[cfg(table_format = "q128_128")]
const RAW_BYTE_LEN: usize = 32;
#[cfg(table_format = "q256_256")]
const RAW_BYTE_LEN: usize = 64;
#[cfg(table_format = "q32_32")]
const RAW_BYTE_LEN: usize = 8;
#[cfg(table_format = "q16_16")]
const RAW_BYTE_LEN: usize = 4;

// ============================================================================
// FixedPoint serialization
// ============================================================================

impl FixedPoint {
    /// Serialize to bytes with profile tag prefix (big-endian).
    ///
    /// Format: `[u8 profile_tag][raw bytes big-endian]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(1 + RAW_BYTE_LEN);
        bytes.push(PROFILE_TAG);
        bytes.extend_from_slice(&raw_to_be_bytes(self.raw()));
        bytes
    }

    /// Deserialize from bytes with profile tag prefix.
    ///
    /// Returns `Err(InvalidInput)` if the profile tag doesn't match or
    /// the byte slice is too short.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.len() < 1 + RAW_BYTE_LEN {
            return Err(OverflowDetected::InvalidInput);
        }
        if bytes[0] != PROFILE_TAG {
            return Err(OverflowDetected::InvalidInput);
        }
        Ok(Self::from_raw(be_bytes_to_raw(&bytes[1..1 + RAW_BYTE_LEN])))
    }

    /// Serialize raw storage only (no profile tag), big-endian.
    pub fn to_raw_bytes(&self) -> Vec<u8> {
        raw_to_be_bytes(self.raw())
    }

    /// Deserialize raw storage only (no profile tag), big-endian.
    pub fn from_raw_bytes(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.len() < RAW_BYTE_LEN {
            return Err(OverflowDetected::InvalidInput);
        }
        Ok(Self::from_raw(be_bytes_to_raw(&bytes[..RAW_BYTE_LEN])))
    }

    /// The profile tag byte for the current compilation profile.
    pub fn profile_tag() -> u8 {
        PROFILE_TAG
    }

    /// Size in bytes of the raw storage (without profile tag).
    pub fn raw_byte_len() -> usize {
        RAW_BYTE_LEN
    }
}

// ============================================================================
// FixedVector serialization
// ============================================================================

impl FixedVector {
    /// Serialize to bytes.
    ///
    /// Format: `[u32 len BE][raw bytes × len]` (no per-element profile tags).
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.len();
        let mut bytes = Vec::with_capacity(4 + n * RAW_BYTE_LEN);
        bytes.extend_from_slice(&(n as u32).to_be_bytes());
        for i in 0..n {
            bytes.extend_from_slice(&raw_to_be_bytes(self[i].raw()));
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.len() < 4 {
            return Err(OverflowDetected::InvalidInput);
        }
        let n = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let expected = 4 + n * RAW_BYTE_LEN;
        if bytes.len() < expected {
            return Err(OverflowDetected::InvalidInput);
        }
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let start = 4 + i * RAW_BYTE_LEN;
            let raw = be_bytes_to_raw(&bytes[start..start + RAW_BYTE_LEN]);
            data.push(FixedPoint::from_raw(raw));
        }
        Ok(FixedVector::from_slice(&data))
    }
}

// ============================================================================
// FixedMatrix serialization
// ============================================================================

impl FixedMatrix {
    /// Serialize to bytes.
    ///
    /// Format: `[u32 rows BE][u32 cols BE][raw bytes × rows*cols]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.rows() * self.cols();
        let mut bytes = Vec::with_capacity(8 + n * RAW_BYTE_LEN);
        bytes.extend_from_slice(&(self.rows() as u32).to_be_bytes());
        bytes.extend_from_slice(&(self.cols() as u32).to_be_bytes());
        for r in 0..self.rows() {
            for c in 0..self.cols() {
                bytes.extend_from_slice(&raw_to_be_bytes(self.get(r, c).raw()));
            }
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.len() < 8 {
            return Err(OverflowDetected::InvalidInput);
        }
        let rows = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let cols = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let n = rows * cols;
        let expected = 8 + n * RAW_BYTE_LEN;
        if bytes.len() < expected {
            return Err(OverflowDetected::InvalidInput);
        }
        let mut m = FixedMatrix::new(rows, cols);
        for i in 0..n {
            let start = 8 + i * RAW_BYTE_LEN;
            let raw = be_bytes_to_raw(&bytes[start..start + RAW_BYTE_LEN]);
            let r = i / cols;
            let c = i % cols;
            m.set(r, c, FixedPoint::from_raw(raw));
        }
        Ok(m)
    }
}

// ============================================================================
// Tensor serialization
// ============================================================================

impl Tensor {
    /// Serialize a tensor to bytes.
    ///
    /// Format: `[u8 rank][u32 × rank (shape dims BE)][raw bytes × product(shape)]`
    ///
    /// The rank byte limits tensors to 255 dimensions (more than enough for any
    /// practical use — Riemann curvature is rank 4, highest in differential geometry).
    pub fn to_bytes(&self) -> Vec<u8> {
        let rank = self.rank();
        let total = self.len();
        let mut bytes = Vec::with_capacity(1 + rank * 4 + total * RAW_BYTE_LEN);

        // Rank
        bytes.push(rank as u8);

        // Shape dimensions
        for &dim in self.shape() {
            bytes.extend_from_slice(&(dim as u32).to_be_bytes());
        }

        // Data (flat, row-major)
        for val in self.data() {
            bytes.extend_from_slice(&raw_to_be_bytes(val.raw()));
        }

        bytes
    }

    /// Deserialize a tensor from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.is_empty() {
            return Err(OverflowDetected::InvalidInput);
        }

        let rank = bytes[0] as usize;
        let header_len = 1 + rank * 4;
        if bytes.len() < header_len {
            return Err(OverflowDetected::InvalidInput);
        }

        // Read shape
        let mut shape = Vec::with_capacity(rank);
        for i in 0..rank {
            let offset = 1 + i * 4;
            let dim = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            shape.push(dim);
        }

        let total: usize = shape.iter().product::<usize>().max(1);
        let expected = header_len + total * RAW_BYTE_LEN;
        if bytes.len() < expected {
            return Err(OverflowDetected::InvalidInput);
        }

        // Read data
        let mut data = Vec::with_capacity(total);
        for i in 0..total {
            let start = header_len + i * RAW_BYTE_LEN;
            let raw = be_bytes_to_raw(&bytes[start..start + RAW_BYTE_LEN]);
            data.push(FixedPoint::from_raw(raw));
        }

        Ok(Tensor::from_data(&shape, &data))
    }
}

// ============================================================================
// ManifoldPoint serialization
// ============================================================================

/// Manifold type tags for serialization.
pub const MANIFOLD_TAG_EUCLIDEAN: u8 = 0x01;
pub const MANIFOLD_TAG_SPHERE: u8 = 0x02;
pub const MANIFOLD_TAG_HYPERBOLIC: u8 = 0x03;
pub const MANIFOLD_TAG_SPD: u8 = 0x04;
pub const MANIFOLD_TAG_GRASSMANNIAN: u8 = 0x05;

/// A serializable point on a manifold with its manifold type.
///
/// This bundles the manifold identity (type + parameters) with the
/// coordinate vector, so a receiver can reconstruct the full context.
#[derive(Clone, Debug)]
pub struct ManifoldPoint {
    /// Manifold type tag.
    pub manifold_tag: u8,
    /// Manifold parameters (interpretation depends on tag).
    /// - Euclidean: [dim]
    /// - Sphere: [dim]
    /// - Hyperbolic: [dim]
    /// - SPD: [n]
    /// - Grassmannian: [k, n]
    pub params: Vec<u32>,
    /// Coordinate vector on the manifold.
    pub coordinates: FixedVector,
}

impl ManifoldPoint {
    /// Create a ManifoldPoint for Euclidean space R^n.
    pub fn euclidean(dim: usize, coords: FixedVector) -> Self {
        Self {
            manifold_tag: MANIFOLD_TAG_EUCLIDEAN,
            params: vec![dim as u32],
            coordinates: coords,
        }
    }

    /// Create a ManifoldPoint for the n-sphere S^n.
    pub fn sphere(dim: usize, coords: FixedVector) -> Self {
        Self {
            manifold_tag: MANIFOLD_TAG_SPHERE,
            params: vec![dim as u32],
            coordinates: coords,
        }
    }

    /// Create a ManifoldPoint for hyperbolic space H^n.
    pub fn hyperbolic(dim: usize, coords: FixedVector) -> Self {
        Self {
            manifold_tag: MANIFOLD_TAG_HYPERBOLIC,
            params: vec![dim as u32],
            coordinates: coords,
        }
    }

    /// Create a ManifoldPoint for the SPD manifold Sym⁺(n).
    pub fn spd(n: usize, coords: FixedVector) -> Self {
        Self {
            manifold_tag: MANIFOLD_TAG_SPD,
            params: vec![n as u32],
            coordinates: coords,
        }
    }

    /// Create a ManifoldPoint for the Grassmannian Gr(k, n).
    pub fn grassmannian(k: usize, n: usize, coords: FixedVector) -> Self {
        Self {
            manifold_tag: MANIFOLD_TAG_GRASSMANNIAN,
            params: vec![k as u32, n as u32],
            coordinates: coords,
        }
    }

    /// Serialize to bytes.
    ///
    /// Format: `[u8 tag][u8 num_params][u32 × num_params BE][FixedVector bytes]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let vec_bytes = self.coordinates.to_bytes();
        let mut bytes = Vec::with_capacity(2 + self.params.len() * 4 + vec_bytes.len());

        bytes.push(self.manifold_tag);
        bytes.push(self.params.len() as u8);
        for &p in &self.params {
            bytes.extend_from_slice(&p.to_be_bytes());
        }
        bytes.extend_from_slice(&vec_bytes);

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.len() < 2 {
            return Err(OverflowDetected::InvalidInput);
        }

        let tag = bytes[0];
        let num_params = bytes[1] as usize;
        let params_end = 2 + num_params * 4;
        if bytes.len() < params_end {
            return Err(OverflowDetected::InvalidInput);
        }

        let mut params = Vec::with_capacity(num_params);
        for i in 0..num_params {
            let offset = 2 + i * 4;
            let p = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap());
            params.push(p);
        }

        let coordinates = FixedVector::from_bytes(&bytes[params_end..])?;

        Ok(Self {
            manifold_tag: tag,
            params,
            coordinates,
        })
    }
}

// ============================================================================
// Compact encoding — variable-length for bandwidth-constrained protocols
// ============================================================================

/// Compact encoding flags byte:
/// - bit 0: sign (0 = non-negative, 1 = negative)
/// - bits 1-2: size class
///   - 00 = zero (no payload)
///   - 01 = small (elide trailing zero bytes, 1-byte length prefix)
///   - 10 = medium (elide trailing zero bytes, 2-byte length prefix)
///   - 11 = full (no elision, full RAW_BYTE_LEN payload)
const COMPACT_ZERO: u8 = 0b000;
const COMPACT_SMALL: u8 = 0b010;
const COMPACT_MEDIUM: u8 = 0b100;
const COMPACT_FULL: u8 = 0b110;
const COMPACT_SIGN_BIT: u8 = 0b001;

impl FixedPoint {
    /// Encode a FixedPoint value in compact format.
    ///
    /// Elides trailing zero bytes from the big-endian representation.
    /// For values near zero (common in tangent vectors, small deltas),
    /// this can save 50-90% of wire bytes.
    ///
    /// Format: `[u8 flags][payload]`
    /// - Zero: 1 byte total
    /// - Small (≤255 significant bytes): 1 + 1 + N bytes
    /// - Medium (≤65535 significant bytes): 1 + 2 + N bytes
    /// - Full: 1 + RAW_BYTE_LEN bytes
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        if self.is_zero() {
            return vec![COMPACT_ZERO];
        }

        let is_neg = self.is_negative();
        let abs_val = self.abs();
        let full_bytes = raw_to_be_bytes(abs_val.raw());

        // Find the first non-zero byte (skip leading zeros in big-endian)
        let first_nonzero = full_bytes.iter().position(|&b| b != 0).unwrap_or(full_bytes.len());
        let significant = &full_bytes[first_nonzero..];
        let sig_len = significant.len();

        let sign_bit = if is_neg { COMPACT_SIGN_BIT } else { 0 };

        if sig_len == 0 {
            // Shouldn't happen (we checked is_zero above), but handle gracefully
            vec![COMPACT_ZERO | sign_bit]
        } else if sig_len <= 255 {
            let mut bytes = Vec::with_capacity(2 + sig_len);
            bytes.push(COMPACT_SMALL | sign_bit);
            bytes.push(sig_len as u8);
            bytes.extend_from_slice(significant);
            bytes
        } else if sig_len <= 65535 {
            let mut bytes = Vec::with_capacity(3 + sig_len);
            bytes.push(COMPACT_MEDIUM | sign_bit);
            bytes.extend_from_slice(&(sig_len as u16).to_be_bytes());
            bytes.extend_from_slice(significant);
            bytes
        } else {
            let mut bytes = Vec::with_capacity(1 + full_bytes.len());
            bytes.push(COMPACT_FULL | sign_bit);
            bytes.extend_from_slice(&full_bytes);
            bytes
        }
    }

    /// Decode a FixedPoint value from compact format.
    ///
    /// Returns (value, bytes_consumed) so the caller can advance through a buffer.
    pub fn from_compact_bytes(bytes: &[u8]) -> Result<(Self, usize), OverflowDetected> {
        if bytes.is_empty() {
            return Err(OverflowDetected::InvalidInput);
        }

        let flags = bytes[0];
        let is_neg = (flags & COMPACT_SIGN_BIT) != 0;
        let size_class = flags & 0b110;

        match size_class {
            COMPACT_ZERO => {
                // Zero value, no payload
                Ok((FixedPoint::ZERO, 1))
            }
            COMPACT_SMALL => {
                // 1-byte length + payload
                if bytes.len() < 2 {
                    return Err(OverflowDetected::InvalidInput);
                }
                let sig_len = bytes[1] as usize;
                if bytes.len() < 2 + sig_len {
                    return Err(OverflowDetected::InvalidInput);
                }

                // Reconstruct full big-endian bytes (zero-pad on the left)
                let mut full = vec![0u8; RAW_BYTE_LEN];
                let offset = RAW_BYTE_LEN.saturating_sub(sig_len);
                let copy_len = sig_len.min(RAW_BYTE_LEN);
                let src_offset = sig_len.saturating_sub(RAW_BYTE_LEN);
                full[offset..offset + copy_len].copy_from_slice(&bytes[2 + src_offset..2 + src_offset + copy_len]);

                let raw = be_bytes_to_raw(&full);
                let val = FixedPoint::from_raw(raw);
                let result = if is_neg { -val } else { val };
                Ok((result, 2 + sig_len))
            }
            COMPACT_MEDIUM => {
                // 2-byte length + payload
                if bytes.len() < 3 {
                    return Err(OverflowDetected::InvalidInput);
                }
                let sig_len = u16::from_be_bytes(bytes[1..3].try_into().unwrap()) as usize;
                if bytes.len() < 3 + sig_len {
                    return Err(OverflowDetected::InvalidInput);
                }

                let mut full = vec![0u8; RAW_BYTE_LEN];
                let offset = RAW_BYTE_LEN.saturating_sub(sig_len);
                let copy_len = sig_len.min(RAW_BYTE_LEN);
                let src_offset = sig_len.saturating_sub(RAW_BYTE_LEN);
                full[offset..offset + copy_len].copy_from_slice(&bytes[3 + src_offset..3 + src_offset + copy_len]);

                let raw = be_bytes_to_raw(&full);
                let val = FixedPoint::from_raw(raw);
                let result = if is_neg { -val } else { val };
                Ok((result, 3 + sig_len))
            }
            COMPACT_FULL | _ => {
                // Full payload, no elision
                if bytes.len() < 1 + RAW_BYTE_LEN {
                    return Err(OverflowDetected::InvalidInput);
                }
                let raw = be_bytes_to_raw(&bytes[1..1 + RAW_BYTE_LEN]);
                let val = FixedPoint::from_raw(raw);
                let result = if is_neg { -val } else { val };
                Ok((result, 1 + RAW_BYTE_LEN))
            }
        }
    }
}

/// Compact-encode a FixedVector (sequence of compact FixedPoint values).
///
/// Format: `[u32 len BE][compact FixedPoint × len]`
///
/// For vectors with many near-zero components (sparse tangent vectors,
/// small perturbations), this can be dramatically smaller than the fixed format.
impl FixedVector {
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let n = self.len();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(n as u32).to_be_bytes());
        for i in 0..n {
            bytes.extend_from_slice(&self[i].to_compact_bytes());
        }
        bytes
    }

    pub fn from_compact_bytes(bytes: &[u8]) -> Result<(Self, usize), OverflowDetected> {
        if bytes.len() < 4 {
            return Err(OverflowDetected::InvalidInput);
        }
        let n = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let mut data = Vec::with_capacity(n);
        let mut pos = 4;
        for _ in 0..n {
            let (val, consumed) = FixedPoint::from_compact_bytes(&bytes[pos..])?;
            data.push(val);
            pos += consumed;
        }
        Ok((FixedVector::from_slice(&data), pos))
    }
}

// ============================================================================
// Profile-specific raw byte conversion helpers
// ============================================================================

use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;

#[cfg(table_format = "q64_64")]
fn raw_to_be_bytes(raw: BinaryStorage) -> Vec<u8> {
    raw.to_be_bytes().to_vec()
}

#[cfg(table_format = "q64_64")]
fn be_bytes_to_raw(bytes: &[u8]) -> BinaryStorage {
    let arr: [u8; 16] = bytes[..16].try_into().unwrap();
    i128::from_be_bytes(arr)
}

#[cfg(table_format = "q128_128")]
fn raw_to_be_bytes(raw: BinaryStorage) -> Vec<u8> {
    // I256 stores words in little-endian order; convert to big-endian bytes
    let le = raw.to_bytes_le();
    le.into_iter().rev().collect()
}

#[cfg(table_format = "q128_128")]
fn be_bytes_to_raw(bytes: &[u8]) -> BinaryStorage {
    // Reverse big-endian bytes to little-endian, then parse
    let le: Vec<u8> = bytes.iter().rev().copied().collect();
    crate::fixed_point::I256::from_bytes_le(&le)
}

#[cfg(table_format = "q256_256")]
fn raw_to_be_bytes(raw: BinaryStorage) -> Vec<u8> {
    let le = raw.to_bytes_le();
    le.into_iter().rev().collect()
}

#[cfg(table_format = "q256_256")]
fn be_bytes_to_raw(bytes: &[u8]) -> BinaryStorage {
    let le: Vec<u8> = bytes.iter().rev().copied().collect();
    crate::fixed_point::I512::from_bytes_le(&le)
}

#[cfg(table_format = "q32_32")]
fn raw_to_be_bytes(raw: BinaryStorage) -> Vec<u8> {
    raw.to_be_bytes().to_vec()
}

#[cfg(table_format = "q32_32")]
fn be_bytes_to_raw(bytes: &[u8]) -> BinaryStorage {
    let arr: [u8; 8] = bytes[..8].try_into().unwrap();
    i64::from_be_bytes(arr)
}

#[cfg(table_format = "q16_16")]
fn raw_to_be_bytes(raw: BinaryStorage) -> Vec<u8> {
    raw.to_be_bytes().to_vec()
}

#[cfg(table_format = "q16_16")]
fn be_bytes_to_raw(bytes: &[u8]) -> BinaryStorage {
    let arr: [u8; 4] = bytes[..4].try_into().unwrap();
    i32::from_be_bytes(arr)
}

// ============================================================================
// serde support (behind "serde" feature gate)
// ============================================================================

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::{Serialize, Deserialize, Serializer, Deserializer};
    use serde::de::{self, Visitor, SeqAccess};
    use serde::ser::SerializeStruct;
    use std::fmt;

    // --- FixedPoint: serialize as tagged raw bytes ---

    impl Serialize for FixedPoint {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let bytes = self.to_bytes();
            serializer.serialize_bytes(&bytes)
        }
    }

    impl<'de> Deserialize<'de> for FixedPoint {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct FixedPointVisitor;
            impl<'de> Visitor<'de> for FixedPointVisitor {
                type Value = FixedPoint;
                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("FixedPoint bytes (profile tag + raw BE)")
                }
                fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<FixedPoint, E> {
                    FixedPoint::from_bytes(v).map_err(|_| E::custom("invalid FixedPoint bytes"))
                }
                fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<FixedPoint, E> {
                    FixedPoint::from_bytes(&v).map_err(|_| E::custom("invalid FixedPoint bytes"))
                }
            }
            deserializer.deserialize_bytes(FixedPointVisitor)
        }
    }

    // --- FixedVector: serialize as struct {len, data} ---

    impl Serialize for FixedVector {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let bytes = self.to_bytes();
            serializer.serialize_bytes(&bytes)
        }
    }

    impl<'de> Deserialize<'de> for FixedVector {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct VecVisitor;
            impl<'de> Visitor<'de> for VecVisitor {
                type Value = FixedVector;
                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("FixedVector bytes (u32 len BE + raw elements)")
                }
                fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<FixedVector, E> {
                    FixedVector::from_bytes(v).map_err(|_| E::custom("invalid FixedVector bytes"))
                }
                fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<FixedVector, E> {
                    FixedVector::from_bytes(&v).map_err(|_| E::custom("invalid FixedVector bytes"))
                }
            }
            deserializer.deserialize_bytes(VecVisitor)
        }
    }

    // --- FixedMatrix: serialize as bytes ---

    impl Serialize for FixedMatrix {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let bytes = self.to_bytes();
            serializer.serialize_bytes(&bytes)
        }
    }

    impl<'de> Deserialize<'de> for FixedMatrix {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct MatVisitor;
            impl<'de> Visitor<'de> for MatVisitor {
                type Value = FixedMatrix;
                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("FixedMatrix bytes (rows/cols u32 BE + raw elements)")
                }
                fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<FixedMatrix, E> {
                    FixedMatrix::from_bytes(v).map_err(|_| E::custom("invalid FixedMatrix bytes"))
                }
                fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<FixedMatrix, E> {
                    FixedMatrix::from_bytes(&v).map_err(|_| E::custom("invalid FixedMatrix bytes"))
                }
            }
            deserializer.deserialize_bytes(MatVisitor)
        }
    }

    // --- Tensor: serialize as bytes ---

    impl Serialize for Tensor {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let bytes = self.to_bytes();
            serializer.serialize_bytes(&bytes)
        }
    }

    impl<'de> Deserialize<'de> for Tensor {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct TensorVisitor;
            impl<'de> Visitor<'de> for TensorVisitor {
                type Value = Tensor;
                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Tensor bytes (rank + shape + data)")
                }
                fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Tensor, E> {
                    Tensor::from_bytes(v).map_err(|_| E::custom("invalid Tensor bytes"))
                }
                fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Tensor, E> {
                    Tensor::from_bytes(&v).map_err(|_| E::custom("invalid Tensor bytes"))
                }
            }
            deserializer.deserialize_bytes(TensorVisitor)
        }
    }

    // --- ManifoldPoint: serialize as bytes ---

    impl Serialize for ManifoldPoint {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let bytes = self.to_bytes();
            serializer.serialize_bytes(&bytes)
        }
    }

    impl<'de> Deserialize<'de> for ManifoldPoint {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct MpVisitor;
            impl<'de> Visitor<'de> for MpVisitor {
                type Value = ManifoldPoint;
                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("ManifoldPoint bytes (tag + params + coords)")
                }
                fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<ManifoldPoint, E> {
                    ManifoldPoint::from_bytes(v).map_err(|_| E::custom("invalid ManifoldPoint bytes"))
                }
                fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<ManifoldPoint, E> {
                    ManifoldPoint::from_bytes(&v).map_err(|_| E::custom("invalid ManifoldPoint bytes"))
                }
            }
            deserializer.deserialize_bytes(MpVisitor)
        }
    }
}
