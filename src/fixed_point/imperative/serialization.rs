//! Binary serialization for `Fixed<S, F>`, `FixedVector`, `FixedMatrix`,
//! `Tensor` and `ManifoldPoint`, generic over the format.
//!
//! Wire format: big-endian byte order with a 3-byte format tag on scalars.
//!
//! ## Scalar format (0.7.0)
//! `[u8 0x20 | storage code][u16 F big-endian][raw bytes in big-endian]`
//!
//! Storage codes: 1 = i32 (4 bytes), 2 = i64 (8), 3 = i128 (16),
//! 4 = I256 (32), 5 = I512 (64). `F` is the number of fractional bits (two
//! bytes, since Q256.256 has 256 of them), so a
//! value written at Q16.16 is refused by a Q22.10 reader instead of being
//! read at the wrong scale, and every `Fixed<S, F>` in one binary writes its
//! own tag.
//!
//! ## Legacy scalar tags (0.6.x)
//! Up to 0.6.7 the tag was one byte naming the build's profile: 0x01 Q64.64,
//! 0x02 Q128.128, 0x03 Q256.256, 0x04 Q32.32, 0x05 Q16.16 and, from 0.6.5,
//! 0x80 | F for a realtime build with another split. [`Fixed::from_bytes`]
//! refuses them; [`Fixed::from_bytes_legacy`] reads them when the storage
//! word matches and, where the tag names a split (0x80 | F), the split
//! matches too (0x05 names Q16.16 by convention, but up to 0.6.4 every
//! realtime split wrote it, so the legacy reader takes it for the reader's
//! own split). [`Fixed::legacy_tag`] is the tag 0.6.x wrote for this format.
//!
//! ## FixedVector format
//! `[u32 len (big-endian)][raw bytes × len]` (no per-element tag)
//!
//! The vector, matrix, tensor and manifold-point formats carry no tag and
//! are unchanged since 0.6.x: the reader must know the format they were
//! written with.
//!
//! ## FixedMatrix format
//! `[u32 rows (big-endian)][u32 cols (big-endian)][raw bytes × rows*cols]`
//!
//! ## Tensor format
//! `[u8 rank][u32 × rank (shape dims)][raw bytes × product(shape)]`
//!
//! ## ManifoldPoint format
//! `[u8 manifold_tag][u8 num_params][u32 × num_params][FixedVector bytes]`
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
//! Small/medium elide leading zero bytes of the magnitude.

use super::FixedVector;
use super::FixedMatrix;
use super::tensor::Tensor;
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::fixed_num::FixedNum;
use crate::fixed_point::width::Storage;
use crate::fixed_point::Fixed;

/// The high bits of the first tag byte of the 0.7.0 scalar format.
const FORMAT_TAG: u8 = 0x20;

// ============================================================================
// Scalar serialization
// ============================================================================

impl<S: Storage, const F: u32> Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    /// Serialize with the format tag (big-endian).
    ///
    /// Format: `[u8 0x20 | storage code][u16 F big-endian][raw bytes big-endian]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(3 + S::BYTES);
        bytes.extend_from_slice(&Self::format_tag());
        bytes.extend_from_slice(&self.raw().to_be_vec());
        bytes
    }

    /// Deserialize bytes written by [`Self::to_bytes`] for this format.
    ///
    /// `Err(InvalidInput)` when the tag names another storage word or split
    /// (a 0.6.x tag included: see [`Self::from_bytes_legacy`]) or the slice
    /// is too short.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.len() < 3 + S::BYTES || bytes[..3] != Self::format_tag() {
            return Err(OverflowDetected::InvalidInput);
        }
        Ok(Self::from_raw(S::from_be_slice(&bytes[3..3 + S::BYTES])))
    }

    /// Deserialize bytes written by `to_bytes` of a 0.6.x build: a one-byte
    /// profile tag and the raw bytes.
    ///
    /// Accepted when the tag names this storage word (0x05 or 0x80 | F for
    /// i32, 0x04 for i64, 0x01 for i128, 0x02 for I256, 0x03 for I512) and,
    /// for 0x80 | F, the split is this one; 0x05 is taken for the reader's
    /// split, as every realtime split wrote it up to 0.6.4. `Err(InvalidInput)`
    /// otherwise, and for a 0.7.0 tag (use [`Self::from_bytes`]).
    pub fn from_bytes_legacy(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.len() < 1 + S::BYTES {
            return Err(OverflowDetected::InvalidInput);
        }
        let accepted = match (S::CODE, bytes[0]) {
            (1, 0x05) => true,
            (1, t) => t & 0x80 != 0 && (t & 0x7F) as u32 == F,
            (2, 0x04) | (3, 0x01) | (4, 0x02) | (5, 0x03) => true,
            _ => false,
        };
        if !accepted {
            return Err(OverflowDetected::InvalidInput);
        }
        Ok(Self::from_raw(S::from_be_slice(&bytes[1..1 + S::BYTES])))
    }

    /// Serialize raw storage only (no tag), big-endian.
    pub fn to_raw_bytes(&self) -> Vec<u8> {
        self.raw().to_be_vec()
    }

    /// Deserialize raw storage only (no tag), big-endian.
    pub fn from_raw_bytes(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.len() < S::BYTES {
            return Err(OverflowDetected::InvalidInput);
        }
        Ok(Self::from_raw(S::from_be_slice(&bytes[..S::BYTES])))
    }

    /// The three tag bytes [`Self::to_bytes`] writes: `0x20 | storage code`
    /// and `F` as two big-endian bytes (Q256.256 has F = 256).
    pub fn format_tag() -> [u8; 3] {
        let f = (F as u16).to_be_bytes();
        [FORMAT_TAG | S::CODE, f[0], f[1]]
    }

    /// The one-byte tag a 0.6.x build of this format wrote (0x05 for Q16.16,
    /// 0x80 | F for another split on i32, 0x04 for i64, 0x01 for i128, 0x02
    /// for I256, 0x03 for I512); what [`Self::from_bytes_legacy`] accepts.
    pub fn legacy_tag() -> u8 {
        match S::CODE {
            1 => if F == 16 { 0x05 } else { 0x80 | F as u8 },
            2 => 0x04,
            3 => 0x01,
            4 => 0x02,
            _ => 0x03,
        }
    }

    /// Size in bytes of the raw storage (without tag).
    pub fn raw_byte_len() -> usize {
        S::BYTES
    }
}

// ============================================================================
// FixedVector serialization
// ============================================================================

impl<S: Storage, const F: u32> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Serialize to bytes.
    ///
    /// Format: `[u32 len BE][raw bytes × len]` (no per-element tag).
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.len();
        let mut bytes = Vec::with_capacity(4 + n * S::BYTES);
        bytes.extend_from_slice(&(n as u32).to_be_bytes());
        for i in 0..n {
            bytes.extend_from_slice(&self[i].raw().to_be_vec());
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OverflowDetected> {
        if bytes.len() < 4 {
            return Err(OverflowDetected::InvalidInput);
        }
        let n = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let expected = 4 + n * S::BYTES;
        if bytes.len() < expected {
            return Err(OverflowDetected::InvalidInput);
        }
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let start = 4 + i * S::BYTES;
            data.push(Fixed::from_raw(S::from_be_slice(&bytes[start..start + S::BYTES])));
        }
        Ok(FixedVector::from_slice(&data))
    }
}

// ============================================================================
// FixedMatrix serialization
// ============================================================================

impl<S: Storage, const F: u32> FixedMatrix<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Serialize to bytes.
    ///
    /// Format: `[u32 rows BE][u32 cols BE][raw bytes × rows*cols]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.rows() * self.cols();
        let mut bytes = Vec::with_capacity(8 + n * S::BYTES);
        bytes.extend_from_slice(&(self.rows() as u32).to_be_bytes());
        bytes.extend_from_slice(&(self.cols() as u32).to_be_bytes());
        for r in 0..self.rows() {
            for c in 0..self.cols() {
                bytes.extend_from_slice(&self.get(r, c).raw().to_be_vec());
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
        let expected = 8 + n * S::BYTES;
        if bytes.len() < expected {
            return Err(OverflowDetected::InvalidInput);
        }
        let mut m = FixedMatrix::new(rows, cols);
        for i in 0..n {
            let start = 8 + i * S::BYTES;
            m.set(i / cols, i % cols, Fixed::from_raw(S::from_be_slice(&bytes[start..start + S::BYTES])));
        }
        Ok(m)
    }
}

// ============================================================================
// Tensor serialization
// ============================================================================

impl<S: Storage, const F: u32> Tensor<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    /// Serialize a tensor to bytes.
    ///
    /// Format: `[u8 rank][u32 × rank (shape dims BE)][raw bytes × product(shape)]`
    ///
    /// The rank byte limits tensors to 255 dimensions (more than enough for any
    /// practical use: Riemann curvature is rank 4, highest in differential geometry).
    pub fn to_bytes(&self) -> Vec<u8> {
        let rank = self.rank();
        let total = self.len();
        let mut bytes = Vec::with_capacity(1 + rank * 4 + total * S::BYTES);
        bytes.push(rank as u8);
        for &dim in self.shape() {
            bytes.extend_from_slice(&(dim as u32).to_be_bytes());
        }
        for val in self.data() {
            bytes.extend_from_slice(&val.raw().to_be_vec());
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
        let mut shape = Vec::with_capacity(rank);
        for i in 0..rank {
            let offset = 1 + i * 4;
            shape.push(u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize);
        }
        let total: usize = shape.iter().product::<usize>().max(1);
        let expected = header_len + total * S::BYTES;
        if bytes.len() < expected {
            return Err(OverflowDetected::InvalidInput);
        }
        let mut data = Vec::with_capacity(total);
        for i in 0..total {
            let start = header_len + i * S::BYTES;
            data.push(Fixed::from_raw(S::from_be_slice(&bytes[start..start + S::BYTES])));
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
pub struct ManifoldPoint<T> {
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
    pub coordinates: FixedVector<T>,
}

impl<T> ManifoldPoint<T> {
    /// Create a ManifoldPoint for Euclidean space R^n.
    pub fn euclidean(dim: usize, coords: FixedVector<T>) -> Self {
        Self { manifold_tag: MANIFOLD_TAG_EUCLIDEAN, params: vec![dim as u32], coordinates: coords }
    }

    /// Create a ManifoldPoint for the n-sphere S^n.
    pub fn sphere(dim: usize, coords: FixedVector<T>) -> Self {
        Self { manifold_tag: MANIFOLD_TAG_SPHERE, params: vec![dim as u32], coordinates: coords }
    }

    /// Create a ManifoldPoint for hyperbolic space H^n.
    pub fn hyperbolic(dim: usize, coords: FixedVector<T>) -> Self {
        Self { manifold_tag: MANIFOLD_TAG_HYPERBOLIC, params: vec![dim as u32], coordinates: coords }
    }

    /// Create a ManifoldPoint for the SPD manifold Sym⁺(n).
    pub fn spd(n: usize, coords: FixedVector<T>) -> Self {
        Self { manifold_tag: MANIFOLD_TAG_SPD, params: vec![n as u32], coordinates: coords }
    }

    /// Create a ManifoldPoint for the Grassmannian Gr(k, n).
    pub fn grassmannian(k: usize, n: usize, coords: FixedVector<T>) -> Self {
        Self { manifold_tag: MANIFOLD_TAG_GRASSMANNIAN, params: vec![k as u32, n as u32], coordinates: coords }
    }
}

impl<S: Storage, const F: u32> ManifoldPoint<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
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
            params.push(u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()));
        }
        let coordinates = FixedVector::from_bytes(&bytes[params_end..])?;
        Ok(Self { manifold_tag: tag, params, coordinates })
    }
}

// ============================================================================
// Compact encoding — variable-length for bandwidth-constrained protocols
// ============================================================================

/// Compact encoding flags byte:
/// - bit 0: sign (0 = non-negative, 1 = negative)
/// - bits 1-2: size class
///   - 00 = zero (no payload)
///   - 01 = small (elide leading zero bytes, 1-byte length prefix)
///   - 10 = medium (elide leading zero bytes, 2-byte length prefix)
///   - 11 = full (no elision, full raw payload)
const COMPACT_ZERO: u8 = 0b000;
const COMPACT_SMALL: u8 = 0b010;
const COMPACT_MEDIUM: u8 = 0b100;
const COMPACT_FULL: u8 = 0b110;
const COMPACT_SIGN_BIT: u8 = 0b001;

impl<S: Storage, const F: u32> Fixed<S, F>
where
    Self: FixedNum<Storage = S>,
{
    /// Encode a value in compact format.
    ///
    /// Elides leading zero bytes of the magnitude's big-endian representation.
    /// For values near zero (common in tangent vectors, small deltas),
    /// this can save 50-90% of wire bytes.
    ///
    /// Format: `[u8 flags][payload]`
    /// - Zero: 1 byte total
    /// - Small (≤255 significant bytes): 1 + 1 + N bytes
    /// - Medium (≤65535 significant bytes): 1 + 2 + N bytes
    /// - Full: 1 + raw bytes
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        if self.is_zero() {
            return vec![COMPACT_ZERO];
        }
        let is_neg = self.is_negative();
        let Some(magnitude) = self.raw().checked_neg_wide().map(|neg| if is_neg { neg } else { self.raw() }) else {
            // the word's minimum has no magnitude in the word: the full class
            // carries its two's complement bytes with no sign bit (up to 0.6.7
            // the magnitude was formed with `abs`, which panicked here)
            let mut bytes = Vec::with_capacity(1 + S::BYTES);
            bytes.push(COMPACT_FULL);
            bytes.extend_from_slice(&self.raw().to_be_vec());
            return bytes;
        };
        let full_bytes = magnitude.to_be_vec();
        let first_nonzero = full_bytes.iter().position(|&b| b != 0).unwrap_or(full_bytes.len());
        let significant = &full_bytes[first_nonzero..];
        let sig_len = significant.len();
        let sign_bit = if is_neg { COMPACT_SIGN_BIT } else { 0 };
        if sig_len == 0 {
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

    /// Decode a value from compact format.
    ///
    /// Returns (value, bytes_consumed) so the caller can advance through a buffer.
    pub fn from_compact_bytes(bytes: &[u8]) -> Result<(Self, usize), OverflowDetected> {
        if bytes.is_empty() {
            return Err(OverflowDetected::InvalidInput);
        }
        let flags = bytes[0];
        let is_neg = (flags & COMPACT_SIGN_BIT) != 0;
        let size_class = flags & 0b110;
        // the magnitude from `sig_len` big-endian bytes at `payload`
        let magnitude = |payload: &[u8], sig_len: usize| -> Self {
            let mut full = vec![0u8; S::BYTES];
            let offset = S::BYTES.saturating_sub(sig_len);
            let copy_len = sig_len.min(S::BYTES);
            let src_offset = sig_len.saturating_sub(S::BYTES);
            full[offset..offset + copy_len].copy_from_slice(&payload[src_offset..src_offset + copy_len]);
            Self::from_raw(S::from_be_slice(&full))
        };
        let signed = |val: Self| if is_neg { -val } else { val };
        match size_class {
            COMPACT_ZERO => Ok((Self::ZERO, 1)),
            COMPACT_SMALL => {
                if bytes.len() < 2 {
                    return Err(OverflowDetected::InvalidInput);
                }
                let sig_len = bytes[1] as usize;
                if bytes.len() < 2 + sig_len {
                    return Err(OverflowDetected::InvalidInput);
                }
                Ok((signed(magnitude(&bytes[2..2 + sig_len], sig_len)), 2 + sig_len))
            }
            COMPACT_MEDIUM => {
                if bytes.len() < 3 {
                    return Err(OverflowDetected::InvalidInput);
                }
                let sig_len = u16::from_be_bytes(bytes[1..3].try_into().unwrap()) as usize;
                if bytes.len() < 3 + sig_len {
                    return Err(OverflowDetected::InvalidInput);
                }
                Ok((signed(magnitude(&bytes[3..3 + sig_len], sig_len)), 3 + sig_len))
            }
            _ => {
                if bytes.len() < 1 + S::BYTES {
                    return Err(OverflowDetected::InvalidInput);
                }
                Ok((signed(Self::from_raw(S::from_be_slice(&bytes[1..1 + S::BYTES]))), 1 + S::BYTES))
            }
        }
    }
}

/// Compact-encode a FixedVector (sequence of compact values).
///
/// Format: `[u32 len BE][compact value × len]`
///
/// For vectors with many near-zero components (sparse tangent vectors,
/// small perturbations), this can be dramatically smaller than the fixed format.
impl<S: Storage, const F: u32> FixedVector<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
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
            let (val, consumed) = Fixed::from_compact_bytes(&bytes[pos..])?;
            data.push(val);
            pos += consumed;
        }
        Ok((FixedVector::from_slice(&data), pos))
    }
}

// ============================================================================
// serde support (behind "serde" feature gate)
// ============================================================================

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::de::{self, Visitor};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::fmt;
    use std::marker::PhantomData;

    /// Every form serializes as the bytes of its `to_bytes`.
    macro_rules! bytes_serde {
        ($ty:ident, $what:expr) => {
            impl<S: Storage, const F: u32> Serialize for $ty<Fixed<S, F>>
            where
                Fixed<S, F>: FixedNum<Storage = S>,
            {
                fn serialize<Z: Serializer>(&self, serializer: Z) -> Result<Z::Ok, Z::Error> {
                    serializer.serialize_bytes(&self.to_bytes())
                }
            }

            impl<'de, S: Storage, const F: u32> Deserialize<'de> for $ty<Fixed<S, F>>
            where
                Fixed<S, F>: FixedNum<Storage = S>,
            {
                fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                    struct V<S, const F: u32>(PhantomData<S>);
                    impl<'de, S: Storage, const F: u32> Visitor<'de> for V<S, F>
                    where
                        Fixed<S, F>: FixedNum<Storage = S>,
                    {
                        type Value = $ty<Fixed<S, F>>;
                        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                            formatter.write_str($what)
                        }
                        fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                            $ty::from_bytes(v).map_err(|_| E::custom(concat!("invalid ", stringify!($ty), " bytes")))
                        }
                        fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                            self.visit_bytes(&v)
                        }
                    }
                    deserializer.deserialize_bytes(V::<S, F>(PhantomData))
                }
            }
        };
    }

    impl<S: Storage, const F: u32> Serialize for Fixed<S, F>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        fn serialize<Z: Serializer>(&self, serializer: Z) -> Result<Z::Ok, Z::Error> {
            serializer.serialize_bytes(&self.to_bytes())
        }
    }

    impl<'de, S: Storage, const F: u32> Deserialize<'de> for Fixed<S, F>
    where
        Fixed<S, F>: FixedNum<Storage = S>,
    {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct V<S, const F: u32>(PhantomData<S>);
            impl<'de, S: Storage, const F: u32> Visitor<'de> for V<S, F>
            where
                Fixed<S, F>: FixedNum<Storage = S>,
            {
                type Value = Fixed<S, F>;
                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("Fixed bytes (format tag + raw BE)")
                }
                fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
                    Fixed::from_bytes(v).map_err(|_| E::custom("invalid Fixed bytes"))
                }
                fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                    self.visit_bytes(&v)
                }
            }
            deserializer.deserialize_bytes(V::<S, F>(PhantomData))
        }
    }

    bytes_serde!(FixedVector, "FixedVector bytes (len + raw BE)");
    bytes_serde!(FixedMatrix, "FixedMatrix bytes (rows, cols + raw BE)");
    bytes_serde!(Tensor, "Tensor bytes (rank, shape + raw BE)");
    bytes_serde!(ManifoldPoint, "ManifoldPoint bytes (tag, params + vector)");
}
