//! Weight bit patterns to fixed point, by integer shifts only.
//!
//! Model weight files carry IEEE binary16 or bfloat16 **bit patterns**. This
//! module decodes those bits into Q-format raw values and TQ1.9 raw values
//! without a float type: a pattern is split into sign, integer mantissa and
//! binary exponent, and every conversion is a shift of that mantissa.
//!
//! - [`to_raw`]: `trunc(v * 2^frac_bits)` toward zero, at any fractional
//!   width (`to_q64_raw` is the Q64.64 form).
//! - [`to_storage_raw`] / [`to_fixed`]: the same at the build's storage
//!   format (realtime and compact profiles).
//! - [`to_tq19_raw`]: `round(v * 3^9)`, half away from zero.
//! - [`WeightBits`]: a row-major matrix of patterns as read from a file, the
//!   input of the quantisers in [`super::quantize`].
//!
//! Infinities and NaNs are refused with `InvalidInput`; a value outside the
//! target range is `TierOverflow`.

use crate::fixed_point::core_types::errors::OverflowDetected;
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
use crate::fixed_point::imperative::FixedPoint;

/// Which 16-bit float format a bit pattern is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HalfKind {
    /// IEEE 754 binary16: 1 sign, 5 exponent, 10 mantissa bits.
    Binary16,
    /// bfloat16: 1 sign, 8 exponent, 7 mantissa bits.
    BFloat16,
}

/// `|v| = m * 2^e` with `m` the integer mantissa (implicit bit included) and
/// the sign separately: `(negative, m, e)`. Zero gives `m = 0`; subnormals
/// have no implicit bit. `Err(InvalidInput)` for an infinity or a NaN.
#[inline]
pub fn decompose(bits: u16, kind: HalfKind) -> Result<(bool, u64, i32), OverflowDetected> {
    let sign = bits & 0x8000 != 0;
    match kind {
        HalfKind::Binary16 => {
            let exp = ((bits >> 10) & 0x1F) as i32;
            let mant = (bits & 0x3FF) as u64;
            if exp == 0x1F {
                return Err(OverflowDetected::InvalidInput);
            }
            Ok(if exp == 0 { (sign, mant, -24) } else { (sign, mant | 0x400, exp - 25) })
        }
        HalfKind::BFloat16 => {
            let exp = ((bits >> 7) & 0xFF) as i32;
            let mant = (bits & 0x7F) as u64;
            if exp == 0xFF {
                return Err(OverflowDetected::InvalidInput);
            }
            Ok(if exp == 0 { (sign, mant, -133) } else { (sign, mant | 0x80, exp - 134) })
        }
    }
}

/// `trunc(v * 2^frac_bits)` toward zero as an i128.
///
/// Exact whenever the value has no bits below `2^-frac_bits`.
/// `Err(TierOverflow)` when the result does not fit i128, `Err(InvalidInput)`
/// for a non-finite pattern.
pub fn to_raw(bits: u16, kind: HalfKind, frac_bits: u32) -> Result<i128, OverflowDetected> {
    let (sign, m, e) = decompose(bits, kind)?;
    if m == 0 {
        return Ok(0);
    }
    let shift = e as i64 + frac_bits as i64;
    let mag: i128 = if shift >= 0 {
        // m has at most 11 significant bits
        let top = 64 - m.leading_zeros() as i64;
        if top + shift > 127 {
            return Err(OverflowDetected::TierOverflow);
        }
        (m as i128) << shift
    } else if -shift >= 64 {
        0
    } else {
        (m >> (-shift) as u32) as i128
    };
    Ok(if sign { -mag } else { mag })
}

/// The value at Q64.64: `trunc(v * 2^64)`. Exact for every finite pattern of
/// either format (the smallest subnormal is `2^-133`, truncated to zero below
/// `2^-64`), `Err(TierOverflow)` from `2^63` up.
#[inline]
pub fn to_q64_raw(bits: u16, kind: HalfKind) -> Result<i128, OverflowDetected> {
    to_raw(bits, kind, 64)
}

/// The value at the build's storage format: `trunc(v * 2^FRAC_BITS)` toward
/// zero. `Err(TierOverflow)` when the magnitude exceeds the largest storage
/// value.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
pub fn to_storage_raw(bits: u16, kind: HalfKind) -> Result<BinaryStorage, OverflowDetected> {
    #[cfg(table_format = "q16_16")]
    let frac_bits = crate::fixed_point::frac_config::FRAC_BITS;
    #[cfg(table_format = "q32_32")]
    let frac_bits = 32;
    let raw = to_raw(bits, kind, frac_bits)?;
    // the magnitude is compared, so -2^(W-1) is refused like +2^(W-1)
    if raw > BinaryStorage::MAX as i128 || raw < -(BinaryStorage::MAX as i128) {
        return Err(OverflowDetected::TierOverflow);
    }
    Ok(raw as BinaryStorage)
}

/// The value as a `FixedPoint`, truncated toward zero; see [`to_storage_raw`].
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[inline]
pub fn to_fixed(bits: u16, kind: HalfKind) -> Result<FixedPoint, OverflowDetected> {
    Ok(FixedPoint::from_raw(to_storage_raw(bits, kind)?))
}

/// TQ1.9 raw value: `round(v * 3^9)`, half away from zero, unclamped (the
/// caller clamps to `MAX_RAW` / `MIN_RAW`). A value from `2^21` up, far
/// outside the TQ1.9 range, returns `i64::MAX / 4` with its sign.
pub fn to_tq19_raw(bits: u16, kind: HalfKind) -> Result<i64, OverflowDetected> {
    let (sign, m, e) = decompose(bits, kind)?;
    if m == 0 {
        return Ok(0);
    }
    let n = (m as i64) * super::SCALE as i64;
    let mag: i64 = if e >= 0 {
        if e > 20 { i64::MAX / 4 } else { n << e }
    } else {
        let k = (-e) as u32;
        if k >= 63 { 0 } else { (n + (1i64 << (k - 1))) >> k }
    };
    Ok(if sign { -mag } else { mag })
}

/// A matrix of weight bit patterns as read from a file: the float-free form
/// projections are quantised from and embeddings are decoded from.
#[derive(Clone)]
pub struct WeightBits {
    pub rows: usize,
    pub cols: usize,
    pub kind: HalfKind,
    /// Row-major, one u16 per element (the file's little-endian pair).
    pub bits: Vec<u16>,
}

impl WeightBits {
    /// From the file's little-endian bytes. `Err(InvalidInput)` when `bytes`
    /// holds fewer than `rows * cols` elements; extra bytes are ignored.
    pub fn from_le_bytes(rows: usize, cols: usize, kind: HalfKind, bytes: &[u8]) -> Result<Self, OverflowDetected> {
        let n = rows * cols;
        if bytes.len() < n * 2 {
            return Err(OverflowDetected::InvalidInput);
        }
        let bits = bytes[..n * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        Ok(Self { rows, cols, kind, bits })
    }

    /// Storage raw of element `i`; see [`to_storage_raw`].
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    #[inline]
    pub fn storage_raw(&self, i: usize) -> Result<BinaryStorage, OverflowDetected> {
        to_storage_raw(self.bits[i], self.kind)
    }

    /// TQ1.9 raw of element `i`; see [`to_tq19_raw`].
    #[inline]
    pub fn tq19_raw(&self, i: usize) -> Result<i64, OverflowDetected> {
        to_tq19_raw(self.bits[i], self.kind)
    }

    /// `(negative, m, e)` of element `i`, for exact rational arithmetic.
    #[inline]
    pub fn decompose(&self, i: usize) -> Result<(bool, u64, i32), OverflowDetected> {
        decompose(self.bits[i], self.kind)
    }
}
