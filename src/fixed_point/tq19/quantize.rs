//! Quantisers from weight bit patterns to the ternary matrix formats, by
//! exact integer arithmetic (no float anywhere).
//!
//! Each returns `None` when a pattern is not finite, and [`quantize_tq19`]
//! also when the values do not fit the format.

use super::bits::WeightBits;
use super::{TQ19Matrix, MAX_RAW, MIN_RAW};

/// `round(num / den)` for non-negative `num`, positive `den`: half up.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[inline]
fn half_up(num: i128, den: i128) -> i128 {
    (2 * num + den) / (2 * den)
}

/// Mantissas are below `2^MANTISSA_BITS` (11 for binary16, 8 for bfloat16).
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
const MANTISSA_BITS: i32 = 11;

/// `m * 2^e > mm * 2^me` for non-zero mantissas below `2^MANTISSA_BITS`.
///
/// From an exponent gap of `MANTISSA_BITS` the larger exponent is the larger
/// value whatever the mantissas are (`m * 2^gap >= 2^11 > mm`), so a mantissa
/// is only ever shifted by less than that. Shifting by the gap itself left
/// 128 bits from a gap of a little over 100, which bfloat16 rows reach
/// (0.6.5).
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[inline]
fn greater(m: u64, e: i32, mm: u64, me: i32) -> bool {
    let gap = e - me;
    if gap >= MANTISSA_BITS {
        true
    } else if gap <= -MANTISSA_BITS {
        false
    } else if gap >= 0 {
        (m << gap) > mm
    } else {
        m > (mm << -gap)
    }
}

/// The largest magnitude in a row as `(m, e)`, value `m * 2^e`; `None` for an
/// all-zero row, `Err(())` for a non-finite pattern.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
fn row_max(w: &WeightBits, base: usize) -> Result<Option<(u64, i32)>, ()> {
    let mut mx: Option<(u64, i32)> = None;
    for c in 0..w.cols {
        let (_, m, e) = w.decompose(base + c).map_err(|_| ())?;
        if m == 0 {
            continue;
        }
        mx = Some(match mx {
            None => (m, e),
            Some((mm, me)) => {
                if greater(m, e, mm, me) { (m, e) } else { (mm, me) }
            }
        });
    }
    Ok(mx)
}

/// `round(|w| * levels / max|w|)` for one element, capped at `levels`.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[inline]
fn row_code(m: u64, e: i32, mm: u64, me: i32, levels: i128) -> i128 {
    let d = e - me;
    // m * levels < 2^26 (m < 2^11, levels < 2^15), so from 2^-64 of the row
    // maximum down the quotient is far below one half: the code is 0. (The
    // shifted denominator would not fit i128 from about 2^-116.)
    if d <= -64 {
        return 0;
    }
    // (mm, me) is the row maximum, so m * 2^d <= mm < 2^11 and a non-negative
    // d is below 11: the numerator below is under 2^37.
    assert!(d < MANTISSA_BITS, "row_code: element above the row maximum");
    let (num, den) = if d >= 0 { ((m as i128) * levels * (1i128 << d), mm as i128) } else { ((m as i128) * levels, (mm as i128) << (-d)) };
    half_up(num, den).min(levels)
}

/// `round(n * 2^(me + 32) / levels)` as an unsigned Q32.32 row scale, for
/// `n < 2^26`; `None` when the scale does not fit 64 bits (a row maximum of
/// about 2^32 or more, which no weight matrix has).
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
fn row_scale_q32(n: i128, me: i32, levels: i128) -> Option<u64> {
    let sh = me + 32;
    if sh > 64 {
        return None;
    }
    // sh >= -101 (the smallest exponent is -133): levels << 101 fits i128
    let (num, den) = if sh >= 0 { (n << sh, levels) } else { (n, levels << (-sh)) };
    u64::try_from(half_up(num, den)).ok()
}

/// Quantise to a `TQ19Matrix` with the global TQ1.9 scale: each element is
/// `round(v * 3^9)`, half away from zero.
///
/// Values outside the TQ1.9 range (about +-1.5) are clamped. Returns `None`
/// if more than 1% of the elements were clamped, or on a non-finite pattern.
pub fn quantize_tq19(w: &WeightBits) -> Option<TQ19Matrix> {
    let total = w.rows * w.cols;
    let mut data = Vec::with_capacity(total);
    let mut clamped = 0u64;
    for i in 0..total {
        let scaled = w.tq19_raw(i).ok()?;
        if scaled > MAX_RAW as i64 {
            data.push(MAX_RAW);
            clamped += 1;
        } else if scaled < MIN_RAW as i64 {
            data.push(MIN_RAW);
            clamped += 1;
        } else {
            data.push(scaled as i16);
        }
    }
    if clamped > (total as u64 / 100) {
        return None;
    }
    Some(TQ19Matrix::new(w.rows, w.cols, data))
}

/// Quantise to a row-scaled TQ1.9 matrix by exact rational rounding: each
/// row quantises against its own largest magnitude,
/// `q = round(v * MAX_RAW / max|w|)` (half away from zero), and carries the
/// scale `round(max|w| * 3^9 * 2^32 / MAX_RAW)` in unsigned Q32.32. An
/// all-zero row gets codes 0 and scale 0.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
pub fn quantize_tq19_rowscaled(w: &WeightBits) -> Option<super::RowScaledTQ19> {
    let levels = MAX_RAW as i128;
    let mut data = vec![0i16; w.rows * w.cols];
    let mut scales_q32 = vec![0u64; w.rows];
    for r in 0..w.rows {
        let base = r * w.cols;
        let (mm, me) = match row_max(w, base).ok()? {
            Some(x) => x,
            None => continue,
        };
        for c in 0..w.cols {
            let (sign, m, e) = w.decompose(base + c).ok()?;
            if m == 0 {
                continue;
            }
            let q = row_code(m, e, mm, me, levels);
            data[base + c] = if sign { -(q as i16) } else { q as i16 };
        }
        scales_q32[r] = row_scale_q32((mm as i128) * super::SCALE as i128, me, levels)?;
    }
    Some(super::RowScaledTQ19::from_parts(w.rows, w.cols, data, scales_q32))
}

/// Quantise to five trits with a per-row scale by exact rational rounding:
/// `s_r = max|w|_r / 121`, `code = round(w / s_r)` (half away from zero),
/// `scale_q32 = round(s_r * 2^32)`. An all-zero row gets codes 0 and scale 0.
#[cfg(table_format = "q16_16")]
pub fn quantize_tq5_rowscaled(w: &WeightBits) -> Option<super::RowScaledTQ5> {
    let levels = super::TQ5_MAX as i128;
    let mut data = vec![0i8; w.rows * w.cols];
    let mut scales_q32 = vec![0u64; w.rows];
    for r in 0..w.rows {
        let base = r * w.cols;
        let (mm, me) = match row_max(w, base).ok()? {
            Some(x) => x,
            None => continue,
        };
        for c in 0..w.cols {
            let (sign, m, e) = w.decompose(base + c).ok()?;
            if m == 0 {
                continue;
            }
            let q = row_code(m, e, mm, me, levels);
            data[base + c] = if sign { -(q as i8) } else { q as i8 };
        }
        // scale_q32 = round(max|w| * 2^32 / 121) = round(mm * 2^(me + 32) / 121)
        scales_q32[r] = row_scale_q32(mm as i128, me, levels)?;
    }
    Some(super::RowScaledTQ5::from_parts(w.rows, w.cols, data, scales_q32))
}
