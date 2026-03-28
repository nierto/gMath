//! # TQ1.9 — Compact Ternary Arithmetic Module
//!
//! Standalone fixed-point ternary operations optimized for throughput.
//! Decoupled from FASC routing, shadow values, and domain dispatch.
//!
//! ## Format
//!
//! TQ1.9 stores values as `i16` scaled by 3^9 = 19683:
//! - 1 integer trit + 9 fractional trits (10 balanced ternary digits)
//! - Range: ±1.5 (±29524/19683)
//! - Precision: ~4.3 uniform decimal digits in 2 bytes
//!
//! ## Operations
//!
//! - [`TQ19Matrix::matvec`] — matrix-vector product with compute-tier accumulation
//! - [`TQ19Matrix::matvec_batch`] — batch matvec (weight matrix stays in cache)
//! - [`tq19_dot`] — single dot product (weights × activations / SCALE)
//! - [`trit_dot`] — zero-multiply dot for pre-decoded trits
//! - [`packed_trit_dot`] — zero-multiply dot for packed trits (5/byte)
//! - [`packed_trit_matvec`] — matvec for packed trit format with per-row scales
//!
//! All operations accumulate at ComputeStorage (tier N+1) with a single
//! division/downscale at the end, matching gMath's precision contract.
//!
//! ## Parallelism
//!
//! With `features = ["parallel"]`, row-parallel variants use rayon:
//! - [`TQ19Matrix::matvec_par`], [`TQ19Matrix::matvec_batch_par`]
//! - [`packed_trit_matvec_par`]
//!
//! ## SIMD
//!
//! On x86_64 with AVX2, the realtime profile (Q16.16, i32 activations) gets
//! hardware-accelerated inner loops:
//! - TQ1.9 dot: 8× multiply-accumulate per cycle via `_mm256_mul_epi32`
//! - Trit dot: 8× zero-multiply per cycle via `_mm256_sign_epi32`
//!
//! Detection is automatic at runtime with scalar fallback.

mod ops;

#[cfg(target_arch = "x86_64")]
pub(crate) mod simd;

use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::imperative::FixedPoint;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================================
// Constants
// ============================================================================

/// TQ1.9 scale factor: 3^9 = 19683.
pub const SCALE: i32 = 19_683;

/// Maximum raw i16 value: (3^10 - 1) / 2.
pub const MAX_RAW: i16 = 29_524;

/// Minimum raw i16 value.
pub const MIN_RAW: i16 = -29_524;

// ============================================================================
// Trit decode table — 256-entry const lookup, 5 balanced trits per byte
// ============================================================================

/// Pre-decoded trit table: maps each byte to 5 balanced trits in {-1, 0, +1}.
///
/// Encoding: `byte = d[0]*81 + d[1]*27 + d[2]*9 + d[3]*3 + d[4]`
/// where `d[i]` ∈ {0,1,2} maps to {-1, 0, +1} via `d - 1`.
///
/// Valid input range: 0..=242 (3^5 - 1). Entries 243..=255 produce
/// undefined trit values and must not be used.
pub const TRIT_DECODE_TABLE: [[i8; 5]; 256] = generate_trit_decode_table();

const fn generate_trit_decode_table() -> [[i8; 5]; 256] {
    let mut table = [[0i8; 5]; 256];
    let mut byte_val: u16 = 0;
    while byte_val < 256 {
        let mut v = byte_val as u8;
        // Unpack least-significant trit first, then reverse for MSB-first order
        let d4 = (v % 3) as i8 - 1; v /= 3;
        let d3 = (v % 3) as i8 - 1; v /= 3;
        let d2 = (v % 3) as i8 - 1; v /= 3;
        let d1 = (v % 3) as i8 - 1; v /= 3;
        let d0 = (v % 3) as i8 - 1;
        table[byte_val as usize] = [d0, d1, d2, d3, d4];
        byte_val += 1;
    }
    table
}

// ============================================================================
// TQ19Matrix — row-major i16 weight matrix
// ============================================================================

/// Row-major TQ1.9 weight matrix.
///
/// Each weight is an `i16` value representing `value * SCALE` in balanced
/// ternary fixed-point. The matrix is stored as a flat `Vec<i16>` in
/// row-major order.
///
/// # Construction
///
/// ```rust,no_run
/// use g_math::fixed_point::tq19::TQ19Matrix;
///
/// // 3×4 matrix from flat data
/// let m = TQ19Matrix::new(3, 4, vec![0i16; 12]);
/// ```
#[derive(Debug, Clone)]
pub struct TQ19Matrix {
    rows: usize,
    cols: usize,
    data: Vec<i16>,
}

impl TQ19Matrix {
    /// Create from flat row-major data.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<i16>) -> Self {
        assert_eq!(data.len(), rows * cols, "TQ19Matrix: data.len() must equal rows × cols");
        Self { rows, cols, data }
    }

    /// Create from a generator function `f(row, col) -> i16`.
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> i16) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self { rows, cols, data }
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize { self.rows }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize { self.cols }

    /// Raw weight data (row-major).
    #[inline]
    pub fn data(&self) -> &[i16] { &self.data }

    /// Slice of weights for a single row.
    #[inline]
    pub fn row_slice(&self, row: usize) -> &[i16] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Get weight at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i16 {
        self.data[row * self.cols + col]
    }

    // ========================================================================
    // Core operations
    // ========================================================================

    /// Matrix-vector product: `result[i] = sum_j(W[i][j] * x[j]) / SCALE`
    ///
    /// Accumulates at ComputeStorage (tier N+1). Single division per row.
    ///
    /// # Panics
    /// Panics if `activations.len() != self.cols()`.
    pub fn matvec(&self, activations: &[BinaryStorage]) -> Vec<BinaryStorage> {
        assert_eq!(activations.len(), self.cols, "TQ19Matrix::matvec: activation length mismatch");
        ops::tq19_matvec(&self.data, self.rows, self.cols, activations)
    }

    /// Batch matrix-vector: same weights applied to multiple activation vectors.
    ///
    /// Weight data stays in cache across batch vectors (weight-centric iteration).
    /// Returns one output vector per input vector.
    ///
    /// # Panics
    /// Panics if any activation vector length != `self.cols()`.
    pub fn matvec_batch(&self, batch: &[&[BinaryStorage]]) -> Vec<Vec<BinaryStorage>> {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(v.len(), self.cols, "TQ19Matrix::matvec_batch: activation[{i}] length mismatch");
        }
        ops::tq19_matvec_batch(&self.data, self.rows, self.cols, batch)
    }

    /// Convenience: matvec returning `FixedPoint` values.
    pub fn matvec_fp(&self, activations: &[BinaryStorage]) -> Vec<FixedPoint> {
        self.matvec(activations).into_iter().map(FixedPoint::from_raw).collect()
    }

    // ========================================================================
    // Parallel variants (rayon feature)
    // ========================================================================

    /// Row-parallel matvec. Each row computed independently via rayon.
    #[cfg(feature = "parallel")]
    pub fn matvec_par(&self, activations: &[BinaryStorage]) -> Vec<BinaryStorage> {
        assert_eq!(activations.len(), self.cols, "TQ19Matrix::matvec_par: activation length mismatch");
        ops::tq19_matvec_par(&self.data, self.rows, self.cols, activations)
    }

    /// Row-parallel batch matvec.
    ///
    /// Parallelizes across rows. Each row processes all batch vectors sequentially
    /// (keeping row weights in L1 cache), then results are transposed.
    #[cfg(feature = "parallel")]
    pub fn matvec_batch_par(&self, batch: &[&[BinaryStorage]]) -> Vec<Vec<BinaryStorage>> {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(v.len(), self.cols, "TQ19Matrix::matvec_batch_par: activation[{i}] length mismatch");
        }
        ops::tq19_matvec_batch_par(&self.data, self.rows, self.cols, batch)
    }
}

// ============================================================================
// Free functions — re-export from ops
// ============================================================================

/// TQ1.9 dot product: `sum(weights[i] * activations[i]) / SCALE`
///
/// Accumulates at ComputeStorage (tier N+1). Single division at end.
/// On x86_64 realtime profile, dispatches to AVX2 when available.
///
/// # Panics
/// Panics (debug) if lengths differ.
#[inline]
pub fn tq19_dot(weights: &[i16], activations: &[BinaryStorage]) -> BinaryStorage {
    ops::tq19_dot(weights, activations)
}

/// Zero-multiply trit dot product for pre-decoded trits.
///
/// Trits must be `i8` values in {-1, 0, +1}. No multiplications —
/// only add, subtract, or skip per element.
///
/// # Panics
/// Panics (debug) if lengths differ.
#[inline]
pub fn trit_dot(trits: &[i8], activations: &[BinaryStorage]) -> BinaryStorage {
    ops::trit_dot(trits, activations)
}

/// Packed trit dot product with per-block scale factor.
///
/// Unpacks 5 trits per byte from base-3 encoding, applies zero-multiply,
/// then multiplies the accumulated result by `scale` at compute tier.
pub fn packed_trit_dot(
    packed: &[u8],
    count: usize,
    activations: &[BinaryStorage],
    scale: BinaryStorage,
) -> BinaryStorage {
    ops::packed_trit_dot(packed, count, activations, scale)
}

/// Packed trit matrix-vector product with per-row scale factors.
///
/// Each row: unpack trits, zero-multiply dot against activations, apply scale.
pub fn packed_trit_matvec(
    packed_trits: &[u8],
    rows: usize,
    cols: usize,
    activations: &[BinaryStorage],
    scales: &[BinaryStorage],
) -> Vec<BinaryStorage> {
    ops::packed_trit_matvec(packed_trits, rows, cols, activations, scales)
}

/// Row-parallel packed trit matvec.
#[cfg(feature = "parallel")]
pub fn packed_trit_matvec_par(
    packed_trits: &[u8],
    rows: usize,
    cols: usize,
    activations: &[BinaryStorage],
    scales: &[BinaryStorage],
) -> Vec<BinaryStorage> {
    ops::packed_trit_matvec_par(packed_trits, rows, cols, activations, scales)
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trit_decode_table_spot_check() {
        // Byte 0 = all Neg: {-1,-1,-1,-1,-1}
        assert_eq!(TRIT_DECODE_TABLE[0], [-1, -1, -1, -1, -1]);

        // Byte 121 = all Zero: 1*81+1*27+1*9+1*3+1 = 121
        assert_eq!(TRIT_DECODE_TABLE[121], [0, 0, 0, 0, 0]);

        // Byte 242 = all Pos: 2*81+2*27+2*9+2*3+2 = 242
        assert_eq!(TRIT_DECODE_TABLE[242], [1, 1, 1, 1, 1]);

        // Byte 1 = {-1,-1,-1,-1,0}: 0*81+0*27+0*9+0*3+1 = 1
        assert_eq!(TRIT_DECODE_TABLE[1], [-1, -1, -1, -1, 0]);

        // Byte 2 = {-1,-1,-1,-1,+1}: 0*81+0*27+0*9+0*3+2 = 2
        assert_eq!(TRIT_DECODE_TABLE[2], [-1, -1, -1, -1, 1]);

        // Byte 3 = {-1,-1,-1,0,-1}: 0*81+0*27+0*9+1*3+0 = 3
        assert_eq!(TRIT_DECODE_TABLE[3], [-1, -1, -1, 0, -1]);
    }

    #[test]
    fn trit_decode_roundtrip() {
        // Verify encode→decode roundtrip for all valid bytes
        for byte in 0u8..=242 {
            let trits = TRIT_DECODE_TABLE[byte as usize];
            // Re-encode: map {-1,0,1} → {0,1,2}, then d[0]*81+d[1]*27+d[2]*9+d[3]*3+d[4]
            let re_encoded = ((trits[0] + 1) as u8) * 81
                + ((trits[1] + 1) as u8) * 27
                + ((trits[2] + 1) as u8) * 9
                + ((trits[3] + 1) as u8) * 3
                + ((trits[4] + 1) as u8);
            assert_eq!(re_encoded, byte, "roundtrip failed for byte {byte}");
        }
    }

    #[test]
    fn tq19_matrix_construction() {
        let m = TQ19Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.get(0, 0), 1);
        assert_eq!(m.get(1, 2), 6);
        assert_eq!(m.row_slice(0), &[1, 2, 3]);
        assert_eq!(m.row_slice(1), &[4, 5, 6]);
    }

    #[test]
    fn tq19_matrix_from_fn() {
        let m = TQ19Matrix::from_fn(3, 3, |r, c| if r == c { SCALE as i16 } else { 0 });
        assert_eq!(m.get(0, 0), SCALE as i16);
        assert_eq!(m.get(0, 1), 0);
        assert_eq!(m.get(1, 1), SCALE as i16);
    }

    #[test]
    #[should_panic(expected = "data.len() must equal rows × cols")]
    fn tq19_matrix_size_mismatch() {
        TQ19Matrix::new(2, 3, vec![0; 5]);
    }
}
