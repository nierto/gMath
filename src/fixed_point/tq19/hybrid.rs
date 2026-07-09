//! # HybridTQ19 — 12-bit low part + sparse high corrections
//!
//! The batch-1 performance form of the trit-plane idea (see `planar.rs`).
//! Measured plane densities on real LLM tensors show digit planes k=0..6 are
//! near-maximum entropy (incompressible noise) while k=7..9 are sparse or
//! empty. Separating the low planes therefore buys nothing — so this format
//! stores them **fused** as one 12-bit field and keeps only the high digits
//! as sparse corrections:
//!
//! ```text
//! raw = hi·3^7 + lo,   lo ∈ [-1093, +1093] (balanced remainder mod 2187),
//!                      hi ∈ [-13, +13]     (zero for ~96-99% of weights)
//! ```
//!
//! - **Low part**: `lo + 1093 ∈ [0, 2186]` packed 12 bits/weight in blocks of
//!   16 weights = 24 bytes (16 low-bytes, then 8 nibble-bytes where byte j
//!   holds the high nibbles of weights j and j+8). One block unpacks to 16
//!   exact i16 lanes in ~11 SSE2 ops.
//! - **High part**: CSR per row, column index (u32) + correction value (i8),
//!   applied as `buf[col] += hi · 2187`.
//!
//! Size: 12 bits/weight + ~40 bits per high-correction ≈ 12.4–13.4 bits/weight
//! on measured models (vs 16 dense, ~11.4-11.9 for trit planes). Reconstruction
//! is ~10× cheaper than the plane kernel — this is the format intended to
//! reach batch-1 dense parity while keeping the bandwidth win.
//!
//! Matvec is bit-identical to [`TQ19Matrix::matvec`]: rows are reconstructed
//! to their exact i16 weights, then the standard `tq19_dot` runs.

use super::ops::tq19_dot;
use super::{TQ19Matrix, MAX_RAW, MIN_RAW};
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;

use rayon::prelude::*;

// ============================================================================
// Constants
// ============================================================================

/// Number of low balanced-ternary digits fused into the 12-bit field.
pub const HYBRID_LOW_TRITS: usize = 7;

/// 3^7 — modulus of the low part.
pub const LOW_MOD: i32 = 2187;

/// Bias added to the balanced low remainder: biased = lo + 1093 ∈ [0, 2186].
pub const LOW_BIAS: i16 = 1093;

/// Weights per packed block.
const BLOCK_WEIGHTS: usize = 16;

/// Bytes per packed block: 16 low-bytes + 8 nibble-bytes.
const BLOCK_BYTES: usize = 24;

// ============================================================================
// HybridTQ19
// ============================================================================

/// A TQ1.9 weight matrix in hybrid 12-bit + sparse-correction form.
///
/// Construct with [`HybridTQ19::from_tq19`]; matvec results are bit-identical
/// to the source [`TQ19Matrix`].
#[derive(Clone, Debug)]
pub struct HybridTQ19 {
    rows: usize,
    cols: usize,
    /// Packed 12-bit low parts, row-aligned: each row occupies
    /// `blocks_per_row * BLOCK_BYTES` bytes. Padding weights encode zero.
    packed: Vec<u8>,
    /// CSR row pointers into `hi_cols`/`hi_vals` (`rows + 1` entries).
    hi_row_ptr: Vec<u32>,
    /// Column indices of nonzero high corrections.
    hi_cols: Vec<u32>,
    /// High correction values (multiplied by 3^7 at reconstruction).
    hi_vals: Vec<i8>,
}

/// Split a raw TQ1.9 value into (hi, biased_lo).
///
/// `raw = hi·2187 + (biased_lo - 1093)`, exact for the full TQ1.9 range.
#[inline]
fn split_raw(raw: i16) -> (i8, u16) {
    debug_assert!((MIN_RAW..=MAX_RAW).contains(&raw));
    let r = (raw as i32).rem_euclid(LOW_MOD); // 0..2186
    let lo = if r > LOW_BIAS as i32 { r - LOW_MOD } else { r };
    let hi = (raw as i32 - lo) / LOW_MOD;
    (hi as i8, (lo + LOW_BIAS as i32) as u16)
}

impl HybridTQ19 {
    /// Convert a dense [`TQ19Matrix`] (lossless; see `to_tq19`).
    pub fn from_tq19(m: &TQ19Matrix) -> Self {
        let rows = m.rows();
        let cols = m.cols();
        let data = m.data();
        let blocks_per_row = (cols + BLOCK_WEIGHTS - 1) / BLOCK_WEIGHTS;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;

        let mut packed = vec![0u8; rows * bytes_per_row];
        let mut hi_row_ptr = Vec::with_capacity(rows + 1);
        let mut hi_cols = Vec::new();
        let mut hi_vals = Vec::new();
        hi_row_ptr.push(0);

        let mut biased = [LOW_BIAS as u16; BLOCK_WEIGHTS];
        for row in 0..rows {
            for blk in 0..blocks_per_row {
                let base_col = blk * BLOCK_WEIGHTS;
                for j in 0..BLOCK_WEIGHTS {
                    let col = base_col + j;
                    if col < cols {
                        let (hi, b) = split_raw(data[row * cols + col]);
                        biased[j] = b;
                        if hi != 0 {
                            hi_cols.push(col as u32);
                            hi_vals.push(hi);
                        }
                    } else {
                        biased[j] = LOW_BIAS as u16; // padding encodes zero
                    }
                }
                let out = &mut packed[row * bytes_per_row + blk * BLOCK_BYTES..];
                for j in 0..BLOCK_WEIGHTS {
                    out[j] = (biased[j] & 0xFF) as u8;
                }
                for j in 0..8 {
                    let n_lo = (biased[j] >> 8) as u8; // weight j
                    let n_hi = (biased[j + 8] >> 8) as u8; // weight j+8
                    out[BLOCK_WEIGHTS + j] = n_lo | (n_hi << 4);
                }
            }
            hi_row_ptr.push(hi_cols.len() as u32);
        }

        Self { rows, cols, packed, hi_row_ptr, hi_cols, hi_vals }
    }

    /// Reconstruct the original dense [`TQ19Matrix`] (lossless inverse).
    pub fn to_tq19(&self) -> TQ19Matrix {
        let mut data = vec![0i16; self.rows * self.cols];
        let mut buf = vec![0i16; self.buf_len()];
        for row in 0..self.rows {
            self.reconstruct_row_into(row, &mut buf);
            data[row * self.cols..(row + 1) * self.cols].copy_from_slice(&buf[..self.cols]);
        }
        TQ19Matrix::new(self.rows, self.cols, data)
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Total heap bytes (packed low parts + CSR high corrections).
    pub fn size_bytes(&self) -> usize {
        self.packed.len()
            + self.hi_row_ptr.len() * 4
            + self.hi_cols.len() * 4
            + self.hi_vals.len()
    }

    /// Number of nonzero high corrections (diagnostics).
    pub fn num_high_corrections(&self) -> usize {
        self.hi_vals.len()
    }

    /// Raw parts accessor for consumer serialization:
    /// `(packed, hi_row_ptr, hi_cols, hi_vals)`.
    pub fn parts(&self) -> (&[u8], &[u32], &[u32], &[i8]) {
        (&self.packed, &self.hi_row_ptr, &self.hi_cols, &self.hi_vals)
    }

    /// Construct from raw parts (consumer deserialization). The parts must
    /// have been produced by [`HybridTQ19::from_tq19`] for the same shape.
    pub fn from_parts(
        rows: usize,
        cols: usize,
        packed: Vec<u8>,
        hi_row_ptr: Vec<u32>,
        hi_cols: Vec<u32>,
        hi_vals: Vec<i8>,
    ) -> Self {
        Self { rows, cols, packed, hi_row_ptr, hi_cols, hi_vals }
    }

    // ========================================================================
    // Reconstruction + matvec
    // ========================================================================

    /// Buffer length for reconstruction: full blocks (up to 15 lanes overhang).
    #[inline]
    fn buf_len(&self) -> usize {
        ((self.cols + BLOCK_WEIGHTS - 1) / BLOCK_WEIGHTS) * BLOCK_WEIGHTS
    }

    /// Reconstruct one row's exact i16 weights into `buf`
    /// (`buf.len() >= self.buf_len()`). Every block lane is overwritten, so
    /// no pre-zeroing is needed; padding lanes decode to 0.
    fn reconstruct_row_into(&self, row: usize, buf: &mut [i16]) {
        let blocks_per_row = (self.cols + BLOCK_WEIGHTS - 1) / BLOCK_WEIGHTS;
        let bytes_per_row = blocks_per_row * BLOCK_BYTES;
        let row_bytes = &self.packed[row * bytes_per_row..(row + 1) * bytes_per_row];
        debug_assert!(buf.len() >= blocks_per_row * BLOCK_WEIGHTS);

        #[cfg(target_arch = "x86_64")]
        // SSE2 is baseline on x86_64 — no runtime detection needed.
        unsafe {
            use std::arch::x86_64::*;
            let zero = _mm_setzero_si128();
            let bias = _mm_set1_epi16(LOW_BIAS);
            let nib_mask = _mm_set1_epi8(0x0F);
            for blk in 0..blocks_per_row {
                let p = row_bytes.as_ptr().add(blk * BLOCK_BYTES);
                let lo = _mm_loadu_si128(p as *const __m128i); // 16 low-bytes
                let nb = _mm_loadl_epi64(p.add(BLOCK_WEIGHTS) as *const __m128i); // 8 nibble-bytes
                // Low nibbles → weights 0..8, high nibbles → weights 8..16.
                // (srli_epi16 shifts across byte pairs; the & 0x0F per byte
                // discards the bits that crossed the boundary.)
                let n_a = _mm_and_si128(nb, nib_mask);
                let n_b = _mm_and_si128(_mm_srli_epi16(nb, 4), nib_mask);
                let w_a = _mm_sub_epi16(
                    _mm_or_si128(
                        _mm_unpacklo_epi8(lo, zero),
                        _mm_slli_epi16(_mm_unpacklo_epi8(n_a, zero), 8),
                    ),
                    bias,
                );
                let w_b = _mm_sub_epi16(
                    _mm_or_si128(
                        _mm_unpackhi_epi8(lo, zero),
                        _mm_slli_epi16(_mm_unpacklo_epi8(n_b, zero), 8),
                    ),
                    bias,
                );
                let dst = buf.as_mut_ptr().add(blk * BLOCK_WEIGHTS);
                _mm_storeu_si128(dst as *mut __m128i, w_a);
                _mm_storeu_si128(dst.add(8) as *mut __m128i, w_b);
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            for blk in 0..blocks_per_row {
                let p = &row_bytes[blk * BLOCK_BYTES..];
                for j in 0..8 {
                    let n = p[BLOCK_WEIGHTS + j];
                    let v_a = p[j] as u16 | (((n & 0x0F) as u16) << 8);
                    let v_b = p[j + 8] as u16 | (((n >> 4) as u16) << 8);
                    buf[blk * BLOCK_WEIGHTS + j] = v_a as i16 - LOW_BIAS;
                    buf[blk * BLOCK_WEIGHTS + j + 8] = v_b as i16 - LOW_BIAS;
                }
            }
        }

        // Sparse high corrections: buf[col] += hi · 3^7.
        let start = self.hi_row_ptr[row] as usize;
        let end = self.hi_row_ptr[row + 1] as usize;
        for i in start..end {
            buf[self.hi_cols[i] as usize] += self.hi_vals[i] as i16 * LOW_MOD as i16;
        }
    }

    /// Matrix-vector product, bit-identical to [`TQ19Matrix::matvec`].
    ///
    /// # Panics
    /// Panics if `activations.len() != self.cols()`.
    pub fn matvec(&self, activations: &[BinaryStorage]) -> Vec<BinaryStorage> {
        assert_eq!(
            activations.len(),
            self.cols,
            "HybridTQ19::matvec: activation length mismatch"
        );
        let mut buf = vec![0i16; self.buf_len()];
        (0..self.rows)
            .map(|row| {
                self.reconstruct_row_into(row, &mut buf);
                tq19_dot(&buf[..self.cols], activations)
            })
            .collect()
    }

    /// Row-parallel matvec (rayon). One reconstruction buffer per worker.
    pub fn matvec_par(&self, activations: &[BinaryStorage]) -> Vec<BinaryStorage> {
        assert_eq!(
            activations.len(),
            self.cols,
            "HybridTQ19::matvec_par: activation length mismatch"
        );
        (0..self.rows)
            .into_par_iter()
            .map_init(
                || vec![0i16; self.buf_len()],
                |buf, row| {
                    self.reconstruct_row_into(row, buf);
                    tq19_dot(&buf[..self.cols], activations)
                },
            )
            .collect()
    }

    /// Batch matvec: each row reconstructed once, dotted per batch vector.
    ///
    /// # Panics
    /// Panics if any activation vector length != `self.cols()`.
    pub fn matvec_batch(&self, batch: &[&[BinaryStorage]]) -> Vec<Vec<BinaryStorage>> {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(
                v.len(),
                self.cols,
                "HybridTQ19::matvec_batch: activation[{i}] length mismatch"
            );
        }
        let mut buf = vec![0i16; self.buf_len()];
        let by_row: Vec<Vec<BinaryStorage>> = (0..self.rows)
            .map(|row| {
                self.reconstruct_row_into(row, &mut buf);
                let w = &buf[..self.cols];
                batch.iter().map(|acts| tq19_dot(w, acts)).collect()
            })
            .collect();
        transpose(by_row, batch.len())
    }

    /// Row-parallel batch matvec.
    pub fn matvec_batch_par(&self, batch: &[&[BinaryStorage]]) -> Vec<Vec<BinaryStorage>> {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(
                v.len(),
                self.cols,
                "HybridTQ19::matvec_batch_par: activation[{i}] length mismatch"
            );
        }
        let by_row: Vec<Vec<BinaryStorage>> = (0..self.rows)
            .into_par_iter()
            .map_init(
                || vec![0i16; self.buf_len()],
                |buf, row| {
                    self.reconstruct_row_into(row, buf);
                    let w = &buf[..self.cols];
                    batch.iter().map(|acts| tq19_dot(w, acts)).collect()
                },
            )
            .collect();
        transpose(by_row, batch.len())
    }
}

/// Transpose row-major per-row results into one output vector per batch input.
fn transpose(by_row: Vec<Vec<BinaryStorage>>, batch_len: usize) -> Vec<Vec<BinaryStorage>> {
    (0..batch_len)
        .map(|b| by_row.iter().map(|r| r[b]).collect())
        .collect()
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::imperative::FixedPoint;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn raw(&mut self) -> i16 {
            ((self.next() % (2 * MAX_RAW as u64 + 1)) as i64 - MAX_RAW as i64) as i16
        }
        fn raw_small(&mut self) -> i16 {
            let a = (self.next() % 2001) as i64 - 1000;
            let b = (self.next() % 2001) as i64 - 1000;
            (a + b) as i16
        }
        fn activation(&mut self) -> BinaryStorage {
            let n = (self.next() % 2001) as i32 - 1000;
            FixedPoint::from_int(n).raw()
        }
    }

    #[test]
    fn split_raw_reconstructs_exhaustively() {
        for raw in MIN_RAW..=MAX_RAW {
            let (hi, biased) = split_raw(raw);
            assert!(biased <= 2186, "biased out of 12-bit range for {raw}");
            assert!((-13..=13).contains(&(hi as i32)), "hi out of range for {raw}");
            let lo = biased as i32 - LOW_BIAS as i32;
            assert_eq!(hi as i32 * LOW_MOD + lo, raw as i32, "raw {raw} split mismatch");
        }
    }

    #[test]
    fn roundtrip_full_range() {
        let mut rng = Lcg(42);
        for &(rows, cols) in &[(7usize, 33usize), (16, 16), (5, 3), (3, 100)] {
            let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw()).collect();
            let m = TQ19Matrix::new(rows, cols, data);
            let h = HybridTQ19::from_tq19(&m);
            assert_eq!(h.to_tq19().data(), m.data(), "{rows}x{cols} roundtrip");
        }
    }

    #[test]
    fn roundtrip_bell_and_compresses() {
        let mut rng = Lcg(7);
        let (rows, cols) = (32, 64);
        // Wide bell (±2000): exercises the high-correction path heavily.
        let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw_small()).collect();
        let m = TQ19Matrix::new(rows, cols, data.clone());
        let h = HybridTQ19::from_tq19(&m);
        assert_eq!(h.to_tq19().data(), m.data());

        // Realistic bell (σ≈400, like measured LLM tensors: ~1-4% corrections):
        // must compress below dense i16.
        let narrow: Vec<i16> = (0..rows * cols)
            .map(|_| {
                let a = (rng.next() % 1001) as i64 - 500;
                let b = (rng.next() % 1001) as i64 - 500;
                (a + b) as i16
            })
            .collect();
        let m2 = TQ19Matrix::new(rows, cols, narrow.clone());
        let h2 = HybridTQ19::from_tq19(&m2);
        assert_eq!(h2.to_tq19().data(), m2.data());
        assert!(
            h2.size_bytes() < rows * cols * 2,
            "no compression on realistic bell data: {} vs {}",
            h2.size_bytes(),
            rows * cols * 2
        );
    }

    #[test]
    fn matvec_bit_identical() {
        let mut rng = Lcg(1234);
        for &(rows, cols) in &[(3usize, 5usize), (8, 17), (16, 128), (9, 48), (5, 3)] {
            let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw()).collect();
            let m = TQ19Matrix::new(rows, cols, data);
            let h = HybridTQ19::from_tq19(&m);
            let acts: Vec<BinaryStorage> = (0..cols).map(|_| rng.activation()).collect();
            assert_eq!(h.matvec(&acts), m.matvec(&acts), "{rows}x{cols} mismatch");
        }
    }

    #[test]
    fn matvec_bit_identical_bell() {
        let mut rng = Lcg(99);
        let (rows, cols) = (24, 96);
        let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw_small()).collect();
        let m = TQ19Matrix::new(rows, cols, data);
        let h = HybridTQ19::from_tq19(&m);
        let acts: Vec<BinaryStorage> = (0..cols).map(|_| rng.activation()).collect();
        assert_eq!(h.matvec(&acts), m.matvec(&acts));
    }

    #[test]
    fn parallel_and_batch_match_sequential() {
        let mut rng = Lcg(2026);
        let (rows, cols) = (19, 37);
        let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw_small()).collect();
        let m = TQ19Matrix::new(rows, cols, data);
        let h = HybridTQ19::from_tq19(&m);

        let acts: Vec<BinaryStorage> = (0..cols).map(|_| rng.activation()).collect();
        assert_eq!(h.matvec_par(&acts), h.matvec(&acts));

        let batch_data: Vec<Vec<BinaryStorage>> = (0..4)
            .map(|_| (0..cols).map(|_| rng.activation()).collect())
            .collect();
        let batch: Vec<&[BinaryStorage]> = batch_data.iter().map(|v| v.as_slice()).collect();
        let expected: Vec<Vec<BinaryStorage>> = batch.iter().map(|v| m.matvec(v)).collect();
        assert_eq!(h.matvec_batch(&batch), expected);
        assert_eq!(h.matvec_batch_par(&batch), expected);
    }

    #[test]
    fn from_parts_roundtrip() {
        let mut rng = Lcg(11);
        let (rows, cols) = (9, 30);
        let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw_small()).collect();
        let m = TQ19Matrix::new(rows, cols, data);
        let h = HybridTQ19::from_tq19(&m);
        let (packed, rp, hc, hv) = h.parts();
        let rebuilt = HybridTQ19::from_parts(
            rows,
            cols,
            packed.to_vec(),
            rp.to_vec(),
            hc.to_vec(),
            hv.to_vec(),
        );
        assert_eq!(rebuilt.to_tq19().data(), m.data());
    }
}
