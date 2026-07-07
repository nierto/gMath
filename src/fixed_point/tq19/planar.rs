//! # PlanarTQ19 — trit-plane decomposed TQ1.9 weight matrix
//!
//! Every TQ1.9 weight (`i16` raw, |raw| ≤ 29,524 = (3^10 − 1)/2) is exactly a
//! 10-digit balanced-ternary number:
//!
//! ```text
//! raw = Σₖ 3^k · dₖ          k = 0..9,   dₖ ∈ {-1, 0, +1}
//! ```
//!
//! `PlanarTQ19` stores each digit position k as its own ternary matrix
//! ("plane"), with a per-plane encoding chosen by density:
//!
//! - **Empty**  — no nonzero trits: zero storage.
//! - **Sparse** — density below [`SPARSE_DENSITY_THRESHOLD`]: CSR column
//!   indices, split into positive and negative sets (no sign storage).
//! - **Dense**  — row-aligned packed trits, 5 per byte (1.6 bits/weight),
//!   decoded via [`TRIT_DECODE_TABLE`].
//!
//! For bell-shaped LLM weight distributions the high planes (k ≥ 7) are
//! empty or sparse, giving ~11.4–11.9 bits/weight vs 16 for dense `i16`
//! (measured on Qwen2.5-3B / Llama-3.2-3B) with **bit-identical** matvec
//! results: the per-row accumulator
//!
//! ```text
//! acc = Σₖ 3^k · Σᵢ tₖᵢ · xᵢ  =  Σᵢ wᵢ · xᵢ
//! ```
//!
//! is the same integer as [`TQ19Matrix::matvec`] computes, so the single
//! truncating division by SCALE yields the same result on every profile.

use super::ops::tq19_dot;
use super::{TQ19Matrix, MAX_RAW, MIN_RAW, TRIT_DECODE_TABLE};
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;

use rayon::prelude::*;

// ============================================================================
// Constants
// ============================================================================

/// Number of balanced-ternary digit planes in a TQ1.9 value.
pub const NUM_PLANES: usize = 10;

/// Powers of three, 3^0 .. 3^9.
pub const POW3: [i16; NUM_PLANES] = [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683];

/// Density below which a plane is stored sparse (CSR) instead of dense packed.
///
/// In-memory break-even: dense costs 1.6 bits/weight, sparse costs ~32 bits
/// per nonzero (u32 column index) → 1.6/32 = 0.05.
pub const SPARSE_DENSITY_THRESHOLD: f64 = 0.05;

/// Base-243 packing weights for trit positions within a byte (MSB-first,
/// matching [`TRIT_DECODE_TABLE`]).
const BYTE_WEIGHTS: [u8; 5] = [81, 27, 9, 3, 1];

/// The packed byte encoding five zero trits: 1·81 + 1·27 + 1·9 + 1·3 + 1.
const ZERO_BYTE: u8 = 121;

/// Per-plane weight-contribution LUT for the reconstruction kernel.
///
/// `CONTRIB_LUT[k][byte]` = the five weight contributions `3^k · tⱼ` of that
/// packed byte, padded to 8 i16 lanes (zeros in lanes 5..8) so one entry is
/// exactly one 128-bit register. Reconstruction of a row tile is then a
/// single unaligned load+add+store per byte per plane.
///
/// Size: 10 planes × 256 entries × 16 B = 40 KB static.
/// Bytes 243..=255 are invalid encodings and map to all-zero contributions.
static CONTRIB_LUT: [[[i16; 8]; 256]; NUM_PLANES] = build_contrib_luts();

const fn build_contrib_luts() -> [[[i16; 8]; 256]; NUM_PLANES] {
    let mut luts = [[[0i16; 8]; 256]; NUM_PLANES];
    let mut k = 0;
    while k < NUM_PLANES {
        let w = POW3[k];
        let mut b = 0usize;
        while b < 243 {
            // Decode 5 balanced digits, MSB-first (same order as TRIT_DECODE_TABLE).
            let mut v = b as u16;
            let mut j = 5usize;
            while j > 0 {
                j -= 1;
                luts[k][b][j] = ((v % 3) as i16 - 1) * w;
                v /= 3;
            }
            b += 1;
        }
        k += 1;
    }
    luts
}

// ============================================================================
// Plane storage
// ============================================================================

/// Storage for one trit plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneData {
    /// All trits zero — nothing stored.
    Empty,
    /// Row-aligned packed trits, 5 per byte. Row `r` occupies bytes
    /// `r * bytes_per_row .. (r+1) * bytes_per_row` with zero-trit padding
    /// in the final byte of each row.
    Dense(Vec<u8>),
    /// CSR column indices, positive and negative trits stored separately
    /// so no sign bits are needed. `row_ptr_*` has `rows + 1` entries.
    Sparse {
        row_ptr_pos: Vec<u32>,
        cols_pos: Vec<u32>,
        row_ptr_neg: Vec<u32>,
        cols_neg: Vec<u32>,
    },
}

impl PlaneData {
    /// Heap bytes used by this plane's storage.
    pub fn size_bytes(&self) -> usize {
        match self {
            PlaneData::Empty => 0,
            PlaneData::Dense(bytes) => bytes.len(),
            PlaneData::Sparse { row_ptr_pos, cols_pos, row_ptr_neg, cols_neg } => {
                (row_ptr_pos.len() + cols_pos.len() + row_ptr_neg.len() + cols_neg.len()) * 4
            }
        }
    }
}

// ============================================================================
// PlanarTQ19
// ============================================================================

/// A TQ1.9 weight matrix decomposed into 10 balanced-ternary trit planes.
///
/// Construct with [`PlanarTQ19::from_tq19`]; matvec results are bit-identical
/// to the source [`TQ19Matrix`].
#[derive(Clone, Debug)]
pub struct PlanarTQ19 {
    rows: usize,
    cols: usize,
    /// `planes[k]` holds digit position k (weight 3^k).
    planes: [PlaneData; NUM_PLANES],
}

/// Extract all 10 balanced-ternary digits of a TQ1.9 raw value.
///
/// Exact for the full TQ1.9 range (|raw| ≤ 29,524).
#[inline]
fn bt_digits(raw: i16) -> [i8; NUM_PLANES] {
    debug_assert!((MIN_RAW..=MAX_RAW).contains(&raw));
    let mut digits = [0i8; NUM_PLANES];
    let mut v = raw as i32;
    let mut k = 0;
    while v != 0 {
        let mut d = v.rem_euclid(3);
        if d == 2 {
            d = -1;
        }
        digits[k] = d as i8;
        v = (v - d) / 3;
        k += 1;
    }
    digits
}

impl PlanarTQ19 {
    /// Decompose a [`TQ19Matrix`] into trit planes.
    ///
    /// Lossless: [`PlanarTQ19::to_tq19`] reconstructs the input exactly, and
    /// matvec results are bit-identical.
    pub fn from_tq19(m: &TQ19Matrix) -> Self {
        let rows = m.rows();
        let cols = m.cols();
        let data = m.data();
        let total = rows * cols;

        // Pass 1: count nonzero trits per plane.
        let mut nnz = [0u64; NUM_PLANES];
        for &raw in data {
            let digits = bt_digits(raw);
            for k in 0..NUM_PLANES {
                if digits[k] != 0 {
                    nnz[k] += 1;
                }
            }
        }

        // Choose encoding per plane.
        #[derive(Clone, Copy, PartialEq)]
        enum Mode {
            Empty,
            Sparse,
            Dense,
        }
        let mut modes = [Mode::Empty; NUM_PLANES];
        for k in 0..NUM_PLANES {
            modes[k] = if nnz[k] == 0 {
                Mode::Empty
            } else if (nnz[k] as f64) < SPARSE_DENSITY_THRESHOLD * total as f64 {
                Mode::Sparse
            } else {
                Mode::Dense
            };
        }

        // Pass 2: fill plane storage.
        let bytes_per_row = (cols + 4) / 5;
        let mut dense: [Vec<u8>; NUM_PLANES] = Default::default();
        let mut row_ptr_pos: [Vec<u32>; NUM_PLANES] = Default::default();
        let mut cols_pos: [Vec<u32>; NUM_PLANES] = Default::default();
        let mut row_ptr_neg: [Vec<u32>; NUM_PLANES] = Default::default();
        let mut cols_neg: [Vec<u32>; NUM_PLANES] = Default::default();
        for k in 0..NUM_PLANES {
            match modes[k] {
                Mode::Dense => dense[k].reserve_exact(rows * bytes_per_row),
                Mode::Sparse => {
                    row_ptr_pos[k].reserve_exact(rows + 1);
                    row_ptr_neg[k].reserve_exact(rows + 1);
                    row_ptr_pos[k].push(0);
                    row_ptr_neg[k].push(0);
                }
                Mode::Empty => {}
            }
        }

        // Per-plane byte accumulator for dense packing.
        let mut byte_acc = [0u8; NUM_PLANES];

        for row in 0..rows {
            for col in 0..cols {
                let digits = bt_digits(data[row * cols + col]);
                let pos_in_byte = col % 5;
                for k in 0..NUM_PLANES {
                    match modes[k] {
                        Mode::Dense => {
                            byte_acc[k] += (digits[k] + 1) as u8 * BYTE_WEIGHTS[pos_in_byte];
                            if pos_in_byte == 4 {
                                dense[k].push(byte_acc[k]);
                                byte_acc[k] = 0;
                            }
                        }
                        Mode::Sparse => match digits[k] {
                            1 => cols_pos[k].push(col as u32),
                            -1 => cols_neg[k].push(col as u32),
                            _ => {}
                        },
                        Mode::Empty => {}
                    }
                }
            }
            // Flush row remainder: pad with zero trits (digit 0 → +1 offset).
            let rem = cols % 5;
            if rem != 0 {
                for k in 0..NUM_PLANES {
                    if modes[k] == Mode::Dense {
                        for pad in rem..5 {
                            byte_acc[k] += BYTE_WEIGHTS[pad];
                        }
                        dense[k].push(byte_acc[k]);
                        byte_acc[k] = 0;
                    }
                }
            }
            for k in 0..NUM_PLANES {
                if modes[k] == Mode::Sparse {
                    row_ptr_pos[k].push(cols_pos[k].len() as u32);
                    row_ptr_neg[k].push(cols_neg[k].len() as u32);
                }
            }
        }

        let mut planes: [PlaneData; NUM_PLANES] =
            std::array::from_fn(|_| PlaneData::Empty);
        for k in 0..NUM_PLANES {
            planes[k] = match modes[k] {
                Mode::Empty => PlaneData::Empty,
                Mode::Dense => PlaneData::Dense(std::mem::take(&mut dense[k])),
                Mode::Sparse => PlaneData::Sparse {
                    row_ptr_pos: std::mem::take(&mut row_ptr_pos[k]),
                    cols_pos: std::mem::take(&mut cols_pos[k]),
                    row_ptr_neg: std::mem::take(&mut row_ptr_neg[k]),
                    cols_neg: std::mem::take(&mut cols_neg[k]),
                },
            };
        }

        Self { rows, cols, planes }
    }

    /// Reconstruct the original dense [`TQ19Matrix`] (lossless inverse).
    pub fn to_tq19(&self) -> TQ19Matrix {
        let mut data = vec![0i16; self.rows * self.cols];
        let bytes_per_row = (self.cols + 4) / 5;

        for (k, plane) in self.planes.iter().enumerate() {
            let w = POW3[k];
            match plane {
                PlaneData::Empty => {}
                PlaneData::Dense(bytes) => {
                    for row in 0..self.rows {
                        let base = row * bytes_per_row;
                        for (byte_idx, &b) in
                            bytes[base..base + bytes_per_row].iter().enumerate()
                        {
                            let trits = TRIT_DECODE_TABLE[b as usize];
                            for (j, &t) in trits.iter().enumerate() {
                                let col = byte_idx * 5 + j;
                                if col < self.cols && t != 0 {
                                    data[row * self.cols + col] += t as i16 * w;
                                }
                            }
                        }
                    }
                }
                PlaneData::Sparse { row_ptr_pos, cols_pos, row_ptr_neg, cols_neg } => {
                    for row in 0..self.rows {
                        for &c in &cols_pos
                            [row_ptr_pos[row] as usize..row_ptr_pos[row + 1] as usize]
                        {
                            data[row * self.cols + c as usize] += w;
                        }
                        for &c in &cols_neg
                            [row_ptr_neg[row] as usize..row_ptr_neg[row + 1] as usize]
                        {
                            data[row * self.cols + c as usize] -= w;
                        }
                    }
                }
            }
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

    /// Access the plane storage (for serialization by consumers).
    pub fn planes(&self) -> &[PlaneData; NUM_PLANES] {
        &self.planes
    }

    /// Construct from raw parts (for deserialization by consumers).
    ///
    /// The caller is responsible for supplying planes that were produced by
    /// [`PlanarTQ19::from_tq19`] (or an equivalent encoder) for a matrix of
    /// the given shape.
    pub fn from_parts(rows: usize, cols: usize, planes: [PlaneData; NUM_PLANES]) -> Self {
        Self { rows, cols, planes }
    }

    /// Total heap bytes of plane storage.
    pub fn size_bytes(&self) -> usize {
        self.planes.iter().map(PlaneData::size_bytes).sum()
    }

    // ========================================================================
    // Matvec — bit-identical to TQ19Matrix::matvec
    //
    // Kernel: reconstruct each row's exact i16 weights into a thread-local
    // buffer (one SIMD load+add per packed byte per plane via CONTRIB_LUT,
    // plus scattered adds for sparse planes), then run the standard tq19_dot
    // (AVX2 on realtime). Reconstruction is exact — buf[j] = Σₖ 3^k·tₖⱼ is
    // the original i16 raw value — so results are bit-identical to the dense
    // path by definition, on every profile.
    // ========================================================================

    /// Reconstruct one row's exact i16 weights into `buf`.
    ///
    /// `buf.len()` must be at least `self.cols + 7`: the dense-plane kernel
    /// writes 8 lanes per 5-weight step (lanes 5..8 of each LUT entry are
    /// zero), overhanging up to 7 slots past the last real column.
    fn reconstruct_row_into(&self, row: usize, buf: &mut [i16]) {
        debug_assert!(buf.len() >= self.cols + 7);
        buf.fill(0);
        let bytes_per_row = (self.cols + 4) / 5;

        for (k, plane) in self.planes.iter().enumerate() {
            match plane {
                PlaneData::Empty => {}
                PlaneData::Dense(bytes) => {
                    let lut = &CONTRIB_LUT[k];
                    let base = row * bytes_per_row;
                    let row_bytes = &bytes[base..base + bytes_per_row];

                    #[cfg(target_arch = "x86_64")]
                    // SSE2 is baseline on x86_64 — no runtime detection needed.
                    unsafe {
                        use std::arch::x86_64::*;
                        let mut j = 0usize;
                        for &b in row_bytes {
                            if b != ZERO_BYTE {
                                let contrib =
                                    _mm_loadu_si128(lut[b as usize].as_ptr() as *const __m128i);
                                let dst = buf.as_mut_ptr().add(j) as *mut __m128i;
                                let cur = _mm_loadu_si128(dst as *const __m128i);
                                _mm_storeu_si128(dst, _mm_add_epi16(cur, contrib));
                            }
                            j += 5;
                        }
                    }

                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        let mut j = 0usize;
                        for &b in row_bytes {
                            if b != ZERO_BYTE {
                                let c = &lut[b as usize];
                                for t in 0..5 {
                                    buf[j + t] += c[t];
                                }
                            }
                            j += 5;
                        }
                    }
                }
                PlaneData::Sparse { row_ptr_pos, cols_pos, row_ptr_neg, cols_neg } => {
                    let w = POW3[k];
                    for &c in
                        &cols_pos[row_ptr_pos[row] as usize..row_ptr_pos[row + 1] as usize]
                    {
                        buf[c as usize] += w;
                    }
                    for &c in
                        &cols_neg[row_ptr_neg[row] as usize..row_ptr_neg[row + 1] as usize]
                    {
                        buf[c as usize] -= w;
                    }
                }
            }
        }
    }

    /// Buffer length for `reconstruct_row_into` (cols + 8-lane overhang pad).
    #[inline]
    fn buf_len(&self) -> usize {
        self.cols + 8
    }

    /// Matrix-vector product: `result[i] = sum_j(W[i][j] * x[j]) / SCALE`.
    ///
    /// Bit-identical to [`TQ19Matrix::matvec`]: each row is reconstructed to
    /// its exact i16 weights, then dotted with the standard TQ1.9 kernel
    /// (AVX2 on the realtime profile).
    ///
    /// # Panics
    /// Panics if `activations.len() != self.cols()`.
    pub fn matvec(&self, activations: &[BinaryStorage]) -> Vec<BinaryStorage> {
        assert_eq!(
            activations.len(),
            self.cols,
            "PlanarTQ19::matvec: activation length mismatch"
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
            "PlanarTQ19::matvec_par: activation length mismatch"
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

    /// Batch matvec: same weights applied to multiple activation vectors.
    ///
    /// Each row is reconstructed **once** and dotted against every batch
    /// vector — reconstruction cost amortizes across the batch.
    ///
    /// # Panics
    /// Panics if any activation vector length != `self.cols()`.
    pub fn matvec_batch(&self, batch: &[&[BinaryStorage]]) -> Vec<Vec<BinaryStorage>> {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(
                v.len(),
                self.cols,
                "PlanarTQ19::matvec_batch: activation[{i}] length mismatch"
            );
        }
        let mut buf = vec![0i16; self.buf_len()];
        let results_by_row: Vec<Vec<BinaryStorage>> = (0..self.rows)
            .map(|row| {
                self.reconstruct_row_into(row, &mut buf);
                let w = &buf[..self.cols];
                batch.iter().map(|acts| tq19_dot(w, acts)).collect()
            })
            .collect();
        transpose(results_by_row, batch.len())
    }

    /// Row-parallel batch matvec: rows in parallel, reconstruction amortized
    /// across the batch within each row.
    pub fn matvec_batch_par(&self, batch: &[&[BinaryStorage]]) -> Vec<Vec<BinaryStorage>> {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(
                v.len(),
                self.cols,
                "PlanarTQ19::matvec_batch_par: activation[{i}] length mismatch"
            );
        }
        let results_by_row: Vec<Vec<BinaryStorage>> = (0..self.rows)
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
        transpose(results_by_row, batch.len())
    }
}

/// Transpose row-major per-row results into one output vector per batch input.
fn transpose(results_by_row: Vec<Vec<BinaryStorage>>, batch_len: usize) -> Vec<Vec<BinaryStorage>> {
    (0..batch_len)
        .map(|b| results_by_row.iter().map(|r| r[b]).collect())
        .collect()
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::imperative::FixedPoint;

    /// Deterministic LCG so tests are reproducible without RNG deps.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        /// Raw TQ1.9 value across the full ±29,524 range.
        fn raw(&mut self) -> i16 {
            ((self.next() % (2 * MAX_RAW as u64 + 1)) as i64 - MAX_RAW as i64) as i16
        }
        /// Bell-ish small raw value (sum of two smaller uniforms).
        fn raw_small(&mut self) -> i16 {
            let a = (self.next() % 2001) as i64 - 1000;
            let b = (self.next() % 2001) as i64 - 1000;
            (a + b) as i16
        }
        /// Profile-portable activation: integer FixedPoint in ±1000.
        fn activation(&mut self) -> BinaryStorage {
            let n = (self.next() % 2001) as i32 - 1000;
            FixedPoint::from_int(n).raw()
        }
    }

    #[test]
    fn bt_digits_reconstruct() {
        // Every raw value in the full TQ1.9 range decomposes exactly.
        for raw in MIN_RAW..=MAX_RAW {
            let digits = bt_digits(raw);
            let mut v: i32 = 0;
            for k in (0..NUM_PLANES).rev() {
                v = v * 3 + digits[k] as i32;
            }
            assert_eq!(v, raw as i32, "raw {raw} did not reconstruct");
        }
    }

    #[test]
    fn roundtrip_full_range() {
        let mut rng = Lcg(42);
        let (rows, cols) = (17, 23); // cols not a multiple of 5
        let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw()).collect();
        let m = TQ19Matrix::new(rows, cols, data.clone());
        let planar = PlanarTQ19::from_tq19(&m);
        assert_eq!(planar.to_tq19().data(), m.data());
    }

    #[test]
    fn roundtrip_bell_distribution() {
        let mut rng = Lcg(7);
        let (rows, cols) = (32, 64);
        let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw_small()).collect();
        let m = TQ19Matrix::new(rows, cols, data.clone());
        let planar = PlanarTQ19::from_tq19(&m);
        assert_eq!(planar.to_tq19().data(), m.data());
        // Bell-shaped data must engage the sparse/empty paths for high planes
        // and compress below dense i16.
        assert!(matches!(planar.planes()[9], PlaneData::Empty | PlaneData::Sparse { .. }));
        assert!(planar.size_bytes() < rows * cols * 2, "no compression on bell data");
    }

    #[test]
    fn matvec_bit_identical_full_range() {
        let mut rng = Lcg(1234);
        for &(rows, cols) in &[(3usize, 5usize), (8, 17), (16, 128), (5, 3)] {
            let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw()).collect();
            let m = TQ19Matrix::new(rows, cols, data);
            let planar = PlanarTQ19::from_tq19(&m);
            let acts: Vec<BinaryStorage> = (0..cols).map(|_| rng.activation()).collect();
            assert_eq!(planar.matvec(&acts), m.matvec(&acts), "{rows}x{cols} mismatch");
        }
    }

    #[test]
    fn matvec_bit_identical_bell_distribution() {
        // Bell data exercises Empty + Sparse + Dense plane mix.
        let mut rng = Lcg(99);
        let (rows, cols) = (24, 96);
        let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw_small()).collect();
        let m = TQ19Matrix::new(rows, cols, data);
        let planar = PlanarTQ19::from_tq19(&m);
        let acts: Vec<BinaryStorage> = (0..cols).map(|_| rng.activation()).collect();
        assert_eq!(planar.matvec(&acts), m.matvec(&acts));
    }

    #[test]
    fn matvec_bit_identical_sparse_only() {
        // A few large weights → only high planes populated, all sparse.
        let (rows, cols) = (6, 40);
        let mut data = vec![0i16; rows * cols];
        data[3] = 19683; // +3^9
        data[45] = -6561; // -3^8
        data[200] = 2187 + 729; // 3^7 + 3^6
        let m = TQ19Matrix::new(rows, cols, data);
        let planar = PlanarTQ19::from_tq19(&m);
        let mut rng = Lcg(5);
        let acts: Vec<BinaryStorage> = (0..cols).map(|_| rng.activation()).collect();
        assert_eq!(planar.matvec(&acts), m.matvec(&acts));
        assert_eq!(planar.planes()[0], PlaneData::Empty);
    }

    #[test]
    fn parallel_and_batch_match_sequential() {
        let mut rng = Lcg(2026);
        let (rows, cols) = (19, 37);
        let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw_small()).collect();
        let m = TQ19Matrix::new(rows, cols, data);
        let planar = PlanarTQ19::from_tq19(&m);

        let acts: Vec<BinaryStorage> = (0..cols).map(|_| rng.activation()).collect();
        assert_eq!(planar.matvec_par(&acts), planar.matvec(&acts));

        let batch_data: Vec<Vec<BinaryStorage>> = (0..4)
            .map(|_| (0..cols).map(|_| rng.activation()).collect())
            .collect();
        let batch: Vec<&[BinaryStorage]> = batch_data.iter().map(|v| v.as_slice()).collect();
        let expected: Vec<Vec<BinaryStorage>> = batch.iter().map(|v| m.matvec(v)).collect();
        assert_eq!(planar.matvec_batch(&batch), expected);
        assert_eq!(planar.matvec_batch_par(&batch), expected);
    }

    #[test]
    fn from_parts_roundtrip() {
        let mut rng = Lcg(11);
        let (rows, cols) = (9, 14);
        let data: Vec<i16> = (0..rows * cols).map(|_| rng.raw_small()).collect();
        let m = TQ19Matrix::new(rows, cols, data);
        let planar = PlanarTQ19::from_tq19(&m);
        let rebuilt = PlanarTQ19::from_parts(rows, cols, planar.planes().clone());
        assert_eq!(rebuilt.to_tq19().data(), m.data());
    }
}
