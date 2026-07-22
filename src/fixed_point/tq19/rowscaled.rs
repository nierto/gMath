//! Row-scaled TQ1.9 ("TQ1.9-R") — per-row quantization scales.
//!
//! Motivation (Maniference O27, 2026-07-21): plain TQ19 quantizes every
//! tensor with ONE global step (1/SCALE ≈ 5.1e-5, zero-floor 2.5e-5).
//! BF16-trained models (Mixtral) put substantial weight mass below that
//! floor; the measured consequence is a 0.207%/layer router top-2 flip
//! rate that compounds to ~6.4% of tokens executing a wrong expert.
//! A per-row scale adapts the step to each row's own max: measured on
//! Mixtral, matvec output error drops ~20× and wrong-expert tokens drop
//! 6.4% → 0.22%, at unchanged 2 bytes/weight plus one i128
//! multiply-shift per output element.
//!
//! Semantics: `row_value = raw × s_row`, with `raw = round(w / s_row)`,
//! `s_row = max|w_row| / MAX_RAW`. Scales are stored RELATIVE to the
//! global step as unsigned Q32.32: `s_rel = s_row × SCALE`, so
//! `matvec_out = tq19_dot(raw_row, x) × s_rel` — the existing (SIMD)
//! `tq19_dot` is reused verbatim and the scale is applied to its result
//! with an i128 multiply and >>32 arithmetic shift. Fully deterministic;
//! truncation error ≤ ~1.5 storage units per output element.
//!
//! Profile support: q16_16 (realtime) and q32_32 (compact) — the i128
//! multiply covers BinaryStorage = i32/i64. Wider profiles need bigint
//! scale arithmetic and are deliberately not implemented yet.

#![cfg(any(table_format = "q16_16", table_format = "q32_32"))]

use super::ops;
use super::MAX_RAW;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;

use rayon::prelude::*;

/// TQ1.9 matrix with one quantization scale per row.
#[derive(Clone)]
pub struct RowScaledTQ19 {
    rows: usize,
    cols: usize,
    /// Row-major quantized weights: `raw = round(w / s_row)`, |raw| ≤ MAX_RAW.
    data: Vec<i16>,
    /// Per-row relative scale `s_row × SCALE` in unsigned Q32.32.
    scales_q32: Vec<u64>,
}

impl RowScaledTQ19 {
    /// Construct from parts. `data.len()` must be `rows × cols`,
    /// `scales_q32.len()` must be `rows`.
    pub fn from_parts(rows: usize, cols: usize, data: Vec<i16>, scales_q32: Vec<u64>) -> Self {
        assert_eq!(data.len(), rows * cols, "RowScaledTQ19: data length mismatch");
        assert_eq!(scales_q32.len(), rows, "RowScaledTQ19: scales length mismatch");
        debug_assert!(data.iter().all(|&w| (w as i32).abs() <= MAX_RAW as i32));
        Self { rows, cols, data, scales_q32 }
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn data(&self) -> &[i16] { &self.data }
    pub fn scales_q32(&self) -> &[u64] { &self.scales_q32 }

    /// Bytes of weight + scale storage (2 B/weight + 8 B/row).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * 2 + self.scales_q32.len() * 8
    }

    #[inline(always)]
    fn scale_row(dot: BinaryStorage, s_q32: u64) -> BinaryStorage {
        // (dot × s_rel) at Q32.32, arithmetic-shifted back. i128 covers
        // i32/i64 storage × u64 scale with headroom: |i64|·u64 < 2^127.
        let scaled = (dot as i128 * s_q32 as i128) >> 32;
        // Oversized scales via from_parts could exceed the storage range;
        // fail loud rather than wrap (house rule — silent wraps corrupted
        // fused dist² and exp downscales before 0.4.28/0.4.29).
        if scaled > BinaryStorage::MAX as i128 || scaled < BinaryStorage::MIN as i128 {
            panic!("RowScaledTQ19: scaled output exceeds storage range");
        }
        scaled as BinaryStorage
    }

    /// Row-scaled matvec: `out[r] = tq19_dot(row_r, x) × s_rel[r]`.
    pub fn matvec(&self, activations: &[BinaryStorage]) -> Vec<BinaryStorage> {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec: activation length mismatch");
        (0..self.rows)
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                Self::scale_row(ops::tq19_dot(row, activations), self.scales_q32[r])
            })
            .collect()
    }

    /// Row-parallel matvec.
    pub fn matvec_par(&self, activations: &[BinaryStorage]) -> Vec<BinaryStorage> {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec_par: activation length mismatch");
        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                Self::scale_row(ops::tq19_dot(row, activations), self.scales_q32[r])
            })
            .collect()
    }

    /// Row-parallel batch matvec (row weights stay in cache across the batch).
    pub fn matvec_batch_par(&self, batch: &[&[BinaryStorage]]) -> Vec<Vec<BinaryStorage>> {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(v.len(), self.cols, "RowScaledTQ19::matvec_batch_par: activation[{i}] length mismatch");
        }
        let per_row: Vec<Vec<BinaryStorage>> = (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                let s = self.scales_q32[r];
                batch
                    .iter()
                    .map(|x| Self::scale_row(ops::tq19_dot(row, x), s))
                    .collect()
            })
            .collect();
        // Transpose rows×batch → batch×rows.
        (0..batch.len())
            .map(|b| per_row.iter().map(|row| row[b]).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tq19::{TQ19Matrix, SCALE};

    /// A row-scaled matrix whose every scale equals the global step must
    /// reproduce plain TQ19 matvec bit-for-bit (s_rel = 1.0 in Q32.32).
    #[test]
    fn unit_scale_matches_plain_tq19() {
        let rows = 7;
        let cols = 64;
        let data: Vec<i16> = (0..rows * cols)
            .map(|i| ((i as i64 * 2654435761 % 59049) - 29524) as i16)
            .collect();
        let tq = TQ19Matrix::new(rows, cols, data.clone());
        let rs = RowScaledTQ19::from_parts(rows, cols, data, vec![1u64 << 32; rows]);
        let x: Vec<BinaryStorage> = (0..cols)
            .map(|i| ((i as i64 * 40503 % 2049) - 1024) as BinaryStorage)
            .collect();
        assert_eq!(tq.matvec(&x), rs.matvec(&x));
    }

    /// Independent oracle: recompute the matvec with plain i128 arithmetic
    /// (no tq19_dot, no SIMD) — Σ raw·x accumulated at i128, then the same
    /// Q32.32 multiply-shift. Pins both the tq19_dot reuse and the scale
    /// application against a path that shares no code with the kernel.
    #[test]
    fn rowscaled_matvec_matches_i128_oracle() {
        let rows = 5;
        let cols = 97; // not divisible by SIMD lane widths
        let data: Vec<i16> = (0..rows * cols)
            .map(|i| ((i as i64 * 48271 % 59049) - 29524) as i16)
            .collect();
        // Scales around 1.0: 0.5×, 1×, ~1.3×, tiny, and exactly 1 LSB above 1.
        let scales: Vec<u64> = vec![
            1u64 << 31,
            1u64 << 32,
            (1u64 << 32) + (1u64 << 30) + 12345,
            1u64 << 20,
            (1u64 << 32) + 1,
        ];
        let x: Vec<BinaryStorage> = (0..cols)
            .map(|i| ((i as i64 * 69621 % 4001) - 2000) as BinaryStorage)
            .collect();
        let rs = RowScaledTQ19::from_parts(rows, cols, data.clone(), scales.clone());
        let got = rs.matvec(&x);
        for r in 0..rows {
            let mut acc: i128 = 0;
            for c in 0..cols {
                acc += data[r * cols + c] as i128 * x[c] as i128;
            }
            // tq19_dot spec: truncate-toward-zero division by the global
            // SCALE, narrow, THEN the Q32.32 row-scale multiply-shift.
            let dot = acc / SCALE as i128;
            let expected = ((dot * scales[r] as i128) >> 32) as BinaryStorage;
            assert_eq!(got[r], expected, "row {} diverged from i128 oracle", r);
        }
        // matvec_par must agree with sequential.
        assert_eq!(got, rs.matvec_par(&x));
    }

    /// Scaling all weights up and the row scale down must approximate the
    /// same values with far finer granularity (small-weight fidelity).
    #[test]
    fn halved_scale_doubles_resolution() {
        let cols = 32;
        // value 3 raw units at global step, stored at s_rel = 0.5 with raw ×2
        let data: Vec<i16> = vec![6; cols];
        let rs = RowScaledTQ19::from_parts(1, cols, data, vec![1u64 << 31]);
        let x: Vec<BinaryStorage> = vec![SCALE as BinaryStorage; cols];
        // per element: raw 6 × x SCALE / SCALE × 0.5 = 3 → sum = 96
        assert_eq!(rs.matvec(&x)[0], 3 * cols as BinaryStorage);
    }
}
