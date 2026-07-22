//! Row-scaled TQ1.9 ("TQ1.9-R") — per-row quantization scales.
//!
//! Motivation: plain TQ19 quantizes every
//! tensor with ONE global step (1/SCALE ≈ 5.1e-5, zero-floor 2.5e-5).
//! Some BF16-trained models put substantial weight mass below that
//! floor; the measured consequence is a 0.207%/layer router top-2 flip
//! rate that compounds to ~6.4% of tokens executing a wrong expert.
//! A per-row scale adapts the step to each row's own max: measured on
//! a large MoE model, matvec output error drops ~20× and wrong-expert tokens drop
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
use crate::fixed_point::universal::fasc::stack_evaluator::{BinaryStorage, ComputeStorage};

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

    /// Wide scale application: `floor(wide_dot · s_rel / 2^32)` at full
    /// precision. Same floor rule as [`Self::scale_row`], applied to the
    /// wide dot instead of the narrowed one.
    #[inline(always)]
    fn scale_row_q2f(wide_dot: ComputeStorage, s_q32: u64) -> ComputeStorage {
        #[cfg(table_format = "q16_16")]
        {
            // i64 × u64 always fits i128 (|i64|·u64 < 2^127).
            let scaled = (wide_dot as i128 * s_q32 as i128) >> 32;
            if scaled > i64::MAX as i128 || scaled < i64::MIN as i128 {
                panic!("RowScaledTQ19: q2f scaled output exceeds compute range");
            }
            scaled as i64
        }
        #[cfg(table_format = "q32_32")]
        {
            // i128 × u64 can overflow i128: split wide_dot = h·2^32 + l
            // (h = floor shift, l ∈ [0, 2^32)); then
            // floor(wide_dot·s/2^32) = h·s + floor(l·s/2^32) exactly.
            let h = wide_dot >> 32;
            let l = (wide_dot & 0xFFFF_FFFF) as i128;
            let low = (l * s_q32 as i128) >> 32;
            match h.checked_mul(s_q32 as i128).and_then(|hs| hs.checked_add(low)) {
                Some(v) => v,
                None => panic!("RowScaledTQ19: q2f scaled output exceeds compute range"),
            }
        }
    }

    /// Wide-output row-scaled matvec: each row at 2·FRAC_BITS precision.
    ///
    /// `floor(tq19_dot_q2f(row, x) · s_rel / 2^32)` — the wide dot keeps
    /// FRAC_BITS extra fractional bits *through* the scale multiply, so this
    /// is strictly more precise than scaling the narrowed dot. Consequence
    /// (documented deliberately): narrowing this result can differ from
    /// [`Self::matvec`] by ±1 storage LSB for non-unit scales — the narrow
    /// path scales an already-rounded dot. For `s_rel = 1.0` (`1u64 << 32`)
    /// the two agree bit-for-bit.
    pub fn matvec_q2f(&self, activations: &[BinaryStorage]) -> Vec<ComputeStorage> {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec_q2f: activation length mismatch");
        (0..self.rows)
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                Self::scale_row_q2f(ops::tq19_dot_q2f(row, activations), self.scales_q32[r])
            })
            .collect()
    }

    /// Row-parallel wide-output matvec. See [`Self::matvec_q2f`].
    pub fn matvec_q2f_par(&self, activations: &[BinaryStorage]) -> Vec<ComputeStorage> {
        assert_eq!(activations.len(), self.cols, "RowScaledTQ19::matvec_q2f_par: activation length mismatch");
        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                Self::scale_row_q2f(ops::tq19_dot_q2f(row, activations), self.scales_q32[r])
            })
            .collect()
    }

    /// Row-parallel wide-output batch matvec. See [`Self::matvec_q2f`].
    pub fn matvec_q2f_batch_par(&self, batch: &[&[BinaryStorage]]) -> Vec<Vec<ComputeStorage>> {
        for (i, v) in batch.iter().enumerate() {
            assert_eq!(v.len(), self.cols, "RowScaledTQ19::matvec_q2f_batch_par: activation[{i}] length mismatch");
        }
        let per_row: Vec<Vec<ComputeStorage>> = (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                let s = self.scales_q32[r];
                batch
                    .iter()
                    .map(|x| Self::scale_row_q2f(ops::tq19_dot_q2f(row, x), s))
                    .collect()
            })
            .collect();
        (0..batch.len())
            .map(|b| per_row.iter().map(|row| row[b]).collect())
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

    fn narrow_q2f(v: ComputeStorage) -> BinaryStorage {
        #[cfg(table_format = "q16_16")]
        { (v / (1i64 << crate::fixed_point::frac_config::FRAC_BITS)) as i32 }
        #[cfg(table_format = "q32_32")]
        { (v / (1i128 << 32)) as i64 }
    }

    /// Wide-output contract for the row-scaled form: with unit scales the
    /// narrow relationship is exact; with general scales the wide path keeps
    /// FRAC_BITS extra bits through the scale multiply, so narrow(q2f) may
    /// differ from matvec by at most 1 storage LSB (documented behavior).
    /// The wide value itself is pinned against an independent i128 oracle.
    #[test]
    fn q2f_unit_scale_exact_general_within_one_lsb_and_oracle() {
        let rows = 5;
        let cols = 97;
        let data: Vec<i16> = (0..rows * cols)
            .map(|i| ((i as i64 * 48271 % 59049) - 29524) as i16)
            .collect();
        let x: Vec<BinaryStorage> = (0..cols)
            .map(|i| ((i as i64 * 69621 % 4001) - 2000) as BinaryStorage)
            .collect();

        // Unit scales: narrow(q2f) == matvec bit-for-bit.
        let unit = RowScaledTQ19::from_parts(rows, cols, data.clone(), vec![1u64 << 32; rows]);
        let narrow = unit.matvec(&x);
        let wide = unit.matvec_q2f(&x);
        for r in 0..rows {
            assert_eq!(narrow_q2f(wide[r]), narrow[r], "unit-scale row {r}");
        }

        // General scales: oracle equality + ≤1 LSB narrow relationship.
        let scales: Vec<u64> = vec![
            1u64 << 31,
            1u64 << 32,
            (1u64 << 32) + (1u64 << 30) + 12345,
            1u64 << 20,
            (1u64 << 32) + 1,
        ];
        let rs = RowScaledTQ19::from_parts(rows, cols, data.clone(), scales.clone());
        let narrow = rs.matvec(&x);
        let wide = rs.matvec_q2f(&x);
        #[cfg(table_format = "q16_16")]
        let f = crate::fixed_point::frac_config::FRAC_BITS;
        #[cfg(table_format = "q32_32")]
        let f = 32u32;
        for r in 0..rows {
            // Independent oracle: exact accumulator → one truncating /SCALE
            // at 2F precision → floor scale multiply, all in plain i128.
            let mut acc: i128 = 0;
            for c in 0..cols {
                acc += data[r * cols + c] as i128 * x[c] as i128;
            }
            let wide_dot = (acc << f) / SCALE as i128;
            let expected = (wide_dot * scales[r] as i128) >> 32;
            assert_eq!(wide[r] as i128, expected, "row {r} diverged from i128 oracle");
            // Narrow relationship: within 1 storage LSB of the narrow path.
            let diff = (narrow_q2f(wide[r]) as i128 - narrow[r] as i128).abs();
            assert!(diff <= 1, "row {r}: narrow(q2f) off by {diff} LSB");
        }
        assert_eq!(wide, rs.matvec_q2f_par(&x));
        let batch: Vec<&[BinaryStorage]> = vec![&x];
        assert_eq!(rs.matvec_q2f_batch_par(&batch)[0], wide);
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
