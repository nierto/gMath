//! Five-trit row-scaled matrices: one i8 code in `[-121, 121]` per weight
//! plus one per-row scale (unsigned Q32.32).
//!
//! `value(r, c) = data[r * cols + c] * scales_q32[r] / 2^32`. Int8-class
//! precision at one byte per weight, exact integer accumulation, one scale
//! multiply per row: half the bytes per weight of TQ1.9.
//! [`RowScaledTQ19`](super::RowScaledTQ19) is the same construction with ten
//! trits in an i16; this one trades precision for half the bytes streamed per
//! output element.
//!
//! Every kernel computes the same integer `sum(code * x)` per row before the
//! row scale, whichever path it takes (scalar, AVX2 on split halves, the
//! 4x4 register tile), so the matvec forms agree bit for bit.
//!
//! Profile support: realtime (i32 activations, any `GMATH_FRAC_BITS`). The
//! SIMD kernels split a 32-bit activation into 16-bit halves.

#![cfg(table_format = "q16_16")]

use crate::fixed_point::frac_config::FRAC_BITS;
use rayon::prelude::*;

/// Largest code: `(3^5 - 1) / 2`.
pub const TQ5_MAX: i8 = 121;
/// The wide output keeps `FRAC_BITS` extra fractional bits: the Q32.32 scale
/// is shifted out by `32 - FRAC_BITS` instead of 32.
const Q2F_SHIFT: u32 = 32 - FRAC_BITS;

/// A row-major matrix of five-trit codes with a per-row scale.
///
/// Value of entry `(r, c)` = `data[r * cols + c] * scales_q32[r] / 2^32`.
#[derive(Clone)]
pub struct RowScaledTQ5 {
    rows: usize,
    cols: usize,
    data: Vec<i8>,
    scales_q32: Vec<u64>,
}

/// An activation vector split into signed 16-bit halves: `x = hi * 65536 + lo` exactly.
/// Built once per matvec. `None` when some |x| >= 2^30 (the split would not fit i16 at the
/// extreme): callers use the scalar dot instead.
///
/// The high half is zero for every element below 2^15 in magnitude, so when few
/// elements carry one, a row's dot runs as one 16-bit multiply-add over `lo` plus a scalar
/// correction over the listed high halves: `sum(c*x) = sum(c*lo) + (sum(c*hi) << 16)`, the
/// same integer as the dense two-multiply form.
struct SplitActivation {
    lo: Vec<i16>,
    hi: Vec<i16>,
    /// `(index, hi)` for every element whose high half is not zero.
    hi_nz: Vec<(u32, i16)>,
}

impl SplitActivation {
    fn new(x: &[i32]) -> Option<Self> {
        // unsigned_abs: |i32::MIN| is 2^31, which `abs` cannot represent
        if x.iter().any(|&v| v.unsigned_abs() >= 1 << 30) {
            return None;
        }
        let lo: Vec<i16> = x.iter().map(|&v| v as i16).collect(); // sign-extended low 16 bits
        let hi: Vec<i16> = x
            .iter()
            .zip(&lo)
            .map(|(&v, &l)| ((v - l as i32) >> 16) as i16)
            .collect();
        let hi_nz = hi
            .iter()
            .enumerate()
            .filter(|(_, &h)| h != 0)
            .map(|(i, &h)| (i as u32, h))
            .collect();
        Some(Self { lo, hi, hi_nz })
    }

    /// At most one element in eight carries a high half: the low-half multiply plus the
    /// scalar correction is then cheaper than two dense multiplies.
    #[inline]
    fn sparse(&self) -> bool {
        self.hi_nz.len() * 8 <= self.lo.len()
    }

    /// `sum(code * hi) << 16` over the listed high halves.
    #[inline]
    fn hi_correction(&self, row: &[i8]) -> i64 {
        let mut s = 0i64;
        for &(i, h) in &self.hi_nz {
            s += row[i as usize] as i64 * h as i64;
        }
        s << 16
    }
}

/// Whether the AVX2 kernel is usable on this machine (checked once).
fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `sum(code * x)` from the split halves: `sum(code * lo) + (sum(code * hi) << 16)`.
/// Each product is below 2^22 (|code| <= 121, |half| <= 2^15); a 16-bit multiply-add
/// pairs two, and a 32-bit lane takes at most 64 of those before widening, so no partial
/// sum ever exceeds 2^29. Same integer result as the scalar `dot`, in a different order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_split_avx2(row: &[i8], xs: &SplitActivation) -> i64 {
    use std::arch::x86_64::*;
    let n = row.len();
    let (lo, hi) = (&xs.lo, &xs.hi);
    let mut total: i64 = 0;
    let mut i = 0usize;
    let vec_end = n / 16 * 16;
    while i < vec_end {
        let block_end = (i + 64 * 16).min(vec_end);
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();
        while i < block_end {
            // SAFETY: i + 16 <= n and lo/hi have length n; the loads tolerate any alignment.
            let c = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.as_ptr().add(i) as *const __m128i));
            let l = _mm256_loadu_si256(lo.as_ptr().add(i) as *const __m256i);
            let h = _mm256_loadu_si256(hi.as_ptr().add(i) as *const __m256i);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(c, l));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(c, h));
            i += 16;
        }
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc_lo);
        let sum_lo: i64 = lanes.iter().map(|&v| v as i64).sum();
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc_hi);
        let sum_hi: i64 = lanes.iter().map(|&v| v as i64).sum();
        total += sum_lo + (sum_hi << 16);
    }
    for k in vec_end..n {
        total += row[k] as i64 * (lo[k] as i64 + ((hi[k] as i64) << 16));
    }
    total
}

/// `sum(code * lo)` alone: one 16-bit multiply-add per 16 columns, same lane bound as
/// [`dot_split_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_lo_avx2(row: &[i8], lo: &[i16]) -> i64 {
    use std::arch::x86_64::*;
    let n = row.len();
    let mut total: i64 = 0;
    let mut i = 0usize;
    let vec_end = n / 16 * 16;
    while i < vec_end {
        let block_end = (i + 64 * 16).min(vec_end);
        let mut acc = _mm256_setzero_si256();
        while i < block_end {
            // SAFETY: i + 16 <= n and lo has length n; the loads tolerate any alignment.
            let c = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.as_ptr().add(i) as *const __m128i));
            let l = _mm256_loadu_si256(lo.as_ptr().add(i) as *const __m256i);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(c, l));
            i += 16;
        }
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
        total += lanes.iter().map(|&v| v as i64).sum::<i64>();
    }
    for k in vec_end..n {
        total += row[k] as i64 * lo[k] as i64;
    }
    total
}

/// Register tile: `sum(code * lo)` for 4 rows x 4 activation vectors at once. Each loaded
/// weight vector feeds four tokens and each loaded activation vector four rows. Every
/// accumulator lane has the same bound as [`dot_split_avx2`] (64 multiply-adds of products
/// below 2^22 before widening).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn tile_lo_4x4(rows: [&[i8]; 4], los: [&[i16]; 4], n: usize) -> [[i64; 4]; 4] {
    use std::arch::x86_64::*;
    let vec_end = n / 16 * 16;
    let mut i = 0usize;
    let mut total = [[0i64; 4]; 4];
    while i < vec_end {
        let block_end = (i + 64 * 16).min(vec_end);
        let mut acc = [[_mm256_setzero_si256(); 4]; 4];
        while i < block_end {
            let mut c = [_mm256_setzero_si256(); 4];
            for r in 0..4 {
                // SAFETY: i + 16 <= n and every row has n codes.
                c[r] = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    rows[r].as_ptr().add(i) as *const __m128i
                ));
            }
            for t in 0..4 {
                // SAFETY: i + 16 <= n and every activation has n elements.
                let l = _mm256_loadu_si256(los[t].as_ptr().add(i) as *const __m256i);
                for r in 0..4 {
                    acc[r][t] = _mm256_add_epi32(acc[r][t], _mm256_madd_epi16(c[r], l));
                }
            }
            i += 16;
        }
        let mut lanes = [0i32; 8];
        for r in 0..4 {
            for t in 0..4 {
                _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc[r][t]);
                total[r][t] += lanes.iter().map(|&v| v as i64).sum::<i64>();
            }
        }
    }
    for k in vec_end..n {
        for r in 0..4 {
            for t in 0..4 {
                total[r][t] += rows[r][k] as i64 * los[t][k] as i64;
            }
        }
    }
    total
}

impl RowScaledTQ5 {
    /// Construct from parts; `data.len() == rows * cols`, `scales_q32.len() == rows`,
    /// every code in `[-121, 121]` (the kernels' overflow bounds rest on it).
    pub fn from_parts(rows: usize, cols: usize, data: Vec<i8>, scales_q32: Vec<u64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "RowScaledTQ5: data length mismatch"
        );
        assert_eq!(
            scales_q32.len(),
            rows,
            "RowScaledTQ5: scales length mismatch"
        );
        assert!(data.iter().all(|&w| w != i8::MIN && w.abs() <= TQ5_MAX), "RowScaledTQ5: code outside [-121, 121]");
        // each term of a row sum is below 2^38, so 2^24 columns stay below 2^62
        assert!(cols < 1 << 24, "RowScaledTQ5: more than 2^24 columns");
        Self {
            rows,
            cols,
            data,
            scales_q32,
        }
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn data(&self) -> &[i8] {
        &self.data
    }
    pub fn scales_q32(&self) -> &[u64] {
        &self.scales_q32
    }
    /// Bytes of weight + scale storage (1 B/weight + 8 B/row).
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.scales_q32.len() * 8
    }

    /// Exact integer dot: `sum(code * x)` in i64. Safe for `cols < 2^24`
    /// (`|code| <= 121 < 2^7`, `|x| < 2^31`, so each term is below 2^38). Summed in
    /// 16-wide blocks so the compiler can widen the products in SIMD lanes; integer
    /// addition is associative, so the result is the same in any order.
    #[inline]
    fn dot(row: &[i8], x: &[i32]) -> i64 {
        let n = row.len() / 16 * 16;
        let blocks: i64 = row[..n]
            .chunks_exact(16)
            .zip(x[..n].chunks_exact(16))
            .map(|(r, v)| {
                let mut s = 0i64;
                for k in 0..16 {
                    s += (r[k] as i32 as i64) * (v[k] as i64);
                }
                s
            })
            .sum();
        blocks
            + row[n..]
                .iter()
                .zip(&x[n..])
                .map(|(&c, &v)| c as i64 * v as i64)
                .sum::<i64>()
    }

    /// The row dot through the fastest exact path available: AVX2 on the split halves,
    /// else the scalar loop.
    #[inline(always)]
    fn dot_any(row: &[i8], x: &[i32], split: Option<&SplitActivation>) -> i64 {
        #[cfg(target_arch = "x86_64")]
        if let Some(xs) = split {
            if avx2_available() {
                // SAFETY: avx2_available() checked the CPU feature; slices share the length.
                return unsafe {
                    if xs.sparse() {
                        dot_lo_avx2(row, &xs.lo) + xs.hi_correction(row)
                    } else {
                        dot_split_avx2(row, xs)
                    }
                };
            }
        }
        Self::dot(row, x)
    }

    /// `floor(acc * s / 2^32)` narrowed to storage; fails loud on overflow.
    #[inline(always)]
    fn scale_row(acc: i64, s_q32: u64) -> i32 {
        let scaled = (acc as i128 * s_q32 as i128) >> 32;
        if scaled > i32::MAX as i128 || scaled < i32::MIN as i128 {
            panic!("RowScaledTQ5: scaled output exceeds storage range");
        }
        scaled as i32
    }

    /// Matvec: `out[r] = floor(sum(code * x) * s_r / 2^32)`.
    ///
    /// # Panics
    /// Panics if `x.len() != self.cols()` or a result leaves the storage range.
    pub fn matvec(&self, x: &[i32]) -> Vec<i32> {
        assert_eq!(x.len(), self.cols, "RowScaledTQ5::matvec: activation length mismatch");
        let split = SplitActivation::new(x);
        (0..self.rows)
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                Self::scale_row(Self::dot_any(row, x, split.as_ref()), self.scales_q32[r])
            })
            .collect()
    }

    /// Row-parallel [`matvec`](Self::matvec): the same results.
    pub fn matvec_par(&self, x: &[i32]) -> Vec<i32> {
        assert_eq!(
            x.len(),
            self.cols,
            "RowScaledTQ5::matvec_par: activation length mismatch"
        );
        let split = SplitActivation::new(x);
        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                Self::scale_row(Self::dot_any(row, x, split.as_ref()), self.scales_q32[r])
            })
            .collect()
    }

    /// Batched matvec. Activations whose high halves are sparse go through the 4x4 register
    /// tile (one low-half multiply per element, high halves corrected per row), in token
    /// chunks sized so a chunk's low halves stay in L2; the rest (dense high halves, values
    /// too large to split, rows past the last full group of 4, no AVX2) take the per-row dot.
    /// Every path computes the same integer `sum(code * x)` before the row scale.
    pub fn matvec_batch_par(&self, batch: &[&[i32]]) -> Vec<Vec<i32>> {
        let rows = self.rows;
        let mut flat = vec![0i32; batch.len() * rows];
        self.matvec_batch_par_into(batch, &mut flat);
        if rows == 0 {
            return vec![Vec::new(); batch.len()];
        }
        flat.chunks_exact(rows).map(<[i32]>::to_vec).collect()
    }

    /// [`matvec_batch_par`](Self::matvec_batch_par) writing into a
    /// caller-provided buffer, flat and batch-major: `out[b * rows + r]` is
    /// row `r` of the result for `batch[b]`. The same values; no result
    /// vector is allocated.
    ///
    /// # Panics
    /// Panics on an activation length mismatch or if
    /// `out.len() != batch.len() * rows`.
    pub fn matvec_batch_par_into(&self, batch: &[&[i32]], out: &mut [i32]) {
        for x in batch {
            assert_eq!(
                x.len(),
                self.cols,
                "RowScaledTQ5::matvec_batch_par: activation length mismatch"
            );
        }
        let (rows, cols) = (self.rows, self.cols);
        let sink = super::ops::BatchOut::new(out, rows, batch.len(), "RowScaledTQ5::matvec_batch_par_into");
        let splits: Vec<Option<SplitActivation>> =
            batch.iter().map(|x| SplitActivation::new(x)).collect();
        let tile_ok = cfg!(target_arch = "x86_64") && avx2_available();
        let (fast, slow): (Vec<usize>, Vec<usize>) = (0..batch.len())
            .partition(|&k| tile_ok && splits[k].as_ref().is_some_and(|s| s.sparse()));
        let row = |r: usize| &self.data[r * cols..(r + 1) * cols];
        let groups = rows / 4;
        // tokens per chunk: their low halves (2 bytes per element) within ~256 KiB, a multiple of 4
        let chunk = ((256 * 1024) / (2 * cols.max(1))).clamp(4, 64) / 4 * 4;
        for toks in fast.chunks(chunk) {
            // SAFETY (every write below): each token index appears once in
            // `fast` or `slow`, and each parallel task owns its rows (a group
            // of four here, one row in the per-row path), so no two
            // concurrent writes name the same (token, row).
            (0..groups).into_par_iter().for_each(|g| {
                let r0 = g * 4;
                let rs = [row(r0), row(r0 + 1), row(r0 + 2), row(r0 + 3)];
                for blk in toks.chunks(4) {
                    if blk.len() < 4 {
                        for &k in blk {
                            for r in 0..4 {
                                let v = Self::scale_row(
                                    Self::dot_any(rs[r], batch[k], splits[k].as_ref()),
                                    self.scales_q32[r0 + r],
                                );
                                unsafe { sink.write(k, r0 + r, v) };
                            }
                        }
                    } else {
                        let sp = [
                            splits[blk[0]].as_ref().unwrap(),
                            splits[blk[1]].as_ref().unwrap(),
                            splits[blk[2]].as_ref().unwrap(),
                            splits[blk[3]].as_ref().unwrap(),
                        ];
                        #[cfg(target_arch = "x86_64")]
                        // SAFETY: tile_ok checked AVX2; rows and activations have `cols` elements.
                        let acc = unsafe {
                            tile_lo_4x4(rs, [&sp[0].lo, &sp[1].lo, &sp[2].lo, &sp[3].lo], cols)
                        };
                        #[cfg(not(target_arch = "x86_64"))]
                        let acc = [[0i64; 4]; 4];
                        for t in 0..4 {
                            for r in 0..4 {
                                let v = Self::scale_row(
                                    acc[r][t] + sp[t].hi_correction(rs[r]),
                                    self.scales_q32[r0 + r],
                                );
                                unsafe { sink.write(blk[t], r0 + r, v) };
                            }
                        }
                    }
                }
            });
            // rows past the last full group of 4
            for r in groups * 4..rows {
                for &k in toks {
                    let v = Self::scale_row(
                        Self::dot_any(row(r), batch[k], splits[k].as_ref()),
                        self.scales_q32[r],
                    );
                    unsafe { sink.write(k, r, v) };
                }
            }
        }
        // the per-row path: each row is read once per tile of 8 such tokens, so the row stays in L1
        const TILE: usize = 8;
        for tile in slow.chunks(TILE) {
            (0..rows).into_par_iter().for_each(|r| {
                for &k in tile {
                    let v = Self::scale_row(
                        Self::dot_any(row(r), batch[k], splits[k].as_ref()),
                        self.scales_q32[r],
                    );
                    unsafe { sink.write(k, r, v) };
                }
            });
        }
    }

    /// `floor(acc * s / 2^(32 - FRAC_BITS))`; fails loud past the compute range.
    #[inline(always)]
    fn scale_row_q2f(acc: i64, s_q32: u64) -> i64 {
        let wide = (acc as i128 * s_q32 as i128) >> Q2F_SHIFT;
        if wide > i64::MAX as i128 || wide < i64::MIN as i128 {
            panic!("RowScaledTQ5: q2f scaled output exceeds compute range");
        }
        wide as i64
    }

    /// Wide-output matvec at `2 * FRAC_BITS` fractional bits:
    /// `floor(sum(code * x) * s_r / 2^(32 - FRAC_BITS))`, one rounding.
    ///
    /// # Panics
    /// Panics if `x.len() != self.cols()` or a result leaves the compute range.
    pub fn matvec_q2f(&self, x: &[i32]) -> Vec<i64> {
        assert_eq!(x.len(), self.cols, "RowScaledTQ5::matvec_q2f: activation length mismatch");
        let split = SplitActivation::new(x);
        (0..self.rows)
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                Self::scale_row_q2f(Self::dot_any(row, x, split.as_ref()), self.scales_q32[r])
            })
            .collect()
    }

    /// Row-parallel [`matvec_q2f`](Self::matvec_q2f): the same results.
    pub fn matvec_q2f_par(&self, x: &[i32]) -> Vec<i64> {
        assert_eq!(
            x.len(),
            self.cols,
            "RowScaledTQ5::matvec_q2f_par: activation length mismatch"
        );
        let split = SplitActivation::new(x);
        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let row = &self.data[r * self.cols..(r + 1) * self.cols];
                Self::scale_row_q2f(Self::dot_any(row, x, split.as_ref()), self.scales_q32[r])
            })
            .collect()
    }

    /// Serialize: `rows (u32) | cols (u32) | codes (i8 each) | scales (u64 each)`,
    /// little-endian.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&(self.rows as u32).to_le_bytes())?;
        w.write_all(&(self.cols as u32).to_le_bytes())?;
        let bytes: Vec<u8> = self.data.iter().map(|&v| v as u8).collect();
        w.write_all(&bytes)?;
        for &s in &self.scales_q32 {
            w.write_all(&s.to_le_bytes())?;
        }
        Ok(())
    }

    /// Deserialize (inverse of [`write_to`](Self::write_to)). A code outside
    /// `[-121, 121]` is `InvalidData`.
    pub fn read_from<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let rows = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4)?;
        let cols = u32::from_le_bytes(buf4) as usize;
        let len = rows.checked_mul(cols).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "RowScaledTQ5: dimensions overflow"))?;
        let mut bytes = vec![0u8; len];
        r.read_exact(&mut bytes)?;
        let data: Vec<i8> = bytes.into_iter().map(|b| b as i8).collect();
        if data.iter().any(|&c| c == i8::MIN || c.abs() > TQ5_MAX) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "RowScaledTQ5: code outside [-121, 121]"));
        }
        let mut scales = vec![0u64; rows];
        let mut buf8 = [0u8; 8];
        for s in scales.iter_mut() {
            r.read_exact(&mut buf8)?;
            *s = u64::from_le_bytes(buf8);
        }
        Ok(Self::from_parts(rows, cols, data, scales))
    }
}
