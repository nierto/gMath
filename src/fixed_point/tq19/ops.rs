//! TQ1.9 core operations: profile-conditional implementation.
//!
//! All dot products accumulate at ComputeStorage (tier N+1).
//! Single division/downscale at the end per gMath precision contract.

use crate::fixed_point::universal::fasc::stack_evaluator::{BinaryStorage, ComputeStorage};
#[cfg(table_format = "q16_16")]
use crate::fixed_point::frac_config;

#[allow(unused_imports)]
use crate::fixed_point::I256;
#[allow(unused_imports)]
use crate::fixed_point::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;

use rayon::prelude::*;

use super::{SCALE, TRIT_DECODE_TABLE};

// ============================================================================
// Profile-conditional widening/narrowing helpers
//
// These isolate ALL cfg blocks so that the actual operations are generic.
// ============================================================================

/// Widen i16 weight value to ComputeStorage (type-widen only, no Q-format shift).
#[inline(always)]
pub(super) fn widen_weight(w: i16) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { w as i64 }
    #[cfg(table_format = "q32_32")]
    { w as i128 }
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(w as i128) }
    #[cfg(table_format = "q128_128")]
    { I512::from_i128(w as i128) }
    #[cfg(table_format = "q256_256")]
    { I1024::from_i128(w as i128) }
}

/// Widen BinaryStorage activation to ComputeStorage (type-widen only).
#[inline(always)]
pub(super) fn widen_activation(a: BinaryStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { a as i64 }
    #[cfg(table_format = "q32_32")]
    { a as i128 }
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(a) }
    #[cfg(table_format = "q128_128")]
    { I512::from_i256(a) }
    #[cfg(table_format = "q256_256")]
    { I1024::from_i512(a) }
}

/// SCALE constant at ComputeStorage width.
#[inline(always)]
pub(super) fn compute_scale() -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { SCALE as i64 }
    #[cfg(table_format = "q32_32")]
    { SCALE as i128 }
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(SCALE as i128) }
    #[cfg(table_format = "q128_128")]
    { I512::from_i128(SCALE as i128) }
    #[cfg(table_format = "q256_256")]
    { I1024::from_i128(SCALE as i128) }
}

/// Zero at ComputeStorage width.
#[inline(always)]
pub(super) fn compute_zero() -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { 0i64 }
    #[cfg(table_format = "q32_32")]
    { 0i128 }
    #[cfg(table_format = "q64_64")]
    { I256::zero() }
    #[cfg(table_format = "q128_128")]
    { I512::zero() }
    #[cfg(table_format = "q256_256")]
    { I1024::zero() }
}

const STORAGE_OVERFLOW: &str = "tq19: result exceeds storage range";
#[cfg(table_format = "q16_16")]
const ACC_OVERFLOW: &str = "tq19: accumulator exceeds compute-tier range";

/// Narrow ComputeStorage to BinaryStorage (type-narrow only, no Q-format shift).
///
/// # Panics
/// Panics if the value does not fit the storage type (fail loud, never wrap).
#[inline(always)]
pub(super) fn narrow_to_storage(v: ComputeStorage) -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    { if v > i32::MAX as i64 || v < i32::MIN as i64 { panic!("{}", STORAGE_OVERFLOW) } v as i32 }
    #[cfg(table_format = "q32_32")]
    { if v > i64::MAX as i128 || v < i64::MIN as i128 { panic!("{}", STORAGE_OVERFLOW) } v as i64 }
    #[cfg(table_format = "q64_64")]
    { if !v.fits_in_i128() { panic!("{}", STORAGE_OVERFLOW) } v.as_i128() }
    #[cfg(table_format = "q128_128")]
    { if !v.fits_in_i256() { panic!("{}", STORAGE_OVERFLOW) } v.as_i256() }
    #[cfg(table_format = "q256_256")]
    { if !v.fits_in_i512() { panic!("{}", STORAGE_OVERFLOW) } v.as_i512() }
}

/// Whether `len` worst-case terms can be summed at the compute tier without a
/// per-term check.
///
/// Realtime: a TQ1.9 term is below `2^15 * 2^31 = 2^46` in magnitude, so
/// `2^16` of them stay below `2^62`; a trit term is at most `2^31`, so the
/// same bound covers it. Wider profiles have at least 49 bits of headroom per
/// term, more than any slice length.
#[inline(always)]
pub(super) fn unchecked_sum_fits(len: usize) -> bool {
    #[cfg(table_format = "q16_16")]
    { len <= 1 << 16 }
    #[cfg(not(table_format = "q16_16"))]
    { let _ = len; true }
}

/// Wide matvec epilogue: the exact row value at 2·FRAC_BITS fractional
/// precision with EXACTLY ONE rounding: truncation toward zero of
/// `acc · 2^FRAC_BITS / SCALE`.
///
/// **Narrowing spec**: Rust's truncating division `q2f / (1 << FRAC_BITS)`
/// reproduces the narrow epilogue `acc / SCALE` bit-for-bit: nested
/// truncation toward zero is exact: `trunc(trunc(a·2^F/S)/2^F) ≡ trunc(a/S)`.
/// Downstream consumers narrowing wide outputs must use this rule.
///
/// Gated to q16_16/q32_32: on wider profiles the storage rounding floor is
/// ≤ 2^-64 and the problem this API addresses does not arise.
///
/// # Panics
/// Panics if the wide value exceeds the compute-tier range (fail loud,
/// never wrap). Unreachable at FRAC_BITS ≤ 14 (|acc| ≤ i64::MAX implies
/// |q2f| < 2^63 there); reachable and guarded at larger FRAC_BITS.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[inline(always)]
pub(super) fn wide_output(acc: ComputeStorage) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    {
        // i64 acc → i128 intermediate: acc·2^F is exact, one division.
        let q2f = ((acc as i128) << frac_config::FRAC_BITS) / (SCALE as i128);
        if q2f > i64::MAX as i128 || q2f < i64::MIN as i128 {
            panic!("matvec_q2f: wide output exceeds compute-tier range");
        }
        q2f as i64
    }
    #[cfg(table_format = "q32_32")]
    {
        // i128 acc: acc·2^32 can overflow i128, so stage the single
        // truncating division as q·2^32 + trunc(r·2^32/S) with q = acc/S,
        // r = acc % S (|r| < S, sign of acc) — algebraically identical to
        // trunc(acc·2^32/S) for truncation toward zero.
        let s = SCALE as i128;
        let q = acc / s;
        let frac = ((acc % s) << 32) / s;
        match q.checked_mul(1i128 << 32).and_then(|hi| hi.checked_add(frac)) {
            Some(v) => v,
            None => panic!("matvec_q2f: wide output exceeds compute-tier range"),
        }
    }
}

/// Apply a Q-format per-block scale to an accumulated trit dot:
/// `round((dot * scale) >> FRAC_BITS)`, round to nearest.
///
/// On realtime and compact the product is taken on the accumulator itself, so
/// a dot that exceeds storage but whose scaled value fits is still exact.
///
/// # Panics
/// Panics if the result does not fit storage (fail loud, never wrap).
#[inline]
fn scale_dot(acc: ComputeStorage, scale: BinaryStorage) -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    {
        let p = acc as i128 * scale as i128;
        let r = (p >> frac_config::FRAC_BITS) + ((p >> frac_config::FRAC_ROUND_BIT) & 1);
        if r > i32::MAX as i128 || r < i32::MIN as i128 { panic!("{}", STORAGE_OVERFLOW) }
        r as i32
    }
    #[cfg(table_format = "q32_32")]
    {
        let p = match acc.checked_mul(scale as i128) {
            Some(p) => p,
            None => panic!("{}", STORAGE_OVERFLOW),
        };
        let r = (p >> 32) + ((p >> 31) & 1);
        if r > i64::MAX as i128 || r < i64::MIN as i128 { panic!("{}", STORAGE_OVERFLOW) }
        r as i64
    }
    #[cfg(table_format = "q64_64")]
    {
        // Both factors fit i128 after the checked narrow, so the I256 product is exact.
        let v = widen_activation(narrow_to_storage(acc)) * widen_activation(scale);
        let round_bit = (v & I256::from_i128(1i128 << 63)) != I256::zero();
        let shifted = v >> 64u32;
        let shifted = if round_bit { shifted + I256::from_i128(1) } else { shifted };
        narrow_to_storage(shifted)
    }
    #[cfg(table_format = "q128_128")]
    {
        let v = widen_activation(narrow_to_storage(acc)) * widen_activation(scale);
        let round_bit = (v & (I512::from_i128(1) << 127usize)) != I512::zero();
        let shifted = v >> 128usize;
        let shifted = if round_bit { shifted + I512::from_i128(1) } else { shifted };
        narrow_to_storage(shifted)
    }
    #[cfg(table_format = "q256_256")]
    {
        let v = widen_activation(narrow_to_storage(acc)) * widen_activation(scale);
        let round_bit = (v & (I1024::from_i128(1) << 255usize)) != I1024::zero();
        let shifted = v >> 256usize;
        let shifted = if round_bit { shifted + I1024::from_i128(1) } else { shifted };
        narrow_to_storage(shifted)
    }
}

// ============================================================================
// Inner dot products — return ComputeStorage (pre-division)
// ============================================================================

/// Realtime rows too long for the unchecked i64 sum: exact i128 sum, one check.
#[cfg(table_format = "q16_16")]
#[cold]
#[inline(never)]
fn tq19_dot_exact(weights: &[i16], activations: &[BinaryStorage]) -> ComputeStorage {
    let mut acc = 0i128;
    for i in 0..weights.len() {
        acc += weights[i] as i128 * activations[i] as i128;
    }
    if acc > i64::MAX as i128 || acc < i64::MIN as i128 { panic!("{}", ACC_OVERFLOW) }
    acc as i64
}

/// TQ1.9 inner dot product at compute tier (before SCALE division).
///
/// Returns raw accumulator. Caller divides by SCALE and narrows.
/// On x86_64 realtime profile, dispatches to AVX2 when available.
#[inline]
fn tq19_dot_compute(weights: &[i16], activations: &[BinaryStorage]) -> ComputeStorage {
    // Past the bound a partial sum could leave i64: sum exactly and check once.
    #[cfg(table_format = "q16_16")]
    {
        if !unchecked_sum_fits(weights.len()) {
            return tq19_dot_exact(weights, activations);
        }
    }

    // SIMD dispatch for realtime profile on x86_64
    #[cfg(all(target_arch = "x86_64", table_format = "q16_16"))]
    {
        if std::is_x86_feature_detected!("avx2") && weights.len() >= 8 {
            // Safety: AVX2 detected, length checked
            return unsafe { super::simd::tq19_dot_avx2(weights, activations) };
        }
    }

    // Scalar fallback (all profiles)
    let mut acc = compute_zero();
    for i in 0..weights.len() {
        acc = acc + widen_weight(weights[i]) * widen_activation(activations[i]);
    }
    acc
}

/// Trit inner dot product at compute tier (pre-scale).
///
/// Zero-multiply: only add/sub/skip. Returns raw accumulator.
#[inline]
fn trit_dot_compute(trits: &[i8], activations: &[BinaryStorage]) -> ComputeStorage {
    // A trit term is at most 2^31, so i64 holds 2^32 of them: no slice reaches that.
    // SIMD dispatch for realtime profile on x86_64
    #[cfg(all(target_arch = "x86_64", table_format = "q16_16"))]
    {
        if std::is_x86_feature_detected!("avx2") && trits.len() >= 8 {
            return unsafe { super::simd::trit_dot_avx2(trits, activations) };
        }
    }

    let mut acc = compute_zero();
    for i in 0..trits.len() {
        let t = trits[i];
        if t == 1 {
            acc = acc + widen_activation(activations[i]);
        } else if t == -1 {
            acc = acc - widen_activation(activations[i]);
        }
    }
    acc
}

// ============================================================================
// Public dot products
// ============================================================================

/// TQ1.9 dot: `sum(w[i] * a[i]) / SCALE` at compute tier.
/// Panics if the result exceeds storage.
pub fn tq19_dot(weights: &[i16], activations: &[BinaryStorage]) -> BinaryStorage {
    debug_assert_eq!(weights.len(), activations.len());
    let acc = tq19_dot_compute(weights, activations);
    narrow_to_storage(acc / compute_scale())
}

/// Wide-output TQ1.9 dot: `trunc(sum(weights[i]·activations[i]) · 2^FRAC_BITS / SCALE)`.
///
/// Same inner loop (and SIMD dispatch) as [`tq19_dot`]; only the epilogue
/// differs: see [`wide_output`] for the exact rounding/narrowing contract.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
pub fn tq19_dot_q2f(weights: &[i16], activations: &[BinaryStorage]) -> ComputeStorage {
    debug_assert_eq!(weights.len(), activations.len());
    wide_output(tq19_dot_compute(weights, activations))
}

/// Zero-multiply trit dot for pre-decoded trits.
/// Panics if the result exceeds storage.
pub fn trit_dot(trits: &[i8], activations: &[BinaryStorage]) -> BinaryStorage {
    debug_assert_eq!(trits.len(), activations.len());
    narrow_to_storage(trit_dot_compute(trits, activations))
}

/// Packed trit dot with per-block scale.
///
/// Unpacks 5 trits/byte, accumulates at compute tier, then applies `scale`
/// with one rounding to nearest. Panics if the result exceeds storage.
pub fn packed_trit_dot(
    packed: &[u8],
    count: usize,
    activations: &[BinaryStorage],
    scale: BinaryStorage,
) -> BinaryStorage {
    assert!(activations.len() >= count, "packed_trit_dot: activations shorter than count");

    let mut acc = compute_zero();
    let mut elem = 0;

    for &byte in packed.iter() {
        if elem >= count { break; }
        let trits = TRIT_DECODE_TABLE[byte as usize];
        for k in 0..5 {
            if elem >= count { break; }
            let t = trits[k];
            if t == 1 {
                acc = acc + widen_activation(activations[elem]);
            } else if t == -1 {
                acc = acc - widen_activation(activations[elem]);
            }
            elem += 1;
        }
    }

    scale_dot(acc, scale)
}

// ============================================================================
// Sequential matvec
// ============================================================================

/// TQ1.9 matrix-vector product (sequential).
pub fn tq19_matvec(
    data: &[i16],
    rows: usize,
    cols: usize,
    activations: &[BinaryStorage],
) -> Vec<BinaryStorage> {
    let scale = compute_scale();
    (0..rows)
        .map(|row| {
            let row_weights = &data[row * cols..(row + 1) * cols];
            let acc = tq19_dot_compute(row_weights, activations);
            narrow_to_storage(acc / scale)
        })
        .collect()
}

/// Tile size for batch matvec (elements per tile).
///
/// Chosen so that weight_tile + activation_tiles fit in L1d:
///   512 × 2B (weights) + 512 × 8B × batch_size (activations)
///   = 1 KB + 4 KB × batch_size
/// For batch=8: 33 KB: fits in 32-48 KB L1d.
const BATCH_TILE: usize = 512;

/// Batch TQ1.9 matvec with tiled accumulation.
///
/// For each row, processes BATCH_TILE elements at a time across all batch
/// vectors before advancing to the next tile. This keeps the weight tile
/// and all corresponding activation tiles in L1 cache together.
///
/// Without tiling, batch=4 on compact profile (32KB activation vectors)
/// thrashes L1. With tiling: weight tile (1KB) + activation tiles (4KB × batch)
/// fits comfortably.
pub fn tq19_matvec_batch(
    data: &[i16],
    rows: usize,
    cols: usize,
    batch: &[&[BinaryStorage]],
) -> Vec<Vec<BinaryStorage>> {
    if !unchecked_sum_fits(cols) {
        return batch.iter().map(|x| tq19_matvec(data, rows, cols, &x[..cols])).collect();
    }
    let batch_size = batch.len();
    let scale = compute_scale();
    let mut results: Vec<Vec<BinaryStorage>> = (0..batch_size)
        .map(|_| Vec::with_capacity(rows))
        .collect();

    // Per-batch accumulators, reused across rows
    let mut accs = vec![compute_zero(); batch_size];

    for row in 0..rows {
        // Reset accumulators
        for acc in accs.iter_mut() {
            *acc = compute_zero();
        }

        let row_start = row * cols;

        // Tiled: process BATCH_TILE elements across all batch vectors
        let mut tile_start = 0;
        while tile_start < cols {
            let tile_end = (tile_start + BATCH_TILE).min(cols);
            let tile_weights = &data[row_start + tile_start..row_start + tile_end];

            for b in 0..batch_size {
                let tile_acts = &batch[b][tile_start..tile_end];
                for i in 0..tile_weights.len() {
                    accs[b] = accs[b] + widen_weight(tile_weights[i]) * widen_activation(tile_acts[i]);
                }
            }

            tile_start = tile_end;
        }

        // Finalize: divide by SCALE, narrow, store
        for b in 0..batch_size {
            results[b].push(narrow_to_storage(accs[b] / scale));
        }
    }

    results
}

/// Wide-output TQ1.9 matvec (sequential): 2·FRAC_BITS precision, one rounding.
///
/// Same inner loops as [`tq19_matvec`]; only the epilogue differs: see
/// [`wide_output`] for the exact rounding/narrowing contract.
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
pub fn tq19_matvec_q2f(
    data: &[i16],
    rows: usize,
    cols: usize,
    activations: &[BinaryStorage],
) -> Vec<ComputeStorage> {
    (0..rows)
        .map(|row| {
            let row_weights = &data[row * cols..(row + 1) * cols];
            wide_output(tq19_dot_compute(row_weights, activations))
        })
        .collect()
}

/// Packed trit matvec (sequential).
pub fn packed_trit_matvec(
    packed_trits: &[u8],
    rows: usize,
    cols: usize,
    activations: &[BinaryStorage],
    scales: &[BinaryStorage],
) -> Vec<BinaryStorage> {
    assert!(activations.len() >= cols);
    assert!(scales.len() >= rows);

    let bytes_per_row = (cols + 4) / 5;
    (0..rows)
        .map(|row| {
            let start = row * bytes_per_row;
            let row_trits = &packed_trits[start..start + bytes_per_row];
            packed_trit_dot(row_trits, cols, activations, scales[row])
        })
        .collect()
}

// ============================================================================
// Parallel variants (rayon feature)
// ============================================================================

/// Row-parallel TQ1.9 matvec.
// rayon always available — module is gated behind inference feature
pub fn tq19_matvec_par(
    data: &[i16],
    rows: usize,
    cols: usize,
    activations: &[BinaryStorage],
) -> Vec<BinaryStorage> {
    let scale = compute_scale();
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let row_weights = &data[row * cols..(row + 1) * cols];
            let acc = tq19_dot_compute(row_weights, activations);
            narrow_to_storage(acc / scale)
        })
        .collect()
}

/// Row-parallel wide-output TQ1.9 matvec: 2·FRAC_BITS precision, one rounding.
///
/// Epilogue-only variant of [`tq19_matvec_par`]: see [`wide_output`].
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
pub fn tq19_matvec_q2f_par(
    data: &[i16],
    rows: usize,
    cols: usize,
    activations: &[BinaryStorage],
) -> Vec<ComputeStorage> {
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let row_weights = &data[row * cols..(row + 1) * cols];
            wide_output(tq19_dot_compute(row_weights, activations))
        })
        .collect()
}

/// Row-parallel wide-output batch TQ1.9 matvec with tiled accumulation.
/// Epilogue-only variant of [`tq19_matvec_batch_par`]: see [`wide_output`].
#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
pub fn tq19_matvec_q2f_batch_par(
    data: &[i16],
    rows: usize,
    cols: usize,
    batch: &[&[BinaryStorage]],
) -> Vec<Vec<ComputeStorage>> {
    if !unchecked_sum_fits(cols) {
        return batch.iter().map(|x| tq19_matvec_q2f_par(data, rows, cols, &x[..cols])).collect();
    }
    let batch_size = batch.len();
    let row_results: Vec<Vec<ComputeStorage>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let row_start = row * cols;
            let mut accs = vec![compute_zero(); batch_size];
            let mut tile_start = 0;
            while tile_start < cols {
                let tile_end = (tile_start + BATCH_TILE).min(cols);
                let tile_weights = &data[row_start + tile_start..row_start + tile_end];
                for b in 0..batch_size {
                    let tile_acts = &batch[b][tile_start..tile_end];
                    for i in 0..tile_weights.len() {
                        accs[b] = accs[b] + widen_weight(tile_weights[i]) * widen_activation(tile_acts[i]);
                    }
                }
                tile_start = tile_end;
            }
            accs.into_iter().map(wide_output).collect()
        })
        .collect();
    let mut results: Vec<Vec<ComputeStorage>> = (0..batch_size)
        .map(|_| Vec::with_capacity(rows))
        .collect();
    for row_result in row_results {
        for (b, val) in row_result.into_iter().enumerate() {
            results[b].push(val);
        }
    }
    results
}

/// Row-parallel batch TQ1.9 matvec with tiled accumulation.
///
/// A caller-provided batch output, flat and batch-major: element
/// `b * rows + r` is row `r` of the result for batch vector `b`. Row-parallel
/// kernels write one row's results for every batch vector from one task, so
/// the positions written by different tasks interleave; this hands them out
/// without a lock.
pub(crate) struct BatchOut<'a, T> {
    ptr: *mut T,
    rows: usize,
    len: usize,
    _slice: std::marker::PhantomData<&'a mut [T]>,
}

// SAFETY: the writer only stores `T` values at positions of an exclusively
// borrowed slice; `write`'s contract keeps the positions disjoint.
unsafe impl<T: Send> Send for BatchOut<'_, T> {}
unsafe impl<T: Send> Sync for BatchOut<'_, T> {}

impl<'a, T: Copy> BatchOut<'a, T> {
    /// Panics unless `out.len() == batch_len * rows`.
    pub(crate) fn new(out: &'a mut [T], rows: usize, batch_len: usize, what: &str) -> Self {
        assert_eq!(out.len(), batch_len * rows, "{what}: out length is not batch.len() * rows");
        BatchOut { ptr: out.as_mut_ptr(), rows, len: out.len(), _slice: std::marker::PhantomData }
    }

    /// Store the result for batch vector `b`, row `r`.
    ///
    /// # Safety
    /// No two calls that can run concurrently may name the same `(b, r)`.
    #[inline(always)]
    pub(crate) unsafe fn write(&self, b: usize, r: usize, value: T) {
        let i = b * self.rows + r;
        assert!(r < self.rows && i < self.len, "BatchOut: position out of range");
        // SAFETY: `i` is in bounds of the borrowed slice; the caller keeps
        // concurrent positions distinct.
        unsafe { self.ptr.add(i).write(value) }
    }
}

/// [`tq19_matvec_batch_par`] writing into a caller-provided buffer, flat and
/// batch-major: `out[b * rows + r]` is row `r` of the result for `batch[b]`.
/// The same values; no result vector is allocated.
///
/// # Panics
/// Panics if `out.len() != batch.len() * rows`.
pub(crate) fn tq19_matvec_batch_par_into(
    data: &[i16],
    rows: usize,
    cols: usize,
    batch: &[&[BinaryStorage]],
    out: &mut [BinaryStorage],
) {
    let batch_size = batch.len();
    assert_eq!(out.len(), batch_size * rows, "tq19_matvec_batch_par_into: out length is not batch.len() * rows");
    if !unchecked_sum_fits(cols) {
        for (x, o) in batch.iter().zip(out.chunks_exact_mut(rows.max(1))) {
            o.copy_from_slice(&tq19_matvec_par(data, rows, cols, &x[..cols]));
        }
        return;
    }
    let scale = compute_scale();
    let sink = BatchOut::new(out, rows, batch_size, "tq19_matvec_batch_par_into");
    (0..rows).into_par_iter().for_each_init(
        || vec![compute_zero(); batch_size],
        |accs, row| {
            let row_start = row * cols;
            accs.iter_mut().for_each(|a| *a = compute_zero());

            let mut tile_start = 0;
            while tile_start < cols {
                let tile_end = (tile_start + BATCH_TILE).min(cols);
                let tile_weights = &data[row_start + tile_start..row_start + tile_end];

                for b in 0..batch_size {
                    let tile_acts = &batch[b][tile_start..tile_end];
                    for i in 0..tile_weights.len() {
                        accs[b] = accs[b] + widen_weight(tile_weights[i]) * widen_activation(tile_acts[i]);
                    }
                }

                tile_start = tile_end;
            }

            for (b, acc) in accs.iter().enumerate() {
                // SAFETY: this task is the only one writing row `row`.
                unsafe { sink.write(b, row, narrow_to_storage(*acc / scale)) };
            }
        },
    );
}

/// Parallelizes across rows via rayon. Each row uses tiled accumulation:
/// processes BATCH_TILE elements across all batch vectors before advancing,
/// keeping weight tile + activation tiles in L1 cache together.
// rayon always available — module is gated behind inference feature
pub fn tq19_matvec_batch_par(
    data: &[i16],
    rows: usize,
    cols: usize,
    batch: &[&[BinaryStorage]],
) -> Vec<Vec<BinaryStorage>> {
    if !unchecked_sum_fits(cols) {
        return batch.iter().map(|x| tq19_matvec_par(data, rows, cols, &x[..cols])).collect();
    }
    let batch_size = batch.len();
    let scale = compute_scale();

    // Parallel: each row produces batch_size results with tiled accumulation
    let row_results: Vec<Vec<BinaryStorage>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let row_start = row * cols;
            let mut accs = vec![compute_zero(); batch_size];

            let mut tile_start = 0;
            while tile_start < cols {
                let tile_end = (tile_start + BATCH_TILE).min(cols);
                let tile_weights = &data[row_start + tile_start..row_start + tile_end];

                for b in 0..batch_size {
                    let tile_acts = &batch[b][tile_start..tile_end];
                    for i in 0..tile_weights.len() {
                        accs[b] = accs[b] + widen_weight(tile_weights[i]) * widen_activation(tile_acts[i]);
                    }
                }

                tile_start = tile_end;
            }

            accs.into_iter()
                .map(|acc| narrow_to_storage(acc / scale))
                .collect()
        })
        .collect();

    // Transpose: row_results[row][batch] → results[batch][row]
    let mut results: Vec<Vec<BinaryStorage>> = (0..batch_size)
        .map(|_| Vec::with_capacity(rows))
        .collect();
    for row_result in row_results {
        for (b, val) in row_result.into_iter().enumerate() {
            results[b].push(val);
        }
    }
    results
}

/// Row-parallel packed trit matvec.
// rayon always available — module is gated behind inference feature
pub fn packed_trit_matvec_par(
    packed_trits: &[u8],
    rows: usize,
    cols: usize,
    activations: &[BinaryStorage],
    scales: &[BinaryStorage],
) -> Vec<BinaryStorage> {
    assert!(activations.len() >= cols);
    assert!(scales.len() >= rows);

    let bytes_per_row = (cols + 4) / 5;
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let start = row * bytes_per_row;
            let row_trits = &packed_trits[start..start + bytes_per_row];
            packed_trit_dot(row_trits, cols, activations, scales[row])
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::imperative::FixedPoint;

    /// Helper: create BinaryStorage for a known value via FixedPoint.
    fn fp_raw(s: &str) -> BinaryStorage {
        if s.starts_with('-') {
            (-FixedPoint::from_str(&s[1..])).raw()
        } else {
            FixedPoint::from_str(s).raw()
        }
    }

    /// Profile-aware BinaryStorage constants for assertions.
    fn bs_zero() -> BinaryStorage { narrow_to_storage(compute_zero()) }
    fn bs_one() -> BinaryStorage { narrow_to_storage(compute_scale() / compute_scale()) }

    #[test]
    fn tq19_dot_identity_weight() {
        // Weight = SCALE means TQ1.9 value = 1.0
        // So dot([SCALE], [activation]) / SCALE = activation
        let act = fp_raw("1.5");
        let result = tq19_dot(&[SCALE as i16], &[act]);
        // Should be very close to activation (within 1 ULP from SCALE rounding)
        let diff = if result > act { result - act } else { act - result };
        // Allow 1 ULP tolerance
        assert!(diff <= bs_one(), "identity weight: diff = {diff:?}");
    }

    /// Contract pin for the wide epilogue:
    /// `q2f / (1 << FRAC_BITS)` (truncating division) must reproduce the
    /// narrow matvec BIT-FOR-BIT: a theorem for nested truncation toward
    /// zero, pinned here on random full-range signed data.
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    #[test]
    fn q2f_narrow_reproduces_matvec_bit_for_bit() {
        fn narrow_q2f(v: ComputeStorage) -> BinaryStorage {
            #[cfg(table_format = "q16_16")]
            { (v / (1i64 << crate::fixed_point::frac_config::FRAC_BITS)) as i32 }
            #[cfg(table_format = "q32_32")]
            { (v / (1i128 << 32)) as i64 }
        }
        let rows = 9;
        let cols = 131; // not divisible by SIMD lane widths
        let data: Vec<i16> = (0..rows * cols)
            .map(|i| ((i as i64 * 48271 % 59049) - 29524) as i16)
            .collect();
        let x: Vec<BinaryStorage> = (0..cols)
            .map(|i| ((i as i64 * 40503 % 8191) - 4095) as BinaryStorage)
            .collect();
        let narrow = tq19_matvec(&data, rows, cols, &x);
        let wide = tq19_matvec_q2f(&data, rows, cols, &x);
        for r in 0..rows {
            assert_eq!(narrow_q2f(wide[r]), narrow[r], "row {r}: narrow(q2f) != matvec");
        }
        // Parallel and batch variants must agree with sequential exactly.
        assert_eq!(wide, tq19_matvec_q2f_par(&data, rows, cols, &x));
        let batch: Vec<&[BinaryStorage]> = vec![&x, &x];
        let wb = tq19_matvec_q2f_batch_par(&data, rows, cols, &batch);
        assert_eq!(wb[0], wide);
        assert_eq!(wb[1], wide);
    }

    /// wide_output against a direct i128 reference, including the
    /// fail-loud guard at the compute-range boundary.
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    #[test]
    fn wide_output_matches_reference_and_guards_overflow() {
        #[cfg(table_format = "q16_16")]
        {
            let f = crate::fixed_point::frac_config::FRAC_BITS;
            for acc in [0i64, 1, -1, 12345, -987654321, 1 << 40, -(1i64 << 40)] {
                let reference = ((acc as i128) << f) / (SCALE as i128);
                assert_eq!(wide_output(acc) as i128, reference, "acc = {acc}");
            }
            // i64::MAX: in range for FRAC_BITS ≤ 14 (SCALE > 2^14), must
            // panic above — branch on the actual FRAC configuration.
            let exact = ((i64::MAX as i128) << f) / (SCALE as i128);
            if exact <= i64::MAX as i128 {
                assert_eq!(wide_output(i64::MAX) as i128, exact);
            } else {
                assert!(std::panic::catch_unwind(|| wide_output(i64::MAX)).is_err(),
                    "wide_output must fail loud past the compute range");
            }
        }
        #[cfg(table_format = "q32_32")]
        {
            // Staged division must equal the single division wherever the
            // single division is directly computable in i128.
            for acc in [0i128, 1, -1, 12345, -987654321, (i64::MAX as i128) * 1000, -(i64::MAX as i128) * 1000] {
                let reference = (acc << 32) / (SCALE as i128);
                assert_eq!(wide_output(acc), reference, "acc = {acc}");
            }
            assert!(std::panic::catch_unwind(|| wide_output(i128::MAX)).is_err(),
                "wide_output must fail loud past the compute range");
        }
    }

    #[test]
    fn tq19_dot_zero_weights() {
        let activations: Vec<BinaryStorage> = (0..4).map(|i| fp_raw(&format!("{}.0", i + 1))).collect();
        let weights = vec![0i16; 4];
        let result = tq19_dot(&weights, &activations);
        assert_eq!(result, bs_zero(), "zero weights should produce zero");
    }

    #[test]
    fn trit_dot_all_positive() {
        // All trits = +1: result = sum of activations
        let a1 = fp_raw("1.0");
        let a2 = fp_raw("2.0");
        let a3 = fp_raw("3.0");
        let activations = vec![a1, a2, a3];
        let trits = vec![1i8, 1, 1];
        let result = trit_dot(&trits, &activations);
        let expected = fp_raw("6.0");
        let diff = if result > expected { result - expected } else { expected - result };
        assert!(diff <= bs_one(), "all-positive trits: diff = {diff:?}");
    }

    #[test]
    fn trit_dot_mixed() {
        // [+1, 0, -1] · [1.0, 2.0, 3.0] = 1.0 + 0 - 3.0 = -2.0
        let activations = vec![fp_raw("1.0"), fp_raw("2.0"), fp_raw("3.0")];
        let trits = vec![1i8, 0, -1];
        let result = trit_dot(&trits, &activations);
        let expected = fp_raw("-2.0");
        let diff = if result > expected { result - expected } else { expected - result };
        assert!(diff <= bs_one(), "mixed trits: diff = {diff:?}");
    }

    #[test]
    fn tq19_matvec_identity_matrix() {
        // Identity-like: diagonal = SCALE, off-diagonal = 0
        let n = 3;
        let mut data = vec![0i16; n * n];
        for i in 0..n {
            data[i * n + i] = SCALE as i16;
        }
        let activations: Vec<BinaryStorage> = vec![fp_raw("1.0"), fp_raw("2.0"), fp_raw("3.0")];
        let result = tq19_matvec(&data, n, n, &activations);
        for i in 0..n {
            let diff = if result[i] > activations[i] { result[i] - activations[i] }
                else { activations[i] - result[i] };
            assert!(diff <= bs_one(), "identity matvec row {i}: diff = {diff:?}");
        }
    }

    #[test]
    fn tq19_matvec_batch_matches_sequential() {
        let n = 4;
        let data: Vec<i16> = (0..n * n).map(|i| ((i as i16) * 137) % (SCALE as i16)).collect();
        let v1: Vec<BinaryStorage> = (0..n).map(|i| fp_raw(&format!("{}.5", i))).collect();
        let v2: Vec<BinaryStorage> = (0..n).map(|i| fp_raw(&format!("{}.25", i + 1))).collect();

        let seq1 = tq19_matvec(&data, n, n, &v1);
        let seq2 = tq19_matvec(&data, n, n, &v2);
        let batch = tq19_matvec_batch(&data, n, n, &[&v1, &v2]);

        assert_eq!(batch[0], seq1, "batch[0] must match sequential");
        assert_eq!(batch[1], seq2, "batch[1] must match sequential");
    }

    #[test]
    fn packed_trit_dot_matches_trit_dot() {
        // Encode 7 trits: [+1, -1, 0, +1, +1, -1, 0]
        // Pack: first 5 in byte 0, last 2 in byte 1
        let trits_i8: Vec<i8> = vec![1, -1, 0, 1, 1, -1, 0];
        let packed = encode_trits_for_test(&trits_i8);

        let activations: Vec<BinaryStorage> = (0..7)
            .map(|i| fp_raw(&format!("{}.0", i + 1)))
            .collect();

        // Identity scale (1.0 in Q-format)
        let one_raw = fp_raw("1.0");

        let trit_result = trit_dot(&trits_i8, &activations);
        let packed_result = packed_trit_dot(&packed, 7, &activations, one_raw);

        // packed_trit_dot applies scale via mul_fixed which introduces rounding
        // Allow 2 ULP tolerance
        let diff = if packed_result > trit_result { packed_result - trit_result }
            else { trit_result - packed_result };
        let tolerance = bs_one() + bs_one();
        assert!(diff <= tolerance, "packed vs trit dot: diff = {diff:?}");
    }

    /// Test helper: encode i8 trits to packed bytes.
    fn encode_trits_for_test(trits: &[i8]) -> Vec<u8> {
        let mut packed = Vec::new();
        for chunk in trits.chunks(5) {
            let mut byte = 0u8;
            for (j, &t) in chunk.iter().enumerate() {
                let d = (t + 1) as u8; // {-1,0,1} → {0,1,2}
                byte += d * [81, 27, 9, 3, 1][j];
            }
            // Pad remaining positions with Zero (1)
            for j in chunk.len()..5 {
                byte += [81, 27, 9, 3, 1][j]; // Zero = 1
            }
            packed.push(byte);
        }
        packed
    }
}
