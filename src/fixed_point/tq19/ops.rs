//! TQ1.9 kernels: dot products, matvecs and their batched and parallel
//! forms, generic over the activations' format `Fixed<S, F>`. Weights are
//! i16 codes at scale 3^9; the activations' word and split come from the
//! argument.

use rayon::prelude::*;

use super::{SCALE, TRIT_DECODE_TABLE};
use crate::fixed_point::fixed_num::{Compute, Fixed, FixedNum};
use crate::fixed_point::width::{Storage, WideInt};

const STORAGE_OVERFLOW: &str = "tq19: result exceeds storage range";
const ACC_OVERFLOW: &str = "tq19: accumulator exceeds compute-tier range";

/// The storage words the inference kernels run on: every word. The 32-bit
/// word carries the AVX2 dot kernels, the exact fallback beyond 2^16 terms
/// and the i128 scale path; the 64-bit word its i128 scale path; the wider
/// words run the generic bodies. Sealed through `Storage`.
pub trait TqWord: Storage {
    /// `sum w_i * a_i` on the compute word by a word-specific path, `None`
    /// when there is none; the same integer as the checked loop.
    #[doc(hidden)]
    #[inline]
    fn tq19_dot_fast(_weights: &[i16], _activations: &[Self]) -> Option<Self::Compute> {
        None
    }
    /// `sum t_i * a_i` for trits by a word-specific path, `None` when there is none.
    #[doc(hidden)]
    #[inline]
    fn trit_dot_fast(_trits: &[i8], _activations: &[Self]) -> Option<Self::Compute> {
        None
    }
    /// Whether `len` worst-case terms sum at the compute tier without a
    /// per-term check (the 32-bit word's i64 accumulator holds 2^16 of them).
    #[doc(hidden)]
    #[inline]
    fn unchecked_sum_fits(_len: usize) -> bool {
        true
    }
    /// `round(acc * scale / 2^F)` to storage, nearest with ties toward +inf,
    /// fail-loud. The wide words narrow the accumulator first (both factors
    /// then fit the storage word and the product is exact at the compute width).
    #[doc(hidden)]
    #[inline]
    fn scale_dot<const F: u32>(acc: Self::Compute, scale: Self) -> Self {
        let v = Self::narrow(acc).unwrap_or_else(|| panic!("{}", STORAGE_OVERFLOW)).widen() * scale.widen();
        let round_bit = v.bit(F - 1);
        let shifted = v.shr_bits(F);
        let shifted = if round_bit { shifted + Self::Compute::one() } else { shifted };
        Self::narrow(shifted).unwrap_or_else(|| panic!("{}", STORAGE_OVERFLOW))
    }
}

impl TqWord for i128 {}
impl TqWord for crate::fixed_point::I256 {}
impl TqWord for crate::fixed_point::I512 {}

impl TqWord for i64 {
    #[inline]
    fn scale_dot<const F: u32>(acc: i128, scale: i64) -> i64 {
        let p = match acc.checked_mul(scale as i128) {
            Some(p) => p,
            None => panic!("{}", STORAGE_OVERFLOW),
        };
        let r = (p >> F) + ((p >> (F - 1)) & 1);
        if r > i64::MAX as i128 || r < i64::MIN as i128 { panic!("{}", STORAGE_OVERFLOW) }
        r as i64
    }
}

impl TqWord for i32 {
    #[inline]
    fn tq19_dot_fast(weights: &[i16], activations: &[i32]) -> Option<i64> {
        if !Self::unchecked_sum_fits(weights.len()) {
            return Some(tq19_dot_exact(weights, activations));
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && weights.len() >= 8 {
                return Some(unsafe { super::simd::tq19_dot_avx2(weights, activations) });
            }
        }
        None
    }
    #[inline]
    fn trit_dot_fast(trits: &[i8], activations: &[i32]) -> Option<i64> {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && trits.len() >= 8 {
                return Some(unsafe { super::simd::trit_dot_avx2(trits, activations) });
            }
        }
        let _ = (trits, activations);
        None
    }
    #[inline]
    fn unchecked_sum_fits(len: usize) -> bool {
        len <= 1 << 16
    }
    #[inline]
    fn scale_dot<const F: u32>(acc: i64, scale: i32) -> i32 {
        let p = acc as i128 * scale as i128;
        let r = (p >> F) + ((p >> (F - 1)) & 1);
        if r > i32::MAX as i128 || r < i32::MIN as i128 { panic!("{}", STORAGE_OVERFLOW) }
        r as i32
    }
}

/// The storage words with a wide matvec output (the exact row value at 2F
/// fractional bits, one truncation toward zero) and the row-scaled forms:
/// i32 and i64. Sealed through `Storage`.
pub trait WideOutput: TqWord {
    /// `trunc(acc * 2^F / SCALE)` at the compute tier, fail-loud.
    #[doc(hidden)]
    fn wide_output<const F: u32>(acc: Self::Compute) -> Self::Compute;
    /// `floor(dot * s / 2^32)` to storage for a Q32.32 row scale, fail-loud.
    #[doc(hidden)]
    fn rowscaled_scale(dot: Self, s_q32: u64) -> Self;
    /// `floor(wide * s / 2^32)` at the compute tier for a Q32.32 row scale, fail-loud.
    #[doc(hidden)]
    fn rowscaled_scale_q2f(wide: Self::Compute, s_q32: u64) -> Self::Compute;
}

impl WideOutput for i32 {
    #[inline(always)]
    fn wide_output<const F: u32>(acc: i64) -> i64 {
        // i64 acc, i128 intermediate: acc * 2^F is exact, one division
        let q2f = ((acc as i128) << F) / (SCALE as i128);
        if q2f > i64::MAX as i128 || q2f < i64::MIN as i128 {
            panic!("matvec_q2f: wide output exceeds compute-tier range");
        }
        q2f as i64
    }
    #[inline(always)]
    fn rowscaled_scale(dot: i32, s_q32: u64) -> i32 {
        let scaled = (dot as i128 * s_q32 as i128) >> 32;
        if scaled > i32::MAX as i128 || scaled < i32::MIN as i128 {
            panic!("RowScaledTQ19: scaled output exceeds storage range");
        }
        scaled as i32
    }
    #[inline(always)]
    fn rowscaled_scale_q2f(wide_dot: i64, s_q32: u64) -> i64 {
        let scaled = (wide_dot as i128 * s_q32 as i128) >> 32;
        if scaled > i64::MAX as i128 || scaled < i64::MIN as i128 {
            panic!("RowScaledTQ19: q2f scaled output exceeds compute range");
        }
        scaled as i64
    }
}

impl WideOutput for i64 {
    #[inline(always)]
    fn wide_output<const F: u32>(acc: i128) -> i128 {
        // i128 acc: acc * 2^32 can overflow i128, so stage the single
        // truncating division as q * 2^32 + trunc(r * 2^32 / S) with q = acc / S,
        // r = acc % S (|r| < S, sign of acc), identical to trunc(acc * 2^32 / S)
        let s = SCALE as i128;
        let q = acc / s;
        let frac = ((acc % s) << F) / s;
        match q.checked_mul(1i128 << F).and_then(|hi| hi.checked_add(frac)) {
            Some(v) => v,
            None => panic!("matvec_q2f: wide output exceeds compute-tier range"),
        }
    }
    #[inline(always)]
    fn rowscaled_scale(dot: i64, s_q32: u64) -> i64 {
        let scaled = (dot as i128 * s_q32 as i128) >> 32;
        if scaled > i64::MAX as i128 || scaled < i64::MIN as i128 {
            panic!("RowScaledTQ19: scaled output exceeds storage range");
        }
        scaled as i64
    }
    #[inline(always)]
    fn rowscaled_scale_q2f(wide_dot: i128, s_q32: u64) -> i128 {
        let h = wide_dot >> 32;
        let l = (wide_dot & 0xFFFF_FFFF) as i128;
        let low = (l * s_q32 as i128) >> 32;
        match h.checked_mul(s_q32 as i128).and_then(|hs| hs.checked_add(low)) {
            Some(v) => v,
            None => panic!("RowScaledTQ19: q2f scaled output exceeds compute range"),
        }
    }
}

// ============================================================================
// Helpers over the format
// ============================================================================

/// An i16 weight on the compute word (type-widen only, no Q-format shift).
#[inline(always)]
pub(super) fn widen_weight<S: Storage>(w: i16) -> S::Compute {
    S::Compute::from_i64(w as i64)
}

/// An activation on the compute word (type-widen only).
#[inline(always)]
pub(super) fn widen_activation<S: Storage, const F: u32>(a: Fixed<S, F>) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    a.raw().widen()
}

/// SCALE on the compute word.
#[inline(always)]
pub(super) fn compute_scale<S: Storage>() -> S::Compute {
    S::Compute::from_i64(SCALE as i64)
}

/// Zero on the compute word.
#[inline(always)]
pub(super) fn compute_zero<S: Storage>() -> S::Compute {
    S::Compute::ZERO
}

/// A compute value as a storage value (type-narrow only, no Q-format shift).
///
/// # Panics
/// Panics if the value does not fit the storage word (fail loud, never wrap).
#[inline]
pub(super) fn narrow_to_storage<S: Storage, const F: u32>(v: Compute<Fixed<S, F>>) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::from_raw(S::narrow(v).unwrap_or_else(|| panic!("{}", STORAGE_OVERFLOW)))
}

#[inline(always)]
fn unchecked_sum_fits<S: TqWord>(len: usize) -> bool {
    S::unchecked_sum_fits(len)
}

/// Wide matvec epilogue: the exact row value at 2F fractional bits with
/// exactly one rounding, truncation toward zero of `acc * 2^F / SCALE`.
#[inline(always)]
pub(super) fn wide_output<S: WideOutput, const F: u32>(acc: Compute<Fixed<S, F>>) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    S::wide_output::<F>(acc)
}

/// `round(acc * scale / 2^F)` to storage (nearest, ties toward +inf), fail-loud.
#[inline]
fn scale_dot<S: TqWord, const F: u32>(acc: Compute<Fixed<S, F>>, scale: Fixed<S, F>) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    Fixed::from_raw(S::scale_dot::<F>(acc, scale.raw()))
}

/// The exact i64 sum of a 32-bit row beyond 2^16 terms (i128 accumulator,
/// one range check), the cold path.
#[cold]
#[inline(never)]
fn tq19_dot_exact(weights: &[i16], activations: &[i32]) -> i64 {
    let mut acc = 0i128;
    for i in 0..weights.len() {
        acc += weights[i] as i128 * activations[i] as i128;
    }
    if acc > i64::MAX as i128 || acc < i64::MIN as i128 { panic!("{}", ACC_OVERFLOW) }
    acc as i64
}

/// `sum w_i * a_i` on the compute word: the word's fast path if it has one,
/// the plain loop otherwise.
#[inline]
fn tq19_dot_compute<S: TqWord, const F: u32>(weights: &[i16], activations: &[Fixed<S, F>]) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if let Some(acc) = S::tq19_dot_fast(weights, Fixed::raw_slice(activations)) {
        return acc;
    }
    let mut acc = compute_zero::<S>();
    for i in 0..weights.len() {
        acc = acc + widen_weight::<S>(weights[i]) * widen_activation(activations[i]);
    }
    acc
}

/// `sum t_i * a_i` for trits on the compute word.
#[inline]
fn trit_dot_compute<S: TqWord, const F: u32>(trits: &[i8], activations: &[Fixed<S, F>]) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if let Some(acc) = S::trit_dot_fast(trits, Fixed::raw_slice(activations)) {
        return acc;
    }
    let mut acc = compute_zero::<S>();
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

/// TQ1.9 dot: `sum(w[i] * a[i]) / SCALE` at compute tier.
/// Panics if the result exceeds storage.
pub fn tq19_dot<S: TqWord, const F: u32>(weights: &[i16], activations: &[Fixed<S, F>]) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    debug_assert_eq!(weights.len(), activations.len());
    let acc = tq19_dot_compute::<S, F>(weights, activations);
    narrow_to_storage::<S, F>(acc.divmod(compute_scale::<S>()).0)
}

/// Wide-output TQ1.9 dot: `trunc(sum(weights[i]·activations[i]) · 2^FRAC_BITS / SCALE)`.
///
/// Same inner loop (and SIMD dispatch) as [`tq19_dot`]; only the epilogue
/// differs: see [`wide_output`] for the exact rounding/narrowing contract.
pub fn tq19_dot_q2f<S: WideOutput, const F: u32>(weights: &[i16], activations: &[Fixed<S, F>]) -> Compute<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    debug_assert_eq!(weights.len(), activations.len());
    wide_output::<S, F>(tq19_dot_compute::<S, F>(weights, activations))
}

/// Zero-multiply trit dot for pre-decoded trits.
/// Panics if the result exceeds storage.
pub fn trit_dot<S: TqWord, const F: u32>(trits: &[i8], activations: &[Fixed<S, F>]) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    debug_assert_eq!(trits.len(), activations.len());
    narrow_to_storage::<S, F>(trit_dot_compute::<S, F>(trits, activations))
}

/// Packed trit dot with per-block scale.
///
/// Unpacks 5 trits/byte, accumulates at compute tier, then applies `scale`
/// with one rounding to nearest. Panics if the result exceeds storage.
pub fn packed_trit_dot<S: TqWord, const F: u32>(
    packed: &[u8],
    count: usize,
    activations: &[Fixed<S, F>],
    scale: Fixed<S, F>,
) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    assert!(activations.len() >= count, "packed_trit_dot: activations shorter than count");

    let mut acc = compute_zero::<S>();
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

    scale_dot::<S, F>(acc, scale)
}

// ============================================================================
// Sequential matvec
// ============================================================================

/// TQ1.9 matrix-vector product (sequential).
pub fn tq19_matvec<S: TqWord, const F: u32>(
    data: &[i16],
    rows: usize,
    cols: usize,
    activations: &[Fixed<S, F>],
) -> Vec<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let scale = compute_scale::<S>();
    (0..rows)
        .map(|row| {
            let row_weights = &data[row * cols..(row + 1) * cols];
            let acc = tq19_dot_compute::<S, F>(row_weights, activations);
            narrow_to_storage::<S, F>(acc.divmod(scale).0)
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
pub fn tq19_matvec_batch<S: TqWord, const F: u32>(
    data: &[i16],
    rows: usize,
    cols: usize,
    batch: &[&[Fixed<S, F>]],
) -> Vec<Vec<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if !unchecked_sum_fits::<S>(cols) {
        return batch.iter().map(|x| tq19_matvec(data, rows, cols, &x[..cols])).collect();
    }
    let batch_size = batch.len();
    let scale = compute_scale::<S>();
    let mut results: Vec<Vec<Fixed<S, F>>> = (0..batch_size)
        .map(|_| Vec::with_capacity(rows))
        .collect();

    // Per-batch accumulators, reused across rows
    let mut accs = vec![compute_zero::<S>(); batch_size];

    for row in 0..rows {
        // Reset accumulators
        for acc in accs.iter_mut() {
            *acc = compute_zero::<S>();
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
                    accs[b] = accs[b] + widen_weight::<S>(tile_weights[i]) * widen_activation(tile_acts[i]);
                }
            }

            tile_start = tile_end;
        }

        // Finalize: divide by SCALE, narrow, store
        for b in 0..batch_size {
            results[b].push(narrow_to_storage::<S, F>(accs[b].divmod(scale).0));
        }
    }

    results
}

/// Wide-output TQ1.9 matvec (sequential): 2·FRAC_BITS precision, one rounding.
///
/// Same inner loops as [`tq19_matvec`]; only the epilogue differs: see
/// [`wide_output`] for the exact rounding/narrowing contract.
pub fn tq19_matvec_q2f<S: WideOutput, const F: u32>(
    data: &[i16],
    rows: usize,
    cols: usize,
    activations: &[Fixed<S, F>],
) -> Vec<Compute<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    (0..rows)
        .map(|row| {
            let row_weights = &data[row * cols..(row + 1) * cols];
            wide_output::<S, F>(tq19_dot_compute::<S, F>(row_weights, activations))
        })
        .collect()
}

/// Packed trit matvec (sequential).
pub fn packed_trit_matvec<S: TqWord, const F: u32>(
    packed_trits: &[u8],
    rows: usize,
    cols: usize,
    activations: &[Fixed<S, F>],
    scales: &[Fixed<S, F>],
) -> Vec<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
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
pub fn tq19_matvec_par<S: TqWord, const F: u32>(
    data: &[i16],
    rows: usize,
    cols: usize,
    activations: &[Fixed<S, F>],
) -> Vec<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let scale = compute_scale::<S>();
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let row_weights = &data[row * cols..(row + 1) * cols];
            let acc = tq19_dot_compute::<S, F>(row_weights, activations);
            narrow_to_storage::<S, F>(acc.divmod(scale).0)
        })
        .collect()
}

/// Row-parallel wide-output TQ1.9 matvec: 2·FRAC_BITS precision, one rounding.
///
/// Epilogue-only variant of [`tq19_matvec_par`]: see [`wide_output`].
pub fn tq19_matvec_q2f_par<S: WideOutput, const F: u32>(
    data: &[i16],
    rows: usize,
    cols: usize,
    activations: &[Fixed<S, F>],
) -> Vec<Compute<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    (0..rows)
        .into_par_iter()
        .map(|row| {
            let row_weights = &data[row * cols..(row + 1) * cols];
            wide_output::<S, F>(tq19_dot_compute::<S, F>(row_weights, activations))
        })
        .collect()
}

/// Row-parallel wide-output batch TQ1.9 matvec with tiled accumulation.
/// Epilogue-only variant of [`tq19_matvec_batch_par`]: see [`wide_output`].
pub fn tq19_matvec_q2f_batch_par<S: WideOutput, const F: u32>(
    data: &[i16],
    rows: usize,
    cols: usize,
    batch: &[&[Fixed<S, F>]],
) -> Vec<Vec<Compute<Fixed<S, F>>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if !unchecked_sum_fits::<S>(cols) {
        return batch.iter().map(|x| tq19_matvec_q2f_par(data, rows, cols, &x[..cols])).collect();
    }
    let batch_size = batch.len();
    let row_results: Vec<Vec<Compute<Fixed<S, F>>>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let row_start = row * cols;
            let mut accs = vec![compute_zero::<S>(); batch_size];
            let mut tile_start = 0;
            while tile_start < cols {
                let tile_end = (tile_start + BATCH_TILE).min(cols);
                let tile_weights = &data[row_start + tile_start..row_start + tile_end];
                for b in 0..batch_size {
                    let tile_acts = &batch[b][tile_start..tile_end];
                    for i in 0..tile_weights.len() {
                        accs[b] = accs[b] + widen_weight::<S>(tile_weights[i]) * widen_activation(tile_acts[i]);
                    }
                }
                tile_start = tile_end;
            }
            accs.into_iter().map(wide_output::<S, F>).collect()
        })
        .collect();
    let mut results: Vec<Vec<Compute<Fixed<S, F>>>> = (0..batch_size)
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
pub(crate) fn tq19_matvec_batch_par_into<S: TqWord, const F: u32>(
    data: &[i16],
    rows: usize,
    cols: usize,
    batch: &[&[Fixed<S, F>]],
    out: &mut [Fixed<S, F>],
)
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    let batch_size = batch.len();
    assert_eq!(out.len(), batch_size * rows, "tq19_matvec_batch_par_into: out length is not batch.len() * rows");
    if !unchecked_sum_fits::<S>(cols) {
        for (x, o) in batch.iter().zip(out.chunks_exact_mut(rows.max(1))) {
            o.copy_from_slice(&tq19_matvec_par(data, rows, cols, &x[..cols]));
        }
        return;
    }
    let scale = compute_scale::<S>();
    let sink = BatchOut::new(out, rows, batch_size, "tq19_matvec_batch_par_into");
    (0..rows).into_par_iter().for_each_init(
        || vec![compute_zero::<S>(); batch_size],
        |accs, row| {
            let row_start = row * cols;
            accs.iter_mut().for_each(|a| *a = compute_zero::<S>());

            let mut tile_start = 0;
            while tile_start < cols {
                let tile_end = (tile_start + BATCH_TILE).min(cols);
                let tile_weights = &data[row_start + tile_start..row_start + tile_end];

                for b in 0..batch_size {
                    let tile_acts = &batch[b][tile_start..tile_end];
                    for i in 0..tile_weights.len() {
                        accs[b] = accs[b] + widen_weight::<S>(tile_weights[i]) * widen_activation(tile_acts[i]);
                    }
                }

                tile_start = tile_end;
            }

            for (b, acc) in accs.iter().enumerate() {
                // SAFETY: this task is the only one writing row `row`.
                unsafe { sink.write(b, row, narrow_to_storage::<S, F>((*acc).divmod(scale).0)) };
            }
        },
    );
}

/// Parallelizes across rows via rayon. Each row uses tiled accumulation:
/// processes BATCH_TILE elements across all batch vectors before advancing,
/// keeping weight tile + activation tiles in L1 cache together.
// rayon always available — module is gated behind inference feature
pub fn tq19_matvec_batch_par<S: TqWord, const F: u32>(
    data: &[i16],
    rows: usize,
    cols: usize,
    batch: &[&[Fixed<S, F>]],
) -> Vec<Vec<Fixed<S, F>>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    if !unchecked_sum_fits::<S>(cols) {
        return batch.iter().map(|x| tq19_matvec_par(data, rows, cols, &x[..cols])).collect();
    }
    let batch_size = batch.len();
    let scale = compute_scale::<S>();

    // Parallel: each row produces batch_size results with tiled accumulation
    let row_results: Vec<Vec<Fixed<S, F>>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let row_start = row * cols;
            let mut accs = vec![compute_zero::<S>(); batch_size];

            let mut tile_start = 0;
            while tile_start < cols {
                let tile_end = (tile_start + BATCH_TILE).min(cols);
                let tile_weights = &data[row_start + tile_start..row_start + tile_end];

                for b in 0..batch_size {
                    let tile_acts = &batch[b][tile_start..tile_end];
                    for i in 0..tile_weights.len() {
                        accs[b] = accs[b] + widen_weight::<S>(tile_weights[i]) * widen_activation(tile_acts[i]);
                    }
                }

                tile_start = tile_end;
            }

            accs.into_iter()
                .map(|acc| narrow_to_storage::<S, F>(acc.divmod(scale).0))
                .collect()
        })
        .collect();

    // Transpose: row_results[row][batch] → results[batch][row]
    let mut results: Vec<Vec<Fixed<S, F>>> = (0..batch_size)
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
pub fn packed_trit_matvec_par<S: TqWord, const F: u32>(
    packed_trits: &[u8],
    rows: usize,
    cols: usize,
    activations: &[Fixed<S, F>],
    scales: &[Fixed<S, F>],
) -> Vec<Fixed<S, F>>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
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
    #[allow(unused_imports)]
    use crate::fixed_point::universal::fasc::stack_evaluator::{BinaryStorage, ComputeStorage};
    #[allow(unused_imports)]
    use crate::fixed_point::FixedPoint;

    /// Helper: create BinaryStorage for a known value via FixedPoint.
    fn fp_raw(s: &str) -> BinaryStorage {
        if s.starts_with('-') {
            (-FixedPoint::from_str(&s[1..])).raw()
        } else {
            FixedPoint::from_str(s).raw()
        }
    }

    /// Profile-aware BinaryStorage constants for assertions.
    fn bs_zero() -> BinaryStorage { <BinaryStorage as crate::fixed_point::width::WideInt>::ZERO }
    fn bs_one() -> BinaryStorage { <BinaryStorage as crate::fixed_point::width::WideInt>::one() }

    #[test]
    fn tq19_dot_identity_weight() {
        // Weight = SCALE means TQ1.9 value = 1.0
        // So dot([SCALE], [activation]) / SCALE = activation
        let act = fp_raw("1.5");
        let result = tq19_dot(&[SCALE as i16], &[FixedPoint::from_raw(act)]);
        // Should be very close to activation (within 1 ULP from SCALE rounding)
        let diff = if result > FixedPoint::from_raw(act) { result - FixedPoint::from_raw(act) } else { FixedPoint::from_raw(act) - result };
        // Allow 1 ULP tolerance
        assert!(diff <= FixedPoint::from_raw(bs_one()), "identity weight: diff = {diff:?}");
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
        let narrow = tq19_matvec(&data, rows, cols, FixedPoint::from_raw_slice(&x));
        let wide = tq19_matvec_q2f(&data, rows, cols, FixedPoint::from_raw_slice(&x));
        for r in 0..rows {
            assert_eq!(narrow_q2f(wide[r]), narrow[r].raw(), "row {r}: narrow(q2f) != matvec");
        }
        // Parallel and batch variants must agree with sequential exactly.
        assert_eq!(wide, tq19_matvec_q2f_par(&data, rows, cols, FixedPoint::from_raw_slice(&x)));
        let batch: Vec<&[BinaryStorage]> = vec![&x, &x];
        let wb = tq19_matvec_q2f_batch_par(&data, rows, cols, &(&batch).iter().map(|v| FixedPoint::from_raw_slice(v)).collect::<Vec<_>>());
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
                assert_eq!(wide_output::<BinaryStorage, { crate::fixed_point::frac_config::FRAC_BITS }>(acc) as i128, reference, "acc = {acc}");
            }
            // i64::MAX: in range for FRAC_BITS ≤ 14 (SCALE > 2^14), must
            // panic above — branch on the actual FRAC configuration.
            let exact = ((i64::MAX as i128) << f) / (SCALE as i128);
            if exact <= i64::MAX as i128 {
                assert_eq!(wide_output::<BinaryStorage, { crate::fixed_point::frac_config::FRAC_BITS }>(i64::MAX) as i128, exact);
            } else {
                assert!(std::panic::catch_unwind(|| wide_output::<BinaryStorage, { crate::fixed_point::frac_config::FRAC_BITS }>(i64::MAX)).is_err(),
                    "wide_output must fail loud past the compute range");
            }
        }
        #[cfg(table_format = "q32_32")]
        {
            // Staged division must equal the single division wherever the
            // single division is directly computable in i128.
            for acc in [0i128, 1, -1, 12345, -987654321, (i64::MAX as i128) * 1000, -(i64::MAX as i128) * 1000] {
                let reference = (acc << 32) / (SCALE as i128);
                assert_eq!(wide_output::<BinaryStorage, 32>(acc), reference, "acc = {acc}");
            }
            assert!(std::panic::catch_unwind(|| wide_output::<BinaryStorage, 32>(i128::MAX)).is_err(),
                "wide_output must fail loud past the compute range");
        }
    }

    #[test]
    fn tq19_dot_zero_weights() {
        let activations: Vec<BinaryStorage> = (0..4).map(|i| fp_raw(&format!("{}.0", i + 1))).collect();
        let weights = vec![0i16; 4];
        let result = tq19_dot(&weights, FixedPoint::from_raw_slice(&activations));
        assert_eq!(result, FixedPoint::from_raw(bs_zero()), "zero weights should produce zero");
    }

    #[test]
    fn trit_dot_all_positive() {
        // All trits = +1: result = sum of activations
        let a1 = fp_raw("1.0");
        let a2 = fp_raw("2.0");
        let a3 = fp_raw("3.0");
        let activations = vec![a1, a2, a3];
        let trits = vec![1i8, 1, 1];
        let result = trit_dot(&trits, FixedPoint::from_raw_slice(&activations));
        let expected = fp_raw("6.0");
        let diff = if result > FixedPoint::from_raw(expected) { result - FixedPoint::from_raw(expected) } else { FixedPoint::from_raw(expected) - result };
        assert!(diff <= FixedPoint::from_raw(bs_one()), "all-positive trits: diff = {diff:?}");
    }

    #[test]
    fn trit_dot_mixed() {
        // [+1, 0, -1] · [1.0, 2.0, 3.0] = 1.0 + 0 - 3.0 = -2.0
        let activations = vec![fp_raw("1.0"), fp_raw("2.0"), fp_raw("3.0")];
        let trits = vec![1i8, 0, -1];
        let result = trit_dot(&trits, FixedPoint::from_raw_slice(&activations));
        let expected = fp_raw("-2.0");
        let diff = if result > FixedPoint::from_raw(expected) { result - FixedPoint::from_raw(expected) } else { FixedPoint::from_raw(expected) - result };
        assert!(diff <= FixedPoint::from_raw(bs_one()), "mixed trits: diff = {diff:?}");
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
        let result = tq19_matvec(&data, n, n, FixedPoint::from_raw_slice(&activations));
        for i in 0..n {
            let diff = if result[i] > FixedPoint::from_raw(activations[i]) { result[i] - FixedPoint::from_raw(activations[i]) }
                else { FixedPoint::from_raw(activations[i]) - result[i] };
            assert!(diff <= FixedPoint::from_raw(bs_one()), "identity matvec row {i}: diff = {diff:?}");
        }
    }

    #[test]
    fn tq19_matvec_batch_matches_sequential() {
        let n = 4;
        let data: Vec<i16> = (0..n * n).map(|i| ((i as i16) * 137) % (SCALE as i16)).collect();
        let v1: Vec<BinaryStorage> = (0..n).map(|i| fp_raw(&format!("{}.5", i))).collect();
        let v2: Vec<BinaryStorage> = (0..n).map(|i| fp_raw(&format!("{}.25", i + 1))).collect();

        let seq1 = tq19_matvec(&data, n, n, FixedPoint::from_raw_slice(&v1));
        let seq2 = tq19_matvec(&data, n, n, FixedPoint::from_raw_slice(&v2));
        let batch = tq19_matvec_batch(&data, n, n, &[FixedPoint::from_raw_slice(&v1), FixedPoint::from_raw_slice(&v2)]);

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

        let trit_result = trit_dot(&trits_i8, FixedPoint::from_raw_slice(&activations));
        let packed_result = packed_trit_dot(&packed, 7, FixedPoint::from_raw_slice(&activations), FixedPoint::from_raw(one_raw));

        // packed_trit_dot applies scale via mul_fixed which introduces rounding
        // Allow 2 ULP tolerance
        let diff = if packed_result > trit_result { packed_result - trit_result }
            else { trit_result - packed_result };
        let tolerance = bs_one() + bs_one();
        assert!(diff <= FixedPoint::from_raw(tolerance), "packed vs trit dot: diff = {diff:?}");
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
