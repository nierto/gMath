//! TQ1.9 core operations — profile-conditional implementation.
//!
//! All dot products accumulate at ComputeStorage (tier N+1).
//! Single division/downscale at the end per gMath precision contract.

use crate::fixed_point::universal::fasc::stack_evaluator::{BinaryStorage, ComputeStorage};

#[allow(unused_imports)]
use crate::fixed_point::I256;
#[allow(unused_imports)]
use crate::fixed_point::I512;
#[allow(unused_imports)]
use crate::fixed_point::I1024;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{SCALE, TRIT_DECODE_TABLE};

// ============================================================================
// Profile-conditional widening/narrowing helpers
//
// These isolate ALL cfg blocks so that the actual operations are generic.
// ============================================================================

/// Widen i16 weight value to ComputeStorage (type-widen only, no Q-format shift).
#[inline(always)]
fn widen_weight(w: i16) -> ComputeStorage {
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
fn widen_activation(a: BinaryStorage) -> ComputeStorage {
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
fn compute_scale() -> ComputeStorage {
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
fn compute_zero() -> ComputeStorage {
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

/// Narrow ComputeStorage to BinaryStorage (type-narrow only, no Q-format shift).
/// Truncates upper bits if value exceeds storage range.
#[inline(always)]
fn narrow_to_storage(v: ComputeStorage) -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    { v as i32 }
    #[cfg(table_format = "q32_32")]
    { v as i64 }
    #[cfg(table_format = "q64_64")]
    { v.as_i128() }
    #[cfg(table_format = "q128_128")]
    { v.as_i256() }
    #[cfg(table_format = "q256_256")]
    { v.as_i512() }
}

/// Multiply two Q-format BinaryStorage values at compute tier.
///
/// `result = (a * b) >> FRAC_BITS` with round-to-nearest.
/// Used for applying per-block scale factors to trit dot products.
#[inline]
fn mul_fixed(a: BinaryStorage, b: BinaryStorage) -> BinaryStorage {
    let a_wide = widen_activation(a);
    let b_wide = widen_activation(b);
    let product = a_wide * b_wide;
    shift_right_frac_and_narrow(product)
}

/// Shift ComputeStorage right by FRAC_BITS with rounding, narrow to BinaryStorage.
#[inline(always)]
fn shift_right_frac_and_narrow(v: ComputeStorage) -> BinaryStorage {
    #[cfg(table_format = "q16_16")]
    {
        let round = (v >> 15) & 1;
        ((v >> 16) + round) as i32
    }
    #[cfg(table_format = "q32_32")]
    {
        let round = (v >> 31) & 1;
        ((v >> 32) + round) as i64
    }
    #[cfg(table_format = "q64_64")]
    {
        let round_bit = (v & I256::from_i128(1i128 << 63)) != I256::zero();
        let shifted = (v >> 64u32).as_i128();
        if round_bit { shifted + 1 } else { shifted }
    }
    #[cfg(table_format = "q128_128")]
    {
        let round_bit = (v & (I512::from_i128(1) << 127usize)) != I512::zero();
        let shifted = (v >> 128usize).as_i256();
        if round_bit { shifted + I256::from_i128(1) } else { shifted }
    }
    #[cfg(table_format = "q256_256")]
    {
        let round_bit = (v & (I1024::from_i128(1) << 255usize)) != I1024::zero();
        let shifted = (v >> 256usize).as_i512();
        if round_bit { shifted + I512::from_i128(1) } else { shifted }
    }
}

// ============================================================================
// Inner dot products — return ComputeStorage (pre-division)
// ============================================================================

/// TQ1.9 inner dot product at compute tier (before SCALE division).
///
/// Returns raw accumulator. Caller divides by SCALE and narrows.
/// On x86_64 realtime profile, dispatches to AVX2 when available.
#[inline]
fn tq19_dot_compute(weights: &[i16], activations: &[BinaryStorage]) -> ComputeStorage {
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
pub fn tq19_dot(weights: &[i16], activations: &[BinaryStorage]) -> BinaryStorage {
    debug_assert_eq!(weights.len(), activations.len());
    let acc = tq19_dot_compute(weights, activations);
    narrow_to_storage(acc / compute_scale())
}

/// Zero-multiply trit dot for pre-decoded trits.
pub fn trit_dot(trits: &[i8], activations: &[BinaryStorage]) -> BinaryStorage {
    debug_assert_eq!(trits.len(), activations.len());
    narrow_to_storage(trit_dot_compute(trits, activations))
}

/// Packed trit dot with per-block scale.
///
/// Unpacks 5 trits/byte, accumulates at compute tier, downscales,
/// then multiplies by `scale` at compute tier.
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

    // Narrow accumulated dot, then apply Q-format scale multiply
    let dot = narrow_to_storage(acc);
    mul_fixed(dot, scale)
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
/// For batch=8: 33 KB — fits in 32-48 KB L1d.
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
#[cfg(feature = "parallel")]
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

/// Row-parallel batch TQ1.9 matvec with tiled accumulation.
///
/// Parallelizes across rows via rayon. Each row uses tiled accumulation:
/// processes BATCH_TILE elements across all batch vectors before advancing,
/// keeping weight tile + activation tiles in L1 cache together.
#[cfg(feature = "parallel")]
pub fn tq19_matvec_batch_par(
    data: &[i16],
    rows: usize,
    cols: usize,
    batch: &[&[BinaryStorage]],
) -> Vec<Vec<BinaryStorage>> {
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
#[cfg(feature = "parallel")]
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
