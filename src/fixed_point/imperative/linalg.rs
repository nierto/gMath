//! Linear algebra helpers with compute-tier precision.
//!
//! Core routines:
//! - `compute_tier_dot`: accumulates dot products at tier N+1 (double width)
//! - `compute_tier_sub_dot_compute`: init-minus-dot at compute tier
//! - `Rotation`, `householder_vector_compute`, `reflect_compute`: orthogonal
//!   transforms on compute-tier state, each output rounded once at the
//!   compute tier from its exact value
//!
//! These are the matrix-operation analog of BinaryCompute chain persistence.

use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
#[cfg(table_format = "q16_16")]
use crate::fixed_point::frac_config;
#[cfg(table_format = "q64_64")]
use crate::fixed_point::I256;

#[cfg(table_format = "q128_128")]
use crate::fixed_point::I512;

#[cfg(table_format = "q256_256")]
use crate::fixed_point::I1024;

// Q32.32 and Q16.16 use native integer types only — no I256/I512/I1024 imports needed.
// BinaryStorage = i64 / i32, ComputeStorage = i128 / i64 (already native).

// Re-export ComputeStorage for fused operations
pub(crate) use crate::fixed_point::universal::fasc::stack_evaluator::ComputeStorage;
pub(crate) use crate::fixed_point::universal::fasc::stack_evaluator::compute::downscale_to_storage;

// Re-export fused sincos for imperative-module consumers (lie_group, etc.)

// ============================================================================
// Rounding helper: round-to-nearest when downscaling from compute to storage
// ============================================================================

/// Downscale a compute-tier accumulator to storage tier with round-to-nearest.
/// This matches FASC's `downscale_to_storage` behavior, NOT truncation.
///
/// Panics if the result exceeds the storage tier's range, matching the
/// infallible imperative transcendentals. The previous shift-and-cast
/// fallback silently wrapped, which on narrow profiles turned an oversized
/// fused result into garbage (e.g. a negative squared distance).
#[inline]
pub(crate) fn round_to_storage(acc: ComputeStorage) -> BinaryStorage {
    downscale_to_storage(acc)
        .expect("round_to_storage: result exceeds storage tier range")
}

/// Upscale a storage-tier value to compute tier (shift left by FRAC_BITS).
#[inline]
pub(crate) fn upscale_to_compute(val: BinaryStorage) -> ComputeStorage {
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(val) << 64usize }
    #[cfg(table_format = "q32_32")]
    { (val as i128) << 32 }
    #[cfg(table_format = "q16_16")]
    { (val as i64) << frac_config::FRAC_BITS }
    #[cfg(table_format = "q128_128")]
    { I512::from_i256(val) << 128usize }
    #[cfg(table_format = "q256_256")]
    { I1024::from_i512(val) << 256usize }
}

// ============================================================================
// Compute-tier dot product
// ============================================================================
/// Below this length the per-term check is cheaper than the bound pass
/// (measured: the two cross between 16 and 32 elements). The length test
/// is made inline by `Storage::dot_acc_fast`; only longer vectors reach the
/// out-of-line [`long_dot_acc_i32`].
#[cfg(target_arch = "x86_64")]
pub(crate) const BOUNDED_DOT_MIN_LEN: usize = 32;

/// The 32-bit word's fast dot accumulator for `BOUNDED_DOT_MIN_LEN` or more
/// elements: the bounded unchecked AVX2 sum where AVX2 is present and the
/// bound holds, `None` otherwise (the generic checked loop then runs; it
/// panics where the sum really overflows). Out of line, so the short-vector
/// path of the callers stays as small as it was.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub(crate) fn long_dot_acc_i32(a: &[i32], b: &[i32]) -> Option<i64> {
    if !std::is_x86_feature_detected!("avx2") {
        return None;
    }
    // SAFETY: AVX2 was just detected.
    unsafe { bounded_dot_avx2(a, b) }
}

/// Largest magnitude in a slice (`|i32::MIN|` is `2^31`, hence unsigned).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn max_abs_avx2(x: &[i32]) -> u32 {
    let mut m = 0u32;
    for &v in x {
        m = m.max(v.unsigned_abs());
    }
    m
}

/// `sum a_i * b_i` in i64 with no per-term check, for inputs whose bound
/// `len * max|a| * max|b| < 2^63` proves that no partial sum can leave i64 in
/// any order of summation: the same integer as the checked loop, in a form
/// the compiler vectorises. `None` when the bound does not hold (the caller
/// then takes the checked loop, which panics where the sum really overflows).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bounded_dot_avx2(a: &[i32], b: &[i32]) -> Option<i64> {
    let bound = (a.len() as u128) * (max_abs_avx2(a) as u128) * (max_abs_avx2(b) as u128);
    if bound >= 1u128 << 63 {
        return None;
    }
    Some(unchecked_dot_avx2(a, b))
}

/// `sum a_i * b_i` in i64, wrapping adds: the caller has proven the bound.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn unchecked_dot_avx2(a: &[i32], b: &[i32]) -> i64 {
    let mut acc = 0i64;
    for (&x, &y) in a.iter().zip(b) {
        acc = acc.wrapping_add((x as i64) * (y as i64));
    }
    acc
}


// ============================================================================
// Compute-tier orthogonal transforms for the iterative decompositions
// ============================================================================
//
// Jacobi, Golub-Kahan and Francis converge only if the transforms they apply
// inject less rounding noise than their convergence tests resolve. A rotation
// coefficient or Householder factor rounded to storage precision injects about
// |x| ulp into every entry it touches (the rotation is no longer orthogonal to
// storage precision). Here every coefficient stays at the compute tier and
// every transformed entry is narrowed once from an exact accumulator, so a
// transform adds at most about half an ulp per entry. Every step is checked:
// leaving the storage range is a `TierOverflow`, never a wrap.


/// Consecutive non-improving iterations after which an iterative decomposition
/// is taken to sit at its precision floor.
pub(crate) const STAGNATION_SWEEPS: usize = 5;

/// Storage fraction bits of the build.
// used by the fused kernels on the wider profiles
#[allow(dead_code)]
#[inline]
pub(crate) fn storage_frac_bits() -> u32 {
    <crate::fixed_point::fixed_num::ProfileFormat as crate::fixed_point::fixed_num::FixedNum>::FRAC_BITS
}
/// Significant bits of `|v|` for a compute raw.
// used by the fused kernels on the wider profiles
#[allow(dead_code)]
#[inline]
pub(crate) fn compute_bit_length(v: ComputeStorage) -> u32 {
    super::kernels::compute_bit_length::<crate::fixed_point::fixed_num::ProfileFormat>(v)
}

/// `v >> shift` (arithmetic) for a compute raw.
// used by the fused kernels on the wider profiles
#[allow(dead_code)]
#[inline]
pub(crate) fn compute_shl(v: ComputeStorage, shift: u32) -> ComputeStorage {
    crate::fixed_point::width::WideInt::shl_bits(v, shift)
}

// used by the fused kernels on the wider profiles
#[allow(dead_code)]
#[inline]
pub(crate) fn compute_shr(v: ComputeStorage, shift: u32) -> ComputeStorage {
    crate::fixed_point::width::WideInt::shr_bits(v, shift)
}

// ============================================================================
// Convergence threshold for fixed-point iterative algorithms
// ============================================================================
// ============================================================================
// Compute-tier trit-weighted operations (zero-multiply dot product)
// Infrastructure ready for FASC ternary integration.
// ============================================================================

#[allow(unused_imports)]
use crate::fixed_point::domains::balanced_ternary::trit_packing::Trit;
// upscale_to_compute is defined locally above — no import needed

/// Trit-weighted dot product accumulated at compute tier (tier N+1).
///
/// For each trit in `packed_trits` (5 per byte, base-3 encoding):
///   - `+1` → `acc += widen(values[i])`
///   - `-1` → `acc -= widen(values[i])`
///   - ` 0` → skip (no operation)
///
/// The accumulator runs at ComputeStorage width. A single downscale
/// with rounding occurs at the very end. **Zero multiplications** in the
/// inner loop: only add/sub/skip.
///
/// After downscaling, the result is multiplied by `scale` (per-block
/// dequantization factor). The final multiply also uses compute-tier
/// intermediate to preserve precision.
///
/// # Arguments
/// - `packed_trits`: 5 trits per byte, base-3 encoded ({-1,0,+1} → {0,1,2})
/// - `num_elements`: exact number of trits (may be less than 5 × packed.len())
/// - `values`: activation vector in BinaryStorage format
/// - `scale`: per-block scale factor in BinaryStorage format
///
/// # Panics
/// Panics if `values.len() < num_elements`.
#[allow(dead_code)]
pub fn compute_tier_trit_dot_raw(
    packed_trits: &[u8],
    num_elements: usize,
    values: &[BinaryStorage],
    scale: BinaryStorage,
) -> BinaryStorage {
    assert!(values.len() >= num_elements, "compute_tier_trit_dot_raw: values shorter than num_elements");

    // Accumulate at compute tier (tier N+1) for full precision
    let mut acc = compute_zero();
    let mut trit_idx = 0;

    for &byte in packed_trits {
        if trit_idx >= num_elements {
            break;
        }

        // Unpack 5 trits from this byte (most-significant first)
        let mut remaining = byte;
        let mut chunk_trits = [1u8; 5]; // 1 = Zero (no-op)
        for j in (0..5).rev() {
            chunk_trits[j] = remaining % 3;
            remaining /= 3;
        }

        for j in 0..5 {
            if trit_idx >= num_elements {
                break;
            }

            let trit = chunk_trits[j];
            if trit == 2 {
                // Trit::Pos (+1): acc += widen(value)
                let widened = upscale_to_compute(values[trit_idx]);
                acc = compute_add(acc, widened);
            } else if trit == 0 {
                // Trit::Neg (-1): acc -= widen(value)
                let widened = upscale_to_compute(values[trit_idx]);
                acc = compute_sub(acc, widened);
            }
            // trit == 1 → Trit::Zero: skip (zero multiply eliminated)

            trit_idx += 1;
        }
    }

    // Single downscale of the accumulated dot product
    let dot_storage = round_to_storage(acc);

    // Apply per-block scale: result = dot * scale, at compute tier
    compute_tier_mul_pair(dot_storage, scale)
}

/// Row-wise trit-weighted matrix-vector product at compute tier.
///
/// Computes `result[row] = sum_j(trit[row][j] * values[j]) * scales[row]`
/// for each row, where the inner sum is a zero-multiply trit dot product.
///
/// # Arguments
/// - `packed_trits`: row-major packed trit matrix (each row = ceil(cols/5) bytes)
/// - `rows`: number of matrix rows
/// - `cols`: number of columns (= length of values vector)
/// - `values`: input activation vector
/// - `scales`: per-row scale factors (one per row)
///
/// # Returns
/// Output vector of length `rows`.
#[allow(dead_code)]
pub fn compute_tier_trit_matvec_raw(
    packed_trits: &[u8],
    rows: usize,
    cols: usize,
    values: &[BinaryStorage],
    scales: &[BinaryStorage],
) -> Vec<BinaryStorage> {
    assert!(values.len() >= cols, "compute_tier_trit_matvec_raw: values shorter than cols");
    assert!(scales.len() >= rows, "compute_tier_trit_matvec_raw: scales shorter than rows");

    let bytes_per_row = (cols + 4) / 5;
    let mut result = Vec::with_capacity(rows);

    for row in 0..rows {
        let row_start = row * bytes_per_row;
        let row_end = row_start + bytes_per_row;
        let row_trits = &packed_trits[row_start..row_end];

        let dot = compute_tier_trit_dot_raw(row_trits, cols, values, scales[row]);
        result.push(dot);
    }

    result
}

// Compute-tier helpers for trit operations
#[allow(dead_code)]
#[inline]
fn compute_zero() -> ComputeStorage {
    #[cfg(table_format = "q64_64")]
    { I256::zero() }
    #[cfg(table_format = "q32_32")]
    { 0i128 }
    #[cfg(table_format = "q16_16")]
    { 0i64 }
    #[cfg(table_format = "q128_128")]
    { I512::zero() }
    #[cfg(table_format = "q256_256")]
    { I1024::zero() }
}

#[allow(dead_code)]
#[inline]
fn compute_add(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage {
    a + b
}

#[allow(dead_code)]
#[inline]
fn compute_sub(a: ComputeStorage, b: ComputeStorage) -> ComputeStorage {
    a - b
}

/// Multiply two BinaryStorage values at compute tier with single downscale.
#[allow(dead_code)]
#[inline]
fn compute_tier_mul_pair(a: BinaryStorage, b: BinaryStorage) -> BinaryStorage {
    #[cfg(table_format = "q64_64")]
    {
        let a_wide = I256::from_i128(a);
        let b_wide = I256::from_i128(b);
        let product = a_wide * b_wide;
        round_to_storage(product)
    }
    #[cfg(table_format = "q32_32")]
    {
        // i64 × i64 → i128 (native widening, no I256 needed)
        let product = (a as i128) * (b as i128);
        round_to_storage(product)
    }
    #[cfg(table_format = "q16_16")]
    {
        // i32 × i32 → i64 (native widening, no I128 needed)
        let product = (a as i64) * (b as i64);
        round_to_storage(product)
    }
    #[cfg(table_format = "q128_128")]
    {
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let product = abs_a.mul_to_i512(abs_b);
        let product = if result_neg { -product } else { product };
        round_to_storage(product)
    }
    #[cfg(table_format = "q256_256")]
    {
        let a_neg = a.is_negative();
        let b_neg = b.is_negative();
        let result_neg = a_neg != b_neg;
        let abs_a = if a_neg { -a } else { a };
        let abs_b = if b_neg { -b } else { b };
        let product = abs_a.mul_to_i1024(abs_b);
        let product = if result_neg { -product } else { product };
        round_to_storage(product)
    }
}
