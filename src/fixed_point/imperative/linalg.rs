//! Linear algebra helpers with compute-tier precision.
//!
//! Core routines:
//! - `compute_tier_dot` — accumulates dot products at tier N+1 (double width)
//! - `compute_tier_sub_dot_raw` — fused init-minus-dot at compute tier
//! - `givens` — Givens rotation without trig (ratio + sqrt only)
//!
//! These are the matrix-operation analog of BinaryCompute chain persistence.

use super::FixedPoint;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
#[cfg(table_format = "q64_64")]
use crate::fixed_point::I256;

#[cfg(table_format = "q128_128")]
use crate::fixed_point::{I256, I512};

#[cfg(table_format = "q256_256")]
use crate::fixed_point::{I512, I1024};

// Q32.32 and Q16.16 use native integer types only — no I256/I512/I1024 imports needed.
// BinaryStorage = i64 / i32, ComputeStorage = i128 / i64 (already native).

// Re-export ComputeStorage for fused operations
pub(crate) use crate::fixed_point::universal::fasc::stack_evaluator::ComputeStorage;
use crate::fixed_point::universal::fasc::stack_evaluator::compute::downscale_to_storage;

// Re-export fused sincos for imperative-module consumers (lie_group, etc.)
pub(crate) use crate::fixed_point::universal::fasc::stack_evaluator::compute::sincos_at_compute_tier;

// ============================================================================
// Rounding helper: round-to-nearest when downscaling from compute to storage
// ============================================================================

/// Downscale a compute-tier accumulator to storage tier with round-to-nearest.
/// This matches FASC's `downscale_to_storage` behavior — NOT truncation.
#[inline]
pub(crate) fn round_to_storage(acc: ComputeStorage) -> BinaryStorage {
    // Use the existing FASC downscale which includes rounding.
    // If the value overflows storage tier, this returns Err — we unwrap
    // because accumulator overflow should have been caught earlier.
    // For decomposition inner sums, overflow is extremely unlikely.
    match downscale_to_storage(acc) {
        Ok(v) => v,
        Err(_) => {
            // Fallback: truncate without rounding (matches old behavior)
            #[cfg(table_format = "q64_64")]
            { (acc >> 64u32).as_i128() }
            #[cfg(table_format = "q32_32")]
            { (acc >> 32) as i64 }
            #[cfg(table_format = "q16_16")]
            { (acc >> 16) as i32 }
            #[cfg(table_format = "q128_128")]
            { (acc >> 128usize).as_i256() }
            #[cfg(table_format = "q256_256")]
            { (acc >> 256usize).as_i512() }
        }
    }
}

/// Upscale a storage-tier value to compute tier (shift left by FRAC_BITS).
#[inline]
pub(crate) fn upscale_to_compute(val: BinaryStorage) -> ComputeStorage {
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(val) << 64usize }
    #[cfg(table_format = "q32_32")]
    { (val as i128) << 32 }
    #[cfg(table_format = "q16_16")]
    { (val as i64) << 16 }
    #[cfg(table_format = "q128_128")]
    { I512::from_i256(val) << 128usize }
    #[cfg(table_format = "q256_256")]
    { I1024::from_i512(val) << 256usize }
}

// ============================================================================
// Compute-tier dot product
// ============================================================================

/// Dot product accumulated at tier N+1 (compute tier).
///
/// Each product a_i * b_i is computed at double width without truncation.
/// The entire sum is accumulated at double width. Only one rounding step
/// occurs at the very end when truncating back to storage tier.
///
/// For an n-element dot product, this gives 1 ULP of rounding error instead
/// of the n ULP that storage-tier accumulation would produce.
///
/// Panics if the slices have different lengths.
pub fn compute_tier_dot(a: &[FixedPoint], b: &[FixedPoint]) -> FixedPoint {
    assert_eq!(a.len(), b.len(), "compute_tier_dot: length mismatch");

    #[cfg(table_format = "q64_64")]
    {
        // i128 × i128 → I256 (Q128.128), accumulate in I256, shift >> 64
        let mut acc = I256::zero();
        for i in 0..a.len() {
            let a_wide = I256::from_i128(a[i].raw());
            let b_wide = I256::from_i128(b[i].raw());
            // I256 * I256 → I256 (no overflow: inputs are sign-extended i128,
            // so the mathematical product fits in 256 bits)
            acc = acc + (a_wide * b_wide);
        }
        // Shift right by FRAC_BITS (64) to convert from Q128.128 to Q64.64
        FixedPoint::from_raw((acc >> 64u32).as_i128())
    }

    #[cfg(table_format = "q32_32")]
    {
        // i64 × i64 → i128 (Q64.64), accumulate in i128, shift >> 32
        let mut acc: i128 = 0;
        for i in 0..a.len() {
            acc += (a[i].raw() as i128) * (b[i].raw() as i128);
        }
        FixedPoint::from_raw((acc >> 32) as i64)
    }

    #[cfg(table_format = "q16_16")]
    {
        // i32 × i32 → i64 (Q32.32), accumulate in i64, shift >> 16
        let mut acc: i64 = 0;
        for i in 0..a.len() {
            acc += (a[i].raw() as i64) * (b[i].raw() as i64);
        }
        FixedPoint::from_raw((acc >> 16) as i32)
    }

    #[cfg(table_format = "q128_128")]
    {
        // I256 × I256 → I512 (Q256.256), accumulate in I512, shift >> 128
        let mut acc = I512::zero();
        for i in 0..a.len() {
            let a_raw = a[i].raw();
            let b_raw = b[i].raw();
            // Signed widening multiply: I256 × I256 → I512
            let a_neg = a_raw.is_negative();
            let b_neg = b_raw.is_negative();
            let result_neg = a_neg != b_neg;
            let abs_a = if a_neg { -a_raw } else { a_raw };
            let abs_b = if b_neg { -b_raw } else { b_raw };
            let product = abs_a.mul_to_i512(abs_b);
            let signed_product = if result_neg { -product } else { product };
            acc = acc + signed_product;
        }
        FixedPoint::from_raw((acc >> 128usize).as_i256())
    }

    #[cfg(table_format = "q256_256")]
    {
        // I512 × I512 → I1024 (Q512.512), accumulate in I1024, shift >> 256
        let mut acc = I1024::zero();
        for i in 0..a.len() {
            let a_raw = a[i].raw();
            let b_raw = b[i].raw();
            // Signed widening multiply: I512 × I512 → I1024
            let a_neg = a_raw.is_negative();
            let b_neg = b_raw.is_negative();
            let result_neg = a_neg != b_neg;
            let abs_a = if a_neg { -a_raw } else { a_raw };
            let abs_b = if b_neg { -b_raw } else { b_raw };
            let product = abs_a.mul_to_i1024(abs_b);
            let signed_product = if result_neg { -product } else { product };
            acc = acc + signed_product;
        }
        FixedPoint::from_raw((acc >> 256usize).as_i512())
    }
}

/// Compute-tier multiply-accumulate: acc += a_i * b_i for matrix operations.
///
/// Same as `compute_tier_dot` but takes raw BinaryStorage slices for
/// internal use where the FixedPoint wrapper would add unnecessary overhead.
#[inline]
pub(crate) fn compute_tier_dot_raw(a: &[BinaryStorage], b: &[BinaryStorage]) -> BinaryStorage {
    assert_eq!(a.len(), b.len(), "compute_tier_dot_raw: length mismatch");

    #[cfg(table_format = "q64_64")]
    {
        let mut acc = I256::zero();
        for i in 0..a.len() {
            acc = acc + (I256::from_i128(a[i]) * I256::from_i128(b[i]));
        }
        round_to_storage(acc)
    }

    #[cfg(table_format = "q32_32")]
    {
        // i64 × i64 → i128, accumulate in i128
        let mut acc: i128 = 0;
        for i in 0..a.len() {
            acc += (a[i] as i128) * (b[i] as i128);
        }
        round_to_storage(acc)
    }

    #[cfg(table_format = "q16_16")]
    {
        // i32 × i32 → i64, accumulate in i64
        let mut acc: i64 = 0;
        for i in 0..a.len() {
            acc += (a[i] as i64) * (b[i] as i64);
        }
        round_to_storage(acc)
    }

    #[cfg(table_format = "q128_128")]
    {
        let mut acc = I512::zero();
        for i in 0..a.len() {
            let a_neg = a[i].is_negative();
            let b_neg = b[i].is_negative();
            let result_neg = a_neg != b_neg;
            let abs_a = if a_neg { -a[i] } else { a[i] };
            let abs_b = if b_neg { -b[i] } else { b[i] };
            let product = abs_a.mul_to_i512(abs_b);
            acc = acc + if result_neg { -product } else { product };
        }
        round_to_storage(acc)
    }

    #[cfg(table_format = "q256_256")]
    {
        let mut acc = I1024::zero();
        for i in 0..a.len() {
            let a_neg = a[i].is_negative();
            let b_neg = b[i].is_negative();
            let result_neg = a_neg != b_neg;
            let abs_a = if a_neg { -a[i] } else { a[i] };
            let abs_b = if b_neg { -b[i] } else { b[i] };
            let product = abs_a.mul_to_i1024(abs_b);
            acc = acc + if result_neg { -product } else { product };
        }
        round_to_storage(acc)
    }
}

// ============================================================================
// Fused init-minus-dot at compute tier
// ============================================================================

/// Compute `init - dot(a, b)` entirely at compute tier (tier N+1).
///
/// Widens `init` to compute-tier format, subtracts each product a_i * b_i
/// accumulated at compute tier, then rounds back to storage tier once.
///
/// This is the core primitive for Gaussian elimination, forward/back
/// substitution, and Cholesky inner sums. It avoids the n ULP accumulation
/// error that would result from element-wise storage-tier operations.
///
/// Panics if `a.len() != b.len()`.
pub(crate) fn compute_tier_sub_dot_raw(
    init: BinaryStorage,
    a: &[BinaryStorage],
    b: &[BinaryStorage],
) -> BinaryStorage {
    assert_eq!(a.len(), b.len(), "compute_tier_sub_dot_raw: length mismatch");

    let acc = compute_tier_sub_dot_compute(init, a, b);
    round_to_storage(acc)
}

/// Same as `compute_tier_sub_dot_raw` but returns the result at compute tier
/// (ComputeStorage) WITHOUT downscaling. Used for fused operations where the
/// compute-tier intermediate feeds directly into sqrt or divide at compute tier.
pub(crate) fn compute_tier_sub_dot_compute(
    init: BinaryStorage,
    a: &[BinaryStorage],
    b: &[BinaryStorage],
) -> ComputeStorage {
    assert_eq!(a.len(), b.len(), "compute_tier_sub_dot_compute: length mismatch");

    #[cfg(table_format = "q64_64")]
    {
        let mut acc = I256::from_i128(init) << 64usize;
        for i in 0..a.len() {
            acc = acc - (I256::from_i128(a[i]) * I256::from_i128(b[i]));
        }
        acc
    }

    #[cfg(table_format = "q32_32")]
    {
        // i64 upscaled to i128, then subtract i64×i64→i128 products
        let mut acc: i128 = (init as i128) << 32;
        for i in 0..a.len() {
            acc -= (a[i] as i128) * (b[i] as i128);
        }
        acc
    }

    #[cfg(table_format = "q16_16")]
    {
        // i32 upscaled to i64, then subtract i32×i32→i64 products
        let mut acc: i64 = (init as i64) << 16;
        for i in 0..a.len() {
            acc -= (a[i] as i64) * (b[i] as i64);
        }
        acc
    }

    #[cfg(table_format = "q128_128")]
    {
        let mut acc = I512::from_i256(init) << 128usize;
        for i in 0..a.len() {
            let a_neg = a[i].is_negative();
            let b_neg = b[i].is_negative();
            let result_neg = a_neg != b_neg;
            let abs_a = if a_neg { -a[i] } else { a[i] };
            let abs_b = if b_neg { -b[i] } else { b[i] };
            let product = abs_a.mul_to_i512(abs_b);
            acc = acc - if result_neg { -product } else { product };
        }
        acc
    }

    #[cfg(table_format = "q256_256")]
    {
        let mut acc = I1024::from_i512(init) << 256usize;
        for i in 0..a.len() {
            let a_neg = a[i].is_negative();
            let b_neg = b[i].is_negative();
            let result_neg = a_neg != b_neg;
            let abs_a = if a_neg { -a[i] } else { a[i] };
            let abs_b = if b_neg { -b[i] } else { b[i] };
            let product = abs_a.mul_to_i1024(abs_b);
            acc = acc - if result_neg { -product } else { product };
        }
        acc
    }
}

// ============================================================================
// Givens rotation (no trig — ratio + sqrt only)
// ============================================================================

/// Compute Givens rotation coefficients (cs, sn) such that:
///   [[cs, sn], [-sn, cs]]^T * [a, b]^T = [r, 0]
///
/// Uses ratio-based computation with a single `sqrt` call.
/// No `atan`, `sin`, or `cos` — avoids layering transcendental approximation
/// error on top of fixed-point quantization.
pub(crate) fn givens(a: FixedPoint, b: FixedPoint) -> (FixedPoint, FixedPoint) {
    let one = FixedPoint::one();
    let zero = FixedPoint::ZERO;

    if b.is_zero() {
        return (one, zero);
    }
    if a.is_zero() {
        let sn = if b.is_negative() { -one } else { one };
        return (zero, sn);
    }

    // Ratio-based approach avoids overflow from a^2 + b^2
    if b.abs() > a.abs() {
        let tau = a / b;
        let sn = one / (one + tau * tau).sqrt();
        let cs = sn * tau;
        (cs, sn)
    } else {
        let tau = b / a;
        let cs = one / (one + tau * tau).sqrt();
        let sn = cs * tau;
        (cs, sn)
    }
}

// ============================================================================
// Compute-tier Givens rotation application
// ============================================================================

/// Apply a 2×2 Givens rotation at compute tier (tier N+1).
///
/// Computes:
///   new_x = cs*x + sn*y
///   new_y = -sn*x + cs*y
///
/// Each result is a 2-element dot product accumulated at tier N+1 with a
/// single downscale. This gives 1 ULP per output instead of the 3 ULP
/// from storage-tier `cs * x + sn * y` (2 multiplies + 1 add, each rounding).
///
/// This is the rotation analog of `compute_tier_dot_raw`.
#[inline]
pub(crate) fn apply_givens_compute(
    cs: FixedPoint, sn: FixedPoint, x: FixedPoint, y: FixedPoint,
) -> (FixedPoint, FixedPoint) {
    let cs_raw = cs.raw();
    let sn_raw = sn.raw();
    let neg_sn_raw = (-sn).raw();
    let x_raw = x.raw();
    let y_raw = y.raw();
    let new_x = FixedPoint::from_raw(compute_tier_dot_raw(
        &[cs_raw, sn_raw], &[x_raw, y_raw]
    ));
    let new_y = FixedPoint::from_raw(compute_tier_dot_raw(
        &[neg_sn_raw, cs_raw], &[x_raw, y_raw]
    ));
    (new_x, new_y)
}

// ============================================================================
// Convergence threshold for fixed-point iterative algorithms
// ============================================================================

/// Compute the convergence threshold for iterative algorithms.
///
/// In floating-point, convergence is tested against machine epsilon.
/// In fixed-point, we use `magnitude >> (FRAC_BITS / 2)`, which gives
/// sqrt(quantum) relative precision — the tightest achievable by iterative
/// multiply-based algorithms at storage tier.
///
/// Floored at 1 quantum (the smallest nonzero representable value).
///
/// Profile-dependent precision:
/// - Q64.64:  ~2^-32 relative (~9.5 decimal digits)
/// - Q128.128: ~2^-64 relative (~19 decimal digits)
/// - Q256.256: ~2^-128 relative (~38 decimal digits)
pub(crate) fn convergence_threshold(magnitude: FixedPoint) -> FixedPoint {
    let quantum = FixedPoint::from_raw(quantum_raw());
    let shifted = magnitude.abs().raw() >> half_frac_bits();
    let result = FixedPoint::from_raw(shifted);
    if result.is_zero() { quantum } else { result }
}

/// Tighter convergence threshold for compute-tier iterative algorithms.
///
/// When all rotation/accumulation steps happen at tier N+1 (via
/// `apply_givens_compute`, `compute_tier_dot_raw`), each step introduces
/// only 1 ULP of error — NOT √quantum. So we can converge to
/// `magnitude >> (2 * FRAC_BITS / 3)` instead of `>> (FRAC_BITS / 2)`.
///
/// Profile-dependent precision:
/// - Q64.64:  ~2^-42 relative (~12.6 decimal digits)
/// - Q128.128: ~2^-85 relative (~25.6 decimal digits)
/// - Q256.256: ~2^-170 relative (~51.2 decimal digits)
pub(crate) fn convergence_threshold_tight(magnitude: FixedPoint) -> FixedPoint {
    let quantum = FixedPoint::from_raw(quantum_raw());
    let shifted = magnitude.abs().raw() >> two_thirds_frac_bits();
    let result = FixedPoint::from_raw(shifted);
    if result.is_zero() { quantum } else { result }
}

#[cfg(table_format = "q64_64")]
fn two_thirds_frac_bits() -> u32 { 42 }
#[cfg(table_format = "q32_32")]
fn two_thirds_frac_bits() -> u32 { 21 }
#[cfg(table_format = "q16_16")]
fn two_thirds_frac_bits() -> u32 { 10 }
#[cfg(table_format = "q128_128")]
fn two_thirds_frac_bits() -> u32 { 85 }
#[cfg(table_format = "q256_256")]
fn two_thirds_frac_bits() -> usize { 170 }

#[cfg(table_format = "q64_64")]
fn half_frac_bits() -> u32 { 32 }
#[cfg(table_format = "q32_32")]
fn half_frac_bits() -> u32 { 16 }
#[cfg(table_format = "q16_16")]
fn half_frac_bits() -> u32 { 8 }
#[cfg(table_format = "q128_128")]
fn half_frac_bits() -> u32 { 64 }
#[cfg(table_format = "q256_256")]
fn half_frac_bits() -> usize { 128 }

#[cfg(table_format = "q64_64")]
fn quantum_raw() -> BinaryStorage { 1i128 }
#[cfg(table_format = "q32_32")]
fn quantum_raw() -> BinaryStorage { 1i64 }
#[cfg(table_format = "q16_16")]
fn quantum_raw() -> BinaryStorage { 1i32 }
#[cfg(table_format = "q128_128")]
fn quantum_raw() -> BinaryStorage { I256::from_i128(1) }
#[cfg(table_format = "q256_256")]
fn quantum_raw() -> BinaryStorage { I512::from_i128(1) }

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
/// inner loop — only add/sub/skip.
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
