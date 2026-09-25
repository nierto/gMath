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

use super::FixedPoint;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
#[cfg(table_format = "q16_16")]
use crate::fixed_point::frac_config;
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
pub(crate) use crate::fixed_point::universal::fasc::stack_evaluator::compute::downscale_to_storage;

// Re-export fused sincos for imperative-module consumers (lie_group, etc.)
pub(crate) use crate::fixed_point::universal::fasc::stack_evaluator::compute::sincos_at_compute_tier;

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

/// Dot product accumulated at tier N+1 (compute tier).
///
/// Each product a_i * b_i is computed at double width without truncation.
/// The entire sum is accumulated at double width (checked: a sum beyond the
/// compute tier panics, never wraps) and rounded to storage once, to nearest
/// with ties toward +infinity like every binary result. Before 0.6.4 this
/// floored (`acc >> F`), up to one unit below the matrix path's nearest.
///
/// Panics if the slices have different lengths.
pub fn compute_tier_dot(a: &[FixedPoint], b: &[FixedPoint]) -> FixedPoint {
    assert_eq!(a.len(), b.len(), "compute_tier_dot: length mismatch");
    FixedPoint::from_raw(round_to_storage(compute_tier_dot_acc_pairs(
        a.iter().zip(b).map(|(x, y)| (x.raw(), y.raw())),
    )))
}

/// Compute-tier multiply-accumulate: acc += a_i * b_i for matrix operations.
///
/// Same as `compute_tier_dot` but takes raw BinaryStorage slices for
/// internal use where the FixedPoint wrapper would add unnecessary overhead.
#[inline]
pub(crate) fn compute_tier_dot_raw(a: &[BinaryStorage], b: &[BinaryStorage]) -> BinaryStorage {
    round_to_storage(compute_tier_dot_acc(a, b))
}

/// sqrt(sum a_i b_i) with the sum AND the root at the compute tier and one
/// rounding to storage. Rounding the sum to storage first (then `.sqrt()`)
/// rounds twice, amplifies the first rounding by 1 / (2 |x|) for small norms,
/// and overflows storage once the SQUARED norm leaves the range even though
/// the norm fits. `Err(DomainError)` for a negative sum.
pub(crate) fn compute_tier_sqrt_dot(a: &[BinaryStorage], b: &[BinaryStorage]) -> Result<BinaryStorage, OverflowDetected> {
    use crate::fixed_point::universal::fasc::stack_evaluator::compute::{compute_is_negative, sqrt_at_compute_tier};
    let acc = compute_tier_dot_acc(a, b);
    if compute_is_negative(&acc) { return Err(OverflowDetected::DomainError); }
    downscale_to_storage(sqrt_at_compute_tier(acc))
}

/// sum a_i b_i at the compute tier, unrounded. The accumulation is checked:
/// a sum beyond the compute tier panics instead of wrapping (the plain
/// additions wrapped silently in release builds before 0.6.4).
#[inline]
pub(crate) fn compute_tier_dot_acc(a: &[BinaryStorage], b: &[BinaryStorage]) -> ComputeStorage {
    assert_eq!(a.len(), b.len(), "compute_tier_dot_raw: length mismatch");
    compute_tier_dot_acc_pairs(a.iter().copied().zip(b.iter().copied()))
}

/// The checked compute-tier accumulator over (a_i, b_i) pairs.
#[inline]
fn compute_tier_dot_acc_pairs(pairs: impl Iterator<Item = (BinaryStorage, BinaryStorage)>) -> ComputeStorage {
    const OVERFLOW: &str = "compute_tier_dot: sum exceeds the compute tier";

    #[cfg(table_format = "q64_64")]
    {
        let mut acc = I256::zero();
        for (x, y) in pairs {
            acc = acc.checked_add(I256::from_i128(x) * I256::from_i128(y)).expect(OVERFLOW);
        }
        acc
    }

    #[cfg(table_format = "q32_32")]
    {
        let mut acc: i128 = 0;
        for (x, y) in pairs {
            acc = acc.checked_add((x as i128) * (y as i128)).expect(OVERFLOW);
        }
        acc
    }

    #[cfg(table_format = "q16_16")]
    {
        let mut acc: i64 = 0;
        for (x, y) in pairs {
            acc = acc.checked_add((x as i64) * (y as i64)).expect(OVERFLOW);
        }
        acc
    }

    #[cfg(table_format = "q128_128")]
    {
        let mut acc = I512::zero();
        for (x, y) in pairs {
            let a_neg = x.is_negative();
            let b_neg = y.is_negative();
            let result_neg = a_neg != b_neg;
            // `-MIN` is MIN again, whose bit pattern read unsigned by the
            // word-wise `mul_to_i512` is 2^(W-1), the true magnitude: the
            // product (at most 2^(2W-2)) stays exact and non-negative
            let abs_a = if a_neg { -x } else { x };
            let abs_b = if b_neg { -y } else { y };
            let product = abs_a.mul_to_i512(abs_b);
            acc = acc.checked_add(if result_neg { -product } else { product }).expect(OVERFLOW);
        }
        acc
    }

    #[cfg(table_format = "q256_256")]
    {
        let mut acc = I1024::zero();
        for (x, y) in pairs {
            let a_neg = x.is_negative();
            let b_neg = y.is_negative();
            let result_neg = a_neg != b_neg;
            // `-MIN` is MIN again, whose bit pattern read unsigned by the
            // word-wise `mul_to_i1024` is 2^(W-1), the true magnitude: the
            // product (at most 2^(2W-2)) stays exact and non-negative
            let abs_a = if a_neg { -x } else { x };
            let abs_b = if b_neg { -y } else { y };
            let product = abs_a.mul_to_i1024(abs_b);
            acc = acc.checked_add(if result_neg { -product } else { product }).expect(OVERFLOW);
        }
        acc
    }
}


/// `init - sum a_i b_i` of storage raws at the compute tier, WITHOUT
/// downscaling (the products are exact at the compute tier). Used where the
/// compute-tier value feeds a further compute-tier step.
pub(crate) fn compute_tier_sub_dot_compute(
    init: BinaryStorage,
    a: &[BinaryStorage],
    b: &[BinaryStorage],
) -> ComputeStorage {
    const OVERFLOW: &str = "compute_tier_sub_dot: sum exceeds the compute tier";
    assert_eq!(a.len(), b.len(), "compute_tier_sub_dot_compute: length mismatch");

    #[cfg(table_format = "q64_64")]
    {
        let mut acc = I256::from_i128(init) << 64usize;
        for i in 0..a.len() {
            acc = acc.checked_sub(I256::from_i128(a[i]) * I256::from_i128(b[i])).expect(OVERFLOW);
        }
        acc
    }

    #[cfg(table_format = "q32_32")]
    {
        // i64 upscaled to i128, then subtract i64×i64→i128 products
        let mut acc: i128 = (init as i128) << 32;
        for i in 0..a.len() {
            acc = acc.checked_sub((a[i] as i128) * (b[i] as i128)).expect(OVERFLOW);
        }
        acc
    }

    #[cfg(table_format = "q16_16")]
    {
        // i32 upscaled to i64, then subtract i32×i32→i64 products
        let mut acc: i64 = (init as i64) << frac_config::FRAC_BITS;
        for i in 0..a.len() {
            acc = acc.checked_sub((a[i] as i64) * (b[i] as i64)).expect(OVERFLOW);
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
            acc = acc.checked_add(if result_neg { product } else { -product }).expect(OVERFLOW);
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
            acc = acc.checked_add(if result_neg { product } else { -product }).expect(OVERFLOW);
        }
        acc
    }
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

use super::interval::exact_product;
use super::wide_acc::{divide_to_compute_nearest, exact_dot_compute, narrow_product_to_compute, widen_product, Wide};
use crate::fixed_point::core_types::errors::OverflowDetected;
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_checked_add, compute_divide, compute_is_negative,
    compute_is_zero, compute_multiply, compute_negate, make_compute_int, sqrt_at_compute_tier,
};

/// Consecutive non-improving iterations after which an iterative decomposition
/// is taken to sit at its precision floor.
pub(crate) const STAGNATION_SWEEPS: usize = 5;

/// Storage fraction bits of the build.
#[inline]
fn storage_frac_bits() -> u32 {
    #[cfg(table_format = "q16_16")]
    { frac_config::FRAC_BITS }
    #[cfg(table_format = "q32_32")]
    { 32 }
    #[cfg(table_format = "q64_64")]
    { 64 }
    #[cfg(table_format = "q128_128")]
    { 128 }
    #[cfg(table_format = "q256_256")]
    { 256 }
}

/// `|v| >> shift` for a compute raw.
#[inline]
pub(crate) fn compute_abs_shr(v: ComputeStorage, shift: u32) -> ComputeStorage {
    let v = compute_abs(v);
    #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
    { v >> shift }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    { v >> shift as usize }
}

/// `2^shift` compute quanta.
#[inline]
fn compute_quanta(shift: u32) -> ComputeStorage {
    #[cfg(table_format = "q16_16")]
    { 1i64 << shift }
    #[cfg(table_format = "q32_32")]
    { 1i128 << shift }
    #[cfg(table_format = "q64_64")]
    { I256::from_i128(1) << shift as usize }
    #[cfg(table_format = "q128_128")]
    { I512::from_i128(1) << shift as usize }
    #[cfg(table_format = "q256_256")]
    { I1024::from_i128(1) << shift as usize }
}

/// Deflation bound of the iterations that keep their state at the compute
/// tier: `magnitude * 2^-(3F/2)`, floored at `2^-(3F/2)` absolute.
///
/// The compute tier resolves `2^-2F`, so the iteration reaches this bound
/// with `F/2` bits to spare; dropping an entry this small moves an eigen- or
/// singular value of the magnitude's size by at most `2^-(F/2)` units (Weyl;
/// quadratically less for a symmetric matrix), below the final rounding.
#[inline]
pub(crate) fn compute_deflation_threshold(magnitude: ComputeStorage) -> ComputeStorage {
    let shift = storage_frac_bits() * 3 / 2;
    compute_abs_shr(magnitude, shift).max(compute_quanta(storage_frac_bits() / 2))
}

/// Absolute noise floor of the compute-tier iterations, `2^-(3F/2)`: a
/// diagonal entry at or below it is an exact zero computed as rounding noise.
#[inline]
pub(crate) fn compute_noise_floor() -> ComputeStorage {
    compute_quanta(storage_frac_bits() / 2)
}

/// The looser bound accepted once a compute-tier iteration has stopped
/// improving: one storage unit relative to the magnitude, with the same
/// absolute floor as [`compute_deflation_threshold`].
#[inline]
pub(crate) fn compute_stagnation_threshold(magnitude: ComputeStorage) -> ComputeStorage {
    compute_abs_shr(magnitude, storage_frac_bits()).max(compute_quanta(storage_frac_bits() / 2))
}

/// Magnitude of a compute-tier value.
#[inline]
pub(crate) fn compute_abs(v: ComputeStorage) -> ComputeStorage {
    if compute_is_negative(&v) { compute_negate(v) } else { v }
}

/// A plane rotation `[cs sn; -sn cs]` whose coefficients stay at the compute
/// tier (`2 * FRAC_BITS` fractional bits).
#[derive(Clone, Copy)]
pub(crate) struct Rotation {
    cs: ComputeStorage,
    sn: ComputeStorage,
}

impl Rotation {
    /// A rotation from coefficients already at the compute tier.
    #[inline]
    pub(crate) fn from_parts(cs: ComputeStorage, sn: ComputeStorage) -> Self {
        Rotation { cs, sn }
    }

    /// The rotation taking `(a, b)` to `(r, 0)`: `cs = a / |r|`, `sn = b / |r|`
    /// (`cs = 1`, `sn = 0` when `b` is zero). `a` and `b` are compute raws at a
    /// common scale. Ratio form with one square root; neither input is squared,
    /// and no trig function is involved.
    pub(crate) fn zeroing_compute(a: ComputeStorage, b: ComputeStorage) -> Result<Self, OverflowDetected> {
        let one = make_compute_int(1);
        let zero = make_compute_int(0);
        if compute_is_zero(&b) {
            return Ok(Rotation { cs: one, sn: zero });
        }
        if compute_is_zero(&a) {
            let sn = if compute_is_negative(&b) { compute_negate(one) } else { one };
            return Ok(Rotation { cs: zero, sn });
        }
        if compute_abs(b) > compute_abs(a) {
            let tau = compute_divide(a, b)?;
            let inv = compute_divide(one, sqrt_at_compute_tier(compute_add(one, compute_multiply(tau, tau))))?;
            let sn = if compute_is_negative(&b) { compute_negate(inv) } else { inv };
            Ok(Rotation { cs: compute_multiply(sn, tau), sn })
        } else {
            let tau = compute_divide(b, a)?;
            let inv = compute_divide(one, sqrt_at_compute_tier(compute_add(one, compute_multiply(tau, tau))))?;
            let cs = if compute_is_negative(&a) { compute_negate(inv) } else { inv };
            Ok(Rotation { cs, sn: compute_multiply(cs, tau) })
        }
    }

    /// `(cs x + sn y, -sn x + cs y)` on compute raws, each rounded once at the
    /// compute tier from its exact value.
    pub(crate) fn apply_compute(
        &self, x: ComputeStorage, y: ComputeStorage,
    ) -> Result<(ComputeStorage, ComputeStorage), OverflowDetected> {
        let first = widen_product(self.cs, x).add_exact(widen_product(self.sn, y))?;
        let second = widen_product(compute_negate(self.sn), x).add_exact(widen_product(self.cs, y))?;
        Ok((narrow_product_to_compute(first)?, narrow_product_to_compute(second)?))
    }

    /// `cs x + sn y` alone, on compute raws.
    pub(crate) fn combine_compute(&self, x: ComputeStorage, y: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
        narrow_product_to_compute(widen_product(self.cs, x).add_exact(widen_product(self.sn, y))?)
    }

    /// `cs x` on a compute raw.
    #[inline]
    pub(crate) fn cos_times_compute(&self, x: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
        compute_product(self.cs, x)
    }

    /// `sn x` on a compute raw.
    #[inline]
    pub(crate) fn sin_times_compute(&self, x: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
        compute_product(self.sn, x)
    }

    /// `-sn x` on a compute raw.
    #[inline]
    pub(crate) fn neg_sin_times_compute(&self, x: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
        compute_product(compute_negate(self.sn), x)
    }
}

/// `a b` of two compute raws, rounded once at the compute tier from the exact
/// product. A product beyond the compute tier is a `TierOverflow`.
#[inline]
pub(crate) fn compute_product(a: ComputeStorage, b: ComputeStorage) -> Result<ComputeStorage, OverflowDetected> {
    narrow_product_to_compute(widen_product(a, b))
}

/// Exact `sum a_i b_i` of storage values as a compute raw (`2 * FRAC_BITS`
/// fractional bits). A sum beyond the compute tier is a `TierOverflow`.
pub(crate) fn exact_dot(a: &[BinaryStorage], b: &[BinaryStorage]) -> Result<ComputeStorage, OverflowDetected> {
    assert_eq!(a.len(), b.len(), "exact_dot: length mismatch");
    let mut acc = make_compute_int(0);
    for i in 0..a.len() {
        acc = compute_checked_add(acc, exact_product(a[i], b[i]))?;
    }
    Ok(acc)
}

/// A compute raw scaled by `2^-shift`, rounded once to storage (nearest,
/// ties toward +infinity, checked): for iterations that scaled a block up by
/// a power of two to keep relative precision.
pub(crate) fn downscale_shifted_to_storage(v: ComputeStorage, shift: u32) -> Result<BinaryStorage, OverflowDetected> {
    if shift == 0 {
        return downscale_to_storage(v);
    }
    let f = storage_frac_bits();
    let half = compute_quanta(f + shift - 1);
    let q = compute_shr(compute_checked_add(v, half)?, f + shift);
    // a multiple of 2^F at the compute scale: the downscale is exact
    downscale_to_storage(compute_shl(q, f))
}

/// `v * 2^shift` for a compute raw (exact; the caller keeps it in range).
#[inline]
pub(crate) fn compute_scale_up(v: ComputeStorage, shift: u32) -> ComputeStorage {
    compute_shl(v, shift)
}

/// The power-of-two exponent that brings the largest of `values` into
/// `[1/2, 1)` at the compute scale when it is below `1/2`, else 0.
pub(crate) fn scale_up_exponent(values: &[ComputeStorage]) -> u32 {
    let bits = values.iter().map(|&v| compute_bit_length(v)).max().unwrap_or(0);
    let unit_bits = 2 * storage_frac_bits();
    if bits == 0 || bits >= unit_bits { 0 } else { unit_bits - bits }
}

/// Significant bits of `|v|` for a compute raw.
#[inline]
fn compute_bit_length(v: ComputeStorage) -> u32 {
    #[cfg(table_format = "q16_16")]
    { 64 - v.unsigned_abs().leading_zeros() }
    #[cfg(table_format = "q32_32")]
    { 128 - v.unsigned_abs().leading_zeros() }
    #[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256"))]
    {
        let words = compute_abs(v).words;
        (0..words.len()).rev().find(|&i| words[i] != 0).map_or(0, |i| i as u32 * 64 + (64 - words[i].leading_zeros()))
    }
}

/// `v << shift` for a compute raw (exact; the caller keeps it in range).
#[inline]
fn compute_shl(v: ComputeStorage, shift: u32) -> ComputeStorage {
    #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
    { v << shift }
    #[cfg(any(table_format = "q64_64", table_format = "q128_128", table_format = "q256_256"))]
    { v << shift as usize }
}

/// `v >> shift` (arithmetic) for a compute raw.
#[inline]
fn compute_shr(v: ComputeStorage, shift: u32) -> ComputeStorage {
    #[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
    { v >> shift }
    #[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
    { v >> shift as usize }
}

/// Householder direction for a column held at the compute tier: `v = x -
/// alpha e_1`, `alpha = -sign(x_0) ||x||`, and `v.v` (both compute raws).
/// The sums of squares are exact (4F) and narrowed once, the root taken at
/// the compute tier. `None` when `x` (or `v`) is zero.
///
/// The column is first scaled by a power of two so its largest entry lies in
/// `[1/2, 1)`: the reflection depends only on the direction of `v`, while
/// `x.x` and `v.v` narrowed to the compute tier keep their relative precision
/// only at that size (at `F = 10` an entry near `2^-8` left `x.x` a few
/// significant bits, and the reflection lost orthogonality), and fit the
/// compute tier only below it (on realtime the compute tier has the storage
/// range, so `x.x` of a large column overflowed). Scaling up is exact;
/// scaling down keeps `2F` significant bits of the largest entry. The
/// returned `v` is the scaled one.
pub(crate) fn householder_vector_compute(
    x: &[ComputeStorage],
) -> Result<Option<(Vec<ComputeStorage>, ComputeStorage)>, OverflowDetected> {
    let bits = x.iter().map(|&xi| compute_bit_length(xi)).max().unwrap_or(0);
    if bits == 0 {
        return Ok(None);
    }
    let unit_bits = 2 * storage_frac_bits();
    let scaled: Vec<ComputeStorage>;
    let x = if bits < unit_bits {
        scaled = x.iter().map(|&xi| compute_shl(xi, unit_bits - bits)).collect();
        &scaled[..]
    } else if bits > unit_bits {
        scaled = x.iter().map(|&xi| compute_shr(xi, bits - unit_bits)).collect();
        &scaled[..]
    } else {
        x
    };
    let xx = narrow_product_to_compute(exact_dot_compute(x, x)?)?;
    if compute_is_zero(&xx) {
        return Ok(None);
    }
    let norm = sqrt_at_compute_tier(xx);
    let alpha = if compute_is_negative(&x[0]) { norm } else { compute_negate(norm) };
    let mut v = x.to_vec();
    v[0] = compute_checked_add(x[0], compute_negate(alpha))?;
    let vv = narrow_product_to_compute(exact_dot_compute(&v, &v)?)?;
    if compute_is_zero(&vv) {
        return Ok(None);
    }
    Ok(Some((v, vv)))
}

/// Reflect a compute-tier vector `w` in the hyperplane orthogonal to `v`:
/// each `w_k -= 2 (v.w) v_k / (v.v)` as one exact quotient at the compute
/// tier. No rounding to storage: callers carry the state across reflections
/// and round once at the end.
pub(crate) fn reflect_compute(
    w: &mut [ComputeStorage], v: &[ComputeStorage], v_dot_v: ComputeStorage,
) -> Result<(), OverflowDetected> {
    let vw = narrow_product_to_compute(exact_dot_compute(v, w)?)?;
    if compute_is_zero(&vw) {
        return Ok(());
    }
    let two_vw = compute_checked_add(vw, vw)?;
    for (wk, vk) in w.iter_mut().zip(v) {
        let update = divide_to_compute_nearest(widen_product(two_vw, *vk), v_dot_v)?;
        *wk = compute_checked_add(*wk, compute_negate(update))?;
    }
    Ok(())
}

// ============================================================================
// Convergence threshold for fixed-point iterative algorithms
// ============================================================================

/// Compute the convergence threshold for iterative algorithms.
///
/// In floating-point, convergence is tested against machine epsilon.
/// In fixed-point, we use `magnitude >> (FRAC_BITS / 2)`, which gives
/// sqrt(quantum) relative precision: the tightest achievable by iterative
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
