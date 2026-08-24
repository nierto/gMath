//! # Compute-Tier Transcendentals: Public Wide-Precision Surface
//!
//! Direct access to the tier-N+1 transcendental engines that back every
//! other API in this crate, operating on raw [`ComputeStorage`] values at
//! `2 × FRAC_BITS` fractional precision. Requires the `inference` feature.
//!
//! ## Format
//!
//! A [`ComputeStorage`] value represents `raw / 2^COMPUTE_FRAC_BITS`, where
//! `COMPUTE_FRAC_BITS = 2 × FRAC_BITS` (see [`FRAC_BITS`] and
//! [`COMPUTE_FRAC_BITS`]). This is exactly the format produced by the
//! `tq19` wide-output `matvec_q2f` family, so those accumulators feed these
//! functions with no conversion. Use [`from_fixed`]/[`to_fixed`] to cross
//! between `FixedPoint` storage and the compute tier; a chain of calls at
//! the compute tier rounds once at the final [`to_fixed`], not once per step.
//!
//! ## Contract
//!
//! - Same engines, same rounding as the canonical and imperative paths;
//!   results are path-independent with the rest of the crate.
//! - `exp` **saturates** at [`ceiling`] when the true result exceeds the
//!   compute tier's range; it never wraps. Detect saturation by comparing
//!   against [`ceiling`].
//! - `ln`, `sqrt`, and `ln1p` **panic** on domain violations (`x <= 0`,
//!   `x < 0`, `x <= -1` respectively) rather than return sentinel values.
//! - [`to_fixed`] **panics** if the value does not fit storage;
//!   [`try_to_fixed`] returns `None` instead. Nothing here wraps silently.
//!
//! ## Example (any profile)
//!
//! ```
//! # #[cfg(feature = "inference")] {
//! use g_math::fixed_point::compute_tier as ct;
//! use g_math::fixed_point::FixedPoint;
//!
//! let x = ct::from_fixed(FixedPoint::from_str("0.5"));
//! let y = ct::sigmoid(x);                 // stays at 2×FRAC_BITS precision
//! let s = ct::to_fixed(y);                // single rounding, back to storage
//! # let _ = s;
//! # }
//! ```

use crate::fixed_point::imperative::FixedPoint;
use crate::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
use crate::fixed_point::universal::fasc::stack_evaluator::compute::{
    compute_add, compute_checked_add, compute_ceiling, compute_divide,
    compute_is_negative, compute_is_zero, compute_negate, downscale_to_storage,
    exp_at_compute_tier, ln_at_compute_tier, make_compute_int,
    sinhcosh_at_compute_tier, sqrt_at_compute_tier, upscale_to_compute,
};

/// The wide integer type of the compute tier for the active profile
/// (`i64`/`i128`/`I256`/`I512`/`I1024` for
/// realtime/compact/embedded/balanced/scientific respectively).
pub use crate::fixed_point::universal::fasc::stack_evaluator::ComputeStorage;

/// Fractional bits of the storage tier (`FixedPoint`).
pub use crate::fixed_point::frac_config::FRAC_BITS;

/// Fractional bits of the compute tier (`= 2 × FRAC_BITS`).
pub use crate::fixed_point::frac_config::COMPUTE_FRAC_BITS;

/// The value `1.0` at compute-tier scale (`1 << COMPUTE_FRAC_BITS`).
#[inline]
pub fn one() -> ComputeStorage {
    make_compute_int(1)
}

/// The compute tier's maximum value: the saturation ceiling for [`exp`].
#[inline]
pub fn ceiling() -> ComputeStorage {
    compute_ceiling()
}

/// Promote a `FixedPoint` (storage tier) to the compute tier. Exact.
#[inline]
pub fn from_fixed(x: FixedPoint) -> ComputeStorage {
    upscale_to_compute(x.raw())
}

/// Round a compute-tier value to the nearest `FixedPoint` (single rounding).
///
/// # Panics
/// Panics if the value does not fit the storage tier. Use [`try_to_fixed`]
/// for a non-panicking variant.
#[inline]
pub fn to_fixed(x: ComputeStorage) -> FixedPoint {
    try_to_fixed(x).expect("compute_tier::to_fixed: value does not fit the storage tier")
}

/// Round a compute-tier value to the nearest `FixedPoint`, or `None` on storage overflow.
#[inline]
pub fn try_to_fixed(x: ComputeStorage) -> Option<FixedPoint> {
    let raw: BinaryStorage = downscale_to_storage(x).ok()?;
    Some(FixedPoint::from_raw(raw))
}

/// `e^x` at the compute tier.
///
/// Saturates at [`ceiling`] when the true result exceeds the compute tier's
/// range; never wraps. A subsequent [`to_fixed`] on a saturated value panics
/// (it does not fit storage), which is the intended loud failure.
#[inline]
pub fn exp(x: ComputeStorage) -> ComputeStorage {
    exp_at_compute_tier(x)
}

/// `ln(x)` at the compute tier.
///
/// # Panics
/// Panics if `x <= 0`.
#[inline]
pub fn ln(x: ComputeStorage) -> ComputeStorage {
    assert!(
        !(compute_is_negative(&x) || compute_is_zero(&x)),
        "compute_tier::ln: x <= 0 is outside the domain"
    );
    ln_at_compute_tier(x)
}

/// `sqrt(x)` at the compute tier.
///
/// # Panics
/// Panics if `x < 0`.
#[inline]
pub fn sqrt(x: ComputeStorage) -> ComputeStorage {
    assert!(
        !compute_is_negative(&x),
        "compute_tier::sqrt: x < 0 is outside the domain"
    );
    sqrt_at_compute_tier(x)
}

/// `(sinh(x), cosh(x))` at the compute tier from one shared exponential pair.
///
/// Both values derive from the same `(e^x, e^-x)` computation, so systematic
/// rounding bias cancels in expressions that use both.
#[inline]
pub fn sinhcosh(x: ComputeStorage) -> (ComputeStorage, ComputeStorage) {
    sinhcosh_at_compute_tier(x)
}

/// `1 / (1 + e^-x)` at the compute tier.
///
/// Uses the sign-split form so the denominator stays in `[1, 2]`: no
/// intermediate can overflow, and large `|x|` saturates smoothly to 0 or 1.
#[inline]
pub fn sigmoid(x: ComputeStorage) -> ComputeStorage {
    let one = make_compute_int(1);
    if compute_is_negative(&x) {
        // e^x <= 1, denominator in [1, 2]
        let e = exp_at_compute_tier(x);
        compute_divide(e, compute_add(one, e))
            .expect("sigmoid: denominator in [1,2] cannot overflow")
    } else {
        // e^-x <= 1, denominator in [1, 2]
        let e = exp_at_compute_tier(compute_negate(x));
        compute_divide(one, compute_add(one, e))
            .expect("sigmoid: denominator in [1,2] cannot overflow")
    }
}

/// `ln(1 + e^x)` (softplus) at the compute tier.
///
/// Uses the stable form `max(x, 0) + ln(1 + e^-|x|)`: the exponential
/// argument is never positive, so nothing saturates for any input.
///
/// # Panics
/// Panics only if `x` is within `ln 2` of [`ceiling`] (the exact result
/// itself would not fit the compute tier).
#[inline]
pub fn softplus(x: ComputeStorage) -> ComputeStorage {
    let one = make_compute_int(1);
    let neg_abs = if compute_is_negative(&x) { x } else { compute_negate(x) };
    let e = exp_at_compute_tier(neg_abs); // in (0, 1]
    let corr = ln_at_compute_tier(compute_add(one, e)); // arg in (1, 2], always in domain
    if compute_is_negative(&x) {
        corr
    } else {
        compute_checked_add(x, corr)
            .expect("softplus: result does not fit the compute tier")
    }
}

/// `ln(1 + x)` at the compute tier.
///
/// The `1 + x` addition is exact in fixed point (integer add), so unlike
/// floating point there is no cancellation regime; this is a convenience
/// wrapper with a domain check, not a different algorithm.
///
/// # Panics
/// Panics if `x <= -1` (outside the domain) or if `1 + x` does not fit the
/// compute tier.
#[inline]
pub fn ln1p(x: ComputeStorage) -> ComputeStorage {
    let arg = compute_checked_add(make_compute_int(1), x)
        .expect("ln1p: 1 + x does not fit the compute tier");
    assert!(
        !(compute_is_negative(&arg) || compute_is_zero(&arg)),
        "compute_tier::ln1p: x <= -1 is outside the domain"
    );
    ln_at_compute_tier(arg)
}
