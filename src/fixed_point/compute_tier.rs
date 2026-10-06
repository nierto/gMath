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
//! let y = ct::sigmoid::<FixedPoint>(x);                 // stays at 2×FRAC_BITS precision
//! let s = ct::to_fixed::<FixedPoint>(y);                // single rounding, back to storage
//! # let _ = s;
//! # }
//! ```

use crate::fixed_point::fixed_num::{Compute, FixedNum};

pub use crate::fixed_point::universal::fasc::stack_evaluator::ComputeStorage;

pub use crate::fixed_point::frac_config::FRAC_BITS;

pub use crate::fixed_point::frac_config::COMPUTE_FRAC_BITS;

/// 1.0 at the compute tier of the format `T`.
#[inline]
pub fn one<T: FixedNum>() -> Compute<T> {
    T::make_compute_int(1)
}

/// The exponential's saturation value at the compute tier of `T`.
#[inline]
pub fn ceiling<T: FixedNum>() -> Compute<T> {
    T::compute_ceiling()
}

/// A storage value lifted to its compute tier (exact).
#[inline]
pub fn from_fixed<T: FixedNum>(x: T) -> Compute<T> {
    T::upscale_to_compute(x.raw_storage())
}

/// A compute-tier value rounded to storage (nearest, ties toward +inf).
///
/// # Panics
/// Panics if the value does not fit the storage tier.
#[inline]
pub fn to_fixed<T: FixedNum>(x: Compute<T>) -> T {
    try_to_fixed(x).expect("compute_tier::to_fixed: value does not fit the storage tier")
}

/// A compute-tier value rounded to storage, `None` if it does not fit.
#[inline]
pub fn try_to_fixed<T: FixedNum>(x: Compute<T>) -> Option<T> {
    Some(T::from_raw_storage(T::downscale_to_storage(x).ok()?))
}

/// e^x at the compute tier; saturates at [`ceiling`], never wraps.
#[inline]
pub fn exp<T: FixedNum>(x: Compute<T>) -> Compute<T> {
    T::exp_at_compute_tier(x)
}

/// ln x at the compute tier.
///
/// # Panics
/// Panics if `x <= 0`.
#[inline]
pub fn ln<T: FixedNum>(x: Compute<T>) -> Compute<T> {
    assert!(
        !(T::compute_is_negative(&x) || T::compute_is_zero(&x)),
        "compute_tier::ln: x <= 0 is outside the domain"
    );
    T::ln_at_compute_tier(x)
}

/// sqrt x at the compute tier.
///
/// # Panics
/// Panics if `x < 0`.
#[inline]
pub fn sqrt<T: FixedNum>(x: Compute<T>) -> Compute<T> {
    assert!(
        !T::compute_is_negative(&x),
        "compute_tier::sqrt: x < 0 is outside the domain"
    );
    T::sqrt_at_compute_tier(x)
}

/// (sinh x, cosh x) at the compute tier from one exponential pair.
#[inline]
pub fn sinhcosh<T: FixedNum>(x: Compute<T>) -> (Compute<T>, Compute<T>) {
    T::sinhcosh_at_compute_tier(x)
}

/// 1 / (1 + e^-x) at the compute tier (sign-split: the exponential argument
/// is never positive and the denominator stays in [1, 2]).
#[inline]
pub fn sigmoid<T: FixedNum>(x: Compute<T>) -> Compute<T> {
    let one = T::make_compute_int(1);
    if T::compute_is_negative(&x) {
        // e^x <= 1, denominator in [1, 2]
        let e = T::exp_at_compute_tier(x);
        T::compute_divide(e, T::compute_add(one, e))
            .expect("sigmoid: denominator in [1,2] cannot overflow")
    } else {
        // e^-x <= 1, denominator in [1, 2]
        let e = T::exp_at_compute_tier(T::compute_negate(x));
        T::compute_divide(one, T::compute_add(one, e))
            .expect("sigmoid: denominator in [1,2] cannot overflow")
    }
}

/// ln(1 + e^x) at the compute tier (sign-split: `x + ln(1 + e^-|x|)` for
/// positive x, `ln(1 + e^x)` otherwise).
///
/// # Panics
/// Panics if the result does not fit the compute tier.
#[inline]
pub fn softplus<T: FixedNum>(x: Compute<T>) -> Compute<T> {
    let one = T::make_compute_int(1);
    let neg_abs = if T::compute_is_negative(&x) { x } else { T::compute_negate(x) };
    let e = T::exp_at_compute_tier(neg_abs); // in (0, 1]
    let corr = T::ln_at_compute_tier(T::compute_add(one, e)); // arg in (1, 2], always in domain
    if T::compute_is_negative(&x) {
        corr
    } else {
        T::compute_checked_add(x, corr)
            .expect("softplus: result does not fit the compute tier")
    }
}

/// ln(1 + x) at the compute tier.
///
/// # Panics
/// Panics if `x <= -1` or if `1 + x` does not fit the compute tier.
#[inline]
pub fn ln1p<T: FixedNum>(x: Compute<T>) -> Compute<T> {
    let arg = T::compute_checked_add(T::make_compute_int(1), x)
        .expect("ln1p: 1 + x does not fit the compute tier");
    assert!(
        !(T::compute_is_negative(&arg) || T::compute_is_zero(&arg)),
        "compute_tier::ln1p: x <= -1 is outside the domain"
    );
    T::ln_at_compute_tier(arg)
}
