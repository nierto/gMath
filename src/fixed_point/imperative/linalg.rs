//! The 32-bit word's long dot product path (the bounded unchecked AVX2 sum
//! behind `Storage::dot_acc_fast`) and the stagnation constant of the
//! iterative decompositions. Everything else that lived here is generic
//! over the format in `kernels` since the width design.

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


/// Consecutive non-improving iterations after which an iterative decomposition
/// is taken to sit at its precision floor.
pub(crate) const STAGNATION_SWEEPS: usize = 5;
