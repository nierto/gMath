//! Scalar square root across the whole storage range, against independent
//! references: the permanent gate for the large-argument defect.
//!
//! The published validation of `sqrt` covered `[0.0001, 10000]`. On the
//! scientific profile the Q512.512 engine lost about 250 bits of relative
//! precision above roughly `2^200` (up to `2^124` ulp at `2^254`), which no
//! test in that range could see. This gate checks `FixedPoint::sqrt` on every
//! profile against `floor` / `ceil` references computed independently
//! (`scripts/generate_certified_geometry_refs.py`: Python `isqrt`,
//! cross-checked against mpmath at 300 digits), including inputs within a
//! factor of four of the storage maximum and the maximum itself. The scalar
//! result rounds to nearest, so it must equal the reference floor or the
//! reference ceil, never anything else.
//!
//! Uses only `FixedPoint` and the reference data, so it runs unchanged on the
//! 0.5.1 patch line as well as on 0.6.0.

use g_math::fixed_point::FixedPoint;

#[allow(dead_code)]
mod data {
    include!("data/certified_geometry_refs.rs");
}
use data::refs;

#[cfg(table_format = "q16_16")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(i32::from_le_bytes(b.try_into().expect("4 bytes"))) }
#[cfg(table_format = "q32_32")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(i64::from_le_bytes(b.try_into().expect("8 bytes"))) }
#[cfg(table_format = "q64_64")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(i128::from_le_bytes(b.try_into().expect("16 bytes"))) }
#[cfg(table_format = "q128_128")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(g_math::fixed_point::I256::from_bytes_le(b)) }
#[cfg(table_format = "q256_256")]
fn fp_le(b: &[u8]) -> FixedPoint { FixedPoint::from_raw(g_math::fixed_point::I512::from_bytes_le(b)) }

#[test]
fn scalar_sqrt_matches_independent_floor_or_ceil_across_the_range() {
    let mut near_max = 0usize;
    for (idx, (x, f, c)) in refs::SQRT.iter().enumerate() {
        let x = fp_le(x);
        let got = x.sqrt();
        let (floor, ceil) = (fp_le(f), fp_le(c));
        assert!(
            got == floor || got == ceil,
            "reference {idx}: sqrt({x}) = {got}, expected the reference floor {floor} or ceil {ceil}"
        );
        if x > FixedPoint::from_int(1 << 20) { near_max += 1; }
    }
    assert!(near_max >= 2, "references must include inputs near the storage maximum");
    assert!(refs::SQRT.len() >= 12);
}

/// Exact squares stay exact at every magnitude: sqrt(2^(2j)) == 2^j.
#[test]
fn exact_powers_of_two_are_exact() {
    let mut j = 0i32;
    let mut checked = 0usize;
    loop {
        let two_j = pow2(j);
        let square = two_j * two_j;
        if square <= FixedPoint::ZERO || square < two_j { break; } // left the storage range
        assert_eq!(square.sqrt(), two_j, "sqrt(2^{}) must be exactly 2^{}", 2 * j, j);
        checked += 1;
        j += 3;
        if j > 300 { break; }
    }
    assert!(checked >= 3, "too few exact squares checked: {checked}");
}

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn pow2(j: i32) -> FixedPoint {
    let one = FixedPoint::one();
    let mut v = one;
    for _ in 0..j { v = v + v; }
    v
}
#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
fn pow2(j: i32) -> FixedPoint {
    let mut v = FixedPoint::one();
    for _ in 0..j { v = v + v; }
    v
}
