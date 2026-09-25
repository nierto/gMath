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

/// Bit length of the magnitude of little-endian two's-complement bytes.
fn magnitude_bits(b: &[u8]) -> u32 {
    let negative = b[b.len() - 1] & 0x80 != 0;
    let mag: Vec<u8> = b.iter().map(|x| if negative { !x } else { *x }).collect();
    for (i, byte) in mag.iter().enumerate().rev() {
        if *byte != 0 { return i as u32 * 8 + (8 - byte.leading_zeros()); }
    }
    0
}

#[test]
fn scalar_sqrt_matches_independent_floor_or_ceil_across_the_range() {
    let mut near_max = 0usize;
    for (idx, (xb, f, c)) in refs::SQRT.iter().enumerate() {
        let x = fp_le(xb);
        let got = x.sqrt();
        let (floor, ceil) = (fp_le(f), fp_le(c));
        assert!(
            got == floor || got == ceil,
            "reference {idx}: sqrt({x}) = {got}, expected the reference floor {floor} or ceil {ceil}"
        );
        // the generator's near-maximum inputs: at least W - 2 bits of magnitude
        if magnitude_bits(xb) >= refs::STORAGE_BITS - 2 { near_max += 1; }
    }
    assert!(near_max >= 2, "references must include inputs near the storage maximum");
    assert!(refs::SQRT.len() >= 9);
}

/// Exact squares stay exact at every magnitude: sqrt(2^(2j)) == 2^j.
#[test]
fn exact_powers_of_two_are_exact() {
    // every 2^(2j) the build represents: 2^-F <= 2^(2j) < 2^(integer bits)
    // (the old loop detected the range edge by letting the multiply wrap)
    let f = g_math::fixed_point::frac_config::FRAC_BITS as i32;
    let int_bits = refs::STORAGE_BITS as i32 - 1 - f;
    let mut checked = 0usize;
    for j in (-f / 2)..=((int_bits - 1) / 2) {
        let two_j = pow2(j);
        assert_eq!((two_j * two_j).sqrt(), two_j, "sqrt(2^{}) must be exactly 2^{}", 2 * j, j);
        checked += 1;
    }
    assert!(checked >= 3, "too few exact squares checked: {checked}");
}

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn pow2(j: i32) -> FixedPoint {
    let mut v = FixedPoint::one();
    for _ in 0..j.max(0) { v = v + v; }
    for _ in 0..(-j).max(0) { v = v / FixedPoint::from_int(2); }
    v
}
#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
fn pow2(j: i32) -> FixedPoint {
    let mut v = FixedPoint::one();
    for _ in 0..j.max(0) { v = v + v; }
    for _ in 0..(-j).max(0) { v = v / FixedPoint::from_int(2); }
    v
}
