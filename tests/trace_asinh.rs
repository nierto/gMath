//! Pin of the canonical asinh(1) at Q64.64 (mpmath), on every build.
use g_math::fixed_point::canonical::{evaluate_in, gmath_in};
use g_math::fixed_point::Q64_64;

#[test]
fn canonical_asinh_one_pin_at_q64_64() {
    let result = evaluate_in(&gmath_in::<Q64_64>("1.0").asinh()).unwrap();
    let raw = result.as_binary_storage().unwrap();
    let expected: i128 = 16258472993076885089;
    let ulp = (raw - expected).abs();
    assert!(ulp <= 1, "Q64.64 asinh(1.0) ULP={ulp}");
}
