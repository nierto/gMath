//! Pins of the canonical ln at Q32.32 (mpmath), on every build.
use g_math::fixed_point::canonical::{evaluate_in, gmath_in};
use g_math::fixed_point::Q32_32;

#[test]
fn canonical_ln_pins_at_q32_32() {
    let cases: [(&str, i64); 4] = [("1", 0), ("2", 2977044472), ("3", 4718503851), ("10", 9889527671)];
    for (input, expected) in cases {
        let result = evaluate_in(&gmath_in::<Q32_32>(input).ln()).unwrap();
        let raw = result.as_binary_storage().unwrap();
        let ulp = (raw - expected).abs();
        assert!(ulp <= 1, "ln({input}) ULP={ulp} exceeds 1");
    }
    // a chain: ln(1 + sqrt(2)) = asinh(1)
    let chain = (gmath_in::<Q32_32>("1") + gmath_in::<Q32_32>("2").sqrt()).ln();
    let raw = evaluate_in(&chain).unwrap().as_binary_storage().unwrap();
    assert!((raw - 3785470732i64).abs() <= 1, "ln(1 + sqrt 2) raw {raw}");
}
