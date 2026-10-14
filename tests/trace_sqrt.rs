//! Pins of the canonical sqrt at Q32.32 (mpmath, verified at 60 digits), on
//! every build.
use g_math::fixed_point::canonical::{evaluate_in, gmath_in};
use g_math::fixed_point::Q32_32;

#[test]
fn canonical_sqrt_pins_at_q32_32() {
    let cases: [(&str, i64); 7] = [
        ("1.0", 4294967296),
        ("4.0", 8589934592),
        ("9.0", 12884901888),
        ("2.0", 6074001000),
        ("3.0", 7439101574),
        ("0.25", 2147483648),
        ("0.5", 3037000500),
    ];
    for (input, expected) in cases {
        let result = evaluate_in(&gmath_in::<Q32_32>(input).sqrt()).unwrap();
        let raw = result.as_binary_storage().unwrap();
        let ulp = (raw - expected).abs();
        assert!(ulp <= 1, "sqrt({input}) has {ulp} ULP, expected 0-1");
    }
}
