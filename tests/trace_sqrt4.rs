/// The Q64.64 square root engine on every build: sqrt(3) within one unit of
/// mpmath (the engine has one meaning in every build since the width design).
#[test]
fn sqrt_q64_64_engine_is_within_one_unit_on_every_build() {
    use g_math::fixed_point::domains::binary_fixed::transcendental::sqrt_tier_n_plus_1::sqrt_q64_64_native;
    let three_q64: i128 = 3_i128 << 64;
    let result = sqrt_q64_64_native(three_q64);
    let expected: i128 = 31950697969885030203; // mpmath sqrt(3) * 2^64 (verified 60-digit precision)
    let ulp = (result - expected).abs();
    assert!(ulp <= 1, "sqrt_q64_64_native(3) should be within 1 unit, got {} units", ulp);
}
