/// Compare sqrt_binary_i128(3 << 64) result across profiles
/// This should be identical since the function implementation is the same
#[test]
fn trace_sqrt_i128_consistency() {
    use g_math::fixed_point::domains::binary_fixed::transcendental::sqrt_binary_i128;

    // 3.0 in Q64.64
    let three_q64: i128 = 3_i128 << 64;
    let result = sqrt_binary_i128(three_q64);
    let expected: i128 = 31950697969885030203; // mpmath sqrt(3) * 2^64 (verified 60-digit precision)

    let ulp = (result - expected).abs();
    println!("Profile: Q{} sqrt_binary_i128(3.0 Q64.64)",
        if cfg!(table_format = "q64_64") { "64.64" }
        else if cfg!(table_format = "q32_32") { "32.32" }
        else { "other" });
    println!("result: {}", result);
    println!("expected: {}", expected);
    println!("ULP: {}", ulp);

    // This MUST be 0 ULP on any profile — it's the same function with the same input
    assert!(ulp <= 1, "sqrt_binary_i128 should be 0-1 ULP regardless of profile, got {} ULP", ulp);
}
