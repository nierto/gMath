mod final_check {
    use g_math::fixed_point::domains::binary_fixed::transcendental::sin_binary_i128;
    
    #[test]
    fn verify_sin_across_profiles() {
        let three_q64: i128 = 3_i128 << 64;
        let result = sin_binary_i128(three_q64);
        // mpmath: sin(3) * 2^64 = 2603204672360199838
        let expected: i128 = 2603204672360199838;
        let ulp = (result - expected).abs();
        println!("sin_binary_i128(3 << 64) = {}", result);
        println!("expected (mpmath)        = {}", expected);
        println!("ULP at Q64.64            = {}", ulp);
        
        // Downscale to Q32.32
        let q32 = (result >> 32) as i64;
        // sin(3) * 2^32 = 606105819
        let expected_q32: i64 = 606105819;
        println!("Q32.32 = {}, expected = {}, ULP = {}", q32, expected_q32, (q32 - expected_q32).abs());

        // Downscale to Q16.16
        let q16 = (result >> 48) as i32;
        // sin(3) * 2^16 = 9248 (0.14112 * 65536)
        let expected_q16: i32 = 9247;
        println!("Q16.16 = {}, expected = {}, ULP = {}", q16, expected_q16, (q16 - expected_q16).abs());
    }
}
