#[cfg(table_format = "q16_16")]
mod trace {
    #[test]
    fn test_sin_q64_native_directly() {
        // Call sin_q64_64 with exact Q64.64 input: 3.0 = 3 << 64
        // This is the NATIVE Q64.64 function — same code as embedded profile
        use g_math::fixed_point::domains::binary_fixed::transcendental::sin_binary_i128;
        
        let three_q64: i128 = 3_i128 << 64;
        let result = sin_binary_i128(three_q64);
        
        // mpmath: sin(3) * 2^64 = 2603255189406976154
        // This is what the embedded profile produces at 0 ULP
        println!("sin_binary_i128(3.0 Q64.64) = {}", result);
        println!("expected = 2603255189406976154");
        println!("ULP at Q64.64 = {}", (result - 2603255189406976154_i128).abs());
        
        // Now downscale to Q32.32
        let q32 = (result >> 32) as i64;
        println!("Downscaled to Q32.32: {}", q32);
        // sin(3) * 2^32 = 606218427
        println!("expected Q32.32: 606218427");
        println!("ULP at Q32.32: {}", (q32 as i64 - 606218427).abs());
    }
}
