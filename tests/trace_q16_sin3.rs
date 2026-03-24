#[cfg(table_format = "q16_16")]
mod trace {
    use g_math::fixed_point::canonical::{gmath, evaluate};

    #[test]
    fn trace_sin3_directly() {
        // Call sin_binary_i64 directly with exact Q32.32 input
        use g_math::fixed_point::domains::binary_fixed::transcendental::sin_binary_i64;
        
        // 3.0 in Q32.32 = 3 * 2^32 = 12884901888
        // But ComputeStorage for Q16.16 is i64 — so the value in BinaryCompute is i64
        // 3.0 in Q32.32 as i64: 3 * 2^32 = 12884901888 — this DOESN'T fit in i64!
        // i64 max = 9223372036854775807
        // 3 * 2^32 = 12884901888 — this DOES fit in i64 actually
        let three_q32: i64 = 3 * (1i64 << 32);
        println!("3.0 in Q32.32: {}", three_q32);
        
        let result = sin_binary_i64(three_q32);
        println!("sin_binary_i64(3.0 Q32.32) = {}", result);
        
        // Expected: sin(3) * 2^32 = 0.14112... * 2^32 = 605969011
        // Wait no — sin_binary_i64 takes Q32.32 and returns Q32.32
        // sin(3) in Q32.32 = sin(3) * 2^32
        // But the Q16.16 downscale will do >> 16
        // Final Q16.16 = sin(3) * 2^16 = 9246.07 → 9246
        
        // sin_binary_i64 returns Q32.32:
        // sin(3) * 2^32 = 0.14112000806... * 4294967296 = 606218427
        println!("Expected Q32.32: 606218427");
        println!("ULP at Q32.32: {}", (result - 606218427_i64).abs());
        
        // Downscale to Q16.16
        let q16 = (result >> 16) as i32;
        println!("Downscaled to Q16.16: {}", q16);
        println!("Expected Q16.16: 9246");
        println!("ULP at Q16.16: {}", (q16 as i64 - 9246).abs());
    }
}
