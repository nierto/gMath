#[cfg(table_format = "q32_32")]
mod trace {
    use g_math::fixed_point::domains::binary_fixed::transcendental::{
        sqrt_binary_i64, sqrt_binary_i128,
    };

    #[test]
    fn trace_sqrt_computation() {
        // 3.0 in Q32.32 = 3 * 2^32 = 12884901888
        let three_q32: i64 = 3 * (1i64 << 32);

        let result_i64 = sqrt_binary_i64(three_q32);
        let expected: i64 = 7439101574; // round(sqrt(3) * 2^32) from mpmath (verified 60-digit precision)
        println!("sqrt_binary_i64(3.0 Q32.32) = {}, expected {}, ULP = {}",
            result_i64, expected, (result_i64 - expected).abs());

        // Same via i128 compute-tier function directly
        // 3.0 in Q64.64 = 3 * 2^64
        let three_q64: i128 = 3 * (1i128 << 64);
        let result_i128 = sqrt_binary_i128(three_q64);
        // sqrt(3) in Q64.64 from mpmath
        let expected_q64: i128 = 31950697969885030203; // sqrt(3) * 2^64 from mpmath (verified 60-digit)
        println!("sqrt_binary_i128(3.0 Q64.64) = {}, expected {}, ULP_q64 = {}",
            result_i128, expected_q64, (result_i128 - expected_q64).abs());

        // Manual downscale
        let round_bit = (result_i128 & (1i128 << 31)) != 0;
        let mut downscaled = (result_i128 >> 32) as i64;
        if round_bit { downscaled += 1; }
        println!("manual downscale = {}, expected {}, ULP = {}",
            downscaled, expected, (downscaled - expected).abs());

        // Check: is the Q32.32 input correct?
        println!("\n3.0 in Q32.32: {} (expected {})", three_q32, 3i64 << 32);
        // Upscale manually: (three_q32 as i128) << 32
        let upscaled = (three_q32 as i128) << 32;
        println!("upscaled to Q64.64: {} (expected {})", upscaled, three_q64);
        println!("match: {}", upscaled == three_q64);
    }
}
