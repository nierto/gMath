#[cfg(table_format = "q16_16")]
mod trace {
    use g_math::fixed_point::canonical::{gmath, evaluate};

    #[test]
    fn trace_sin_pi_approx() {
        // sin(3.14159265) — near pi, result should be ~0
        // If compute stays at Q32.32 throughout, downscale to Q16.16 MUST be 0 ULP
        
        // First: what does "3.14159265" parse as?
        let parsed = evaluate(&gmath("3.14159265")).unwrap();
        println!("'3.14159265' tier={}, raw={:?}", parsed.tier(), parsed.as_binary_storage());
        
        // sin of it
        let sin_result = evaluate(&gmath("3.14159265").sin()).unwrap();
        println!("sin('3.14159265') tier={}, raw={:?}", sin_result.tier(), sin_result.as_binary_storage());
        
        // What about sin(3) — integer input, definitely binary
        let sin3 = evaluate(&gmath("3").sin()).unwrap();
        println!("sin('3') tier={}, raw={:?}", sin3.tier(), sin3.as_binary_storage());
        
        // sin(1) — should be 0 ULP
        let sin1 = evaluate(&gmath("1").sin()).unwrap();
        let sin1_raw = sin1.as_binary_storage().unwrap() as i64;
        // mpmath: sin(1) * 2^16 = 55141
        println!("sin('1') raw={}, expected=55141, ULP={}", sin1_raw, (sin1_raw - 55141).abs());
    }
}
