// Compare asinh(1.0) on embedded vs compact to find where precision loss occurs
use g_math::fixed_point::canonical::{gmath, evaluate};

#[test]
fn trace_asinh_cross_profile() {
    // asinh(1.0) = ln(1 + sqrt(1 + 1)) = ln(1 + sqrt(2))
    let result = evaluate(&gmath("1.0").asinh()).unwrap();
    let _raw = result.as_binary_storage().unwrap();

    #[cfg(table_format = "q64_64")]
    {
        let expected: i128 = 16258472993076885089;
        let ulp = (_raw - expected).abs();
        println!("[Q64.64] asinh(1.0) raw={}, expected={}, ULP={}", _raw, expected, ulp);
        assert!(ulp <= 1, "Q64.64 asinh(1.0) ULP={}", ulp);
    }
    #[cfg(table_format = "q32_32")]
    {
        let expected: i64 = 3785470732;
        let ulp = (_raw as i64 - expected).abs();
        println!("[Q32.32] asinh(1.0) raw={}, expected={}, ULP={}", _raw, expected, ulp);
        // With tier N+1, this SHOULD be 0-1 ULP
        println!("Result tier: {}", result.tier());
    }
}
