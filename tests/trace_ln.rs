#[cfg(table_format = "q32_32")]
mod trace {
    use g_math::fixed_point::canonical::{gmath, evaluate};

    #[test]
    fn trace_ln_precision() {
        let cases: &[(&str, i64)] = &[
            ("1", 0),
            ("2", 2977044472),
            ("3", 4718503851),
            ("10", 9889527671),
        ];

        println!("=== ln() direct ULP tests on Q32.32 ===");
        for (input, expected) in cases {
            let result = evaluate(&gmath(input).ln()).unwrap();
            let raw = result.as_binary_storage().unwrap() as i64;
            let ulp = (raw - expected).abs();
            println!("ln({}) raw={} expected={} ULP={}", input, raw, expected, ulp);
            assert!(ulp <= 1, "ln({}) ULP={} exceeds 1", input, ulp);
        }

        // Now test ln with a fractional input via the canonical API
        // 1+sqrt(2) ≈ 2.41421356...
        println!("\n=== ln() chain test ===");
        let one_plus_sqrt2 = gmath("1") + gmath("2").sqrt();
        let ln_chain = one_plus_sqrt2.ln();
        let result = evaluate(&ln_chain).unwrap();
        let raw = result.as_binary_storage().unwrap() as i64;
        let expected: i64 = 3785470732;
        let ulp = (raw - expected).abs();
        println!("ln(1+sqrt(2)) raw={} expected={} ULP={}", raw, expected, ulp);
    }
}
