#[cfg(table_format = "q32_32")]
mod direct {
    use g_math::fixed_point::canonical::{gmath, evaluate};

    #[test]
    fn test_sqrt_values() {
        let cases = [
            ("1.0", 4294967296_i64),
            ("4.0", 8589934592_i64),
            ("9.0", 12884901888_i64),
            ("2.0", 6074001000_i64),
            ("3.0", 7439101574_i64), // mpmath: round(sqrt(3) * 2^32) verified at 60 digits
            ("0.25", 2147483648_i64),
            ("0.5", 3037000500_i64),
        ];

        for (input, expected) in &cases {
            let result = evaluate(&gmath(input).sqrt()).unwrap();
            let raw = result.as_binary_storage().unwrap();
            let ulp = (raw as i64 - expected).abs();
            println!("sqrt({}) raw={} expected={} ULP={}", input, raw, expected, ulp);
            assert!(ulp <= 1, "sqrt({}) has {} ULP, expected 0-1", input, ulp);
        }
    }
}
