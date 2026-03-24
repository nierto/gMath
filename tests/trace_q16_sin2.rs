#[cfg(table_format = "q16_16")]
mod trace {
    use g_math::fixed_point::canonical::{gmath, evaluate};

    #[test]
    fn trace_sin_path() {
        // The input "3.14159265" parses as tier 4 (Decimal with many decimal places)
        // What does evaluate see after parsing?
        
        // Test: sin of integer 3 (unambiguously binary)
        let sin3 = evaluate(&gmath("3").sin()).unwrap();
        let sin3_raw = sin3.as_binary_storage().unwrap();
        println!("sin(3) raw={}", sin3_raw);
        // mpmath: sin(3) * 2^16 = 9246.07... → 9246
        println!("sin(3) expected=9246, ULP={}", (sin3_raw as i64 - 9246).abs());

        // Test: cos of integer 2 (binary)
        let cos2 = evaluate(&gmath("2").cos()).unwrap();
        let cos2_raw = cos2.as_binary_storage().unwrap();
        println!("cos(2) raw={}", cos2_raw);
        // mpmath: cos(2) * 2^16 = -27253.7... → -27254
        println!("cos(2) expected=-27254, ULP={}", (cos2_raw as i64 - (-27254)).abs());
        
        // Test: tan(1) = sin(1)/cos(1) — composed chain
        let tan1 = evaluate(&gmath("1").tan()).unwrap();
        let tan1_raw = tan1.as_binary_storage().unwrap();
        println!("tan(1) raw={}", tan1_raw);
        // mpmath: tan(1) * 2^16 = 102041.92... → 102042
        println!("tan(1) expected=102042, ULP={}", (tan1_raw as i64 - 102042).abs());
    }
}
