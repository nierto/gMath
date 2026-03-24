#[cfg(table_format = "q32_32")]
mod trace {
    use g_math::fixed_point::canonical::{gmath, evaluate};

    #[test]
    fn trace_sqrt_input_path() {
        // "3.0" is parsed as Decimal → then sqrt converts to binary
        // "3" is parsed as Integer → Binary directly

        // Test with integer input (parsed as Binary)
        let sqrt3_int = evaluate(&gmath("3").sqrt()).unwrap();
        let raw_int = sqrt3_int.as_binary_storage().unwrap();
        let expected: i64 = 7439101574; // mpmath: round(sqrt(3) * 2^32) verified at 60 digits
        println!("sqrt(3) [integer parse] raw={} expected={} ULP={}",
            raw_int, expected, (raw_int as i64 - expected).abs());

        // Test with decimal input (parsed as Decimal)
        let sqrt3_dec = evaluate(&gmath("3.0").sqrt()).unwrap();
        let raw_dec = sqrt3_dec.as_binary_storage().unwrap();
        println!("sqrt(3.0) [decimal parse] raw={} expected={} ULP={}",
            raw_dec, expected, (raw_dec as i64 - expected).abs());

        // Test parsing: what does "3.0" actually become?
        let three_dec = evaluate(&gmath("3.0")).unwrap();
        let three_int = evaluate(&gmath("3")).unwrap();
        println!("3.0 raw: {:?}", three_dec.as_binary_storage());
        println!("3   raw: {:?}", three_int.as_binary_storage());
        println!("3.0 tier: {}", three_dec.tier());
        println!("3   tier: {}", three_int.tier());
    }
}
