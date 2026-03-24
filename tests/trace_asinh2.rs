#[cfg(table_format = "q32_32")]
mod trace {
    use g_math::fixed_point::canonical::{gmath, evaluate};

    #[test]
    fn trace_asinh_step_by_step() {
        // asinh(1) = ln(1 + sqrt(1 + 1)) = ln(1 + sqrt(2))

        // Step 1: x^2 = 1*1 = 1
        let x_sq = evaluate(&(gmath("1") * gmath("1"))).unwrap();
        println!("x^2 type: {:?}, tier: {}",
            if x_sq.as_binary_storage().is_some() { "Binary" } else { "Other" },
            x_sq.tier());

        // Step 2: x^2 + 1 = 2
        let x_sq_p1 = evaluate(&(gmath("1") * gmath("1") + gmath("1"))).unwrap();
        println!("x^2+1 raw: {:?}, tier: {}", x_sq_p1.as_binary_storage(), x_sq_p1.tier());

        // Step 3: sqrt(2)
        let sqrt2 = evaluate(&(gmath("1") * gmath("1") + gmath("1")).sqrt()).unwrap();
        let sqrt2_raw = sqrt2.as_binary_storage().unwrap();
        // mpmath: sqrt(2) * 2^32 = 6074001000
        let sqrt2_expected: i64 = 6074001000;
        println!("sqrt(2) raw={}, expected={}, ULP={}", sqrt2_raw, sqrt2_expected, (sqrt2_raw as i64 - sqrt2_expected).abs());

        // Step 4: 1 + sqrt(2)
        let one_plus_sqrt2 = evaluate(&(gmath("1") + (gmath("1") * gmath("1") + gmath("1")).sqrt())).unwrap();
        let sum_raw = one_plus_sqrt2.as_binary_storage().unwrap();
        // mpmath: (1+sqrt(2)) * 2^32 = 10368968296
        let sum_expected: i64 = 10368968296;
        println!("1+sqrt(2) raw={}, expected={}, ULP={}", sum_raw, sum_expected, (sum_raw as i64 - sum_expected).abs());

        // Step 5: ln(1 + sqrt(2)) = asinh(1)
        let result = evaluate(&(gmath("1") + (gmath("1") * gmath("1") + gmath("1")).sqrt()).ln()).unwrap();
        let raw = result.as_binary_storage().unwrap();
        let expected: i64 = 3785470732;
        println!("ln(1+sqrt(2)) raw={}, expected={}, ULP={}", raw, expected, (raw as i64 - expected).abs());

        // Also test direct asinh
        let asinh_direct = evaluate(&gmath("1").asinh()).unwrap();
        let asinh_raw = asinh_direct.as_binary_storage().unwrap();
        println!("asinh(1) raw={}, expected={}, ULP={}", asinh_raw, expected, (asinh_raw as i64 - expected).abs());

        // Test JUST sqrt(2) directly
        let sqrt2_direct = evaluate(&gmath("2").sqrt()).unwrap();
        let sqrt2d_raw = sqrt2_direct.as_binary_storage().unwrap();
        println!("\nsqrt(2) direct: raw={}, expected={}, ULP={}",
            sqrt2d_raw, sqrt2_expected, (sqrt2d_raw as i64 - sqrt2_expected).abs());
    }
}
