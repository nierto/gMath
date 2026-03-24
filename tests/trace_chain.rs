//! Trace composed transcendental chain to find ULP source
#[cfg(table_format = "q32_32")]
mod trace {
    use g_math::fixed_point::canonical::{gmath, evaluate};

    #[test]
    fn trace_acosh_chain() {
        // acosh(2.0) = ln(2 + sqrt(2^2 - 1)) = ln(2 + sqrt(3))
        // mpmath: acosh(2.0) = 1.31695789692481670862504...
        // Q32.32 expected: 5656291098

        // Step 1: sqrt(3) directly
        let sqrt3 = evaluate(&gmath("3.0").sqrt()).unwrap();
        let sqrt3_raw = sqrt3.as_binary_storage().unwrap();
        // mpmath sqrt(3) = 1.7320508075688772... → raw = round(sqrt(3)*2^32) = 7439101574
        let sqrt3_expected: i64 = 7439101574;
        println!("sqrt(3) raw: {}, expected: {}, ULP: {}", sqrt3_raw, sqrt3_expected, (sqrt3_raw as i64 - sqrt3_expected).abs());

        // Step 2: 2 + sqrt(3)
        let sum = evaluate(&(gmath("2.0") + gmath("3.0").sqrt())).unwrap();
        let sum_raw = sum.as_binary_storage().unwrap();
        // 2 + sqrt(3) = 3.7320508... → raw = round((2+sqrt(3))*2^32) = 16029036166
        let sum_expected: i64 = 16029036166;
        println!("2+sqrt(3) raw: {}, expected: {}, ULP: {}", sum_raw, sum_expected, (sum_raw as i64 - sum_expected).abs());

        // Step 3: ln(2 + sqrt(3)) = acosh(2)
        let acosh_via_chain = evaluate(&(gmath("2.0") + gmath("3.0").sqrt()).ln()).unwrap();
        let chain_raw = acosh_via_chain.as_binary_storage().unwrap();
        let acosh_expected: i64 = 5656291098;
        println!("ln(2+sqrt(3)) raw: {}, expected: {}, ULP: {}", chain_raw, acosh_expected, (chain_raw as i64 - acosh_expected).abs());

        // Step 4: acosh(2) directly via FASC
        let acosh_direct = evaluate(&gmath("2.0").acosh()).unwrap();
        let direct_raw = acosh_direct.as_binary_storage().unwrap();
        println!("acosh(2.0) raw: {}, expected: {}, ULP: {}", direct_raw, acosh_expected, (direct_raw as i64 - acosh_expected).abs());

        // Step 5: ln(3.732) to test ln with a specific input
        let ln_val = evaluate(&gmath("3.7320508").ln()).unwrap();
        let ln_raw = ln_val.as_binary_storage().unwrap();
        println!("ln(3.7320508) raw: {}, expected ~acosh: {}, ULP: {}", ln_raw, acosh_expected, (ln_raw as i64 - acosh_expected).abs());
    }
}
