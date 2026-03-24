//! Q16.16 Realtime Profile ULP Validation
//!
//! Validates all transcendental functions against mpmath 50-digit reference values.
//! Run with: GMATH_PROFILE=realtime cargo test --test realtime_profile_validation -- --nocapture

#[cfg(table_format = "q16_16")]
mod realtime_tests {
    use g_math::fixed_point::canonical::{gmath, evaluate};

    include!("data/fasc_ulp_refs_q16.rs");

    fn gmath_safe(input: &'static str) -> g_math::fixed_point::canonical::LazyExpr {
        if input.starts_with('-') {
            let positive: &'static str = unsafe {
                std::str::from_utf8_unchecked(
                    std::slice::from_raw_parts(input.as_ptr().add(1), input.len() - 1)
                )
            };
            -gmath(positive)
        } else {
            gmath(input)
        }
    }

    #[test]
    fn test_realtime_transcendentals_vs_mpmath() {
        let mut total = 0;
        let mut max_ulp = 0i64;
        let mut failures = Vec::new();

        for &(input_str, _input_raw, expected_raw, func_name) in Q16_REFS {
            let expr = match func_name {
                "exp" => gmath_safe(input_str).exp(),
                "ln" => gmath_safe(input_str).ln(),
                "sqrt" => gmath_safe(input_str).sqrt(),
                "sin" => gmath_safe(input_str).sin(),
                "cos" => gmath_safe(input_str).cos(),
                "tan" => gmath_safe(input_str).tan(),
                "atan" => gmath_safe(input_str).atan(),
                "asin" => gmath_safe(input_str).asin(),
                "acos" => gmath_safe(input_str).acos(),
                "sinh" => gmath_safe(input_str).sinh(),
                "cosh" => gmath_safe(input_str).cosh(),
                "tanh" => gmath_safe(input_str).tanh(),
                "asinh" => gmath_safe(input_str).asinh(),
                "acosh" => gmath_safe(input_str).acosh(),
                "atanh" => gmath_safe(input_str).atanh(),
                _ => panic!("unknown function: {}", func_name),
            };

            match evaluate(&expr) {
                Ok(result) => {
                    match result.as_binary_storage() {
                        Some(got_raw) => {
                            let ulp = (got_raw as i64 - expected_raw as i64).abs();
                            if ulp > max_ulp { max_ulp = ulp; }
                            total += 1;

                            // ALL transcendentals MUST achieve 0-1 ULP with tier N+1 computation.
                            // ComputeStorage = i64 (Q32.32), computed via Q128.128 (I256).
                            // 96 guard bits above Q16.16 storage — error CANNOT survive downscale.
                            if ulp > 1 {
                                failures.push(format!(
                                    "{}({}) = {} (expected {}), {} ULP",
                                    func_name, input_str, got_raw, expected_raw, ulp
                                ));
                            }
                        }
                        None => {
                            let display = result.to_decimal_string(4);
                            failures.push(format!(
                                "{}({}) -> non-binary result: {}",
                                func_name, input_str, display
                            ));
                        }
                    }
                }
                Err(e) => {
                    failures.push(format!(
                        "{}({}) -> error: {:?}",
                        func_name, input_str, e
                    ));
                }
            }
        }

        println!("\n=== Q16.16 Realtime Profile ULP Validation ===");
        println!("Total test points: {}", total);
        println!("Max ULP: {}", max_ulp);
        println!("Failures (>1 ULP): {}", failures.len());
        for f in &failures {
            println!("  FAIL: {}", f);
        }

        assert!(failures.is_empty(),
            "Q16.16 realtime profile: {} failures out of {} tests (max ULP: {})\n{}",
            failures.len(), total, max_ulp,
            failures.join("\n"));
    }
}
