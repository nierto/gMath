//! Q32.32 Compact Profile ULP Validation
//!
//! Validates all transcendental functions against mpmath 50-digit reference values.
//! Run with: cargo test --test transcendentals compact_profile_validation:: -- --nocapture


macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            type FixedPoint = Fixed<$S, $F>;
            #[allow(dead_code)]
            const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
            #[allow(dead_code)]
            const FRAC_BITS: u32 = $F;
            use g_math::fixed_point::Fixed;
            mod compact_tests {
                #[allow(unused_imports)]
                use super::{BITS, FRAC_BITS, FixedPoint};
                use g_math::fixed_point::canonical::{gmath, evaluate};

                include!("data/fasc_ulp_refs_q32.rs");

                fn gmath_safe(input: &'static str) -> g_math::fixed_point::canonical::LazyExpr<FixedPoint> {
                    if input.starts_with('-') {
                        let positive: &'static str = unsafe {
                            std::str::from_utf8_unchecked(
                                std::slice::from_raw_parts(input.as_ptr().add(1), input.len() - 1)
                            )
                        };
                        -gmath::<FixedPoint>(positive)
                    } else {
                        gmath::<FixedPoint>(input)
                    }
                }

                #[test]
                fn test_compact_transcendentals_vs_mpmath() {
                    let mut total = 0;
                    let mut max_ulp = 0i64;
                    let mut failures = Vec::new();

                    for &(input_str, _input_raw, expected_raw, func_name) in Q32_REFS {
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
                                        let ulp = (got_raw as i64 - expected_raw).abs();
                                        if ulp > max_ulp { max_ulp = ulp; }
                                        total += 1;

                                        // Dedicated engines (exp, ln, sqrt, sin, cos, atan) should achieve 0-1 ULP
                                        // Composed functions (tan, asin, acos, sinh, cosh, tanh, asinh, acosh, atanh)
                                        // may have higher ULP due to FASC chain materialization at Q32.32 precision
                                        // ALL transcendentals should achieve 0-1 ULP with tier N+1 computation.
                                        // BinaryCompute chain persistence keeps intermediates at Q64.64 (i128)
                                        // and only downscales once to Q32.32 at materialization.
                                        if ulp > 1 {
                                            failures.push(format!(
                                                "{}({}) = {} (expected {}), {} ULP",
                                                func_name, input_str, got_raw, expected_raw, ulp
                                            ));
                                        }
                                    }
                                    None => {
                                        let display = result.to_decimal_string(9);
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

                    println!("\n=== Q32.32 Compact Profile ULP Validation ===");
                    println!("Total test points: {}", total);
                    println!("Max ULP: {}", max_ulp);
                    println!("Failures (>1 ULP): {}", failures.len());
                    for f in &failures {
                        println!("  FAIL: {}", f);
                    }

                    assert!(failures.is_empty(),
                        "Q32.32 compact profile: {} failures out of {} tests (max ULP: {})\n{}",
                        failures.len(), total, max_ulp,
                        failures.join("\n"));
                }
            }
        }
    };
}

// the word this suite's references are for
suite!(q32_32, i64, 32);
