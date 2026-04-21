//! Fused sinhcosh validation — mpmath-backed, correlation, and overflow gates.
//!
//! Covers:
//! - `FixedPoint::sinhcosh` / `try_sinhcosh` (imperative binary)
//! - `DecimalFixed::sinhcosh`                (imperative decimal)
//! - `canonical::evaluate_sinhcosh`          (FASC surface, domain routing)
//!
//! ULP references are reused from the per-profile `fasc_ulp_refs_q{16,32}.rs`
//! datasets (generated at mpmath 50-digit precision). Sinhcosh must produce
//! the same storage-tier values as separate `sinh` + `cosh` — its benefit is
//! throughput (one exp-pair vs two) and error correlation, not different
//! output values.
//!
//! Run with:
//!   GMATH_PROFILE=realtime cargo test --test sinhcosh_validation -- --nocapture
//!   GMATH_PROFILE=compact  cargo test --test sinhcosh_validation -- --nocapture

#[cfg(table_format = "q16_16")]
mod realtime_tests {
    use g_math::fixed_point::FixedPoint;
    use g_math::fixed_point::canonical::{gmath, evaluate, evaluate_sinhcosh};

    include!("data/fasc_ulp_refs_q16.rs");

    fn gmath_safe(input: &'static str) -> g_math::fixed_point::canonical::LazyExpr {
        if input.starts_with('-') {
            let positive: &'static str = unsafe {
                std::str::from_utf8_unchecked(
                    std::slice::from_raw_parts(input.as_ptr().add(1), input.len() - 1),
                )
            };
            -gmath(positive)
        } else {
            gmath(input)
        }
    }

    /// FixedPoint::sinhcosh must match the mpmath references for sinh and cosh.
    #[test]
    fn binary_sinhcosh_matches_mpmath() {
        let mut failures = Vec::new();
        let mut sinh_map: Vec<(i32, i32)> = Vec::new();
        let mut cosh_map: Vec<(i32, i32)> = Vec::new();

        for &(_input_str, input_raw, expected_raw, func_name) in Q16_REFS {
            match func_name {
                "sinh" => sinh_map.push((input_raw, expected_raw)),
                "cosh" => cosh_map.push((input_raw, expected_raw)),
                _ => {}
            }
        }
        assert!(!sinh_map.is_empty(), "no sinh references found");
        assert!(!cosh_map.is_empty(), "no cosh references found");

        for &(input_raw, sinh_expected) in &sinh_map {
            let cosh_expected = cosh_map
                .iter()
                .find(|(x, _)| *x == input_raw)
                .map(|(_, e)| *e)
                .expect("no matching cosh ref");

            let x = FixedPoint::from_raw(input_raw);
            let (s, c) = x.sinhcosh();

            let sinh_ulp = (s.raw() as i64 - sinh_expected as i64).abs();
            let cosh_ulp = (c.raw() as i64 - cosh_expected as i64).abs();

            if sinh_ulp > 1 {
                failures.push(format!(
                    "sinh({}) = {} (expected {}), {} ULP",
                    input_raw, s.raw(), sinh_expected, sinh_ulp
                ));
            }
            if cosh_ulp > 1 {
                failures.push(format!(
                    "cosh({}) = {} (expected {}), {} ULP",
                    input_raw, c.raw(), cosh_expected, cosh_ulp
                ));
            }
        }

        assert!(failures.is_empty(), "ULP failures:\n{}", failures.join("\n"));
    }

    /// Fused sinhcosh must be bit-identical to separate sinh() + cosh() calls.
    /// (They share the same compute-tier exp pair; downstream rounding is the same.)
    #[test]
    fn binary_sinhcosh_matches_separate_calls() {
        for &(_input_str, input_raw, _expected_raw, func_name) in Q16_REFS {
            if func_name != "sinh" { continue; }
            let x = FixedPoint::from_raw(input_raw);
            let (fused_s, fused_c) = x.sinhcosh();
            let sep_s = x.sinh();
            let sep_c = x.cosh();
            assert_eq!(fused_s.raw(), sep_s.raw(),
                "sinhcosh.sinh != sinh for raw={}", input_raw);
            assert_eq!(fused_c.raw(), sep_c.raw(),
                "sinhcosh.cosh != cosh for raw={}", input_raw);
        }
    }

    /// Identity: cosh²(x) - sinh²(x) = 1. Use relative tightness vs sep calls.
    #[test]
    fn binary_sinhcosh_identity_tight() {
        for &(_input_str, input_raw, _expected, func_name) in Q16_REFS {
            if func_name != "sinh" { continue; }
            let x = FixedPoint::from_raw(input_raw);
            let (s, c) = x.sinhcosh();
            let one = FixedPoint::from_int(1);
            let diff = (c * c) - (s * s) - one;
            // Q16.16 ULP tolerance for the identity: 8 ULP absolute at storage tier.
            // (cosh amplifies input ULP, so identity residual scales with cosh²).
            assert!(diff.raw().abs() <= 8,
                "identity cosh²-sinh²=1 residual too large for raw={}: diff={}",
                input_raw, diff.raw());
        }
    }

    /// try_sinhcosh must return Err on inputs large enough that cosh overflows storage.
    #[test]
    fn binary_try_sinhcosh_overflow_gate() {
        // Q16.16 storage max ~32768. cosh(12) ≈ 81377, already overflows.
        let big = FixedPoint::from_int(15);
        let res = big.try_sinhcosh();
        assert!(res.is_err(), "try_sinhcosh(15) should overflow Q16.16 storage");
    }

    /// FASC evaluate_sinhcosh via gmath("...").sinhcosh() must match imperative path.
    #[test]
    fn fasc_evaluate_sinhcosh_matches_imperative() {
        for &(input_str, input_raw, _exp, func_name) in Q16_REFS {
            if func_name != "sinh" { continue; }
            let (fasc_s, fasc_c) = evaluate_sinhcosh(&gmath_safe(input_str))
                .expect("FASC sinhcosh eval");
            let fasc_s_raw = fasc_s.as_binary_storage().expect("binary storage");
            let fasc_c_raw = fasc_c.as_binary_storage().expect("binary storage");

            let (imp_s, imp_c) = FixedPoint::from_raw(input_raw).sinhcosh();
            // FASC path may route through decimal-compute for "0.5"-style literals and
            // downscale back; imperative path is pure binary. Tolerate 1 ULP drift.
            assert!((fasc_s_raw as i64 - imp_s.raw() as i64).abs() <= 1,
                "FASC vs imperative sinh mismatch for {}: {} vs {}",
                input_str, fasc_s_raw, imp_s.raw());
            assert!((fasc_c_raw as i64 - imp_c.raw() as i64).abs() <= 1,
                "FASC vs imperative cosh mismatch for {}: {} vs {}",
                input_str, fasc_c_raw, imp_c.raw());

            // Also cross-check FASC pair against separate FASC sinh/cosh.
            let sin_only = evaluate(&gmath_safe(input_str).sinh()).unwrap()
                .as_binary_storage().unwrap();
            let cos_only = evaluate(&gmath_safe(input_str).cosh()).unwrap()
                .as_binary_storage().unwrap();
            assert!((fasc_s_raw as i64 - sin_only as i64).abs() <= 1,
                "FASC pair sinh vs FASC single sinh drift at {}", input_str);
            assert!((fasc_c_raw as i64 - cos_only as i64).abs() <= 1,
                "FASC pair cosh vs FASC single cosh drift at {}", input_str);
        }
    }
}

#[cfg(table_format = "q32_32")]
mod compact_tests {
    use g_math::fixed_point::FixedPoint;
    use g_math::fixed_point::canonical::{gmath, evaluate, evaluate_sinhcosh};

    include!("data/fasc_ulp_refs_q32.rs");

    fn gmath_safe(input: &'static str) -> g_math::fixed_point::canonical::LazyExpr {
        if input.starts_with('-') {
            let positive: &'static str = unsafe {
                std::str::from_utf8_unchecked(
                    std::slice::from_raw_parts(input.as_ptr().add(1), input.len() - 1),
                )
            };
            -gmath(positive)
        } else {
            gmath(input)
        }
    }

    #[test]
    fn binary_sinhcosh_matches_mpmath() {
        let mut failures = Vec::new();
        let mut sinh_map: Vec<(i64, i64)> = Vec::new();
        let mut cosh_map: Vec<(i64, i64)> = Vec::new();

        for &(_input_str, input_raw, expected_raw, func_name) in Q32_REFS {
            match func_name {
                "sinh" => sinh_map.push((input_raw, expected_raw)),
                "cosh" => cosh_map.push((input_raw, expected_raw)),
                _ => {}
            }
        }
        assert!(!sinh_map.is_empty() && !cosh_map.is_empty(),
            "missing sinh/cosh refs in Q32 dataset");

        for &(input_raw, sinh_expected) in &sinh_map {
            let cosh_expected = cosh_map.iter()
                .find(|(x, _)| *x == input_raw)
                .map(|(_, e)| *e)
                .expect("no matching cosh ref");

            let x = FixedPoint::from_raw(input_raw);
            let (s, c) = x.sinhcosh();

            let sinh_ulp = (s.raw() as i128 - sinh_expected as i128).abs();
            let cosh_ulp = (c.raw() as i128 - cosh_expected as i128).abs();

            if sinh_ulp > 1 {
                failures.push(format!(
                    "sinh({}) = {} (expected {}), {} ULP",
                    input_raw, s.raw(), sinh_expected, sinh_ulp
                ));
            }
            if cosh_ulp > 1 {
                failures.push(format!(
                    "cosh({}) = {} (expected {}), {} ULP",
                    input_raw, c.raw(), cosh_expected, cosh_ulp
                ));
            }
        }

        assert!(failures.is_empty(), "ULP failures:\n{}", failures.join("\n"));
    }

    #[test]
    fn binary_sinhcosh_matches_separate_calls() {
        for &(_input_str, input_raw, _expected, func_name) in Q32_REFS {
            if func_name != "sinh" { continue; }
            let x = FixedPoint::from_raw(input_raw);
            let (fused_s, fused_c) = x.sinhcosh();
            let sep_s = x.sinh();
            let sep_c = x.cosh();
            assert_eq!(fused_s.raw(), sep_s.raw());
            assert_eq!(fused_c.raw(), sep_c.raw());
        }
    }

    #[test]
    fn binary_sinhcosh_identity_tight() {
        for &(_input_str, input_raw, _expected, func_name) in Q32_REFS {
            if func_name != "sinh" { continue; }
            let x = FixedPoint::from_raw(input_raw);
            let (s, c) = x.sinhcosh();
            let one = FixedPoint::from_int(1);
            let diff = (c * c) - (s * s) - one;
            // Q32.32 storage: tighter tolerance. cosh(2) ≈ 3.76, cosh² ≈ 14.
            // Per-coord ULP multiplied by cosh² ≈ a few dozen ULP at most.
            assert!(diff.raw().abs() <= 64,
                "identity residual too large for raw={}: diff={}",
                input_raw, diff.raw());
        }
    }

    #[test]
    fn binary_try_sinhcosh_overflow_gate() {
        // Q32.32 storage max ~2.1e9. cosh(22) ≈ 1.78e9 (fits). cosh(25) ≈ 3.6e10 (overflows).
        let big = FixedPoint::from_int(25);
        let res = big.try_sinhcosh();
        assert!(res.is_err(), "try_sinhcosh(25) should overflow Q32.32 storage");
    }

    #[test]
    fn fasc_evaluate_sinhcosh_matches_imperative() {
        for &(input_str, input_raw, _exp, func_name) in Q32_REFS {
            if func_name != "sinh" { continue; }
            let (fasc_s, fasc_c) = evaluate_sinhcosh(&gmath_safe(input_str))
                .expect("FASC sinhcosh eval");
            let fasc_s_raw = fasc_s.as_binary_storage().expect("binary storage");
            let fasc_c_raw = fasc_c.as_binary_storage().expect("binary storage");

            let (imp_s, imp_c) = FixedPoint::from_raw(input_raw).sinhcosh();
            assert!((fasc_s_raw as i128 - imp_s.raw() as i128).abs() <= 1);
            assert!((fasc_c_raw as i128 - imp_c.raw() as i128).abs() <= 1);

            let sin_only = evaluate(&gmath_safe(input_str).sinh()).unwrap()
                .as_binary_storage().unwrap();
            let cos_only = evaluate(&gmath_safe(input_str).cosh()).unwrap()
                .as_binary_storage().unwrap();
            assert!((fasc_s_raw as i128 - sin_only as i128).abs() <= 1);
            assert!((fasc_c_raw as i128 - cos_only as i128).abs() <= 1);
        }
    }
}

// ============================================================================
// DECIMAL DOMAIN — DecimalFixed::sinhcosh
// ============================================================================
//
// DecimalFixed<DECIMALS> values: compare raw scaled i128 against separate sinh
// + cosh results. sinhcosh must produce identical storage values, since both
// routes ultimately materialize the same (exp(x) ± exp(-x))/2 at decimal
// compute tier. Independent mpmath references are redundant for the pair test
// — the separate sinh/cosh already have 0-ULP validation in their own suites.

#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
mod decimal_tests {
    use g_math::fixed_point::DecimalFixed;
    use std::str::FromStr;

    fn approx_equal<const DECIMALS: u8>(
        a: DecimalFixed<DECIMALS>,
        b: DecimalFixed<DECIMALS>,
        tol_raw: i128,
    ) -> bool {
        (a.raw_value() - b.raw_value()).abs() <= tol_raw
    }

    #[test]
    fn decimal_sinhcosh_matches_separate_calls() {
        // DECIMALS=4: 4 decimal places, comfortably representable in i128 at Q16.16.
        let inputs = ["0.0", "0.5", "1.0", "-0.5", "-1.0", "2.0"];
        for s in &inputs {
            let x = DecimalFixed::<4>::from_str(s).expect("decimal parse");
            let (fused_s, fused_c) = x.sinhcosh();
            let sep_s = x.sinh();
            let sep_c = x.cosh();
            // Decimal sinhcosh uses compute-tier fused path; sinh/cosh use storage-tier
            // recomputation. Both should land within 1 raw unit at storage dp.
            assert!(approx_equal(fused_s, sep_s, 1),
                "decimal sinhcosh.sinh drifts from sinh() at {}: fused={} sep={}",
                s, fused_s.raw_value(), sep_s.raw_value());
            assert!(approx_equal(fused_c, sep_c, 1),
                "decimal sinhcosh.cosh drifts from cosh() at {}: fused={} sep={}",
                s, fused_c.raw_value(), sep_c.raw_value());
        }
    }

    #[test]
    fn decimal_sinhcosh_identity_tight() {
        // cosh²(x) - sinh²(x) = 1 — decimal identity should hold to within a handful
        // of storage ULP. DecimalFixed<4>::SCALE = 10000, so "1.0000" raw = 10000.
        let inputs = ["0.0", "0.5", "1.0", "-0.5", "-1.0", "2.0"];
        for s in &inputs {
            let x = DecimalFixed::<4>::from_str(s).expect("decimal parse");
            let (sh, ch) = x.sinhcosh();
            let one = DecimalFixed::<4>::from_str("1.0").unwrap();
            let residual = (ch * ch) - (sh * sh) - one;
            assert!(residual.raw_value().abs() <= 8,
                "decimal identity residual too large at {}: {}",
                s, residual.raw_value());
        }
    }

    #[test]
    fn decimal_sinhcosh_zero_is_exact() {
        let zero = DecimalFixed::<4>::from_str("0.0").unwrap();
        let (s, c) = zero.sinhcosh();
        assert_eq!(s.raw_value(), 0, "sinh(0) must be 0");
        // cosh(0) = 1 → raw = SCALE = 10000.
        assert_eq!(c.raw_value(), DecimalFixed::<4>::SCALE,
            "cosh(0) must be 1.0 exactly");
    }
}
