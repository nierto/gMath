//! 0.5.0 item 0b — negative-operand battery for the unsigned widening-
//! multiply audit (docs/design/ROUNDING_CENSUS.md appendix).
//!
//! The I256/I512/I1024 widening multiplies are unsigned by convention;
//! call sites either sign-wrap or are positive-by-construction (now
//! debug_assert-enforced at every such site). This battery drives
//! negative values through the transcendental compute chains — the paths
//! where a violated invariant would corrupt sign extension — and pins
//! the odd/even symmetries that corruption would break first. Most
//! valuable on q128_128/q256_256 (the wide-kernel profiles), runs on all.

use g_math::canonical::{evaluate, gmath, LazyExpr};
use g_math::fixed_point::FixedPoint;

fn disp(e: &LazyExpr) -> String {
    format!("{}", evaluate(e).expect("evaluate"))
}

#[test]
fn odd_even_symmetries_bit_exact() {
    for x in ["0.25", "0.5", "1", "1.5", "2.5", "7.75"] {
        assert_eq!(disp(&gmath(x).sin()), disp(&(-(-gmath(x)).sin())), "sin odd at {x}");
        assert_eq!(disp(&gmath(x).cos()), disp(&(-gmath(x)).cos()), "cos even at {x}");
        assert_eq!(disp(&gmath(x).atan()), disp(&(-(-gmath(x)).atan())), "atan odd at {x}");
        assert_eq!(disp(&gmath(x).tanh()), disp(&(-(-gmath(x)).tanh())), "tanh odd at {x}");
        assert_eq!(disp(&gmath(x).sinh()), disp(&(-(-gmath(x)).sinh())), "sinh odd at {x}");
        assert_eq!(disp(&gmath(x).asinh()), disp(&(-(-gmath(x)).asinh())), "asinh odd at {x}");
    }
}

#[test]
fn negative_intermediate_chains_roundtrip() {
    // x < 1 makes ln(x) NEGATIVE — exp(ln(x)) pushes a negative value
    // through the exp compute chain (the audit's highest-value path).
    for s in ["0.125", "0.3", "0.5", "0.9"] {
        let x = FixedPoint::from_str(s);
        let rt = x.ln().exp();
        let diff = (rt - x).abs().raw();
        let one_ulp = FixedPoint::from_raw({
            // 1 raw ulp in the profile's storage
            #[cfg(any(table_format = "q128_128"))]
            { g_math::fixed_point::I256::from_i128(1) }
            #[cfg(any(table_format = "q256_256"))]
            { g_math::fixed_point::I512::from_i128(1) }
            #[cfg(not(any(table_format = "q128_128", table_format = "q256_256")))]
            { 1 }
        })
        .raw();
        assert!(diff <= one_ulp, "exp(ln({s})) drifted beyond 1 ulp");
    }
    // Negative base through sinh/cosh/tanh chains (exp(-x) internally).
    for s in ["0.5", "2", "5"] {
        let x = FixedPoint::from_str(s);
        let (sh, ch) = (-x).sinhcosh();
        let (shp, chp) = x.sinhcosh();
        assert_eq!(sh.raw(), (-shp).raw(), "sinh(-x) != -sinh(x) at {s}");
        assert_eq!(ch.raw(), chp.raw(), "cosh(-x) != cosh(x) at {s}");
    }
}

#[test]
fn negative_fasc_compute_chain_products() {
    // sin(x)·sin(x) with x chosen so sin < 0: the FASC multiply of two
    // negative BinaryCompute intermediates exercises compute_multiply's
    // wide signed path (q256_256: multiply_i1024_q512_512, the sign-safe
    // one — its unsigned shadow was renamed in this audit).
    for x in ["4", "5", "10"] {
        // sin(4), sin(5) < 0; product must be positive and equal sin²|computed once
        let prod = disp(&(gmath(x).sin() * gmath(x).sin()));
        assert!(
            !prod.starts_with('-'),
            "sin({x})² must be non-negative, got {prod}"
        );
        // (−sin)·(−sin) == sin·sin, bit-exact through the router.
        let prod_neg = disp(&((-gmath(x).sin()) * (-gmath(x).sin())));
        assert_eq!(prod, prod_neg, "(-a)(-a) != a·a for a = sin({x})");
    }
}
