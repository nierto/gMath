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

#[macro_use]
mod common;

macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            type FixedPoint = Fixed<$S, $F>;
            type LazyExpr<T = FixedPoint> = g_math::canonical::LazyExpr<T>;
            #[allow(dead_code)]
            const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
            #[allow(dead_code)]
            const FRAC_BITS: u32 = $F;
            use g_math::fixed_point::Fixed;
            use g_math::canonical::{evaluate_in, gmath_in};

            fn disp(e: &LazyExpr) -> String {
                format!("{}", evaluate_in(e).expect("evaluate"))
            }

            #[test]
            fn odd_even_symmetries_bit_exact() {
                for x in ["0.25", "0.5", "1", "1.5", "2.5", "7.75"] {
                    assert_eq!(disp(&gmath_in::<FixedPoint>(x).sin()), disp(&(-(-gmath_in::<FixedPoint>(x)).sin())), "sin odd at {x}");
                    assert_eq!(disp(&gmath_in::<FixedPoint>(x).cos()), disp(&(-gmath_in::<FixedPoint>(x)).cos()), "cos even at {x}");
                    assert_eq!(disp(&gmath_in::<FixedPoint>(x).atan()), disp(&(-(-gmath_in::<FixedPoint>(x)).atan())), "atan odd at {x}");
                    assert_eq!(disp(&gmath_in::<FixedPoint>(x).tanh()), disp(&(-(-gmath_in::<FixedPoint>(x)).tanh())), "tanh odd at {x}");
                    assert_eq!(disp(&gmath_in::<FixedPoint>(x).sinh()), disp(&(-(-gmath_in::<FixedPoint>(x)).sinh())), "sinh odd at {x}");
                    assert_eq!(disp(&gmath_in::<FixedPoint>(x).asinh()), disp(&(-(-gmath_in::<FixedPoint>(x)).asinh())), "asinh odd at {x}");
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
                    // 1 raw ulp of the format
                    let one_ulp = { let unit = { let half = FixedPoint::from_str("0.5"); let mut u = FixedPoint::one(); for _ in 0..FRAC_BITS { u = u * half; } u }; unit }.raw();
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
                    let prod = disp(&(gmath_in::<FixedPoint>(x).sin() * gmath_in::<FixedPoint>(x).sin()));
                    assert!(
                        !prod.starts_with('-'),
                        "sin({x})² must be non-negative, got {prod}"
                    );
                    // (−sin)·(−sin) == sin·sin, bit-exact through the router.
                    let prod_neg = disp(&((-gmath_in::<FixedPoint>(x).sin()) * (-gmath_in::<FixedPoint>(x).sin())));
                    assert_eq!(prod, prod_neg, "(-a)(-a) != a·a for a = sin({x})");
                }
            }
        }
    };
}

for_each_format!(suite);
