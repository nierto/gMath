//! Narrowing defects fixed in 0.6.4: every narrowing is checked (a loud
//! `Err(TierOverflow)` or a panic, never a plausible wrong value), decimal
//! rounding is half to even wherever it occurs, and compute-tier division
//! rounds to nearest (ties toward +infinity) like every binary result.
//!
//! One regression per defect (gated by storage width where the defect is width
//! specific), the loud paths, and in-range bit identity. Reference values:
//! mpmath at 80 digits (`scripts/generate_narrowing_defects_refs.py`) or exact
//! integer arithmetic. No floats.


macro_rules! shared_items {
    ($S:ty, $F:literal) => {
            #[allow(dead_code)]
            type FixedPoint = Fixed<$S, $F>;
            type ComputeStorage = <$S as g_math::fixed_point::Storage>::Compute;
            type BinaryStorage = $S;
            type FixedVector<T = FixedPoint> = g_math::fixed_point::FixedVector<T>;
            const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
            const FRAC_BITS: u32 = $F;
            use g_math::fixed_point::universal::fasc::stack_evaluator::FascFormat;
            use g_math::fixed_point::domains::decimal_fixed::transcendental::decimal_compute::{decimal_compute_div_int, decimal_compute_from_int, decimal_compute_neg, decimal_compute_to_i128, i128_upscale_to_compute, try_decimal_compute_to_i128, try_i128_upscale_to_compute};
            use g_math::fixed_point::{Fixed, DecimalFixed, OverflowDetected, I512};

            // ============================================================================
            // Item 1: decimal compute -> i128 downscale (checked, half to even)
            // ============================================================================

            fn q256(int: i128) -> I512 {
                I512::from_i128(int) << 256usize
            }
            fn min_raw() -> BinaryStorage {
                <$S as g_math::fixed_point::WideInt>::shl_bits(<$S as g_math::fixed_point::WideInt>::one(), BITS - 1)
            }
    };
}

macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            #![allow(unused_imports, dead_code)]
            shared_items!($S, $F);
            /// e^30 and e^40 at 19 decimals. mpmath: 106864745815244621469904686507414
            /// and 2353852668370199854078999107490348045. Embedded (38 compute decimals)
            /// returned ...042 for e^40 (3 units low) until the 0.6.4 exp rework; every
            /// format is now exact (full-range gate: tests/decimal_exp_range_validation.rs).
            #[test]
            fn decimal_exp_in_range_is_unchanged() {
                if !((BITS == 128 || BITS == 256 || BITS == 512)) { return; }
                assert_eq!(DecimalFixed::<19>::from_integer(30).exp().raw_value(), 106864745815244621469904686507414);
                let r = DecimalFixed::<19>::from_integer(40).exp().raw_value();
                assert_eq!(r, 2353852668370199854078999107490348045);
            }

            /// e^3 at the largest storage dp of the word (mpmath 20.0855369231876677409285...).
            #[test]
            fn decimal_exp_small_is_unchanged() {
                if BITS == 32 {
                assert_eq!(DecimalFixed::<4>::from_integer(3).exp().raw_value(), 200855);
                }
                if BITS == 64 {
                assert_eq!(DecimalFixed::<9>::from_integer(3).exp().raw_value(), 20085536923);
                }
                if (BITS == 128 || BITS == 256 || BITS == 512) {
                assert_eq!(DecimalFixed::<19>::from_integer(3).exp().raw_value(), 200855369231876677409);
                }
            }

            /// e^50 = 5.18e21 fits the compute tier but not i128 at 19 decimals: the
            /// unchecked `as_i128` returned a wrapped value on embedded and wider up to
            /// 0.6.3. The compute word follows D now, so the case is the same on every
            /// build and the infallible boundary reports the ladder's `TierOverflow`.
            #[test]
            #[should_panic(expected = "decimal exp overflow: TierOverflow")]
            fn decimal_exp_beyond_i128_panics() {
                let _ = DecimalFixed::<19>::from_integer(50).exp();
            }

            #[test]
            fn decimal_compute_to_i128_reports_overflow() {
                if !((BITS == 128 || BITS == 256 || BITS == 512)) { return; }
                // 10^10 * 10^10 = 10^20 at compute dp, 10^39 raw at 19 decimals
                let big = g_math::fixed_point::domains::decimal_fixed::transcendental::decimal_compute_mul(
                    decimal_compute_from_int::<ComputeStorage>(10_000_000_000),
                    decimal_compute_from_int::<ComputeStorage>(10_000_000_000),
                );
                assert_eq!(try_decimal_compute_to_i128(big, 19), Err(OverflowDetected::TierOverflow));
                assert_eq!(try_decimal_compute_to_i128(big, 0), Ok(100_000_000_000_000_000_000));
            }

            /// Exact halves at the compute tier round half to even (they rounded away
            /// from zero before 0.6.4: 2.5 -> 3, -2.5 -> -3).
            #[test]
            fn decimal_compute_to_i128_rounds_half_to_even() {
                let half_of = |n: i64| decimal_compute_div_int(decimal_compute_from_int::<ComputeStorage>(n), 2);
                assert_eq!(decimal_compute_to_i128(half_of(5), 0), 2);
                assert_eq!(decimal_compute_to_i128(half_of(7), 0), 4);
                assert_eq!(decimal_compute_to_i128(decimal_compute_neg(half_of(5)), 0), -2);
                assert_eq!(decimal_compute_to_i128(decimal_compute_neg(half_of(7)), 0), -4);
                assert_eq!(decimal_compute_to_i128(half_of(5), 1), 25);
                // not ties: 2.5 + 0.1 and 2.5 - 0.1
                let tenth = decimal_compute_div_int(decimal_compute_from_int::<ComputeStorage>(1), 10);
                assert_eq!(decimal_compute_to_i128(half_of(5) + tenth, 0), 3);
                assert_eq!(decimal_compute_to_i128(half_of(5) - tenth, 0), 2);
            }

            // ============================================================================
            // Item 2: i128 -> decimal compute upscale (checked, half to even)
            // ============================================================================

            /// 20 at 18 decimals is 2e19 raw, past i64: realtime cast it to i64
            /// (2e19 - 2^64) before rescaling, so sqrt(20) came out near 1.2468.
            /// mpmath sqrt(20) = 4.47213595499957939282..., 9 compute decimals.
            #[test]
            fn realtime_decimal_sqrt_of_wide_raw() {
                if !(BITS == 32) { return; }
                let r = DecimalFixed::<18>::from_integer(20).sqrt().raw_value();
                let expected: i128 = 4_472_135_955 * 1_000_000_000;
                assert!((r - expected).abs() <= 1_000_000_000, "sqrt(20) at 18 dp: {r}");
            }

            /// 10^20 at 0 decimals exceeded the realtime (i64 at 9 dp) and compact
            /// (i128 at 19 dp) compute tiers up to 0.6.7; `DecimalFixed` now computes on
            /// the word `D` selects (I256 at 38 digits up to D = 19), which every i128
            /// raw fits, so the engine helper on the narrow word is what still
            /// reports the overflow.
            #[test]
            fn narrow_profile_upscale_reports_overflow() {
                if !((BITS == 32 || BITS == 64)) { return; }
                assert_eq!(
                    try_i128_upscale_to_compute::<ComputeStorage>(100_000_000_000_000_000_000, 0),
                    Err(OverflowDetected::TierOverflow)
                );
            }

            /// A dp one past the compute dp divides by 10, rounding half to even (it
            /// truncated before 0.6.4: 15 -> 1, 16 -> 1).
            #[test]
            fn upscale_past_compute_dp_rounds_half_to_even() {
                let dp = <FixedPoint as FascFormat>::DECIMAL_COMPUTE_DP + 1;
                let unit = |n: i128| i128_upscale_to_compute::<ComputeStorage>(n * 10, dp);
                assert_eq!(i128_upscale_to_compute::<ComputeStorage>(15, dp), unit(2));
                assert_eq!(i128_upscale_to_compute::<ComputeStorage>(25, dp), unit(2));
                assert_eq!(i128_upscale_to_compute::<ComputeStorage>(16, dp), unit(2));
                assert_eq!(i128_upscale_to_compute::<ComputeStorage>(14, dp), unit(1));
                assert_eq!(i128_upscale_to_compute::<ComputeStorage>(-15, dp), unit(-2));
                assert_eq!(i128_upscale_to_compute::<ComputeStorage>(-16, dp), unit(-2));
                // beyond 38 extra digits every i128 rounds to zero
                assert_eq!(i128_upscale_to_compute::<ComputeStorage>(i128::MAX, <FixedPoint as FascFormat>::DECIMAL_COMPUTE_DP + 39), unit(0));
                assert_eq!(try_i128_upscale_to_compute::<ComputeStorage>(-15, dp), Ok(unit(-2)));
            }

            // ============================================================================
            // Item 3: DecimalFixed <-> Q256.256 (independent of the format)
            // ============================================================================


            #[test]
            fn from_binary_q256_reports_overflow() {
                // 2^44 at 30 decimals = 1.76e43 raw: wrapped through the unchecked as_i128
                let big = I512::from_i128(1) << 300usize;
                assert_eq!(DecimalFixed::<30>::try_from_binary_q256(big), Err(OverflowDetected::TierOverflow));
                assert_eq!(DecimalFixed::<30>::try_from_binary_q256(-big), Err(OverflowDetected::TierOverflow));
                // I512::MIN: `-q256` wrapped to itself and lost the sign
                assert_eq!(DecimalFixed::<0>::try_from_binary_q256(I512::min_value()), Err(OverflowDetected::TierOverflow));
                // the largest magnitudes that fit
                assert_eq!(DecimalFixed::<0>::try_from_binary_q256(q256(i128::MAX)).unwrap().raw_value(), i128::MAX);
                assert_eq!(DecimalFixed::<0>::try_from_binary_q256(q256(i128::MIN)).unwrap().raw_value(), i128::MIN);
            }

            #[test]
            #[should_panic(expected = "DecimalFixed::from_binary_q256: value outside the i128 range")]
            fn from_binary_q256_beyond_i128_panics() {
                let _ = DecimalFixed::<30>::from_binary_q256(I512::from_i128(1) << 300usize);
            }

            /// Exact halves round half to even (half away from zero before 0.6.4).
            #[test]
            fn from_binary_q256_rounds_half_to_even() {
                let half = I512::from_i128(1) << 255usize;
                let at0 = |v: I512| DecimalFixed::<0>::from_binary_q256(v).raw_value();
                assert_eq!(at0(half), 0);
                assert_eq!(at0(q256(1) + half), 2);
                assert_eq!(at0(q256(2) + half), 2);
                assert_eq!(at0(-half), 0);
                assert_eq!(at0(-(q256(1) + half)), -2);
                // not ties: one quantum either side
                assert_eq!(at0(half + I512::from_i128(1)), 1);
                assert_eq!(at0(half - I512::from_i128(1)), 0);
                // 1.5 at 2 decimals, and -1.5
                assert_eq!(DecimalFixed::<2>::from_binary_q256(q256(1) + half).raw_value(), 150);
                assert_eq!(DecimalFixed::<2>::from_binary_q256(-(q256(1) + half)).raw_value(), -150);
            }

            #[test]
            fn to_binary_q256_at_the_raw_minimum() {
                // i128::MIN.abs() overflowed before 0.6.4
                let v = DecimalFixed::<0>::from_raw(i128::MIN).to_binary_q256();
                assert_eq!(v, q256(i128::MIN));
                assert_eq!(DecimalFixed::<0>::from_binary_q256(v).raw_value(), i128::MIN);
                assert_eq!(DecimalFixed::<2>::from_raw(-150).to_binary_q256(), -(q256(1) + (I512::from_i128(1) << 255usize)));
            }

            // ============================================================================
            // Item 4: construction and accessors
            // ============================================================================

            #[test]
            fn from_integer_reports_overflow() {
                assert_eq!(DecimalFixed::<20>::try_from_integer(i64::MAX), Err(OverflowDetected::TierOverflow));
                assert_eq!(DecimalFixed::<20>::try_from_integer(1_000_000).unwrap().raw_value(), 1_000_000 * 10i128.pow(20));
                assert_eq!(DecimalFixed::<2>::from_integer(i64::MIN).raw_value(), (i64::MIN as i128) * 100);
            }

            #[test]
            #[should_panic(expected = "DecimalFixed::from_integer: value outside the i128 range")]
            fn from_integer_beyond_i128_panics() {
                let _ = DecimalFixed::<20>::from_integer(i64::MAX);
            }

            #[test]
            #[should_panic(expected = "DecimalFixed::from_parts: value outside the i128 range")]
            fn from_parts_beyond_i128_panics() {
                let _ = DecimalFixed::<20>::from_parts(i64::MAX, 0);
            }

            /// An oversized fraction was clamped to 10^DECIMALS - 1 (1.99 here).
            #[test]
            #[should_panic(expected = "DecimalFixed::from_parts: fractional part must be below 10^DECIMALS")]
            fn from_parts_oversized_fraction_panics() {
                let _ = DecimalFixed::<2>::from_parts(1, 100);
            }

            #[test]
            fn from_parts_in_range_is_unchanged() {
                assert_eq!(DecimalFixed::<2>::from_parts(19, 99).raw_value(), 1999);
                assert_eq!(DecimalFixed::<2>::from_parts(-19, 99).raw_value(), -1999);
                assert_eq!(DecimalFixed::<20>::from_parts(1, 5).raw_value(), 10i128.pow(20) + 5);
            }

            #[test]
            #[should_panic(expected = "DecimalFixed::integer_part: integer part outside the i64 range")]
            fn integer_part_beyond_i64_panics() {
                let _ = DecimalFixed::<0>::from_raw(1i128 << 100).integer_part();
            }

            #[test]
            #[should_panic(expected = "DecimalFixed::fractional_part: fractional digits outside the u64 range")]
            fn fractional_part_beyond_u64_panics() {
                let _ = "0.1234567890123456789012345".parse::<DecimalFixed<25>>().unwrap().fractional_part();
            }

            #[test]
            fn display_at_full_width() {
                // integer part past i64 (was cast to i64)
                assert_eq!(DecimalFixed::<0>::from_raw(1i128 << 100).to_string(), "1267650600228229401496703205376");
                assert_eq!(DecimalFixed::<0>::from_raw(i128::MIN).to_string(), "-170141183460469231731687303715884105728");
                // fraction past u64 (was cast to u64)
                let v: DecimalFixed<25> = "0.1234567890123456789012345".parse().unwrap();
                assert_eq!(v.raw_value(), 1234567890123456789012345);
                assert_eq!(v.to_string(), "0.1234567890123456789012345");
                // a value in (-1, 0) lost its sign
                assert_eq!(DecimalFixed::<1>::from_raw(-5).to_string(), "-0.5");
                assert_eq!(DecimalFixed::<2>::from_raw(-1999).to_string(), "-19.99");
                assert_eq!(DecimalFixed::<2>::from_raw(7).to_string(), "0.07");
            }

            #[test]
            fn parser_is_exact_and_strict() {
                type D2 = DecimalFixed<2>;
                type D20 = DecimalFixed<20>;
                use g_math::fixed_point::domains::decimal_fixed::ParseError;
                assert_eq!("-12.34".parse::<D2>().unwrap().raw_value(), -1234);
                assert_eq!("+12.3".parse::<D2>().unwrap().raw_value(), 1230);
                assert_eq!("-0.5".parse::<D2>().unwrap().raw_value(), -50);
                assert_eq!("5.".parse::<D2>().unwrap().raw_value(), 500);
                // a second sign was accepted: "--5" gave 5, "1.+5" gave 1.50
                assert_eq!("--5".parse::<D2>(), Err(ParseError::InvalidFormat));
                assert_eq!("-+5".parse::<D2>(), Err(ParseError::InvalidFormat));
                assert_eq!("1.+5".parse::<D2>(), Err(ParseError::InvalidFormat));
                assert_eq!("1.234".parse::<D2>(), Err(ParseError::TooManyDecimals));
                // integers past i64 parse where the value fits (were InvalidFormat)
                assert_eq!("10000000000000000000000".parse::<D2>().unwrap().raw_value(), 10i128.pow(24));
                // fractions of 20+ digits parse (were InvalidFormat through u64)
                assert_eq!("1.99999999999999999999".parse::<D20>().unwrap().raw_value(), 2 * 10i128.pow(20) - 1);
                // the extremes of the i128 range, then one past (fraction overflow wrapped)
                assert_eq!(
                    "1701411834604692317.31687303715884105727".parse::<D20>().unwrap().raw_value(),
                    i128::MAX
                );
                assert_eq!(
                    "1701411834604692317.31687303715884105728".parse::<D20>(),
                    Err(ParseError::Overflow)
                );
                assert_eq!("1701411834604692318".parse::<D20>(), Err(ParseError::Overflow));
            }

            // ============================================================================
            // Item 5: DecimalFixed operators are loud, try_ twins
            // ============================================================================

            #[test]
            fn decimal_try_twins_report_overflow() {
                type D = DecimalFixed<2>;
                let max = D::from_raw(i128::MAX);
                let min = D::from_raw(i128::MIN);
                let one = D::ONE;
                assert_eq!(max.try_add(D::from_raw(1)), Err(OverflowDetected::TierOverflow));
                assert_eq!(min.try_sub(D::from_raw(1)), Err(OverflowDetected::TierOverflow));
                assert_eq!(min.try_neg(), Err(OverflowDetected::TierOverflow));
                assert_eq!(max.try_mul(D::from_integer(2)), Err(OverflowDetected::TierOverflow));
                assert_eq!(one.try_div(D::ZERO), Err(OverflowDetected::DivisionByZero));
                assert_eq!(max.try_div(D::from_raw(1)), Err(OverflowDetected::TierOverflow));
                // exact boundary results fit
                assert_eq!(max.try_mul(one).unwrap(), max);
                assert_eq!(min.try_mul(one).unwrap(), min);
                assert_eq!(min.try_div(one).unwrap(), min);
                assert_eq!(D::from_raw(i128::MIN).try_div(D::from_integer(-1)), Err(OverflowDetected::TierOverflow));
                // the raw minimum as divisor or dividend (its i128 magnitude overflowed)
                type D0 = DecimalFixed<0>;
                assert_eq!(D0::from_raw(i128::MIN).try_div(D0::from_integer(2)).unwrap().raw_value(), i128::MIN / 2);
                assert_eq!(D0::from_integer(5).try_div(D0::from_raw(i128::MIN)).unwrap().raw_value(), 0);
                assert_eq!(D0::from_raw(i128::MIN).try_div(D0::from_raw(i128::MIN)).unwrap().raw_value(), 1);
            }

            #[test]
            #[should_panic(expected = "DecimalFixed: division by zero")]
            fn decimal_division_by_zero_panics() {
                let _ = DecimalFixed::<2>::ONE / DecimalFixed::<2>::ZERO;
            }

            #[test]
            #[should_panic(expected = "DecimalFixed: multiplication overflow")]
            fn decimal_multiplication_overflow_panics() {
                // 10^10 * 10^10 = 10^20 > i128::MAX / 10^19 (it saturated to i128::MAX)
                let a = DecimalFixed::<19>::from_integer(10_000_000_000);
                let _ = a * a;
            }

            #[test]
            #[should_panic(expected = "DecimalFixed: division overflow")]
            fn decimal_division_overflow_panics() {
                // 1000 / 10^-18 = 10^21 > i128::MAX / 10^18
                let _ = DecimalFixed::<18>::from_integer(1000) / DecimalFixed::<18>::from_raw(1);
            }

            #[test]
            #[should_panic(expected = "DecimalFixed: addition overflow")]
            fn decimal_addition_overflow_panics() {
                let _ = DecimalFixed::<2>::from_raw(i128::MAX) + DecimalFixed::<2>::from_raw(1);
            }

            /// In range, the operators and the try_ twins agree, banker's ties included,
            /// on both the narrow i128 path and the 256-bit path. Exact integer model:
            /// round-half-even of the exact rational, computed on magnitudes in i128 or
            /// u128 where it fits.
            #[test]
            fn decimal_in_range_bit_identity() {
                type D = DecimalFixed<2>;
                let t = |a: i128, b: i128| (D::from_raw(a), D::from_raw(b));
                // ties: 0.05 * 0.5 = 0.025 -> 0.02, 0.15 * 0.5 = 0.075 -> 0.08
                let (a, b) = t(5, 50);
                assert_eq!((a * b).raw_value(), 2);
                let (a, b) = t(15, 50);
                assert_eq!((a * b).raw_value(), 8);
                let (a, b) = t(-15, 50);
                assert_eq!((a * b).raw_value(), -8);
                // 0.01 / 0.08 = 0.125 -> 0.12; 0.03 / 0.08 = 0.375 -> 0.38
                let (a, b) = t(1, 8);
                assert_eq!((a / b).raw_value(), 12);
                let (a, b) = t(3, 8);
                assert_eq!((a / b).raw_value(), 38);
                let (a, b) = t(-3, 8);
                assert_eq!((a / b).raw_value(), -38);
                // 256-bit path (raw product past i128): 2^125 raw * 1.5
                let x = D::from_raw(1i128 << 125);
                let h = D::from_raw(150);
                assert_eq!((x * h).raw_value(), (1i128 << 125) + (1i128 << 124));
                assert_eq!(x.try_mul(h).unwrap(), x * h);
                // 256-bit division: 2^125 raw / 3.00 = nearest of 2^125 / 3 (fraction 2/3)
                let q = (x / D::from_raw(300)).raw_value();
                assert_eq!(q, (1i128 << 125) / 3 + 1);
                assert_eq!((-x / D::from_raw(300)).raw_value(), -((1i128 << 125) / 3 + 1));
                // 256-bit exact ties: (2^125 + 1) raw * 0.5 = 2^124 + 0.5 raw -> even 2^124
                let half = D::from_raw(50);
                assert_eq!((D::from_raw((1i128 << 125) + 1) * half).raw_value(), 1i128 << 124);
                let odd3 = D::from_raw((1i128 << 125) + 3);
                assert_eq!((odd3 * half).raw_value(), (1i128 << 124) + 2);
                assert_eq!(((-odd3) * half).raw_value(), -((1i128 << 124) + 2));
                for (a, b) in [(1999i128, 500i128), (-1234, 77), (5, -3), (1 << 90, 12345)] {
                    let (x, y) = t(a, b);
                    assert_eq!(x.try_add(y).unwrap(), x + y);
                    assert_eq!(x.try_sub(y).unwrap(), x - y);
                    assert_eq!(x.try_mul(y).unwrap(), x * y);
                    assert_eq!(x.try_div(y).unwrap(), x / y);
                    assert_eq!(x.try_neg().unwrap(), -x);
                }
            }

            // ============================================================================
            // Item 6: FixedPoint::to_int
            // ============================================================================

            #[test]
            fn to_int_in_range_is_unchanged() {
                assert_eq!(FixedPoint::from_str("2.5").to_int(), 2);
                assert_eq!(FixedPoint::from_str("-2.5").to_int(), -3);
                assert_eq!(FixedPoint::from_str("-2").to_int(), -2);
                assert_eq!(FixedPoint::from_str("0.25").try_to_int(), Ok(0));
                assert_eq!(FixedPoint::from_str("-0.25").try_to_int(), Ok(-1));
            }

            /// |x| >= 2^31 fits storage on embedded and wider; `as i32` wrapped
            /// (3000000000 gave -1294967296).
            #[test]
            fn to_int_beyond_i32_reports_overflow() {
                if !((BITS == 128 || BITS == 256 || BITS == 512)) { return; }
                assert_eq!(FixedPoint::from_str("3000000000").try_to_int(), Err(OverflowDetected::TierOverflow));
                assert_eq!(FixedPoint::from_str("-2147483648.5").try_to_int(), Err(OverflowDetected::TierOverflow));
                assert_eq!(FixedPoint::from_str("2147483648").try_to_int(), Err(OverflowDetected::TierOverflow));
                assert_eq!(FixedPoint::from_str("-2147483648").try_to_int(), Ok(i32::MIN));
                assert_eq!(FixedPoint::from_str("2147483647.75").try_to_int(), Ok(i32::MAX));
            }

            /// On scientific the floor has 256 integer bits: 2^200 + 5 was reduced
            /// mod 2^128 and then mod 2^32 (giving 5).
            #[test]
            fn to_int_scientific_high_bits() {
                if !(BITS == 512) { return; }
                let v = FixedPoint::from_str("1606938044258990275541962092341162602522202993782792835301381");
                assert_eq!(v.try_to_int(), Err(OverflowDetected::TierOverflow));
            }

            #[test]
            #[should_panic(expected = "Fixed::to_int: integer part outside the i32 range")]
            fn to_int_beyond_i32_panics() {
                if !((BITS == 128 || BITS == 256 || BITS == 512)) { panic!("Fixed::to_int: integer part outside the i32 range (not this format)"); }
                let _ = FixedPoint::from_str("3000000000").to_int();
            }

            // ============================================================================
            // Item 7: compute-tier division rounds to nearest, ties toward +infinity
            // ============================================================================


            // ============================================================================
            // Item 8: compute-tier dot at the storage minimum
            // ============================================================================

            /// `-x` of the storage minimum is itself; the unsigned `mul_to_*` reads that
            /// bit pattern as 2^(W-1), the true magnitude, so the product is exact.
            #[test]
            fn dot_at_the_storage_minimum() {
                let min = FixedPoint::from_raw(min_raw());
                let one = FixedPoint::from_int(1);
                let half = FixedPoint::from_str("0.5");
                let v = |x: &[FixedPoint]| FixedVector::from_slice(x);
                assert_eq!(v(&[min]).dot_precise(&v(&[one])), min);
                assert_eq!(v(&[one]).dot_precise(&v(&[min])), min);
                assert_eq!(v(&[min, one]).dot_precise(&v(&[half, one])).raw(), (min.try_div(FixedPoint::from_int(2)).unwrap() + one).raw());
                assert_eq!(v(&[min, min]).dot_precise(&v(&[half, -half])), FixedPoint::ZERO);
            }

            #[test]
            #[should_panic]
            fn dot_at_the_storage_minimum_overflowing() {
                let min = FixedPoint::from_raw(min_raw());
                let neg_one = -FixedPoint::from_int(1);
                let _ = FixedVector::from_slice(&[min]).dot_precise(&FixedVector::from_slice(&[neg_one]));
            }
        }
    };
}

// The part of the suite written for a subset of the words.
macro_rules! narrow_suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            #![allow(unused_imports, dead_code)]
            shared_items!($S, $F);
            /// sigmoid(x >= 0) = one / (one + e^-x) and sigmoid(x < 0) = e / (one + e)
            /// are single compute-tier quotients; the expected raw is the exact nearest
            /// (ties toward +infinity) of the integer quotient. Before 0.6.4 it was the
            /// truncated quotient, one unit low on about half the inputs.
            #[test]
            fn compute_divide_rounds_to_nearest() {
                use g_math::compute_tier::{exp, one, sigmoid};
                let compute_frac_bits = 2 * FRAC_BITS;
                let nearest = |n: i128, d: i128| (2 * n + d).div_euclid(2 * d);
                let one = one::<FixedPoint>();
                let mut differs = 0;
                for k in 0..400i64 {
                    // x from 0 to about 8 in uneven steps
                    let x = k * (one / 50) + k * k;
                    for x in [x, -x] {
                        let (n, d) = if x >= 0 {
                            let e = exp::<FixedPoint>(-x) as i128;
                            ((one as i128) << compute_frac_bits, one as i128 + e)
                        } else {
                            let e = exp::<FixedPoint>(x) as i128;
                            (e << compute_frac_bits, one as i128 + e)
                        };
                        let expected = nearest(n, d);
                        assert_eq!(sigmoid::<FixedPoint>(x) as i128, expected, "sigmoid raw {x}");
                        if expected != n / d { differs += 1; }
                    }
                }
                assert!(differs > 100, "the gate must see truncation and nearest disagree ({differs})");
            }
        }
    };
}

for_each_format!(suite);
narrow_suite!(narrow_suite_q22_10, i32, 10);
narrow_suite!(narrow_suite_q16_16, i32, 16);

