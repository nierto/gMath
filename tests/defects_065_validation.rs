//! Defects fixed in 0.6.5: one regression per defect, the loud path, and the
//! in-range value where the fix must not move it.
//!
//! - `rms_norm_factor` / `rms_norm_factor_eps_wide` returned `Ok(0)` for a
//!   negative radicand.
//! - The serialization tag did not record the realtime fractional split.
//! - The top of the decimal UGOD ladder kept the low words of an oversized
//!   product or quotient.
//! - `Currency` / `HighPrecisionCurrency` were documented public but
//!   reachable only through a hidden path.
//! - A whitespace-only decimal string reported `InvalidFormat`.
//!
//! References are exact integer arithmetic. No floats.


macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            type FixedPoint = Fixed<$S, $F>;
            #[allow(dead_code)]
            const FRAC_BITS: u32 = $F;
            #[allow(dead_code)]
            const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
            use g_math::fixed_point::domains::decimal_fixed::{DecimalRaw, UniversalDecimalTiered};
            use g_math::fixed_point::imperative::fused;
            use g_math::fixed_point::{Fixed, Currency, DecimalFixed, HighPrecisionCurrency, OverflowDetected, I256, I512};

            fn fp(s: &str) -> FixedPoint {
                if let Some(rest) = s.strip_prefix('-') { -FixedPoint::from_str(rest) } else { FixedPoint::from_str(s) }
            }

            // ============================================================================
            // rms_norm_factor: negative radicand is a domain error
            // ============================================================================

            // Values stay inside [-2, 2) so the tests hold at every realtime split.
            #[test]
            fn rms_negative_radicand_is_a_domain_error() {
                let zeros = [FixedPoint::ZERO; 4];
                // eps = -1.0 in Q64.64
                assert_eq!(fused::rms_norm_factor_eps_wide(&zeros, -(1i128 << 64)), Err(OverflowDetected::DomainError));
                assert_eq!(fused::rms_norm_factor(&zeros, fp("-1")), Err(OverflowDetected::DomainError));
                // mean(x^2) = 1, eps = -1.5: still negative
                let ones = [fp("1"); 4];
                assert_eq!(fused::rms_norm_factor(&ones, fp("-1.5")), Err(OverflowDetected::DomainError));
            }

            #[test]
            fn rms_negative_eps_with_positive_radicand_is_a_value() {
                // mean(x^2) = 2.25, eps = -1.25: 1/sqrt(1) = 1 exactly
                assert_eq!(fused::rms_norm_factor(&[fp("1.5")], fp("-1.25")), Ok(fp("1")));
                // unchanged: mean(x^2) = 1, eps = 0: 1
                assert_eq!(fused::rms_norm_factor(&[fp("1"); 4], FixedPoint::ZERO), Ok(fp("1")));
                // unchanged: zero radicand is DivisionByZero
                assert_eq!(fused::rms_norm_factor(&[FixedPoint::ZERO; 3], FixedPoint::ZERO), Err(OverflowDetected::DivisionByZero));
            }

            // ============================================================================
            // Serialization tag names the realtime split
            // ============================================================================

            #[test]
            fn profile_tag_names_the_split() {
                let tag = FixedPoint::legacy_tag();
                if BITS == 128 { assert_eq!(tag, 0x01); }
                if BITS == 256 { assert_eq!(tag, 0x02); }
                if BITS == 512 { assert_eq!(tag, 0x03); }
                if BITS == 64 { assert_eq!(tag, 0x04); }
                if BITS == 32 {
                    let f = FRAC_BITS;
                    if f == 16 {
                        assert_eq!(tag, 0x05);
                    } else {
                        assert_eq!(tag, 0x80 | f as u8);
                    }
                }
                // since 0.7.0 the written tag is the two-byte format tag; the 0.6.x
                // tag is what the legacy reader accepts
                let x = fp("1.5");
                let bytes = x.to_bytes();
                assert_eq!([bytes[0], bytes[1], bytes[2]], FixedPoint::format_tag());
                assert_eq!(FixedPoint::from_bytes(&bytes), Ok(x));
                let mut old = vec![tag];
                old.extend_from_slice(&bytes[3..]);
                assert_eq!(FixedPoint::from_bytes_legacy(&old), Ok(x));
            }

            /// A non-default realtime split refuses another split's tag instead of
            /// reading the payload at the wrong scale, through both readers; the
            /// explicit raw read still works.
            #[test]
            fn non_default_split_refuses_q16_16_bytes() {
                if !(BITS == 32) { return; }
                let x = fp("1.5");
                let bytes = x.to_bytes();
                let f = FRAC_BITS;
                let mut other = bytes.clone();
                other[2] = if f == 16 { 10 } else { 16 };
                assert_eq!(FixedPoint::from_bytes(&other), Err(OverflowDetected::InvalidInput));
                let mut legacy_other = vec![0x80 | if f == 16 { 10u8 } else { 16u8 }];
                legacy_other.extend_from_slice(&bytes[3..]);
                assert_eq!(FixedPoint::from_bytes_legacy(&legacy_other), Err(OverflowDetected::InvalidInput));
                // 0x05 was written by every split up to 0.6.4: taken for the reader's split
                let mut legacy_default = vec![0x05];
                legacy_default.extend_from_slice(&bytes[3..]);
                assert_eq!(FixedPoint::from_bytes_legacy(&legacy_default), Ok(x));
                assert_eq!(FixedPoint::from_raw_bytes(&bytes[3..]), Ok(x));
            }

            // ============================================================================
            // Decimal UGOD ladder top
            // ============================================================================

            fn large(raw: (u8, DecimalRaw)) -> (u8, I512) {
                match raw {
                    (t, DecimalRaw::Large(v)) => (t, v),
                    (t, other) => panic!("expected a tier-6 value, got tier {t}: {other:?}"),
                }
            }

            #[test]
            fn decimal_tier6_product_past_i512_is_tier_overflow() {
                let big = UniversalDecimalTiered::from_tier_raw(6, 0, DecimalRaw::Large(I512::from_i128(1) << 300usize)).unwrap();
                // 2^600 does not fit I512 (0.6.4 returned its low 512 bits: zero)
                assert_eq!(big.multiply(&big).err(), Some(OverflowDetected::TierOverflow));

                // in range: 2^200 * 2^200 = 2^400
                let ok = UniversalDecimalTiered::from_tier_raw(6, 0, DecimalRaw::Large(I512::from_i128(1) << 200usize)).unwrap();
                let (tier, v) = large(ok.multiply(&ok).unwrap().to_tier_raw());
                assert_eq!(tier, 6);
                assert!(v == I512::from_i128(1) << 400usize);
            }

            #[test]
            fn decimal_tier5_quotient_past_i256_promotes() {
                // (2^250 at 30 dp) / (1e-30): the quotient raw is 2^250 * 10^30, past I256
                let a = UniversalDecimalTiered::from_tier_raw(5, 30, DecimalRaw::Medium(I256::from_i128(1) << 250usize)).unwrap();
                let b = UniversalDecimalTiered::from_tier_raw(5, 30, DecimalRaw::Small(1)).unwrap();
                let (tier, v) = large(a.divide(&b).unwrap().to_tier_raw());
                assert_eq!(tier, 6);
                let expected = (I512::from_i128(1) << 250usize) * I512::from_i128(10i128.pow(30));
                assert!(v == expected);
            }

            #[test]
            fn decimal_tier6_quotient_past_i512_is_tier_overflow() {
                // (2^500 at 38 dp) / (1e-38): 2^500 * 10^38 does not fit I512
                let a = UniversalDecimalTiered::from_tier_raw(6, 38, DecimalRaw::Large(I512::from_i128(1) << 500usize)).unwrap();
                let b = UniversalDecimalTiered::from_tier_raw(6, 38, DecimalRaw::Small(1)).unwrap();
                assert_eq!(a.divide(&b).err(), Some(OverflowDetected::TierOverflow));

                // in range: dividing by 1.0 (raw 10^38 at 38 dp) returns the raw unchanged
                let a = UniversalDecimalTiered::from_tier_raw(6, 38, DecimalRaw::Large(I512::from_i128(1) << 300usize)).unwrap();
                let one = UniversalDecimalTiered::from_tier_raw(6, 38, DecimalRaw::Small(10i128.pow(38))).unwrap();
                let (tier, v) = large(a.divide(&one).unwrap().to_tier_raw());
                assert_eq!(tier, 6);
                assert!(v == I512::from_i128(1) << 300usize);
            }

            // ============================================================================
            // Money aliases and the decimal parser
            // ============================================================================

            #[test]
            fn currency_aliases_are_public() {
                let price: Currency = DecimalFixed::<2>::from_raw(1999);
                assert_eq!(price.to_string(), "19.99");
                let rate: HighPrecisionCurrency = DecimalFixed::<6>::from_raw(-70_000);
                assert_eq!(rate.to_string(), "-0.070000");
            }

            #[test]
            fn whitespace_only_decimal_string_is_empty() {
                use g_math::fixed_point::domains::decimal_fixed::ParseError;
                assert!(matches!(Currency::from_decimal_str_decimal(""), Err(ParseError::EmptyString)));
                assert!(matches!(Currency::from_decimal_str_decimal("   "), Err(ParseError::EmptyString)));
                assert_eq!(Currency::from_decimal_str_decimal(" 19.99 ").unwrap().raw_value(), 1999);
            }
        }
    };
}

for_each_format!(suite);

