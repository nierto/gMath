//! Defects fixed in 0.6.5: one regression per defect, the loud path, and the
//! in-range value where the fix must not move it.
//!
//! - tq19 narrowing wrapped (`as` casts) where every other infallible
//!   narrowing in the crate panics; the AVX2 trit kernel negated `i32::MIN`
//!   in 32 bits.
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

use g_math::fixed_point::domains::decimal_fixed::{DecimalRaw, UniversalDecimalTiered};
use g_math::fixed_point::imperative::fused;
use g_math::fixed_point::{Currency, DecimalFixed, FixedPoint, HighPrecisionCurrency, OverflowDetected, I256, I512};

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
    let tag = FixedPoint::profile_tag();
    #[cfg(table_format = "q64_64")]
    assert_eq!(tag, 0x01);
    #[cfg(table_format = "q128_128")]
    assert_eq!(tag, 0x02);
    #[cfg(table_format = "q256_256")]
    assert_eq!(tag, 0x03);
    #[cfg(table_format = "q32_32")]
    assert_eq!(tag, 0x04);
    #[cfg(table_format = "q16_16")]
    {
        let f = g_math::fixed_point::frac_config::FRAC_BITS;
        if f == 16 {
            assert_eq!(tag, 0x05);
        } else {
            assert_eq!(tag, 0x80 | f as u8);
        }
    }
    let x = fp("1.5");
    let bytes = x.to_bytes();
    assert_eq!(bytes[0], tag);
    assert_eq!(FixedPoint::from_bytes(&bytes), Ok(x));
}

/// A non-default realtime split refuses the Q16.16 tag instead of reading the
/// payload at the wrong scale; the explicit raw read still works.
#[cfg(table_format = "q16_16")]
#[test]
fn non_default_split_refuses_q16_16_bytes() {
    let x = fp("1.5");
    let mut bytes = x.to_bytes();
    if g_math::fixed_point::frac_config::FRAC_BITS != 16 {
        bytes[0] = 0x05;
        assert_eq!(FixedPoint::from_bytes(&bytes), Err(OverflowDetected::InvalidInput));
    }
    assert_eq!(FixedPoint::from_raw_bytes(&bytes[1..]), Ok(x));
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

// ============================================================================
// tq19 narrowing (realtime: the profile with 32-bit activations)
// ============================================================================

#[cfg(all(feature = "inference", table_format = "q16_16"))]
mod tq19_narrowing {
    use g_math::fixed_point::frac_config::FRAC_BITS;
    use g_math::tq19::{packed_trit_dot, tq19_dot, tq19_dot_q2f, trit_dot, TQ19Matrix, MAX_RAW, SCALE};

    /// 0.6.4 returned -2147483648 (the true sum is 2^31).
    #[test]
    #[should_panic(expected = "exceeds storage range")]
    fn trit_dot_past_storage_panics() {
        let _ = trit_dot(&[1, 1], &[1 << 30, 1 << 30]);
    }

    /// 0.6.4 returned -436426 (the true value is 25,769,367,350).
    #[test]
    #[should_panic(expected = "exceeds storage range")]
    fn tq19_dot_past_storage_panics() {
        let _ = tq19_dot(&[MAX_RAW; 8], &[i32::MAX; 8]);
    }

    #[test]
    #[should_panic(expected = "exceeds storage range")]
    fn tq19_matvec_past_storage_panics() {
        let m = TQ19Matrix::new(1, 8, vec![MAX_RAW; 8]);
        let _ = m.matvec(&[i32::MAX; 8]);
    }

    /// Byte 229 decodes to trits [1, 1, 0, 0, 0].
    #[test]
    #[should_panic(expected = "exceeds storage range")]
    fn packed_trit_dot_past_storage_panics() {
        let _ = packed_trit_dot(&[229], 5, &[1 << 30, 1 << 30, 0, 0, 0], 1 << FRAC_BITS);
    }

    /// The unscaled dot (2^31) exceeds storage but the scaled value (2^30)
    /// fits: one exact product, one rounding. 0.6.4 narrowed the dot first
    /// and returned -2^30.
    #[test]
    fn packed_trit_dot_scales_before_narrowing() {
        let half = 1 << (FRAC_BITS - 1);
        assert_eq!(packed_trit_dot(&[229], 5, &[1 << 30, 1 << 30, 0, 0, 0], half), 1 << 30);
        // unchanged in range: (3 + 4) * 0.5 = 3.5 raw -> ties toward +inf
        assert_eq!(packed_trit_dot(&[229], 5, &[3, 4, 0, 0, 0], half), 4);
        assert_eq!(packed_trit_dot(&[229], 5, &[-3, -4, 0, 0, 0], half), -3);
    }

    /// -(i32::MIN) is 2^31. The AVX2 kernel flipped the sign in 32 bits, which
    /// wraps back to i32::MIN; with eight elements it disagreed with the
    /// scalar path by 2^32.
    #[test]
    fn trit_dot_negates_i32_min_exactly() {
        let trits = [-1i8, 1, 0, 0, 0, 0, 0, 0, 1];
        let acts = [i32::MIN, -5, 9, 9, 9, 9, 9, 9, 2];
        // 2^31 - 5 + 2 = 2147483645
        assert_eq!(trit_dot(&trits, &acts), 2147483645);
        assert_eq!(trit_dot(&trits[..2], &acts[..2]), 2147483643);
    }

    fn reference(weights: &[i16], acts: &[i32]) -> i128 {
        weights.iter().zip(acts).map(|(&w, &a)| w as i128 * a as i128).sum()
    }

    /// Rows past 2^16 columns take the exactly-summed path; rows below it the
    /// unchecked one. Both equal the i128 reference.
    #[test]
    fn long_rows_match_the_exact_reference() {
        for &n in &[131usize, 65_536, 65_536 + 24, 150_000] {
            let weights: Vec<i16> = (0..n).map(|i| ((i as i64 * 48_271 % 59_049) - 29_524) as i16).collect();
            let acts: Vec<i32> = (0..n).map(|i| ((i as i64 * 40_503 % 8_191) - 4_095) as i32).collect();
            let acc = reference(&weights, &acts);
            assert_eq!(tq19_dot(&weights, &acts) as i128, acc / SCALE as i128, "n = {n}");
            assert_eq!(tq19_dot_q2f(&weights, &acts) as i128, (acc << FRAC_BITS) / SCALE as i128, "n = {n}");
            let m = TQ19Matrix::new(1, n, weights.clone());
            let batch: Vec<&[i32]> = vec![&acts[..], &acts[..]];
            let expected = vec![vec![(acc / SCALE as i128) as i32]; 2];
            assert_eq!(m.matvec_batch(&batch), expected, "n = {n}");
            assert_eq!(m.matvec_batch_par(&batch), expected, "n = {n}");
        }
    }

    /// A long row whose exact sum leaves i64 panics instead of wrapping.
    #[test]
    #[should_panic(expected = "exceeds")]
    fn long_row_accumulator_overflow_panics() {
        let n = (1usize << 17) + 8;
        let _ = tq19_dot(&vec![i16::MIN; n], &vec![i32::MIN; n]);
    }
}
