//! `DecimalRounding` and the one-rounding `mul_div` on `DecimalFixed` (0.6.5).
//!
//! References are exact integer arithmetic written independently of the
//! library's rounding code (`2r` against the divisor on a floored quotient),
//! plus literals and counts from an exact-rational model (Python `Fraction`;
//! the model is described on each pin). No floats. `DecimalFixed` arithmetic
//! is the same on every profile, so this suite is too.

use g_math::fixed_point::{Currency, DecimalFixed, DecimalRounding, OverflowDetected};

use DecimalRounding::{HalfEven, HalfUp};

/// round(n / d) for d > 0, any sign of n: nearest, ties by `mode`.
fn reference(n: i128, d: i128, mode: DecimalRounding) -> i128 {
    let neg = n < 0;
    let (q, r) = (n.abs() / d, n.abs() % d);
    let up = match (2 * r).cmp(&d) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => match mode {
            HalfUp => true,
            HalfEven => q % 2 == 1,
        },
    };
    let m = if up { q + 1 } else { q };
    if neg { -m } else { m }
}

/// Rates in hundredths of a percent: whole and half percent rates, and the
/// two-decimal class that arises from dividing a rounded tax by a rounded net.
const RATES: [i128; 24] = [
    0, 500, 600, 700, 800, 900, 1000, 1200, 1300, 1700, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2650, 2700,
    1, 99, 1001, 2199, 2501,
];

/// The included-tax share `t * R / (100 + R)` of a gross amount, one rounding.
fn vat(t: i128, rate: i128, mode: DecimalRounding) -> i128 {
    let total = Currency::from_raw(t);
    let pct = Currency::from_raw(rate); // 21.00
    let hundred_plus = Currency::from_raw(10_000 + rate); // 121.00
    total.try_mul_div_with(pct, hundred_plus, mode).unwrap().raw_value()
}

#[test]
fn mul_div_rounds_once_under_both_modes() {
    // 24 rates x 50,000 amounts. Model pins: the two modes differ on 5,060
    // amounts, 4,167 at 20% and 893 at 12%, nowhere else.
    let mut differ = [0u32; 24];
    for (i, &rate) in RATES.iter().enumerate() {
        for t in 1..=50_000i128 {
            let he = vat(t, rate, HalfEven);
            let hu = vat(t, rate, HalfUp);
            assert_eq!(he, reference(t * rate, 10_000 + rate, HalfEven), "half even t={t} R={rate}");
            assert_eq!(hu, reference(t * rate, 10_000 + rate, HalfUp), "half up t={t} R={rate}");
            // odd symmetry (credit lines)
            assert_eq!(vat(-t, rate, HalfEven), -he);
            assert_eq!(vat(-t, rate, HalfUp), -hu);
            if he != hu {
                differ[i] += 1;
            }
        }
    }
    for (i, &rate) in RATES.iter().enumerate() {
        let want = match rate { 2000 => 4_167, 1200 => 893, _ => 0 };
        assert_eq!(differ[i], want, "tie count at rate {rate}");
    }
}

#[test]
fn counterparty_cases() {
    // 49.94 at 26.50% -> 10.46; 2799.20 at 25.01% -> 560.02 (both modes)
    for mode in [HalfEven, HalfUp] {
        assert_eq!(vat(4_994, 2_650, mode), 1_046);
        assert_eq!(vat(279_920, 2_501, mode), 56_002);
    }
    // a tie: 0.03 at 20% is 0.005 exactly
    assert_eq!(vat(3, 2_000, HalfEven), 0);
    assert_eq!(vat(3, 2_000, HalfUp), 1);
    assert_eq!(vat(-3, 2_000, HalfUp), -1);
}

/// The staged form (product at six decimals, divide, narrow to two) rounds
/// three times. Model pin: it lands one unit from the single rounding on 22
/// of the 1,200,000 cases, all at two-decimal rates.
#[test]
fn staged_form_differs_from_one_rounding_on_22_cases() {
    let mut differ = 0u32;
    for &rate in &RATES {
        let mut here = 0u32;
        for t in 1..=50_000i128 {
            let a = DecimalFixed::<6>::from_raw(t * 10_000);
            let r = DecimalFixed::<6>::from_raw(rate * 10_000);
            let d = DecimalFixed::<6>::from_raw((10_000 + rate) * 10_000);
            let staged = a.try_mul(r).unwrap().try_div(d).unwrap().convert_with_rounding::<2>().raw_value();
            if staged != vat(t, rate, HalfEven) {
                here += 1;
            }
        }
        if rate % 50 == 0 {
            assert_eq!(here, 0, "whole and half percent rates never differ (rate {rate})");
        }
        differ += here;
    }
    assert_eq!(differ, 22);
}

/// The documented `try_div` contract: one rounding from the exact quotient.
#[test]
fn try_div_is_one_rounding_from_the_exact_quotient() {
    for &rate in &RATES {
        for t in (1..=50_000i128).step_by(7) {
            let n = Currency::from_raw(t * rate);
            let d = Currency::from_raw(100 * (10_000 + rate));
            for mode in [HalfEven, HalfUp] {
                // round(n * SCALE / d) = round(t * R / (10000 + R))
                assert_eq!(n.try_div_with(d, mode).unwrap().raw_value(), reference(t * rate, 10_000 + rate, mode));
            }
            assert_eq!(n.try_div(d), n.try_div_with(d, HalfEven));
        }
    }
}

#[test]
fn mul_and_convert_modes() {
    // 0.25 * 0.50 = 0.125 at two decimals
    let (a, b) = (Currency::from_raw(25), Currency::from_raw(50));
    assert_eq!(a.try_mul(b).unwrap().raw_value(), 12);
    assert_eq!(a.try_mul_with(b, HalfEven).unwrap().raw_value(), 12);
    assert_eq!(a.try_mul_with(b, HalfUp).unwrap().raw_value(), 13);
    assert_eq!((-a).try_mul_with(b, HalfUp).unwrap().raw_value(), -13);
    assert_eq!((-a).try_mul_with(b, HalfEven).unwrap().raw_value(), -12);
    // 0.35 * 0.50 = 0.175: both modes agree on 0.18
    assert_eq!(Currency::from_raw(35).try_mul_with(b, HalfEven).unwrap().raw_value(), 18);
    // not a tie
    assert_eq!(Currency::from_raw(26).try_mul_with(b, HalfUp).unwrap().raw_value(), 13);
    assert_eq!(Currency::from_raw(24).try_mul_with(b, HalfUp).unwrap().raw_value(), 12);

    // narrowing 6 -> 2 decimals
    let x = DecimalFixed::<6>::from_raw(125_000); // 0.125
    assert_eq!(x.convert_with_rounding::<2>().raw_value(), 12);
    assert_eq!(x.convert_with_rounding_mode::<2>(HalfEven).raw_value(), 12);
    assert_eq!(x.convert_with_rounding_mode::<2>(HalfUp).raw_value(), 13);
    assert_eq!((-x).convert_with_rounding_mode::<2>(HalfUp).raw_value(), -13);
    assert_eq!(DecimalFixed::<6>::from_raw(124_999).convert_with_rounding_mode::<2>(HalfUp).raw_value(), 12);
    // widening is exact, and loud past i128
    assert_eq!(Currency::from_raw(1999).try_convert_with_rounding::<6>(HalfUp).unwrap().raw_value(), 19_990_000);
    assert_eq!(
        Currency::from_raw(i128::MAX / 100).try_convert_with_rounding::<6>(HalfEven),
        Err(OverflowDetected::TierOverflow)
    );
}

#[test]
fn mul_div_precision_of_the_ratio_is_free() {
    // the ratio may carry its own precision: 21% as 21.0000 / 121.0000
    let total = Currency::from_raw(12_100);
    let pct = DecimalFixed::<4>::from_raw(210_000);
    let den = DecimalFixed::<4>::from_raw(1_210_000);
    assert_eq!(total.mul_div(pct, den).raw_value(), 2_100);
    assert_eq!(total.try_mul_div(pct, DecimalFixed::<4>::from_raw(0)), Err(OverflowDetected::DivisionByZero));
    assert_eq!(Currency::from_raw(0).mul_div(pct, den).raw_value(), 0);
}

/// Products past i128 take the 256-bit path. Literals from the exact-rational
/// model (seed 20261005): `round(a * n / d)`, the same under both modes
/// because none is a tie.
#[test]
fn mul_div_wide_path() {
    let cases: [(i128, i128, i128, i128); 4] = [
        (50987350244276324608789381139341911040, 843051498110757625083500703289, 492946242421774578553371480848553, 87200100759377071104911789351147270),
        (-34168591640737577598615938210799409260, 187581866841201927644758353204, 195400114344096569511050715534188, -32801455765872138495058022193466444),
        (20683631814350712241646387788486864176, 1249852834188406801386204947619, 965296476742258876628343268711642, 26780886978600536600824941493725437),
        (-39023157006791933280479483486913923941, 361907239418471079257305622936, 83447728523265381097351233584087, -169240832262848029348599890221758852),
    ];
    for &(a, n, d, want) in &cases {
        for mode in [HalfEven, HalfUp] {
            let got = Currency::from_raw(a).try_mul_div_with(Currency::from_raw(n), Currency::from_raw(d), mode);
            assert_eq!(got.unwrap().raw_value(), want);
            // sign of the divisor
            let got = Currency::from_raw(a).try_mul_div_with(Currency::from_raw(n), Currency::from_raw(-d), mode);
            assert_eq!(got.unwrap().raw_value(), -want);
        }
    }

    // a constructed wide tie: a = M(2k+1), n = N, d = 2MN gives k + 1/2 exactly
    let (m, nn) = ((1i128 << 70) + 1, (1i128 << 50) + 3);
    for (k, even, up) in [(1i128 << 40, 1i128 << 40, (1i128 << 40) + 1), ((1i128 << 40) + 1, (1i128 << 40) + 2, (1i128 << 40) + 2)] {
        let a = Currency::from_raw(m * (2 * k + 1));
        let (n, d) = (Currency::from_raw(nn), Currency::from_raw(2 * m * nn));
        assert_eq!(a.try_mul_div_with(n, d, HalfEven).unwrap().raw_value(), even);
        assert_eq!(a.try_mul_div_with(n, d, HalfUp).unwrap().raw_value(), up);
        assert_eq!((-a).try_mul_div_with(n, d, HalfEven).unwrap().raw_value(), -even);
        assert_eq!((-a).try_mul_div_with(n, d, HalfUp).unwrap().raw_value(), -up);
    }

    // result past i128
    let big = Currency::from_raw(i128::MAX);
    assert_eq!(big.try_mul_div(Currency::from_raw(3), Currency::from_raw(2)), Err(OverflowDetected::TierOverflow));
}
