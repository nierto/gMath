//! Certified decimal interval arithmetic: the permanent gate.
//!
//! Profile-independent (`DecimalFixed` is an i128 scaled by `10^DECIMALS` on
//! every profile), so every test here runs everywhere. Three properties:
//! 1. Directed rounding: for point operands the interval is `[floor, ceil]`
//!    of the exact result, the scalar (banker's) lies inside, and the two
//!    endpoints are at most one unit in the last place apart. Negative
//!    operands, exact halves (where banker's picks either side) and the
//!    256-bit intermediate path (products and scaled dividends beyond i128)
//!    are all exercised.
//! 2. Enclosure: `+ - * /`, `dot` and `sqrt` contain the exact result,
//!    checked against i128 / u128 references.
//! 3. The sqrt certificate `k^2 <= x * 10^D < (k+1)^2` holds on sampled input.
//!
//! Overflow and domain errors are typed, never wrapped, never saturated.

use g_math::fixed_point::{DecimalFixed, DecimalInterval, OverflowDetected};

type D9 = DecimalFixed<9>;
type I9 = DecimalInterval<9>;
type D2 = DecimalFixed<2>;
type I2 = DecimalInterval<2>;

const S9: i128 = 1_000_000_000;

/// Deterministic generator, integer only.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = self.0;
        (x >> 33) ^ x
    }
    fn raw(&mut self, bits: u32) -> i128 {
        let v = ((self.next() as u128) & ((1u128 << bits) - 1)) as i128;
        if self.next() & 1 == 1 { -v } else { v }
    }
}

fn d9(s: &str) -> D9 { s.parse().unwrap() }
fn p9(s: &str) -> I9 { I9::point(d9(s)) }
fn d2(s: &str) -> D2 { s.parse().unwrap() }
fn p2(s: &str) -> I2 { I2::point(d2(s)) }
fn ulp9() -> D9 { D9::from_raw(1) }
/// `10^k` at nine places, for integer parts beyond what `from_str` (i64) accepts.
fn pow10_9(k: u32) -> D9 { D9::from_raw(10i128.pow(k + 9)) }
fn ulp2() -> D2 { D2::from_raw(1) }

/// Floor and ceil of `num / den` for `den > 0`, in i128.
fn floor_ceil_pos(num: i128, den: i128) -> (i128, i128) {
    let f = num.div_euclid(den);
    let c = if num.rem_euclid(den) != 0 { f + 1 } else { f };
    (f, c)
}

// ----------------------------------------------------------------------------
// Property 1: directed rounding brackets the banker's scalar by at most 1 ulp
// ----------------------------------------------------------------------------

#[test]
fn mul_interval_is_floor_ceil_of_exact_product() {
    let mut rng = Rng(0xDEC1_3A7);
    let mut inexact = 0usize;
    let mut negatives = 0usize;
    for _ in 0..20_000 {
        let a = rng.raw(60);
        let b = rng.raw(60);
        let (floor, ceil) = floor_ceil_pos(a * b, S9);
        let iv = I9::point(D9::from_raw(a)) * I9::point(D9::from_raw(b));
        assert_eq!(iv.lo().raw_value(), floor, "floor mismatch for {a} * {b}");
        assert_eq!(iv.hi().raw_value(), ceil, "ceil mismatch for {a} * {b}");
        let nearest = D9::from_raw(a) * D9::from_raw(b);
        assert!(iv.contains(nearest), "banker's result escaped [{floor}, {ceil}]");
        if ceil != floor { inexact += 1; }
        if a * b < 0 { negatives += 1; }
    }
    assert!(inexact > 15_000, "sweep must exercise inexact products, got {inexact}");
    assert!(negatives > 8_000, "sweep must exercise negative products, got {negatives}");
}

#[test]
fn div_interval_is_floor_ceil_of_exact_quotient() {
    let mut rng = Rng(0xD1_0D1);
    let mut inexact = 0usize;
    let mut negatives = 0usize;
    for _ in 0..20_000 {
        let a = rng.raw(60);
        let mut b = rng.raw(40);
        if b == 0 { b = 7; }
        let num = a * S9;
        // Euclidean division gives a non-negative remainder for either sign of b.
        let q = num.div_euclid(b);
        let r = num.rem_euclid(b);
        let (floor, ceil) = if r == 0 {
            (q, q)
        } else if b > 0 {
            (q, q + 1)
        } else {
            (q - 1, q)
        };
        let iv = I9::point(D9::from_raw(a)) / I9::point(D9::from_raw(b));
        assert_eq!(iv.lo().raw_value(), floor, "floor mismatch for {a} / {b}");
        assert_eq!(iv.hi().raw_value(), ceil, "ceil mismatch for {a} / {b}");
        let nearest = D9::from_raw(a) / D9::from_raw(b);
        assert!(iv.contains(nearest), "banker's quotient escaped [{floor}, {ceil}]");
        if ceil != floor { inexact += 1; }
        if (a < 0) != (b < 0) { negatives += 1; }
    }
    assert!(inexact > 15_000, "sweep must exercise inexact quotients, got {inexact}");
    assert!(negatives > 8_000, "sweep must exercise negative quotients, got {negatives}");
}

/// Exact halves at two decimals: banker's rounding lands on either side, and
/// the interval brackets it both ways.
#[test]
fn exact_halves_bracket_bankers_rounding() {
    // 0.05 * 0.1 = 0.005: banker's -> 0.00 (even), interval [0.00, 0.01]
    let iv = p2("0.05") * p2("0.1");
    assert_eq!(iv.lo(), d2("0.00"));
    assert_eq!(iv.hi(), d2("0.01"));
    assert_eq!(iv.width(), ulp2());
    let nearest = d2("0.05") * d2("0.1");
    assert!(iv.contains(nearest));
    assert_eq!(nearest, iv.lo());

    // 0.15 * 0.1 = 0.015: banker's -> 0.02 (even), interval [0.01, 0.02]
    let iv = p2("0.15") * p2("0.1");
    assert_eq!(iv.lo(), d2("0.01"));
    assert_eq!(iv.hi(), d2("0.02"));
    let nearest = d2("0.15") * d2("0.1");
    assert!(iv.contains(nearest));
    assert_eq!(nearest, iv.hi());

    // negative half: -0.05 * 0.1 = -0.005, interval [-0.01, 0.00]
    let iv = p2("-0.05") * p2("0.1");
    assert_eq!(iv.lo(), d2("-0.01"));
    assert_eq!(iv.hi(), d2("0.00"));
    assert!(iv.contains(d2("-0.05") * d2("0.1")));

    // exact: 0.10 * 0.10 = 0.01, a point
    assert!((p2("0.10") * p2("0.10")).is_point());
    assert_eq!((p2("0.10") * p2("0.10")).lo(), d2("0.01"));

    // 1/3 at two places: [0.33, 0.34], and -1/3 mirrors it
    let third = p2("1") / p2("3");
    assert_eq!(third.lo(), d2("0.33"));
    assert_eq!(third.hi(), d2("0.34"));
    assert_eq!(p2("-1") / p2("3"), -third);
}

/// Products and scaled dividends beyond i128 take the 256-bit path.
#[test]
fn wide_intermediates_are_exact() {
    // 10^25 * 3 = 3 * 10^25: raw product 3e43 exceeds i128, exact result is a point
    let iv = I9::point(pow10_9(25)) * p9("3");
    assert!(iv.is_point());
    assert_eq!(iv.lo().raw_value(), 3 * 10i128.pow(34));

    // 10^25 / 3: scaled dividend 1e43 exceeds i128; one ulp wide, and
    // multiplying the endpoints back by 3 (exact, integer factor) brackets 10^25
    let a = pow10_9(25);
    let iv = I9::point(a) / p9("3");
    assert_eq!(iv.width(), ulp9());
    let back_lo = I9::point(iv.lo()) * p9("3");
    let back_hi = I9::point(iv.hi()) * p9("3");
    assert!(back_lo.is_point() && back_hi.is_point());
    assert!(back_lo.lo() <= a && a <= back_hi.lo());

    // the scalar path agrees with the enclosure on the same inputs
    assert!(iv.contains(a / d9("3")));
    assert!((I9::point(a) * p9("3")).contains(a * d9("3")));
}

// ----------------------------------------------------------------------------
// Property 2: enclosure of the exact result
// ----------------------------------------------------------------------------

#[test]
fn interval_mul_encloses_exact_product_of_interval_operands() {
    let mut rng = Rng(0xE7C_10D);
    for _ in 0..5_000 {
        let a0 = rng.raw(58);
        let a1 = a0 + (rng.next() & 0xFFF) as i128;
        let b0 = rng.raw(58);
        let b1 = b0 + (rng.next() & 0xFFF) as i128;
        let a = I9::new(D9::from_raw(a0), D9::from_raw(a1));
        let b = I9::new(D9::from_raw(b0), D9::from_raw(b1));
        let iv = a * b;
        for &x in &[a0, a1, (a0 + a1) / 2] {
            for &y in &[b0, b1, (b0 + b1) / 2] {
                let exact = x * y; // at 18 places
                assert!(iv.lo().raw_value() * S9 <= exact, "product {x}*{y} below the enclosure");
                assert!(exact <= iv.hi().raw_value() * S9, "product {x}*{y} above the enclosure");
            }
        }
    }
}

#[test]
fn dot_encloses_exact_sum_with_one_narrowing() {
    let mut rng = Rng(0xD07_DEC);
    for _ in 0..2_000 {
        let n = 1 + (rng.next() % 16) as usize;
        let a: Vec<i128> = (0..n).map(|_| rng.raw(56)).collect();
        let b: Vec<i128> = (0..n).map(|_| rng.raw(56)).collect();
        let exact: i128 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let (floor, ceil) = floor_ceil_pos(exact, S9);
        let av: Vec<D9> = a.iter().map(|&r| D9::from_raw(r)).collect();
        let bv: Vec<D9> = b.iter().map(|&r| D9::from_raw(r)).collect();
        let iv = I9::dot(&av, &bv);
        assert_eq!(iv.lo().raw_value(), floor);
        assert_eq!(iv.hi().raw_value(), ceil);
        assert!(iv.width() <= ulp9(), "dot widened beyond one narrowing");
    }
}

/// Chains: the scalar path rounds (banker's) at every step and must still lie
/// inside the interval path.
#[test]
fn composed_chain_contains_scalar_result() {
    let cases = [
        ("0.1", "0.2", "0.3", "0.7"),
        ("-1.5", "2.25", "0.125", "3"),
        ("123.456", "-0.001", "7", "-0.5"),
        ("0.333333333", "3", "-0.000000001", "1.000000001"),
        ("19.99", "1.0825", "0.005", "3"),
    ];
    for (a, b, c, d) in cases {
        let scalar = (d9(a) * d9(b) + d9(c)) / d9(d) - d9(a);
        let iv = (p9(a) * p9(b) + p9(c)) / p9(d) - p9(a);
        assert!(iv.contains(scalar), "scalar escaped chain enclosure for ({a},{b},{c},{d})");
        // two narrowings (mul, div); add and sub are exact
        assert!(iv.width() <= ulp9() + ulp9(), "chain width too large for ({a},{b},{c},{d})");
    }
}

// ----------------------------------------------------------------------------
// Property 3: the sqrt certificate
// ----------------------------------------------------------------------------

#[test]
fn sqrt_certificate_holds_for_sampled_inputs() {
    let mut rng = Rng(0x5A4_7DEC);
    let mut inexact = 0usize;
    for _ in 0..10_000 {
        // x * 10^9 must fit u128 for the reference: x < 2^96
        let x = ((rng.next() as u128) & ((1u128 << 63) - 1)) as i128 * ((rng.next() & 0xFFFF_FFFF) as i128 + 1);
        let n = (x as u128) * (S9 as u128);
        let iv = I9::point(D9::from_raw(x)).sqrt();
        let k = iv.lo().raw_value() as u128;
        let c = iv.hi().raw_value() as u128;
        assert!(k * k <= n, "floor certificate failed: k^2 > n for x = {x}");
        assert!(n < (k + 1) * (k + 1), "floor certificate failed: (k+1)^2 <= n for x = {x}");
        assert!(c == k || c == k + 1, "ceil is not floor or floor + 1 for x = {x}");
        assert_eq!(c == k, k * k == n, "ceil equals floor iff exact for x = {x}");
        if c != k { inexact += 1; }
    }
    assert!(inexact > 9_000, "sqrt sweep must exercise inexact roots, got {inexact}");
}

/// Structural sqrt pins, including digits that a reader can check by hand.
#[test]
fn sqrt_pins() {
    for (x, r) in [("4", "2"), ("0.25", "0.5"), ("2.25", "1.5"), ("0", "0"), ("1", "1")] {
        let iv = p9(x).sqrt();
        assert!(iv.is_point(), "sqrt({x}) should be exact");
        assert_eq!(iv.lo(), d9(r));
    }
    // sqrt(2) = 1.41421356237...: at nine places [1.414213562, 1.414213563]
    let s2 = p9("2").sqrt();
    assert_eq!(s2.lo(), d9("1.414213562"));
    assert_eq!(s2.hi(), d9("1.414213563"));
    // sqrt(10^20) = 10^10 exactly; sqrt(10^28) = 10^14, whose scaled input 10^37 takes the 256-bit path
    let s20 = I9::point(pow10_9(20)).sqrt();
    assert!(s20.is_point());
    assert_eq!(s20.lo(), pow10_9(10));
    let s28 = I9::point(pow10_9(28)).sqrt();
    assert!(s28.is_point());
    assert_eq!(s28.lo(), pow10_9(14));
    // interval input: monotone endpoints
    let iv = I9::new(d9("4"), d9("9")).sqrt();
    assert_eq!(iv.lo(), d9("2"));
    assert_eq!(iv.hi(), d9("3"));
    // the scalar engine (guard digits available at two places on every profile) lies inside
    let s = p2("2").sqrt();
    assert_eq!(s.lo(), d2("1.41"));
    assert_eq!(s.hi(), d2("1.42"));
    assert!(s.contains(d2("2").sqrt()));
}

// ----------------------------------------------------------------------------
// Errors are typed: never wrapped, never saturated
// ----------------------------------------------------------------------------

#[test]
fn errors_are_typed_where_the_scalar_saturates() {
    let straddle = I9::new(d9("-1"), d9("1"));
    assert_eq!(p9("1").try_div(straddle), Err(OverflowDetected::DivisionByZero));
    assert_eq!(p9("1").try_div(p9("0")), Err(OverflowDetected::DivisionByZero));
    // the scalar saturates on the same input; the interval refuses
    assert_eq!((d9("1") / d9("0")).raw_value(), i128::MAX);

    assert_eq!(I9::new(d9("-1"), d9("4")).try_sqrt(), Err(OverflowDetected::DomainError));
    assert_eq!(I9::try_new(d9("2"), d9("1")), Err(OverflowDetected::InvalidInput));

    let max = I9::point(D9::from_raw(i128::MAX));
    assert_eq!(max.try_add(I9::point(ulp9())), Err(OverflowDetected::TierOverflow));
    // the scalar saturates on the same input
    assert_eq!((D9::from_raw(i128::MAX) + ulp9()).raw_value(), i128::MAX);
    assert_eq!(max.try_mul(p9("2")), Err(OverflowDetected::TierOverflow));
    let min = I9::point(D9::from_raw(i128::MIN));
    assert_eq!(min.try_neg(), Err(OverflowDetected::TierOverflow));
    assert_eq!(min.try_sub(I9::point(ulp9())), Err(OverflowDetected::TierOverflow));

    // a quotient beyond i128 after scaling: 10^28 / 10^-9 = 10^37 at nine places = 10^46 raw
    assert_eq!(
        I9::point(pow10_9(28)).try_div(I9::point(ulp9())),
        Err(OverflowDetected::TierOverflow)
    );
    // a dot product whose exact sum fits D256 but whose narrowing does not fit i128
    let big = pow10_9(28);
    let v = [big, big, big, big];
    assert_eq!(I9::try_dot(&v, &v), Err(OverflowDetected::TierOverflow));
    // ceil of a value whose floor is i128::MAX has no home
    assert_eq!(
        max.try_mul(I9::new(D9::ONE, D9::ONE + ulp9())),
        Err(OverflowDetected::TierOverflow)
    );
}
