//! Certified interval arithmetic: the permanent gate.
//!
//! Three properties, on every profile:
//! 1. Directed rounding: for every product and quotient, the interval of the
//!    point operands is `[floor, ceil]` of the exact result, the scalar
//!    (nearest, ties toward +infinity) lies inside it, and the two endpoints
//!    are at most one ulp apart. Negative operands and constructed exact ties
//!    are included, because arithmetic-shift floor is where a directed
//!    rounding goes wrong.
//! 2. Enclosure: for `+ - * /`, `dot`, `quadratic_form` and `sqrt`, the exact
//!    result lies within the interval. On the narrow profiles the exact result
//!    is computed in i128/u128 from raw values; on the wide profiles the
//!    scalar path (whose enclosure is implied) and structural pins are used.
//! 3. The sqrt certificate `k^2 <= n < (k+1)^2` holds for every sampled input.
//!
//! Overflow and domain errors are typed, never wrapped.

use g_math::fixed_point::{FixedMatrix, FixedPoint, FixedVector, Interval, OverflowDetected};

#[cfg(table_format = "q16_16")]
const FB: u32 = 16; // default FRAC_BITS in CI; custom splits share the kernel
#[cfg(table_format = "q32_32")]
const FB: u32 = 32;
#[cfg(table_format = "q64_64")]
const FB: u32 = 64;

/// Storage width in bits on the narrow profiles; results must fit it.
#[cfg(table_format = "q16_16")]
const SB: u32 = 32;
#[cfg(table_format = "q32_32")]
const SB: u32 = 64;
#[cfg(table_format = "q64_64")]
const SB: u32 = 128;

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn fits_storage(v: i128) -> bool {
    if SB >= 128 { return true; }
    let max = (1i128 << (SB - 1)) - 1;
    v >= -max - 1 && v <= max
}

/// Deterministic generator, integer only.
#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
struct Rng(u64);
#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = self.0;
        (x >> 33) ^ x
    }
}

fn fp(s: &str) -> FixedPoint { FixedPoint::from_str(s) }
fn pt(s: &str) -> Interval { Interval::point(fp(s)) }
fn ulp() -> FixedPoint { FixedPoint::from_raw(one_raw()) }

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn one_raw() -> g_math::fixed_point::imperative::BinaryStorage { 1 }
#[cfg(table_format = "q128_128")]
fn one_raw() -> g_math::fixed_point::imperative::BinaryStorage { g_math::fixed_point::I256::from_i128(1) }
#[cfg(table_format = "q256_256")]
fn one_raw() -> g_math::fixed_point::imperative::BinaryStorage { g_math::fixed_point::I512::from_i128(1) }

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn raw_i128(x: FixedPoint) -> i128 { x.raw() as i128 }

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn from_i128(r: i128) -> FixedPoint { FixedPoint::from_raw(r as _) }

/// Sample a raw value whose products with another sample stay within i128
/// at the compute scale: |raw| < 2^(FB + FB/2 - 2) keeps a*b < 2^(3FB - 4),
/// which is 2^188 at FB = 64; so at q64_64 the sweep restricts to |raw| < 2^62.
#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
fn sample_raw(rng: &mut Rng) -> i128 {
    let bits = if FB >= 64 { 62 } else { FB + FB / 2 - 2 };
    let mask = (1u128 << bits) - 1;
    let mag = (rng.next() as u128) & mask;
    let v = mag as i128;
    if rng.next() & 1 == 1 { -v } else { v }
}

// ----------------------------------------------------------------------------
// Property 1: directed rounding brackets the nearest scalar by at most 1 ulp
// ----------------------------------------------------------------------------

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
#[test]
fn mul_interval_is_floor_ceil_of_exact_product() {
    let mut rng = Rng(0x1A7E_57ED);
    let mut inexact = 0usize;
    let mut negatives = 0usize;
    for _ in 0..20_000 {
        let a = sample_raw(&mut rng);
        let b = sample_raw(&mut rng);
        let wide = a * b;
        let floor = wide >> FB;
        let ceil = if wide & ((1i128 << FB) - 1) != 0 { floor + 1 } else { floor };
        let iv = Interval::point(from_i128(a)) * Interval::point(from_i128(b));
        assert_eq!(raw_i128(iv.lo()), floor, "floor mismatch for {a} * {b}");
        assert_eq!(raw_i128(iv.hi()), ceil, "ceil mismatch for {a} * {b}");
        let nearest = from_i128(a) * from_i128(b);
        assert!(iv.contains(nearest), "nearest {} outside [{}, {}]", raw_i128(nearest), floor, ceil);
        if ceil != floor { inexact += 1; }
        if wide < 0 { negatives += 1; }
    }
    assert!(inexact > 15_000, "sweep must exercise inexact products, got {inexact}");
    assert!(negatives > 8_000, "sweep must exercise negative products, got {negatives}");
}

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
#[test]
fn div_interval_is_floor_ceil_of_exact_quotient() {
    let mut rng = Rng(0xD1F1_DED);
    let mut inexact = 0usize;
    let mut negatives = 0usize;
    let mut overflowed = 0usize;
    for _ in 0..20_000 {
        let a = sample_raw(&mut rng);
        let mut b = sample_raw(&mut rng) >> (FB / 2);
        if b == 0 { b = 3; }
        let num = a << FB;
        let q = num / b;
        let rem = num - q * b;
        let (floor, ceil) = if rem == 0 {
            (q, q)
        } else if (num < 0) == (b < 0) {
            (q, q + 1)
        } else {
            (q - 1, q)
        };
        let iv = Interval::point(from_i128(a)).try_div(Interval::point(from_i128(b)));
        if !fits_storage(floor) || !fits_storage(ceil) {
            // the quotient leaves the storage tier: typed, never wrapped
            assert_eq!(iv, Err(OverflowDetected::TierOverflow), "quotient {a} / {b} should overflow storage");
            overflowed += 1;
            continue;
        }
        let iv = iv.expect("quotient fits storage");
        assert_eq!(raw_i128(iv.lo()), floor, "floor mismatch for {a} / {b}");
        assert_eq!(raw_i128(iv.hi()), ceil, "ceil mismatch for {a} / {b}");
        let nearest = from_i128(a) / from_i128(b);
        assert!(iv.contains(nearest), "nearest {} outside [{}, {}]", raw_i128(nearest), floor, ceil);
        if ceil != floor { inexact += 1; }
        if (num < 0) != (b < 0) { negatives += 1; }
    }
    assert!(inexact > 14_000, "sweep must exercise inexact quotients, got {inexact}");
    assert!(negatives > 7_000, "sweep must exercise negative quotients, got {negatives}");
    if SB == 32 {
        assert!(overflowed > 0, "realtime sweep must reach the storage boundary");
    }
}

/// Constructed ties, both signs, on every profile.
#[test]
fn constructed_ties_bracket_nearest_and_are_one_ulp_wide() {
    let half = fp("0.5");
    let half_plus = half + ulp();
    let one = FixedPoint::one();

    // exact: 1.5 * 0.5 = 0.75, a point interval
    let exact = pt("1.5") * pt("0.5");
    assert!(exact.is_point());
    assert_eq!(exact.lo(), fp("0.75"));

    // positive tie: (0.5 + ulp) * 0.5 = 0.25 + half an ulp
    let iv = Interval::point(half_plus) * Interval::point(half);
    assert_eq!(iv.lo(), fp("0.25"));
    assert_eq!(iv.hi(), fp("0.25") + ulp());
    assert_eq!(iv.width(), ulp());
    let nearest = half_plus * half; // ties toward +infinity -> the ceil
    assert!(iv.contains(nearest));
    assert_eq!(nearest, iv.hi());

    // negative tie: -(0.5 + ulp) * 0.5 = -0.25 - half an ulp
    let iv = Interval::point(-half_plus) * Interval::point(half);
    assert_eq!(iv.hi(), -fp("0.25"));
    assert_eq!(iv.lo(), -fp("0.25") - ulp());
    assert_eq!(iv.width(), ulp());
    let nearest = (-half_plus) * half; // tie toward +infinity -> the ceil
    assert!(iv.contains(nearest));
    assert_eq!(nearest, iv.hi());

    // 1/3 is inexact on every binary profile: one ulp wide, scalar inside
    let third = Interval::point(one) / pt("3");
    assert_eq!(third.width(), ulp());
    assert!(third.contains(one / fp("3")));
    // and -1/3 mirrors it exactly (floor/ceil swap under negation)
    let neg_third = Interval::point(-one) / pt("3");
    assert_eq!(neg_third, -third);
}

// ----------------------------------------------------------------------------
// Property 2: enclosure of the exact result
// ----------------------------------------------------------------------------

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
#[test]
fn interval_mul_encloses_exact_product_of_interval_operands() {
    let mut rng = Rng(0xE7C1_05E);
    for _ in 0..5_000 {
        let a0 = sample_raw(&mut rng) >> 1;
        let a1 = a0 + ((rng.next() & 0xFF) as i128);
        let b0 = sample_raw(&mut rng) >> 1;
        let b1 = b0 + ((rng.next() & 0xFF) as i128);
        let a = Interval::new(from_i128(a0), from_i128(a1));
        let b = Interval::new(from_i128(b0), from_i128(b1));
        let iv = a * b;
        // every corner, and a few interior points, at exact scale
        for &x in &[a0, a1, (a0 + a1) / 2] {
            for &y in &[b0, b1, (b0 + b1) / 2] {
                let wide = x * y;
                let lo_scaled = raw_i128(iv.lo()) << FB;
                let hi_scaled = raw_i128(iv.hi()) << FB;
                assert!(lo_scaled <= wide && wide <= hi_scaled, "product {x}*{y} escaped the enclosure");
            }
        }
    }
}

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
#[test]
fn dot_encloses_exact_sum_with_one_narrowing() {
    let mut rng = Rng(0xD07_D07);
    for _ in 0..2_000 {
        let n = 1 + (rng.next() % 16) as usize;
        let a: Vec<i128> = (0..n).map(|_| sample_raw(&mut rng) >> 4).collect();
        let b: Vec<i128> = (0..n).map(|_| sample_raw(&mut rng) >> 4).collect();
        let exact: i128 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let floor = exact >> FB;
        let ceil = if exact & ((1i128 << FB) - 1) != 0 { floor + 1 } else { floor };
        let av: Vec<FixedPoint> = a.iter().map(|&r| from_i128(r)).collect();
        let bv: Vec<FixedPoint> = b.iter().map(|&r| from_i128(r)).collect();
        let iv = Interval::dot(&av, &bv);
        assert_eq!(raw_i128(iv.lo()), floor);
        assert_eq!(raw_i128(iv.hi()), ceil);
        assert!(iv.width() <= ulp(), "dot widened beyond one narrowing");
    }
}

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
#[test]
fn quadratic_form_encloses_exact_value_and_stays_narrow() {
    let mut rng = Rng(0x0F0_12A);
    for _ in 0..300 {
        let n = 2 + (rng.next() % 6) as usize;
        let v: Vec<i128> = (0..n).map(|_| sample_raw(&mut rng) >> (FB / 2 + 2)).collect();
        let m: Vec<Vec<i128>> = (0..n).map(|_| (0..n).map(|_| sample_raw(&mut rng) >> (FB / 2 + 2)).collect()).collect();
        // exact v^T M v at scale 2^(2 FB): sum_i v_i * (sum_j m_ij v_j), inner sum at 2^(2FB)
        // then divided by 2^FB exactly (rational); compare via cross-multiplication.
        let mut num: i128 = 0; // numerator at scale 2^(3 FB)
        for i in 0..n {
            let mut mv: i128 = 0;
            for j in 0..n { mv += m[i][j] * v[j]; }
            num += v[i] * mv;
        }
        // exact value in raw units = num / 2^(2 FB)
        let fv = FixedVector::from_slice(&v.iter().map(|&r| from_i128(r)).collect::<Vec<_>>());
        let mut fm = FixedMatrix::new(n, n);
        for i in 0..n { for j in 0..n { fm.set(i, j, from_i128(m[i][j])); } }
        let iv = Interval::quadratic_form(&fv, &fm);
        // lo * 2^(2FB) <= num  <=>  lo * 2^FB <= floor(num / 2^FB), since the
        // left side is an integer; likewise ceil(num / 2^FB) <= hi * 2^FB.
        let lo_scaled = raw_i128(iv.lo()) << FB;
        let hi_scaled = raw_i128(iv.hi()) << FB;
        let num_floor = num >> FB;
        let num_ceil = if num & ((1i128 << FB) - 1) != 0 { num_floor + 1 } else { num_floor };
        assert!(lo_scaled <= num_floor, "quadratic form lower endpoint above the exact value");
        assert!(num_ceil <= hi_scaled, "quadratic form upper endpoint below the exact value");
        // two narrowings, each 1 ulp, times |v_i| < 1 unit in these samples: a few ulp at most
        let width = raw_i128(iv.width());
        assert!(width <= (n as i128) + 2, "quadratic form width {width} ulp exceeds the two-narrowing bound for n = {n}");
    }
}

/// Structural pin on every profile: with the identity metric the quadratic
/// form's first stage is exact, so `quadratic_form(v, I) == dot(v, v)`.
#[test]
fn quadratic_form_with_identity_equals_dot() {
    let v = FixedVector::from_slice(&[fp("0.3"), fp("-1.25"), fp("2.5"), fp("0.001")]);
    let id = FixedMatrix::identity(4);
    let qf = Interval::quadratic_form(&v, &id);
    let vs: Vec<FixedPoint> = (0..4).map(|i| v[i]).collect();
    let d = Interval::dot(&vs, &vs);
    assert_eq!(qf, d);
    assert!(qf.is_certainly_positive());
    assert!(qf.width() <= ulp());
}

/// Chains: the scalar path rounds at every step and must still lie inside the
/// interval path on every profile.
#[test]
fn composed_chain_contains_scalar_result() {
    let cases = [
        ("0.1", "0.2", "0.3", "0.7"),
        ("-1.5", "2.25", "0.125", "3"),
        ("123.456", "-0.001", "7", "-0.5"),
        ("0.333333", "3", "-0.000001", "1.000001"),
    ];
    for (a, b, c, d) in cases {
        let (fa, fb, fc, fd) = (fp(a), fp(b), fp(c), fp(d));
        let scalar = (fa * fb + fc) / fd - fa;
        let iv = (pt(a) * pt(b) + pt(c)) / pt(d) - pt(a);
        assert!(iv.contains(scalar), "scalar escaped chain enclosure for ({a},{b},{c},{d})");
        // three narrowings: mul, div, and nothing else rounds (add/sub exact)
        assert!(iv.width() <= ulp() + ulp() + ulp(), "chain width too large for ({a},{b},{c},{d})");
    }
}

// ----------------------------------------------------------------------------
// Property 3: the sqrt certificate
// ----------------------------------------------------------------------------

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
#[test]
fn sqrt_certificate_holds_for_sampled_inputs() {
    let mut rng = Rng(0x5A4_7C3);
    let mut inexact = 0usize;
    for _ in 0..10_000 {
        // x must fit the storage tier, and n = x_raw << FB must fit u128
        let bits = (127 - FB).min(SB - 1).min(63);
        let x = ((rng.next() as u128) & ((1u128 << bits) - 1)) as i128;
        let n = (x as u128) << FB;
        let iv = Interval::point(from_i128(x)).sqrt();
        let k = raw_i128(iv.lo()) as u128;
        let c = raw_i128(iv.hi()) as u128;
        assert!(k * k <= n, "floor certificate failed: k^2 > n for x = {x}");
        assert!(n < (k + 1) * (k + 1), "floor certificate failed: (k+1)^2 <= n for x = {x}");
        assert!(c == k || c == k + 1, "ceil is not floor or floor + 1 for x = {x}");
        assert_eq!(c == k, k * k == n, "ceil equals floor iff exact for x = {x}");
        if c != k { inexact += 1; }
        // the scalar sqrt lies inside
        assert!(iv.contains(from_i128(x).sqrt()));
    }
    assert!(inexact > 9_000, "sqrt sweep must exercise inexact roots, got {inexact}");
}

/// Structural sqrt pins on every profile.
#[test]
fn sqrt_pins() {
    // perfect squares are points
    for (x, r) in [("4", "2"), ("0.25", "0.5"), ("2.25", "1.5"), ("0", "0"), ("1", "1")] {
        let iv = pt(x).sqrt();
        assert!(iv.is_point(), "sqrt({x}) should be exact");
        assert_eq!(iv.lo(), fp(r));
    }
    // sqrt(2) is one ulp wide and contains the scalar
    let s2 = pt("2").sqrt();
    assert_eq!(s2.width(), ulp());
    assert!(s2.contains(fp("2").sqrt()));
    // interval input: monotone endpoints
    let iv = Interval::new(fp("4"), fp("9")).sqrt();
    assert_eq!(iv.lo(), fp("2"));
    assert_eq!(iv.hi(), fp("3"));
    // certificate squared back, using interval arithmetic (itself sound):
    // lo^2 <= x and x <= hi^2 hold in the enclosed products
    let x = fp("2");
    let lo2 = Interval::point(s2.lo()) * Interval::point(s2.lo());
    let hi2 = Interval::point(s2.hi()) * Interval::point(s2.hi());
    assert!(lo2.lo() <= x, "lo^2 exceeds x");
    assert!(hi2.hi() >= x, "hi^2 below x");
}

// ----------------------------------------------------------------------------
// Errors are typed, never wrapped
// ----------------------------------------------------------------------------

#[test]
fn errors_are_typed() {
    let zero_straddle = Interval::new(fp("-1"), fp("1"));
    assert_eq!(pt("1").try_div(zero_straddle), Err(OverflowDetected::DivisionByZero));
    assert_eq!(pt("1").try_div(pt("0")), Err(OverflowDetected::DivisionByZero));
    assert_eq!(Interval::new(fp("-1"), fp("4")).try_sqrt(), Err(OverflowDetected::DomainError));
    assert_eq!(Interval::try_new(fp("2"), fp("1")), Err(OverflowDetected::InvalidInput));

    let max = Interval::point(FixedPoint::from_raw(max_raw()));
    assert_eq!(max.try_add(Interval::point(ulp())), Err(OverflowDetected::TierOverflow));
    assert_eq!(max.try_mul(pt("2")), Err(OverflowDetected::TierOverflow));
    let min = Interval::point(FixedPoint::from_raw(min_raw()));
    assert_eq!(min.try_neg(), Err(OverflowDetected::TierOverflow));
    assert_eq!(min.try_sub(Interval::point(ulp())), Err(OverflowDetected::TierOverflow));
    // ceil of a value whose floor is the storage maximum has no home
    assert_eq!(max.try_mul(Interval::new(FixedPoint::one(), FixedPoint::one() + ulp())), Err(OverflowDetected::TierOverflow));
}

#[cfg(table_format = "q16_16")]
fn max_raw() -> g_math::fixed_point::imperative::BinaryStorage { i32::MAX }
#[cfg(table_format = "q32_32")]
fn max_raw() -> g_math::fixed_point::imperative::BinaryStorage { i64::MAX }
#[cfg(table_format = "q64_64")]
fn max_raw() -> g_math::fixed_point::imperative::BinaryStorage { i128::MAX }
#[cfg(table_format = "q128_128")]
fn max_raw() -> g_math::fixed_point::imperative::BinaryStorage { g_math::fixed_point::I256::max_value() }
#[cfg(table_format = "q256_256")]
fn max_raw() -> g_math::fixed_point::imperative::BinaryStorage { g_math::fixed_point::I512::max_value() }

#[cfg(table_format = "q16_16")]
fn min_raw() -> g_math::fixed_point::imperative::BinaryStorage { i32::MIN }
#[cfg(table_format = "q32_32")]
fn min_raw() -> g_math::fixed_point::imperative::BinaryStorage { i64::MIN }
#[cfg(table_format = "q64_64")]
fn min_raw() -> g_math::fixed_point::imperative::BinaryStorage { i128::MIN }
#[cfg(table_format = "q128_128")]
fn min_raw() -> g_math::fixed_point::imperative::BinaryStorage { g_math::fixed_point::I256::min_value() }
#[cfg(table_format = "q256_256")]
fn min_raw() -> g_math::fixed_point::imperative::BinaryStorage { g_math::fixed_point::I512::min_value() }
