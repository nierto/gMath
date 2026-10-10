//! The canonical layer on every width in one build: `evaluate_in::<T>` on a
//! corpus of expressions for six formats. Exact-domain results (decimal and
//! rational literals under `+ - * /`) are the same rational on every width;
//! transcendental results agree with the widest format within the narrower
//! format's own resolution. The per-format evaluators live in one thread-local
//! registry, so the formats are interleaved on purpose.

use g_math::canonical::{evaluate_in, gmath_in, FascFormat, LazyExpr, StackValue};
use g_math::fixed_point::domains::symbolic::rational::rational_number::RationalNumber;
use g_math::fixed_point::{Fixed, I256, I512, WideInt};

type Ref = Fixed<I512, 256>;

/// The corpus: a builder per case so every format gets its own tree.
fn corpus<T: FascFormat>() -> Vec<(&'static str, LazyExpr<T>, bool)> {
    // (label, tree, exact: the result is a rational the decimal/symbolic domains hold exactly)
    vec![
        ("0.1 + 0.2", gmath_in::<T>("0.1") + gmath_in::<T>("0.2"), true),
        ("1.5 * 2.5 + 0.1", gmath_in::<T>("1.5") * gmath_in::<T>("2.5") + gmath_in::<T>("0.1"), true),
        ("3 * 7 - 2", gmath_in::<T>("3") * gmath_in::<T>("7") - gmath_in::<T>("2"), true),
        ("1 / 3", gmath_in::<T>("1") / gmath_in::<T>("3"), true),
        ("(0.1 + 0.2) * 3", (gmath_in::<T>("0.1") + gmath_in::<T>("0.2")) * gmath_in::<T>("3"), true),
        ("-(2.75) / 4", -gmath_in::<T>("2.75") / gmath_in::<T>("4"), true),
        ("2.sqrt()", gmath_in::<T>("2").sqrt(), false),
        ("1.exp()", gmath_in::<T>("1").exp(), false),
        ("10.ln()", gmath_in::<T>("10").ln(), false),
        ("0.5.sin()", gmath_in::<T>("0.5").sin(), false),
        ("0.5.cos()", gmath_in::<T>("0.5").cos(), false),
        ("0.3.atan()", gmath_in::<T>("0.3").atan(), false),
        ("1.5.sinh()", gmath_in::<T>("1.5").sinh(), false),
        ("-0.5.tanh()", gmath_in::<T>("-0.5").tanh(), false),
        ("0.25.asin()", gmath_in::<T>("0.25").asin(), false),
        ("2.pow(0.5)", gmath_in::<T>("2").pow(gmath_in::<T>("0.5")), false),
        ("1.5.exp().ln()", gmath_in::<T>("1.5").exp().ln(), false),
        ("0.7.exp() * 0.7.exp()", gmath_in::<T>("0.7").exp() * gmath_in::<T>("0.7").exp(), false),
    ]
}

fn rational<T: FascFormat>(v: &StackValue<T>) -> RationalNumber {
    v.to_rational().expect("a finite result has a rational value")
}

const DIGITS: u32 = 45;

/// A displayed value as an integer scaled by 10^DIGITS (the wide rationals
/// of Q128.128 and Q256.256 results exceed the i128 rational arithmetic).
fn scaled(s: &str) -> I512 {
    let (neg, s) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
    let mut v = I512::from_i128(0);
    let ten = I512::from_i128(10);
    for d in int.bytes() {
        v = v * ten + I512::from_i128((d - b'0') as i128);
    }
    for i in 0..DIGITS as usize {
        let d = frac.as_bytes().get(i).map_or(0, |d| (d - b'0') as i128);
        v = v * ten + I512::from_i128(d);
    }
    if neg { I512::from_i128(0) - v } else { v }
}

fn pow10(n: u32) -> I512 {
    let mut v = I512::from_i128(1);
    for _ in 0..n {
        v = v * I512::from_i128(10);
    }
    v
}

/// One format against the reference: exact cases equal as rationals, the
/// rest within 4 units of the binary storage plus one unit of the decimal
/// storage, measured on 45 displayed digits.
fn check<T: FascFormat>(label: &str, frac_bits: u32, refs: &[(&'static str, StackValue<Ref>, bool)]) {
    let mut tol = pow10(DIGITS) * I512::from_i128(4);
    for _ in 0..frac_bits {
        tol = tol.shr_bits(1);
    }
    let tol = tol + pow10(DIGITS - T::DECIMAL_STORAGE_MAX_DP.min(DIGITS as u8) as u32);
    for (case, tree, exact) in corpus::<T>() {
        let got = evaluate_in::<T>(&tree).unwrap_or_else(|e| panic!("{label} {case}: {e:?}"));
        let want = &refs.iter().find(|(c, _, _)| *c == case).unwrap().1;
        if exact {
            assert_eq!(rational(&got), rational(want), "{label} {case}: exact-domain result differs from the reference");
        } else {
            let (g, w) = (scaled(&got.to_decimal_string(DIGITS as usize)), scaled(&want.to_decimal_string(DIGITS as usize)));
            let diff = if g > w { g - w } else { w - g };
            assert!(diff <= tol, "{label} {case}: {got} vs reference {want}, off by {diff:?} > {tol:?} (x 10^-45)");
        }
    }
}

#[test]
fn every_width_agrees_with_the_widest() {
    let refs: Vec<(&'static str, StackValue<Ref>, bool)> = corpus::<Ref>()
        .into_iter()
        .map(|(c, t, exact)| (c, evaluate_in::<Ref>(&t).unwrap_or_else(|e| panic!("reference {c}: {e:?}")), exact))
        .collect();
    // interleaved on purpose: six evaluators share one thread's registry
    for _ in 0..2 {
        check::<Fixed<i32, 10>>("Q22.10", 10, &refs);
        check::<Fixed<i128, 64>>("Q64.64", 64, &refs);
        check::<Fixed<i32, 16>>("Q16.16", 16, &refs);
        check::<Fixed<I256, 128>>("Q128.128", 128, &refs);
        check::<Fixed<i64, 32>>("Q32.32", 32, &refs);
        check::<Fixed<I512, 256>>("Q256.256", 256, &refs);
    }
}

#[test]
fn a_value_of_one_format_stays_in_its_tree() {
    // feeding a result back keeps its format: the types do not admit a mix
    let v = evaluate_in::<Fixed<i64, 32>>(&(gmath_in("0.1") + gmath_in("0.2"))).unwrap();
    let again = evaluate_in::<Fixed<i64, 32>>(&(LazyExpr::from(v) * gmath_in("10"))).unwrap();
    assert_eq!(rational(&again), RationalNumber::new(3, 1));
    assert!(format!("{again}").starts_with("3.0"), "{again}");
}

#[test]
fn display_follows_the_format_digits() {
    let narrow = format!("{}", evaluate_in::<Fixed<i32, 10>>(&(gmath_in("1") / gmath_in("4"))).unwrap());
    assert!(narrow.starts_with("0.25"), "{narrow}");
    let wide = format!("{}", evaluate_in::<Fixed<I256, 128>>(&(gmath_in("1") / gmath_in("4"))).unwrap());
    assert!(wide.starts_with("0.25"), "{wide}");
}
