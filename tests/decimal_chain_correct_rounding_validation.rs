//! Canonical chains with decimal results against mpmath
//! (`scripts/generate_decimal_chain_refs.py`, 600 digits): decimal literal
//! leaves, a transcendental at the root. The result, read as a rational, must
//! be the exact value rounded once, half to even, at the format's storage
//! decimal places. The tally per format is printed with `--nocapture`.


use num_bigint::BigInt;

/// "num/den" or "num", reduced.
fn fraction(s: &str) -> (BigInt, BigInt) {
    let (n, d) = match s.split_once('/') { Some((n, d)) => (n.trim(), d.trim()), None => (s.trim(), "1") };
    let (n, d): (BigInt, BigInt) = (n.parse().expect("numerator"), d.parse().expect("denominator"));
    let g = gcd(n.clone(), d.clone());
    let (mut n, mut d) = (n / &g, d / &g);
    if d < BigInt::from(0) { n = -n; d = -d; }
    (n, d)
}

fn gcd(a: BigInt, b: BigInt) -> BigInt {
    let (mut a, mut b) = (if a < BigInt::from(0) { -a } else { a }, if b < BigInt::from(0) { -b } else { b });
    while b != BigInt::from(0) { let t = &a % &b; a = b; b = t; }
    if a == BigInt::from(0) { BigInt::from(1) } else { a }
}

macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            #![allow(dead_code, unused_imports)]
            type FixedPoint = g_math::fixed_point::Fixed<$S, $F>;
            const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
            const FRAC_BITS: u32 = $F;
            use g_math::canonical::{evaluate, gmath_parse, LazyExpr, StackValue};

            mod data {
                #[allow(unused_imports)]
                use super::{BITS, FRAC_BITS};
                include!("data/decimal_chain_refs.rs");
            }

            type E = LazyExpr<FixedPoint>;

            fn node(t: &[String], i: &mut usize) -> E {
                let tok = t[*i].clone();
                *i += 1;
                if tok != "(" {
                    let lit = tok.strip_prefix('d').expect("decimal leaf");
                    return gmath_parse::<FixedPoint>(lit).expect("literal");
                }
                let op = t[*i].clone();
                *i += 1;
                let a = node(t, i);
                let e = match op.as_str() {
                    "exp" => a.exp(), "ln" => a.ln(), "sqrt" => a.sqrt(), "sin" => a.sin(), "cos" => a.cos(), "tan" => a.tan(),
                    "atan" => a.atan(), "asin" => a.asin(), "acos" => a.acos(), "sinh" => a.sinh(), "cosh" => a.cosh(),
                    "tanh" => a.tanh(), "asinh" => a.asinh(), "acosh" => a.acosh(), "atanh" => a.atanh(),
                    _ => { let b = node(t, i); match op.as_str() { "add" => a + b, "sub" => a - b, "mul" => a * b, "div" => a / b, o => panic!("{o}") } }
                };
                assert_eq!(t[*i], ")");
                *i += 1;
                e
            }

            #[test]
            fn decimal_chains_are_correctly_rounded() {
                let (mut cases, mut off, mut not_decimal) = (0, 0, 0);
                let mut failures = Vec::new();
                for &(tree, want) in data::REFS {
                    let toks: Vec<String> = tree.replace('(', " ( ").replace(')', " ) ").split_whitespace().map(String::from).collect();
                    let mut i = 0;
                    let e = node(&toks, &mut i);
                    cases += 1;
                    let got = std::panic::catch_unwind(|| evaluate(&e).ok().map(|v| (matches!(v, StackValue::<FixedPoint>::Decimal(..) | StackValue::<FixedPoint>::DecimalCompute(..)), v.to_rational().ok().map(|r| format!("{}", r)))));
                    match got {
                        Ok(Some((true, Some(r)))) => {
                            if super::fraction(&r) != super::fraction(want) {
                                off += 1;
                                if failures.len() < 12 { failures.push(format!("{tree}: got {r}, want {want}")); }
                            }
                        }
                        Ok(Some((false, _))) => { not_decimal += 1; }
                        _ => { off += 1; if failures.len() < 12 { failures.push(format!("{tree}: error or panic")); } }
                    }
                }
                println!("decimal chains Q{}.{}: cases {cases}, not correctly rounded {off}, not decimal {not_decimal}", BITS - FRAC_BITS, FRAC_BITS);
                assert!(off == 0, "Q{}.{}: {:#?}", BITS - FRAC_BITS, FRAC_BITS, failures);
            }
        }
    };
}

for_each_format!(suite);
suite!(q8_24, i32, 24);
