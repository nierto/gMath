//! Canonical expression chains against mpmath (`scripts/generate_chain_refs.py`,
//! 600 digits): random trees of depth 2 to 4 over the 17 functions and the four
//! operations, with binary leaves, small integer literals and pi. Each result
//! must be the exact value of the tree on its exact leaves rounded once to
//! storage (nearest, ties toward +infinity). The tally per format is printed
//! with `--nocapture`.

#[macro_use]
mod common;

macro_rules! suite {
    ($m:ident, $S:ty, $F:literal) => {
        mod $m {
            #![allow(dead_code, unused_imports)]
            type FixedPoint = g_math::fixed_point::Fixed<$S, $F>;
            const BITS: u32 = <$S as g_math::fixed_point::WideInt>::BITS;
            const FRAC_BITS: u32 = $F;
            use g_math::fixed_point::WideInt;
            use g_math::canonical::{evaluate, gmath_parse, CompactShadow, ConstantId, LazyExpr, StackValue};
            use g_math::fixed_point::FixedNum;

            mod data {
                #[allow(unused_imports)]
                use super::{BITS, FRAC_BITS};
                include!("data/chain_refs.rs");
            }

            fn raw(s: &str) -> $S {
                let (neg, digits) = match s.strip_prefix('-') { Some(r) => (true, r), None => (false, s) };
                let ten = <$S as WideInt>::from_i128(10);
                let mut v = <$S as WideInt>::from_i128(0);
                for ch in digits.chars() {
                    v = v * ten + <$S as WideInt>::from_i128(ch.to_digit(10).expect("digit") as i128);
                }
                if neg { -v } else { v }
            }

            fn units(a: $S, b: $S) -> u128 {
                let d = match <$S as WideInt>::checked_sub_wide(a, b) { Some(d) => d, None => return u128::MAX };
                let m = if <$S as WideInt>::is_negative(d) { -d } else { d };
                let n = <$S as WideInt>::bit_length(m);
                if n > 128 { return u128::MAX; }
                let mut acc = 0u128;
                for k in 0..n { if <$S as WideInt>::bit(m, k) { acc |= 1u128 << k; } }
                acc
            }

            type E = LazyExpr<FixedPoint>;

            /// Parse the prefix tree text of the generator.
            fn parse(s: &str) -> E {
                let toks: Vec<String> = s.replace('(', " ( ").replace(')', " ) ").split_whitespace().map(String::from).collect();
                let mut i = 0;
                let e = node(&toks, &mut i);
                assert_eq!(i, toks.len(), "trailing tokens in {s}");
                e
            }

            fn node(t: &[String], i: &mut usize) -> E {
                let tok = t[*i].clone();
                *i += 1;
                if tok == "(" {
                    let op = t[*i].clone();
                    *i += 1;
                    let a = node(t, i);
                    let e = match op.as_str() {
                        "exp" => a.exp(), "ln" => a.ln(), "sqrt" => a.sqrt(), "sin" => a.sin(), "cos" => a.cos(),
                        "tan" => a.tan(), "atan" => a.atan(), "asin" => a.asin(), "acos" => a.acos(), "sinh" => a.sinh(),
                        "cosh" => a.cosh(), "tanh" => a.tanh(), "asinh" => a.asinh(), "acosh" => a.acosh(),
                        "atanh" => a.atanh(), "neg" => -a,
                        _ => {
                            let b = node(t, i);
                            match op.as_str() {
                                "add" => a + b, "sub" => a - b, "mul" => a * b, "div" => a / b,
                                "pow" => a.pow(b), "atan2" => a.atan2(b),
                                other => panic!("unknown op {other}"),
                            }
                        }
                    };
                    assert_eq!(t[*i], ")");
                    *i += 1;
                    e
                } else if tok == "pi" {
                    LazyExpr::Constant(ConstantId::Pi)
                } else if let Some(k) = tok.strip_prefix('i') {
                    gmath_parse::<FixedPoint>(k).expect("integer literal")
                } else if let Some(r) = tok.strip_prefix('#') {
                    E::from(StackValue::<FixedPoint>::Binary(<FixedPoint as FixedNum>::STORAGE_TIER, raw(r), CompactShadow::None))
                } else {
                    panic!("bad token {tok}")
                }
            }

            /// Evaluate every tree in one output mode: (cases, off by one,
            /// off by more, errors, not binary, worst, examples).
            fn run(mode: &str, require_binary: bool) -> (usize, usize, usize, usize, usize, u128, Vec<String>) {
                g_math::canonical::set_gmath_mode(mode).expect("mode");
                let mut t = (0usize, 0usize, 0usize, 0usize, 0usize, 0u128, Vec::new());
                for &(tree, want) in data::REFS {
                    let e = parse(tree);
                    let want = raw(want);
                    let got = std::panic::catch_unwind(|| evaluate(&e).map(|v| (v.as_binary_storage(), matches!(v, StackValue::<FixedPoint>::Binary(..)))));
                    t.0 += 1;
                    match got {
                        Ok(Ok((Some(g), true))) => {
                            let u = units(g, want);
                            if u == 1 { t.1 += 1 } else if u > 1 { t.2 += 1 }
                            if u > t.5 { t.5 = u }
                            if u > 0 && t.6.len() < 16 { t.6.push(format!("{mode} {tree}: off by {u}")); }
                        }
                        Ok(Ok(_)) => {
                            t.4 += 1;
                            if require_binary && t.6.len() < 16 { t.6.push(format!("{mode} {tree}: not binary")); }
                        }
                        _ => { t.3 += 1; if t.6.len() < 16 { t.6.push(format!("{mode} {tree}: error or panic")); } }
                    }
                }
                g_math::canonical::reset_gmath_mode();
                t
            }

            #[test]
            fn canonical_chains_are_correctly_rounded() {
                let auto = run("auto:auto", false);
                let bin = run("auto:binary", true);
                println!("chains Q{}.{}: mode, cases, off by one, off by more, errors, not binary, worst", BITS - FRAC_BITS, FRAC_BITS);
                println!("  auto   {:4} {:4} {:4} {:4} {:4} {}", auto.0, auto.1, auto.2, auto.3, auto.4, auto.5);
                println!("  binary {:4} {:4} {:4} {:4} {:4} {}", bin.0, bin.1, bin.2, bin.3, bin.4, bin.5);
                assert!(!data::REFS.is_empty(), "no references for this format");
                let bad = auto.1 + auto.2 + auto.3 + bin.1 + bin.2 + bin.3 + bin.4;
                let mut ex = auto.6.clone();
                ex.extend(bin.6.iter().cloned());
                assert!(bad == 0, "Q{}.{}: {} chain results not correctly rounded, e.g. {:#?}", BITS - FRAC_BITS, FRAC_BITS, bad, ex);
            }
        }
    };
}

for_each_format!(suite);
suite!(q8_24, i32, 24);
