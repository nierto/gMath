//! The exp family across the whole output range of every binary format,
//! against mpmath (`scripts/generate_exp_full_range_refs.py`, 400 digits).
//!
//! The compute tiers' exp engines carry an error relative to the result, so a
//! result near the top of a format's range has the least margin. The earlier
//! gates used small arguments only. Every case here must be the nearest
//! storage value (ties toward +infinity); the tally per function is printed
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

            mod data {
                #[allow(unused_imports)]
                use super::{BITS, FRAC_BITS};
                include!("data/exp_full_range_refs.rs");
            }

            /// A decimal integer string as a raw of the word.
            fn raw(s: &str) -> $S {
                let (neg, digits) = match s.strip_prefix('-') { Some(r) => (true, r), None => (false, s) };
                let ten = <$S as WideInt>::from_i128(10);
                let mut v = <$S as WideInt>::from_i128(0);
                for ch in digits.chars() {
                    v = v * ten + <$S as WideInt>::from_i128(ch.to_digit(10).expect("digit") as i128);
                }
                if neg { -v } else { v }
            }

            /// |a - b| in storage units, saturating at u128::MAX.
            fn units(a: $S, b: $S) -> u128 {
                let d = match <$S as WideInt>::checked_sub_wide(a, b) { Some(d) => d, None => return u128::MAX };
                let m = if <$S as WideInt>::is_negative(d) { -d } else { d };
                let n = <$S as WideInt>::bit_length(m);
                if n > 128 { return u128::MAX; }
                let mut acc = 0u128;
                for k in 0..n { if <$S as WideInt>::bit(m, k) { acc |= 1u128 << k; } }
                acc
            }

            fn eval(f: &str, x: FixedPoint, y: FixedPoint) -> Option<$S> {
                let r = std::panic::catch_unwind(|| match f {
                    "exp" => x.try_exp(),
                    "sinh" => x.try_sinh(),
                    "cosh" => x.try_cosh(),
                    "tanh" => x.try_tanh(),
                    "pow" => x.try_pow(y),
                    "ln" => x.try_ln(),
                    "sqrt" => x.try_sqrt(),
                    "sin" => Ok(x.sin()),
                    "cos" => Ok(x.cos()),
                    "tan" => x.try_tan(),
                    "atan" => x.try_atan(),
                    "asin" => x.try_asin(),
                    "acos" => x.try_acos(),
                    "asinh" => x.try_asinh(),
                    "acosh" => x.try_acosh(),
                    "atanh" => x.try_atanh(),
                    "atan2" => x.try_atan2(y),
                    other => panic!("unknown function {other}"),
                });
                match r { Ok(Ok(v)) => Some(v.raw()), _ => None }
            }

            /// The same function through the canonical evaluator (one-node tree on a binary value).
            fn eval_canonical(f: &str, x: FixedPoint, y: FixedPoint) -> Option<$S> {
                use g_math::canonical::{evaluate_in, LazyExpr, StackValue};
                use g_math::canonical::CompactShadow;
                use g_math::fixed_point::FixedNum;
                let leaf = |v: FixedPoint| LazyExpr::<FixedPoint>::from(StackValue::<FixedPoint>::Binary(<FixedPoint as FixedNum>::STORAGE_TIER, v.raw(), CompactShadow::None));
                let r = std::panic::catch_unwind(|| {
                    let e = match f {
                        "exp" => leaf(x).exp(), "sinh" => leaf(x).sinh(), "cosh" => leaf(x).cosh(), "tanh" => leaf(x).tanh(),
                        "pow" => leaf(x).pow(leaf(y)), "ln" => leaf(x).ln(), "sqrt" => leaf(x).sqrt(), "sin" => leaf(x).sin(),
                        "cos" => leaf(x).cos(), "tan" => leaf(x).tan(), "atan" => leaf(x).atan(), "asin" => leaf(x).asin(),
                        "acos" => leaf(x).acos(), "asinh" => leaf(x).asinh(), "acosh" => leaf(x).acosh(), "atanh" => leaf(x).atanh(),
                        "atan2" => leaf(x).atan2(leaf(y)),
                        other => panic!("unknown function {other}"),
                    };
                    evaluate_in(&e).ok().and_then(|v| v.as_binary_storage())
                });
                r.ok().flatten()
            }

            #[test]
            fn canonical_path_is_correctly_rounded_across_the_whole_output_range() {
                let mut tally: Vec<(&str, usize, usize, usize, usize, u128)> = Vec::new();
                let mut failures = Vec::new();
                for &(f, xs, ys, want) in data::REFS {
                    let (x, y, want) = (FixedPoint::from_raw(raw(xs)), FixedPoint::from_raw(raw(ys)), raw(want));
                    let i = match tally.iter().position(|t| t.0 == f) { Some(i) => i, None => { tally.push((f, 0, 0, 0, 0, 0)); tally.len() - 1 } };
                    let t = &mut tally[i];
                    t.1 += 1;
                    match eval_canonical(f, x, y) {
                        Some(got) => {
                            let u = units(got, want);
                            if u == 1 { t.2 += 1; } else if u > 1 { t.3 += 1; }
                            if u > t.5 { t.5 = u; }
                            if u > 0 && failures.len() < 12 { failures.push(format!("{f}({xs}, {ys}): off by {u}")); }
                        }
                        None => { t.4 += 1; if failures.len() < 12 { failures.push(format!("{f}({xs}, {ys}): error or panic")); } }
                    }
                }
                println!("canonical Q{}.{}: function, cases, off by one, off by more, errors, worst", BITS - FRAC_BITS, FRAC_BITS);
                for t in &tally { println!("  {:5} {:4} {:4} {:4} {:4} {}", t.0, t.1, t.2, t.3, t.4, t.5); }
                assert!(failures.is_empty(), "canonical Q{}.{}: {} of {} cases not correctly rounded, e.g. {:?}",
                    BITS - FRAC_BITS, FRAC_BITS, tally.iter().map(|t| t.2 + t.3 + t.4).sum::<usize>(), data::REFS.len(), failures);
            }

            #[test]
            fn exp_family_is_correctly_rounded_across_the_whole_output_range() {
                // (function, cases, off by one, off by more, errors or panics, worst units)
                let mut tally: Vec<(&str, usize, usize, usize, usize, u128)> = Vec::new();
                let mut failures = Vec::new();
                for &(f, xs, ys, want) in data::REFS {
                    let x = FixedPoint::from_raw(raw(xs));
                    let y = FixedPoint::from_raw(raw(ys));
                    let want = raw(want);
                    let i = match tally.iter().position(|t| t.0 == f) { Some(i) => i, None => { tally.push((f, 0, 0, 0, 0, 0)); tally.len() - 1 } };
                    let t = &mut tally[i];
                    t.1 += 1;
                    match eval(f, x, y) {
                        Some(got) => {
                            let u = units(got, want);
                            if u == 1 { t.2 += 1; } else if u > 1 { t.3 += 1; }
                            if u > t.5 { t.5 = u; }
                            if u > 0 && failures.len() < 12 { failures.push(format!("{f}({xs}, {ys}): off by {u}")); }
                        }
                        None => { t.4 += 1; if failures.len() < 12 { failures.push(format!("{f}({xs}, {ys}): error or panic")); } }
                    }
                }
                println!("Q{}.{}: function, cases, off by one, off by more, errors, worst", BITS - FRAC_BITS, FRAC_BITS);
                for t in &tally { println!("  {:5} {:4} {:4} {:4} {:4} {}", t.0, t.1, t.2, t.3, t.4, t.5); }
                assert!(!data::REFS.is_empty(), "no references for this format");
                assert!(failures.is_empty(), "Q{}.{}: {} of {} cases not correctly rounded, e.g. {:?}",
                    BITS - FRAC_BITS, FRAC_BITS, tally.iter().map(|t| t.2 + t.3 + t.4).sum::<usize>(), data::REFS.len(), failures);
            }
        }
    };
}

for_each_format!(suite);
suite!(q8_24, i32, 24);
