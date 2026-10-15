//! Differential gate: bit-identity with the published 0.6.7 (the scalar surface is unchanged since 0.6.5).
//!
//! `g_math` 0.6.7 from crates.io is a dev-dependency (`g_math_065`). Both
//! crates read the same `GMATH_PROFILE` / `GMATH_FRAC_BITS`, so in each build
//! the old `FixedPoint` and the new one are the same format, and every
//! operation is run on both with the same raw inputs. Required: the same
//! integers, the same `Err` variants, and a panic exactly where the old
//! crate panicked. Run once per profile and split.
//!
//! Inputs are raw integers of every magnitude: a random bit length, random
//! bits below it, a random sign, plus the extremes and the values around
//! zero and one.

use g_math::fixed_point::{Fixed, FixedNum, Storage, WideInt};
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---- the format of the published crate's build, mirrored in this crate -----
//
// The published crate still reads `GMATH_PROFILE` / `GMATH_FRAC_BITS`; this
// crate has no profile. The new type is chosen from the old crate's own
// storage word and split, so one build of this test compares the two crates
// at whatever format the old one was built for.

type OldWord = g_math_065::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
const OLD_FRAC: u32 = g_math_065::fixed_point::frac_config::FRAC_BITS;

/// The new crate's word for an old crate's word, and the raw in both as bytes.
trait Mirror: Copy {
    type New: Storage;
    fn mirror_to_be(self) -> Vec<u8>;
    fn mirror_from_be(bytes: &[u8]) -> Self;
}
macro_rules! mirror_native {
    ($t:ty) => {
        impl Mirror for $t {
            type New = $t;
            fn mirror_to_be(self) -> Vec<u8> { self.to_be_bytes().to_vec() }
            fn mirror_from_be(bytes: &[u8]) -> Self { <$t>::from_be_bytes(bytes.try_into().expect("word bytes")) }
        }
    };
}
mirror_native!(i32);
mirror_native!(i64);
mirror_native!(i128);
macro_rules! mirror_wide {
    ($old:ty, $new:ty, $words:expr) => {
        impl Mirror for $old {
            type New = $new;
            fn mirror_to_be(self) -> Vec<u8> {
                let mut be = Vec::with_capacity($words * 8);
                for w in self.words.iter().rev() { be.extend_from_slice(&w.to_be_bytes()); }
                be
            }
            fn mirror_from_be(bytes: &[u8]) -> Self {
                let mut words = [0u64; $words];
                for (i, chunk) in bytes.rchunks(8).enumerate() { words[i] = u64::from_be_bytes(chunk.try_into().expect("word bytes")); }
                Self { words }
            }
        }
    };
}
mirror_wide!(g_math_065::fixed_point::I256, g_math::fixed_point::I256, 4);
mirror_wide!(g_math_065::fixed_point::I512, g_math::fixed_point::I512, 8);

type New = Fixed<<OldWord as Mirror>::New, OLD_FRAC>;
use g_math_065::fixed_point::FixedPoint as Old;

// ---- the same raw integer in both crates --------------------------------

mod raw {
    use super::{Mirror, New, Old, OldWord, Storage};
    /// A raw of the new crate's word.
    pub type Raw = <OldWord as Mirror>::New;
    pub const WORDS: usize = (<Raw as super::WideInt>::BITS / 64) as usize + (<Raw as super::WideInt>::BITS % 64 != 0) as usize;
    /// Little-endian 64-bit words into a raw (the words beyond the width are ignored).
    pub fn from_words(w: &[u64]) -> Raw {
        let bytes = (<Raw as super::WideInt>::BITS / 8) as usize;
        let mut be = vec![0u8; bytes];
        for (i, x) in w.iter().enumerate() {
            for k in 0..8 {
                let pos = i * 8 + k;
                if pos < bytes { be[bytes - 1 - pos] = (x >> (8 * k)) as u8; }
            }
        }
        <Raw as Storage>::from_be_slice(&be)
    }
    pub fn old(r: Raw) -> Old { Old::from_raw(OldWord::mirror_from_be(&<Raw as Storage>::to_be_vec(r))) }
    pub fn new(r: Raw) -> New { New::from_raw(r) }
    pub fn old_raw(v: Old) -> Raw { <Raw as Storage>::from_be_slice(&v.raw().mirror_to_be()) }
    pub fn new_raw(v: New) -> Raw { v.raw() }
}

use raw::{Raw, WORDS};

#[allow(dead_code)]
const BITS: u32 = <Raw as WideInt>::BITS;
const WIDTH: u32 = BITS;
const FRAC: u32 = <New as FixedNum>::FRAC_BITS;

// ---- inputs ---------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = self.0;
        (x ^ (x >> 33)).wrapping_mul(0xff51afd7ed558ccd) ^ (x >> 29)
    }
    /// A raw of a random magnitude: `bits` significant bits, random sign.
    fn raw(&mut self, max_bits: u32) -> Raw {
        let bits = (self.next() % (max_bits as u64 + 1)) as u32;
        let mut w = [0u64; 8];
        for i in 0..WORDS {
            w[i] = self.next();
        }
        // keep `bits` low bits, force the top one
        for i in 0..WORDS as u32 {
            let lo = i * 64;
            if bits <= lo {
                w[i as usize] = 0;
            } else if bits < lo + 64 {
                w[i as usize] &= (1u64 << (bits - lo)) - 1;
                w[i as usize] |= 1u64 << (bits - lo - 1);
            }
        }
        if self.next() & 1 == 1 {
            // two's complement negation over the storage width
            let mut carry = 1u64;
            for x in w.iter_mut().take(WORDS) {
                let (v, c) = (!*x).overflowing_add(carry);
                *x = v;
                carry = c as u64;
            }
        }
        if WIDTH < 64 {
            // sign-extend the low word's WIDTH bits
            w[0] = ((w[0] as i64) << (64 - WIDTH) >> (64 - WIDTH)) as u64;
        }
        raw::from_words(&w)
    }
}

/// Cases per function, scaled down where one evaluation is slow.
fn count(narrow: usize) -> usize {
    let scale = std::env::var("GMATH_DIFF_SCALE").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let base = match WIDTH {
        32 | 64 => narrow,
        128 => narrow / 4,
        256 => narrow / 20,
        _ => narrow / 60,
    };
    (base * scale).max(24)
}

/// Raws to try: every magnitude up to `max_bits`, then the special values.
fn inputs(seed: u64, n: usize, max_bits: u32) -> Vec<Raw> {
    let mut rng = Rng(seed);
    let mut v: Vec<Raw> = (0..n).map(|_| rng.raw(max_bits)).collect();
    v.extend(specials());
    v
}

fn specials() -> Vec<Raw> {
    let mut out = Vec::new();
    let mut push = |f: &dyn Fn(&mut [u64; 8])| {
        let mut w = [0u64; 8];
        f(&mut w);
        if WIDTH < 64 {
            w[0] = ((w[0] as i64) << (64 - WIDTH) >> (64 - WIDTH)) as u64;
        }
        out.push(raw::from_words(&w));
    };
    let set = |w: &mut [u64; 8], bit: u32| w[(bit / 64) as usize] |= 1u64 << (bit % 64);
    push(&|_| {}); // 0
    push(&|w| w[0] = 1); // one unit
    push(&|w| w.iter_mut().take(WORDS).for_each(|x| *x = u64::MAX)); // minus one unit
    push(&|w| set(w, FRAC)); // 1.0
    push(&|w| { set(w, FRAC); w[0] |= 1 }); // 1.0 + one unit
    push(&|w| set(w, FRAC - 1)); // 0.5
    push(&|w| set(w, FRAC + 1)); // 2.0
    push(&|w| { // -1.0
        for b in FRAC..WIDTH { set(w, b) }
    });
    push(&|w| { // -0.5
        for b in FRAC - 1..WIDTH { set(w, b) }
    });
    push(&|w| set(w, WIDTH - 1)); // minimum
    push(&|w| { // maximum
        for b in 0..WIDTH - 1 { set(w, b) }
    });
    push(&|w| { // minimum + one unit
        set(w, WIDTH - 1);
        w[0] |= 1
    });
    out
}

// ---- comparing outcomes -----------------------------------------------------

/// What a call did: a value, an error (by its Debug text, the variants are
/// distinct types in the two crates), or a panic.
#[derive(PartialEq, Debug)]
enum Outcome<T> {
    Value(T),
    Error(String),
    Panic,
}

thread_local! {
    /// Set while a call that may legitimately panic is being compared.
    static EXPECTING_PANICS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn quiet<T>(f: impl FnOnce() -> T) -> Option<T> {
    EXPECTING_PANICS.with(|e| e.set(true));
    let result = catch_unwind(AssertUnwindSafe(f)).ok();
    EXPECTING_PANICS.with(|e| e.set(false));
    result
}

fn value<T>(f: impl FnOnce() -> T) -> Outcome<T> {
    quiet(f).map_or(Outcome::Panic, Outcome::Value)
}

fn fallible<T, E: std::fmt::Debug>(f: impl FnOnce() -> Result<T, E>) -> Outcome<T> {
    match quiet(f) {
        None => Outcome::Panic,
        Some(Ok(v)) => Outcome::Value(v),
        Some(Err(e)) => Outcome::Error(format!("{e:?}")),
    }
}

struct Tally {
    name: &'static str,
    cases: usize,
    values: usize,
    errors: usize,
    panics: usize,
    /// Differences where 0.6.7 was not correctly rounded and the new crate
    /// equals the sure path (transcendentals), or where a composition moved
    /// by at most `NEAR_UNITS` because a transcendental inside it now rounds
    /// correctly (geometry). Every other difference fails the gate.
    corrected: usize,
    worst: u128,
}

/// The largest move of a geometric composition allowed when the
/// transcendentals inside it became correctly rounded.
const NEAR_UNITS: u128 = 4;

/// Distance in storage units between two results.
trait Units {
    fn units(&self, other: &Self) -> u128;
}
impl Units for Raw {
    fn units(&self, other: &Self) -> u128 {
        let d = match self.checked_sub_wide(*other) { Some(d) => d, None => return u128::MAX };
        let m = if d.is_negative() { -d } else { d };
        let n = m.bit_length();
        if n > 127 { return u128::MAX; }
        (0..n).fold(0u128, |acc, k| if m.bit(k) { acc | (1u128 << k) } else { acc })
    }
}
impl Units for usize {
    fn units(&self, other: &Self) -> u128 { if self == other { 0 } else { u128::MAX } }
}
impl<A: Units, B: Units> Units for (A, B) {
    fn units(&self, other: &Self) -> u128 { self.0.units(&other.0).max(self.1.units(&other.1)) }
}
impl<A: Units, B: Units, C: Units> Units for (A, B, C) {
    fn units(&self, other: &Self) -> u128 { self.0.units(&other.0).max(self.1.units(&other.1)).max(self.2.units(&other.2)) }
}
impl<T: Units> Units for Vec<T> {
    fn units(&self, other: &Self) -> u128 {
        if self.len() != other.len() { return u128::MAX; }
        self.iter().zip(other).map(|(a, b)| a.units(b)).max().unwrap_or(0)
    }
}

impl Tally {
    fn new(name: &'static str) -> Self {
        Tally { name, cases: 0, values: 0, errors: 0, panics: 0, corrected: 0, worst: 0 }
    }
    fn check<T: PartialEq + std::fmt::Debug>(&mut self, old: Outcome<T>, new: Outcome<T>, input: &dyn std::fmt::Debug) {
        assert_eq!(old, new, "{} differs from 0.6.5 on input {:?}", self.name, input);
        self.cases += 1;
        match old {
            Outcome::Value(_) => self.values += 1,
            Outcome::Error(_) => self.errors += 1,
            Outcome::Panic => self.panics += 1,
        }
    }
    fn report(&self) {
        println!(
            "{:<22} {:>6} cases: {:>6} values, {:>5} errors, {:>5} panics, {} corrected (worst {} units), 0 other differences",
            self.name, self.cases, self.values, self.errors, self.panics, self.corrected, self.worst
        );
    }
    /// As `check`, where the new crate may differ from 0.6.7 only by being
    /// the correctly rounded value the sure path computes (`oracle`, `None`
    /// where the result does not exist or leaves storage).
    /// Raw compute-tier words (compared as their Debug text): a difference
    /// passes only when every new component is no farther from the exact
    /// value (`Mpf` at 1088 bits) than the 0.6.7 one, and counts as corrected.
    fn check_closer(&mut self, old: Outcome<String>, new: Outcome<String>, frac: u32, exact: impl FnOnce() -> Vec<ExactValue>, input: &dyn std::fmt::Debug) {
        if old == new { return self.check(old, new, input); }
        let (Outcome::Value(o), Outcome::Value(n)) = (&old, &new) else {
            panic!("{} differs from 0.6.7 on input {:?} in kind: old {:?}, new {:?}", self.name, input, old, new);
        };
        let (ov, nv, ev) = (debug_words(o), debug_words(n), exact());
        assert!(ov.len() == ev.len() && nv.len() == ev.len(), "{}: component counts", self.name);
        for ((ow, nw), e) in ov.iter().zip(&nv).zip(&ev) {
            let e = match e { Ok(e) => *e, Err(_) => continue };
            let dist = |w: &I1024| Mpf::<C17>::from_raw(*w, frac).sub(e).abs();
            let worse = dist(ow).sub(dist(nw)).is_negative();
            assert!(!worse, "{} differs from 0.6.7 on input {:?} and is farther from the exact value: old {}, new {}", self.name, input, o, n);
        }
        self.cases += 1;
        self.corrected += 1;
        self.values += 1;
    }

    fn check_oracle<T: PartialEq + std::fmt::Debug + Units>(&mut self, old: Outcome<T>, new: Outcome<T>, oracle: impl FnOnce() -> Option<T>, input: &dyn std::fmt::Debug) {
        if old == new { return self.check(old, new, input); }
        let want = oracle();
        let ok = match (&new, &want) {
            (Outcome::Value(v), Some(w)) => *v == *w,
            (Outcome::Value(_), None) => false,
            (_, None) => true,
            (_, Some(_)) => false,
        };
        assert!(ok, "{} differs from 0.6.7 on input {:?} and is not the correctly rounded value: old {:?}, new {:?}, sure {:?}", self.name, input, old, new, want);
        if let (Outcome::Value(o), Outcome::Value(n)) = (&old, &new) { self.worst = self.worst.max(o.units(n)); }
        self.cases += 1;
        self.corrected += 1;
        if let Outcome::Value(_) = new { self.values += 1 } else { self.errors += 1 }
    }
    /// As `check`, where a geometric composition may move by at most
    /// `NEAR_UNITS` (a transcendental inside it now rounds correctly).
    fn check_near<T: PartialEq + std::fmt::Debug + Units>(&mut self, old: Outcome<T>, new: Outcome<T>, input: &dyn std::fmt::Debug) {
        if old == new { return self.check(old, new, input); }
        let u = match (&old, &new) { (Outcome::Value(o), Outcome::Value(n)) => o.units(n), _ => u128::MAX };
        assert!(u <= NEAR_UNITS, "{} differs from 0.6.7 by {} units on input {:?}: old {:?}, new {:?}", self.name, u, input, old, new);
        self.worst = self.worst.max(u);
        self.cases += 1;
        self.corrected += 1;
        self.values += 1;
    }
    /// As `report`, for inputs chosen inside the function's domain: a gate
    /// that only ever compared errors or panics would prove nothing.
    fn report_in_domain(&self) {
        self.report();
        assert!(self.values * 2 >= self.cases, "{}: too few in-range cases", self.name);
    }
}

/// Keep the expected panics out of the test output; report every other one.
fn silence_panics() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !EXPECTING_PANICS.with(|e| e.get()) {
                default(info);
            }
        }));
    });
}

// ---- the gates ----------------------------------------------------------------

/// The correctly rounded value of a unary function at the new crate's
/// format, from the sure path (`None` outside the domain or the word).
fn oracle(name: &str, r: Raw) -> Option<Raw> {
    use g_math::fixed_point::domains::binary_fixed::transcendental::sure;
    let f = FRAC;
    let one: Raw = <Raw as WideInt>::one().shl_bits(f);
    // |v|, with the word's minimum (no magnitude) mapped to the maximum
    let mag = |v: Raw| if v.is_negative() { v.checked_neg_wide().unwrap_or(<Raw as WideInt>::one().shl_bits(<Raw as WideInt>::BITS - 2)) } else { v };
    match name {
        "exp" => sure::exp_raw_to_raw(r, f),
        "sinh" => sure::sinh_cosh_to_raw(r, f, false),
        "cosh" => sure::sinh_cosh_to_raw(r, f, true),
        "tanh" => sure::tanh_to_raw(r, f),
        "sin" => sure::sin_cos_to_raw(r, f, false),
        "cos" => sure::sin_cos_to_raw(r, f, true),
        "tan" => sure::tan_to_raw(r, f),
        "atan" => sure::atan_to_raw(r, f),
        "asinh" => sure::asinh_to_raw(r, f),
        "ln" => sure::ln_to_raw(r, f),
        "sqrt" => sure::sqrt_to_raw(r, f),
        "asin" => if mag(r) > one { None } else { sure::asin_acos_to_raw(r, f, false) },
        "acos" => if mag(r) > one { None } else { sure::asin_acos_to_raw(r, f, true) },
        "atanh" => if !(mag(r) < one) { None } else { sure::atanh_to_raw(r, f) },
        "acosh" => sure::acosh_to_raw(r, f),
        _ => None,
    }
}

macro_rules! unary_gate {
    ($t:ident, $report:ident, $inputs:expr; $($name:literal => $try_m:ident / $m:ident),+ $(,)?) => {$(
        let mut tally = Tally::new($name);
        let mut infallible = Tally::new(concat!($name, " (panicking)"));
        for &r in $inputs.iter() {
            tally.check_oracle(
                fallible(|| raw::old(r).$try_m().map(raw::old_raw)),
                fallible(|| raw::new(r).$try_m().map(raw::new_raw)),
                || oracle($name, r),
                &r,
            );
            infallible.check_oracle(value(|| raw::old_raw(raw::old(r).$m())), value(|| raw::new_raw(raw::new(r).$m())), || oracle($name, r), &r);
        }
        tally.$report();
        infallible.$report();
        $t += tally.cases + infallible.cases;
    )+};
}

#[test]
fn arithmetic_is_bit_identical_to_0_6_5() {
    silence_panics();
    let n = count(40_000);
    let a = inputs(0x0A11_CE55, n, WIDTH - 1);
    let b = inputs(0xB0B0_B0B0, n, WIDTH - 1);
    // operands of a size whose product or quotient usually fits
    let small_a = inputs(0x5EED_0001, n, WIDTH / 2 + FRAC / 2);
    let small_b = inputs(0x5EED_0002, n, WIDTH / 2 + FRAC / 2);

    let mut add = Tally::new("add");
    let mut sub = Tally::new("sub");
    let mut neg = Tally::new("neg");
    let mut mul = Tally::new("mul");
    let mut div = Tally::new("div");
    let mut ops = Tally::new("operators");
    let mut cmp = Tally::new("ordering");
    for (pairs, tight) in [(a.iter().zip(b.iter()), false), (small_a.iter().zip(small_b.iter()), true)] {
        for (&x, &y) in pairs {
            let (ox, oy, nx, ny) = (raw::old(x), raw::old(y), raw::new(x), raw::new(y));
            if !tight {
                add.check(fallible(|| ox.try_add(oy).map(raw::old_raw)), fallible(|| nx.try_add(ny).map(raw::new_raw)), &(x, y));
                sub.check(fallible(|| ox.try_sub(oy).map(raw::old_raw)), fallible(|| nx.try_sub(ny).map(raw::new_raw)), &(x, y));
                neg.check(fallible(|| ox.try_neg().map(raw::old_raw)), fallible(|| nx.try_neg().map(raw::new_raw)), &x);
                cmp.check(
                    value(|| (ox < oy, ox == oy, ox.is_negative(), ox.is_zero(), raw::old_raw(ox.abs_or_self()))),
                    value(|| (nx < ny, nx == ny, nx.is_negative(), nx.is_zero(), raw::new_raw(nx.abs_or_self()))),
                    &(x, y),
                );
            }
            mul.check(fallible(|| ox.try_mul(oy).map(raw::old_raw)), fallible(|| nx.try_mul(ny).map(raw::new_raw)), &(x, y));
            div.check(fallible(|| ox.try_div(oy).map(raw::old_raw)), fallible(|| nx.try_div(ny).map(raw::new_raw)), &(x, y));
            // the panicking operators panic exactly where the old ones did
            ops.check(
                value(|| [raw::old_raw(ox + oy), raw::old_raw(ox - oy), raw::old_raw(ox * oy)]),
                value(|| [raw::new_raw(nx + ny), raw::new_raw(nx - ny), raw::new_raw(nx * ny)]),
                &(x, y),
            );
            ops.check(value(|| raw::old_raw(ox / oy)), value(|| raw::new_raw(nx / ny)), &(x, y));
        }
    }
    for t in [&add, &sub, &neg, &mul, &div, &cmp] {
        t.report();
    }
    println!("{:<16} {:>6} cases, {} panics, 0 differences", ops.name, ops.cases, ops.panics);
}

/// `abs` panics at the minimum; the comparison above wants a total function.
trait AbsOrSelf: Sized {
    fn abs_or_self(self) -> Self;
}
impl AbsOrSelf for Old {
    fn abs_or_self(self) -> Self { quiet(|| self.abs()).unwrap_or(self) }
}
impl AbsOrSelf for New {
    fn abs_or_self(self) -> Self { quiet(|| self.abs()).unwrap_or(self) }
}

#[test]
fn conversions_are_bit_identical_to_0_6_5() {
    silence_panics();
    let n = count(40_000);
    let raws = inputs(0xC0DE_C0DE, n, WIDTH - 1);

    let mut to_int = Tally::new("to_int");
    let mut to_float = Tally::new("to_f64/to_f32");
    let mut display = Tally::new("to_string");
    for &r in raws.iter() {
        to_int.check(fallible(|| raw::old(r).try_to_int()), fallible(|| raw::new(r).try_to_int()), &r);
        to_float.check(
            value(|| (raw::old(r).to_f64().to_bits(), raw::old(r).to_f32().to_bits())),
            value(|| (raw::new(r).to_f64().to_bits(), raw::new(r).to_f32().to_bits())),
            &r,
        );
    }
    // Display runs the canonical formatter: fewer cases
    for &r in raws.iter().step_by(8) {
        display.check(value(|| raw::old(r).to_string()), value(|| raw::new(r).to_string()), &r);
    }

    let mut rng = Rng(0x1234_5678);
    let mut from_int = Tally::new("from_int");
    let mut from_float = Tally::new("from_f64/f32");
    for i in 0..n {
        let bits = rng.next();
        // integers of every magnitude; floats from raw bit patterns (NaN,
        // infinities, subnormals and all) and from in-range values
        let v = (bits as i32) >> (rng.next() % 32);
        from_int.check(fallible(|| Old::try_from_int(v).map(raw::old_raw)), fallible(|| New::try_from_int(v).map(raw::new_raw)), &v);
        let f = if i % 2 == 0 {
            f64::from_bits(bits)
        } else {
            // sign, an exponent near the format's range, random fraction
            let exp = 1023 + (rng.next() % (2 * WIDTH as u64 + 8)) as i64 - (WIDTH as i64 + 4);
            f64::from_bits((bits & 0x800F_FFFF_FFFF_FFFF) | ((exp as u64 & 0x7FF) << 52))
        };
        from_float.check(fallible(|| Old::try_from_f64(f).map(raw::old_raw)), fallible(|| New::try_from_f64(f).map(raw::new_raw)), &f.to_bits());
        let g = f as f32;
        from_float.check(fallible(|| Old::try_from_f32(g).map(raw::old_raw)), fallible(|| New::try_from_f32(g).map(raw::new_raw)), &g.to_bits());
    }

    // decimal literals: digit strings of every length, with and without exponents
    let mut parse = Tally::new("try_from_str");
    for _ in 0..n / 8 {
        let mut s = String::new();
        if rng.next() % 2 == 0 {
            s.push('-');
        }
        for _ in 0..1 + rng.next() % 6 {
            s.push((b'0' + (rng.next() % 10) as u8) as char);
        }
        let frac = rng.next() % 40;
        if frac > 0 {
            s.push('.');
            for _ in 0..frac {
                s.push((b'0' + (rng.next() % 10) as u8) as char);
            }
        }
        if rng.next() % 4 == 0 {
            s.push_str(&format!("e{}", (rng.next() % 13) as i64 - 8));
        }
        parse.check(fallible(|| Old::try_from_str(&s).map(raw::old_raw)), fallible(|| New::try_from_str(&s).map(raw::new_raw)), &s);
    }
    // named constants: a difference must be the correctly rounded constant,
    // its 120 mpmath digits through the exact decimal parser (0.6.7's sqrt 2
    // carried 253 bits, 9 units off at Q256.256)
    let digits = |name: &str| -> Option<&'static str> { match name {
        "pi" => Some("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647"),
        "e" => Some("2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427427466391932003059922"),
        "sqrt2" => Some("1.414213562373095048801688724209698078569671875376948073176679737990732478462107038850387534327641572735013846230912297025"),
        _ => None,
    } };
    for literal in ["pi", "1/3", "0x1F", "0b101", "0t12", "0.3...", "", " ", "abc", "1e", "--1", "e", "sqrt2"] {
        let old = fallible(|| Old::try_from_str(literal).map(raw::old_raw));
        let new = fallible(|| New::try_from_str(literal).map(raw::new_raw));
        match digits(literal) {
            Some(d) => parse.check_oracle(old, new, || New::try_from_str(d).ok().map(raw::new_raw), &literal),
            None => parse.check(old, new, &literal),
        }
    }
    for t in [&to_int, &to_float, &display, &from_int, &from_float, &parse] {
        t.report();
    }
}

#[test]
fn transcendentals_are_bit_identical_to_0_6_5() {
    silence_panics();
    let n = count(6_000);
    let mut total = 0usize;
    // Arguments inside each function's domain, at magnitudes where the
    // result is in range; then arguments of every magnitude and sign, where
    // most calls must fail the same way in both crates.
    let near = inputs(0x7EA_0001, n, FRAC + 2);
    let positive: Vec<Raw> = near.iter().map(|&r| raw::new_raw(raw::new(r).abs_or_self())).collect();
    let unit = inputs(0x7EA_0004, n, FRAC);
    let above_one: Vec<Raw> = unit
        .iter()
        .filter_map(|&r| {
            let a = raw::new(r).abs_or_self();
            New::one().try_add(a).and_then(|v| v.try_add(a)).ok().map(raw::new_raw)
        })
        .collect();
    let any = inputs(0x7EA_0002, n / 4, WIDTH - 1);
    unary_gate!(total, report_in_domain, &near;
        "exp" => try_exp / exp,
        "sin" => try_sin / sin,
        "cos" => try_cos / cos,
        "tan" => try_tan / tan,
        "atan" => try_atan / atan,
        "sinh" => try_sinh / sinh,
        "cosh" => try_cosh / cosh,
        "tanh" => try_tanh / tanh,
        "asinh" => try_asinh / asinh,
    );
    unary_gate!(total, report_in_domain, &positive;
        "ln" => try_ln / ln,
        "sqrt" => try_sqrt / sqrt,
        "inv_sqrt" => try_inv_sqrt / inv_sqrt,
    );
    unary_gate!(total, report_in_domain, &unit;
        "asin" => try_asin / asin,
        "acos" => try_acos / acos,
        "atanh" => try_atanh / atanh,
    );
    unary_gate!(total, report_in_domain, &above_one;
        "acosh" => try_acosh / acosh,
    );
    unary_gate!(total, report, &any;
        "exp" => try_exp / exp,
        "ln" => try_ln / ln,
        "sqrt" => try_sqrt / sqrt,
        "inv_sqrt" => try_inv_sqrt / inv_sqrt,
        "sin" => try_sin / sin,
        "cos" => try_cos / cos,
        "tan" => try_tan / tan,
        "atan" => try_atan / atan,
        "asin" => try_asin / asin,
        "acos" => try_acos / acos,
        "sinh" => try_sinh / sinh,
        "cosh" => try_cosh / cosh,
        "tanh" => try_tanh / tanh,
        "asinh" => try_asinh / asinh,
        "acosh" => try_acosh / acosh,
        "atanh" => try_atanh / atanh,
    );

    // fused pairs and the two-argument functions
    let mut sincos = Tally::new("sincos");
    let mut sinhcosh = Tally::new("sinhcosh");
    let mut pow = Tally::new("pow");
    let mut atan2 = Tally::new("atan2");
    let ys = inputs(0x7EA_0003, n, FRAC + 2);
    for (&x, &y) in near.iter().zip(ys.iter()) {
        let (ox, oy, nx, ny) = (raw::old(x), raw::old(y), raw::new(x), raw::new(y));
        sincos.check_oracle(
            fallible(|| ox.try_sincos().map(|(s, c)| (raw::old_raw(s), raw::old_raw(c)))),
            fallible(|| nx.try_sincos().map(|(s, c)| (raw::new_raw(s), raw::new_raw(c)))),
            || Some((oracle("sin", x)?, oracle("cos", x)?)),
            &x,
        );
        sinhcosh.check_oracle(
            fallible(|| ox.try_sinhcosh().map(|(s, c)| (raw::old_raw(s), raw::old_raw(c)))),
            fallible(|| nx.try_sinhcosh().map(|(s, c)| (raw::new_raw(s), raw::new_raw(c)))),
            || Some((oracle("sinh", x)?, oracle("cosh", x)?)),
            &x,
        );
        let pow_oracle = || if x > <Raw as WideInt>::ZERO { g_math::fixed_point::domains::binary_fixed::transcendental::sure::pow_to_raw(x, y, FRAC) } else { None };
        pow.check_oracle(fallible(|| ox.try_pow(oy).map(raw::old_raw)), fallible(|| nx.try_pow(ny).map(raw::new_raw)), pow_oracle, &(x, y));
        // pow with a non-positive base: ln's sentinel times the exponent
        // overflows the compute tier and `pow` panics on every profile. On
        // Q256.256 the published crate's I2048 fits check read the wrong sign
        // word (fixed in width-types phase 3a), so it wrapped the product and
        // returned a plausible value (pow(-1, -1) = 1.0); there the panic is
        // the corrected outcome and the published value is not compared.
        // Whether the product overflows depends on the exponent (pow(-1, 0) is
        // 1.0 everywhere), so only the pair (published value, tree panic) is
        // the accepted divergence; every other outcome must still agree.
        let base_non_positive = nx <= g_math::fixed_point::FixedPoint::ZERO;
        let (old_pow, new_pow) = (value(|| raw::old_raw(ox.pow(oy))), value(|| raw::new_raw(nx.pow(ny))));
        let wrapped_in_published = cfg!(table_format = "q256_256")
            && base_non_positive
            && matches!(old_pow, Outcome::Value(_))
            && matches!(new_pow, Outcome::Panic);
        if wrapped_in_published {
            pow.check(Outcome::Panic, new_pow, &(x, y));
        } else {
            pow.check_oracle(old_pow, new_pow, pow_oracle, &(x, y));
        }
        let atan2_oracle = || g_math::fixed_point::domains::binary_fixed::transcendental::sure::atan2_to_raw(y, x, FRAC);
        atan2.check_oracle(fallible(|| oy.try_atan2(ox).map(raw::old_raw)), fallible(|| ny.try_atan2(nx).map(raw::new_raw)), atan2_oracle, &(y, x));
        atan2.check_oracle(value(|| raw::old_raw(oy.atan2(ox))), value(|| raw::new_raw(ny.atan2(nx))), atan2_oracle, &(y, x));
    }
    for t in [&sincos, &sinhcosh, &pow, &atan2] {
        t.report();
        total += t.cases;
    }
    println!("{total} transcendental comparisons at {WIDTH}-bit storage, {FRAC} fractional bits: 0 differences");
}


// ---- vectors and matrices ---------------------------------------------------

use g_math::fixed_point::{FixedMatrix as NewMatrix, FixedVector as NewVector};
use g_math_065::fixed_point::{FixedMatrix as OldMatrix, FixedVector as OldVector};

fn old_vec(raws: &[Raw]) -> OldVector { OldVector::from_slice(&raws.iter().map(|&r| raw::old(r)).collect::<Vec<_>>()) }
fn new_vec(raws: &[Raw]) -> NewVector { NewVector::from_slice(&raws.iter().map(|&r| raw::new(r)).collect::<Vec<_>>()) }
fn old_raws(v: &OldVector) -> Vec<Raw> { v.as_slice().iter().map(|&x| raw::old_raw(x)).collect() }
fn new_raws(v: &NewVector) -> Vec<Raw> { v.as_slice().iter().map(|&x| raw::new_raw(x)).collect() }
fn old_mat(r: usize, c: usize, raws: &[Raw]) -> OldMatrix { OldMatrix::from_slice(r, c, &raws.iter().map(|&x| raw::old(x)).collect::<Vec<_>>()) }
fn new_mat(r: usize, c: usize, raws: &[Raw]) -> NewMatrix { NewMatrix::from_slice(r, c, &raws.iter().map(|&x| raw::new(x)).collect::<Vec<_>>()) }
fn old_mraws(m: &OldMatrix) -> Vec<Raw> { (0..m.rows()).flat_map(|i| (0..m.cols()).map(move |j| (i, j))).map(|(i, j)| raw::old_raw(m.get(i, j))).collect() }
fn new_mraws(m: &NewMatrix) -> Vec<Raw> { (0..m.rows()).flat_map(|i| (0..m.cols()).map(move |j| (i, j))).map(|(i, j)| raw::new_raw(m.get(i, j))).collect() }

#[test]
fn vectors_and_matrices_are_bit_identical_to_0_6_5() {
    silence_panics();
    let n = count(600);
    let mut rng = Rng(0x7EC7_0001);
    let mut dot = Tally::new("vector dot/length");
    let mut norm = Tally::new("normalize");
    let mut cross = Tally::new("cross/rotate_pairs");
    let mut ops = Tally::new("vector operators");
    let mut matvec = Tally::new("matvec/matmul");
    let mut mops = Tally::new("matrix operators");
    for i in 0..n {
        // vectors of 1..=40 elements, elements a few units in magnitude; every
        // tenth case long (up to 200, past the realtime AVX2 bound) and every
        // seventh at the full storage range (overflow paths)
        let len = if i % 10 == 9 { 33 + (rng.next() % 168) as usize } else { 1 + (rng.next() % 40) as usize };
        let bits = if i % 7 == 6 { WIDTH - 1 } else { FRAC + 3 };
        let a: Vec<Raw> = (0..len).map(|_| rng.raw(bits)).collect();
        let b: Vec<Raw> = (0..len).map(|_| rng.raw(bits)).collect();
        let (oa, ob, na, nb) = (old_vec(&a), old_vec(&b), new_vec(&a), new_vec(&b));
        dot.check(
            value(|| (raw::old_raw(oa.dot(&ob)), raw::old_raw(oa.length_squared()), raw::old_raw(oa.length()), raw::old_raw(oa.metric_distance_safe(&ob)))),
            value(|| (raw::new_raw(na.dot(&nb)), raw::new_raw(na.length_squared()), raw::new_raw(na.length()), raw::new_raw(na.metric_distance_safe(&nb)))),
            &(&a, &b),
        );
        norm.check(value(|| old_raws(&oa.normalized())), value(|| new_raws(&na.normalized())), &a);
        ops.check(
            value(|| (old_raws(&(&oa + &ob)), old_raws(&(&oa - &ob)), old_raws(&(-&oa)), old_raws(&(oa.clone() * ob[0])))),
            value(|| (new_raws(&(&na + &nb)), new_raws(&(&na - &nb)), new_raws(&(-&na)), new_raws(&(na.clone() * nb[0])))),
            &(&a, &b),
        );
        if len >= 4 {
            let half = len / 2 * 2 / 2;
            let rot = |v: &[Raw]| v[..half].to_vec();
            let (s, c) = (rot(&a), rot(&b));
            cross.check(
                value(|| { let mut v = oa.clone(); v.rotate_pairs(&old_vec(&s).as_slice(), &old_vec(&c).as_slice(), half * 2); old_raws(&v) }),
                value(|| { let mut v = na.clone(); v.rotate_pairs(&new_vec(&s).as_slice(), &new_vec(&c).as_slice(), half * 2); new_raws(&v) }),
                &(&a, &s, &c),
            );
        }
        if len == 3 {
            cross.check(value(|| old_raws(&oa.cross(&ob))), value(|| new_raws(&na.cross(&nb))), &(&a, &b));
        }
        // matrices up to 6 x 6 from the same raws
        let (r, c) = (1 + (rng.next() % 6) as usize, 1 + (rng.next() % 6) as usize);
        let m: Vec<Raw> = (0..r * c).map(|_| rng.raw(bits)).collect();
        let k: Vec<Raw> = (0..c * r).map(|_| rng.raw(bits)).collect();
        let x: Vec<Raw> = (0..c).map(|_| rng.raw(bits)).collect();
        let (om, ok, onm, nk) = (old_mat(r, c, &m), old_mat(c, r, &k), new_mat(r, c, &m), new_mat(c, r, &k));
        matvec.check(
            value(|| (old_raws(&om.mul_vector(&old_vec(&x))), old_mraws(&(&om * &ok)), old_raws(&om.row(0)), old_raws(&om.col(0)))),
            value(|| (new_raws(&onm.mul_vector(&new_vec(&x))), new_mraws(&(&onm * &nk)), new_raws(&onm.row(0)), new_raws(&onm.col(0)))),
            &(&m, &k, &x),
        );
        let (om2, nm2) = (old_mat(r, c, &k[..r * c]), new_mat(r, c, &k[..r * c]));
        mops.check(
            value(|| (old_mraws(&(&om + &om2)), old_mraws(&(&om - &om2)), old_mraws(&(-&om)), old_mraws(&(om.clone() * om2.get(0, 0))), old_mraws(&om.transpose()), old_mraws(&om.kronecker(&om2)))),
            value(|| (new_mraws(&(&onm + &nm2)), new_mraws(&(&onm - &nm2)), new_mraws(&(-&onm)), new_mraws(&(onm.clone() * nm2.get(0, 0))), new_mraws(&onm.transpose()), new_mraws(&onm.kronecker(&nm2)))),
            &(&m, &k),
        );
        if r == c {
            mops.check(value(|| raw::old_raw(om.trace())), value(|| raw::new_raw(onm.trace())), &m);
        }
    }
    for t in [&dot, &norm, &cross, &ops, &matvec, &mops] {
        t.report();
    }
}


// ---- decompositions and derived operations ----------------------------------

#[test]
fn decompositions_are_bit_identical_to_0_6_5() {
    use g_math::fixed_point::imperative::decompose as nd;
    use g_math::fixed_point::imperative::derived as nv;
    use g_math_065::fixed_point::imperative::decompose as od;
    use g_math_065::fixed_point::imperative::derived as ov;
    silence_panics();
    let n = count(240);
    let mut rng = Rng(0xDEC0_0001);
    let mut lu = Tally::new("lu/solve/det/inverse");
    let mut qr = Tally::new("qr");
    let mut chol = Tally::new("cholesky/inverse_spd");
    let mut eig = Tally::new("eigen_symmetric");
    let mut svd = Tally::new("svd/pseudoinverse/rank");
    let mut schur = Tally::new("schur");
    let mut norms = Tally::new("norms/condition");
    for i in 0..n {
        // 1 x 1 to 5 x 5, entries a few units in magnitude; every ninth at
        // the storage range (overflow and refusal paths)
        let d = 1 + (rng.next() % 5) as usize;
        let bits = if i % 9 == 8 { WIDTH - 1 } else { FRAC + 2 };
        let m: Vec<Raw> = (0..d * d).map(|_| rng.raw(bits)).collect();
        let b: Vec<Raw> = (0..d).map(|_| rng.raw(bits)).collect();
        let (om, nm) = (old_mat(d, d, &m), new_mat(d, d, &m));
        let (ob, nb) = (old_vec(&b), new_vec(&b));
        // a symmetric form from the same raws (mirrored, not summed: no overflow)
        let sym: Vec<Raw> = (0..d * d).map(|k| { let (i, j) = (k / d, k % d); m[if i <= j { i * d + j } else { j * d + i }] }).collect();
        let (os, ns) = (old_mat(d, d, &sym), new_mat(d, d, &sym));
        lu.check(
            fallible(|| od::lu_decompose(&om).and_then(|f| Ok((old_mraws(&f.l), old_mraws(&f.u), f.perm.clone(), f.num_swaps, old_raws(&f.solve(&ob)?), raw::old_raw(f.determinant()), old_mraws(&f.inverse()?))))),
            fallible(|| nd::lu_decompose(&nm).and_then(|f| Ok((new_mraws(&f.l), new_mraws(&f.u), f.perm.clone(), f.num_swaps, new_raws(&f.solve(&nb)?), raw::new_raw(f.determinant()), new_mraws(&f.inverse()?))))),
            &m,
        );
        lu.check(
            fallible(|| -> Result<_, g_math_065::fixed_point::OverflowDetected> { Ok((old_raws(&ov::solve(&om, &ob)?), raw::old_raw(ov::determinant(&om)?), old_mraws(&ov::inverse(&om)?))) }),
            fallible(|| -> Result<_, g_math::fixed_point::OverflowDetected> { Ok((new_raws(&nv::solve(&nm, &nb)?), raw::new_raw(nv::determinant(&nm)?), new_mraws(&nv::inverse(&nm)?))) }),
            &m,
        );
        qr.check(
            fallible(|| od::qr_decompose(&om).and_then(|f| Ok((old_mraws(&f.q), old_mraws(&f.r), old_raws(&f.solve(&ob)?))))),
            fallible(|| nd::qr_decompose(&nm).and_then(|f| Ok((new_mraws(&f.q), new_mraws(&f.r), new_raws(&f.solve(&nb)?))))),
            &m,
        );
        chol.check(
            fallible(|| od::cholesky_decompose(&os).and_then(|f| Ok((old_mraws(&f.l), old_raws(&f.solve(&ob)?))))),
            fallible(|| nd::cholesky_decompose(&ns).and_then(|f| Ok((new_mraws(&f.l), new_raws(&f.solve(&nb)?))))),
            &m,
        );
        chol.check(fallible(|| ov::inverse_spd(&os).map(|x| old_mraws(&x))), fallible(|| nv::inverse_spd(&ns).map(|x| new_mraws(&x))), &m);
        if bits < WIDTH - 1 {
            // a positive definite form: M^T M plus the identity
            let (opd, npd) = (&(&om.transpose() * &om) + &OldMatrix::identity(d), &(&nm.transpose() * &nm) + &NewMatrix::<g_math::fixed_point::FixedPoint>::identity(d));
            chol.check(
                fallible(|| od::cholesky_decompose(&opd).and_then(|f| Ok((old_mraws(&f.l), old_raws(&f.solve(&ob)?), raw::old_raw(f.determinant()))))),
                fallible(|| nd::cholesky_decompose(&npd).and_then(|f| Ok((new_mraws(&f.l), new_raws(&f.solve(&nb)?), raw::new_raw(f.determinant()))))),
                &m,
            );
            chol.check(fallible(|| ov::solve_spd(&opd, &ob).map(|x| old_raws(&x))), fallible(|| nv::solve_spd(&npd, &nb).map(|x| new_raws(&x))), &m);
        }
        if bits < WIDTH - 1 {
            // the iterative ones on in-range entries only (debug builds are slow at the range)
            eig.check(
                fallible(|| od::eigen_symmetric(&os).map(|e| (old_raws(&e.values), old_mraws(&e.vectors)))),
                fallible(|| nd::eigen_symmetric(&ns).map(|e| (new_raws(&e.values), new_mraws(&e.vectors)))),
                &m,
            );
            svd.check(
                fallible(|| od::svd_decompose(&om).map(|s| (old_mraws(&s.u), old_raws(&s.sigma), old_mraws(&s.vt)))),
                fallible(|| nd::svd_decompose(&nm).map(|s| (new_mraws(&s.u), new_raws(&s.sigma), new_mraws(&s.vt)))),
                &m,
            );
            svd.check(
                fallible(|| -> Result<_, g_math_065::fixed_point::OverflowDetected> { Ok((old_mraws(&ov::pseudoinverse(&om)?), ov::rank(&om)?, raw::old_raw(ov::condition_number_2(&om)?))) }),
                fallible(|| -> Result<_, g_math::fixed_point::OverflowDetected> { Ok((new_mraws(&nv::pseudoinverse(&nm)?), nv::rank(&nm)?, raw::new_raw(nv::condition_number_2(&nm)?))) }),
                &m,
            );
            schur.check(
                fallible(|| od::schur_decompose(&om).map(|s| (old_mraws(&s.q), old_mraws(&s.t)))),
                fallible(|| nd::schur_decompose(&nm).map(|s| (new_mraws(&s.q), new_mraws(&s.t)))),
                &m,
            );
        }
        norms.check(
            fallible(|| -> Result<_, g_math_065::fixed_point::OverflowDetected> { Ok((raw::old_raw(ov::frobenius_norm(&om)), raw::old_raw(ov::norm_1(&om)), raw::old_raw(ov::norm_inf(&om)), raw::old_raw(ov::condition_number_1(&om)?))) }),
            fallible(|| -> Result<_, g_math::fixed_point::OverflowDetected> { Ok((raw::new_raw(nv::frobenius_norm(&nm)), raw::new_raw(nv::norm_1(&nm)), raw::new_raw(nv::norm_inf(&nm)), raw::new_raw(nv::condition_number_1(&nm)?))) }),
            &m,
        );
    }
    for t in [&lu, &qr, &chol, &eig, &svd, &schur, &norms] {
        t.report();
    }
}


// ---- matrix functions ---------------------------------------------------------

#[test]
fn matrix_functions_are_bit_identical_to_0_6_5() {
    use g_math::fixed_point::imperative::matrix_functions as nf;
    use g_math_065::fixed_point::imperative::matrix_functions as of;
    silence_panics();
    let n = count(120);
    let mut rng = Rng(0xF00D_0001);
    let mut exp = Tally::new("matrix_exp");
    let mut sqrt = Tally::new("matrix_sqrt");
    let mut log = Tally::new("matrix_log");
    let mut pow = Tally::new("matrix_pow");
    for _ in 0..n {
        let d = 1 + (rng.next() % 4) as usize;
        // entries below one in magnitude: the functions converge and stay in range
        let m: Vec<Raw> = (0..d * d).map(|_| rng.raw(FRAC - 1)).collect();
        let (om, nm) = (old_mat(d, d, &m), new_mat(d, d, &m));
        // I + M for the square root and the logarithm (near the identity)
        let (oi, ni) = (&OldMatrix::identity(d) + &om, &NewMatrix::<g_math::fixed_point::FixedPoint>::identity(d) + &nm);
        exp.check(fallible(|| of::matrix_exp(&om).map(|x| old_mraws(&x))), fallible(|| nf::matrix_exp(&nm).map(|x| new_mraws(&x))), &m);
        sqrt.check(fallible(|| of::matrix_sqrt(&oi).map(|x| old_mraws(&x))), fallible(|| nf::matrix_sqrt(&ni).map(|x| new_mraws(&x))), &m);
        log.check(fallible(|| of::matrix_log(&oi).map(|x| old_mraws(&x))), fallible(|| nf::matrix_log(&ni).map(|x| new_mraws(&x))), &m);
        let p = rng.raw(FRAC + 1);
        pow.check(fallible(|| of::matrix_pow(&oi, raw::old(p)).map(|x| old_mraws(&x))), fallible(|| nf::matrix_pow(&ni, raw::new(p)).map(|x| new_mraws(&x))), &(&m, p));
    }
    for t in [&exp, &sqrt, &log, &pow] {
        t.report();
    }
}

// ---- geometry layers, dynamics, tensors (width-types phase 4a) -------------

mod geo_old {
    pub use g_math_065::fixed_point::imperative::curvature::{christoffel, riemann_curvature, scalar_curvature, HyperbolicMetric, SphereMetric};
    pub use g_math_065::fixed_point::imperative::fiber_bundle::{BundleConnection, VectorBundle};
    pub use g_math_065::fixed_point::imperative::lie_group::{LieGroup, SE3, SO3, SOn};
    pub use g_math_065::fixed_point::imperative::manifold::{HyperbolicSpace, Manifold, SPDManifold, Sphere};
    pub use g_math_065::fixed_point::imperative::ode::{ode_fn, rk4_step, rk45_integrate, verlet_integrate, HamiltonianSystem, Rk45Config};
    pub use g_math_065::fixed_point::imperative::projective::{cross_ratio_1d, from_homogeneous, stereo_project, stereo_unproject, to_homogeneous, Moebius};
    pub use g_math_065::fixed_point::imperative::tensor::{contract, symmetrize, trace, Tensor};
    pub use g_math_065::fixed_point::imperative::tensor_decompose::{cp_decompose, truncated_svd, tucker_decompose};
}
mod geo_new {
    pub use g_math::fixed_point::imperative::curvature::{christoffel, riemann_curvature, scalar_curvature, HyperbolicMetric, SphereMetric};
    pub use g_math::fixed_point::imperative::fiber_bundle::{BundleConnection, VectorBundle};
    pub use g_math::fixed_point::imperative::lie_group::{LieGroup, SE3, SO3, SOn};
    pub use g_math::fixed_point::imperative::manifold::{HyperbolicSpace, Manifold, SPDManifold, Sphere};
    pub use g_math::fixed_point::imperative::ode::{ode_fn, rk4_step, rk45_integrate, verlet_integrate, HamiltonianSystem, Rk45Config};
    pub use g_math::fixed_point::imperative::projective::{cross_ratio_1d, from_homogeneous, stereo_project, stereo_unproject, to_homogeneous, Moebius};
    pub use g_math::fixed_point::imperative::tensor::{contract, symmetrize, trace, Tensor};
    pub use g_math::fixed_point::imperative::tensor_decompose::{cp_decompose, truncated_svd, tucker_decompose};
}

/// The harmonic oscillator as a Hamiltonian system for both crates.
struct Oscillator;
impl geo_old::HamiltonianSystem for Oscillator {
    fn force(&self, q: &OldVector, _p: &OldVector) -> OldVector { -q }
    fn velocity(&self, _q: &OldVector, p: &OldVector) -> OldVector { p.clone() }
    fn energy(&self, q: &OldVector, p: &OldVector) -> Old { (q.dot(q) + p.dot(p)) / Old::from_int(2) }
}
impl geo_new::HamiltonianSystem for Oscillator {
    fn force(&self, q: &NewVector, _p: &NewVector) -> NewVector { -q }
    fn velocity(&self, _q: &NewVector, p: &NewVector) -> NewVector { p.clone() }
    fn energy(&self, q: &NewVector, p: &NewVector) -> New { (q.dot(q) + p.dot(p)) / New::from_int(2) }
}

fn old_tensor(shape: &[usize], raws: &[Raw]) -> geo_old::Tensor { geo_old::Tensor::from_data(shape, &raws.iter().map(|&r| raw::old(r)).collect::<Vec<_>>()) }
fn new_tensor(shape: &[usize], raws: &[Raw]) -> geo_new::Tensor { geo_new::Tensor::from_data(shape, &raws.iter().map(|&r| raw::new(r)).collect::<Vec<_>>()) }
fn old_traws(t: &geo_old::Tensor) -> Vec<Raw> { t.data().iter().map(|&x| raw::old_raw(x)).collect() }
fn new_traws(t: &geo_new::Tensor) -> Vec<Raw> { t.data().iter().map(|&x| raw::new_raw(x)).collect() }

#[test]
fn geometry_is_bit_identical_to_0_6_5() {
    use geo_new as n_;
    use geo_new::{BundleConnection as _, LieGroup as _, Manifold as _};
    use geo_old as o_;
    use geo_old::{BundleConnection as _, LieGroup as _, Manifold as _};
    silence_panics();
    let n = count(60);
    let mut rng = Rng(0x6E0_0001);
    let t = |name: &'static str| Tally::new(name);
    let (mut sph, mut hyp, mut spd, mut so3, mut se3, mut son, mut curv, mut ode, mut proj, mut ten, mut fib) = (
        t("sphere"), t("hyperbolic"), t("spd"), t("so3"), t("se3"), t("son"), t("curvature"), t("ode"), t("projective"), t("tensors"), t("fiber bundle"),
    );
    let unit_old = |v: OldVector| { let l = v.length(); OldVector::from_slice(&(0..v.len()).map(|i| v[i] / l).collect::<Vec<_>>()) };
    let unit_new = |v: NewVector| { let l = v.length(); NewVector::from_slice(&(0..v.len()).map(|i| v[i] / l).collect::<Vec<_>>()) };
    for _ in 0..n {
        let small = |rng: &mut Rng, k: usize, bits: u32| (0..k).map(|_| rng.raw(bits)).collect::<Vec<Raw>>();
        // sphere S^2: unit points, the tangent of q at p
        let (a, b) = (small(&mut rng, 3, FRAC), small(&mut rng, 3, FRAC));
        let (op, oq) = (unit_old(old_vec(&a)), unit_old(old_vec(&b)));
        let (np, nq) = (unit_new(new_vec(&a)), unit_new(new_vec(&b)));
        let s_o = o_::Sphere { dim: 2 };
        let s_n = n_::Sphere { dim: 2 };
        sph.check_near(fallible(|| s_o.distance(&op, &oq).map(raw::old_raw)), fallible(|| s_n.distance(&np, &nq).map(raw::new_raw)), &(&a, &b));
        sph.check_near(fallible(|| s_o.log_map(&op, &oq).map(|v| old_raws(&v))), fallible(|| s_n.log_map(&np, &nq).map(|v| new_raws(&v))), &(&a, &b));
        let (ot, nt) = (o_::Manifold::log_map(&s_o, &op, &oq), n_::Manifold::log_map(&s_n, &np, &nq));
        if let (Ok(ot), Ok(nt)) = (ot, nt) {
            sph.check_near(fallible(|| s_o.exp_map(&op, &ot).map(|v| old_raws(&v))), fallible(|| s_n.exp_map(&np, &nt).map(|v| new_raws(&v))), &(&a, &b));
            sph.check_near(fallible(|| s_o.parallel_transport(&op, &oq, &ot).map(|v| old_raws(&v))), fallible(|| s_n.parallel_transport(&np, &nq, &nt).map(|v| new_raws(&v))), &(&a, &b));
        }
        // hyperboloid H^2: x0 = sqrt(1 + x1^2 + x2^2)
        let (a, b) = (small(&mut rng, 2, FRAC - 1), small(&mut rng, 2, FRAC - 1));
        let lift_old = |r: &[Raw]| { let (x, y) = (raw::old(r[0]), raw::old(r[1])); OldVector::from_slice(&[(Old::one() + x * x + y * y).sqrt(), x, y]) };
        let lift_new = |r: &[Raw]| { let (x, y) = (raw::new(r[0]), raw::new(r[1])); NewVector::from_slice(&[(New::one() + x * x + y * y).sqrt(), x, y]) };
        let (op, oq, np, nq) = (lift_old(&a), lift_old(&b), lift_new(&a), lift_new(&b));
        let h_o = o_::HyperbolicSpace { dim: 2 };
        let h_n = n_::HyperbolicSpace { dim: 2 };
        hyp.check_near(fallible(|| h_o.distance(&op, &oq).map(raw::old_raw)), fallible(|| h_n.distance(&np, &nq).map(raw::new_raw)), &(&a, &b));
        hyp.check_near(fallible(|| h_o.log_map(&op, &oq).map(|v| old_raws(&v))), fallible(|| h_n.log_map(&np, &nq).map(|v| new_raws(&v))), &(&a, &b));
        if let (Ok(ot), Ok(nt)) = (h_o.log_map(&op, &oq), h_n.log_map(&np, &nq)) {
            hyp.check_near(fallible(|| h_o.exp_map(&op, &ot).map(|v| old_raws(&v))), fallible(|| h_n.exp_map(&np, &nt).map(|v| new_raws(&v))), &(&a, &b));
        }
        // SPD(2): M^T M + I, tangents symmetric
        let (m, w) = (small(&mut rng, 4, FRAC - 1), small(&mut rng, 4, FRAC - 2));
        let spd_old = |r: &[Raw]| { let m = old_mat(2, 2, r); &(&m.transpose() * &m) + &OldMatrix::identity(2) };
        let spd_new = |r: &[Raw]| { let m = new_mat(2, 2, r); &(&m.transpose() * &m) + &NewMatrix::<New>::identity(2) };
        let (op, oq) = (spd_old(&m), spd_old(&w));
        let (np, nq) = (spd_new(&m), spd_new(&w));
        let sym = |r: &[Raw]| vec![r[0], r[1], r[1], r[3]];
        let sv = sym(&w);
        let (opv, oqv, otv) = (old_vec(&old_mraws(&op)), old_vec(&old_mraws(&oq)), old_vec(&sv));
        let (npv, nqv, ntv) = (new_vec(&new_mraws(&np)), new_vec(&new_mraws(&nq)), new_vec(&sv));
        let p_o = o_::SPDManifold { n: 2 };
        let p_n = n_::SPDManifold { n: 2 };
        spd.check_near(fallible(|| p_o.distance(&opv, &oqv).map(raw::old_raw)), fallible(|| p_n.distance(&npv, &nqv).map(raw::new_raw)), &(&m, &w));
        spd.check_near(fallible(|| p_o.exp_map(&opv, &otv).map(|v| old_raws(&v))), fallible(|| p_n.exp_map(&npv, &ntv).map(|v| new_raws(&v))), &(&m, &w));
        spd.check_near(fallible(|| p_o.log_map(&opv, &oqv).map(|v| old_raws(&v))), fallible(|| p_n.log_map(&npv, &nqv).map(|v| new_raws(&v))), &(&m, &w));
        // SO(3), SE(3), SO(n)
        let xi = small(&mut rng, 3, FRAC);
        let (oxi, nxi) = (old_vec(&xi), new_vec(&xi));
        so3.check_near(fallible(|| o_::SO3.lie_exp(&oxi).map(|g| old_mraws(&g))), fallible(|| n_::SO3.lie_exp(&nxi).map(|g| new_mraws(&g))), &xi);
        if let (Ok(og), Ok(ng)) = (o_::SO3.lie_exp(&oxi), n_::SO3.lie_exp(&nxi)) {
            so3.check_near(fallible(|| o_::SO3.lie_log(&og).map(|v| old_raws(&v))), fallible(|| n_::SO3.lie_log(&ng).map(|v| new_raws(&v))), &xi);
            so3.check_near(fallible(|| o_::SO3.adjoint(&og, &oxi).map(|v| old_raws(&v))), fallible(|| n_::SO3.adjoint(&ng, &nxi).map(|v| new_raws(&v))), &xi);
            son.check_near(fallible(|| o_::SOn { n: 3 }.lie_log(&og).map(|v| old_raws(&v))), fallible(|| n_::SOn { n: 3 }.lie_log(&ng).map(|v| new_raws(&v))), &xi);
        }
        son.check_near(fallible(|| o_::SOn { n: 3 }.lie_exp(&oxi).map(|g| old_mraws(&g))), fallible(|| n_::SOn { n: 3 }.lie_exp(&nxi).map(|g| new_mraws(&g))), &xi);
        let xi6 = small(&mut rng, 6, FRAC);
        let (oxi6, nxi6) = (old_vec(&xi6), new_vec(&xi6));
        se3.check_near(fallible(|| o_::SE3.lie_exp(&oxi6).map(|g| old_mraws(&g))), fallible(|| n_::SE3.lie_exp(&nxi6).map(|g| new_mraws(&g))), &xi6);
        if let (Ok(og), Ok(ng)) = (o_::SE3.lie_exp(&oxi6), n_::SE3.lie_exp(&nxi6)) {
            se3.check_near(fallible(|| o_::SE3.lie_log(&og).map(|v| old_raws(&v))), fallible(|| n_::SE3.lie_log(&ng).map(|v| new_raws(&v))), &xi6);
        }
        // curvature on the round sphere (radius in (1, 2)) and the hyperbolic plane
        let r = rng.raw(FRAC - 1);
        let ang = small(&mut rng, 2, FRAC - 1); // theta, phi in (-0.5, 0.5) + offsets
        let (otp, ntp) = (
            OldVector::from_slice(&[raw::old(ang[0]).abs_or_self() + Old::one() / Old::from_int(2), raw::old(ang[1])]),
            NewVector::from_slice(&[raw::new(ang[0]).abs_or_self() + New::one() / New::from_int(2), raw::new(ang[1])]),
        );
        let (sm_o, sm_n) = (o_::SphereMetric { radius: Old::one() + raw::old(r).abs_or_self() }, n_::SphereMetric { radius: New::one() + raw::new(r).abs_or_self() });
        curv.check_near(fallible(|| o_::christoffel(&sm_o, &otp).map(|t| old_traws(&t))), fallible(|| n_::christoffel(&sm_n, &ntp).map(|t| new_traws(&t))), &(r, &ang));
        curv.check_near(fallible(|| o_::riemann_curvature(&sm_o, &otp).map(|t| old_traws(&t))), fallible(|| n_::riemann_curvature(&sm_n, &ntp).map(|t| new_traws(&t))), &(r, &ang));
        curv.check_near(fallible(|| o_::scalar_curvature(&sm_o, &otp).map(raw::old_raw)), fallible(|| n_::scalar_curvature(&sm_n, &ntp).map(raw::new_raw)), &(r, &ang));
        let (ohp, nhp) = (OldVector::from_slice(&[raw::old(ang[0]), raw::old(ang[1])]), NewVector::from_slice(&[raw::new(ang[0]), raw::new(ang[1])]));
        curv.check_near(fallible(|| o_::scalar_curvature(&o_::HyperbolicMetric, &ohp).map(raw::old_raw)), fallible(|| n_::scalar_curvature(&n_::HyperbolicMetric, &nhp).map(raw::new_raw)), &ang);
        // ODE: one RK4 step and a short RK45 run of the oscillator, Verlet
        let x = small(&mut rng, 2, FRAC);
        let h = rng.raw(FRAC - 3);
        let (ox, nx, oh, nh) = (old_vec(&x), new_vec(&x), raw::old(h).abs_or_self(), raw::new(h).abs_or_self());
        let osc_o = o_::ode_fn(|_t: Old, x: &OldVector| OldVector::from_slice(&[x[1], -x[0]]));
        let osc_n = n_::ode_fn(|_t: New, x: &NewVector| NewVector::from_slice(&[x[1], -x[0]]));
        ode.check_near(value(|| old_raws(&o_::rk4_step(&osc_o, Old::ZERO, &ox, oh))), value(|| new_raws(&n_::rk4_step(&osc_n, New::ZERO, &nx, nh))), &(&x, h));
        let end_o = oh * Old::from_int(3);
        let end_n = nh * New::from_int(3);
        ode.check_near(
            fallible(|| o_::rk45_integrate(&osc_o, &ox, Old::ZERO, end_o, &o_::Rk45Config::new(oh, oh)).map(|(pts, s)| (pts.iter().map(|p| (raw::old_raw(p.t), old_raws(&p.x))).collect::<Vec<_>>(), s))),
            fallible(|| n_::rk45_integrate(&osc_n, &nx, New::ZERO, end_n, &n_::Rk45Config::new(nh, nh)).map(|(pts, s)| (pts.iter().map(|p| (raw::new_raw(p.t), new_raws(&p.x))).collect::<Vec<_>>(), s))),
            &(&x, h),
        );
        let (oq, nq) = (old_vec(&x[..1]), new_vec(&x[..1]));
        ode.check_near(
            value(|| o_::verlet_integrate(&Oscillator, &oq, &oq, Old::ZERO, end_o, oh).iter().map(|p| (old_raws(&p.q), old_raws(&p.p), raw::old_raw(p.energy))).collect::<Vec<_>>()),
            value(|| n_::verlet_integrate(&Oscillator, &nq, &nq, New::ZERO, end_n, nh).iter().map(|p| (new_raws(&p.q), new_raws(&p.p), raw::new_raw(p.energy))).collect::<Vec<_>>()),
            &(&x, h),
        );
        // projective: Moebius maps, the cross ratio, stereographic projection
        let c = small(&mut rng, 5, FRAC);
        let (mo, mn) = (o_::Moebius::new(raw::old(c[0]), raw::old(c[1]), raw::old(c[2]), raw::old(c[3])), n_::Moebius::new(raw::new(c[0]), raw::new(c[1]), raw::new(c[2]), raw::new(c[3])));
        proj.check_near(fallible(|| mo.apply(raw::old(c[4])).map(raw::old_raw)), fallible(|| mn.apply(raw::new(c[4])).map(raw::new_raw)), &c);
        proj.check_near(value(|| old_mraws(&mo.compose(&mo.inverse()).to_matrix())), value(|| new_mraws(&mn.compose(&mn.inverse()).to_matrix())), &c);
        let q4 = small(&mut rng, 4, FRAC + 1);
        proj.check_near(
            fallible(|| o_::cross_ratio_1d(raw::old(q4[0]), raw::old(q4[1]), raw::old(q4[2]), raw::old(q4[3])).map(raw::old_raw)),
            fallible(|| n_::cross_ratio_1d(raw::new(q4[0]), raw::new(q4[1]), raw::new(q4[2]), raw::new(q4[3])).map(raw::new_raw)),
            &q4,
        );
        let (a3, b3) = (small(&mut rng, 3, FRAC), small(&mut rng, 2, FRAC));
        let (op3, np3) = (unit_old(old_vec(&a3)), unit_new(new_vec(&a3)));
        proj.check_near(fallible(|| o_::stereo_project(&op3).map(|v| old_raws(&v))), fallible(|| n_::stereo_project(&np3).map(|v| new_raws(&v))), &a3);
        proj.check_near(value(|| old_raws(&o_::stereo_unproject(&old_vec(&b3)))), value(|| new_raws(&n_::stereo_unproject(&new_vec(&b3)))), &b3);
        proj.check_near(fallible(|| o_::from_homogeneous(&o_::to_homogeneous(&old_vec(&a3))).map(|v| old_raws(&v))), fallible(|| n_::from_homogeneous(&n_::to_homogeneous(&new_vec(&a3))).map(|v| new_raws(&v))), &a3);
        // tensors: contraction, trace, symmetrisation, the three decompositions
        let (ta, tb) = (small(&mut rng, 6, FRAC), small(&mut rng, 6, FRAC));
        ten.check_near(value(|| old_traws(&o_::contract(&old_tensor(&[2, 3], &ta), &old_tensor(&[3, 2], &tb), &[(1, 0)]))), value(|| new_traws(&n_::contract(&new_tensor(&[2, 3], &ta), &new_tensor(&[3, 2], &tb), &[(1, 0)]))), &(&ta, &tb));
        let t8 = small(&mut rng, 8, FRAC);
        ten.check_near(value(|| old_traws(&o_::trace(&old_tensor(&[2, 2, 2], &t8), 0, 2))), value(|| new_traws(&n_::trace(&new_tensor(&[2, 2, 2], &t8), 0, 2))), &t8);
        ten.check_near(value(|| old_traws(&o_::symmetrize(&old_tensor(&[2, 2, 2], &t8), &[0, 1]))), value(|| new_traws(&n_::symmetrize(&new_tensor(&[2, 2, 2], &t8), &[0, 1]))), &t8);
        let m9 = small(&mut rng, 9, FRAC);
        ten.check_near(fallible(|| o_::truncated_svd(&old_mat(3, 3, &m9), 2).map(|d| old_mraws(&d.reconstruct()))), fallible(|| n_::truncated_svd(&new_mat(3, 3, &m9), 2).map(|d| new_mraws(&d.reconstruct()))), &m9);
        ten.check_near(fallible(|| o_::tucker_decompose(&old_tensor(&[2, 2, 2], &t8), &[2, 2, 2]).map(|d| old_traws(&d.reconstruct()))), fallible(|| n_::tucker_decompose(&new_tensor(&[2, 2, 2], &t8), &[2, 2, 2]).map(|d| new_traws(&d.reconstruct()))), &t8);
        ten.check_near(
            fallible(|| o_::cp_decompose(&old_tensor(&[2, 2, 2], &t8), 2, 8, Old::ZERO).map(|d| old_traws(&d.reconstruct(&[2, 2, 2])))),
            fallible(|| n_::cp_decompose(&new_tensor(&[2, 2, 2], &t8), 2, 8, New::ZERO).map(|d| new_traws(&d.reconstruct(&[2, 2, 2])))),
            &t8,
        );
        // a vector bundle with a connection: transport along a short path
        let coeffs = small(&mut rng, 8, FRAC - 2); // A[a, b, i] for k = 2, n = 2
        let path = [small(&mut rng, 2, FRAC - 2), small(&mut rng, 2, FRAC - 2), small(&mut rng, 2, FRAC - 2)];
        let f0 = small(&mut rng, 2, FRAC);
        let vb_o = o_::VectorBundle::with_connection(2, 2, coeffs.iter().map(|&r| raw::old(r)).collect());
        let vb_n = n_::VectorBundle::with_connection(2, 2, coeffs.iter().map(|&r| raw::new(r)).collect());
        let (opath, npath): (Vec<OldVector>, Vec<NewVector>) = (path.iter().map(|p| old_vec(p)).collect(), path.iter().map(|p| new_vec(p)).collect());
        fib.check_near(fallible(|| vb_o.parallel_transport_along(&opath, &old_vec(&f0)).map(|v| old_raws(&v))), fallible(|| vb_n.parallel_transport_along(&npath, &new_vec(&f0)).map(|v| new_raws(&v))), &(&coeffs, &path, &f0));
    }
    for t in [&sph, &hyp, &spd, &so3, &se3, &son, &curv, &ode, &proj, &ten, &fib] {
        t.report();
    }
}

// ---- fused kernels and the compute tier (width-types phase 4b) ------------

#[test]
fn fused_and_compute_tier_are_bit_identical_to_0_6_5() {
    use g_math::fixed_point::compute_tier as nct;
    use g_math::fixed_point::imperative::fused as nf;
    use g_math_065::fixed_point::compute_tier as oct;
    use g_math_065::fixed_point::imperative::fused as of;
    silence_panics();
    let n = count(80);
    let mut rng = Rng(0xF05E_D001);
    let t = |name: &'static str| Tally::new(name);
    let (mut norms, mut soft, mut mix, mut gates, mut rms, mut ct) = (t("norms"), t("softmax/silu"), t("softmax_mix"), t("sigmoid_mul/entropy"), t("rms_norm"), t("compute_tier"));
    for _ in 0..n {
        let len = 1 + (rng.next() % 12) as usize;
        let small = |rng: &mut Rng, k: usize, bits: u32| (0..k).map(|_| rng.raw(bits)).collect::<Vec<Raw>>();
        let (a, b) = (small(&mut rng, len, FRAC + 2), small(&mut rng, len, FRAC + 2));
        let (oa, na, ob, nb) = (old_vec(&a), new_vec(&a), old_vec(&b), new_vec(&b));
        let (oas, obs, nas, nbs) = (oa.as_slice(), ob.as_slice(), na.as_slice(), nb.as_slice());
        norms.check(value(|| raw::old_raw(of::sqrt_sum_sq(oas))), value(|| raw::new_raw(nf::sqrt_sum_sq(nas))), &a);
        norms.check(value(|| raw::old_raw(of::inv_sqrt_sum_sq(oas))), value(|| raw::new_raw(nf::inv_sqrt_sum_sq(nas))), &a);
        norms.check(value(|| raw::old_raw(of::euclidean_distance(oas, obs))), value(|| raw::new_raw(nf::euclidean_distance(nas, nbs))), &(&a, &b));
        norms.check(value(|| raw::old_raw(of::euclidean_distance_squared(oas, obs))), value(|| raw::new_raw(nf::euclidean_distance_squared(nas, nbs))), &(&a, &b));
        norms.check(value(|| raw::old_raw(of::dot(oas, obs))), value(|| raw::new_raw(nf::dot(nas, nbs))), &(&a, &b));
        norms.check(value(|| raw::old_raw(of::mobius_denominator_sq(oas, obs))), value(|| raw::new_raw(nf::mobius_denominator_sq(nas, nbs))), &(&a, &b));
        let m = small(&mut rng, len * len, FRAC);
        norms.check(fallible(|| of::try_quadratic_form(&oa, &old_mat(len, len, &m)).map(raw::old_raw)), fallible(|| nf::try_quadratic_form(&na, &new_mat(len, len, &m)).map(raw::new_raw)), &(&a, &m));
        norms.check(value(|| raw::old_raw(of::quadratic_form(&oa, &old_mat(len, len, &m)))), value(|| raw::new_raw(nf::quadratic_form(&na, &new_mat(len, len, &m)))), &(&a, &m));
        norms.check(value(|| old_raws(&OldVector::from_slice(&of::dot_many(oas, &[obs, oas].concat(), len)))), value(|| new_raws(&NewVector::from_slice(&nf::dot_many(nas, &[nbs, nas].concat(), len)))), &(&a, &b));
        // softmax, silu
        let s = small(&mut rng, len, FRAC + 3);
        let (os, ns) = (old_vec(&s), new_vec(&s));
        soft.check(fallible(|| of::softmax(os.as_slice()).map(|v| old_raws(&OldVector::from_slice(&v)))), fallible(|| nf::softmax(ns.as_slice()).map(|v| new_raws(&NewVector::from_slice(&v)))), &s);
        soft.check(value(|| raw::old_raw(of::silu(raw::old(s[0])))), value(|| raw::new_raw(nf::silu(raw::new(s[0])))), &s[0]);
        let eps = rng.raw(FRAC - 4);
        soft.check(fallible(|| of::rms_norm_factor(oas, raw::old(eps).abs_or_self()).map(raw::old_raw)), fallible(|| nf::rms_norm_factor(nas, raw::new(eps).abs_or_self()).map(raw::new_raw)), &(&a, eps));
        let eps_q64 = (rng.next() as i128) << 20;
        soft.check(fallible(|| of::rms_norm_factor_eps_wide(oas, eps_q64).map(raw::old_raw)), fallible(|| nf::rms_norm_factor_eps_wide(nas, eps_q64).map(raw::new_raw)), &(&a, eps_q64));
        // softmax_mix over `len` rows of width `dim`
        let dim = 1 + (rng.next() % 5) as usize;
        let flat = small(&mut rng, len * dim, FRAC + 2);
        let (oflat, nflat) = (old_vec(&flat), new_vec(&flat));
        mix.check(
            fallible(|| of::softmax_mix_flat(os.as_slice(), oflat.as_slice(), dim).map(|(o, w)| (old_raws(&OldVector::from_slice(&o)), old_raws(&OldVector::from_slice(&w))))),
            fallible(|| nf::softmax_mix_flat(ns.as_slice(), nflat.as_slice(), dim).map(|(o, w)| (new_raws(&NewVector::from_slice(&o)), new_raws(&NewVector::from_slice(&w))))),
            &(&s, &flat, dim),
        );
        let orows: Vec<&[Old]> = oflat.as_slice().chunks_exact(dim).collect();
        let nrows: Vec<&[New]> = nflat.as_slice().chunks_exact(dim).collect();
        mix.check(fallible(|| of::softmax_mix_values(os.as_slice(), &orows).map(|o| old_raws(&OldVector::from_slice(&o)))), fallible(|| nf::softmax_mix_values(ns.as_slice(), &nrows).map(|o| new_raws(&NewVector::from_slice(&o)))), &(&s, &flat, dim));
        // sigmoid_mul, entropy
        let g = small(&mut rng, len, FRAC + 3);
        let (og, ng) = (old_vec(&g), new_vec(&g));
        gates.check(value(|| old_raws(&OldVector::from_slice(&of::sigmoid_mul_slice(oas, og.as_slice())))), value(|| new_raws(&NewVector::from_slice(&nf::sigmoid_mul_slice(nas, ng.as_slice())))), &(&a, &g));
        let w: Vec<Raw> = a.iter().map(|&r| r).collect();
        let (ow, nw) = (OldVector::from_slice(&oa.as_slice().iter().map(|x| x.abs_or_self()).collect::<Vec<_>>()), NewVector::from_slice(&na.as_slice().iter().map(|x| x.abs_or_self()).collect::<Vec<_>>()));
        gates.check(fallible(|| of::entropy(ow.as_slice()).map(raw::old_raw)), fallible(|| nf::entropy(nw.as_slice()).map(raw::new_raw)), &w);
        // rms_norm
        rms.check(fallible(|| of::rms_norm(oas, obs, eps_q64).map(|v| old_raws(&OldVector::from_slice(&v)))), fallible(|| nf::rms_norm(nas, nbs, eps_q64).map(|v| new_raws(&NewVector::from_slice(&v)))), &(&a, &b, eps_q64));
        // compute tier (the compute words are different types per crate: compared as their Debug text)
        let x = rng.raw(FRAC + 3);
        let (ox, nx) = (oct::from_fixed(raw::old(x)), nct::from_fixed(raw::new(x)));
        let cf = 2 * FRAC;
        let xm = || Mpf::<C17>::from_raw(x, FRAC);
        let one = || Mpf::<C17>::from_int(1);
        ct.check_closer(value(|| format!("{:?}", oct::exp(ox))), value(|| format!("{:?}", nct::exp::<New>(nx))), cf, || vec![xm().exp()], &x);
        ct.check_closer(value(|| format!("{:?}", oct::sigmoid(ox))), value(|| format!("{:?}", nct::sigmoid::<New>(nx))), cf, || vec![xm().neg().exp().and_then(|e| one().div(one().add(e)))], &x);
        ct.check_closer(value(|| format!("{:?}", oct::softplus(ox))), value(|| format!("{:?}", nct::softplus::<New>(nx))), cf, || vec![xm().exp().and_then(|e| e.ln1p())], &x);
        ct.check_closer(value(|| format!("{:?}", oct::sinhcosh(ox))), value(|| format!("{:?}", nct::sinhcosh::<New>(nx))), cf, || vec![xm().sinh(), xm().cosh()], &x);
        ct.check_closer(value(|| format!("{:?}", oct::ln(ox))), value(|| format!("{:?}", nct::ln::<New>(nx))), cf, || vec![xm().ln()], &x);
        ct.check_closer(value(|| format!("{:?}", oct::sqrt(ox))), value(|| format!("{:?}", nct::sqrt::<New>(nx))), cf, || vec![xm().sqrt()], &x);
        ct.check_closer(value(|| format!("{:?}", oct::ln1p(ox))), value(|| format!("{:?}", nct::ln1p::<New>(nx))), cf, || vec![xm().ln1p()], &x);
        ct.check(value(|| oct::try_to_fixed(oct::exp(ox)).map(raw::old_raw)), value(|| nct::try_to_fixed::<New>(nct::exp::<New>(nx)).map(raw::new_raw)), &x);
    }
    for t in [&norms, &soft, &mix, &gates, &rms, &ct] {
        t.report();
    }
}

use g_math::fixed_point::domains::binary_fixed::transcendental::mpf::{Mpf, MpfErr, C17};
use g_math::fixed_point::I1024;
type ExactValue = Result<Mpf<C17>, MpfErr>;

/// Every integer in a compute word's Debug text, two's complement words
/// (`words: [..]`) or a native integer, as I1024 values.
fn debug_words(s: &str) -> Vec<I1024> {
    let mut out = Vec::new();
    if s.contains("words") {
        for part in s.split("words: [").skip(1) {
            let list = &part[..part.find(']').expect("]")];
            let ws: Vec<u64> = list.split(',').map(|t| t.trim().parse().expect("word")).collect();
            let neg = ws.last().map_or(false, |w| w >> 63 == 1);
            let mut all = [if neg { u64::MAX } else { 0 }; 16];
            all[..ws.len()].copy_from_slice(&ws);
            out.push(I1024::from_words(all));
        }
    } else {
        for t in s.split(|c: char| !(c.is_ascii_digit() || c == '-')).filter(|t| !t.is_empty()) {
            out.push(I1024::from_i128(t.parse().expect("integer")));
        }
    }
    out
}

// ---- the certified interval and the exact predicates (width-types phase 4c) --

#[test]
fn intervals_and_predicates_are_bit_identical_to_0_6_5() {
    use g_math::fixed_point::imperative::predicates as np;
    use g_math::fixed_point::Interval as NI;
    use g_math_065::fixed_point::imperative::predicates as op;
    use g_math_065::fixed_point::Interval as OI;
    silence_panics();
    let n = count(100);
    let mut rng = Rng(0x1A7E_0001);
    let t = |name: &'static str| Tally::new(name);
    let (mut iv, mut sq, mut dots, mut ori, mut pd) = (t("interval ops"), t("interval sqrt"), t("interval dot/qf"), t("orientation"), t("pd_verdict"));
    let oiv = |r: &[Raw]| OI::new(raw::old(r[0]).min(raw::old(r[1])), raw::old(r[0]).max(raw::old(r[1])));
    let niv = |r: &[Raw]| NI::new(raw::new(r[0]).min(raw::new(r[1])), raw::new(r[0]).max(raw::new(r[1])));
    let oraws = |i: OI| (raw::old_raw(i.lo()), raw::old_raw(i.hi()));
    let nraws = |i: NI| (raw::new_raw(i.lo()), raw::new_raw(i.hi()));
    for _ in 0..n {
        let small = |rng: &mut Rng, k: usize, bits: u32| (0..k).map(|_| rng.raw(bits)).collect::<Vec<Raw>>();
        let (a, b) = (small(&mut rng, 2, FRAC + 4), small(&mut rng, 2, FRAC + 4));
        let (oa, ob, na, nb) = (oiv(&a), oiv(&b), niv(&a), niv(&b));
        iv.check(fallible(|| oa.try_add(ob).map(oraws)), fallible(|| na.try_add(nb).map(nraws)), &(&a, &b));
        iv.check(fallible(|| oa.try_sub(ob).map(oraws)), fallible(|| na.try_sub(nb).map(nraws)), &(&a, &b));
        iv.check(fallible(|| oa.try_mul(ob).map(oraws)), fallible(|| na.try_mul(nb).map(nraws)), &(&a, &b));
        iv.check(fallible(|| oa.try_div(ob).map(oraws)), fallible(|| na.try_div(nb).map(nraws)), &(&a, &b));
        iv.check(fallible(|| oa.try_neg().map(oraws)), fallible(|| na.try_neg().map(nraws)), &a);
        iv.check(value(|| oraws(oa * ob)), value(|| nraws(na * nb)), &(&a, &b));
        sq.check(fallible(|| oa.try_sqrt().map(oraws)), fallible(|| na.try_sqrt().map(nraws)), &a);
        let (oab, nab) = (
            OI::new(raw::old(a[0]).abs_or_self().min(raw::old(a[1]).abs_or_self()), raw::old(a[0]).abs_or_self().max(raw::old(a[1]).abs_or_self())),
            NI::new(raw::new(a[0]).abs_or_self().min(raw::new(a[1]).abs_or_self()), raw::new(a[0]).abs_or_self().max(raw::new(a[1]).abs_or_self())),
        );
        sq.check(fallible(|| oab.try_sqrt().map(oraws)), fallible(|| nab.try_sqrt().map(nraws)), &a);
        let (p, q) = (small(&mut rng, 4, FRAC + 2), small(&mut rng, 4, FRAC + 2));
        let (op_, oq, np_, nq) = (old_vec(&p), old_vec(&q), new_vec(&p), new_vec(&q));
        dots.check(fallible(|| OI::try_dot(op_.as_slice(), oq.as_slice()).map(oraws)), fallible(|| NI::try_dot(np_.as_slice(), nq.as_slice()).map(nraws)), &(&p, &q));
        let (opi, npi): (Vec<OI>, Vec<NI>) = (p.chunks(2).map(oiv).collect(), p.chunks(2).map(niv).collect());
        let (oqi, nqi): (Vec<OI>, Vec<NI>) = (q.chunks(2).map(oiv).collect(), q.chunks(2).map(niv).collect());
        dots.check(fallible(|| OI::try_dot_intervals(&opi, &oqi).map(oraws)), fallible(|| NI::try_dot_intervals(&npi, &nqi).map(nraws)), &(&p, &q));
        let m = small(&mut rng, 16, FRAC);
        dots.check(fallible(|| OI::try_quadratic_form(&op_, &old_mat(4, 4, &m)).map(oraws)), fallible(|| NI::try_quadratic_form(&np_, &new_mat(4, 4, &m)).map(nraws)), &(&p, &m));
        // orientation predicates on random points (collinear cases from duplicated coordinates)
        let pts = small(&mut rng, 12, FRAC + 3);
        let pt2 = |r: &[Raw]| ([raw::old(r[0]), raw::old(r[1])], [raw::new(r[0]), raw::new(r[1])]);
        let pt3 = |r: &[Raw]| ([raw::old(r[0]), raw::old(r[1]), raw::old(r[2])], [raw::new(r[0]), raw::new(r[1]), raw::new(r[2])]);
        let ((oa2, na2), (ob2, nb2), (oc2, nc2), (od2, nd2)) = (pt2(&pts[0..2]), pt2(&pts[2..4]), pt2(&pts[4..6]), pt2(&pts[0..2]));
        ori.check(value(|| format!("{:?}", op::orient2d(oa2, ob2, oc2))), value(|| format!("{:?}", np::orient2d(na2, nb2, nc2))), &pts);
        ori.check(value(|| format!("{:?}", op::orient2d(oa2, ob2, od2))), value(|| format!("{:?}", np::orient2d(na2, nb2, nd2))), &pts);
        let ((oa3, na3), (ob3, nb3), (oc3, nc3), (od3, nd3)) = (pt3(&pts[0..3]), pt3(&pts[3..6]), pt3(&pts[6..9]), pt3(&pts[9..12]));
        ori.check(value(|| format!("{:?}", op::orient3d(oa3, ob3, oc3, od3))), value(|| format!("{:?}", np::orient3d(na3, nb3, nc3, nd3))), &pts);
        #[cfg(not(table_format = "q256_256"))]
        {
            let ((oe2, ne2), (of2, nf2)) = (pt2(&pts[6..8]), pt2(&pts[8..10]));
            ori.check(value(|| format!("{:?}", op::incircle(oa2, ob2, oc2, oe2))), value(|| format!("{:?}", np::incircle(na2, nb2, nc2, ne2))), &pts);
            ori.check(value(|| format!("{:?}", op::incircle(oa2, ob2, oe2, of2))), value(|| format!("{:?}", np::incircle(na2, nb2, ne2, nf2))), &pts);
            let (oe3, ne3) = pt3(&pts[1..4]);
            ori.check(value(|| format!("{:?}", op::insphere(oa3, ob3, oc3, od3, oe3))), value(|| format!("{:?}", np::insphere(na3, nb3, nc3, nd3, ne3))), &pts);
        }
        // definiteness of M^T M + c I and of a shifted copy
        let mm = small(&mut rng, 9, FRAC - 1);
        let (om, nm) = (old_mat(3, 3, &mm), new_mat(3, 3, &mm));
        let (ospd, nspd) = (&(&om.transpose() * &om) + &OldMatrix::identity(3), &(&nm.transpose() * &nm) + &NewMatrix::<New>::identity(3));
        let (onot, nnot) = (&(&om.transpose() * &om) - &OldMatrix::identity(3), &(&nm.transpose() * &nm) - &NewMatrix::<New>::identity(3));
        let overdict = |v: Result<op::PdVerdict, _>| v.map(|v| format!("{v:?}"));
        let nverdict = |v: Result<np::PdVerdict, _>| v.map(|v| format!("{v:?}"));
        pd.check(fallible(|| overdict(op::pd_verdict(&ospd))), fallible(|| nverdict(np::pd_verdict(&nspd))), &mm);
        pd.check(fallible(|| overdict(op::pd_verdict(&onot))), fallible(|| nverdict(np::pd_verdict(&nnot))), &mm);
    }
    for t in [&iv, &sq, &dots, &ori, &pd] {
        t.report();
    }
}

// ---- the inference module (width-types phase 4d) ----------------------------

#[test]
fn inference_kernels_are_bit_identical_to_0_6_5() {
    use g_math::tq19 as nt;
    use g_math_065::tq19 as ot;
    silence_panics();
    let n = count(40);
    let mut rng = Rng(0x7019_0001);
    let t = |name: &'static str| Tally::new(name);
    #[allow(unused_mut)] // the row-scaled forms exist on the 32- and 64-bit words only
    let (mut dots, mut dense, mut planar, mut hybrid, mut rowscaled, mut bits) = (t("dots"), t("TQ19Matrix"), t("PlanarTQ19"), t("HybridTQ19"), t("RowScaledTQ19"), t("bits/quantize"));
    let weights = |rng: &mut Rng, k: usize| (0..k).map(|_| ((rng.next() % 59_049) as i32 - 29_524) as i16).collect::<Vec<i16>>();
    let trits = |rng: &mut Rng, k: usize| (0..k).map(|_| (rng.next() % 3) as i8 - 1).collect::<Vec<i8>>();
    // the published crate takes raw words, the tree takes Fixed slices over the same words
    let old_acts = |r: &[Raw]| r.iter().map(|&x| raw::old(x).raw()).collect::<Vec<_>>();
    let old_res = |v| raw::old_raw(Old::from_raw(v));
    let old_vec_res = |v: Vec<_>| v.into_iter().map(|x| raw::old_raw(Old::from_raw(x))).collect::<Vec<Raw>>();
    let new_acts = |r: &[Raw]| r.iter().map(|&x| raw::new(x)).collect::<Vec<_>>();
    let new_raws_of = |v: Vec<New>| v.iter().map(|&x| raw::new_raw(x)).collect::<Vec<Raw>>();
    for _ in 0..n {
        let cols = 1 + (rng.next() % 40) as usize;
        let rows = 1 + (rng.next() % 6) as usize;
        let a = (0..cols).map(|_| rng.raw(FRAC + 4)).collect::<Vec<Raw>>();
        let (oa, na) = (old_acts(&a), new_acts(&a));
        let w = weights(&mut rng, cols);
        let tr = trits(&mut rng, cols);
        dots.check(value(|| old_res(ot::tq19_dot(&w, &oa))), value(|| raw::new_raw(nt::tq19_dot(&w, &na))), &(&w, &a));
        dots.check(value(|| old_res(ot::trit_dot(&tr, &oa))), value(|| raw::new_raw(nt::trit_dot(&tr, &na))), &(&tr, &a));
        let packed: Vec<u8> = tr.chunks(5).map(|c| c.iter().fold(0u8, |acc, &t| acc * 3 + (t + 1) as u8)).collect();
        let s = rng.raw(FRAC + 1);
        dots.check(value(|| old_res(ot::packed_trit_dot(&packed, cols, &oa, raw::old(s).raw()))), value(|| raw::new_raw(nt::packed_trit_dot(&packed, cols, &na, raw::new(s)))), &(&tr, &a, s));
        #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
        dots.check(value(|| format!("{:?}", ot::tq19_dot_q2f(&w, &oa))), value(|| format!("{:?}", nt::tq19_dot_q2f(&w, &na))), &(&w, &a));
        // the dense matrix and its repacked forms
        let data = weights(&mut rng, rows * cols);
        let (om, nm) = (ot::TQ19Matrix::new(rows, cols, data.clone()), nt::TQ19Matrix::new(rows, cols, data.clone()));
        let b = (0..3).map(|_| (0..cols).map(|_| rng.raw(FRAC + 4)).collect::<Vec<Raw>>()).collect::<Vec<_>>();
        let ob: Vec<Vec<_>> = b.iter().map(|r| old_acts(r)).collect();
        let nb: Vec<Vec<New>> = b.iter().map(|r| new_acts(r)).collect();
        let obr: Vec<&[_]> = ob.iter().map(|v| v.as_slice()).collect();
        let nbr: Vec<&[New]> = nb.iter().map(|v| v.as_slice()).collect();
        dense.check(value(|| old_vec_res(om.matvec(&oa))), value(|| new_raws_of(nm.matvec(&na))), &(&data, &a));
        dense.check(value(|| old_vec_res(om.matvec_par(&oa))), value(|| new_raws_of(nm.matvec_par(&na))), &(&data, &a));
        dense.check(value(|| om.matvec_batch(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| nm.matvec_batch(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b));
        dense.check(value(|| om.matvec_batch_par(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| nm.matvec_batch_par(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b));
        #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
        {
            dense.check(value(|| format!("{:?}", om.matvec_q2f(&oa))), value(|| format!("{:?}", nm.matvec_q2f(&na))), &(&data, &a));
            dense.check(value(|| format!("{:?}", om.matvec_q2f_batch_par(&obr))), value(|| format!("{:?}", nm.matvec_q2f_batch_par(&nbr))), &(&data, &b));
        }
        let (op, np_) = (ot::PlanarTQ19::from_tq19(&om), nt::PlanarTQ19::from_tq19(&nm));
        planar.check(value(|| old_vec_res(op.matvec(&oa))), value(|| new_raws_of(np_.matvec(&na))), &(&data, &a));
        planar.check(value(|| op.matvec_batch_par(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| np_.matvec_batch_par(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b));
        let (oh, nh) = (ot::HybridTQ19::from_tq19(&om), nt::HybridTQ19::from_tq19(&nm));
        hybrid.check(value(|| old_vec_res(oh.matvec(&oa))), value(|| new_raws_of(nh.matvec(&na))), &(&data, &a));
        hybrid.check(value(|| oh.matvec_batch_par(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| nh.matvec_batch_par(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b));
        #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
        {
            planar.check(value(|| format!("{:?}", op.matvec_q2f(&oa))), value(|| format!("{:?}", np_.matvec_q2f(&na))), &(&data, &a));
            hybrid.check(value(|| format!("{:?}", oh.matvec_q2f(&oa))), value(|| format!("{:?}", nh.matvec_q2f(&na))), &(&data, &a));
            let scales: Vec<u64> = (0..rows).map(|_| rng.next() % (1u64 << 30)).collect();
            let (or, nr) = (ot::RowScaledTQ19::from_parts(rows, cols, data.clone(), scales.clone()), nt::RowScaledTQ19::from_parts(rows, cols, data.clone(), scales.clone()));
            rowscaled.check(value(|| old_vec_res(or.matvec(&oa))), value(|| new_raws_of(nr.matvec(&na))), &(&data, &a, &scales));
            rowscaled.check(value(|| or.matvec_batch_par(&obr).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| nr.matvec_batch_par(&nbr).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&data, &b, &scales));
            rowscaled.check(value(|| format!("{:?}", or.matvec_q2f(&oa))), value(|| format!("{:?}", nr.matvec_q2f(&na))), &(&data, &a, &scales));
            rowscaled.check(value(|| format!("{:?}", or.matvec_q2f_batch_par(&obr))), value(|| format!("{:?}", nr.matvec_q2f_batch_par(&nbr))), &(&data, &b, &scales));
        }
        #[cfg(table_format = "q16_16")]
        {
            let codes: Vec<i8> = (0..rows * cols).map(|_| ((rng.next() % 243) as i32 - 121) as i8).collect();
            let scales: Vec<u64> = (0..rows).map(|_| rng.next() % (1u64 << 28)).collect();
            let (o5, n5) = (ot::RowScaledTQ5::from_parts(rows, cols, codes.clone(), scales.clone()), nt::RowScaledTQ5::from_parts(rows, cols, codes.clone(), scales.clone()));
            let x: Vec<i32> = a.iter().map(|&r| (r as i64 >> 2) as i32).collect();
            let nx: Vec<New> = x.iter().map(|&v| raw::new(v as Raw)).collect();
            rowscaled.check(value(|| old_vec_res(o5.matvec(&x))), value(|| new_raws_of(n5.matvec(&nx))), &(&codes, &x));
            rowscaled.check(value(|| o5.matvec_q2f(&x)), value(|| n5.matvec_q2f(&nx)), &(&codes, &x));
            let xb: Vec<&[i32]> = vec![&x, &x];
            let nxb: Vec<&[New]> = vec![&nx, &nx];
            rowscaled.check(value(|| o5.matvec_batch_par(&xb).into_iter().map(old_vec_res).collect::<Vec<_>>()), value(|| n5.matvec_batch_par(&nxb).into_iter().map(new_raws_of).collect::<Vec<_>>()), &(&codes, &x));
        }
        // the half-precision decoders and the quantisers
        let hb = (rng.next() & 0xFFFF) as u16;
        for kind in [ot::bits::HalfKind::BFloat16, ot::bits::HalfKind::Binary16] {
            let nkind = match kind { ot::bits::HalfKind::BFloat16 => nt::bits::HalfKind::BFloat16, _ => nt::bits::HalfKind::Binary16 };
            bits.check(value(|| ot::bits::to_tq19_raw(hb, kind).map_err(|e| format!("{e:?}"))), value(|| nt::bits::to_tq19_raw(hb, nkind).map_err(|e| format!("{e:?}"))), &hb);
            #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
            bits.check(value(|| ot::bits::to_fixed(hb, kind).map(raw::old_raw).map_err(|e| format!("{e:?}"))), value(|| nt::bits::to_fixed::<_, FRAC>(hb, nkind).map(raw::new_raw).map_err(|e| format!("{e:?}"))), &hb);
        }
        let wb: Vec<u8> = (0..rows * cols * 2).map(|_| (rng.next() & 0xFF) as u8).collect();
        let (ow, nw) = (ot::bits::WeightBits::from_le_bytes(rows, cols, ot::bits::HalfKind::BFloat16, &wb).unwrap(), nt::bits::WeightBits::from_le_bytes(rows, cols, nt::bits::HalfKind::BFloat16, &wb).unwrap());
        bits.check(value(|| ot::quantize::quantize_tq19(&ow).map(|m| m.data().to_vec())), value(|| nt::quantize::quantize_tq19(&nw).map(|m| m.data().to_vec())), &wb);
        #[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
        bits.check(value(|| ot::quantize::quantize_tq19_rowscaled(&ow).map(|m| (m.data().to_vec(), m.scales_q32().to_vec()))), value(|| nt::quantize::quantize_tq19_rowscaled(&nw).map(|m| (m.data().to_vec(), m.scales_q32().to_vec()))), &wb);
    }
    for t in [&dots, &dense, &planar, &hybrid, &rowscaled, &bits] {
        t.report();
    }
}

// ============================================================================
// The canonical layer: expression strings through both crates (phase 5b)
// ============================================================================

/// Both crates evaluate the same tree; the outcome is the displayed value
/// plus 30 decimal digits, or the error, or a panic.
macro_rules! fasc_case {
    ($tally:expr, $label:expr, |$p:ident| $e:expr) => {{
        let old = fallible(|| -> Result<(String, String), String> {
            let $p = |s: &str| g_math_065::canonical::gmath_parse(s).map_err(|e| format!("{e:?}"));
            let tree = $e;
            g_math_065::canonical::evaluate(&tree)
                .map(|v| (format!("{v}"), v.to_decimal_string(30)))
                .map_err(|e| format!("{e:?}"))
        });
        let new = fallible(|| -> Result<(String, String), String> {
            let $p = |s: &str| g_math::canonical::gmath_parse(s).map_err(|e| format!("{e:?}"));
            let tree = $e;
            g_math::canonical::evaluate(&tree)
                .map(|v| (format!("{v}"), v.to_decimal_string(30)))
                .map_err(|e| format!("{e:?}"))
        });
        let label: String = ($label).to_string();
        if old != new && DOMAIN_CORRECTIONS.contains(&label.as_str()) && new == Outcome::Error(format!("{:?}", "DomainError")) {
            // 0.6.7 returned a value for an expression with no value; the
            // exact evaluation reports the domain error
            $tally.cases += 1;
            $tally.corrected += 1;
        } else if old != new && VALUE_CORRECTIONS.iter().any(|&(l, bits, frac, want)| l == label && bits == 8 * std::mem::size_of::<New>() && frac == FRAC
            && matches!(&new, Outcome::Value((d, _)) if d == want)) {
            // 0.6.7's chain was off; the new value is the exact one (mpmath)
            $tally.cases += 1;
            $tally.corrected += 1;
        } else if old != new {
            FASC_DIFFS.with(|d| d.borrow_mut().push(format!("{label}: old {old:?}, new {new:?}")));
            $tally.cases += 1;
        } else {
            $tally.check(old, new, &label);
        }
    }};
}

thread_local! {
    /// Every canonical difference from 0.6.7 not in a corrections list,
    /// reported together at the end of the test.
    static FASC_DIFFS: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
}

/// Canonical expressions that are undefined (a function outside its domain
/// inside the tree), for which 0.6.7 returned a value: ln of zero or of a
/// negative number has no real value.
/// Canonical chains where 0.6.7 was off and the new display is the exactly
/// rounded value: (label, word bits, fraction bits, display), each checked
/// with mpmath (sin(e^100) = 0.14219812365823863777...).
/// The scientific rows are symbolic results on the 0.7.0 constants (0.6.7
/// carried pi, sqrt 2 and the rest at 253 to 265 bits).
const VALUE_CORRECTIONS: &[(&str, usize, u32, &str)] = &[
    ("sin(exp(100))", 32, 10, "0.142"),
    ("sin(exp(100))", 32, 16, "0.1421"),
    ("sin(exp(100))", 32, 24, "0.14219"),
    ("sin(exp(100))", 64, 32, "0.142198123"),
    ("sin(exp(100))", 128, 64, "0.1421981236582386377"),
    ("sin(exp(100))", 256, 128, "0.14219812365823863777245030647868525633"),
    ("sin(exp(100))", 512, 256, "0.1421981236582386377724503064786852563347376095638780236589265895941923231058"),
    // 1e-9 is exact (symbolic); 0.6.7 rounded it to 0 first and reported a domain error
    ("ln(0.000000001)", 32, 10, "-20.723"),
    ("pow(0.000000001, 0.5)", 32, 10, "0.000"),
    ("ln(0.000000001)", 32, 16, "-20.7232"),
    ("ln(0.000000001)", 32, 24, "-20.72326"),
    ("sqrt2", 512, 256, "1.4142135623730950488016887242096980785696718753769480731766797379907324784621"),
    ("0t12 * pi", 512, 256, "37.6991118430775188615517205993540346103660327925012698516993351076937968754345"),
    ("pi * 0t12", 512, 256, "37.6991118430775188615517205993540346103660327925012698516993351076937968754345"),
    ("pi / 0.1", 512, 256, "31.41592653589793238462643383279502884197169399375105820974944592307816406286208"),
    ("pi / 0.000000001", 512, 256, "3141592653.58979323846264338327950288419716939937510582097494459230781640628620899862803"),
    ("pi * 214748.3647", 512, 256, "674651.88491194168257096980859854856741027560088468517694382031087121177227779418261"),
    ("214748.3647 * pi", 512, 256, "674651.88491194168257096980859854856741027560088468517694382031087121177227779418261"),
    ("214748.3647 / pi", 512, 256, "68356.52752581217035985041065725843234832794794180868600026185015248069555603395431"),
    ("pi * 0x10", 512, 256, "50.26548245743669181540229413247204614715471039000169313559911347692506250057934"),
    ("0x10 * pi", 512, 256, "50.26548245743669181540229413247204614715471039000169313559911347692506250057934"),
];

const DOMAIN_CORRECTIONS: &[&str] = &["exp(ln(0))", "exp(ln(-0.5))", "exp(ln(-2))"];

#[test]
fn fasc_evaluation_is_identical_to_0_6_7() {
    silence_panics();
    let literals = [
        "0", "1", "-1", "42", "0.1", "0.2", "-0.5", "2.75", "1.5", "2.5", "3.14159265358979323846264338327950288",
        "0.000000001", "214748.3647", "214748.3648", "467295.6470", "1000000", "99999999999999999999999999999999999999999.5",
        "1/3", "-2/7", "22/7", "0x1F", "0b101", "0t10", "0t-12", "pi", "e", "sqrt2", "1e-5", "abc", "", "1.2.3",
    ];
    let mut parse = Tally::new("fasc parse");
    for s in literals {
        fasc_case!(parse, s, |p| p(s)?);
    }
    parse.report();

    let operands = ["0.1", "0.2", "1.5", "2.5", "-0.5", "3", "7", "1/3", "0t12", "pi", "0.000000001", "214748.3647", "0x10"];
    let mut arith = Tally::new("fasc arithmetic");
    for a in operands {
        for b in operands {
            fasc_case!(arith, format!("{a} + {b}"), |p| p(a)? + p(b)?);
            fasc_case!(arith, format!("{a} - {b}"), |p| p(a)? - p(b)?);
            fasc_case!(arith, format!("{a} * {b}"), |p| p(a)? * p(b)?);
            fasc_case!(arith, format!("{a} / {b}"), |p| p(a)? / p(b)?);
        }
    }
    arith.report_in_domain();

    let args = ["0", "0.1", "0.5", "1", "1.5", "2", "-0.5", "-2", "10", "0.25", "1/3", "pi", "0t1", "100", "0.000000001"];
    let mut unary = Tally::new("fasc transcendental");
    for a in args {
        fasc_case!(unary, format!("exp({a})"), |p| p(a)?.exp());
        fasc_case!(unary, format!("ln({a})"), |p| p(a)?.ln());
        fasc_case!(unary, format!("sqrt({a})"), |p| p(a)?.sqrt());
        fasc_case!(unary, format!("sin({a})"), |p| p(a)?.sin());
        fasc_case!(unary, format!("cos({a})"), |p| p(a)?.cos());
        fasc_case!(unary, format!("tan({a})"), |p| p(a)?.tan());
        fasc_case!(unary, format!("atan({a})"), |p| p(a)?.atan());
        fasc_case!(unary, format!("asin({a})"), |p| p(a)?.asin());
        fasc_case!(unary, format!("acos({a})"), |p| p(a)?.acos());
        fasc_case!(unary, format!("sinh({a})"), |p| p(a)?.sinh());
        fasc_case!(unary, format!("cosh({a})"), |p| p(a)?.cosh());
        fasc_case!(unary, format!("tanh({a})"), |p| p(a)?.tanh());
        fasc_case!(unary, format!("asinh({a})"), |p| p(a)?.asinh());
        fasc_case!(unary, format!("acosh({a})"), |p| p(a)?.acosh());
        fasc_case!(unary, format!("atanh({a})"), |p| p(a)?.atanh());
        fasc_case!(unary, format!("exp(ln({a}))"), |p| p(a)?.ln().exp());
        fasc_case!(unary, format!("sin(exp({a}))"), |p| p(a)?.exp().sin());
        fasc_case!(unary, format!("sqrt({a}) * sqrt({a})"), |p| p(a)?.sqrt() * p(a)?.sqrt());
        fasc_case!(unary, format!("pow({a}, 0.5)"), |p| p(a)?.pow(p("0.5")?));
        fasc_case!(unary, format!("pow({a}, 3)"), |p| p(a)?.pow(p("3")?));
        fasc_case!(unary, format!("atan2({a}, 2)"), |p| p(a)?.atan2(p("2")?));
    }
    unary.report_in_domain();
    let diffs = FASC_DIFFS.with(|d| d.borrow().clone());
    assert!(diffs.is_empty(), "{} canonical differences from 0.6.7:\n{}", diffs.len(), diffs.join("\n"));
}

// ============================================================================
// DecimalFixed<D>: the compute word follows D (phase 6a)
// ============================================================================

/// The decimal compute digits of this build's profile up to 0.6.7 (the
/// word every `DecimalFixed<D>` ran on, whatever `D`).
#[cfg(table_format = "q16_16")]
const OLD_DECIMAL_DP: u8 = 9;
#[cfg(table_format = "q32_32")]
const OLD_DECIMAL_DP: u8 = 19;
#[cfg(table_format = "q64_64")]
const OLD_DECIMAL_DP: u8 = 38;
#[cfg(table_format = "q128_128")]
const OLD_DECIMAL_DP: u8 = 77;
#[cfg(table_format = "q256_256")]
const OLD_DECIMAL_DP: u8 = 154;

/// Both crates' `try_` transcendentals at one `D` on the same raws. Where
/// the word `D` selects is the word 0.6.7's profile used, the outcomes must
/// be identical. Elsewhere the two words are compared: where 0.6.7's word
/// had at least ten guard digits beyond `D`, both are correctly rounded and
/// the outcomes must still be identical; where it had fewer (the narrow
/// profiles at the larger `D`, where 0.6.7 misrounded or overflowed a
/// compute tier narrower than the storage), differences are counted and
/// reported, and an error in 0.6.7 may become a value.
fn decimal_at_d<const D: u8>(same: &mut Tally, other: &mut (usize, usize, u128, Vec<String>))
where
    g_math::fixed_point::Dp<D>: g_math::fixed_point::DecimalTier,
{
    use g_math::fixed_point::DecimalFixed as NewDec;
    use g_math_065::fixed_point::DecimalFixed as OldDec;
    let new_dp = <<g_math::fixed_point::Dp<D> as g_math::fixed_point::DecimalTier>::Fast as g_math::fixed_point::domains::decimal_fixed::transcendental::DecimalWord>::DP;
    let word = std::any::type_name::<<g_math::fixed_point::Dp<D> as g_math::fixed_point::DecimalTier>::Fast>();
    // identical outcomes required where 0.6.7's word was well guarded; where
    // it IS the fast word, the ladder's fallbacks (ties, overflows, tan next
    // to a pole) are the only differences and are counted, not required
    let same_word = OLD_DECIMAL_DP >= D + 10 && new_dp != OLD_DECIMAL_DP;
    let grid: &[(i128, u32)] = &[
        (0, 0), (5, 1), (-5, 1), (25, 2), (-25, 2), (75, 2), (-75, 2), (9999, 4), (-9999, 4), (1, 3), (-1, 3),
        (1, 0), (-1, 0), (10001, 4), (15, 1), (-15, 1), (2, 0), (-225, 2), (3, 0), (73, 1), (-73, 1), (10, 0),
        (-10, 0), (123456, 3), (-123456, 3), (1_000_000, 0), (1, 1), (33, 2), (-99, 2), (999, 3), (1, 2),
    ];
    let raws: Vec<i128> = grid
        .iter()
        .filter(|(_, k)| *k <= D as u32)
        .filter_map(|&(n, k)| n.checked_mul(10i128.pow(D as u32 - k)))
        .collect();
    macro_rules! case {
        ($label:expr, $raw:expr, |$o:ident| $old:expr, |$n:ident| $new:expr) => {{
            let old = fallible(|| { let $o = OldDec::<D>::from_raw($raw); $old });
            let new = fallible(|| { let $n = NewDec::<D>::from_raw($raw); $new });
            if same_word {
                same.check(old, new, &format!("{}<{}>({})", $label, D, $raw));
            } else {
                other.0 += 1;
                if old != new {
                    other.1 += 1;
                    if let (Outcome::Value(a), Outcome::Value(b)) = (&old, &new) {
                        let units = a.iter().zip(b.iter()).map(|(x, y)| (x - y).unsigned_abs()).max().unwrap_or(0);
                        other.2 = other.2.max(units);
                        // 0.6.7 under-guarded: counted above, not bounded (atanh near 1
                        // at D = 6 on 9 digits was 5 units off)
                    } else if !matches!((&old, &new), (Outcome::Error(_), Outcome::Value(_))) {
                        other.3.push(format!("{}<{}>({}): 0.6.7 {:?}, tree {:?}", $label, D, $raw, old, new));
                    }
                }
            }
        }};
    }
    for &raw in &raws {
        case!("exp", raw, |o| o.try_exp().map(|v| vec![v.raw_value()]), |n| n.try_exp().map(|v| vec![v.raw_value()]));
        case!("ln", raw, |o| o.try_ln().map(|v| vec![v.raw_value()]), |n| n.try_ln().map(|v| vec![v.raw_value()]));
        case!("sqrt", raw, |o| o.try_sqrt().map(|v| vec![v.raw_value()]), |n| n.try_sqrt().map(|v| vec![v.raw_value()]));
        case!("sin", raw, |o| o.try_sin().map(|v| vec![v.raw_value()]), |n| n.try_sin().map(|v| vec![v.raw_value()]));
        case!("cos", raw, |o| o.try_cos().map(|v| vec![v.raw_value()]), |n| n.try_cos().map(|v| vec![v.raw_value()]));
        case!("sincos", raw, |o| o.try_sincos().map(|(a, b)| vec![a.raw_value(), b.raw_value()]), |n| n.try_sincos().map(|(a, b)| vec![a.raw_value(), b.raw_value()]));
        case!("tan", raw, |o| o.try_tan().map(|v| vec![v.raw_value()]), |n| n.try_tan().map(|v| vec![v.raw_value()]));
        case!("atan", raw, |o| o.try_atan().map(|v| vec![v.raw_value()]), |n| n.try_atan().map(|v| vec![v.raw_value()]));
        case!("asin", raw, |o| o.try_asin().map(|v| vec![v.raw_value()]), |n| n.try_asin().map(|v| vec![v.raw_value()]));
        case!("acos", raw, |o| o.try_acos().map(|v| vec![v.raw_value()]), |n| n.try_acos().map(|v| vec![v.raw_value()]));
        case!("sinh", raw, |o| o.try_sinh().map(|v| vec![v.raw_value()]), |n| n.try_sinh().map(|v| vec![v.raw_value()]));
        case!("cosh", raw, |o| o.try_cosh().map(|v| vec![v.raw_value()]), |n| n.try_cosh().map(|v| vec![v.raw_value()]));
        case!("sinhcosh", raw, |o| o.try_sinhcosh().map(|(a, b)| vec![a.raw_value(), b.raw_value()]), |n| n.try_sinhcosh().map(|(a, b)| vec![a.raw_value(), b.raw_value()]));
        case!("tanh", raw, |o| o.try_tanh().map(|v| vec![v.raw_value()]), |n| n.try_tanh().map(|v| vec![v.raw_value()]));
        case!("asinh", raw, |o| o.try_asinh().map(|v| vec![v.raw_value()]), |n| n.try_asinh().map(|v| vec![v.raw_value()]));
        case!("acosh", raw, |o| o.try_acosh().map(|v| vec![v.raw_value()]), |n| n.try_acosh().map(|v| vec![v.raw_value()]));
        case!("atanh", raw, |o| o.try_atanh().map(|v| vec![v.raw_value()]), |n| n.try_atanh().map(|v| vec![v.raw_value()]));
        for &x in &raws[..raws.len().min(8)] {
            case!("atan2", raw, |o| o.try_atan2(OldDec::<D>::from_raw(x)).map(|v| vec![v.raw_value()]), |n| n.try_atan2(NewDec::<D>::from_raw(x)).map(|v| vec![v.raw_value()]));
        }
    }
    println!(
        "DecimalFixed<{D}> fast on {word} ({new_dp} digits; 0.6.7 {OLD_DECIMAL_DP}): {}",
        if same_word { "identical required" } else { "compared" }
    );
}

#[test]
fn decimal_fixed_transcendentals_against_0_6_7() {
    silence_panics();
    let mut same = Tally::new("decimal same word");
    let mut other = (0usize, 0usize, 0u128, Vec::new());
    decimal_at_d::<0>(&mut same, &mut other);
    decimal_at_d::<2>(&mut same, &mut other);
    decimal_at_d::<4>(&mut same, &mut other);
    decimal_at_d::<6>(&mut same, &mut other);
    decimal_at_d::<9>(&mut same, &mut other);
    decimal_at_d::<12>(&mut same, &mut other);
    decimal_at_d::<19>(&mut same, &mut other);
    decimal_at_d::<28>(&mut same, &mut other);
    decimal_at_d::<38>(&mut same, &mut other);
    if same.cases > 0 {
        same.report_in_domain();
    } else {
        same.report();
    }
    println!(
        "decimal other word     {:>6} cases: {:>6} differ from 0.6.7, at most {} unit(s) at D",
        other.0, other.1, other.2
    );
    assert!(other.3.is_empty(), "{} unexpected differences:\n{}", other.3.len(), other.3[..other.3.len().min(30)].join("\n"));
}

// ============================================================================
// Serialization (phase 6b): 0.6.7 bytes through the legacy reader, the
// untagged forms identical in both directions
// ============================================================================

#[test]
fn serialization_reads_0_6_7_bytes_and_writes_the_same_untagged_forms() {
    silence_panics();
    let raws = {
        let mut v = specials();
        v.extend(inputs(0x5e71a1, 40, 0));
        v
    };
    // scalars: 0.6.7's tagged bytes read through the legacy entry and refused
    // by the 0.7.0 reader; the raw forms identical
    let mut scalar = Tally::new("scalar bytes");
    for &r in &raws {
        let old = raw::old(r);
        let new = raw::new(r);
        let old_bytes = old.to_bytes();
        assert_eq!(old_bytes[0], New::legacy_tag(), "0.6.7 tag is the legacy tag");
        assert_eq!(New::from_bytes_legacy(&old_bytes).map(raw::new_raw), Ok(r), "legacy read of 0.6.7 bytes");
        assert!(New::from_bytes(&old_bytes).is_err(), "0.7.0 reader refuses a 0.6.7 tag");
        let new_bytes = new.to_bytes();
        assert_eq!(new_bytes[3..], old_bytes[1..], "the raw payload is unchanged");
        assert!(Old::from_bytes(&new_bytes).is_err(), "0.6.7 refuses the 0.7.0 tag");
        scalar.check(value(|| old.to_raw_bytes()), value(|| new.to_raw_bytes()), &r);
        // the compact form of the word's minimum panicked in 0.6.7 (abs); it
        // is the full class now, checked by the round trip below instead
        if quiet(|| old.to_compact_bytes()).is_some() {
            scalar.check(value(|| old.to_compact_bytes()), value(|| new.to_compact_bytes()), &r);
            scalar.check(
                fallible(|| Old::from_compact_bytes(&old.to_compact_bytes()).map(|(v, n)| (raw::old_raw(v), n))),
                fallible(|| New::from_compact_bytes(&new.to_compact_bytes()).map(|(v, n)| (raw::new_raw(v), n))),
                &r,
            );
        } else {
            assert_eq!(New::from_compact_bytes(&new.to_compact_bytes()).map(|(v, _)| raw::new_raw(v)), Ok(r), "compact round trip of the minimum");
        }
    }
    scalar.report();
    // vectors, matrices, tensors, manifold points: the same bytes, readable
    // by the other crate
    let mut forms = Tally::new("container bytes");
    for n in [0usize, 1, 3, 7] {
        let chunk = &raws[..n.min(raws.len())];
        let (ov, nv) = (old_vec(chunk), new_vec(chunk));
        let (ob, nb) = (ov.to_bytes(), nv.to_bytes());
        forms.check(value(|| ob.clone()), value(|| nb.clone()), &n);
        assert_eq!(new_raws(&NewVector::from_bytes(&ob).unwrap()), chunk, "vector from 0.6.7 bytes");
        assert_eq!(old_raws(&OldVector::from_bytes(&nb).unwrap()), chunk, "0.6.7 vector from 0.7.0 bytes");
        forms.check(value(|| ov.to_compact_bytes()), value(|| nv.to_compact_bytes()), &n);
        assert_eq!(new_raws(&NewVector::from_compact_bytes(&ov.to_compact_bytes()).unwrap().0), chunk);
        if n > 0 {
            let (om, nm) = (old_mat(1, n, chunk), new_mat(1, n, chunk));
            let (ob, nb) = (om.to_bytes(), nm.to_bytes());
            forms.check(value(|| ob.clone()), value(|| nb.clone()), &n);
            assert_eq!(new_mraws(&NewMatrix::from_bytes(&ob).unwrap()), chunk, "matrix from 0.6.7 bytes");
            assert_eq!(old_mraws(&OldMatrix::from_bytes(&nb).unwrap()), chunk);
            let (ot, nt) = (old_tensor(&[n], chunk), new_tensor(&[n], chunk));
            let (ob, nb) = (ot.to_bytes(), nt.to_bytes());
            forms.check(value(|| ob.clone()), value(|| nb.clone()), &n);
            assert_eq!(new_traws(&geo_new::Tensor::from_bytes(&ob).unwrap()), chunk, "tensor from 0.6.7 bytes");
            assert_eq!(old_traws(&geo_old::Tensor::from_bytes(&nb).unwrap()), chunk);
        }
        let op = g_math_065::fixed_point::imperative::ManifoldPoint::sphere(n, ov.clone());
        let np = g_math::fixed_point::imperative::ManifoldPoint::sphere(n, nv.clone());
        let (ob, nb) = (op.to_bytes(), np.to_bytes());
        forms.check(value(|| ob.clone()), value(|| nb.clone()), &n);
        let back = g_math::fixed_point::imperative::ManifoldPoint::<New>::from_bytes(&ob).unwrap();
        assert_eq!((back.manifold_tag, back.params.clone(), new_raws(&back.coordinates)), (op.manifold_tag, op.params.clone(), chunk.to_vec()));
    }
    forms.report();
}
