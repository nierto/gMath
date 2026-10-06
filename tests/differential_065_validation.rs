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

use g_math::fixed_point::FixedPoint as New;
use g_math_065::fixed_point::FixedPoint as Old;
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---- the same raw integer in both crates --------------------------------

#[cfg(any(table_format = "q16_16", table_format = "q32_32", table_format = "q64_64"))]
mod raw {
    #[cfg(table_format = "q16_16")]
    pub type Raw = i32;
    #[cfg(table_format = "q32_32")]
    pub type Raw = i64;
    #[cfg(table_format = "q64_64")]
    pub type Raw = i128;
    pub const WORDS: usize = (Raw::BITS / 64) as usize + (Raw::BITS % 64 != 0) as usize;
    pub fn from_words(w: &[u64]) -> Raw {
        let mut v: u128 = 0;
        for (i, x) in w.iter().enumerate().take(2) {
            v |= (*x as u128) << (64 * i);
        }
        v as Raw
    }
    pub fn old(r: Raw) -> super::Old { super::Old::from_raw(r) }
    pub fn new(r: Raw) -> super::New { super::New::from_raw(r) }
    pub fn old_raw(v: super::Old) -> Raw { v.raw() }
    pub fn new_raw(v: super::New) -> Raw { v.raw() }
}

#[cfg(any(table_format = "q128_128", table_format = "q256_256"))]
mod raw {
    #[cfg(table_format = "q128_128")]
    pub const WORDS: usize = 4;
    #[cfg(table_format = "q256_256")]
    pub const WORDS: usize = 8;
    #[cfg(table_format = "q128_128")]
    use g_math::fixed_point::I256 as NewInt;
    #[cfg(table_format = "q256_256")]
    use g_math::fixed_point::I512 as NewInt;
    #[cfg(table_format = "q128_128")]
    use g_math_065::fixed_point::I256 as OldInt;
    #[cfg(table_format = "q256_256")]
    use g_math_065::fixed_point::I512 as OldInt;
    pub type Raw = [u64; WORDS];
    pub fn from_words(w: &[u64]) -> Raw {
        let mut r = [0u64; WORDS];
        r.copy_from_slice(&w[..WORDS]);
        r
    }
    pub fn old(r: Raw) -> super::Old { super::Old::from_raw(OldInt { words: r }) }
    pub fn new(r: Raw) -> super::New { super::New::from_raw(NewInt { words: r }) }
    pub fn old_raw(v: super::Old) -> Raw { v.raw().words }
    pub fn new_raw(v: super::New) -> Raw { v.raw().words }
}

use raw::{Raw, WORDS};

#[allow(dead_code)]
const BITS: u32 = WORDS as u32 * 64;

#[cfg(table_format = "q16_16")]
const WIDTH: u32 = 32;
#[cfg(not(table_format = "q16_16"))]
const WIDTH: u32 = BITS;
const FRAC: u32 = g_math::fixed_point::frac_config::FRAC_BITS;

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
}

impl Tally {
    fn new(name: &'static str) -> Self {
        Tally { name, cases: 0, values: 0, errors: 0, panics: 0 }
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
            "{:<22} {:>6} cases: {:>6} values, {:>5} errors, {:>5} panics, 0 differences",
            self.name, self.cases, self.values, self.errors, self.panics
        );
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

macro_rules! unary_gate {
    ($t:ident, $report:ident, $inputs:expr; $($name:literal => $try_m:ident / $m:ident),+ $(,)?) => {$(
        let mut tally = Tally::new($name);
        let mut infallible = Tally::new(concat!($name, " (panicking)"));
        for &r in $inputs.iter() {
            tally.check(
                fallible(|| raw::old(r).$try_m().map(raw::old_raw)),
                fallible(|| raw::new(r).$try_m().map(raw::new_raw)),
                &r,
            );
            infallible.check(value(|| raw::old_raw(raw::old(r).$m())), value(|| raw::new_raw(raw::new(r).$m())), &r);
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
    for literal in ["pi", "1/3", "0x1F", "0b101", "0t12", "0.3...", "", " ", "abc", "1e", "--1", "e", "sqrt2"] {
        parse.check(
            fallible(|| Old::try_from_str(literal).map(raw::old_raw)),
            fallible(|| New::try_from_str(literal).map(raw::new_raw)),
            &literal,
        );
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
        sincos.check(
            fallible(|| ox.try_sincos().map(|(s, c)| (raw::old_raw(s), raw::old_raw(c)))),
            fallible(|| nx.try_sincos().map(|(s, c)| (raw::new_raw(s), raw::new_raw(c)))),
            &x,
        );
        sinhcosh.check(
            fallible(|| ox.try_sinhcosh().map(|(s, c)| (raw::old_raw(s), raw::old_raw(c)))),
            fallible(|| nx.try_sinhcosh().map(|(s, c)| (raw::new_raw(s), raw::new_raw(c)))),
            &x,
        );
        pow.check(fallible(|| ox.try_pow(oy).map(raw::old_raw)), fallible(|| nx.try_pow(ny).map(raw::new_raw)), &(x, y));
        pow.check(value(|| raw::old_raw(ox.pow(oy))), value(|| raw::new_raw(nx.pow(ny))), &(x, y));
        atan2.check(fallible(|| oy.try_atan2(ox).map(raw::old_raw)), fallible(|| ny.try_atan2(nx).map(raw::new_raw)), &(y, x));
        atan2.check(value(|| raw::old_raw(oy.atan2(ox))), value(|| raw::new_raw(ny.atan2(nx))), &(y, x));
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
