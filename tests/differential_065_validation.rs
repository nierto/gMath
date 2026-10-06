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
            pow.check(old_pow, new_pow, &(x, y));
        }
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
        sph.check(fallible(|| s_o.distance(&op, &oq).map(raw::old_raw)), fallible(|| s_n.distance(&np, &nq).map(raw::new_raw)), &(&a, &b));
        sph.check(fallible(|| s_o.log_map(&op, &oq).map(|v| old_raws(&v))), fallible(|| s_n.log_map(&np, &nq).map(|v| new_raws(&v))), &(&a, &b));
        let (ot, nt) = (o_::Manifold::log_map(&s_o, &op, &oq), n_::Manifold::log_map(&s_n, &np, &nq));
        if let (Ok(ot), Ok(nt)) = (ot, nt) {
            sph.check(fallible(|| s_o.exp_map(&op, &ot).map(|v| old_raws(&v))), fallible(|| s_n.exp_map(&np, &nt).map(|v| new_raws(&v))), &(&a, &b));
            sph.check(fallible(|| s_o.parallel_transport(&op, &oq, &ot).map(|v| old_raws(&v))), fallible(|| s_n.parallel_transport(&np, &nq, &nt).map(|v| new_raws(&v))), &(&a, &b));
        }
        // hyperboloid H^2: x0 = sqrt(1 + x1^2 + x2^2)
        let (a, b) = (small(&mut rng, 2, FRAC - 1), small(&mut rng, 2, FRAC - 1));
        let lift_old = |r: &[Raw]| { let (x, y) = (raw::old(r[0]), raw::old(r[1])); OldVector::from_slice(&[(Old::one() + x * x + y * y).sqrt(), x, y]) };
        let lift_new = |r: &[Raw]| { let (x, y) = (raw::new(r[0]), raw::new(r[1])); NewVector::from_slice(&[(New::one() + x * x + y * y).sqrt(), x, y]) };
        let (op, oq, np, nq) = (lift_old(&a), lift_old(&b), lift_new(&a), lift_new(&b));
        let h_o = o_::HyperbolicSpace { dim: 2 };
        let h_n = n_::HyperbolicSpace { dim: 2 };
        hyp.check(fallible(|| h_o.distance(&op, &oq).map(raw::old_raw)), fallible(|| h_n.distance(&np, &nq).map(raw::new_raw)), &(&a, &b));
        hyp.check(fallible(|| h_o.log_map(&op, &oq).map(|v| old_raws(&v))), fallible(|| h_n.log_map(&np, &nq).map(|v| new_raws(&v))), &(&a, &b));
        if let (Ok(ot), Ok(nt)) = (h_o.log_map(&op, &oq), h_n.log_map(&np, &nq)) {
            hyp.check(fallible(|| h_o.exp_map(&op, &ot).map(|v| old_raws(&v))), fallible(|| h_n.exp_map(&np, &nt).map(|v| new_raws(&v))), &(&a, &b));
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
        spd.check(fallible(|| p_o.distance(&opv, &oqv).map(raw::old_raw)), fallible(|| p_n.distance(&npv, &nqv).map(raw::new_raw)), &(&m, &w));
        spd.check(fallible(|| p_o.exp_map(&opv, &otv).map(|v| old_raws(&v))), fallible(|| p_n.exp_map(&npv, &ntv).map(|v| new_raws(&v))), &(&m, &w));
        spd.check(fallible(|| p_o.log_map(&opv, &oqv).map(|v| old_raws(&v))), fallible(|| p_n.log_map(&npv, &nqv).map(|v| new_raws(&v))), &(&m, &w));
        // SO(3), SE(3), SO(n)
        let xi = small(&mut rng, 3, FRAC);
        let (oxi, nxi) = (old_vec(&xi), new_vec(&xi));
        so3.check(fallible(|| o_::SO3.lie_exp(&oxi).map(|g| old_mraws(&g))), fallible(|| n_::SO3.lie_exp(&nxi).map(|g| new_mraws(&g))), &xi);
        if let (Ok(og), Ok(ng)) = (o_::SO3.lie_exp(&oxi), n_::SO3.lie_exp(&nxi)) {
            so3.check(fallible(|| o_::SO3.lie_log(&og).map(|v| old_raws(&v))), fallible(|| n_::SO3.lie_log(&ng).map(|v| new_raws(&v))), &xi);
            so3.check(fallible(|| o_::SO3.adjoint(&og, &oxi).map(|v| old_raws(&v))), fallible(|| n_::SO3.adjoint(&ng, &nxi).map(|v| new_raws(&v))), &xi);
            son.check(fallible(|| o_::SOn { n: 3 }.lie_log(&og).map(|v| old_raws(&v))), fallible(|| n_::SOn { n: 3 }.lie_log(&ng).map(|v| new_raws(&v))), &xi);
        }
        son.check(fallible(|| o_::SOn { n: 3 }.lie_exp(&oxi).map(|g| old_mraws(&g))), fallible(|| n_::SOn { n: 3 }.lie_exp(&nxi).map(|g| new_mraws(&g))), &xi);
        let xi6 = small(&mut rng, 6, FRAC);
        let (oxi6, nxi6) = (old_vec(&xi6), new_vec(&xi6));
        se3.check(fallible(|| o_::SE3.lie_exp(&oxi6).map(|g| old_mraws(&g))), fallible(|| n_::SE3.lie_exp(&nxi6).map(|g| new_mraws(&g))), &xi6);
        if let (Ok(og), Ok(ng)) = (o_::SE3.lie_exp(&oxi6), n_::SE3.lie_exp(&nxi6)) {
            se3.check(fallible(|| o_::SE3.lie_log(&og).map(|v| old_raws(&v))), fallible(|| n_::SE3.lie_log(&ng).map(|v| new_raws(&v))), &xi6);
        }
        // curvature on the round sphere (radius in (1, 2)) and the hyperbolic plane
        let r = rng.raw(FRAC - 1);
        let ang = small(&mut rng, 2, FRAC - 1); // theta, phi in (-0.5, 0.5) + offsets
        let (otp, ntp) = (
            OldVector::from_slice(&[raw::old(ang[0]).abs_or_self() + Old::one() / Old::from_int(2), raw::old(ang[1])]),
            NewVector::from_slice(&[raw::new(ang[0]).abs_or_self() + New::one() / New::from_int(2), raw::new(ang[1])]),
        );
        let (sm_o, sm_n) = (o_::SphereMetric { radius: Old::one() + raw::old(r).abs_or_self() }, n_::SphereMetric { radius: New::one() + raw::new(r).abs_or_self() });
        curv.check(fallible(|| o_::christoffel(&sm_o, &otp).map(|t| old_traws(&t))), fallible(|| n_::christoffel(&sm_n, &ntp).map(|t| new_traws(&t))), &(r, &ang));
        curv.check(fallible(|| o_::riemann_curvature(&sm_o, &otp).map(|t| old_traws(&t))), fallible(|| n_::riemann_curvature(&sm_n, &ntp).map(|t| new_traws(&t))), &(r, &ang));
        curv.check(fallible(|| o_::scalar_curvature(&sm_o, &otp).map(raw::old_raw)), fallible(|| n_::scalar_curvature(&sm_n, &ntp).map(raw::new_raw)), &(r, &ang));
        let (ohp, nhp) = (OldVector::from_slice(&[raw::old(ang[0]), raw::old(ang[1])]), NewVector::from_slice(&[raw::new(ang[0]), raw::new(ang[1])]));
        curv.check(fallible(|| o_::scalar_curvature(&o_::HyperbolicMetric, &ohp).map(raw::old_raw)), fallible(|| n_::scalar_curvature(&n_::HyperbolicMetric, &nhp).map(raw::new_raw)), &ang);
        // ODE: one RK4 step and a short RK45 run of the oscillator, Verlet
        let x = small(&mut rng, 2, FRAC);
        let h = rng.raw(FRAC - 3);
        let (ox, nx, oh, nh) = (old_vec(&x), new_vec(&x), raw::old(h).abs_or_self(), raw::new(h).abs_or_self());
        let osc_o = o_::ode_fn(|_t: Old, x: &OldVector| OldVector::from_slice(&[x[1], -x[0]]));
        let osc_n = n_::ode_fn(|_t: New, x: &NewVector| NewVector::from_slice(&[x[1], -x[0]]));
        ode.check(value(|| old_raws(&o_::rk4_step(&osc_o, Old::ZERO, &ox, oh))), value(|| new_raws(&n_::rk4_step(&osc_n, New::ZERO, &nx, nh))), &(&x, h));
        let end_o = oh * Old::from_int(3);
        let end_n = nh * New::from_int(3);
        ode.check(
            fallible(|| o_::rk45_integrate(&osc_o, &ox, Old::ZERO, end_o, &o_::Rk45Config::new(oh, oh)).map(|(pts, s)| (pts.iter().map(|p| (raw::old_raw(p.t), old_raws(&p.x))).collect::<Vec<_>>(), s))),
            fallible(|| n_::rk45_integrate(&osc_n, &nx, New::ZERO, end_n, &n_::Rk45Config::new(nh, nh)).map(|(pts, s)| (pts.iter().map(|p| (raw::new_raw(p.t), new_raws(&p.x))).collect::<Vec<_>>(), s))),
            &(&x, h),
        );
        let (oq, nq) = (old_vec(&x[..1]), new_vec(&x[..1]));
        ode.check(
            value(|| o_::verlet_integrate(&Oscillator, &oq, &oq, Old::ZERO, end_o, oh).iter().map(|p| (old_raws(&p.q), old_raws(&p.p), raw::old_raw(p.energy))).collect::<Vec<_>>()),
            value(|| n_::verlet_integrate(&Oscillator, &nq, &nq, New::ZERO, end_n, nh).iter().map(|p| (new_raws(&p.q), new_raws(&p.p), raw::new_raw(p.energy))).collect::<Vec<_>>()),
            &(&x, h),
        );
        // projective: Moebius maps, the cross ratio, stereographic projection
        let c = small(&mut rng, 5, FRAC);
        let (mo, mn) = (o_::Moebius::new(raw::old(c[0]), raw::old(c[1]), raw::old(c[2]), raw::old(c[3])), n_::Moebius::new(raw::new(c[0]), raw::new(c[1]), raw::new(c[2]), raw::new(c[3])));
        proj.check(fallible(|| mo.apply(raw::old(c[4])).map(raw::old_raw)), fallible(|| mn.apply(raw::new(c[4])).map(raw::new_raw)), &c);
        proj.check(value(|| old_mraws(&mo.compose(&mo.inverse()).to_matrix())), value(|| new_mraws(&mn.compose(&mn.inverse()).to_matrix())), &c);
        let q4 = small(&mut rng, 4, FRAC + 1);
        proj.check(
            fallible(|| o_::cross_ratio_1d(raw::old(q4[0]), raw::old(q4[1]), raw::old(q4[2]), raw::old(q4[3])).map(raw::old_raw)),
            fallible(|| n_::cross_ratio_1d(raw::new(q4[0]), raw::new(q4[1]), raw::new(q4[2]), raw::new(q4[3])).map(raw::new_raw)),
            &q4,
        );
        let (a3, b3) = (small(&mut rng, 3, FRAC), small(&mut rng, 2, FRAC));
        let (op3, np3) = (unit_old(old_vec(&a3)), unit_new(new_vec(&a3)));
        proj.check(fallible(|| o_::stereo_project(&op3).map(|v| old_raws(&v))), fallible(|| n_::stereo_project(&np3).map(|v| new_raws(&v))), &a3);
        proj.check(value(|| old_raws(&o_::stereo_unproject(&old_vec(&b3)))), value(|| new_raws(&n_::stereo_unproject(&new_vec(&b3)))), &b3);
        proj.check(fallible(|| o_::from_homogeneous(&o_::to_homogeneous(&old_vec(&a3))).map(|v| old_raws(&v))), fallible(|| n_::from_homogeneous(&n_::to_homogeneous(&new_vec(&a3))).map(|v| new_raws(&v))), &a3);
        // tensors: contraction, trace, symmetrisation, the three decompositions
        let (ta, tb) = (small(&mut rng, 6, FRAC), small(&mut rng, 6, FRAC));
        ten.check(value(|| old_traws(&o_::contract(&old_tensor(&[2, 3], &ta), &old_tensor(&[3, 2], &tb), &[(1, 0)]))), value(|| new_traws(&n_::contract(&new_tensor(&[2, 3], &ta), &new_tensor(&[3, 2], &tb), &[(1, 0)]))), &(&ta, &tb));
        let t8 = small(&mut rng, 8, FRAC);
        ten.check(value(|| old_traws(&o_::trace(&old_tensor(&[2, 2, 2], &t8), 0, 2))), value(|| new_traws(&n_::trace(&new_tensor(&[2, 2, 2], &t8), 0, 2))), &t8);
        ten.check(value(|| old_traws(&o_::symmetrize(&old_tensor(&[2, 2, 2], &t8), &[0, 1]))), value(|| new_traws(&n_::symmetrize(&new_tensor(&[2, 2, 2], &t8), &[0, 1]))), &t8);
        let m9 = small(&mut rng, 9, FRAC);
        ten.check(fallible(|| o_::truncated_svd(&old_mat(3, 3, &m9), 2).map(|d| old_mraws(&d.reconstruct()))), fallible(|| n_::truncated_svd(&new_mat(3, 3, &m9), 2).map(|d| new_mraws(&d.reconstruct()))), &m9);
        ten.check(fallible(|| o_::tucker_decompose(&old_tensor(&[2, 2, 2], &t8), &[2, 2, 2]).map(|d| old_traws(&d.reconstruct()))), fallible(|| n_::tucker_decompose(&new_tensor(&[2, 2, 2], &t8), &[2, 2, 2]).map(|d| new_traws(&d.reconstruct()))), &t8);
        ten.check(
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
        fib.check(fallible(|| vb_o.parallel_transport_along(&opath, &old_vec(&f0)).map(|v| old_raws(&v))), fallible(|| vb_n.parallel_transport_along(&npath, &new_vec(&f0)).map(|v| new_raws(&v))), &(&coeffs, &path, &f0));
    }
    for t in [&sph, &hyp, &spd, &so3, &se3, &son, &curv, &ode, &proj, &ten, &fib] {
        t.report();
    }
}
